"""
RAG Query Engine - FastAPI Application

POST /query endpoint that:
  1. Embeds the question via SentenceTransformer
  2. Retrieves top_k candidates from Qdrant
  3. Reranks with FlashRank
  4. Enforces token_budget deterministically
  5. Calls Groq LLM for grounded answer synthesis
  6. Returns QueryResponse with citations and timings
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

from ops import (
    call_groq_with_retry,
    health_check,
    log_timings,
    ready_check,
    setup_logging,
)
from shared.schema import (
    AnswerType,
    Citation,
    ContextItem,
    QueryRequest,
    QueryResponse,
    TimingsMS,
)

# ---------------------------------------------------------------------------
# Environment & logging
# ---------------------------------------------------------------------------
load_dotenv()
setup_logging()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------
MODEL_PATH = os.getenv("MODEL_PATH", "sentence-transformers/all-MiniLM-L6-v2")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
COLLECTION_NAME = "knowledge_base"

# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------
PROMPT_PATH = Path(__file__).parent / "prompt.md"


def _load_system_prompt() -> str:
    """Read prompt.md and return its contents as the system prompt."""
    if not PROMPT_PATH.exists():
        raise FileNotFoundError(f"prompt.md not found at {PROMPT_PATH}")
    return PROMPT_PATH.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Application state (populated during lifespan)
# ---------------------------------------------------------------------------
class AppState:
    """Mutable container for resources initialised at startup."""

    def __init__(self) -> None:
        self.embed_model: Any = None
        self.qdrant_client: Any = None
        self.reranker: Any = None
        self.groq_client: Any = None
        self.system_prompt: str = ""


state = AppState()

# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise heavy resources once at startup, tear down on shutdown."""
    from sentence_transformers import SentenceTransformer
    from qdrant_client import QdrantClient
    from flashrank import Ranker
    from groq import Groq

    logger.info("Loading system prompt from prompt.md ...")
    state.system_prompt = _load_system_prompt()

    logger.info("Loading SentenceTransformer model: %s ...", MODEL_PATH)
    state.embed_model = SentenceTransformer(MODEL_PATH)

    logger.info("Connecting to Qdrant at %s:%s ...", QDRANT_HOST, QDRANT_PORT)
    state.qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    logger.info("Initialising FlashRank reranker ...")
    state.reranker = Ranker()

    logger.info("Initialising Groq client ...")
    state.groq_client = Groq(api_key=GROQ_API_KEY)

    logger.info("Startup complete.")
    yield
    logger.info("Shutting down.")


app = FastAPI(title="RAG Query Engine", lifespan=lifespan)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: 1 token ~ 4 characters."""
    return len(text) // 4


def _ms_since(start: float) -> int:
    """Milliseconds elapsed since *start* (time.perf_counter timestamp)."""
    return int((time.perf_counter() - start) * 1000)


# ---------------------------------------------------------------------------
# POST /query
# ---------------------------------------------------------------------------


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest) -> QueryResponse:
    """
    Full RAG pipeline: embed -> retrieve -> rerank -> budget-trim -> LLM -> respond.
    """
    from flashrank import RerankRequest

    total_start = time.perf_counter()

    # ------------------------------------------------------------------
    # (a) Embed the question
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    query_vector = state.embed_model.encode(req.question).tolist()
    embed_ms = _ms_since(t0)

    # ------------------------------------------------------------------
    # (b) Qdrant vector search (top_k candidates)
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    search_results = state.qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=req.top_k,
        with_payload=True,
    )
    retrieve_ms = _ms_since(t0)

    if not search_results:
        # Nothing in the collection -- return NOT_FOUND immediately
        return QueryResponse(
            answer_type="NOT_FOUND",
            answer="No relevant documents found in the knowledge base.",
            citations=[],
            contexts=[],
            timings_ms=TimingsMS(
                embed=embed_ms,
                retrieve=retrieve_ms,
                rerank=0,
                llm=0,
                total=_ms_since(total_start),
            ),
        )

    # ------------------------------------------------------------------
    # (c) FlashRank rerank
    # ------------------------------------------------------------------
    t0 = time.perf_counter()

    # Build passage list for FlashRank
    passages: List[Dict[str, Any]] = []
    for hit in search_results:
        payload = hit.payload or {}
        passages.append(
            {
                "id": payload.get("chunk_id", str(hit.id)),
                "text": payload.get("text", ""),
                "meta": payload,
            }
        )

    rerank_request = RerankRequest(query=req.question, passages=passages)
    reranked = state.reranker.rerank(rerank_request)
    rerank_ms = _ms_since(t0)

    # ------------------------------------------------------------------
    # (d) Select top_n results subject to token_budget
    #     Budget accounts for FULL prompt: system + question + contexts
    # ------------------------------------------------------------------
    system_tokens = _estimate_tokens(state.system_prompt)
    question_tokens = _estimate_tokens(req.question)
    formatting_tokens = _estimate_tokens("CONTEXT:\n\n---\n\nQUESTION:\n")
    prompt_overhead = system_tokens + question_tokens + formatting_tokens
    available_for_contexts = max(0, req.token_budget - prompt_overhead)

    original_top_n = min(req.top_n, len(reranked))
    selected: List[Dict[str, Any]] = []
    token_count = 0

    for item in reranked[: req.top_n]:
        text = item.get("text", "") if isinstance(item, dict) else getattr(item, "text", "")
        meta = item.get("meta", {}) if isinstance(item, dict) else getattr(item, "meta", {})
        score = item.get("score", 0.0) if isinstance(item, dict) else getattr(item, "score", 0.0)
        item_id = item.get("id", "") if isinstance(item, dict) else getattr(item, "id", "")

        estimated_tokens = _estimate_tokens(text)
        if token_count + estimated_tokens > available_for_contexts:
            break
        token_count += estimated_tokens
        selected.append(
            {
                "chunk_id": meta.get("chunk_id", item_id),
                "text": text,
                "score": float(score),
                "source_path": meta.get("source_path", ""),
                "page_index": meta.get("page_index"),
                "page_label": meta.get("page_label"),
                "text_preview": meta.get("text_preview", text[:120]),
            }
        )

    # If no contexts fit but we have candidates, truncate the first to fit
    if not selected and reranked and available_for_contexts > 0:
        item = reranked[0]
        text = item.get("text", "") if isinstance(item, dict) else getattr(item, "text", "")
        meta = item.get("meta", {}) if isinstance(item, dict) else getattr(item, "meta", {})
        score = item.get("score", 0.0) if isinstance(item, dict) else getattr(item, "score", 0.0)
        item_id = item.get("id", "") if isinstance(item, dict) else getattr(item, "id", "")
        max_chars = available_for_contexts * 4
        truncated = text[:max_chars]
        token_count = _estimate_tokens(truncated)
        logger.info(
            "Token policy: truncated context 0 from %d to %d chars to fit budget",
            len(text), len(truncated),
        )
        selected.append(
            {
                "chunk_id": meta.get("chunk_id", item_id),
                "text": truncated,
                "score": float(score),
                "source_path": meta.get("source_path", ""),
                "page_index": meta.get("page_index"),
                "page_label": meta.get("page_label"),
                "text_preview": meta.get("text_preview", truncated[:120]),
            }
        )

    if len(selected) < original_top_n:
        logger.info(
            "Token policy: reduced contexts from %d to %d "
            "(budget=%d, overhead=%d, available=%d, used=%d)",
            original_top_n, len(selected), req.token_budget,
            prompt_overhead, available_for_contexts, token_count,
        )

    # Build ContextItem list for the response
    contexts = [
        ContextItem(
            chunk_id=s["chunk_id"],
            score=s["score"],
            source_path=s["source_path"],
            page_index=s["page_index"],
            page_label=s["page_label"],
            text_preview=s["text_preview"],
        )
        for s in selected
    ]

    # ------------------------------------------------------------------
    # (e) Build context string
    # ------------------------------------------------------------------
    context_parts: List[str] = []
    for s in selected:
        header = f"[chunk_id={s['chunk_id']}]"
        context_parts.append(f"{header}\n{s['text']}")
    context_block = "\n\n---\n\n".join(context_parts)

    # ------------------------------------------------------------------
    # (f) Call Groq LLM
    # ------------------------------------------------------------------
    user_message = (
        f"CONTEXT:\n{context_block}\n\n"
        f"QUESTION:\n{req.question}"
    )

    # Final budget verification before Groq call
    total_prompt_tokens = prompt_overhead + token_count
    logger.info(
        "Token policy: total_prompt_est=%d, budget=%d, contexts=%d",
        total_prompt_tokens, req.token_budget, len(selected),
    )

    t0 = time.perf_counter()
    try:
        chat_completion = call_groq_with_retry(
            client=state.groq_client,
            messages=[
                {"role": "system", "content": state.system_prompt},
                {"role": "user", "content": user_message},
            ],
            model=GROQ_MODEL,
            temperature=0.0,
        )
        raw_reply = chat_completion.choices[0].message.content or ""
    except Exception as exc:
        logger.error("Groq call failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"LLM call failed: {exc}") from exc
    llm_ms = _ms_since(t0)

    # ------------------------------------------------------------------
    # (g) Parse Groq JSON response
    # ------------------------------------------------------------------
    answer_type: AnswerType = "PARTIAL"
    answer_text: str = raw_reply
    used_chunk_ids: List[str] = []

    try:
        parsed = json.loads(raw_reply)
        raw_answer_type = parsed.get("answer_type", "PARTIAL")
        if raw_answer_type in ("COMPLETE", "PARTIAL", "NOT_FOUND"):
            answer_type = raw_answer_type  # type: ignore[assignment]
        else:
            answer_type = "PARTIAL"
        answer_text = parsed.get("answer", raw_reply)
        used_chunk_ids = parsed.get("used_chunk_ids", [])
    except (json.JSONDecodeError, TypeError, KeyError) as exc:
        logger.warning("Could not parse Groq response as JSON: %s", exc)
        # Fallback: answer_type stays PARTIAL, answer_text stays raw_reply

    # ------------------------------------------------------------------
    # (h) Map used_chunk_ids -> Citation objects
    # ------------------------------------------------------------------
    chunk_lookup: Dict[str, Dict[str, Any]] = {s["chunk_id"]: s for s in selected}
    citations: List[Citation] = []
    for cid in used_chunk_ids:
        if cid in chunk_lookup:
            c = chunk_lookup[cid]
            citations.append(
                Citation(
                    source_path=c["source_path"],
                    chunk_id=cid,
                    page_index=c.get("page_index"),
                    page_label=c.get("page_label"),
                )
            )

    # ------------------------------------------------------------------
    # (i) Return QueryResponse
    # ------------------------------------------------------------------
    total_ms = _ms_since(total_start)

    timings = TimingsMS(
        embed=embed_ms,
        retrieve=retrieve_ms,
        rerank=rerank_ms,
        llm=llm_ms,
        total=total_ms,
    )

    # Log latency split
    log_timings(timings.model_dump())

    return QueryResponse(
        answer_type=answer_type,
        answer=answer_text,
        citations=citations,
        contexts=contexts,
        timings_ms=timings,
    )


# ---------------------------------------------------------------------------
# Health & Readiness
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    """Liveness probe."""
    return health_check()


@app.get("/ready")
async def ready():
    """Readiness probe — checks Qdrant connectivity."""
    return ready_check(state.qdrant_client)
