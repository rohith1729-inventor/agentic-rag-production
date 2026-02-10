"""
RAG Query Engine - FastAPI Application

POST /query endpoint backed by universal rag_core components.
This file is thin wiring only — no business logic, no domain wording.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

from ops import health_check, log_timings, ready_check, setup_logging
from rag_core.config import get_config
from rag_core.generation import generate
from rag_core.prompts import load_prompt
from rag_core.retrieval.strategy import get_profile, retrieve_and_rank
from shared.schema import (
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
# Application state (populated during lifespan)
# ---------------------------------------------------------------------------


class AppState:
    """Mutable container for resources initialised at startup."""

    def __init__(self) -> None:
        self.embed_model: Any = None
        self.qdrant_client: Any = None
        self.reranker: Any = None
        self.groq_client: Any = None


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

    cfg = get_config()

    logger.info("Loading SentenceTransformer model: %s ...", cfg.model_path)
    state.embed_model = SentenceTransformer(cfg.model_path)

    logger.info("Connecting to Qdrant at %s:%s ...", cfg.qdrant_host, cfg.qdrant_port)
    state.qdrant_client = QdrantClient(host=cfg.qdrant_host, port=cfg.qdrant_port)

    logger.info("Initialising FlashRank reranker ...")
    state.reranker = Ranker()

    logger.info("Initialising Groq client ...")
    state.groq_client = Groq(api_key=cfg.groq_api_key)

    # Warm the default prompt cache
    load_prompt("default", cfg.prompts_path())
    logger.info("Startup complete.")
    yield
    logger.info("Shutting down.")


app = FastAPI(title="RAG Query Engine", lifespan=lifespan)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ms_since(start: float) -> int:
    """Milliseconds elapsed since *start* (time.perf_counter timestamp)."""
    return int((time.perf_counter() - start) * 1000)


# ---------------------------------------------------------------------------
# POST /query
# ---------------------------------------------------------------------------


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest) -> QueryResponse:
    """Full RAG pipeline: embed -> retrieve -> rerank -> budget-trim -> LLM -> respond."""
    cfg = get_config()
    total_start = time.perf_counter()

    # (a) Embed the question
    t0 = time.perf_counter()
    query_vector = state.embed_model.encode(req.question).tolist()
    embed_ms = _ms_since(t0)

    # (b) Resolve retrieval profile
    profile = get_profile(req.retrieval_profile)

    # (c) Load prompt
    prompt_id = req.prompt_id or "default"
    system_prompt = load_prompt(prompt_id, cfg.prompts_path())

    # (d) Retrieve, rerank, and apply token budget
    selected, retrieve_ms, rerank_ms = retrieve_and_rank(
        qdrant_client=state.qdrant_client,
        reranker=state.reranker,
        collection_name=cfg.collection_name,
        query_vector=query_vector,
        question=req.question,
        profile=profile,
        system_prompt=system_prompt,
        token_budget=req.token_budget,
        top_k_override=req.top_k,
        top_n_override=req.top_n,
    )

    if not selected:
        return QueryResponse(
            answer_type="NOT_FOUND",
            answer="No relevant documents found in the knowledge base.",
            citations=[],
            contexts=[],
            timings_ms=TimingsMS(
                embed=embed_ms,
                retrieve=retrieve_ms,
                rerank=rerank_ms,
                llm=0,
                total=_ms_since(total_start),
            ),
        )

    # (e) Build ContextItem list for the response
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

    # (f) Call LLM via generation module
    try:
        answer_type, answer_text, used_chunk_ids, llm_ms = generate(
            groq_client=state.groq_client,
            system_prompt=system_prompt,
            selected_contexts=selected,
            question=req.question,
            model=cfg.groq_model,
            temperature=0.0,
        )
    except Exception as exc:
        logger.error("LLM call failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"LLM call failed: {exc}") from exc

    # (g) Map used_chunk_ids -> Citation objects
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

    # (h) Return QueryResponse
    total_ms = _ms_since(total_start)
    timings = TimingsMS(
        embed=embed_ms,
        retrieve=retrieve_ms,
        rerank=rerank_ms,
        llm=llm_ms,
        total=total_ms,
    )
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
