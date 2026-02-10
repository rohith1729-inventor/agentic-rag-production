"""
Retrieval strategy configuration and execution.

RetrievalProfile defines how retrieval is performed.
Default profile reproduces current pipeline behavior exactly.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class RetrievalProfile(BaseModel):
    """Configurable retrieval strategy."""

    strategy: Literal["vector_only"] = Field(
        default="vector_only",
        description="Retrieval strategy. Currently only vector_only is implemented.",
    )
    rerank: bool = Field(
        default=True,
        description="Whether to apply FlashRank reranking after retrieval.",
    )
    top_k: int = Field(default=20, description="Number of candidates to retrieve from vector search.")
    top_n: int = Field(default=5, description="Number of results to keep after reranking/selection.")


# Named profiles registry
PROFILES: Dict[str, RetrievalProfile] = {
    "default": RetrievalProfile(),
    "vector_only": RetrievalProfile(rerank=False),
    "precise": RetrievalProfile(top_k=40, top_n=3),
}


def get_profile(name: Optional[str] = None) -> RetrievalProfile:
    """Resolve a named profile or return default."""
    if name is None or name not in PROFILES:
        return PROFILES["default"]
    return PROFILES[name]


def retrieve_candidates(
    qdrant_client: Any,
    collection_name: str,
    query_vector: List[float],
    top_k: int,
) -> List[Any]:
    """Execute vector search against Qdrant.

    Returns list of Qdrant search result hits.
    """
    return qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=top_k,
        with_payload=True,
    )


def rerank_results(
    reranker: Any,
    question: str,
    search_results: List[Any],
) -> List[Any]:
    """Rerank search results using FlashRank.

    Returns reranked list of passage dicts/objects.
    """
    from flashrank import RerankRequest

    passages: List[Dict[str, Any]] = []
    for hit in search_results:
        payload = hit.payload or {}
        passages.append({
            "id": payload.get("chunk_id", str(hit.id)),
            "text": payload.get("text", ""),
            "meta": payload,
        })

    rerank_request = RerankRequest(query=question, passages=passages)
    return reranker.rerank(rerank_request)


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: 1 token ~ 4 characters."""
    return len(text) // 4


def apply_token_budget(
    candidates: List[Any],
    top_n: int,
    token_budget: int,
    system_prompt: str,
    question: str,
) -> List[Dict[str, Any]]:
    """Select top_n candidates subject to token_budget.

    Budget accounts for FULL prompt: system + question + contexts.
    Returns list of selected context dicts.

    Deterministic: iterates in rank order, stops when budget exhausted.
    If no full context fits, truncates the first to fit.
    """
    system_tokens = _estimate_tokens(system_prompt)
    question_tokens = _estimate_tokens(question)
    formatting_tokens = _estimate_tokens("CONTEXT:\n\n---\n\nQUESTION:\n")
    prompt_overhead = system_tokens + question_tokens + formatting_tokens
    available_for_contexts = max(0, token_budget - prompt_overhead)

    original_top_n = min(top_n, len(candidates))
    selected: List[Dict[str, Any]] = []
    token_count = 0

    for item in candidates[:top_n]:
        text = item.get("text", "") if isinstance(item, dict) else getattr(item, "text", "")
        meta = item.get("meta", {}) if isinstance(item, dict) else getattr(item, "meta", {})
        score = item.get("score", 0.0) if isinstance(item, dict) else getattr(item, "score", 0.0)
        item_id = item.get("id", "") if isinstance(item, dict) else getattr(item, "id", "")

        estimated_tokens = _estimate_tokens(text)
        if token_count + estimated_tokens > available_for_contexts:
            break
        token_count += estimated_tokens
        selected.append({
            "chunk_id": meta.get("chunk_id", item_id),
            "text": text,
            "score": float(score),
            "source_path": meta.get("source_path", ""),
            "page_index": meta.get("page_index"),
            "page_label": meta.get("page_label"),
            "text_preview": meta.get("text_preview", text[:120]),
        })

    # If no contexts fit but we have candidates, truncate the first to fit
    if not selected and candidates and available_for_contexts > 0:
        item = candidates[0]
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
        selected.append({
            "chunk_id": meta.get("chunk_id", item_id),
            "text": truncated,
            "score": float(score),
            "source_path": meta.get("source_path", ""),
            "page_index": meta.get("page_index"),
            "page_label": meta.get("page_label"),
            "text_preview": meta.get("text_preview", truncated[:120]),
        })

    if len(selected) < original_top_n:
        logger.info(
            "Token policy: reduced contexts from %d to %d "
            "(budget=%d, overhead=%d, available=%d, used=%d)",
            original_top_n, len(selected), token_budget,
            prompt_overhead, available_for_contexts, token_count,
        )

    return selected


def retrieve_and_rank(
    qdrant_client: Any,
    reranker: Any,
    collection_name: str,
    query_vector: List[float],
    question: str,
    profile: RetrievalProfile,
    system_prompt: str,
    token_budget: int,
    top_k_override: Optional[int] = None,
    top_n_override: Optional[int] = None,
) -> tuple[List[Dict[str, Any]], int, int]:
    """Full retrieval pipeline: search → optional rerank → budget trim.

    Parameters
    ----------
    top_k_override, top_n_override:
        If provided, override the profile's top_k/top_n (from QueryRequest).

    Returns
    -------
    (selected_contexts, retrieve_ms, rerank_ms)
    """
    import time

    top_k = top_k_override if top_k_override is not None else profile.top_k
    top_n = top_n_override if top_n_override is not None else profile.top_n

    # Vector search
    t0 = time.perf_counter()
    search_results = retrieve_candidates(qdrant_client, collection_name, query_vector, top_k)
    retrieve_ms = int((time.perf_counter() - t0) * 1000)

    if not search_results:
        return [], retrieve_ms, 0

    # Rerank (if enabled)
    t0 = time.perf_counter()
    if profile.rerank and reranker is not None:
        ranked = rerank_results(reranker, question, search_results)
    else:
        # No reranking — convert search results to passage-like dicts
        ranked = []
        for hit in search_results:
            payload = hit.payload or {}
            ranked.append({
                "id": payload.get("chunk_id", str(hit.id)),
                "text": payload.get("text", ""),
                "score": hit.score,
                "meta": payload,
            })
    rerank_ms = int((time.perf_counter() - t0) * 1000)

    # Token budget enforcement
    selected = apply_token_budget(ranked, top_n, token_budget, system_prompt, question)

    return selected, retrieve_ms, rerank_ms
