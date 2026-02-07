"""
Ops utilities: health checks, Groq retry wrapper, and structured logging.

Imported by main.py -- these are plain functions, not FastAPI routes.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception,
)


# ---------------------------------------------------------------------------
# Structured logging
# ---------------------------------------------------------------------------

def setup_logging() -> None:
    """Configure root logger with a structured format."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )


def log_timings(timings_dict: Dict[str, Any]) -> None:
    """Log the latency split at INFO level.

    Expected keys: embed, retrieve, rerank, llm, total (values in ms).
    """
    logger = logging.getLogger(__name__)
    parts = []
    for key in ("embed", "retrieve", "rerank", "llm", "total"):
        value = timings_dict.get(key)
        if value is not None:
            parts.append(f"{key}={value}ms")
    logger.info("Timings | %s", " | ".join(parts))


# ---------------------------------------------------------------------------
# Health / readiness checks
# ---------------------------------------------------------------------------

def health_check() -> Dict[str, str]:
    """Simple liveness probe."""
    return {"status": "ok"}


def ready_check(qdrant_client: Any) -> Dict[str, str]:
    """Readiness probe -- verifies Qdrant connectivity.

    Parameters
    ----------
    qdrant_client:
        A ``qdrant_client.QdrantClient`` instance (or compatible).

    Returns
    -------
    dict with ``status`` key ("ready" or "not_ready").
    """
    try:
        # get_collections is a lightweight call that confirms the server is up.
        qdrant_client.get_collections()
        return {"status": "ready"}
    except Exception as exc:  # noqa: BLE001
        return {"status": "not_ready", "error": str(exc)}


# ---------------------------------------------------------------------------
# Groq retry wrapper
# ---------------------------------------------------------------------------

def _is_retryable_error(exc: BaseException) -> bool:
    """Return True for rate-limit (429) and server (5xx) errors."""
    # groq library raises groq.RateLimitError (429) and
    # groq.InternalServerError / groq.APIStatusError for 5xx.
    status = getattr(exc, "status_code", None)
    if status is not None:
        return status == 429 or status >= 500
    return False


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=4),
    retry=retry_if_exception(_is_retryable_error),
    reraise=True,
)
def call_groq_with_retry(
    client: Any,
    messages: list,
    model: str,
    temperature: float = 0.1,
    max_tokens: int = 1024,
) -> Any:
    """Call the Groq chat completions API with automatic retry.

    Retries up to 3 times with exponential back-off (1 s, 2 s, 4 s)
    on 429 (rate limit) and 5xx (server error) responses.

    Parameters
    ----------
    client:
        A ``groq.Groq`` client instance.
    messages:
        List of message dicts for the chat completion.
    model:
        Model identifier (e.g. ``"llama3-70b-8192"``).
    temperature:
        Sampling temperature (default 0.1).
    max_tokens:
        Maximum tokens in the response (default 1024).

    Returns
    -------
    The raw chat completion response object.
    """
    return client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
