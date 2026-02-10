"""
LLM generation wrapper.

Accepts prompt text and context, calls Groq, returns raw completion.
Contains ZERO domain wording — prompt content comes from the registry.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Tuple

from ops import call_groq_with_retry
from shared.schema import AnswerType

logger = logging.getLogger(__name__)


def build_user_message(context_block: str, question: str) -> str:
    """Build the user message from context and question."""
    return f"CONTEXT:\n{context_block}\n\nQUESTION:\n{question}"


def build_context_block(selected: List[Dict[str, Any]]) -> str:
    """Build the context string from selected chunks."""
    parts: List[str] = []
    for s in selected:
        header = f"[chunk_id={s['chunk_id']}]"
        parts.append(f"{header}\n{s['text']}")
    return "\n\n---\n\n".join(parts)


def generate(
    groq_client: Any,
    system_prompt: str,
    selected_contexts: List[Dict[str, Any]],
    question: str,
    model: str,
    temperature: float = 0.0,
) -> Tuple[AnswerType, str, List[str], int]:
    """Call LLM and parse the structured response.

    Parameters
    ----------
    groq_client:
        Initialized Groq client.
    system_prompt:
        Full system prompt text (loaded from prompt registry).
    selected_contexts:
        List of context dicts from retrieval.
    question:
        The user question.
    model:
        Groq model identifier.
    temperature:
        Sampling temperature.

    Returns
    -------
    (answer_type, answer_text, used_chunk_ids, llm_ms)
    """
    import time

    context_block = build_context_block(selected_contexts)
    user_message = build_user_message(context_block, question)

    t0 = time.perf_counter()
    chat_completion = call_groq_with_retry(
        client=groq_client,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        model=model,
        temperature=temperature,
    )
    llm_ms = int((time.perf_counter() - t0) * 1000)

    raw_reply = chat_completion.choices[0].message.content or ""

    # Parse structured JSON response
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
        logger.warning("Could not parse LLM response as JSON: %s", exc)

    return answer_type, answer_text, used_chunk_ids, llm_ms
