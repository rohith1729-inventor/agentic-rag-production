"""
Prompt registry.

Maps prompt_id to a file in the prompts/ directory.
Core code never contains domain wording — it loads prompt text at runtime.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Cache loaded prompts to avoid repeated disk reads
_cache: dict[str, str] = {}


def load_prompt(prompt_id: str, prompts_dir: Optional[Path] = None) -> str:
    """Load prompt text by id from the prompts directory.

    Parameters
    ----------
    prompt_id:
        Identifier mapping to ``prompts/{prompt_id}.md``.
    prompts_dir:
        Override for the prompts directory path. If None, resolves
        to ``<project_root>/prompts/``.

    Returns
    -------
    The prompt text content.

    Raises
    ------
    FileNotFoundError
        If the prompt file does not exist.
    """
    if prompt_id in _cache:
        return _cache[prompt_id]

    if prompts_dir is None:
        # Default: <project_root>/prompts/
        prompts_dir = Path(__file__).parent.parent.parent / "prompts"

    prompt_path = prompts_dir / f"{prompt_id}.md"
    if not prompt_path.exists():
        raise FileNotFoundError(
            f"Prompt '{prompt_id}' not found at {prompt_path}. "
            f"Create prompts/{prompt_id}.md to register it."
        )

    text = prompt_path.read_text(encoding="utf-8")
    _cache[prompt_id] = text
    logger.info("Loaded prompt '%s' from %s", prompt_id, prompt_path)
    return text


def clear_cache() -> None:
    """Clear the prompt cache (useful for testing or hot-reload)."""
    _cache.clear()
