"""
Abstract ingestion adapter interface.

Every adapter must implement load() and return a list of page/section entries
with standardized metadata keys.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple


class IngestionAdapter(ABC):
    """Base class for ingestion adapters.

    Subclasses implement load() to produce page entries from any source.
    Each entry dict must contain:
        - source: str (identifier for the source)
        - text: str (extracted text content)
        - page_index: Optional[int] (physical page index for PDFs, None otherwise)
        - source_type: str (e.g. 'pdf', 'md', 'txt')
    """

    @abstractmethod
    def load(self, source: str) -> Tuple[List[Dict], int]:
        """Load documents from *source*.

        Parameters
        ----------
        source:
            Source identifier (e.g. directory path, URL, database connection string).

        Returns
        -------
        (page_entries, doc_count)
            page_entries: list of dicts with keys: source, text, page_index, source_type
            doc_count: number of documents loaded
        """
        ...
