"""
Filesystem ingestion adapter.

Loads PDF, Markdown, and plain text files from a local directory.
This is the first concrete implementation of IngestionAdapter.
"""

from __future__ import annotations

import glob
import os
from typing import Dict, List, Tuple

from pypdf import PdfReader

from rag_core.ingest.base import IngestionAdapter


class FilesystemAdapter(IngestionAdapter):
    """Load documents from a local filesystem directory.

    Supports: *.pdf, *.md, *.txt
    """

    SUPPORTED_PATTERNS = ["**/*.pdf", "**/*.md", "**/*.txt"]

    def load(self, source: str) -> Tuple[List[Dict], int]:
        """Load all supported files from directory *source*.

        Returns (page_entries, doc_count) where each entry has:
            source, text, page_index, source_type
        """
        data_dir = source

        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            print(f"Created '{data_dir}' directory. Please add documents and re-run.")
            return [], 0

        files: List[str] = []
        for pat in self.SUPPORTED_PATTERNS:
            files.extend(glob.glob(os.path.join(data_dir, pat), recursive=True))
        files = sorted(set(files))

        print(f"Found {len(files)} document(s) in '{data_dir}'.")

        all_entries: List[Dict] = []
        for fpath in files:
            source_type = self._detect_type(fpath)
            if source_type == "pdf":
                entries = self._load_pdf(fpath)
            else:
                entries = self._load_text_file(fpath, source_type)
            all_entries.extend(entries)

        return all_entries, len(files)

    @staticmethod
    def _detect_type(file_path: str) -> str:
        """Detect source_type from file extension."""
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            return "pdf"
        elif ext == ".md":
            return "md"
        else:
            return "txt"

    @staticmethod
    def _load_pdf(file_path: str) -> List[Dict]:
        """Extract text from a PDF, one entry per physical page."""
        pages: List[Dict] = []
        reader = PdfReader(file_path)
        for page_index, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            text = text.strip()
            if text:
                pages.append({
                    "source": file_path,
                    "text": text,
                    "page_index": page_index,
                    "source_type": "pdf",
                })
        return pages

    @staticmethod
    def _load_text_file(file_path: str, source_type: str) -> List[Dict]:
        """Load a plain text or markdown file."""
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        if not text:
            return []
        return [{
            "source": file_path,
            "text": text,
            "page_index": None,
            "source_type": source_type,
        }]
