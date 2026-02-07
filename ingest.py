"""
Ingestion pipeline for RAG project.

Loads PDF, Markdown, and plain text documents from a data directory,
chunks them, generates embeddings, and upserts into Qdrant.

Usage:
    python ingest.py                  # uses default "data" directory
    python ingest.py --path ./docs    # uses specified directory
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import os
import sys
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from shared.schema import ChunkMeta, QdrantPoint

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
COLLECTION_NAME = "knowledge_base"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
MIN_CHUNK_SIZE = 100
DEFAULT_DATA_DIR = "data"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256(text: str) -> str:
    """Return the hex SHA-256 digest of *text*."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _make_doc_id(source_path: str) -> str:
    """Stable document id derived from the source path."""
    return _sha256(source_path)


def _make_chunk_id(doc_id: str, page_index: Optional[int], chunk_index: int) -> str:
    """Format: {doc_id}_{page_index}_{chunk_index}. Uses 'none' when page_index is None."""
    page_part = "none" if page_index is None else str(page_index)
    return f"{doc_id}_{page_part}_{chunk_index}"


def _stable_uuid(doc_id: str, page_index: Optional[int], chunk_index: int) -> str:
    """Deterministic UUID5 from (doc_id, page_index, chunk_index) for idempotent upserts."""
    page_part = "none" if page_index is None else str(page_index)
    name = f"{doc_id}:{page_part}:{chunk_index}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, name))


def _text_preview(text: str, max_len: int = 100) -> str:
    """First *max_len* characters of *text*."""
    return text[:max_len]


# ---------------------------------------------------------------------------
# Document loading
# ---------------------------------------------------------------------------


def _load_pdf(file_path: str) -> List[Dict]:
    """
    Extract text from a PDF file, one entry per physical page.
    Returns a list of dicts: {"source": str, "text": str, "page_index": int}
    """
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
            })
    return pages


def _load_text_file(file_path: str) -> List[Dict]:
    """
    Load a plain text or markdown file.
    Returns a list with one dict: {"source": str, "text": str, "page_index": None}
    """
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        return []
    return [{"source": file_path, "text": text, "page_index": None}]


def load_documents(data_dir: str) -> Tuple[List[Dict], int]:
    """
    Discover and load all supported files (*.pdf, *.md, *.txt) from *data_dir*.

    Returns:
        (page_entries, doc_count)
        Each page_entry has keys: source, text, page_index.
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created '{data_dir}' directory. Please add documents and re-run.")
        return [], 0

    patterns = ["**/*.pdf", "**/*.md", "**/*.txt"]
    files: List[str] = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(data_dir, pat), recursive=True))
    # Deduplicate (a file could match multiple patterns theoretically) and sort
    files = sorted(set(files))

    print(f"Found {len(files)} document(s) in '{data_dir}'.")
    all_entries: List[Dict] = []
    for fpath in files:
        if fpath.lower().endswith(".pdf"):
            entries = _load_pdf(fpath)
        else:
            entries = _load_text_file(fpath)
        all_entries.extend(entries)

    return all_entries, len(files)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def _merge_small_chunks(chunks: List[str], min_size: int) -> List[str]:
    """Merge consecutive tiny chunks (< *min_size* characters) to preserve context."""
    if not chunks:
        return chunks

    merged: List[str] = []
    current = ""

    for chunk in chunks:
        if not current:
            current = chunk
            continue

        if len(chunk) < min_size:
            # Tiny chunk -- merge into current
            current = current + "\n" + chunk
        elif len(current) < min_size:
            # Current is tiny -- merge incoming into it
            current = current + "\n" + chunk
        else:
            merged.append(current)
            current = chunk

    if current:
        merged.append(current)

    return merged


def chunk_entries(
    entries: List[Dict],
) -> List[Dict]:
    """
    Split each page entry into overlapping chunks using RecursiveCharacterTextSplitter.
    Returns a flat list of dicts with keys:
        source, page_index, chunk_index, text
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )

    # Group entries by source to assign chunk_index per document
    # (chunk_index is global across all pages of a single document)
    source_entries: Dict[str, List[Dict]] = {}
    for entry in entries:
        source_entries.setdefault(entry["source"], []).append(entry)

    all_chunks: List[Dict] = []

    for source, page_entries in source_entries.items():
        chunk_counter = 0
        for entry in page_entries:
            raw_chunks = splitter.split_text(entry["text"])
            merged_chunks = _merge_small_chunks(raw_chunks, MIN_CHUNK_SIZE)
            for chunk_text in merged_chunks:
                all_chunks.append({
                    "source": source,
                    "page_index": entry["page_index"],
                    "chunk_index": chunk_counter,
                    "text": chunk_text,
                })
                chunk_counter += 1

    return all_chunks


# ---------------------------------------------------------------------------
# Build QdrantPoints
# ---------------------------------------------------------------------------


def build_points(
    chunks: List[Dict],
    model: SentenceTransformer,
) -> List[QdrantPoint]:
    """
    Generate embeddings and construct QdrantPoint objects for each chunk.
    """
    if not chunks:
        return []

    texts = [c["text"] for c in chunks]
    print(f"Encoding {len(texts)} chunk(s) ...")
    vectors = model.encode(texts, show_progress_bar=True)

    now_iso = datetime.now(timezone.utc).isoformat()
    points: List[QdrantPoint] = []

    for chunk, vector in zip(chunks, vectors):
        doc_id = _make_doc_id(chunk["source"])
        page_index: Optional[int] = chunk["page_index"]
        chunk_index: int = chunk["chunk_index"]

        chunk_id = _make_chunk_id(doc_id, page_index, chunk_index)
        point_id = _stable_uuid(doc_id, page_index, chunk_index)
        content_hash = _sha256(chunk["text"])

        meta = ChunkMeta(
            doc_id=doc_id,
            source_path=chunk["source"],
            chunk_id=chunk_id,
            chunk_index=chunk_index,
            page_index=page_index,
            page_label=None,
            text=chunk["text"],
            text_preview=_text_preview(chunk["text"]),
            content_hash=content_hash,
            created_at=now_iso,
        )

        points.append(
            QdrantPoint(
                id=point_id,
                vector=vector.tolist(),
                payload=meta,
            )
        )

    return points


# ---------------------------------------------------------------------------
# Qdrant upsert
# ---------------------------------------------------------------------------


def upsert_to_qdrant(points: List[QdrantPoint]) -> int:
    """
    Connect to Qdrant, ensure collection exists, and upsert all points.
    Returns the total point count in the collection after upsert.
    """
    host = os.getenv("QDRANT_HOST", "localhost")
    port = int(os.getenv("QDRANT_PORT", "6333"))

    print(f"Connecting to Qdrant at {host}:{port} ...")
    client = QdrantClient(host=host, port=port)

    # Determine vector size from the first point
    vector_size = len(points[0].vector) if points else 384  # MiniLM default

    # Create collection if it does not exist
    collections = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in collections:
        print(f"Creating collection '{COLLECTION_NAME}' (dim={vector_size}, COSINE) ...")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
    else:
        print(f"Collection '{COLLECTION_NAME}' already exists.")

    # Upsert in batches of 100
    batch_size = 100
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        qdrant_points = [
            PointStruct(
                id=p.id,
                vector=p.vector,
                payload=p.payload.model_dump(),
            )
            for p in batch
        ]
        client.upsert(collection_name=COLLECTION_NAME, points=qdrant_points)
        print(f"  Upserted batch {i // batch_size + 1} ({len(batch)} points)")

    # Get final count
    info = client.get_collection(collection_name=COLLECTION_NAME)
    return info.points_count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest documents into Qdrant for RAG."
    )
    parser.add_argument(
        "--path",
        type=str,
        default=DEFAULT_DATA_DIR,
        help=f"Path to data directory (default: {DEFAULT_DATA_DIR})",
    )
    args = parser.parse_args()
    data_dir = args.path

    print("=" * 60)
    print("RAG Ingestion Pipeline")
    print("=" * 60)

    # 1. Load documents
    entries, doc_count = load_documents(data_dir)
    if not entries:
        print("No document content found. Exiting.")
        sys.exit(0)

    print(f"Loaded {len(entries)} page/section(s) from {doc_count} document(s).")

    # 2. Chunk
    chunks = chunk_entries(entries)
    print(f"Produced {len(chunks)} chunk(s) after splitting and merging.")

    # 3. Load embedding model
    model_path = os.getenv("MODEL_PATH")
    if model_path and os.path.isdir(model_path):
        print(f"Loading embedding model from MODEL_PATH: {model_path}")
        model = SentenceTransformer(model_path)
    else:
        print(f"Loading embedding model: {MODEL_NAME}")
        model = SentenceTransformer(MODEL_NAME)

    # 4. Build QdrantPoint objects (embed + metadata)
    points = build_points(chunks, model)
    print(f"Built {len(points)} QdrantPoint(s).")

    # 5. Upsert to Qdrant
    total_in_collection = upsert_to_qdrant(points)

    # 6. Summary
    print()
    print("=" * 60)
    print("Ingestion Summary")
    print("=" * 60)
    print(f"  Documents processed : {doc_count}")
    print(f"  Chunks created      : {len(chunks)}")
    print(f"  Qdrant point count  : {total_in_collection}")
    print("=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
