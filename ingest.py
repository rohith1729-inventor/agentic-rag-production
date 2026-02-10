"""
Ingestion pipeline for RAG project.

Loads documents via pluggable adapters, chunks them, generates embeddings,
and upserts into Qdrant.

Usage:
    python ingest.py                  # uses default "data" directory
    python ingest.py --path ./docs    # uses specified directory
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from rag_core.config import get_config
from rag_core.ingest import FilesystemAdapter
from shared.schema import ChunkMeta, QdrantPoint

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
            current = current + "\n" + chunk
        elif len(current) < min_size:
            current = current + "\n" + chunk
        else:
            merged.append(current)
            current = chunk

    if current:
        merged.append(current)

    return merged


def chunk_entries(entries: List[Dict]) -> List[Dict]:
    """
    Split each page entry into overlapping chunks using RecursiveCharacterTextSplitter.
    Returns a flat list of dicts with keys:
        source, page_index, chunk_index, text, source_type
    """
    cfg = get_config()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )

    source_entries: Dict[str, List[Dict]] = {}
    for entry in entries:
        source_entries.setdefault(entry["source"], []).append(entry)

    all_chunks: List[Dict] = []

    for source, page_entries in source_entries.items():
        chunk_counter = 0
        for entry in page_entries:
            raw_chunks = splitter.split_text(entry["text"])
            merged_chunks = _merge_small_chunks(raw_chunks, cfg.min_chunk_size)
            for chunk_text in merged_chunks:
                all_chunks.append({
                    "source": source,
                    "page_index": entry["page_index"],
                    "chunk_index": chunk_counter,
                    "text": chunk_text,
                    "source_type": entry.get("source_type"),
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
    """Generate embeddings and construct QdrantPoint objects for each chunk."""
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
            source_type=chunk.get("source_type"),
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
    cfg = get_config()

    print(f"Connecting to Qdrant at {cfg.qdrant_host}:{cfg.qdrant_port} ...")
    client = QdrantClient(host=cfg.qdrant_host, port=cfg.qdrant_port)

    vector_size = len(points[0].vector) if points else 384

    collections = [c.name for c in client.get_collections().collections]
    if cfg.collection_name not in collections:
        print(f"Creating collection '{cfg.collection_name}' (dim={vector_size}, COSINE) ...")
        client.create_collection(
            collection_name=cfg.collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
    else:
        print(f"Collection '{cfg.collection_name}' already exists.")

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
        client.upsert(collection_name=cfg.collection_name, points=qdrant_points)
        print(f"  Upserted batch {i // batch_size + 1} ({len(batch)} points)")

    info = client.get_collection(collection_name=cfg.collection_name)
    return info.points_count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    cfg = get_config()

    parser = argparse.ArgumentParser(
        description="Ingest documents into Qdrant for RAG."
    )
    parser.add_argument(
        "--path",
        type=str,
        default=cfg.default_data_dir,
        help=f"Path to data directory (default: {cfg.default_data_dir})",
    )
    args = parser.parse_args()
    data_dir = args.path

    print("=" * 60)
    print("RAG Ingestion Pipeline")
    print("=" * 60)

    # 1. Load documents via adapter
    adapter = FilesystemAdapter()
    entries, doc_count = adapter.load(data_dir)
    if not entries:
        print("No document content found. Exiting.")
        sys.exit(0)

    print(f"Loaded {len(entries)} page/section(s) from {doc_count} document(s).")

    # 2. Chunk
    chunks = chunk_entries(entries)
    print(f"Produced {len(chunks)} chunk(s) after splitting and merging.")

    # 3. Load embedding model
    model_path = cfg.model_path
    if os.path.isdir(model_path):
        print(f"Loading embedding model from path: {model_path}")
    else:
        print(f"Loading embedding model: {model_path}")
    model = SentenceTransformer(model_path)

    # 4. Build QdrantPoint objects
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
