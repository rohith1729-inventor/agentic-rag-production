from __future__ import annotations

from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field


class ChunkMeta(BaseModel):
    doc_id: str = Field(..., description="Stable document id derived from source_path or hash")
    source_path: str = Field(..., description="Original file path")
    chunk_id: str = Field(..., description="Unique chunk id")
    chunk_index: int = Field(..., ge=0)

    # PDF specific
    page_index: Optional[int] = Field(
        default=None,
        description="Physical page index, 0-based. Source of truth for citations."
    )
    page_label: Optional[str] = Field(
        default=None,
        description="Optional logical page label if detected."
    )

    text: str = Field(..., description="Chunk text stored for retrieval and synthesis")
    text_preview: str = Field(..., description="Short preview for debugging")
    content_hash: Optional[str] = Field(default=None, description="Optional hash for dedupe")
    created_at: Optional[str] = Field(default=None, description="ISO timestamp")


class QdrantPoint(BaseModel):
    id: str = Field(..., description="Point id used in Qdrant")
    vector: List[float] = Field(..., description="Embedding vector")
    payload: ChunkMeta


class QueryRequest(BaseModel):
    question: str
    top_k: int = 20
    top_n: int = 5
    token_budget: int = 6000


class Citation(BaseModel):
    source_path: str
    chunk_id: str
    page_index: Optional[int] = None
    page_label: Optional[str] = None


class ContextItem(BaseModel):
    chunk_id: str
    score: float
    source_path: str
    page_index: Optional[int] = None
    page_label: Optional[str] = None
    text_preview: str


class TimingsMS(BaseModel):
    embed: int
    retrieve: int
    rerank: int
    llm: int
    total: int


AnswerType = Literal["COMPLETE", "PARTIAL", "NOT_FOUND"]


class QueryResponse(BaseModel):
    answer_type: AnswerType
    answer: str
    citations: List[Citation]
    contexts: Optional[List[ContextItem]] = None
    timings_ms: TimingsMS
    debug: Optional[Dict[str, Any]] = None
