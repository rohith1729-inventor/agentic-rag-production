# RAG System Architecture

## Overview

A Retrieval Augmented Generation (RAG) system combines information retrieval with language model generation to produce accurate, grounded answers. The system ingests documents, splits them into chunks, generates embeddings, and stores them in a vector database.

## Components

### Ingestion Pipeline

The ingestion pipeline processes documents through several stages:

1. **Document Loading**: Reads PDF and Markdown files from the data directory.
2. **Text Extraction**: Extracts raw text content, preserving page boundaries for PDFs.
3. **Chunking**: Splits text into overlapping chunks of approximately 500 characters.
4. **Embedding**: Generates dense vector representations using sentence-transformers (all-MiniLM-L6-v2).
5. **Indexing**: Upserts vectors with metadata into Qdrant vector database.

### Query Pipeline

The query pipeline handles user questions:

1. **Embedding**: The user question is embedded using the same model.
2. **Retrieval**: Top-K similar chunks are retrieved from Qdrant.
3. **Reranking**: FlashRank reranks the candidates for relevance.
4. **Token Budget**: Contexts are trimmed to fit within the token budget.
5. **Synthesis**: Groq LLM generates an answer grounded in the context.
6. **Citations**: Source references are extracted and returned.

## Vector Database

Qdrant is used as the vector database. It stores embeddings with associated metadata (ChunkMeta) including:
- doc_id: Stable document identifier
- source_path: Original file path
- chunk_id: Unique chunk identifier
- page_index: Physical 0-based page index for PDFs
- text: Full chunk text
- text_preview: Short preview for debugging

## Answer Types

The system classifies answers into three categories:
- **COMPLETE**: The context fully supports the answer.
- **PARTIAL**: The context contains relevant but incomplete information.
- **NOT_FOUND**: The context is irrelevant or empty.

## Technology Stack

- **Language**: Python 3.11+
- **Framework**: FastAPI with async support
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2), local inference
- **Vector DB**: Qdrant (Docker container)
- **Reranking**: FlashRank (local, no API needed)
- **LLM**: Groq (fast inference via LPU)
- **Container**: Docker Compose for local development
