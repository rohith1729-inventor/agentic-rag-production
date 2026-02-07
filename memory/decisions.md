# Architectural Decisions Log

| Date | Decision | Options Considered | Reason | Impact |
|------|----------|-------------------|--------|--------|
| 2026-02-05 | **Language:** Python 3.11+ | Python, TypeScript | Ecosystem maturity for AI/ML libraries. | Standard Python tooling required. |
| 2026-02-05 | **Framework:** FastAPI | Flask, Django | High performance, async support, auto-docs. | Async patterns required in code. |
| 2026-02-05 | **Vector DB:** Qdrant | Chroma, PGVector | Production-grade, fast, Rust-based, Docker-ready. | Requires Qdrant container in compose. |
| 2026-02-05 | **Embeddings:** Local HF (all-MiniLM-L6-v2) | OpenAI, Cohere | Privacy, zero cost, low latency, no API dependency. | Increased container image size (model baked in). |
| 2026-02-05 | **Reranking:** FlashRank (Local) | Cohere, Cross-Encoder | Privacy, zero cost, no API latency. | Removed Cohere dependency. |
| 2026-02-05 | **LLM:** Groq | OpenAI, Anthropic | Extreme speed (LPU), low latency focus. | Requires Groq API Key. |
| 2026-02-06 | **Schema Contract:** shared/schema.py | Inline models, separate per-module | Single source of truth prevents drift. | All modules import from shared.schema. |
| 2026-02-06 | **Token Budget:** len/4 estimate | tiktoken, regex | Zero-dep, fast, good-enough for budget gating. | Slightly inaccurate but deterministic. |
| 2026-02-06 | **Idempotent IDs:** UUID5 from (doc_id, page, chunk) | Random UUID, sequential | Enables re-ingestion without duplicates. | Stable across runs. |
| 2026-02-06 | **Retry Strategy:** tenacity 3x exponential | Manual retry, no retry | Handles Groq 429/5xx gracefully. | Added tenacity dependency. |
| 2026-02-06 | **PDF Text:** pypdf | pdfplumber, pdfminer | Lightweight, maintained, sufficient for text. | Added pypdf dependency. |
| 2026-02-07 | **Qdrant Client Pin:** >=1.7.3,<1.12 | Unpinned (latest), upgrade server | Client 1.16.2 removed `.search()` method; server 1.7.4 lacks `query_points` endpoint. | Pin ensures API compatibility with server v1.7.4. |
| 2026-02-07 | **Full-prompt token budget** | Context-only budget | System+question+formatting overhead must be subtracted before allocating context tokens. | Prevents over-budget Groq requests; logs reductions. |
