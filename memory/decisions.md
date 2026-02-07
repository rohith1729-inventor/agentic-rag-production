# Architectural Decisions Log

| Date | Decision | Options Considered | Reason | Impact |
|------|----------|-------------------|--------|--------|
| 2026-02-05 | **Language:** Python 3.11+ | Python, TypeScript | Ecosystem maturity for AI/ML libraries. | Standard Python tooling required. |
| 2026-02-05 | **Framework:** FastAPI | Flask, Django | High performance, async support, auto-docs. | Async patterns required in code. |
| 2026-02-05 | **Vector DB:** Qdrant | Chroma, PGVector | Production-grade, fast, Rust-based, Docker-ready. | Requires Qdrant container in compose. |
| 2026-02-05 | **Embeddings:** Local HF (all-MiniLM-L6-v2) | OpenAI, Cohere | Privacy, zero cost, low latency, no API dependency. | Increased container image size (model baked in). |
| 2026-02-05 | **Reranking:** FlashRank (Local) | Cohere, Cross-Encoder | Privacy, zero cost, no API latency. | Removed Cohere dependency. |
| 2026-02-05 | **LLM:** Groq | OpenAI, Anthropic | Extreme speed (LPU), low latency focus. | Requires Groq API Key. |
