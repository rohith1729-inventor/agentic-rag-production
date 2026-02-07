# Changelog

## [0.1.0] - 2026-02-07

### Added
- **Ingestion pipeline** (`ingest.py`): PDF + Markdown + TXT extraction, chunking (~500 chars), sentence-transformers embeddings, Qdrant upsert with ChunkMeta payload, idempotent UUID5 point IDs.
- **Query engine** (`main.py`): FastAPI POST /query with embed -> retrieve -> FlashRank rerank -> token_budget enforcement -> Groq LLM synthesis -> citations + timings.
- **Ops** (`ops.py`): /health liveness probe, /ready readiness probe (Qdrant connectivity), Groq retry with tenacity (3x exponential backoff on 429/5xx), structured logging with latency split.
- **Evaluation** (`eval.py`): 20 test questions (12 answerable, 8 NOT_FOUND), report saved to reports/eval_report.md.
- **Interface contract** (`shared/schema.py`): ChunkMeta, QdrantPoint, QueryRequest, QueryResponse, Citation, ContextItem, TimingsMS, AnswerType.
- **Documentation**: Complete README with quick start, API endpoints, configuration, tech stack.
- **Infrastructure**: Updated Dockerfile, docker-compose.yml, requirements.txt, .env.example.
- **Skills**: ingest, query, ops, eval, release skill definitions.
- **Sample data**: data/sample.md for smoke testing.

### Fixed
- **qdrant-client version pin**: `>=1.7.3,<1.12` for server v1.7.4 compatibility.
- **Token policy enforcement**: Full prompt (system+question+contexts) budgeted before Groq call, with logging of reductions/truncations.

### Previous
- Initial repository setup with control files.
