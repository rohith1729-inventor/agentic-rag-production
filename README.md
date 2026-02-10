# RAG System

A production-ready Retrieval Augmented Generation system that ingests documents (PDF and Markdown), indexes them as vector embeddings in Qdrant, and answers user questions with grounded, cited responses synthesized by a Groq-hosted LLM. The system enforces strict answer grounding -- every response is classified as COMPLETE, PARTIAL, or NOT_FOUND based on the retrieved context, ensuring no hallucinated answers.

## Architecture

### Ingest Pipeline

Documents are loaded from the `data/` directory, text is extracted (with page boundary preservation for PDFs), split into overlapping chunks (~500 characters), embedded using a local sentence-transformers model (all-MiniLM-L6-v2), and upserted into a Qdrant vector database with full metadata (doc ID, source path, chunk ID, page index, text preview).

### Query Pipeline

A user question is embedded with the same model, the top-K most similar chunks are retrieved from Qdrant, FlashRank reranks the candidates for relevance, contexts are trimmed to fit within the token budget, and Groq LLM synthesizes a grounded answer with citations extracted from the source chunks.

## Quick Start

```bash
# 1. Set up environment
cp .env.example .env  # Add your GROQ_API_KEY

# 2. Start services
docker compose up -d

# 3. Ingest documents
python ingest.py --path data

# 4. Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the RAG system architecture?"}'
```

## API Endpoints

### POST /query

Submit a question to the RAG system.

**Request body:**

```json
{
  "question": "What embedding model is used?",
  "top_k": 20,
  "top_n": 5,
  "token_budget": 6000,
  "retrieval_profile": null,
  "prompt_id": "default"
}
```

`retrieval_profile` and `prompt_id` are optional. Omit them to use default behavior.

**Response:**

```json
{
  "answer_type": "COMPLETE",
  "answer": "The system uses sentence-transformers (all-MiniLM-L6-v2).",
  "citations": [
    {
      "source_path": "data/sample.md",
      "chunk_id": "...",
      "page_index": null,
      "page_label": null
    }
  ],
  "timings_ms": {
    "embed": 12,
    "retrieve": 8,
    "rerank": 15,
    "llm": 320,
    "total": 355
  }
}
```

### GET /health

Liveness probe. Returns `{"status": "ok"}`.

### GET /ready

Readiness probe. Checks Qdrant connectivity. Returns `{"status": "ready"}` or `{"status": "not_ready", "error": "..."}`.

## Evaluation

Run the evaluation script to test the system against 20 predefined questions (mix of answerable and out-of-scope):

```bash
python eval.py
```

Optionally specify a different API URL:

```bash
python eval.py --url http://localhost:8000
```

The report is saved to `reports/eval_report.md` and includes per-question results, answer type distribution, citation coverage, and average latency.

## Configuration

All configuration is centralized in `rag_core/config.py` via Pydantic Settings, loaded from environment variables or `.env`.

| Variable | Description | Default |
|---|---|---|
| `GROQ_API_KEY` | API key for Groq LLM service | *(required)* |
| `QDRANT_HOST` | Hostname of the Qdrant instance | `localhost` |
| `QDRANT_PORT` | Port of the Qdrant instance | `6333` |
| `MODEL_PATH` | Path to the local embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| `GROQ_MODEL` | Groq model identifier | `llama-3.1-8b-instant` |
| `COLLECTION_NAME` | Qdrant collection name | `knowledge_base` |
| `CHUNK_SIZE` | Chunk size for text splitting | `500` |
| `CHUNK_OVERLAP` | Overlap between chunks | `50` |
| `PROMPTS_DIR` | Directory containing prompt templates | `prompts` |

## Extension Points

### Adding a New Ingestion Adapter

1. Create a new file in `rag_core/ingest/`, e.g. `rag_core/ingest/s3.py`.
2. Subclass `IngestionAdapter` and implement `load()`:

```python
from rag_core.ingest.base import IngestionAdapter

class S3Adapter(IngestionAdapter):
    def load(self, source: str):
        # source = s3://bucket/prefix
        # Return (page_entries, doc_count)
        # Each entry: {"source": ..., "text": ..., "page_index": ..., "source_type": "s3"}
        ...
```

3. Register it in `rag_core/ingest/__init__.py`.
4. Use it in `ingest.py` by swapping `FilesystemAdapter()` for your adapter.

### Adding a New Retrieval Profile

1. Open `rag_core/retrieval/strategy.py`.
2. Add a new entry to the `PROFILES` dict:

```python
PROFILES["fast"] = RetrievalProfile(
    strategy="vector_only",
    rerank=False,
    top_k=10,
    top_n=3,
)
```

3. Use it in queries: `{"question": "...", "retrieval_profile": "fast"}`.

### Adding a New Prompt Template

1. Create a file in the `prompts/` directory, e.g. `prompts/legal.md`.
2. Write the system prompt content (grounding rules, output format, etc.).
3. Use it in queries: `{"question": "...", "prompt_id": "legal"}`.

The prompt registry (`rag_core/prompts/registry.py`) automatically resolves `prompt_id` to `prompts/{prompt_id}.md`.

## Project Structure

```
rag-project/
├── shared/schema.py          # Interface contract (single source of truth)
├── rag_core/                  # Universal RAG core (domain-agnostic)
│   ├── config.py              # Unified Pydantic Settings configuration
│   ├── ingest/                # Ingestion adapters
│   │   ├── base.py            # Abstract IngestionAdapter interface
│   │   └── filesystem.py      # FilesystemAdapter (PDF, MD, TXT)
│   ├── retrieval/             # Retrieval strategies
│   │   └── strategy.py        # RetrievalProfile + retrieve/rerank/budget logic
│   ├── generation/            # LLM generation
│   │   └── llm.py             # Groq call wrapper (accepts prompt text)
│   └── prompts/               # Prompt registry
│       └── registry.py        # prompt_id -> file loader
├── prompts/                   # Prompt templates
│   └── default.md             # Default grounding prompt
├── main.py                    # FastAPI wiring (thin layer)
├── ingest.py                  # CLI ingestion pipeline
├── ops.py                     # Utilities (health, logging, retry)
├── eval.py                    # Evaluation harness
└── docker-compose.yml         # Qdrant + API services
```

## Technology Stack

- **Language:** Python 3.11+
- **API Framework:** FastAPI with async support
- **Embeddings:** sentence-transformers (all-MiniLM-L6-v2), local inference
- **Vector Database:** Qdrant (Docker container)
- **Reranking:** FlashRank (local, no external API)
- **LLM:** Groq (fast inference via LPU)
- **HTTP Client:** httpx
- **Retry Logic:** tenacity (exponential backoff)
- **Containerization:** Docker Compose for local development
- **Schema Validation:** Pydantic v2
