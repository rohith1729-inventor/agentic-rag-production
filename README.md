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
  "token_budget": 6000
}
```

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

| Variable | Description | Default |
|---|---|---|
| `GROQ_API_KEY` | API key for Groq LLM service | *(required)* |
| `QDRANT_HOST` | Hostname of the Qdrant instance | `localhost` |
| `QDRANT_PORT` | Port of the Qdrant instance | `6333` |
| `MODEL_PATH` | Path to the local embedding model | `./models/all-MiniLM-L6-v2` |

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
