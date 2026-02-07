# Project Requirements

## Problem Statement
Build an end-to-end Retrieval Augmented Generation (RAG) system with a clean repository structure, strict rules, test gates, and production-ready basics. The system must ingest documents, index them, and answer queries with citations.

## Users and Use Cases
* **User:** Anna (Technical Owner)
* **Use Case:** Upload technical documents (PDF, Markdown, etc.) and ask questions to retrieve accurate answers with source tracing.

## Supported Document Types (Options - To Be Selected)
* [ ] PDF
* [ ] DOCX
* [ ] TXT
* [ ] HTML
* [ ] Markdown

## Goals
* **Retrieval Quality:** High precision (answers must be contained in retrieved chunks).
* **Latency:** < 200ms for retrieval (excluding LLM generation time).
* **Cost:** Minimize token usage and infrastructure costs.
* **Security:** No data leaks; strict environment variable handling.

## Constraints
* Deployable via Docker.
* Must run locally for development.
* Strict separation of concerns (Ingest vs. Query).

## Definition of Done (DoD)
1. [ ] Ingestion pipeline works for selected file types.
2. [ ] Query endpoint returns answer + correct citations.
3. [ ] Evaluation script exists and produces metrics.
4. [ ] All tests pass (Unit + Integration).
5. [ ] README is complete and runnable.
6. [ ] Minimal observability (logging/tracing) is implemented.
