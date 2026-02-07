# Skill: query

Purpose:
Implement and verify the query API with retrieval, rerank, synthesis, citations.

Inputs:
- Qdrant populated collection knowledge_base
- Groq API key configured via .env

Commands:
1) docker compose up -d
2) run the FastAPI app
3) curl POST /query with a known question and verify citations
4) curl POST /query with an unknown question and verify NOT_FOUND

Checklist:
- Uses shared/schema.py QueryRequest and QueryResponse
- Enforces token_budget
- Returns answer_type COMPLETE|PARTIAL|NOT_FOUND
- Returns citations with source_path, chunk_id, page_index optional
