# Skill: ingest

Purpose:
Build and verify the ingestion pipeline end to end.

Inputs:
- data/ directory contents (PDF and md)
- Qdrant running via docker-compose

Commands:
1) docker compose up -d qdrant
2) python ingest.py --path data
3) verify qdrant point count increased

Checklist:
- PDF extraction captures page_index (0-based)
- Payload matches shared/schema.py ChunkMeta
- Upsert goes to collection knowledge_base
- Smoke run exits 0
