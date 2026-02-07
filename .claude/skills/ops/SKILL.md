# Skill: ops

Purpose:
Add production gating and observability.

Commands:
1) Add /health and /ready endpoints
2) Add Groq retries/backoff
3) Add logging: embed, retrieve, rerank, llm, total timings

Checklist:
- /health returns ok if process alive
- /ready validates Qdrant connectivity
- Logs show latency split
