# Threat Model

## Assets
- User uploaded documents (potentially private/sensitive).
- API Keys (LLM provider, Vector DB).

## Threats
- Data Leakage: Private docs appearing in logs or error messages.
- Injection: Prompt injection via user queries.
- DoS: Large file uploads crashing the ingest worker.

## Mitigations
- [ ] Environment variable strictness.
- [ ] Input validation for file types and sizes.
