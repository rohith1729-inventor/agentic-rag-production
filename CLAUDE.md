# RAG Project: Operating Contract (Persistent)

Read in order:
1) rules.md
2) requirements.md
3) role.md
4) prompt.md
5) memory/tasks.md

Hard rules:
- Never invent file paths, APIs, or configs. Verify by reading files or searching the repo.
- shared/schema.py is the single interface contract. Do not redefine schemas elsewhere.
- Only Team Lead edits files under memory/. Teammates must not touch memory/tasks.md.
- Answers must be grounded in retrieved context only.
- Use answer_type: COMPLETE, PARTIAL, NOT_FOUND.

RAG citations:
- Use page_index as truth for PDFs. page_index is physical 0-based index.
- page_label is optional best-effort only.

Token budget:
- Estimate tokens for contexts before calling Groq.
- Enforce token_budget by reducing top_n or truncating deterministically.

Workflow:
- Use Skills for repeatable workflows.
- Use Agent Teams only after the interface contract exists.
