# Roles & Operating Model

## Roster
* **Owner:** Anna (Strategic decision maker)
* **Manager:** Thammudu (Me) / OpenClaw (Planner, Reviewer, Prompter)
* **Executor:** Claude Code (Implementation, Coding, CLI operation)

## Workflow Loop
1. **Approval:** Owner (Anna) approves the current phase or task.
2. **Plan:** Manager (Thammudu) writes the specific steps into memory/tasks.md.
3. **Execute:** Executor (Claude Code) implements one task from the list.
4. **Report:** Manager reviews and reports proof (files changed, commands run, test results).

## Communication Format
When completing a task, use this template:
**Task [ID] Complete**
* **Summary:** [Brief description of what was done]
* **Files Changed:** [List of files]
* **Commands Run:** [e.g., `pytest tests/test_ingest.py`]
* **Result:** [Output snippet or Success/Fail status]
* **Next Task:** [Suggestion for the next step]
