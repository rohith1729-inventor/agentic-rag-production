# Execution Rules

## General
1. **Owner Protocol:** Always address the Owner as Anna.
2. **Phase Gating:** No feature coding is permitted until Phase 0 is fully complete and verified.
3. **Single Task Flow:** Execute one task at a time. Do not multitask.

## Workflow
4. **Proof of Work:** Every task must end with proof:
   * Commands run.
   * Test execution or smoke check.
   * Output logs.
5. **Dependency Control:** No new dependencies (pip/npm/maven) unless explicitly logged and justified in memory/decisions.md.
6. **Refactoring:** No refactoring is allowed during active feature work. Refactor only as a dedicated task.

## Security & Privacy
7. **Data Safety:** Never log secrets, API keys, or raw private document contents to the console or git.

## Stop Conditions (Ask Anna)
8. **Stop immediately and ask for guidance if:**
   * Requirements are ambiguous or unclear.
   * Multiple valid architecture options exist (requires a decision).
   * A choice involving payment or specific API keys is needed.
   * A test fails twice in a row (do not loop).
