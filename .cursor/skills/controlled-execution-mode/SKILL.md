---
name: controlled-execution-mode
description: Enforces a discuss-before-implement workflow for code changes, architecture, refactoring, config, database migrations, or infrastructure. Use when the user requests code changes, architecture modifications, refactoring, configuration changes, database migration, or infrastructure updates. Agent must summarize, discuss, and wait for explicit execution trigger before implementing.
---

# Controlled Execution Mode

When operating in this mode, the agent **must not** implement changes until the user explicitly triggers execution.

## Trigger Scope

Apply this workflow when the user's request relates to:

- Code changes
- Architecture modification
- Refactoring
- Configuration changes
- Database migration
- Infrastructure updates

## Workflow (Mandatory)

### STEP 1 — Summarize

- Summarize the request in your own words.
- Identify scope of impact (files, modules, systems).
- Identify risks and side effects.
- Identify assumptions.

### STEP 2 — Discussion Phase

- Ask clarifying questions if needed.
- Propose alternative approaches if relevant.
- **Do NOT** modify or generate final production code.
- **Do NOT** output full implementation.
- You may show small illustrative snippets **only** if needed for discussion.

### STEP 3 — Wait for Explicit Execution Trigger

Only proceed to actual implementation when the user explicitly says one of:

- "请执行"
- "执行"
- "apply"
- "implement"
- "请修改"
- "go ahead"

**Without explicit execution trigger:**

- Do NOT generate full code.
- Do NOT rewrite files.
- Do NOT output full patches.

### STEP 4 — Execution (After Trigger)

Once execution is triggered:

- Provide complete implementation.
- Include full code blocks.
- Include migration steps if needed.
- Include rollback considerations.
- Include verification steps.

## Quick Reference

| Phase   | Action                                             | Forbidden                    |
|---------|----------------------------------------------------|-----------------------------|
| Step 1  | Summarize, scope, risks, assumptions               | —                            |
| Step 2  | Discuss, ask questions, show tiny snippets         | Full code, file rewrites     |
| Step 3  | Wait for trigger phrase                           | Any implementation           |
| Step 4  | Implement fully with migrations, rollback, verify  | —                            |
