# DUAT Wiki — Schema & Maintainer Conventions

> LLM-maintained knowledge base for Project Duat (Karpathy llm-wiki pattern).
> The LLM owns `wiki/` entirely. Raw sources live in `dev-docs/` (immutable,
> never quoted verbatim into git history beyond what is needed) and the live
> World Labs docs on the web. This schema tells future agents how to maintain
> the wiki.

## Layers

1. **Raw sources (immutable):** `dev-docs/*.md|*.py|*.tsx`, `.env` (secret —
   never copy the key into wiki), https://atlas-beta.worldlabs.ai/docs/*.
   Never edit raw sources from wiki workflows.
2. **Wiki (`wiki/` — LLM-owned):** interlinked markdown pages listed in
   `index.md`. One page per entity/concept/source-sync. Chronology in `log.md`.
3. **Schema (this file):** structure, conventions, workflows.

## Conventions

- Every page: `# Title`, one-line summary, body, `## Sources` + `## Links`
  (wikilinks to related pages).
- Wikilinks style: `[Epistemic Stack](epistemic-stack.md)`.
- `index.md` is content-oriented (catalog). `log.md` is chronological,
  entries prefixed `## [YYYY-MM-DD] type | title`.
- Code references use `path:line` patterns, never pasted secrets.
- Anti-vividness-bias rule: every claim about Atlas behavior cites the live
  docs page; every deviation between the original brief and the live API is
  flagged in [Atlas Compliance](atlas-compliance.md).
- Never create `wiki/dev-docs*`, never commit `dev-docs/` to git
  (see `.gitignore`). The wiki *synthesizes* dev-docs, it does not mirror them.

## Workflows

- **Ingest:** read source → discuss takeaways → write/update pages →
  update `index.md` → append `log.md`. One source may touch many pages.
- **Query:** read `index.md` → drill into pages → synthesize answer with
  citations → file valuable answers back as new pages.
- **Lint:** check contradictions, stale claims vs live Atlas docs, orphans,
  missing cross-refs, gaps. Suggest new sources/questions.

## Frontend / backend source of truth

- Backend: `apps/api/` (FastAPI + pipeline). Frontend: `apps/web/`.
- `dev-docs/` drafts are *inputs*, not outputs — production code lives in
  `apps/` and may deliberately diverge where drafts contradict live docs.
