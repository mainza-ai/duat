# MILIMO DUAT — Production Implementation Plan (for review)

**Status:** draft for review · **Principle:** live Atlas is the product; mock is a thin offline
fallback, never a substitute. No demo shortcuts — every item below is build-once, production-grade.

Companion: `docs/AUDIT.md` (verified findings). Live API contract: `wiki/atlas-compliance.md`.

---

## 0. Terminology & cost model

- **Billable unit = one accepted task** (`POST /tasks:atlasChisel`, `images2PosedRGBD`, …).
  Asset/operation/poll calls are lifecycle traffic, not model work (no separate price published).
- **Price unknown to this repo:** the per-task credit rate card lives only in the platform
  (Settings → Billing) and can change during beta. 1,000 credits = US$1.
- **Cost formula:** `USD = (Σ tasks × rate_credits_per_task) / 1000`.
  Until the rate card is read, every phase below budgets **task counts**, not dollars.
  **Action for you:** open Settings → Billing, note the `atlasChisel` credit cost for a 1-target
  and a 32-target call, and fill the two blanks in §5. All estimates then resolve to dollars.
- **Guardrails (do these in the platform today, $0):** account-wide monthly USD cap +
  a tight `duat-dev` project cap; Athena: all estimates below assume a dev project cap of ~$25
  and a pilot cap of ~$150 until real burn is measured.

Key platform facts the plan relies on (verified in live docs):
- Submit once with a stable `Idempotency-Key`; reuse on transport retry; `409` on key+new-body.
- `?wait=true` risks LB `504` past 60 s → long work submits **without** wait, then `:wait`
  (`timeoutSecs`) + `GET /operations/{id}` backoff. `done` is the only terminal signal.
- `402 CREDITS_UNAVAILABLE / SPEND_LIMIT_EXCEEDED` → stop, top up/raise cap, retry with the
  **same** idempotency key. Never tight-loop. `429` → honor `Retry-After`, backoff + jitter.
- Rollout budget: context frame = 3 views, photo/target = 1, max 128 (32 targets fill it alone).
- Retention: operations/traces 30 days; outputs persist unless purged — our artifact store copies
  everything inside the window. `returnDepth` needs ≥2 targets with a translation baseline.

---

## Phase 0 — Safety & CI foundation (Atlas spend: $0)

1. **Explicit execution mode.** Replace `mock_mode: Optional[bool]` with `mode: "mock" | "live"`
   (default `mock`). Live requires a key present; without one the server refuses live with 409
   + clear message instead of silently simulating. UI toggle becomes three-state
   (Simulated / Live-1-target probe / Live full), default Simulated, with a credit warning.
2. **Deterministic idempotency.** Key = `duat-{sha256(canonical request)}` (we already hash —
   promote it to the `Idempotency-Key` header on every submit).
3. **Retry/backoff client.** httpx transport: capped exponential backoff + jitter on 429/5xx/504,
   honoring `Retry-After`; map 402/409/422 to typed 502s with operator action strings.
4. **Pins + CI.** `pip-compile` + lockfile; GitHub Actions: `pytest`, `tsc`, `vite build`,
   headless visual suite (`scripts/visual-verify.mjs`, zero-console-error gate).
5. **`.env.example`** (no secrets), dev/prod key guidance (one project per environment).

*Acceptance:* CI green; live paths unreachable in mock mode by construction; 402 simulation test.

## Phase 1 — Live verification probe (Atlas spend: ~2–5 tasks)

1. `respx` transport tests simulating submit → running → done / failed / timeout / 429 / 402.
2. **One real 1-target live call** (single shaft frame, `mock_mode:false`), record request/response
   shape; assert decode path (`frames[].imageAsset`, `camera`, optional `depth`).
3. Confirm actual credit deduction on the Usage page; fill the §5 rate blanks with real numbers.
4. Contract test in CI using the recording (never hits network).

*Acceptance:* live decode path covered; real per-task cost known; spend ≤ 5 tasks.

## Phase 2 — Persistence + jobs (Atlas spend: $0)

1. SQLite (SQLModel): `jobs`, `operations`, `ensembles`, `artifacts` tables; content-addressed
   `data/sha256/...` store (PNG/EXR/GLB/manifest.json); `GET /api/provenance/{sha}` re-serves.
2. `POST /jobs` (ensemble/chain/descent) → `GET /jobs/{id}` progress; frontend progress UI;
   cancel endpoint wired to `operations:cancel`.
3. Startup: lazy volume init + readiness probe; keep `khafre_volume_loaded` semantics.

*Acceptance:* kill -9 mid-run loses nothing billable (op IDs persisted; resume polls, never resubmits).

## Phase 3 — Production chained descent (Atlas spend: pilot budget, §5)

1. Full-batch chaining: 32 cams/batch, `returnDepth:true` (≥2 targets, translation baseline) so
   **geometry propagates**, not just RGB; per-hop artifacts; resume-from-hop; NBV recompute/hop.
2. Posterior resampling on by default (measured <2 s/render); structural weight stays 0.3 pending
   A/B on real outputs.
3. Rollout-budget planner: given targets+photos, validate ≤128 views client-side before submit.

*Acceptance:* top-to-chamber descent runs unattended; every hop reproducible from stored manifest.

## Phase 4 — Live-first frontend (Atlas spend: ~5–10 probe tasks)

1. Mode-aware UI (Phase 0 control), live progress + cancel + cost preview
   (`tasks × last-known rate`, clearly labeled estimate).
2. Export card uses **current viewport state**; e2e Playwright suite from the visual harness.
3. `manualChunks` + lazy tabs (bundle 1.17 MB → budgeted <500 KB initial); loading skeletons;
   deck scroll; mobile pass; keyboard nav.
4. Replace 3D labels approach permanently (done: HUD chip over portals) — keep portal-free rule.

*Acceptance:* zero-console-error e2e on mock; one live 1-target e2e behind a flag.

## Phase 5 — Hardening & deploy (Atlas spend: $0)

Operator API keys, per-IP rate limits, 25 MB upload cap + magic bytes, CORS allow-list,
structured logs + request IDs, `/health` depth checks, pinned prod Docker (nginx + workers),
load-test against the TBA limits with a representative workload.

## Phase 6 — Pilot production run (Atlas spend: the budget below)

Full Khafre Shaft #2: 32-target descent × 4 seeds × 3 hypotheses + NBV physical-frame plan.
Count tasks first, multiply by your rate card, set the project cap, then run.

---

## 5. Spend estimate worksheet

| Workflow | Tasks counted | Notes |
|---|---|---|
| 1-target live probe | 1 | Phase 1 smoke test |
| Contract recording set | 2–4 | 1-target + 2-target (depth) + 1 failure-path |
| Ensemble, 1 frame (4 seeds) | 4 | one task per seed (current design) |
| Ensemble, 1 frame (4 seeds × 3 hypotheses) | 12 | full mixer comparison |
| Descent, 32 targets × 4 seeds | 4–128 | 1 task per request if batched by request; per-seed-per-batch if by seed |
| Chained descent, B batches | B × (above) | each hop billed fresh |
| Pilot (descent 4 seeds + mixer 3 hyp on 4 key frames) | ~28–40 | recommended first real run |

`USD = tasks × R / 1000`, where `R` = your platform rate card (credits per atlasChisel call
at the target count you use — **fill in: R₁ = ___, R₃₂ = ___**).
**Suggested caps:** dev project $25; pilot project $150; account ceiling per your comfort.
Re-estimate after Phase 1 measures real burn. Rate card can change in beta — re-check before
any large run; never embed prices in client code.

---

## 6. Open decisions for you

1. Approve Phase 0–1 as specified (includes 1 real billed probe call)?
2. Confirm dev/pilot spend caps to set in the platform.
3. Paste your `R₁` / `R₃₂` rate-card numbers so §5 resolves to dollars.
4. Merge `develop` → `main` now, or after Phase 0?
