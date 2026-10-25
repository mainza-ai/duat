# MILIMO DUAT — Production Audit Report & Implementation Plan

**Date:** 2026-10-25 · **Auditor:** Staff Full-Stack Graphics & AI Systems Architect (automated deep dive)
**Method:** headless-Chrome visual verification (lens / helical / NBV screenshots + console-error capture),
`pytest` (13 passed), `tsc --noEmit` (clean), production `vite build` + static serve, live-endpoint probing.
**Standard:** production deployment. No demo shortcuts — every simplification listed below must be
resolved properly, not worked around.

---

## 1. Defects found & fixed in this session (verified visually)

| # | Defect | Root cause | Fix | Proof |
|---|--------|-----------|-----|-------|
| 1 | 3D view "broken" (empty-looking) | Backend Z-up trajectory rendered raw in Y-up scene: frustums/points ~40 m off-axis, below mesh | `GET /trajectory` gains `z_start/z_end/orbit_radius` + frame metadata; frontend maps via rigid R_x(−90°) (quaternion left-multiplied) into mesh bounds | All 32 mapped poses orbit r≈2.4 inside shaft; screenshots show frustums + wall-hugging cloud |
| 2 | "Jump to NBV" didn't move the camera | Button only re-selected index; orbit target snapped, external camera never flew | Fly-to rig: chase pose = NBV + outward×12/up×6, damped lerp of position + controls target; user drag cancels; button shows flight state | NBV screenshot: view glides to NBV frustum |
| 3 | Point cloud was fake geometry | Synthetic spiral pattern, mismatched with active camera | New `POST /api/khafre/depth` (scaled intrinsics, 0.94 valid fraction); points unprojected from MEASURED depth; colors derived (hillshade / dielectric / iso-depth / occlusion proxy), honestly labeled | `PTS:MEASURED` badge in HUD; wall-hugging cloud in screenshots |
| 4 | Lens quad small / letterboxed | Fixed zoom 45 + flex-height canvas + stretched column | `ZoomFit` (zoom = min(w/16, h/9)) + exact 16:9 canvas | Fills viewport width, zero internal bands |
| 5 | Tab switch could blank the app | R3F root unmount race (dev StrictMode double-mount; one non-fatal portal race remains in prod) | Both Canvases stay mounted (display toggle) + `frameloop never` when hidden + `ViewportErrorBoundary` per tab | Prod build: app renders through all tabs |
| 6 | Thematic violet + "Quantum" naming | Generic-AI aesthetic | Signal orange `#FF6B1A`, "Variance Shimmer", honest labels everywhere | Repo-wide sweep: zero matches |

---

## 2. Remaining gaps — P0 (production blockers)

- **P0-1 · Live Atlas path is unverified and unguarded.** No live `atlasChisel` call has ever been
  made; `_decode_live_operation_result`, `:wait` fallback, and polling exist but have zero tests.
  Worse, omitting `mock_mode` silently spends credits (`ATLAS_CLIENT` is live when a key exists).
  **Plan:** explicit `mode: "mock" | "live"` (default `mock`); live requires key + explicit opt-in;
  add `respx`-mocked transport tests simulating submit → running → done, failure, and timeout;
  make one real 1-target live call (1 credit-scale) and record the response shape; add cost logging.
- **P0-2 · No background jobs.** Live Atlas ops run minutes; ensemble holds HTTP workers; no timeout
  UX. **Plan:** job queue (FastAPI `BackgroundTasks` → DB-backed jobs table as v1; Redis/Celery if
  scale demands), `POST /jobs`, `GET /jobs/{id}` with progress, frontend polling + progress UI.
- **P0-3 · No persistence.** `PROVENANCE_STORE` is in-memory; `data/` cache unused; manifests die
  with the process. **Plan:** SQLite (SQLModel) for manifests/jobs/ensembles + content-addressed
  artifact dir `data/sha256/...` (PNG/EXR/GLB/manifest.json); `GET /api/provenance/{sha}` re-serves.
- **P0-4 · No API hardening.** CORS `*`, no auth, no rate limits, unbounded uploads, temp files.
  **Plan:** API-key middleware (operator key, separate from WLT key), per-IP rate limit on ensemble/
  upload routes, 25 MB upload cap + magic-byte validation, `CORS` allow-list from env.
- **P0-5 · Chained descent is a 2-hop demo.** `chain_descent` supports N batches but the endpoint
  hardcodes 2 hops on a 4-cam mini-trajectory and never uses returned depth. **Plan:** full-batch
  chaining job (32 cams/batch, `returnDepth:true` with ≥2 targets so GEOMETRY propagates, not just
  RGB), persisted per-hop artifacts, resume-from-hop.
- **P0-6 · `returnDepth` never used.** Contradicts the chained-descent design (geometry reuse).
  Covered by P0-5 implementation.
- **P0-7 · Unpinned dependencies, no CI.** `requirements.txt`/`package.json` float; no GitHub
  Actions. **Plan:** pin (`pip-compile` + lockfile — frontend already has one), CI running
  `pytest`, `tsc`, `vite build`, and the headless visual suite (this audit's `scripts/` harness).
- **P0-8 · Production container.** `docker-compose.web` runs a dev server. **Plan:** multi-stage
  Dockerfile (build → nginx static), pinned base images, `VITE_API_URL` build arg, backend gunicorn/
  uvicorn workers + env config; add `.env.example` (never commit keys).
- **P0-9 · Startup cost.** Full 160×160×240 synthesis on boot (~50 MB+ arrays, seconds of import
  time). **Plan:** lazy volume init with readiness endpoint, smaller default grid + on-demand
  full-res, startup timing log.

## 3. Remaining gaps — P1 (important)

- **P1-1 · `Html` portal `removeChild` race** (once per session, non-fatal, contained by boundary).
  Replace 3D-anchored labels with canvas-space sprites or a single fixed HUD chip.
- **P1-2 · Promote the visual harness to a Playwright suite** (`scripts/visual-verify.mjs` seeds it):
  console-error-free load, both tabs render, NBV flight arrives, ensemble round-trip, export download.
- **P1-3 · Export card uses defaults**, not live viewport state (hypothesis/weights/voxel/σ).
  Pass current deck state into the export request.
- **P1-4 · Bundle un-split** (1.17 MB, three.js + R3F in one chunk). `manualChunks` + lazy tab imports.
- **P1-5 · Defaults are demo-tuned:** `posterior_resample=false`, single-geometry ensembles.
  Measure render cost; if <2 s, default resampling ON for honest variance.
- **P1-6 · Upload endpoint hardening** (magic bytes already extension-checked; add size caps — see P0-4).
- **P1-7 · Observability:** request IDs, structured logs, `/api/health` depth checks (disk, memory,
  Atlas reachability already reported — extend).
- **P1-8 · `voxelSize` validation** against the 0–0.4 Atlas guidance (currently 0–2).

## 4. Enhancements — P2

Mobile/responsive pass · keyboard/a11y nav · PWA offline · telemetry single-source-of-truth
(shader/TS duplication) · right-deck internal scroll · tutorial overlay · multi-language prompts.

## 5. Atlas live-rendering utilization (how to actually use it)

The backend boots **live-capable** when `WLT_API_KEY` is set (`worldlabs_mock_mode:false` —
the header badge proves it). Every UI path today forces `mock_mode:true` (free simulation) by
design, so **no path currently spends credits**. To render for real:

```bash
# 1. Single live view, 1 target (cheapest probe; depth NOT returned — needs ≥2 targets)
curl -s -X POST localhost:8001/api/chisel/ensemble \
  -H 'Content-Type: application/json' \
  -d '{"camera":{"pose":{"position":[-38.5,-20.0,-45.0],"quaternion":[0,0,0,1]}},
       "hypothesis":"megalithic","instrument_rig":"drop_flare",
       "seeds":[7],"mock_mode":false}' | python3 -c "import json,sys; print(json.load(sys.stdin)['provenance_sha256'])"

# 2. Live + returned depth (≥2 DISTINCT target centers required by Atlas, else scale 0/NaN)
#    Use two cameras with translation baseline; add "returnDepth": true via modelParameters
#    (expose as API flag — see P0-5).

# 3. Chained descent: take response.frames[i].imageAsset + camera from batch k and pass
#    as sourceFrames/contextFrames of batch k+1; reuse promptUsed with enhancePrompt:false
#    to skip recaptioning (latency win documented in Atlas guides).
```

Rules that govern cost and correctness (live docs, encoded in `wiki/atlas-compliance.md`):
context = 3 rollout views, photo/target = 1 each, max 128 (32 targets alone fill it);
inverted log-depth (`255` = `zMin`, `0` = empty); `WLT-Api-Key` header; poll `GET /operations/{id}`;
reuse `promptUsed`; never send `depth.confidenceAsset`.

## 6. Suggested build order

P0-7 (CI) → P0-1 (live safety + one real call) → P0-3 (persistence) → P0-2 (jobs) → P0-5/P0-6
(chaining with depth) → P0-4 (hardening) → P0-8/P0-9 (deploy) → P1 batch → P2.
