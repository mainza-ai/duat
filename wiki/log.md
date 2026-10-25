## [2026-10-25] build | K1–K6 Khafre program implemented ($0 spend)

- Claim registry (7 records, retraction + outlet variance) + /api/claims;
  null prior everywhere; chamber-size ranges (80/90) with cache; mixer chips.
- Validated void benchmarks (NFC 9×2×2, Big Void 30×6) + endpoint; wiki rewritten;
  UNVERIFIED-CLAIM chip in UI; NBV callout driven by live responses.
- Geometry remodel: shared shaft_model (cylinder minus helical ramp solid),
  renderer matches by construction; 3D prop is plain cylinder + instanced steps;
  near/far 1/400, polygonOffset, display-range control, GLB posterior overlay.
- Suite 36 passed; visual gate green on prod build. Plan statuses updated in dev-docs.

## [2026-10-25] build | Phases 0–5 implemented under 1,000-credit cap (1 live task)

- Mode flag (mock/live/probe) + spend confirm, deterministic idempotency keys,
  retry/backoff + typed Atlas errors (402/409/422/429/504), pinned deps, CI, `.env.example`.
- ONE live 1-target probe fired (op `ae53860d…`); fixture recorded; respx suite green.
- SQLite store + artifacts + jobs API (deduplicating IDs, progress, cancel); lazy volume init.
- Full-trajectory chaining (1:1 context frames, promptUsed reuse, resume hints,
  rollout guard, returnDepth), resample default ON.
- Frontend: 3-mode renderer, cost preview, job progress + cancel, export-from-current-view,
  code-split, zero-console-error e2e gate on prod build.
- Hardening: operator key, rate limits, upload caps, CORS allow-list, request IDs,
  structured logs, prod Docker, mock-only load script. Pilot HELD (over budget).

## [2026-10-25] fix | 3D view broken — backend/scene frame mismatch

- Root cause: the navigator fetched live backend cameras (Z-up Khafre frame,
  shaft #2 at x=-40/y=-20, z down to -180) and rendered them raw in the
  Y-up demo scene (shaft at origin, mesh y in [-115,-5]) — frustums and
  point cloud landed ~40 m off-axis and below the mesh: empty-looking view.
- Fix: `GET /api/khafre/trajectory` accepts `z_start/z_end/orbit_radius`
  + returns `shaft_center_xy`/`coordinate_frame`; frontend maps cameras via
  the Z-up→Y-up rigid transform (R_x(-90°), quaternion left-multiplied) and
  requests z in [-15,-105] to fit the 110 m shaft prop. Verified all 32
  mapped poses orbit r≈2.4 inside the mesh. Suite 12 passed, tsc clean.

## [2026-10-25] audit | Full plan-vs-implementation audit + gap closure

- Gaps found: exporter was `alert()` stub; no voxel/σ sliders; no click
  inspector; no mesh→volume ingest; std-only disagreement (wiki claimed
  SSIM hook); no degradation endpoint; single geometry sample per ensemble;
  torch path stub-only.
- Fixed: real PNG comparison-card exporter with baked-in provenance bar
  (`Export Card`); voxel + σ sliders wired to backend; click-to-pin
  inspector with simulated 3-hypothesis comparison; `POST /api/khafre/mesh`
  (mesh→`EpistemicVolumeBundle`, trimesh positive-inside sign corrected);
  `GET /api/khafre/degradation` ladder (RMS 0 → 0.014 → 0.58 → 2.72m);
  `posterior_resample` + `structural_weight` (SSIM blend) flags; defensive
  torch MPS/CUDA path; navigator prefers backend trajectory w/ fallback.
- Suite now **12 passed** (was 8); tsc clean; e2e verified through proxy.

## [2026-10-25] repo | Initialized git + pushed to mainza-ai/duat

- `git init -b main`, root commit `27b68bd` (42 files: backend, frontend,
  wiki, blueprint, orchestration). Remote `origin` via SSH (HTTPS helper
  broken locally); `git push -u origin main` succeeded.
- Excluded from history: `dev-docs/`, `.env` (+secrets), `.DS_Store`,
  caches. Grep-verified no live key material outside `.env`.

## [2026-10-25] verify | Backend suite green + lifespan migration

- `pytest apps/api/tests -q`: **8 passed** (depth math, inverted codec
  endpoints/round-trip/zero-mask, ensemble shapes, mock determinism, prompt
  composer, API contract incl. nested Atlas cameras + `coordinateSystem: rub`).
- Replaced deprecated `@app.on_event("startup")` with `lifespan` handler.
- Spot-checked official inverted codec: `z=0.5→255, z=40→1, z=5→121`;
  helical cameras emit `{intrinsics, extrinsics}`; prompt rigs map to
  correct `zMax` (borescope 6 / flare 40 / laser 25).
- Phases 5–9 closed: SAR synthesizer, lens viewport, helical navigator,
  worldlabs client, and ensemble/main integrations all landed in `apps/`
  with Atlas-compliant corrections (see atlas-compliance.md).

## [2026-10-25] ingest | Initial wiki + Atlas compliance audit + monorepo scaffold

- Ingested `dev-docs/` (7 md/py/tsx drafts), `.env` (key presence confirmed,
  value not recorded), `requirements.txt`, and live docs:
  `quickstart`, `how-to-use-atlas`, `api-reference`, `atlas-chisel`,
  `depth-guided-generation`, `cameras-and-posed-images`, `assets`.
- Flagged normative deviations: inverted log-depth direction, camera schema
  (`intrinsics/extrinsics` + `coordinateSystem`), asset upload flow, auth
  header `WLT-Api-Key`, `enhancePrompt`/`voxelGridOrientation`/`returnDepth`
  semantics, rollout budget (ctx=3 views). See `atlas-compliance.md`.
- Created wiki pages: epistemic-stack, atlas-compliance, khafre-dataset,
  pipeline, frontend, provenance-ensemble, verification.
- Scaffolded `apps/api/pipeline/*`, `apps/api/main.py`, `apps/web/*`,
  `Makefile`, `docker-compose.yml`, `.gitignore` (excludes `dev-docs/`, `.env`).
