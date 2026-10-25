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
