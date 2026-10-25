# DUAT — Epistemic World-Model Exploration Platform

Navigable, photorealistic, provenance-tracked 3D interiors from non-optical
volumetric sensor geometry, conditioned through World Labs Marble 2 / Atlas
`atlasChisel`. Flagship dataset: **Khafre Pyramid Subterranean SAR Anomalies**.

Every pixel is one of three tiers — **Measured** (cyan `#00F0FF`), **Inferred**
(amber `#FFB800`), **Imagined + variance** (violet `#BD00FF`) — with a permanent
"MODEL PREDICTION — CONDITIONED ON SENSOR GEOMETRY" provenance badge.

## Quickstart

```bash
make install   # python deps
make test      # mock-mode suite, no API key needed
make api       # :8000  (second terminal)
make web       # :5173  (cd apps/web && npm install && npm run dev)
# or: docker compose up --build
```

Live Atlas: set `WLT_API_KEY` in `.env` (already has `WLT_API_BASE_URL`).
Without a key the stack runs fully offline via the deterministic mock client.

## Compliance note (read first)

The live Atlas API uses **inverted** log-depth (`255 = zMin, 0 = empty`),
nested cameras (`intrinsics/extrinsics/coordinateSystem: rub`), `WLT-Api-Key`
auth, and the `assets -> :createUploadUrl -> :completeUpload` flow. The
implementation follows the live docs; see `wiki/atlas-compliance.md` and
`docs/BLUEPRINT.md`. Early drafts with the non-inverted codec and flat camera
dicts are superseded.

## Layout

- `apps/api/` — FastAPI + pipeline (geometry, depth_renderer, prompts,
  worldlabs_client, ensemble) + `tests/`.
- `apps/web/` — React 19 + R3F observatory (Epistemic Lens + Helical Navigator).
- `wiki/` — LLM-maintained knowledge base (Karpathy pattern).
- `docs/BLUEPRINT.md` — Phase 1 blueprint (tree, schemas, math).
- `data/` — local content-addressable cache (gitignored artifacts).
