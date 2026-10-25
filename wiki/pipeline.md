# Pipeline

> One-line: `geometry → depth_renderer → prompts → worldlabs_client →
> ensemble → main.py` with strict 0-masking and mock/offline parity.

## Modules (`apps/api/pipeline/`)

- `geometry.py` — re-export facade over `khafre_sar_volume.py` + degradation
  sweep + mesh I/O (keeps the brief's import path working).
- `khafre_sar_volume.py` — `KhafreSARVolumeEngine`, `EpistemicVolumeBundle`,
  posterior mesh extraction, helical cameras (Atlas camera dicts).
- `depth_renderer.py` — official inverted log-depth codec
  (`encode_log_depth_png`, `decode_log_depth_png`), EXR planar-z I/O,
  pinhole `raycast_planar_depth` (mesh via trimesh ray or analytic shaft
  raymarch fallback), `render_shaft_planar_depth_and_reflectivity`.
- `prompts.py` — `HYPOTHESIS_PRIORS` (megalithic/karst/artifact) ×
  `INSTRUMENT_RIGS` (borescope_led zMax 6 / drop_flare 40 / structured_laser
  25) → `(prompt, recommended_zmax)`; `ENHANCE_DEFAULT=True`.
- `worldlabs_client.py` — async client: `.env` (`WLT_API_KEY`/`WLT_API_BASE_URL`,
  legacy `WORLDLABS_*` fallback), `WLT-Api-Key` auth, asset create/upload/
  complete + base64-inline path, `tasks:atlasChisel` submit, `:wait`+poll,
  `run_atlas_chisel` (≤32 targets), `chain_descent` (target #32 →
  source+context), deterministic `_synthesize_mock_chisel_output`.
- `ensemble.py` — consensus (median), disagreement (normalized mean-std +
  optional SSIM/LPIPS hook), next-best-view scoring, SHA-256 manifest.
- `main.py` — FastAPI: `GET /api/health`, `GET /api/khafre/trajectory`,
  `POST /api/chisel/ensemble`, `POST /api/chisel/chain`,
  `GET /api/provenance/{sha}`; CORS open; lazy volume cache.

## Request/response (Pydantic + TS mirror in `apps/web/src/lib/api.ts`)

- `EnsembleExploreRequest {camera, hypothesis, instrument_rig, seeds[],
  voxel_size, sensor_noise_sigma_m, z_min, z_max?, mock_mode?}` →
  `EnsembleExploreResponse {provenance_sha256, prompt_used, z_min/max,
  layers{measured_log_depth_b64, inferred_reflectivity_b64,
  imagined_consensus_rgb_b64, disagreement_map_b64}, next_best_view{...}}`.

## Sources

- Drafts: `worldlabs_client_and_main.md`, `worldlabs_client.py.py`, `main.py`.

## Links

- [Atlas Compliance](atlas-compliance.md), [Epistemic Stack](epistemic-stack.md),
  [Provenance & Ensemble](provenance-ensemble.md), [Verification](verification.md).
