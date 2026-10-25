# DUAT — Phase 1 Implementation Blueprint

## 1. Directory tree (monorepo root = current dir)

```
duat/
  .env                      # WLT_API_BASE_URL + WLT_API_KEY (never commit key)
  .gitignore                # excludes dev-docs/, .env, node_modules, __pycache__
  Makefile
  docker-compose.yml
  requirements.txt
  README.md
  docs/BLUEPRINT.md         # this file
  wiki/                     # LLM wiki (index, log, schema, 6 pages)
  data/khafre_showcase/     # generated GLB/PLY/manifest (gitignored artifacts)
  apps/api/
    __init__.py
    main.py                 # FastAPI server
    pipeline/
      __init__.py
      khafre_sar_volume.py  # SAR volume + posterior meshes + helical cameras
      geometry.py           # facade + degradation sweep + mesh I/O
      depth_renderer.py     # official inverted log-depth codec + raycasting
      prompts.py            # hypothesis × instrument rig composer
      worldlabs_client.py   # Atlas client (live + mock), assets, chaining
      ensemble.py           # consensus / disagreement / NBV / manifest
    tests/
      __init__.py
      test_depth_math.py
      test_ensemble.py
      test_api.py
  apps/web/
    package.json
    vite.config.ts
    index.html
    src/
      main.tsx
      App.tsx
      lib/api.ts
      components/
        EpistemicLensViewport.tsx
        HelicalDescentNavigator.tsx
```

## 2. API contract schemas

### Atlas (live, normative) — see wiki/atlas-compliance.md

- `POST /api/v2/tasks:atlasChisel` body: `{contextFrames[], sourceFrames[],
  targetCameras[1..32], prompt, enhancePrompt=true, voxelSize?,
  voxelGridOrientation?, returnDepth?, modelParameters{seed?, numSteps?}}`.
- Camera: `{intrinsics:{fx,fy,cx,cy,width:1280,height:720},
  extrinsics:{position:[x,y,z], quaternion:[x,y,z,w] XYZW, coordinateSystem:'rub'}}`.
- Depth: `depth:{logDepthAsset:{assetId}, zMin, zMax}` (inverted PNG) or
  `depth:{depthAsset:{assetId}}` (EXR planar-z). `0`=empty.
- Op envelope `{id, done, error, response:{frames[{imageAsset,camera,depth?}], promptUsed}}`.

### Duat backend (Pydantic v2 + TS mirror in apps/web/src/lib/api.ts)

```python
CameraPose: {position: [x,y,z], quaternion: [x,y,z,w]}
PinholeCameraSpec: {width=1280, height=720, fov_deg=70,
  coordinate_system='rub', pose} + fx property + to_atlas_dict()
EnsembleExploreRequest: {camera, dataset: khafre|nfc|big-void|uploaded_mesh,
  hypothesis: megalithic|karst|artifact|bedrock_null, force_geometry_on_null=False,
  instrument_rig: borescope_led|drop_flare|structured_laser,
  seeds=[7,42,108,256], voxel_size=0.25, sensor_noise_sigma_m=0.18,
  z_min=0.5, z_max?=rig_default, mode: mock|live}
EnsembleExploreResponse: {provenance_sha256, timestamp_utc, prompt_used,
  z_min, z_max, voxel_size, empty_void_pixel_ratio, mean_ensemble_disagreement,
  layers{measured_log_depth_b64, inferred_reflectivity_b64,
    imagined_consensus_rgb_b64, disagreement_map_b64},
  next_best_view{target_z_m, predicted_variance_reduction_pct, rationale}}
GET /api/health → {status, project, worldlabs_mock_mode, khafre_volume_loaded}
GET /api/khafre/trajectory?num_cameras=32 → {bounds_min/max_m, voxel_size_m, targetCameras[]}
POST /api/chisel/ensemble → EnsembleExploreResponse
POST /api/chisel/chain → chained-descent batch (target#32 promoted)
GET /api/provenance/{sha} → stored manifest record
```

## 3. Mathematical verification (planar depth + log codec)

Pinhole (undistorted), pixel `(u,v)`, ray distance `d`:
`dir = ((u-cx)/fx, (v-cy)/fy, 1)/|..|`, planar depth = ray z-component:
`z = d / sqrt(((u-cx)/fx)^2 + ((v-cy)/fy)^2 + 1)`. ∎ (implemented +
round-trip-tested in test_depth_math.py).

Official inverted log codec (`255`=near):
`p = clip(round(255*(1 - ln(z/zMin)/ln(zMax/zMin))), 1, 255)`, `0`=empty;
`z = zMin*(zMax/zMin)^(1 - p/255)`. Endpoints: `zMin→255`, `zMax→1`
(clamped, never 0). Max quantization rel-error <1% across typical ranges;
provenance records `(zMin,zMax)` per frame. GLSL + NumPy decoders share the
formula. The brief's non-inverted variant is documented as superseded.
