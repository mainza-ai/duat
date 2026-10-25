# Atlas Compliance (normative)

> One-line: the live Marble 2 / Atlas API overrides the original brief
> wherever they disagree. This page lists every deviation found 2026-10-25.

## Endpoints & auth

- Base: `https://api.atlas-beta.worldlabs.ai/api/v2` (canonical OpenAPI host; overridden by `WLT_API_BASE_URL`).
- Task: `POST /api/v2/tasks:atlasChisel` (+ optional `?wait=true`).
- Poll: `GET /api/v2/operations/{id}` (or `POST :wait`); envelope
  `{id, done, error, response}` with `response: AtlasChiselResponse`.
- Headers: `WLT-Api-Key: <key>` (NOT `Authorization: Bearer`),
  `Content-Type: application/json`, deterministic `Idempotency-Key` (`duat-<sha256[:48]>` ≤128 chars).
- Health: `GET /api/v2/health` (no auth).

## Camera schema (all context + target cameras)

```json
{
  "intrinsics": {"fx": 914.0, "fy": 914.0, "cx": 640, "cy": 360, "width": 1280, "height": 720},
  "extrinsics": {"position": [x, y, z], "quaternion": [x, y, z, w], "coordinateSystem": "rub"}
}
```

- Pinhole only, 1280×720 for every context/target camera. `quaternion` XYZW
  unit. `rub` = looks along −Z (Three.js/Blender); `rdf` = +Z (OpenCV).
- Pose is camera-to-world, single shared world frame + distance unit.
- Drafts used flat `{width,height,fx,fy,cx,cy,pose:{position,quaternion}}` —
  rejected; production code uses the nested form above with `rub`.
- Source photos: any resolution (cover-resized to 1280×720), same frame/unit.

## Depth encoding — CRITICAL correction

- Official (inverted): `255` = `zMin`, darker = farther.
  Encode: `p = round(255*(1 - ln(z/zMin)/ln(zMax/zMin)))`, clamp ≥1.
  Decode: `z = zMin*(zMax/zMin)^(1 - p/255)`. `0` = no surface.
- EXR alternative: single-channel `image/x-exr`, planar-z, `0/NaN/Inf` = empty.
- Planar-z definition (agrees with brief):
  `z = d / sqrt(((u-cx)/fx)^2 + ((v-cy)/fy)^2 + 1)`.
- The brief + dev-docs drafts use the NON-inverted form
  (`pixel = round(254*ln(z/zMin)/ln(zMax/zMin))+1`, `z = zMin*(zMax/zMin)^((b-1)/254)`);
  that contradicts live docs and would render depth backwards. Production code
  implements the official inverted form; draft decoders kept only for reading
  legacy assets (flagged in UI).
- Never send `depth.confidenceAsset` to `atlasChisel` (rejected). 16-bit PNG
  rejected. `zMin/zMax`: `0 < zMin <= zMax`, per-frame nearest/farthest surface.

## Context/source/target composition

- `contextFrames`: 0 or exactly one posed depth frame per target camera, in
  target order. Each: `camera` + `depth:{depthAsset|logDepthAsset + zMin/zMax}`
  + optional `imageAsset` + optional `maskAsset` (white = known). Either both
  or neither of image/mask on ALL frames when present; no mixing.
- `sourceFrames`: `[{imageAsset, camera}]`, no mask/depth. Rollout budget:
  context = 3 views, photo/target = 1 each; max 128 views → 32 targets alone
  fill it; 30 targets leave room for 8 photos.
- `targetCameras`: 1–32 pinhole 1280×720.
- `prompt` required non-empty; `enhancePrompt` default true (reuse
  `promptUsed` + `enhancePrompt:false` for chained descents);
  `voxelSize` in normalized gauge (90th-pct disparity ≈ 1), 0–0.4
  (0/omit = keep depth; start 0.05–0.15); `voxelGridOrientation`:
  `world` (default, axis-aligned) | `gauge` (legacy).
- `modelParameters`: `{seed?, numSteps?}` (`additionalProperties: false`).
- `returnDepth`: top-level boolean on `AtlasChiselRequest` (NOT inside `modelParameters`);
  needs ≥2 targets with distinct centers (pure rotation → scale 0/NaN). Returned
  `depthAsset` (linear EXR) and `logDepthAsset` (PNG/EXR) are decoded to metric planar
  depth in meters (`read_planar_exr_bytes` / `decode_log_depth_png`), with zero-confidence
  pixels masked to `0.0`.

## Assets (draft flow was wrong)

- Drafts: `POST /assets {filename, contentType, dataBase64}` + `Bearer`.
  Real: `POST /assets {"asset": {displayName, url?|base64?|parts?}}` with
  `WLT-Api-Key`; response gives `assetId`. Small inline: `asset.base64`
  (<10 MiB). Large: create record → `:createUploadUrl` → PUT bytes (exact
  method+headers, no API key) → `:completeUpload`. Typed parts declared up
  front; one `:createReadUrl` per asset or `:createReadPrefix` for many.
- Task outputs: `{imageAsset: {assetId, url}}` (url valid 24 h); reuse
  `assetId` afterwards.
- Chained descent: promote returned `frames[i].imageAsset + camera`
  (+ `returnDepth` depth when available) into next request's
  `sourceFrames`/`contextFrames`.

## Sources

- `quickstart`, `how-to-use-atlas`, `api-reference/atlas-chisel`,
  `depth-guided-generation`, `cameras-and-posed-images`, `assets`,
  `operations`, `errors-and-retries` (all atlas-beta.worldlabs.ai/docs).

## Links

- [Epistemic Stack](epistemic-stack.md), [Pipeline](pipeline.md),
  [Provenance & Ensemble](provenance-ensemble.md).
