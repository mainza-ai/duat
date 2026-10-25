# Epistemic Stack

> One-line: every displayed pixel belongs to exactly one of three tiers —
> Measured, Inferred, Imagined — plus an ensemble variance overlay.

## Tier 1 — Measured (geometry & uncertainty)

- Raw sensor boundary mesh/volume, error bars, planar depth conditioning maps.
- Frontend color: radar-cyan `#00F0FF`. Transport: 8-bit inverted log-depth
  PNG (`0` = empty, `255` = `zMin`) or linear EXR planar-z with `0/NaN/Inf` =
  empty. Never model emptiness as a distant dome.
- Decoding (normative, per live docs): `z = zMin * (zMax/zMin)^(1 - p/255)`.

## Tier 2 — Inferred (material & physical proxies)

- Dielectric reflectivity, acoustic impedance, muon density mapped onto geometry.
- High reflectivity → water/metal/dense granite; low → porous limestone/air.
- Frontend color: amber `#FFB800`. Transport: 8-bit grayscale proxy PNG.

## Tier 3 — Imagined (generative appearance & ensemble variance)

- `atlasChisel` RGB conditioned on depth + pose + lighting rig prompt.
- Always shown with Consensus View (median across seeds/posteriors/prompts)
  and Disagreement Heatmap (normalized variance → emerald `#34D399`
  shimmer). Clicking high-disagreement regions opens a per-seed inspector.
- Mandatory badge: "MODEL PREDICTION — CONDITIONED ON SENSOR GEOMETRY".

## Blending contract

- Weights `(w1, w2, w3)` normalized by sum; shimmer applied only where
  `valid && disagreement > threshold`; X-ray loupe overrides locally.
- Telemetry readout shows pixel `(u,v)`, planar `z`, reflectivity,
  disagreement % on hover.

## Sources

- Live: `depth-guided-generation` (encode/mark-sky sections).
- Drafts: `Epistemic_Lens_Viewport.md` (shader), `unify_frontend.md`.

## Links

- [Atlas Compliance](atlas-compliance.md), [Pipeline](pipeline.md),
  [Frontend](frontend.md), [Provenance & Ensemble](provenance-ensemble.md).
