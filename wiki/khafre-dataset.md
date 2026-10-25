# Khafre Dataset

> One-line: procedural SAR showcase — 8 helical shafts + 2 basal chambers —
> CONDITIONED ON A CONTESTED, UNVERIFIED PRESS CLAIM. Read this page before
> citing any number below.

## Evidential status (normative for all dataset copy)

- Source claim `khafre-2025-press`: press conference 2025-03-15, **never
  peer-reviewed** → status **contested**. Foundation paper (`biondi-malanga-2022`)
  → **RETRACTED Aug 2026** for serious methodological flaws. Expert response:
  rejected (Hawass: no authorized work/evidence; Conyers: "huge exaggeration";
  AFP: "unfounded"). Full record: `GET /api/claims`, per-claim detail at
  `GET /api/claims/{id}`.
- Outlet variance is data, not noise: basal chambers **80 m** (JPost/Ahram) vs
  **90 m** (NatGeo/TRT); depths 610–640 m (some reports: 2 km). The synthesizer
  defaults to 80 m and accepts 40–120 m; `VolumeMetadata.chamber_size_m`
  records the choice; the trajectory response echoes it.
- Sensor realism: our SAR σ defaults (0.35–0.6 m) are *modeling choices*, not
  demonstrated precision of the contested method (orbital SAR cannot resolve
  such features at such depths — see claim `physical_notes`). The degradation
  ladder cites validated figures instead (ScanPyramids ~3 m voids).

## Validated benchmarks (shipped alongside)

- `synthesize_reference_void("nfc")` — 9×2×2 m corridor (Nature 2023, cm precision).
- `synthesize_reference_void("big-void")` — 30×6 m void (Nature 2017).
- Served at `GET /api/khafre/reference-voids` with cited dims. Use these, not
  Khafre, for fidelity claims. Ongoing legitimate Khafre work: ScIDEP muon
  scan (arXiv 2025); historic null: Alvarez 1960s (solid rock → our null prior).

## Complex (synthesizer defaults — all overrideable, all cited as ranges)

- Bounds `x,y ∈ [-100,100] m`, `z ∈ [-650,0] m`; grid default
  `(160,160,240)` (override for memory).
- 8 shafts: `x ∈ {-40,+40}`, `y ∈ {-60,-20,+20,+60}`, radius `5.5 m`
  (claimed "cylindrical"; radius value is our modeling choice, NOT a cited figure),
  `z ∈ [-550,-15]`, spiral wall modulation
  `1.1*sin(2π z/28 + 2θ)` (see audit: remodel to swept ramp in K3).
- 2 chambers: centers `(-40,0,-565)`, `(40,0,-565)`, footprint parameter
  (default 80), 50 m tall. Upper corridor at `z=-45` linking rows.
- Tier-2 reflectivity: boundary proximity + water-table boost at `z≈-585`
  + stratigraphy bands + smoothed noise, clipped [0,1].

## Posterior sampling / ingestion / degradation sweep / helical cameras

Unchanged from prior revision (see pipeline page): posterior SDF perturbation +
Marching Cubes with inward normals; 2D slice ingest; `degrade_for_sweep`
ladder; 32-cam helical rigs (Atlas camera dicts, `coordinateSystem: rub`).

## Sources

- Drafts: `Khafre_SAR_Volume_Synthesizer.md`, `khafre_sar_volume.py`.
- Live registry: `GET /api/claims` (stable IDs, URLs, dates).

## Links

- [Pipeline](pipeline.md), [Atlas Compliance](atlas-compliance.md),
  [Provenance & Ensemble](provenance-ensemble.md), [Verification](verification.md).
