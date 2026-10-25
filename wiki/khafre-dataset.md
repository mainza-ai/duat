# Khafre Dataset

> One-line: procedural SAR showcase — 8 helical shafts + 2 basal chambers —
> with posterior sampling for ensembles and a degradation sweep harness.

## Complex (synthesizer defaults)

- Bounds `x,y ∈ [-100,100] m`, `z ∈ [-650,0] m`; grid default
  `(160,160,240)` (override for memory).
- 8 shafts: `x ∈ {-40,+40}`, `y ∈ {-60,-20,+20,+60}`, radius `5.5 m`,
  `z ∈ [-550,-15]`, spiral wall modulation
  `1.1*sin(2π z/28 + 2θ)` (double-helix ledge).
- 2 chambers: centers `(-40,0,-565)`, `(40,0,-565)`, `80×80×50 m`.
- Upper corridor at `z=-45` linking rows.
- Tier-2 reflectivity: boundary proximity + water-table boost at `z≈-585`
  + stratigraphy bands + smoothed noise, clipped [0,1].

## Posterior sampling

- Perturb SDF with spatially-correlated Gaussian
  (`σ = sensor_uncertainty_m * scale`, smooth σ≈1.6 voxels), then Marching
  Cubes at iso 0, flip winding → inward normals, attach per-vertex
  reflectivity + amber/cyan preview colors.
- ROI crops (e.g. Shaft #2 `[-52,-32]×[-32,-8]×[-220,-15]`) for dense
  ensemble meshes; full-complex GLB for the 3D HUD.

## Ingestion & degradation sweep

- `reconstruct_from_2d_sar_slices`: HSV+gray anomaly response → resize →
  zoom-interp → smooth → `(threshold - field)*15` SDF + reflectivity bundle.
- Sweep harness (`geometry.degrade_for_sweep`): Gaussian blur + quantization
  ladder LiDAR→GPR→muon; metric: fidelity vs sensor σ for the demo report.

## Helical cameras

- 32 cams max, 1280×720, `fx = (640)/tan(fov/2)`, helix inside Shaft #2,
  `position/quaternion(XYZW)` + `coordinateSystem: rub` (Atlas form).

## Sources

- Drafts: `Khafre_SAR_Volume_Synthesizer.md`, `khafre_sar_volume.py`.

## Links

- [Pipeline](pipeline.md), [Provenance & Ensemble](provenance-ensemble.md),
  [Verification](verification.md).
