# Provenance & Ensemble

> One-line: median consensus + normalized variance + NBV + SHA-256 manifest
> make every imagined pixel auditable.

## Math

- Stack `K` RGB runs → consensus = pixel-wise median (LAB/linear option).
- Disagreement = mean over channels of per-pixel std, normalized by P99 over
  valid mask → [0,1]; optional multiscale-SSIM/LPIPS hook for structural term.
- NBV: score candidate poses by mean disagreement in frustum (or by
  variance-reduction estimate `min(92, 45 + 80*mean_dis)`); top pose returned
  with `target_z_m` + rationale.
- Manifest: `SHA256(canonical_json{camera, prompt, seeds, voxel_size,
  z_min/max, child_hashes})`; `/api/provenance/{sha}` re-serves the record.

## Chained descent

- Request k's target #32 (+ returned RGB/depth) → request k+1's
  `sourceFrame` + `contextFrame`; reuse `promptUsed` with
  `enhancePrompt:false` to cut latency; keeps 32/batch within rollout budget.

## Sources

- Drafts: `worldlabs_client_and_main.md` (ensemble section).

## Links

- [Pipeline](pipeline.md), [Verification](verification.md).
