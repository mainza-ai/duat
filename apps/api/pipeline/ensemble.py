"""
duat/apps/api/pipeline/ensemble.py

Consensus / disagreement / next-best-view / provenance manifest.
Vectorized NumPy; MPS/MLX hook attempted opportunistically, CPU fallback default.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Tuple

import numpy as np


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


def _try_mps() -> bool:
    try:
        import torch

        return bool(torch.backends.mps.is_available())
    except Exception:
        return False


def _accelerated_stats(rgb_stack: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Median consensus + per-pixel mean-channel std, via torch (MPS/CUDA) if
    importable, else vectorized NumPy. Returns (consensus_u8, std_map)."""
    if _torch_available():
        try:
            import torch

            device = "mps" if _try_mps() else ("cuda" if torch.cuda.is_available() else "cpu")
            t = torch.from_numpy(rgb_stack.astype(np.float32)).to(device)
            consensus = torch.median(t, dim=0).values.clamp(0, 255).byte().cpu().numpy()
            std_map = torch.std(t, dim=0).mean(dim=-1).cpu().numpy()
            return consensus, std_map.astype(np.float32)
        except Exception:
            pass
    consensus = np.median(rgb_stack, axis=0).astype(np.uint8)
    std_map = np.mean(np.std(rgb_stack.astype(np.float32), axis=0), axis=-1)
    return consensus, std_map.astype(np.float32)


def compute_consensus(rgb_stack: np.ndarray, method: str = "median") -> np.ndarray:
    """(K,H,W,3) float32 -> (H,W,3) uint8 consensus."""
    if method == "mean":
        return np.clip(np.mean(rgb_stack, axis=0), 0, 255).astype(np.uint8)
    consensus, _ = _accelerated_stats(rgb_stack)
    return consensus


def _structural_divergence(rgb_stack: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Mean pairwise (1 - SSIM) on downsampled grayscale — the structural half
    of the disagreement signal (cheap multiscale-SSIM proxy)."""
    from skimage.metrics import structural_similarity as ssim
    from skimage.transform import resize

    k = rgb_stack.shape[0]
    if k < 2:
        return np.zeros(rgb_stack.shape[1:3], dtype=np.float32)
    gray = rgb_stack.mean(axis=-1).astype(np.float32)
    h, w = gray.shape[1], gray.shape[2]
    scale = min(1.0, 320.0 / max(h, w))
    if scale < 1.0:
        small = np.stack([resize(g, (int(h * scale), int(w * scale)),
                                 anti_aliasing=True) for g in gray], axis=0)
    else:
        small = gray
    div = np.zeros(small.shape[1:], dtype=np.float64)
    pairs = 0
    data_range = float(max(small.max() - small.min(), 1.0))
    for i in range(k):
        for j in range(i + 1, k):
            _, s = ssim(small[i], small[j], data_range=data_range, full=True)
            div += (1.0 - s)
            pairs += 1
    div /= max(pairs, 1)
    if scale < 1.0:
        div = resize(div, (h, w), anti_aliasing=True).astype(np.float32)
    return np.where(valid_mask, np.clip(div, 0.0, 1.0), 0.0).astype(np.float32)


def compute_disagreement(rgb_stack: np.ndarray, valid_mask: np.ndarray,
                         structural_weight: float = 0.0) -> Tuple[np.ndarray, float]:
    """Per-pixel mean-channel std normalized by P99 over valid -> [0,1],
    optionally blended with pairwise structural (1-SSIM) divergence."""
    _, std_px = _accelerated_stats(rgb_stack)
    eff_mask = valid_mask if np.any(valid_mask) else np.ones_like(valid_mask, dtype=bool)
    denom = max(float(np.percentile(std_px[eff_mask], 99)), 1.0)
    norm = np.where(eff_mask, np.clip(std_px / denom, 0.0, 1.0), 0.0).astype(np.float32)
    if structural_weight > 0.0:
        struct = _structural_divergence(rgb_stack, eff_mask)
        s_denom = max(float(np.percentile(struct[eff_mask], 99)), 1e-6)
        struct_n = np.where(eff_mask, np.clip(struct / s_denom, 0.0, 1.0), 0.0)
        w = float(np.clip(structural_weight, 0.0, 1.0))
        norm = ((1.0 - w) * norm + w * struct_n).astype(np.float32)
    mean_dis = float(np.mean(norm[eff_mask]))
    return norm, mean_dis


def disagreement_to_u8(norm: np.ndarray) -> np.ndarray:
    return np.round(np.clip(norm, 0.0, 1.0) * 255.0).astype(np.uint8)


def score_next_best_view(disagreement_norm: np.ndarray, valid_mask: np.ndarray,
                         candidate_depths_m: List[float],
                         planar_z: np.ndarray | None = None,
                         cam_z_m: float = -45.0) -> Tuple[float, float]:
    """Pick candidate descent depth (m) that maximizes expected information gain
    (highest local epistemic disagreement in the corresponding depth band)."""
    if not candidate_depths_m:
        return round(cam_z_m - 6.0, 1), 60.0
    h, w = disagreement_norm.shape
    eff_mask = valid_mask if np.any(valid_mask) else np.ones_like(valid_mask, dtype=bool)
    n_cands = len(candidate_depths_m)
    scores: List[float] = []
    for idx, z_cand in enumerate(candidate_depths_m):
        target_dist = max(abs(float(z_cand) - float(cam_z_m)), 1.0)
        if planar_z is not None and planar_z.shape == disagreement_norm.shape and np.any(planar_z > 0):
            band_mask = eff_mask & (np.abs(planar_z - target_dist) <= 3.5)
            if np.any(band_mask):
                scores.append(float(np.mean(disagreement_norm[band_mask])))
                continue
        # Partition image vertically across candidate horizons
        band_h = max(h // max(n_cands, 1), 1)
        r0 = min(idx * band_h, max(h - band_h, 0))
        r1 = min(r0 + band_h, h)
        band = disagreement_norm[r0:r1, :]
        band_m = eff_mask[r0:r1, :]
        scores.append(float(np.mean(band[band_m])) if np.any(band_m) else float(np.mean(band)))
    best = int(np.argmax(scores)) if scores else 0
    best_score = float(scores[best]) if scores else 0.0
    mean_dis = float(np.mean(disagreement_norm[eff_mask]))
    reduction = round(min(94.0, max(35.0, 42.0 + 48.0 * max(mean_dis, best_score))), 1)
    return float(candidate_depths_m[best]), reduction


def provenance_manifest(payload: Dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()
