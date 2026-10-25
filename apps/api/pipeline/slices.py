"""
duat/apps/api/pipeline/slices.py

Tomographic slice decoding: the evidence artifact of SAR anomaly claims is 2D
imagery (jet-colormap heatmaps, grayscale scans) with colorbars, legends and
annotation overlays. This module decodes them into anomaly fields honestly:

- jet pixels are inverted through the actual jet transfer function
  (nearest-LUT, exact — not an HSV heuristic);
- flat border-connected regions (legends, frames, annotation boxes) are masked
  via connected components, never interpolated as geology;
- vertical shaft striping is quantified with peak finding, not eyeballed.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.ndimage import label as cc_label


def jet_lut(n: int = 256) -> np.ndarray:
    """Classic jet transfer function, (n,3) float BGR in [0,1] to match OpenCV."""
    x = np.linspace(0.0, 1.0, n, dtype=np.float64)
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    rgb = np.stack([r, g, b], axis=-1)
    return rgb[..., ::-1]  # BGR for OpenCV convention


def _quantize(img: np.ndarray, levels: int = 16) -> np.ndarray:
    q = np.clip((img * levels).astype(np.int32), 0, levels - 1)
    return (q[..., 0] * levels * levels + q[..., 1] * levels + q[..., 2]).astype(np.int32)


def mask_legends(bgr: np.ndarray, mask_rects: Optional[List[Tuple[float, float, float, float]]] = None,
                 min_area_frac: float = 0.02) -> np.ndarray:
    """Valid-pixel mask: False for flat border-connected legend/frame regions
    and caller-supplied normalized rects. Annotations fully inside the field
    (small text) are kept and reported, not silently erased."""
    h, w = bgr.shape[:2]
    valid = np.ones((h, w), dtype=bool)
    if mask_rects:
        for x0, y0, x1, y1 in mask_rects:
            valid[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)] = False
    # Flat components touching the border (colorbars, frames, boxes).
    # Coarse quantization (4 levels): true gradients span multiple bands and
    # survive via high within-component variance, while solid legend fills
    # stay flat and are masked. Fine quantization would dice smooth fields
    # into flat border-touching stripes and eat them — never do that.
    flat = _quantize(bgr.astype(np.float32) / 255.0, levels=4)
    # NOTE: scipy label() groups by nonzero, not by value — segment each
    # quantized color separately so distinct regions never merge.
    for code in np.unique(flat):
        if code == 0:
            continue
        lab, _ = cc_label(flat == code)
        touches = set(np.unique(np.concatenate([lab[0, :], lab[-1, :], lab[:, 0], lab[:, -1]])))
        touches.discard(0)
        for cid in touches:
            cells = lab == cid
            if cells.mean() < min_area_frac:
                continue
            vals = bgr[cells].astype(np.float32)
            if vals.std(axis=0).max() < 12.0:
                valid[cells] = False
    # Code 0 (near-black) regions touching the border (letterboxing) are field,
    # not legend — never mask them here.
    return valid


def decode_jet_anomaly(bgr: np.ndarray,
                       mask_rects: Optional[List[Tuple[float, float, float, float]]] = None,
                       gray_fallback_sat: float = 0.15) -> Tuple[np.ndarray, np.ndarray]:
    """Decode a tomography image to (anomaly [0,1] float32, valid_mask).

    Jet pixels invert through the transfer function; near-grayscale inputs
    fall back to luminance (some press panels are desaturated scans)."""
    h, w = bgr.shape[:2]
    f = bgr.astype(np.float32) / 255.0
    mx, mn = f.max(axis=-1), f.min(axis=-1)
    sat = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    valid = mask_legends(bgr, mask_rects)
    anomaly = np.zeros((h, w), dtype=np.float32)
    if (float(sat[valid].mean()) if np.any(valid) else 0.0) < gray_fallback_sat:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        anomaly[valid] = gray[valid]
        return anomaly.astype(np.float32), valid
    lut = jet_lut(256).astype(np.float32)  # BGR
    flat = f.reshape(-1, 3)
    uniq, inv = np.unique((flat * 255).astype(np.int32), axis=0, return_inverse=True)
    d = ((uniq[:, None, :].astype(np.float32) / 255.0 - lut[None, :, :]) ** 2).sum(axis=-1)
    best = d.argmin(axis=1).astype(np.float32) / 255.0
    anomaly = best[inv].reshape(h, w)
    anomaly[~valid] = 0.0
    return anomaly.astype(np.float32), valid


def stripe_peaks(anomaly: np.ndarray, valid_mask: np.ndarray,
                 min_prominence: float = 0.08) -> Dict[str, object]:
    """Vertical shaft striping: column-mean profile over valid pixels + peaks."""
    from scipy.signal import find_peaks

    col_valid = valid_mask.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        profile = np.where(col_valid > 0, (anomaly * valid_mask).sum(axis=0) / np.maximum(col_valid, 1), 0.0)
    peaks, props = find_peaks(profile, prominence=min_prominence, distance=3)
    return {"profile": [round(float(v), 4) for v in profile],
            "peaks": [int(p) for p in peaks],
            "peak_positions_norm": [round(float(p) / max(len(profile) - 1, 1), 4) for p in peaks],
            "peak_count": int(len(peaks)),
            "prominences": [round(float(v), 4) for v in props.get("prominences", [])]}
