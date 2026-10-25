"""
duat/apps/api/pipeline/shaft_model.py

Single source of truth for the Khafre shaft void SDF (bundle convention:
< 0 inside void, > 0 in solid rock, 0 at the boundary).

The void is a plain vertical cylinder MINUS a helical ramp solid (the
"descending spiral pathway" is rock you walk on, protruding from the wall —
not a whole-wall undulation). An optional whole-wall relief term reproduces
the legacy look and is reserved for the radar-artifact hypothesis variant.

Used by both the volume synthesizer (grid evaluation) and the depth raymarcher
(per-step evaluation), so conditioning geometry can never diverge from the mesh.
All functions are vectorized over numpy arrays.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ShaftParams:
    radius_m: float = 5.5
    spiral_pitch_m: float = 28.0
    ramp_width_m: float = 2.0       # radial extent of the walkway shelf
    ramp_thickness_m: float = 0.8   # vertical thickness of the shelf
    wall_relief_m: float = 0.0      # legacy whole-wall undulation (artifact variant)
    wall_relief_lobes: int = 2
    z_top_m: float = -15.0
    z_bottom_m: float = -550.0
    handedness: int = 1             # +1 descending with CCW angle


def _helix_gap(pz: np.ndarray, theta: np.ndarray, p: ShaftParams) -> np.ndarray:
    """Signed vertical distance from each point to the nearest ramp turn,
    slope-corrected to approximate perpendicular distance."""
    u = (p.z_top_m - pz) / p.spiral_pitch_m + p.handedness * theta / (2.0 * np.pi)
    gap = (u - np.round(u)) * p.spiral_pitch_m
    r_ramp = max(p.radius_m - p.ramp_width_m / 2.0, 0.5)
    cos_a = p.spiral_pitch_m / np.sqrt(p.spiral_pitch_m ** 2 + (2.0 * np.pi * r_ramp) ** 2)
    return gap * cos_a


def ramp_box_sdf(r: np.ndarray, gap: np.ndarray, p: ShaftParams) -> np.ndarray:
    """SDF of the helical shelf solid in (radial, vertical) coords (< 0 solid)."""
    dr = np.maximum((p.radius_m - p.ramp_width_m) - r, r - (p.radius_m + 0.5))
    dv = np.abs(gap) - p.ramp_thickness_m / 2.0
    ax, ay = np.maximum(dr, 0.0), np.maximum(dv, 0.0)
    return np.sqrt(ax * ax + ay * ay) + np.minimum(np.maximum(dr, dv), 0.0)


def shaft_void_sdf(
    px: np.ndarray, py: np.ndarray, pz: np.ndarray,
    cx: float, cy: float, p: ShaftParams,
) -> np.ndarray:
    """Bundle-convention void SDF for one shaft centered at (cx, cy)."""
    dx = px - cx
    dy = py - cy
    r = np.sqrt(dx * dx + dy * dy)
    theta = np.arctan2(dy, dx)

    cyl = r - p.radius_m
    if p.wall_relief_m != 0.0:
        omega = (2.0 * np.pi) / p.spiral_pitch_m
        cyl = cyl - p.wall_relief_m * np.sin(omega * pz + p.wall_relief_lobes * theta)
    zcap = np.maximum(pz - p.z_top_m, p.z_bottom_m - pz)
    void = np.maximum(cyl, zcap)

    gap = _helix_gap(pz, theta, p)
    # Only the turns inside the shaft's vertical span can matter; far turns
    # already produce large |gap|, so no explicit clamp is needed.
    ramp = ramp_box_sdf(r, gap, p)
    return np.maximum(void, -ramp)


def ramp_proximity(r: np.ndarray, gap: np.ndarray, p: ShaftParams, scale_m: float = 2.0) -> np.ndarray:
    """Tier-2 dielectric boost field peaking at the ramp surface (dressed stone)."""
    rb = ramp_box_sdf(r, gap, p)
    return np.exp(-(rb / max(scale_m, 1e-6)) ** 2)
