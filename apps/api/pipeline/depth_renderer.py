"""
duat/apps/api/pipeline/depth_renderer.py

Pinhole planar-depth renderer + OFFICIAL Atlas inverted log-depth codec.

Official spec (depth-guided-generation):
  planar z = d / sqrt(((u-cx)/fx)^2 + ((v-cy)/fy)^2 + 1)
  encode: p = round(255 * (1 - ln(z/zMin)/ln(zMax/zMin))), clamp [1,255], 0 = empty
  decode: z = zMin * (zMax/zMin)^(1 - p/255)
  EXR: planar z, 0/NaN/Inf = empty. Never send confidenceAsset to atlasChisel.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


# ---------------- codec ----------------

def encode_log_depth_png(planar_z: np.ndarray, z_min: float, z_max: float) -> np.ndarray:
    """Planar-z float32 (0/NaN/Inf = empty) -> uint8 inverted log-depth (0 = empty)."""
    assert 0 < z_min <= z_max, "require 0 < z_min <= z_max"
    out = np.zeros(planar_z.shape, dtype=np.uint8)
    valid = np.isfinite(planar_z) & (planar_z >= z_min) & (planar_z <= z_max)
    if np.any(valid):
        zc = np.clip(planar_z[valid], z_min, z_max)
        denom = np.log(z_max / z_min)
        if denom <= 0:
            out[valid] = 255
        else:
            log_norm = np.log(zc / z_min) / denom
            out[valid] = np.clip(np.round(255.0 * (1.0 - log_norm)), 1, 255).astype(np.uint8)
    return out


def decode_log_depth_png(log_depth_u8: np.ndarray, z_min: float, z_max: float) -> np.ndarray:
    """uint8 inverted log-depth -> planar-z float32 (0.0 = empty)."""
    planar = np.zeros(log_depth_u8.shape, dtype=np.float32)
    valid = log_depth_u8 > 0
    if np.any(valid):
        p = log_depth_u8[valid].astype(np.float32)
        planar[valid] = (z_min * ((z_max / z_min) ** (1.0 - p / 255.0))).astype(np.float32)
    return planar


def write_planar_exr(planar_z: np.ndarray, path: str | Path) -> None:
    arr = np.where(np.isfinite(planar_z) & (planar_z > 0), planar_z, 0.0).astype(np.float32)
    try:
        import OpenEXR  # type: ignore
        import Imath  # type: ignore

        h, w = arr.shape
        header = OpenEXR.Header(w, h)
        header["channels"] = {"R": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))}
        exr = OpenEXR.OutputFile(str(path), header)
        exr.writePixels({"R": arr.tobytes()})
        exr.close()
    except Exception:
        import imageio.v2 as imageio  # type: ignore

        imageio.imwrite(str(path), arr)


def read_planar_exr(path: str | Path) -> np.ndarray:
    try:
        import OpenEXR  # type: ignore
        import Imath  # type: ignore

        exr = OpenEXR.InputFile(str(path))
        dw = exr.header()["dataWindow"]
        w, h = dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1
        raw = exr.channel("R", Imath.PixelType(Imath.PixelType.FLOAT))
        return np.frombuffer(raw, dtype=np.float32).reshape(h, w)
    except Exception:
        import imageio.v2 as imageio  # type: ignore

        arr = imageio.imread(str(path)).astype(np.float32)
        if arr.ndim == 3:
            arr = arr[..., 0]
        return np.where(np.isfinite(arr) & (arr > 0), arr, 0.0).astype(np.float32)


# ---------------- pinhole helpers ----------------

def pinhole_ray_directions(w: int, h: int, fx: float, fy: float, cx: float, cy: float) -> Tuple[np.ndarray, np.ndarray]:
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    x_norm = (u - cx) / fx
    y_norm = (v - cy) / fy
    inv_norm = 1.0 / np.sqrt(x_norm**2 + y_norm**2 + 1.0)
    dirs = np.stack([x_norm * inv_norm, y_norm * inv_norm, inv_norm], axis=-1).astype(np.float32)
    return dirs, inv_norm.astype(np.float32)


def ray_distance_to_planar_depth(ray_d: np.ndarray, inv_norm: np.ndarray) -> np.ndarray:
    return (ray_d * inv_norm).astype(np.float32)


# ---------------- shaft raymarcher (analytic showcase) ----------------

def render_shaft_planar_depth_and_reflectivity(
    cam_origin_xyz: Tuple[float, float, float],
    width: int = 1280,
    height: int = 720,
    fx: Optional[float] = None,
    fy: Optional[float] = None,
    cx: Optional[float] = None,
    cy: Optional[float] = None,
    z_min: float = 0.5,
    z_max: float = 40.0,
    noise_sigma_m: float = 0.15,
    seed: int = 42,
    shaft_center_xy: Tuple[float, float] = (-40.0, -20.0),
    shaft_radius_m: float = 5.5,
    spiral_amp_m: float = 0.95,
    spiral_pitch_m: float = 28.0,
    n_steps: int = 96,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Raymarch analytic helical shaft; returns (log_depth_u8, refl_u8, valid_mask)."""
    rng = np.random.default_rng(seed)
    fx = fx or float((width / 2.0) / np.tan(np.radians(70.0 / 2.0)))
    fy = fy or fx
    cx = width / 2.0 if cx is None else cx
    cy = height / 2.0 if cy is None else cy
    dirs, inv_norm = pinhole_ray_directions(width, height, fx, fy, cx, cy)

    rel = np.array([cam_origin_xyz[0] - shaft_center_xy[0],
                    cam_origin_xyz[1] - shaft_center_xy[1], cam_origin_xyz[2]], dtype=np.float32)
    omega = (2.0 * np.pi) / spiral_pitch_m
    t = np.full((height, width), z_min, dtype=np.float32)
    hit = np.zeros((height, width), dtype=bool)
    for _ in range(n_steps):
        pts = rel + dirs * t[..., None]
        ang = np.arctan2(pts[..., 1], pts[..., 0])
        spiral = spiral_amp_m * np.sin(omega * pts[..., 2] + 2.0 * ang)
        sdf = (shaft_radius_m + spiral) - np.sqrt(pts[..., 0] ** 2 + pts[..., 1] ** 2)
        newly = (sdf <= 0.025) & (~hit) & (t < z_max)
        hit |= newly
        t = np.where(hit, t, t + np.maximum(sdf * 0.65, 0.06))
    if noise_sigma_m > 0:
        t = t + rng.normal(0.0, noise_sigma_m, size=t.shape).astype(np.float32)
    planar_z = ray_distance_to_planar_depth(t, inv_norm)
    valid = hit & np.isfinite(planar_z) & (planar_z >= z_min) & (planar_z <= z_max)
    planar_z = np.where(valid, planar_z, 0.0).astype(np.float32)

    log_depth_u8 = encode_log_depth_png(planar_z, z_min, z_max)
    hit_pts = rel + dirs * t[..., None]
    hit_ang = np.arctan2(hit_pts[..., 1], hit_pts[..., 0])
    spiral_band = 0.5 + 0.5 * np.sin(omega * hit_pts[..., 2] + 2.0 * hit_ang)
    moisture = np.clip((-hit_pts[..., 2]) / 600.0, 0.0, 1.0) * 0.35
    refl = np.where(valid, np.clip(0.25 + 0.5 * spiral_band + moisture, 0.0, 1.0), 0.0)
    return log_depth_u8, np.round(refl * 255.0).astype(np.uint8), valid


def backproject_planar_depth(
    log_depth_u8: np.ndarray, rgb: Optional[np.ndarray],
    fx: float, fy: float, cx: float, cy: float,
    z_min: float, z_max: float,
    cam_position: Tuple[float, float, float], cam_quaternion_xyzw: Tuple[float, float, float, float],
    max_points: int = 60000, seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Unproject valid pixels to world points (+ optional colors). Subsamples to max_points."""
    planar = decode_log_depth_png(log_depth_u8, z_min, z_max)
    ys, xs = np.nonzero(planar > 0)
    if len(xs) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3))
    if len(xs) > max_points:
        rng = np.random.default_rng(seed)
        sel = rng.choice(len(xs), max_points, replace=False)
        xs, ys = xs[sel], ys[sel]
    z = planar[ys, xs]
    x_cam = ((xs.astype(np.float32) - cx) / fx) * z
    y_cam = -((ys.astype(np.float32) - cy) / fy) * z
    pts_cam = np.stack([x_cam, y_cam, -z], axis=-1)
    qx, qy, qz, qw = cam_quaternion_xyzw
    # quat rotate (vectorized)
    uvw = np.stack([qx, qy, qz])
    t = 2.0 * np.cross(uvw, pts_cam)
    pts_world = pts_cam + qw * t + np.cross(uvw, t) + np.asarray(cam_position, dtype=np.float32)
    if rgb is not None:
        cols = rgb[ys, xs].astype(np.float32) / 255.0
    else:
        d = np.clip((z - z_min) / max(z_max - z_min, 1e-6), 0, 1)
        cols = np.stack([np.zeros_like(d), 0.94 * (0.35 + 0.65 * d), (0.35 + 0.65 * d)], axis=-1)
    return pts_world.astype(np.float32), cols.astype(np.float32)
