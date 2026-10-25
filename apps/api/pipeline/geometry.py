"""
duat/apps/api/pipeline/geometry.py

Facade (keeps brief import path `pipeline.geometry` working) + degradation
sweep harness + mesh I/O helpers.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import trimesh
from scipy.ndimage import gaussian_filter

from .khafre_sar_volume import EpistemicVolumeBundle, KhafreSARVolumeEngine, VolumeMetadata

__all__ = ["KhafreSARVolumeEngine", "EpistemicVolumeBundle", "VolumeMetadata",
           "degrade_for_sweep", "load_mesh", "save_mesh", "mesh_to_bundle"]


def mesh_to_bundle(
    mesh: trimesh.Trimesh,
    grid_shape: Tuple[int, int, int] = (96, 96, 96),
    padding_m: float = 2.0,
    sensor_uncertainty_m: float = 0.35,
) -> EpistemicVolumeBundle:
    """Convert a raw surface mesh (.obj/.ply/.glb/.stl) into an EpistemicVolumeBundle.

    Samples a signed-distance field on a regular grid via vectorized
    closest-point queries (chunked for memory), then derives a Tier-2
    reflectivity proxy from vertex colors when present, else from boundary
    proximity. SDF convention matches the synthesizer: < 0 inside void.
    """
    from trimesh.proximity import signed_distance

    mesh = mesh.copy()
    for cleanup in ("remove_duplicate_faces", "remove_infinite_values", "fill_holes"):
        try:
            getattr(mesh, cleanup)()
        except Exception:
            pass

    lo, hi = mesh.bounds
    lo = lo - padding_m
    hi = hi + padding_m
    nx, ny, nz = grid_shape
    x = np.linspace(lo[0], hi[0], nx)
    y = np.linspace(lo[1], hi[1], ny)
    z = np.linspace(lo[2], hi[2], nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1).astype(np.float64)

    sdf = np.empty(pts.shape[0], dtype=np.float64)
    chunk = 1 << 16
    for i in range(0, pts.shape[0], chunk):
        sdf[i:i + chunk] = signed_distance(mesh, pts[i:i + chunk])
    # trimesh signed_distance is POSITIVE inside / NEGATIVE outside; the bundle
    # convention is < 0 inside the void, so flip (surface stays at iso 0).
    sdf = -sdf.reshape(grid_shape).astype(np.float32)

    if mesh.visual is not None and hasattr(mesh.visual, "vertex_colors"):
        try:
            vc = np.asarray(mesh.visual.vertex_colors)[..., :3].astype(np.float32) / 255.0
            refl_scalar = float(np.clip(vc.mean(), 0.0, 1.0))
        except Exception:
            refl_scalar = 0.5
        reflectivity = np.full(grid_shape, refl_scalar, dtype=np.float32)
    else:
        boundary = np.exp(-(sdf / max(padding_m, 1e-6)) ** 2).astype(np.float32)
        reflectivity = np.clip(0.15 + 0.45 * boundary, 0.0, 1.0)

    dx = (hi[0] - lo[0]) / max(nx - 1, 1)
    dy = (hi[1] - lo[1]) / max(ny - 1, 1)
    dz = (hi[2] - lo[2]) / max(nz - 1, 1)
    metadata = VolumeMetadata(
        grid_shape=grid_shape, bounds_min_m=tuple(lo.tolist()),
        bounds_max_m=tuple(hi.tolist()), voxel_size_m=(dx, dy, dz),
        iso_level=0.0, sensor_uncertainty_m=sensor_uncertainty_m)
    return EpistemicVolumeBundle(sdf_volume=sdf, reflectivity_volume=reflectivity, metadata=metadata)


def degrade_for_sweep(mesh: trimesh.Trimesh, levels: List[Tuple[str, float, float]]) -> List[Tuple[str, trimesh.Trimesh]]:
    """3D spatial Gaussian blur + voxel-clustering ladder: LiDAR → SAR → GPR → muon.

    Level sigmas are modeling choices anchored to validated sensing figures
    (ScanPyramids: ~3 m voids resolvable with ~2 yr muon exposure; ERT/NFC work
    at meter-to-centimeter scale), NOT to the contested orbital-SAR precision
    claims (see claims khafre-2025-press physical_notes)."""
    from scipy.spatial import cKDTree

    out: List[Tuple[str, trimesh.Trimesh]] = []
    v = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if len(v) == 0:
        return [(name, mesh.copy()) for name, _, _ in levels]

    k_nn = min(24, len(v))
    tree = cKDTree(v)
    dists, nn_idx = tree.query(v, k=k_nn)
    if k_nn == 1:
        dists = dists[:, None]
        nn_idx = nn_idx[:, None]
    median_nn = float(np.median(dists[:, 1])) if k_nn > 1 else 0.0

    for name, sigma, qstep in levels:
        dull = v.copy()
        if sigma > 0.0:
            # True 3D spatial Gaussian kernel over k-nearest surface neighbors
            w = np.exp(-0.5 * (dists / max(float(sigma), 1e-6)) ** 2)
            w_sum = np.maximum(np.sum(w, axis=1, keepdims=True), 1e-12)
            w = w / w_sum
            dull = np.einsum("ij,ijk->ik", w, v[nn_idx])
        if qstep > 0.0:
            metric_dull = np.round(dull / qstep) * qstep
            rms_val = float(np.sqrt(np.mean(np.sum((metric_dull - v) ** 2, axis=-1))))
            eff_qstep = max(float(qstep), median_nn * (0.85 + 1.5 * float(qstep)))
            clustered = np.round(dull / eff_qstep) * eff_qstep
            # Deduplicate vertices sharing a quantized 3D voxel cell and drop degenerate triangles
            uniq_v, inv = np.unique(clustered, axis=0, return_inverse=True)
            new_faces = inv[faces]
            non_deg = (
                (new_faces[:, 0] != new_faces[:, 1])
                & (new_faces[:, 1] != new_faces[:, 2])
                & (new_faces[:, 2] != new_faces[:, 0])
            )
            new_faces = new_faces[non_deg]
            m = trimesh.Trimesh(vertices=uniq_v, faces=new_faces, process=False)
        else:
            rms_val = float(np.sqrt(np.mean(np.sum((dull - v) ** 2, axis=-1))))
            m = trimesh.Trimesh(vertices=dull, faces=faces.copy(), process=False)
        # Preserve per-original-vertex RMS displacement in meters
        m.metadata["rms_displacement_m"] = rms_val
        out.append((name, m))
    return out


def load_mesh(path: str | Path) -> trimesh.Trimesh:
    return trimesh.load(str(path), process=False)


def save_mesh(mesh: trimesh.Trimesh, path: str | Path) -> str:
    mesh.export(str(path))
    return str(path)


DEFAULT_SWEEP: List[Tuple[str, float, float]] = [
    ("lidar", 0.0, 0.0),
    ("sar_high", 0.15, 0.05),
    ("gpr", 0.45, 0.25),
    ("muon", 1.20, 0.80),
]
