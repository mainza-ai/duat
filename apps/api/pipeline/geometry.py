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
           "degrade_for_sweep", "load_mesh", "save_mesh"]


def degrade_for_sweep(mesh: trimesh.Trimesh, levels: List[Tuple[str, float, float]]) -> List[Tuple[str, trimesh.Trimesh]]:
    """Blur+quantize vertex positions: [(name, sigma_m, quantum_m)] ladder LiDAR->GPR->muon."""
    out = []
    v = mesh.vertices.astype(np.float64)
    for name, sigma, quantum in levels:
        dull = v.copy()
        if sigma > 0:
            dull = gaussian_filter(dull, sigma=(sigma, 0)) if dull.ndim == 2 else dull
        if quantum > 0:
            dull = np.round(dull / quantum) * quantum
        m = trimesh.Trimesh(vertices=dull, faces=mesh.faces, process=False)
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
