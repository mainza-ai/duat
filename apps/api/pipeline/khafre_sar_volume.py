"""
duat/apps/api/pipeline/khafre_sar_volume.py

Volumetric SAR Tomography Ingestor, Procedural Khafre Subterranean Anomaly
Synthesizer, and Posterior Cavity Mesh Extractor for Project Duat.

Compliant with the live Marble 2 / Atlas API:
- Helical cameras emitted as nested Atlas PinholeCamera dicts
  {intrinsics:{fx,fy,cx,cy,width,height}, extrinsics:{position, quaternion XYZW,
  coordinateSystem:'rub'}} at 1280x720.
- Depth handled by depth_renderer (official INVERTED log-depth: 255=zMin).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import trimesh
from scipy.ndimage import gaussian_filter
from skimage.measure import marching_cubes

from .shaft_model import ShaftParams, shaft_void_sdf, spiral_feature_proximity


@dataclass
class VolumeMetadata:
    """Physical coordinate bounds and voxel gauge for the Epistemic Stack."""

    grid_shape: Tuple[int, int, int]  # (nx, ny, nz)
    bounds_min_m: Tuple[float, float, float]
    bounds_max_m: Tuple[float, float, float]
    voxel_size_m: Tuple[float, float, float]
    iso_level: float = 0.0
    sensor_uncertainty_m: float = 0.35
    # Cited outlet variance for the basal chambers (80 m per JPost/Ahram vs
    # 90 m per NatGeo/TRT); None when the bundle did not come from the synth.
    chamber_size_m: Optional[float] = None


@dataclass
class EpistemicVolumeBundle:
    """Tier 1 SDF + Tier 2 reflectivity scalar fields."""

    sdf_volume: np.ndarray
    reflectivity_volume: np.ndarray
    metadata: VolumeMetadata


class KhafreSARVolumeEngine:
    """Procedural Khafre anomaly synthesizer + posterior cavity extractor."""

    def __init__(
        self,
        grid_shape: Tuple[int, int, int] = (160, 160, 240),
        bounds_min_m: Tuple[float, float, float] = (-100.0, -100.0, -650.0),
        bounds_max_m: Tuple[float, float, float] = (100.0, 100.0, 0.0),
    ):
        self.grid_shape = grid_shape
        self.bounds_min_m = bounds_min_m
        self.bounds_max_m = bounds_max_m
        nx, ny, nz = grid_shape
        dx = (bounds_max_m[0] - bounds_min_m[0]) / max(nx - 1, 1)
        dy = (bounds_max_m[1] - bounds_min_m[1]) / max(ny - 1, 1)
        dz = (bounds_max_m[2] - bounds_min_m[2]) / max(nz - 1, 1)
        self.voxel_size_m = (dx, dy, dz)

    # ------------------------------------------------------------------
    def synthesize_khafre_subterranean_complex(
        self,
        shaft_radius_m: float = 5.5,
        spiral_pitch_m: float = 28.0,
        ramp_width_m: float = 2.0,
        ramp_thickness_m: float = 0.8,
        wall_relief_m: float = 0.0,
        spiral_style: str = "interior_ramp",
        chamber_size_m: float = 80.0,
        seed: int = 42,
    ) -> EpistemicVolumeBundle:
        rng = np.random.default_rng(seed)
        nx, ny, nz = self.grid_shape
        x = np.linspace(self.bounds_min_m[0], self.bounds_max_m[0], nx, dtype=np.float32)
        y = np.linspace(self.bounds_min_m[1], self.bounds_max_m[1], ny, dtype=np.float32)
        z = np.linspace(self.bounds_min_m[2], self.bounds_max_m[2], nz, dtype=np.float32)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

        sdf = np.full(self.grid_shape, 100.0, dtype=np.float32)

        shaft_x_coords = [-40.0, 40.0]
        shaft_y_coords = [-60.0, -20.0, 20.0, 60.0]
        params = ShaftParams(
            radius_m=shaft_radius_m, spiral_pitch_m=spiral_pitch_m,
            ramp_width_m=ramp_width_m, ramp_thickness_m=ramp_thickness_m,
            wall_relief_m=wall_relief_m, spiral_style=spiral_style)
        for sx in shaft_x_coords:
            for sy in shaft_y_coords:
                shaft_sdf = shaft_void_sdf(X, Y, Z, sx, sy, params)
                sdf = np.minimum(sdf, shaft_sdf)

        chamber_centers = [(-40.0, 0.0, -565.0), (40.0, 0.0, -565.0)]
        half_xy = chamber_size_m / 2.0
        half_z = 25.0
        for cx, cy, cz in chamber_centers:
            q_x = np.abs(X - cx) - half_xy
            q_y = np.abs(Y - cy) - half_xy
            q_z = np.abs(Z - cz) - half_z
            box_sdf = np.minimum(np.maximum(q_x, np.maximum(q_y, q_z)), 0.0) + np.sqrt(
                np.maximum(q_x, 0.0) ** 2
                + np.maximum(q_y, 0.0) ** 2
                + np.maximum(q_z, 0.0) ** 2
            )
            sdf = np.minimum(sdf, box_sdf)

        for sx in shaft_x_coords:
            corridor_sdf = np.maximum(
                np.abs(X - sx) - 3.5,
                np.maximum(np.abs(Y) - 65.0, np.abs(Z - (-45.0)) - 4.0),
            )
            sdf = np.minimum(sdf, corridor_sdf)

        water_table_boost = np.exp(-((Z - (-585.0)) / 12.0) ** 2) * 0.45
        boundary_proximity = np.exp(-(sdf / 4.0) ** 2) * 0.40
        # Feature dressing evaluated only near shafts (12 m box): the full-grid
        # alternative costs minutes of transcendentals for identical output.
        feat_dressing = np.zeros(self.grid_shape, dtype=np.float32)
        for sx in shaft_x_coords:
            for sy in shaft_y_coords:
                near = ((np.abs(X - sx) < 12.0) & (np.abs(Y - sy) < 12.0)
                        & (Z > -600.0) & (Z < 0.0))
                if np.any(near):
                    prox = spiral_feature_proximity(
                        X[near], Y[near], Z[near], sx, sy, params)
                    feat_dressing[near] = np.maximum(feat_dressing[near], prox)
        feat_dressing = feat_dressing * 0.25
        stratigraphy_bands = 0.10 * (0.5 + 0.5 * np.sin(Z * 0.08))
        noise_field = gaussian_filter(
            rng.normal(0.0, 0.08, size=self.grid_shape).astype(np.float32), sigma=1.5
        )
        reflectivity = np.clip(
            0.15 + boundary_proximity + water_table_boost + stratigraphy_bands
            + feat_dressing + noise_field,
            0.0, 1.0,
        ).astype(np.float32)

        metadata = VolumeMetadata(
            grid_shape=self.grid_shape,
            bounds_min_m=self.bounds_min_m,
            bounds_max_m=self.bounds_max_m,
            voxel_size_m=self.voxel_size_m,
            iso_level=0.0,
            sensor_uncertainty_m=0.45,
            chamber_size_m=chamber_size_m,
        )
        return EpistemicVolumeBundle(sdf_volume=sdf, reflectivity_volume=reflectivity, metadata=metadata)

    # ------------------------------------------------------------------
    def reconstruct_from_2d_sar_slices(
        self,
        slice_paths: List[Path],
        slice_axis: str = "y",
        void_intensity_threshold: float = 0.42,
    ) -> EpistemicVolumeBundle:
        if not slice_paths:
            raise ValueError("slice_paths must contain at least one image path.")
        nx, ny, nz = self.grid_shape
        loaded_slices = []
        for p in slice_paths:
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"Could not read SAR slice image: {p}")
            # True jet-transfer inversion (slices.decode_jet_anomaly), not HSV.
            from .slices import decode_jet_anomaly

            anomaly_response, _slice_valid = decode_jet_anomaly(img)
            target_w = nx if slice_axis.lower() == "y" else ny
            resized = cv2.resize(anomaly_response, (target_w, nz), interpolation=cv2.INTER_CUBIC)
            loaded_slices.append(np.flipud(resized).T)
        stack = np.stack(loaded_slices, axis=1 if slice_axis.lower() == "y" else 0)
        from scipy.ndimage import zoom

        zoom_factors = (nx / stack.shape[0], ny / stack.shape[1], nz / stack.shape[2])
        reflectivity = np.clip(zoom(stack, zoom_factors, order=1), 0.0, 1.0).astype(np.float32)
        smoothed = gaussian_filter(reflectivity, sigma=1.2)
        sdf = (void_intensity_threshold - smoothed) * 15.0
        metadata = VolumeMetadata(
            grid_shape=self.grid_shape, bounds_min_m=self.bounds_min_m,
            bounds_max_m=self.bounds_max_m, voxel_size_m=self.voxel_size_m,
            iso_level=0.0, sensor_uncertainty_m=0.60,
        )
        return EpistemicVolumeBundle(sdf_volume=sdf, reflectivity_volume=reflectivity, metadata=metadata)

    # ------------------------------------------------------------------
    def extract_posterior_cavity_mesh(
        self,
        bundle: EpistemicVolumeBundle,
        posterior_seed: Optional[int] = None,
        uncertainty_scale: float = 1.0,
        roi_bounds_min_m: Optional[Tuple[float, float, float]] = None,
        roi_bounds_max_m: Optional[Tuple[float, float, float]] = None,
    ) -> trimesh.Trimesh:
        sdf = bundle.sdf_volume.copy()
        meta = bundle.metadata
        if posterior_seed is not None and uncertainty_scale > 0.0:
            rng = np.random.default_rng(posterior_seed)
            raw_noise = rng.normal(0.0, meta.sensor_uncertainty_m * uncertainty_scale, size=sdf.shape).astype(np.float32)
            sdf = sdf + gaussian_filter(raw_noise, sigma=1.6)

        origin_m = np.array(meta.bounds_min_m, dtype=np.float64)
        spacing_m = np.array(meta.voxel_size_m, dtype=np.float64)

        if roi_bounds_min_m is not None and roi_bounds_max_m is not None:
            idx_min = np.clip(np.floor((np.array(roi_bounds_min_m) - origin_m) / spacing_m).astype(int), 0, np.array(meta.grid_shape) - 2)
            idx_max = np.clip(np.ceil((np.array(roi_bounds_max_m) - origin_m) / spacing_m).astype(int), idx_min + 2, np.array(meta.grid_shape))
            sdf_sub = sdf[idx_min[0]:idx_max[0], idx_min[1]:idx_max[1], idx_min[2]:idx_max[2]]
            refl_sub = bundle.reflectivity_volume[idx_min[0]:idx_max[0], idx_min[1]:idx_max[1], idx_min[2]:idx_max[2]]
            origin_m = origin_m + idx_min * spacing_m
        else:
            sdf_sub = sdf
            refl_sub = bundle.reflectivity_volume

        verts_idx, faces, normals, _ = marching_cubes(
            sdf_sub, level=meta.iso_level, spacing=tuple(spacing_m), gradient_direction="ascent",
        )
        verts_world = verts_idx + origin_m
        faces_inward = faces[:, ::-1]
        normals_inward = -normals

        # Correct world->grid index mapping for reflectivity sampling
        grid_idx = np.clip(np.round((verts_world - origin_m) / spacing_m).astype(int), 0, np.array(sdf_sub.shape) - 1)
        vertex_reflectivity = refl_sub[grid_idx[:, 0], grid_idx[:, 1], grid_idx[:, 2]]

        colors = np.zeros((len(verts_world), 4), dtype=np.uint8)
        colors[:, 0] = np.clip(vertex_reflectivity * 255.0, 0, 255).astype(np.uint8)
        colors[:, 1] = np.clip((1.0 - np.abs(vertex_reflectivity - 0.5)) * 220.0, 0, 255).astype(np.uint8)
        colors[:, 2] = np.clip((1.0 - vertex_reflectivity) * 255.0, 0, 255).astype(np.uint8)
        colors[:, 3] = 255

        mesh = trimesh.Trimesh(vertices=verts_world, faces=faces_inward,
                               vertex_normals=normals_inward, vertex_colors=colors, process=False)
        mesh.vertex_attributes["reflectivity"] = vertex_reflectivity.astype(np.float32)
        return mesh

    # ------------------------------------------------------------------
    def generate_helical_descent_cameras(
        self,
        shaft_center_xy: Tuple[float, float] = (-40.0, -20.0),
        z_start_m: float = -30.0,
        z_end_m: float = -180.0,
        num_cameras: int = 32,
        orbit_radius_m: float = 2.2,
        coordinate_system: str = "rub",
    ) -> List[Dict[str, Any]]:
        """Atlas-compliant nested PinholeCamera dicts (1280x720, rub)."""
        num_cameras = min(max(num_cameras, 1), 32)
        width, height = 1280, 720
        fov_deg = 70.0
        fx = round(float((width / 2.0) / np.tan(np.radians(fov_deg / 2.0))), 2)
        z_vals = np.linspace(z_start_m, z_end_m, num_cameras)
        angles = np.linspace(0.0, 2.0 * np.pi, num_cameras)
        cameras: List[Dict[str, Any]] = []
        for z_pos, ang in zip(z_vals, angles):
            cx = shaft_center_xy[0] + orbit_radius_m * np.cos(ang)
            cy = shaft_center_xy[1] + orbit_radius_m * np.sin(ang)
            half_yaw = (ang + np.pi * 0.75) * 0.5
            quat = [0.0, 0.0, round(float(np.sin(half_yaw)), 5), round(float(np.cos(half_yaw)), 5)]
            cameras.append({
                "intrinsics": {"fx": fx, "fy": fx, "cx": width / 2.0, "cy": height / 2.0,
                               "width": width, "height": height},
                "extrinsics": {"position": [round(float(cx), 3), round(float(cy), 3), round(float(z_pos), 3)],
                               "quaternion": quat, "coordinateSystem": coordinate_system},
            })
        return cameras


def _box_sdf_field(X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                   center: Tuple[float, float, float],
                   size: Tuple[float, float, float]) -> np.ndarray:
    qx = np.abs(X - center[0]) - size[0] / 2.0
    qy = np.abs(Y - center[1]) - size[1] / 2.0
    qz = np.abs(Z - center[2]) - size[2] / 2.0
    return (np.minimum(np.maximum(qx, np.maximum(qy, qz)), 0.0)
            + np.sqrt(np.maximum(qx, 0.0) ** 2 + np.maximum(qy, 0.0) ** 2
                      + np.maximum(qz, 0.0) ** 2))


# Validated-void benchmarks (peer-reviewed muography): the fidelity anchors
# against which contested-claim renders are judged. Dimensions from the
# Nature papers (see claims scanpyramids-2017 / scanpyramids-2023).
REFERENCE_VOIDS: Dict[str, Dict[str, Any]] = {
    "nfc": {"dims_m": (9.0, 2.0, 2.0), "claim_id": "scanpyramids-2023",
            "bounds": ((-12.0, -12.0, -12.0), (12.0, 12.0, 12.0)),
            "grid": (64, 64, 64), "uncertainty_m": 0.05},
    "big-void": {"dims_m": (30.0, 6.0, 6.0), "claim_id": "scanpyramids-2017",
                 "bounds": ((-20.0, -8.0, -8.0), (20.0, 8.0, 8.0)),
                 "grid": (100, 40, 40), "uncertainty_m": 0.10},
}


def synthesize_reference_void(kind: str = "nfc") -> EpistemicVolumeBundle:
    """Procedural bundle of a VALIDATED void (box approximation at cited dims)."""
    if kind not in REFERENCE_VOIDS:
        raise ValueError(f"Unknown reference void: {kind}")
    spec = REFERENCE_VOIDS[kind]
    lo, hi = spec["bounds"]
    nx, ny, nz = spec["grid"]
    x = np.linspace(lo[0], hi[0], nx, dtype=np.float32)
    y = np.linspace(lo[1], hi[1], ny, dtype=np.float32)
    z = np.linspace(lo[2], hi[2], nz, dtype=np.float32)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    sdf = _box_sdf_field(X, Y, Z, (0.0, 0.0, 0.0), spec["dims_m"]).astype(np.float32)
    reflectivity = np.clip(0.15 + 0.45 * np.exp(-(sdf / 1.5) ** 2), 0.0, 1.0).astype(np.float32)
    metadata = VolumeMetadata(
        grid_shape=spec["grid"], bounds_min_m=lo, bounds_max_m=hi,
        voxel_size_m=((hi[0] - lo[0]) / (nx - 1), (hi[1] - lo[1]) / (ny - 1),
                      (hi[2] - lo[2]) / (nz - 1)),
        iso_level=0.0, sensor_uncertainty_m=spec["uncertainty_m"], chamber_size_m=None)
    return EpistemicVolumeBundle(sdf_volume=sdf, reflectivity_volume=reflectivity, metadata=metadata)


if __name__ == "__main__":  # pragma: no cover
    out_dir = Path("data/khafre_showcase")
    out_dir.mkdir(parents=True, exist_ok=True)
    engine = KhafreSARVolumeEngine()
    print("[Duat] Synthesizing Khafre subterranean SAR volume ...")
    bundle = engine.synthesize_khafre_subterranean_complex(seed=42)
    full_mesh = engine.extract_posterior_cavity_mesh(bundle, posterior_seed=None)
    full_mesh.export(str(out_dir / "khafre_full_complex.glb"))
    print(f"[Duat] Exported full complex mesh ({len(full_mesh.faces):,} faces)")
    for i, seed in enumerate([101, 202, 303], start=1):
        m = engine.extract_posterior_cavity_mesh(bundle, posterior_seed=seed,
            roi_bounds_min_m=(-52.0, -32.0, -220.0), roi_bounds_max_m=(-28.0, -8.0, -15.0))
        m.export(str(out_dir / f"khafre_shaft_posterior_{i}.ply"))
    with open(out_dir / "khafre_descent_manifest.json", "w") as f:
        json.dump({"metadata": asdict(bundle.metadata),
                   "targetCameras": engine.generate_helical_descent_cameras()}, f, indent=2)
    print("[Duat] Saved 32-camera descent manifest")
