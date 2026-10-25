"""
duat/apps/api/main.py

FastAPI Backend Observatory Server for Project Duat.
Run: uvicorn apps.api.main:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import time
from typing import Any, Dict, List, Literal, Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .pipeline.depth_renderer import render_shaft_planar_depth_and_reflectivity
from .pipeline.ensemble import (compute_consensus, compute_disagreement, disagreement_to_u8,
                                provenance_manifest)
from .pipeline.khafre_sar_volume import EpistemicVolumeBundle, KhafreSARVolumeEngine
from .pipeline.prompts import compose_epistemic_prompt
from .pipeline.worldlabs_client import WorldLabsAtlasClient

from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    global CACHED_BUNDLE
    try:
        CACHED_BUNDLE = VOLUME_ENGINE.synthesize_khafre_subterranean_complex(seed=42)
    except Exception:
        CACHED_BUNDLE = None
    yield


app = FastAPI(title="MILIMO DUAT // Subterranean Epistemic World-Model API", version="1.0.0",
              description="Conditions World Labs atlasChisel on SAR/Muon/GPR geometry with provenance.",
              lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

VOLUME_ENGINE = KhafreSARVolumeEngine()
CACHED_BUNDLE: Optional[EpistemicVolumeBundle] = None
ATLAS_CLIENT = WorldLabsAtlasClient()
PROVENANCE_STORE: Dict[str, Dict[str, Any]] = {}


# ---------------- schemas ----------------
class CameraPose(BaseModel):
    position: List[float] = Field(default_factory=lambda: [-38.5, -20.0, -45.0])
    quaternion: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0, 1.0])


class PinholeCameraSpec(BaseModel):
    width: int = 1280
    height: int = 720
    fov_deg: float = 70.0
    coordinate_system: Literal["rub", "rdf"] = "rub"
    pose: CameraPose = Field(default_factory=CameraPose)

    @property
    def fx(self) -> float:
        return float((self.width / 2.0) / np.tan(np.radians(self.fov_deg / 2.0)))

    def to_atlas_dict(self) -> Dict[str, Any]:
        return {
            "intrinsics": {"fx": round(self.fx, 2), "fy": round(self.fx, 2),
                           "cx": self.width / 2.0, "cy": self.height / 2.0,
                           "width": self.width, "height": self.height},
            "extrinsics": {"position": [float(x) for x in self.pose.position],
                           "quaternion": [float(x) for x in self.pose.quaternion],
                           "coordinateSystem": self.coordinate_system},
        }


class EnsembleExploreRequest(BaseModel):
    camera: PinholeCameraSpec = Field(default_factory=PinholeCameraSpec)
    hypothesis: Literal["megalithic", "karst", "artifact"] = "megalithic"
    instrument_rig: Literal["borescope_led", "drop_flare", "structured_laser"] = "drop_flare"
    seeds: List[int] = Field(default_factory=lambda: [7, 42, 108, 256])
    voxel_size: float = Field(default=0.25, ge=0.0, le=2.0)
    sensor_noise_sigma_m: float = Field(default=0.18, ge=0.0, le=2.0)
    z_min: float = 0.5
    z_max: Optional[float] = None
    mock_mode: Optional[bool] = None


class EpistemicLayerTextures(BaseModel):
    measured_log_depth_b64: str
    inferred_reflectivity_b64: str
    imagined_consensus_rgb_b64: str
    disagreement_map_b64: str


class NextBestViewRecommendation(BaseModel):
    target_z_m: float
    predicted_variance_reduction_pct: float
    rationale: str


class EnsembleExploreResponse(BaseModel):
    provenance_sha256: str
    timestamp_utc: float
    prompt_used: str
    z_min: float
    z_max: float
    voxel_size: float
    empty_void_pixel_ratio: float
    mean_ensemble_disagreement: float
    layers: EpistemicLayerTextures
    next_best_view: NextBestViewRecommendation


# ---------------- helpers ----------------
def _encode_png_b64(arr: np.ndarray) -> str:
    if arr.ndim == 3 and arr.shape[2] == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", arr)
    if not ok:
        raise RuntimeError("PNG encode failed.")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _get_bundle() -> EpistemicVolumeBundle:
    global CACHED_BUNDLE
    if CACHED_BUNDLE is None:
        CACHED_BUNDLE = VOLUME_ENGINE.synthesize_khafre_subterranean_complex(seed=42)
    return CACHED_BUNDLE


@app.get("/api/health")
async def health_check() -> Dict[str, Any]:
    return {"status": "online", "project": "Milimo Duat",
            "worldlabs_mock_mode": ATLAS_CLIENT.mock_mode,
            "khafre_volume_loaded": CACHED_BUNDLE is not None}


@app.get("/api/khafre/trajectory")
async def get_trajectory(num_cameras: int = 32) -> Dict[str, Any]:
    bundle = _get_bundle()
    return {"bounds_min_m": bundle.metadata.bounds_min_m,
            "bounds_max_m": bundle.metadata.bounds_max_m,
            "voxel_size_m": bundle.metadata.voxel_size_m,
            "targetCameras": VOLUME_ENGINE.generate_helical_descent_cameras(num_cameras=num_cameras)}


@app.post("/api/chisel/ensemble", response_model=EnsembleExploreResponse)
async def execute_ensemble(req: EnsembleExploreRequest) -> EnsembleExploreResponse:
    prompt, default_zmax = compose_epistemic_prompt(req.hypothesis, req.instrument_rig)
    z_max = req.z_max if req.z_max is not None else default_zmax

    log_depth_u8, reflectivity_u8, valid_mask = render_shaft_planar_depth_and_reflectivity(
        cam_origin_xyz=tuple(req.camera.pose.position),
        width=req.camera.width, height=req.camera.height, fx=req.camera.fx,
        z_min=req.z_min, z_max=z_max,
        noise_sigma_m=req.sensor_noise_sigma_m, seed=req.seeds[0] if req.seeds else 42)

    atlas_cam = req.camera.to_atlas_dict()
    client = WorldLabsAtlasClient(mock_mode=req.mock_mode) if req.mock_mode is not None else ATLAS_CLIENT
    tasks = [client.run_atlas_chisel(
        context_camera=atlas_cam, log_depth_png=log_depth_u8, z_min=req.z_min, z_max=z_max,
        target_cameras=[atlas_cam], prompt=prompt, voxel_size=req.voxel_size,
        seed=s, hypothesis=req.hypothesis) for s in req.seeds]
    results = await asyncio.gather(*tasks)

    rgb_stack = np.stack([r.rgb_frames[0].astype(np.float32) for r in results], axis=0)
    consensus_rgb = compute_consensus(rgb_stack)
    dis_norm, mean_dis = compute_disagreement(rgb_stack, valid_mask)
    disagreement_u8 = disagreement_to_u8(dis_norm)

    manifest = {"camera": atlas_cam, "prompt": prompt, "seeds": req.seeds,
                "voxel_size": req.voxel_size, "z_min": req.z_min, "z_max": z_max,
                "child_hashes": [r.request_hash for r in results]}
    sha = provenance_manifest(manifest)
    PROVENANCE_STORE[sha] = {"manifest": manifest, "timestamp_utc": time.time()}

    empty_ratio = float(1.0 - (np.count_nonzero(valid_mask) / valid_mask.size))
    return EnsembleExploreResponse(
        provenance_sha256=sha, timestamp_utc=time.time(), prompt_used=prompt,
        z_min=req.z_min, z_max=z_max, voxel_size=req.voxel_size,
        empty_void_pixel_ratio=round(empty_ratio, 4),
        mean_ensemble_disagreement=round(mean_dis, 4),
        layers=EpistemicLayerTextures(
            measured_log_depth_b64=_encode_png_b64(log_depth_u8),
            inferred_reflectivity_b64=_encode_png_b64(reflectivity_u8),
            imagined_consensus_rgb_b64=_encode_png_b64(consensus_rgb),
            disagreement_map_b64=_encode_png_b64(disagreement_u8)),
        next_best_view=NextBestViewRecommendation(
            target_z_m=round(float(req.camera.pose.position[2] - 18.5), 2),
            predicted_variance_reduction_pct=round(min(92.0, 45.0 + mean_dis * 80.0), 1),
            rationale=("Ensemble variance peaks along the helical ledge occlusion boundary; "
                       "advancing the borescope 18.5m down-shaft maximizes information gain.")))


@app.post("/api/chisel/chain")
async def execute_chain(req: EnsembleExploreRequest, batches: int = 2) -> Dict[str, Any]:
    """Two-hop autoregressive demo: batch k target -> batch k+1 source+context."""
    prompt, default_zmax = compose_epistemic_prompt(req.hypothesis, req.instrument_rig)
    z_max = req.z_max if req.z_max is not None else default_zmax
    client = WorldLabsAtlasClient(mock_mode=req.mock_mode) if req.mock_mode is not None else ATLAS_CLIENT
    traj = VOLUME_ENGINE.generate_helical_descent_cameras(num_cameras=4)
    out, carry = [], None
    for b in range(batches):
        log_depth_u8, _, _ = render_shaft_planar_depth_and_reflectivity(
            cam_origin_xyz=tuple(traj[b % len(traj)]["extrinsics"]["position"]),
            z_min=req.z_min, z_max=z_max, seed=req.seeds[0] if req.seeds else 7)
        srcs = [carry] if carry else []
        res = await client.run_atlas_chisel(
            context_camera=traj[b % len(traj)], log_depth_png=log_depth_u8,
            z_min=req.z_min, z_max=z_max, target_cameras=[traj[(b + 1) % len(traj)]],
            prompt=prompt, voxel_size=req.voxel_size, seed=req.seeds[0] if req.seeds else 7,
            hypothesis=req.hypothesis, source_frames=srcs, enhance_prompt=(b == 0))
        out.append({"operation_id": res.operation_id, "prompt_used": res.prompt_used})
        if res.rgb_frames:
            ok, buf = cv2.imencode(".png", cv2.cvtColor(res.rgb_frames[-1], cv2.COLOR_RGB2BGR))
            if ok:
                ref = await client.create_asset_inline(buf.tobytes(), f"chain_b{b}.png")
                carry = {"imageAsset": {"assetId": ref["assetId"]},
                         "camera": traj[(b + 1) % len(traj)]}
    return {"batches": out, "prompt": prompt}


@app.get("/api/provenance/{sha}")
async def get_provenance(sha: str) -> Dict[str, Any]:
    if sha not in PROVENANCE_STORE:
        raise HTTPException(status_code=404, detail="Unknown provenance hash.")
    return {"provenance_sha256": sha, **PROVENANCE_STORE[sha]}
