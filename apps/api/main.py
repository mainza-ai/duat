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
import os
import shutil
import time
import uuid
from typing import Any, Callable, Dict, List, Literal, Optional

import cv2
import numpy as np
import trimesh
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .pipeline import store as store

from .pipeline.depth_renderer import render_shaft_planar_depth_and_reflectivity
from .pipeline.ensemble import (compute_consensus, compute_disagreement, disagreement_to_u8,
                                provenance_manifest)
from .pipeline.khafre_sar_volume import EpistemicVolumeBundle, KhafreSARVolumeEngine
from .pipeline.prompts import compose_epistemic_prompt
from .pipeline.worldlabs_client import AtlasError, WorldLabsAtlasClient

from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Production: never synthesize the volume at boot. It is built lazily on
    # first use (see _get_bundle) so deploys stay fast and small instances
    # don't OOM before serving traffic. Only the DB is initialized here.
    store.init_db()
    yield


app = FastAPI(title="MILIMO DUAT // Subterranean Epistemic World-Model API", version="1.0.0",
              description="Conditions World Labs atlasChisel on SAR/Muon/GPR geometry with provenance.",
              lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=[o.strip() for o in os.getenv(
    "DUAT_CORS_ORIGINS",
    "http://localhost:5173,http://localhost:5174,http://127.0.0.1:5173,http://127.0.0.1:5174"
).split(",") if o.strip()], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

OPERATOR_KEY = os.getenv("OPERATOR_API_KEY", "")
if not OPERATOR_KEY:
    print("[Duat] WARNING: OPERATOR_API_KEY unset — API is open. Set it in production.")

# In-memory sliding-window rate limiter (per IP). Production single-instance
# default: 30 expensive POSTs/min; tune via DUAT_RATE_LIMIT_PER_MIN.
import threading as _th

_RATE_BUCKETS: Dict[str, List[float]] = {}
_RATE_LOCK = _th.Lock()


def _rate_limit_ok(ip: str, limit: int, window_s: float = 60.0) -> bool:
    now = time.monotonic()
    with _RATE_LOCK:
        hits = _RATE_BUCKETS.setdefault(ip, [])
        while hits and hits[0] <= now - window_s:
            hits.pop(0)
        if len(hits) >= limit:
            return False
        hits.append(now)
        return True


@app.middleware("http")
async def observability_middleware(request: Request, call_next: Callable) -> Any:
    request_id = request.headers.get("X-Request-ID", uuid.uuid4().hex[:12])
    # Operator auth (skipped when unset = local dev).
    if OPERATOR_KEY and request.url.path.startswith("/api/"):
        if request.url.path not in ("/api/health", "/api/ready"):
            if request.headers.get("X-Duat-Key") != OPERATOR_KEY:
                return JSONResponse(status_code=401,
                                    content={"code": "UNAUTHORIZED",
                                             "detail": "Missing or invalid X-Duat-Key."})
    # Rate-limit expensive mutating endpoints.
    if request.method == "POST" and any(
            request.url.path.startswith(p) for p in
            ("/api/chisel/", "/api/jobs", "/api/khafre/mesh", "/api/khafre/depth")):
        try:
            limit = int(os.getenv("DUAT_RATE_LIMIT_PER_MIN", "30"))
        except Exception:
            limit = 30
        ip = (request.client.host if request.client else "unknown")
        if not _rate_limit_ok(ip, limit):
            return JSONResponse(status_code=429,
                                headers={"Retry-After": "60"},
                                content={"code": "RATE_LIMITED",
                                         "detail": "Local rate limit exceeded; retry after 60s."})
    started = time.monotonic()
    try:
        response = await call_next(request)
    except AtlasError as exc:
        return JSONResponse(status_code=502,
                            content={"code": exc.code, "detail": str(exc)})
    ms = (time.monotonic() - started) * 1000.0
    response.headers["X-Request-ID"] = request_id
    print(json.dumps({"ts": round(time.time(), 3), "req": request_id,
                      "m": request.method, "path": request.url.path,
                      "status": response.status_code, "ms": round(ms, 1)}))
    return response

VOLUME_ENGINE = KhafreSARVolumeEngine()
CACHED_BUNDLE: Optional[EpistemicVolumeBundle] = None
ATLAS_CLIENT = WorldLabsAtlasClient()


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


@app.exception_handler(AtlasError)
async def _atlas_error_handler(request: Any, exc: AtlasError) -> Any:
    from fastapi.responses import JSONResponse

    return JSONResponse(status_code=502,
                        content={"code": exc.code, "detail": str(exc),
                                 "hint": "Out-of-credits/spend-limit: top up or raise the cap, then retry with the same request (idempotency is stable)."})


class EnsembleExploreRequest(BaseModel):
    camera: PinholeCameraSpec = Field(default_factory=PinholeCameraSpec)
    hypothesis: Literal["megalithic", "karst", "artifact"] = "megalithic"
    instrument_rig: Literal["borescope_led", "drop_flare", "structured_laser"] = "drop_flare"
    seeds: List[int] = Field(default_factory=lambda: [7, 42, 108, 256])
    voxel_size: float = Field(default=0.25, ge=0.0, le=2.0)
    sensor_noise_sigma_m: float = Field(default=0.18, ge=0.0, le=2.0)
    structural_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    posterior_resample: bool = True
    chain_batches: int = Field(default=2, ge=1, le=16)
    chain_targets: int = Field(default=1, ge=1, le=32)
    return_depth: bool = False
    start_batch: int = Field(default=0, ge=0)
    resume_source_asset_id: Optional[str] = None
    resume_source_camera: Optional[Dict[str, Any]] = None
    z_min: float = 0.5
    z_max: Optional[float] = None
    mode: Literal["mock", "live"] = "mock"


class EpistemicLayerTextures(BaseModel):
    measured_log_depth_b64: str
    inferred_reflectivity_b64: str
    imagined_consensus_rgb_b64: str
    disagreement_map_b64: str


class NextBestViewRecommendation(BaseModel):
    target_z_m: float
    predicted_variance_reduction_pct: float
    rationale: str


class DepthRenderRequest(BaseModel):
    """Render Tier-1/Tier-2 layers for an arbitrary pinhole pose (backend frame)."""

    camera: PinholeCameraSpec = Field(default_factory=PinholeCameraSpec)
    z_min: float = 0.5
    z_max: Optional[float] = None
    out_width: int = Field(default=160, ge=32, le=640)
    out_height: int = Field(default=90, ge=18, le=360)
    seed: int = 7
    shaft_center_xy: List[float] = Field(default_factory=lambda: [-40.0, -20.0])


class DepthRenderResponse(BaseModel):
    log_depth_b64: str
    reflectivity_b64: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    z_min: float
    z_max: float
    valid_fraction: float


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
def _client_for_mode(mode: str) -> WorldLabsAtlasClient:
    """Mock is free and default. Live requires a configured key — otherwise 409,
    never a silent simulation and never a surprise charge."""
    if mode == "live":
        if ATLAS_CLIENT.mock_mode:
            raise HTTPException(
                status_code=409,
                detail="Live Atlas requested but no WLT_API_KEY is configured. "
                       "Set the key or use mode='mock'.")
        return ATLAS_CLIENT
    return WorldLabsAtlasClient(mock_mode=True)


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
    try:
        free_gb = round(shutil.disk_usage(store.db_path().parent).free / 1e9, 2)
    except Exception:
        free_gb = -1.0
    try:
        store.get_job("__none__")
        db_ok = True
    except Exception:
        db_ok = False
    return {"status": "online", "project": "Milimo Duat",
            "worldlabs_mock_mode": ATLAS_CLIENT.mock_mode,
            "khafre_volume_loaded": CACHED_BUNDLE is not None,
            "disk_free_gb": free_gb, "db_ok": db_ok}


@app.get("/api/khafre/trajectory")
async def get_trajectory(num_cameras: int = 32, z_start_m: float = -30.0,
                         z_end_m: float = -180.0, orbit_radius_m: float = 2.2) -> Dict[str, Any]:
    bundle = _get_bundle()
    shaft_center_xy = (-40.0, -20.0)
    return {"bounds_min_m": bundle.metadata.bounds_min_m,
            "bounds_max_m": bundle.metadata.bounds_max_m,
            "voxel_size_m": bundle.metadata.voxel_size_m,
            "shaft_center_xy": list(shaft_center_xy),
            "coordinate_frame": "z_up",
            "targetCameras": VOLUME_ENGINE.generate_helical_descent_cameras(
                num_cameras=num_cameras, shaft_center_xy=shaft_center_xy,
                z_start_m=z_start_m, z_end_m=z_end_m, orbit_radius_m=orbit_radius_m)}


def _check_rollout(n_context: int, n_source: int, n_targets: int) -> None:
    """Atlas rollout budget: context=3 views, photo/target=1, max 128."""
    total = 3 * n_context + n_source + n_targets
    if n_targets < 1 or n_targets > 32:
        raise HTTPException(status_code=422, detail="atlasChisel requires 1..32 targetCameras.")
    if total > 128:
        raise HTTPException(
            status_code=422,
            detail=f"Rollout budget exceeded: 3*{n_context}+{n_source}+{n_targets}={total} > 128. "
                   "Reduce targets or sourceFrames.")


@app.post("/api/chisel/ensemble", response_model=EnsembleExploreResponse)
async def execute_ensemble(req: EnsembleExploreRequest) -> EnsembleExploreResponse:
    client = _client_for_mode(req.mode)
    return await _run_ensemble(req, client)


async def _run_ensemble(
    req: EnsembleExploreRequest,
    client: WorldLabsAtlasClient,
    job_id: Optional[str] = None,
    progress_cb: Optional[Callable[[float], None]] = None,
) -> EnsembleExploreResponse:
    """Shared ensemble core: sync endpoint and background job worker use this."""
    if job_id and store.get_job(job_id) and store.get_job(job_id)["status"] == "cancelled":
        raise asyncio.CancelledError()
    prompt, default_zmax = compose_epistemic_prompt(req.hypothesis, req.instrument_rig)
    z_max = req.z_max if req.z_max is not None else default_zmax

    # Render depth once (fast path), or once per seed from independent
    # posterior draws when the caller wants true geometry variance.
    render_seeds = list(req.seeds) if req.posterior_resample else [req.seeds[0] if req.seeds else 42]
    depth_runs = []
    for rs in render_seeds:
        log_u8, refl_u8, valid = render_shaft_planar_depth_and_reflectivity(
            cam_origin_xyz=tuple(req.camera.pose.position),
            width=req.camera.width, height=req.camera.height, fx=req.camera.fx,
            z_min=req.z_min, z_max=z_max,
            noise_sigma_m=req.sensor_noise_sigma_m, seed=rs)
        depth_runs.append((log_u8, refl_u8, valid))
    log_depth_u8, reflectivity_u8, valid_mask = depth_runs[0]

    atlas_cam = req.camera.to_atlas_dict()
    if progress_cb:
        progress_cb(0.2)
    if job_id and store.get_job(job_id) and store.get_job(job_id)["status"] == "cancelled":
        raise asyncio.CancelledError()
    _check_rollout(n_context=1, n_source=0, n_targets=1)
    tasks = [client.run_atlas_chisel(
        context_camera=atlas_cam, log_depth_png=depth_runs[i % len(depth_runs)][0],
        z_min=req.z_min, z_max=z_max,
        target_cameras=[atlas_cam], prompt=prompt, voxel_size=req.voxel_size,
        seed=s, hypothesis=req.hypothesis) for i, s in enumerate(req.seeds)]
    results = await asyncio.gather(*tasks)
    for r in results:
        store.record_operation(r.operation_id, "atlasChisel", r.request_hash, job_id=job_id)
    if progress_cb:
        progress_cb(0.7)

    rgb_stack = np.stack([r.rgb_frames[0].astype(np.float32) for r in results], axis=0)
    consensus_rgb = compute_consensus(rgb_stack)
    dis_norm, mean_dis = compute_disagreement(rgb_stack, valid_mask,
                                              structural_weight=req.structural_weight)
    disagreement_u8 = disagreement_to_u8(dis_norm)

    manifest = {"camera": atlas_cam, "prompt": prompt, "seeds": req.seeds,
                "voxel_size": req.voxel_size, "z_min": req.z_min, "z_max": z_max,
                "child_hashes": [r.request_hash for r in results]}
    sha = provenance_manifest(manifest)
    record = {"manifest": manifest, "timestamp_utc": time.time()}
    store.record_ensemble(sha, record)
    for r in results:
        store.update_operation(r.operation_id, "succeeded")

    empty_ratio = float(1.0 - (np.count_nonzero(valid_mask) / valid_mask.size))
    if progress_cb:
        progress_cb(1.0)
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
async def execute_chain(req: EnsembleExploreRequest, batches: Optional[int] = None) -> Dict[str, Any]:
    """Synchronous chained descent (small batch counts). For production runs
    use POST /api/jobs (kind=chain) with progress + resume + cancel."""
    client = _client_for_mode(req.mode)
    return await _run_chain(req, batches if batches is not None else req.chain_batches, client)


async def _run_chain(
    req: EnsembleExploreRequest,
    batches: int,
    client: WorldLabsAtlasClient,
    job_id: Optional[str] = None,
    progress_cb: Optional[Callable[[float], None]] = None,
) -> Dict[str, Any]:
    """Production chained descent over the full 32-camera trajectory.

    Each batch covers `chain_targets` consecutive cameras with per-target
    rendered depth (1:1 contextFrames, Atlas-compliant). The previous batch's
    last frame is promoted to a sourceFrame appearance anchor, and its
    `promptUsed` becomes the next prompt with enhancement off (latency win).
    Resume via start_batch + resume asset/camera (assets persist server-side).
    """
    traj = VOLUME_ENGINE.generate_helical_descent_cameras(num_cameras=32)
    _, default_zmax = compose_epistemic_prompt(req.hypothesis, req.instrument_rig)
    z_max = req.z_max if req.z_max is not None else default_zmax
    _check_rollout(n_context=1, n_source=1, n_targets=req.chain_targets)
    traj = VOLUME_ENGINE.generate_helical_descent_cameras(num_cameras=32)
    prompt = compose_epistemic_prompt(req.hypothesis, req.instrument_rig)[0]
    out, carry, carry_depth = [], None, None
    if req.resume_source_asset_id and req.resume_source_camera:
        carry = {"imageAsset": {"assetId": req.resume_source_asset_id},
                 "camera": req.resume_source_camera}
    total_batches = min(batches, max(1, (len(traj) + req.chain_targets - 1) // req.chain_targets))
    for b in range(req.start_batch, total_batches):
        if job_id and store.get_job(job_id) and store.get_job(job_id)["status"] == "cancelled":
            raise asyncio.CancelledError()
        seg = traj[b * req.chain_targets:(b + 1) * req.chain_targets]
        if not seg:
            break
        # Per-target depth, rendered AT each target pose (1:1 contextFrames).
        ctx = []
        for t in seg:
            log_u8, _, _ = render_shaft_planar_depth_and_reflectivity(
                cam_origin_xyz=tuple(t["extrinsics"]["position"]),
                z_min=req.z_min, z_max=z_max,
                seed=(req.seeds[0] if req.seeds else 7) + b)
            ctx.append((t, log_u8))
        srcs = [carry] if carry else []
        res = await client.run_atlas_chisel(
            context_camera=seg[0], log_depth_png=ctx[0][1],
            z_min=req.z_min, z_max=z_max, target_cameras=seg,
            prompt=prompt, voxel_size=req.voxel_size,
            seed=req.seeds[0] if req.seeds else 7,
            hypothesis=req.hypothesis, source_frames=srcs,
            enhance_prompt=(b == 0 and req.start_batch == 0),
            return_depth=req.return_depth and len(seg) >= 2,
            extra_context=ctx if len(ctx) == len(seg) else None)
        store.record_operation(res.operation_id, "atlasChisel", res.request_hash, job_id=job_id)
        batch_rec: Dict[str, Any] = {"batch": b, "operation_id": res.operation_id,
                                    "prompt_used": res.prompt_used,
                                    "targets": len(seg),
                                    "returned_depth": len(res.predicted_depths) > 0}
        out.append(batch_rec)
        prompt = res.prompt_used or prompt
        if res.rgb_frames:
            ok, buf = cv2.imencode(".png", cv2.cvtColor(res.rgb_frames[-1], cv2.COLOR_RGB2BGR))
            if ok:
                ref = await client.create_asset_inline(buf.tobytes(), f"chain_b{b}.png")
                carry = {"imageAsset": {"assetId": ref["assetId"]}, "camera": seg[-1]}
                batch_rec["carry_asset_id"] = ref["assetId"]
        if progress_cb:
            progress_cb((b - req.start_batch + 1) / max(total_batches - req.start_batch, 1))
    return {"batches": out, "prompt": prompt,
            "resume_hint": {"start_batch": total_batches,
                            "resume_source_asset_id": (carry or {}).get("imageAsset", {}).get("assetId"),
                            "resume_source_camera": (carry or {}).get("camera")}}


def _canonical_job_id(kind: str, payload: Dict[str, Any]) -> str:
    """Deterministic job IDs: resubmitting the same work returns the same job
    instead of billing a duplicate task."""
    canonical = json.dumps({"kind": kind, "payload": payload}, sort_keys=True).encode()
    return "job_" + hashlib.sha256(canonical).hexdigest()[:16]


async def _job_worker(job_id: str, kind: str, request: Dict[str, Any]) -> None:
    try:
        client = _client_for_mode(request.get("mode", "mock"))

        def progress(p: float) -> None:
            store.update_job(job_id, progress=round(p, 3))

        if kind == "ensemble":
            req = EnsembleExploreRequest(**request)
            resp = await _run_ensemble(req, client, job_id=job_id, progress_cb=progress)
            store.update_job(job_id, status="done", progress=1.0,
                             result=json.loads(resp.model_dump_json()))
        elif kind == "chain":
            req = EnsembleExploreRequest(**{k: v for k, v in request.items() if k != "batches"})
            resp = await _run_chain(req, int(request.get("batches", 2)), client,
                                     job_id=job_id, progress_cb=progress)
            store.update_job(job_id, status="done", progress=1.0, result=resp)
        else:
            store.update_job(job_id, status="failed",
                             error={"code": "UNKNOWN_KIND", "detail": f"Unknown job kind: {kind}"})
    except asyncio.CancelledError:
        store.update_job(job_id, status="cancelled")
    except AtlasError as exc:
        store.update_job(job_id, status="failed",
                         error={"code": exc.code, "detail": str(exc)})
    except Exception as exc:  # never leave a job stuck in running
        store.update_job(job_id, status="failed",
                         error={"code": "WORKER", "detail": f"{type(exc).__name__}: {exc}"})


class JobCreateRequest(BaseModel):
    kind: Literal["ensemble", "chain"] = "ensemble"
    request: Dict[str, Any] = Field(default_factory=dict)


@app.post("/api/jobs")
async def create_job(body: JobCreateRequest, background: BackgroundTasks) -> Dict[str, Any]:
    job_id = _canonical_job_id(body.kind, body.request)
    existing = store.get_job(job_id)
    if existing is None:
        store.create_job(job_id, body.kind, body.request)
        background.add_task(_job_worker, job_id, body.kind, body.request)
        return {"id": job_id, "status": "pending", "deduplicated": False}
    return {"id": job_id, "status": existing["status"], "deduplicated": True}


def _job_public(job: Dict[str, Any]) -> Dict[str, Any]:
    out = {"id": job["id"], "kind": job["kind"], "status": job["status"],
           "progress": job["progress"], "created_ts": job["created_ts"],
           "updated_ts": job["updated_ts"]}
    if job.get("request_json"):
        try:
            out["request"] = json.loads(job["request_json"])
        except Exception:
            pass
    if job.get("result_json"):
        out["result"] = json.loads(job["result_json"])
    if job.get("error_json"):
        out["error"] = json.loads(job["error_json"])
    return out


@app.get("/api/jobs")
async def list_jobs(limit: int = 50) -> Dict[str, Any]:
    return {"jobs": [_job_public(j) for j in store.list_jobs(limit=limit)]}


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str) -> Dict[str, Any]:
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job.")
    return _job_public(job)


@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: str) -> Dict[str, Any]:
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job.")
    if job["status"] not in ("pending", "running"):
        raise HTTPException(status_code=409, detail=f"Job is already {job['status']}.")
    store.update_job(job_id, status="cancelled")
    return {"id": job_id, "status": "cancelled"}


@app.get("/api/pricing")
async def pricing() -> Dict[str, Any]:
    """Cost transparency: per-task rates come from the operator's platform
    rate card (env), never hardcoded — rates change during beta."""
    import os as _os

    def _rate(name: str) -> Optional[float]:
        try:
            v = _os.getenv(name)
            return float(v) if v else None
        except Exception:
            return None

    return {"credits_per_usd": 1000.0,
            "atlas_chisel_1t_credits": _rate("ATLAS_CREDITS_PER_CHISEL_1T"),
            "atlas_chisel_32t_credits": _rate("ATLAS_CREDITS_PER_CHISEL_32T"),
            "note": "Set ATLAS_CREDITS_PER_CHISEL_1T/32T from Settings → Billing. "
                    "Unknown = cannot estimate; live calls still prompt for confirmation."}


@app.get("/api/ready")
async def readiness() -> Dict[str, Any]:
    ready = CACHED_BUNDLE is not None
    if not ready:
        try:
            _get_bundle()
            ready = True
        except Exception:
            ready = False
    from fastapi.responses import JSONResponse

    return JSONResponse(status_code=200 if ready else 202,
                        content={"ready": ready, "volume_loaded": CACHED_BUNDLE is not None})


@app.get("/api/provenance/{sha}")
async def get_provenance(sha: str) -> Dict[str, Any]:
    rec = store.get_ensemble(sha)
    if rec is None:
        raise HTTPException(status_code=404, detail="Unknown provenance hash.")
    return {"provenance_sha256": sha, **json.loads(rec["manifest_json"])}


_DEGRADATION_MESH = None


def _get_degradation_base() -> trimesh.Trimesh:
    """Small cached ROI mesh (Shaft #2 segment) for the sweep harness."""
    global _DEGRADATION_MESH
    if _DEGRADATION_MESH is None:
        bundle = _get_bundle()
        roi_mesh = VOLUME_ENGINE.extract_posterior_cavity_mesh(
            bundle, posterior_seed=None,
            roi_bounds_min_m=(-52.0, -32.0, -60.0),
            roi_bounds_max_m=(-28.0, -8.0, -30.0))
        _DEGRADATION_MESH = roi_mesh
    return _DEGRADATION_MESH


@app.get("/api/khafre/degradation")
async def degradation_sweep() -> Dict[str, Any]:
    """LiDAR -> SAR -> GPR -> muon downsampling ladder with fidelity proxies.

    Returns per-level vertex counts plus RMS vertex displacement vs the
    full-resolution base mesh (fidelity proxy: prediction error grows as
    sensor resolution coarsens).
    """
    import numpy as np

    from .pipeline.geometry import DEFAULT_SWEEP, degrade_for_sweep

    base = _get_degradation_base()
    base_v = base.vertices.astype(np.float64)
    levels = []
    for name, sigma, quantum in DEFAULT_SWEEP:
        _, m = degrade_for_sweep(base, [(name, sigma, quantum)])[0]
        rms = float(np.sqrt(np.mean((m.vertices.astype(np.float64) - base_v) ** 2)))
        levels.append({"name": name, "blur_sigma_m": sigma, "quant_m": quantum,
                       "vertices": int(len(m.vertices)), "faces": int(len(m.faces)),
                       "rms_displacement_m": round(rms, 4)})
    return {"base_vertices": int(len(base.vertices)), "base_faces": int(len(base.faces)),
            "levels": levels}


@app.post("/api/khafre/depth", response_model=DepthRenderResponse)
async def render_depth(req: DepthRenderRequest) -> DepthRenderResponse:
    """Render official-spec inverted log-depth + dielectric proxy for any pose.

    Intrinsics are scaled from the camera's native resolution to the output
    grid so the pinhole geometry stays exact at any downsample factor.
    """
    from .pipeline.prompts import INSTRUMENT_RIGS

    z_max = req.z_max if req.z_max is not None else INSTRUMENT_RIGS["drop_flare"][1]
    sx = req.out_width / req.camera.width
    sy = req.out_height / req.camera.height
    fx, fy = req.camera.fx * sx, req.camera.fx * sy
    cx, cy = (req.camera.width / 2.0) * sx, (req.camera.height / 2.0) * sy
    log_u8, refl_u8, valid = render_shaft_planar_depth_and_reflectivity(
        cam_origin_xyz=tuple(req.camera.pose.position),
        width=req.out_width, height=req.out_height, fx=fx, fy=fy, cx=cx, cy=cy,
        z_min=req.z_min, z_max=z_max, shaft_center_xy=tuple(req.shaft_center_xy),
        noise_sigma_m=0.0, seed=req.seed)
    return DepthRenderResponse(
        log_depth_b64=_encode_png_b64(log_u8), reflectivity_b64=_encode_png_b64(refl_u8),
        width=req.out_width, height=req.out_height, fx=fx, fy=fy, cx=cx, cy=cy,
        z_min=req.z_min, z_max=z_max,
        valid_fraction=round(float(np.count_nonzero(valid) / valid.size), 4))


@app.post("/api/khafre/mesh")
async def ingest_mesh(file: UploadFile) -> Dict[str, Any]:
    """Ingest a raw surface mesh (.obj/.ply/.glb/.stl) into an EpistemicVolumeBundle."""
    from .pipeline.geometry import load_mesh, mesh_to_bundle

    suffix = (file.filename or "mesh.obj").lower().rsplit(".", 1)[-1]
    if suffix not in ("obj", "ply", "glb", "stl", "off"):
        raise HTTPException(status_code=422, detail=f"Unsupported mesh format: .{suffix}")
    raw = await file.read(26 * 1024 * 1024 + 1)
    if len(raw) > 25 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Mesh exceeds the 25 MB upload cap.")
    if not raw:
        raise HTTPException(status_code=422, detail="Empty upload.")
    if suffix == "glb" and not raw.startswith(b"glTF"):
        raise HTTPException(status_code=422, detail="Not a GLB file (bad magic).")
    import tempfile
    from pathlib import Path

    with tempfile.NamedTemporaryFile(suffix=f".{suffix}", delete=False) as tmp:
        tmp.write(raw)
        tmp_path = tmp.name
    try:
        mesh = load_mesh(tmp_path)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
        bundle = mesh_to_bundle(mesh)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not parse mesh: {exc}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    inside = int(np.count_nonzero(bundle.sdf_volume < 0))
    total = int(bundle.sdf_volume.size)
    return {"grid_shape": list(bundle.metadata.grid_shape),
            "bounds_min_m": list(bundle.metadata.bounds_min_m),
            "bounds_max_m": list(bundle.metadata.bounds_max_m),
            "void_fraction": round(inside / total, 4),
            "mean_reflectivity": round(float(bundle.reflectivity_volume.mean()), 4)}
