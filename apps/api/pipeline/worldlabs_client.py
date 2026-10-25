"""
duat/apps/api/pipeline/worldlabs_client.py

Async client for the World Labs Marble 2 / Atlas API (`atlasChisel`).

Strictly follows live docs (see wiki/atlas-compliance.md):
- Auth header `WLT-Api-Key` (NOT Bearer).
- Assets: POST /assets {"asset": {...}} -> :createUploadUrl -> PUT -> :completeUpload
  (inline base64 fast-path for <10 MiB); task refs use {"assetId": ...}.
- Task: POST /tasks:atlasChisel {contextFrames, sourceFrames, targetCameras,
  prompt, enhancePrompt, voxelSize, voxelGridOrientation, modelParameters}.
- Cameras: {intrinsics:{fx,fy,cx,cy,width,height},
  extrinsics:{position, quaternion XYZW, coordinateSystem}}.
- Depth: logDepthAsset = INVERTED 8-bit PNG (255=zMin, 0=empty) + zMin/zMax,
  or depthAsset = linear EXR planar-z.
- Poll: GET /operations/{id} (fallback POST :wait).
- Mock mode: deterministic physics-inspired synthesizer, zero API cost.

Env: WLT_API_KEY / WLT_API_BASE_URL (legacy WORLDLABS_API_KEY fallback).
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import httpx
import numpy as np


def _load_dotenv_minimal() -> None:
    for name in (".env", "../.env", "../../.env"):
        p = Path(__file__).resolve().parent / name
        alt = Path.cwd() / ".env"
        for cand in (p, alt):
            if cand.exists():
                try:
                    for line in cand.read_text().splitlines():
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
                except Exception:
                    pass


_load_dotenv_minimal()

DEFAULT_BASE_URL = "https://atlas-beta.worldlabs.ai/api/v2"


# ----------------------------------------------------------------------
# Typed errors + retrying transport (production: never tight-loop, always
# reuse the caller's Idempotency-Key so transport retries can't double-bill).
# ----------------------------------------------------------------------
class AtlasError(RuntimeError):
    """Typed World Labs API failure. `code` mirrors the service contract."""

    def __init__(self, code: str, message: str, status: Optional[int] = None):
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.status = status


def _retry_after_secs(resp: httpx.Response) -> float:
    try:
        return max(0.0, min(30.0, float(resp.headers.get("Retry-After", "0"))))
    except Exception:
        return 0.0


async def _request_json(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    headers: Dict[str, str],
    payload: Optional[Dict[str, Any]] = None,
    *,
    attempts: int = 4,
    base_delay: float = 1.0,
) -> httpx.Response:
    """POST/GET with capped exponential backoff + jitter. 4xx (except 429) fail
    fast; 429 honors Retry-After; 402 maps to payment codes; 5xx/timeouts retry."""
    last: Optional[Exception] = None
    for attempt in range(attempts):
        try:
            if method == "POST":
                resp = await client.post(url, headers=headers, json=payload)
            else:
                resp = await client.get(url, headers=headers)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            last = exc
            await asyncio.sleep(min(8.0, base_delay * (2.0 ** attempt)) + random.uniform(0, 1))
            continue
        status = resp.status_code
        if status < 400:
            return resp
        if status == 402:
            try:
                code = (resp.json().get("code") or "CREDITS_OR_LIMIT")
            except Exception:
                code = "CREDITS_OR_LIMIT"
            raise AtlasError(code, "Out of credits or spend limit reached — top up or raise the cap, then retry with the same Idempotency-Key.", status)
        if status == 409:
            raise AtlasError("CONFLICT", "Idempotency key reused with a different body.", status)
        if status == 422:
            raise AtlasError("VALIDATION", f"Request rejected: {resp.text[:400]}", status)
        if status == 429:
            wait = _retry_after_secs(resp) or min(15.0, base_delay * (2.0 ** attempt))
            await asyncio.sleep(wait + random.uniform(0, 1))
            last = AtlasError("RATE_LIMITED", "Rate or concurrency limit refused the request.", status)
            continue
        if status >= 500 or status == 408:
            await asyncio.sleep(min(8.0, base_delay * (2.0 ** attempt)) + random.uniform(0, 1))
            last = AtlasError("SERVER", f"Server error {status}: {resp.text[:200]}", status)
            continue
        raise AtlasError("REQUEST", f"HTTP {status}: {resp.text[:400]}", status)
    raise last if last else AtlasError("UNKNOWN", "Request failed with no response.")


# ----------------------------------------------------------------------
@dataclass
class ChiselFrame:
    rgb: np.ndarray  # (H, W, 3) uint8
    camera: Dict[str, Any]
    depth_planar: Optional[np.ndarray] = None  # (H, W) float32
    asset_id: Optional[str] = None


@dataclass
class ChiselRunResult:
    seed: int
    hypothesis: str
    rgb_frames: List[np.ndarray] = field(default_factory=list)
    predicted_depths: List[np.ndarray] = field(default_factory=list)
    cameras: List[Dict[str, Any]] = field(default_factory=list)
    operation_id: str = ""
    request_hash: str = ""
    prompt_used: str = ""


class WorldLabsAtlasClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        mock_mode: Optional[bool] = None,
        timeout_s: float = 120.0,
    ):
        key = (
            api_key
            or os.getenv("WLT_API_KEY")
            or os.getenv("WORLDLABS_API_KEY", "")
        )
        base = (
            base_url
            or os.getenv("WLT_API_BASE_URL")
            or os.getenv("WORLDLABS_BASE_URL", "")
            or DEFAULT_BASE_URL
        )
        self.api_key = (key or "").strip()
        self.base_url = base.rstrip("/")
        self.mock_mode = mock_mode if mock_mode is not None else (not bool(self.api_key))
        self.timeout_s = timeout_s

    # ---------------- headers ----------------
    def _headers(self, idempotency_key: Optional[str] = None) -> Dict[str, str]:
        h = {"WLT-Api-Key": self.api_key, "Content-Type": "application/json", "Accept": "application/json"}
        if idempotency_key:
            h["Idempotency-Key"] = idempotency_key[:128]
        return h

    # ---------------- assets ----------------
    async def create_asset_inline(self, png_bytes: bytes, display_name: str = "depth.png") -> Dict[str, Any]:
        """Small-asset path: POST /assets {"asset": {displayName, base64}} (<10 MiB)."""
        if self.mock_mode:
            digest = hashlib.sha256(png_bytes).hexdigest()[:16]
            return {"assetId": f"mock_asset_{digest}", "displayName": display_name}
        if len(png_bytes) >= 10 * 1024 * 1024:
            return await self.upload_asset_resumable(png_bytes, display_name, content_type="image/png")
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await _request_json(
                client, "POST", f"{self.base_url}/assets", headers=self._headers(),
                payload={"asset": {"displayName": display_name,
                                    "base64": base64.b64encode(png_bytes).decode("ascii")}},
            )
            data = resp.json()
            asset = data.get("asset", data)
            asset_id = asset.get("assetId") or asset.get("name", "").split("/")[-1] or asset.get("id")
            return {"assetId": asset_id, **asset}

    async def upload_asset_resumable(self, raw_bytes: bytes, display_name: str,
                                     content_type: str = "image/png") -> Dict[str, Any]:
        """Large-asset path: create record -> :createUploadUrl -> PUT -> :completeUpload."""
        if self.mock_mode:
            digest = hashlib.sha256(raw_bytes).hexdigest()[:16]
            return {"assetId": f"mock_asset_{digest}", "displayName": display_name}
        async with httpx.AsyncClient(timeout=60.0) as client:
            create = await _request_json(
                client, "POST", f"{self.base_url}/assets", headers=self._headers(),
                payload={"asset": {"displayName": display_name}})
            asset = create.json().get("asset", create.json())
            asset_id = asset.get("assetId") or asset.get("name", "").split("/")[-1]
            up = await _request_json(
                client, "POST", f"{self.base_url}/assets/{asset_id}:createUploadUrl",
                headers=self._headers(),
                payload={"contentType": content_type, "contentLength": len(raw_bytes)})
            grant = up.json()
            upload_url, method = grant["uploadUrl"], grant.get("method", "PUT").upper()
            put_headers = {k: v for k, v in grant.get("headers", {}).items()}
            # Byte upload: retry 5xx/timeouts, fail fast otherwise.
            last: Optional[Exception] = None
            for attempt in range(3):
                try:
                    put = await client.request(method, upload_url, content=raw_bytes, headers=put_headers)
                    if put.status_code < 400:
                        break
                    if put.status_code < 500:
                        raise AtlasError("REQUEST", f"Upload refused: HTTP {put.status_code}", put.status_code)
                    last = AtlasError("SERVER", f"Upload HTTP {put.status_code}", put.status_code)
                except (httpx.TimeoutException, httpx.TransportError) as exc:
                    last = exc
                await asyncio.sleep(min(8.0, 1.0 * (2.0 ** attempt)) + random.uniform(0, 1))
            else:
                raise last if last else AtlasError("UNKNOWN", "Upload failed with no response.")
            done = await _request_json(
                client, "POST", f"{self.base_url}/assets/{asset_id}:completeUpload",
                headers=self._headers(), payload={})
            done.json()
            return {"assetId": asset_id, **asset}

    async def upload_depth_asset(self, png_bytes: bytes, filename: str = "depth.png") -> Dict[str, Any]:
        return await self.create_asset_inline(png_bytes, filename)

    # ---------------- task ----------------
    def _camera(self, cam: Dict[str, Any]) -> Dict[str, Any]:
        if "intrinsics" in cam and "extrinsics" in cam:
            return cam
        # legacy flat {width,height,fx,fy,cx,cy,pose:{position,quaternion}} -> nested
        pose = cam.get("pose", {})
        w, h = int(cam.get("width", 1280)), int(cam.get("height", 720))
        return {
            "intrinsics": {"fx": float(cam.get("fx", 914.0)), "fy": float(cam.get("fy", cam.get("fx", 914.0))),
                           "cx": float(cam.get("cx", w / 2.0)), "cy": float(cam.get("cy", h / 2.0)),
                           "width": w, "height": h},
            "extrinsics": {"position": [float(x) for x in pose.get("position", [0, 0, 0])],
                           "quaternion": [float(x) for x in pose.get("quaternion", [0, 0, 0, 1])],
                           "coordinateSystem": cam.get("coordinateSystem", "rub")},
        }

    async def run_atlas_chisel(
        self,
        context_camera: Dict[str, Any],
        log_depth_png: np.ndarray,
        z_min: float,
        z_max: float,
        target_cameras: List[Dict[str, Any]],
        prompt: str,
        voxel_size: float = 0.25,
        voxel_grid_orientation: str = "world",
        seed: int = 7,
        hypothesis: str = "megalithic",
        source_frames: Optional[List[Dict[str, Any]]] = None,
        enhance_prompt: bool = True,
        return_depth: bool = False,
        idempotency_key: Optional[str] = None,
        extra_context: Optional[List[Tuple[Dict[str, Any], np.ndarray]]] = None,
    ) -> ChiselRunResult:
        if not (1 <= len(target_cameras) <= 32):
            raise ValueError("atlasChisel requires 1..32 targetCameras.")
        if not prompt.strip():
            raise ValueError("prompt must be non-empty.")
        if extra_context is not None and len(extra_context) != len(target_cameras):
            raise ValueError("extra_context must carry one (camera, depth) per target camera.")
        if return_depth and len(target_cameras) < 2:
            raise ValueError("returnDepth needs at least 2 target cameras with a translation baseline.")
        # Rollout budget guard (context=3 views, photo/target=1, max 128).
        n_src = len(source_frames or [])
        n_ctx = len(extra_context) if extra_context is not None else 1
        if 3 * n_ctx + n_src + len(target_cameras) > 128:
            raise AtlasError("VALIDATION", "Rollout budget exceeded (>128 views). Reduce targets or sources.")

        tgts = [self._camera(c) for c in target_cameras]
        srcs = []
        for s in (source_frames or []):
            srcs.append({"imageAsset": s["imageAsset"], "camera": self._camera(s["camera"])})

        async def _upload_depth(png: np.ndarray, tag: str) -> str:
            ok, encoded = cv2.imencode(".png", png)
            if not ok:
                raise RuntimeError("Failed to encode log_depth_png to PNG bytes.")
            ref = await self.upload_depth_asset(encoded.tobytes(), filename=f"depth_{tag}.png")
            return ref.get("assetId")

        if extra_context is not None:
            context_frames = []
            for i, (ccam, cpng) in enumerate(extra_context):
                aid = await _upload_depth(cpng, f"seed_{seed}_t{i}")
                context_frames.append({
                    "camera": self._camera(ccam),
                    "depth": {"logDepthAsset": {"assetId": aid},
                              "zMin": float(z_min), "zMax": float(z_max)},
                })
        else:
            asset_id = await _upload_depth(log_depth_png, f"seed_{seed}")
            ctx_cam = self._camera(context_camera)
            context_frames = [{
                "camera": ctx_cam,
                "depth": {"logDepthAsset": {"assetId": asset_id}, "zMin": float(z_min), "zMax": float(z_max)},
            }]
        payload: Dict[str, Any] = {
            "contextFrames": context_frames,
            "sourceFrames": srcs,
            "targetCameras": tgts,
            "prompt": prompt,
            "enhancePrompt": bool(enhance_prompt),
            "modelParameters": {"seed": int(seed)},
        }
        if voxel_size and voxel_size > 0:
            payload["voxelSize"] = float(voxel_size)
            payload["voxelGridOrientation"] = voxel_grid_orientation
        if return_depth:
            payload["modelParameters"]["returnDepth"] = True

        req_hash = hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()

        if self.mock_mode:
            await asyncio.sleep(0.05)
            depth_for_mock = extra_context[0][1] if extra_context else log_depth_png
            rgb_frames, pred_depths = self._synthesize_mock_chisel_output(
                depth_for_mock, z_min, z_max, len(tgts), seed, hypothesis)
            return ChiselRunResult(seed=seed, hypothesis=hypothesis, rgb_frames=rgb_frames,
                                   predicted_depths=pred_depths, cameras=tgts,
                                   operation_id=f"op_mock_{req_hash[:12]}",
                                   request_hash=req_hash, prompt_used=prompt)

        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            # Submit once with the caller's stable idempotency key: transport
            # retries inside _request_json reuse it, so retries can't double-bill.
            submit = await _request_json(
                client, "POST", f"{self.base_url}/tasks:atlasChisel",
                headers=self._headers(idempotency_key), payload=payload, attempts=3)
            op = submit.json()
            op_id = op.get("id") or op.get("name", "").split("/")[-1]
            if op.get("done") and not op.get("error"):
                return await self._decode_live_result(client, op, seed, hypothesis, req_hash, op_id, prompt)
            # poll: try :wait first (best-effort, may 504 past the 60 s LB cap),
            # then GET loop with backoff + jitter.
            try:
                wait = await client.post(f"{self.base_url}/operations/{op_id}:wait",
                                         headers=self._headers(),
                                         json={"timeoutSecs": 25}, timeout=45.0)
                if wait.status_code < 300:
                    wop = wait.json()
                    if wop.get("done"):
                        if wop.get("error"):
                            raise AtlasError("FAILED", f"atlasChisel failed: {wop['error']}")
                        return await self._decode_live_result(client, wop, seed, hypothesis, req_hash, op_id, prompt)
            except (httpx.TimeoutException, httpx.TransportError, AtlasError):
                pass  # wait is best-effort; polling continues below
            for attempt in range(90):
                await asyncio.sleep(min(1.5 * (1.15 ** attempt), 6.0) + random.uniform(0, 1))
                try:
                    poll = await _request_json(
                        client, "GET", f"{self.base_url}/operations/{op_id}",
                        headers=self._headers(), attempts=3)
                except AtlasError as exc:
                    if exc.code in ("RATE_LIMITED", "SERVER"):
                        continue
                    raise
                status = poll.json()
                if status.get("done"):
                    if status.get("error"):
                        raise AtlasError("FAILED", f"atlasChisel task failed: {status['error']}")
                    return await self._decode_live_result(client, status, seed, hypothesis, req_hash, op_id, prompt)
            raise AtlasError("TIMEOUT", f"Operation {op_id} timed out.")

    async def _decode_live_result(self, client: httpx.AsyncClient, status: Dict[str, Any],
                                  seed: int, hypothesis: str, req_hash: str, op_id: str,
                                  prompt: str) -> ChiselRunResult:
        resp_block = status.get("response") or status.get("result") or {}
        frames_meta = resp_block.get("frames", [])
        prompt_used = resp_block.get("promptUsed", prompt)
        rgbs, depths, cams = [], [], []
        for fr in frames_meta:
            cams.append(fr.get("camera", {}))
            img = fr.get("imageAsset", {}) or {}
            url = img.get("url")
            if url:
                try:
                    r = await client.get(url, timeout=60.0)
                    if r.status_code >= 400:
                        raise AtlasError("SERVER", f"Output download HTTP {r.status_code}", r.status_code)
                    arr = np.frombuffer(r.content, dtype=np.uint8)
                    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if bgr is not None:
                        rgbs.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                except (httpx.TimeoutException, httpx.TransportError, AtlasError) as exc:
                    raise AtlasError("SERVER", f"Output download failed: {exc}")
            d = (fr.get("depth") or {})
            for key in ("depthAsset", "logDepthAsset"):
                dasset = d.get(key) or {}
                durl = (dasset or {}).get("url")
                if durl:
                    try:
                        rr = await client.get(durl)
                        rr.raise_for_status()
                        if key == "logDepthAsset":
                            arr = np.frombuffer(rr.content, dtype=np.uint8)
                            gray = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
                            if gray is not None:
                                depths.append(gray.astype(np.float32))
                    except Exception:
                        pass
        return ChiselRunResult(seed=seed, hypothesis=hypothesis, rgb_frames=rgbs,
                               predicted_depths=depths, cameras=cams,
                               operation_id=str(op_id), request_hash=req_hash, prompt_used=prompt_used)

    # ---------------- mock synthesizer ----------------
    def _synthesize_mock_chisel_output(self, log_depth_png, z_min, z_max,
                                       num_targets, seed, hypothesis):
        from .depth_renderer import decode_log_depth_png

        rng = np.random.default_rng(seed)
        h, w = log_depth_png.shape[:2]
        valid = log_depth_png > 0
        planar_z = decode_log_depth_png(log_depth_png, z_min, z_max)
        logn = np.zeros_like(planar_z)
        logn[valid] = np.log(np.clip(planar_z[valid], z_min, z_max) / z_min) / max(np.log(z_max / z_min), 1e-9)
        dz_dx = cv2.Sobel(planar_z, cv2.CV_32F, 1, 0, ksize=3)
        dz_dy = cv2.Sobel(planar_z, cv2.CV_32F, 0, 1, ksize=3)
        normal_z = 1.0 / np.sqrt(dz_dx**2 + dz_dy**2 + 1.0)
        falloff = np.where(valid, np.clip(1.35 - 0.85 * logn, 0.08, 1.0), 0.0)
        grain = cv2.GaussianBlur(rng.normal(0.0, 1.0, size=(h, w)).astype(np.float32), (5, 5), 1.2)
        edge = np.hypot(dz_dx, dz_dy)
        var = np.sin(planar_z * 2.5 + seed * 1.7) * np.clip(edge * 0.4, 0.0, 0.35)
        shading = np.clip(normal_z * 0.85 + grain * 0.08 + var, 0.0, 1.2) * falloff
        base = {"megalithic": np.array([182.0, 156.0, 120.0]),
                "karst": np.array([98.0, 142.0, 155.0])}.get(hypothesis, np.array([85.0, 200.0, 235.0]))
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for c in range(3):
            rgb[..., c] = np.where(valid, np.clip(base[c] * shading, 0, 255), 0).astype(np.uint8)
        return [rgb.copy() for _ in range(num_targets)], [planar_z.copy() for _ in range(num_targets)]

    # ---------------- chained descent ----------------
    async def cancel_operation(self, op_id: str, reason: str = "Superseded.") -> bool:
        """Best-effort async cancellation. Returns True if the cancel was accepted."""
        if self.mock_mode:
            return True
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await _request_json(
                    client, "POST", f"{self.base_url}/operations/{op_id}:cancel",
                    headers=self._headers(),
                    payload={"requestId": f"cancel-{op_id[:12]}", "reason": reason},
                    attempts=2)
                return resp.status_code < 300
        except AtlasError:
            return False

    async def chain_descent(self, batches: List[Dict[str, Any]]) -> List[ChiselRunResult]:
        """Legacy helper (prefer the jobs API for production chaining)."""
        results: List[ChiselRunResult] = []
        carry_source: Optional[Dict[str, Any]] = None
        for b in batches:
            srcs = list(b.get("source_frames") or [])
            if carry_source is not None:
                srcs = [carry_source] + srcs
            res = await self.run_atlas_chisel(
                context_camera=b["context_camera"], log_depth_png=b["log_depth_png"],
                z_min=b["z_min"], z_max=b["z_max"], target_cameras=b["target_cameras"],
                prompt=b["prompt"], voxel_size=b.get("voxel_size", 0.25), seed=b.get("seed", 7),
                hypothesis=b.get("hypothesis", "megalithic"), source_frames=srcs,
                enhance_prompt=b.get("enhance_prompt", False))
            results.append(res)
            if res.rgb_frames:
                last_cam = (b["target_cameras"] or [b["context_camera"]])[-1]
                # re-upload last RGB as source photo asset for next hop
                ok, buf = cv2.imencode(".png", cv2.cvtColor(res.rgb_frames[-1], cv2.COLOR_RGB2BGR))
                if ok:
                    ref = await self.create_asset_inline(buf.tobytes(), "chain_source.png")
                    carry_source = {"imageAsset": {"assetId": ref["assetId"]}, "camera": self._camera(last_cam)}
        return results
