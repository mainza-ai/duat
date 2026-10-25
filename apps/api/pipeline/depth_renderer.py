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


def _encode_openexr_float32_bytes(arr: np.ndarray) -> bytes:
    """Write a standards-compliant uncompressed OpenEXR v2 single-channel ('R' FLOAT) image."""
    import struct

    h, w = arr.shape
    ch_val = b"R\x00" + struct.pack("<iB3sii", 2, 0, b"\x00\x00\x00", 1, 1) + b"\x00"

    def _attr(name: bytes, atype: bytes, val: bytes) -> bytes:
        return name + b"\x00" + atype + b"\x00" + struct.pack("<i", len(val)) + val

    header = bytearray(b"\x76\x2f\x31\x01\x02\x00\x00\x00")
    header += _attr(b"channels", b"chlist", ch_val)
    header += _attr(b"compression", b"compression", b"\x00")  # NO_COMPRESSION
    header += _attr(b"dataWindow", b"box2i", struct.pack("<4i", 0, 0, w - 1, h - 1))
    header += _attr(b"displayWindow", b"box2i", struct.pack("<4i", 0, 0, w - 1, h - 1))
    header += _attr(b"lineOrder", b"lineOrder", b"\x00")
    header += _attr(b"pixelAspectRatio", b"float", struct.pack("<f", 1.0))
    header += _attr(b"screenWindowCenter", b"v2f", struct.pack("<2f", 0.0, 0.0))
    header += _attr(b"screenWindowWidth", b"float", struct.pack("<f", 1.0))
    header += b"\x00"

    row_bytes = w * 4
    chunk_size = 8 + row_bytes
    first_offset = len(header) + h * 8
    offsets = [first_offset + y * chunk_size for y in range(h)]
    out = bytearray(header)
    out += struct.pack(f"<{h}Q", *offsets)
    arr_le = np.ascontiguousarray(arr, dtype="<f4")
    for y in range(h):
        out += struct.pack("<ii", y, row_bytes)
        out += arr_le[y].tobytes()
    return bytes(out)


def _decode_openexr_bytes(raw_bytes: bytes) -> np.ndarray:
    """Decode an OpenEXR v2 scanline image (NO_COMPRESSION, ZIPS, or ZIP; FLOAT or HALF)."""
    import struct
    import zlib

    if len(raw_bytes) < 8 or raw_bytes[:4] != b"\x76\x2f\x31\x01":
        raise ValueError("Not a valid OpenEXR magic header.")
    pos = 8
    w = h = 0
    compression = 0
    pix_type = 2  # 1=HALF, 2=FLOAT
    n_channels = 0
    while pos < len(raw_bytes):
        if raw_bytes[pos] == 0:
            pos += 1
            break
        name_end = raw_bytes.index(b"\x00", pos)
        name = raw_bytes[pos:name_end]
        type_end = raw_bytes.index(b"\x00", name_end + 1)
        size = struct.unpack_from("<i", raw_bytes, type_end + 1)[0]
        val_start = type_end + 5
        val = raw_bytes[val_start:val_start + size]
        pos = val_start + size
        if name == b"dataWindow" and size == 16:
            xmin, ymin, xmax, ymax = struct.unpack("<4i", val)
            w, h = xmax - xmin + 1, ymax - ymin + 1
        elif name == b"compression" and size == 1:
            compression = val[0]
        elif name == b"channels":
            cpos = 0
            while cpos < len(val) and val[cpos] != 0:
                cend = val.index(b"\x00", cpos)
                ptype = struct.unpack_from("<i", val, cend + 1)[0]
                if n_channels == 0:
                    pix_type = ptype
                n_channels += 1
                cpos = cend + 1 + 16
    if w <= 0 or h <= 0:
        raise ValueError("Invalid OpenEXR dataWindow.")
    n_channels = max(n_channels, 1)
    lines_per_block = 16 if compression == 3 else 1
    n_blocks = (h + lines_per_block - 1) // lines_per_block
    offsets = struct.unpack_from(f"<{n_blocks}Q", raw_bytes, pos)
    dtype = np.float16 if pix_type == 1 else np.float32
    bytes_per_sample = 2 if pix_type == 1 else 4
    out = np.zeros((h, w), dtype=np.float32)

    def _undo_exr_zip_predictor(buf: bytes) -> bytes:
        arr_u8 = np.frombuffer(buf, dtype=np.uint8).copy()
        if len(arr_u8) <= 1:
            return bytes(arr_u8)
        # OpenEXR ZIP predictor: delta-encoded with +128 bias, then byte-transposed (even/odd split)
        deltas = (arr_u8[1:].astype(np.int32) - 128) & 0xFF
        recon = np.empty_like(arr_u8)
        recon[0] = arr_u8[0]
        recon[1:] = ( int(arr_u8[0]) + np.cumsum(deltas) ) & 0xFF
        half = (len(recon) + 1) // 2
        interleaved = np.empty_like(recon)
        interleaved[0::2] = recon[:half]
        interleaved[1::2] = recon[half:]
        return interleaved.tobytes()

    for b_idx, off in enumerate(offsets):
        y_start, psize = struct.unpack_from("<ii", raw_bytes, off)
        chunk = raw_bytes[off + 8:off + 8 + psize]
        block_lines = min(lines_per_block, h - b_idx * lines_per_block)
        expected = block_lines * n_channels * w * bytes_per_sample
        if compression in (2, 3) and psize < expected:
            decomp = zlib.decompress(chunk)
            raw_line = _undo_exr_zip_predictor(decomp)
        else:
            raw_line = chunk
        line_arr = np.frombuffer(raw_line, dtype=dtype).reshape(block_lines, n_channels, w)
        row0 = b_idx * lines_per_block
        out[row0:row0 + block_lines] = line_arr[:, 0, :].astype(np.float32)
    return np.where(np.isfinite(out) & (out > 0), out, 0.0).astype(np.float32)


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
        Path(path).write_bytes(_encode_openexr_float32_bytes(arr))


def read_planar_exr(path: str | Path) -> np.ndarray:
    try:
        import OpenEXR  # type: ignore
        import Imath  # type: ignore

        exr = OpenEXR.InputFile(str(path))
        dw = exr.header()["dataWindow"]
        w, h = dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1
        raw = exr.channel("R", Imath.PixelType(Imath.PixelType.FLOAT))
        arr = np.frombuffer(raw, dtype=np.float32).reshape(h, w)
        return np.where(np.isfinite(arr) & (arr > 0), arr, 0.0).astype(np.float32)
    except Exception:
        raw_bytes = Path(path).read_bytes()
        if raw_bytes[:4] == b"\x76\x2f\x31\x01":
            return _decode_openexr_bytes(raw_bytes)
        import imageio.v2 as imageio  # type: ignore

        arr = imageio.imread(str(path)).astype(np.float32)
        if arr.ndim == 3:
            arr = arr[..., 0]
        return np.where(np.isfinite(arr) & (arr > 0), arr, 0.0).astype(np.float32)


def write_planar_exr_bytes(planar_z: np.ndarray) -> bytes:
    """Encode a float32 planar-z array (0/NaN/Inf = empty) into OpenEXR bytes in memory."""
    arr = np.where(np.isfinite(planar_z) & (planar_z > 0), planar_z, 0.0).astype(np.float32)
    return _encode_openexr_float32_bytes(arr)


def read_planar_exr_bytes(raw_bytes: bytes) -> np.ndarray:
    """Decode linear planar-depth EXR (or fallback float/gray image) bytes in memory,
    masking NaN/Inf/<=0 floater pixels to 0.0 per Atlas cameras-and-posed-images spec."""
    if raw_bytes[:4] == b"\x76\x2f\x31\x01":
        return _decode_openexr_bytes(raw_bytes)
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".exr", delete=False) as tmp:
        tmp.write(raw_bytes)
        tmp_path = Path(tmp.name)
    try:
        return read_planar_exr(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)


# ---------------- pinhole helpers ----------------

def rotate_vectors_by_quaternion(
    vecs: np.ndarray,
    quat_xyzw: Tuple[float, float, float, float],
) -> np.ndarray:
    """Rotate (..., 3) vectors by a unit XYZW quaternion (camera-to-world)."""
    qx, qy, qz, qw = (float(quat_xyzw[0]), float(quat_xyzw[1]),
                      float(quat_xyzw[2]), float(quat_xyzw[3]))
    norm = float(np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw))
    if norm > 1e-8:
        qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
    else:
        qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0
    uvw = np.array([qx, qy, qz], dtype=np.float32)
    t = 2.0 * np.cross(uvw, vecs)
    return (vecs + qw * t + np.cross(uvw, t)).astype(np.float32)


def pinhole_ray_directions(
    w: int,
    h: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    coordinate_system: str = "rub",
    cam_quaternion_xyzw: Optional[Tuple[float, float, float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute unit ray directions (H, W, 3) and planar conversion factor inv_norm (H, W).

    - In 'rub' (OpenGL/Three.js/Atlas default): +X right, +Y up, -Z forward.
      Pixel (0,0) is top-left (+u right, +v down), so camera-space ray is
      [(u - cx)/fx, -(v - cy)/fy, -1] * inv_norm.
    - In 'rdf' (OpenCV): +X right, +Y down, +Z forward ->
      [(u - cx)/fx, (v - cy)/fy, +1] * inv_norm.
    - When cam_quaternion_xyzw is provided, camera-space unit rays are rotated
      into world space while inv_norm stays in camera space (planar z is always
      measured along the camera's optical axis).
    """
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    x_norm = (u - cx) / fx
    y_norm = (v - cy) / fy
    inv_norm = (1.0 / np.sqrt(x_norm**2 + y_norm**2 + 1.0)).astype(np.float32)
    if coordinate_system.lower() == "rdf":
        dirs = np.stack([x_norm * inv_norm, y_norm * inv_norm, inv_norm], axis=-1).astype(np.float32)
    else:
        dirs = np.stack([x_norm * inv_norm, -y_norm * inv_norm, -inv_norm], axis=-1).astype(np.float32)
    if cam_quaternion_xyzw is not None:
        dirs = rotate_vectors_by_quaternion(dirs, cam_quaternion_xyzw)
    return dirs, inv_norm


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
    spiral_pitch_m: float = 28.0,
    ramp_width_m: float = 2.0,
    ramp_thickness_m: float = 0.8,
    wall_relief_m: float = 0.0,
    spiral_style: str = "interior_ramp",
    n_steps: int = 96,
    cam_quaternion_xyzw: Optional[Tuple[float, float, float, float]] = None,
    coordinate_system: str = "rub",
    null_void: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Raymarch the shared shaft model; returns (log_depth_u8, refl_u8, valid_mask).

    Marching uses the bundle-convention SDF negated (positive = safe void),
    so the conditioning geometry is identical to the extracted meshes by
    construction — they share `shaft_model.shaft_void_sdf`.
    When `null_void=True` (Alvarez 1969 bedrock_null hypothesis: no void detected),
    returns strict 0-masked depth (`log_depth_u8 = 0`, `valid = False`) and
    stratigraphic bedrock baseline reflectivity.
    """
    from .shaft_model import ShaftParams, spiral_feature_proximity, shaft_void_sdf

    rng = np.random.default_rng(seed)
    fx = fx or float((width / 2.0) / np.tan(np.radians(70.0 / 2.0)))
    fy = fy or fx
    cx = width / 2.0 if cx is None else cx
    cy = height / 2.0 if cy is None else cy

    if null_void:
        log_depth_u8 = np.zeros((height, width), dtype=np.uint8)
        refl_u8 = np.zeros((height, width), dtype=np.uint8)
        valid = np.zeros((height, width), dtype=bool)
        return log_depth_u8, refl_u8, valid

    dirs, inv_norm = pinhole_ray_directions(
        width, height, fx, fy, cx, cy,
        coordinate_system=coordinate_system,
        cam_quaternion_xyzw=cam_quaternion_xyzw,
    )

    rel = np.array([cam_origin_xyz[0] - shaft_center_xy[0],
                    cam_origin_xyz[1] - shaft_center_xy[1], cam_origin_xyz[2]], dtype=np.float32)
    params = ShaftParams(radius_m=shaft_radius_m, spiral_pitch_m=spiral_pitch_m,
                         ramp_width_m=ramp_width_m, ramp_thickness_m=ramp_thickness_m,
                         wall_relief_m=wall_relief_m, spiral_style=spiral_style)
    t = np.full((height, width), z_min, dtype=np.float32)
    hit = np.zeros((height, width), dtype=bool)
    for _ in range(n_steps):
        pts = rel + dirs * t[..., None]
        sdf = -shaft_void_sdf(pts[..., 0], pts[..., 1], pts[..., 2], 0.0, 0.0, params)
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
    feat_band = spiral_feature_proximity(
        hit_pts[..., 0], hit_pts[..., 1], hit_pts[..., 2],
        0.0, 0.0, params)
    moisture = np.clip((-hit_pts[..., 2]) / 600.0, 0.0, 1.0) * 0.35
    refl = np.where(valid, np.clip(0.25 + 0.35 * feat_band + moisture, 0.0, 1.0), 0.0)
    return log_depth_u8, np.round(refl * 255.0).astype(np.uint8), valid


def raycast_mesh_planar_depth_and_reflectivity(
    mesh: "object",
    cam_origin_xyz: Tuple[float, float, float],
    cam_quaternion_xyzw: Optional[Tuple[float, float, float, float]] = None,
    width: int = 1280,
    height: int = 720,
    fx: Optional[float] = None,
    fy: Optional[float] = None,
    cx: Optional[float] = None,
    cy: Optional[float] = None,
    z_min: float = 0.5,
    z_max: float = 40.0,
    noise_sigma_m: float = 0.0,
    seed: int = 42,
    coordinate_system: str = "rub",
    max_rays: int = 23040,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pinhole mesh raycaster (for reference voids & uploaded .obj/.ply/.glb/.stl meshes).

    Traces rays at a working resolution (up to `max_rays`, e.g. 192x120) against
    `mesh` via trimesh ray-triangle intersection, converts Euclidean hit distance
    `d` to planar depth `z = d * inv_norm`, and upsamples with nearest-neighbor
    (preserving strict 0-masks) to `(height, width)`.
    """
    import trimesh

    assert isinstance(mesh, trimesh.Trimesh), "mesh must be a trimesh.Trimesh"
    rng = np.random.default_rng(seed)
    fx = fx or float((width / 2.0) / np.tan(np.radians(70.0 / 2.0)))
    fy = fy or fx
    cx = width / 2.0 if cx is None else cx
    cy = height / 2.0 if cy is None else cy

    total_px = width * height
    if total_px > max_rays:
        scale = float(np.sqrt(max_rays / total_px))
        rw = max(16, int(round(width * scale)))
        rh = max(9, int(round(height * scale)))
    else:
        rw, rh = width, height

    sx, sy = rw / width, rh / height
    rfx, rfy = fx * sx, fy * sy
    rcx, rcy = cx * sx, cy * sy

    dirs, inv_norm = pinhole_ray_directions(
        rw, rh, rfx, rfy, rcx, rcy,
        coordinate_system=coordinate_system,
        cam_quaternion_xyzw=cam_quaternion_xyzw,
    )
    origins = np.broadcast_to(
        np.asarray(cam_origin_xyz, dtype=np.float32), (rh * rw, 3)
    ).copy()
    ray_dirs = dirs.reshape(-1, 3)

    ray_d = np.zeros(rh * rw, dtype=np.float32)
    hit_mask = np.zeros(rh * rw, dtype=bool)
    refl_flat = np.zeros(rh * rw, dtype=np.float32)

    try:
        locs, idx_ray, idx_tri = mesh.ray.intersects_location(
            ray_origins=origins, ray_directions=ray_dirs, multiple_hits=False
        )
        if len(idx_ray) > 0:
            dists = np.linalg.norm(locs - origins[idx_ray], axis=-1).astype(np.float32)
            if noise_sigma_m > 0:
                dists = dists + rng.normal(0.0, noise_sigma_m, size=dists.shape).astype(np.float32)
            ray_d[idx_ray] = dists
            hit_mask[idx_ray] = True
            # Derive Tier-2 reflectivity from face normals + vertex attributes if present
            if "reflectivity" in mesh.vertex_attributes:
                v_refl = np.asarray(mesh.vertex_attributes["reflectivity"], dtype=np.float32)
                tri_verts = mesh.faces[idx_tri]
                refl_flat[idx_ray] = np.clip(v_refl[tri_verts].mean(axis=1), 0.0, 1.0)
            else:
                normals = mesh.face_normals[idx_tri]
                cos_inc = np.abs(np.sum(normals * ray_dirs[idx_ray], axis=-1))
                refl_flat[idx_ray] = np.clip(0.25 + 0.55 * cos_inc, 0.0, 1.0)
    except Exception:
        pass

    ray_d_2d = ray_d.reshape(rh, rw)
    hit_2d = hit_mask.reshape(rh, rw)
    refl_2d = refl_flat.reshape(rh, rw)

    planar_z = ray_distance_to_planar_depth(ray_d_2d, inv_norm)
    valid_small = hit_2d & np.isfinite(planar_z) & (planar_z >= z_min) & (planar_z <= z_max)
    planar_z = np.where(valid_small, planar_z, 0.0).astype(np.float32)
    refl_2d = np.where(valid_small, refl_2d, 0.0).astype(np.float32)

    if (rw, rh) != (width, height):
        planar_z = cv2.resize(planar_z, (width, height), interpolation=cv2.INTER_NEAREST)
        refl_2d = cv2.resize(refl_2d, (width, height), interpolation=cv2.INTER_NEAREST)
        valid = planar_z >= z_min
    else:
        valid = valid_small

    log_depth_u8 = encode_log_depth_png(planar_z, z_min, z_max)
    refl_u8 = np.round(np.clip(refl_2d, 0.0, 1.0) * 255.0).astype(np.uint8)
    return log_depth_u8, refl_u8, valid


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
