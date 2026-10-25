"""Planar-depth math + official inverted log-depth codec tests."""
import numpy as np

from apps.api.pipeline.depth_renderer import (
    decode_log_depth_png,
    encode_log_depth_png,
    pinhole_ray_directions,
    ray_distance_to_planar_depth,
)


def test_planar_conversion_matches_pinhole_ray():
    w, h, fx, fy, cx, cy = 1280, 720, 914.0, 914.0, 640.0, 360.0
    dirs, inv = pinhole_ray_directions(w, h, fx, fy, cx, cy)
    d = np.full((h, w), 10.0, dtype=np.float32)
    z = ray_distance_to_planar_depth(d, inv)
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    expect = d / np.sqrt(((u - cx) / fx) ** 2 + ((v - cy) / fy) ** 2 + 1.0)
    assert np.allclose(z, expect, atol=1e-5)
    # center pixel: planar == ray distance
    assert abs(float(z[360, 640]) - 10.0) < 1e-4


def test_inverted_codec_endpoints_and_roundtrip():
    z_min, z_max = 0.5, 40.0
    assert int(encode_log_depth_png(np.array([[z_min]], np.float32), z_min, z_max)[0, 0]) == 255
    assert int(encode_log_depth_png(np.array([[z_max]], np.float32), z_min, z_max)[0, 0]) == 1
    assert int(encode_log_depth_png(np.array([[0.0]], np.float32), z_min, z_max)[0, 0]) == 0
    zs = np.geomspace(z_min, z_max, 50).astype(np.float32).reshape(-1, 1)
    back = decode_log_depth_png(encode_log_depth_png(zs, z_min, z_max), z_min, z_max)
    rel = np.abs(back.flatten() - zs.flatten()) / zs.flatten()
    assert float(rel.max()) < 0.02


def test_strict_zero_masking_misses_and_out_of_range():
    z_min, z_max = 0.5, 40.0
    planar = np.array([[0.0, np.nan, np.inf, 100.0, 0.3]], dtype=np.float32)
    enc = encode_log_depth_png(planar, z_min, z_max)
    assert (enc == 0).all()
    assert (decode_log_depth_png(enc, z_min, z_max) == 0.0).all()
