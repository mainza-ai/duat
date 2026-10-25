"""Ensemble + prompts + mock client determinism tests."""
import asyncio

import numpy as np

from apps.api.pipeline.depth_renderer import render_shaft_planar_depth_and_reflectivity
from apps.api.pipeline.ensemble import compute_consensus, compute_disagreement, provenance_manifest
from apps.api.pipeline.prompts import compose_epistemic_prompt
from apps.api.pipeline.worldlabs_client import WorldLabsAtlasClient


def test_ensemble_shapes_and_ranges():
    log_u8, _, valid = render_shaft_planar_depth_and_reflectivity(
        (-38.5, -20.0, -45.0), width=320, height=180, z_min=0.5, z_max=40.0, seed=7)
    assert log_u8.shape == (180, 320) and log_u8.dtype == np.uint8
    assert (log_u8[~valid] == 0).all()
    client = WorldLabsAtlasClient(mock_mode=True)
    res = asyncio.run(client.run_atlas_chisel(
        context_camera={"intrinsics": {"fx": 200, "fy": 200, "cx": 160, "cy": 90, "width": 320, "height": 180},
                        "extrinsics": {"position": [0, 0, 0], "quaternion": [0, 0, 0, 1], "coordinateSystem": "rub"}},
        log_depth_png=log_u8, z_min=0.5, z_max=40.0,
        target_cameras=[{"intrinsics": {"fx": 200, "fy": 200, "cx": 160, "cy": 90, "width": 320, "height": 180},
                         "extrinsics": {"position": [0, 0, 0], "quaternion": [0, 0, 0, 1], "coordinateSystem": "rub"}}],
        prompt="test", seed=7, hypothesis="megalithic"))
    assert res.rgb_frames[0].shape == (180, 320, 3)
    stack = np.stack([res.rgb_frames[0].astype(np.float32)] * 3, axis=0)
    cons = compute_consensus(stack)
    assert cons.shape == (180, 320, 3)
    norm, mean = compute_disagreement(stack, valid)
    assert norm.shape == (180, 320) and 0.0 <= mean <= 1.0


def test_mock_determinism_and_prompt_hash():
    client = WorldLabsAtlasClient(mock_mode=True)
    kw = dict(context_camera={"intrinsics": {"fx": 200, "fy": 200, "cx": 160, "cy": 90, "width": 320, "height": 180},
                              "extrinsics": {"position": [0, 0, 0], "quaternion": [0, 0, 0, 1], "coordinateSystem": "rub"}},
              log_depth_png=np.full((180, 320), 200, np.uint8), z_min=0.5, z_max=40.0,
              target_cameras=[{"intrinsics": {"fx": 200, "fy": 200, "cx": 160, "cy": 90, "width": 320, "height": 180},
                               "extrinsics": {"position": [0, 0, 0], "quaternion": [0, 0, 0, 1], "coordinateSystem": "rub"}}],
              prompt="x", seed=11, hypothesis="karst")
    a = asyncio.run(client.run_atlas_chisel(**kw))
    b = asyncio.run(client.run_atlas_chisel(**kw))
    assert np.array_equal(a.rgb_frames[0], b.rgb_frames[0])
    assert a.request_hash == b.request_hash
    prompt, zmax = compose_epistemic_prompt("karst", "borescope_led")
    assert zmax == 6.0 and "phreatic" in prompt
    h = provenance_manifest({"a": 1})
    assert len(h) == 64
