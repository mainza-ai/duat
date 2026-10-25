"""Atlas transport contract tests (respx-mocked httpx; zero credits spent).

Covers: submit -> running -> done success, :wait 504 fallthrough to polling,
429-then-success retry, 402 payment mapping, done+error failure mapping.
"""
import asyncio

import cv2
import httpx
import numpy as np
import respx

from apps.api.pipeline.worldlabs_client import AtlasError, WorldLabsAtlasClient

BASE = "https://atlas-beta.worldlabs.ai/api/v2"
DEPTH_PNG = cv2.imencode(".png", np.full((8, 8), 200, np.uint8))[1].tobytes()
RGB_PNG = cv2.imencode(".png", np.zeros((4, 4, 3), np.uint8))[1].tobytes()

CAM = {"intrinsics": {"fx": 200, "fy": 200, "cx": 160, "cy": 90, "width": 320, "height": 180},
       "extrinsics": {"position": [0, 0, 0], "quaternion": [0, 0, 0, 1], "coordinateSystem": "rub"}}


def _kw(**over):
    kw = dict(context_camera=CAM, log_depth_png=np.full((180, 320), 200, np.uint8),
              z_min=0.5, z_max=40.0, target_cameras=[CAM], prompt="x",
              seed=7, hypothesis="megalithic")
    kw.update(over)
    return kw


def _client():
    return WorldLabsAtlasClient(api_key="test-key", base_url=BASE, mock_mode=False)


@respx.mock
def test_submit_running_done_success():
    respx.post(f"{BASE}/assets").mock(return_value=httpx.Response(200, json={"asset": {"assetId": "a1"}}))
    respx.post(f"{BASE}/tasks:atlasChisel").mock(
        return_value=httpx.Response(200, json={"id": "op1", "done": False}))
    respx.post(f"{BASE}/operations/op1:wait").mock(
        return_value=httpx.Response(200, json={"id": "op1", "done": True,
            "response": {"frames": [{"imageAsset": {"url": "https://cdn/x/img.png"}, "camera": CAM}],
                          "promptUsed": "used prompt"}}))
    respx.get("https://cdn/x/img.png").mock(return_value=httpx.Response(200, content=RGB_PNG))
    res = asyncio.run(_client().run_atlas_chisel(**_kw()))
    assert res.operation_id == "op1"
    assert res.prompt_used == "used prompt"
    assert res.rgb_frames[0].shape == (4, 4, 3)


@respx.mock
def test_wait_504_falls_through_to_poll():
    respx.post(f"{BASE}/assets").mock(return_value=httpx.Response(200, json={"asset": {"assetId": "a1"}}))
    respx.post(f"{BASE}/tasks:atlasChisel").mock(
        return_value=httpx.Response(200, json={"id": "op2", "done": False}))
    respx.post(f"{BASE}/operations/op2:wait").mock(return_value=httpx.Response(504))
    respx.get(f"{BASE}/operations/op2").mock(
        return_value=httpx.Response(200, json={"id": "op2", "done": True,
            "response": {"frames": [{"imageAsset": {"url": "https://cdn/x/i2.png"}, "camera": CAM}],
                          "promptUsed": "p"}}))
    respx.get("https://cdn/x/i2.png").mock(return_value=httpx.Response(200, content=RGB_PNG))
    res = asyncio.run(_client().run_atlas_chisel(**_kw()))
    assert res.operation_id == "op2"


@respx.mock
def test_429_then_success_retries_with_same_key():
    respx.post(f"{BASE}/assets").mock(return_value=httpx.Response(200, json={"asset": {"assetId": "a1"}}))
    route = respx.post(f"{BASE}/tasks:atlasChisel")
    route.side_effect = [
        httpx.Response(429, headers={"Retry-After": "0"}, json={"code": "RATE_LIMIT_EXCEEDED"}),
        httpx.Response(200, json={"id": "op3", "done": True,
            "response": {"frames": [{"imageAsset": {"url": "https://cdn/x/i3.png"}, "camera": CAM}],
                          "promptUsed": "p"}}),
    ]
    respx.get("https://cdn/x/i3.png").mock(return_value=httpx.Response(200, content=RGB_PNG))
    res = asyncio.run(_client().run_atlas_chisel(**_kw(), idempotency_key="k-same"))
    assert res.operation_id == "op3"
    keys = {c.request.headers.get("Idempotency-Key") for c in route.calls}
    assert keys == {"k-same"}  # stable key across the transport retry


def test_live_probe_fixture_schema():
    """Real 1-target probe response (recorded, no network): decoder inputs valid."""
    import json
    from pathlib import Path

    fix = json.loads(Path(__file__).parent.joinpath(
        "fixtures/atlas_chisel_1target_probe.json").read_text())
    assert len(fix["operation_id"]) >= 8 and len(fix["request_hash"]) == 64
    assert isinstance(fix["prompt_used"], str) and fix["prompt_used"]
    fr = fix["frames"][0]
    assert fr["rgb_shape"] == [720, 1280, 3]
    intr = fr["camera"]["intrinsics"]
    assert (intr["width"], intr["height"]) == (1280, 720)
    assert fr["camera"]["extrinsics"]["coordinateSystem"] == "rub"


@respx.mock
def test_splats_submit_done_returns_mesh():
    respx.post(f"{BASE}/tasks:splats2Mesh").mock(
        return_value=httpx.Response(200, json={"id": "ops1", "done": True,
            "response": {"mesh": {"assetId": "asset_mesh1", "url": "https://cdn/x/m.glb"},
                          "anchorPositions": [[0, 0, 0]]}}))
    res = asyncio.run(_client().run_splats_to_mesh("asset_spz1"))
    assert res.mesh_asset_id == "asset_mesh1"
    assert res.anchor_positions == [[0, 0, 0]]


@respx.mock
def test_generate_submit_done_returns_frames():
    respx.post(f"{BASE}/tasks:atlasGenerate").mock(
        return_value=httpx.Response(200, json={"id": "opg1", "done": True,
            "response": {"frames": [{"imageAsset": {"url": "https://cdn/x/g.png"}, "camera": CAM}],
                          "promptUsed": "gen"}}))
    respx.get("https://cdn/x/g.png").mock(return_value=httpx.Response(200, content=RGB_PNG))
    res = asyncio.run(_client().run_atlas_generate(
        context_frames=[{"image_asset_id": "a", "camera": CAM}],
        target_cameras=[CAM], prompt="x"))
    assert len(res.rgb_frames) == 1 and res.prompt_used == "gen"


def test_splats_endpoint_mock():
    from fastapi.testclient import TestClient

    from apps.api.main import app

    c = TestClient(app)
    r = c.post("/api/world/splats2mesh", json={"splat_asset_id": "asset_spz1", "mode": "mock"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["mesh_asset_id"] == "mock_asset_splats"
    assert body["mesh_glb_b64"].startswith("data:model/gltf-binary;base64,")
    bad = c.post("/api/world/splats2mesh",
                 json={"splat_asset_id": "x", "texture_mode": "bogus", "mode": "mock"})
    assert bad.status_code == 422


@respx.mock
def test_402_maps_to_payment_error():
    respx.post(f"{BASE}/assets").mock(return_value=httpx.Response(200, json={"asset": {"assetId": "a1"}}))
    respx.post(f"{BASE}/tasks:atlasChisel").mock(
        return_value=httpx.Response(402, json={"code": "SPEND_LIMIT_EXCEEDED"}))
    try:
        asyncio.run(_client().run_atlas_chisel(**_kw()))
    except AtlasError as exc:
        assert exc.code == "SPEND_LIMIT_EXCEEDED"
    else:
        raise AssertionError("expected AtlasError")


@respx.mock
def test_done_error_maps_to_failed():
    respx.post(f"{BASE}/assets").mock(return_value=httpx.Response(200, json={"asset": {"assetId": "a1"}}))
    respx.post(f"{BASE}/tasks:atlasChisel").mock(
        return_value=httpx.Response(200, json={"id": "op4", "done": False}))
    respx.post(f"{BASE}/operations/op4:wait").mock(
        return_value=httpx.Response(200, json={"id": "op4", "done": False}))
    respx.get(f"{BASE}/operations/op4").mock(
        return_value=httpx.Response(200, json={"id": "op4", "done": True, "error": {"message": "boom"}}))
    try:
        asyncio.run(_client().run_atlas_chisel(**_kw()))
    except AtlasError as exc:
        assert exc.code == "FAILED"
    else:
        raise AssertionError("expected AtlasError")
