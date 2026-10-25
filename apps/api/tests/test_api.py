"""API contract tests (mock mode — no Atlas key needed)."""
from fastapi.testclient import TestClient

from apps.api.main import app

client = TestClient(app)


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["project"] == "Milimo Duat"


def test_trajectory_shape_and_frame_params():
    r = client.get("/api/khafre/trajectory?num_cameras=4&z_start_m=-15&z_end_m=-105&orbit_radius_m=2.4")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["coordinate_frame"] == "z_up"
    assert body["shaft_center_xy"] == [-40.0, -20.0]
    zs = [c["extrinsics"]["position"][2] for c in body["targetCameras"]]
    assert min(zs) >= -105.01 and max(zs) <= -14.99
    r = client.get("/api/khafre/trajectory?num_cameras=4")
    assert r.status_code == 200
    body = r.json()
    assert len(body["targetCameras"]) == 4
    cam = body["targetCameras"][0]
    assert cam["intrinsics"]["width"] == 1280
    assert cam["extrinsics"]["coordinateSystem"] == "rub"


def test_depth_render_endpoint_scaled_intrinsics():
    r = client.post("/api/khafre/depth", json={
        "camera": {"width": 1280, "height": 720, "fov_deg": 70.0,
                   "pose": {"position": [-38.5, -20.0, -45.0], "quaternion": [0, 0, 0, 1]}},
        "out_width": 160, "out_height": 90})
    assert r.status_code == 200, r.text
    body = r.json()
    assert (body["width"], body["height"]) == (160, 90)
    assert abs(body["fx"] - 914.29 * 160 / 1280) < 1.0
    assert body["log_depth_b64"].startswith("data:image/png;base64,")
    assert 0.0 < body["valid_fraction"] < 1.0


def test_chain_multi_batch_mock():
    r = client.post("/api/chisel/chain", params={"batches": 2}, json={
        "camera": {"pose": {"position": [-38.5, -20.0, -45.0], "quaternion": [0, 0, 0, 1]}},
        "seeds": [7], "voxel_size": 0.25, "mode": "mock",
        "chain_targets": 2, "return_depth": True})
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["batches"]) == 2
    assert all(b["returned_depth"] for b in body["batches"])
    assert "resume_hint" in body and body["resume_hint"]["start_batch"] == 2


def test_rollout_budget_rejected():
    r = client.post("/api/chisel/chain", params={"batches": 1}, json={
        "camera": {"pose": {"position": [-38.5, -20.0, -45.0], "quaternion": [0, 0, 0, 1]}},
        "seeds": [7], "mode": "mock", "chain_targets": 33})
    assert r.status_code == 422


def test_operator_auth_gate(monkeypatch):
    import apps.api.main as main_mod

    monkeypatch.setenv("OPERATOR_API_KEY", "secret-123")
    # main.OPERATOR_KEY was bound at import; patch module attr directly.
    monkeypatch.setattr(main_mod, "OPERATOR_KEY", "secret-123")
    assert client.get("/api/health").status_code == 200  # health stays open
    r = client.get("/api/provenance/unknown")
    assert r.status_code == 401
    r2 = client.get("/api/provenance/unknown", headers={"X-Duat-Key": "secret-123"})
    assert r2.status_code == 404  # authed: passes gate, misses record


def test_rate_limiter_unit():
    from apps.api.main import _rate_limit_ok

    ip = "test-unit-ip-1"
    assert all(_rate_limit_ok(ip, 3) for _ in range(3))
    assert not _rate_limit_ok(ip, 3)


def test_degradation_sweep_ladder():
    r = client.get("/api/khafre/degradation")
    assert r.status_code == 200, r.text
    body = r.json()
    assert [lv["name"] for lv in body["levels"]] == ["lidar", "sar_high", "gpr", "muon"]
    rms = [lv["rms_displacement_m"] for lv in body["levels"]]
    assert rms[0] == 0.0 and all(v >= 0.0 for v in rms)
    assert rms[-1] >= rms[0]


def test_ensemble_posterior_resample_flag():
    r = client.post("/api/chisel/ensemble", json={
        "camera": {"pose": {"position": [-38.5, -20.0, -45.0], "quaternion": [0, 0, 0, 1]}},
        "seeds": [7, 42], "voxel_size": 0.25,
        "mode": "mock", "posterior_resample": True, "structural_weight": 0.5})
    assert r.status_code == 200, r.text
    assert len(r.json()["provenance_sha256"]) == 64


def test_ensemble_mock_roundtrip():
    r = client.post("/api/chisel/ensemble", json={
        "camera": {"pose": {"position": [-38.5, -20.0, -45.0], "quaternion": [0, 0, 0, 1]}},
        "hypothesis": "megalithic", "instrument_rig": "drop_flare",
        "seeds": [7, 42], "voxel_size": 0.25, "mode": "mock"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["provenance_sha256"]) == 64
    for k in ("measured_log_depth_b64", "inferred_reflectivity_b64",
              "imagined_consensus_rgb_b64", "disagreement_map_b64"):
        assert body["layers"][k].startswith("data:image/png;base64,")
