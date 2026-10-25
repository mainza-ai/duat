"""API contract tests (mock mode — no Atlas key needed)."""
from fastapi.testclient import TestClient

from apps.api.main import app

client = TestClient(app)


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["project"] == "Milimo Duat"


def test_trajectory_chamber_param_and_null_prompt():
    from apps.api.pipeline.prompts import compose_epistemic_prompt

    r = client.get("/api/khafre/trajectory?num_cameras=2&chamber_size_m=90")
    assert r.status_code == 200, r.text
    assert r.json()["volume"] == {"seed": 42, "chamber_size_m": 90.0}
    assert client.get("/api/khafre/trajectory?num_cameras=2&chamber_size_m=10").status_code == 422
    prompt, _ = compose_epistemic_prompt("bedrock_null", "drop_flare")
    assert "no void" in prompt
    # null hypothesis accepted end-to-end (mock)
    r2 = client.post("/api/chisel/ensemble", json={
        "camera": {"pose": {"position": [-38.5, -20.0, -45.0], "quaternion": [0, 0, 0, 1]}},
        "hypothesis": "bedrock_null", "seeds": [7], "mode": "mock"})
    assert r2.status_code == 200, r2.text


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
    assert 0.5 < body["valid_fraction"] <= 1.0


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


def test_complex_glb_and_metadata():
    import json as _json
    import struct

    r = client.get("/api/khafre/complex-glb")
    assert r.status_code == 200, r.text
    assert r.content[:4] == b"glTF"
    json_len = struct.unpack("<I", r.content[12:16])[0]
    scene = _json.loads(r.content[20:20 + json_len])
    names = {n.get("name", "") for n in scene.get("nodes", [])}
    assert any("shaft" in n for n in names) and any("chamber" in n or "basal" in n for n in names)
    meta = client.get("/api/khafre/complex").json()
    assert meta["water_z_m"] == -585.0
    assert len([p for p in meta["parts"] if p["kind"] == "shaft"]) == 8


def test_posterior_mesh_glb():
    r = client.get("/api/khafre/mesh-glb")
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "model/gltf-binary"
    assert r.content[:4] == b"glTF"
    assert len(r.content) > 10000
    assert "x-mesh-sha256" in {k.lower() for k in r.headers.keys()}


def test_frames_persist_and_download():
    # Seed one ensemble (mock) to record a frame, then list + download.
    r = client.post("/api/chisel/ensemble", json={
        "camera": {"pose": {"position": [-38.5, -20.0, -45.0], "quaternion": [0, 0, 0, 1]}},
        "seeds": [7], "mode": "mock"})
    assert r.status_code == 200, r.text
    sha = r.json()["provenance_sha256"]
    fr = client.get("/api/frames?limit=5").json()["frames"]
    assert len(fr) >= 1
    mine = [f for f in fr if f["provenance_sha"] == sha]
    assert mine, "ensemble run must record its frame"
    rgb = client.get(f"/api/frames/{mine[0]['id']}/rgb")
    assert rgb.status_code == 200 and rgb.content[:8] == b"\x89PNG\r\n\x1a\n"
    assert rgb.headers["content-type"] == "image/png"
    assert client.get("/api/frames/nope/rgb").status_code == 404


def test_reference_voids_cited_dims():
    from apps.api.pipeline.khafre_sar_volume import synthesize_reference_void

    nfc = synthesize_reference_void("nfc")
    assert nfc.sdf_volume.shape == (64, 64, 64)
    assert float(nfc.sdf_volume[32, 32, 32]) < 0.0  # center inside the corridor
    assert nfc.metadata.sensor_uncertainty_m == 0.05
    big = synthesize_reference_void("big-void")
    assert big.metadata.sensor_uncertainty_m == 0.10
    try:
        synthesize_reference_void("nope")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
    r = client.get("/api/khafre/reference-voids")
    assert r.status_code == 200, r.text
    voids = {v["kind"]: v for v in r.json()["voids"]}
    assert voids["nfc"]["dims_m"] == [9.0, 2.0, 2.0]
    assert voids["big-void"]["dims_m"] == [30.0, 6.0, 6.0]


def test_claims_registry_integrity():
    r = client.get("/api/claims")
    assert r.status_code == 200, r.text
    claims = r.json()["claims"]
    ids = [c["id"] for c in claims]
    assert len(ids) == len(set(ids)) and len(ids) >= 6
    statuses = {"unverified", "contested", "retracted-foundation", "validated", "ongoing"}
    for c in claims:
        assert c["status"] in statuses, c["id"]
        assert c["title"] and c["claimants"] and c["date"] and c["venue_status"]
        for s in c["sources"]:
            assert s["url"].startswith("http"), s
    assert client.get("/api/claims/khafre-2025-press").status_code == 200
    assert client.get("/api/claims/nope").status_code == 404
    assert client.get("/api/claims?status=bogus").status_code == 422
    # chamber-size outlet variance is recorded, not hardcoded
    params = client.get("/api/claims/khafre-2025-press").json()["parameters"]
    assert "80" in params["chamber_size_m"]["cited"] and "90" in params["chamber_size_m"]["cited"]
    # trajectory links the dataset to its claims
    traj = client.get("/api/khafre/trajectory?num_cameras=2").json()
    assert "khafre-2025-press" in traj["claim_ids"]


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
