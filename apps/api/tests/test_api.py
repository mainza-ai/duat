"""API contract tests (mock mode — no Atlas key needed)."""
from fastapi.testclient import TestClient

from apps.api.main import app

client = TestClient(app)


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["project"] == "Milimo Duat"


def test_trajectory_shape():
    r = client.get("/api/khafre/trajectory?num_cameras=4")
    assert r.status_code == 200
    body = r.json()
    assert len(body["targetCameras"]) == 4
    cam = body["targetCameras"][0]
    assert cam["intrinsics"]["width"] == 1280
    assert cam["extrinsics"]["coordinateSystem"] == "rub"


def test_ensemble_mock_roundtrip():
    r = client.post("/api/chisel/ensemble", json={
        "camera": {"pose": {"position": [-38.5, -20.0, -45.0], "quaternion": [0, 0, 0, 1]}},
        "hypothesis": "megalithic", "instrument_rig": "drop_flare",
        "seeds": [7, 42], "voxel_size": 0.25, "mock_mode": True})
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["provenance_sha256"]) == 64
    for k in ("measured_log_depth_b64", "inferred_reflectivity_b64",
              "imagined_consensus_rgb_b64", "disagreement_map_b64"):
        assert body["layers"][k].startswith("data:image/png;base64,")
