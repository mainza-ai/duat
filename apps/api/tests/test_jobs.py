"""Store + jobs API tests (isolated temp DB via conftest)."""
import time

from fastapi.testclient import TestClient

from apps.api.main import app
from apps.api.pipeline import store as store

client = TestClient(app)


def test_store_crud_roundtrip():
    store.create_job("job_t1", "ensemble", {"a": 1})
    store.update_job("job_t1", status="running", progress=0.5)
    job = store.get_job("job_t1")
    assert job["status"] == "running" and abs(job["progress"] - 0.5) < 1e-9
    store.record_operation("op_t1", "atlasChisel", "hash1", job_id="job_t1")
    store.update_operation("op_t1", "succeeded", {"done": True})
    op = store.get_operation("op_t1")
    assert op["status"] == "succeeded" and op["job_id"] == "job_t1"
    store.record_ensemble("sha_t1", {"manifest": True})
    assert store.get_ensemble("sha_t1")["manifest_json"] == '{"manifest": true}'
    ref = store.store_bytes(b"hello-duat", kind="png")
    assert store.load_bytes(ref["sha256"]) == b"hello-duat"
    assert len(store.list_jobs(limit=10)) >= 1


def test_jobs_ensemble_end_to_end_mock():
    body = {"kind": "ensemble",
            "request": {"camera": {"pose": {"position": [-38.5, -20.0, -45.0],
                                                    "quaternion": [0, 0, 0, 1]}},
                        "seeds": [7], "voxel_size": 0.25, "mode": "mock"}}
    r1 = client.post("/api/jobs", json=body)
    assert r1.status_code == 200, r1.text
    jid = r1.json()["id"]
    assert r1.json()["deduplicated"] is False
    # Deterministic IDs: resubmitting the same work returns the same job.
    r2 = client.post("/api/jobs", json=body)
    assert r2.json()["id"] == jid and r2.json()["deduplicated"] is True
    for _ in range(100):
        st = client.get(f"/api/jobs/{jid}").json()
        if st["status"] == "done":
            break
        time.sleep(0.1)
    assert st["status"] == "done", st
    assert st["result"]["provenance_sha256"]
    assert st["progress"] == 1.0


def test_jobs_unknown_and_bad_kind():
    assert client.get("/api/jobs/nope").status_code == 404
    assert client.post("/api/jobs/nope/cancel").status_code == 404
    r = client.post("/api/jobs", json={"kind": "ensemble", "request": {}})
    assert r.status_code == 200  # empty request still validates with defaults


def test_provenance_served_from_store():
    body = {"kind": "ensemble",
            "request": {"camera": {"pose": {"position": [-38.5, -20.0, -45.0],
                                                    "quaternion": [0, 0, 0, 1]}},
                        "seeds": [11], "mode": "mock"}}
    jid = client.post("/api/jobs", json=body).json()["id"]
    for _ in range(100):
        st = client.get(f"/api/jobs/{jid}").json()
        if st["status"] == "done":
            break
        time.sleep(0.1)
    sha = st["result"]["provenance_sha256"]
    pr = client.get(f"/api/provenance/{sha}")
    assert pr.status_code == 200
    assert pr.json()["provenance_sha256"] == sha
    assert client.get("/api/provenance/unknown").status_code == 404
    assert client.get("/api/ready").status_code == 200
