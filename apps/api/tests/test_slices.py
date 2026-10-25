"""Slice decoding: jet inversion accuracy, legend rejection, stripe peaks."""
import cv2
import numpy as np

from apps.api.pipeline.slices import decode_jet_anomaly, jet_lut, mask_legends, stripe_peaks


def _jet_bar(w: int = 200, h: int = 60) -> np.ndarray:
    lut = (jet_lut(256) * 255).astype(np.uint8)  # BGR
    idx = np.clip((np.linspace(0, 1, w) * 255).astype(int), 0, 255)
    bar = np.tile(lut[idx][None, :, :], (h, 1, 1))
    return bar


def test_jet_inversion_is_exact():
    truth = np.linspace(0, 1, 200, dtype=np.float32)
    bar = _jet_bar()
    anomaly, valid = decode_jet_anomaly(bar)
    assert valid.all()
    est = anomaly[:, :].mean(axis=0)
    assert float(np.abs(est - truth).max()) < 0.02


def test_flat_border_legends_masked_field_kept():
    bar = _jet_bar(200, 60).copy()
    frame = np.full((100, 240, 3), 128, np.uint8)  # flat gray frame = legend-like
    frame[20:80, 20:220] = cv2.resize(bar, (200, 60))
    anomaly, valid = decode_jet_anomaly(frame)
    assert not valid[0, 0] and not valid[99, 239]
    assert valid[50, 120]
    masked = 1.0 - float(valid.mean())
    assert 0.25 < masked < 0.60


def test_grayscale_fallback():
    gray = np.tile(np.linspace(0, 255, 160, dtype=np.uint8)[None, :], (80, 1))
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    anomaly, valid = decode_jet_anomaly(bgr)
    assert valid.all()
    assert abs(float(anomaly.mean()) - 0.5) < 0.05


def test_stripe_peaks_count_shafts():
    h, w = 120, 320
    field = np.zeros((h, w), dtype=np.float32)
    for cx in (40, 120, 200, 280):
        field[:, cx - 6:cx + 6] = 0.9
    out = stripe_peaks(field, np.ones_like(field, dtype=bool), min_prominence=0.2)
    assert out["peak_count"] == 4
    assert all(abs(a - b) < 0.03 for a, b in zip(out["peak_positions_norm"], [0.125, 0.375, 0.625, 0.875]))


def test_slices_endpoint_roundtrip():
    from fastapi.testclient import TestClient

    from apps.api.main import app

    bar = _jet_bar(160, 80)
    ok, buf = cv2.imencode(".png", bar)
    assert ok
    c = TestClient(app)
    r = c.post("/api/khafre/slices", files={"files": ("bar.png", buf.tobytes(), "image/png")})
    assert r.status_code == 200, r.text
    s = r.json()["slices"][0]
    assert s["anomaly_b64"].startswith("data:image/png;base64,")
    assert s["masked_fraction"] < 0.05


def test_press_fixture_images_decode():
    import glob
    import os

    paths = sorted(glob.glob("dev-docs/images/*"))
    assert len(paths) >= 3, "expected press fixtures in dev-docs/images"
    n_ok = 0
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            continue
        anomaly, valid = decode_jet_anomaly(img)
        assert anomaly.min() >= 0.0 and anomaly.max() <= 1.0
        masked = 1.0 - float(valid.mean())
        assert masked < 0.90, f"{p}: over-masked ({masked:.2f})"
        n_ok += 1
    assert n_ok >= 3
