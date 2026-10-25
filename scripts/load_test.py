#!/usr/bin/env python3
"""Mock-only load probe (zero Atlas spend): hammers local-compute endpoints with
concurrency and reports p50/p95/max latency + error count.

Usage: python3 scripts/load_test.py [--base http://127.0.0.1:8001] [--n 20] [--workers 4]
"""
import json
import statistics
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = "http://127.0.0.1:8001"
N = 20
WORKERS = 4
for a in sys.argv[1:]:
    if a.startswith("--base="):
        BASE = a.split("=", 1)[1]
    elif a.startswith("--n="):
        N = int(a.split("=", 1)[1])
    elif a.startswith("--workers="):
        WORKERS = int(a.split("=", 1)[1])


def post(path, body):
    t = time.monotonic()
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            r.read()
            return time.monotonic() - t, r.status
    except Exception as e:
        return time.monotonic() - t, f"ERR {e}"[:120]


def get(path):
    t = time.monotonic()
    try:
        with urllib.request.urlopen(BASE + path, timeout=60) as r:
            r.read()
            return time.monotonic() - t, r.status
    except Exception as e:
        return time.monotonic() - t, f"ERR {e}"[:120]


JOBS = [
    ("GET /api/health", lambda: get("/api/health")),
    ("GET /trajectory", lambda: get("/api/khafre/trajectory?num_cameras=4")),
    ("POST /depth", lambda: post("/api/khafre/depth", {
        "camera": {"pose": {"position": [-38.5, -20.0, -45.0], "quaternion": [0, 0, 0, 1]}},
        "out_width": 160, "out_height": 90})),
    ("POST /ensemble mock", lambda: post("/api/chisel/ensemble", {
        "camera": {"pose": {"position": [-38.5, -20.0, -45.0], "quaternion": [0, 0, 0, 1]}},
        "seeds": [7], "mode": "mock"})),
]

with ThreadPoolExecutor(max_workers=WORKERS) as ex:
    for name, fn in JOBS:
        futs = [ex.submit(fn) for _ in range(N)]
        res = [f.result() for f in futs]
        lat = sorted(r[0] for r in res)
        errs = [r[1] for r in res if r[1] != 200]
        p50 = statistics.median(lat)
        p95 = lat[min(len(lat) - 1, int(len(lat) * 0.95))]
        print(f"{name:22s} n={N} p50={p50:6.2f}s p95={p95:6.2f}s max={lat[-1]:6.2f}s errors={len(errs)}")
        for e in errs[:3]:
            print("   ", e)
