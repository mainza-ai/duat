# Verification

> One-line: prove the math, enforce 0-masking, and boot the demo with one command.

## Math proofs (implemented in `apps/api/tests/test_depth_math.py`)

1. Pinhole round-trip: `z = d/sqrt(((u-cx)/fx)^2+((v-cy)/fy)^2+1)` inverts
   raymarch distance exactly (analytic shaft test, tolerance 1e-4).
2. Inverted log codec round-trip: `p=round(255*(1-ln(z/zMin)/ln(zMax/zMin)))`,
   `z=p→zMin*(zMax/zMin)^(1-p/255)`; `zMin→255`, `zMax→0→clamped 1`,
   empty→`0`; max rel-error < 1% over `[zMin,zMax]`.
3. Zero-miss: rays missing geometry / outside `[zMin,zMax]` encode strictly `0`.

## Test suite

- `pytest apps/api/tests -q`: **12 passed** — depth math, inverted codec
  prompt composer, mock client determinism, API contract (mock mode, no key).

## Quickstart

- `make install && make api` (port 8000) + `make web` (port 5173);
  or `docker compose up --build`. Demo loads Khafre showcase zero-config;
  set `WLT_API_KEY` in `.env` for live Atlas, else deterministic mock.
- Optimize: NumPy-vectorized raymarch; Apple Silicon MPS/MLX hook where
  available (fallback CPU); ensemble tensor ops batched.

## Links

- [Atlas Compliance](atlas-compliance.md), [Pipeline](pipeline.md).
