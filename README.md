<div align="center">

<img src="assets/milimo_logo.png" alt="Milimo Duat" width="120" />

# MILIMO DUAT

### *Subterranean Epistemic Observatory*

Turn **non-optical sensor geometry** — SAR tomography, muon, GPR, sonar, LiDAR —
into **navigable, photorealistic, provenance-tracked 3D interiors** via
World Labs Marble 2 / Atlas `atlasChisel`.

[![Status](https://img.shields.io/badge/status-beta-00F0FF?style=flat-square)](docs/BLUEPRINT.md)
[![Atlas](https://img.shields.io/badge/Atlas-atlasChisel-FF2D55?style=flat-square)](https://atlas-beta.worldlabs.ai/docs/depth-guided-generation)
[![Tests](https://img.shields.io/badge/tests-8_passing-00c853?style=flat-square)](apps/api/tests)
[![Stack](https://img.shields.io/badge/stack-React_19_%2B_R3F_%2B_FastAPI-FFB800?style=flat-square)](docs/BLUEPRINT.md)

**Every pixel is one of three tiers — never mistaken for ground truth.**

| Tier | Meaning | Color |
|:-----|:--------|:------|
| **1 · Measured** | Sensor boundary mesh, error bars, planar-depth maps | 🟦 `#00F0FF` |
| **2 · Inferred** | Dielectric / density / moisture proxies | 🟧 `#FFB800` |
| **3 · Imagined** | `atlasChisel` render + ensemble consensus & disagreement | 🟥 `#FF2D55` |

> `MODEL PREDICTION — CONDITIONED ON SENSOR GEOMETRY`

</div>

---

## ⚡ Quickstart

```bash
make install && make test     # deps + mock-mode suite (no API key needed)
make api                       # → :8000  (terminal 1)
make web                       # → :5173  (terminal 2)
# or: docker compose up --build
```

Set `WLT_API_KEY` in `.env` for live Atlas — otherwise the stack runs fully
offline on a deterministic mock client. Zero credits burned either way.

## 🏛️ Flagship dataset

**Khafre Pyramid subterranean SAR anomalies** — 8 helical shafts descending
hundreds of meters into twin 80 m basal chambers. Procedural, zero-config,
posterior-sampled (`σ`-perturbed meshes → ensemble variance → next-best-view
borescope targeting).

## 🧭 Two viewports

- **Epistemic Lens** — 2D shader scrubber blending Measured / Inferred /
  Imagined with variance-shimmer disagreement overlay, X-ray loupe, and
  hypothesis mixer (*Megalithic × Karst × Radar-artifact* on identical geometry).
- **Helical Descent** — 3D R3F observatory: shaft cutaway, 32 camera frustums,
  live RGBD point-cloud back-projection, borescope POV fly-through, chained-descent anchors.

## 🗺️ Repo map

```
apps/api/    FastAPI + pipeline (geometry · depth · prompts · atlas client · ensemble)
apps/web/    React 19 + Three.js observatory
assets/      brand logos
wiki/        LLM-maintained knowledge base
docs/        implementation blueprint + math proofs
data/        local content-addressable cache
```

📖 Start with [`docs/BLUEPRINT.md`](docs/BLUEPRINT.md), then [`wiki/`](wiki/) —
especially [`atlas-compliance.md`](wiki/atlas-compliance.md), the normative
record of where the live Atlas API overrides earlier drafts.

---

<div align="center">

**Author — [Mainza Kangombe](https://www.linkedin.com/in/mainza-kangombe-6214295)**

*Built for open-ended wonder, epistemic rigor, and World Labs Marble 2 / Atlas.*

</div>
