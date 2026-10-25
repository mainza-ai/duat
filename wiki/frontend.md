# Frontend

> One-line: dark observatory (`#090A0F`) with 2D Epistemic Lens shader +
> 3D helical navigator + hypothesis/instrument deck, all provenance-badged.

## Screens (`apps/web/src/`)

- `App.tsx` — header (DUAT mark, tab switcher, API status, export manifest),
  body (`EpistemicLensViewport` | `HelicalDescentNavigator`), footer.
- `components/EpistemicLensViewport.tsx` — R3F quad + GLSL:
  `decodePlanarDepth` (official inverted form, `0` → obsidian hatch),
  Tier1 contours + grid, Tier2 thermal ramp, Tier3 RGB, variance shimmer
  (`disagreement > 0.18`), X-ray loupe modes (off/measured/inferred/variance);
  procedural fallback textures; live `POST /api/chisel/ensemble` wiring via
  `lib/api.ts` + `TextureLoader`.
- `components/HelicalDescentNavigator.tsx` — shaft cutaway + spiral mesh,
  32 frustums (cyan selected, crimson NBV), `RGBDPointCloud` back-projection
  (`x=(u-cx)z/fx, y=-(v-cy)z/fy` in camera space → world), orbit vs borescope
  POV, auto-descend playback, intrinsics card, chaining callout.
- `lib/api.ts` — typed client + Atlas camera builders + PNG data-URL →
  THREE texture helpers.

## Palette

- Obsidian `#090A0F`, cyan `#00F0FF` (Measured), amber `#FFB800` (Inferred),
  crimson `#FF2D55` (variance), emerald RGB accent.

## Sources

- Drafts: `Epistemic_Lens_Viewport.md`, `EpistemicLensViewport.tsx`,
  `HelicalDescentNavigator.md`, `HelicalDescentNavigator.tsx`, `App.tsx`,
  `unify_frontend.md`.

## Links

- [Epistemic Stack](epistemic-stack.md), [Pipeline](pipeline.md).
