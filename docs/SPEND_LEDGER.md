# Atlas Spend Ledger — hard cap: **1,000 credits** for the entire implementation.

1,000 credits = US$1. Per-task credit prices live only in the platform
(Settings → Billing); the ledger counts **tasks submitted** and converts once
the rate card (`R₁`, `R₃₂`) is known: `USD = tasks × R / 1000`.

## Guardrails in force

- Dev project monthly cap: **$25** (set in platform before any live call).
- Pilot cap: **$150** (not authorized — pilot deferred).
- No transport retry may reuse a *different* idempotency key (client enforces).
- 402 → stop immediately, do not retry in a loop.

## Entries (every live submission, no exceptions)

| # | Date (UTC) | Task | Targets | Seeds | Idempotency key prefix | Operation ID | Est. credits |
|---|------------|------|---------|-------|------------------------|--------------|--------------|
| 1 | 2026-10-25 | atlasChisel probe (shaft #2, cam 04, megalithic) | 1 | 1 | duat-probe-… | `ae53860d17ff4c79b0f72512b2688531` | TBD (read Usage page) |

**Tasks submitted so far: 1.** Remaining budget: 999 tasks-worth of headroom is NOT
the right math — remaining *credits* are unknown until entry #1's cost is read
from Usage. **Do not submit any further live task until R₁ is filled in below.**

## Rate card (fill from Settings → Billing)

- `R₁` (1-target atlasChisel): ___
- `R₃₂` (32-target atlasChisel): ___
- Credits consumed by entry #1 (from Usage): ___

## Budget holds

- ⏸ Phase 6 pilot (28–40 tasks): HELD — exceeds the 1,000-credit cap at any plausible rate.
- ⏸ Any further live call: HELD until R₁ is recorded above.
