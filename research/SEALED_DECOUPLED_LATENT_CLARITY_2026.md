# Sealed 2026 Decoupled-Latent Clarity Check

Date: 2026-06-19

A new mechanism was found and proven on 2025: **decouple the set-piece latent
between paths**. Shrink the latent on the point/XV/captain path (regularises the
noisy lineout-steal / scrum latent for MAE), but let the bench/supersub head keep
the *full* unshrunk latent (impact bench forwards' value is set-piece driven).
Implemented via the new `supersub_latent_shrink` config knob.

## Why it looked promising (2025 dev)

Decomposing latent-shrink's team-value loss showed it is *entirely* the supersub:
XV +10, captain +7, but supersub ×3 **−78**. Holding the supersub on the unshrunk
signal recovers the loss, giving a clean two-leg 2025 Pareto win.

Promoted base `bench_points_twostage_13`: value_team 0.707 / value_xv 0.716 / MAE 7.550.

| candidate | 2025 team | 2025 xv | 2025 MAE | 2025 verdict |
|---|---:|---:|---:|---|
| `decoupled_latent025_sub100` | 0.714 | 0.722 | 7.517 | reject (MAE round robustness 2/5) |
| `decoupled_latent040_sub100` | 0.714 | 0.722 | 7.506 | **accept + all stability passed** |
| `decoupled_latent050_sub100` | 0.713 | 0.720 | 7.503 | **accept + all stability passed** |
| `decoupled_latent060_sub100` | 0.710 | 0.719 | 7.506 | reject (no Pareto) |
| `decoupled_latent075_sub100` | 0.710 | 0.719 | 7.516 | accept |
| `decoupled_latent050_sub075` | 0.689 | 0.720 | 7.503 | reject — partial unshrink collapses team value |

The `sub075` row is a clean mechanism control: the supersub needs the *full*
unshrunk latent, not a partial one.

## Why it is vetoed (sealed 2026)

The supersub decoupling generalises (team value holds flat on both seasons), but the
point-path latent shrink does **not** — the set-piece latent is real signal in 2026,
so shrinking it for MAE is a 2025-only trap.

Promoted sealed baseline: value_team 0.6875 / value_xv 0.7145 / MAE 7.3476.

| candidate | sealed team | dteam | sealed xv | dxv | sealed MAE | dMAE |
|---|---:|---:|---:|---:|---:|---:|
| `decoupled_latent040_sub100` | 0.6875 | +0.0000 | 0.7145 | −0.0000 | 7.3928 | **+0.0453** |
| `decoupled_latent050_sub100` | 0.6875 | +0.0000 | 0.7145 | −0.0000 | 7.3675 | **+0.0199** |

Lighter shrink → smaller sealed MAE penalty, but no shrink level yields a sealed
gain. Both are added to `KNOWN_SEALED_VETOES`. The `supersub_latent_shrink`
mechanism is retained in the code: it is sound and could be paired with a future
point-path change that *does* generalise.
