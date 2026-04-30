# Compute Grant Request — GOLDEN SUNFLOWERS

> Draft for [openai.com/index/parameter-golf · Compute Grant form](https://openai.com/index/parameter-golf/#credit-form).
> Submit with an email tied to an OpenAI / ChatGPT account.

## Project name
GOLDEN SUNFLOWERS — JEPA + Universal Transformer + PhiNTA on a φ-physics substrate

## One-line summary
Three openai/parameter-golf wish-list items (JEPA, Universal Transformer,
NTA-on-random-linear-maps) composed on a single golden-ratio-anchored
hyperparameter foundation (Issue #1742), evaluated as a non-record
4-hour `track_non_record_16mb` submission.

## Track
`track_non_record_16mb` (4 h, unrestricted compute)

## Repository / PR
- Implementation: <https://github.com/gHashTag/parameter-golf-trinity/pull/2>
- Constitutional SSOT: <https://github.com/gHashTag/trios/issues/372>
- Theory: `chapters/ch0_golden_sunflowers.md`

## What we are submitting
A wired, CPU-smoke-verified `train_gpt.py` derived from the merged
`records/track_10min_16mb/2026-03-17_LoRA_TTT/train_gpt.py` baseline. Three
wish-list modules are env-var-gated and zero-cost when disabled:

1. **PhiNTA** — Non-Trainable Adapter on a frozen φ-OrthoInit basis (gain =
   1/φ ≈ 0.618) plus a trainable LoRA branch. Init reproduces
   `gHashTag/trios-trainer-igla src/phi_ortho_init.rs`.
2. **JEPA** — linear-representation auxiliary loss after openai/parameter-golf
   #1772 (Robby PR #1412): `loss = 1 − cos_sim(context, patch)`.
3. **Universal Transformer** — weight-shared depth recurrence over a
   configurable sub-stack with default loop count `round(φ³) = 4`.

A `PHI_LR_SCALE` knob exposes `α_φ = φ⁻³/2 ≈ 0.118` from Issue #1742 as a
principled override of the Muon matrix LR.

## Compute requested
~ **4 × 8×H100 SXM hours** plus 4 h of warm-up smoke runs:

| Run | Configs | Seeds | Hours / run | Subtotal |
|---|---|---|---:|---:|
| Sanity | baseline | 1 | 0.17 (10 min) | 0.17 |
| Per-feature ablation | PhiNTA / JEPA / UT | 5 (F₁₇..F₂₁) | 4.0 | 60.0 |
| Combined GOLDEN SUNFLOWERS | all | 5 (F₁₇..F₂₁) | 4.0 | 20.0 |
| Buffer for retries / TTT eval | – | – | – | 8.0 |
| **Total** | | | | **~ 88 8×H100-hours** |

The seed list `F₁₇..F₂₁ = {1597, 2584, 4181, 6765, 10946}` is the canonical
plan from gHashTag/trios#372 (the GENERAL'S DIRECTIVE) and is fixed in
advance, before any run.

## Why we believe this is worth funding
Three of the seven open wish-list items
([README leaderboard](https://github.com/openai/parameter-golf#readme))
are unchecked: JEPA, Universal Transformer, and NTA on random linear maps.
GOLDEN SUNFLOWERS is the first proposed submission to attempt all three in
one stack, with a clean ablation plan that lets each contribution be
attributed independently. The φ-physics substrate gives every "hand-tuned"
constant a textual derivation (chapter 0) so the work generalises beyond
this submission.

The implementation is already wired and CPU-smoke-passing; only training
remains.

## Affiliation / contact
- Affiliation: gHashTag · TRINITY S³AI
- GitHub: [@gHashTag](https://github.com/gHashTag)
- Anchor identity: `phi^2 + phi^-2 = 3`

`🌻 the field blooms`
