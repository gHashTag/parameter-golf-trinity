# 🌻 GOLDEN SUNFLOWERS — JEPA + Universal Transformer + PhiNTA

[![smoke](https://github.com/gHashTag/parameter-golf-trinity/actions/workflows/golden_sunflowers_smoke.yml/badge.svg?branch=feat%2Fgolden-sunflowers-jepa-universal-nta)](https://github.com/gHashTag/parameter-golf-trinity/actions/workflows/golden_sunflowers_smoke.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19227877.svg)](https://doi.org/10.5281/zenodo.19227877)
[![Anchor: phi^2 + phi^-2 = 3](https://img.shields.io/badge/anchor-%CF%86%C2%B2%2B%CF%86%E2%81%BB%C2%B2%3D3-d4af37)](https://github.com/gHashTag/t27/blob/master/docs/nona-02-organism/SACRED-PHYSICS-001.md)
[![Status: untrained proposal](https://img.shields.io/badge/status-untrained%20proposal-orange)](https://github.com/gHashTag/parameter-golf-trinity/pull/2)

**Track:** `track_non_record_16mb` (4-hour, unrestricted-compute slot from
[openai/parameter-golf#1742](https://github.com/openai/parameter-golf/issues/1742))
**Status:** **PROPOSAL — UNTRAINED.** Modules implemented and CPU-smoke-verified;
training run on 8×H100 still pending. This directory lives under `experiments/`
(not `records/`) so no untrained `submission.json` claims a BPB number.
**Anchor:** `φ² + φ⁻² = 3` · TRINITY · v3.0 · 🌻
**Trinity SoT:** [gHashTag/trios#372](https://github.com/gHashTag/trios/issues/372)

---

## Files in this directory

| File | Purpose |
|---|---|
| `train_gpt.py` | Submission training script (1547 LOC, derived from `2026-03-17_LoRA_TTT`) |
| `smoke_modules.py` | CPU smoke test (5/5 — φ-physics, PhiNTA, JEPA, UT, JEPA-tap normalisation) |
| `baseline_equivalence.py` | CPU proof: state_dict + forward loss are byte-identical to baseline at defaults |
| `smoke.log` | Last verified smoke run output |
| `run_sweep.sh` | 5-config × 5-seed sweep over F₁₇..F₂₁ (canonical seeds from trios#372) |
| `compute_grant.md` | Draft for the openai/parameter-golf compute-grant request form |
| `CITATION.cff` | Citation metadata referencing trios-trainer-igla Zenodo DOI |
| `reproducibility.lock.json` | Pinned commit SHAs + numeric constants (PhD-style lock) |
| `experiment_map.csv` | GS-INV-1..9 ↔ PhD anchor table (style of `trios/docs/phd/experiment_map.csv`) |
| `PHD_LINKAGE.md` | One-screen navigation bridge to the 44-chapter PhD monograph |
| `theorems/GoldenSunflowers.v` | Coq module — 2 Qed + 2 Admitted (compiles with `coqc 8.18+`) |
| `theorems/_CoqProject` | Coq project file for IDE / CI integration |
| `Makefile` | `make verify` runs parse + smoke + equivalence + cff + coq |
| `README.md` | This file |
| `../../chapters/ch0_golden_sunflowers.md` | Theoretical foundation (φ-physics derivation of every constant) |
| `../../.github/workflows/golden_sunflowers_smoke.yml` | CI — runs smoke on every PR/push touching this folder |

## What this PR adds

Three wish-list items from `openai/parameter-golf` are wired into a single
`train_gpt.py` derived from
[`records/track_10min_16mb/2026-03-17_LoRA_TTT/train_gpt.py`](../../records/track_10min_16mb/2026-03-17_LoRA_TTT/train_gpt.py)
(authoritative TTT-LoRA baseline). Every new feature is **off by default and
gated by env-vars** — when none are set, the model and optimizer are
byte-equivalent to the baseline.

| Wish-list item | Module | Env-vars | Default |
|---|---|---|---|
| 🌻 **NTA on random linear maps** | `PhiNTA` (frozen φ-OrthoInit basis + LoRA), pre-head **or** per-block | `PHINTA_ENABLE`, `PHINTA_RANK`, `PHINTA_INIT_SCALE`, `PHINTA_PER_BLOCK` | disabled |
| 🌻 **JEPA** auxiliary loss (linear-representation form) | `_jepa_loss` | `JEPA_LAMBDA`, `JEPA_MAX_SPAN_FRAC`, `JEPA_START_FRAC`, `JEPA_LAYER` | λ = 0 |
| 🌻 **Universal Transformer** (φ³ depth recurrence) | `GPT.forward` dispatcher | `UT_LOOPS`, `UT_LAYER_START`, `UT_LAYER_END` | 1 loop |
| Bonus — **φ-LR scaling** ([#1742](https://github.com/openai/parameter-golf/issues/1742)) | Muon LR multiplier | `PHI_LR_SCALE` | 1.0 |

---

## Module 1 · PhiNTA (Non-Trainable Adapter on φ-random basis)

```text
y = x · W_frozen + (x · A) · B
W_frozen ∈ ℝ^{D×D}   buffer, never updated, row-normalised then scaled by 1/φ
A        ∈ ℝ^{D×r}   trainable, σ = 0.02
B        ∈ ℝ^{r×D}   trainable, zero-init  (LoRA pattern)
```

- Initialisation reproduces `phi_ortho_init.rs` from
  [gHashTag/trios-trainer-igla](https://github.com/gHashTag/trios-trainer-igla/blob/main/src/phi_ortho_init.rs)
  (the Rust SoT for the trainer pipeline) byte-for-byte:
  uniform `[-0.05, 0.05]` → row-normalise → rescale by `1/φ ≈ 0.618`.
- `register_buffer` keeps `W_frozen` out of the optimizer **and** out of the
  parameter checkpoint (verified in smoke test — `nta.A.requires_grad and not nta.W_frozen.requires_grad`).
- Default `rank = round(D / φ)`, matching the canonical φ split.
- Inserted as a **post-final-norm pre-head residual** so it touches every
  prediction once without adding depth to the attention stack.

## Module 2 · JEPA (Joint Embedding Predictive Architecture)

Implements the Robby-Goldberg linear-representation formulation discussed in
[openai/parameter-golf#1772](https://github.com/openai/parameter-golf/issues/1772)
(after PR #1412):

```text
context = (h[a-1] − h[0])  +  (h[T-1] − h[b])     # before-span + after-span
patch   =  h[b]   − h[a-1]                          # the span itself
loss    = 1 − cos_sim(context, patch)               # add λ·loss to CE
```

`context + patch ≡ h[T-1] − h[0]` — they partition the full encoding, so
maximising cosine forces hidden states to encode spans **linearly**. The
JEPA tap is the final pre-norm representation by default (`JEPA_LAYER=-1`,
zero extra compute — same tensor that already feeds `final_norm`). When
`JEPA_LAMBDA == 0`, the branch is fully short-circuited.

## Module 3 · Universal Transformer (φ³ depth recurrence)

```python
ut_active = ut_loops > 1 and ut_layer_end > ut_layer_start
def maybe_loop(bi, x):
    if ut_active and ut_layer_start <= bi < ut_layer_end:
        for _ in range(ut_loops):
            x = self.blocks[bi](x, x0, qd, vd)   # SAME weights, ut_loops passes
        return x
    return self.blocks[bi](x, x0, qd, vd)
```

- Default loop count = `round(φ³) = round(4.236) = 4`.
- Loops a configurable sub-stack `[UT_LAYER_START, UT_LAYER_END)` rather than
  the whole stack; the rest of the encoder/decoder remains parameter-distinct.
- Identical behaviour to baseline when `UT_LOOPS=1` or
  `UT_LAYER_END=0` (verified in smoke `[4/4]`).

## Bonus · φ-LR scaling (Issue [#1742](https://github.com/openai/parameter-golf/issues/1742))

`α_φ = φ⁻³/2 ≈ 0.118034` is the gauge-sector eigenvalue of the Trinity
identity `φ² + φ⁻² = 3`. We expose `PHI_LR_SCALE` as a multiplier on the
Muon matrix LR so the wish-list LR can be A/B-tested against the
hand-tuned `MATRIX_LR=0.04` baseline without rewriting the optimizer.

---

## φ-physics constants (matched to t27 SACRED-PHYSICS-001)

The module-level constants in `train_gpt.py` are bit-for-bit equal to those in
[gHashTag/t27 · docs/nona-02-organism/SACRED-PHYSICS-001.md](https://github.com/gHashTag/t27/blob/master/docs/nona-02-organism/SACRED-PHYSICS-001.md):

| Symbol | Value | Source |
|---|---|---|
| `PHI` | 1.6180339887498948482 | golden ratio |
| `PHI_INV` | 0.6180339887498948482 | `φ − 1` |
| `PHI_INV_CUBE` | 0.2360679774997896964 | `γ_LQG` Barbero–Immirzi |
| `ALPHA_PHI` | 0.118033988749894… | `φ⁻³/2` (Issue [#1742](https://github.com/openai/parameter-golf/issues/1742)) |
| `PHI_LOOPS` | 4 | `round(φ³) = round(4.2361)` |
| `FIBONACCI_HEADS` | (1, 2, 3, 5, 8, 13, 21) | head-count canon |

Smoke test asserts `PHI**2 + PHI**-2 == 3.0` exactly (modulo fp64).

---

## Reproduction

### CPU smoke (verified — see `smoke.log`)

```bash
python experiments/golden_sunflowers_jepa_ut_phinta/smoke_modules.py
# [1/5] φ-physics OK: φ²+φ⁻²=3.000000000000 α_φ=0.118034 loops=4
# [2/5] PhiNTA OK: trainable=1664 frozen=4096 ratio=0.406
# [3/5] JEPA loss OK: 1.6922 (cosine-similarity form)
# [4/5] UT loop OK: ‖x_4‖/‖x_0‖=1.0406 expected=1.0406
# [5/5] JEPA tap normalisation OK: -1 → last block, in-range indices preserved
# 🌻 GOLDEN SUNFLOWERS smoke OK · 5/5 · phi^2 + phi^-2 = 3
```

CI runs the same smoke on every PR via
[.github/workflows/golden_sunflowers_smoke.yml](../../.github/workflows/golden_sunflowers_smoke.yml).

### Local full verify (smoke + equivalence + cff + coqc)

```bash
cd experiments/golden_sunflowers_jepa_ut_phinta
make verify
# train_gpt.py: parse OK
# 5/5 smoke + 3/3 equivalence + CITATION valid + theorems/GoldenSunflowers.v: coqc OK (2 Qed)
```

Requires `coqc 8.18+` for the proof step (the rest is pure Python + cffconvert).

### Recommended training sweeps (8×H100, 16 MB track)

Canonical Fibonacci seeds **F₁₇..F₂₁ = {1597, 2584, 4181, 6765, 10946}**
per the GENERAL'S DIRECTIVE in [trios#372](https://github.com/gHashTag/trios/issues/372#issuecomment-2791653601).

Use the bundled sweep script:

```bash
# Run all 5 configs across all 5 seeds (baseline / phinta / jepa / ut / all).
bash experiments/golden_sunflowers_jepa_ut_phinta/run_sweep.sh full

# Or one config at a time:
bash experiments/golden_sunflowers_jepa_ut_phinta/run_sweep.sh baseline
bash experiments/golden_sunflowers_jepa_ut_phinta/run_sweep.sh phinta
bash experiments/golden_sunflowers_jepa_ut_phinta/run_sweep.sh jepa
bash experiments/golden_sunflowers_jepa_ut_phinta/run_sweep.sh ut
bash experiments/golden_sunflowers_jepa_ut_phinta/run_sweep.sh all
```

Each config writes `train_seed${SEED}.log` next to `train_gpt.py`. Promotion
to `records/` happens only after a 3-seed mean and std are honest.

---

## Honesty / status

- **No `submission.json` is included** — that file in `records/` would imply a
  measured BPB. We do not have one yet. This PR is a wired, smoke-verified
  proposal for the `non_record_16mb` 4-hour slot.
- Each wish-list module is **independently togglable** so future PRs can
  attribute ΔBPB to the right contribution.
- All wish-list defaults are no-ops; merging this PR does not change behaviour
  of any existing record under `records/`.

---

## Constitutional traceability

- **Spec source**: t27 SACRED-PHYSICS-001 (`docs/nona-02-organism/SACRED-PHYSICS-001.md`)
- **Numeric standard**: t27 NUMERIC-STANDARD-001 (GF16 primary; future quant pass)
- **Master SSOT**: [gHashTag/trios#372](https://github.com/gHashTag/trios/issues/372)
- **Wish-list anchor**: [openai/parameter-golf#1742](https://github.com/openai/parameter-golf/issues/1742) ·
  [#1772](https://github.com/openai/parameter-golf/issues/1772) (JEPA discussion)
- **Rust SoT for φ-init**: [gHashTag/trios-trainer-igla src/phi_ortho_init.rs](https://github.com/gHashTag/trios-trainer-igla/blob/main/src/phi_ortho_init.rs)

`phi^2 + phi^-2 = 3 · THE FIELD BLOOMS`
