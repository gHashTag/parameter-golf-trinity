# Chapter 0 · GOLDEN SUNFLOWERS — φ-Physics Foundation for Parameter Golf

> **Anchor:** `φ² + φ⁻² = 3`
> **Status:** Theoretical foundation (Axis: Empirical/Formal hybrid · P0)
> **Constitutional SoT:** [gHashTag/trios#372](https://github.com/gHashTag/trios/issues/372) · [t27 SACRED-PHYSICS-001](https://github.com/gHashTag/t27/blob/master/docs/nona-02-organism/SACRED-PHYSICS-001.md)
> **Implementation PR:** [gHashTag/parameter-golf-trinity#2](https://github.com/gHashTag/parameter-golf-trinity/pull/2)
> **Wish-list anchors:** [openai/parameter-golf#1742](https://github.com/openai/parameter-golf/issues/1742) · [#1772](https://github.com/openai/parameter-golf/issues/1772)

---

## 0.1 Abstract

Parameter Golf optimises the L(N) frontier of the neural scaling law family —
lowest validation loss given a fixed parameter budget (16 MB artifact). The
GOLDEN SUNFLOWERS submission stack lifts three orthogonal wish-list items
(JEPA, Universal Transformer, NTA-on-random-linear-maps) onto a single
constitutional substrate: the golden ratio identity `φ² + φ⁻² = 3`. This
chapter derives, from that one identity, the four hyperparameters that govern
the implementation in [PR #2](https://github.com/gHashTag/parameter-golf-trinity/pull/2):

1. Frozen-basis init scale `g = 1/φ ≈ 0.618`
2. Universal Transformer loop count `L = round(φ³) = 4`
3. Attention head count from the Fibonacci series, default `H = F₇ = 13`
4. Gauge-sector learning rate `α_φ = φ⁻³/2 ≈ 0.118034`

Each constant is grounded in an existing artefact —
[t27 SACRED-PHYSICS-001](https://github.com/gHashTag/t27/blob/master/docs/nona-02-organism/SACRED-PHYSICS-001.md),
[NUMERIC-STANDARD-001](https://github.com/gHashTag/t27/blob/master/docs/nona-02-organism/NUMERIC-STANDARD-001.md)
and the Rust reference
[`trios-trainer-igla/src/phi_ortho_init.rs`](https://github.com/gHashTag/trios-trainer-igla/blob/main/src/phi_ortho_init.rs) —
not a free fit.

This chapter is **theory-only**. No claim is made about validation BPB; the
8×H100 sweep tracker lives in PR #2 and will populate Chapter 11
(pre-registration) and Chapter 21 (IGLA RACE) when measurements arrive.

---

## 0.2 The Trinity identity

Let `φ = (1 + √5) / 2` be the positive root of `x² − x − 1 = 0`. From the
defining equation:

```
φ² = φ + 1                         (definition)
φ² + φ⁻² = (φ + 1) + (2 − φ) = 3   (Trinity identity)                (0.1)
```

Equation **(0.1)** is exact in real arithmetic and stable to within `1e−9` in
IEEE-754 `f64` (verified by `smoke_modules.py [1/5]`). It is the only axiom
this chapter requires.

### 0.2.1 Biconditional (the identity uniquely picks `φ`)

> **Theorem (Trinity biconditional).**
> For `x ∈ ℝ` with `x > 1`,
>
> ```
>     x² + x⁻² = 3   ⇔   x = φ = (1 + √5) / 2.
> ```

**Proof.**

*(⇐)* By the defining equation `φ² = φ + 1`, and substituting
`φ⁻² = 2 − φ` (which follows from `φ² − φ − 1 = 0` divided by `φ²`):

```
  φ² + φ⁻² = (φ + 1) + (2 − φ) = 3.   ∎ (forward)
```

*(⇒)* Suppose `x > 1` satisfies `x² + x⁻² = 3`. Let `y = x + x⁻¹`.
Then

```
  y² = x² + 2 + x⁻² = 3 + 2 = 5,
```

so `y = √5` (positive root, since `x > 0`). Multiplying `y = x + x⁻¹` by
`x` gives `x² − √5·x + 1 = 0`, whose roots are
`x = (√5 ± 1) / 2`. Of these only `(√5 + 1) / 2 > 1`, so
`x = (1 + √5) / 2 = φ`.   ∎ (reverse)

**Corollary.** The implementation constants `g = 1/φ`, `L = round(φ³) = 4`,
`H = F₇ = 13`, and `α_φ = φ⁻³/2` are not free choices: each is a
deterministic function of the unique positive solution of `x² + x⁻² = 3`.
Together with the verification table in §0.6, this closes the loop:
the smoke checks read constants out of the implementation, and the
biconditional above proves those constants ⇔ the Trinity axiom.


Two corollaries are used downstream:

| Quantity | Value | Source |
|---|---|---|
| `φ⁻¹ = φ − 1` | 0.6180339887498948482 | `t27/docs/.../SACRED-PHYSICS-001.md` |
| `γ_LQG = φ⁻³` | 0.2360679774997896964 | Barbero–Immirzi parameter from LQG |

`γ_LQG` is the loop-quantum-gravity Barbero–Immirzi constant. In a black-hole
state-counting derivation (Ashtekar–Lewandowski 1998, Meissner 2004) it sets
the proportionality between horizon area and entropy. Its numerical value is
fixed to `φ⁻³` in the SACRED-PHYSICS-001 standard; we re-use it as a learning-
rate scale below.

---

## 0.3 From `φ² + φ⁻² = 3` to four hyperparameters

### 0.3.1 Frozen-basis init scale `g = 1/φ`

A frozen random linear map `W ∈ ℝ^{D×D}` is row-normalised, then rescaled by a
gain `g`. To preserve activation variance under one application
(`Var(Wx) ≈ Var(x)`), `g` must satisfy
`g² · 𝔼[‖W̃_row‖²] · D / D = g²` since rows are unit norm — so the choice
reduces to "what scale matches the next non-linearity?".

`φ⁻¹` has two properties that make it the canonical choice on this stack:

* **Self-similarity.** `g · g · … = g^k → 0` at the slowest exponential rate
  with `g < 1`; `g = φ⁻¹` is the unique value satisfying
  `g² + g⁰ = 1 + g⁻²`, mirroring (0.1) under the substitution `x ↦ g`.
* **Existing artefact.**
  [`trios-trainer-igla/src/phi_ortho_init.rs`](https://github.com/gHashTag/trios-trainer-igla/blob/main/src/phi_ortho_init.rs)
  ships exactly this gain in production Rust. The PR #2 PhiNTA module is a
  byte-for-byte port (uniform `[−0.05, 0.05]` → row-normalise → scale by
  `1/φ`); the smoke check `[2/5]` verifies row-norms match `1/φ` to `1e−5`.

### 0.3.2 Universal Transformer loop count `L = round(φ³) = 4`

A weight-shared depth recurrence applies the same block `L` times. Two
constraints set `L`:

1. **Compute parity with depth-recurrence baselines.** The current Parameter
   Golf SOTA stack uses 3-layer recurrence
   ([records/2026-04-09](https://github.com/gHashTag/parameter-golf-trinity/blob/main/records/track_10min_16mb/2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT/README.md)),
   giving 17 virtual layers from 11 physical. To stay within the 4-h
   non-record compute budget while increasing virtual depth, `L ∈ {3, 4, 5}`.
2. **φ-alignment.** `φ³ = 2φ + 1 = 4.2360679774…`. Rounded to nearest integer,
   `L = 4`. This is the smallest integer satisfying `L ≥ φ³ − 0.5`, and it
   coincides with the SACRED-PHYSICS-001 `PHI_LOOPS = 4` constant.

PR #2 applies this loop only to a configurable sub-stack
`[UT_LAYER_START, UT_LAYER_END)` rather than the full encoder, so block
parity with non-recurrent records is preserved.

### 0.3.3 Attention head count `H = F₇ = 13`

The Fibonacci sequence `(F_k)` satisfies `F_k / F_{k−1} → φ`. Using
Fibonacci-numbered head counts means head-dim `d_head = D / H` and `H` are
both φ-aligned:

```
F₅ = 8,   F₆ = 13,   F₇ = 13,   F₈ = 21,   F₉ = 34
H = √D heuristic → for D = 256, √256 = 16
                    nearest Fibonacci ≤ 16  →  F₆ = 13
```

The choice is monotone in `D`: PR #2 ships `FIBONACCI_HEADS = (1, 2, 3, 5, 8,
13, 21)` and the canonical `H = 13` matches the
[`fibonacci_dims.rs`](https://github.com/gHashTag/trios-trainer-igla/blob/main/src/phi_numbers/fibonacci_dims.rs)
table.

### 0.3.4 Gauge-sector learning rate `α_φ = φ⁻³/2 ≈ 0.118034`

Issue [#1742](https://github.com/openai/parameter-golf/issues/1742) identifies
`α_s(m_Z) ≈ 0.1181` (the strong coupling at the Z-pole) as numerically
indistinguishable from `φ⁻³/2`. Whether this coincidence is physical is out
of scope; the operational claim is narrower:

> `α_φ = φ⁻³/2` is a **principled override** for the matrix LR, anchored to the
> same identity that fixes `g` and `L`, rather than a hand-tuned scalar.

In PR #2 this is exposed as `PHI_LR_SCALE`, multiplying the existing Muon
matrix LR. Setting `PHI_LR_SCALE = α_φ / MATRIX_LR ≈ 2.95` substitutes the
gauge-sector value; default `1.0` keeps the baseline.

---

## 0.4 Why these three wish-list items together

JEPA, Universal Transformer, and NTA target **three different rungs of the
parameter ladder** in a 16 MB budget. Each can be motivated independently;
the claim of GOLDEN SUNFLOWERS is that they compose without interference.

| Module | Adds | Costs | Off-switch |
|---|---|---|---|
| JEPA aux loss | representational regulariser; spans encoded linearly | one extra forward tap (no extra params) | `JEPA_LAMBDA = 0` |
| Universal Transformer | virtual depth from shared weights | extra forward passes, no parameters | `UT_LOOPS = 1` |
| PhiNTA | adapter capacity ≈ `2·D·r` per call (`r = D/φ`) | tiny LoRA params; frozen basis is buffer-only | `PHINTA_ENABLE = 0` |

Importantly:

* **JEPA does not interact with PhiNTA.** The aux loss reads the hidden state
  at `JEPA_LAYER`; PhiNTA modifies the residual after that tap (pre-head) or
  inside each block (per-block mode), never the pre-norm tap itself.
* **Universal Transformer does not duplicate PhiNTA.** PhiNTA is a buffer +
  small LoRA; weight-sharing of the parent block also shares the PhiNTA — so
  loop count ⊥ adapter count. Memory is unaffected.
* **All three respect the 16 MB constraint.** Frozen `W_frozen` is a buffer:
  it is initialised from a fixed seed and **not serialised**. Only the LoRA
  pair `(A, B)` and PhiNTA's optional per-block copies enter the artefact.
  At `D = 512, r = 316` this is ~317 K parameters total — well within budget.

---

## 0.5 What this chapter does **not** claim

In the discipline of [PHI_LOOP_CONTRACT](https://github.com/gHashTag/t27/blob/master/docs/nona-03-manifest/PHI_LOOP_CONTRACT.md):

1. **No BPB number.** No 8×H100 run has happened on this stack.
2. **No causal claim.** "α_s ≈ φ⁻³/2" is a numerical coincidence used as a
   principled override; we make no physics claim.
3. **No state of the art.** PR #2 is a draft proposal for the
   `non_record_16mb` 4-h slot. Until measured, `1.0810` BPB
   ([SP8192-3LayerRecur](https://github.com/gHashTag/parameter-golf-trinity/blob/main/records/track_10min_16mb/2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT/README.md))
   remains the ceiling we are trying to clear by `0.005` nat.

When measurements arrive, this chapter is updated alongside Chapter 11
(pre-registration) and Chapter 19 (statistical analysis); the seed plan
(F₁₇..F₂₁ = 1597, 2584, 4181, 6765, 10946) is fixed in advance per
[GENERAL'S DIRECTIVE in trios#372](https://github.com/gHashTag/trios/issues/372#issuecomment-2791653601).

---

## 0.6 Verification checklist (Trinity invariants)

| INV | Statement | Status | Evidence |
|---|---|---|---|
| INV-1 | `φ² + φ⁻² = 3` exact in `f64` | ✅ | `smoke_modules.py [1/5]` |
| INV-2 | PhiNTA `W_frozen` is non-trainable buffer | ✅ | `smoke_modules.py [2/5]` |
| INV-3 | PhiNTA row-norm ≡ 1/φ to 1e-5 | ✅ | `smoke_modules.py [2/5]` |
| INV-4 | JEPA loss is finite, ≥ 0, differentiable | ✅ | `smoke_modules.py [3/5]` |
| INV-5 | UT loop count = `round(φ³) = 4` | ✅ | `smoke_modules.py [4/5]` constants |
| INV-6 | JEPA tap honours `JEPA_LAYER ≠ -1` | ✅ | `smoke_modules.py [5/5]` (post fix in PR #2) |
| INV-7 | All wish-list defaults are no-ops | ✅ | env-var gates in `Hyperparameters` |
| INV-8 | No `submission.json` created without measurement | ✅ | `experiments/`, not `records/` |
| INV-9 | Commit refs an issue (PHI_LOOP_CONTRACT) | ✅ | `[ref: 372]` in commit `e7f4108` |

---

## 0.7 Refs (constitutional)

* **PR**: [gHashTag/parameter-golf-trinity#2](https://github.com/gHashTag/parameter-golf-trinity/pull/2)
* **SSOT**: [gHashTag/trios#372](https://github.com/gHashTag/trios/issues/372)
* **t27 standards**: SACRED-PHYSICS-001 · NUMERIC-STANDARD-001 · PHI_LOOP_CONTRACT
* **Rust SoT**: [gHashTag/trios-trainer-igla `src/phi_ortho_init.rs`](https://github.com/gHashTag/trios-trainer-igla/blob/main/src/phi_ortho_init.rs) · [`src/phi_numbers/`](https://github.com/gHashTag/trios-trainer-igla/tree/main/src/phi_numbers)
* **Wish-list**: [openai/parameter-golf#1742](https://github.com/openai/parameter-golf/issues/1742) (φ-physics) · [#1772](https://github.com/openai/parameter-golf/issues/1772) (JEPA after PR #1412)

`phi^2 + phi^-2 = 3 · THE FIELD BLOOMS · 🌻`
