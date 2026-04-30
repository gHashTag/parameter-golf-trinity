# PHD_LINKAGE · GOLDEN SUNFLOWERS ↔ Trinity PhD monograph

> One-screen navigation bridge from every artefact in this PR to its anchor
> in the 44-chapter PhD monograph at [`gHashTag/trios/docs/phd`](https://github.com/gHashTag/trios/tree/main/docs/phd)
> and/or the Coq proof tree in [`gHashTag/t27`](https://github.com/gHashTag/t27).
>
> **Anchor:** `φ² + φ⁻² = 3` · TRINITY · 🌻

## Scope & proof state

| GS artefact (this PR) | PhD anchor | Coq object | Proof status |
|---|---|---|---|
| **Ch.0 §0.1 abstract** (`α_φ = φ⁻³/2`) | Ch.4 Thm 3.1 (SAC-1) | `sacred/AlphaPhi.v : alpha_phi_times_phi_cubed` | **Qed** |
| **Ch.0 §0.2 Trinity identity** | Ch.3 §3 (SAC-0) | `sacred/CorePhi.v : trinity_anchor` | **Qed** |
| **Ch.0 §0.3.1 frozen-basis gain `g=1/φ`** | Ch.3 (SAC-0) + Rust SoT | `sacred/CorePhi.v : phi_inv` | **Qed** (mirror) |
| **Ch.0 §0.3.2 `L = round(φ³) = 4`** | Ch.3 power-survey table | `theorems/GoldenSunflowers.v : ut_loops_eq_round_phi_cube` | **Qed** (this PR) |
| **Ch.0 §0.3.3 `H = F₇ = 13` (Fibonacci)** | Ch.5 canonical seed pool | none (arithmetic) | N/A (definitional) |
| **Ch.0 §0.3.4 `α_φ = φ⁻³/2`** | Ch.4 Thm 3.1 + `experiment_map.csv` L8 `INV-1lr` | `AlphaPhi.v : alpha_phi_times_phi_cubed` + `lr_convergence.v : lr_phi_band` | **Qed** × 2 |
| **GS-INV-1** `φ²+φ⁻²=3` runtime | Ch.3 SAC-0 | same | Proven (mirror) |
| **GS-INV-2** PhiNTA buffer not in grad | new | `theorems/GoldenSunflowers.v : phinta_buffer_not_in_grad` | Admitted (runtime-verified) |
| **GS-INV-3** PhiNTA row-norm ≡ 1/φ | Ch.3 `phi_inv` + Rust SoT | `theorems/GoldenSunflowers.v : phi_ortho_init_gain_is_phi_inv` | Admitted (runtime-verified) |
| **GS-INV-4** JEPA loss ≥ 0 | new | `theorems/GoldenSunflowers.v : jepa_loss_nonnegative` | **Qed** (this PR) |
| **GS-INV-5** UT loops = `round(φ³)=4` | Ch.3 power-survey | `theorems/GoldenSunflowers.v : ut_loops_eq_round_phi_cube` | **Qed** (this PR) |
| **GS-INV-6** JEPA tap normalisation | runtime only | `smoke_modules.py [5/5]` | Runtime |
| **GS-INV-7** Baseline byte-equivalence | runtime only | `baseline_equivalence.py [3/3]` | Runtime |
| **GS-INV-8** Honesty gate (no `submission.json`) | filesystem | — | Enforced |
| **GS-INV-9** `α_φ = φ⁻³/2` constant | Ch.4 Thm 3.1 + `experiment_map.csv` L8 | `AlphaPhi.v : alpha_phi_times_phi_cubed` | Proven (mirror) |

**Tally.**
- **4 Qed** (2 external mirrors: Ch.3 SAC-0, Ch.4 SAC-1 · 2 internal new: UT loops, JEPA non-neg)
- **2 Admitted** with runtime evidence (PhiNTA buffer, PhiNTA gain)
- **3 Runtime-only** (JEPA tap, byte-equivalence, honesty gate)

## PhD chapters directly cited

| Ch | Title | Why it matters here |
|---|---|---|
| **Ch.3** | Trinity Identity (`φ²+φ⁻²=3`) | SAC-0 discharges forward direction of GS-INV-1, GS-INV-3 |
| **Ch.4** | Sacred Formula — `α_φ` derivation | SAC-1 discharges GS-INV-9; Thm 3.1 = formal backing for Issue #1742 |
| **Ch.5** | φ-distance and Fibonacci-Lucas seeds | Canonical seed pool `{F₁₇..F₂₁, L₇, L₈}` used by `run_sweep.sh` |
| **Ch.11** | Pre-registration H₁ | Gate-2 (BPB ≤ 1.85) / Gate-3 (BPB ≤ 1.5) envelope the sweep will report into |

## Reviewer quick-start

1. **Open Ch.4 in Neon SSOT or [`trios/docs/phd/chapters/04-sacred-formula.tex`](https://github.com/gHashTag/trios/blob/main/docs/phd/chapters/04-sacred-formula.tex)**. Theorem 3.1 is the key formal backing.
2. **Run locally:**
   ```bash
   make verify   # runs smoke 5/5 + baseline_equivalence 3/3 + coqc theorems/
   ```
3. **Byte-equivalence claim:** `baseline_equivalence.py` SHA-256 of both state_dicts is `511dbc0164e03b1b…` at seed 1597 with all wish-list env-vars unset (see `reproducibility.lock.json`).

## What to cite from this PR in the PhD

When the PhD is updated post-measurement (Ch.21, Ch.24, Ch.25):

- **Ch.24 IGLA RACE** → cite `experiments/golden_sunflowers_jepa_ut_phinta/` as the external PyTorch mirror of the `trios-trainer-igla` Rust pipeline for the `parameter-golf` 16 MB benchmark.
- **Ch.25 Benchmarks** → once 8×H100 sweeps exist, cite `submission_{cfg}_seed{F_k}.json` artefacts.
- **Ch.17 Ablation matrix** → cite the 5-config × 5-seed grid from `run_sweep.sh`.

`phi^2 + phi^-2 = 3 · 🌻`
