# MHSA Continuation Plan (v2 — grounded in the 2026 paper)

Status: revised after reading the source paper.
Date: 2026-09-14. Repo `github.com/ins-amu/vbjax`.

## Source of truth

Babu, Granier, Proix, **Woodman**, Jirsa, Balvočius, Saudargienė, Senn.
*"Cortico-subcortical multi-head self-attention as a substrate for cognitive performance."*
bioRxiv 2026.08.06.743205 (posted 2026-08-10), 38 pp.
Local copy: `~/Downloads/2026.08.06.743205v1.full.pdf` (text: `/tmp/mhsa2026.txt`).
Note: M. Woodman (Aix-Marseille) is a co-author — the paper is the spec, and `vbjax` is the intended JAX platform.

## What the paper actually specifies

**Mapping (eqs. 1–6).**
- L2/3 pyramidal ensembles = recurrent key–value memory `M_t^h ∈ R^{dv×dk}`, held in **neuronal activity** (not synapses). `dv` macrocolumns × `dk` microcolumns = `dv×dk` ensembles per area.
- L5IT = query gain-modulated retrieval: `L5IT = M_t^h ⊙ diag(φ(q_t^h))` (product of memory with apical query).
- L5ET = head output `o_t^h = Σ_j L5IT`.
- One **cortical area = one attention head**; a **block** = several areas fed by one thalamic nucleus, outputs integrated in a **higher-order thalamic nucleus** → token for the next block (thalamo-cortico-thalamic chains).
- Thalamo-cortical split: **core** projections → values (L2/3 basal, focused); **matrix** projections → keys (L1) and queries (L5a), diffuse.

**Core equations.**
- `M_t = (1 − Δt/τ^h) M_{t−Δt} + (Δt/τ^h) v_t φ(k_t)^T` (eq. 1). Retention factor `λ^h = 1 − Δt/τ^h`. τ^h head-specific.
- `o_t^h = M_t^h φ(q_t^h)` (eq. 2); `y_t = Σ_h W_y^h o_t^h` (eq. 3).
- **Apical afferent matrix A** (Gaussian on diagonal, width σ): `o = M A φ(q)`, `α = φ(k)^T A φ(q)`. Provides robustness to synaptic drop-in/out. **Currently missing in repo.**
- φ = threshold-linear (not ReLU).

**Learning.**
- **Only `W_q` and `W_y` are learned; `W_k`, `W_v` random & fixed** → high memory capacity, and *no BPTT needed*.
- Local rules: `ΔW_y ∝ e_y o^T` (eq. 12); query error `e_q = M^T W_y^T e_y`; `ΔW_q ∝ e_q x^T` (eq. 13), represented by **L6IT/L6ET error neurons** (eq. 14–15). Duality inference/learning via neuronal least-action (S6).

**Extensions beyond vanilla transformer.**
- **Cortico-cortical cross-attention**: L2/3 in area h → L5IT in area h′; matrix `C` Gaussian (σ_c). Adds redundancy/robustness vs random connectivity (fig. 3).
- **Unsigned RPE** `|δ|` from basal ganglia/midbrain gain-modulates L5ET: `L5ET = e^{α|δ|} · Σ L5IT + η` (eq. 9). Gating of reward-relevant output.
- **Hippocampal (Hopfield) memory** recalled into L2/3, gated by unsigned RPE (eqs. 10–11), extending memory horizon.
- Pathology framing (Discussion): epilepsy = excessive L2/3 recurrence; schizophrenia = over-strong self-attention to past; Parkinson's = aberrant movement gain.

**Concrete numbers (validation targets).**
- *Tiny Shakespeare (fig. 2)*: 6 heads, **32² = 1024** L2/3 + L5IT ensembles per area (dv=dk=32), d=192, BPE vocab 4096, λ=0.95 (Δt=50 ms, τ=1 s), batch 32, 5000 epochs, Adam + early stopping. Learn E, P, W_q, W_y; W_k,W_v fixed. → match fig. 2b validation loss.
- *En→De translation (fig. 3e)*: 6 heads, **100² = 10000** ensembles/area (dk=dq=dv=100), d=600, BPE 16834, 1.6e5 sentence pairs, batch 16, 5–6 epochs, 8:2 split.
- *RPE task*: α=6, γ^h=0.7, noise η0=140; German rewarded (chrF), French silent.
- *Hippocampus task*: Polyak τ=0.005 target net; α_hc coupling.
- *ECoG (fig. 4)*: 4 epilepsy patients, TIMIT sentences; BHA 70–150 Hz; predict PC1/PC2 of head embeddings from 5×50 ms bins; LOO 3-fold + shuffle surrogates + FDR. Trained model maps to superior temporal gyrus, each area → distinct head.

**Paper's own stated future work**: embed these networks in a large-scale **The Virtual Brain (TVB)** simulation with a realistic 3D connectome. (Matches the TVB session grouping and `vbjax`.)

## Corrections to earlier assumptions (DeepSeek chat vs paper)

- **Right**: head = cortical area; macrocolumns are the `d_v` value units, not heads.
- **Wrong/speculative (not in paper)**: "180–360 areas, 10–30 blocks, d_v≈1000–2000, ~400k macrocolumns, 32 heads/area → 500k". The paper uses **6 area-heads per block by choice**, and treats `dv×dk` as *model* dimensions (32 or 100). Do **not** build the plan on the chat's biological numerology.
- **Paper does not do** a connectome-scaffold whole-brain simulation or a visual-search task — those are *our* extensions.

## Repo vs paper — gap analysis

`vbjax/ct_mhsa.py` (both branches) is a reasonable but **simplified** rendition:
- [ ] **Missing apical A matrix** (eq. after 5; Algorithm 3 uses `M A φ(q)`).
- [ ] **λ/tau semantics differ.** Repo: `M ← M + lam·(target − M)`, i.e. `lam ≡ Δt/τ` (update rate). Paper: `λ = 1 − Δt/τ` (retention). Repo default `lam=0.9` ⇒ Δt/τ=0.9 (τ≈1.1Δt), whereas paper λ=0.95 ⇒ Δt/τ=0.05 (τ=20Δt). **Directly incompatible defaults — fix and document.**
- [ ] φ = ReLU in repo; paper threshold-linear (nonnegative, bounded below). Minor but affects A/α.
- [ ] Repo adds **LayerNorm + residual inside the block**; paper has neither in the core (normalization is at the token/embedding stage; residual is `x_t + y_t` at token level).
- [ ] Repo trains **all** params via autodiff; paper learns **only W_q, W_y** (W_k, W_v frozen random) using local L6 error rules.
- [ ] Repo's `C` is a **generic output→input transmission matrix** (network coupling, with delays); paper's `C` is **cross-attention apical gating** (L2/3→L5IT), Gaussian σ_c. Different role — conceptual conflation to resolve.
- [x] Linear (non-softmax) attention, outer-product memory, `scan` rollout — consistent with paper.

`feat/mhsa-gemini` ≈ faithful core skeleton (closest to paper). `feat/mhsa-vsearch` = core + **our own** connectome/visual-search agent (explicitly beyond the paper).

## Revised tracks

**A — Paper-faithful core (first, cheap ground truth).**
Merge core to main; fix `ct_mhsa.py` to eqs 1–3: add **A**, correct **λ/τ** convention, threshold-linear φ, make residual/LN optional (off by default), support **frozen W_k/W_v + learned W_q/W_y**. Persist Hyperparameters with any checkpoint.

**B — Reproduce Tiny Shakespeare (fig. 2b).**
Config exactly: 6 heads, 32²=1024/area, d=192, BPE-4096, λ=0.95, batch 32. Compare validation loss shape to fig. 2b. Then En→De translation (6×100², d=600). This is the paper's actual validation.

**C — Paper's later experiments (as budget allows).**
RPE gain modulation (eq. 9) → hippocampal recall (eqs. 10–11) → ECoG alignment (fig. 4).

**D — Our own extensions / TVB (long-term, explicitly beyond paper).**
Connectome-scaffold whole-brain model and active visual search stay as separate tracks; label them "beyond paper". Target: TVB + 3D connectome, which is the paper's stated next step.

## Technical debt
- HP not persisted with params (train vs analyze config mismatch) — fix for all checkpoints.
- No trained checkpoints exist; re-training required.
- Decide repo layout: keep paper-faithful core (`vbjax/ct_mhsa.py`) clean and isolate the visual-search app so paper reproduction isn't entangled with extensions.
