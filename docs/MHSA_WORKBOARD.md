# MHSA Workboard — paper↔code mapping + whole-brain reference

Status: **superseded for current work — see `docs/mhsa-session-2026-09-18.md`.** This file remains the
E-item planning board written on 2026-09-14; the live work is coordinated on the `mhsa` workboard
(16 cards) and summarised in the session doc.
Owner session group: TVB. Repo: `github.com/ins-amu/vbjax`.
Created: 2026-09-14.

## North star

The 2026 paper (Babu/Granier/Proix/**Woodman**/Jirsa/... , bioRxiv 2026.08.06.743205)
does **not** fully specify the whole-brain integration. That gap is exactly where
`vbjax` should serve as a *reference implementation*. Two complementary goals:

1. **Map the paper's math faithfully to code** (the core is a simplified rendition today — see `docs/mhsa-continuation-plan.md` "Repo vs paper").
2. **Use that faithful core as the seed for a whole-brain/TVB model** — the paper's own stated next step ("embed in The Virtual Brain with a realistic 3D connectome"). The visual-search / connectome work in `feat/mhsa-vsearch` lives here, explicitly labelled "beyond the paper".

Guiding rule: **paper-faithful core stays clean and separate from extensions**, so
reproduction is never entangled with our own agent/connectome additions.

## Compute convention (fixed, per user)

- Cluster: `inslurm` (= `slurm.lnn.timone.univ-mrs.fr`, user `marmaduke`), partition `run`.
- GPU nodes: `g1`, `g2`, `g3` (each `gres=gpu:1`; g1/g2/g3 63 GB, g2 31 GB).
- Apptainer + singularity available on the login node. Existing container pattern to reuse: `~/src/tvb-hybrid/tvb.def` (debian:bookworm-slim + `uv` venv in `/opt/venv`).
- **Not yet built**: a GPU-enabled JAX apptainer container. This is the first infra task.
- Cluster already has a `vbjax` checkout at `~/src/vbjax` (rev `e3c84a2`, main).
- No git credentials on the cluster → ship trees via `git archive` (see `remote-cluster-test-runs` skill); defeats of the home mount / NFS to be confirmed when we first execute.

## Legend
- `[ ]` not started · `[~]` in progress · `[x]` done (acceptance captured in the task)
- **bite-size**: each item is one checkable outcome, not a phase.

---

## E0 — Foundation / cluster & container (hard dependency, do first)

**E0.1** `[ ]` Write `apptainer/vbjax-jax.def` — debian:bookworm-slim + uv venv with a **CUDA-enabled jax/jaxlib** pin matching the GPU driver on `g1/g2/g3` (verify `nvidia-smi` driver/CUDA on a GPU node first — `ssh` from login node to compute nodes returned nothing, so confirm reachability method).
   *Acceptance:* built sif imports `jax` and `jaxlib>=x` and reports a CUDA device.
**E0.2** `[ ]` Build the container to a stable path (e.g. `~/src/vbjax-container/vbjax.sif`) and document a one-line rebuild command.
**E0.3** `[ ]` Write a reusable `sbatch` runner: `--partition=run --gres=gpu:1` (+ `-w g1/g2/g3` when pinned), mounts the sif, sets `MPLBACKEND=Agg`, points any JAX/XLA cache at a per-job `/tmp/<tag>_${SLURM_JOB_ID}` dir.
**E0.4** `[ ]` Prove GPU is actually used (not inferred): tiny JIT that prints `jax.devices()` + runs one matmul; grep the log for the GPU device. Reuse `gpu-performance-proof` skill.
**E0.5** `[ ]` Ship recipe: a one-line `git archive | ssh inslurm ...` that syncs the needed tree to a scratch dir on the cluster (per `remote-cluster-test-runs`).

---

## A — Paper-faithful core (`vbjax/ct_mhsa.py`)

**A.1** `[ ]` Correct λ/τ convention: repo `lam` currently means `Δt/τ` (update rate) but paper `λ=1−Δt/τ` (retention, 0.95). Decide representation, fix defaults, add a docstring + unit test pinning the recurrence matches eq. 1.
**A.2** `[ ]` Add the apical afferent matrix `A` (Gaussian on diagonal, width σ): `o = M A φ(q)`, `α = φ(k)^T A φ(q)`. Parameterize σ; default off (identity) to match "transformers id" case, on for robustness experiments.
**A.3** `[ ]` φ → threshold-linear (paper), not ReLU; keep configurable. Update projection code + tests.
**A.4** `[ ]` Make in-block LayerNorm + residual **optional, off by default** (paper has neither inside the block; token-level `x_t + y_t` residual + normalization at embedding/decoding stage).
**A.5** `[ ]` Learning split: support **frozen `W_k`,`W_v` (random) + learned `W_q`,`W_y`**; expose local L6 error-neuron rules (eqs. 12–15) as an alternative to autodiff. Keep autodiff available for reference/cross-check.
**A.6** `[ ]` Persist `Hyperparameters` (and task config) **with** every checkpoint; make train/analyze/eval load from the checkpoint instead of reconstructing (fixes the known d_model 64 vs 32 bug).
**A.7** `[ ]` Keep core importable without the visual-search app; add a `try: from .app...` guard or split module so paper tasks run dependency-light.
   *Acceptance (A):* a CPU unit test + a GPU smoke test assert eqs 1–3 (with A) reproduce a reference loop; checkpoint round-trips hyperparameters.

---

## B — Reproduce the paper: Tiny Shakespeare (fig. 2b) — first real validation

**B.1** `[ ]` BPE tokenizer (vocab 4096) for the provided `examples/input.txt`.
**B.2** `[ ]` Model config exactly as paper: 6 heads, **32²=1024** L2/3+L5IT ensembles per area (dv=dk=32), d=192, λ=0.95 (Δt=50ms, τ=1s), batch 32, Adam + early stop; learn `E`, `P`, `W_q`, `W_y`; keep `W_k`,`W_v` fixed.
**B.3** `[ ]` Training run on GPU (E0 container) with loss/val logging + seed pinned.
**B.4** `[ ]` Report validation-loss curve vs the paper's fig. 2b shape (not necessarily exact values — note where it differs and why).
   *Acceptance (B):* loss decreases and generation is non-trivial; curve + seed + config committed.

---

## C — Reproduce the paper: En→De translation (fig. 3e) incl. cross-attention

**C.1** `[ ]` Data pipeline: ManyThings En→De subset (≈1.6e5 pairs), BPE-16834, d=600.
**C.2** `[ ]` Encoder/decoder block config: 6 heads, **100²=10000** ensembles/area, dk=dq=dv=100; learn `W_q`,`W_y`; frozen `W_k`,`W_v`.
**C.3** `[ ]` Add cortico-cortical **cross-attention** `C` (Gaussian σ_c, eq. 17); confirm the redundancy/robustness claim vs random drop-in/out (fig. 3b,c).
**C.4** `[ ]` GPU training + validation-loss curves vs fig. 3e; illustrative translations.
   *Acceptance (C):* sensible translations; σ_c robustness effect reproduced qualitatively.

---

## D — Paper later experiments (as budget allows)

**D.1** `[ ]` Unsigned-RPE gain modulation (eq. 9): basal-ganglia TD value net (eqs. 18–20), `L5ET = e^{α|δ|}·ΣL5IT + η`, α=6, γ=0.7, η0=140; German-rewarded / French-silent task (fig. 5).
**D.2** `[ ]` Hippocampal (Hopfield) recall into L2/3, RPE-gated (eqs. 10–11); Polyak target net (τ=0.005) (fig. 5g,h).
**D.3** `[ ]` Mismatch with ECoG: 4 epilepsy patients, TIMIT, BHA 70–150 Hz, PC1/PC2 classification of head embeddings; LOO 3-fold + 1000 shuffle surrogates + FDR (fig. 4). Verify the STG/head mapping reproduces qualitatively.
   *Acceptance (D):* each sub-experiment produces the figure's qualitative effect + a documented comparison.

---

## E — Whole-brain / TVB reference (our extension — north-star goal)

**E.1** `[ ]` Define the whole-brain model spec as a doc: how areas=heads, blocks, thalamic nuclei, tract-length delays, and cross-attention map onto a real 3D connectome (the paper leaves this open — this is vbjax's original contribution).
**E.2** `[ ]` Decide head↔area mapping (one area = one head vs sub-heads per area) and block↔thalamic-nucleus mapping; document the reasoning + chosen parcellation.
**E.3** `[ ]` Reconcile the two roles of `C`: paper's cross-attention gating vs the existing `network_coupling` transmission-with-delays. Decide/merge deliberately.
**E.4** `[ ]` Port the active visual-search agent (`vbjax/app/visual_search/`, from `feat/mhsa-vsearch`) onto the faithful core, keeping it isolated as a "beyond the paper" app.
**E.5** `[ ]` Re-train the visual-search baseline (saccade clone → active RL) since checkpoints are lost; persist HP+config properly this time.
**E.6** `[ ]` Define TVB integration: data contract between the faithful core and TVB (signals/state), target environment (`~/src/tvb-hybrid/tvb.sif` exists on cluster — reuse path).
   *Acceptance (E):* a running whole-brain model on `inslurm` GPU whose core is the validated paper-faithful `ct_mhsa`, plus a doc explaining every design choice beyond the paper.

---

## Cross-cutting / hygiene

**X.1** `[ ]` Decide branch strategy: keep `feat/mhsa-gemini` core as base for A–C vs rebase `feat/mhsa-vsearch`; merge cleanly; never let extensions into the faithful core.
**X.2** `[ ]` Add GPU CI/proof harness (the E0 runner) as THE way heavy suites run on `inslurm` — update `remote-cluster-test-runs` skill with the GPU-container variant once proven.
**X.3** `[ ]` Checkpoint registry: artifact paths, seeds, configs, and loss curves for every run, committed alongside results.

## Immediate next step (when you green-light execution)
Start **E0.1** (container def + driver/CUDA pin). It's the only hard prerequisite for everything else and is low-risk.
