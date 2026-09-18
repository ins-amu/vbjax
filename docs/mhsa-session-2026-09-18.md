# MHSA session state — 2026-09-18

State of the whole-brain CT-MHSA fit work as of 2026-09-18. Companion docs:
`docs/mhsa-continuation-plan.md` (paper↔code mapping, 09-14) and
`docs/MHSA_WORKBOARD.md` (planning board, 09-14 — see its status note).

Coordinates
- Repo: `/home/duke/src/vbjax`, branch `feat/mhsa-paper-core`, remote `git@github.com:ins-amu/vbjax`.
- Coordination: workboard **`mhsa`** (16 cards); hourly automation `mhsa-workboard-hourly`.
- Cluster: `inslurm` (`slurm.lnn.timone.univ-mrs.fr`, user `marmaduke`); container `~/mhsa_e0/vbjax-jax.sif`;
  work trees `~/mhsa_e0/work/tree-*`; logs `~/mhsa_e0/logs/`; artifacts `~/mhsa_e0/runs_persist/`;
  data `~/mhsa_e0/data/`.

---

## 1. Two defects dominated everything

**1a. The trainer had no coupling (fixed).** `wb_train.py` computed the EMA-normalised
output `y_out = y / (y_ema + 1e-8)` and then discarded it, coupling regions through the
raw `y`. With `kv_scale` collapsed to 0.0032, raw `y` rms ≈ 5.7e-09, so the drive was
≈3.4e-09 against noise 0.145 (SNR ≈ 2e-08): 100 independent noise-driven blocks, and the
reported fc1 ≈ 0.209 came from the shared random projection matrices. Fix: `--coupling ema`
+ `--persist-carry`. Both attribution factors then rose ~300x and `kv_scale` stopped
collapsing (0.0032 → 0.40–0.65).

**1b. The metric measured startup transients (certified).** Legacy `fc_match` scored the
whole 600 s trace from the default balloon IC: contamination 103x–541x the resting std in
6 of 7 arms, and the settle cap bound in 5/12 runs at 1200 s and 8/12 at 2400 s.
**Certified rule** (`vbjax/app/wb_eval_std.py`, commit b5026a9) — 2400 s; 10-TR rolling
std; ref = median over the final 50%; W = first TR starting 50 consecutive TRs within 10%
of ref; NO cap; FC on `bold[W:]`. Use it for every comparison; never mix legacy numbers in.
Rankings inverted under it: `r3_fcy300` 0.172 → −0.002; `r3_fc2`'s lead was a cap artefact.
Certified standings: `ema_yout` +0.081/+0.179 · `fcy_noreg` +0.114/+0.021 ·
`current_logg` −0.082/+0.210 · `r3_fc2` +0.035/+0.066 · baseline −0.049/−0.007.

## 2. Why the loop gain G is not learnable by BPTT (measured)

`vbjax/app/wb_gradflow.py` exploits that G enters only as `aff = G·(C@y_prev)`, so
`dL/dG = Σ_t ⟨dL/daff_t, daff_t/dG⟩` with `aff` injected as an explicit scan input. The
identity reproduces `dL/dG` to **9 significant digits** (verified in 3 configs).

`ema_yout`, burn-in 200: `dL/dG = −5.0e-06`; `‖dL/daff‖ = 0.044`; `‖daff/dG‖ = 0.076` —
both healthy. Stage log-sensitivity `dlog‖stage‖/dlogG`: **aff 1.0000 → x 6.2e-04 →
e 3.0e-04**; 96.6% of the residual sits in the per-region z-score null space; the
per-chunk contributions cancel in time; direct (−5.1e-06) ≫ recursive (+4.5e-08);
removing the z-score makes `dL/dG` *smaller* (−1.8e-07), so the normaliser is not the cause.
**Mechanism:** G is a pure scalar gain (magnitude only, never direction), so its channel is
discarded three times over — noise dilution, the scale-free loss, and any drive
normalisation. Falsification test: `--drive-norm` drove `dlog‖x‖/dlogG` from 6.2e-04 to
**1.4e-07** and `dL/dG` to +1.1e-13 — it removed the very channel it was meant to expose.

## 3. What the cohort gives us

`vbjax/app/wb_cohort_build.py` extracted **K=261 subjects** (session _1: BOLD 300×100,
streamline counts, tract lengths) from the 1000Brains Schaefer-100 archive into
`~/mhsa_e0/data/kb_cohort/`. `vbjax/app/wb_cohort_fit.py` then selects (G, σ) on a cohort
objective and reports low-rank diagnostics.

- Selection: cohort ranking is stable (no subj1/subj2 flip as with n=2). The objective
  **peaks near G ≈ 4 with small σ** (search 0.163 at G=4/σ=0.005) and is flat/noisy
  elsewhere — a broad optimum, not a point (holdout top combos 0.15–0.17, sd 0.08–0.11).
  NOTE: those confirms used the 1200 s strict window, **not** the certified 2400 s rule —
  a certified re-score is filed as a card.

## 4. The structural finding: the simulated FC is a global mode

At the cohort-best combos the simulated FC is **almost exactly rank-1**:
`eig_sim top-10 = 97.9, 0.95, 0.55, 0.22, …` vs empirical `18.5, 11.2, 8.8, 6.9, …`;
within-network block FC **≈0.99 for all 17 blocks** (empirical 0.44–0.84); principal
angles between leading eigenspaces 47–56°; R² of the empirical FC reconstructed in the
model's rank-r eigenbasis 0.196 (r=1) → 0.487 (r=20); block-corr 0.357.
`wb_rank1_diag` (job 4828980) then showed this is **robust to every noise intervention**
(σ 0.05–1.0, noise variants, region-scaled empirical noise) with the readout `e` staying
**92–99% correlated across regions**, and the extended sweep showed it **persists at every
G** (worse block agreement at G=16: 0.222). So the scalar fc_match is earned by a single
global mode. **VERDICT (dispatcher worker on card 6490d046, from the 12-variant artifact): the
readout/input construction is the cause.** (a) coupling saturation REFUTED — at G=0.5 the spectrum
is still rank-1 (eig_sim_top5 [97.2, 0.57, 0.31, 0.15, 0.11], block_within_max 0.992); (b)/(d) noise
level and structure REFUTED — every noise variant stays ~95–99% first eigenvalue; (c) READOUT common
mode CONFIRMED — the raw surprise trace is already dominated by one component ([90.9, …]) at G=0.5,
with e_off 0.90–0.99 and cos_x ≈0.87–0.89 in every condition. `e = ‖ΔM‖_F` averaged over heads
collapses each region's directional surprise into one shared scalar, compounded by z-scoring the
balloon input (which drops inter-region amplitude). σ=3.0 grades the spectrum only by drowning the
network in noise (block 0.66, fc 0.049) — decoupling-by-noise, not a fix. Artifact:
`~/mhsa_e0/runs_persist/wbrank1_4828980/rank1_summary.json`.

## 5. Dead ends (do not retry)

Balloon initial conditions (touch only the first ~30 s; per-region ICs inflate fc 5x with
no model change) · `--drive-norm` (falsified above) · FC-shaped losses (fcy/fce/fc2) and
pulse excitation (no robust certified gain) · log-G reparameterisation (G moves; that is
not identifiability) · legacy whole-trace scoring · two-stage "fit G on one subject's FC
then BPTT" (optimum flips between subjects; best arm 0.116 vs 0.130).

## 6. Operating conventions (learned the hard way)

- **One cluster tree per workstream** (`~/mhsa_e0/work/tree-<name>`); never `rm -rf`
  another workstream's tree, and never re-ship a tree while its job runs (that deleted a
  tree under a running job once).
- **Single-command ssh only** — chained/multi-line ssh gets truncated.
- Ship trees with `git archive --format=tar --prefix=tree-<name>/ HEAD <paths> | ssh …`.
- Verify a spawned worker actually started (re-check its session after ~60 s); if it dies
  without a summary, collect its cluster artifacts directly (a provider stall did exactly
  that once — run 8c373665).
- Certified numbers only; flag any cap-bound window as unsettled.

## 7. Open threads (see the `mhsa` board)

1. Certified 2400 s re-score of the corner optima (G=4/σ=0.005, G=16/σ=0.05, G=4/σ=0.05).
2. ~~G-dependence test of the spectrum~~ **answered (refuted)**: lowering G to 0.5 leaves the
   spectrum rank-1 (eig_sim_top5 [97.2, …], block_within_max 0.992, e_off 0.907).
3. Redefine the **readout/input** against the rank-1 collapse (card `ef935186`): keep the per-head
   direction of ΔM instead of the scalar norm, stop z-scoring away inter-region amplitude, give each
   region an independent surprise — variants and acceptance criteria are merged into that card.
4. G policy: cohort-fitted range vs fixed constant.
5. Freeze cohort G/σ, BPTT the shape controls (W_q, W_y, kv_scale, τ), certified re-score.
6. Deck v3 (embed the cohort + low-rank figures), housekeeping (retire dead knobs).