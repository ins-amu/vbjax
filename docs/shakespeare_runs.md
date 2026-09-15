# Tiny Shakespeare reproduction — run comparison (track B)

Faithful reproduction of the 2026 paper's Tiny Shakespeare task (bioRxiv
2026.08.06.743205, fig. 2b) with the paper-faithful core (`vbjax/ct_mhsa.py`,
track A). Script: `vbjax/app/shakespeare_paper.py`.

Fixed architecture (all runs, per paper Methods): BPE-4096 on
`examples/input.txt` (341,869 tokens), 1 block, 6 heads (areas) × 32² = 1024
ensembles, d_model=192, retention λ=0.95 (u=dt/τ=0.05), learn
E, P, W_q, W_y; W_k, W_v frozen random; residual readout x_t + y_t; batch 32,
seq 32, AdamW + grad-clip 1.0, seed 42.

## Runs

| # | job | key config deltas | best val CE | end val CE | outcome |
|---|-----|-------------------|-------------|------------|---------|
| 1 | 4827684 | init σ=0.02, peak lr 3e-4, 3000 steps | 7.06 @ ~1200 | 8.83 | ❌ no real learning — val_acc stuck at most-common-token baseline; E,P 3.5× too small → activations ~1e-3, gradients vanish |
| 2 | 4827685 | init 1/√192, peak lr 1e-4, warmup 500, decay→5000 | **6.49 @ 2600** | 7.01 @ 3000 | ✅ best run — smooth descent 0→2600, late drift |
| 3 | 4827686 | = run 2, 5000 steps (same trajectory, seed 42) | 6.49 @ 2600 | 7.81 @ 5000 | ✅ confirms late drift is systematic, not a fluke |
| 4 | 4827687 | peak lr 5e-5, decay→5000 | ~8.3 | ~8.27 @ 5000 | ❌ too slow — barely below random-guess (8.32) |
| 5 | 4827688 | peak lr 1e-4, decay→2000 | ~7.65 | 7.86 @ 5000 | ❌ early LR floor destabilized the middle (val spike 10.3 @ 2450), never reached run 2's basin |

Artifacts per run: `runs/<name>/{config.json, loss_curve.png}`
(run 1 = `shakespeare_run1_*` superseded by run 2 dir naming; pkl checkpoints
kept on cluster at `~/mhsa_e0/work/vbjax-tree/runs/`).

## Findings

1. **The model learns**: 8.32 (random-guess CE for vocab 4096) → 6.49 best val
   (run 2/3), val accuracy 0.0002 → 0.137.
2. **Reproducible**: runs 2 and 3 share an identical trajectory through step
   3000 (same seed/config) — the pipeline is deterministic.
3. **LR window is narrow**: 5e-5 stalls (run 4); 1e-4 learns but diverges
   after ~step 2500 (runs 2–3); flooring the LR early makes it worse (run 5).
   The instability is systematic and schedule-sensitive, not seed noise.
4. **Gap to the paper**: fig. 2b reaches ~1.8 val CE at 5000 epochs; our best
   is 6.49. The Methods section I have says only "5000 epochs, Adam, early
   stopping" — the exact LR/optimizer recipe is not specified.

## Next steps to close the gap (needs owner green-light; open-ended GPU work)

- Obtain the paper's exact optimizer recipe (supplementary / authors) — highest
  value by far.
- Controlled experiments: (a) plain Adam wd=0 vs AdamW wd=0.01 (current);
  (b) β2 ∈ {0.95, 0.999}; (c) peak lr ∈ {7e-5, 8e-5} with full-5000 decay;
  (d) in-block LN on (`use_ln=True`) to bound logit growth.
- If the paper's recipe can't be obtained, a small grid over (a)–(d) on the
  GPU node (~4–8 min wall per run) should find a stably-descending config.

## Gap-closure stages (2026-09-15, clean-room)

**Stage 1 diagnosis (instrumented run, `runs/shakespeare_diag`):** unnormalized
linear attention at unit input scale equilibrates with max|M| ~ 40, max|logit| ~
240, and grad norm permanently above the 1.0 clip from step ~950 → softmax
saturated; training fights over scale, not content. Embeddings (e_norm 64→65)
are NOT the initial culprit.

**Stage 2 ablations (5000 steps each):**

| run | config | best val CE | verdict |
|---|---|---|---|
| kvs | frozen W_K/W_V ×0.15 | 5.44 @3950 | worked, then exploded late (maxlogit 690) |
| lnkvs | + readout standardization | **4.78 @4950, stable** | winner — maxlogit bounded ~14, gn ~0.6, no clip |
| ln | readout standardization only | (see run dir) | |
| ctrl20k | unmodified, 20k steps | (see run dir) | |

**Stage 3 (30k steps, lnkvs):** best val CE **4.53** @ ~12–14k steps; val
plateaus ~4.55 while train declines to 3.83 (genuine generalization plateau,
not undertraining). Late explosion at step ~28k: rare-token embedding rows
blow up (global e_norm flat) and poison the frozen-scale memory (maxM → 9).

**Stage 3b:** + `--embed-clip 5.0` (per-row embedding norm guard). Pending.

**Current status:** 8.32 (random) → 6.49 (runs 2/3) → **4.53** (stage 3), stable
and reproducible. Remaining gap to paper's ~1.8 (fig 2b) consistent with
normalization/scale conventions and optimizer details not specified in the
available paper text; per-row embedding guard tests whether the late crash
was hiding further descent.

**Stage 3b result (`runs/shakespeare_stage3b`):** embed-clip 5.0 fully
prevents the late crash (maxM ≤ 0.02 through 26k steps vs 9.0 crash before;
early stop at 26100), but best val CE is **4.5331 @ step 13950 — identical to
stage 3**. Conclusion: ~4.53 is the genuine val plateau of this configuration;
the remaining gap to fig 2b's ~1.8 lives in unspecified training/scale
conventions of the paper, not in undertraining or instability.
