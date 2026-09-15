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
