"""Build the whole-brain CT-MHSA G-identifiability deck (PDF) with matplotlib.

Run:  python3 docs/deck_mhsa.py [out.pdf]
Numbers are constants below so the deck can be regenerated as results land.

v3 (2026-09-19): cohort G/sigma selection verdict (replaces the two-stage "running"
slide) + the rank-1 degeneracy headline slide with the low-rank figures.
"""
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.image import imread

W, H = 13.333, 7.5
INK, MUTED, ACCENT = "#1a1a1a", "#5a5a5a", "#b03030"
IMG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
FIG_LOWRANK = os.path.join(IMG_DIR, "cohort_lowrank.png")
FIG_BLOCKS = os.path.join(IMG_DIR, "cohort_blocks.png")

# ------------------------------------------------------------------ results
CERTIFIED = [
    ["arm (certified: 2400 s, strict settle, no cap)", "fc1", "fc2_holdout"],
    ["ema_yout  — loss=current, ema coupling, persist carry", "+0.081", "+0.179"],
    ["fcy_noreg — FC-shaped loss on y-traces", "+0.114", "+0.021"],
    ["current_logg — log-G per-region MSE", "-0.082", "+0.210"],
    ["r3_fc2 — combined y+e FC loss", "+0.035", "+0.066"],
    ["r3_fcy300 — FC loss, 300 iters", "-0.010", "+0.002"],
    ["wb_train — original baseline (raw coupling)", "-0.049", "-0.007"],
]
LEGACY = [
    ["arm (legacy: 600 s whole trace, no settle)", "fc1", "fc2"],
    ["wb_train baseline", "+0.209", "+0.118"],
    ["r3_fc2", "+0.224", "+0.180"],
    ["r3_fcy300", "+0.172", "+0.106"],
]
ATTR = [
    ["checkpoint", "dL/dG", "||dL/daff||", "||daff/dG||", "null_frac", "dL/dG no z-score"],
    ["ema_yout (burn-in 200)", "-5.0e-06", "0.044", "0.076", "0.966", "-1.8e-07"],
    ["ema_yout (burn-in 0)", "-2.1e-06", "0.043", "0.054", "0.964", "-1.1e-07"],
    ["baseline wb_train", "-3.7e-12", "1.4e-04", "9.2e-06", "0.972", "-4.6e-15"],
    ["drive-norm fix (falsified)", "+1.1e-13", "1.4e-04", "2.3e-05", "0.968", "+4.6e-15"],
]
STAGES = [
    ["stage", "dlog||stage|| / dlogG"],
    ["aff  (the coupling drive)", "1.0000"],
    ["x = aff + noise  (block input)", "6.2e-04"],
    ["e  (surprise readout)", "3.0e-04"],
    ["e_vol  (TR-averaged)", "2.6e-04"],
    ["y (block output)", "1.4e-03"],
    ["M (memory)", "1.2e-03"],
]


def slide(pdf, title, blocks, subtitle=None, footer=None):
    """Render one 13.333 x 7.5 slide and save it into the PdfPages."""
    fig = plt.figure(figsize=(W, H))
    fig.patch.set_facecolor("white")
    fig.text(0.045, 0.945, title, fontsize=23, weight="bold", va="top", color=INK)
    if subtitle:
        fig.text(0.045, 0.885, subtitle, fontsize=12.5, va="top", color=MUTED)
    y = 0.83 if subtitle else 0.87
    for blk in blocks:
        k = blk.get("kind", "text")
        if k == "text":
            txt = blk["text"]
            fig.text(0.05, y, txt, fontsize=blk.get("fs", 12.5), va="top",
                     color=blk.get("color", INK), linespacing=1.55)
            y -= blk.get("dy", 0.042) * (txt.count("\n") + 1) + 0.015
        elif k == "table":
            if "rect" in blk:
                x, y0, w0, h0 = blk["rect"]
            else:  # flow-anchored: place just above the current flow line
                x = blk.get("x", 0.05)
                w0, h0 = blk["w"], blk["h"]
                y0 = y - h0
            ax = fig.add_axes([x, y0, w0, h0])
            ax.axis("off")
            t = ax.table(cellText=blk["data"], loc="center", cellLoc="left")
            t.auto_set_font_size(False)
            t.set_fontsize(blk.get("fs", 10.5))
            t.scale(1, 1.45)
            for (r, c), cell in t.get_celld().items():
                cell.set_edgecolor("#dddddd")
                if r == 0:
                    cell.set_text_props(weight="bold")
            y = y0 - 0.045
        elif k == "imagerow":
            # A row of side-by-side images that participates in the vertical flow.
            top = y
            cap = blk.get("maxh", 1.0)
            maxh = 0.0
            for it in blk["items"]:
                arr = imread(it["src"])
                ar = arr.shape[0] / arr.shape[1]
                x, w = it["x"], it["w"]
                h = w * ar
                if h > cap:                      # keep the row within the cap
                    w *= cap / h
                    h = cap
                ax = fig.add_axes([x, top - h, w, h])
                ax.imshow(arr)
                ax.axis("off")
                if it.get("label"):
                    fig.text(x + w / 2, top - h - 0.015, it["label"], fontsize=9.5,
                             va="top", ha="center", color=MUTED)
                maxh = max(maxh, h)
            y = top - maxh - 0.03
        elif k == "rule":
            fig.add_artist(plt.Line2D([0.045, 0.955], [y, y], color="#dddddd"))
            y -= 0.03
    if footer:
        fig.text(0.045, 0.035, footer, fontsize=10, color=MUTED)
    pdf.savefig(fig)
    plt.close(fig)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "docs/mhsa-g-identifiability.pdf"
    with PdfPages(out) as pdf:
        slide(pdf, "Whole-brain CT-MHSA: why the loop gain G is unidentifiable",
              [{"text": """Intermediate report v3 — 2026-09-19
marmaduke woodman · vbjax (feat/mhsa-paper-core)""", "fs": 14},
               {"kind": "rule"},
               {"text": """Bottom line
  1. Two of the three problems were BUGS, not science: the trainer had no
     functional coupling at all, and the metric was scoring startup transients.
  2. The loop gain G is structurally unidentifiable under BPTT: it acts only as
     a scalar gain on the drive, and this readout+loss pipeline discards amplitude.
  3. Cohort selection (K = 261) settles the G policy: the objective sits on a broad
     plateau at small sigma, so G is reported as a RANGE (~4), not a point (slide 10).
  4. HEADLINE structural finding: the simulated FC is essentially RANK-1 — the scalar
     fc_match was earned by a single global mode, not by network structure (slide 11).""",
                "fs": 13}],
              footer="All numbers on the following slides are reproducible from committed code; the certified metric rule is defined on slide 5.")

        slide(pdf, "The model and the fit",
              [{"text": """Per integration step (dt_train = 100 ms, window = 64 TR, BPTT with remat):
    aff_t  = G * (C @ y_{t-1})        C = structural connectome (spectral-radius 1)
    x_t    = aff_t + noise_t          OU/AR(1) noise, stationary std sigma
    M     += u (v k^T - M)            u = dt/tau;  memory (per region, per head)
    y      = W_Y M A phi(q);  y_coupled = y / (ema_rms(y) + 1e-8)
    e_t    = ||Delta M_t||_F  averaged over heads     ('surprise', >= 0)

Training loss   per-region z-scored MSE between TR-averaged e and empirical BOLD
Evaluation      Pearson corr between simulated and empirical FC matrices (fc_match)
                after a Windkessel balloon, scored on a settled window""", "fs": 12.5},
               {"kind": "rule"},
               {"text": "Tunable parameters: W_q, W_y, kv_scale, log_tau, log_sigma, log_noise_tc, and G (the loop gain).",
                "fs": 12.5, "color": ACCENT}])

        slide(pdf, "Bug 1 — the model had no coupling",
              [{"text": """The reference simulator couples regions through the EMA-normalised output:
    wholebrain_fc.py:   y_out = y / (y_ema + 1e-8);   history.at[t % L].set(y_out)

The trainer computed the identical quantity and then discarded it:
    wb_train.py:        y_out = y / (y_ema + 1e-8)
                        return (state.M, noise, y_ema, y), e      # couples through raw y""", "fs": 12.5},
               {"kind": "rule"},
               {"text": """With kv_scale collapsed to 0.0032, raw y has rms ~5.7e-09 -> the afferent drive is
~3.4e-09 against noise 0.145 (SNR ~2e-08).  The 'whole-brain network' was 100
independent noise-driven blocks, and fc1 = 0.209 came from the SHARED random
projection matrices, not from coupling.

Fix (--coupling ema + persistent carry): both attribution factors rose ~300x.
kv_scale then stopped collapsing: 0.0032 -> 0.40-0.65.""", "fs": 12.5}])

        slide(pdf, "Bug 2 — the metric was measuring startup transients",
              [{"text": """Legacy fc_match integrated 600 s from the default balloon initial condition and
scored the WHOLE trace.  Transient contamination, first-10-TR std / resting std:
103x - 541x in 6 of 7 arms; only the baseline (1.5x) and one arm (5.4x) were clean.

Settled-window cap bound in 5/12 subject-runs at 1200 s and 8/12 at 2400 s ->
those windows were truncated, not certified settled.""", "fs": 12.5},
               {"kind": "table", "data": LEGACY, "rect": [0.06, 0.40, 0.62, 0.22], "fs": 10.5},
               {"text": """Certified rule (used for all comparisons from here):
2400 s sim; 10-TR rolling std; ref = median over the final 50% of the trace;
W = first TR starting 50 consecutive TRs within 10% of ref; NO cap; FC on bold[W:].""",
                "fs": 12.5, "color": ACCENT}])

        slide(pdf, "The ranking inverted under the certified metric",
              [{"kind": "table", "data": CERTIFIED, "rect": [0.06, 0.34, 0.88, 0.45], "fs": 11.5},
               {"text": """r3_fcy300 was our 'best quality' arm on legacy numbers (fc1 0.172) and collapses to
-0.002 once settled; r3_fc2's lead was a cap artefact (W = 54 TR at 1200 s).
The robust winner is the SIMPLEST recipe: the original per-region MSE loss with
ema coupling + persistent carry.  The raw-coupling baseline is last.""", "fs": 12.5}])

        slide(pdf, "Instrument: exact attribution of dL/dG",
              [{"text": """G enters at exactly one place per step (aff_t = G * C @ y_{t-1}), so
    dL/dG = sum_t < dL/daff_t , daff_t/dG >
with aff injected as an explicit scan input (reverse demand exact per step) and
daff/dG obtained by jax.jvp (forward supply).  The identity reproduces dL/dG to
9 significant digits in every configuration -> the decomposition is exact.""", "fs": 12.5},
               {"kind": "table", "data": ATTR, "rect": [0.06, 0.34, 0.88, 0.30], "fs": 10.5},
               {"text": """Both factors are healthy at the drive (0.044 and 0.076) — so it is neither that the
loss ignores the drive, nor that G fails to move it.""", "fs": 12.5}])

        slide(pdf, "Where the identifiability is lost",
              [{"kind": "table", "data": STAGES, "rect": [0.06, 0.46, 0.52, 0.34], "fs": 11},
               {"text": """G controls aff perfectly, but x = aff + noise only 6.2e-04 — the drive is 4-5 orders
of magnitude below the noise floor, so the dilution happens at the sum, and
everything downstream inherits that factor.

Of what survives, 96.6% lies in the null space of the per-region z-score
(span{1, e_r} per region), and the residual cancels in time: the per-chunk
contributions alternate (+2.8e-6, +1.8e-6, -2.4e-6, -1.8e-6, +2.7e-7, +5.3e-6,
-1.0e-5, -5.8e-7) and sum to ~5e-6.  The recurrent path transmits ~nothing
(direct -5.1e-6 vs recursive +4.5e-8).""", "fs": 12.5}])

        slide(pdf, "The mechanism: G is a pure gain, and amplitude is discarded",
              [{"text": """aff = G * (C @ y_prev): G scales the drive's MAGNITUDE but never its DIRECTION.
So G can only be seen through amplitude — and this pipeline throws amplitude away
three times over:""", "fs": 12.5},
               {"kind": "rule"},
               {"text": """   (a) noise dilution      drive rms 7e-06 vs noise 0.145  (SNR ~5e-05)
   (b) scale-free loss     the z-scored MSE is invariant to per-region scaling
   (c) any normalisation   dividing the drive by its own rms removes G entirely""", "fs": 12.5},
               {"kind": "rule"},
               {"text": """Prediction and test: normalising the drive (the fix designed to raise SNR) should
destroy G — and it did.  log-sensitivity of x fell from 6.2e-04 to 1.4e-07, and
dL/dG went from -5e-06 to +1e-13.  Removing the z-score ALSO made it smaller
(-1.8e-07), so the normaliser is not the culprit: the amplitude channel itself is.""",
                "fs": 12.5}])

        slide(pdf, "Dead ends (recorded so they are not retried)",
              [{"text": """  balloon initial conditions   nuisance: they touch only the first ~30 s
                              (frozen-tail max |dBOLD| ~ 8e-06); per-region ICs
                              inflate fc2 from 0.118 to 0.623 with no model change
  FC-shaped losses (fcy/fce/fc2)  no robust gain under the certified metric
  pulse excitation            no effect on the certified score
  log-G reparameterisation     G moves, but that is not identifiability
  drive normalisation          falsified its own hypothesis (slide 8)
  normalisation asymmetry      real, but not causal (removing the z-score hurts)""", "fs": 12.5},
               {"kind": "rule"},
               {"text": """Consistent with the reference pipeline: wholebrain_fc.py never learns G — it SWEEPS
G and selects by fc_match, a stationary, amplitude-sensitive statistic.""", "fs": 12.5}])

        slide(pdf, "Cohort selection: fit G, sigma on fc_match (no gradients)",
              [{"text": """K = 261 subjects (1000Brains, Schaefer-100). Search: mean fc_match over 6 FIT subjects
across the G x sigma grid; top combos confirmed on 6 HELD-OUT subjects. Shape controls
frozen from ema_yout.""", "fs": 12},
               {"kind": "table",
                "data": [["G \\ sigma", "0.05", "0.15", "0.30"],
                         ["0.5", "0.077", "0.028", "-"],
                         ["1.0", "0.083", "0.063", "0.054"],
                         ["1.5", "0.092", "0.069", "0.046"],
                         ["2.5", "0.121", "0.114", "0.086"],
                         ["4.0", "0.150", "0.145", "0.133"]],
                "x": 0.05, "w": 0.52, "h": 0.22, "fs": 10.5},
               {"text": """Best by HELD-OUT (job 4828968): G=4.0 / sigma=0.05 -> fit 0.141 / holdout 0.103
(sd 0.077). The objective peaks at G~4 with small sigma and is flat/noisy elsewhere — a broad
plateau, not a sharp point; cohort ranking is consistent across combos (no subj1/subj2 flip).
The extended sweep (job 4828987) confirms G~4-6 / sigma~0.005 as a saturated plateau ~0.15
(holdout sds 0.08-0.11).""", "fs": 11, "dy": 0.038},
               {"text": """Certified 2400 s re-score of the corner optima (job 4829003):
  (G, sigma)          fit      holdout (sd)
  4.0 / 0.005       0.087     0.154 (0.126)   <-- BEST by holdout
  4.0 / 0.05        0.143     0.076
  16.0 / 0.05       0.158     0.061     (this REVERSED the 1200 s ranking: 16/0.05 led at 0.172)""",
                "fs": 10.5, "dy": 0.035}],
              footer="All cohort numbers from runs_persist/wb_cohort/cohort.json + re-score card 039556c9.")

        slide(pdf, "G policy verdict (finished): cohort-fitted range, not a point",
              [{"text": """Decision (card b8dbc2bf): adopt (a) — fit G/sigma on a cohort objective and report G as a
RANGE with the subject-to-subject spread, not a fixed point and not a per-subject fit.""",
                "fs": 12.5},
               {"kind": "rule"},
               {"text": """Evidence
  1. No sharp optimum. The certified re-score overturned the 1200 s best (16/0.05
     collapsed 0.172 -> 0.061) while 4/0.005 held at 0.154; certified holdouts
     0.154 / 0.076 / 0.061 with per-subject sds 0.08–0.13 = a noise-dominated
     plateau ~0.15, not a peak.
  2. Subject-to-subject spread is intrinsic. Single-subject fitting flips the ranking;
     only cohort selection is stable, and the cohort optimum sits on a plateau as
     wide as the holdout sd. No single biophysical G — the data support a range.
  3. BPTT cannot fit G. G is a pure scalar gain (wb_gradflow.py), unidentifiable in
     the training gradient, and widening it only modestly raises the scalar while
     the rank-1 degeneracy persists (next slide).  So G must be cohort-chosen.""",
                "fs": 12},
               {"kind": "rule"},
               {"text": """Two-stage BPTT (fit shape controls at frozen G) was tested and did NOT beat ema_yout
(best arm 0.116 vs 0.130); the optimum flips between the two single subjects, which is
exactly why the cohort is needed.  The scalar ceiling points to the structural fix,
not to more G tuning.""", "fs": 11.5, "color": ACCENT}],
              footer="Verdict follows the G-policy decision card b8dbc2bf.")

        slide(pdf, "Headline finding: the simulated FC is essentially RANK-1",
              [{"text": """At the cohort-best combos the simulated FC is almost exactly rank-1:
eig_sim top-10 = [97.9, 0.95, 0.55, 0.22, …] vs empirical [18.5, 11.2, 8.8, 6.9, …] — a single
global mode earns the scalar fc_match, not network structure (robust to every noise
intervention, job 4828980).""", "fs": 12},
               {"kind": "imagerow", "maxh": 0.27,
                "items": [
                  {"src": FIG_LOWRANK, "x": 0.05, "w": 0.52,
                   "label": "FC heatmaps, eigenspectra, R^2-vs-rank (cohort_lowrank.png)"},
                  {"src": FIG_BLOCKS, "x": 0.60, "w": 0.36,
                   "label": "within-block FC: sim ~0.99 vs emp 0.44-0.84 (cohort_blocks.png)"}]},
               {"text": """One global mode: within-network block FC ~0.99 for all 17 networks (empirical
0.44-0.84); principal angles 47-56 deg (poor alignment); R^2 of the empirical FC in the model's
rank-r basis 0.196 (r=1) -> 0.487 (r=20); block-FC corr 0.357.  The eig_corr_top10 = 0.83 is
misleading (driven by the single dominant eigenvalue).""", "fs": 11, "dy": 0.038},
               {"kind": "rule"},
               {"text": """The scalar fc_match is NOT evidence of recovered network structure — it is earned
by a trivial common mode.  This structural ceiling limits the model and points at the
readout/input construction as the cause (card ef935186).""", "fs": 11.5,
                "color": ACCENT}],
              footer="Figures from runs_persist/wb_cohort/cohort_lowrank.png and cohort_blocks.png.")

        slide(pdf, "Status, caveats, next steps",
              [{"text": """Established: the machinery is correct and the metric is fair; the certified winner is
ema_yout (fc1 +0.081 / fc2 +0.179); G's unidentifiability has a measured mechanism,
the G policy is settled (cohort-fitted range G~4), and the rank-1 degeneracy is now
the headline structural finding.""", "fs": 12.5},
               {"kind": "rule"},
               {"text": """Caveats
  - the fits are still weak: best certified number is ~0.18 on one subject;
  - some certified windows still hit the cap (noted per-arm) and are not certified settled;
  - the winning recipe is n=3 seeds with a wide spread (legacy fc1 0.078 - 0.301);
  - the cohort optimum sits on a plateau as wide as the holdout sd (no sharp peak).""",
                "fs": 12.5},
               {"kind": "rule"},
               {"text": """Next
  1. attack the RANK-1 degeneracy at the readout/input: keep the per-head direction
     of Delta M instead of the scalar norm, stop z-scoring away inter-region
     amplitude, give each region an independent surprise (card ef935186);
  2. freeze cohort G/sigma and BPTT only the shape controls, certified re-score
     (card 4ec1331b);
  3. more seeds on the ema_yout recipe to bound its spread;
  4. retire the dead knobs; register the certified metric as the standard.""",
                "fs": 12.5, "color": ACCENT}])
    print("wrote", out)


if __name__ == "__main__":
    main()
