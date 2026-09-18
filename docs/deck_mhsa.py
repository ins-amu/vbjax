"""Build the whole-brain CT-MHSA G-identifiability deck (PDF) with matplotlib.

Run:  python3 docs/deck_mhsa.py [out.pdf]
Numbers are constants below so the deck can be regenerated as results land.
"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

W, H = 13.333, 7.5
INK, MUTED, ACCENT = "#1a1a1a", "#5a5a5a", "#b03030"

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
            ax = fig.add_axes(blk.get("rect", [0.05, y - 0.32, 0.9, 0.3]))
            ax.axis("off")
            t = ax.table(cellText=blk["data"], loc="center", cellLoc="left")
            t.auto_set_font_size(False)
            t.set_fontsize(blk.get("fs", 10.5))
            t.scale(1, 1.45)
            for (r, c), cell in t.get_celld().items():
                cell.set_edgecolor("#dddddd")
                if r == 0:
                    cell.set_text_props(weight="bold")
            y -= blk.get("dy", 0.05) * (len(blk["data"]) + 1) + 0.03
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
              [{"text": "Intermediate report — 2026-09-17\n"
                        "marmaduke woodman · vbjax (feat/mhsa-paper-core)", "fs": 14},
               {"kind": "rule"},
               {"text": "Bottom line\n"
                        "  1. Two of the three problems were BUGS, not science: the trainer had no\n"
                        "     functional coupling at all, and the metric was scoring startup transients.\n"
                        "  2. The loop gain G is structurally unidentifiable under BPTT: it acts only as\n"
                        "     a scalar gain on the drive, and this readout+loss pipeline discards amplitude.\n"
                        "  3. Proposal: fit G and sigma on the FC metric by an outer loop; keep BPTT for\n"
                        "     the shape controls (W_q, W_y, kv_scale, tau).  Running now.", "fs": 13}],
              footer="All numbers on the following slides are reproducible from committed code; the certified metric rule is defined on slide 5.")

        slide(pdf, "The model and the fit",
              [{"text": "Per integration step (dt_train = 100 ms, window = 64 TR, BPTT with remat):\n"
                        "    aff_t  = G * (C @ y_{t-1})        C = structural connectome (spectral-radius 1)\n"
                        "    x_t    = aff_t + noise_t          OU/AR(1) noise, stationary std sigma\n"
                        "    M     += u (v k^T - M)            u = dt/tau;  memory (per region, per head)\n"
                        "    y      = W_Y M A phi(q);  y_coupled = y / (ema_rms(y) + 1e-8)\n"
                        "    e_t    = ||Delta M_t||_F  averaged over heads     ('surprise', >= 0)\n\n"
                        "Training loss   per-region z-scored MSE between TR-averaged e and empirical BOLD\n"
                        "Evaluation      Pearson corr between simulated and empirical FC matrices (fc_match)\n"
                        "                after a Windkessel balloon, scored on a settled window", "fs": 12.5},
               {"kind": "rule"},
               {"text": "Tunable parameters: W_q, W_y, kv_scale, log_tau, log_sigma, log_noise_tc, and G (the loop gain).",
                "fs": 12.5, "color": ACCENT}])

        slide(pdf, "Bug 1 — the model had no coupling",
              [{"text": "The reference simulator couples regions through the EMA-normalised output:\n"
                        "    wholebrain_fc.py:   y_out = y / (y_ema + 1e-8);   history.at[t % L].set(y_out)\n\n"
                        "The trainer computed the identical quantity and then discarded it:\n"
                        "    wb_train.py:        y_out = y / (y_ema + 1e-8)\n"
                        "                        return (state.M, noise, y_ema, y), e      # couples through raw y", "fs": 12.5},
               {"kind": "rule"},
               {"text": "With kv_scale collapsed to 0.0032, raw y has rms ~5.7e-09 -> the afferent drive is\n"
                        "~3.4e-09 against noise 0.145 (SNR ~2e-08).  The 'whole-brain network' was 100\n"
                        "independent noise-driven blocks, and fc1 = 0.209 came from the SHARED random\n"
                        "projection matrices, not from coupling.\n\n"
                        "Fix (--coupling ema + persistent carry): both attribution factors rose ~300x.\n"
                        "kv_scale then stopped collapsing: 0.0032 -> 0.40-0.65.", "fs": 12.5}])

        slide(pdf, "Bug 2 — the metric was measuring startup transients",
              [{"text": "Legacy fc_match integrated 600 s from the default balloon initial condition and\n"
                        "scored the WHOLE trace.  Transient contamination, first-10-TR std / resting std:\n"
                        "103x - 541x in 6 of 7 arms; only the baseline (1.5x) and one arm (5.4x) were clean.\n\n"
                        "Settled-window cap bound in 5/12 subject-runs at 1200 s and 8/12 at 2400 s ->\n"
                        "those windows were truncated, not certified settled.", "fs": 12.5},
               {"kind": "table", "data": LEGACY, "rect": [0.06, 0.40, 0.62, 0.22], "fs": 10.5},
               {"text": "Certified rule (used for all comparisons from here):\n"
                        "2400 s sim; 10-TR rolling std; ref = median over the final 50% of the trace;\n"
                        "W = first TR starting 50 consecutive TRs within 10% of ref; NO cap; FC on bold[W:].",
                "fs": 12.5, "color": ACCENT}])

        slide(pdf, "The ranking inverted under the certified metric",
              [{"kind": "table", "data": CERTIFIED, "rect": [0.06, 0.34, 0.88, 0.45], "fs": 11.5},
               {"text": "r3_fcy300 was our 'best quality' arm on legacy numbers (fc1 0.172) and collapses to\n"
                        "-0.002 once settled; r3_fc2's lead was a cap artefact (W = 54 TR at 1200 s).\n"
                        "The robust winner is the SIMPLEST recipe: the original per-region MSE loss with\n"
                        "ema coupling + persistent carry.  The raw-coupling baseline is last.", "fs": 12.5}])

        slide(pdf, "Instrument: exact attribution of dL/dG",
              [{"text": "G enters at exactly one place per step (aff_t = G * C @ y_{t-1}), so\n"
                        "    dL/dG = sum_t < dL/daff_t , daff_t/dG >\n"
                        "with aff injected as an explicit scan input (reverse demand exact per step) and\n"
                        "daff/dG obtained by jax.jvp (forward supply).  The identity reproduces dL/dG to\n"
                        "9 significant digits in every configuration -> the decomposition is exact.", "fs": 12.5},
               {"kind": "table", "data": ATTR, "rect": [0.06, 0.34, 0.88, 0.30], "fs": 10.5},
               {"text": "Both factors are healthy at the drive (0.044 and 0.076) — so it is neither that the\n"
                        "loss ignores the drive, nor that G fails to move it.", "fs": 12.5}])

        slide(pdf, "Where the identifiability is lost",
              [{"kind": "table", "data": STAGES, "rect": [0.06, 0.46, 0.52, 0.34], "fs": 11},
               {"text": "G controls aff perfectly, but x = aff + noise only 6.2e-04 — the drive is 4-5 orders\n"
                        "of magnitude below the noise floor, so the dilution happens at the sum, and\n"
                        "everything downstream inherits that factor.\n\n"
                        "Of what survives, 96.6% lies in the null space of the per-region z-score\n"
                        "(span{1, e_r} per region), and the residual cancels in time: the per-chunk\n"
                        "contributions alternate (+2.8e-6, +1.8e-6, -2.4e-6, -1.8e-6, +2.7e-7, +5.3e-6,\n"
                        "-1.0e-5, -5.8e-7) and sum to ~5e-6.  The recurrent path transmits ~nothing\n"
                        "(direct -5.1e-6 vs recursive +4.5e-8).", "fs": 12.5}])

        slide(pdf, "The mechanism: G is a pure gain, and amplitude is discarded",
              [{"text": "aff = G * (C @ y_prev): G scales the drive's MAGNITUDE but never its DIRECTION.\n"
                        "So G can only be seen through amplitude — and this pipeline throws amplitude away\n"
                        "three times over:", "fs": 12.5},
               {"kind": "rule"},
               {"text": "   (a) noise dilution      drive rms 7e-06 vs noise 0.145  (SNR ~5e-05)\n"
                        "   (b) scale-free loss     the z-scored MSE is invariant to per-region scaling\n"
                        "   (c) any normalisation   dividing the drive by its own rms removes G entirely", "fs": 12.5},
               {"kind": "rule"},
               {"text": "Prediction and test: normalising the drive (the fix designed to raise SNR) should\n"
                        "destroy G — and it did.  log-sensitivity of x fell from 6.2e-04 to 1.4e-07, and\n"
                        "dL/dG went from -5e-06 to +1e-13.  Removing the z-score ALSO made it smaller\n"
                        "(-1.8e-07), so the normaliser is not the culprit: the amplitude channel itself is.", "fs": 12.5}])

        slide(pdf, "Dead ends (recorded so they are not retried)",
              [{"text": "  balloon initial conditions   nuisance: they touch only the first ~30 s\n"
                        "                              (frozen-tail max |dBOLD| ~ 8e-06); per-region ICs\n"
                        "                              inflate fc2 from 0.118 to 0.623 with no model change\n"
                        "  FC-shaped losses (fcy/fce/fc2)  no robust gain under the certified metric\n"
                        "  pulse excitation            no effect on the certified score\n"
                        "  log-G reparameterisation     G moves, but that is not identifiability\n"
                        "  drive normalisation          falsified its own hypothesis (slide 8)\n"
                        "  normalisation asymmetry      real, but not causal (removing the z-score hurts)", "fs": 12.5},
               {"kind": "rule"},
               {"text": "Consistent with the reference pipeline: wholebrain_fc.py never learns G — it SWEEPS\n"
                        "G and selects by fc_match, a stationary, amplitude-sensitive statistic.", "fs": 12.5}])

        slide(pdf, "Two-stage test: stage 1 (G, sigma on FC, no gradients)",
              [{"text": "Shape parameters frozen from ema_yout; G in {0.5 ... 4.0} x sigma in {0.05 ... 0.30}.\n"
                        "Search scored on a settled window (600 s, 20% rule); top 5 re-scored with the\n"
                        "certified rule (2400 s, strict, no cap).", "fs": 12.5},
               {"kind": "table",
                "data": [["(G, sigma)", "lenient fc1 (search)", "certified fc1", "certified fc2"],
                         ["2.0 / 0.05", "0.298", "-0.046", "+0.101"],
                         ["4.0 / 0.15", "0.254", "-0.019", "+0.084"],
                         ["2.0 / 0.10", "0.227", "+0.012", "+0.026"],
                         ["2.5 / 0.10", "0.217", "-0.008", "+0.047"],
                         ["3.0 / 0.15", "0.206", "+0.021", "-0.041"]],
                "rect": [0.06, 0.40, 0.80, 0.26], "fs": 10.5},
               {"text": "Sensitivity (certified fc1, subj1) at sigma = 0.15 across G:\n"
                        "    0.5 +0.074 | 1.0 +0.085 | 1.5 +0.098 | 2.0 +0.074 | 2.5 +0.023 | 3.0 +0.021 | 4.0 -0.019\n"
                        "    sd 0.040, span 0.117  ->  fc_match is NOT flat in G (3-4x the seed scatter)\n"
                        "At G = 3.0 across sigma: sd 0.042, span 0.129  ->  sigma is observable too.\n\n"
                        "BUT the holdout subject peaks at G = 4, not at subj1's G = 1.5: the metric MOVES G,\n"
                        "it does not PIN it.  And the lenient search ranks the certified winner LAST.", "fs": 12}],
              footer="Stage 1 took ~1 h on one GPU; the sweep is cheap because it needs no gradients.")

        slide(pdf, "Two-stage test: stage 2 and verdict (NO)",
              [{"kind": "table",
                "data": [["arm", "final loss", "frozen G / sigma", "cert fc1", "cert fc2", "mean"],
                         ["BPTT, seed 1234", "1.998", "1.5 / 0.15", "+0.017", "+0.214", "+0.116"],
                         ["BPTT, seed 7", "2.003", "1.5 / 0.15", "-0.015", "+0.043", "+0.014"],
                         ["ema_yout (re-scored)", "-", "2.23 / 0.19", "+0.081", "+0.179", "+0.130"],
                         ["stage-1 only (no BPTT)", "-", "1.5 / 0.15", "+0.098", "+0.026", "+0.062"]],
                "rect": [0.06, 0.55, 0.86, 0.24], "fs": 10.5},
               {"text": "Verdict: the two-stage split does NOT beat ema_yout (best arm mean 0.116 vs 0.130),\n"
                        "and both BPTT arms collapse on subj1 (fc1 +0.017 / -0.015).  Freezing G is sound,\n"
                        "but choosing it on FC is not: the optimum flips between subjects, so the choice is\n"
                        "a coin flip on the holdout.  Stage-1-only (no BPTT) is worse still (mean 0.062).", "fs": 12.5},
               {"kind": "rule"},
               {"text": "What it DOES establish: fc_match is genuinely sensitive to G and to sigma, so the loop\n"
                        "gain is not a pure nuisance parameter -- it is under-determined by a single subject's\n"
                        "FC, not invisible.  That is the thread worth pulling next.", "fs": 12.5, "color": ACCENT}])

        slide(pdf, "The simulated FC is a global mode — and it is not the noise",
              [{"text": "Diagnostic over the cohort (job 4828980): the simulated FC stays almost rank-1 under EVERY\n"
                        "noise intervention tried, so the 'the noise is i.i.d. per region' explanation is dead.",
                "fs": 12.5},
               {"kind": "table",
                "data": [["condition (all at G = 4.0)", "eig_sim top-1", "eig_emp top-1", "within-max", "block_corr", "fc"],
                         ["sigma = 0.05  (anchor)", "97.9", "18.5", "0.997", "0.352", "0.102"],
                         ["sigma = 0.30", "96.3", "19.2", "0.986", "0.175", "0.100"],
                         ["sigma = 1.00", "95.5", "19.2", "0.985", "0.012", "0.019"],
                         ["baseline-std variant", "99.3", "19.2", "0.998", "0.432", "0.208"],
                         ["noise variant 2", "94.8", "19.2", "0.997", "0.349", "0.125"],
                         ["region-scaled empirical noise", "99.1", "19.2", "0.999", "0.455", "0.192"]],
                "rect": [0.06, 0.34, 0.88, 0.34], "fs": 10},
               {"text": "In every condition the readout e is 92-99% correlated ACROSS REGIONS, and cos(x, aff) ~ 0.85:\n"
                        "the readout carries almost no region-specific information. The readout common mode is\n"
                        "the strongest remaining candidate, but it is entangled with the coupling drive.", "fs": 12},
               {"kind": "rule"},
               {"text": "Result of the G test (extended sweep, job 4828987): the objective PEAKS near G = 4 with\n"
                        "small sigma and is flat/noisy elsewhere (top combos 0.15-0.17 at 1200 s, sd 0.08-0.11).\n"
                        "The simulated spectrum stays rank-1 at EVERY G tested, and block agreement is WORSE\n"
                        "at G = 16 (0.222 vs 0.357) — a broad optimum, not a point.\n"
                        "Also note the trap this exposed: the baseline-std and empirical-noise conditions RAISE\n"
                        "the scalar (fc 0.208, block_corr 0.46-0.58) while leaving the rank-1 structure untouched —\n"
                        "exactly the failure mode the low-rank diagnostics exist to catch.", "fs": 12,
                "color": ACCENT}])

        slide(pdf, "Status, caveats, next steps",
              [{"text": "Established: the machinery is correct and the metric is fair; the certified winner is\n"
                        "ema_yout (fc1 +0.081 / fc2 +0.179); G's unidentifiability has a measured mechanism.", "fs": 12.5},
               {"kind": "rule"},
               {"text": "Caveats\n"
                        "  - the fits are still weak: best certified number is ~0.18 on one subject;\n"
                        "  - some certified windows still hit the cap (noted per-arm) and are not certified settled;\n"
                        "  - the winning recipe is n=3 seeds with a wide spread (legacy fc1 0.078 - 0.301);\n"
                        "  - attribution used one window per config (three configs agreed).", "fs": 12.5},
               {"kind": "rule"},
               {"text": "Next\n"
                        "  1. look at the REGIONAL structure, not the scalar: fc_match is one number over\n"
                        "     4950 edges, and a 0.2 upper-triangle correlation can hide good structure;\n"
                        "  2. fit G/sigma with BOTH subjects in the objective (the flip says one subject\n"
                        "     under-determines G) -- or report G as a range, not a point;\n"
                        "  3. more seeds on the ema_yout recipe to bound its spread;\n"
                        "  4. retire the dead knobs; register the certified metric as the standard.",
                "fs": 12.5, "color": ACCENT}])
    print("wrote", out)


if __name__ == "__main__":
    main()