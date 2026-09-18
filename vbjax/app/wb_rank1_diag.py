"""Why is the simulated FC nearly rank-1? (workboard card 6490d046)

Diagnosis only: no fitting here.  For each (G, sigma, noise-model) variant it
records

  * the certified fc_match (strict settled window, T_s = 1200 s, as in the cohort
    confirmation run),
  * the cohort-mean simulated FC, its eigenspectrum, the principal angles to the
    empirical leading eigenspaces and the within-block FC on the 17 data-driven
    blocks (exactly the parent run's low-rank diagnostics),
  * the cross-region correlation of the raw e (surprise) traces, and
  * the time-averaged cross-region cosine similarity of the three stages
    aff (coupling drive), x (block input), y (block output), accumulated inside
    the scan without storing any trace.

That last set localises the collapse: if cos_x is already ~1 the drive itself is
a scalar-gain common mode (coupling saturation); if cos_x is graded but the e
correlation is ~1, the readout collapses the structure.

Phases
  a  coupling saturation      : sweep G at sigma = 0.05, base noise
  b' global sigma             : sweep sigma at G = 4.0, base noise
  b  noise structure          : per-region sigma (and tc) scaled by the empirical
                                per-region BOLD std
  d  spatially structured noise: noise innovations mixed with the sqrt of the
                                fit-cohort FC (is i.i.d. noise the blocker?)

Run inside the GPU container from the tree root:
  python -m vbjax.app.wb_rank1_diag --cohort COH --params PARAMS.pkl --out runs/wb_rank1
"""
import argparse, json, os, pickle, time
import numpy as onp
import jax
import jax.numpy as np

from vbjax.ct_mhsa import Hyperparameters
from vbjax.app.wholebrain_fc import simulate, bold_forward, fc_from_ts, fc_match
from vbjax.app.wb_train import zscore, build_mhsa
from vbjax.app.wb_eval_std import settled_window_strict
from vbjax.app.wb_cohort_fit import load_cohort, fc_of, eigs_angles, lowrank_r2, kmeans

# (label, G, sigma, noise-mode, use the full 6-subject holdout set)
VARIANTS = [
    ("a_G0.5_s0.05", 0.5, 0.05, "base", False),
    ("a_G1.0_s0.05", 1.0, 0.05, "base", False),
    ("a_G2.0_s0.05", 2.0, 0.05, "base", False),
    ("a_G4.0_s0.05", 4.0, 0.05, "base", True),      # reproduces the parent baseline
    ("b_G4.0_s0.3", 4.0, 0.30, "base", False),
    ("b_G4.0_s1.0", 4.0, 1.00, "base", False),
    ("b_G4.0_s3.0", 4.0, 3.00, "base", False),
    ("c_bstd_G4.0", 4.0, 0.05, "bstd", False),
    ("c_bstd2_G4.0", 4.0, 0.05, "bstd2", False),
    ("c_bstdtc_G4.0", 4.0, 0.05, "bstd_tc", False),
    ("d_empnoise_G4.0_s0.05", 4.0, 0.05, "empnoise", False),
    ("d_empnoise_G4.0_s0.30", 4.0, 0.30, "empnoise", False),
]
BASE_SUBJ = [6, 7, 8, 9]                 # holdout subjects used for every variant
FULL_SUBJ = [6, 7, 8, 9, 10, 11]         # the parent run's holdout set
K_FIT = 6                                # fit subjects -> noise model / emp FC stats
RS = [1, 2, 3, 5, 8, 12, 20]


def sqrtm_psd(A):
    w, V = onp.linalg.eigh(A)
    w = onp.clip(w, 0, None)
    return (V * onp.sqrt(w)) @ V.T


def offdiag_stats(M):
    iu = onp.triu_indices(M.shape[0], k=1)
    w = onp.linalg.eigvalsh((M + M.T) / 2)[::-1]
    return {"mean_offdiag": float(M[iu].mean()),
            "top5_eig": [float(x) for x in w[:5]]}


def corr_matrix(trace):
    z = trace - trace.mean(0, keepdims=True)
    z = z / (z.std(0, keepdims=True) + 1e-12)
    return (z.T @ z) / trace.shape[0]


def spec_for(mode, sigma, tc0, r, Lmix):
    """(per-region sigma, per-region noise tc, noise mixing matrix)."""
    if mode == "base":
        return float(sigma), float(tc0), None
    if mode == "bstd":
        return (sigma * r), float(tc0), None
    if mode == "bstd2":
        return (sigma * r ** 2), float(tc0), None
    if mode == "bstd_tc":
        return (sigma * r), (tc0 / r), None
    if mode == "empnoise":
        return float(sigma), float(tc0), Lmix
    raise ValueError(mode)


def run_one(shape, hp0, C, lags, L, ts_fc, args, G, sigma_spec, tc_spec, mix, subj):
    hp = hp0._replace(lam=args.dt / (shape["tau_s"] * 1000.0),
                      tau=shape["tau_s"] * 1000.0)
    key = jax.random.PRNGKey(args.seed + subj)
    mhsa = build_mhsa(hp, key, shape["W_q"], shape["W_y"], shape["kv_scale"])
    T = int(args.T_s * 1000 / args.dt)
    t0 = time.time()
    e, _, grams = simulate(mhsa, np.asarray(C), jax.numpy.asarray(lags), L, hp,
                           float(G), sigma_spec, T, key, tc_spec, args.dt, "e",
                           collect_grams=True, want_y=False, noise_mix=mix)
    e = onp.asarray(e)
    if not onp.isfinite(e).all():
        return {"subj": subj, "status": "nonfinite_e", "secs": time.time() - t0}
    bold = onp.asarray(bold_forward(zscore(e), args.dt, args.tr))
    if not onp.isfinite(bold).all():
        return {"subj": subj, "status": "nonfinite_bold", "secs": time.time() - t0}
    W, ref, W_unc, found = settled_window_strict(bold, win=10, tol=0.1, run=50,
                                                cap_frac=0.0)
    every = int(round(args.tr * 1000.0 / args.dt))
    settled = bool(found and W is not None and bold.shape[0] - W >= 10)
    use = bold[W:] if settled else bold
    rec = {"subj": subj, "status": "ok", "secs": round(time.time() - t0, 1),
           "settled": settled, "W": (int(W) if W is not None else None),
           "n_tr": int(bold.shape[0]), "marker": "settled" if settled else "whole"}
    e_use = e[W * every:] if settled else e
    rec["fc_match"] = float(fc_match(fc_from_ts(use), ts_fc))
    rec["e_corr"] = offdiag_stats(corr_matrix(e_use))
    names = ("aff", "x", "y")
    for gi, nm in enumerate(names):
        rec[f"cos_{nm}"] = float(grams[gi][onp.triu_indices(grams.shape[1], 1)].mean())
    rec["_fc_sim"] = fc_from_ts(use)
    rec["_fc_emp"] = ts_fc
    return rec


def summarise(recs):
    ok = [r for r in recs if r.get("_fc_sim") is not None]
    if not ok:
        return {"n_ok": 0}
    FCs = onp.mean([r["_fc_sim"] for r in ok], 0)
    FCe = onp.mean([r["_fc_emp"] for r in ok], 0)
    we, ws, angles = eigs_angles(FCe, FCs, RS)
    r2 = lowrank_r2(FCe, FCs, RS)
    lab = kmeans(FCe, 17, seed=0)

    def block(M):
        B = onp.zeros((17, 17))
        for i in range(17):
            for j in range(17):
                B[i, j] = M[lab == i][:, lab == j].mean()
        return B
    Be, Bs = block(FCe), block(FCs)
    fm = [r["fc_match"] for r in ok]
    out = {"n_ok": len(ok), "n_settled": sum(1 for r in ok if r["settled"]),
           "fc_match_mean": float(onp.mean(fm)),
           "fc_match_sd": float(onp.std(fm)),
           "fc_offdiag": float(offdiag_stats(FCs)["mean_offdiag"]),
           "e_offdiag": float(onp.mean([r["e_corr"]["mean_offdiag"] for r in ok])),
           "e_top5": [float(x) for x in onp.mean(
               [onp.asarray(r["e_corr"]["top5_eig"]) for r in ok], 0)],
           "cos_aff": float(onp.mean([r["cos_aff"] for r in ok])),
           "cos_x": float(onp.mean([r["cos_x"] for r in ok])),
           "cos_y": float(onp.mean([r["cos_y"] for r in ok])),
           "eig_sim_top10": [float(x) for x in ws[:10]],
           "eig_emp_top10": [float(x) for x in we[:10]],
           "principal_angles_deg": angles,
           "r2_vs_rank": r2,
           "block_within_sim": [float(Bs[i, i]) for i in range(17)],
           "block_within_sim_max": float(max(Bs[i, i] for i in range(17))),
           "block_within_emp": [float(Be[i, i]) for i in range(17)],
           "block_fc_corr": float(onp.corrcoef(Be.ravel(), Bs.ravel())[0, 1])}
    out["_FCs"] = FCs
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--params", required=True)
    ap.add_argument("--out", default="runs/wb_rank1")
    ap.add_argument("--dt", type=float, default=2.0)
    ap.add_argument("--tr", type=float, default=2.0)
    ap.add_argument("--v-c", type=float, default=5.0)
    ap.add_argument("--d-model", type=int, default=16)
    ap.add_argument("--T-s", type=float, default=1200.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--only", default=None, help="comma-separated variant labels")
    args = ap.parse_args()

    bold, Cs, Ls = load_cohort(args.cohort, args.v_c, args.dt)
    K, T, N = bold.shape
    print(f"[cohort] K={K} T={T} N={N}", flush=True)
    with open(args.params, "rb") as f:
        p = pickle.load(f)
    shape = {"W_q": np.asarray(p["W_q"]), "W_y": np.asarray(p["W_y"]),
             "kv_scale": np.asarray(p["kv_scale"]),
             "tau_s": float(onp.exp(onp.asarray(p["log_tau"]))),
             "noise_tc": float(onp.exp(onp.asarray(p["log_noise_tc"])))}
    print(f"[params] tau_s={shape['tau_s']:.2f} noise_tc={shape['noise_tc']:.2f} "
          f"kv_scale={float(shape['kv_scale']):.4f} "
          f"sigma_ckpt={float(onp.exp(onp.asarray(p['log_sigma']))):.4f} "
          f"G_ckpt={(float(onp.exp(onp.asarray(p['log_G']))) if 'log_G' in p else float(onp.asarray(p['G_rel']))):.4f} "
          f"G_key={'log_G' if 'log_G' in p else 'G_rel'} "
          f"keys={sorted(p.keys())}", flush=True)

    fcs = [fc_of(bold[k]) for k in range(K)]
    bstd = bold[:K_FIT].std(axis=1).mean(0)          # (N,) fit-cohort per-region std
    r = bstd / bstd.mean()
    print(f"[bstd] per-region empirical BOLD std: min={bstd.min():.3f} "
          f"mean={bstd.mean():.3f} max={bstd.max():.3f} "
          f"ratio r in [{r.min():.2f}, {r.max():.2f}]", flush=True)
    FCfit = onp.mean([fcs[k] for k in range(K_FIT)], 0)
    Lmix = sqrtm_psd(FCfit)
    print(f"[noise] fit-cohort FC sqrt: top5 eig "
          f"{[round(float(x),2) for x in onp.linalg.eigvalsh(FCfit)[::-1][:5]]}",
          flush=True)
    d = args.d_model
    hp0 = Hyperparameters(n_regions=N, n_heads=N, d_model=d, d_k=d, d_v=d, lam=1.0,
                          dt=args.dt, tau=1e6, leak=1.0, act="relu",
                          steps_per_token=1, sigma_a=0.0)

    os.makedirs(args.out, exist_ok=True)
    wanted = set(args.only.split(",")) if args.only else None
    summary = {}
    for label, G, sig, mode, full in VARIANTS:
        if wanted is not None and label not in wanted:
            continue
        subjects = FULL_SUBJ if full else BASE_SUBJ
        sig_spec, tc_spec, mix = spec_for(mode, sig, shape["noise_tc"], r, Lmix)
        recs = []
        for subj in subjects:
            rec = run_one(shape, hp0, Cs[subj], Ls[subj],
                          int(Ls[subj].max()) + 1, fcs[subj], args, G, sig_spec,
                          tc_spec, mix, subj)
            recs.append(rec)
            print(f"[{label}] s{subj} status={rec['status']} "
                  f"settled={rec.get('settled')} W={rec.get('W')} "
                  f"fc={rec.get('fc_match')} e_off={rec.get('e_corr', {}).get('mean_offdiag')} "
                  f"cos(x)={rec.get('cos_x')} ({rec['secs']}s)", flush=True)
            with open(os.path.join(args.out, f"rec_{label}_s{subj}.json"), "w") as fh:
                json.dump({k: v for k, v in rec.items() if not k.startswith("_")}, fh)
        s = summarise(recs)
        base = summarise([x for x in recs if x["subj"] in BASE_SUBJ])
        s["base4"] = {k: v for k, v in base.items() if not k.startswith("_")}
        if full:
            s["base4"]["_note"] = "same variant restricted to subjects 6-9"
        summary[label] = s
        FCs = s.pop("_FCs", None)
        if label == "a_G4.0_s0.05" and FCs is not None:
            onp.save(os.path.join(args.out, "FC_sim_REF_G4_s0.05.npy"), FCs)
        print(f"[{label}] SUMMARY n={s['n_ok']}/{len(recs)} settled={s['n_settled']} "
              f"fc={s['fc_match_mean']:.4f}+-{s['fc_match_sd']:.4f} "
              f"fc_off={s['fc_offdiag']:.4f} e_off={s['e_offdiag']:.4f} "
              f"cos(x/aff/y)={s['cos_x']:.4f}/{s['cos_aff']:.4f}/{s['cos_y']:.4f}",
              flush=True)
        print(f"[{label}] eig_sim_top5="
              f"{[round(x, 3) for x in s['eig_sim_top10'][:5]]} "
              f"eig_emp_top5={[round(x, 3) for x in s['eig_emp_top10'][:5]]} "
              f"block_within_sim_max={s['block_within_sim_max']:.3f} "
              f"block_corr={s['block_fc_corr']:.3f} "
              f"angles={ {k: round(v, 1) for k, v in s['principal_angles_deg'].items()} }",
              flush=True)
        with open(os.path.join(args.out, "rank1_summary.json"), "w") as fh:
            json.dump(summary, fh, indent=1)

    print("WB_RANK1_DIAG_DONE", flush=True)


if __name__ == "__main__":
    main()
