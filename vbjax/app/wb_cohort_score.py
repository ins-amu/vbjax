"""Certified + low-rank holdout score for a cohort-BPTT shape checkpoint.

The cohort training card (4ec1331b) freezes the cohort-fitted (G, sigma) and
BPTTs only the shape controls. Its acceptance requires BOTH a certified
fc_match on HELD-OUT cohort subjects AND the low-rank structural metrics
(graded eigenspectrum, within-block FC, principal angles), reported as
per-subject spread, not just the mean.

wb_cohort_fit.py has the certified 2400 s strict rule (score_subject) and the
structure metrics (eigs_angles, lowrank_r2, kmeans block FC), but its main()
crashes at k_fit=0 and its structure path only reports a held-out COHORT MEAN.
This driver calls the same helpers directly so a single trained checkpoint is
scored per-subject on the held-out split, and reports the certified per-subject
spread that the card explicitly demands.

  python -m vbjax.app.wb_cohort_score --cohort ... --params <shape.pkl> \
      --g 4.0 --sigma 0.005 --k-hold 6 --out runs/wb_cohort_bptt_X/holdout
"""
import argparse, json, os, pickle
import numpy as onp
import jax
import jax.numpy as np

from vbjax.ct_mhsa import Hyperparameters
from vbjax.app.wholebrain_fc import fc_from_ts
from vbjax.app.wb_cohort_fit import (load_cohort, score_subject, eigs_angles,
                                     lowrank_r2, kmeans)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cohort", required=True)
    p.add_argument("--params", required=True, help="trained SHAPE checkpoint")
    p.add_argument("--out", default="runs/wb_cohort_score")
    p.add_argument("--g", type=float, required=True)
    p.add_argument("--sigma", type=float, required=True)
    p.add_argument("--dt", type=float, default=2.0)
    p.add_argument("--tr", type=float, default=2.0)
    p.add_argument("--v-c", type=float, default=5.0)
    p.add_argument("--d-model", type=int, default=16)
    p.add_argument("--k-fit", type=int, default=0,
                   help="# leading cohort subjects treated as FIT (not scored)")
    p.add_argument("--k-hold", type=int, default=6,
                   help="# HELD-OUT subjects to certify + structure-score")
    p.add_argument("--confirm-s", type=float, default=2400.0)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    bold, Cs, Ls = load_cohort(a.cohort, a.v_c, a.dt, rho_norm=True)
    K, T, N = bold.shape
    with open(a.params, "rb") as f:
        pk = pickle.load(f)
    shape = {"W_q": np.asarray(pk["W_q"]), "W_y": np.asarray(pk["W_y"]),
             "kv_scale": np.asarray(pk["kv_scale"]),
             "tau_s": float(onp.exp(onp.asarray(pk["log_tau"]))),
             "noise_tc": float(onp.exp(onp.asarray(pk["log_noise_tc"])))}
    d = a.d_model
    hp0 = Hyperparameters(n_regions=N, n_heads=N, d_model=d, d_k=d, d_v=d,
                          lam=1.0, dt=a.dt, tau=1e6, leak=1.0, act="relu",
                          steps_per_token=1, sigma_a=0.0)
    hold = list(range(a.k_fit, a.k_fit + a.k_hold))
    fcs = [fc_from_ts(bold[k]) for k in range(K)]

    print(f"[score] K={K} N={N} hold={a.k_fit}..{a.k_fit+a.k_hold-1} "
          f"G={a.g} sigma={a.sigma} confirm_s={a.confirm_s}", flush=True)

    # ---- per-subject certified fc_match (2400 s strict, no cap)
    per_subj = []
    vals = []
    for k in hold:
        sc, b = score_subject(shape, Cs[k], Ls[k], hp0, a.g, a.sigma, fcs[k],
                              a.dt, a.tr, a.confirm_s,
                              jax.random.PRNGKey(a.seed + k), "strict")
        rec = {"subject": int(k),
               "certified_fc_match": (float(sc) if sc is not None else None),
               "settled_tr": None}
        if b is not None:
            rec["settled_tr"] = int(b.shape[0])
        if sc is not None:
            vals.append(float(sc))
            print(f"[holdout] subj {k}: certified fc_match={sc:.4f} "
                  f"n_tr={b.shape[0]}", flush=True)
        else:
            print(f"[holdout] subj {k}: unsettled/non-finite", flush=True)
        per_subj.append(rec)

    cert = {"per_subject_certified": per_subj,
            "n_subjects": len(hold),
            "certified_mean": (float(onp.mean(vals)) if vals else None),
            "certified_sd": (float(onp.std(vals)) if vals else None),
            "certified_list": vals,
            "certified_range": ([min(vals), max(vals)] if vals else None)}
    if vals:
        print(f"[holdout] CERTIFIED mean={cert['certified_mean']:.4f} "
              f"sd={cert['certified_sd']:.4f} n={len(vals)}", flush=True)

    # ---- low-rank structure on the held-out COHORT MEAN (mirrors wb_cohort_fit)
    FCe, FCs = [], []
    for k in hold:
        sc, b = score_subject(shape, Cs[k], Ls[k], hp0, a.g, a.sigma, fcs[k],
                              a.dt, a.tr, a.confirm_s,
                              jax.random.PRNGKey(a.seed + k), "strict")
        if b is None:
            continue
        FCs.append(fc_from_ts(b)); FCe.append(fcs[k])
    if FCe:
        FCe = onp.mean(FCe, 0); FCs = onp.mean(FCs, 0)
        rs = [1, 2, 3, 5, 8, 12, 20]
        we, ws, angles = eigs_angles(FCe, FCs, rs)
        r2 = lowrank_r2(FCe, FCs, rs)
        lab = kmeans(FCe, 17, seed=0)

        def block(M):
            B = onp.zeros((17, 17))
            for i in range(17):
                for jj in range(17):
                    B[i, jj] = M[lab == i][:, lab == jj].mean()
            return B
        Be, Bs = block(FCe), block(FCs)
        structure = {
            "eig_emp_top10": [float(x) for x in we[:10]],
            "eig_sim_top10": [float(x) for x in ws[:10]],
            "eig_corr_top10": float(onp.corrcoef(we[:10], ws[:10])[0, 1]),
            "principal_angles_deg": angles,
            "r2_vs_rank": r2,
            "block_fc_corr": float(onp.corrcoef(Be.ravel(), Bs.ravel())[0, 1]),
            "block_within_emp": [float(Be[i, i]) for i in range(17)],
            "block_within_sim": [float(Bs[i, i]) for i in range(17)],
        }
        s1 = ws[0]; s2 = ws[1] if len(ws) > 1 else 0.0
        structure["sim_eig_ratio_1_to_2"] = (float(s2 / s1) if s1 else None)
        structure["sim_eig_1st_frac"] = (float(s1 / max(sum(ws[:10]), 1e-30)))
        cert["structure"] = structure
        print(f"[holdout] eig_sim_top5={[round(x, 3) for x in ws[:5]]} "
              f"eig_corr={structure['eig_corr_top10']:.3f} "
              f"block_corr={structure['block_fc_corr']:.3f}", flush=True)
    else:
        cert["structure"] = None

    cert["shape_frozen_G"] = a.g
    cert["shape_frozen_sigma"] = a.sigma
    cert["settings"] = {"dt": a.dt, "tr": a.tr, "v_c": a.v_c,
                        "d_model": d, "k_hold": a.k_hold,
                        "confirm_s": a.confirm_s, "seed": a.seed}
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "holdout_certified.json"), "w") as f:
        json.dump(cert, f, indent=1)
    print(json.dumps({k: cert.get(k) for k in
                      ("certified_mean", "certified_sd", "certified_range",
                       "certified_list", "structure")}, indent=1)[:1800],
          flush=True)
    print("WB_COHORT_SCORE_DONE", flush=True)


if __name__ == "__main__":
    main()
