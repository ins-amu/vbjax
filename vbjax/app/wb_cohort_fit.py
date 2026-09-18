"""Cohort selection of G/sigma + low-rank structure diagnostics for the FC fit.

Replaces the single-subject scalar fit:
  * stage 1'  : sweep (G, sigma) maximising the MEAN fc_match over a cohort of
                FIT subjects; confirm the top combos on a held-out cohort split.
  * low-rank  : characterise the fit by structure, not one scalar —
                  - empirical vs simulated eigenspectra
                  - principal angles between the leading eigenspaces
                  - "low-rank mapping": reconstruct the empirical FC in the
                    model's own leading eigenbasis and report R^2 vs rank r
                    (does the model capture the leading modes, or only a
                    low-rank-corrupted version of them?)
                  - network-level agreement on data-driven blocks (k=17)

Runs on a GPU (the sweeps are many short simulations).
"""
import argparse, json, os, pickle
import numpy as onp
import jax
import jax.numpy as np

from vbjax.ct_mhsa import Hyperparameters
from vbjax.app.wholebrain_fc import simulate, bold_forward, fc_from_ts, fc_match
from vbjax.app.wb_train import zscore, build_mhsa
from vbjax.app.wb_eval_std import settled_window_strict


def load_cohort(d, v_c, dt, rho_norm=True):
    bold = onp.load(os.path.join(d, "bold_all.npy"))
    counts = onp.load(os.path.join(d, "counts_all.npy"))
    lengths = onp.load(os.path.join(d, "lengths_all.npy"))
    Cs, Ls = [], []
    for k in range(counts.shape[0]):
        W = counts[k] - counts[k].min()
        W = W / (onp.percentile(W, 99) + 1e-12)
        W = onp.clip(onp.sqrt(W), 0, 1)
        onp.fill_diagonal(W, 0.0)
        if rho_norm:
            W = W / float(onp.max(onp.abs(onp.linalg.eigvals(W))))
        lag = onp.floor(lengths[k] / v_c / dt).astype("i")
        Cs.append(W.astype("f4")); Ls.append(lag)
    return bold, Cs, Ls


def fc_of(bold):
    return fc_from_ts(bold)


def score_subject(shape, C, lags, hp0, G, sigma, ts_fc, dt, tr, T_s, key, hp_rule="strict"):
    L = int(lags.max()) + 1
    hp = hp0._replace(lam=dt / (shape["tau_s"] * 1000.0), tau=shape["tau_s"] * 1000.0)
    mhsa = build_mhsa(hp, key, shape["W_q"], shape["W_y"], shape["kv_scale"])
    T = int(T_s * 1000 / dt)
    e, _ = simulate(mhsa, np.asarray(C), jax.numpy.asarray(lags), L, hp, float(G),
                    float(sigma), T, key, shape["noise_tc"], dt, "e")
    e = onp.asarray(e)
    if not onp.isfinite(e).all():
        return None, None
    bold = onp.asarray(bold_forward(zscore(e), dt, tr))
    if not onp.isfinite(bold).all():
        return None, None
    if hp_rule == "strict":
        W, ref, W_unc, found = settled_window_strict(bold, win=10, tol=0.1, run=50,
                                                     cap_frac=0.0)
        if not found or W is None or bold.shape[0] - W < 10:
            return None, None
    else:  # lenient 20% rule with cap, for the coarse search
        from vbjax.app.wb_eval_std import settled_window
        W, rolling, ref, W_unc = settled_window(bold, win=10, tol=0.2, cap_frac=0.2)
        if W is None or bold.shape[0] - W < 10:
            return None, None
    return float(fc_match(fc_from_ts(bold[W:]), ts_fc)), bold[W:]


def eigs_angles(A, B, rs):
    """Eigenvalues and principal angles between leading eigenspaces."""
    wa, Va = onp.linalg.eigh(A); wb, Vb = onp.linalg.eigh(B)
    wa, Va = wa[::-1], Va[:, ::-1]; wb, Vb = wb[::-1], Vb[:, ::-1]
    out = {}
    for r in rs:
        s = onp.linalg.svd(Va[:, :r].T @ Vb[:, :r], compute_uv=False)
        s = onp.clip(s, -1, 1)
        out[int(r)] = float(onp.degrees(onp.mean(onp.arccos(s))))
    return wa, wb, out


def lowrank_r2(FC_emp, FC_sim, rs):
    """Reconstruct the empirical FC in the model's leading eigenbasis.

    For each r: keep the top-r eigenvectors of the SIMULATED FC, express the
    empirical FC in that basis, zero everything except the top-r block, and
    report R^2 of the reconstruction.  A high R^2 at small r means the model's
    leading modes span the empirical structure.
    """
    w, V = onp.linalg.eigh(FC_sim)
    V = V[:, ::-1]
    out = {}
    denom = float(onp.sum(FC_emp ** 2))
    for r in rs:
        Vr = V[:, :r]
        block = Vr.T @ FC_emp @ Vr
        rec = Vr @ block @ Vr.T
        out[int(r)] = float(1.0 - onp.sum((FC_emp - rec) ** 2) / (denom + 1e-30))
    return out


def kmeans(X, k, iters=100, seed=0):
    rs = onp.random.RandomState(seed)
    C = X[rs.choice(X.shape[0], k, replace=False)]
    for _ in range(iters):
        d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        lab = d.argmin(1)
        for j in range(k):
            m = lab == j
            if m.any():
                C[j] = X[m].mean(0)
    return lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, help="dir with bold_all/counts_all/lengths_all")
    ap.add_argument("--params", required=True, help="checkpoint for SHAPE params")
    ap.add_argument("--out", default="runs/wb_cohort")
    ap.add_argument("--dt", type=float, default=2.0)
    ap.add_argument("--tr", type=float, default=2.0)
    ap.add_argument("--v-c", type=float, default=5.0)
    ap.add_argument("--d-model", type=int, default=16)
    ap.add_argument("--k-fit", type=int, default=6)
    ap.add_argument("--k-hold", type=int, default=6)
    ap.add_argument("--search-s", type=float, default=400.0)
    ap.add_argument("--confirm-s", type=float, default=1200.0)
    ap.add_argument("--g-grid", default="0.5,1.0,1.5,2.5,4.0")
    ap.add_argument("--sigma-grid", default="0.05,0.15,0.30")
    ap.add_argument("--seed", type=int, default=42)
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
    d = args.d_model
    hp0 = Hyperparameters(n_regions=N, n_heads=N, d_model=d, d_k=d, d_v=d, lam=1.0,
                          dt=args.dt, tau=1e6, leak=1.0, act="relu",
                          steps_per_token=1, sigma_a=0.0)
    fit = list(range(args.k_fit))
    hold = list(range(args.k_fit, args.k_fit + args.k_hold))
    fcs = [fc_of(bold[k]) for k in range(K)]

    Gs = [float(x) for x in args.g_grid.split(",")]
    Ss = [float(x) for x in args.sigma_grid.split(",")]
    grid = []
    for G in Gs:
        for s in Ss:
            vals = []
            for k in fit:
                sc, _ = score_subject(shape, Cs[k], Ls[k], hp0, G, s, fcs[k],
                                      args.dt, args.tr, args.search_s,
                                      jax.random.PRNGKey(args.seed + k), "rule20")
                if sc is not None:
                    vals.append(sc)
            grid.append({"G": G, "sigma": s, "mean_fc1": float(onp.mean(vals)) if vals
                         else None, "sd": float(onp.std(vals)) if vals else None,
                         "n": len(vals)})
            print(f"[search] G={G:<4} sigma={s:<5} cohort fc1="
                  f"{grid[-1]['mean_fc1'] if vals else None} (n={len(vals)})", flush=True)

    ok = [r for r in grid if r["mean_fc1"] is not None]
    ok.sort(key=lambda r: -r["mean_fc1"])
    top = ok[:3]
    for r in top:
        hv, fv = [], []
        for k in hold:
            sc, _ = score_subject(shape, Cs[k], Ls[k], hp0, r["G"], r["sigma"],
                                  fcs[k], args.dt, args.tr, args.confirm_s,
                                  jax.random.PRNGKey(args.seed + k), "strict")
            if sc is not None:
                hv.append(sc)
        for k in fit:
            sc, _ = score_subject(shape, Cs[k], Ls[k], hp0, r["G"], r["sigma"],
                                  fcs[k], args.dt, args.tr, args.confirm_s,
                                  jax.random.PRNGKey(args.seed + k), "strict")
            if sc is not None:
                fv.append(sc)
        r["holdout_mean"] = float(onp.mean(hv)) if hv else None
        r["holdout_sd"] = float(onp.std(hv)) if hv else None
        r["fit_mean_confirmed"] = float(onp.mean(fv)) if fv else None
        print(f"[confirm] G={r['G']} sigma={r['sigma']} fit={r['fit_mean_confirmed']} "
              f"holdout={r['holdout_mean']}", flush=True)

    best = max(top, key=lambda r: (r["holdout_mean"] if r["holdout_mean"] is not None
                                   else -9))
    print(f"[cohort] BEST by holdout: G={best['G']} sigma={best['sigma']}", flush=True)

    # ---------------- low-rank structure on the holdout cohort
    FCe, FCs = [], []
    for k in hold:
        sc, b = score_subject(shape, Cs[k], Ls[k], hp0, best["G"], best["sigma"],
                              fcs[k], args.dt, args.tr, args.confirm_s,
                              jax.random.PRNGKey(args.seed + k), "strict")
        if b is None:
            continue
        FCs.append(fc_from_ts(b)); FCe.append(fcs[k])
    FCe, FCs = onp.mean(FCe, 0), onp.mean(FCs, 0)
    rs = [1, 2, 3, 5, 8, 12, 20]
    we, ws, angles = eigs_angles(FCe, FCs, rs)
    r2 = lowrank_r2(FCe, FCs, rs)
    lab = kmeans(FCe, 17, seed=0)
    def block(M):
        B = onp.zeros((17, 17))
        for i in range(17):
            for j in range(17):
                B[i, j] = M[lab == i][:, lab == j].mean()
        return B
    Be, Bs = block(FCe), block(FCs)
    out = {"best": best, "grid": grid, "top": top,
           "lowrank": {"eig_emp_top10": [float(x) for x in we[:10]],
                       "eig_sim_top10": [float(x) for x in ws[:10]],
                       "principal_angles_deg": angles,
                       "r2_vs_rank": r2,
                       "block_fc_corr": float(onp.corrcoef(Be.ravel(), Bs.ravel())[0, 1]),
                       "block_within_emp": [float(Be[i, i]) for i in range(17)],
                       "block_within_sim": [float(Bs[i, i]) for i in range(17)],
                       "eig_corr_top10": float(onp.corrcoef(we[:10], ws[:10])[0, 1])}}
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "cohort.json"), "w") as fh:
        json.dump(out, fh, indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(11, 9))
    ax[0, 0].imshow(FCe, cmap="RdBu_r", vmin=-0.5, vmax=0.5); ax[0, 0].set_title("empirical FC (cohort mean)")
    im = ax[0, 1].imshow(FCs, cmap="RdBu_r", vmin=-0.5, vmax=0.5); ax[0, 1].set_title(f"simulated FC (G={best['G']}, sigma={best['sigma']})")
    ax[1, 0].plot(range(1, 11), we[:10], "o-", label="empirical")
    ax[1, 0].plot(range(1, 11), ws[:10], "s-", label="simulated")
    ax[1, 0].set_xlabel("eigenvalue index"); ax[1, 0].set_ylabel("eigenvalue")
    ax[1, 0].set_title(f"eigenspectra (corr of top-10 = {out['lowrank']['eig_corr_top10']:.2f})")
    ax[1, 0].legend()
    ax[1, 1].plot(rs, [r2[r] for r in rs], "o-")
    ax[1, 1].set_xlabel("rank r"); ax[1, 1].set_ylabel("R^2 of empirical FC")
    ax[1, 1].set_title("empirical FC reconstructed in the model's rank-r eigenbasis")
    ax[1, 1].set_ylim(-0.2, 1.0); ax[1, 1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "cohort_lowrank.png"), dpi=110)
    fig2, ax2 = plt.subplots(figsize=(6, 5.5))
    ax2.scatter(Be.ravel(), Bs.ravel(), s=18)
    lim = [min(Be.min(), Bs.min()), max(Be.max(), Bs.max())]
    ax2.plot(lim, lim, "k--", lw=1)
    ax2.set_xlabel("empirical block FC (17 networks)"); ax2.set_ylabel("simulated block FC")
    ax2.set_title(f"network-block FC, r={out['lowrank']['block_fc_corr']:.2f}")
    fig2.tight_layout(); fig2.savefig(os.path.join(args.out, "cohort_blocks.png"), dpi=110)
    print(json.dumps(out["lowrank"], indent=1)[:1400], flush=True)
    print("WB_COHORT_FIT_DONE", flush=True)


if __name__ == "__main__":
    main()