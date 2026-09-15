"""Hopf whole-brain baseline control: what fc_match does the STANDARD model achieve
on the same 1000Brains subject-1 data? (Deco-style Hopf oscillator per region,
delayed SC coupling, vbjax balloon BOLD.) Calibrates the CT-MHSA numbers.

  python -m vbjax.app.wb_hopf_control --sc SC.npy --lengths LEN.npy --emp-ts TS.npy \
      --out runs/wb_fc/hopf --g-rel-sweep 0.3,0.7,1.0 --noise-tc 2.0
"""
import argparse, os, json, itertools
import numpy as onp
import jax
import jax.numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vbjax.app.wholebrain_fc import bold_forward


def hopf_sim(C, lags, L, G, sigma, T, dt_ms, key, noise_tc_s=2.0, a_param=1.0):
    """Deco Hopf: dz_i dot = (a - |z_i|^2) z_i + i w z_i + G SC z_delayed + noise.
    Super critical (a>0) limit-cycle oscillator per region; coupling on x."""
    N = C.shape[0]
    ar = np.arange(N)
    dt_s = dt_ms / 1000.0
    w = 2.0 * np.pi * 0.04          # ~0.04 Hz (well within BOLD band)

    def step(carry, key_t):
        (x, yv), hist, n_prev = carry
        t = hist_t[0]  # replaced below; use explicit counter via lags trick
        raise NotImplementedError

    # simpler: explicit counter in carry
    def step2(carry, key_t):
        (x, yv), hist, n_prev, t = carry
        idx = (t - lags) % L
        x_hist = hist[idx, ar]                     # (N, N) delayed x_j
        aff = G * np.einsum("ij,ij->i", C, x_hist)
        na = float(onp.exp(-dt_s / noise_tc_s))
        eps = jax.random.normal(key_t, (2 * N,))
        noise = na * n_prev + sigma * np.sqrt(1 - na**2) * eps
        nx, ny = noise[:N], noise[N:]
        dx = (a_param - x * x - yv * yv) * x + w * yv + aff + nx
        dy = (a_param - x * x - yv * yv) * yv - w * x + ny
        x2, y2 = x + dt_s * dx, yv + dt_s * dy
        hist = hist.at[t % L].set(x2)
        return ((x2, y2), hist, noise, t + 1), (x2, y2)

    x0 = 0.5 * onp.ones(N, "f4"); y0 = onp.zeros(N, "f4")
    hist0 = onp.zeros((L, N), "f4"); n0 = onp.zeros(2 * N, "f4")
    keys = jax.random.split(jax.random.fold_in(key, 7), T)
    (fx, _, _, _), (xs, ys) = jax.lax.scan(step2, ((x0, y0), hist0, n0, 0), keys)
    return xs, ys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sc", required=True)
    p.add_argument("--lengths", required=True)
    p.add_argument("--emp-ts", required=True)
    p.add_argument("--out", default="runs/wb_fc/hopf")
    p.add_argument("--dt", type=float, default=2.0)
    p.add_argument("--tr", type=float, default=2.0)
    p.add_argument("--v-c", type=float, default=5.0)
    p.add_argument("--sim-s", type=float, default=600.0)
    p.add_argument("--g-rel-sweep", default="0.3,0.7,1.0,1.2")
    p.add_argument("--noise-tc", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    W = onp.load(a.sc) if a.sc.endswith(".npy") else onp.loadtxt(a.sc)
    Ll = onp.load(a.lengths) if a.lengths.endswith(".npy") else onp.loadtxt(a.lengths)
    N = W.shape[0]
    W = W - W.min(); W = W / onp.percentile(W, 99); W = onp.clip(onp.sqrt(W), 0, 1)
    onp.fill_diagonal(W, 0.0)
    rho = float(onp.max(onp.abs(onp.linalg.eigvals(W))))
    W = W / rho                                   # unit spectral radius
    lags = onp.floor(Ll / a.v_c / a.dt).astype("i")
    L = int(lags.max()) + 1
    emp_ts = onp.load(a.emp_ts) if a.emp_ts.endswith(".npy") else onp.loadtxt(a.emp_ts)
    z = (emp_ts - emp_ts.mean(0)) / (emp_ts.std(0) + 1e-8)
    fc_emp = (z.T @ z) / emp_ts.shape[0]
    iu = onp.triu_indices(N, 1)

    T = int(a.sim_s * 1000 / a.dt)
    key = jax.random.PRNGKey(a.seed)
    os.makedirs(a.out, exist_ok=True)
    results = []
    best = None
    for G_rel in [float(x) for x in a.g_rel_sweep.split(",")]:
        xs, ys = hopf_sim(np.asarray(W), jax.numpy.asarray(lags), L, G_rel, 0.3,
                          T, a.dt, key, a.noise_tc)
        xs = onp.asarray(xs); ys = onp.asarray(ys)
        if not onp.isfinite(xs).all() or not onp.isfinite(ys).all():
            print(f"G_rel={G_rel} DIVERGED"); results.append({"G_rel": G_rel, "fc_match": None, "diverged": True}); continue
        # Feed the POSITIVE amplitude r = sqrt(x^2 + y^2) to the balloon:
        # the raw x oscillates around 0 and net-negative neural input drives
        # the balloon volume V negative -> V**alpha NaN (round-3 lesson).
        r = onp.sqrt(xs * xs + ys * ys)
        # Window the input per TR and sample the balloon once per window, exactly
        # like wholebrain_fc.bold_forward. make_offline's inner scan consumes a
        # WINDOW of per-step inputs; feeding it single (N,) steps made that scan
        # iterate over the N regions and broadcast each scalar region value to
        # ALL regions -> identical BOLD traces -> all-ones FC -> corrcoef NaN.
        bold = bold_forward(r, a.dt, a.tr)
        if not onp.isfinite(bold).all():
            print(f"G_rel={G_rel} BOLD_NAN"); results.append({"G_rel": G_rel, "fc_match": None, "bold_nan": True}); continue
        zs = (bold - bold.mean(0)) / (bold.std(0) + 1e-8)
        fc_sim = (zs.T @ zs) / bold.shape[0]
        fc_vec = fc_sim[iu]
        if not onp.isfinite(fc_vec).all() or float(fc_vec.std()) == 0.0:
            print(f"G_rel={G_rel} FC_DEGENERATE (constant/non-finite fc_sim)")
            results.append({"G_rel": G_rel, "fc_match": None, "fc_degenerate": True})
            continue
        score = float(onp.corrcoef(fc_sim[iu], fc_emp[iu])[0, 1])
        results.append({"G_rel": G_rel, "fc_match": score})
        print(f"G_rel={G_rel} fc_match={score:.4f}")
        if best is None or (isinstance(score, float) and score > (best.get("fc_match") or -9)):
            best = dict(results[-1])
            onp.save(os.path.join(a.out, "fc_sim_best.npy"), fc_sim)

    best = best or max((r for r in results if r.get("fc_match") is not None),
                       key=lambda r: r["fc_match"], default=None)
    with open(os.path.join(a.out, "results.json"), "w") as f:
        json.dump({"results": results, "best": best, "N": N, "T": T}, f, indent=2)
    print("best:", best)


if __name__ == "__main__":
    main()
