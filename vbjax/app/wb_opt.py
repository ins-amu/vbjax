"""Learn the whole-brain CT-MHSA hyperparameters from the BOLD data.

Optimizes (G_rel, tau_s, sigma, noise_tc_s, kv_scale) to maximize the FC match
on subject 1 (Powell, gradient-free — each eval is a full forward simulation),
then scores the learned hyperparameters on held-out subject 2.

  python -m vbjax.app.wb_opt --sc SC.npy --lengths LEN.npy \
      --ts1 subj1.npy --ts2 subj2.npy --out runs/wb_opt
"""
import argparse, os, json, time
import numpy as onp
import jax
import jax.numpy as np
from scipy.optimize import minimize

from vbjax.ct_mhsa import Hyperparameters, init_ct_mhsa
from vbjax.app.wholebrain_fc import (
    load_connectome, simulate, bold_forward, fc_from_ts, fc_match,
)

BOUNDS = [(0.3, 5.0), (5.0, 100.0), (0.02, 0.3), (0.5, 10.0), (0.05, 0.5)]
NAMES = ["G_rel", "tau_s", "sigma", "noise_tc_s", "kv_scale"]


def clip_params(x):
    return [float(min(max(v, lo), hi)) for v, (lo, hi) in zip(x, BOUNDS)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sc", required=True)
    p.add_argument("--lengths", required=True)
    p.add_argument("--ts1", required=True, help="subject-1 BOLD TS (optimization target)")
    p.add_argument("--ts2", required=True, help="subject-2 BOLD TS (held-out)")
    p.add_argument("--out", default="runs/wb_opt")
    p.add_argument("--dt", type=float, default=2.0)
    p.add_argument("--tr", type=float, default=2.0)
    p.add_argument("--v-c", type=float, default=5.0)
    p.add_argument("--d-model", type=int, default=16)
    p.add_argument("--sim-s", type=float, default=300.0, help="sim seconds during optimization")
    p.add_argument("--final-sim-s", type=float, default=1200.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--maxfev", type=int, default=60)
    a = p.parse_args()

    C, lags, L = load_connectome(a.sc, a.lengths, a.dt, a.v_c, "sqrt99")
    N = C.shape[0]
    rho = float(onp.max(onp.abs(onp.linalg.eigvals(C))))
    C = C / rho
    ts1 = onp.load(a.ts1); ts2 = onp.load(a.ts2)
    fc1 = fc_from_ts(ts1); fc2 = fc_from_ts(ts2)
    iu = onp.triu_indices(N, 1)
    T_opt = int(a.sim_s * 1000 / a.dt)
    T_fin = int(a.final_sim_s * 1000 / a.dt)

    os.makedirs(a.out, exist_ok=True)
    progress = []
    neval = [0]

    def eval_params(x, T, ts_fc):
        G_rel, tau, sigma, noise_tc, kv_scale = clip_params(x)
        hp = Hyperparameters(n_regions=N, n_heads=N, d_model=a.d_model,
                             d_k=a.d_model, d_v=a.d_model,
                             lam=a.dt / (tau * 1000.0), dt=a.dt, tau=tau * 1000.0,
                             leak=1.0, act="relu", steps_per_token=1, sigma_a=0.0)
        k_m, k_sim = jax.random.split(jax.random.PRNGKey(a.seed))
        mhsa, _, _ = init_ct_mhsa(hp, k_m, batch_size=1)
        mhsa = mhsa._replace(W_K=mhsa.W_K * kv_scale, W_V=mhsa.W_V * kv_scale)
        e, y = simulate(mhsa, np.asarray(C), jax.numpy.asarray(lags), L, hp,
                        G_rel, sigma, T, k_sim, noise_tc, a.dt, "e")
        e = onp.asarray(e)
        if not onp.isfinite(e).all():
            return None
        src = (e - e.mean(0)) / (e.std(0) + 1e-8)
        bold = bold_forward(src, a.dt, a.tr)
        fc_sim = fc_from_ts(bold)
        return float(onp.corrcoef(fc_sim[iu], ts_fc[iu])[0, 1])

    def objective(x):
        t0 = time.time()
        s = eval_params(x, T_opt, fc1)
        neval[0] += 1
        if s is None:
            print(f"eval {neval[0]:3d} x={clip_params(x)} DIVERGED", flush=True)
            progress.append({"eval": neval[0], "x": clip_params(x), "fc1": None})
            return 1.0   # penalize divergence
        print(f"eval {neval[0]:3d} x={clip_params(x)} fc1={s:.4f} ({time.time()-t0:.0f}s)", flush=True)
        progress.append({"eval": neval[0], "x": clip_params(x), "fc1": s})
        with open(os.path.join(a.out, "progress.json"), "w") as f:
            json.dump(progress, f, indent=1)
        return -s

    x0 = [2.5, 10.0, 0.1, 2.0, 0.15]
    print(f"optimizing {NAMES} over {a.maxfev} evals (sim {a.sim_s}s each), start={x0}", flush=True)
    res = minimize(objective, x0, method="Powell", bounds=BOUNDS,
                   options={"maxfev": a.maxfev, "xtol": 1e-3, "ftol": 1e-4})

    best_x = clip_params(res.x)
    fc1_fin = eval_params(best_x, T_fin, fc1)
    fc2_fin = eval_params(best_x, T_fin, fc2)
    print(f"DONE best={dict(zip(NAMES, best_x))} fc1={fc1_fin} fc2(holdout)={fc2_fin}", flush=True)
    with open(os.path.join(a.out, "results.json"), "w") as f:
        json.dump({"best_x": dict(zip(NAMES, best_x)), "fc1_final": fc1_fin,
                   "fc2_holdout": fc2_fin, "n_evals": neval[0],
                   "res_message": str(res.message)}, f, indent=1)


if __name__ == "__main__":
    main()
