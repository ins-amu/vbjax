"""Whole-brain CT-MHSA fit to resting-state BOLD: prediction error as the fMRI forward model.

Dynamics (N cortical areas, one attention head per area, connectome coupling with delays):
    aff_i(t) = G * sum_j C_ij * y_j(t - lag_ij)          (delayed afferent prediction)
    x_i(t)   = aff_i(t) + sigma * noise                  (thalamic drive + fluctuations)
    y_i(t)   = mhsa block output for area i              (recurrent key-value memory)
    e_i(t)   = ||Delta M_i||_F                           (LOCAL PREDICTION ERROR / surprise:
                                                          how much the new key-value pair
                                                          deviates from what memory predicts)
    BOLD_i   = balloon(e_i) sampled at TR                (vbjax make_bold)

Fit: sweep coupling gain G, memory time constant tau, noise sigma; metric =
corr(upper-tri(FC_sim), upper-tri(FC_emp)) where FC = Pearson correlations of BOLD traces.

Run (inside the GPU container, from the tree root):
  python -m vbjax.app.wholebrain_fc --sc SC.npy --lengths LEN.npy --emp-ts TS.npy \
      --out runs/wb_fc --g-sweep 0.5,1.0,2.0 --tau-sweep 20,40
"""
from typing import NamedTuple, Tuple
import argparse, os, pickle, json, itertools
import numpy as onp
import jax
import jax.numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vbjax import make_bold, bold_default_theta, make_offline
from vbjax.ct_mhsa import (
    Hyperparameters, NetworkState, init_ct_mhsa, mhsa_step,
)


def load_connectome(sc_path: str, lengths_path: str, dt: float, v_c: float):
    """Structural weights + tract lengths -> normalized C (sqrt, 99th-pct, aaicd.py style)
    and integer per-pair delays in integration steps."""
    W = onp.load(sc_path) if sc_path.endswith(".npy") else onp.loadtxt(sc_path)
    L = onp.load(lengths_path) if lengths_path.endswith(".npy") else onp.loadtxt(lengths_path)
    nreg = W.shape[0]
    assert W.shape == L.shape == (nreg, nreg)
    W = W - W.min()
    W = W / onp.percentile(W, 99)
    W = onp.clip(onp.sqrt(W), 0, 1)
    onp.fill_diagonal(W, 0.0)
    lags = onp.floor(L / v_c / dt).astype("i")
    Lmax = int(lags.max())
    return W.astype("f4"), lags, Lmax + 1


def make_sim_fn(C: jax.Array, lags: jax.Array, L: int, hp: Hyperparameters,
                G: float, sigma: float):
    """Jitted whole-brain scan. Carry = (state, history (L, N, D) of past outputs)."""
    N = C.shape[0]
    D = hp.d_model
    ar = np.arange(N)

    def step(carry, key_t):
        state, history = carry
        t = state.step
        idx = (t - lags) % L                       # (N, N)
        y_hist = history[idx, ar, :]               # (N, N, D) y_j(t - lags[i,j])
        aff = G * np.einsum("ij,ijd->id", C, y_hist)          # delayed afferent prediction
        noise = sigma * jax.random.normal(key_t, (N, D))
        x = (aff + noise)[None, :, :]              # (B=1, N, D)
        state, y, surprise = mhsa_step(wb_mhsa, state, x, hp)  # y (1,N,D), surprise (1,N,H)
        y = y[0]
        e = surprise[0].mean(-1)                   # local prediction error per area (N,)
        history = history.at[t % L].set(y)
        state = state._replace(step=t + 1)
        return (state, history), (y, e)

    wb_mhsa = None  # closed over via init below (set by caller through partial)
    def build(mhsa_params):
        nonlocal wb_mhsa
        wb_mhsa = mhsa_params
        return jax.jit(lambda carry, keys: jax.lax.scan(step, carry, keys))
    return build


def simulate(mhsa_params, C, lags, L, hp, G, sigma, T, key):
    """Run T integration steps; returns e (T, N) local prediction errors, y (T, N, D)."""
    N = C.shape[0]
    _, state, _ = init_ct_mhsa(hp, key, batch_size=1)   # zero memory, no lags (we manage history)
    history = np.zeros((L, N, hp.d_model))
    keys = jax.random.split(jax.random.fold_in(key, 1), T)
    sim = make_sim_fn(C, lags, L, hp, G, sigma)(mhsa_params)
    (fs, _), (y_seq, e_seq) = sim((state, history), keys)
    return e_seq, y_seq


def bold_forward(signal: onp.ndarray, dt_ms: float, tr_s: float) -> onp.ndarray:
    """Balloon model: input (T, N) neural signal at step dt_ms -> BOLD sampled at TR.

    One offline_sample per TR window yields the BOLD at each window end
    (aaicd.py make_bold_hopf pattern). Balloon ODE time constants are in
    seconds, so the Heun step uses dt_s = dt_ms/1000.
    NOTE: installed jax's scan signature is scan(f, init, xs).
    """
    T, N = signal.shape
    every = int(round(tr_s * 1000.0 / dt_ms))
    W = T // every
    windowed = signal[:W * every].reshape(W, every, N)
    bold_buf, bold_step, bold_sample = make_bold(
        shape=(N,), dt=dt_ms / 1000.0, p=bold_default_theta)
    bold_offline = make_offline(step_fn=bold_step, sample_fn=bold_sample)
    _, bold = jax.lax.scan(bold_offline, bold_buf, np.asarray(windowed))
    return onp.asarray(bold)  # (W, N), one sample per TR


def fc_from_ts(ts: onp.ndarray) -> onp.ndarray:
    z = (ts - ts.mean(0)) / (ts.std(0) + 1e-8)
    return (z.T @ z) / ts.shape[0]


def fc_match(fc_sim: onp.ndarray, fc_emp: onp.ndarray) -> float:
    n = fc_sim.shape[0]
    iu = onp.triu_indices(n, k=1)
    return float(onp.corrcoef(fc_sim[iu], fc_emp[iu])[0, 1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sc", required=True, help="structural connectome (txt/npy), (N,N)")
    p.add_argument("--lengths", required=True, help="tract lengths (N,N)")
    p.add_argument("--emp-ts", default=None, help="empirical BOLD time series (T_vol, N)")
    p.add_argument("--emp-fc", default=None,
                   help="empirical FC matrix (N,N) directly; skips the BOLD forward model "
                        "on the empirical side (use when only FC matrices exist)")
    p.add_argument("--out", default="runs/wb_fc")
    p.add_argument("--dt", type=float, default=2.0, help="integration step ms")
    p.add_argument("--tr", type=float, default=2.0, help="fMRI repetition time s")
    p.add_argument("--v-c", type=float, default=5.0, help="conduction velocity mm/ms")
    p.add_argument("--d-model", type=int, default=16)
    p.add_argument("--sim-s", type=float, default=600.0, help="simulated seconds")
    p.add_argument("--g-sweep", default="0.5,1.0,2.0")
    p.add_argument("--tau-sweep", default="20,40")
    p.add_argument("--sigma-sweep", default="0.3")
    p.add_argument("--kv-scale", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    C, lags, L = load_connectome(a.sc, a.lengths, a.dt, a.v_c)
    N = C.shape[0]
    if a.emp_fc is not None:
        fc_emp = onp.load(a.emp_fc) if a.emp_fc.endswith(".npy") else onp.loadtxt(a.emp_fc)
        assert fc_emp.shape == (N, N), f"emp fc {fc_emp.shape} vs N={N}"
        emp_source = a.emp_fc
    elif a.emp_ts is not None:
        emp_ts = onp.load(a.emp_ts) if a.emp_ts.endswith(".npy") else onp.loadtxt(a.emp_ts)
        assert emp_ts.shape[1] == N, f"emp ts {emp_ts.shape} vs N={N}"
        fc_emp = fc_from_ts(emp_ts)
        emp_source = a.emp_ts
    else:
        # no empirical side: fixed-seed random FC -> results are PIPELINE VALIDATION, not a fit
        dkey = onp.random.RandomState(a.seed)
        fc_emp = dkey.randn(N, N)
        iu = onp.triu_indices(N, 1)
        fc_emp = (fc_emp + fc_emp.T) / 2
        onp.fill_diagonal(fc_emp, 1.0)
        fc_emp[iu] /= (onp.abs(fc_emp[iu]).max() + 1e-8)
        emp_source = "SYNTHETIC-DUMMY (no empirical TS/FC available; pipeline validation only)"
    T = int(a.sim_s * 1000 / a.dt)
    key = jax.random.PRNGKey(a.seed)

    os.makedirs(a.out, exist_ok=True)
    results = []
    best = None
    for G, tau, sigma in itertools.product(
            [float(x) for x in a.g_sweep.split(",")],
            [float(x) for x in a.tau_sweep.split(",")],
            [float(x) for x in a.sigma_sweep.split(",")]):
        hp = Hyperparameters(n_regions=N, n_heads=N, d_model=a.d_model,
                             d_k=a.d_model, d_v=a.d_model,
                             lam=1.0 / tau, dt=1.0, tau=tau, leak=1.0,
                             act="relu", steps_per_token=1, sigma_a=0.0)
        k_m, k_sim = jax.random.split(key)
        mhsa, _, _ = init_ct_mhsa(hp, k_m, batch_size=1)
        if a.kv_scale != 1.0:
            mhsa = mhsa._replace(W_K=mhsa.W_K * a.kv_scale, W_V=mhsa.W_V * a.kv_scale)
        e, y = simulate(mhsa, np.asarray(C), jax.numpy.asarray(lags), L, hp,
                        G, sigma, T, k_sim)
        bold = bold_forward(onp.asarray(e), a.dt, a.tr)
        fc_sim = fc_from_ts(bold)
        score = fc_match(fc_sim, fc_emp)
        results.append({"G": G, "tau": tau, "sigma": sigma, "fc_match": score})
        print(f"G={G} tau={tau} sigma={sigma} fc_match={score:.4f}")
        if best is None or score > best["fc_match"]:
            best = dict(results[-1])
            onp.save(os.path.join(a.out, "fc_sim_best.npy"), fc_sim)

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "results.json"), "w") as f:
        json.dump({"results": results, "best": best, "N": N, "T": T,
                   "dt": a.dt, "tr": a.tr, "kv_scale": a.kv_scale, "seed": a.seed,
                   "emp_source": emp_source}, f, indent=2)
    emp_label = "synthetic dummy" if emp_source.startswith("SYN") else ("FC matrix" if a.emp_fc else "BOLD")
    # heatmap comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(fc_emp); axes[0].set_title(f"empirical FC ({emp_label})")
    axes[1].imshow(onp.load(os.path.join(a.out, "fc_sim_best.npy"))); axes[1].set_title(f"sim FC (best, r={best['fc_match']:.3f})")
    plt.tight_layout(); plt.savefig(os.path.join(a.out, "fc_comparison.png"), dpi=120)
    print("best:", best)


if __name__ == "__main__":
    main()
