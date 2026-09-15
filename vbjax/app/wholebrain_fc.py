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


def load_connectome(sc_path: str, lengths_path: str, dt: float, v_c: float,
                    c_norm: str = "sqrt99"):
    """Structural weights + tract lengths -> normalized C and integer per-pair delays.
    c_norm: 'sqrt99' (aaicd.py style: sqrt, clip at 99th pct) or 'row' (row-stochastic,
    each area's afferents sum to 1 — TVB-style; shape-preserving coupling)."""
    W = onp.load(sc_path) if sc_path.endswith(".npy") else onp.loadtxt(sc_path)
    L = onp.load(lengths_path) if lengths_path.endswith(".npy") else onp.loadtxt(lengths_path)
    nreg = W.shape[0]
    assert W.shape == L.shape == (nreg, nreg)
    W = W - W.min()
    if c_norm == "row":
        rs = W.sum(axis=1, keepdims=True)
        W = W / (rs + (rs == 0) * 1.0)      # each target area's afferents sum to 1
    else:
        W = W / onp.percentile(W, 99)
        W = onp.clip(onp.sqrt(W), 0, 1)
    onp.fill_diagonal(W, 0.0)
    lags = onp.floor(L / v_c / dt).astype("i")
    Lmax = int(lags.max())
    return W.astype("f4"), lags, Lmax + 1


def make_sim_fn(C: jax.Array, lags: jax.Array, L: int, hp: Hyperparameters,
                G: float, sigma: float, noise_a: float, bold_src: str):
    """Jitted whole-brain scan. Carry = (state, history (L, N, D) of past outputs,
    AR(1) noise state). noise_a: OU-like coefficient; bold_src in {'e','y','ey'}"""
    N = C.shape[0]
    D = hp.d_model
    ar = np.arange(N)

    def step(carry, key_t):
        state, history, n_prev, y_ema = carry
        t = state.step
        idx = (t - lags) % L                       # (N, N)
        y_hist = history[idx, ar, :]               # (N, N, D) y_j(t - lags[i,j])
        aff = G * np.einsum("ij,ijd->id", C, y_hist)          # delayed afferent prediction
        # AR(1) (OU-like) noise, stationary variance sigma^2
        eps = jax.random.normal(key_t, (N, D))
        noise = noise_a * n_prev + sigma * np.sqrt(1.0 - noise_a ** 2) * eps
        x = (aff + noise)[None, :, :]              # (B=1, N, D)
        state, y, surprise = mhsa_step(wb_mhsa, state, x, hp)  # y (1,N,D), surprise (1,N,H)
        y = y[0]
        e = surprise[0].mean(-1)                   # local prediction error per area (N,)
        # per-area EMA normalization: local output -> O(1) so the coupling gain is
        # meaningful (neural-mass-model convention; the raw block output scale
        # depends on u, kv_scale etc. and is otherwise ~1e-6 -> uncoupled areas)
        rms = np.sqrt(np.mean(y ** 2, axis=-1, keepdims=True))   # (N, 1)
        y_ema = 0.999 * y_ema + 0.001 * rms
        y_out = y / (y_ema + 1e-8)
        history = history.at[t % L].set(y_out)
        state = state._replace(step=t + 1)
        return (state, history, noise, y_ema), (y_out, e)

    wb_mhsa = None  # closed over via build() below
    def build(mhsa_params):
        nonlocal wb_mhsa
        wb_mhsa = mhsa_params
        return jax.jit(lambda carry, keys: jax.lax.scan(step, carry, keys))
    return build


def simulate(mhsa_params, C, lags, L, hp, G, sigma, T, key,
             noise_tc_s: float = 2.0, dt_ms: float = 2.0, bold_src: str = "e"):
    """Run T integration steps; returns e (T, N) local prediction errors, y (T, N, D)."""
    N = C.shape[0]
    _, state, _ = init_ct_mhsa(hp, key, batch_size=1)   # zero memory, no lags (we manage history)
    history = np.zeros((L, N, hp.d_model))
    keys = jax.random.split(jax.random.fold_in(key, 1), T)
    noise_a = float(onp.exp(-(dt_ms / 1000.0) / max(noise_tc_s, 1e-6)))
    sim = make_sim_fn(C, lags, L, hp, G, sigma, noise_a, bold_src)(mhsa_params)
    (fs, _, _, _), (y_seq, e_seq) = sim((state, history, np.zeros((N, hp.d_model)),
                                         np.ones((N, 1))), keys)
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
    p.add_argument("--noise-tc", type=float, default=2.0, help="AR(1) noise time constant (s)")
    p.add_argument("--bold-input", default="e", choices=["e", "y", "ey"],
                   help="signal fed to the balloon: prediction error (e), mean area output (y), or sum")
    p.add_argument("--g-rel", action="store_true", help="interpret g-sweep as fractions of Gmax=1/rho(C)")
    p.add_argument("--spectral-norm", action="store_true", default=True,
                   help="normalize C to unit spectral radius (G then = loop gain)")
    p.add_argument("--no-spectral-norm", dest="spectral_norm", action="store_false")
    p.add_argument("--c-norm", default="sqrt99", choices=["sqrt99", "row"],
                   help="SC normalization: sqrt99 (aaicd style) or row (row-stochastic)")
    p.add_argument("--g-sweep", default="0.5,1.0,2.0")
    p.add_argument("--tau-sweep", default="20,40")
    p.add_argument("--sigma-sweep", default="0.3")
    p.add_argument("--kv-scale", type=float, default=0.15)
    p.add_argument("--delta-rule", action="store_true",
                   help="error-gated memory write M += u(v - M phi(k)) phi(k)^T")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    C, lags, L = load_connectome(a.sc, a.lengths, a.dt, a.v_c, a.c_norm)
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
    # stability cap: recurrent loop gain G * rho(C) must stay below 1
    rho = float(onp.max(onp.abs(onp.linalg.eigvals(C))))
    if a.spectral_norm and rho > 1.0:
        C = C / rho
        rho = 1.0
        print("C spectrally normalized to rho=1")
    Gmax = 1.0 / max(rho, 1e-8)
    print(f"N={N} T={T} rho(C)={rho:.3f} Gmax={Gmax:.3f}")

    os.makedirs(a.out, exist_ok=True)
    results = []
    best = None
    for G, tau, sigma in itertools.product(
            [float(x) for x in a.g_sweep.split(",")],
            [float(x) for x in a.tau_sweep.split(",")],
            [float(x) for x in a.sigma_sweep.split(",")]):
        # tau in SECONDS on the CLI; memory_update_rate = dt/tau with dt in ms
        hp = Hyperparameters(n_regions=N, n_heads=N, d_model=a.d_model,
                             d_k=a.d_model, d_v=a.d_model,
                             lam=a.dt / (tau * 1000.0), dt=a.dt, tau=tau * 1000.0,
                             leak=1.0, act="relu", steps_per_token=1, sigma_a=0.0,
                             delta_rule=1 if a.delta_rule else 0)
        k_m, k_sim = jax.random.split(key)
        mhsa, _, _ = init_ct_mhsa(hp, k_m, batch_size=1)
        if a.kv_scale != 1.0:
            mhsa = mhsa._replace(W_K=mhsa.W_K * a.kv_scale, W_V=mhsa.W_V * a.kv_scale)
        G_abs = G * Gmax if a.g_rel else G
        e, y = simulate(mhsa, np.asarray(C), jax.numpy.asarray(lags), L, hp,
                        G_abs, sigma, T, k_sim, a.noise_tc, a.dt, a.bold_input)
        e = onp.asarray(e); y = onp.asarray(y)
        if not onp.isfinite(e).all() or not onp.isfinite(y).all():
            print(f"G={G_abs:.3f} tau={tau} sigma={sigma} DIVERGED (NaN/inf) -- skipped")
            results.append({"G": round(G_abs, 4), "tau": tau, "sigma": sigma,
                            "fc_match": None, "diverged": True})
            continue
        src = {"e": e, "y": y.mean(-1), "ey": e + y.mean(-1)}[a.bold_input]
        src = onp.asarray(src, dtype="f4")
        # Rescale each area to [0.2, 1.0] before the balloon. Net-negative input
        # drives the balloon volume V negative and V**alpha (alpha=3.2) of a
        # negative base -> NaN (this NaNed the entire round-3 sweep, where the
        # input was z-scored to zero mean). Positive affine rescale preserves
        # the signal's correlation structure.
        lo = src.min(0, keepdims=True)
        hi = src.max(0, keepdims=True)
        src = 0.2 + 0.8 * (src - lo) / (hi - lo + 1e-8)
        bold = bold_forward(src, a.dt, a.tr)
        if not onp.isfinite(bold).all():
            print(f"G={G_abs:.3f} tau={tau} sigma={sigma} BOLD_NAN (non-finite BOLD output)")
            results.append({"G": round(G_abs, 4), "tau": tau, "sigma": sigma,
                            "bold_input": a.bold_input, "fc_match": None, "bold_nan": True})
            continue
        fc_sim = fc_from_ts(bold)
        score = fc_match(fc_sim, fc_emp)
        results.append({"G": round(G_abs, 4), "tau": tau, "sigma": sigma,
                        "bold_input": a.bold_input, "fc_match": score})
        print(f"G={G_abs:.3f} tau={tau} sigma={sigma} in={a.bold_input} fc_match={score:.4f}")
        if best is None or score > best["fc_match"]:
            best = dict(results[-1])
            onp.save(os.path.join(a.out, "fc_sim_best.npy"), fc_sim)

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "results.json"), "w") as f:
        json.dump({"results": results, "best": best, "N": N, "T": T, "Gmax": Gmax,
                   "dt": a.dt, "tr": a.tr, "kv_scale": a.kv_scale, "seed": a.seed,
                   "bold_input": a.bold_input, "noise_tc": a.noise_tc,
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
