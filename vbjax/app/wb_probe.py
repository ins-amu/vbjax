"""Localise where d(loss)/d(G_rel) vanishes, and compare candidate loss variants.

Context: in the 150-iter run G_rel moved 2.090 -> 2.096 while log_tau/log_sigma
moved 0.21-0.37. This probe measures, with jax, the fraction of G_rel's
pushforward on the window surprise trace `e_vol` that lies in the null space of
the per-region z-score used by the loss.

The z-score `zscore(e)` (per region, over the window) is invariant to
e -> a_r * 1 + b_r * e  for per-region constants a_r, b_r. Its Jacobian null
space at e is therefore span{1, e_r} per region. If G_rel's pushforward
de = d(e_vol)/d(G) lies in that span (i.e. it is a per-region affine change),
then d(loss)/d(G) = 0 exactly, no matter what the optimiser does.

Reported per loss variant: loss value, per-parameter gradient norms, and
`null_frac` = ||P_null(de)|| / ||de|| for the shipped forward.

Run inside the GPU container from the tree root:
  python -m vbjax.app.wb_probe --sc SC.npy --lengths LEN.npy --ts1 TS.npy \
      --params runs_persist/wb_train/params.pkl --out runs/wb_probe
"""
import argparse, json, os, pickle
import numpy as onp
import jax
import jax.numpy as np

from vbjax import make_bold, bold_default_theta, make_offline
from vbjax.ct_mhsa import Hyperparameters, NetworkState, init_ct_mhsa, mhsa_step
from vbjax.app.wholebrain_fc import load_connectome
from vbjax.app.wb_train import zscore, build_mhsa


# ---------------------------------------------------------------- forward

def window_forward(params, carry, keys, C, hp0, win_vols, dt_ms, chunk=25):
    """One window: returns (e_vol (V,N) TR-averaged surprise, e_seq (T,N))."""
    N = C.shape[0]
    D = params["W_q"].shape[-1]
    G = params["G_rel"]
    tau_s = np.exp(params["log_tau"])
    sigma = np.exp(params["log_sigma"])
    noise_tc = np.exp(params["log_noise_tc"])
    dt_s = dt_ms / 1000.0
    u = dt_s / tau_s
    noise_a = np.exp(-dt_s / noise_tc)
    sq1 = np.sqrt(1.0 - noise_a ** 2 + 1e-12)
    mhsa = build_mhsa(hp0, jax.random.PRNGKey(0), params["W_q"], params["W_y"],
                      params["kv_scale"])
    hp_eff = hp0._replace(lam=u, tau=tau_s * 1000.0)
    steps_per_vol = int(round(2000.0 / dt_ms))
    T_int = win_vols * steps_per_vol
    if T_int % chunk:
        chunk = max(c for c in range(1, chunk + 1) if T_int % c == 0)
    n_chunks = T_int // chunk
    M0, n0, fe0, y0 = carry
    keys_c = jax.random.split(keys, T_int).reshape(n_chunks, chunk, 2)

    def neural_step(carry_n, key_t):
        M, n_prev, y_ema, y_prev = carry_n
        state = NetworkState(M=M, history=None, step=0)
        aff = G * np.einsum("ij,jd->id", C, y_prev)
        eps = jax.random.normal(key_t, (N, D))
        noise = noise_a * n_prev + sigma * sq1 * eps
        x = (aff + noise)[None]
        state, y, e = mhsa_step(mhsa, state, x, hp_eff)
        y = y[0]
        e = e[0].mean(-1)
        rms = np.sqrt(np.mean(y ** 2, axis=-1, keepdims=True))
        y_ema = 0.999 * y_ema + 0.001 * rms
        return (state.M, noise, y_ema, y), e

    def chunk_step(carry_c, kc):
        return jax.lax.scan(neural_step, carry_c, kc)

    (fM, fn, fe, fyp), e_chunks = jax.lax.scan(
        jax.checkpoint(chunk_step), (M0, n0, fe0, y0), keys_c)
    e_seq = e_chunks.reshape(T_int, N)
    e_vol = e_seq.reshape(win_vols, steps_per_vol, N).mean(axis=1)
    return e_vol, e_seq


def bold_forward_jax(signal, dt_ms, tr_s):
    """Differentiable balloon: (T,N) neural signal at dt_ms -> BOLD at TR."""
    T, N = signal.shape
    every = int(round(tr_s * 1000.0 / dt_ms))
    W = T // every
    windowed = signal[:W * every].reshape(W, every, N)
    bold_buf, bold_step, bold_sample = make_bold(
        shape=(N,), dt=dt_ms / 1000.0, p=bold_default_theta)
    offline = make_offline(step_fn=bold_step, sample_fn=bold_sample)
    _, bold = jax.lax.scan(offline, bold_buf, windowed)
    return bold


def rescale_per_region(x, lo=0.2, hi=1.0):
    mn = x.min(0, keepdims=True)
    mx = x.max(0, keepdims=True)
    return lo + (hi - lo) * (x - mn) / (mx - mn + 1e-8)


def rescale_shared(x, lo=0.2, hi=1.0):
    mn = x.min()
    mx = x.max()
    return lo + (hi - lo) * (x - mn) / (mx - mn + 1e-8)


# ---------------------------------------------------------------- losses

def make_loss(mode, emp_win, emp_raw_win, dt_ms, tr_ms, lam_amp=0.5):
    """Return loss(e_vol, e_seq) -> scalar."""

    def loss_fn(e_vol, e_seq):
        if mode == "current":
            pred, tgt = zscore(e_vol), emp_win
        elif mode == "sym_win":
            pred, tgt = zscore(e_vol), zscore(emp_raw_win)
        elif mode == "gscale":
            pred = e_vol / (np.std(e_vol) + 1e-8)
            tgt = emp_win
        elif mode == "balloon_pr":
            bold = bold_forward_jax(rescale_per_region(e_seq), dt_ms, tr_ms / 1000.0)
            pred, tgt = zscore(bold), zscore(emp_raw_win[: bold.shape[0]])
        elif mode == "balloon_gs":
            bold = bold_forward_jax(rescale_shared(e_seq), dt_ms, tr_ms / 1000.0)
            pred, tgt = zscore(bold), zscore(emp_raw_win[: bold.shape[0]])
        elif mode == "amp":
            pred, tgt = zscore(e_vol), zscore(emp_raw_win)
            ls_p = np.log(pred.std(0) + 1e-8)
            ls_t = np.log(tgt.std(0) + 1e-8)
            return np.mean((pred - tgt) ** 2) + lam_amp * np.mean((ls_p - ls_t) ** 2)
        else:
            raise ValueError(mode)
        return np.mean((pred - tgt) ** 2)

    return loss_fn


def null_fraction(de, e_vol):
    """Fraction of de that lies in span{1, e_r} per region (z-score null space)."""
    resid_sq, tot_sq = 0.0, 0.0
    for r in range(de.shape[1]):
        d = onp.asarray(de[:, r])
        e = onp.asarray(e_vol[:, r])
        X = onp.stack([onp.ones_like(e), e], axis=1)
        coef, *_ = onp.linalg.lstsq(X, d, rcond=None)
        res = d - X @ coef
        resid_sq += float(res @ res)
        tot_sq += float(d @ d)
    return (resid_sq ** 0.5) / (tot_sq ** 0.5 + 1e-30), tot_sq ** 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sc", required=True)
    ap.add_argument("--lengths", required=True)
    ap.add_argument("--ts1", required=True)
    ap.add_argument("--params", default=None, help="trained params.pkl (else init)")
    ap.add_argument("--out", default="runs/wb_probe")
    ap.add_argument("--dt", type=float, default=2.0)
    ap.add_argument("--dt-train", type=float, default=100.0)
    ap.add_argument("--tr", type=float, default=2.0)
    ap.add_argument("--v-c", type=float, default=5.0)
    ap.add_argument("--d-model", type=int, default=16)
    ap.add_argument("--win", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--modes", default="current,sym_win,gscale,balloon_pr,balloon_gs,amp")
    args = ap.parse_args()

    C, lags, L = load_connectome(args.sc, args.lengths, args.dt, args.v_c, "sqrt99")
    N = C.shape[0]
    C = C / float(onp.max(onp.abs(onp.linalg.eigvals(C))))
    d = args.d_model
    hp0 = Hyperparameters(n_regions=N, n_heads=N, d_model=d, d_k=d, d_v=d,
                          lam=1.0, dt=args.dt, tau=1e6, leak=1.0, act="relu",
                          steps_per_token=1, sigma_a=0.0)
    key = jax.random.PRNGKey(args.seed)
    k_m, _ = jax.random.split(key)
    mhsa0, _, _ = init_ct_mhsa(hp0, k_m, batch_size=1)

    if args.params:
        with open(args.params, "rb") as f:
            params = pickle.load(f)
        src = args.params
    else:
        params = {"W_q": mhsa0.W_Q, "W_y": mhsa0.W_Y,
                  "kv_scale": np.asarray(0.15), "G_rel": np.asarray(2.09),
                  "log_tau": np.asarray(onp.log(36.0)),
                  "log_sigma": np.asarray(onp.log(0.18)),
                  "log_noise_tc": np.asarray(onp.log(2.0))}
        src = "init"
    params = {k: np.asarray(v) for k, v in params.items()}

    ts1 = onp.load(args.ts1)
    n_win = ts1.shape[0] // args.win
    emp_raw_wins = ts1[:n_win * args.win].reshape(n_win, args.win, N)
    emp_z_wins = zscore(ts1)[:n_win * args.win].reshape(n_win, args.win, N)

    def full_carry():
        # mirror wb_train.full_carry exactly: M from init_ct_mhsa is (B=1, N, H=N, Dv, Dk)
        return (np.zeros((1, N, N, d, d)), np.zeros((N, d)), np.ones((N, 1)),
                np.zeros((N, d)))

    carry = full_carry()
    kw = jax.random.fold_in(key, 0)
    emp_win, emp_raw = emp_z_wins[0], emp_raw_wins[0]

    out = {"params_source": src, "N": N, "win": args.win,
           "dt_train": args.dt_train, "tr": args.tr, "modes": {}}

    e_vol, e_seq = window_forward(params, carry, kw, C, hp0, args.win, args.dt_train)
    e_vol = jax.lax.stop_gradient(e_vol)
    e_seq = jax.lax.stop_gradient(e_seq)
    out["e_vol_stats"] = {
        "mean": float(e_vol.mean()), "std": float(e_vol.std()),
        "min": float(e_vol.min()), "max": float(e_vol.max()),
        "per_region_std_of_mean": float(e_vol.mean(0).std()),
    }

    # --- localisation: does d(e_vol)/d(G) lie in the z-score null space?
    def fwd_G(g):
        p = dict(params)
        p["G_rel"] = g
        return window_forward(p, carry, kw, C, hp0, args.win, args.dt_train)[0]

    de = jax.jvp(fwd_G, (params["G_rel"],), (np.asarray(1.0),))[1]
    nf, dn = null_fraction(de, e_vol)
    out["G_pushforward"] = {
        "null_frac": nf,
        "norm": dn,
        "rel_norm": dn / (float(e_vol.std()) + 1e-30),
        "note": "null_frac ~1 => d(e_vol)/d(G) is a per-region affine change, "
                "exactly annihilated by the per-region z-score",
    }

    # --- per-mode loss + gradient norms
    for mode in args.modes.split(","):
        loss_fn = make_loss(mode, emp_win, emp_raw, args.dt_train, args.tr)

        def total(p):
            ev, es = window_forward(p, carry, kw, C, hp0, args.win, args.dt_train)
            return loss_fn(ev, es)

        val, grads = jax.value_and_grad(total)(params)
        gnorm = {k: float(np.sqrt(np.sum(np.asarray(g) ** 2)))
                 for k, g in grads.items()}
        out["modes"][mode] = {
            "loss": float(val),
            "grad_norm": gnorm,
            "grad_rel_to_param": {
                k: float(np.sqrt(np.sum(np.asarray(grads[k]) ** 2))
                         / (float(np.abs(np.asarray(params[k])).mean()) + 1e-12))
                for k in ("W_q", "W_y", "kv_scale", "G_rel", "log_tau",
                          "log_sigma", "log_noise_tc")},
        }
        print(f"[{mode}] loss={float(val):.5f} "
              f"|gWq|={gnorm['W_q']:.3e} |gWy|={gnorm['W_y']:.3e} "
              f"|gkv|={gnorm['kv_scale']:.3e} |gG|={gnorm['G_rel']:.3e} "
              f"|gtau|={gnorm['log_tau']:.3e} |gsig|={gnorm['log_sigma']:.3e} "
              f"|gtc|={gnorm['log_noise_tc']:.3e}", flush=True)

    print(f"G pushforward null_frac={nf:.4f} (norm {dn:.4e})", flush=True)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "probe.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("WB_PROBE_DONE", flush=True)


if __name__ == "__main__":
    main()
