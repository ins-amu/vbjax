"""Localise where d(loss)/d(G_rel) vanishes, and compare candidate loss variants.

Findings so far (trained params, window 0, dt_train=100ms):
  mode        loss      |gG_rel|   |gkv_scale|  |glog_tau|
  current    0.774     6.07e-13    2.51e-02     3.83e-05
  sym_win    1.000     4.90e-13    2.32e-02     3.49e-05
  gscale     0.777     2.52e-13    1.52e+00     2.42e-03
=> G_rel's gradient is ~1e-13 under *every* normalisation tried, so the
   prediction normalisation is not the cause. This probe adds the upstream
   diagnostics that decide between two candidates:
   (a) the coupling drive G*C@y is swamped by the injected noise before it can
       shape the surprise readout, and/or
   (b) G's pushforward is a per-region affine change, exactly in the null space
       of any per-region normalisation.

Reported: aff/noise RMS ratio, ||d(e_vol)/d(G)|| and its null-space fraction,
the loss-vs-G scan (flat => unidentifiable, sloped => gradient pathology), and
per-mode loss + gradient norms including the balloon variants.

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

def window_forward(params, carry, keys, C, hp0, win_vols, dt_ms, chunk=25,
                   with_diag=False, coupling="raw", burnin=0):
    """One window. Returns (e_vol (V,N), e_seq (T,N)) and optional diagnostics.

    coupling: 'raw' = couple through the raw block output (historical trainer
    behaviour); 'ema' = couple through the EMA-normalised output, which is what
    wholebrain_fc.simulate stores in its history buffer.
    burnin: stop-gradient warm-up steps before the differentiable window; lets
    the EMA normaliser (time constant 1000 steps) converge to the block output
    scale, as it has in the eval simulator (300k steps).
    """
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
        if with_diag:
            M, n_prev, y_ema, y_prev, s_aff, s_noi, s_y, cnt = carry_n
        else:
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
        # coupling input for the NEXT step. wholebrain_fc.simulate couples
        # through the EMA-normalised output; the training window historically
        # coupled through raw y, which is ~1e-9 once kv_scale collapses.
        y_coupled = y / (y_ema + 1e-8) if coupling == "ema" else y
        if with_diag:
            return (state.M, noise, y_ema, y_coupled,
                    s_aff + np.mean(aff ** 2), s_noi + np.mean(noise ** 2),
                    s_y + np.mean(y_coupled ** 2), cnt + 1), e
        return (state.M, noise, y_ema, y_coupled), e

    def chunk_step(carry_c, kc):
        return jax.lax.scan(neural_step, carry_c, kc)

    if burnin > 0:
        # warm-up (no gradient): converge M, noise and y_ema to their stationary
        # regime so the differentiable window starts from a realistic state
        bk = jax.random.split(jax.random.fold_in(keys, 99), burnin)
        burnin_scan = jax.jit(
            lambda c, k: jax.lax.scan(neural_step, c, k)[0])
        M0, n0, fe0, y0 = jax.lax.stop_gradient(
            burnin_scan((M0, n0, fe0, y0), bk))

    carry0 = ((M0, n0, fe0, y0, np.zeros(()), np.zeros(()), np.zeros(()),
               np.zeros(())) if with_diag else (M0, n0, fe0, y0))
    fin, e_chunks = jax.lax.scan(jax.checkpoint(chunk_step), carry0, keys_c)
    e_seq = e_chunks.reshape(T_int, N)
    e_vol = e_seq.reshape(win_vols, steps_per_vol, N).mean(axis=1)
    if with_diag:
        aff_rms = np.sqrt(fin[4] / fin[7])
        noise_rms = np.sqrt(fin[5] / fin[7])
        y_rms = np.sqrt(fin[6] / fin[7])
        return e_vol, e_seq, (aff_rms, noise_rms, y_rms)
    return e_vol, e_seq


def bold_forward_jax(signal, dt_ms, tr_s):
    """Differentiable balloon: (T,N) neural signal at dt_ms -> BOLD sampled at TR."""
    T, N = signal.shape
    every = int(round(tr_s * 1000.0 / dt_ms))
    if every < 1:
        raise ValueError(f"tr_s={tr_s} too small for dt_ms={dt_ms}")
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

def make_loss(mode, emp_win, emp_raw_win, dt_ms, tr_s, lam_amp=0.5):
    """tr_s in SECONDS."""

    def loss_fn(e_vol, e_seq):
        if mode == "current":
            pred, tgt = zscore(e_vol), emp_win
        elif mode == "sym_win":
            pred, tgt = zscore(e_vol), zscore(emp_raw_win)
        elif mode == "gscale":
            pred, tgt = e_vol / (np.std(e_vol) + 1e-8), emp_win
        elif mode == "raw":
            # no prediction normalisation at all: the only variant in which a
            # pure gain on the drive is directly observable
            pred, tgt = e_vol, emp_raw_win
        elif mode == "balloon_pr":
            bold = bold_forward_jax(rescale_per_region(e_seq), dt_ms, tr_s)
            pred, tgt = zscore(bold), zscore(emp_raw_win[: bold.shape[0]])
        elif mode == "balloon_gs":
            bold = bold_forward_jax(rescale_shared(e_seq), dt_ms, tr_s)
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
    """Fraction of de in span{1, e_r} per region (the per-region z-score null space)."""
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
    ap.add_argument("--params", default=None)
    ap.add_argument("--out", default="runs/wb_probe")
    ap.add_argument("--dt", type=float, default=2.0)
    ap.add_argument("--dt-train", type=float, default=100.0)
    ap.add_argument("--tr", type=float, default=2.0, help="TR in seconds")
    ap.add_argument("--v-c", type=float, default=5.0)
    ap.add_argument("--d-model", type=int, default=16)
    ap.add_argument("--win", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--modes", default="current,balloon_gs")
    ap.add_argument("--couplings", default="raw,ema")
    ap.add_argument("--burnin", type=int, default=0,
                    help="stop-gradient warm-up steps before the window")
    ap.add_argument("--g-scan", default="0.5,1.0,1.5,2.09,3.0,4.0")
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
        return (np.zeros((1, N, N, d, d)), np.zeros((N, d)), np.ones((N, 1)),
                np.zeros((N, d)))

    carry = full_carry()
    kw = jax.random.fold_in(key, 0)
    emp_win, emp_raw = emp_z_wins[0], emp_raw_wins[0]

    out = {"params_source": src, "N": N, "win": args.win,
           "dt_train": args.dt_train, "tr": args.tr,
           "learned": {k: float(v) for k, v in params.items()
                       if k not in ("W_q", "W_y")},
           "coupling": {}, "g_scan": {}, "modes": {}}
    os.makedirs(args.out, exist_ok=True)

    def save():
        with open(os.path.join(args.out, "probe.json"), "w") as f:
            json.dump(out, f, indent=1)

    gvals = [float(x) for x in args.g_scan.split(",")]

    for coupling in args.couplings.split(","):
        tag = f"{coupling}-b{args.burnin}"
        out["coupling"][tag] = {}
        out["g_scan"][tag] = {}

        # --- diagnostics: is the coupling drive visible against the noise?
        e_vol, e_seq, diag = window_forward(
            params, carry, kw, C, hp0, args.win, args.dt_train,
            with_diag=True, coupling=coupling, burnin=args.burnin)
        e_vol_s = jax.lax.stop_gradient(e_vol)
        aff_rms, noise_rms, y_rms = (float(diag[0]), float(diag[1]),
                                     float(diag[2]))
        out["coupling"][tag]["diag"] = {
            "aff_rms": aff_rms, "noise_rms": noise_rms, "y_coupled_rms": y_rms,
            "snr_aff_over_noise": aff_rms / (noise_rms + 1e-30),
            "e_vol_mean": float(e_vol.mean()), "e_vol_std": float(e_vol.std()),
        }
        print(f"[{tag}] aff_rms={aff_rms:.4e} noise_rms={noise_rms:.4e} "
              f"snr={aff_rms/(noise_rms+1e-30):.4f} y_rms={y_rms:.4e}", flush=True)

        # --- G pushforward: magnitude and null-space fraction
        def fwd_G(g, _c=coupling, _b=args.burnin):
            p = dict(params)
            p["G_rel"] = g
            return window_forward(p, carry, kw, C, hp0, args.win,
                                  args.dt_train, coupling=_c, burnin=_b)[0]

        de = jax.jvp(fwd_G, (params["G_rel"],), (np.asarray(1.0),))[1]
        nf, dn = null_fraction(de, onp.asarray(e_vol_s))
        out["coupling"][tag]["G_pushforward"] = {
            "norm": dn, "null_frac": nf,
            "rel_to_e_vol_std": dn / (float(e_vol_s.std()) + 1e-30),
        }
        print(f"[{tag}] ||d(e_vol)/dG||={dn:.4e} null_frac={nf:.4f}",
              flush=True)
        save()

        # --- loss vs G: flat => unidentifiable
        for mode in ("current", "gscale"):
            loss_fn = make_loss(mode, emp_win, emp_raw, args.dt_train, args.tr)
            row = {}
            for g in gvals:
                p = dict(params)
                p["G_rel"] = np.asarray(g)
                ev, es = window_forward(p, carry, kw, C, hp0, args.win,
                                        args.dt_train, coupling=coupling,
                                        burnin=args.burnin)
                row[g] = float(loss_fn(ev, es))
            out["g_scan"][coupling][mode] = row
            print(f"[gscan:{tag}:{mode}] "
                  + " ".join(f"G={g}:{v:.5f}" for g, v in row.items()),
                  flush=True)
        save()

        # --- per-mode loss + gradient norms (isolated)
        for mode in args.modes.split(","):
            key_m = f"{coupling}:{mode}"
            try:
                loss_fn = make_loss(mode, emp_win, emp_raw, args.dt_train, args.tr)

                def total(p, _c=coupling, _lf=loss_fn, _b=args.burnin):
                    ev, es = window_forward(p, carry, kw, C, hp0, args.win,
                                            args.dt_train, coupling=_c,
                                            burnin=_b)
                    return _lf(ev, es)

                val, grads = jax.value_and_grad(total)(params)
                gnorm = {k: float(np.sqrt(np.sum(np.asarray(g) ** 2)))
                         for k, g in grads.items()}
                out["modes"][key_m] = {"loss": float(val), "grad_norm": gnorm}
                print(f"[{key_m}] loss={float(val):.5f} "
                      f"|gkv|={gnorm['kv_scale']:.3e} |gG|={gnorm['G_rel']:.3e} "
                      f"|gtau|={gnorm['log_tau']:.3e} "
                      f"|gsig|={gnorm['log_sigma']:.3e}", flush=True)
            except Exception as exc:  # noqa: BLE001
                out["modes"][key_m] = {"error": f"{type(exc).__name__}: {exc}"}
                print(f"[{key_m}] FAILED: {type(exc).__name__}: {exc}", flush=True)
            save()

    print("WB_PROBE_DONE", flush=True)


if __name__ == "__main__":
    main()
