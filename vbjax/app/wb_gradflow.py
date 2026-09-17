"""Stage-by-stage autograd attribution for d(loss)/dG_rel.

G enters the model at exactly one place per step:
    aff_t = G * (C @ y_{t-1})        (the afferent coupling drive)
Everything downstream (x -> k,v -> dM -> surprise e -> e_vol -> loss) then
depends on G only through aff_t (and the recurrent state).

This script splits dL/dG into its two factors and measures them stage by stage:

  forward  (supply): ||d(stage)/dG||      via jax.jvp  -- how much G moves the stage
  reverse  (demand): dL/d(stage)          via jax.vjp  -- how much the loss cares
  product:           <dL/dstage, dstage/dG>             -- the exact contribution

`aff` is injected as an explicit input (the step takes aff_t from the scan xs
instead of computing it), so its cotangent dL/daff is exact per step and per
region. Summing <dL/daff, daff/dG> over steps must reproduce dL/dG -- we verify
that, so the decomposition is exact rather than illustrative.

Also reported:
  * per-step contribution profile (where in time the sensitivity lives/dies)
  * direct vs recursive split: with the trajectory frozen, daff/dG = C@y_prev,
    so the direct term is sum_t <dL/daff_t, (C y_prev)_t>; the remainder is the
    recursive (state-feedback) part
  * null-space fraction of d(e_vol)/dG inside span{1, e_vol_r} per region --
    the kernel of the per-region z-score used by the loss
  * log-sensitivity of each stage: dlog||stage||/dlog G

CPU is sufficient (no GPU needed) -- run it in an sbatch job:
  python -m vbjax.app.wb_gradflow --sc SC.npy --lengths LEN.npy --ts1 TS.npy \
      --params runs_persist/wb_arm_ema_yout/params.pkl --out runs/wb_gradflow
"""
import argparse, json, os, pickle
import numpy as onp
import jax
import jax.numpy as np

from vbjax.ct_mhsa import Hyperparameters, NetworkState, init_ct_mhsa, mhsa_step
from vbjax.app.wholebrain_fc import load_connectome
from vbjax.app.wb_train import zscore, build_mhsa


def make_setup(args):
    C, lags, L = load_connectome(args.sc, args.lengths, args.dt, args.v_c, "sqrt99")
    N = C.shape[0]
    C = C / float(onp.max(onp.abs(onp.linalg.eigvals(C))))
    d = args.d_model
    hp0 = Hyperparameters(n_regions=N, n_heads=N, d_model=d, d_k=d, d_v=d,
                          lam=1.0, dt=args.dt, tau=1e6, leak=1.0, act="relu",
                          steps_per_token=1, sigma_a=0.0)
    with open(args.params, "rb") as f:
        params = pickle.load(f)
    params = {k: np.asarray(v) for k, v in params.items()}
    ts1 = onp.load(args.ts1)
    win = args.win
    n_win = ts1.shape[0] // win
    emp_raw = ts1[:n_win * win].reshape(n_win, win, N)[0]
    return C, hp0, params, N, d, emp_raw


def window(params, carry, keys, C, hp0, win, dt_ms, burnin=0, inject=None):
    """Run one window.

    inject=None  -> endogenous: aff_t = G * (C @ y_prev)   (G differentiates)
    inject=array -> the drive is taken from `inject` (T_int, N, D) instead,
                    so the caller can differentiate w.r.t. it directly.

    Returns (loss, stages) with stages carrying light arrays only.
    """
    N = C.shape[0]
    D = params["W_q"].shape[-1]
    G = np.exp(params["log_G"]) if "log_G" in params else params["G_rel"]
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
    T_int = win * steps_per_vol
    M0, n0, fe0, y0 = carry

    def step_core(M, n_prev, y_ema, y_prev, key_t, aff_t):
        state = NetworkState(M=M, history=None, step=0)
        eps = jax.random.normal(key_t, (N, D))
        noise = noise_a * n_prev + sigma * sq1 * eps
        x = (aff_t + noise)[None]
        state, y, e = mhsa_step(mhsa, state, x, hp_eff)
        y = y[0]
        e = e[0].mean(-1)
        rms = np.sqrt(np.mean(y ** 2, axis=-1, keepdims=True))
        y_ema = 0.999 * y_ema + 0.001 * rms
        y_coupled = y / (y_ema + 1e-8)
        return (state.M, noise, y_ema, y_coupled), (e, x[0], aff_t)

    def step_endogenous(carry_n, key_t):
        M, n_prev, y_ema, y_prev = carry_n
        aff_t = G * np.einsum("ij,jd->id", C, y_prev)
        new, (e, x_t, aff_o) = step_core(M, n_prev, y_ema, y_prev, key_t, aff_t)
        return new, (e, x_t, aff_o)

    def step_injected(carry_n, xs_t):
        key_t, aff_t = xs_t
        M, n_prev, y_ema, y_prev = carry_n
        new, (e, x_t, aff_o) = step_core(M, n_prev, y_ema, y_prev, key_t, aff_t)
        return new, (e, x_t, aff_o)

    keys_w = jax.random.split(keys, T_int)
    if burnin > 0:
        bk = jax.random.split(jax.random.fold_in(keys, 77), burnin)

        def burn_step(c, key_t):
            M, n_prev, y_ema, y_prev = c
            aff_t = G * np.einsum("ij,jd->id", C, y_prev)
            return step_core(M, n_prev, y_ema, y_prev, key_t, aff_t)[0]

        carry = jax.lax.stop_gradient(jax.lax.scan(burn_step, carry, bk)[0])

    if inject is None:
        carry, (e_seq, x_seq, aff_seq) = jax.lax.scan(
            step_endogenous, carry, keys_w)
    else:
        carry, (e_seq, x_seq, aff_seq) = jax.lax.scan(
            step_injected, carry, (keys_w, inject))

    e_vol = e_seq.reshape(win, steps_per_vol, N).mean(axis=1)
    loss = np.mean((zscore(e_vol) - zscore(emp_raw)) ** 2)
    stages = {"aff": aff_seq, "x": x_seq, "e": e_seq, "e_vol": e_vol,
              "y_coupled": carry[3], "M": carry[0]}
    return loss, stages


def log_sens(f, g_val):
    """dlog||f||/dlog G  (finite-difference-free first-order version)."""
    val = f(g_val)
    tang = jax.jvp(f, (g_val,), (np.asarray(1.0),))[1]
    nv = float(np.sqrt(np.sum(val ** 2)) + 1e-30)
    nt = float(np.sqrt(np.sum(tang ** 2)) + 1e-30)
    return nt / nv * abs(float(g_val))     # dlog||stage||/dlogG


def null_frac(de, e_vol):
    resid, tot = 0.0, 0.0
    for r in range(de.shape[1]):
        d = onp.asarray(de[:, r]); e = onp.asarray(e_vol[:, r])
        X = onp.stack([onp.ones_like(e), e], axis=1)
        coef, *_ = onp.linalg.lstsq(X, d, rcond=None)
        res = d - X @ coef
        resid += float(res @ res); tot += float(d @ d)
    return (resid ** 0.5) / (tot ** 0.5 + 1e-30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sc", required=True)
    ap.add_argument("--lengths", required=True)
    ap.add_argument("--ts1", required=True)
    ap.add_argument("--params", required=True)
    ap.add_argument("--out", default="runs/wb_gradflow")
    ap.add_argument("--dt", type=float, default=2.0)
    ap.add_argument("--dt-train", type=float, default=100.0)
    ap.add_argument("--v-c", type=float, default=5.0)
    ap.add_argument("--d-model", type=int, default=16)
    ap.add_argument("--win", type=int, default=64)
    ap.add_argument("--burnin", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    C, hp0, params, N, d, emp_raw = make_setup(args)
    key = jax.random.PRNGKey(args.seed)
    _, kk = jax.random.split(key)
    carry0 = (np.zeros((1, N, N, d, d)), np.zeros((N, d)), np.ones((N, 1)),
              np.zeros((N, d)))
    G = (float(np.exp(params["log_G"])) if "log_G" in params
         else float(params["G_rel"]))
    out = {"params": args.params, "G": G, "burnin": args.burnin,
           "win": args.win, "dt_train": args.dt_train}

    # ---- endogenous: loss, stages, total dL/dG
    def f_end(g):
        p = dict(params)
        if "log_G" in p:
            p["log_G"] = np.log(g)
        else:
            p["G_rel"] = g
        return window(p, carry0, kk, C, hp0, args.win, args.dt_train,
                      args.burnin)[0]

    loss0 = float(f_end(np.asarray(G)))
    dLdG = float(jax.grad(f_end)(np.asarray(G)))
    _, st = window(params, carry0, kk, C, hp0, args.win, args.dt_train, args.burnin)
    aff_true = jax.lax.stop_gradient(st["aff"])
    e_vol = jax.lax.stop_gradient(st["e_vol"])
    out["loss"] = loss0
    out["dLdG_total"] = dLdG

    # ---- reverse demand at the injected drive: dL/daff (exact, per step/region)
    def f_inject(a):
        return window(params, carry0, kk, C, hp0, args.win, args.dt_train,
                      args.burnin, inject=a)[0]

    g_aff = jax.grad(f_inject)(aff_true)                 # (T,N,D)
    out["dLdaff_norm"] = float(np.sqrt(np.sum(g_aff ** 2)))

    # ---- forward supply: daff/dG (endogenous), incl. the recursive part
    def aff_of(g):
        p = dict(params)
        if "log_G" in p:
            p["log_G"] = np.log(g)
        else:
            p["G_rel"] = g
        return window(p, carry0, kk, C, hp0, args.win, args.dt_train,
                      args.burnin)[1]["aff"]

    daff = jax.jvp(aff_of, (np.asarray(G),), (np.asarray(1.0),))[1]
    out["daffdG_norm"] = float(np.sqrt(np.sum(daff ** 2)))
    contrib = onp.asarray(g_aff) * onp.asarray(daff)      # (T,N,D)
    per_step = contrib.sum(axis=(1, 2))
    out["dLdG_via_aff"] = float(per_step.sum())
    ts = max(1, per_step.size // 8)
    out["contrib_profile"] = [float(per_step[i:i + ts].sum())
                              for i in range(0, per_step.size, ts)]
    # direct (trajectory frozen: daff/dG = C@y_prev) vs recursive
    y_prev_seq = onp.asarray(jax.lax.stop_gradient(st["y_coupled"]))
    ys = onp.concatenate([onp.zeros((1, N, d), dtype="f4"), y_prev_seq[:-1]], 0)
    C_y = onp.einsum("ij,tjd->tid", onp.asarray(C), ys)
    direct = float((onp.asarray(g_aff) * C_y).sum())
    out["direct_part"] = direct
    out["recursive_part"] = float(per_step.sum()) - direct

    # ---- stage log-sensitivities (supply side), incl. the readout
    W_K = onp.asarray(params["W_K"]); W_V = onp.asarray(params["W_V"])

    def norm_of(fn):
        return lambda g: np.sqrt(np.sum(fn(g) ** 2))

    def stage_fns(g):
        p = dict(params)
        if "log_G" in p:
            p["log_G"] = np.log(g)
        else:
            p["G_rel"] = g
        return window(p, carry0, kk, C, hp0, args.win, args.dt_train,
                      args.burnin)[1]

    def nrm(arr):
        return np.sqrt(np.sum(arr ** 2))

    gg = np.asarray(G)
    sens = {}
    for name, fn in (
            ("aff", lambda g: stage_fns(g)["aff"]),
            ("x", lambda g: stage_fns(g)["x"]),
            ("e", lambda g: stage_fns(g)["e"]),
            ("e_vol", lambda g: stage_fns(g)["e_vol"]),
            ("y_coupled", lambda g: stage_fns(g)["y_coupled"]),
            ("M", lambda g: stage_fns(g)["M"])):
        try:
            sens[name] = log_sens(fn, gg)
        except Exception as exc:  # noqa: BLE001
            sens[name] = f"error: {type(exc).__name__}"
    out["log_sens_stages"] = sens

    # ---- null-space of the per-region z-score (de vs span{1, e_r})
    de = jax.jvp(lambda g: stage_fns(g)["e_vol"], (gg,), (np.asarray(1.0),))[1]
    out["e_vol_null_frac"] = null_frac(de, onp.asarray(e_vol))
    out["e_vol_rel_sens"] = (float(np.sqrt(np.sum(de ** 2)))
                             / (float(np.sqrt(np.sum(e_vol ** 2))) + 1e-30))

    # ---- sanity: loss with the z-score removed (does the normaliser hide G?)
    def f_raw(g):
        p = dict(params)
        if "log_G" in p:
            p["log_G"] = np.log(g)
        else:
            p["G_rel"] = g
        return window_raw(p, carry0, kk, C, hp0, args.win, args.dt_train,
                          args.burnin, emp_raw)

    out["dLdG_no_zscore"] = float(jax.grad(f_raw)(np.asarray(G)))

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "gradflow.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1)[:1800], flush=True)
    print("WB_GRADFLOW_DONE", flush=True)


def window_raw(params, carry, keys, C, hp0, win, dt_ms, burnin, emp_raw):
    """Same as window() but the loss is the un-z-scored MSE (control condition)."""
    import functools
    # reuse window() internals by monkey-patching the loss via a copy
    _, st = window(params, carry, keys, C, hp0, win, dt_ms, burnin)
    return np.mean((st["e_vol"] - emp_raw) ** 2)


if __name__ == "__main__":
    main()