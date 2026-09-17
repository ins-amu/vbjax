"""Stage-by-stage autograd attribution for d(loss)/dG_rel.

G enters at exactly one place per step:  aff_t = G * (C @ y_{t-1}).
Everything downstream (x -> k,v -> dM -> surprise e -> e_vol -> loss) depends on
G only through aff_t and the recurrent state, so

    dL/dG = sum_t < dL/daff_t , daff_t/dG >

and both factors are measurable exactly:
  * reverse demand dL/daff_t -- aff is injected as an explicit scan input, so its
    cotangent is exact per step and per region;
  * forward supply daff_t/dG -- jax.jvp of the endogenous drive.
Summing the product over steps reproduces dL/dG (checked), so the decomposition
is exact rather than illustrative.

Also reported: the per-step contribution profile, the direct (trajectory frozen:
daff/dG = C@y_prev) vs recursive split, log-sensitivity dlog||stage||/dlogG per
stage, the null-space fraction of d(e_vol)/dG inside span{1, e_vol_r} (the kernel
of the per-region z-score), and the same dL/dG with the z-score removed as a
control condition.

CPU is enough (no GPU). Run it as an sbatch job without --gres.
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
    emp_raw = onp.asarray(ts1[:n_win * win].reshape(n_win, win, N)[0])
    emp_z = onp.asarray(zscore(ts1)[:n_win * win].reshape(n_win, win, N)[0])
    return C, hp0, params, N, d, emp_raw, emp_z


def make_window(C, hp0, emp_raw, emp_z, win, dt_ms):
    """f(params, carry, keys, burnin, inject=None, zscore_loss=True)
       -> (loss, stages). inject=None: endogenous drive (G differentiates),
       else the drive is an explicit input array (T_int, N, D)."""
    N = C.shape[0]

    def f(params, carry, keys, burnin, inject=None, zscore_loss=True):
        D_ = params["W_q"].shape[-1]
        G = np.exp(params["log_G"]) if "log_G" in params else params["G_rel"]
        tau_s = np.exp(params["log_tau"])
        sigma = np.exp(params["log_sigma"])
        noise_tc = np.exp(params["log_noise_tc"])
        dt_s = dt_ms / 1000.0
        u = dt_s / tau_s
        noise_a = np.exp(-dt_s / noise_tc)
        sq1 = np.sqrt(1.0 - noise_a ** 2 + 1e-12)
        mhsa = build_mhsa(hp0, jax.random.PRNGKey(0), params["W_q"],
                          params["W_y"], params["kv_scale"])
        hp_eff = hp0._replace(lam=u, tau=tau_s * 1000.0)
        steps_per_vol = int(round(2000.0 / dt_ms))
        T_int = win * steps_per_vol

        def step_core(M, n_prev, y_ema, y_prev, key_t, aff_t):
            state = NetworkState(M=M, history=None, step=0)
            eps = jax.random.normal(key_t, (N, D_))
            noise = noise_a * n_prev + sigma * sq1 * eps
            x = (aff_t + noise)[None]
            state, y, e = mhsa_step(mhsa, state, x, hp_eff)
            y = y[0]
            e = e[0].mean(-1)
            rms = np.sqrt(np.mean(y ** 2, axis=-1, keepdims=True))
            y_ema = 0.999 * y_ema + 0.001 * rms
            y_coupled = y / (y_ema + 1e-8)
            return (state.M, noise, y_ema, y_coupled), (e, x[0], aff_t, y_coupled)

        def step_end(carry_n, key_t):
            M, n_prev, y_ema, y_prev = carry_n
            aff_t = G * np.einsum("ij,jd->id", C, y_prev)
            return step_core(M, n_prev, y_ema, y_prev, key_t, aff_t)

        def step_inj(carry_n, xs_t):
            key_t, aff_t = xs_t
            M, n_prev, y_ema, y_prev = carry_n
            return step_core(M, n_prev, y_ema, y_prev, key_t, aff_t)

        keys_w = jax.random.split(keys, T_int)
        carry = jax.lax.stop_gradient(carry)
        if burnin > 0:
            bk = jax.random.split(jax.random.fold_in(keys, 77), burnin)

            def burn_step(c, key_t):
                M, n_prev, y_ema, y_prev = c
                aff_t = G * np.einsum("ij,jd->id", C, y_prev)
                out_carry, _ = step_core(M, n_prev, y_ema, y_prev, key_t, aff_t)
                return out_carry, None       # scan body must return a pair

            carry = jax.lax.stop_gradient(jax.lax.scan(burn_step, carry, bk)[0])

        if inject is None:
            carry, (e_seq, x_seq, aff_seq, y_seq) = jax.lax.scan(
                step_end, carry, keys_w)
        else:
            carry, (e_seq, x_seq, aff_seq, y_seq) = jax.lax.scan(
                step_inj, carry, (keys_w, inject))

        e_vol = e_seq.reshape(win, steps_per_vol, N).mean(axis=1)
        tgt = emp_z if zscore_loss else emp_raw
        pred = zscore(e_vol) if zscore_loss else e_vol
        loss = np.mean((pred - tgt) ** 2)
        stages = {"aff": aff_seq, "x": x_seq, "e": e_seq, "e_vol": e_vol,
                  "y": y_seq, "M": carry[0]}
        return loss, stages

    return f


def log_sens(fn, g):
    val, tang = jax.jvp(fn, (g,), (np.asarray(1.0),))
    nv = float(np.sqrt(np.sum(val ** 2)) + 1e-30)
    nt = float(np.sqrt(np.sum(tang ** 2)) + 1e-30)
    return nt / nv * abs(float(g))


def null_frac(de, e_vol):
    resid, tot = 0.0, 0.0
    for r in range(de.shape[1]):
        dd = onp.asarray(de[:, r]); ee = onp.asarray(e_vol[:, r])
        X = onp.stack([onp.ones_like(ee), ee], axis=1)
        coef, *_ = onp.linalg.lstsq(X, dd, rcond=None)
        res = dd - X @ coef
        resid += float(res @ res); tot += float(dd @ dd)
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

    C, hp0, params, N, d, emp_raw, emp_z = make_setup(args)
    f = make_window(C, hp0, emp_raw, emp_z, args.win, args.dt_train)
    key = jax.random.PRNGKey(args.seed)
    _, kk = jax.random.split(key)
    carry0 = (np.zeros((1, N, N, d, d)), np.zeros((N, d)), np.ones((N, 1)),
              np.zeros((N, d)))
    G = (float(np.exp(params["log_G"])) if "log_G" in params
         else float(params["G_rel"]))
    gv = np.asarray(G)

    def with_g(g):
        p = dict(params)
        if "log_G" in p:
            p["log_G"] = np.log(g)
        else:
            p["G_rel"] = g
        return p

    out = {"params": args.params, "G": G, "burnin": args.burnin,
           "win": args.win, "dt_train": args.dt_train, "N": N}

    loss0, st = f(params, carry0, kk, args.burnin)
    aff_true = jax.lax.stop_gradient(st["aff"])
    e_vol = jax.lax.stop_gradient(st["e_vol"])
    y_seq = jax.lax.stop_gradient(st["y"])
    out["loss"] = float(loss0)

    def fz(g):
        return f(with_g(g), carry0, kk, args.burnin)[0]

    out["dLdG_total"] = float(jax.grad(fz)(gv))

    def fi(a):
        return f(params, carry0, kk, args.burnin, inject=a)[0]

    g_aff = jax.grad(fi)(aff_true)          # reverse demand, per step/region

    # ONE forward-mode pass yields every stage tangent: the drive supply
    # (daff/dG), the readout (d(e_vol)/dG) and the stage log-sensitivities.
    def stages_all(g):
        st = f(with_g(g), carry0, kk, args.burnin)[1]
        return (st["aff"], st["x"], st["e"], st["e_vol"], st["y"], st["M"])

    val_s, tang_s = jax.jvp(stages_all, (gv,), (np.asarray(1.0),))
    daff = onp.asarray(tang_s[0])                          # forward supply
    out["dLdaff_norm"] = float(np.sqrt(np.sum(g_aff ** 2)))
    out["daffdG_norm"] = float(np.sqrt(np.sum(daff ** 2)))
    contrib = onp.asarray(g_aff) * onp.asarray(daff)
    per_step = contrib.sum(axis=(1, 2))
    out["dLdG_via_aff"] = float(per_step.sum())
    ns = max(1, per_step.size // 8)
    out["contrib_profile"] = [float(per_step[i:i + ns].sum())
                              for i in range(0, per_step.size, ns)]

    ys = onp.asarray(y_seq)
    y_prev_seq = onp.concatenate([onp.zeros((1, N, d), dtype="f4"), ys[:-1]], 0)
    C_y = onp.einsum("ij,tjd->tid", onp.asarray(C), y_prev_seq)
    direct = float((onp.asarray(g_aff) * C_y).sum())
    out["direct_part"] = direct
    out["recursive_part"] = float(per_step.sum()) - direct

    sens = {}
    for name, vv, tt in zip(("aff", "x", "e", "e_vol", "y", "M"),
                            val_s, tang_s):
        nv = float(np.sqrt(np.sum(vv ** 2)) + 1e-30)
        nt = float(np.sqrt(np.sum(tt ** 2)) + 1e-30)
        sens[name] = nt / nv * abs(float(gv))          # dlog||stage||/dlogG
    out["log_sens_stages"] = sens

    de = tang_s[3]                                          # d(e_vol)/dG
    out["e_vol_null_frac"] = null_frac(de, onp.asarray(e_vol))
    out["e_vol_rel_sens"] = (float(np.sqrt(np.sum(de ** 2)))
                             / (float(np.sqrt(np.sum(e_vol ** 2))) + 1e-30))

    def f_raw(g):
        return f(with_g(g), carry0, kk, args.burnin, zscore_loss=False)[0]

    out["dLdG_no_zscore"] = float(jax.grad(f_raw)(gv))

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "gradflow.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1)[:2000], flush=True)
    print("WB_GRADFLOW_DONE", flush=True)


if __name__ == "__main__":
    main()