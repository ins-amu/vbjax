"""Balloon (Windkessel) initial-condition study for the whole-brain fit.

The eval forward starts the balloon at s=0, f=1, v=1, q=1 and integrates the
whole 600 s simulation, then computes FC over the WHOLE trace. The fixed point
under a positive drive is s=0, f=1+tau_f*x, v=f**(1/alpha) -- quite different
from the shipped initial condition, and the v^3.1 nonlinearity makes the
approach stiff. Any startup transient therefore enters fc_match.

Variants measured here on a fixed (already simulated) e trace:
  default  : shipped ICs (0,1,1,1), FC over the whole trace
  steady   : ICs from integrating the balloon to its fixed point under the mean drive
  discard  : shipped ICs, FC computed only after dropping the first --discard-s seconds
  steady+d : steady ICs AND the same discard
  rescale  : shipped ICs but the balloon drive is the per-region [0.2,1] affine
             rescale (wholebrain_fc's convention) instead of zscore(e)

Run inside the GPU container from the tree root:
  python -m vbjax.app.wb_bold_ic --sc SC.npy --lengths LEN.npy --ts1 TS.npy --ts2 TS2.npy \
      --params runs_persist/wb_train/params.pkl --out runs/wb_ic
"""
import argparse, json, os, pickle
import numpy as onp
import jax
import jax.numpy as np

from vbjax import make_bold, bold_default_theta, make_offline
from vbjax.ct_mhsa import Hyperparameters, NetworkState, init_ct_mhsa
from vbjax.app.wholebrain_fc import load_connectome, fc_from_ts, fc_match
from vbjax.app.wb_train import zscore, build_mhsa


def balloon(src, dt_ms, tr_s, ic=None):
    """Balloon on (T,N) jax drive -> BOLD sampled at TR. ic: (4,) or (4,N)."""
    T, N = src.shape
    every = int(round(tr_s * 1000.0 / dt_ms))
    W = T // every
    windowed = src[:W * every].reshape(W, every, N)
    buf, step, sample = make_bold(shape=(N,), dt=dt_ms / 1000.0,
                                  p=bold_default_theta)
    if ic is not None:
        ic = np.asarray(ic, dtype=np.float32)
        ic = ic.reshape(4, 1) if ic.size == 4 else ic.reshape(4, N)
        buf = np.broadcast_to(ic, (4, N))
    offline = make_offline(step_fn=step, sample_fn=sample)
    _, bold = jax.lax.scan(offline, buf, windowed)
    return bold


def fc_of_ic(ic, src, dt_ms, tr_s, tsfc, iu):
    """fc_match between the balloon output (from ICs `ic`) and an empirical FC."""
    bold = balloon(src, dt_ms, tr_s, ic)
    z = (bold - bold.mean(0)) / (bold.std(0) + 1e-8)
    fc = z.T @ z / bold.shape[0]
    fu, tu = fc[iu], tsfc[iu]
    return (np.mean(fu * tu) - np.mean(fu) * np.mean(tu)) / (
        np.std(fu) * np.std(tu) + 1e-8)


def optimise_ic(src, tsfc, dt_ms, tr_s, steps, lr=0.05, per_region=False):
    """Gradient-ascent the balloon initial conditions on fc_match (the shipped ICs
    are only the fixed point of the mean drive; the fluctuating part of the drive
    leaves room to improve the startup)."""
    N = src.shape[1]
    iu = np.triu_indices(N, 1)
    base = np.asarray([0.0, 1.0, 1.0, 1.0], dtype=np.float32)
    ic = (np.tile(base.reshape(4, 1), (1, N)) if per_region
          else base.reshape(4, 1))

    @jax.jit
    def obj(ic_):
        return fc_of_ic(ic_, src, dt_ms, tr_s, tsfc, iu)

    m = np.zeros_like(ic)
    v = np.zeros_like(ic)
    hist = []
    for t in range(1, steps + 1):
        val, g = jax.value_and_grad(obj)(ic)
        g = np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g ** 2
        ic = ic + lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
        ic = ic.at[2].set(np.clip(ic[2], 1e-3, None))   # v > 0 (v^3.1)
        ic = ic.at[3].set(np.clip(ic[3], 1e-3, None))   # q > 0
        if t % 10 == 0 or t == 1:
            hist.append([t, float(val)])
    return ic, hist


def fixed_point_ic(drive_mean, dt_ms=2.0, t_conv_s=400.0):
    """ICs obtained by integrating the balloon with a constant drive to convergence."""
    N = drive_mean.shape[0]
    n = int(t_conv_s * 1000.0 / dt_ms)
    xs = np.broadcast_to(drive_mean, (n, N))
    buf, step, sample = make_bold(shape=(N,), dt=dt_ms / 1000.0,
                                  p=bold_default_theta)
    final = jax.lax.scan(lambda b, x: (step(b, x), None), buf, xs)[0]
    return final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sc", required=True)
    ap.add_argument("--lengths", required=True)
    ap.add_argument("--ts1", required=True)
    ap.add_argument("--ts2", required=True)
    ap.add_argument("--params", required=True)
    ap.add_argument("--out", default="runs/wb_ic")
    ap.add_argument("--dt", type=float, default=2.0)
    ap.add_argument("--tr", type=float, default=2.0)
    ap.add_argument("--v-c", type=float, default=5.0)
    ap.add_argument("--d-model", type=int, default=16)
    ap.add_argument("--sim-s", type=float, default=600.0)
    ap.add_argument("--discard-s", type=float, default=60.0)
    ap.add_argument("--opt-steps", type=int, default=40,
                    help="Adam steps for gradient-descending the balloon ICs on fc_match")
    ap.add_argument("--opt-lr", type=float, default=0.05)
    ap.add_argument("--per-region", action="store_true",
                    help="optimise one IC set per region instead of one shared set")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

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
    G = (float(np.exp(params["log_G"])) if "log_G" in params
         else float(params["G_rel"]))
    tau_s = float(np.exp(params["log_tau"]))
    sigma = float(np.exp(params["log_sigma"]))
    noise_tc = float(np.exp(params["log_noise_tc"]))
    ts1 = onp.load(args.ts1)
    ts2 = onp.load(args.ts2)
    fc1, fc2 = fc_from_ts(ts1), fc_from_ts(ts2)

    from vbjax.app.wholebrain_fc import simulate
    T = int(args.sim_s * 1000 / args.dt)

    out = {"params": args.params, "G": G, "tau_s": tau_s, "sigma": sigma,
           "sim_s": args.sim_s, "discard_s": args.discard_s, "variants": {}}

    e_all = {}
    for name, seed, tsfc in (("subj1", args.seed, fc1), ("subj2", args.seed + 1, fc2)):
        hp = hp0._replace(lam=args.dt / (tau_s * 1000.0), tau=tau_s * 1000.0)
        mhsa = build_mhsa(hp, jax.random.PRNGKey(seed), params["W_q"],
                          params["W_y"], float(params["kv_scale"]))
        e, _ = simulate(mhsa, np.asarray(C), jax.numpy.asarray(lags), L, hp,
                        G, sigma, T, jax.random.PRNGKey(seed), noise_tc,
                        args.dt, "e")
        e = jax.lax.stop_gradient(e)
        e_all[name] = (e, tsfc)

    for name, (e, tsfc) in e_all.items():
        src_z = zscore(e)
        mean_drive = np.mean(src_z, axis=0)
        ic_fp = fixed_point_ic(mean_drive, args.dt)
        lo = src_z.min(0, keepdims=True)
        hi = src_z.max(0, keepdims=True)
        src_rs = 0.2 + 0.8 * (src_z - lo) / (hi - lo + 1e-8)

        drop = int(args.discard_s / args.tr)
        res = {}
        for tag, s, ic, d0 in (
                ("default", src_z, None, 0),
                ("steady", src_z, ic_fp, 0),
                ("discard", src_z, None, drop),
                ("steady+discard", src_z, ic_fp, drop),
                ("rescale_default", src_rs, None, 0),
                ("rescale_steady", src_rs, ic_fp, 0)):
            try:
                bold = onp.asarray(balloon(s, args.dt, args.tr, ic))
                if not onp.isfinite(bold).all():
                    res[tag] = {"fc": None, "note": "non-finite BOLD"}
                    continue
                b = bold[d0:]
                if b.shape[0] < 10:
                    res[tag] = {"fc": None, "note": "too few TRs after discard"}
                    continue
                res[tag] = {"fc": fc_match(fc_from_ts(b), tsfc), "n_tr": int(b.shape[0])}
            except Exception as exc:  # noqa: BLE001
                res[tag] = {"fc": None, "note": f"{type(exc).__name__}: {exc}"}
        # transient size: spread of the first few TRs relative to the rest
        bold0 = onp.asarray(balloon(src_z, args.dt, args.tr, None))
        res["transient"] = {
            "first_10_tr_std": float(onp.std(bold0[:10])),
            "rest_std": float(onp.std(bold0[10:])),
            "ic_fixed_point": [float(x) for x in onp.asarray(ic_fp).mean(-1)],
        }
        out["variants"][name] = res
        print(f"[{name}] " + " ".join(
            f"{k}={v.get('fc'):.4f}" if isinstance(v, dict) and v.get("fc") is not None
            else f"{k}=n/a" for k, v in res.items() if k != "transient"), flush=True)
        print(f"[{name}] IC fixed point (mean over regions) = "
              f"{res['transient']['ic_fixed_point']}", flush=True)

    if args.opt_steps > 0:
        iu = np.triu_indices(N, 1)
        ic_opt, hist = optimise_ic(zscore(e_all["subj1"][0]), fc1, args.dt,
                                   args.tr, args.opt_steps, args.opt_lr,
                                   args.per_region)
        o1 = float(fc_of_ic(ic_opt, zscore(e_all["subj1"][0]), args.dt, args.tr,
                            fc1, iu))
        o2 = float(fc_of_ic(ic_opt, zscore(e_all["subj2"][0]), args.dt, args.tr,
                            fc2, iu))
        d1 = out["variants"]["subj1"]["default"]["fc"]
        d2 = out["variants"]["subj2"]["default"]["fc"]
        out["opt_ic"] = {
            "per_region": bool(args.per_region),
            "ic_mean_over_regions": [float(x) for x in onp.asarray(ic_opt).mean(-1)],
            "fc1_fit": o1, "fc2_holdout": o2, "hist": hist}
        print(f"[opt_ic] fit on subj1: fc1={o1:.4f} (was {d1:.4f}), "
              f"fc2_holdout={o2:.4f} (was {d2:.4f})", flush=True)
        print(f"[opt_ic] ICs (mean over regions) = "
              f"{out['opt_ic']['ic_mean_over_regions']}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "bold_ic.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("WB_BOLD_IC_DONE", flush=True)


if __name__ == "__main__":
    main()