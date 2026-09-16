"""Balloon (Windkessel) initial-condition optimisation for the whole-brain fit.

The eval forward starts the balloon at s=0, f=1, v=1, q=1 and integrates the
whole 600 s simulation, then computes FC over the WHOLE trace.  The fixed point
under the mean (z-scored) drive equals those shipped ICs, so "steady" ICs change
nothing; the fluctuating part of the drive, however, leaves room to pick a
startup that happens to score better on fc_match.  This file measures that.

Variants measured here on a fixed (already simulated) e trace:
  default  : shipped ICs (0,1,1,1), FC over the whole trace
  steady   : ICs from integrating the balloon to its fixed point under the mean drive
  discard  : shipped ICs, FC computed only after dropping the first --discard-s seconds
  steady+d : steady ICs AND the same discard
  rescale  : shipped ICs but the balloon drive is the per-region [0.2,1] affine
             rescale (wholebrain_fc's convention) instead of zscore(e)

IC optimisation protocol
------------------------
  * Adam ascent on fc_match, lr --opt-lr (default 0.01), --opt-steps (default 220),
    with a running best-by-objective.
  * the FINAL IC is the mean of the iterates over the last 25% of steps; the
    best-by-objective single iterate is also reported.
  * --opt-restarts (default 3) random restarts around the shipped IC; all are
    reported, the best-by-objective one (of its averaged iterate) is chosen.
  * symmetry check: optimise on subj1 -> report fc1 (fit) + fc2 (holdout), then
    optimise on subj2 -> report fc2 (fit) + fc1 (holdout).  A real improvement
    has to show on the held-out subject in BOTH directions.
  * the objective history is printed every --opt-log-every steps (default 10).
  * with --per-region one IC set per region is optimised; the across-region
    spread of the chosen ICs is reported to show what the shared-IC restriction costs.

Efficiency: the balloon is contractive (time constants 0.41-0.98 s), so the BOLD
is independent of the IC after a few seconds.  The optimiser therefore
differentiates only through the first --opt-warm-s seconds and appends a frozen
tail computed once with the shipped IC, while still scoring the FULL 600 s trace.
The frozen-tail approximation is verified at the end against an exact full scan,
and every reported fc number is the exact full-trace value.

Run inside the GPU container from the tree root:
  python -m vbjax.app.wb_bold_ic --sc SC.npy --lengths LEN.npy --ts1 TS.npy --ts2 TS2.npy \
      --params runs_persist/wb_train/params.pkl --out runs/wb_ic2
"""
import argparse, json, os, pickle, time
import numpy as onp
import jax
import jax.numpy as np

from vbjax import make_bold, bold_default_theta, make_offline
from vbjax.ct_mhsa import Hyperparameters
from vbjax.app.wholebrain_fc import load_connectome, fc_from_ts, fc_match, simulate
from vbjax.app.wb_train import zscore, build_mhsa


IC_NAMES = ("s", "f", "v", "q")
DEFAULT_IC = onp.asarray([0.0, 1.0, 1.0, 1.0], dtype="f4")
# restart perturbations around the shipped IC, one scale per balloon component
RESTART_SCALE = onp.asarray([0.05, 0.05, 0.05, 0.05], dtype="f4")


# --------------------------------------------------------------------------- #
# balloon forward
# --------------------------------------------------------------------------- #
def _windows(src, dt_ms, tr_s):
    """(W, every, N) windowed drive plus a fresh BOLD monitor."""
    T, N = src.shape
    every = int(round(tr_s * 1000.0 / dt_ms))
    W = T // every
    windowed = src[:W * every].reshape(W, every, N)
    buf, step, sample = make_bold(shape=(N,), dt=dt_ms / 1000.0,
                                  p=bold_default_theta)
    return windowed, buf, step, sample, W, every, N


def _as_ic(ic, N):
    ic = np.asarray(ic, dtype=np.float32)
    ic = ic.reshape(4, 1) if ic.size == 4 else ic.reshape(4, N)
    return np.broadcast_to(ic, (4, N))


def balloon(src, dt_ms, tr_s, ic=None):
    """Exact balloon over the whole trace -> BOLD (W, N). ic: (4,) or (4,N)."""
    windowed, buf, step, sample, W, every, N = _windows(src, dt_ms, tr_s)
    if ic is not None:
        buf = _as_ic(ic, N)
    offline = make_offline(step_fn=step, sample_fn=sample)
    _, bold = jax.lax.scan(offline, buf, windowed)
    return bold


def _corr_triu(fc, tsfc, iu):
    fu, tu = fc[iu], tsfc[iu]
    return (np.mean(fu * tu) - np.mean(fu) * np.mean(tu)) / (
        np.std(fu) * np.std(tu) + 1e-8)


def _fc_of_bold(bold, tsfc, iu):
    z = (bold - bold.mean(0)) / (bold.std(0) + 1e-8)
    return _corr_triu(z.T @ z / bold.shape[0], tsfc, iu)


def fc_of_ic(ic, src, dt_ms, tr_s, tsfc, iu):
    """fc_match between the balloon output (from ICs `ic`) and an empirical FC."""
    return _fc_of_bold(balloon(src, dt_ms, tr_s, ic), tsfc, iu)


def exact_fc(ic, src, dt_ms, tr_s, tsfc):
    """Exact full-trace fc_match (numpy); None if the balloon produced non-finite BOLD."""
    bold = onp.asarray(balloon(src, dt_ms, tr_s, ic))
    if not onp.isfinite(bold).all():
        return None
    return float(fc_match(fc_from_ts(bold), tsfc))


# --------------------------------------------------------------------------- #
# differentiable objective (head from the IC + frozen, IC-independent tail)
# --------------------------------------------------------------------------- #
def make_ic_objective(src, tsfc, dt_ms, tr_s, tail_bold, warm_windows):
    """fc_match over the whole trace, differentiated through the first
    `warm_windows` TRs only.  The remaining TRs are IC-independent (contractive
    balloon) and are supplied frozen, so the score is unchanged while the
    backprop-through-the-scan cost drops by (W - warm_windows)/W."""
    windowed, buf, step, sample, W, every, N = _windows(src, dt_ms, tr_s)
    iu = np.triu_indices(N, 1)
    head_src = windowed[:warm_windows]
    offline = make_offline(step_fn=step, sample_fn=sample)
    tail = np.asarray(tail_bold, dtype=np.float32)

    def obj(ic):
        _, head = jax.lax.scan(offline, _as_ic(ic, N), head_src)
        return _fc_of_bold(np.concatenate([head, tail], axis=0), tsfc, iu)
    return obj, iu


def optimise_ic(src, tsfc, dt_ms, tr_s, steps, lr=0.01, per_region=False, *,
                restarts=3, seed=42, warm_windows=None, tail_bold=None,
                log_every=10, time_budget_s=None):
    """Adam-ascent the balloon ICs on fc_match, with restarts.

    Returns the chosen IC (mean over the last 25% of steps of the best restart),
    the best-by-objective iterate, the across-restart report and the given
    objective histories.  Caller supplies the frozen tail / warm window."""
    windowed, buf, step, sample, W, every, N = _windows(src, dt_ms, tr_s)
    if warm_windows is None or tail_bold is None:
        warm_windows, tail_bold = W, np.zeros((0, N), dtype="f4")
    obj, iu = make_ic_objective(src, tsfc, dt_ms, tr_s, tail_bold, warm_windows)

    base = (np.broadcast_to(DEFAULT_IC.reshape(4, 1), (4, N)) if per_region
            else DEFAULT_IC.reshape(4, 1))
    key = jax.random.PRNGKey(seed)
    noise = jax.random.normal(key, (restarts,) + tuple(base.shape))
    noise = noise * np.asarray(RESTART_SCALE).reshape(4, 1)
    ics = base[None] + noise

    vobj = jax.jit(jax.vmap(jax.value_and_grad(obj)))
    m = np.zeros_like(ics)
    v = np.zeros_like(ics)
    best_val = np.full((restarts,), -np.inf)
    best_ic = ics
    hist_steps, hist_vals = [], []
    win = max(1, int(round(0.25 * steps)))
    ics_hist = []
    t0 = time.time()
    done = 0
    for t in range(1, steps + 1):
        vals, grads = vobj(ics)
        grads = np.nan_to_num(grads, nan=0.0, posinf=0.0, neginf=0.0)
        vals = np.nan_to_num(vals, nan=-np.inf)
        m = 0.9 * m + 0.1 * grads
        v = 0.999 * v + 0.001 * grads ** 2
        ics = ics + lr * (m / (1 - 0.9 ** t)) / (
            np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
        ics = ics.at[:, 2].set(np.clip(ics[:, 2], 1e-3, None))   # v > 0 (v**3.125)
        ics = ics.at[:, 3].set(np.clip(ics[:, 3], 1e-3, None))   # q > 0
        improved = vals > best_val
        best_val = np.where(improved, vals, best_val)
        best_ic = np.where(improved.reshape(-1, 1, 1), ics, best_ic)
        ics_hist.append(ics)
        if t == 1 or t % log_every == 0:
            hist_steps.append(t)
            hist_vals.append([float(x) for x in vals])
        done = t
        if time_budget_s and (time.time() - t0) > time_budget_s:
            break

    # final IC = mean of the iterates over the last 25% of the (completed) steps
    n_tail = max(1, min(win, len(ics_hist)))
    mean_ic = np.mean(np.stack(ics_hist[-n_tail:], 0), 0)
    obj_b = jax.jit(jax.vmap(obj))
    val_mean = [float(x) for x in obj_b(mean_ic)]
    val_best = [float(x) for x in obj_b(best_ic)]
    pick = int(onp.argmax(onp.asarray(val_mean)))
    chosen = onp.asarray(mean_ic[pick])
    report = {
        "per_region": bool(per_region),
        "steps_requested": int(steps), "steps_done": int(done), "lr": float(lr),
        "restarts": int(restarts), "seed": int(seed),
        "avg_window": int(n_tail),
        "warm_windows": int(warm_windows), "total_windows": int(W),
        "wall_s": round(time.time() - t0, 1),
        "chosen_restart": pick,
        "restart_report": [
            {"restart": r,
             "ic_avg25": [float(x) for x in onp.asarray(mean_ic[r]).mean(-1)],
             "ic_best_step": [float(x) for x in onp.asarray(best_ic[r]).mean(-1)],
             "obj_avg25": val_mean[r], "obj_best_step": val_best[r]} for r in range(restarts)],
        "history": {"step": hist_steps, "obj": hist_vals},
        "ic_chosen": chosen,
        "ic_chosen_best_step": onp.asarray(best_ic[pick]),
        "ic_chosen_avg25": chosen,
    }
    if per_region:
        report["ic_spread_std_over_regions"] = {
            IC_NAMES[i]: float(onp.std(chosen[i])) for i in range(4)}
        report["ic_mean_over_regions"] = [float(x) for x in onp.asarray(chosen).mean(-1)]
    return report


def fixed_point_ic(drive_mean, dt_ms=2.0, t_conv_s=400.0):
    """ICs obtained by integrating the balloon with a constant drive to convergence."""
    N = drive_mean.shape[0]
    n = int(t_conv_s * 1000.0 / dt_ms)
    xs = np.broadcast_to(drive_mean, (n, N))
    buf, step, sample = make_bold(shape=(N,), dt=dt_ms / 1000.0,
                                  p=bold_default_theta)
    final = jax.lax.scan(lambda b, x: (step(b, x), None), buf, xs)[0]
    # v**recip_alpha (recip_alpha = 3.125) is NaN for v <= 0, and the integrated
    # fixed point can land a hair below zero per region; clip to the physical range
    final = final.at[2].set(np.maximum(final[2], 1e-3))
    final = final.at[3].set(np.maximum(final[3], 1e-3))
    return final


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def _f(x):
    return "n/a" if x is None else f"{x:.4f}"


def _dump(out, outdir):
    """Write results so far (progressive, so a wall-clock kill loses nothing)."""
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "bold_ic.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sc", required=True)
    ap.add_argument("--lengths", required=True)
    ap.add_argument("--ts1", required=True)
    ap.add_argument("--ts2", required=True)
    ap.add_argument("--params", required=True)
    ap.add_argument("--out", default="runs/wb_ic2")
    ap.add_argument("--dt", type=float, default=2.0)
    ap.add_argument("--tr", type=float, default=2.0)
    ap.add_argument("--v-c", type=float, default=5.0)
    ap.add_argument("--d-model", type=int, default=16)
    ap.add_argument("--sim-s", type=float, default=600.0)
    ap.add_argument("--discard-s", type=float, default=60.0)
    ap.add_argument("--opt-steps", type=int, default=220,
                    help="Adam steps for ascending the balloon ICs on fc_match")
    ap.add_argument("--opt-lr", type=float, default=0.01)
    ap.add_argument("--opt-restarts", type=int, default=3)
    ap.add_argument("--opt-warm-s", type=float, default=60.0,
                    help="seconds of the trace differentiated through from the IC "
                         "(the rest is a frozen IC-independent tail). "
                         "0 => differentiate through the whole trace")
    ap.add_argument("--opt-log-every", type=int, default=10)
    ap.add_argument("--opt-budget-s", type=float, default=0.0,
                    help="wall-clock budget per optimisation; 0 = no limit")
    ap.add_argument("--per-region", action="store_true",
                    help="optimise one IC set per region instead of one shared set")
    ap.add_argument("--skip-variants", action="store_true",
                    help="skip the default/steady/discard/rescale variant study")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    t_start = time.time()

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
    T = int(args.sim_s * 1000 / args.dt)

    out = {"params": args.params, "G": G, "tau_s": tau_s, "sigma": sigma,
           "N": int(N), "sim_s": args.sim_s, "discard_s": args.discard_s,
           "dt": args.dt, "tr": args.tr,
           "default_ic": [float(x) for x in DEFAULT_IC],
           "opt": {"steps": args.opt_steps, "lr": args.opt_lr,
                   "restarts": args.opt_restarts, "warm_s": args.opt_warm_s,
                   "per_region": bool(args.per_region),
                   "budget_s": args.opt_budget_s},
           "variants": {}}

    # ---- simulate the drive for both subjects ---------------------------------
    e_all = {}
    for name, seed, tsfc in (("subj1", args.seed, fc1), ("subj2", args.seed + 1, fc2)):
        hp = hp0._replace(lam=args.dt / (tau_s * 1000.0), tau=tau_s * 1000.0)
        mhsa = build_mhsa(hp, jax.random.PRNGKey(seed), params["W_q"],
                          params["W_y"], float(params["kv_scale"]))
        e, _ = simulate(mhsa, np.asarray(C), jax.numpy.asarray(lags), L, hp,
                        G, sigma, T, jax.random.PRNGKey(seed), noise_tc,
                        args.dt, "e")
        e = jax.lax.stop_gradient(e)
        e_all[name] = {"src": zscore(e), "tsfc": tsfc, "name": name}
        print(f"[sim] {name}: e {e.shape} in {time.time() - t_start:.0f}s", flush=True)

    # ---- variant study (startup transient size / discard / rescale) -----------
    if not args.skip_variants:
        for name, sub in e_all.items():
            e, tsfc = sub["src"], sub["tsfc"]
            src_z = e
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
        _dump(out, args.out)

    # ---- IC optimisation -----------------------------------------------------
    if args.opt_steps > 0:
        W = int(args.sim_s / args.tr)
        k = W if args.opt_warm_s <= 0 else int(round(args.opt_warm_s / args.tr))
        k = max(1, min(k, W))
        for name, sub in e_all.items():
            bold_def = onp.asarray(balloon(sub["src"], args.dt, args.tr, None))
            sub["bold_default"] = bold_def
            sub["fc_default"] = float(fc_match(fc_from_ts(bold_def), sub["tsfc"]))
            sub["k_windows"] = k
            sub["tail_bold"] = np.asarray(bold_def[k:], dtype="f4")
            print(f"[{name}] default fc = {sub['fc_default']:.4f} "
                  f"({W} TRs, differentiating through first {k} = {k * args.tr:.0f}s)",
                  flush=True)

        opt = {}
        for fit_name, hold_name in (("subj1", "subj2"), ("subj2", "subj1")):
            fit, hold = e_all[fit_name], e_all[hold_name]
            t0 = time.time()
            rep = optimise_ic(fit["src"], fit["tsfc"], args.dt, args.tr,
                              args.opt_steps, args.opt_lr, args.per_region,
                              restarts=args.opt_restarts, seed=args.seed,
                              warm_windows=fit["k_windows"],
                              tail_bold=fit["tail_bold"],
                              log_every=args.opt_log_every,
                              time_budget_s=(args.opt_budget_s or None))
            print(f"[opt fit={fit_name}] {rep['steps_done']}/{rep['steps_requested']} "
                  f"steps in {time.time() - t0:.0f}s; restarts="
                  + " | ".join(f"avg25={r['obj_avg25']:.4f} best={r['obj_best_step']:.4f}"
                               for r in rep["restart_report"]), flush=True)
            for i, (s, vals) in enumerate(zip(rep["history"]["step"],
                                              rep["history"]["obj"])):
                print(f"[hist fit={fit_name} step {s}] "
                      + " ".join(f"r{j}={v:.5f}" for j, v in enumerate(vals)), flush=True)

            entry = {"fit_subject": fit_name, "holdout_subject": hold_name,
                     "opt": {kk: vv for kk, vv in rep.items()
                             if kk not in ("ic_chosen", "ic_chosen_best_step",
                                           "ic_chosen_avg25")},
                     "restart_report": rep["restart_report"],
                     "history": rep["history"],
                     "evals": {}}
            for tag, ic in (("avg25", rep["ic_chosen_avg25"]),
                            ("best_step", rep["ic_chosen_best_step"])):
                f_fit = exact_fc(ic, fit["src"], args.dt, args.tr, fit["tsfc"])
                f_hold = exact_fc(ic, hold["src"], args.dt, args.tr, hold["tsfc"])
                entry["evals"][tag] = {
                    "ic_mean_over_regions": [float(x) for x in onp.asarray(ic).mean(-1)],
                    "ic_full": onp.asarray(ic).tolist(),
                    "fc_fit": f_fit, "fc_holdout": f_hold,
                    "d_fit": (None if f_fit is None
                              else f_fit - fit["fc_default"]),
                    "d_holdout": (None if f_hold is None
                                  else f_hold - hold["fc_default"])}
                print(f"[opt fit={fit_name} {tag}] "
                      f"fc_{fit_name}(fit)={_f(f_fit)} (default {fit['fc_default']:.4f}) "
                      f"fc_{hold_name}(holdout)={_f(f_hold)} "
                      f"(default {hold['fc_default']:.4f})", flush=True)
            # frozen-tail validity: exact full scan vs head+frozen-tail for the chosen IC
            head = onp.asarray(balloon(fit["src"][:k * int(round(args.tr * 1000.0 / args.dt))],
                                       args.dt, args.tr, rep["ic_chosen_avg25"]))
            approx = onp.concatenate([head, onp.asarray(fit["tail_bold"])], 0)
            full = onp.asarray(balloon(fit["src"], args.dt, args.tr,
                                       rep["ic_chosen_avg25"]))
            entry["frozen_tail_maxdiff"] = float(onp.max(onp.abs(full - approx)))
            print(f"[opt fit={fit_name}] frozen-tail max|BOLD diff| = "
                  f"{entry['frozen_tail_maxdiff']:.3e}", flush=True)
            opt[f"fit_{fit_name}"] = entry
            out["opt_results"] = opt
            _dump(out, args.out)
        out["opt_results"] = opt

    _dump(out, args.out)

    # ---- compact summary -----------------------------------------------------
    print("\n=== SUMMARY (exact full-trace fc_match) ===", flush=True)
    for name, sub in e_all.items():
        print(f"[{name}] default fc = {sub.get('fc_default', float('nan')):.4f}", flush=True)
    for key, ent in out.get("opt_results", {}).items():
        print(f"-- fit on {ent['fit_subject']} --", flush=True)
        for tag, ev in ent["evals"].items():
            print(f"   {tag:9s} IC(mean)={[round(x, 4) for x in ev['ic_mean_over_regions']]} "
                  f"fc_fit={_f(ev['fc_fit'])} (d={_f(ev['d_fit'])}) "
                  f"fc_holdout={_f(ev['fc_holdout'])} (d={_f(ev['d_holdout'])})", flush=True)
    print(f"total wall = {time.time() - t_start:.0f}s", flush=True)
    print("WB_BOLD_IC_DONE", flush=True)


if __name__ == "__main__":
    main()
