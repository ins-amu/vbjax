"""Two-stage whole-brain CT-MHSA fit, STAGE 1: fit the loop gain G and the noise
sigma on the FC metric (outer loop, no gradients).

Why this split.  Autograd attribution showed G is structurally unidentifiable
through BPTT: G enters only as a SCALAR GAIN on the drive (aff = G*C@y_prev), so
it changes the drive's magnitude and never its direction; the z-scored window
loss discards magnitude (log-sens(x) 6.2e-4 -> 1.4e-7), the injected noise
(drive rms ~7e-6 vs noise ~0.145) dilutes it, and per-region drive normalisation
removes its channel entirely.  The shape controls (W_q, W_y, kv_scale, tau) do
move under BPTT.  The reference pipeline (wholebrain_fc.py) never learns G: it
SWEEPS G and selects by fc_match.  So: fit (G, sigma) on the FC metric here,
then BPTT the shape controls with G and sigma FROZEN (wb_train --freeze-g
--freeze-sigma --g-init --sigma-init).

What this driver does
  1. GRID SEARCH: 7 G values x 5 sigma values, each a 600 s simulation at
     dt = 2 ms (wholebrain_fc.simulate with bold_src='e'), ballooned exactly as
     wholebrain_fc.bold_forward (default ICs) and scored by fc_match against
     fc_from_ts(ts1) on a settled window (lenient 20%-of-ref rule; SEARCH only).
     The pre-balloon signal is zscore(e), exactly as wb_eval_std.score does, so
     the search score is a cheap proxy for the certified one.
  2. CERTIFIED CONFIRMATION of the top 5 by subj1: 2400 s, dt 2 ms, 10-TR
     rolling std, ref = median rolling std over the final 50% of the trace,
     W = first TR starting 50 consecutive TRs within 10% of ref, NO cap, FC on
     bold[W:]; reported for subj1 (fit) and subj2 (holdout).
  3. SENSITIVITY of the certified score to G (whole G grid at the best sigma)
     and to sigma (whole sigma grid at the best G), plus the lenient-rule spread
     over the full 35-point grid.  This decides whether the two-stage idea can
     work at all: a certified fc_match that is flat in G is a decisive negative
     result.

The SHAPE parameters (W_q, W_y, kv_scale, log_tau, log_noise_tc) come from a
params.pkl; G and sigma are taken from the grid, never from the checkpoint.

Run (inside the GPU container, from the tree root):
  python -m vbjax.app.wb_fit_gs --sc .../sc.npy --lengths .../lengths.npy \
      --ts1 .../emp_ts.npy --ts2 .../kb_100sch_subj2/emp_ts.npy \
      --params ~/mhsa_e0/runs_persist/wb_arm_ema_yout/params.pkl \
      --out runs/wb_fitgs
"""
import argparse, json, os, pickle, time

import numpy as onp
import jax
import jax.numpy as np

from vbjax.ct_mhsa import Hyperparameters
from vbjax.app.wholebrain_fc import (load_connectome, simulate, bold_forward,
                                     fc_from_ts, fc_match)
from vbjax.app.wb_train import build_mhsa, zscore
from vbjax.app.wb_eval_std import settled_window, settled_window_strict

SHAPE_KEYS = ("W_q", "W_y", "kv_scale", "log_tau", "log_noise_tc")


def load_shape_params(path):
    """Shape-only parameter dict from a trainer checkpoint (G/sigma dropped)."""
    with open(path, "rb") as f:
        p = pickle.load(f)
    missing = [k for k in SHAPE_KEYS if k not in p]
    if missing:
        raise SystemExit(f"{path}: missing shape keys {missing}")
    return {k: p[k] for k in SHAPE_KEYS}


def slice_stats(vals):
    """Spread statistics of one grid slice (None = unsettled/diverged entries)."""
    v = [float(x) for x in vals if x is not None and onp.isfinite(x)]
    if not v:
        return {"n": 0}
    v = onp.asarray(v)
    return {"n": int(v.size), "mean": float(v.mean()), "std": float(v.std()),
            "min": float(v.min()), "max": float(v.max()),
            "span": float(v.max() - v.min())}


def search_score(bold, ts_fc, win, tol, cap):
    """Lenient 20%-of-ref settled-window fc_match (SEARCH stage only)."""
    W, ref, W_unc = settled_window(bold, win, tol, cap)
    W = int(W)
    return {"W": W, "cap_binds": bool(W_unc > W), "ref": float(ref),
            "fc_match": fc_match(fc_from_ts(bold[W:]), ts_fc),
            "whole": fc_match(fc_from_ts(bold), ts_fc)}


def strict_score(bold, ts_fc, win, tol, run):
    """CERTIFIED rule: run consecutive TRs within tol of ref, no cap."""
    W, ref, W_unc, found = settled_window_strict(bold, win, tol, run, 0.0)
    if not found or W is None:
        return {"W": None, "found": False, "ref": float(ref),
                "fc_match": None, "settled_frac": None}
    W = int(W)
    return {"W": W, "found": True, "ref": float(ref),
            "fc_match": fc_match(fc_from_ts(bold[W:]), ts_fc),
            "settled_frac": float((bold.shape[0] - W) / bold.shape[0])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sc", required=True)
    ap.add_argument("--lengths", required=True)
    ap.add_argument("--ts1", required=True)
    ap.add_argument("--ts2", required=True)
    ap.add_argument("--params", required=True,
                    help="params.pkl supplying the SHAPE parameters only")
    ap.add_argument("--out", default="runs/wb_fitgs")
    ap.add_argument("--dt", type=float, default=2.0)
    ap.add_argument("--tr", type=float, default=2.0)
    ap.add_argument("--v-c", type=float, default=5.0)
    ap.add_argument("--d-model", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--search-s", type=float, default=600.0)
    ap.add_argument("--confirm-s", type=float, default=2400.0)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--g-sweep", default="0.5,1.0,1.5,2.0,2.5,3.0,4.0")
    ap.add_argument("--sigma-sweep", default="0.05,0.10,0.15,0.20,0.30")
    ap.add_argument("--search-win", type=int, default=10)
    ap.add_argument("--search-tol", type=float, default=0.2)
    ap.add_argument("--search-cap", type=float, default=0.2)
    ap.add_argument("--strict-win", type=int, default=10)
    ap.add_argument("--strict-tol", type=float, default=0.1)
    ap.add_argument("--strict-run", type=int, default=50)
    ap.add_argument("--certified-slices", dest="certified_slices",
                    action="store_true", default=True,
                    help="score the whole G grid / sigma grid with the CERTIFIED "
                         "rule at the best combo (the sensitivity deliverable)")
    ap.add_argument("--no-certified-slices", dest="certified_slices",
                    action="store_false")
    ap.add_argument("--probe", action="store_true",
                    help="time one search point and one certified point, then exit")
    a = ap.parse_args()

    t_start = time.time()
    C, lags, L = load_connectome(a.sc, a.lengths, a.dt, a.v_c, "sqrt99")
    N = C.shape[0]
    rho_raw = float(onp.max(onp.abs(onp.linalg.eigvals(C))))
    C = C / rho_raw                    # unit spectral radius, as wb_train/wb_eval_std
    d = a.d_model
    hp0 = Hyperparameters(n_regions=N, n_heads=N, d_model=d, d_k=d, d_v=d,
                          lam=1.0, dt=a.dt, tau=1e6, leak=1.0, act="relu",
                          steps_per_token=1, sigma_a=0.0)
    shape = load_shape_params(a.params)
    tau_s = float(onp.exp(shape["log_tau"]))
    hp = hp0._replace(lam=a.dt / (tau_s * 1000.0), tau=tau_s * 1000.0)
    mhsa = build_mhsa(hp, jax.random.PRNGKey(a.seed), shape["W_q"],
                      shape["W_y"], float(shape["kv_scale"]))
    noise_tc_s = float(onp.exp(shape["log_noise_tc"]))
    ts1 = onp.load(a.ts1)
    ts2 = onp.load(a.ts2)
    fc1, fc2 = fc_from_ts(ts1), fc_from_ts(ts2)
    Gs = [float(x) for x in a.g_sweep.split(",") if x.strip()]
    Ss = [float(x) for x in a.sigma_sweep.split(",") if x.strip()]

    os.makedirs(a.out, exist_ok=True)
    print(f"[init] N={N} rho(C)_raw={rho_raw:.4f} L={L} dt={a.dt} tr={a.tr} "
          f"shape_from={a.params} tau_s={tau_s:.2f} sigma_ck={shape['log_sigma'] if 'log_sigma' in shape else 'n/a'} "
          f"noise_tc_s={noise_tc_s:.3f}", flush=True)
    print(f"[init] G grid {Gs}", flush=True)
    print(f"[init] sigma grid {Ss}", flush=True)
    print(f"[init] search={a.search_s}s (lenient win={a.search_win} tol={a.search_tol} "
          f"cap={a.search_cap}), certified={a.confirm_s}s "
          f"(win={a.strict_win} tol={a.strict_tol} run={a.strict_run} no cap), "
          f"top_k={a.top_k}, certified_slices={a.certified_slices}", flush=True)

    def make_sim(T_s, seed):
        """jit(G, sigma) -> simulate(...); one compile per T_s, reused for the
        whole sweep (inside an outer trace the inner simulate jit is elided)."""
        T = int(T_s * 1000 / a.dt)
        key = jax.random.PRNGKey(seed)

        def run(G, sigma):
            e, _y = simulate(mhsa, np.asarray(C), jax.numpy.asarray(lags), L,
                             hp, G, sigma, T, key, noise_tc_s, a.dt, "e")
            return e                     # y is dead code for scoring (XLA DCEs it)

        return jax.jit(run)

    bold_jit = jax.jit(lambda sig: bold_forward(sig, a.dt, a.tr))

    def bold_at(sim, G, sigma):
        """One point: simulate -> zscore(e) (wb_eval_std convention) -> balloon."""
        e = onp.asarray(sim(float(G), float(sigma)))
        if not onp.isfinite(e).all():
            return None, "non-finite e"
        bold = onp.asarray(bold_jit(np.asarray(zscore(e), dtype="f4")))
        if not onp.isfinite(bold).all():
            return None, "non-finite BOLD"
        return bold, None

    if a.probe:
        for T_s, tag in ((a.search_s, "search"), (a.confirm_s, "certified")):
            t0 = time.time()
            sim = make_sim(T_s, a.seed)
            bold, err = bold_at(sim, 2.0, 0.19)
            t1 = time.time()
            _ = bold_at(sim, 2.0, 0.19)          # 2nd call: compile-free speed
            t2 = time.time()
            print(f"[probe] {tag} T_s={T_s} first={t1-t0:.1f}s second={t2-t1:.1f}s "
                  f"err={err} bold={None if bold is None else bold.shape}", flush=True)
        try:
            print(f"[probe] device memory {jax.devices()[0].memory_stats()}", flush=True)
        except Exception as ex:
            print(f"[probe] device memory unavailable: {ex}", flush=True)
        print("WB_FITGS_PROBE_DONE", flush=True)
        return

    # ---------------- stage 1a: lenient grid search (subj1) ----------------
    sim_search = make_sim(a.search_s, a.seed)
    grid = []
    for sig in Ss:
        for G in Gs:
            t0 = time.time()
            bold, err = bold_at(sim_search, G, sig)
            row = {"G": G, "sigma": sig}
            if bold is None:
                row.update({"status": "diverged", "reason": err})
            else:
                row.update({"status": "ok"})
                row.update(search_score(bold, fc1, a.search_win, a.search_tol,
                                        a.search_cap))
            dtp = time.time() - t0
            row["secs"] = round(dtp, 1)
            grid.append(row)
            fcv, wv = row.get("fc_match"), row.get("whole")
            s_fc = "n/a" if fcv is None else f"{fcv:+.4f}"
            s_w = "n/a" if wv is None else f"{wv:+.4f}"
            print(f"[search] G={G:<5} sigma={sig:<5} fc1_20%={s_fc} "
                  f"whole={s_w} W={row.get('W')} ({dtp:.1f}s)", flush=True)

    ok = [r for r in grid if r.get("fc_match") is not None]
    ok.sort(key=lambda r: -r["fc_match"])
    top = ok[:a.top_k]
    print(f"[search] {len(ok)}/{len(grid)} scored; top-{len(top)}: "
          f"{[(r['G'], r['sigma'], round(r['fc_match'], 4)) for r in top]}",
          flush=True)

    # ---------------- stage 1b: certified confirmation + sensitivity --------
    sim_s1 = make_sim(a.confirm_s, a.seed)
    sim_s2 = make_sim(a.confirm_s, a.seed + 1)
    cert = {}
    cert_log = []

    def certified(G, sigma):
        """Certified (G, sigma) score for both subjects, cached."""
        k = (float(G), float(sigma))
        if k in cert:
            return cert[k]
        t0 = time.time()
        out = {"G": k[0], "sigma": k[1], "subj1": None, "subj2": None}
        for name, sim, fc, seed in (("subj1", sim_s1, fc1, a.seed),
                                    ("subj2", sim_s2, fc2, a.seed + 1)):
            bold, err = bold_at(sim, k[0], k[1])
            if bold is None:
                out[name] = {"fc_match": None, "W": None, "found": False,
                             "reason": err}
            else:
                out[name] = strict_score(bold, fc, a.strict_win, a.strict_tol,
                                         a.strict_run)
                del bold
        out["secs"] = round(time.time() - t0, 1)
        cert[k] = out
        cert_log.append(out)
        f1 = out["subj1"].get("fc_match") if out["subj1"] else None
        f2 = out["subj2"].get("fc_match") if out["subj2"] else None
        s1 = "n/a" if f1 is None else f"{f1:+.4f}"
        s2 = "n/a" if f2 is None else f"{f2:+.4f}"
        w1 = out["subj1"].get("W") if out["subj1"] else None
        w2 = out["subj2"].get("W") if out["subj2"] else None
        print(f"[certified] G={k[0]:<5} sigma={k[1]:<5} fc1={s1} (W={w1}) "
              f"fc2={s2} (W={w2}) ({out['secs']}s)", flush=True)
        with open(os.path.join(a.out, "wb_fitgs.json"), "w") as f:
            json.dump({"partial": True, "grid": grid, "certified": cert_log},
                      f, indent=1)
        return out

    confirmed = [certified(r["G"], r["sigma"]) for r in top]

    def cert_fc(entry, s):
        e = entry.get(s)
        return None if not e else e.get("fc_match")

    best = None
    for c in confirmed:
        if cert_fc(c, "subj1") is None:
            continue
        if best is None or cert_fc(c, "subj1") > cert_fc(best, "subj1"):
            best = c
    if best is None and top:
        best = {"G": top[0]["G"], "sigma": top[0]["sigma"], "subj1": None,
                "subj2": None, "note": "no settled certified run; search best"}
    best_G, best_sigma = (best["G"], best["sigma"]) if best else (None, None)

    # sensitivity: lenient rule, full grid, subj1
    lenient_by_sigma = {str(s): slice_stats(
        [r.get("fc_match") for r in grid
         if r["sigma"] == s and r["status"] == "ok"]) for s in Ss}
    lenient_by_G = {str(g): slice_stats(
        [r.get("fc_match") for r in grid
         if r["G"] == g and r["status"] == "ok"]) for g in Gs}
    lenient_all = slice_stats([r.get("fc_match") for r in grid])

    sensitivity = {
        "lenient_rule_full_grid_subj1": {
            "by_G_slice": lenient_by_G, "by_sigma_slice": lenient_by_sigma,
            "all": lenient_all,
            "note": "lenient 20%-of-ref rule, 600 s sims; G slice = spread over "
                    "the G grid at fixed sigma (and vice versa)"},
    }

    # sensitivity: CERTIFIED rule, whole G grid at best sigma / whole sigma grid
    # at best G (both subjects), the decisive measurement
    if a.certified_slices and best_G is not None:
        G_slice = [certified(g, best_sigma) for g in Gs]
        S_slice = [certified(best_G, s) for s in Ss]
        sensitivity["certified_rule_subj1"] = {
            "at_best_sigma_by_G": {
                "sigma": best_sigma,
                "fc_match": {str(e["G"]): cert_fc(e, "subj1") for e in G_slice},
                "spread": slice_stats([cert_fc(e, "subj1") for e in G_slice])},
            "at_best_G_by_sigma": {
                "G": best_G,
                "fc_match": {str(e["sigma"]): cert_fc(e, "subj1")
                             for e in S_slice},
                "spread": slice_stats([cert_fc(e, "subj1") for e in S_slice])},
        }
        sensitivity["certified_rule_subj2"] = {
            "at_best_sigma_by_G": {
                "sigma": best_sigma,
                "fc_match": {str(e["G"]): cert_fc(e, "subj2") for e in G_slice},
                "spread": slice_stats([cert_fc(e, "subj2") for e in G_slice])},
            "at_best_G_by_sigma": {
                "G": best_G,
                "fc_match": {str(e["sigma"]): cert_fc(e, "subj2")
                             for e in S_slice},
                "spread": slice_stats([cert_fc(e, "subj2") for e in S_slice])},
        }

    # spread of the certified score over the top-k confirmations (subj1)
    sensitivity["certified_topk_spread_subj1"] = slice_stats(
        [cert_fc(c, "subj1") for c in confirmed])

    summary = {
        "settings": {"sc": a.sc, "lengths": a.lengths, "ts1": a.ts1, "ts2": a.ts2,
                     "params": a.params, "dt": a.dt, "tr": a.tr, "v_c": a.v_c,
                     "d_model": d, "seed": a.seed, "search_s": a.search_s,
                     "confirm_s": a.confirm_s, "top_k": a.top_k,
                     "g_sweep": Gs, "sigma_sweep": Ss,
                     "search_rule": {"win": a.search_win, "tol": a.search_tol,
                                     "cap": a.search_cap},
                     "certified_rule": {"win": a.strict_win, "tol": a.strict_tol,
                                        "run": a.strict_run, "cap": 0.0,
                                        "sim_s": a.confirm_s, "dt": a.dt},
                     "c_norm": "sqrt99+unit_spectral_radius", "bold_src": "e",
                     "pre_balloon": "zscore(e) (wb_eval_std.score convention)",
                     "N": N, "rho_C_raw": rho_raw,
                     "shape_params_from_ckpt": {
                         "W_q": "as checkpoint", "W_y": "as checkpoint",
                         "kv_scale": float(shape["kv_scale"]),
                         "tau_s": tau_s, "noise_tc_s": noise_tc_s}},
        "grid": grid,
        "search_top_k": [{"G": r["G"], "sigma": r["sigma"],
                          "fc1_lenient": r.get("fc_match")} for r in top],
        "confirmed": cert_log,
        "best": best,
        "sensitivity": sensitivity,
        "elapsed_s": round(time.time() - t_start, 1),
    }
    with open(os.path.join(a.out, "wb_fitgs.json"), "w") as f:
        json.dump(summary, f, indent=1)

    # compact markdown table for the deck
    lines = ["| G \\ sigma | " + " | ".join(f"{s}" for s in Ss) + " |",
             "|---|" + "---|" * len(Ss)]
    for G in Gs:
        cells = []
        for s in Ss:
            r = next((x for x in grid if x["G"] == G and x["sigma"] == s), None)
            cells.append("n/a" if not r or r.get("fc_match") is None
                         else f"{r['fc_match']:+.4f}")
        lines.append(f"| {G} | " + " | ".join(cells) + " |")
    with open(os.path.join(a.out, "grid_lenient_fc1.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    print(json.dumps({"best": best, "sensitivity": sensitivity}, indent=1)[:4000],
          flush=True)
    print(f"WB_FITGS_DONE elapsed={time.time()-t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
