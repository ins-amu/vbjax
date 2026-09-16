"""Standardised evaluation of whole-brain CT-MHSA checkpoints.

Problem
-------
wb_train.eval_fc integrates 600 s at dt=2 ms, runs the Windkessel balloon from
the default ICs (s=0, f=1, v=1, q=1) and computes FC over the WHOLE trace. The
startup transient is checkpoint-dependent, so the reported fc_match is not
comparable across arms: rankings are metric artefacts.

This driver re-scores every checkpoint with explicit, identical rules:

  legacy       : T = 600 s, no discard, FC over the whole BOLD trace
                 (reproduces what the trainer reports)
  r1200_20     : T = 1200 s, discard to a settled window W, FC over bold[W:]
  r2400_20     : T = 2400 s, same 20%-of-ref rule (does the cap stop binding?)
  r2400_strict : T = 2400 s, STRICTER rule: rolling std within 10% of ref for
                 50 consecutive TRs, no cap (a certified-settled window)

Settled-window rules (identical for every checkpoint and subject)
-----------------------------------------------------------------
Shared prelude. After building the BOLD (TR-sampled), compute a rolling std
with a 10-TR window across the trace. Let ``ref`` be the median of the
rolling-std series over the final 50% of the trace. The rolling-std series is
"ok" where it is within ``tol`` of ``ref``.

  rule20 (tol=20%, cap=20% of trace)
      W = smallest TR index after which the rolling std NEVER leaves ``ref``
      +-20% again (i.e. last violating window index + 1); W is capped at
      ``cap_frac`` * trace length. A W sitting on the cap is NOT certified
      settled; the driver reports the uncapped W and a cap-binding flag.

  strict (tol=10%, run=50 consecutive TRs, no cap)
      W = smallest TR index where at least ``run`` consecutive rolling-std
      windows are all within 10% of ``ref``; FC is scored on ``bold[W:]``. If
      no such run exists, W is None and the run is reported unsettled.
      ``--strict-cap`` can impose a cap (default 0 = no cap); cap binding is
      reported either way.

Every simulation setting below is copied verbatim from wb_train.main/eval_fc so
that the legacy column is the trainer's own number and the settled columns
differ only by (duration, discard rule).
"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as onp
import jax
import jax.numpy as np

from vbjax.ct_mhsa import Hyperparameters
from vbjax.app.wholebrain_fc import (load_connectome, bold_forward,
                                     fc_from_ts, fc_match, simulate)
# Reuse the trainer's own builders; wb_train is imported, never modified.
from vbjax.app.wb_train import build_mhsa, zscore

# Tags whose runs NaN-skipped every step: params are still the initialisation,
# not a fit. Recorded in the report as skipped, never scored.
SKIP_TAGS = {
    "wb_arm_balloon_gs":
        "run NaN-skipped every step; params are the initialisation, not a fit",
    "wb_arm_r3_balloon200":
        "run NaN-skipped every step; params are the initialisation, not a fit",
}

DEFAULT_TAGS = [
    "wb_train",
    "wb_arm_fcy_noreg",
    "wb_arm_fce_logg",
    "wb_arm_fcy_pulse",
    "wb_arm_current_logg",
    "wb_arm_ema_yout",
    "wb_arm_r3_fcy300",
    "wb_arm_r3_fc2",
]


def rolling_std(bold, win=10):
    """Rolling std of a (T, N) BOLD trace over a `win`-TR boxcar (length T-win+1)."""
    T = bold.shape[0]
    n_win = max(T - win + 1, 1)
    if T < win:
        return onp.asarray([float(bold.std())]), float(bold.std())
    rolling = onp.array([float(bold[t:t + win].std()) for t in range(n_win)])
    ref = float(onp.median(rolling[len(rolling) // 2:]))
    return rolling, ref


def settled_window(bold, win=10, tol=0.2, cap_frac=0.2):
    """20%-of-ref settled window (rule20). Returns (W, rolling, ref, W_uncapped).

    W = smallest TR index after which the rolling std stays within `tol` of
    `ref`, capped at `cap_frac` * T. ``W_uncapped`` is the same index before the
    cap, so ``W_uncapped > W`` means the cap bound (not certified settled).
    """
    T = bold.shape[0]
    rolling, ref = rolling_std(bold, win)
    if not onp.isfinite(ref) or ref == 0.0:
        return 0, rolling, ref, 0
    dev = onp.nonzero(onp.abs(rolling - ref) > tol * abs(ref))[0]
    W_unc = int(dev[-1]) + 1 if dev.size else 0
    cap = int(cap_frac * T) if cap_frac and cap_frac > 0 else None
    W = min(W_unc, cap) if cap is not None else W_unc
    return W, rolling, ref, W_unc


def settled_window_strict(bold, win=10, tol=0.1, run=50, cap_frac=0.0):
    """Strict settled window: `run` consecutive TRs within `tol` of ref, no cap.

    Returns (W, ref, W_uncapped, found). W = smallest TR index starting a run of
    `run` consecutive ok rolling-std windows; None if no such run exists.
    ``cap_frac`` <= 0 means no cap (the default).
    """
    T = bold.shape[0]
    rolling, ref = rolling_std(bold, win)
    n_win = rolling.shape[0]
    if not onp.isfinite(ref) or ref == 0.0:
        return None, ref, None, False
    ok = onp.abs(rolling - ref) <= tol * abs(ref)
    W_unc = None
    if n_win >= run:
        for t in range(n_win - run + 1):
            if ok[t:t + run].all():
                W_unc = t
                break
    if W_unc is None:
        return None, ref, None, False
    cap = int(cap_frac * T) if cap_frac and cap_frac > 0 else None
    W = min(W_unc, cap) if cap is not None else W_unc
    return W, ref, W_unc, True


def transient_ratio(bold, W, n_first=10):
    """First-`n_first`-TR std vs the resting std over the settled segment.

    Returns (ratio, transient_std, resting_std).
    """
    trans = float(bold[:n_first].std())
    rest = (float(bold[W:].std()) if W is not None and W < bold.shape[0]
            else float("nan"))
    ratio = trans / rest if rest and onp.isfinite(rest) and rest > 0 else float("nan")
    return ratio, trans, rest


def settled_entry(bold, ts_fc, T_s, tr, rule, **kw):
    """One settled-window row: rule in {'rule20','strict'} over a given BOLD trace."""
    entry = {"T_s": T_s, "n_tr": int(bold.shape[0])}
    if rule == "rule20":
        W, rolling, ref, W_unc = settled_window(
            bold, kw["win"], kw["tol"], kw["cap"])
        cap = int(kw["cap"] * bold.shape[0]) if kw["cap"] and kw["cap"] > 0 else None
        entry.update({"rule": f"rolling std within {kw['tol']:.0%} of ref, "
                              f"cap {kw['cap']:.0%} of trace",
                      "tol": kw["tol"], "cap_frac": kw["cap"],
                      "W": int(W), "W_uncapped": int(W_unc),
                      "W_s": float(W * tr), "cap_binds": bool(W_unc > W),
                      "cap_value_tr": cap, "rolling_std_ref": ref})
    else:
        W, ref, W_unc, found = settled_window_strict(
            bold, kw["win"], kw["tol"], kw["run"], kw["cap"])
        cap = int(kw["cap"] * bold.shape[0]) if kw["cap"] and kw["cap"] > 0 else None
        entry.update({"rule": f"rolling std within {kw['tol']:.0%} of ref for "
                              f"{kw['run']} consecutive TRs, "
                              f"cap {kw['cap']:.0%} of trace",
                      "tol": kw["tol"], "run": kw["run"], "cap_frac": kw["cap"],
                      "W": (int(W) if W is not None else None),
                      "W_uncapped": (int(W_unc) if W_unc is not None else None),
                      "W_s": (float(W * tr) if W is not None else None),
                      "cap_binds": bool(found and W_unc is not None and
                                        W is not None and W_unc > W),
                      "cap_value_tr": cap, "found": bool(found),
                      "rolling_std_ref": ref})
    if entry["W"] is None:
        entry["fc_match"] = None
        entry["settled_frac"] = None
        entry["transient_ratio"] = None
    else:
        entry["fc_match"] = fc_match(fc_from_ts(bold[entry["W"]:]), ts_fc)
        entry["settled_frac"] = float((bold.shape[0] - entry["W"]) /
                                      bold.shape[0])
        tr_ratio, tr_std, rest_std = transient_ratio(bold, entry["W"])
        entry["transient_ratio"] = tr_ratio
        entry["transient_std_first10"] = tr_std
        entry["resting_std"] = rest_std
    return entry


def score(params, C, lags, L, hp0, ts_fc, T_s, dt, tr, seed):
    """One simulate() + balloon, mirroring wb_train.eval_fc exactly.

    Returns a dict with the raw BOLD trace and the whole-trace FC score.
    """
    tau_s = float(np.exp(params["log_tau"]))
    sigma = float(np.exp(params["log_sigma"]))
    hp = hp0._replace(lam=dt / (tau_s * 1000.0), tau=tau_s * 1000.0)
    mhsa = build_mhsa(hp, jax.random.PRNGKey(seed), params["W_q"],
                      params["W_y"], float(params["kv_scale"]))
    G_eval = (float(np.exp(params["log_G"])) if "log_G" in params
              else float(params["G_rel"]))
    e, _ = simulate(mhsa, np.asarray(C), jax.numpy.asarray(lags), L, hp,
                    G_eval, sigma, int(T_s * 1000 / dt),
                    jax.random.PRNGKey(seed),
                    float(np.exp(params["log_noise_tc"])), dt, "e")
    e = onp.asarray(e)
    if not onp.isfinite(e).all():
        return None
    bold = bold_forward(zscore(e), dt, tr)
    del e
    return {"bold": bold,
            "whole": fc_match(fc_from_ts(bold), ts_fc),
            "T_s": T_s}


def init_params_like_trainer(hp0, N, d, key_seed=42, log_g=False):
    """The params dict wb_train builds before training (initialisation probe)."""
    from vbjax.ct_mhsa import init_ct_mhsa
    key = jax.random.PRNGKey(key_seed)
    k_m, _ = jax.random.split(key)
    mhsa0, _, _ = init_ct_mhsa(hp0, k_m, batch_size=1)
    p = {"W_q": mhsa0.W_Q, "W_y": mhsa0.W_Y,
         "kv_scale": np.asarray(0.15),
         "log_tau": np.asarray(onp.log(36.0)),
         "log_sigma": np.asarray(onp.log(0.18)),
         "log_noise_tc": np.asarray(onp.log(2.0))}
    if log_g:
        p["log_G"] = np.asarray(onp.log(2.09))
    else:
        p["G_rel"] = np.asarray(2.09)
    return p


def compare_init(params, init_p):
    """Max relative deviation of `params` from the trainer's initialisation."""
    worst, key_at = 0.0, None
    for k, v0 in init_p.items():
        if k not in params:
            return float("inf"), k
        a = onp.asarray(params[k], dtype="f8")
        b = onp.asarray(v0, dtype="f8")
        if a.shape != b.shape:
            return float("inf"), k
        den = max(float(onp.abs(b).max()), 1e-12)
        dv = float(onp.abs(a - b).max()) / den
        if dv > worst:
            worst, key_at = dv, k
    return worst, key_at


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sc", required=True)
    p.add_argument("--lengths", required=True)
    p.add_argument("--ts1", required=True)
    p.add_argument("--ts2", required=True)
    p.add_argument("--runs-root", default=os.path.expanduser(
        "~/mhsa_e0/runs_persist"))
    p.add_argument("--out", default="runs/wb_metstd")
    p.add_argument("--tags", default=None,
                   help="comma-separated tag subset; default = all target tags")
    p.add_argument("--dt", type=float, default=2.0)
    p.add_argument("--tr", type=float, default=2.0)
    p.add_argument("--v-c", type=float, default=5.0)
    p.add_argument("--d-model", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sim-s", type=float, default=2400.0,
                   help="primary standardised simulation length (s)")
    p.add_argument("--mid-s", type=float, default=1200.0,
                   help="earlier standardised length for comparison (s); "
                        "set equal to --sim-s to reuse the primary trace")
    p.add_argument("--legacy-s", type=float, default=600.0,
                   help="legacy simulation length (s), no discard")
    p.add_argument("--settle-win", type=int, default=10)
    p.add_argument("--settle-tol", type=float, default=0.2)
    p.add_argument("--settle-cap", type=float, default=0.2)
    p.add_argument("--strict-win", type=int, default=10)
    p.add_argument("--strict-tol", type=float, default=0.1)
    p.add_argument("--strict-run", type=int, default=50,
                   help="consecutive TRs that must stay within strict tol")
    p.add_argument("--strict-cap", type=float, default=0.0,
                   help="cap for the strict rule as a fraction of trace "
                        "length; 0 = no cap (default)")
    p.add_argument("--skip-legacy", action="store_true",
                   help="omit the 600 s legacy runs (halves the cost)")
    a = p.parse_args()

    tags = (a.tags.split(",") if a.tags else list(DEFAULT_TAGS))
    tags = [t.strip() for t in tags if t.strip()]

    C, lags, L = load_connectome(a.sc, a.lengths, a.dt, a.v_c, "sqrt99")
    N = C.shape[0]
    rho = float(onp.max(onp.abs(onp.linalg.eigvals(C))))
    C = C / rho                      # unit spectral radius, as in wb_train
    d = a.d_model
    hp0 = Hyperparameters(n_regions=N, n_heads=N, d_model=d, d_k=d, d_v=d,
                          lam=1.0, dt=a.dt, tau=1e6, leak=1.0, act="relu",
                          steps_per_token=1, sigma_a=0.0)

    ts1 = onp.load(a.ts1)
    ts2 = onp.load(a.ts2)
    fc1, fc2 = fc_from_ts(ts1), fc_from_ts(ts2)
    subjects = [("subj1", fc1, a.seed), ("subj2", fc2, a.seed + 1)]

    os.makedirs(a.out, exist_ok=True)
    print(f"N={N} L={L} dt={a.dt} tr={a.tr} v_c={a.v_c} d_model={d} "
          f"sim_s={a.sim_s} mid_s={a.mid_s} legacy_s={a.legacy_s} "
          f"settle=(win={a.settle_win},tol={a.settle_tol},cap={a.settle_cap}) "
          f"strict=(win={a.strict_win},tol={a.strict_tol},run={a.strict_run},"
          f"cap={a.strict_cap})", flush=True)

    reuse_mid = abs(a.mid_s - a.sim_s) < 1e-9
    rows = []
    for tag in tags:
        if tag in SKIP_TAGS:
            print(f"[{tag}] SKIPPED: {SKIP_TAGS[tag]}", flush=True)
            rows.append({"tag": tag, "status": "skipped",
                         "reason": SKIP_TAGS[tag]})
            continue
        pkl = os.path.join(a.runs_root, tag, "params.pkl")
        if not os.path.exists(pkl):
            print(f"[{tag}] MISSING {pkl}", flush=True)
            rows.append({"tag": tag, "status": "missing", "reason": pkl})
            continue

        t0 = time.time()
        with open(pkl, "rb") as f:
            params = pickle.load(f)
        try:
            init_p = init_params_like_trainer(
                hp0, N, d, a.seed, log_g=("log_G" in params))
            init_dev, init_at = compare_init(params, init_p)
        except Exception as ex:                      # diagnostics only
            init_dev, init_at = float("nan"), f"error:{ex}"

        row = {"tag": tag, "status": "ok",
               "G": (float(np.exp(params["log_G"])) if "log_G" in params
                     else float(params["G_rel"])),
               "G_param": "log_G" if "log_G" in params else "G_rel",
               "tau_s": float(np.exp(params["log_tau"])),
               "sigma": float(np.exp(params["log_sigma"])),
               "noise_tc_s": float(np.exp(params["log_noise_tc"])),
               "kv_scale": float(params["kv_scale"]),
               "init_max_rel_dev": init_dev, "init_dev_at": init_at,
               "params": {}}

        for name, ts_fc, seed in subjects:
            entry = {}
            # ---- legacy: 600 s, whole trace (the trainer's own metric)
            if not a.skip_legacy:
                r = score(params, C, lags, L, hp0, ts_fc, a.legacy_s,
                          a.dt, a.tr, seed)
                if r is None:
                    entry["legacy"] = None
                    print(f"[{tag}][{name}] legacy NON-FINITE e", flush=True)
                else:
                    entry["legacy"] = {"fc_match": r["whole"],
                                       "T_s": a.legacy_s,
                                       "n_tr": int(r["bold"].shape[0])}
                    del r
            # ---- primary standardised run: sim_s s (2400), both rules
            r = score(params, C, lags, L, hp0, ts_fc, a.sim_s, a.dt, a.tr, seed)
            if r is None:
                entry["r2400_20"] = None
                entry["r2400_strict"] = None
                print(f"[{tag}][{name}] primary NON-FINITE e", flush=True)
            else:
                bold = r["bold"]
                entry[f"r{int(a.sim_s)}_20"] = settled_entry(
                    bold, ts_fc, a.sim_s, a.tr, "rule20",
                    win=a.settle_win, tol=a.settle_tol, cap=a.settle_cap)
                entry[f"r{int(a.sim_s)}_strict"] = settled_entry(
                    bold, ts_fc, a.sim_s, a.tr, "strict",
                    win=a.strict_win, tol=a.strict_tol, run=a.strict_run,
                    cap=a.strict_cap)
                if reuse_mid:
                    entry[f"r{int(a.mid_s)}_20"] = dict(
                        entry[f"r{int(a.sim_s)}_20"],
                        note="reused primary trace (mid_s == sim_s)")
                del bold, r
            # ---- mid standardised run: mid_s s (1200), 20% rule
            if not reuse_mid:
                r = score(params, C, lags, L, hp0, ts_fc, a.mid_s,
                          a.dt, a.tr, seed)
                if r is None:
                    entry["r1200_20"] = None
                    print(f"[{tag}][{name}] mid NON-FINITE e", flush=True)
                else:
                    bold = r["bold"]
                    entry[f"r{int(a.mid_s)}_20"] = settled_entry(
                        bold, ts_fc, a.mid_s, a.tr, "rule20",
                        win=a.settle_win, tol=a.settle_tol, cap=a.settle_cap)
                    del bold, r
            row["params"][name] = entry
            print(f"[{tag}][{name}] legacy={_fm(entry.get('legacy'))} "
                  f"mid20={_fm(entry.get(f'r{int(a.mid_s)}_20'))}"
                  f"(W={_w(entry.get(f'r{int(a.mid_s)}_20'))}) "
                  f"prim20={_fm(entry.get(f'r{int(a.sim_s)}_20'))}"
                  f"(W={_w(entry.get(f'r{int(a.sim_s)}_20'))}) "
                  f"strict={_fm(entry.get(f'r{int(a.sim_s)}_strict'))}"
                  f"(W={_w(entry.get(f'r{int(a.sim_s)}_strict'))}) "
                  f"({time.time()-t0:.0f}s)", flush=True)

        rows.append(row)
        with open(os.path.join(a.out, f"metstd_{tag}.json"), "w") as f:
            json.dump(row, f, indent=1)
        del params
        print(f"[{tag}] done in {time.time()-t0:.0f}s", flush=True)

    variants = ["legacy", f"r{int(a.mid_s)}_20", f"r{int(a.sim_s)}_20",
                f"r{int(a.sim_s)}_strict"]
    summary = {
        "settled_rule_20": (
            "rolling std, window 10 TRs, across the TR-sampled BOLD trace; "
            "ref = median of the rolling-std series over the final 50% of the "
            "trace; W = smallest TR index after which the rolling std stays "
            f"within {a.settle_tol:.0%} of ref; W capped at "
            f"{a.settle_cap:.0%} of the trace length; FC scored on bold[W:]; "
            "cap_binds flags W_uncapped > W"),
        "settled_rule_strict": (
            "same rolling std and same ref; W = smallest TR index starting "
            f"{a.strict_run} consecutive TRs all within {a.strict_tol:.0%} of "
            f"ref; cap {a.strict_cap:.0%} of trace (0 = no cap); no run => W "
            "None (unsettled); FC scored on bold[W:]"),
        "settings": {"dt": a.dt, "tr": a.tr, "v_c": a.v_c, "d_model": d,
                     "seed": a.seed, "sim_s": a.sim_s, "mid_s": a.mid_s,
                     "legacy_s": a.legacy_s,
                     "settle_win": a.settle_win, "settle_tol": a.settle_tol,
                     "settle_cap": a.settle_cap, "strict_win": a.strict_win,
                     "strict_tol": a.strict_tol, "strict_run": a.strict_run,
                     "strict_cap": a.strict_cap,
                     "c_norm": "sqrt99+unit_spectral_radius", "bold_src": "e",
                     "N": N},
        "rows": rows,
        "rankings": rank_variants(rows, variants),
    }
    with open(os.path.join(a.out, "metstd_all.json"), "w") as f:
        json.dump(summary, f, indent=1)

    table = make_table(rows, variants, a)
    with open(os.path.join(a.out, "table.md"), "w") as f:
        f.write(table)
    print("\n" + table, flush=True)
    print(rank_report(summary["rankings"], variants), flush=True)
    print(f"WB_METSTD_DONE rows={len(rows)}", flush=True)


# --------------------------------------------------------------------------
# reporting helpers
# --------------------------------------------------------------------------
def _fm(e):
    return "n/a" if not e or e.get("fc_match") is None else f"{e['fc_match']:+.4f}"


def _w(e):
    return "n/a" if not e or e.get("W") is None else e["W"]


def _subj_mean(row, name, variant):
    e = (row.get("params", {}).get(name) or {}).get(variant) or {}
    v = e.get("fc_match")
    return v


def rank_variants(rows, variants):
    """Per variant: tags ranked by mean fc_match over subjects (n/a last)."""
    out = {}
    for var in variants:
        scored = []
        for row in rows:
            if row.get("status") != "ok":
                continue
            vals = [_subj_mean(row, s, var) for s in ("subj1", "subj2")]
            vals = [v for v in vals if v is not None and onp.isfinite(v)]
            mean = float(onp.mean(vals)) if vals else None
            scored.append({"tag": row["tag"], "mean": mean,
                           "subj1": _subj_mean(row, "subj1", var),
                           "subj2": _subj_mean(row, "subj2", var)})
        scored.sort(key=lambda r: (r["mean"] is None,
                                   -(r["mean"] if r["mean"] is not None else 0)))
        out[var] = scored
    return out


def rank_report(rankings, variants):
    lines = ["RANKINGS (mean fc_match over subj1+subj2, descending)"]
    for var in variants:
        lines.append(f"  {var}:")
        for i, r in enumerate(rankings.get(var, []), 1):
            m = "n/a" if r["mean"] is None else f"{r['mean']:+.4f}"
            lines.append(f"    {i}. {r['tag']:<24} mean={m}")
    return "\n".join(lines)


def make_table(rows, variants, a):
    def cell(e):
        if not e:
            return "n/a"
        if e.get("fc_match") is None:
            return "unsettled"
        s = f"{e['fc_match']:+.4f}"
        return s

    def wcell(e):
        if not e or e.get("W") is None:
            return "n/a"
        return (f"{e['W']}" + ("*" if e.get("cap_binds") else ""))

    vmid = f"r{int(a.mid_s)}_20"
    vprim = f"r{int(a.sim_s)}_20"
    vstrict = f"r{int(a.sim_s)}_strict"
    hdr = ("| tag | subj | legacy " + f"{int(a.legacy_s)}s | "
           f"{int(a.mid_s)}s+20% | W | {int(a.sim_s)}s+20% | W | "
           f"{int(a.sim_s)}s+strict | W |")
    sep = "|---|---|---|---|---|---|---|---|---|"
    lines = [hdr, sep]
    for row in rows:
        if row["status"] != "ok":
            lines.append(f"| {row['tag']} | - | skipped: {row['reason']} "
                         "| | | | | | |")
            continue
        for name in ("subj1", "subj2"):
            e = row["params"].get(name, {})
            lg = e.get("legacy") or {}
            lgs = ("n/a" if lg.get("fc_match") is None
                   else f"{lg['fc_match']:+.4f}")
            lines.append(
                f"| {row['tag']} | {name} | {lgs} | "
                f"{cell(e.get(vmid))} | {wcell(e.get(vmid))} | "
                f"{cell(e.get(vprim))} | {wcell(e.get(vprim))} | "
                f"{cell(e.get(vstrict))} | {wcell(e.get(vstrict))} |")
    lines.append("")
    lines.append("* in the W columns = the cap bound (W_uncapped > W), so that "
                 "window is capped, not certified settled.")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
