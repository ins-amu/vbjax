"""Standardised evaluation of whole-brain CT-MHSA checkpoints.

Problem
-------
wb_train.eval_fc integrates 600 s at dt=2 ms, runs the Windkessel balloon from
the default ICs (s=0, f=1, v=1, q=1) and computes FC over the WHOLE trace. The
startup transient is checkpoint-dependent, so the reported fc_match is not
comparable across arms: rankings are metric artefacts.

This driver re-scores every checkpoint with one explicit, identical rule:

  legacy : T = 600 s, no discard, FC over the whole BOLD trace
           (reproduces what the trainer reports)
  std    : T = 1200 s, discard to a settled window W, FC over bold[W:]

Settled-window rule (identical for every checkpoint and subject)
----------------------------------------------------------------
After building the BOLD (TR-sampled), compute a rolling std with a 10-TR
window across the trace. Let ``ref`` be the median of the rolling-std series
over the final 50% of the trace. Then W is the smallest TR index after which
the rolling std stays within 20% of ``ref``; W is capped at 20% of the trace
length. FC is scored on ``bold[W:]``.

Every setting below is copied verbatim from wb_train.main/eval_fc so that the
legacy column is the trainer's own number and the std column differs only by
(duration, discard).
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


def settled_window(bold, win=10, tol=0.2, cap_frac=0.2):
    """Rolling-std settled window for a (T, N) BOLD trace.

    Returns (W, rolling_std, ref). W = smallest TR index after which the
    rolling std stays within `tol` of `ref` (the median rolling std over the
    final 50% of the trace), capped at `cap_frac` * T.
    """
    T = bold.shape[0]
    n_win = max(T - win + 1, 1)
    rolling = onp.array([float(bold[t:t + win].std()) for t in range(n_win)])
    tail = rolling[len(rolling) // 2:]
    ref = float(onp.median(tail))
    if not onp.isfinite(ref) or ref == 0.0:
        return 0, rolling, ref
    dev = onp.nonzero(onp.abs(rolling - ref) > tol * abs(ref))[0]
    W = int(dev[-1]) + 1 if dev.size else 0
    W = min(W, int(cap_frac * T))
    return W, rolling, ref


def transient_ratio(bold, W, n_first=10):
    """First-`n_first`-TR std vs the resting std over the settled segment.

    Returns (ratio, transient_std, resting_std).
    """
    trans = float(bold[:n_first].std())
    rest = float(bold[W:].std()) if W < bold.shape[0] else float("nan")
    ratio = trans / rest if rest and onp.isfinite(rest) and rest > 0 else float("nan")
    return ratio, trans, rest


def score(params, C, lags, L, hp0, ts_fc, T_s, dt, tr, seed):
    """One simulate() + balloon + FC, mirroring wb_train.eval_fc exactly.

    Returns a dict with the raw BOLD trace and both whole-trace and
    settled-window FC scores.
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
    p.add_argument("--out", default="runs/wb_evalstd")
    p.add_argument("--tags", default=None,
                   help="comma-separated tag subset; default = all target tags")
    p.add_argument("--dt", type=float, default=2.0)
    p.add_argument("--tr", type=float, default=2.0)
    p.add_argument("--v-c", type=float, default=5.0)
    p.add_argument("--d-model", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sim-s", type=float, default=1200.0,
                   help="standardised simulation length (s)")
    p.add_argument("--legacy-s", type=float, default=600.0,
                   help="legacy simulation length (s), no discard")
    p.add_argument("--settle-win", type=int, default=10)
    p.add_argument("--settle-tol", type=float, default=0.2)
    p.add_argument("--settle-cap", type=float, default=0.2)
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
          f"sim_s={a.sim_s} legacy_s={a.legacy_s} "
          f"settle=(win={a.settle_win},tol={a.settle_tol},cap={a.settle_cap})",
          flush=True)

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
            # ---- standardised: 1200 s, discard to settled window W
            r = score(params, C, lags, L, hp0, ts_fc, a.sim_s, a.dt, a.tr, seed)
            if r is None:
                entry["std"] = None
                print(f"[{tag}][{name}] std NON-FINITE e", flush=True)
            else:
                bold = r["bold"]
                W, rolling, ref = settled_window(bold, a.settle_win,
                                                 a.settle_tol, a.settle_cap)
                tr_ratio, tr_std, rest_std = transient_ratio(bold, W)
                entry["std"] = {
                    "fc_match": fc_match(fc_from_ts(bold[W:]), ts_fc),
                    "T_s": a.sim_s, "n_tr": int(bold.shape[0]), "W": int(W),
                    "W_s": float(W * a.tr),
                    "rolling_std_ref": ref,
                    "settled_frac": float((bold.shape[0] - W) / bold.shape[0]),
                    "transient_ratio": tr_ratio,
                    "transient_std_first10": tr_std,
                    "resting_std": rest_std,
                }
                del bold, rolling, r
            row["params"][name] = entry
            lg = (entry.get("legacy") or {}).get("fc_match")
            sd = (entry.get("std") or {}).get("fc_match")
            Wv = (entry.get("std") or {}).get("W")
            trv = (entry.get("std") or {}).get("transient_ratio")
            print(f"[{tag}][{name}] legacy={lg} std={sd} W={Wv} "
                  f"transient={trv} ({time.time()-t0:.0f}s)", flush=True)

        rows.append(row)
        with open(os.path.join(a.out, f"evalstd_{tag}.json"), "w") as f:
            json.dump(row, f, indent=1)
        del params
        print(f"[{tag}] done in {time.time()-t0:.0f}s", flush=True)

    summary = {
        "settled_rule": (
            "rolling std, window 10 TRs, across the TR-sampled BOLD trace; "
            "ref = median of the rolling-std series over the final 50% of the "
            "trace; W = smallest TR index after which the rolling std stays "
            "within 20% of ref; W capped at 20% of the trace length; FC scored "
            "on bold[W:]"),
        "settings": {"dt": a.dt, "tr": a.tr, "v_c": a.v_c, "d_model": d,
                     "seed": a.seed, "sim_s": a.sim_s, "legacy_s": a.legacy_s,
                     "c_norm": "sqrt99+unit_spectral_radius", "bold_src": "e",
                     "N": N},
        "rows": rows,
    }
    with open(os.path.join(a.out, "evalstd_all.json"), "w") as f:
        json.dump(summary, f, indent=1)

    # ---- markdown table -------------------------------------------------
    def fmt(x, nd=4):
        return "n/a" if x is None or not onp.isfinite(x) else f"{x:+.{nd}f}"

    hdr = ("| tag | legacy fc1 | legacy fc2 | std fc1 | std fc2 | "
           "W1 (TR) | W2 (TR) | transient1 | transient2 |")
    sep = "|---|---|---|---|---|---|---|---|---|"
    lines = [hdr, sep]
    for row in rows:
        if row["status"] != "ok":
            lines.append(f"| {row['tag']} | skipped: {row['reason']} "
                         "| | | | | | | |")
            continue
        s1 = row["params"].get("subj1", {})
        s2 = row["params"].get("subj2", {})
        L1 = (s1.get("legacy") or {}).get("fc_match")
        L2 = (s2.get("legacy") or {}).get("fc_match")
        S1 = (s1.get("std") or {}).get("fc_match")
        S2 = (s2.get("std") or {}).get("fc_match")
        W1 = (s1.get("std") or {}).get("W")
        W2 = (s2.get("std") or {}).get("W")
        T1 = (s1.get("std") or {}).get("transient_ratio")
        T2 = (s2.get("std") or {}).get("transient_ratio")
        lines.append(f"| {row['tag']} | {fmt(L1)} | {fmt(L2)} | {fmt(S1)} | "
                     f"{fmt(S2)} | {W1} | {W2} | "
                     f"{'n/a' if T1 is None else f'{T1:.1f}x'} | "
                     f"{'n/a' if T2 is None else f'{T2:.1f}x'} |")
    table = "\n".join(lines) + "\n"
    with open(os.path.join(a.out, "table.md"), "w") as f:
        f.write(table)
    print("\n" + table, flush=True)
    print(f"WB_EVALSTD_DONE rows={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
