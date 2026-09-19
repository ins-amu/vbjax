"""Scan (G, sigma) corners for the window-3 kv_scale gradient NaN at the frozen
regime. Single subject, fresh carry per window, reports per-key finiteness on
window 3 under each (G, sigma). Fast GPU probe to localise the instability.
"""
import os, pickle, sys
import numpy as onp
import jax
import jax.numpy as np

from vbjax.ct_mhsa import Hyperparameters, init_ct_mhsa
from vbjax.app.wb_train import window_loss, zscore
from vbjax.app.wb_cohort_fit import load_cohort
from vbjax.app.wholebrain_fc import fc_from_ts

cohort = os.path.expanduser("~/mhsa_e0/data/kb_cohort")
params_pkl = os.path.expanduser(
    "~/mhsa_e0/runs_persist/wb_arm_ema_yout/params.pkl")
N, d, dt_train, tr, v_c, win, dt, seed = 100, 16, 100.0, 2.0, 5.0, 64, 2.0, 42

bold, Cs, Ls = load_cohort(cohort, v_c, dt, rho_norm=True)
k = 0
C = np.asarray(Cs[k]); lags = np.asarray(Ls[k])
L = int(lags.max()) + 1
ts = bold[k]
n_win = ts.shape[0] // win
zw = zscore(ts)[:n_win * win].reshape(n_win, win, N)
rw = ts[:n_win * win].reshape(n_win, win, N)
fc_emp = np.asarray(fc_from_ts(ts))
hp0 = Hyperparameters(n_regions=N, n_heads=N, d_model=d, d_k=d, d_v=d,
                      lam=1.0, dt=dt, tau=1e6, leak=1.0, act="relu",
                      steps_per_token=1, sigma_a=0.0)

with open(params_pkl, "rb") as f:
    base = pickle.load(f)

key = jax.random.PRNGKey(seed)
_, st, _ = init_ct_mhsa(hp0, key, batch_size=1)
carry0 = (st.M, np.zeros((N, d)), np.ones((N, 1)), np.zeros((N, d)),
          np.ones((N, 1)))

# chained carry (as the real loss_fn does) and per-window fresh-carry
def chained_carry(params_arr):
    c = carry0
    for wi in range(n_win):
        kw = jax.random.fold_in(key, wi)
        c, _ = window_loss(params_arr, c, zw[wi], rw[wi], kw, C, lags, L, hp0,
                           dt_train, "current", tr * 1000.0, 0.5, "ema",
                           {"fc_emp": fc_emp, "pulse_sched": None,
                            "pulse_amp": 0.0}, False)
    return tuple(jax.lax.stop_gradient(x) for x in c)

def make_params(g, sig):
    tr_d = {kk: np.asarray(base[kk]) for kk in ("W_q", "W_y", "kv_scale", "log_tau")}
    tr_d["log_sigma"] = np.asarray(onp.log(sig))
    tr_d["log_noise_tc"] = np.asarray(base["log_noise_tc"])
    tr_d["G_rel"] = np.asarray(g)
    return tr_d

grid = [(4.0, 0.005), (4.0, 0.05), (2.0, 0.005), (2.0, 0.05), (4.0, 0.01)]
for g, sig in grid:
    pp = make_params(g, sig)
    # chained-carry loss over all windows
    c = carry0
    tot = 0.0
    for wi in range(n_win):
        kw = jax.random.fold_in(key, wi)
        c, l = window_loss(pp, c, zw[wi], rw[wi], kw, C, lags, L, hp0,
                           dt_train, "current", tr * 1000.0, 0.5, "ema",
                           {"fc_emp": fc_emp, "pulse_sched": None,
                            "pulse_amp": 0.0}, False)
        tot = tot + l
    val, grads = jax.value_and_grad(
        lambda p2: (lambda _c: sum(
            window_loss(p2, _c, zw[wi], rw[wi], jax.random.fold_in(key, wi),
                        C, lags, L, hp0, dt_train, "current", tr * 1000.0,
                        0.5, "ema", {"fc_emp": fc_emp, "pulse_sched": None,
                                     "pulse_amp": 0.0}, False)[1]
            for wi in range(n_win)) / n_win)(carry0))(pp)
    fin = {kk: bool(onp.isfinite(onp.asarray(grads[kk])).all())
           for kk in ("W_q", "W_y", "kv_scale", "log_tau")}
    print(f"G={g} sigma={sig}: chained_loss={float(val):.5f} "
          f"finite={ {kk: fin[kk] for kk in ('kv_scale','W_q','W_y','log_tau')} }",
          flush=True)

print("WB_COBPTT_DIAG2_DONE", flush=True)
