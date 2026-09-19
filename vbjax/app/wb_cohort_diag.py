"""Diagnose which trainable key gets a non-finite BPTT gradient at the frozen
G=4/sigma=0.005 regime (cpu, single subject, few windows). Fast iteration vs GPU.
"""
import os, pickle, sys
import numpy as onp
import jax
jax.config.update("jax_platform_name", "cpu")
import jax.numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vbjax.ct_mhsa import Hyperparameters, init_ct_mhsa
from vbjax.app.wb_train import window_loss, zscore
from vbjax.app.wb_cohort_fit import load_cohort

cohort = os.path.expanduser("~/mhsa_e0/data/kb_cohort")
params_pkl = os.path.expanduser(
    "~/mhsa_e0/runs_persist/wb_arm_ema_yout/params.pkl")
g_init, sigma_init, seed = 4.0, 0.005, 42
N = 100
d = 16
dt_train = 100.0
tr = 2.0
v_c = 5.0
win = 64
dt = 2.0

bold, Cs, Ls = load_cohort(cohort, v_c, dt, rho_norm=True)
k = 0
C = np.asarray(Cs[k]); lags = np.asarray(Ls[k])
L = int(lags.max()) + 1
ts = bold[k]
n_win = ts.shape[0] // win
zw = zscore(ts)[:n_win * win].reshape(n_win, win, N)
rw = ts[:n_win * win].reshape(n_win, win, N)

hp0 = Hyperparameters(n_regions=N, n_heads=N, d_model=d, d_k=d, d_v=d,
                      lam=1.0, dt=dt, tau=1e6, leak=1.0, act="relu",
                      steps_per_token=1, sigma_a=0.0)

with open(params_pkl, "rb") as f:
    base = pickle.load(f)
trainable = {kk: np.asarray(base[kk]) for kk in ("W_q", "W_y", "kv_scale", "log_tau")}
frozen = {"log_sigma": np.asarray(onp.log(sigma_init)),
          "log_noise_tc": np.asarray(base["log_noise_tc"]),
          "G_rel": np.asarray(g_init)}
params = dict(trainable); params.update(frozen)

key = jax.random.PRNGKey(seed)
_, st, _ = init_ct_mhsa(hp0, key, batch_size=1)
carry = (st.M, np.zeros((N, d)), np.ones((N, 1)), np.zeros((N, d)), np.ones((N, 1)))

def ws(kw, wi=0):
    fc_emp = None
    from vbjax.app.wholebrain_fc import fc_from_ts
    aux = {"fc_emp": np.asarray(fc_from_ts(ts)), "pulse_sched": None, "pulse_amp": 0.0}
    return window_loss(params, carry, zw[wi], rw[wi], kw, C, lags, L, hp0,
                       dt_train, "current", tr * 1000.0, 0.5, "ema", aux, False)

kw = jax.random.fold_in(key, 0)
val, grads = jax.value_and_grad(
    lambda pp: window_loss(pp, carry, zw[0], rw[0], kw, C, lags, L, hp0,
                           dt_train, "current", tr * 1000.0, 0.5, "ema",
                           {"fc_emp": np.asarray(fc_from_ts(ts)),
                            "pulse_sched": None, "pulse_amp": 0.0}, False)[1])(params)
print("loss", float(val))
for kk, g in grads.items():
    g = onp.asarray(g)
    finite = bool(onp.isfinite(g).all())
    print(f"  {kk}: shape={g.shape} finite={finite} "
          f"nan={int(onp.isnan(g).sum()) if not finite else 0} "
          f"inf={int(onp.isinf(g).sum()) if not finite else 0} "
          f"max_abs={float(onp.abs(g).max()) if finite else float('nan')}")

# now check individual window losses at each window & each stage
print("--- per-window loss + per-key finite ---")
for wi in range(n_win):
    kwx = jax.random.fold_in(key, wi)
    val, grads = jax.value_and_grad(
        lambda pp: window_loss(pp, carry, zw[wi], rw[wi], kwx, C, lags, L, hp0,
                               dt_train, "current", tr * 1000.0, 0.5, "ema",
                               {"fc_emp": np.asarray(fc_from_ts(ts)),
                                "pulse_sched": None, "pulse_amp": 0.0}, False)[1])(params)
    fin = {kk: bool(onp.isfinite(onp.asarray(g)).all()) for kk, g in grads.items()}
    print(f"  win {wi}: loss={float(val):.5f} finite={fin}")
