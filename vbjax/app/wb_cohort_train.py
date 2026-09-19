"""Cohort BPTT of the SHAPE controls with G/sigma frozen (card 4ec1331b).

The parent card established the G policy: fit G/sigma on a cohort objective and
report G as a RANGE with subject-to-subject spread. This driver implements the
next step: with G/sigma chosen on the cohort, FREEZE them and backprop only the
shape controls (W_q, W_y, kv_scale, tau) through the exact same differentiable
window the single-subject trainer uses (vbjax.app.wb_train.window_loss).

It starts from a supplied SHAPE checkpoint (the one the cohort fit used,
wb_arm_ema_yout), overwrites G and sigma with the frozen cohort values, and
BPTTs W_q/W_y/kv_scale/log_tau over the FIT subjects, chaining the carry across
windows within each subject and persisting it across iterations (--persist-carry
semantics, the config that made coupling live). log_noise_tc and G/sigma are
pinned (the card lists exactly the four shape controls); drive-norm is NOT used
(it was falsified).

The trained checkpoint is evaluated OUTSIDE this driver with the certified
2400 s strict rule and the low-rank structural metrics, both via the single-point
confirm path of wb_cohort_fit.py on the HELD-OUT split (see the runner).

  python -m vbjax.app.wb_cohort_train --cohort ~/mhsa_e0/data/kb_cohort \
      --params ~/mhsa_e0/runs_persist/wb_arm_ema_yout/params.pkl \
      --g-init 4.0 --sigma-init 0.005 --loss-mode current \
      --out runs/wb_cohort_bptt_current
"""
import argparse, json, os, pickle, time
import numpy as onp
import jax
import jax.numpy as np

from vbjax.ct_mhsa import Hyperparameters, init_ct_mhsa
from vbjax.app.wb_train import Adam, window_loss, zscore, build_mhsa
from vbjax.app.wb_cohort_fit import load_cohort, fc_of

# Keys held exactly at their frozen values in BOTH the momentum state and the
# params dict; the driver only ever optimises the four shape controls.
FROZEN = {"G_rel", "log_G", "log_sigma", "log_noise_tc"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cohort", required=True)
    p.add_argument("--params", required=True,
                   help="SHAPE checkpoint to initialise from (e.g. ema_yout)")
    p.add_argument("--out", default="runs/wb_cohort_bptt")
    p.add_argument("--dt", type=float, default=2.0)
    p.add_argument("--tr", type=float, default=2.0)
    p.add_argument("--dt-train", type=float, default=100.0)
    p.add_argument("--v-c", type=float, default=5.0)
    p.add_argument("--d-model", type=int, default=16)
    p.add_argument("--win", type=int, default=64)
    p.add_argument("--k-fit-train", type=int, default=6,
                   help="number of FIT subjects to train on (first of the cohort)")
    p.add_argument("--g-init", type=float, default=4.0,
                   help="frozen loop gain G (cohort-fitted)")
    p.add_argument("--sigma-init", type=float, default=0.005,
                   help="frozen noise sigma (cohort-fitted)")
    p.add_argument("--loss-mode", default="current",
                   choices=["current", "sym_win", "gscale", "amp", "fcy",
                            "fce", "fcy_noreg", "fc2", "fc2r"],
                   help="objective; fc2/fc2r add an explicit FC-structure term")
    p.add_argument("--loss-amp-lambda", type=float, default=0.5)
    p.add_argument("--coupling", default="ema", choices=["ema", "raw"])
    p.add_argument("--iters", type=int, default=60)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    bold, Cs, Ls = load_cohort(a.cohort, a.v_c, a.dt, rho_norm=True)
    K, T, N = bold.shape
    print(f"[cohort_bptt] K={K} T={T} N={N} fit_train={a.k_fit_train} "
          f"frozen G={a.g_init} sigma={a.sigma_init} "
          f"loss_mode={a.loss_mode} coupling={a.coupling} iters={a.iters} "
          f"lr={a.lr}", flush=True)

    # ---- shape checkpoint init, force pinned G parameterisation (G_rel)
    with open(a.params, "rb") as f:
        base = pickle.load(f)
    params = {}
    for k in ("W_q", "W_y", "kv_scale", "log_tau", "log_noise_tc"):
        if k not in base:
            raise SystemExit(f"shape checkpoint missing {k!r}")
        params[k] = np.asarray(base[k])
    params["log_sigma"] = np.asarray(onp.log(a.sigma_init))
    params.pop("log_G", None)
    params["G_rel"] = np.asarray(a.g_init)

    print(f"[cohort_bptt] init shape from {a.params}", flush=True)
    for k, v in params.items():
        av = np.asarray(v)
        print(f"  {k}: shape={av.shape} "
              f"val={'%.5g' % float(av) if av.ndim == 0 else 'matrix'}",
              flush=True)

    d = a.d_model
    hp0 = Hyperparameters(n_regions=N, n_heads=N, d_model=d, d_k=d, d_v=d,
                          lam=1.0, dt=a.dt, tau=1e6, leak=1.0, act="relu",
                          steps_per_token=1, sigma_a=0.0)

    fit = list(range(min(a.k_fit_train, K)))
    subj = []          # per subject: (C, lags, L, z_wins, raw_wins, n_win, fc_emp)
    for k in fit:
        C = np.asarray(Cs[k]); lags = np.asarray(Ls[k])
        l = int(lags.max()) + 1
        ts = bold[k]
        n_win = ts.shape[0] // a.win
        zw = zscore(ts)[:n_win * a.win].reshape(n_win, a.win, N)
        rw = ts[:n_win * a.win].reshape(n_win, a.win, N)
        subj.append((C, lags, l, zw, rw, n_win, np.asarray(fc_of(ts))))
        print(f"[cohort_bptt] fit subj {k}: n_win={n_win}", flush=True)

    # per-subject persisted carry (matching ema_yout's --persist-carry)
    def full_carry():
        key = jax.random.PRNGKey(a.seed)
        _, st, _ = init_ct_mhsa(hp0, key, batch_size=1)
        return (st.M, np.zeros((N, d)), np.ones((N, 1)),
                np.zeros((N, d)), np.ones((N, 1)))
    carries = [full_carry() for _ in fit]

    # ---- one window of one subject
    def ws_window(p_in, carry, si, wi, kw):
        C, lags, L, zw, rw, _, fc_emp = subj[si]
        aux = {"fc_emp": fc_emp, "pulse_sched": None, "pulse_amp": 0.0}
        return window_loss(p_in, carry, zw[wi], rw[wi], kw, C, lags, L, hp0,
                           a.dt_train, a.loss_mode, a.tr * 1000.0,
                           a.loss_amp_lambda, a.coupling, aux, False)

    def forward_all(p_in, carries_in):
        """Stop-grad carries after chaining every window of every fit subject."""
        keys = jax.random.split(jax.random.PRNGKey(a.seed), len(fit))
        out = []
        for i in range(len(fit)):
            c = carries_in[i]
            nw = subj[i][5]
            for wi in range(nw):
                k_w = jax.random.fold_in(keys[i], wi)
                c, _ = ws_window(p_in, c, i, wi, k_w)
            out.append(jax.lax.stop_gradient(c))
        return tuple(out)

    def loss_fn(p_in, carries_in):
        keys = jax.random.split(jax.random.PRNGKey(a.seed), len(fit))
        total = np.asarray(0.0)
        n_total = 0
        for i in range(len(fit)):
            c = carries_in[i]
            nw = subj[i][5]
            for wi in range(nw):
                k_w = jax.random.fold_in(keys[i], wi)
                c, l = ws_window(p_in, c, i, wi, k_w)
                total = total + l
                n_total += 1
        return total / n_total

    fwd_all = jax.jit(forward_all)

    # Adam only sees the trainable (shape) keys; frozen keys never enter it.
    trainable = {k: v for k, v in params.items() if k not in FROZEN}
    opt = Adam(trainable, lr=a.lr)
    log = []

    t_start = time.time()
    for it in range(a.iters):
        t0 = time.time()
        key, _ = jax.random.split(jax.random.PRNGKey(a.seed))
        val, grads = jax.value_and_grad(
            lambda pp: loss_fn(pp, tuple(carries)))(params)
        v = float(val)
        finite = bool(onp.isfinite(v)) and all(
            bool(onp.isfinite(onp.asarray(g)).all()) for g in grads.values())
        if not finite:
            print(f"iter {it:4d} SKIP non-finite loss={v} ({time.time()-t0:.1f}s)",
                  flush=True)
            log.append({"iter": it, "skipped": True, "loss": None})
            continue
        # pin the frozen keys' gradients to zero so they cannot move
        for k in FROZEN:
            if k in grads:
                grads[k] = np.zeros_like(grads[k])
        gnorm = float(np.sqrt(sum(float(np.sum(g ** 2)) for g in grads.values())))
        if gnorm > 1.0:
            grads = {k: g / gnorm for k, g in grads.items()}
        # update only the trainable sub-dict with Adam
        g_trainable = {k: grads[k] for k in trainable}
        new_train = opt.step(trainable, g_trainable)
        params = {k: (new_train[k] if k in new_train else params[k])
                  for k in params}
        trainable = new_train
        # persist carry (stop-grad forward) for the next iteration
        carries = list(fwd_all(params, tuple(carries)))
        G_val = float(np.asarray(params["G_rel"]))
        msg = (f"iter {it:4d}/{a.iters} loss={v:.5f} "
               f"G={G_val:.3f}(frozen) "
               f"tau={float(np.asarray(np.exp(params['log_tau']))):.1f}s "
               f"sigma={float(np.asarray(np.exp(params['log_sigma']))):.4f}(frozen) "
               f"kv_scale={float(np.asarray(params['kv_scale'])):.4f} "
               f"gnorm={gnorm:.2e} ({time.time()-t0:.1f}s)")
        print(msg, flush=True)
        log.append({"iter": it, "loss": v, "gnorm": gnorm,
                    "kv_scale": float(np.asarray(params['kv_scale'])),
                    "tau_s": float(np.asarray(np.exp(params['log_tau']))),
                    "G": G_val, "sigma": float(np.asarray(np.exp(params['log_sigma'])))})
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, "log.json"), "w") as f:
            json.dump(log, f, indent=1)
        with open(os.path.join(a.out, "params_ckpt.pkl"), "wb") as f:
            pickle.dump(params, f)

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "params.pkl"), "wb") as f:
        pickle.dump(params, f)
    with open(os.path.join(a.out, "results.json"), "w") as f:
        json.dump({"learned": {k: float(np.asarray(v)) for k, v in params.items()
                               if np.asarray(v).ndim == 0},
                   "shape_frozen_G": a.g_init, "shape_frozen_sigma": a.sigma_init,
                   "loss_mode": a.loss_mode, "iters": a.iters,
                   "log": log}, f, indent=1)
    print(f"WB_COHORT_BPTT_DONE iters={a.iters} loss_final={v:.5f} "
          f"elapsed={time.time()-t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
