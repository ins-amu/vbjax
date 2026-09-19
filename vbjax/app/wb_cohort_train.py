"""Cohort BPTT of the SHAPE controls with G/sigma frozen (card 4ec1331b).

The parent card established the G policy: fit G/sigma on a cohort objective and
report G as a RANGE with subject-to-subject spread. This driver implements the
next step: with G/sigma chosen on the cohort, FREEZE them and backprop only the
shape controls (W_q, W_y, kv_scale, tau) through the exact same differentiable
window the single-subject trainer uses (vbjax.app.wb_train.window_loss).

It starts from a supplied SHAPE checkpoint (the one the cohort fit used,
wb_arm_ema_yout), overwrites G and sigma with the frozen cohort values, and
BPTTs the four shape controls over the FIT subjects, with the carry persisted
across iterations (--persist-carry semantics, the config that made coupling
live). log_noise_tc and G/sigma are pinned (the card lists exactly the four
shape controls); drive-norm is NOT used (it was falsified).

Freezing is structural: gradients are computed ONLY with respect to the four
trainable keys (the frozen values are constants in the forward).

Robustness: window_loss returns a stop-grad carry, so every window is an
independent gradient term. Measured (wb_cohort_diag): a single data segment can
NaN its scalar-channel (kv_scale/log_tau) gradient even though the forward loss
is finite (subject 0 window 3, at every (G, sigma)). We therefore compute
gradients per window and average only over non-finite windows, so one
pathological segment cannot stop all training. The persisted carry is advanced
with the *post-step* params via a single forward pass each iteration.

The trained checkpoint is evaluated OUTSIDE this driver with the certified
2400 s strict rule and the low-rank structural metrics, both via
wb_cohort_score.py on the HELD-OUT split (see the runner).

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
from vbjax.app.wb_train import Adam, window_loss, zscore
from vbjax.app.wb_cohort_fit import load_cohort, fc_of

# The four shape controls that are BPTT'd. Everything else is a frozen constant.
TRAINABLE = ("W_q", "W_y", "kv_scale", "log_tau")


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

    # ---- shape checkpoint init; separate TRAINABLE from FROZEN constants
    with open(a.params, "rb") as f:
        base = pickle.load(f)
    trainable = {}
    for k in TRAINABLE:
        if k not in base:
            raise SystemExit(f"shape checkpoint missing {k!r}")
        trainable[k] = np.asarray(base[k])
    frozen = {
        "log_sigma": np.asarray(onp.log(a.sigma_init)),
        "log_noise_tc": np.asarray(base["log_noise_tc"]),
        "G_rel": np.asarray(a.g_init),
    }

    def merge(tt):
        pp = dict(tt); pp.update(frozen)
        return pp

    print(f"[cohort_bptt] init shape from {a.params}", flush=True)
    print(f"[cohort_bptt] BPTT targets: {list(TRAINABLE)}", flush=True)
    print(f"[cohort_bptt] frozen: G={a.g_init} (G_rel)  "
          f"sigma={a.sigma_init}  noise_tc="
          f"{float(np.exp(np.asarray(frozen['log_noise_tc']))):.4f}s", flush=True)
    for k, v in trainable.items():
        av = np.asarray(v)
        print(f"  {k}: shape={av.shape} "
              f"val={'%.5g' % float(av) if av.ndim == 0 else 'matrix'}",
              flush=True)

    d = a.d_model
    hp0 = Hyperparameters(n_regions=N, n_heads=N, d_model=d, d_k=d, d_v=d,
                          lam=1.0, dt=a.dt, tau=1e6, leak=1.0, act="relu",
                          steps_per_token=1, sigma_a=0.0)

    fit = list(range(min(a.k_fit_train, K)))
    # windows[(i, wi)] = (C, lags, L, zw_win, rw_win, fc_emp, kw)
    windows = {}
    for k in fit:
        C = np.asarray(Cs[k]); lags = np.asarray(Ls[k])
        L = int(lags.max()) + 1
        ts = bold[k]
        n_win = ts.shape[0] // a.win
        zw = zscore(ts)[:n_win * a.win].reshape(n_win, a.win, N)
        rw = ts[:n_win * a.win].reshape(n_win, a.win, N)
        fc_emp = np.asarray(fc_of(ts))
        for wi in range(n_win):
            kw = jax.random.fold_in(jax.random.PRNGKey(a.seed + k), wi)
            windows[(k, wi)] = (C, lags, L, zw[wi], rw[wi], fc_emp, kw)
        print(f"[cohort_bptt] fit subj {k}: n_win={n_win}", flush=True)
    n_subj = len(fit)
    n_win = bold.shape[1] // a.win
    win_index = [(i, wi) for i in fit for wi in range(n_win)]

    # per-subject persisted carry (matching ema_yout's --persist-carry)
    def full_carry():
        key = jax.random.PRNGKey(a.seed)
        _, st, _ = init_ct_mhsa(hp0, key, batch_size=1)
        return (st.M, np.zeros((N, d)), np.ones((N, 1)),
                np.zeros((N, d)), np.ones((N, 1)))
    carries = {i: full_carry() for i in fit}

    # ---- one window (data passed as args so the jit key is shape-only and the
    # compile is reused across every window of identical shapes)
    def _one_window(pp, carry, C, lags, L, zw_win, rw_win, fc_emp, kw):
        aux = {"fc_emp": fc_emp, "pulse_sched": None, "pulse_amp": 0.0}
        c2, l = window_loss(merge(pp), carry, zw_win, rw_win, kw, C, lags, L,
                            hp0, a.dt_train, a.loss_mode, a.tr * 1000.0,
                            a.loss_amp_lambda, a.coupling, aux, False)
        return l, jax.lax.stop_gradient(c2)

    val_loss = jax.jit(lambda pp, c, C, lags, L, z, r, f, kw: _one_window(
        pp, c, C, lags, L, z, r, f, kw)[0])

    def forward_all(pp, carries_in):
        """Advance every persisted carry with the given params."""
        out = {}
        for i in fit:
            c = carries_in[i]
            for wi in range(n_win):
                C, lags, L, z, r, f, kw = windows[(i, wi)]
                _, c = _one_window(pp, c, C, lags, L, z, r, f, kw)
            out[i] = jax.lax.stop_gradient(c)
        return out

    fwd_all = jax.jit(forward_all)

    # ---- training loop: per-window grads, keep only finite windows
    opt = Adam(trainable, lr=a.lr)
    log = []

    t_start = time.time()
    for it in range(a.iters):
        t0 = time.time()
        acc = {k: np.zeros_like(v) for k, v in trainable.items()}
        n_finite = 0
        sum_loss = 0.0
        bad = []
        for (i, wi) in win_index:
            C, lags, L, z, r, f, kw = windows[(i, wi)]
            c = carries[i]
            v, g = jax.value_and_grad(val_loss)(trainable, c, C, lags, L, z,
                                                r, f, kw)
            vf = float(v)
            gf = all(bool(onp.isfinite(onp.asarray(g[k])).all())
                     for k in g) and onp.isfinite(vf)
            if gf:
                for k in acc:
                    acc[k] = acc[k] + g[k]
                n_finite += 1
                sum_loss += vf
            else:
                bad.append((i, wi, vf))
        n_win_total = len(win_index)
        if n_finite == 0:
            print(f"iter {it:4d}/{a.iters} ALL {n_win_total} windows non-finite "
                  f"- skip ({time.time()-t0:.1f}s)", flush=True)
            log.append({"iter": it, "skipped": True, "loss": None,
                        "n_finite": 0, "n_windows": n_win_total})
            continue
        v = sum_loss / n_finite
        grads = {k: acc[k] / n_finite for k in acc}
        gnorm = float(np.sqrt(sum(float(np.sum(g ** 2)) for g in grads.values())))
        if gnorm > 1.0:
            grads = {k: g / gnorm for k, g in grads.items()}
        trainable = opt.step(trainable, grads)
        carries = fwd_all(merge(trainable), carries)
        G_val = float(np.asarray(frozen["G_rel"]))
        msg = (f"iter {it:4d}/{a.iters} loss={v:.5f} "
               f"G={G_val:.3f}(frozen) "
               f"tau={float(np.asarray(np.exp(trainable['log_tau']))):.1f}s "
               f"sigma={float(np.asarray(np.exp(frozen['log_sigma']))):.4f}(frozen) "
               f"kv_scale={float(np.asarray(trainable['kv_scale'])):.4f} "
               f"gnorm={gnorm:.2e} kept={n_finite}/{n_win_total} "
               f"({time.time()-t0:.1f}s)")
        print(msg, flush=True)
        log.append({"iter": it, "loss": v, "gnorm": gnorm,
                    "kv_scale": float(np.asarray(trainable['kv_scale'])),
                    "tau_s": float(np.asarray(np.exp(trainable['log_tau']))),
                    "G": G_val, "sigma": float(np.asarray(np.exp(frozen['log_sigma']))),
                    "n_finite": n_finite, "n_windows": n_win_total,
                    "skipped_windows": bad})
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, "log.json"), "w") as f:
            json.dump(log, f, indent=1)
        with open(os.path.join(a.out, "params_ckpt.pkl"), "wb") as f:
            pickle.dump(merge(trainable), f)

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "params.pkl"), "wb") as f:
        pickle.dump(merge(trainable), f)
    with open(os.path.join(a.out, "results.json"), "w") as f:
        json.dump({"learned": {k: float(np.asarray(v)) for k, v in trainable.items()
                               if np.asarray(v).ndim == 0},
                   "shape_frozen_G": a.g_init, "shape_frozen_sigma": a.sigma_init,
                   "loss_mode": a.loss_mode, "iters": a.iters,
                   "log": log}, f, indent=1)
    print(f"WB_COHORT_BPTT_DONE iters={a.iters} loss_final={v:.5f} "
          f"elapsed={time.time()-t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
