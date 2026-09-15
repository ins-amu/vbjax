"""Truncated-BPTT trainer for the whole-brain CT-MHSA resting-state BOLD model.

Owner design: reshape time into batches of 64 TR volumes (128 s each); BPTT
through each window with model state carried across windows (stop_gradient at
boundaries); everything else vectorized. Gradients flow to ALL tunable
parameters: W_q, W_y, kv_scale, G_rel, tau, sigma, noise_tc.

Integration: dt_train = 20 ms by default, so a 64-volume window is 6400
integration steps — tractable with jax.checkpoint (recompute-on-backward).
Tract delays resolve as floor(length/v_c/dt); keep dt consistent between
training and evaluation.

Loss: per-window MSE between simulated BOLD (balloon applied to the z-scored
prediction-error signal, sampled at TR) and empirical BOLD, both z-scored per
region. Evaluation metric: fc_match on the full simulation (never in the loss).

Stage 1 trains the scalars only (weights' gradients zeroed); stage 2 trains
everything. Hand-rolled Adam with global-norm clipping (no optax dependency).

  python -m vbjax.app.wb_train --sc SC.npy --lengths LEN.npy \
      --ts1 subj1.npy --ts2 subj2.npy --out runs/wb_train
  python -m vbjax.app.wb_train ... --eval-only --params runs/wb_train/params.pkl
"""
import argparse, os, pickle, json, time
import numpy as onp
import jax
import jax.numpy as np

from vbjax import make_bold, bold_default_theta, make_offline
from vbjax.ct_mhsa import Hyperparameters, NetworkState, init_ct_mhsa, mhsa_step
from vbjax.app.wholebrain_fc import load_connectome, fc_from_ts, fc_match


def zscore(x):
    return (x - x.mean(0, keepdims=True)) / (x.std(0, keepdims=True) + 1e-8)


def build_mhsa(hp0, key, W_q, W_y, kv_scale):
    mhsa, _, _ = init_ct_mhsa(hp0, key, batch_size=1)
    return mhsa._replace(W_Q=W_q, W_Y=W_y,
                         W_K=mhsa.W_K * kv_scale, W_V=mhsa.W_V * kv_scale)


def window_loss(params, carry, emp_z_win, keys, C, lags, L, hp0, dt_ms):
    """One window: differentiable neural sim + balloon + MSE.

    params: dict(W_q, W_y, kv_scale, G_rel, log_tau, log_sigma, log_noise_tc)
    carry:  (M (N,H,Dv,Dk), n_prev (N,D), y_ema (N,1), y_prev (N,D))
    Returns (new_carry (stop-grad), loss, bold_vol)."""
    N = C.shape[0]
    D = params["W_q"].shape[-1]
    G = params["G_rel"]
    tau_s = np.exp(params["log_tau"])
    sigma = np.exp(params["log_sigma"])
    noise_tc = np.exp(params["log_noise_tc"])
    dt_s = dt_ms / 1000.0
    u = dt_s / tau_s
    noise_a = np.exp(-dt_s / noise_tc)
    sq1 = np.sqrt(1.0 - noise_a ** 2 + 1e-12)
    mhsa = build_mhsa(hp0, jax.random.PRNGKey(0), params["W_q"], params["W_y"],
                      params["kv_scale"])
    hp_eff = hp0._replace(lam=u, tau=tau_s * 1000.0)
    win_vols = emp_z_win.shape[0]
    steps_per_vol = int(round(2000.0 / dt_ms))
    T_int = win_vols * steps_per_vol
    ar = np.arange(N)
    M0, n0, fe0, y0 = carry[0], carry[1], carry[2], carry[3]  # carry[0] = M array

    # NOTE: coupling inside the differentiable window uses the previous-step
    # output (causal, instantaneous) instead of the tract-delay history buffer:
    # dynamic fancy-indexing of a traced buffer breaks jax.checkpoint remat
    # (negative-index bounds checks become tracer bool conversions). Delays
    # (ms-scale) are kept in the evaluation simulation; the FC-relevant
    # dynamics live at seconds scale, so this approximation is benign.
    def neural_step(carry_n, key_t):
        M, n_prev, y_ema, y_prev = carry_n
        state = NetworkState(M=M, history=None, step=0)
        aff = G * np.einsum("ij,jd->id", C, y_prev)
        eps = jax.random.normal(key_t, (N, D))
        noise = noise_a * n_prev + sigma * sq1 * eps
        x = (aff + noise)[None]
        state, y, e = mhsa_step(mhsa, state, x, hp_eff)
        y = y[0]
        e = e[0].mean(-1)                                    # (N,) prediction error
        rms = np.sqrt(np.mean(y ** 2, axis=-1, keepdims=True))
        y_ema = 0.999 * y_ema + 0.001 * rms
        y_out = y / (y_ema + 1e-8)
        return (state.M, noise, y_ema, y), e

    # Chunked remat: checkpoint at the 100-step chunk level (inner scan runs
    # uncheckpointed). Per-step checkpoint over thousands of steps OOMs on the
    # gradient reduction; per-chunk keeps backward memory ~100x smaller.
    chunk = 25
    n_chunks = T_int // chunk
    keys_w = jax.random.split(keys, T_int)
    keys_c = keys_w.reshape(n_chunks, chunk, 2)

    def chunk_step(carry_c, keys_chunk):
        return jax.lax.scan(neural_step, carry_c, keys_chunk)

    (fM, fn, fe, fyp), e_chunks = jax.lax.scan(
        jax.checkpoint(chunk_step), (M0, n0, fe0, y0), keys_c)
    e_seq = e_chunks.reshape(n_chunks * chunk, N)            # (T_int, N)

    # Balloon at dt_train (20 ms) destabilizes (V**alpha NaN) — the training
    # loss fits the NEURAL prediction-error trace: average e within each TR
    # volume, z-score, match to empirical BOLD. The balloon stays in the
    # fc_match evaluation (bold_forward at fine dt).
    e_vol = e_seq.reshape(win_vols, steps_per_vol, N).mean(axis=1)   # (win_vols, N)
    bold_vol = zscore(e_vol)

    loss = np.mean((bold_vol - emp_z_win) ** 2)
    new_carry = (jax.lax.stop_gradient(fM),
                 jax.lax.stop_gradient(fn),
                 jax.lax.stop_gradient(fe),
                 jax.lax.stop_gradient(fyp))
    return new_carry, loss, bold_vol


class Adam:
    def __init__(self, params, lr):
        self.lr = lr
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.b1, self.b2, self.eps, self.t = 0.9, 0.999, 1e-8, 0

    def reset(self):
        """Zero the moment state. Called when a step is discarded so a
        poisoned m/v (from a non-finite gradient) cannot re-inject NaN."""
        self.m = {k: np.zeros_like(v) for k, v in self.m.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.v.items()}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        out = {}
        for k in params:
            g = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g ** 2
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            out[k] = params[k] - self.lr * mh / (np.sqrt(vh) + self.eps)
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sc", required=True)
    ap.add_argument("--lengths", required=True)
    ap.add_argument("--ts1", required=True)
    ap.add_argument("--ts2", required=True)
    ap.add_argument("--out", default="runs/wb_train")
    ap.add_argument("--dt", type=float, default=2.0, help="eval integration step (ms)")
    ap.add_argument("--dt-train", type=float, default=100.0,
                    help="training integration step (ms); coarse dt keeps the BPTT window shallow" )
    ap.add_argument("--tr", type=float, default=2.0)
    ap.add_argument("--v-c", type=float, default=5.0)
    ap.add_argument("--d-model", type=int, default=16)
    ap.add_argument("--win", type=int, default=64, help="window size in TR volumes")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--final-sim-s", type=float, default=600.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--params", default=None)
    args = ap.parse_args()

    C, lags, L = load_connectome(args.sc, args.lengths, args.dt, args.v_c, "sqrt99")
    N = C.shape[0]
    rho = float(onp.max(onp.abs(onp.linalg.eigvals(C))))
    C = C / rho
    ts1 = onp.load(args.ts1); ts2 = onp.load(args.ts2)
    fc1, fc2 = fc_from_ts(ts1), fc_from_ts(ts2)
    iu = onp.triu_indices(N, 1)
    d = args.d_model
    hp0 = Hyperparameters(n_regions=N, n_heads=N, d_model=d, d_k=d, d_v=d,
                          lam=1.0, dt=args.dt, tau=1e6, leak=1.0, act="relu",
                          steps_per_token=1, sigma_a=0.0)
    key = jax.random.PRNGKey(args.seed)
    k_m, _ = jax.random.split(key)
    mhsa0, _, _ = init_ct_mhsa(hp0, k_m, batch_size=1)

    params = {"W_q": mhsa0.W_Q, "W_y": mhsa0.W_Y,
              "kv_scale": np.asarray(0.15), "G_rel": np.asarray(2.09),
              "log_tau": np.asarray(onp.log(36.0)),
              "log_sigma": np.asarray(onp.log(0.18)),
              "log_noise_tc": np.asarray(onp.log(2.0))}

    n_win = ts1.shape[0] // args.win
    emp_z_wins = zscore(ts1)[:n_win * args.win].reshape(n_win, args.win, N)
    L_win = max(int(lags.max()), 1) + 1

    def full_carry(params):
        _, st, _ = init_ct_mhsa(hp0, key, batch_size=1)
        n_prev = np.zeros((N, d))
        y_ema = np.ones((N, 1))
        y_prev = np.zeros((N, d))
        return (st.M, n_prev, y_ema, y_prev)

    def losses_all_windows(params, carry, kw):
        losses = []
        c = carry
        for w in range(emp_z_wins.shape[0]):
            k_w = jax.random.fold_in(kw, w)
            c, loss, _ = window_loss(params, c, emp_z_wins[w], k_w,
                                     C, lags, L_win, hp0, args.dt_train)
            losses.append(loss)
        return c, np.mean(np.stack(losses))

    def eval_fc(params, T_s, ts_fc, seed):
        tau_s = float(np.exp(params["log_tau"]))
        sigma = float(np.exp(params["log_sigma"]))
        hp = hp0._replace(lam=args.dt / (tau_s * 1000.0), tau=tau_s * 1000.0)
        mhsa = build_mhsa(hp, jax.random.PRNGKey(seed), params["W_q"],
                          params["W_y"], float(params["kv_scale"]))
        from vbjax.app.wholebrain_fc import simulate, bold_forward
        e, _ = simulate(mhsa, np.asarray(C), jax.numpy.asarray(lags), L, hp,
                        float(params["G_rel"]), sigma, int(T_s * 1000 / args.dt),
                        jax.random.PRNGKey(seed),
                        float(np.exp(params["log_noise_tc"])), args.dt, "e")
        e = onp.asarray(e)
        if not onp.isfinite(e).all():
            return None
        bold = bold_forward(zscore(e), args.dt, args.tr)
        return fc_match(fc_from_ts(bold), ts_fc)

    os.makedirs(args.out, exist_ok=True)

    if args.eval_only:
        with open(args.params, "rb") as f:
            params = pickle.load(f)
        s1 = eval_fc(params, args.final_sim_s, fc1, args.seed)
        s2 = eval_fc(params, args.final_sim_s, fc2, args.seed + 1)
        print(json.dumps({"subj1": s1, "subj2": s2}), flush=True)
        return

    opt = Adam(params, lr=1e-3)
    opt_s = Adam(params, lr=1e-2)
    MIN_LR = 1e-5
    BAD_TO_HALVE = 3
    iters1, iters2 = args.iters // 2, args.iters - args.iters // 2
    log = []
    last_good = {k: v for k, v in params.items()}
    bad_streak = 0
    for it in range(iters1 + iters2):
        stage_all = it >= iters1
        cur_opt = opt if stage_all else opt_s
        t0 = time.time()
        key, kw = jax.random.split(key)
        carry = full_carry(params)

        def loss_fn(p):
            _, l = losses_all_windows(p, carry, kw)
            return l

        val, grads = jax.value_and_grad(loss_fn)(params)
        v = float(val)
        # Guard the STEP, not just the loss: a finite loss can still carry a
        # non-finite gradient. If either is non-finite, discard this step
        # entirely so neither params nor the Adam moments get poisoned.
        finite = bool(onp.isfinite(v)) and all(
            bool(onp.isfinite(onp.asarray(g)).all()) for g in grads.values())
        if not finite:
            bad_streak += 1
            lr_now = cur_opt.lr
            cur_opt.reset()          # drop any poisoned m/v moment state
            print(f"iter {it:4d}/{args.iters} SKIP non-finite loss/grad "
                  f"loss={v} bad_streak={bad_streak} lr={lr_now:.2e} "
                  f"({time.time()-t0:.1f}s)", flush=True)
            if bad_streak >= BAD_TO_HALVE:
                cur_opt.lr = max(lr_now * 0.5, MIN_LR)
                bad_streak = 0
                print(f"iter {it:4d} {BAD_TO_HALVE} consecutive bad steps -> "
                      f"lr={cur_opt.lr:.2e} (floor {MIN_LR:.0e})", flush=True)
            log.append({"iter": it, "skipped": True, "loss": None,
                        "bad_streak": bad_streak, "lr": cur_opt.lr})
            with open(os.path.join(args.out, "log.json"), "w") as f:
                json.dump(log, f, indent=1)
            continue
        bad_streak = 0
        last_good = {k: v2 for k, v2 in params.items()}
        if not stage_all:
            grads = {k: (np.zeros_like(g) if k in ("W_q", "W_y") else g)
                     for k, g in grads.items()}
        gnorm = float(np.sqrt(sum(float(np.sum(g ** 2)) for g in grads.values())))
        if gnorm > 1.0:
            grads = {k: g / gnorm for k, g in grads.items()}
        params = cur_opt.step(params, grads)
        msg = (f"iter {it:4d}/{args.iters} loss={v:.5f} "
               f"G={float(params['G_rel']):.2f} "
               f"tau={float(np.exp(params['log_tau'])):.1f}s "
               f"sigma={float(np.exp(params['log_sigma'])):.3f} "
               f"gnorm={gnorm:.2e} lr={cur_opt.lr:.2e} ({time.time()-t0:.1f}s)")
        print(msg, flush=True)
        log.append({"iter": it, "loss": v, "gnorm": gnorm,
                    "lr": cur_opt.lr})
        with open(os.path.join(args.out, "log.json"), "w") as f:
            json.dump(log, f, indent=1)
        with open(os.path.join(args.out, "params_ckpt.pkl"), "wb") as f:
            pickle.dump(params, f)
    s1 = eval_fc(params, args.final_sim_s, fc1, args.seed)
    s2 = eval_fc(params, args.final_sim_s, fc2, args.seed + 1)
    with open(os.path.join(args.out, "params.pkl"), "wb") as f:
        pickle.dump(params, f)
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump({"fc1": s1, "fc2_holdout": s2,
                   "learned": {k: float(v) for k, v in params.items()
                               if k not in ("W_q", "W_y")}, "log": log}, f, indent=1)
    print(f"DONE fc1={s1} fc2_holdout={s2}", flush=True)


if __name__ == "__main__":
    main()
