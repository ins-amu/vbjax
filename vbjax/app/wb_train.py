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


def bold_forward_jax(signal, dt_ms, tr_s):
    """Differentiable balloon: (T, N) neural drive at dt_ms -> BOLD sampled at TR.

    Same monitor as wholebrain_fc.bold_forward but returns a traced array, so the
    hemodynamic forward can live inside the training loss."""
    T, N = signal.shape
    every = int(round(tr_s * 1000.0 / dt_ms))
    W = T // every
    windowed = signal[:W * every].reshape(W, every, N)
    bold_buf, bold_step, bold_sample = make_bold(
        shape=(N,), dt=dt_ms / 1000.0, p=bold_default_theta)
    offline = make_offline(step_fn=bold_step, sample_fn=bold_sample)
    _, bold = jax.lax.scan(offline, bold_buf, windowed)
    return bold


def rescale_per_region(x, lo=0.2, hi=1.0):
    """Per-region affine map into [lo, hi] (balloon needs a non-negative input).

    Matches wholebrain_fc's pre-balloon rescale, which preserves the signal's
    correlation structure but quotients out per-region amplitude."""
    mn = x.min(0, keepdims=True)
    mx = x.max(0, keepdims=True)
    return lo + (hi - lo) * (x - mn) / (mx - mn + 1e-8)


def rescale_shared(x, lo=0.2, hi=1.0):
    """Single shared affine map into [lo, hi]: keeps per-region amplitude ratios,
    so the coupling gain stays observable through the balloon."""
    mn = x.min()
    mx = x.max()
    return lo + (hi - lo) * (x - mn) / (mx - mn + 1e-8)


def window_loss(params, carry, emp_z_win, emp_raw_win, keys, C, lags, L, hp0,
                dt_ms, mode="current", tr_ms=2000.0, lam_amp=0.5,
                coupling="ema", aux=None, drive_norm=False):
    """One window: differentiable neural sim + selectable loss.

    params: dict(W_q, W_y, kv_scale, G_rel, log_tau, log_sigma, log_noise_tc)
    carry:  (M (N,H,Dv,Dk), n_prev (N,D), y_ema (N,1), y_prev (N,D),
             aff_ema (N,1))
    drive_norm: per-region drive normalisation (fix 1). The raw afferent drive
        aff_raw = G (C @ y_prev) is divided by a stop-gradient EMA of its own
        per-region rms and rescaled by sigma*drive_beta (= params['drive_beta'],
        a learnable scalar). Without it the drive sits ~4-5 orders below the
        injected noise and the sum x = aff + noise destroys dL/dG.
    mode:   loss formulation (see --loss-mode); the shipped default is 'current'
    coupling: 'ema' = couple through the EMA-normalised output, as
        wholebrain_fc.simulate does; 'raw' = the historical trainer behaviour
        (raw y), which decouples the regions once kv_scale collapses
    Returns (new_carry (stop-grad), loss)."""
    N = C.shape[0]
    D = params["W_q"].shape[-1]
    G = np.exp(params["log_G"]) if "log_G" in params else params["G_rel"]
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
    a0 = carry[4]                                            # aff_ema (N,1)
    drive_beta = params["drive_beta"] if drive_norm else None

    # NOTE: coupling inside the differentiable window uses the previous-step
    # output (causal, instantaneous) instead of the tract-delay history buffer:
    # dynamic fancy-indexing of a traced buffer breaks jax.checkpoint remat
    # (negative-index bounds checks become tracer bool conversions). Delays
    # (ms-scale) are kept in the evaluation simulation; the FC-relevant
    # dynamics live at seconds scale, so this approximation is benign.
    def neural_step(carry_n, xs_t):
        key_t, p_t = xs_t
        M, n_prev, y_ema, y_prev, aff_ema = carry_n
        state = NetworkState(M=M, history=None, step=0)
        aff_raw = G * np.einsum("ij,jd->id", C, y_prev)
        if drive_norm:
            # Fix 1: per-region drive normalisation. The afferent drive is
            # rescaled to the noise scale by dividing by an EMA of its own
            # per-region rms and multiplying by sigma*drive_beta. The
            # normaliser is stop_gradient'ed on purpose: G and drive_beta still
            # receive gradient through the numerator and the beta factor, while
            # the normaliser cannot trivially cancel the scale (which would
            # re-hide G).
            rms_a = np.sqrt(np.mean(aff_raw ** 2, axis=-1, keepdims=True))
            aff_ema = 0.999 * aff_ema + 0.001 * rms_a
            aff = (aff_raw / (jax.lax.stop_gradient(aff_ema) + 1e-8)
                   * (sigma * drive_beta))
        else:
            aff = aff_raw
        eps = jax.random.normal(key_t, (N, D))
        noise = noise_a * n_prev + sigma * sq1 * eps
        x = (aff + noise + p_t[:, None])[None]
        state, y, e = mhsa_step(mhsa, state, x, hp_eff)
        y = y[0]
        e = e[0].mean(-1)                                    # (N,) prediction error
        rms = np.sqrt(np.mean(y ** 2, axis=-1, keepdims=True))
        y_ema = 0.999 * y_ema + 0.001 * rms
        # Afferent coupling input for the next step. wholebrain_fc.simulate
        # couples through the EMA-normalised output y_out; coupling through raw
        # y made the drive ~1e-9 against O(sigma) noise, i.e. the regions were
        # effectively independent and G_rel had no gradient (~1e-13).
        y_coupled = y / (y_ema + 1e-8) if coupling == "ema" else y
        return ((state.M, noise, y_ema, y_coupled, aff_ema),
                (e, np.mean(y_coupled, -1)))

    # Chunked remat: checkpoint at the chunk level (inner scan runs
    # uncheckpointed). Per-step checkpoint over thousands of steps OOMs on the
    # gradient reduction; per-chunk keeps backward memory far smaller.
    # The chunk must tile T_int exactly (it is only a checkpoint grouping, so
    # the scan arithmetic is identical for any chunk). Default 25 does not
    # divide T_int=1280 (64 vols x 20 steps), so fall back to the largest
    # divisor of T_int <= 25, which is strictly cheaper in backward memory.
    chunk = 25
    if T_int % chunk != 0:
        chunk = max(c for c in range(1, chunk + 1) if T_int % c == 0)
    n_chunks = T_int // chunk
    keys_w = jax.random.split(keys, T_int)
    if aux is not None and aux.get("pulse_sched") is not None:
        sched = aux["pulse_sched"] * aux.get("pulse_amp", 1.0)
    else:
        sched = np.zeros((T_int, N))
    sched_c = sched.reshape(n_chunks, chunk, N)
    keys_c = keys_w.reshape(n_chunks, chunk, 2)

    def chunk_step(carry_c, xs_chunk):
        kc, sc = xs_chunk
        return jax.lax.scan(neural_step, carry_c, (kc, sc))

    (fM, fn, fe, fyp, fa), (e_chunks, y_chunks) = jax.lax.scan(
        jax.checkpoint(chunk_step), (M0, n0, fe0, y0, a0), (keys_c, sched_c))
    e_seq = e_chunks.reshape(n_chunks * chunk, N)            # (T_int, N)
    y_seq = y_chunks.reshape(n_chunks * chunk, N)            # (T_int, N)

    # Balloon at dt_train (20 ms) destabilizes (V**alpha NaN) — the training
    # loss fits the NEURAL prediction-error trace: average e within each TR
    # volume, z-score, match to empirical BOLD. The balloon stays in the
    # fc_match evaluation (bold_forward at fine dt).
    e_vol = e_seq.reshape(win_vols, steps_per_vol, N).mean(axis=1)   # (win_vols, N)

    if mode in ("fcy", "fce", "fcy_noreg", "fc2", "fc2r"):
        # FC-shaped objective: the loop gain lives in inter-region covariance,
        # which the per-region temporal MSE never rewards. Region traces: the
        # EMA-normalised output (fcy) or the surprise (fce), FC over the window
        # vs the empirical FC, loss = 1 - Pearson(upper triangles).
        def _fc_term(tr):
            trz = (tr - tr.mean(0)) / (tr.std(0) + 1e-8)
            fc_sim = trz.T @ trz / tr.shape[0]
            iu = np.triu_indices(N, 1)
            fu, tu = fc_sim[iu], aux["fc_emp"][iu]
            r = (np.mean(fu * tu) - np.mean(fu) * np.mean(tu)) / (
                np.std(fu) * np.std(tu) + 1e-8)
            return 1.0 - r

        base = np.mean((zscore(e_vol) - zscore(emp_raw_win)) ** 2)
        if mode == "fc2":
            loss = _fc_term(y_seq) + _fc_term(e_seq)
        elif mode == "fc2r":
            # fc2 plus a temporal-MSE regulariser: the same FC terms, anchored
            # to the empirical BOLD trace by the window MSE (weight lam_amp),
            # to test whether pinning the temporal fit helps or hurts FC.
            loss = _fc_term(y_seq) + _fc_term(e_seq) + lam_amp * base
        elif mode == "fcy_noreg":
            loss = _fc_term(y_seq)
        elif mode == "fce":
            loss = _fc_term(e_seq) + lam_amp * base
        else:
            loss = _fc_term(y_seq) + lam_amp * base
    else:
        if mode == "current":
            # as shipped: prediction z-scored per region per window, target globally
            # z-scored -> any per-region affine change of the prediction is invisible
            pred, tgt = zscore(e_vol), emp_z_win
        elif mode == "sym_win":
            pred, tgt = zscore(e_vol), zscore(emp_raw_win)
        elif mode == "gscale":
            # one shared scale: per-region amplitude ratios survive
            pred, tgt = e_vol / (np.std(e_vol) + 1e-8), emp_z_win
        elif mode in ("balloon_pr", "balloon_gs"):
            src = (rescale_per_region(e_seq) if mode == "balloon_pr"
                   else rescale_shared(e_seq))
            bold = bold_forward_jax(src, dt_ms, tr_ms)
            pred, tgt = zscore(bold), zscore(emp_raw_win[: bold.shape[0]])
        elif mode == "amp":
            pred, tgt = zscore(e_vol), zscore(emp_raw_win)
        else:
            raise ValueError(f"unknown loss mode {mode!r}")

        loss = np.mean((pred - tgt) ** 2)
        if mode == "amp":
            # explicit scale term: makes the loop gain observable through the
            # per-region dispersion, which the z-scored term discards
            ls_p = np.log(pred.std(0) + 1e-8)
            ls_t = np.log(tgt.std(0) + 1e-8)
            loss = loss + lam_amp * np.mean((ls_p - ls_t) ** 2)
    new_carry = (jax.lax.stop_gradient(fM),
                 jax.lax.stop_gradient(fn),
                 jax.lax.stop_gradient(fe),
                 jax.lax.stop_gradient(fyp),
                 jax.lax.stop_gradient(fa))
    return new_carry, loss


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


def _log_sens(fn, g):
    val, tang = jax.jvp(fn, (g,), (np.asarray(1.0),))
    nv = float(np.sqrt(np.sum(val ** 2)) + 1e-30)
    nt = float(np.sqrt(np.sum(tang ** 2)) + 1e-30)
    return nt / nv * abs(float(g))


def _null_frac(de, e_vol):
    """Fraction of d(e_vol)/dG lying in the kernel of the per-region z-score
    (span{1, e_vol_r}), i.e. readout sensitivity the loss cannot see."""
    resid, tot = 0.0, 0.0
    for r in range(de.shape[1]):
        dd = onp.asarray(de[:, r]); ee = onp.asarray(e_vol[:, r])
        X = onp.stack([onp.ones_like(ee), ee], axis=1)
        coef, *_ = onp.linalg.lstsq(X, dd, rcond=None)
        res = dd - X @ coef
        resid += float(res @ res); tot += float(dd @ dd)
    return (resid ** 0.5) / (tot ** 0.5 + 1e-30)


def run_attribution(C, hp0, params, ts1, win, dt_ms, burnin, coupling="ema",
                    drive_norm=False, seed=42):
    """Stage-by-stage dL/dG attribution on the trainer's own graph.

    Same metric set as vbjax/app/wb_gradflow.py, but honouring drive_norm: the
    frozen copy in wb_gradflow.py embeds its own un-normalised step (it only
    imports zscore/build_mhsa from here), so it cannot see the normaliser. This
    path keeps the attribution on exactly the graph the trainer optimises.
    'aff' is reported as the raw drive G(C@y_prev) in both cases, so its
    log-sensitivity stays comparable across arms.
    """
    N = C.shape[0]
    D = params["W_q"].shape[-1]
    G = np.exp(params["log_G"]) if "log_G" in params else params["G_rel"]
    tau_s = np.exp(params["log_tau"])
    sigma = np.exp(params["log_sigma"])
    noise_tc = np.exp(params["log_noise_tc"])
    beta = params["drive_beta"] if drive_norm else np.asarray(1.0)
    dt_s = dt_ms / 1000.0
    u = dt_s / tau_s
    noise_a = np.exp(-dt_s / noise_tc)
    sq1 = np.sqrt(1.0 - noise_a ** 2 + 1e-12)
    mhsa = build_mhsa(hp0, jax.random.PRNGKey(0), params["W_q"], params["W_y"],
                      params["kv_scale"])
    hp_eff = hp0._replace(lam=u, tau=tau_s * 1000.0)
    steps_per_vol = int(round(2000.0 / dt_ms))
    T_int = win * steps_per_vol
    n_win = ts1.shape[0] // win
    emp_raw = np.asarray(ts1[:n_win * win].reshape(n_win, win, N)[0])
    emp_z = np.asarray(zscore(ts1)[:n_win * win].reshape(n_win, win, N)[0])

    def f(p, carry, keys, burnin_n, inject=None, zscore_loss=True):
        Gp = np.exp(p["log_G"]) if "log_G" in p else p["G_rel"]
        bet = p["drive_beta"] if drive_norm else np.asarray(1.0)

        def step_core(carry_n, key_t, aff_raw):
            M, n_prev, y_ema, y_prev, aff_ema = carry_n
            state = NetworkState(M=M, history=None, step=0)
            if drive_norm:
                rms_a = np.sqrt(np.mean(aff_raw ** 2, axis=-1, keepdims=True))
                aff_ema = 0.999 * aff_ema + 0.001 * rms_a
                aff_t = (aff_raw / (jax.lax.stop_gradient(aff_ema) + 1e-8)
                         * (sigma * bet))
            else:
                aff_t = aff_raw
            eps = jax.random.normal(key_t, (N, D))
            noise = noise_a * n_prev + sigma * sq1 * eps
            x = (aff_t + noise)[None]
            state, y, e = mhsa_step(mhsa, state, x, hp_eff)
            y = y[0]
            e = e[0].mean(-1)
            rms_y = np.sqrt(np.mean(y ** 2, axis=-1, keepdims=True))
            y_ema = 0.999 * y_ema + 0.001 * rms_y
            y_coupled = y / (y_ema + 1e-8) if coupling == "ema" else y
            return ((state.M, noise, y_ema, y_coupled, aff_ema),
                    (e, x[0], aff_raw, y_coupled))

        def step_end(carry_n, key_t):
            M, n_prev, y_ema, y_prev, aff_ema = carry_n
            aff_raw = Gp * np.einsum("ij,jd->id", C, y_prev)
            return step_core(carry_n, key_t, aff_raw)

        def step_inj(carry_n, xs_t):
            key_t, aff_raw = xs_t
            return step_core(carry_n, key_t, aff_raw)

        keys_w = jax.random.split(keys, T_int)
        carry = jax.lax.stop_gradient(carry)
        if burnin_n > 0:
            bk = jax.random.split(jax.random.fold_in(keys, 77), burnin_n)

            def burn_step(c, key_t):
                M, n_prev, y_ema, y_prev, aff_ema = c
                aff_raw = Gp * np.einsum("ij,jd->id", C, y_prev)
                out_carry, _ = step_core(c, key_t, aff_raw)
                return out_carry, None

            carry = jax.lax.stop_gradient(jax.lax.scan(burn_step, carry, bk)[0])

        if inject is None:
            carry, (e_seq, x_seq, aff_seq, y_seq) = jax.lax.scan(
                step_end, carry, keys_w)
        else:
            carry, (e_seq, x_seq, aff_seq, y_seq) = jax.lax.scan(
                step_inj, carry, (keys_w, inject))
        e_vol = e_seq.reshape(win, steps_per_vol, N).mean(axis=1)
        tgt = emp_z if zscore_loss else emp_raw
        pred = zscore(e_vol) if zscore_loss else e_vol
        return np.mean((pred - tgt) ** 2), {
            "aff": aff_seq, "x": x_seq, "e": e_seq, "e_vol": e_vol,
            "y": y_seq, "M": carry[0]}

    key = jax.random.PRNGKey(seed)
    _, kk = jax.random.split(key)
    carry0 = (np.zeros((1, N, N, D, D)), np.zeros((N, D)), np.ones((N, 1)),
              np.zeros((N, D)), np.ones((N, 1)))
    gv = np.asarray(float(G))

    def with_g(g):
        p = dict(params)
        if "log_G" in p:
            p["log_G"] = np.log(g)
        else:
            p["G_rel"] = g
        return p

    out = {"G": float(G), "burnin": burnin, "win": win, "dt_train": dt_ms,
           "N": N, "drive_norm": bool(drive_norm), "coupling": coupling,
           "drive_beta": float(beta), "sigma": float(sigma)}
    loss0, st = f(params, carry0, kk, burnin)
    aff_true = jax.lax.stop_gradient(st["aff"])
    e_vol = jax.lax.stop_gradient(st["e_vol"])
    y_seq = jax.lax.stop_gradient(st["y"])
    out["loss"] = float(loss0)

    out["dLdG_total"] = float(jax.grad(
        lambda g: f(with_g(g), carry0, kk, burnin)[0])(gv))
    g_aff = jax.grad(
        lambda a: f(params, carry0, kk, burnin, inject=a)[0])(aff_true)

    def stages_all(g):
        st2 = f(with_g(g), carry0, kk, burnin)[1]
        return (st2["aff"], st2["x"], st2["e"], st2["e_vol"], st2["y"],
                st2["M"])

    val_s, tang_s = jax.jvp(stages_all, (gv,), (np.asarray(1.0),))
    daff = onp.asarray(tang_s[0])
    out["dLdaff_norm"] = float(np.sqrt(np.sum(g_aff ** 2)))
    out["daffdG_norm"] = float(np.sqrt(np.sum(daff ** 2)))
    contrib = onp.asarray(g_aff) * daff
    per_step = contrib.sum(axis=(1, 2))
    out["dLdG_via_aff"] = float(per_step.sum())
    ns = max(1, per_step.size // 8)
    out["contrib_profile"] = [float(per_step[i:i + ns].sum())
                              for i in range(0, per_step.size, ns)]
    ys = onp.asarray(y_seq)
    y_prev_seq = onp.concatenate([onp.zeros((1, N, D), dtype="f4"), ys[:-1]], 0)
    C_y = onp.einsum("ij,tjd->tid", onp.asarray(C), y_prev_seq)
    direct = float((onp.asarray(g_aff) * C_y).sum())
    out["direct_part"] = direct
    out["recursive_part"] = float(per_step.sum()) - direct

    sens = {}
    for name, vv, tt in zip(("aff", "x", "e", "e_vol", "y", "M"),
                            val_s, tang_s):
        nv = float(np.sqrt(np.sum(vv ** 2)) + 1e-30)
        nt = float(np.sqrt(np.sum(tt ** 2)) + 1e-30)
        sens[name] = nt / nv * abs(float(gv))
    out["log_sens_stages"] = sens

    de = tang_s[3]
    out["e_vol_null_frac"] = _null_frac(de, onp.asarray(e_vol))
    out["e_vol_rel_sens"] = (float(np.sqrt(np.sum(de ** 2)))
                             / (float(np.sqrt(np.sum(e_vol ** 2))) + 1e-30))
    out["dLdG_no_zscore"] = float(jax.grad(
        lambda g: f(with_g(g), carry0, kk, burnin, zscore_loss=False)[0])(gv))
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
    ap.add_argument("--loss-mode", default="current",
                    choices=["current", "sym_win", "gscale", "balloon_pr",
                             "balloon_gs", "amp", "fcy", "fce", "fcy_noreg",
                             "fc2", "fc2r"],
                    help="prediction/target normalisation + forward model. 'current' is "
                         "the shipped objective; others are the probe's candidates")
    ap.add_argument("--loss-amp-lambda", type=float, default=0.5,
                    help="weight of the per-region log-std term in --loss-mode amp; "
                         "also the temporal-MSE regulariser weight in the FC modes")
    ap.add_argument("--log-g", action="store_true",
                    help="parameterise the loop gain as log_G (better conditioning)")
    ap.add_argument("--pulse-step", type=int, default=-1,
                    help="if >= 0, inject an excitation pulse at this in-window step")
    ap.add_argument("--pulse-amp", type=float, default=1.0)
    ap.add_argument("--pulse-regions", type=int, default=10)
    ap.add_argument("--coupling", default="ema", choices=["ema", "raw"],
                    help="afferent coupling input: 'ema' (EMA-normalised output, "
                         "matches wholebrain_fc.simulate) or 'raw' (historical trainer)")
    ap.add_argument("--persist-carry", action="store_true",
                    help="keep (M, noise, y_ema, aff_ema) across iterations instead "
                         "of restarting cold each one; lets the EMA normalisers "
                         "and memory converge so the coupling is actually live")
    ap.add_argument("--drive-norm", action="store_true",
                    help="fix 1: per-region drive normalisation. Divide the raw "
                         "afferent drive by a stop-gradient EMA of its own "
                         "per-region rms and rescale by sigma*drive_beta, so the "
                         "coupling drive reaches the injected-noise scale. Adds "
                         "the learnable scalar drive_beta (init 1.0)")
    ap.add_argument("--freeze-sigma", action="store_true",
                    help="zero the log_sigma gradient in every iteration (both "
                         "stages) so sigma stays at its initialised value "
                         "log(0.18) and cannot absorb the drive-norm slack")
    ap.add_argument("--attrib-json", default=None,
                    help="diagnostic mode: run the stage attribution of dL/dG on "
                         "the --params checkpoint (same metric set as "
                         "wb_gradflow.py, but honouring --drive-norm/--coupling) "
                         "and write it here, then exit")
    ap.add_argument("--attrib-burnin", type=int, default=200,
                    help="burn-in steps for --attrib-json")
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
              "kv_scale": np.asarray(0.15),
              "log_tau": np.asarray(onp.log(36.0)),
              "log_sigma": np.asarray(onp.log(0.18)),
              "log_noise_tc": np.asarray(onp.log(2.0))}
    if args.log_g:
        params["log_G"] = np.asarray(onp.log(2.09))
    else:
        params["G_rel"] = np.asarray(2.09)
    if args.drive_norm:
        params["drive_beta"] = np.asarray(1.0)
    if args.freeze_sigma:
        print(f"[init] freeze-sigma ON: log_sigma pinned at "
              f"{float(onp.log(0.18)):.6f} (sigma=0.18)", flush=True)
    if args.drive_norm:
        print(f"[init] drive-norm ON: drive_beta="
              f"{float(params['drive_beta']):.4f}, sigma*beta="
              f"{float(np.exp(params['log_sigma'])) * float(params['drive_beta']):.4f}",
              flush=True)

    n_win = ts1.shape[0] // args.win
    emp_z_wins = zscore(ts1)[:n_win * args.win].reshape(n_win, args.win, N)
    emp_raw_wins = ts1[:n_win * args.win].reshape(n_win, args.win, N)
    L_win = max(int(lags.max()), 1) + 1

    pulse_sched = None
    if args.pulse_step >= 0:
        T_int = args.win * int(round(2000.0 / args.dt_train))
        _ps = onp.zeros((T_int, N), dtype="f4")
        _ps[args.pulse_step, :args.pulse_regions] = 1.0
        pulse_sched = np.asarray(_ps)
    aux = {"fc_emp": np.asarray(fc1), "pulse_sched": pulse_sched,
           "pulse_amp": args.pulse_amp}

    def full_carry(params):
        _, st, _ = init_ct_mhsa(hp0, key, batch_size=1)
        n_prev = np.zeros((N, d))
        y_ema = np.ones((N, 1))
        y_prev = np.zeros((N, d))
        aff_ema = np.ones((N, 1))
        return (st.M, n_prev, y_ema, y_prev, aff_ema)

    def losses_all_windows(params, carry, kw):
        losses = []
        c = carry
        for w in range(emp_z_wins.shape[0]):
            k_w = jax.random.fold_in(kw, w)
            c, loss = window_loss(params, c, emp_z_wins[w], emp_raw_wins[w], k_w,
                                  C, lags, L_win, hp0, args.dt_train,
                                  args.loss_mode, args.tr * 1000.0,
                                  args.loss_amp_lambda, args.coupling, aux,
                                  args.drive_norm)
            losses.append(loss)
        return c, np.mean(np.stack(losses))

    def eval_fc(params, T_s, ts_fc, seed):
        tau_s = float(np.exp(params["log_tau"]))
        sigma = float(np.exp(params["log_sigma"]))
        hp = hp0._replace(lam=args.dt / (tau_s * 1000.0), tau=tau_s * 1000.0)
        mhsa = build_mhsa(hp, jax.random.PRNGKey(seed), params["W_q"],
                          params["W_y"], float(params["kv_scale"]))
        from vbjax.app.wholebrain_fc import simulate, bold_forward
        G_eval = (float(np.exp(params["log_G"])) if "log_G" in params
                  else float(params["G_rel"]))
        e, _ = simulate(mhsa, np.asarray(C), jax.numpy.asarray(lags), L, hp,
                        G_eval, sigma, int(T_s * 1000 / args.dt),
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

    if args.attrib_json:
        with open(args.params, "rb") as f:
            p_ck = {k: np.asarray(v) for k, v in pickle.load(f).items()}
        aout = run_attribution(C, hp0, p_ck, ts1, args.win, args.dt_train,
                              args.attrib_burnin, args.coupling,
                              args.drive_norm, args.seed)
        aout["params"] = args.params
        with open(args.attrib_json, "w") as f:
            json.dump(aout, f, indent=1)
        print(json.dumps(aout, indent=1)[:2000], flush=True)
        print("WB_TRAIN_ATTRIB_DONE", flush=True)
        return

    opt = Adam(params, lr=1e-3)
    opt_s = Adam(params, lr=1e-2)
    MIN_LR = 1e-5
    BAD_TO_HALVE = 3
    iters1, iters2 = args.iters // 2, args.iters - args.iters // 2
    log = []
    last_good = {k: v for k, v in params.items()}
    bad_streak = 0
    carry = full_carry(params)
    last_good_carry = carry

    @jax.jit
    def fwd_only(p, c, kw):
        """No-grad forward: the persisted carry for the next iteration."""
        c2, _ = losses_all_windows(p, c, kw)
        return jax.lax.stop_gradient(c2)

    for it in range(iters1 + iters2):
        stage_all = it >= iters1
        cur_opt = opt if stage_all else opt_s
        t0 = time.time()
        key, kw = jax.random.split(key)
        if not args.persist_carry:
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
            # Roll back to the last params whose step produced finite grads:
            # the bad gradient was measured *at* the current (post-step) params.
            params = {k: v2 for k, v2 in last_good.items()}
            if args.persist_carry:
                carry = last_good_carry   # the bad carry came from bad params
            lr_now = cur_opt.lr
            cur_opt.reset()          # drop any poisoned m/v moment state
            print(f"iter {it:4d}/{args.iters} SKIP non-finite loss/grad "
                  f"loss={v} -> restore params, bad_streak={bad_streak} "
                  f"lr={lr_now:.2e} ({time.time()-t0:.1f}s)", flush=True)
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
        last_good_carry = carry
        if not stage_all:
            grads = {k: (np.zeros_like(g) if k in ("W_q", "W_y") else g)
                     for k, g in grads.items()}
        if args.freeze_sigma:
            # sigma is held at its initialised value in BOTH stages so the
            # drive-normalisation slack cannot be absorbed into the noise level.
            grads = {k: (np.zeros_like(g) if k == "log_sigma" else g)
                     for k, g in grads.items()}
        gnorm = float(np.sqrt(sum(float(np.sum(g ** 2)) for g in grads.values())))
        if gnorm > 1.0:
            grads = {k: g / gnorm for k, g in grads.items()}
        params = cur_opt.step(params, grads)
        if args.persist_carry:
            carry = fwd_only(params, carry, kw)
        G_val = (float(np.exp(params["log_G"])) if "log_G" in params
                 else float(params["G_rel"]))
        msg = (f"iter {it:4d}/{args.iters} loss={v:.5f} "
               f"G={G_val:.2f} "
               f"tau={float(np.exp(params['log_tau'])):.1f}s "
               f"sigma={float(np.exp(params['log_sigma'])):.3f} "
               f"gnorm={gnorm:.2e} lr={cur_opt.lr:.2e} ({time.time()-t0:.1f}s)")
        if args.drive_norm:
            msg += f" beta={float(params['drive_beta']):.4f}"
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
