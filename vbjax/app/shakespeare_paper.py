"""Faithful reproduction of the paper's Tiny Shakespeare task (Fig 2b).

Babu/Granier/Proix/Woodman/Jirsa et al., "Cortico-subcortical multi-head
self-attention as a substrate for cognitive performance" (bioRxiv 2026.08.06.743205).

Paper Methods (Shakespeare task):
  - ~1M chars, BPE tokenizer, vocab 4096.
  - token embedding x_t in R^192 via linear embed E in R^{192x4096}.
  - single attention block, 6 heads; each area (head) has 32^2=1024 L2/3+L5IT
    ensembles, i.e. d_k = d_v = 32.
  - memory retention lambda=0.95 (dt=50ms, tau=1s) -> update rate u = dt/tau = 0.05.
  - output y_t = sum_h W_y^h o_t^h; next-token probs from softmax(P (x_t + y_t)),
    P in R^{4096x192}.
  - learn E, P, W_q, W_y; W_k, W_v random & FROZEN (no BPTT for memory).
  - avg cross-entropy over 32 time steps, batch 32, Adam, early stopping.

Run on the GPU apptainer container (see docs/MHSA_WORKBOARD.md E0).
Persists hyperparameters + config + seed + val curve alongside the checkpoint (A.6).
"""
from typing import NamedTuple, Tuple
import os, pickle, json
import jax
import jax.numpy as np
import optax
import numpy as onp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

from vbjax.ct_mhsa import (
    Hyperparameters, CTMHSAParams, NetworkState, init_ct_mhsa, mhsa_step,
)


# ---- data ---------------------------------------------------------------

def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def make_bpe_tokenizer(text: str, vocab_size: int = 4096) -> Tokenizer:
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["<unk>"])
    tok.train_from_iterator([text], trainer)
    return tok


def encode(tok: Tokenizer, text: str) -> onp.ndarray:
    enc = tok.encode(text)
    return onp.array(enc.ids, dtype=onp.int32)


def get_batch(data: onp.ndarray, batch_size: int, seq_len: int, key):
    starts = jax.random.randint(key, (batch_size,), 0, len(data) - seq_len - 1)
    xs, ys = [], []
    for s in onp.asarray(starts):
        s = int(s)
        xs.append(data[s:s + seq_len])
        ys.append(data[s + 1:s + seq_len + 1])
    return np.stack(xs), np.stack(ys)  # (B, T)


# ---- model config -------------------------------------------------------

class ShakespeareHP(NamedTuple):
    vocab_size: int = 4096
    d_model: int = 192          # paper token embedding dim
    n_heads: int = 6            # 6 cortical areas / heads
    d_k: int = 32               # 32^2 = 1024 ensembles per area
    d_v: int = 32
    seq_len: int = 32
    batch_size: int = 32
    lr: float = 1e-4
    warmup_steps: int = 500
    train_steps: int = 5000
    decay_steps: int = 0        # 0 => decay over full train_steps
    early_stop_patience: int = 300
    seed: int = 42


class ShakespeareParams(NamedTuple):
    embed: jax.Array            # (V, D)
    mhsa: CTMHSAParams
    head: jax.Array             # (D, V)  == P^T


def init_shakespeare(hp: ShakespeareHP, mhsa_hp: Hyperparameters, key):
    # Embedding/readout init: unit-ish input scale (1/sqrt(d_model)) so the residual
    # readout x_t + y_t has meaningful magnitude and gradients don't vanish.
    # (Core W_* use Xavier in init_ct_mhsa; 0.02 here was ~3.5x too small.)
    k_e, k_m, k_h = jax.random.split(key, 3)
    scale = 1.0 / (hp.d_model ** 0.5)
    embed = jax.random.normal(k_e, (hp.vocab_size, hp.d_model)) * scale
    mhsa, state, lags = init_ct_mhsa(mhsa_hp, k_m, batch_size=hp.batch_size,
                                     initial_c=np.ones((1, 1)))
    head = jax.random.normal(k_h, (hp.d_model, hp.vocab_size)) * scale
    return ShakespeareParams(embed, mhsa, head), state


def new_state(mhsa_hp: Hyperparameters, batch_size: int, key) -> NetworkState:
    _, state, _ = init_ct_mhsa(mhsa_hp, key, batch_size=batch_size,
                               initial_c=np.ones((1, 1)))
    return state


def forward(prm: ShakespeareParams, state: NetworkState, x_idx: jax.Array,
            mhsa_hp: Hyperparameters) -> Tuple[NetworkState, jax.Array]:
    """x_idx (B, T) -> state, logits (B, T, V). Single block, heads sum. No C coupling."""
    B, T = x_idx.shape
    x_emb = prm.embed[x_idx]                       # (B, T, D)
    x_emb_T = np.transpose(x_emb, (1, 0, 2))        # (T, B, D)

    def step_fn(carry, x_t):
        st = carry
        x_t = x_t[:, None, :]  # (B, 1, D) single region (paper: heads sum, no C coupling)
        st_new, y, _ = mhsa_step(prm.mhsa, st, x_t, mhsa_hp)  # y (B,1,D)
        return st_new, y[:, 0, :]                   # squeeze region dim

    final_state, y_seq = jax.lax.scan(step_fn, state, x_emb_T)  # y_seq (T,B,D)
    y_seq = np.transpose(y_seq, (1, 0, 2))          # (B, T, D)

    # paper uses residual x_t + y_t for the readout
    resid = x_emb + y_seq                           # (B, T, D)
    logits = resid @ prm.head                       # (B, T, V)
    return final_state, logits


def loss_fn(prm, state, x, y, mhsa_hp):
    _, logits = forward(prm, state, x, mhsa_hp)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    one_hot = jax.nn.one_hot(y, logits.shape[-1])
    return -np.mean(np.sum(one_hot * log_probs, axis=-1))


def sample_accuracy(prm, state, x, y, mhsa_hp):
    _, logits = forward(prm, state, x, mhsa_hp)
    preds = np.argmax(logits, axis=-1)
    return np.mean(preds == y)


# ---- training -----------------------------------------------------------

def train(hp: ShakespeareHP, mhsa_hp: Hyperparameters,
          train_data: onp.ndarray, val_data: onp.ndarray,
          out_dir: str):
    key = jax.random.PRNGKey(hp.seed)
    k_init, k_train, k_val = jax.random.split(key, 3)

    prm, _ = init_shakespeare(hp, mhsa_hp, k_init)

    # freeze W_k, W_v (paper: memory synapses random & fixed), learn W_q, W_y
    from jax.tree_util import GetAttrKey
    FROZEN = {"W_K", "W_V", "C", "A", "ln_gamma", "ln_beta"}

    def frozen_mask(prm):
        def is_frozen(path):
            names = {getattr(seg, "name", None) for seg in path}
            return bool(names & FROZEN)
        return jax.tree_util.tree_map_with_path(
            lambda path, leaf: onp.float32(0.0 if is_frozen(path) else 1.0), prm)

    mask = frozen_mask(prm)

    # warmup_cosine_decay_schedule internally needs decay_steps > warmup_steps; bound both
    # so short/smoke runs (train_steps small) don't produce a negative internal decay window.
    warmup = min(hp.warmup_steps, max(1, hp.train_steps // 4))
    decay_steps = int(getattr(hp, "decay_steps", hp.train_steps) or hp.train_steps)
    decay_steps = max(decay_steps, warmup + 1)
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=1e-4, peak_value=hp.lr, warmup_steps=warmup,
        decay_steps=decay_steps, end_value=1e-5,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.masked(optax.adamw(learning_rate=schedule), mask=mask),
    )
    opt_state = optimizer.init(prm)

    @jax.jit
    def train_step(prm, opt_state, state, x, y):
        loss, grads = jax.value_and_grad(loss_fn)(prm, state, x, y, mhsa_hp)
        gn = optax.global_norm(grads)  # pre-clip grad norm (diagnostic)
        updates, opt_state = optimizer.update(grads, opt_state, prm)
        prm = optax.apply_updates(prm, updates)
        e_norm = np.sqrt(np.sum(prm.embed ** 2))
        return prm, opt_state, loss, gn, e_norm

    @jax.jit
    def diag_fn(prm, st, x):
        fin_state, logits = forward(prm, st, x, mhsa_hp)
        return np.max(np.abs(logits)), np.max(np.abs(fin_state.M))

    val_bs = hp.batch_size * 4
    val_state = new_state(mhsa_hp, val_bs, k_val)
    val_step = jax.jit(lambda prm, x, y, st: (loss_fn(prm, st, x, y, mhsa_hp),
                                              sample_accuracy(prm, st, x, y, mhsa_hp)))

    # validation slice
    n_val = min(4096, len(val_data) - hp.seq_len - 1)
    val_x, val_y = get_batch(val_data[:n_val + hp.seq_len], hp.batch_size * 4, hp.seq_len, k_val)

    train_losses, val_losses, val_accs = [], [], []
    best_val, best_prm, patience = float("inf"), None, 0

    for i in range(hp.train_steps + 1):
        k_train, k_b = jax.random.split(k_train)
        x, y = get_batch(train_data, hp.batch_size, hp.seq_len, k_b)
        state = new_state(mhsa_hp, hp.batch_size, k_train)
        prm, opt_state, loss, gn, en = train_step(prm, opt_state, state, x, y)
        train_losses.append(float(onp.asarray(loss)))

        if i % 50 == 0:
            vl, va = val_step(prm, val_x, val_y, val_state)
            ml, mm = diag_fn(prm, val_state, val_x)
            vl, va = float(onp.asarray(vl)), float(onp.asarray(va))
            gn_f, en_f = float(onp.asarray(gn)), float(onp.asarray(en))
            ml_f, mm_f = float(onp.asarray(ml)), float(onp.asarray(mm))
            val_losses.append(vl)
            val_accs.append(va)
            print(f"step {i:5d} train_loss={loss:.4f} val_loss={vl:.4f} val_acc={va:.4f} "
                  f"gn={gn_f:.3f} e_norm={en_f:.2f} maxlogit={ml_f:.2f} maxM={mm_f:.4f}")
            if vl < best_val - 1e-4:
                best_val, best_prm, patience = vl, prm, 0
            else:
                patience += 1
                if patience >= hp.early_stop_patience:
                    print("early stop")
                    break

    os.makedirs(out_dir, exist_ok=True)
    ckpt = {"hp": hp._asdict(), "mhsa_hp": mhsa_hp._asdict(), "params": best_prm or prm,
            "seed": hp.seed, "train_losses": train_losses,
            "val_losses": val_losses, "val_accs": val_accs}
    with open(os.path.join(out_dir, "shakespeare.pkl"), "wb") as f:
        pickle.dump(ckpt, f)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump({"hp": hp._asdict(), "mhsa_hp": mhsa_hp._asdict(), "seed": hp.seed}, f, indent=2)

    # loss curve
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(train_losses, label="train")
    ax[0].plot(np.linspace(0, len(train_losses) - 1, len(val_losses)), val_losses, label="val")
    ax[0].set_xlabel("step"); ax[0].set_ylabel("CE loss"); ax[0].legend(); ax[0].set_title("loss")
    ax[1].plot(np.linspace(0, len(train_losses) - 1, len(val_accs)), val_accs)
    ax[1].set_xlabel("step"); ax[1].set_ylabel("val accuracy"); ax[1].set_title("accuracy")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "loss_curve.png"))
    print("saved", out_dir)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="examples/input.txt")
    p.add_argument("--out", default="runs/shakespeare")
    p.add_argument("--train-steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--lr", type=float, default=None, help="peak learning rate (default hp.lr)")
    p.add_argument("--decay-steps", type=int, default=None, help="lr decay horizon in steps (default: train_steps)")
    p.add_argument("--patience", type=int, default=None, help="early-stop patience in steps")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    hp = ShakespeareHP(train_steps=a.train_steps, batch_size=a.batch_size,
                       seq_len=a.seq_len, seed=a.seed)
    if a.lr is not None:
        hp = hp._replace(lr=a.lr)
    if a.patience is not None:
        hp = hp._replace(early_stop_patience=a.patience)
    if a.decay_steps is not None:
        hp = hp._replace(decay_steps=a.decay_steps)
    mhsa_hp = Hyperparameters(n_regions=1, n_heads=hp.n_heads, d_k=hp.d_k,
                              d_v=hp.d_v, d_model=hp.d_model,
                              lam=0.05, dt=1.0, tau=20.0, leak=1.0,
                              act="relu", steps_per_token=1)

    text = load_text(a.input)
    print(f"text {len(text)} chars")
    tok = make_bpe_tokenizer(text, hp.vocab_size)
    ids = encode(tok, text)
    print(f"tokens {len(ids)}")
    split = int(len(ids) * 0.95)
    train_data, val_data = ids[:split], ids[split:]
    train(hp, mhsa_hp, train_data, val_data, a.out)
