"""Generate text from a trained Shakespeare CT-MHSA checkpoint.

Usage (inside the apptainer container, from the repo tree root):
  python -m vbjax.app.shakespeare_gen --ckpt runs/shakespeare_stage3b/shakespeare.pkl \
      --prompt "ROMEO:" --n 300 --temp 0.8

Rebuilds the BPE tokenizer (deterministic, same corpus) and autoregressively
samples from softmax(P(x_t + y_t)) with the recurrent key-value memory carried
across tokens.
"""
import argparse, pickle
import jax
import jax.numpy as np
import numpy as onp

import vbjax.app.shakespeare_paper as sp
from vbjax.ct_mhsa import Hyperparameters, init_ct_mhsa
from vbjax.app.shakespeare_paper import (
    ShakespeareHP, make_bpe_tokenizer, load_text,
)


def generate(prm, mhsa_hp, tok, prompt_ids, n_tokens, temp, key):
    state = None
    out_ids = list(prompt_ids)

    # feed the prompt (single forward keeps the final recurrent memory state)
    x_idx = np.array([out_ids])
    state, logits = sp.forward(prm, state or _new_state(mhsa_hp, key), x_idx, mhsa_hp)

    for i in range(n_tokens):
        l = logits[0, -1] / temp
        key, k_s = jax.random.split(key)
        nxt = int(jax.random.categorical(k_s, l))
        out_ids.append(nxt)
        state, logits = sp.forward(prm, state, np.array([[nxt]]), mhsa_hp)
    return out_ids


def _new_state(mhsa_hp, key):
    _, state, _ = init_ct_mhsa(mhsa_hp, key, batch_size=1, initial_c=np.ones((1, 1)))
    return state


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="runs/shakespeare_stage3b/shakespeare.pkl")
    p.add_argument("--input", default="examples/input.txt")
    p.add_argument("--prompt", default="ROMEO:")
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--temps", type=str, default="0.5,0.8,1.0",
                   help="comma-separated temperatures; one sample each")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    with open(a.ckpt, "rb") as f:
        ck = pickle.load(f)
    hp = ShakespeareHP(**ck["hp"])
    mhsa_hp = Hyperparameters(**ck["mhsa_hp"])
    prm = ck["params"]
    print(f"checkpoint: best val CE {min(ck['val_losses']):.4f}, readout_ln={hp.readout_ln}, kv_scale={hp.kv_scale}")
    assert isinstance(prm, sp.ShakespeareParams)

    # match training flags
    sp._READOUT_LN = bool(hp.readout_ln)
    sp._EMBED_CLIP = 0.0

    tok = make_bpe_tokenizer(load_text(a.input), hp.vocab_size)
    prompt_ids = tok.encode(a.prompt).ids
    print(f"prompt {a.prompt!r} -> {prompt_ids}\n")

    for temp in [float(t) for t in a.temps.split(",")]:
        key = jax.random.PRNGKey(a.seed)
        ids = generate(prm, mhsa_hp, tok, prompt_ids, a.n, temp, key)
        print(f"===== temp {temp} =====")
        print(tok.decode(ids))
        print()
