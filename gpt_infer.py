import sys
import jax
import jax.numpy as jnp
import modules
import loader
import pickle
import tokenizer as tk

# checkpoints
config_path = "checkpoints/260915_0829/config_s25.pkl"
tokenizer_path = "checkpoints/260915_0829/tokenizer.pkl"

with open(config_path, "rb") as f:
    config = pickle.load(f)

num_layers = config["num_layers"]
heads = config["heads"]
seq = config["seq"]
embed = config["embed"]
vocab = config["vocab"]
params = config["params"]
dropout = 0

tokenizer = tk.BPETokenizer.load(tokenizer_path)
model = modules.GPT(num_layers, heads, seq, embed, vocab, dropout)


def generate(
    prompt: str,
    length: int = 150,
    temperature: float = 0.8,
    stop: str | None = None,
    rng_key: jax.Array = jax.random.key(42),
):
    tokens = tokenizer.encode(prompt)
    yield tokenizer.decode(list(tokens))

    window = tokens[-seq:]
    n = len(window)
    curr = jnp.zeros((seq,), dtype=jnp.int32).at[:n].set(jnp.array(window))
    stop_tokens = jnp.array(tokenizer.encode(stop) if stop else [])

    for _ in range(length):
        rng_key, subkey = jax.random.split(rng_key)
        logits = model.apply(params, curr)[n - 1]
        if temperature > 0:
            next_tok = int(jax.random.categorical(subkey, logits / temperature))
        else:
            next_tok = int(jnp.argmax(logits))

        yield tokenizer.decode([next_tok])

        if n < seq:
            curr = curr.at[n].set(next_tok)
            n += 1
        else:
            curr = jnp.concatenate([curr[1:], jnp.array([next_tok], dtype=jnp.int32)])

        if jnp.array_equal(curr[n - len(stop_tokens) : n], stop_tokens):
            return


for x in generate(
    "<|user|>\nQuel est le sens de la vie?\n<|end|>\n<|assistant|>\n",
    length=1000,
    temperature=0.7,
    stop="<|end|>",
):
    print(x, end="")
    sys.stdout.flush()

print()
