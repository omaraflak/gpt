import os
import jax
import fire
import datetime
import modules
import losses
import optimizers
import loader
import training
import pickle
import tokenizer as tk


model_dir = "checkpoints/260915_1806"
step = 11776

with open(f"{model_dir}/config_s{step}.pkl", "rb") as f:
    config = pickle.load(f)

num_layers = config["num_layers"]
heads = config["heads"]
seq = config["seq"]
embed = config["embed"]
vocab = config["vocab"]
params = config["params"]
dropout = 0.1

with open("data/hugo_qa.txt", "r") as f:
    corpus = f.read()

tokenizer = tk.BPETokenizer.load(f"{model_dir}/tokenizer.pkl")
corpus_tokens = tokenizer.encode(corpus)
chars_per_token = len(tk.BPETokenizer.normalize(corpus)) / len(corpus_tokens)
x_train, y_train, x_val, y_val = training.make_dataset(
    corpus_tokens, seq, sequence_overlap=2, train_val_split=0.9, block_size=4
)

model = modules.GPT(num_layers, heads, seq, embed, vocab, dropout)
apply = jax.vmap(model.apply, in_axes=(None, 0, 0))


def on_checkpoint(new_params, step: int, val_loss: float):
    ckpt_config = {
        "seq": seq,
        "vocab": vocab,
        "embed": embed,
        "heads": heads,
        "num_layers": num_layers,
        "params": new_params,
    }
    with open(f"{model_dir}/config_sft_s{int(step)}.pkl", "wb") as f:
        pickle.dump(ckpt_config, f)


epochs = 3
batch_size = 1
total_steps = epochs * (len(x_train) // batch_size)
warmup_steps = min(100, total_steps // 10)
params, opt_state, train_loss, val_loss = training.train(
    params,
    apply,
    losses.cross_entropy_logits,
    optimizers.adam(
        lr=1e-4,
        beta1=0.9,
        beta2=0.95,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr=1e-5,
        weight_decay=0.01,
    ),
    x_train,
    y_train,
    x_val,
    y_val,
    epochs,
    run_val_every=4 * batch_size,
    batch_size=batch_size,
    print_every=2 * batch_size,
    chars_per_token=chars_per_token,
    checkpoint_callback=on_checkpoint,
)

losses = {"train_loss": train_loss, "val_loss": val_loss}
with open(f"{model_dir}/losses_sft.pkl", "wb") as f:
    pickle.dump(losses, f)
