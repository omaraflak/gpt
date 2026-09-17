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

# config
num_layers = 8
heads = 4
seq = 256
embed = 384
epochs = 3
dropout = 0
target_vocab = 4000

date = datetime.datetime.now().strftime("%y%m%d_%H%M")
model_dir = f"checkpoints/{date}"
os.makedirs(model_dir, exist_ok=True)

corpus = loader.download_french_literature(max_books=300)

tokenizer = tk.BPETokenizer.create(corpus, vocab_size=target_vocab)
tokenizer.save(f"{model_dir}/tokenizer.pkl")
vocab = tokenizer.vocab_size

corpus_tokens = tokenizer.encode(corpus)
chars_per_token = len(tk.BPETokenizer.normalize(corpus)) / len(corpus_tokens)
x_train, y_train, x_val, y_val = training.make_dataset(
    corpus_tokens, seq, sequence_overlap=2, train_val_split=0.9, block_size=256
)

model = modules.GPT(num_layers, heads, seq, embed, vocab, dropout)
params = model.init(jax.random.key(42))
apply = jax.vmap(model.apply, in_axes=(None, 0, 0))
num_params = sum(x.size for x in jax.tree.leaves(params))

print("Number of parameters:", num_params)
print("Corpus tokens:", len(corpus_tokens))
print("Chars per token:", chars_per_token)
print("Parameters per token:", num_params / len(corpus_tokens))
print("Number of training examples:", len(x_train))
print("Number of validation examples:", len(x_val))


def on_checkpoint(new_params, opt_state, step: int, val_loss: float):
    ckpt_config = {
        "seq": seq,
        "vocab": vocab,
        "embed": embed,
        "heads": heads,
        "num_layers": num_layers,
        "params": new_params,
        "opt_state": opt_state,
    }
    with open(f"{model_dir}/config_s{int(step)}.pkl", "wb") as f:
        pickle.dump(ckpt_config, f)
    print("[saved model]")


batch_size = 64
total_steps = epochs * (len(x_train) // batch_size)
warmup_steps = min(100, total_steps // 10)
params, opt_state, train_loss, val_loss = training.train(
    params,
    apply,
    losses.cross_entropy_logits,
    optimizers.adam(
        lr=3e-4,
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
    opt_state=opt_state,
    rng_key=jax.random.key(1),
    checkpoint_callback=on_checkpoint,
)

losses = {"train_loss": train_loss, "val_loss": val_loss}
with open(f"{model_dir}/losses.pkl", "wb") as f:
    pickle.dump(losses, f)
