import os
import jax
import datetime
import modules
import losses
import optimizers
import loader
import pickle
import tokenizer as tk

# config
num_layers = 6
heads = 4
seq = 256
embed = 256
epochs = 3
target_vocab = 1000

date = datetime.datetime.now().strftime("%y%m%d_%H%M")
model_dir = f"checkpoints/{date}"
os.makedirs(model_dir, exist_ok=True)

corpus = loader.files_to_str(
    [
        "data/miserables.txt",
        "data/notredame.txt",
        "data/93.txt",
        "data/lhommequirit.txt",
        "data/dernierjour.txt",
        "data/hugo_qa.txt",
    ]
)

tokenizer = tk.BPETokenizer.create(corpus, vocab_size=target_vocab)
tokenizer.save(f"{model_dir}/tokenizer.pkl")
vocab = tokenizer.vocab_size

x_train, y_train = loader.chunk_tokens_for_llm(
    tokenizer.encode(corpus), seq, stride_factor=2
)

model = modules.GPT(num_layers, heads, seq, embed, vocab)
# params = model.init(jax.random.key(42))
with open("checkpoints/260915_0758/config_s100.pkl", "rb") as f:
    params = pickle.load(f)["params"]
apply = jax.vmap(model.apply, in_axes=(None, 0))
num_params = sum(x.size for x in jax.tree.leaves(params))

print(
    f"Training {num_params} parameters on {len(x_train)} sequences of length {seq}..."
)


def on_checkpoint(new_params, step: int, val_loss: float):
    ckpt_config = {
        "seq": seq,
        "vocab": vocab,
        "embed": embed,
        "heads": heads,
        "num_layers": num_layers,
        "params": new_params,
    }
    with open(f"{model_dir}/config_s{int(step)}.pkl", "wb") as f:
        pickle.dump(ckpt_config, f)
    print(f"--> Saved best model at step {int(step)} (val_loss: {float(val_loss):.4f})")


batch_size = 64
total_steps = epochs * int(len(x_train) * 0.9 // batch_size)
# warmup_steps = min(100, total_steps // 10)
warmup_steps = 0
optimizer = optimizers.adam(
    lr=1e-5,
    beta1=0.9,
    beta2=0.95,
    warmup_steps=warmup_steps,
    total_steps=total_steps,
    min_lr=1e-6,
    weight_decay=0.001,
)

# optimizer = optimizers.adam(
#     lr=1e-3,
#     beta1=0.9,
#     beta2=0.95,
#     warmup_steps=warmup_steps,
#     total_steps=total_steps,
#     min_lr=1e-4,
#     weight_decay=0.01,
# )

params, train_loss, val_loss = modules.train(
    params,
    apply,
    losses.cross_entropy_logits,
    optimizer,
    x_train,
    y_train,
    epochs,
    train_val_split=0.9,
    run_val_every=25,
    batch_size=batch_size,
    print_every=25,
    checkpoint_callback=on_checkpoint,
)

losses = {"train_loss": train_loss, "val_loss": val_loss}
with open(f"{model_dir}/losses.pkl", "wb") as f:
    pickle.dump(losses, f)
