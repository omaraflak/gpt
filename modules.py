import jax
import math
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Callable, Union
from losses import Loss
from optimizers import Optimizer

Params = dict[str, Union["Params", jax.Array]]
Init = Callable[[jax.Array], Params]
Apply = Callable[[jax.Array], Params]
Model = Callable[[Params, jax.Array], jax.Array]


@dataclass
class Module:
    name: str
    init: Init
    apply: Apply


def mod(f):
    def fun(*args, **kwargs):
        value = getattr(f, "instances", 0) + 1
        setattr(f, "instances", value)
        init, apply = f(*args, **kwargs)
        name = f.__name__ + str(value)
        return Module(name, init, apply)

    return fun


def _split(key: jax.Array | None, n: int) -> list:
    if key is None:
        return [None] * n
    return list(jax.random.split(key, n))


def make(f) -> Module:
    value = getattr(make, "instances", 0) + 1
    setattr(make, "instances", value)
    name = f.__name__ + str(value)

    def init(key):
        return {}

    def apply(params, x, key=None):
        return f(x)

    return Module(name, init, apply)


@mod
def Sequence(modules: list[Module]):
    def init(key):
        params = dict()
        for mod in modules:
            key, subkey = jax.random.split(key)
            params[mod.name] = mod.init(subkey)
        return params

    def apply(params, x, key=None):
        for mod, subkey in zip(modules, _split(key, len(modules))):
            x = mod.apply(params[mod.name], x, subkey)
        return x

    return init, apply


@mod
def Linear(in_dim: int, out_dim: int):
    std = 1.0 / jnp.sqrt(in_dim)

    def init(key):
        return {
            "w": jax.random.normal(key, (out_dim, in_dim)) * std,
            "b": jnp.zeros((out_dim,)),
        }

    def apply(params, x, key=None):
        return x @ params["w"].T + params["b"]

    return init, apply


@mod
def Dropout(p: float):
    keep_prob = 1.0 - p

    def init(key):
        return {}

    def apply(params, x, key=None):
        if p <= 0.0 or key is None:
            return x
        mask = jax.random.bernoulli(key, p=keep_prob, shape=x.shape)
        return jnp.where(mask, x / keep_prob, 0.0)

    return init, apply


@mod
def LayerNorm(shape: tuple[int, ...]):
    def init(key):
        return {
            "alpha": jnp.ones(shape),
            "beta": jnp.zeros(shape),
        }

    def apply(params, x, key=None):
        eps = 1e-5
        reduction_axes = tuple(range(-len(shape), 0))
        mu = jnp.mean(x, axis=reduction_axes, keepdims=True)
        var = jnp.var(x, axis=reduction_axes, keepdims=True)
        x_hat = (x - mu) / jnp.sqrt(var + eps)
        return x_hat * params["alpha"] + params["beta"]

    return init, apply


@mod
def RMSNorm(shape: tuple[int, ...]):
    def init(key):
        return {"alpha": jnp.ones(shape)}

    def apply(params, x, key=None):
        eps = 1e-5
        rms = jnp.sqrt(jnp.mean(x**2, axis=-1, keepdims=True) + eps)
        return (x / rms) * params["alpha"]

    return init, apply


@mod
def Attention(embed: int, dim: int):
    std = 1.0 / jnp.sqrt(embed)

    def init(key):
        wq, wk, wv = jax.random.split(key, 3)
        return {
            "wq": jax.random.normal(wq, (embed, dim)) * std,
            "wk": jax.random.normal(wk, (embed, dim)) * std,
            "wv": jax.random.normal(wv, (embed, dim)) * std,
        }

    # (T, C) -> (T, dk)
    def apply(params, x, key=None):
        # sequence length
        t = x.shape[0]
        # (T, C) @ (C, dk) -> (T, dk)
        q = x @ params["wq"]
        k = x @ params["wk"]
        v = x @ params["wv"]
        # (T, T)
        upper_tri = jnp.triu(jnp.ones((t, t), dtype=jnp.bool_), k=1)
        mask = jnp.where(upper_tri, -jnp.inf, 0.0)
        # (T, T)
        score = (q @ k.T) / jnp.sqrt(dim) + mask
        attention = jax.nn.softmax(score, axis=-1)
        # (T, dk)
        return attention @ v

    return init, apply


@mod
def MultiHeadAttention(heads: int, embed: int, dim: int):
    attentions = [Attention(embed, dim) for _ in range(heads)]
    norm = RMSNorm((embed,))
    std = 1.0 / jnp.sqrt(embed)

    def init(key):
        key, k1, k2 = jax.random.split(key, 3)
        params = {
            "w": jax.random.normal(k1, (embed, embed)) * std,
            norm.name: norm.init(k2),
        }
        for m in attentions:
            key, subkey = jax.random.split(key)
            params[m.name] = m.init(subkey)
        return params

    # (T, C) -> (T, C)
    def apply(params, x, key=None):
        x_hat = norm.apply(params[norm.name], x)
        # (T, dk)
        outs = [att.apply(params[att.name], x_hat) for att in attentions]
        # head * dk = embed
        # (T, C)
        return x + jnp.concat(outs, axis=-1) @ params["w"]

    return init, apply


@mod
def SwiGLU(dim: int):
    linear1 = Linear(dim, dim)
    linear2 = Linear(dim, dim)

    def init(key):
        k1, k2 = jax.random.split(key)
        return {linear1.name: linear1.init(k1), linear2.name: linear2.init(k2)}

    def apply(params, x, key=None):
        o1 = linear1.apply(params[linear1.name], x)
        o2 = linear2.apply(params[linear2.name], x)
        return o1 * jax.nn.sigmoid(o1) * o2

    return init, apply


@mod
def MLP(embed: int, dropout: float):
    hidden = int(8 / 3 * embed)
    w_gate = Linear(embed, hidden)
    w_up = Linear(embed, hidden)
    w_down = Linear(hidden, embed)
    norm = RMSNorm((embed,))
    dout = Dropout(dropout)

    def init(key):
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        return {
            w_gate.name: w_gate.init(k1),
            w_up.name: w_up.init(k2),
            w_down.name: w_down.init(k3),
            norm.name: norm.init(k4),
            # Holds no parameters, but apply() looks it up by name.
            dout.name: dout.init(k5),
        }

    # (T, C) -> (T, C)
    def apply(params, x, key=None):
        x_norm = norm.apply(params[norm.name], x)
        gate = jax.nn.silu(w_gate.apply(params[w_gate.name], x_norm))
        up = w_up.apply(params[w_up.name], x_norm)
        down = w_down.apply(params[w_down.name], gate * up)
        out = dout.apply(params[dout.name], down, key)
        return x + out

    return init, apply


@mod
def TransformerBlock(size: int, heads: int, embed: int, dim: int, dropout: float):
    modules = []
    for _ in range(size):
        modules.append(MultiHeadAttention(heads, embed, dim))
        modules.append(MLP(embed, dropout))

    def init(key):
        params = dict()
        for m in modules:
            key, subkey = jax.random.split(key)
            params[m.name] = m.init(subkey)
        return params

    # (T, C) -> (T, C)
    def apply(params, x, key=None):
        for m, subkey in zip(modules, _split(key, len(modules))):
            x = m.apply(params[m.name], x, subkey)
        return x

    return init, apply


@mod
def Embeddings(vocab: int, embed: int, seq: int):
    std = 1.0 / jnp.sqrt(embed)

    def init(key):
        ke, kp = jax.random.split(key)
        return {
            "we": jax.random.normal(ke, (vocab, embed)) * std,
            "wp": jax.random.normal(kp, (seq, embed)) * std,
        }

    # (T,) -> (T, C)
    def apply(params, x, key=None):
        return params["we"][x] + params["wp"][: x.shape[0]]

    return init, apply


@mod
def GPT(num_layers: int, heads: int, seq: int, embed: int, vocab: int, dropout: float):
    dim = embed // heads
    emb = Embeddings(vocab, embed, seq)
    head = Linear(embed, vocab)
    model = Sequence(
        [
            emb,
            TransformerBlock(num_layers, heads, embed, dim, dropout),
            RMSNorm((embed,)),
            head,
        ]
    )

    def init(key):
        params = model.init(key)
        params[head.name] = {"b": params[head.name]["b"]}
        return {model.name: params}

    def apply(params, x, key=None):
        p = params[model.name]
        p = {**p, head.name: {"w": p[emb.name]["we"], "b": p[head.name]["b"]}}
        return model.apply(p, x, key)

    return init, apply


def train(
    params: Params,
    apply: Apply,
    loss_fn: Loss,
    optimizer: Optimizer,
    x_data: jax.Array,
    y_data: jax.Array,
    epochs: int,
    train_val_split: float = 0.9,
    run_val_every: int = 128,
    batch_size: int = 64,
    print_every: int = 100,
    shuffle: bool = True,
    chars_per_token: float | None = None,
    rng_key: jax.Array = jax.random.key(0),
    checkpoint_callback: Callable[[Params, int, float], None] | None = None,
):
    x_data = jnp.asarray(x_data)
    y_data = jnp.asarray(y_data)
    num_samples = len(x_data)
    bpc_scale = 1.0 / (chars_per_token * math.log(2)) if chars_per_token else None

    if 0.0 < train_val_split < 1.0 and num_samples >= 10:
        if shuffle:
            rng_key, split_key = jax.random.split(rng_key)
            perm = jax.random.permutation(split_key, num_samples)
            split_idx = int(num_samples * train_val_split)
            train_idx, val_idx = perm[:split_idx], perm[split_idx:]
            x_train, y_train = x_data[train_idx], y_data[train_idx]
            x_val, y_val = x_data[val_idx], y_data[val_idx]
        else:
            split_idx = int(num_samples * train_val_split)
            x_train, y_train = x_data[:split_idx], y_data[:split_idx]
            x_val, y_val = x_data[split_idx:], y_data[split_idx:]
    else:
        x_train, y_train = x_data, y_data
        x_val, y_val = None, None

    num_train = len(x_train)
    batch_size = min(batch_size, num_train) if batch_size > 0 else num_train
    num_batches = num_train // batch_size
    total_steps = epochs * num_batches
    used_samples = num_batches * batch_size

    if shuffle and used_samples > 0:
        keys = jax.random.split(rng_key, epochs)
        epoch_perms = jax.vmap(
            lambda k: jax.random.permutation(k, used_samples).reshape(
                num_batches, batch_size
            )
        )(keys)
        batch_indices = epoch_perms.reshape(total_steps, batch_size)
    else:
        base = jnp.arange(used_samples).reshape(num_batches, batch_size)
        batch_indices = jnp.tile(base, (epochs, 1))

    # One dropout key per step, carried through the scan. Each is split into
    # batch_size sub-keys so every example gets an independent mask.
    # fold_in (rather than another split of rng_key) keeps this stream
    # independent of the shuffle keys derived above.
    step_keys = jax.random.split(jax.random.fold_in(rng_key, 0xD0), total_steps)

    def _loss_fn(p, b_idx, key):
        xb = x_train[b_idx]
        yb = y_train[b_idx]
        return loss_fn(apply(p, xb, jax.random.split(key, batch_size)), yb)

    def _update_step(state, step_input):
        p, opt_s, best_v_loss = state
        step, b_idx, step_key = step_input
        train_loss, grads = jax.value_and_grad(_loss_fn)(p, b_idx, step_key)

        # Global gradient norm clipping (1.0)
        total_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree.leaves(grads)))
        clip_coef = jnp.minimum(1.0, 1.0 / (total_norm + 1e-6))
        grads = jax.tree.map(lambda g: g * clip_coef, grads)

        p, opt_s = optimizer.update(p, grads, opt_s, step)

        if print_every > 0:

            def _print_train():
                if bpc_scale is None:
                    jax.debug.print(
                        "[{step}/{total_steps}] loss: {loss:.7f}",
                        step=step + 1,
                        total_steps=total_steps,
                        loss=train_loss,
                    )
                else:
                    jax.debug.print(
                        "[{step}/{total_steps}] loss: {loss:.7f} ({bpc:.4f} bpc)",
                        step=step + 1,
                        total_steps=total_steps,
                        loss=train_loss,
                        bpc=train_loss * bpc_scale,
                    )

            jax.lax.cond(
                (step + 1) % print_every == 0,
                _print_train,
                lambda: None,
            )

        val_loss = jnp.nan
        new_best_v_loss = best_v_loss
        if x_val is not None and run_val_every > 0:
            is_val_step = (step + 1) % run_val_every == 0

            def _val():
                val_sub_x = x_val[:256]
                val_sub_y = y_val[:256]
                v_loss = loss_fn(apply(p, val_sub_x, None), val_sub_y)
                if bpc_scale is None:
                    jax.debug.print(
                        "[{step}/{total_steps}] val_loss: {v_loss:.7f}",
                        step=step + 1,
                        total_steps=total_steps,
                        v_loss=v_loss,
                    )
                else:
                    jax.debug.print(
                        "[{step}/{total_steps}] val_loss: {v_loss:.7f} ({bpc:.4f} bpc)",
                        step=step + 1,
                        total_steps=total_steps,
                        v_loss=v_loss,
                        bpc=v_loss * bpc_scale,
                    )
                return v_loss

            val_loss = jax.lax.cond(
                is_val_step,
                _val,
                lambda: jnp.nan,
            )

            improved = is_val_step & (val_loss < best_v_loss)
            new_best_v_loss = jnp.where(improved, val_loss, best_v_loss)

            if checkpoint_callback is not None:
                jax.lax.cond(
                    improved,
                    lambda: jax.debug.callback(
                        checkpoint_callback, p, step + 1, val_loss
                    ),
                    lambda: None,
                )

        return (p, opt_s, new_best_v_loss), (train_loss, val_loss)

    @jax.jit
    def _run_scan(p, s):
        return jax.lax.scan(
            _update_step,
            (p, s, jnp.float32(float("inf"))),
            xs=(jnp.arange(total_steps), batch_indices, step_keys),
        )

    opt_state = optimizer.init(params)
    (final_params, _, _), (train_loss, val_loss) = _run_scan(params, opt_state)
    return final_params, train_loss, val_loss
