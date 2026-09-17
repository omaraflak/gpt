import jax
import math
import jax.numpy as jnp
from typing import Callable
from losses import Loss
from optimizers import Optimizer
from modules import Params, Apply


def _chunk_tokens(
    tokens: jax.Array,
    sequence_size: int = 32,
    sequence_overlap: int = 2,
) -> tuple[jax.Array, jax.Array]:
    step = max(1, sequence_size // sequence_overlap)
    n = (tokens.shape[0] - sequence_size) // step
    starts = jnp.arange(n) * step
    idx = starts[:, None] + jnp.arange(sequence_size)[None, :]
    return tokens[idx], tokens[idx + 1]


def make_dataset(
    tokens: list[int],
    sequence_size: int,
    sequence_overlap: int = 2,
    train_val_split: float = 0.9,
    block_size: int = 256,
    rng_key: jax.Array = jax.random.key(0),
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    # 1. break corpus into blocks
    data = jnp.asarray(tokens, dtype=jnp.int32)
    num_blocks = min(block_size, len(data) // (sequence_size + 1))
    block_size = len(data) // num_blocks
    blocks = data[: num_blocks * block_size].reshape(num_blocks, block_size)

    # 2. shuffle blocks so train/val both draw from across the whole corpus
    perm = jax.random.permutation(rng_key, num_blocks)
    blocks = blocks[perm]
    val_key = jax.random.fold_in(rng_key, 1)
    train_key = jax.random.fold_in(rng_key, 2)

    # 3. split blocks in train / val groups
    split_idx = int(num_blocks * train_val_split)
    train_blocks = blocks[:split_idx]
    val_blocks = blocks[split_idx:]

    # 4. for each (train/val) blocks, create x,y pairs by chunking with overlap
    chunk_fn = jax.vmap(
        lambda b: _chunk_tokens(
            b, sequence_size=sequence_size, sequence_overlap=sequence_overlap
        )
    )

    x_train, y_train = chunk_fn(train_blocks)
    x_train = x_train.reshape(-1, sequence_size)
    y_train = y_train.reshape(-1, sequence_size)

    x_val, y_val = chunk_fn(val_blocks)
    x_val = x_val.reshape(-1, sequence_size)
    y_val = y_val.reshape(-1, sequence_size)

    # 5. reshape and shuffle pairs
    train_perm = jax.random.permutation(train_key, len(x_train))
    x_train = x_train[train_perm]
    y_train = y_train[train_perm]

    val_perm = jax.random.permutation(val_key, len(x_val))
    x_val = x_val[val_perm]
    y_val = y_val[val_perm]

    return x_train, y_train, x_val, y_val


def train(
    params: Params,
    model_fn: Apply,
    loss_fn: Loss,
    optimizer: Optimizer,
    x_train: jax.Array,
    y_train: jax.Array,
    x_val: jax.Array,
    y_val: jax.Array,
    epochs: int,
    run_val_every: int = 128,
    batch_size: int = 64,
    print_every: int = 100,
    val_samples: int = 256,
    shuffle: bool = True,
    chars_per_token: float | None = None,
    opt_state=None,
    start_step: int = 0,
    rng_key: jax.Array = jax.random.key(0),
    checkpoint_callback: Callable[[Params, int, float], None] | None = None,
):
    x_train = jnp.asarray(x_train)
    y_train = jnp.asarray(y_train)
    num_samples = len(x_train)
    bpc_scale = 1.0 / (chars_per_token * math.log(2)) if chars_per_token else None

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
        return loss_fn(model_fn(p, xb, jax.random.split(key, batch_size)), yb)

    def _update_step(state, step_input):
        p, opt_s, best_v_loss = state
        step, b_idx, step_key = step_input
        train_loss, grads = jax.value_and_grad(_loss_fn)(p, b_idx, step_key)

        # Global gradient norm clipping (1.0)
        total_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree.leaves(grads)))
        clip_coef = jnp.minimum(1.0, 1.0 / (total_norm + 1e-6))
        grads = jax.tree.map(lambda g: g * clip_coef, grads)

        p, opt_s = optimizer.update(p, grads, opt_s, step + start_step)

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
                val_sub_x = x_val[:val_samples]
                val_sub_y = y_val[:val_samples]
                v_loss = loss_fn(model_fn(p, val_sub_x, None), val_sub_y)
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
                        checkpoint_callback, p, opt_s, step + 1, val_loss
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

    opt_state = optimizer.init(params) if opt_state is None else opt_state
    (final_params, final_opt_state, _), (train_loss, val_loss) = _run_scan(
        params, opt_state
    )
    return final_params, final_opt_state, train_loss, val_loss
