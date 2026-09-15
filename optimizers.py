import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Callable, Union


@dataclass
class Optimizer:
    init: Callable
    update: Callable


def cosine_schedule(
    init_lr: float = 1e-3,
    total_steps: int = 10000,
    min_lr: float = 1e-4,
    warmup_steps: int = 0,
) -> Callable[[jax.Array], jax.Array]:
    def schedule(step):
        if warmup_steps > 0:
            warmup_lr = min_lr + (init_lr - min_lr) * (step / max(1, warmup_steps))
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            progress = jnp.clip(progress, 0.0, 1.0)
            decay_lr = min_lr + (init_lr - min_lr) * 0.5 * (
                1.0 + jnp.cos(jnp.pi * progress)
            )
            return jnp.where(step < warmup_steps, warmup_lr, decay_lr)
        else:
            progress = jnp.clip(step / max(1, total_steps), 0.0, 1.0)
            return min_lr + (init_lr - min_lr) * 0.5 * (
                1.0 + jnp.cos(jnp.pi * progress)
            )

    return schedule


def adam(
    lr: Union[float, Callable] = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    total_steps: int | None = None,
    min_lr: float = 1e-4,
    warmup_steps: int = 0,
    weight_decay: float = 0.0,
) -> Optimizer:
    if total_steps is not None:
        lr_fn = cosine_schedule(
            init_lr=lr if isinstance(lr, (int, float)) else 1e-3,
            total_steps=total_steps,
            min_lr=min_lr,
            warmup_steps=warmup_steps,
        )
    elif callable(lr):
        lr_fn = lr
    else:
        lr_fn = lambda step: lr

    def init(params):
        # m and v have the exact same tree structure and shape as params, initialized to zeros
        m = jax.tree.map(jnp.zeros_like, params)
        v = jax.tree.map(jnp.zeros_like, params)
        return (m, v)

    def update(params, grads, state, step):
        m, v = state
        t = step + 1  # 1-indexed time step for bias correction
        current_lr = lr_fn(step)

        # Update first and second moments
        m_new = jax.tree.map(
            lambda m_i, g_i: beta1 * m_i + (1.0 - beta1) * g_i, m, grads
        )
        v_new = jax.tree.map(
            lambda v_i, g_i: beta2 * v_i + (1.0 - beta2) * (g_i**2), v, grads
        )

        # Bias corrections
        m_hat = jax.tree.map(lambda m_i: m_i / (1.0 - beta1**t), m_new)
        v_hat = jax.tree.map(lambda v_i: v_i / (1.0 - beta2**t), v_new)

        # Update weights (AdamW decoupled weight decay)
        if weight_decay > 0.0:
            params_new = jax.tree.map(
                lambda w_i, m_h, v_h: (1.0 - current_lr * weight_decay) * w_i
                - current_lr * m_h / (jnp.sqrt(v_h) + eps),
                params,
                m_hat,
                v_hat,
            )
        else:
            params_new = jax.tree.map(
                lambda w_i, m_h, v_h: w_i - current_lr * m_h / (jnp.sqrt(v_h) + eps),
                params,
                m_hat,
                v_hat,
            )

        return params_new, (m_new, v_new)

    return Optimizer(init, update)


def sgd(lr: float = 0.1) -> Optimizer:
    def init(params):
        return None

    def update(params, grads, state, step):
        return jax.tree.map(lambda w, g: w - lr * g, params, grads), None

    return Optimizer(init, update)
