import jax
import jax.numpy as jnp
from typing import Callable

Loss = Callable[[jax.Array, jax.Array], jax.Array]

def mse(pred: jax.Array, target: jax.Array) -> jax.Array:
    return jnp.mean((pred - target) ** 2)

def cross_entropy(pred: jax.Array, target: jax.Array) -> jax.Array:
    return -jnp.mean(target * jnp.log(jnp.clip(pred, 1e-9, 1.0)))

def cross_entropy_logits(logits: jax.Array, target: jax.Array) -> jax.Array:
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    if target.ndim == logits.ndim - 1:
        return -jnp.mean(jnp.take_along_axis(log_probs, target[..., None], axis=-1))
    return -jnp.mean(jnp.sum(target * log_probs, axis=-1))
