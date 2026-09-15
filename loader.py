import os
import math
from typing import Callable
import jax
import gzip
import urllib.request
import jax.numpy as jnp
import pickle


def chunk_tokens_for_llm(
    tokens: list[int], seq: int = 32, num_samples: int = 0, stride_factor: int = 2
) -> tuple[jax.Array, jax.Array]:
    data = jnp.array(tokens, dtype=jnp.int32)
    step = seq // stride_factor
    total_possible = (len(data) - seq) // step
    n = min(num_samples, total_possible) if num_samples else total_possible
    x_train = jnp.stack([data[i * step : i * step + seq] for i in range(n)])
    y_train = jnp.stack([data[i * step + 1 : i * step + seq + 1] for i in range(n)])
    return x_train, y_train


def list_to_xy(values: list[float]) -> tuple[float, float]:
    x, y = [], []
    for i, v in enumerate(values):
        if not math.isnan(v):
            x.append(i)
            y.append(v)
    return x, y


def files_to_str(files: list[str]) -> str:
    text = ""
    for file in files:
        with open(file, "r") as f:
            text += f.read()
    return text
