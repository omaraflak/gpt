import math
import pickle
import numpy as np
import matplotlib.pyplot as plt


def list_to_xy(values: list[float]) -> tuple[float, float]:
    x, y = [], []
    for i, v in enumerate(values):
        if not math.isnan(v):
            x.append(i)
            y.append(v)
    return x, y


def smooth(vals: list[float]):
    x_vals, y_vals = list_to_xy(vals)
    nx, ny = [], []
    for i in range(len(x_vals)):
        nx.append(x_vals[i])
        ny.append(np.mean(y_vals[i : i + 5]))
    return nx, ny


paths = [
    "checkpoints/260915_1114/losses.pkl",
]

train_loss = []
val_loss = []

for path in paths:
    with open(path, "rb") as f:
        losses = pickle.load(f)
        train_loss.extend(losses["train_loss"])
        val_loss.extend(losses["val_loss"])

plt.plot(*smooth(train_loss), label="train")
plt.plot(*list_to_xy(val_loss), label="val")
plt.legend()
plt.show()
