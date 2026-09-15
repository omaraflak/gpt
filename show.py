import numpy as np
import pickle
import loader
import matplotlib.pyplot as plt

paths = [
    "checkpoints/260914_2311/losses.pkl",
    "checkpoints/260914_2324/losses.pkl",
    "checkpoints/260914_2337/losses.pkl",
    "checkpoints/260914_2359/losses.pkl",
    "checkpoints/260915_0758/losses.pkl",
]

train_loss = []
val_loss = []

for path in paths:
    with open(path, "rb") as f:
        losses = pickle.load(f)
        train_loss.extend(losses["train_loss"])
        val_loss.extend(losses["val_loss"])


def smooth(vals):
    x_vals, y_vals = loader.list_to_xy(vals)
    nx, ny = [], []
    for i in range(len(x_vals)):
        nx.append(x_vals[i])
        ny.append(np.mean(y_vals[i : i + 50]))
    return nx, ny


plt.plot(*smooth(train_loss), label="train")
plt.plot(*loader.list_to_xy(val_loss), label="val")
plt.legend()
plt.show()
