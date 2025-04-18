import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from PIL import Image
from matplotlib import rcParams

rcParams.update(
    {
        "axes.edgecolor": "white",
        "axes.facecolor": "#EAEAF2",
        "axes.grid": True,
        "grid.color": "white",
        "grid.linestyle": "--",
        "grid.alpha": 0.6,
        "xtick.color": "black",
        "ytick.color": "black",
        "axes.labelcolor": "black",
        "axes.titleweight": "bold",
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.frameon": True,
        "legend.fontsize": 11,
        "figure.facecolor": "white",
    }
)

colors = {"train": "#13034d", "val": "#084d02"}


def plot_training_metrics(train_losses, val_losses, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(
        epochs,
        train_losses,
        label="Training Loss",
        marker="o",
        color=colors["train"],
        linewidth=2,
    )
    plt.plot(
        epochs,
        val_losses,
        label="Validation Loss",
        marker="s",
        color=colors["val"],
        linewidth=2,
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Metrics", fontweight="bold")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    save_path = os.path.join(save_dir, "training_metrics.png")
    plt.savefig(save_path, format="png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Metrics plot saved to {save_path}")
