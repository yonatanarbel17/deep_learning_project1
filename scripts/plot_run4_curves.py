"""
Generate the per-epoch accuracy plot for Run 4 (final, 30 epochs).

Reproduces the epoch-by-epoch summary printed at the end of training and
saves the figure to results/run4_training_curves.png.
"""

import os
import matplotlib.pyplot as plt

EPOCHS = list(range(1, 31))
TRAIN_ACC = [
    37.58, 67.75, 74.30, 78.50, 82.23, 84.62, 85.92, 86.97, 88.04, 88.31,
    89.15, 90.63, 90.72, 91.15, 91.61, 91.82, 92.29, 92.66, 92.82, 93.66,
    93.82, 93.56, 93.67, 93.88, 94.02, 93.80, 94.28, 94.88, 95.00, 94.74,
]
VAL_ACC = [
    70.43, 76.67, 79.57, 83.03, 86.27, 88.95, 89.75, 90.29, 90.71, 91.28,
    92.42, 92.49, 92.57, 93.03, 93.37, 93.95, 94.30, 94.16, 94.39, 94.42,
    94.09, 94.42, 94.39, 94.78, 94.97, 94.98, 95.03, 94.99, 95.09, 95.37,
]
UNFREEZE_EPOCH = 4


def main() -> None:
    output_dir = os.path.join(os.path.dirname(__file__), os.pardir, "results")
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "run4_training_curves.png")

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(EPOCHS, TRAIN_ACC, color="#2563eb", linewidth=2.5,
            marker="o", markersize=5, label="Train Accuracy")
    ax.plot(EPOCHS, VAL_ACC, color="#059669", linewidth=2.5,
            marker="s", markersize=5, linestyle="--", label="Validation Accuracy")

    ax.axvline(UNFREEZE_EPOCH, color="#d97706", linestyle=":", linewidth=2, alpha=0.8)
    ax.text(UNFREEZE_EPOCH + 0.2, 72.5, "Backbone unfrozen\n(epoch 4, LR halved)",
            color="#d97706", fontsize=11, va="bottom")

    best_idx = max(range(len(VAL_ACC)), key=VAL_ACC.__getitem__)
    best_epoch = EPOCHS[best_idx]
    best_val = VAL_ACC[best_idx]
    ax.annotate(
        f"Best: {best_val:.2f}%\n(Epoch {best_epoch})",
        xy=(best_epoch, best_val),
        xytext=(best_epoch - 6, best_val - 6),
        fontsize=13,
        color="#059669",
        arrowprops=dict(arrowstyle="->", color="#059669", lw=1.5),
    )

    ax.set_xlabel("Epoch", fontsize=14)
    ax.set_ylabel("Accuracy (%)", fontsize=14)
    ax.set_title("Run 4 — Training & Validation Accuracy (30 Epochs)", fontsize=16)
    ax.set_xlim(0.5, 30.5)
    ax.set_ylim(35, 100)
    ax.set_xticks(list(range(1, 31, 2)) + [30])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12, loc="lower right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
