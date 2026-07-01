import matplotlib.pyplot as plt
import numpy as np

from generic_image_classifier.logging import get_logger
from generic_image_classifier.schemas import Model

logger = get_logger(__name__)

PLOT_STYLE = {
    "figure.facecolor": "#ffffff",
    "axes.facecolor": "#f8fafc",
    "axes.edgecolor": "#cbd5e1",
    "axes.labelcolor": "#334155",
    "axes.titlecolor": "#0f172a",
    "text.color": "#334155",
    "xtick.color": "#64748b",
    "ytick.color": "#64748b",
    "grid.color": "#e2e8f0",
    "grid.alpha": 0.8,
    "font.size": 10,
}

TRAIN_COLOR = "#0f766e"
VAL_COLOR = "#f59e0b"
CM_CMAP = "YlGnBu"


def _fig_to_array(fig: plt.Figure) -> np.ndarray:
    fig.canvas.draw()
    return np.array(fig.canvas.renderer.buffer_rgba())


def plot_training_history(model_schema: Model) -> np.ndarray:
    history = model_schema.history
    with plt.rc_context(PLOT_STYLE):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

        epochs = range(1, len(history.accuracy) + 1)
        ax1.plot(epochs, history.accuracy, color=TRAIN_COLOR, lw=2, label="Train")
        ax1.plot(epochs, history.val_accuracy, color=VAL_COLOR, lw=2, label="Validation")
        ax1.set_title("Accuracy", fontweight="600", pad=10)
        ax1.set_xlabel("Epoch")
        ax1.set_ylim(0, 1.02)
        ax1.legend(frameon=False, loc="lower right")
        ax1.grid(True, linestyle="--", linewidth=0.6)

        ax2.plot(epochs, history.loss, color=TRAIN_COLOR, lw=2, label="Train")
        ax2.plot(epochs, history.val_loss, color=VAL_COLOR, lw=2, label="Validation")
        ax2.set_title("Loss", fontweight="600", pad=10)
        ax2.set_xlabel("Epoch")
        ax2.legend(frameon=False, loc="upper right")
        ax2.grid(True, linestyle="--", linewidth=0.6)

        fig.tight_layout()
        img = _fig_to_array(fig)
        plt.close(fig)
    return img


def plot_confusion_matrix(model_schema: Model) -> np.ndarray:
    if not model_schema.results or not model_schema.results.confusion_matrix:
        logger.warning("No confusion matrix in schema")
        return np.zeros((10, 10, 4), dtype=np.uint8)

    cm = np.array(model_schema.results.confusion_matrix)
    class_names = [
        model_schema.info.class_mapping.get(i, f"Class {i}")
        for i in range(model_schema.info.num_classes)
    ]

    with plt.rc_context(PLOT_STYLE):
        size = max(6, min(12, len(class_names) * 0.65))
        fig, ax = plt.subplots(figsize=(size, size * 0.85))
        im = ax.imshow(cm, cmap=CM_CMAP)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)

        ax.set_xlabel("Predicted", fontweight="500")
        ax.set_ylabel("True", fontweight="500")
        ax.set_title("Confusion matrix", fontweight="600", pad=12)
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=35, ha="right", fontsize=9)
        ax.set_yticklabels(class_names, fontsize=9)

        threshold = cm.max() / 2 if cm.size else 0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if cm[i, j] > threshold else "#0f172a",
                )

        fig.tight_layout()
        img = _fig_to_array(fig)
        plt.close(fig)
    return img
