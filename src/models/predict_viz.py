"""Prediction visualisation for Stage B.

Each figure has the five required panels: MRI, ground truth, prediction,
ground-truth overlay, prediction overlay. Colours reuse the Sprint 1 semantic
palette so a class means the same thing in every figure across both sprints.

As elsewhere, this module does not call ``matplotlib.use()`` - the entry point
chooses the backend.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.preprocessing.visualize import (
    SEMANTIC_CMAP,
    overlay_mask,
    semantic_legend_handles,
)


def _disagreement_map(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """RGB map of where prediction and truth differ.

    Green = correct foreground, red = predicted foreground that is not there
    (false positive), blue = missed foreground (false negative).
    """
    rgb = np.zeros((*truth.shape, 3), dtype=np.float32)
    truth_fg = truth > 0
    pred_fg = prediction > 0

    rgb[truth_fg & pred_fg] = (0.0, 0.8, 0.2)   # agreement on foreground
    rgb[~truth_fg & pred_fg] = (0.9, 0.1, 0.1)  # false positive
    rgb[truth_fg & ~pred_fg] = (0.1, 0.4, 1.0)  # false negative
    return rgb


def visualize_prediction(
    image: np.ndarray,
    truth: np.ndarray,
    prediction: np.ndarray,
    *,
    title: str,
    dice_text: str | None = None,
    save_path: Path | str | None = None,
    dpi: int = 130,
    include_disagreement: bool = True,
):
    """Five- (or six-) panel prediction figure for one slice.

    Panels
    ------
    1 MRI slice (preprocessed input the model actually saw)
    2 Ground-truth mask
    3 Predicted mask
    4 MRI + ground-truth overlay
    5 MRI + prediction overlay
    6 Agreement map (optional) - makes the error structure readable, which the
      two overlays side by side do not
    """
    n_panels = 6 if include_disagreement else 5
    figure, axes = plt.subplots(1, n_panels, figsize=(3.3 * n_panels, 4.6))

    full_title = title if dice_text is None else f"{title}\n{dice_text}"
    figure.suptitle(full_title, fontsize=11, y=0.99)

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title(f"1. MRI (input)\n{image.shape[0]}x{image.shape[1]} px",
                      fontsize=9)

    axes[1].imshow(truth, cmap=SEMANTIC_CMAP, vmin=0, vmax=3, interpolation="nearest")
    axes[1].set_title(f"2. Ground truth\n{int((truth > 0).sum()):,} labelled px",
                      fontsize=9)

    axes[2].imshow(prediction, cmap=SEMANTIC_CMAP, vmin=0, vmax=3,
                   interpolation="nearest")
    axes[2].set_title(f"3. Prediction\n{int((prediction > 0).sum()):,} labelled px",
                      fontsize=9)

    axes[3].imshow(overlay_mask(image, truth))
    axes[3].set_title("4. MRI + ground truth", fontsize=9)

    axes[4].imshow(overlay_mask(image, prediction))
    axes[4].set_title("5. MRI + prediction", fontsize=9)

    if include_disagreement:
        axes[5].imshow(_disagreement_map(truth, prediction))
        axes[5].set_title(
            "6. Agreement\ngreen=correct, red=false pos, blue=missed", fontsize=8
        )

    for axis in axes:
        axis.axis("off")

    figure.legend(handles=semantic_legend_handles(), loc="lower center", ncol=3,
                  frameon=False, fontsize=9)
    figure.tight_layout(rect=(0, 0.06, 1, 0.93))

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        return None
    return figure


def plot_training_history(
    history,
    *,
    save_path: Path | str | None = None,
    dpi: int = 130,
):
    """Loss and validation Dice curves from the training history table."""
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.4))

    axes[0].plot(history["epoch"], history["train_loss"], marker="o", label="train")
    if "val_loss" in history.columns:
        axes[0].plot(history["epoch"], history["val_loss"], marker="s", label="val")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss (CE + Dice)")
    axes[0].set_title("Training and validation loss", fontsize=11)
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.25)

    if "val_fg_dice" in history.columns:
        axes[1].plot(history["epoch"], history["train_fg_dice"], marker="o",
                     label="train")
        axes[1].plot(history["epoch"], history["val_fg_dice"], marker="s", label="val")
        axes[1].set_xlabel("epoch")
        axes[1].set_ylabel("macro foreground Dice")
        axes[1].set_title("Foreground Dice (background excluded)", fontsize=11)
        axes[1].legend(frameon=False)
        axes[1].grid(alpha=0.25)

    per_class = [c for c in history.columns if c.startswith("val_dice_")]
    for column in per_class:
        axes[2].plot(history["epoch"], history[column], marker=".",
                     label=column.replace("val_dice_", ""))
    axes[2].set_xlabel("epoch")
    axes[2].set_ylabel("validation Dice")
    axes[2].set_title("Per-class validation Dice", fontsize=11)
    axes[2].legend(frameon=False, fontsize=8)
    axes[2].grid(alpha=0.25)

    figure.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        return None
    return figure


def plot_metric_bars(
    per_class: dict,
    *,
    title: str = "Test-set segmentation metrics per class",
    save_path: Path | str | None = None,
    dpi: int = 130,
):
    """Grouped bars of Dice / IoU / precision / recall per class."""
    names = list(per_class)
    metrics = ["dice", "iou", "precision", "recall"]
    x = np.arange(len(names))
    width = 0.2

    figure, axis = plt.subplots(figsize=(9.5, 4.8))
    for offset, metric in enumerate(metrics):
        values = [per_class[n][metric] for n in names]
        bars = axis.bar(x + (offset - 1.5) * width, values, width, label=metric)
        for bar, value in zip(bars, values):
            if np.isfinite(value):
                axis.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}",
                          ha="center", va="bottom", fontsize=7, rotation=90)

    axis.set_xticks(x, names, fontsize=9)
    axis.set_ylabel("score")
    axis.set_ylim(0, 1.15)
    axis.set_title(title, fontsize=11)
    axis.legend(frameon=False, ncol=4)
    axis.grid(alpha=0.2, axis="y")

    figure.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        return None
    return figure
