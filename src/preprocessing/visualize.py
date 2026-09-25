"""Qualitative visualisation of the preprocessing pipeline.

Figures are the main way Sprint 1 progress is demonstrated, so each one is
built to be self-explanatory: panel titles carry the shape and intensity
range, and masks use a fixed colour per class so colours mean the same thing
in every figure.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from src.preprocessing.labels import (
    SEM_CANAL,
    SEM_IVD,
    SEM_VERTEBRA,
    SEMANTIC_CLASSES,
    to_semantic,
)

# NOTE on the matplotlib backend: this module deliberately does *not* call
# ``matplotlib.use()``. Forcing a backend here would switch the notebook away
# from its inline backend on import and silently suppress every figure.
# Choosing the backend is the entry point's job: the headless scripts in
# ``scripts/`` select "Agg" themselves before importing this module.

# Fixed colour per semantic class. Background is transparent-black so it can
# be used both as a standalone image and as an overlay.
SEMANTIC_COLORS: dict[int, str] = {
    0: "#000000",  # background
    SEM_VERTEBRA: "#ff4d4d",  # red
    SEM_IVD: "#4dff88",  # green
    SEM_CANAL: "#4d94ff",  # blue
}

SEMANTIC_CMAP = ListedColormap([SEMANTIC_COLORS[i] for i in range(4)])


def semantic_legend_handles() -> list[Patch]:
    """Legend entries for the semantic classes (background omitted)."""
    return [
        Patch(facecolor=SEMANTIC_COLORS[value], edgecolor="white", label=name)
        for value, name in SEMANTIC_CLASSES.items()
        if value != 0
    ]


def overlay_mask(
    image: np.ndarray, semantic_mask: np.ndarray, alpha: float = 0.45
) -> np.ndarray:
    """Blend a semantic mask over a grayscale image, returning RGB in [0, 1].

    Only labelled pixels are tinted; background pixels keep the original
    grayscale value, so the underlying anatomy stays readable.
    """
    image = np.asarray(image, dtype=np.float32)
    # Rescale for display only - does not affect the saved data.
    span = image.max() - image.min()
    display = (image - image.min()) / span if span > 0 else np.zeros_like(image)
    rgb = np.dstack([display] * 3)

    for value, color in SEMANTIC_COLORS.items():
        if value == 0:
            continue
        selector = semantic_mask == value
        if not selector.any():
            continue
        rgb_color = np.array(
            [int(color[i : i + 2], 16) / 255.0 for i in (1, 3, 5)], dtype=np.float32
        )
        rgb[selector] = (1 - alpha) * rgb[selector] + alpha * rgb_color
    return np.clip(rgb, 0, 1)


def _show_gray(axis, data: np.ndarray, title: str) -> None:
    axis.imshow(data, cmap="gray")
    axis.set_title(title, fontsize=9)
    axis.axis("off")


def _show_mask(axis, mask: np.ndarray, title: str) -> None:
    axis.imshow(mask, cmap=SEMANTIC_CMAP, vmin=0, vmax=3, interpolation="nearest")
    axis.set_title(title, fontsize=9)
    axis.axis("off")


def visualize_sample(
    original_image: np.ndarray,
    original_mask: np.ndarray,
    processed_image: np.ndarray,
    processed_mask: np.ndarray,
    *,
    title: str,
    save_path: Path | str | None = None,
    dpi: int = 130,
):
    """Five-panel figure for one slice: the required A-E comparison.

    Panels
    ------
    A original MRI slice (raw intensities, native resolution)
    B original segmentation mask
    C preprocessed MRI slice
    D preprocessed mask
    E preprocessed MRI with the mask overlaid

    ``original_mask`` is expected in **raw** label values and is collapsed to
    semantic classes for display; ``processed_mask`` is already semantic.
    """
    original_semantic = to_semantic(original_mask)

    figure, axes = plt.subplots(1, 5, figsize=(18, 4.4))
    figure.suptitle(title, fontsize=12, y=0.99)

    _show_gray(
        axes[0],
        original_image,
        f"A. Original MRI\n{original_image.shape[0]}x{original_image.shape[1]} px, "
        f"range [{original_image.min():.0f}, {original_image.max():.0f}]",
    )
    _show_mask(
        axes[1],
        original_semantic,
        f"B. Original mask\n{original_mask.shape[0]}x{original_mask.shape[1]} px, "
        f"{len(np.unique(original_mask)) - 1} structures",
    )
    _show_gray(
        axes[2],
        processed_image,
        f"C. Preprocessed MRI\n{processed_image.shape[0]}x{processed_image.shape[1]} px, "
        f"range [{processed_image.min():.2f}, {processed_image.max():.2f}]",
    )
    _show_mask(
        axes[3],
        processed_mask,
        f"D. Preprocessed mask\n{processed_mask.shape[0]}x{processed_mask.shape[1]} px, "
        f"nearest-neighbour",
    )
    axes[4].imshow(overlay_mask(processed_image, processed_mask))
    axes[4].set_title("E. Preprocessed MRI + mask overlay", fontsize=9)
    axes[4].axis("off")

    figure.legend(
        handles=semantic_legend_handles(),
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.06, 1, 0.96))

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        return None
    return figure


def visualize_before_after(
    original_image: np.ndarray,
    processed_image: np.ndarray,
    *,
    title: str,
    save_path: Path | str | None = None,
    dpi: int = 130,
):
    """Before/after figure focused on the *intensity* effect of preprocessing.

    Shows both slices with their intensity histograms. The histograms are the
    point of this figure: they make the two different raw intensity
    conventions and the effect of normalisation visible, which a side-by-side
    grayscale pair alone does not.
    """
    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    figure.suptitle(title, fontsize=12)

    _show_gray(
        axes[0, 0],
        original_image,
        f"Before: {original_image.shape[0]}x{original_image.shape[1]} px, "
        f"range [{original_image.min():.0f}, {original_image.max():.0f}]",
    )
    _show_gray(
        axes[0, 1],
        processed_image,
        f"After: {processed_image.shape[0]}x{processed_image.shape[1]} px, "
        f"range [{processed_image.min():.2f}, {processed_image.max():.2f}]",
    )

    axes[1, 0].hist(original_image.ravel(), bins=120, color="#666666")
    axes[1, 0].set_title("Before: intensity histogram", fontsize=9)
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_xlabel("raw intensity")
    axes[1, 0].set_ylabel("voxel count (log)")

    axes[1, 1].hist(processed_image.ravel(), bins=120, color="#1f77b4")
    axes[1, 1].set_title("After: intensity histogram (normalised to [0, 1])", fontsize=9)
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xlabel("normalised intensity")
    axes[1, 1].set_ylabel("pixel count (log)")

    figure.tight_layout(rect=(0, 0, 1, 0.95))
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        return None
    return figure


def visualize_preprocessing_stages(
    stages: dict[str, np.ndarray],
    *,
    title: str,
    save_path: Path | str | None = None,
    dpi: int = 130,
):
    """Show the pipeline stage by stage so each step's effect is separable.

    ``stages`` maps a stage name to the image after that stage, in order.
    Used to justify the optional steps (denoising, CLAHE) visually rather
    than asserting they help.
    """
    n = len(stages)
    figure, axes = plt.subplots(1, n, figsize=(3.4 * n, 4.0))
    figure.suptitle(title, fontsize=12)
    if n == 1:
        axes = [axes]

    for axis, (name, data) in zip(axes, stages.items()):
        _show_gray(
            axis,
            data,
            f"{name}\n{data.shape[0]}x{data.shape[1]}, "
            f"[{data.min():.2f}, {data.max():.2f}]",
        )

    figure.tight_layout(rect=(0, 0, 1, 0.93))
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        return None
    return figure


def visualize_class_distribution(
    class_pixel_counts: dict[str, int],
    *,
    title: str = "Semantic class pixel distribution (preprocessed dataset)",
    save_path: Path | str | None = None,
    dpi: int = 130,
):
    """Bar chart of class frequency on a log scale.

    Log scale because background outnumbers the discs by several orders of
    magnitude; on a linear axis the foreground classes are invisible.
    """
    names = list(class_pixel_counts)
    values = [class_pixel_counts[n] for n in names]

    figure, axis = plt.subplots(figsize=(7, 4.4))
    colors = [SEMANTIC_COLORS.get(i, "#888888") for i in range(len(names))]
    bars = axis.bar(names, values, color=colors, edgecolor="black", linewidth=0.6)
    axis.set_yscale("log")
    axis.set_ylabel("pixel count (log scale)")
    axis.set_title(title, fontsize=11)

    total = sum(values)
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:,}\n({100 * value / total:.2f}%)",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    axis.margins(y=0.18)
    figure.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        return None
    return figure


def visualize_split(split_counts, *, save_path: Path | str | None = None, dpi: int = 130):
    """Grouped bar chart of patients / series / slices per split."""
    splits = list(split_counts)
    metrics = ["patients", "series", "slices"]
    x = np.arange(len(splits))
    width = 0.26

    figure, axis = plt.subplots(figsize=(7.5, 4.4))
    for offset, metric in enumerate(metrics):
        values = [split_counts[s][metric] for s in splits]
        bars = axis.bar(x + (offset - 1) * width, values, width, label=metric)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:,}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    axis.set_xticks(x, splits)
    axis.set_ylabel("count")
    axis.set_title("Patient-level train / validation / test split", fontsize=11)
    axis.legend(frameon=False)
    axis.margins(y=0.15)
    figure.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        return None
    return figure


def visualize_interpolation_comparison(
    mask_plane: np.ndarray,
    src_spacing: tuple[float, float],
    config,
    *,
    title: str = "Why masks must use nearest-neighbour interpolation",
    save_path: Path | str | None = None,
    dpi: int = 130,
):
    """Demonstrate the label corruption that averaging interpolation causes.

    Resizes the *same* mask with nearest-neighbour and with bilinear/bicubic,
    and reports how many label values each produces. Bilinear averages
    neighbouring label *numbers*, which invents values that correspond to no
    anatomical structure - e.g. averaging vertebra 3 and vertebra 4 yields 3.5,
    and averaging a disc (201) with background (0) yields ~100, which happens
    to be the spinal canal's label.

    This figure is the visual proof behind the nearest-neighbour-only rule.
    """
    import cv2

    from src.preprocessing.transforms import center_crop_or_pad

    row_mm, col_mm = src_spacing
    rows, cols = mask_plane.shape[:2]
    new_rows = max(1, int(round(rows * row_mm / config.target_spacing_mm)))
    new_cols = max(1, int(round(cols * col_mm / config.target_spacing_mm)))

    original_labels = set(int(v) for v in np.unique(mask_plane))
    variants: dict[str, tuple[np.ndarray, int]] = {}

    for name, flag in (
        ("NEAREST (used)", cv2.INTER_NEAREST),
        ("BILINEAR (wrong)", cv2.INTER_LINEAR),
        ("BICUBIC (wrong)", cv2.INTER_CUBIC),
    ):
        # Float input is required to expose what averaging actually produces.
        resized = cv2.resize(
            mask_plane.astype(np.float32), (new_cols, new_rows), interpolation=flag
        )
        cropped = center_crop_or_pad(resized, config.target_size, pad_value=0)
        produced = set(int(round(v)) for v in np.unique(cropped))
        variants[name] = (cropped, len(produced - original_labels))

    figure, axes = plt.subplots(1, 4, figsize=(17, 4.6))
    figure.suptitle(title, fontsize=12)

    axes[0].imshow(mask_plane, cmap="nipy_spectral", interpolation="nearest")
    axes[0].set_title(
        f"Original mask\n{rows}x{cols} px, {len(original_labels)} label values",
        fontsize=9,
    )
    axes[0].axis("off")

    for axis, (name, (data, n_invented)) in zip(axes[1:], variants.items()):
        axis.imshow(data, cmap="nipy_spectral", interpolation="nearest")
        verdict = "no invented labels" if n_invented == 0 else f"{n_invented} INVENTED labels"
        axis.set_title(f"{name}\n{data.shape[0]}x{data.shape[1]} px, {verdict}", fontsize=9)
        axis.axis("off")

    figure.tight_layout(rect=(0, 0, 1, 0.93))
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        return {name: n for name, (_, n) in variants.items()}
    return figure


def visualize_dataset_grid(
    samples: list[dict],
    *,
    title: str = "Preprocessed dataset overview",
    save_path: Path | str | None = None,
    dpi: int = 130,
):
    """Grid of overlay thumbnails, two rows per sample set.

    ``samples`` is a list of dicts with ``image``, ``mask`` and ``label`` keys.
    Gives an at-a-glance view that the whole dataset came out consistent -
    same size, same orientation, same intensity scale.
    """
    n = len(samples)
    figure, axes = plt.subplots(2, n, figsize=(2.6 * n, 6.0))
    figure.suptitle(title, fontsize=12)

    for column, sample in enumerate(samples):
        top = axes[0, column] if n > 1 else axes[0]
        bottom = axes[1, column] if n > 1 else axes[1]

        _show_gray(top, sample["image"], sample["label"])
        bottom.imshow(overlay_mask(sample["image"], sample["mask"]))
        bottom.set_title("+ mask overlay", fontsize=8)
        bottom.axis("off")

    figure.legend(
        handles=semantic_legend_handles(), loc="lower center", ncol=3,
        frameon=False, fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 0.94))

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        return None
    return figure
