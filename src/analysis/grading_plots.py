"""Figures for the radiological grading analysis.

As in ``src.preprocessing.visualize``, this module deliberately does not call
``matplotlib.use()`` - the entry point chooses the backend so the notebook
keeps its inline rendering.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.gradings import (
    FINDINGS,
    IVD_COLUMN,
    MODIC_CATEGORY_NAMES,
    PFIRRMANN_GRADE_NAMES,
)

# Consistent colours: ordinal scales use a sequential map, binary findings a
# single accent colour, so the reader can tell the kinds apart at a glance.
ORDINAL_CMAP = plt.get_cmap("YlOrRd")
BINARY_PRESENT = "#c44e52"
BINARY_ABSENT = "#cfd4da"
NOMINAL_COLORS = ["#cfd4da", "#dd8452", "#4c72b0", "#55a868"]


def _annotate_bars(axis, bars, values, total: int) -> None:
    """Write count and percentage above each bar."""
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:,}\n{100 * value / total:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def plot_finding_distributions(
    raw: pd.DataFrame,
    *,
    save_path: Path | str | None = None,
    dpi: int = 130,
):
    """One panel per graded finding, showing its full value distribution.

    This is the figure the revised requirements ask for: the distribution of
    every radiological finding in the dataset. Absolute counts are plotted
    with percentages annotated, because both matter - prevalence drives the
    clinical reading, counts drive what is statistically learnable.
    """
    valid = raw[raw[IVD_COLUMN] >= 1]
    total = len(valid)

    n = len(FINDINGS)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    figure, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.9 * nrows))
    axes = np.atleast_1d(axes).ravel()

    figure.suptitle(
        f"Distribution of each radiological finding  "
        f"({total:,} graded disc records, {valid['Patient'].nunique()} patients)",
        fontsize=13,
        y=0.995,
    )

    for axis, finding in zip(axes, FINDINGS):
        counts = valid[finding.column].value_counts().sort_index()
        values = counts.index.tolist()

        if finding.kind == "binary":
            labels = ["absent (0)", "present (1)"]
            colors = [BINARY_ABSENT, BINARY_PRESENT]
            heights = [int(counts.get(0, 0)), int(counts.get(1, 0))]
        elif finding.kind == "nominal":
            labels = [MODIC_CATEGORY_NAMES.get(v, str(v)) for v in values]
            colors = [NOMINAL_COLORS[i % len(NOMINAL_COLORS)] for i in range(len(values))]
            heights = counts.tolist()
        else:  # ordinal
            labels = [str(v) for v in values]
            colors = [ORDINAL_CMAP(0.15 + 0.2 * i) for i in range(len(values))]
            heights = counts.tolist()

        bars = axis.bar(labels, heights, color=colors, edgecolor="black", linewidth=0.6)
        _annotate_bars(axis, bars, heights, total)

        title = f"{finding.column}\n({finding.kind})"
        axis.set_title(title, fontsize=10)
        axis.set_ylabel("disc records")
        axis.margins(y=0.22)
        axis.tick_params(axis="x", labelsize=9)

    for axis in axes[n:]:
        axis.axis("off")

    figure.tight_layout(rect=(0, 0, 1, 0.97))
    return _finish(figure, save_path, dpi)


def plot_prevalence_by_level(
    level_table: pd.DataFrame,
    *,
    save_path: Path | str | None = None,
    dpi: int = 130,
):
    """Finding prevalence against disc index, plus records per level.

    Disc index 1 is the most inferior disc. Plotting against it shows whether
    the labels follow the expected anatomical gradient, and makes the very
    uneven per-level record counts visible.
    """
    binary_findings = [f for f in FINDINGS if f.kind == "binary"]
    figure, axes = plt.subplots(1, 3, figsize=(17, 4.8))

    levels = level_table["ivd_label"].tolist()

    # --- panel 1: binary finding prevalence -----------------------------
    for finding in binary_findings:
        axes[0].plot(
            levels,
            level_table[finding.tidy],
            marker="o",
            linewidth=1.8,
            label=finding.column,
        )
    axes[0].plot(
        levels,
        level_table["modic"],
        marker="s",
        linestyle="--",
        linewidth=1.8,
        color="black",
        label="any Modic change",
    )
    axes[0].set_xlabel("IVD label (1 = most inferior disc)")
    axes[0].set_ylabel("prevalence (%)")
    axes[0].set_title("Finding prevalence by disc level", fontsize=11)
    axes[0].legend(fontsize=8, frameon=False)
    axes[0].set_xticks(levels)
    axes[0].grid(alpha=0.25)

    # --- panel 2: mean Pfirrmann grade ----------------------------------
    axes[1].plot(
        levels,
        level_table["pfirrmann_grade"],
        marker="o",
        color="#c44e52",
        linewidth=2.2,
    )
    axes[1].set_xlabel("IVD label (1 = most inferior disc)")
    axes[1].set_ylabel("mean Pfirrmann grade")
    axes[1].set_title("Mean Pfirrmann grade by disc level", fontsize=11)
    axes[1].set_xticks(levels)
    axes[1].grid(alpha=0.25)

    # --- panel 3: records available per level ---------------------------
    bars = axes[2].bar(
        levels,
        level_table["n_records"],
        color="#4c72b0",
        edgecolor="black",
        linewidth=0.6,
    )
    for bar, value in zip(bars, level_table["n_records"]):
        axes[2].text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:,}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    axes[2].set_xlabel("IVD label (1 = most inferior disc)")
    axes[2].set_ylabel("graded disc records")
    axes[2].set_title("Records available per disc level", fontsize=11)
    axes[2].set_xticks(levels)
    axes[2].margins(y=0.15)

    figure.suptitle(
        "Radiological findings across disc levels", fontsize=13, y=1.02
    )
    figure.tight_layout()
    return _finish(figure, save_path, dpi)


def plot_cooccurrence(
    correlation: pd.DataFrame,
    *,
    save_path: Path | str | None = None,
    dpi: int = 130,
):
    """Heatmap of pairwise Spearman correlation between findings."""
    figure, axis = plt.subplots(figsize=(7.6, 6.4))
    image = axis.imshow(correlation.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)

    labels = list(correlation.columns)
    axis.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=9)
    axis.set_yticks(range(len(labels)), labels, fontsize=9)

    for i in range(len(labels)):
        for j in range(len(labels)):
            value = correlation.iloc[i, j]
            axis.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if abs(value) > 0.55 else "black",
            )

    axis.set_title(
        "Co-occurrence of radiological findings\n(Spearman correlation, "
        "ordinal/binary data)",
        fontsize=11,
    )
    figure.colorbar(image, ax=axis, shrink=0.8, label="Spearman rho")
    figure.tight_layout()
    return _finish(figure, save_path, dpi)


def plot_findings_per_disc(
    per_disc: dict,
    *,
    save_path: Path | str | None = None,
    dpi: int = 130,
):
    """How many findings co-occur on one disc - shows the task is multi-label."""
    distribution = per_disc["distribution"]
    keys = sorted(distribution)
    values = [distribution[k] for k in keys]
    total = sum(values)

    figure, axis = plt.subplots(figsize=(7.6, 4.4))
    bars = axis.bar(
        [str(k) for k in keys],
        values,
        color="#55a868",
        edgecolor="black",
        linewidth=0.6,
    )
    _annotate_bars(axis, bars, values, total)
    axis.set_xlabel("number of findings present on the same disc")
    axis.set_ylabel("disc records")
    axis.set_title(
        "Findings co-occurring per disc\n"
        "(6 binary findings + 'any Modic change'; Pfirrmann excluded as every "
        "disc has a grade)",
        fontsize=11,
    )
    axis.margins(y=0.2)
    figure.tight_layout()
    return _finish(figure, save_path, dpi)


def plot_split_class_balance(
    coverage: pd.DataFrame,
    *,
    save_path: Path | str | None = None,
    dpi: int = 130,
):
    """Per-split class counts for the ordinal/nominal targets.

    Makes the thin-class problem concrete: a class with only a handful of test
    records cannot support a trustworthy per-class metric.
    """
    figure, axes = plt.subplots(1, 2, figsize=(14, 4.6))

    for axis, (prefix, names, title) in zip(
        axes,
        [
            ("pfirrmann_", PFIRRMANN_GRADE_NAMES, "Pfirrmann grade"),
            ("modic_", MODIC_CATEGORY_NAMES, "Modic category"),
        ],
    ):
        classes = sorted(names)
        width = 0.26
        x = np.arange(len(classes))
        for offset, (_, row) in enumerate(coverage.iterrows()):
            heights = [int(row[f"{prefix}{c}"]) for c in classes]
            bars = axis.bar(
                x + (offset - 1) * width, heights, width, label=str(row["split"])
            )
            for bar, value in zip(bars, heights):
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    value,
                    str(value),
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
        axis.set_xticks(x, [str(c) for c in classes])
        axis.set_xlabel(title)
        axis.set_ylabel("disc records")
        axis.set_title(f"{title} per split", fontsize=11)
        axis.legend(frameon=False, fontsize=9)
        axis.margins(y=0.18)

    figure.suptitle(
        "Class balance of the grading targets across the Sprint 1 patient-level split",
        fontsize=12,
        y=1.03,
    )
    figure.tight_layout()
    return _finish(figure, save_path, dpi)


def _finish(figure, save_path: Path | str | None, dpi: int):
    """Save and close, or return the figure for inline display."""
    if save_path is None:
        return figure
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return None
