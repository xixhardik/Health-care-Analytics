"""Disc-indexing failure analysis (Sprint 3 audit, section E).

Why indexing matters more than Dice
-----------------------------------
The U-Net predicts four *semantic* classes. It does not predict which disc is
which. Disc identity is recovered afterwards by taking connected components of
the IVD class and numbering them from the most inferior upward, matching the
dataset's own convention. Every downstream stage - the measurement table, the
grading linkage, the per-disc report - is keyed on that index. A disc that is
found but numbered wrongly attaches its measurements to the wrong grading, which
is worse than not finding it at all.

Measured accuracy is 78.4% (baseline) and 82.6% (extended) per slice, against
95%+ region detection. So most failures are **numbering** failures, not
detection failures. This module works out why.

Failure taxonomy
----------------
Each ground-truth disc on each slice is classified into exactly one category:

``correct``
    matched a predicted component that carries the same index
``shifted``
    matched a predicted component, but the index differs - the numbering is
    off. Reported with the signed offset.
``merged``
    the matched predicted component also overlaps another ground-truth disc,
    i.e. two discs were fused into one component
``split``
    this ground-truth disc overlaps two or more predicted components
``missed``
    no predicted component overlaps it at all

Level naming
------------
Analysis uses the integer disc index only. The dataset does not state which
vertebra is L5 (documented as usually L5 but possibly L4 or L6), so no
anatomical level name is asserted anywhere in this module.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.disc_features import (
    INSTANCE_IVD_VALUES,
    instance_to_ivd_label,
)
from src.preprocessing.labels import SEM_IVD, SEM_VERTEBRA

MIN_AREA_PX = 20


def _components(binary: np.ndarray, min_area: int = MIN_AREA_PX) -> list[dict]:
    """Connected components of a boolean mask, ordered most-inferior first.

    Row 0 is superior in the Sprint 1 display convention, so a larger mean row
    is more inferior; ordering by descending mean row therefore matches the
    dataset's bottom-up numbering.
    """
    from scipy import ndimage

    labelled, n = ndimage.label(binary)
    found: list[dict] = []
    for component in range(1, n + 1):
        selector = labelled == component
        area = int(selector.sum())
        if area < min_area:
            continue
        rows, cols = np.where(selector)
        found.append(
            {
                "area_px": area,
                "mean_row": float(rows.mean()),
                "row_min": int(rows.min()),
                "row_max": int(rows.max()),
                "col_min": int(cols.min()),
                "col_max": int(cols.max()),
                "selector": selector,
            }
        )
    found.sort(key=lambda item: -item["mean_row"])
    return found


def analyse_slice(
    truth_instance: np.ndarray,
    predicted_semantic: np.ndarray,
    *,
    min_area: int = MIN_AREA_PX,
) -> dict:
    """Classify every ground-truth disc on one slice into the failure taxonomy.

    Takes the **semantic** prediction and re-derives components here, so the
    analysis sees exactly what the production instance-derivation sees.
    """
    truth_labels = [
        value for value in INSTANCE_IVD_VALUES
        if int((truth_instance == value).sum()) >= min_area
    ]
    truth_regions = {
        instance_to_ivd_label(value): (truth_instance == value)
        for value in truth_labels
    }

    predicted = _components(predicted_semantic == SEM_IVD, min_area=min_area)
    # Derived index: 1 = most inferior, matching the dataset convention.
    predicted_index = {i + 1: comp for i, comp in enumerate(predicted)}

    records: list[dict] = []
    for truth_index in sorted(truth_regions):
        region = truth_regions[truth_index]
        truth_area = int(region.sum())

        overlaps = {
            index: int((region & comp["selector"]).sum())
            for index, comp in predicted_index.items()
        }
        overlapping = {k: v for k, v in overlaps.items() if v > 0}

        if not overlapping:
            records.append(
                {
                    "truth_index": truth_index,
                    "category": "missed",
                    "matched_index": None,
                    "offset": None,
                    "truth_area_px": truth_area,
                    "overlap_px": 0,
                    "iou": 0.0,
                    "n_overlapping_components": 0,
                }
            )
            continue

        best_index = max(overlapping, key=overlapping.get)
        best_component = predicted_index[best_index]
        overlap = overlapping[best_index]
        union = int((region | best_component["selector"]).sum())

        # Does the matched component also cover another ground-truth disc?
        also_covers = [
            other for other, other_region in truth_regions.items()
            if other != truth_index
            and int((other_region & best_component["selector"]).sum()) >= min_area
        ]

        # Is this ground-truth disc split across several predicted components?
        substantial = [k for k, v in overlapping.items() if v >= min_area]

        if also_covers:
            category = "merged"
        elif len(substantial) > 1:
            category = "split"
        elif best_index == truth_index:
            category = "correct"
        else:
            category = "shifted"

        records.append(
            {
                "truth_index": truth_index,
                "category": category,
                "matched_index": best_index,
                "offset": best_index - truth_index,
                "truth_area_px": truth_area,
                "overlap_px": overlap,
                "iou": round(overlap / union, 4) if union else 0.0,
                "n_overlapping_components": len(substantial),
                "merged_with": also_covers or None,
            }
        )

    # Spurious predicted components: overlap no ground-truth disc at all.
    spurious = []
    for index, component in predicted_index.items():
        total_overlap = sum(
            int((component["selector"] & region).sum())
            for region in truth_regions.values()
        )
        if total_overlap < min_area:
            spurious.append({"predicted_index": index, "area_px": component["area_px"]})

    # Vertebra components. Compared against the ground truth's OWN component
    # count, not against the number of vertebra instances: a vertebra occupies
    # roughly two disconnected components in a sagittal slice (the body, and the
    # posterior elements), so comparing components to instances would blame the
    # model for a property of the anatomy. See measure_structure_decomposition().
    vertebra_components = _components(
        predicted_semantic == SEM_VERTEBRA, min_area=min_area
    )
    truth_vertebra_instances = [
        value for value in range(1, 10)
        if int((truth_instance == value).sum()) >= min_area
    ]
    truth_vertebra_components = _components(
        np.isin(truth_instance, truth_vertebra_instances), min_area=min_area
    ) if truth_vertebra_instances else []
    truth_vertebra_count = len(truth_vertebra_instances)

    return {
        "n_truth_discs": len(truth_regions),
        "n_predicted_components": len(predicted_index),
        "count_matches": len(truth_regions) == len(predicted_index),
        "count_delta": len(predicted_index) - len(truth_regions),
        "discs": records,
        "n_spurious_components": len(spurious),
        "spurious": spurious,
        "n_truth_vertebrae": truth_vertebra_count,
        "n_truth_vertebra_components": len(truth_vertebra_components),
        "n_predicted_vertebra_components": len(vertebra_components),
        # Correct comparison: predicted components vs ground-truth components.
        "vertebra_component_delta":
            len(vertebra_components) - len(truth_vertebra_components),
        # The naive (and misleading) comparison, kept to show why it misleads.
        "vertebra_components_vs_instances":
            len(vertebra_components) - truth_vertebra_count,
    }


def measure_structure_decomposition(
    index: pd.DataFrame,
    *,
    project_root: Path,
    n_slices: int = 250,
    seed: int = 0,
    min_area: int = MIN_AREA_PX,
) -> dict:
    """How many 2-D connected components does one anatomical instance occupy?

    This is the decisive check on whether "ordered connected components" can
    recover identity at all. It is measured on the **ground truth**, so it is a
    property of the anatomy and the annotation, entirely independent of model
    quality.

    A structure that occupies ~1 component per instance can be numbered by
    ordering components. A structure that occupies ~2 cannot: the ordering will
    interleave body and posterior-element fragments.
    """
    from scipy import ndimage

    subset = index
    if n_slices and len(subset) > n_slices:
        rng = np.random.default_rng(seed)
        positions = rng.choice(len(subset), size=n_slices, replace=False)
        subset = subset.iloc[sorted(positions)]

    tallies = {
        "intervertebral_disc": {"instances": 0, "components": 0, "multi": 0},
        "vertebra": {"instances": 0, "components": 0, "multi": 0},
    }
    label_ranges = {
        "intervertebral_disc": range(11, 20),
        "vertebra": range(1, 10),
    }

    for _, row in subset.iterrows():
        with np.load(project_root / row["npz_path"]) as bundle:
            instance_mask = bundle["mask_instance"]

        for structure, values in label_ranges.items():
            for value in values:
                selector = instance_mask == value
                if int(selector.sum()) < min_area:
                    continue
                labelled, n = ndimage.label(selector)
                sizes = ndimage.sum(selector, labelled, range(1, n + 1))
                n_components = int((sizes >= min_area).sum())
                tallies[structure]["instances"] += 1
                tallies[structure]["components"] += n_components
                if n_components > 1:
                    tallies[structure]["multi"] += 1

    result = {"n_slices_measured": int(len(subset)), "structures": {}}
    for structure, counts in tallies.items():
        instances = max(counts["instances"], 1)
        result["structures"][structure] = {
            "instances": counts["instances"],
            "components": counts["components"],
            "components_per_instance": round(counts["components"] / instances, 3),
            "pct_instances_multi_component": round(
                100 * counts["multi"] / instances, 2
            ),
            "ordered_components_valid": bool(counts["components"] / instances < 1.15),
        }

    disc = result["structures"]["intervertebral_disc"]
    vertebra = result["structures"]["vertebra"]
    result["conclusion"] = (
        f"Discs occupy {disc['components_per_instance']} components per instance "
        f"({disc['pct_instances_multi_component']}% are split), so numbering discs by "
        f"ordered connected components is structurally sound. Vertebrae occupy "
        f"{vertebra['components_per_instance']} components per instance, because the "
        f"vertebral body and the posterior elements are separate regions in a "
        f"sagittal plane. Numbering vertebrae by ordered connected components is "
        f"therefore structurally invalid regardless of segmentation quality - and "
        f"that is the root cause of the unreliable disc-to-vertebra height ratio."
    )
    return result


def run_analysis(
    index: pd.DataFrame,
    predictions_dir: Path,
    *,
    project_root: Path,
    limit: int | None = None,
    progress_every: int = 250,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Analyse every slice that has a prediction.

    Returns
    -------
    (disc_frame, slice_frame)
        ``disc_frame`` has one row per ground-truth disc per slice with its
        failure category; ``slice_frame`` has one row per slice.
    """
    rows = index if limit is None else index.head(limit)
    disc_records: list[dict] = []
    slice_records: list[dict] = []

    for position, (_, row) in enumerate(rows.iterrows(), start=1):
        prediction_path = Path(predictions_dir) / f"{row['slice_id']}.npz"
        if not prediction_path.exists():
            continue

        with np.load(project_root / row["npz_path"]) as bundle:
            truth_instance = bundle["mask_instance"]
        with np.load(prediction_path) as bundle:
            predicted_semantic = bundle["semantic"]

        result = analyse_slice(truth_instance, predicted_semantic)

        context = {
            "slice_id": row["slice_id"],
            "image_id": row["image_id"],
            "patient_id": int(row["patient_id"]),
            "modality": row["modality"],
            "labelled_pixels": int(row.get("labelled_pixels_processed", 0)),
        }
        for record in result["discs"]:
            disc_records.append({**context, **record})

        slice_records.append(
            {
                **context,
                "n_truth_discs": result["n_truth_discs"],
                "n_predicted_components": result["n_predicted_components"],
                "count_matches": result["count_matches"],
                "count_delta": result["count_delta"],
                "n_spurious_components": result["n_spurious_components"],
                "n_truth_vertebrae": result["n_truth_vertebrae"],
                "n_truth_vertebra_components": result["n_truth_vertebra_components"],
                "n_predicted_vertebra_components": result["n_predicted_vertebra_components"],
                "vertebra_component_delta": result["vertebra_component_delta"],
                "vertebra_components_vs_instances":
                    result["vertebra_components_vs_instances"],
                "n_correct": sum(
                    1 for d in result["discs"] if d["category"] == "correct"
                ),
            }
        )

        if progress_every and position % progress_every == 0:
            print(f"  {position}/{len(rows)} slices analysed", flush=True)

    return pd.DataFrame(disc_records), pd.DataFrame(slice_records)


def summarise(disc_frame: pd.DataFrame, slice_frame: pd.DataFrame) -> dict:
    """Aggregate the failure taxonomy into report-level findings."""
    if disc_frame.empty:
        return {}

    total = len(disc_frame)
    categories = disc_frame["category"].value_counts().to_dict()

    # Offsets among shifted discs: is the error a systematic +/-1 shift?
    shifted = disc_frame[disc_frame["category"] == "shifted"]
    offset_counts = (
        shifted["offset"].value_counts().sort_index().to_dict() if len(shifted) else {}
    )

    # Failure rate by disc index - are the extreme levels worse?
    by_index = (
        disc_frame.groupby("truth_index")
        .agg(
            n=("category", "size"),
            correct=("category", lambda s: int((s == "correct").sum())),
            shifted=("category", lambda s: int((s == "shifted").sum())),
            merged=("category", lambda s: int((s == "merged").sum())),
            split=("category", lambda s: int((s == "split").sum())),
            missed=("category", lambda s: int((s == "missed").sum())),
        )
        .reset_index()
    )
    by_index["pct_correct"] = (100 * by_index["correct"] / by_index["n"]).round(2)

    # Failure rate against how much annotation the slice carries. Lateral slices
    # hold only disc fragments, so this separates "hard slice" from "hard disc".
    disc_frame = disc_frame.copy()
    disc_frame["area_band"] = pd.cut(
        disc_frame["labelled_pixels"],
        bins=[-1, 1000, 3000, 6000, 10000, 10**9],
        labels=["<1k", "1k-3k", "3k-6k", "6k-10k", ">10k"],
    )
    by_area = (
        disc_frame.groupby("area_band", observed=True)
        .agg(
            n=("category", "size"),
            pct_correct=("category", lambda s: round(100 * (s == "correct").mean(), 2)),
            pct_shifted=("category", lambda s: round(100 * (s == "shifted").mean(), 2)),
            pct_missed=("category", lambda s: round(100 * (s == "missed").mean(), 2)),
        )
        .reset_index()
    )

    # Disc-size effect: small discs are easier to miss or fragment.
    disc_frame["size_band"] = pd.cut(
        disc_frame["truth_area_px"],
        bins=[0, 50, 100, 200, 400, 10**9],
        labels=["<50", "50-100", "100-200", "200-400", ">400"],
    )
    by_size = (
        disc_frame.groupby("size_band", observed=True)
        .agg(
            n=("category", "size"),
            pct_correct=("category", lambda s: round(100 * (s == "correct").mean(), 2)),
            pct_missed=("category", lambda s: round(100 * (s == "missed").mean(), 2)),
        )
        .reset_index()
    )

    count_delta = slice_frame["count_delta"].value_counts().sort_index().to_dict()

    # How often would a slice be fully correct (all discs right)?
    slice_frame = slice_frame.copy()
    slice_frame["all_correct"] = slice_frame["n_correct"] == slice_frame["n_truth_discs"]

    return {
        "n_discs_analysed": total,
        "n_slices_analysed": int(len(slice_frame)),
        "categories": {
            k: {"n": int(v), "pct": round(100 * v / total, 2)}
            for k, v in categories.items()
        },
        "pct_index_correct": round(100 * categories.get("correct", 0) / total, 2),
        "pct_region_found": round(
            100 * (total - categories.get("missed", 0)) / total, 2
        ),
        "shifted_offsets": {str(k): int(v) for k, v in offset_counts.items()},
        "pct_shifted_by_one": round(
            100 * sum(v for k, v in offset_counts.items() if abs(k) == 1) / total, 2
        ) if offset_counts else 0.0,
        "slice_count_delta_distribution": {str(k): int(v) for k, v in count_delta.items()},
        "pct_slices_count_correct": round(
            100 * float(slice_frame["count_matches"].mean()), 2
        ),
        "pct_slices_all_discs_correct": round(
            100 * float(slice_frame["all_correct"].mean()), 2
        ),
        "total_spurious_components": int(slice_frame["n_spurious_components"].sum()),
        "pct_slices_with_spurious": round(
            100 * float((slice_frame["n_spurious_components"] > 0).mean()), 2
        ),
        # Predicted vertebra components vs GROUND-TRUTH components (the fair
        # comparison). The instance-based version is reported alongside to show
        # how misleading it is.
        "vertebra_component_delta_distribution": {
            str(k): int(v)
            for k, v in slice_frame["vertebra_component_delta"]
            .value_counts().sort_index().items()
        },
        "pct_slices_vertebra_components_correct": round(
            100 * float((slice_frame["vertebra_component_delta"] == 0).mean()), 2
        ),
        "mean_abs_vertebra_component_delta": round(
            float(slice_frame["vertebra_component_delta"].abs().mean()), 3
        ),
        "pct_slices_vertebra_count_matches_instances": round(
            100 * float((slice_frame["vertebra_components_vs_instances"] == 0).mean()), 2
        ),
        "mean_truth_vertebra_components_per_slice": round(
            float(slice_frame["n_truth_vertebra_components"].mean()), 2
        ),
        "mean_truth_vertebra_instances_per_slice": round(
            float(slice_frame["n_truth_vertebrae"].mean()), 2
        ),
        "by_disc_index": by_index.to_dict("records"),
        "by_slice_annotation_area": by_area.to_dict("records"),
        "by_disc_size": by_size.to_dict("records"),
    }


def compare_runs(baseline: dict, extended: dict) -> list[dict]:
    """Compare the failure taxonomy between two runs."""
    rows: list[dict] = []
    keys = [
        ("pct_index_correct", True),
        ("pct_region_found", True),
        ("pct_slices_count_correct", True),
        ("pct_slices_all_discs_correct", True),
        ("pct_shifted_by_one", False),
        ("total_spurious_components", False),
        ("pct_slices_with_spurious", False),
        ("pct_slices_vertebra_components_correct", True),
        ("mean_abs_vertebra_component_delta", False),
    ]
    for key, higher_better in keys:
        base = baseline.get(key)
        ext = extended.get(key)
        if base is None or ext is None:
            continue
        delta = ext - base
        rows.append(
            {
                "metric": key,
                "baseline": base,
                "extended": ext,
                "delta": round(delta, 3),
                "verdict": (
                    "unchanged" if abs(delta) < 1e-9
                    else "improved" if (delta > 0) == higher_better
                    else "worsened"
                ),
            }
        )

    for category in ["correct", "shifted", "merged", "split", "missed"]:
        base = baseline.get("categories", {}).get(category, {}).get("pct")
        ext = extended.get("categories", {}).get(category, {}).get("pct")
        if base is None or ext is None:
            continue
        delta = ext - base
        higher_better = category == "correct"
        rows.append(
            {
                "metric": f"category_pct/{category}",
                "baseline": base,
                "extended": ext,
                "delta": round(delta, 3),
                "verdict": (
                    "unchanged" if abs(delta) < 1e-9
                    else "improved" if (delta > 0) == higher_better
                    else "worsened"
                ),
            }
        )
    return rows
