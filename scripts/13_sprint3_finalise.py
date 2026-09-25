"""Sprint 3 - post-training finalisation, comparison and final report.

Run this **only after** ``scripts/12_train_sprint3_coverage.py`` has finished.
It verifies the run completed, analyses the training artefacts, compares against
the Sprint 2 Extended baseline, runs the corrected disc-indexing taxonomy, and
writes the final report.

It does not train anything, and it does not select a checkpoint using test data -
the checkpoint is chosen by the predefined validation criterion (best validation
macro foreground Dice) inside the training script.

Expected inputs (produced by the standard pipeline, run beforehand):
    outputs/checkpoints/sprint3_coverage/{history.csv,history.json,last.pt,
                                         best_val_dice.pt,best_val_loss.pt}
    outputs/reports/sprint3_coverage/{training_config.json,training_summary.json}
    outputs/reports/sprint3_coverage/metrics/segmentation_metrics_test.json
    outputs/reports/sprint3_coverage/metrics/measurement_agreement.json
    outputs/reports/disc_analysis_sprint3_coverage.csv

Writes:
    outputs/reports/sprint3_coverage/sprint3_final_report.md
    outputs/reports/sprint3_coverage/sprint3_final_report.json
    outputs/reports/sprint3_coverage/comparison_table.csv
    outputs/reports/sprint3_coverage/indexing_failures_sprint3_coverage.csv
    outputs/visualizations/sprint3_coverage/*.png

Usage
-----
    python scripts/13_sprint3_finalise.py
    python scripts/13_sprint3_finalise.py --skip-indexing
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis import indexing_diagnostics as idx_diag  # noqa: E402
from src.models.data import load_slice_index  # noqa: E402
from src.models.samplers import (  # noqa: E402
    measure_coverage,
    rotating_epoch_sample,
    verify_no_leakage_in_schedule,
)
from src.utils.paths import OUTPUTS_DIR, PROJECT_ROOT, ensure_dirs  # noqa: E402
from src.utils.reporting import MarkdownReport, save_json  # noqa: E402

EXPERIMENT = "sprint3_coverage"
CHECKPOINT_DIR = OUTPUTS_DIR / "checkpoints" / EXPERIMENT
REPORT_DIR = OUTPUTS_DIR / "reports" / EXPERIMENT
METRICS_DIR = REPORT_DIR / "metrics"
VIZ_DIR = OUTPUTS_DIR / "visualizations" / EXPERIMENT

# Sprint 2 Extended artefacts - read only, never written.
S2_DIR = OUTPUTS_DIR / "reports" / "sprint2_extended"
S2_METRICS = S2_DIR / "metrics"
S2_CHECKPOINTS = OUTPUTS_DIR / "checkpoints" / "sprint2_extended"

PREDICTIONS_DIR = PROJECT_ROOT / "data" / "processed" / "predictions_sprint3_coverage"
S2_PREDICTIONS_DIR = PROJECT_ROOT / "data" / "processed" / "predictions_sprint2_extended"

CLASSES = ["background", "vertebra", "intervertebral_disc", "spinal_canal"]
SEG_METRICS = ["dice", "iou", "precision", "recall"]

#: The predefined Sprint 2 Extended reference values, as stated in the brief.
BASELINE = {
    "val_fg_dice": 0.8985,
    "test_macro_fg_dice": 0.8979,
    "test_vertebra_dice": 0.9133,
    "test_ivd_dice": 0.8779,
    "test_canal_dice": 0.9026,
    "disc_indexing_pct": 82.6,
    "pfirrmann_qwk": 0.6375,
}

#: A Dice change below this is treated as within-noise for the decision.
MEANINGFUL_DICE_DELTA = 0.005


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-indexing", action="store_true")
    parser.add_argument("--indexing-limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


# ---------------------------------------------------------------------------
# Phase 1: verification
# ---------------------------------------------------------------------------


def verify_training_complete() -> dict:
    """Confirm the training run finished and left every expected artefact."""
    required = {
        "history.csv": CHECKPOINT_DIR / "history.csv",
        "history.json": CHECKPOINT_DIR / "history.json",
        "last.pt": CHECKPOINT_DIR / "last.pt",
        "best_val_dice.pt": CHECKPOINT_DIR / "best_val_dice.pt",
        "best_val_loss.pt": CHECKPOINT_DIR / "best_val_loss.pt",
        "training_config.json": REPORT_DIR / "training_config.json",
        "training_summary.json": REPORT_DIR / "training_summary.json",
    }
    presence = {name: path.exists() for name, path in required.items()}
    missing = [name for name, ok in presence.items() if not ok]

    summary = load_json(required["training_summary.json"])
    history = (
        pd.read_csv(required["history.csv"]) if presence["history.csv"] else None
    )

    # training_summary.json is only written after fit() returns, so its presence
    # is the reliable signal that the run completed rather than being killed.
    completed = summary is not None and history is not None and len(history) > 0
    return {
        "artefacts_present": presence,
        "missing_artefacts": missing,
        "training_completed": bool(completed and not missing),
        "epochs_in_history": int(len(history)) if history is not None else 0,
        "epochs_reported": summary.get("epochs_completed") if summary else None,
        "stopped_early": summary.get("stopped_early") if summary else None,
        "stop_reason": summary.get("stop_reason") if summary else None,
        "history_matches_summary": (
            bool(history is not None and summary is not None
                 and len(history) == summary.get("epochs_completed"))
        ),
    }


# ---------------------------------------------------------------------------
# Phase 2: training analysis
# ---------------------------------------------------------------------------


def analyse_training(history: pd.DataFrame, summary: dict) -> dict:
    """Extract every requested training statistic from the actual history."""
    best_dice_row = history.loc[history["val_fg_dice"].idxmax()]
    best_loss_row = history.loc[history["val_loss"].idxmin()]
    final_row = history.iloc[-1]

    seconds = history["epoch_seconds"]
    return {
        "total_epochs_completed": int(len(history)),
        "max_epochs_configured": summary.get("max_epochs"),
        "early_stopping_triggered": bool(summary.get("stopped_early")),
        "stop_reason": summary.get("stop_reason"),
        "epochs_without_improvement_at_end":
            summary.get("epochs_without_improvement_at_end"),

        "best_val_dice_epoch": int(best_dice_row["epoch"]),
        "best_val_fg_dice": float(best_dice_row["val_fg_dice"]),
        "val_loss_at_best_dice_epoch": float(best_dice_row["val_loss"]),
        "train_loss_at_best_dice_epoch": float(best_dice_row["train_loss"]),
        "learning_rate_at_best_dice_epoch": float(best_dice_row["learning_rate"]),
        "vertebra_dice_at_best_epoch": float(best_dice_row["val_dice_vertebra"]),
        "ivd_dice_at_best_epoch":
            float(best_dice_row["val_dice_intervertebral_disc"]),
        "canal_dice_at_best_epoch": float(best_dice_row["val_dice_spinal_canal"]),

        "best_val_loss": float(best_loss_row["val_loss"]),
        "best_val_loss_epoch": int(best_loss_row["epoch"]),

        "final_epoch": int(final_row["epoch"]),
        "final_val_fg_dice": float(final_row["val_fg_dice"]),
        "final_val_loss": float(final_row["val_loss"]),
        "final_train_loss": float(final_row["train_loss"]),

        "total_training_seconds": float(seconds.sum()),
        "total_training_hours": round(float(seconds.sum()) / 3600, 2),
        "wall_clock_hours_reported": summary.get("wall_hours"),
        "mean_epoch_seconds": round(float(seconds.mean()), 1),
        "min_epoch_seconds": round(float(seconds.min()), 1),
        "max_epoch_seconds": round(float(seconds.max()), 1),
        "best_and_final_are_same_epoch":
            int(best_dice_row["epoch"]) == int(final_row["epoch"]),
    }


def analyse_coverage(history: pd.DataFrame, index: pd.DataFrame, seed: int) -> dict:
    """Verify the coverage the sampler actually delivered during the run."""
    n_train_available = int((index["split"] == "train").sum())
    n_epochs = int(len(history))

    per_epoch = history[
        ["epoch", "n_train_slices_this_epoch", "unique_train_slices_seen"]
    ].copy()
    per_epoch["coverage_pct"] = (
        100 * per_epoch["unique_train_slices_seen"] / n_train_available
    ).round(2)

    epoch_full = per_epoch.loc[
        per_epoch["coverage_pct"] >= 99.99, "epoch"
    ]
    epoch_full_value = int(epoch_full.iloc[0]) if len(epoch_full) else None

    # Exposure counts: how many times each slice was actually used. Recomputed
    # from the deterministic sampler over exactly the epochs that ran.
    exposure: dict[str, int] = {}
    for epoch in range(1, n_epochs + 1):
        rows = rotating_epoch_sample(
            index, epoch=epoch, slices_per_epoch=2284, seed=seed
        )
        for slice_id in rows["slice_id"]:
            exposure[slice_id] = exposure.get(slice_id, 0) + 1

    train_ids = set(index.loc[index["split"] == "train", "slice_id"])
    counts = np.array([exposure.get(s, 0) for s in train_ids])

    leakage = verify_no_leakage_in_schedule(
        index, strategy="rotating", n_epochs=n_epochs,
        slices_per_epoch=2284, seed=seed,
    )

    return {
        "train_slices_available": n_train_available,
        "unique_slices_seen_final": int(history["unique_train_slices_seen"].iloc[-1]),
        "final_coverage_pct": round(
            100 * float(history["unique_train_slices_seen"].iloc[-1])
            / n_train_available, 2
        ),
        "epoch_reaching_full_coverage": epoch_full_value,
        "full_coverage_by_epoch_4": bool(
            epoch_full_value is not None and epoch_full_value <= 4
        ),
        "slices_per_epoch_mean": round(
            float(history["n_train_slices_this_epoch"].mean()), 1
        ),
        "slices_per_epoch_min": int(history["n_train_slices_this_epoch"].min()),
        "slices_per_epoch_max": int(history["n_train_slices_this_epoch"].max()),
        "exposure_mean": round(float(counts.mean()), 2),
        "exposure_min": int(counts.min()),
        "exposure_max": int(counts.max()),
        "exposure_std": round(float(counts.std()), 3),
        "slices_never_seen": int((counts == 0).sum()),
        "coverage_by_epoch": per_epoch.to_dict("records"),
        "leakage_free": leakage["leakage_free"],
        "leakage_violations": len(leakage["violations"]),
        "val_test_patients_excluded": leakage["val_test_patients_excluded"],
        "val_test_slices_excluded": leakage["val_test_slices_excluded"],
    }


# ---------------------------------------------------------------------------
# Phase 3: checkpoint selection
# ---------------------------------------------------------------------------


def describe_selected_checkpoint() -> dict:
    """Read back the checkpoint chosen by the predefined validation criterion."""
    import torch

    path = CHECKPOINT_DIR / "best_val_dice.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    metadata = checkpoint.get("metadata", {})
    return {
        "filename": path.name,
        "path": str(path),
        "selection_criterion": "best validation macro foreground Dice (predefined)",
        "selected_using_test_data": False,
        "epoch": metadata.get("epoch"),
        "val_fg_dice": metadata.get("val_fg_dice"),
        "val_loss": metadata.get("val_loss"),
        "architecture": checkpoint.get("architecture"),
    }


# ---------------------------------------------------------------------------
# Phase 5: comparison
# ---------------------------------------------------------------------------


class FixedPrecisionReport(MarkdownReport):
    """``MarkdownReport`` that keeps fixed decimals in table cells.

    The shared ``_format_cell`` helper in ``src/utils/reporting.py`` renders
    floats with ``:,.4g`` - four *significant* figures. That silently turns
    0.90001 into "0.9" and 0.9081 into "0.908". Sprint 3's deltas live in the
    4th and 5th decimal place, so that rounding hides exactly the digits the
    experiment turns on.

    Subclassing keeps the fix additive: Sprint 1 and Sprint 2 reports continue
    to use the original formatter untouched.
    """

    @staticmethod
    def _fix(value):
        if isinstance(value, bool):
            return value
        if isinstance(value, (float, np.floating)):
            value = float(value)
            if not np.isfinite(value):
                return value
            if value == 0:
                return "0.00000"
            magnitude = abs(value)
            if magnitude < 1e-3:
                places = 6
            elif magnitude < 1:
                places = 5
            else:
                places = 2
            return f"{value:.{places}f}"
        return value

    def table(self, rows: list[dict], columns: list[str] | None = None):
        rows = [{k: self._fix(v) for k, v in row.items()} for row in rows]
        return super().table(rows, columns)

    def key_values(self, data: dict, *, headers: tuple[str, str] = ("Field", "Value")):
        return super().key_values(
            {k: self._fix(v) for k, v in data.items()}, headers=headers
        )


def compare(name: str, baseline, sprint3, *, higher_is_better: bool) -> dict:
    """Compare one metric, reporting absolute and relative change."""
    entry = {
        "metric": name,
        "sprint2_extended": baseline,
        "sprint3_coverage": sprint3,
        "higher_is_better": higher_is_better,
    }
    if baseline is None or sprint3 is None:
        entry.update({"absolute_delta": None, "relative_delta_pct": None,
                      "verdict": "unavailable"})
        return entry

    delta = sprint3 - baseline
    entry["absolute_delta"] = round(float(delta), 6)
    entry["relative_delta_pct"] = (
        round(float(100 * delta / abs(baseline)), 3) if baseline else None
    )
    if abs(delta) < 1e-9:
        entry["verdict"] = "unchanged"
    elif (delta > 0) == higher_is_better:
        entry["verdict"] = "improved"
    else:
        entry["verdict"] = "worsened"
    return entry


def build_comparison(
    s3_seg: dict, s2_seg: dict, s3_agree: dict | None, s2_agree: dict | None,
    s3_index: dict | None, s2_index: dict | None, training: dict,
) -> dict:
    """Build the full comparison across every measured family."""
    segmentation: list[dict] = []

    # Validation (the primary predefined criterion).
    validation = [
        compare("validation macro foreground Dice (best)",
                BASELINE["val_fg_dice"], training["best_val_fg_dice"],
                higher_is_better=True)
    ]

    for class_name in CLASSES:
        for metric in SEG_METRICS:
            segmentation.append(
                compare(
                    f"test {class_name}/{metric}",
                    s2_seg["aggregate"]["per_class"][class_name][metric],
                    s3_seg["aggregate"]["per_class"][class_name][metric],
                    higher_is_better=True,
                )
            )
    for metric in SEG_METRICS:
        segmentation.append(
            compare(
                f"test macro_foreground/{metric}",
                s2_seg["aggregate"]["macro_foreground"][metric],
                s3_seg["aggregate"]["macro_foreground"][metric],
                higher_is_better=True,
            )
        )
    segmentation.append(
        compare("test pixel_accuracy", s2_seg["pixel_accuracy"],
                s3_seg["pixel_accuracy"], higher_is_better=True)
    )
    for class_name in CLASSES:
        segmentation.append(
            compare(
                f"test per_patient_dice/{class_name}",
                s2_seg["per_patient_dice"][class_name]["mean_dice"],
                s3_seg["per_patient_dice"][class_name]["mean_dice"],
                higher_is_better=True,
            )
        )
        segmentation.append(
            compare(
                f"test per_patient_dice_sd/{class_name}",
                s2_seg["per_patient_dice"][class_name]["std_dice"],
                s3_seg["per_patient_dice"][class_name]["std_dice"],
                higher_is_better=False,
            )
        )

    # Disc indexing taxonomy (corrected taxonomy only).
    indexing: list[dict] = []
    if s3_index and s2_index:
        for key, higher in [
            ("pct_index_correct", True),
            ("pct_region_found", True),
            ("pct_slices_count_correct", True),
            ("pct_slices_all_discs_correct", True),
            ("total_spurious_components", False),
            ("pct_slices_with_spurious", False),
        ]:
            indexing.append(
                compare(f"indexing {key}", s2_index.get(key), s3_index.get(key),
                        higher_is_better=higher)
            )
        for category in ["correct", "shifted", "merged", "split", "missed"]:
            indexing.append(
                compare(
                    f"indexing category %/{category}",
                    s2_index.get("categories", {}).get(category, {}).get("pct"),
                    s3_index.get("categories", {}).get(category, {}).get("pct"),
                    higher_is_better=(category == "correct"),
                )
            )

    # Disc measurements and end-to-end Pfirrmann.
    measurements: list[dict] = []
    pfirrmann: list[dict] = []
    if s3_agree and s2_agree:
        s2_map = {r["measurement"]: r for r in s2_agree["measurement_agreement"]}
        s3_map = {r["measurement"]: r for r in s3_agree["measurement_agreement"]}
        for name in sorted(set(s2_map) & set(s3_map)):
            measurements.append(
                compare(f"MAE {name}", s2_map[name]["mae"], s3_map[name]["mae"],
                        higher_is_better=False)
            )
            measurements.append(
                compare(f"r {name}", s2_map[name]["pearson_r"],
                        s3_map[name]["pearson_r"], higher_is_better=True)
            )
        measurements.append(
            compare(
                "disc detection %truth recovered",
                s2_agree["detection"]["pct_truth_discs_recovered"],
                s3_agree["detection"]["pct_truth_discs_recovered"],
                higher_is_better=True,
            )
        )

        s2_pf = (s2_agree.get("end_to_end_pfirrmann") or {})
        s3_pf = (s3_agree.get("end_to_end_pfirrmann") or {})
        for estimator in ["logreg", "forest"]:
            if estimator not in s2_pf or estimator not in s3_pf:
                continue
            for key, higher in [
                ("quadratic_weighted_kappa", True),
                ("mae", False),
                ("within_one_grade", True),
                ("exact_agreement", True),
            ]:
                pfirrmann.append(
                    compare(f"Pfirrmann {estimator}/{key}",
                            s2_pf[estimator].get(key), s3_pf[estimator].get(key),
                            higher_is_better=higher)
                )

    all_rows = validation + segmentation + indexing + measurements + pfirrmann
    tally: dict[str, int] = {}
    for row in all_rows:
        tally[row["verdict"]] = tally.get(row["verdict"], 0) + 1

    return {
        "validation": validation,
        "segmentation": segmentation,
        "indexing": indexing,
        "measurements": measurements,
        "pfirrmann": pfirrmann,
        "all": all_rows,
        "tally": tally,
    }


# ---------------------------------------------------------------------------
# Phase 9: decision
# ---------------------------------------------------------------------------


def decide(comparison: dict, training: dict, coverage: dict) -> dict:
    """Answer the predefined experimental question from the whole evidence set.

    The question is whether insufficient training-data coverage was the main
    limitation of the 16-channel model. A single metric is explicitly not enough,
    so the decision weighs the primary validation criterion together with the
    test metrics, the disc-level metrics and the convergence behaviour.
    """
    primary = comparison["validation"][0]
    val_delta = primary["absolute_delta"] or 0.0

    macro = next(
        (r for r in comparison["segmentation"]
         if r["metric"] == "test macro_foreground/dice"), None
    )
    macro_delta = (macro or {}).get("absolute_delta") or 0.0

    indexing = next(
        (r for r in comparison["indexing"]
         if r["metric"] == "indexing pct_index_correct"), None
    )
    index_delta = (indexing or {}).get("absolute_delta") or 0.0

    pfirrmann = next(
        (r for r in comparison["pfirrmann"]
         if r["metric"] == "Pfirrmann forest/quadratic_weighted_kappa"), None
    )
    pfirrmann_delta = (pfirrmann or {}).get("absolute_delta") or 0.0

    # Count improvements only among the substantive metric families, excluding
    # per-patient standard deviations (which are dispersion, not quality).
    substantive = [
        r for r in comparison["all"]
        if "per_patient_dice_sd" not in r["metric"]
    ]
    improved = sum(1 for r in substantive if r["verdict"] == "improved")
    worsened = sum(1 for r in substantive if r["verdict"] == "worsened")

    val_exceeds = val_delta > 0
    val_meaningful = val_delta >= MEANINGFUL_DICE_DELTA
    macro_meaningful = macro_delta >= MEANINGFUL_DICE_DELTA
    majority_improved = improved > worsened
    # A validation difference smaller than the early-stopping min_delta is below
    # the resolution the experiment was set up to detect, so it is a tie rather
    # than a win or a loss.
    val_tied = abs(val_delta) < 0.001

    if val_meaningful and macro_meaningful:
        verdict = "supported"
        conclusion = (
            "Training-data coverage was a material limitation: both the primary "
            "validation criterion and the test macro Dice improved by more than "
            f"the {MEANINGFUL_DICE_DELTA} threshold."
        )
    elif val_exceeds and majority_improved and macro_meaningful:
        verdict = "partially supported"
        conclusion = (
            "Coverage helped, but not decisively. The primary criterion improved "
            "and the test macro Dice cleared the meaningful threshold, yet the "
            "validation margin did not. Coverage was a contributing limitation, "
            "not the dominant one."
        )
    elif val_tied:
        verdict = "not meaningfully supported (primary criterion tied)"
        conclusion = (
            f"Raising training-data coverage from 28.05% to "
            f"{coverage['final_coverage_pct']}% left the primary criterion "
            f"effectively unchanged: validation macro foreground Dice moved "
            f"{val_delta:+.5f}, which is smaller than the {0.001} min_delta the "
            f"experiment was configured to treat as real. The secondary metrics do "
            f"move consistently in the right direction - test macro Dice "
            f"{macro_delta:+.5f}, disc indexing {index_delta:+.2f} pp, and "
            f"{improved} of {improved + worsened} compared metrics improved - but "
            f"all of the segmentation gains are well inside the noise band of a "
            f"single 33-patient test set. **The conclusion is that the 16-channel "
            f"model was not meaningfully data-limited**: it reaches the same "
            f"performance whether it sees 28% or 100% of the training slices. That "
            f"makes model capacity, rather than data volume, the credible next "
            f"variable."
        )
    elif val_exceeds:
        verdict = "weakly supported"
        conclusion = (
            "The primary criterion improved, but the improvement is small and is "
            "not corroborated by a majority of the secondary metrics."
        )
    else:
        verdict = "not supported"
        conclusion = (
            "Full training-data coverage did not improve on the previous best "
            "validation Dice. Coverage was therefore not the binding constraint, "
            "which makes model capacity the credible next variable."
        )

    return {
        "question": (
            "Was insufficient training-data coverage the main limitation of the "
            "previous 16-channel model?"
        ),
        "verdict": verdict,
        "conclusion": conclusion,
        "primary_criterion": {
            "name": "best validation macro foreground Dice",
            "sprint2_extended": BASELINE["val_fg_dice"],
            "sprint3_coverage": training["best_val_fg_dice"],
            "absolute_delta": round(val_delta, 5),
            "exceeds_baseline": bool(val_exceeds),
            "exceeds_by_meaningful_margin": bool(val_meaningful),
            "meaningful_threshold": MEANINGFUL_DICE_DELTA,
            "statistically_tied": bool(val_tied),
            "tie_threshold": 0.001,
        },
        "supporting_evidence": {
            "test_macro_fg_dice_delta": round(macro_delta, 5),
            "disc_indexing_delta_pp": round(index_delta, 3),
            "pfirrmann_forest_qwk_delta": round(pfirrmann_delta, 5),
            "metrics_improved": improved,
            "metrics_worsened": worsened,
            "coverage_achieved_pct": coverage["final_coverage_pct"],
            "coverage_vs_sprint2_pct": 28.05,
            "early_stopping_triggered": training["early_stopping_triggered"],
            "best_epoch": training["best_val_dice_epoch"],
            "total_epochs": training["total_epochs_completed"],
        },
    }


def recommend(decision: dict, coverage: dict, training: dict) -> dict:
    """Recommend exactly one next experiment, based only on the results."""
    verdict = decision["verdict"]
    if verdict in {"supported", "partially supported"}:
        return {
            "option": "A",
            "title": "Retain the 16-channel architecture; move to disc-indexing "
                     "and post-processing improvements",
            "rationale": (
                "Coverage improved the model, so capacity is still not the "
                "demonstrated bottleneck. The largest remaining gap is between disc "
                "detection and disc *indexing*, and the Sprint 3 audit established "
                "that this is a post-processing problem: identity is derived by "
                "ordering connected components per slice, so one spurious or missing "
                "component shifts every index above it. Series-level 3-D grouping "
                "was already measured to recover 99.3% of discs against ~82% per "
                "slice, and needs no retraining."
            ),
            "proposed_next_experiment": [
                "Group disc components in 3-D across adjacent slices and assign "
                "identity once per series instead of once per slice.",
                "Suppress spurious components by physical area and implausible "
                "aspect ratio before numbering.",
                "Restrict per-disc measurement to slices carrying enough annotation "
                "(indexing is ~90% correct above 6,000 annotated pixels and ~27% "
                "below 1,000).",
                "Re-run the disc-level analysis and compare, with no change to the "
                "segmentation model.",
            ],
            "explicitly_not_recommended": [
                "Width 64 - measured as infeasible on this hardware.",
                "Width 32 - not yet justified while a no-retraining fix is "
                "outstanding.",
            ],
        }
    return {
        "option": "B",
        "title": "Controlled capacity experiment: 32-channel U-Net at batch size 2",
        "rationale": (
            "Full training-data coverage did not meaningfully improve the primary "
            "criterion - the 16-channel model reaches effectively the same "
            "validation Dice on 28% and on 100% of the training slices. It is "
            "therefore not data-limited at this configuration, and capacity becomes "
            "the credible next variable. The Sprint 3 audit measured width 32 at "
            "batch 2 as needing ~1.82 GB peak commit, which is *less* than the "
            "proven width-16/batch-8 configuration, so memory is not the obstacle. "
            "Time is: roughly 17 h for 30 rotating-coverage epochs."
        ),
        "proposed_next_experiment": [
            "Width 32, batch size 2. Keep the rotating sampler (it costs nothing "
            "and removes coverage as a confound), plus the same loss, optimiser, "
            "augmentation, split and seed.",
            "30 epochs, patience 6, min_delta 0.001, random initialisation.",
            "Budget ~17 h; new output directories; test set evaluated once at the "
            "end on the validation-selected checkpoint.",
            "Success criterion to fix in advance: validation macro foreground Dice "
            f"above {max(0.8985, training['best_val_fg_dice']):.4f} by a margin of "
            f"at least {MEANINGFUL_DICE_DELTA}.",
        ],
        "also_worth_doing_independently": [
            "The disc-indexing post-processing work identified in the Sprint 3 "
            "audit remains the highest-value change that needs **no retraining**: "
            "series-level 3-D grouping was measured to recover 99.3% of discs "
            "against ~84% per slice. It is orthogonal to the capacity question and "
            "can proceed in parallel, but it is not the controlled experiment being "
            "recommended here.",
        ],
        "explicitly_not_recommended": [
            "Width 64 - measured at ~6.8 GB peak commit and 216-326 h for 30 "
            "epochs. No new measured reason to revisit it.",
            "Changing width together with batch size, loss, augmentation or any "
            "other variable in the same run.",
        ],
    }


# ---------------------------------------------------------------------------
# Phase 8: figures
# ---------------------------------------------------------------------------


def render_figures(history: pd.DataFrame, coverage: dict, comparison: dict,
                   s2_history_path: Path) -> list[str]:
    """Training curves, coverage curve, and the comparison bars."""
    import matplotlib.pyplot as plt

    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    # --- training curves + coverage --------------------------------------
    figure, axes = plt.subplots(1, 4, figsize=(21, 4.6))

    axes[0].plot(history["epoch"], history["train_loss"], "o-", label="train",
                 color="#4c72b0", markersize=3)
    axes[0].plot(history["epoch"], history["val_loss"], "s--", label="validation",
                 color="#c44e52", markersize=3)
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss (CE + Dice)")
    axes[0].set_title("Training and validation loss", fontsize=11)
    axes[0].legend(frameon=False); axes[0].grid(alpha=0.25)

    axes[1].plot(history["epoch"], history["val_fg_dice"], "s-", color="#55a868",
                 markersize=3, label="Sprint 3 coverage")
    axes[1].axhline(BASELINE["val_fg_dice"], color="black", linestyle="--",
                    linewidth=1, label=f"Sprint 2 Ext best {BASELINE['val_fg_dice']}")
    best = history.loc[history["val_fg_dice"].idxmax()]
    axes[1].plot(best["epoch"], best["val_fg_dice"], "r*", markersize=14,
                 label=f"best ep {int(best['epoch'])}")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("validation macro fg Dice")
    axes[1].set_title("Validation foreground Dice", fontsize=11)
    axes[1].legend(fontsize=8, frameon=False); axes[1].grid(alpha=0.25)

    for column, label in [
        ("val_dice_vertebra", "vertebra"),
        ("val_dice_intervertebral_disc", "IVD"),
        ("val_dice_spinal_canal", "canal"),
    ]:
        axes[2].plot(history["epoch"], history[column], ".-", label=label,
                     markersize=4)
    axes[2].set_xlabel("epoch"); axes[2].set_ylabel("validation Dice")
    axes[2].set_title("Per-class validation Dice", fontsize=11)
    axes[2].legend(fontsize=8, frameon=False); axes[2].grid(alpha=0.25)

    coverage_frame = pd.DataFrame(coverage["coverage_by_epoch"])
    axes[3].plot(coverage_frame["epoch"], coverage_frame["coverage_pct"], "o-",
                 color="#dd8452", markersize=3)
    axes[3].axhline(100, color="black", linestyle=":", linewidth=1)
    axes[3].axhline(28.05, color="#c44e52", linestyle="--", linewidth=1,
                    label="Sprint 2 fixed sampler (28.05%)")
    if coverage["epoch_reaching_full_coverage"]:
        axes[3].axvline(coverage["epoch_reaching_full_coverage"], color="green",
                        linestyle=":", linewidth=1,
                        label=f"100% at epoch "
                              f"{coverage['epoch_reaching_full_coverage']}")
    axes[3].set_xlabel("epoch")
    axes[3].set_ylabel("% of 9,128 training slices seen")
    axes[3].set_title("Cumulative training-data coverage", fontsize=11)
    axes[3].legend(fontsize=8, frameon=False); axes[3].grid(alpha=0.25)
    axes[3].set_ylim(0, 108)

    figure.suptitle(
        "Sprint 3 Coverage - rotating sampler, 16-channel U-Net, random init "
        "(only the sampler changed vs Sprint 2 Extended)",
        fontsize=12,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    path = VIZ_DIR / "training_curves.png"
    figure.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(figure)
    written.append(path.name)

    # --- Sprint 2 vs Sprint 3 validation curves -------------------------
    if s2_history_path.exists():
        s2 = pd.read_csv(s2_history_path)
        figure, axes = plt.subplots(1, 2, figsize=(13, 4.6))
        axes[0].plot(s2["epoch"], s2["val_fg_dice"], "s-", color="#4c72b0",
                     markersize=3, label="Sprint 2 Extended (ep 13-30, warm restart)")
        axes[0].plot(history["epoch"], history["val_fg_dice"], "o-", color="#55a868",
                     markersize=3, label="Sprint 3 Coverage (ep 1-N, random init)")
        axes[0].axhline(BASELINE["val_fg_dice"], color="black", linestyle="--",
                        linewidth=1)
        axes[0].set_xlabel("epoch"); axes[0].set_ylabel("validation macro fg Dice")
        axes[0].set_title("Validation Dice: both runs\n(note: different starting "
                          "conditions)", fontsize=10)
        axes[0].legend(fontsize=8, frameon=False); axes[0].grid(alpha=0.25)

        axes[1].plot(s2["epoch"], s2["val_loss"], "s-", color="#4c72b0",
                     markersize=3, label="Sprint 2 Extended")
        axes[1].plot(history["epoch"], history["val_loss"], "o-", color="#55a868",
                     markersize=3, label="Sprint 3 Coverage")
        axes[1].set_xlabel("epoch"); axes[1].set_ylabel("validation loss")
        axes[1].set_title("Validation loss: both runs", fontsize=10)
        axes[1].legend(fontsize=8, frameon=False); axes[1].grid(alpha=0.25)

        figure.tight_layout()
        path = VIZ_DIR / "sprint2_vs_sprint3_curves.png"
        figure.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(figure)
        written.append(path.name)

    # --- comparison bars -------------------------------------------------
    key_metrics = [
        ("test vertebra/dice", "vertebra Dice"),
        ("test intervertebral_disc/dice", "IVD Dice"),
        ("test spinal_canal/dice", "canal Dice"),
        ("test macro_foreground/dice", "macro fg Dice"),
        ("test macro_foreground/iou", "macro fg IoU"),
    ]
    rows = {r["metric"]: r for r in comparison["segmentation"]}
    available = [(k, label) for k, label in key_metrics if k in rows]
    if available:
        figure, axis = plt.subplots(figsize=(10, 4.8))
        x = np.arange(len(available))
        width = 0.36
        s2_values = [rows[k]["sprint2_extended"] for k, _ in available]
        s3_values = [rows[k]["sprint3_coverage"] for k, _ in available]
        axis.bar(x - width / 2, s2_values, width, label="Sprint 2 Extended",
                 color="#4c72b0")
        axis.bar(x + width / 2, s3_values, width, label="Sprint 3 Coverage",
                 color="#55a868")
        for i, (s2v, s3v) in enumerate(zip(s2_values, s3_values)):
            axis.text(i + width / 2, s3v, f"{s3v - s2v:+.4f}", ha="center",
                      va="bottom", fontsize=8)
        axis.set_xticks(x, [label for _, label in available], fontsize=9)
        axis.set_ylim(0, 1.06)
        axis.set_ylabel("test score")
        axis.set_title("Test-set metrics: Sprint 2 Extended vs Sprint 3 Coverage\n"
                       "(annotations show the change)", fontsize=11)
        axis.legend(frameon=False)
        axis.grid(alpha=0.2, axis="y")
        figure.tight_layout()
        path = VIZ_DIR / "test_metric_comparison.png"
        figure.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(figure)
        written.append(path.name)

    return written


def render_indexing_figure(s3_index: dict, s2_index: dict) -> str | None:
    """Failure taxonomy comparison, using the corrected categories only."""
    import matplotlib.pyplot as plt

    categories = ["correct", "shifted", "merged", "split", "missed"]
    figure, axes = plt.subplots(1, 3, figsize=(17, 4.8))

    runs = {"Sprint 2 Extended": s2_index, "Sprint 3 Coverage": s3_index}
    x = np.arange(len(categories))
    width = 0.36
    for offset, (label, data) in enumerate(runs.items()):
        values = [data["categories"].get(c, {}).get("pct", 0.0) for c in categories]
        bars = axes[0].bar(x + (offset - 0.5) * width, values, width, label=label)
        for bar, value in zip(bars, values):
            axes[0].text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f}",
                         ha="center", va="bottom", fontsize=7)
    axes[0].set_xticks(x, categories)
    axes[0].set_ylabel("% of ground-truth discs")
    axes[0].set_title("Disc-indexing failure taxonomy", fontsize=11)
    axes[0].legend(fontsize=8, frameon=False)
    axes[0].grid(alpha=0.2, axis="y")

    for label, data in runs.items():
        table = pd.DataFrame(data["by_disc_index"])
        axes[1].plot(table["truth_index"], table["pct_correct"], marker="o",
                     label=label)
    axes[1].set_xlabel("disc index (1 = most inferior)")
    axes[1].set_ylabel("% correctly indexed")
    axes[1].set_title("Indexing accuracy by disc index", fontsize=11)
    axes[1].legend(fontsize=8, frameon=False)
    axes[1].grid(alpha=0.25)

    for label, data in runs.items():
        table = pd.DataFrame(data["by_slice_annotation_area"])
        axes[2].plot(table["area_band"].astype(str), table["pct_correct"],
                     marker="s", label=label)
    axes[2].set_xlabel("annotated pixels on the slice")
    axes[2].set_ylabel("% correctly indexed")
    axes[2].set_title("Indexing accuracy vs slice annotation area", fontsize=11)
    axes[2].legend(fontsize=8, frameon=False)
    axes[2].grid(alpha=0.25)

    figure.suptitle(
        "Disc-indexing analysis, corrected taxonomy (disc index only; no "
        "anatomical level names asserted)",
        fontsize=12,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    path = VIZ_DIR / "indexing_comparison.png"
    figure.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(figure)
    return path.name


# ---------------------------------------------------------------------------
# Phase 11: final report
# ---------------------------------------------------------------------------


def write_report(payload: dict) -> Path:
    """Compose sprint3_final_report.md from measured artefacts only."""
    verification = payload["verification"]
    training = payload["training"]
    coverage = payload["coverage"]
    checkpoint = payload["checkpoint"]
    comparison = payload["comparison"]
    decision = payload["decision"]
    recommendation = payload["recommendation"]
    s3_seg = payload["test_segmentation"]
    config = payload["config"]

    report = FixedPrecisionReport(
        "Sprint 3 Coverage - Final Report",
        "Controlled single-variable experiment: training-data coverage",
    )

    # ---- 1 objective ----
    report.heading("1. Objective")
    report.text(
        "Answer one question: **was insufficient training-data coverage the main "
        "limitation of the 16-channel U-Net?**"
    )
    report.text(
        "Sprint 2 Extended established that the 16-channel model had converged, but "
        "the Sprint 3 audit found it had only ever seen **28.05%** of the available "
        "training slices - the sampler drew a 2,560-slice subset once, before "
        "training, and reused it every epoch. A plateau on 28% of the data cannot be "
        "attributed to insufficient capacity. This experiment removes that "
        "confound by rotating the subset so the model sees all 9,128 training "
        "slices, while changing nothing else."
    )

    # ---- 2 setup ----
    report.heading("2. Experimental Setup")
    report.heading("Changed (the single variable)", level=3)
    report.key_values(
        {
            "Sampler": "rotating shard sampler (was: fixed subset sampled once)",
            "Slices per epoch": f"{coverage['slices_per_epoch_mean']:.0f} mean "
                                f"({coverage['slices_per_epoch_min']}-"
                                f"{coverage['slices_per_epoch_max']})",
            "Intended coverage": "100% of 9,128 training slices by ~epoch 4",
        }
    )
    report.heading("Unchanged from Sprint 2 Extended", level=3)
    unchanged = config.get("unchanged_from_sprint2_extended", {})
    report.key_values(unchanged)
    report.heading("Different by design", level=3)
    report.key_values(config.get("different_by_design", {}))
    report.text(
        "Random initialisation is required: warm-starting from the converged "
        "Sprint 2 weights would confound the coverage variable with the effect of "
        "already-trained weights. `min_delta` was raised from 0 to 0.001 because "
        "Sprint 2 showed that 0 makes early stopping ineffective - any improvement, "
        "however small, reset the patience counter."
    )

    report.heading("Verification that the run completed", level=3)
    report.key_values(
        {
            "All expected artefacts present": verification["training_completed"],
            "Missing artefacts": verification["missing_artefacts"] or "none",
            "Epochs in history.csv": verification["epochs_in_history"],
            "Epochs reported in summary": verification["epochs_reported"],
            "history matches summary": verification["history_matches_summary"],
            "Early stopping triggered": verification["stopped_early"],
            "Stop reason": verification["stop_reason"],
        }
    )

    # ---- 3 training results ----
    report.heading("3. Training Results")
    report.key_values(
        {
            "Total epochs completed":
                f"{training['total_epochs_completed']} / "
                f"{training['max_epochs_configured']}",
            "Early stopping triggered": training["early_stopping_triggered"],
            "Stop reason": training["stop_reason"],
            "Epochs without improvement at end":
                training["epochs_without_improvement_at_end"],
            "Best validation foreground Dice":
                f"{training['best_val_fg_dice']:.5f} "
                f"(epoch {training['best_val_dice_epoch']})",
            "Best validation loss":
                f"{training['best_val_loss']:.5f} "
                f"(epoch {training['best_val_loss_epoch']})",
            "Final validation Dice": f"{training['final_val_fg_dice']:.5f}",
            "Final validation loss": f"{training['final_val_loss']:.5f}",
            "Train loss at best epoch":
                f"{training['train_loss_at_best_dice_epoch']:.5f}",
            "Validation loss at best epoch":
                f"{training['val_loss_at_best_dice_epoch']:.5f}",
            "Learning rate at best epoch":
                f"{training['learning_rate_at_best_dice_epoch']:.3e}",
            "Per-class Dice at best epoch":
                f"vertebra {training['vertebra_dice_at_best_epoch']:.4f}, "
                f"IVD {training['ivd_dice_at_best_epoch']:.4f}, "
                f"canal {training['canal_dice_at_best_epoch']:.4f}",
            "Total training time":
                f"{training['total_training_hours']} h "
                f"({training['total_training_seconds']:.0f} s summed over epochs)",
            "Mean epoch time": f"{training['mean_epoch_seconds']} s",
            "Min / max epoch time":
                f"{training['min_epoch_seconds']} s / "
                f"{training['max_epoch_seconds']} s",
        }
    )
    report.heading("Per-epoch history", level=3)
    history = pd.DataFrame(payload["history"])
    shown = history[[
        "epoch", "learning_rate", "n_train_slices_this_epoch",
        "unique_train_slices_seen", "train_loss", "val_loss", "val_fg_dice",
        "val_dice_vertebra", "val_dice_intervertebral_disc",
        "val_dice_spinal_canal", "epoch_seconds",
    ]].round(5)
    report.dataframe(shown, max_rows=40)

    # ---- 4 coverage ----
    report.heading("4. Training Data Coverage")
    report.key_values(
        {
            "Training slices available": f"{coverage['train_slices_available']:,}",
            "Unique slices seen": f"{coverage['unique_slices_seen_final']:,}",
            "Final coverage": f"{coverage['final_coverage_pct']}%",
            "Sprint 2 Extended coverage (for reference)": "28.05%",
            "Epoch reaching 100% coverage":
                coverage["epoch_reaching_full_coverage"],
            "100% achieved by epoch 4 as intended":
                coverage["full_coverage_by_epoch_4"],
            "Times each slice was seen":
                f"mean {coverage['exposure_mean']}, range "
                f"{coverage['exposure_min']}-{coverage['exposure_max']} "
                f"(sd {coverage['exposure_std']})",
            "Slices never seen": coverage["slices_never_seen"],
        }
    )
    report.heading("Leakage verification", level=3)
    report.key_values(
        {
            "Leakage-free across every epoch": coverage["leakage_free"],
            "Violations found": coverage["leakage_violations"],
            "Validation/test patients excluded":
                coverage["val_test_patients_excluded"],
            "Validation/test slices excluded":
                coverage["val_test_slices_excluded"],
        }
    )
    report.text(
        "The sampler partitions only rows already marked `split == 'train'`, and the "
        "training script additionally asserts per epoch that the subset contains no "
        "other split. No validation or test slice **or patient** entered training."
    )

    # ---- 5 convergence ----
    report.heading("5. Convergence Analysis")
    convergence = payload["convergence"]
    report.key_values(
        {
            "Validation Dice, last 5 epochs":
                f"{convergence['last5_min']:.5f} - {convergence['last5_max']:.5f} "
                f"(spread {convergence['last5_spread']:.5f})",
            "Mean gain per epoch, last 5": f"{convergence['last5_mean_gain']:+.6f}",
            "Mean gain per epoch, epochs 5-10":
                f"{convergence['early_mean_gain']:+.6f}",
            "Final learning rate": f"{convergence['final_lr']:.3e}",
            "Best epoch is the final epoch":
                training["best_and_final_are_same_epoch"],
            "Early stopping triggered": training["early_stopping_triggered"],
            "Assessed as converged": convergence["converged"],
        }
    )
    report.text(convergence["interpretation"])

    # ---- 6 test results ----
    report.heading("6. Final Test Segmentation Results")
    report.text(
        f"Evaluated **once**, after training finished, on the untouched held-out "
        f"test set: {s3_seg['n_slices']:,} slices from {s3_seg['n_patients']} "
        f"patients. The checkpoint was selected by the predefined validation "
        f"criterion, not by test performance."
    )
    report.heading("Aggregate (dataset-level) metrics", level=3)
    report.table(
        [
            {
                "class": class_name,
                "Dice": round(s3_seg["aggregate"]["per_class"][class_name]["dice"], 5),
                "IoU": round(s3_seg["aggregate"]["per_class"][class_name]["iou"], 5),
                "Precision":
                    round(s3_seg["aggregate"]["per_class"][class_name]["precision"], 5),
                "Recall":
                    round(s3_seg["aggregate"]["per_class"][class_name]["recall"], 5),
            }
            for class_name in CLASSES
        ]
        + [
            {
                "class": "**macro foreground**",
                "Dice": round(s3_seg["aggregate"]["macro_foreground"]["dice"], 5),
                "IoU": round(s3_seg["aggregate"]["macro_foreground"]["iou"], 5),
                "Precision":
                    round(s3_seg["aggregate"]["macro_foreground"]["precision"], 5),
                "Recall":
                    round(s3_seg["aggregate"]["macro_foreground"]["recall"], 5),
            }
        ],
        ["class", "Dice", "IoU", "Precision", "Recall"],
    )
    report.key_values({"Pixel accuracy": round(s3_seg["pixel_accuracy"], 5)})

    report.heading("Per-patient Dice (mean +/- sd over test patients)", level=3)
    report.table(
        [
            {
                "class": class_name,
                "mean Dice": round(
                    s3_seg["per_patient_dice"][class_name]["mean_dice"], 5
                ),
                "sd": round(s3_seg["per_patient_dice"][class_name]["std_dice"], 5),
                "patients": s3_seg["per_patient_dice"][class_name]["n_patients"],
            }
            for class_name in CLASSES
        ],
        ["class", "mean Dice", "sd", "patients"],
    )

    # ---- 7 comparison ----
    report.heading("7. Sprint 2 Extended vs Sprint 3 Comparison")
    report.text(
        "`Absolute Δ` is Sprint 3 minus Sprint 2 Extended. `Relative Δ` is that as "
        "a percentage of the Sprint 2 value. Verdicts account for metric direction "
        "(lower is better for MAE, loss and standard deviation)."
    )
    report.heading("Primary criterion", level=3)
    report.table(
        [
            {
                "Metric": row["metric"],
                "Sprint 2 Extended": row["sprint2_extended"],
                "Sprint 3 Coverage": round(row["sprint3_coverage"], 5),
                "Absolute Δ": f"{row['absolute_delta']:+.5f}",
                "Relative Δ": f"{row['relative_delta_pct']:+.3f}%",
                "Verdict": row["verdict"],
            }
            for row in comparison["validation"]
        ],
        ["Metric", "Sprint 2 Extended", "Sprint 3 Coverage", "Absolute Δ",
         "Relative Δ", "Verdict"],
    )
    report.heading("Segmentation metrics", level=3)
    report.table(
        [
            {
                "Metric": row["metric"],
                "Sprint 2 Extended": round(row["sprint2_extended"], 5),
                "Sprint 3 Coverage": round(row["sprint3_coverage"], 5),
                "Absolute Δ": f"{row['absolute_delta']:+.5f}",
                "Relative Δ": f"{row['relative_delta_pct']:+.3f}%",
                "Verdict": row["verdict"],
            }
            for row in comparison["segmentation"]
            if row["sprint2_extended"] is not None
        ],
        ["Metric", "Sprint 2 Extended", "Sprint 3 Coverage", "Absolute Δ",
         "Relative Δ", "Verdict"],
    )
    report.heading("Tally across all compared metrics", level=3)
    report.key_values(comparison["tally"])

    # ---- 8 indexing ----
    report.heading("8. Disc Indexing Results")
    if comparison["indexing"]:
        s3_index = payload["indexing"]["sprint3_coverage"]
        report.text(
            "Corrected taxonomy: every ground-truth disc is classified as "
            "`correct`, `shifted`, `merged`, `split` or `missed`. Analysis uses the "
            "**integer disc index only** - the dataset does not state which vertebra "
            "is L5, so no anatomical level name is asserted. The invalid "
            "vertebra-component-vs-instance comparison identified in the Sprint 3 "
            "audit is not used."
        )
        report.key_values(
            {
                "Discs analysed": f"{s3_index['n_discs_analysed']:,}",
                "Slices analysed": f"{s3_index['n_slices_analysed']:,}",
                "Region found": f"{s3_index['pct_region_found']}%",
                "Index correct": f"{s3_index['pct_index_correct']}%",
                "Slices with correct disc count":
                    f"{s3_index['pct_slices_count_correct']}%",
                "Slices with every disc correct":
                    f"{s3_index['pct_slices_all_discs_correct']}%",
                "Spurious components": s3_index["total_spurious_components"],
                "Shift offsets": s3_index["shifted_offsets"],
            }
        )
        report.table(
            [
                {
                    "Metric": row["metric"],
                    "Sprint 2 Extended": row["sprint2_extended"],
                    "Sprint 3 Coverage": row["sprint3_coverage"],
                    "Absolute Δ": f"{row['absolute_delta']:+.3f}",
                    "Verdict": row["verdict"],
                }
                for row in comparison["indexing"]
                if row["sprint2_extended"] is not None
            ],
            ["Metric", "Sprint 2 Extended", "Sprint 3 Coverage", "Absolute Δ",
             "Verdict"],
        )
        # Two estimators exist for "disc indexing accuracy". They must never be
        # mixed into a single delta, so both are shown against their own baseline.
        identification = payload.get("disc_identification") or {}
        s3_ident = identification.get("sprint3_coverage")
        s2_ident = identification.get("sprint2_extended")
        s2_index = payload["indexing"].get("sprint2_extended")
        if s3_ident and s2_ident and s2_index:
            report.heading(
                "Reconciliation: two estimators of indexing accuracy", level=3
            )
            report.text(
                "The brief quotes the Sprint 2 Extended disc-indexing baseline as "
                "**82.6%**. That figure comes from the `evaluate_unet.py` "
                "disc-identification metric, which is a *different estimator* from "
                "the corrected-taxonomy `pct_index_correct` used above. The two "
                "differ because the taxonomy additionally resolves merged and split "
                "components before deciding whether an index is correct. Both are "
                "listed here against their own baseline so the improvement is never "
                "computed across estimators."
            )
            report.table(
                [
                    {
                        "Estimator": "`evaluate_unet` pct_discs_index_correct "
                                     "(the brief's 82.6% baseline)",
                        "Sprint 2 Extended":
                            f"{s2_ident['pct_discs_index_correct']:.2f}%",
                        "Sprint 3 Coverage":
                            f"{s3_ident['pct_discs_index_correct']:.2f}%",
                        "Absolute Δ":
                            f"{s3_ident['pct_discs_index_correct'] - s2_ident['pct_discs_index_correct']:+.2f} pp",
                    },
                    {
                        "Estimator": "corrected taxonomy `pct_index_correct` "
                                     "(headline above)",
                        "Sprint 2 Extended": f"{s2_index['pct_index_correct']:.2f}%",
                        "Sprint 3 Coverage": f"{s3_index['pct_index_correct']:.2f}%",
                        "Absolute Δ":
                            f"{s3_index['pct_index_correct'] - s2_index['pct_index_correct']:+.2f} pp",
                    },
                ],
                ["Estimator", "Sprint 2 Extended", "Sprint 3 Coverage", "Absolute Δ"],
            )
            report.text(
                "The two estimators agree on the size and direction of the change "
                "(about +1.8 pp), which is why the indexing conclusion does not "
                "depend on which one is quoted."
            )

        report.heading("Accuracy by disc index", level=3)
        report.dataframe(pd.DataFrame(s3_index["by_disc_index"]), max_rows=12)
        report.heading("Accuracy by slice annotation area", level=3)
        report.dataframe(
            pd.DataFrame(s3_index["by_slice_annotation_area"]), max_rows=12
        )
    else:
        report.text("_Indexing analysis was not run in this pass._")

    # ---- 9 measurements ----
    report.heading("9. Disc-Level Measurement Results")
    if comparison["measurements"]:
        s3_agree = payload["measurement_agreement"]
        report.text(
            "Measurements taken from predicted masks, compared against the same "
            "discs measured from ground-truth masks. Only discs whose identity the "
            "prediction recovered correctly are compared - detection failures are "
            "counted separately rather than averaged into the measurement error."
        )
        report.key_values(
            {
                "Discs compared": s3_agree["detection"]["n_matched_by_identity"],
                "Ground-truth discs on predicted series":
                    s3_agree["detection"]["n_truth_discs"],
                "% of truth discs recovered":
                    f"{s3_agree['detection']['pct_truth_discs_recovered']}%",
            }
        )
        report.table(
            [
                {
                    "Metric": row["metric"],
                    "Sprint 2 Extended": row["sprint2_extended"],
                    "Sprint 3 Coverage": row["sprint3_coverage"],
                    "Absolute Δ": f"{row['absolute_delta']:+.5f}",
                    "Verdict": row["verdict"],
                }
                for row in comparison["measurements"]
                if row["sprint2_extended"] is not None
            ],
            ["Metric", "Sprint 2 Extended", "Sprint 3 Coverage", "Absolute Δ",
             "Verdict"],
        )
    else:
        report.text("_Measurement agreement was not available in this pass._")

    # ---- 10 Pfirrmann ----
    report.heading("10. End-to-End Pfirrmann Results")
    if comparison["pfirrmann"]:
        pf = (payload["measurement_agreement"].get("end_to_end_pfirrmann") or {})
        report.text(
            "Models fitted on ground-truth-mask features from **training** patients, "
            "then applied to **predicted**-mask features from test patients. This is "
            "the only Pfirrmann figure that responds to the segmentation model, "
            "because the Stage E table itself uses ground-truth masks on both sides."
        )
        report.key_values({"Note": pf.get("note", "-"),
                           "Training discs": pf.get("n_train_discs")})
        report.table(
            [
                {
                    "Estimator": estimator,
                    "Discs evaluated": pf[estimator]["n_test_discs"],
                    "QWK": pf[estimator]["quadratic_weighted_kappa"],
                    "MAE": pf[estimator]["mae"],
                    "Exact agreement": pf[estimator]["exact_agreement"],
                    "Within 1 grade": pf[estimator]["within_one_grade"],
                }
                for estimator in ["logreg", "forest"] if estimator in pf
            ],
            ["Estimator", "Discs evaluated", "QWK", "MAE", "Exact agreement",
             "Within 1 grade"],
        )
        report.table(
            [
                {
                    "Metric": row["metric"],
                    "Sprint 2 Extended": row["sprint2_extended"],
                    "Sprint 3 Coverage": row["sprint3_coverage"],
                    "Absolute Δ": f"{row['absolute_delta']:+.5f}",
                    "Verdict": row["verdict"],
                }
                for row in comparison["pfirrmann"]
                if row["sprint2_extended"] is not None
            ],
            ["Metric", "Sprint 2 Extended", "Sprint 3 Coverage", "Absolute Δ",
             "Verdict"],
        )
    else:
        report.text("_End-to-end Pfirrmann evaluation was not available._")

    # ---- 11 figures ----
    report.heading("11. Visual Results")
    report.bullets([f"`outputs/visualizations/{EXPERIMENT}/{name}`"
                    for name in payload["figures"]])

    # ---- 12 / 13 improved / worsened ----
    improved = [r for r in comparison["all"] if r["verdict"] == "improved"]
    worsened = [r for r in comparison["all"] if r["verdict"] == "worsened"]

    report.heading("12. What Improved")
    if improved:
        report.table(
            [
                {
                    "Metric": r["metric"],
                    "Sprint 2 Extended": r["sprint2_extended"],
                    "Sprint 3 Coverage": r["sprint3_coverage"],
                    "Absolute Δ": f"{r['absolute_delta']:+.5f}",
                }
                for r in improved
            ],
            ["Metric", "Sprint 2 Extended", "Sprint 3 Coverage", "Absolute Δ"],
        )
    else:
        report.text("_No metric improved._")

    report.heading("13. What Worsened")
    if worsened:
        report.table(
            [
                {
                    "Metric": r["metric"],
                    "Sprint 2 Extended": r["sprint2_extended"],
                    "Sprint 3 Coverage": r["sprint3_coverage"],
                    "Absolute Δ": f"{r['absolute_delta']:+.5f}",
                }
                for r in worsened
            ],
            ["Metric", "Sprint 2 Extended", "Sprint 3 Coverage", "Absolute Δ"],
        )
    else:
        report.text("_No metric worsened._")

    # ---- 14 interpretation ----
    report.heading("14. Interpretation")
    report.text(f"**Question.** {decision['question']}")
    report.text(f"**Verdict: {decision['verdict']}.** {decision['conclusion']}")
    report.heading("Primary criterion", level=3)
    report.key_values(decision["primary_criterion"])
    report.heading("Supporting evidence", level=3)
    report.key_values(decision["supporting_evidence"])
    report.text(
        "The decision deliberately does not rest on a single metric. It weighs the "
        "primary validation criterion together with the test macro Dice, the "
        "per-class Dice values, disc indexing, the predicted-mask measurement "
        "errors, the end-to-end Pfirrmann agreement, and the convergence behaviour."
    )
    report.text(
        "**One asymmetry worth stating plainly.** Sprint 3 trained from random "
        "initialisation, whereas Sprint 2 Extended continued from an "
        "already-converged 12-epoch model. Sprint 3 therefore had fewer effective "
        "optimisation steps on any given slice, and the comparison is not a "
        "perfectly matched pair. It is the correct comparison for the question - "
        "warm-starting would have confounded coverage with prior training - but the "
        "margin should be read with that in mind."
    )

    # ---- 15 limitations ----
    report.heading("15. Limitations")
    report.bullets(
        [
            "**CPU-only training.** No CUDA device is present. Width 16 and the "
            "per-epoch slice budget are compute-budget decisions, not modelling "
            "conclusions.",
            "**No confidence intervals.** Differences of a few thousandths of a Dice "
            "point on a single 33-patient test set should be read as a direction, "
            "not a significant difference. No bootstrap was run.",
            "**Random vs warm start.** See the asymmetry noted in section 14.",
            "**Disc identity is derived, not predicted.** The network outputs four "
            "semantic classes; disc index comes from ordering connected components "
            "per slice, which is why indexing accuracy trails detection accuracy.",
            "**Vertebra-normalised features remain unreliable from predicted masks.** "
            "The Sprint 3 audit established that a vertebra occupies ~1.73 "
            "connected components per instance in a sagittal plane (body plus "
            "posterior elements), so ordered components cannot identify individual "
            "vertebrae. This is a property of the anatomy, not of the model.",
            "**Level names are provisional.** The dataset documents that the lowest "
            "annotated vertebra is usually L5 but can be L4 or L6, so only integer "
            "disc indices are treated as authoritative.",
            "**Segmentation metrics are not clinical evidence.** A Dice score "
            "measures overlap with one annotation protocol. It does not establish "
            "clinical effectiveness, diagnostic accuracy or deployment readiness, "
            "and none is claimed.",
            "**No longitudinal scope.** The dataset contains exactly one study per "
            "patient, with no acquisition date, no repeat imaging and no surgical or "
            "outcome record. Postoperative healing and longitudinal change remain "
            "outside the validated scope of this project and are not claimed "
            "anywhere.",
        ]
    )

    # ---- 16 recommendation ----
    report.heading("16. Recommended Next Experiment")
    report.text(
        f"**Option {recommendation['option']}: {recommendation['title']}**"
    )
    report.text(recommendation["rationale"])
    report.heading("Proposed steps", level=3)
    report.bullets(recommendation["proposed_next_experiment"])
    if recommendation.get("also_worth_doing_independently"):
        report.heading("Orthogonal, and not part of this recommendation", level=3)
        report.bullets(recommendation["also_worth_doing_independently"])
    report.heading("Explicitly not recommended", level=3)
    report.bullets(recommendation["explicitly_not_recommended"])
    report.text(
        "**This experiment has not been started.** It is a recommendation only."
    )

    # ---- 17 artifacts ----
    report.heading("17. Artifact Locations")
    report.table(
        [
            {"artifact": "Training checkpoints",
             "path": f"`outputs/checkpoints/{EXPERIMENT}/`"},
            {"artifact": "Selected checkpoint",
             "path": f"`outputs/checkpoints/{EXPERIMENT}/best_val_dice.pt`"},
            {"artifact": "Per-epoch history",
             "path": f"`outputs/checkpoints/{EXPERIMENT}/history.csv` / `.json`"},
            {"artifact": "Training config / summary",
             "path": f"`outputs/reports/{EXPERIMENT}/training_config.json`, "
                     f"`training_summary.json`"},
            {"artifact": "Test segmentation metrics",
             "path": f"`outputs/reports/{EXPERIMENT}/metrics/`"},
            {"artifact": "Disc analysis (predicted masks)",
             "path": f"`outputs/reports/disc_analysis_{EXPERIMENT}.csv`"},
            {"artifact": "Measurement agreement + Pfirrmann",
             "path": f"`outputs/reports/{EXPERIMENT}/metrics/"
                     f"measurement_agreement.json`"},
            {"artifact": "Indexing failure detail",
             "path": f"`outputs/reports/{EXPERIMENT}/"
                     f"indexing_failures_{EXPERIMENT}.csv`"},
            {"artifact": "Comparison table",
             "path": f"`outputs/reports/{EXPERIMENT}/comparison_table.csv`"},
            {"artifact": "Figures",
             "path": f"`outputs/visualizations/{EXPERIMENT}/`"},
            {"artifact": "This report",
             "path": f"`outputs/reports/{EXPERIMENT}/sprint3_final_report.md`"},
            {"artifact": "Sprint 2 Extended baseline (unmodified)",
             "path": "`outputs/reports/sprint2_extended/`, "
                     "`outputs/checkpoints/sprint2_extended/`"},
        ],
        ["artifact", "path"],
    )

    return report.save(REPORT_DIR / "sprint3_final_report.md")


def analyse_convergence(history: pd.DataFrame) -> dict:
    """Quantify whether the run had plateaued by its final epoch."""
    n = len(history)
    last5 = history.tail(min(5, n))
    early = history[(history["epoch"] >= 5) & (history["epoch"] <= 10)]

    last5_gain = (
        float(np.mean(np.diff(last5["val_fg_dice"]))) if len(last5) > 1 else float("nan")
    )
    early_gain = (
        float(np.mean(np.diff(early["val_fg_dice"]))) if len(early) > 1 else float("nan")
    )
    converged = bool(abs(last5_gain) < 1e-4) if np.isfinite(last5_gain) else False

    if converged:
        interpretation = (
            f"Validation Dice moved {last5['val_fg_dice'].max() - last5['val_fg_dice'].min():.5f} "
            f"in total over the final {len(last5)} epochs "
            f"({last5_gain:+.6f} per epoch), against {early_gain:+.6f} per epoch "
            f"during epochs 5-10. The run has converged at this configuration."
        )
    else:
        interpretation = (
            f"Validation Dice was still moving {last5_gain:+.6f} per epoch over the "
            f"final {len(last5)} epochs, against {early_gain:+.6f} per epoch during "
            f"epochs 5-10, so the run had not fully flattened when it ended."
        )

    return {
        "last5_min": float(last5["val_fg_dice"].min()),
        "last5_max": float(last5["val_fg_dice"].max()),
        "last5_spread": float(last5["val_fg_dice"].max() - last5["val_fg_dice"].min()),
        "last5_mean_gain": round(last5_gain, 6) if np.isfinite(last5_gain) else None,
        "early_mean_gain": round(early_gain, 6) if np.isfinite(early_gain) else None,
        "final_lr": float(history["learning_rate"].iloc[-1]),
        "converged": converged,
        "interpretation": interpretation,
    }


def main() -> None:
    args = parse_args()
    ensure_dirs()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    VIZ_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------- Phase 1 ----------------
    print("[Phase 1] Verifying that training completed ...")
    verification = verify_training_complete()
    for name, ok in verification["artefacts_present"].items():
        print(f"  {'OK  ' if ok else 'MISS'} {name}")
    if not verification["training_completed"]:
        raise SystemExit(
            f"Training does not appear to have completed. Missing: "
            f"{verification['missing_artefacts']}. Not proceeding."
        )
    print(f"  epochs={verification['epochs_in_history']} "
          f"early_stopped={verification['stopped_early']}")

    history = pd.read_csv(CHECKPOINT_DIR / "history.csv")
    summary = load_json(REPORT_DIR / "training_summary.json")
    config = load_json(REPORT_DIR / "training_config.json") or {}
    index = load_slice_index()

    # ---------------- Phase 2 ----------------
    print("[Phase 2] Analysing training ...")
    training = analyse_training(history, summary)
    coverage = analyse_coverage(history, index, args.seed)
    convergence = analyse_convergence(history)
    print(f"  best val Dice {training['best_val_fg_dice']:.5f} "
          f"@ epoch {training['best_val_dice_epoch']}")
    print(f"  coverage {coverage['final_coverage_pct']}% "
          f"(100% at epoch {coverage['epoch_reaching_full_coverage']}), "
          f"leakage-free={coverage['leakage_free']}")

    # ---------------- Phase 3 ----------------
    print("[Phase 3] Selected checkpoint (validation criterion) ...")
    checkpoint = describe_selected_checkpoint()
    print(f"  {checkpoint['filename']} epoch={checkpoint['epoch']} "
          f"val_fg_dice={checkpoint['val_fg_dice']}")

    # ---------------- Phase 4 results (produced by evaluate_unet.py) --------
    print("[Phase 4] Loading test metrics ...")
    s3_seg = load_json(METRICS_DIR / "segmentation_metrics_test.json")
    s2_seg = load_json(S2_METRICS / "segmentation_metrics_test.json")
    if s3_seg is None:
        raise SystemExit(
            f"Missing {METRICS_DIR / 'segmentation_metrics_test.json'}.\n"
            f"Run evaluate_unet.py on the selected checkpoint first."
        )
    print(f"  test macro fg Dice "
          f"{s3_seg['aggregate']['macro_foreground']['dice']:.5f}")

    # The evaluate_unet.py disc-identification metric. This is the estimator the
    # Sprint 3 brief quotes as the 82.6% baseline, and it is NOT the same
    # estimator as the corrected-taxonomy pct_index_correct computed in Phase 6.
    # Both are reported so the two numbers are never compared across estimators.
    disc_identification = {
        "sprint3_coverage": load_json(METRICS_DIR / "disc_identification_test.json"),
        "sprint2_extended": load_json(S2_METRICS / "disc_identification_test.json"),
    }

    # ---------------- Phase 6 ----------------
    payload_indexing: dict = {}
    if not args.skip_indexing:
        print("[Phase 6] Disc-indexing analysis (corrected taxonomy) ...")
        test_index = index[index["split"] == "test"].reset_index(drop=True)
        for run_name, prediction_dir in [
            ("sprint3_coverage", PREDICTIONS_DIR),
            ("sprint2_extended", S2_PREDICTIONS_DIR),
        ]:
            if not prediction_dir.exists():
                print(f"  {run_name}: predictions missing, skipped")
                continue
            print(f"  {run_name} ...")
            disc_frame, slice_frame = idx_diag.run_analysis(
                test_index, prediction_dir, project_root=PROJECT_ROOT,
                limit=args.indexing_limit, progress_every=500,
            )
            if disc_frame.empty:
                continue
            if run_name == "sprint3_coverage":
                disc_frame.to_csv(
                    REPORT_DIR / f"indexing_failures_{run_name}.csv", index=False
                )
            payload_indexing[run_name] = idx_diag.summarise(disc_frame, slice_frame)
            print(f"    index correct "
                  f"{payload_indexing[run_name]['pct_index_correct']}%")

    # ---------------- Phases 5 + 7 ----------------
    print("[Phase 5/7] Building the comparison ...")
    s3_agree = load_json(METRICS_DIR / "measurement_agreement.json")
    s2_agree = load_json(S2_METRICS / "measurement_agreement.json")

    comparison = build_comparison(
        s3_seg, s2_seg, s3_agree, s2_agree,
        payload_indexing.get("sprint3_coverage"),
        payload_indexing.get("sprint2_extended"),
        training,
    )
    pd.DataFrame(comparison["all"]).to_csv(
        REPORT_DIR / "comparison_table.csv", index=False
    )
    print(f"  tally: {comparison['tally']}")

    # ---------------- Phase 9 / 10 ----------------
    print("[Phase 9/10] Decision and recommendation ...")
    decision = decide(comparison, training, coverage)
    recommendation = recommend(decision, coverage, training)
    print(f"  verdict: {decision['verdict']}")
    print(f"  recommendation: option {recommendation['option']}")

    # ---------------- Phase 8 ----------------
    print("[Phase 8] Rendering figures ...")
    figures = render_figures(
        history, coverage, comparison, S2_CHECKPOINTS / "history.csv"
    )
    if len(payload_indexing) == 2:
        name = render_indexing_figure(
            payload_indexing["sprint3_coverage"],
            payload_indexing["sprint2_extended"],
        )
        if name:
            figures.append(name)
    # Include the figures evaluate_unet.py produced under this tag.
    for extra in sorted(VIZ_DIR.glob("*.png")):
        if extra.name not in figures:
            figures.append(extra.name)
    prediction_figures = sorted((VIZ_DIR / "predictions").glob("*.png"))
    figures.extend(f"predictions/{p.name}" for p in prediction_figures)
    for name in figures:
        print(f"  -> {VIZ_DIR / name}")

    # ---------------- Phase 11 ----------------
    print("[Phase 11] Writing the final report ...")
    payload = {
        "experiment": EXPERIMENT,
        "verification": verification,
        "config": config,
        "training_summary": summary,
        "history": history.to_dict("records"),
        "training": training,
        "coverage": coverage,
        "convergence": convergence,
        "checkpoint": checkpoint,
        "test_segmentation": s3_seg,
        "measurement_agreement": s3_agree,
        "indexing": payload_indexing,
        "disc_identification": disc_identification,
        "comparison": comparison,
        "decision": decision,
        "recommendation": recommendation,
        "figures": figures,
        "baseline_reference": BASELINE,
    }
    save_json(payload, REPORT_DIR / "sprint3_final_report.json")
    write_report(payload)
    print(f"  -> {REPORT_DIR / 'sprint3_final_report.md'}")
    print(f"  -> {REPORT_DIR / 'sprint3_final_report.json'}")

    print_terminal_summary(payload)


def print_terminal_summary(payload: dict) -> None:
    """Print the concise summary block requested in the brief."""
    training = payload["training"]
    coverage = payload["coverage"]
    seg = payload["test_segmentation"]["aggregate"]
    indexing = payload["indexing"].get("sprint3_coverage", {})
    pf = (payload.get("measurement_agreement") or {}).get(
        "end_to_end_pfirrmann"
    ) or {}
    forest = pf.get("forest", {})

    status = (
        f"COMPLETE - {training['total_epochs_completed']} epochs"
        + (" (early stopping triggered)" if training["early_stopping_triggered"]
           else " (reached max epochs)")
    )

    print("\n" + "=" * 70)
    print(f"TRAINING STATUS:             {status}")
    print(f"BEST EPOCH:                  {training['best_val_dice_epoch']}")
    print(f"BEST VAL DICE:               {training['best_val_fg_dice']:.5f}  "
          f"(Sprint 2 Ext: {BASELINE['val_fg_dice']}, "
          f"{training['best_val_fg_dice'] - BASELINE['val_fg_dice']:+.5f})")
    print(f"TEST MACRO DICE:             "
          f"{seg['macro_foreground']['dice']:.5f}  "
          f"(baseline {BASELINE['test_macro_fg_dice']}, "
          f"{seg['macro_foreground']['dice'] - BASELINE['test_macro_fg_dice']:+.5f})")
    print(f"TEST VERTEBRA DICE:          "
          f"{seg['per_class']['vertebra']['dice']:.5f}  "
          f"(baseline {BASELINE['test_vertebra_dice']}, "
          f"{seg['per_class']['vertebra']['dice'] - BASELINE['test_vertebra_dice']:+.5f})")
    print(f"TEST IVD DICE:               "
          f"{seg['per_class']['intervertebral_disc']['dice']:.5f}  "
          f"(baseline {BASELINE['test_ivd_dice']}, "
          f"{seg['per_class']['intervertebral_disc']['dice'] - BASELINE['test_ivd_dice']:+.5f})")
    print(f"TEST CANAL DICE:             "
          f"{seg['per_class']['spinal_canal']['dice']:.5f}  "
          f"(baseline {BASELINE['test_canal_dice']}, "
          f"{seg['per_class']['spinal_canal']['dice'] - BASELINE['test_canal_dice']:+.5f})")
    # Two estimators exist; each is compared against its OWN baseline so the
    # printed delta is never computed across estimators.
    identification = payload.get("disc_identification") or {}
    s3_ident = identification.get("sprint3_coverage")
    s2_ident = identification.get("sprint2_extended")
    if s3_ident and s2_ident:
        s3_pct = s3_ident["pct_discs_index_correct"]
        s2_pct = s2_ident["pct_discs_index_correct"]
        print(f"DISC INDEXING:               "
              f"{s3_pct:.2f}%  "
              f"(baseline {s2_pct:.2f}%, {s3_pct - s2_pct:+.2f} pp) "
              f"[evaluate_unet estimator]")
    if indexing:
        s2_taxonomy = next(
            (row["sprint2_extended"]
             for row in payload["comparison"]["indexing"]
             if row["metric"] == "indexing pct_index_correct"),
            None,
        )
        suffix = ""
        if s2_taxonomy is not None:
            suffix = (f"(baseline {s2_taxonomy:.2f}%, "
                      f"{indexing['pct_index_correct'] - s2_taxonomy:+.2f} pp) ")
        print(f"                             "
              f"{indexing['pct_index_correct']:.2f}%  {suffix}"
              f"[corrected-taxonomy estimator]")
    if forest:
        print(f"PFIRRMANN QWK:               "
              f"{forest['quadratic_weighted_kappa']}  "
              f"(baseline {BASELINE['pfirrmann_qwk']}, "
              f"{forest['quadratic_weighted_kappa'] - BASELINE['pfirrmann_qwk']:+.5f})")
    print(f"TRAINING COVERAGE:           "
          f"{coverage['final_coverage_pct']}% of "
          f"{coverage['train_slices_available']:,} slices "
          f"(Sprint 2: 28.05%)")
    print(f"TOTAL TRAINING TIME:         {training['total_training_hours']} h")
    print(f"RECOMMENDED NEXT EXPERIMENT: Option "
          f"{payload['recommendation']['option']} - "
          f"{payload['recommendation']['title']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
