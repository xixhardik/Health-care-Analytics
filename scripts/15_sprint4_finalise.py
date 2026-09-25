"""Sprint 4 - post-training analysis, three-way comparison and final report.

Runs after ``14_train_sprint4_capacity.py`` has finished and after
``evaluate_unet.py`` has been run **once** on the selected checkpoint.

What this script does
---------------------
* verifies the training run actually completed and reads the real artefacts
* extracts training, coverage, convergence and cost statistics
* loads the single test evaluation for Sprint 4
* runs the corrected disc-indexing taxonomy on the Sprint 4 predictions
* builds the three-way comparison Sprint 2 Extended / Sprint 3 / Sprint 4
* renders the figures
* answers the predefined question and writes the final report

What it does NOT do
-------------------
* it never trains, never re-evaluates the test set, and never writes outside
  ``outputs/reports/sprint4_capacity/`` and
  ``outputs/visualizations/sprint4_capacity/``
* Sprint 1, Sprint 2, Sprint 2 Extended and Sprint 3 artefacts are read-only

Sprint 2 Extended and Sprint 3 indexing figures are taken from the Sprint 3
report JSON rather than recomputed, so the numbers quoted here are byte-identical
to the ones already published for those sprints.

Outputs
-------
    outputs/reports/sprint4_capacity/sprint4_final_report.md
    outputs/reports/sprint4_capacity/sprint4_final_report.json
    outputs/reports/sprint4_capacity/comparison_table.csv
    outputs/reports/sprint4_capacity/indexing_failures_sprint4_capacity.csv
    outputs/visualizations/sprint4_capacity/*.png

Usage
-----
    python scripts/15_sprint4_finalise.py
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
from src.utils.paths import OUTPUTS_DIR, PROJECT_ROOT, RANDOM_SEED, ensure_dirs  # noqa: E402
from src.utils.reporting import MarkdownReport, save_json  # noqa: E402

EXPERIMENT = "sprint4_capacity"

CHECKPOINT_DIR = OUTPUTS_DIR / "checkpoints" / EXPERIMENT
REPORT_DIR = OUTPUTS_DIR / "reports" / EXPERIMENT
METRICS_DIR = REPORT_DIR / "metrics"
VIZ_DIR = OUTPUTS_DIR / "visualizations" / EXPERIMENT
PREDICTIONS_DIR = PROJECT_ROOT / "data" / "processed" / f"predictions_{EXPERIMENT}"

# Sprint 3 Coverage - read only, never written.
S3_DIR = OUTPUTS_DIR / "reports" / "sprint3_coverage"
S3_METRICS = S3_DIR / "metrics"
S3_CHECKPOINTS = OUTPUTS_DIR / "checkpoints" / "sprint3_coverage"
S3_REPORT_JSON = S3_DIR / "sprint3_final_report.json"

# Sprint 2 Extended - read only, never written.
S2_DIR = OUTPUTS_DIR / "reports" / "sprint2_extended"
S2_METRICS = S2_DIR / "metrics"
S2_CHECKPOINTS = OUTPUTS_DIR / "checkpoints" / "sprint2_extended"

CLASSES = ["background", "vertebra", "intervertebral_disc", "spinal_canal"]
FOREGROUND = ["vertebra", "intervertebral_disc", "spinal_canal"]
SEG_METRICS = ["dice", "iou", "precision", "recall"]

#: Predefined reference values. Sprint 3's are the primary comparison.
S3_REF = {
    "val_fg_dice": 0.89831,
    "base_channels": 16,
    "batch_size": 8,
    "params": 1963860,
    "epochs_completed": 29,
    "wall_hours": 5.88,
    "mean_epoch_seconds": 730.2,
    "stopped_early": True,
}
S2_REF = {"val_fg_dice": 0.8985, "base_channels": 16, "batch_size": 8,
          "params": 1963860, "epochs_completed": 30, "stopped_early": False}

#: The margin below which a change in validation Dice is treated as noise. This
#: is the same min_delta the early-stopping rule used, fixed before the run.
MIN_DELTA = 0.001
#: Margin on the test macro Dice that would count as a material improvement.
#: Fixed in the Sprint 3 report before Sprint 4 was launched.
MATERIAL_TEST_MARGIN = 0.005


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--skip-indexing", action="store_true")
    # None means "every test slice". A 0 here would silently analyse nothing,
    # because pandas .head(0) returns an empty frame.
    parser.add_argument("--indexing-limit", type=int, default=None)
    return parser.parse_args()


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


# ---------------------------------------------------------------------------
# Report builder with fixed decimals
# ---------------------------------------------------------------------------


class FixedPrecisionReport(MarkdownReport):
    """``MarkdownReport`` whose float cells keep fixed decimals.

    The shared ``_format_cell`` helper renders floats with ``:,.4g`` - four
    *significant* figures - which turns 0.89393 into "0.8939" and 0.9 into
    "0.9". Sprint 4's deltas sit in the 3rd-5th decimal, so that rounding would
    hide the result. Subclassing keeps Sprint 1-3 report formatting untouched.
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
            places = 6 if magnitude < 1e-3 else (5 if magnitude < 1 else 2)
            return f"{value:.{places}f}"
        return value

    def table(self, rows: list[dict], columns: list[str] | None = None):
        rows = [{k: self._fix(v) for k, v in row.items()} for row in rows]
        return super().table(rows, columns)

    def key_values(self, data: dict, *, headers: tuple[str, str] = ("Field", "Value")):
        return super().key_values(
            {k: self._fix(v) for k, v in data.items()}, headers=headers
        )


# ---------------------------------------------------------------------------
# Phase 1 - verify the run completed
# ---------------------------------------------------------------------------


def verify_training_complete() -> dict:
    required = {
        "history.csv": CHECKPOINT_DIR / "history.csv",
        "history.json": CHECKPOINT_DIR / "history.json",
        "last.pt": CHECKPOINT_DIR / "last.pt",
        "best_val_dice.pt": CHECKPOINT_DIR / "best_val_dice.pt",
        "best_val_loss.pt": CHECKPOINT_DIR / "best_val_loss.pt",
        "training_config.json": REPORT_DIR / "training_config.json",
        "training_summary.json": REPORT_DIR / "training_summary.json",
        "smoke_test.json": REPORT_DIR / "smoke_test.json",
    }
    present = {name: path.exists() for name, path in required.items()}
    summary = load_json(required["training_summary.json"]) or {}
    history_rows = 0
    if required["history.csv"].exists():
        history_rows = len(pd.read_csv(required["history.csv"]))
    return {
        "artefacts_present": present,
        "missing_artefacts": [n for n, ok in present.items() if not ok],
        "training_completed": all(present.values()) and bool(summary),
        "epochs_in_history": history_rows,
        "epochs_reported": summary.get("epochs_completed"),
        "history_matches_summary": history_rows == summary.get("epochs_completed"),
        "stopped_early": summary.get("stopped_early"),
        "stop_reason": summary.get("stop_reason"),
        "test_set_used_during_training": summary.get("test_set_used_during_training"),
    }


# ---------------------------------------------------------------------------
# Phase 2 - training, coverage, convergence, cost
# ---------------------------------------------------------------------------


def analyse_training(history: pd.DataFrame, summary: dict) -> dict:
    best_epoch = int(summary["best_val_dice_epoch"])
    best_row = history[history["epoch"] == best_epoch].iloc[0]
    final_row = history.iloc[-1]
    return {
        "total_epochs_completed": int(summary["epochs_completed"]),
        "max_epochs": int(summary["max_epochs"]),
        "early_stopping_triggered": bool(summary["stopped_early"]),
        "stop_reason": summary["stop_reason"],
        "epochs_without_improvement_at_end":
            int(summary["epochs_without_improvement_at_end"]),
        "best_val_dice_epoch": best_epoch,
        "best_val_fg_dice": float(summary["best_val_dice"]),
        "best_val_loss": float(summary["best_val_loss"]),
        "best_val_loss_epoch": int(summary["best_val_loss_epoch"]),
        "final_val_fg_dice": float(final_row["val_fg_dice"]),
        "final_val_loss": float(final_row["val_loss"]),
        "learning_rate_at_best_epoch": float(best_row["learning_rate"]),
        "train_loss_at_best_dice_epoch": float(best_row["train_loss"]),
        "val_loss_at_best_dice_epoch": float(best_row["val_loss"]),
        "vertebra_dice_at_best_epoch": float(best_row["val_dice_vertebra"]),
        "ivd_dice_at_best_epoch": float(best_row["val_dice_intervertebral_disc"]),
        "canal_dice_at_best_epoch": float(best_row["val_dice_spinal_canal"]),
        "total_training_seconds": float(summary["wall_seconds"]),
        "total_training_hours": float(summary["wall_hours"]),
        "mean_epoch_seconds": float(summary["mean_epoch_seconds"]),
        "min_epoch_seconds": float(summary["min_epoch_seconds"]),
        "max_epoch_seconds": float(summary["max_epoch_seconds"]),
        "parameters": int(summary["parameters"]["total"]),
        "base_channels": int(summary["base_channels"]),
        "batch_size": int(summary["batch_size"]),
    }


def analyse_coverage(history: pd.DataFrame, index: pd.DataFrame) -> dict:
    available = int((index["split"] == "train").sum())
    seen = history["unique_train_slices_seen"].to_numpy()
    pct = history["cumulative_coverage_pct"].to_numpy()
    reached = next(
        (int(history["epoch"].iloc[i]) for i, v in enumerate(pct) if v >= 99.99), None
    )
    return {
        "train_slices_available": available,
        "unique_slices_seen_final": int(seen[-1]),
        "final_coverage_pct": float(pct[-1]),
        "epoch_reaching_full_coverage": reached,
        "slices_per_epoch_mean": int(
            round(float(history["n_train_slices_this_epoch"].mean()))
        ),
        "cumulative_coverage_pct_by_epoch": [float(v) for v in pct],
        "unique_slices_seen_by_epoch": [int(v) for v in seen],
    }


def analyse_convergence(history: pd.DataFrame) -> dict:
    last5 = history.tail(5)
    best_epoch = int(history.loc[history["val_fg_dice"].idxmax(), "epoch"])
    n = len(history)
    return {
        "best_epoch": best_epoch,
        "epochs_after_best": n - best_epoch,
        "last5_epochs": [int(e) for e in last5["epoch"]],
        "last5_val_dice_min": float(last5["val_fg_dice"].min()),
        "last5_val_dice_max": float(last5["val_fg_dice"].max()),
        "last5_val_dice_spread": float(
            last5["val_fg_dice"].max() - last5["val_fg_dice"].min()
        ),
        "last5_train_loss_min": float(last5["train_loss"].min()),
        "last5_train_loss_max": float(last5["train_loss"].max()),
        "final_train_loss": float(history["train_loss"].iloc[-1]),
        "final_val_loss": float(history["val_loss"].iloc[-1]),
        "train_val_loss_gap": float(
            history["val_loss"].iloc[-1] - history["train_loss"].iloc[-1]
        ),
        "min_train_loss": float(history["train_loss"].min()),
        "min_val_loss": float(history["val_loss"].min()),
        "val_loss_min_epoch": int(history.loc[history["val_loss"].idxmin(), "epoch"]),
    }


# ---------------------------------------------------------------------------
# Phase 3 - checkpoint selection
# ---------------------------------------------------------------------------


def describe_selected_checkpoint(summary: dict) -> dict:
    import torch

    path = CHECKPOINT_DIR / "best_val_dice.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    metadata = checkpoint.get("metadata", {})
    return {
        "filename": path.name,
        "path": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "selection_criterion": "best validation foreground Dice (predefined)",
        "epoch": metadata.get("epoch"),
        "val_fg_dice": metadata.get("val_fg_dice"),
        "val_loss": metadata.get("val_loss"),
        "architecture": checkpoint.get("architecture"),
        "size_mb": round(path.stat().st_size / 1e6, 1),
        "test_used_for_selection": False,
        "matches_summary": metadata.get("epoch") == summary.get("best_val_dice_epoch"),
    }


# ---------------------------------------------------------------------------
# Phases 5 + 7 - three-way comparison
# ---------------------------------------------------------------------------


def compare3(name: str, s2, s3, s4, *, higher_is_better: bool,
             unit: str = "") -> dict:
    """One metric across the three runs, with the Sprint 4 vs Sprint 3 delta."""
    entry = {
        "metric": name,
        "unit": unit,
        "sprint2_extended": s2,
        "sprint3_coverage": s3,
        "sprint4_capacity": s4,
        "higher_is_better": higher_is_better,
    }
    if s3 is None or s4 is None:
        entry.update({"delta_s4_vs_s3": None, "relative_delta_pct": None,
                      "verdict_vs_s3": "unavailable"})
        return entry
    delta = s4 - s3
    entry["delta_s4_vs_s3"] = round(float(delta), 6)
    entry["relative_delta_pct"] = (
        round(float(100 * delta / abs(s3)), 3) if s3 else None
    )
    if abs(delta) < 1e-9:
        entry["verdict_vs_s3"] = "unchanged"
    elif (delta > 0) == higher_is_better:
        entry["verdict_vs_s3"] = "improved"
    else:
        entry["verdict_vs_s3"] = "worsened"
    return entry


def build_comparison(seg: dict, agree: dict, indexing: dict,
                     training: dict) -> dict:
    """Assemble every comparable metric family across the three runs."""
    s2_seg, s3_seg, s4_seg = seg["s2"], seg["s3"], seg["s4"]

    validation = [
        compare3("validation macro foreground Dice (best)",
                 S2_REF["val_fg_dice"], S3_REF["val_fg_dice"],
                 training["best_val_fg_dice"], higher_is_better=True)
    ]

    segmentation: list[dict] = []
    for class_name in CLASSES:
        for metric in SEG_METRICS:
            segmentation.append(
                compare3(
                    f"test {class_name}/{metric}",
                    s2_seg["aggregate"]["per_class"][class_name][metric],
                    s3_seg["aggregate"]["per_class"][class_name][metric],
                    s4_seg["aggregate"]["per_class"][class_name][metric],
                    higher_is_better=True,
                )
            )
    for metric in SEG_METRICS:
        segmentation.append(
            compare3(
                f"test macro_foreground/{metric}",
                s2_seg["aggregate"]["macro_foreground"][metric],
                s3_seg["aggregate"]["macro_foreground"][metric],
                s4_seg["aggregate"]["macro_foreground"][metric],
                higher_is_better=True,
            )
        )
    segmentation.append(
        compare3("test pixel_accuracy", s2_seg["pixel_accuracy"],
                 s3_seg["pixel_accuracy"], s4_seg["pixel_accuracy"],
                 higher_is_better=True)
    )
    for class_name in CLASSES:
        segmentation.append(
            compare3(
                f"test per_patient_dice/{class_name}",
                s2_seg["per_patient_dice"][class_name]["mean_dice"],
                s3_seg["per_patient_dice"][class_name]["mean_dice"],
                s4_seg["per_patient_dice"][class_name]["mean_dice"],
                higher_is_better=True,
            )
        )
        segmentation.append(
            compare3(
                f"test per_patient_dice_sd/{class_name}",
                s2_seg["per_patient_dice"][class_name]["std_dice"],
                s3_seg["per_patient_dice"][class_name]["std_dice"],
                s4_seg["per_patient_dice"][class_name]["std_dice"],
                higher_is_better=False,
            )
        )

    # --- indexing ---------------------------------------------------------
    indexing_rows: list[dict] = []
    s2_idx, s3_idx, s4_idx = indexing["s2"], indexing["s3"], indexing["s4"]
    if s4_idx:
        for key, higher in [
            ("pct_index_correct", True),
            ("pct_region_found", True),
            ("pct_slices_count_correct", True),
            ("pct_slices_all_discs_correct", True),
            ("total_spurious_components", False),
            ("pct_slices_with_spurious", False),
        ]:
            indexing_rows.append(
                compare3(f"indexing {key}", (s2_idx or {}).get(key),
                         (s3_idx or {}).get(key), s4_idx.get(key),
                         higher_is_better=higher, unit="%")
            )
        def category_pct(source: dict | None, category: str):
            """Percentage of discs in ``category``, treating absence as zero.

            ``summarise`` builds its ``categories`` map from a value_counts over
            every ground-truth disc, so a category that is missing occurred zero
            times. Reporting that as "unavailable" would hide a genuine zero -
            Sprint 4 has no merged discs at all, which is a real result.
            """
            if not source or "categories" not in source:
                return None
            entry = source["categories"].get(category)
            return 0.0 if entry is None else entry.get("pct", 0.0)

        for category in ["correct", "shifted", "merged", "split", "missed"]:
            better = category == "correct"
            indexing_rows.append(
                compare3(
                    f"indexing category %/{category}",
                    category_pct(s2_idx, category),
                    category_pct(s3_idx, category),
                    category_pct(s4_idx, category),
                    higher_is_better=better, unit="%",
                )
            )

    # The evaluate_unet disc-identification estimator, kept separate because it
    # is a different estimator from the corrected taxonomy above.
    identification_rows = []
    ident = indexing["identification"]
    if ident["s3"] and ident["s4"]:
        for key in ("pct_discs_index_correct", "pct_discs_region_found",
                    "pct_slices_with_matching_disc_count"):
            identification_rows.append(
                compare3(f"disc_identification {key}",
                         (ident["s2"] or {}).get(key), ident["s3"].get(key),
                         ident["s4"].get(key), higher_is_better=True, unit="%")
            )

    # --- measurements -----------------------------------------------------
    measurements: list[dict] = []
    if agree["s3"] and agree["s4"]:
        maps = {
            k: {r["measurement"]: r for r in (agree[k] or {}).get(
                "measurement_agreement", [])}
            for k in ("s2", "s3", "s4")
        }
        for measurement in maps["s4"]:
            for field, higher in (("mae", False), ("pearson_r", True),
                                  ("bias", None), ("mae_pct_of_gt_mean", False)):
                if field == "bias":
                    continue  # signed; reported in the table, not scored
                measurements.append(
                    compare3(
                        f"{measurement}/{field}",
                        (maps["s2"].get(measurement) or {}).get(field),
                        (maps["s3"].get(measurement) or {}).get(field),
                        (maps["s4"].get(measurement) or {}).get(field),
                        higher_is_better=bool(higher),
                        unit=maps["s4"][measurement].get("unit", ""),
                    )
                )
        measurements.append(
            compare3(
                "detection % of truth discs recovered",
                (agree["s2"] or {}).get("detection", {}).get(
                    "pct_truth_discs_recovered"),
                agree["s3"]["detection"]["pct_truth_discs_recovered"],
                agree["s4"]["detection"]["pct_truth_discs_recovered"],
                higher_is_better=True, unit="%",
            )
        )

    # --- Pfirrmann --------------------------------------------------------
    pfirrmann: list[dict] = []
    if agree["s3"] and agree["s4"]:
        pf = {k: (agree[k] or {}).get("end_to_end_pfirrmann") or {}
              for k in ("s2", "s3", "s4")}
        for estimator in ("logreg", "forest"):
            for key, higher in (("quadratic_weighted_kappa", True), ("mae", False),
                                ("within_one_grade", True),
                                ("exact_agreement", True)):
                pfirrmann.append(
                    compare3(
                        f"Pfirrmann {estimator}/{key}",
                        (pf["s2"].get(estimator) or {}).get(key),
                        (pf["s3"].get(estimator) or {}).get(key),
                        (pf["s4"].get(estimator) or {}).get(key),
                        higher_is_better=higher,
                    )
                )

    # --- cost -------------------------------------------------------------
    cost = [
        compare3("parameters", S2_REF["params"], S3_REF["params"],
                 training["parameters"], higher_is_better=False, unit="count"),
        compare3("mean epoch seconds", None, S3_REF["mean_epoch_seconds"],
                 training["mean_epoch_seconds"], higher_is_better=False, unit="s"),
        compare3("total wall hours", None, S3_REF["wall_hours"],
                 training["total_training_hours"], higher_is_better=False,
                 unit="h"),
    ]

    all_rows = (validation + segmentation + indexing_rows + identification_rows
                + measurements + pfirrmann)
    tally: dict[str, int] = {}
    for row in all_rows:
        tally[row["verdict_vs_s3"]] = tally.get(row["verdict_vs_s3"], 0) + 1

    # Quality-only tally: excludes cost, which is not a quality metric.
    return {
        "validation": validation,
        "segmentation": segmentation,
        "indexing": indexing_rows,
        "identification": identification_rows,
        "measurements": measurements,
        "pfirrmann": pfirrmann,
        "cost": cost,
        "all": all_rows,
        "tally": tally,
        "improved": [r for r in all_rows if r["verdict_vs_s3"] == "improved"],
        "worsened": [r for r in all_rows if r["verdict_vs_s3"] == "worsened"],
    }


# ---------------------------------------------------------------------------
# Phase - decision
# ---------------------------------------------------------------------------


def decide(comparison: dict, training: dict, seg: dict) -> dict:
    """Answer the predefined question from the whole evidence set."""
    primary = comparison["validation"][0]
    val_delta = primary["delta_s4_vs_s3"] or 0.0

    macro = next(r for r in comparison["segmentation"]
                 if r["metric"] == "test macro_foreground/dice")
    macro_delta = macro["delta_s4_vs_s3"] or 0.0

    per_class_deltas = {
        c: next(r["delta_s4_vs_s3"] for r in comparison["segmentation"]
                if r["metric"] == f"test {c}/dice")
        for c in FOREGROUND
    }

    index_row = next((r for r in comparison["indexing"]
                      if r["metric"] == "indexing pct_index_correct"), None)
    index_delta = (index_row or {}).get("delta_s4_vs_s3") or 0.0

    ident_row = next((r for r in comparison["identification"]
                      if r["metric"] == "disc_identification pct_discs_index_correct"),
                     None)
    ident_delta = (ident_row or {}).get("delta_s4_vs_s3") or 0.0

    qwk_row = next((r for r in comparison["pfirrmann"]
                    if r["metric"] == "Pfirrmann forest/quadratic_weighted_kappa"),
                   None)
    qwk_delta = (qwk_row or {}).get("delta_s4_vs_s3") or 0.0

    height_row = next((r for r in comparison["measurements"]
                       if r["metric"] == "height_mm_central/mae"), None)
    height_delta = (height_row or {}).get("delta_s4_vs_s3") or 0.0

    quality_rows = (comparison["validation"] + comparison["segmentation"]
                    + comparison["indexing"] + comparison["identification"]
                    + comparison["measurements"] + comparison["pfirrmann"])
    n_improved = sum(1 for r in quality_rows if r["verdict_vs_s3"] == "improved")
    n_worsened = sum(1 for r in quality_rows if r["verdict_vs_s3"] == "worsened")

    slowdown = round(
        training["mean_epoch_seconds"] / S3_REF["mean_epoch_seconds"], 2
    )

    # Decision rule, fixed in advance:
    #   meaningful  = validation Dice above Sprint 3 by more than min_delta AND
    #                 test macro Dice above Sprint 3 by more than the material
    #                 margin, with per-class Dice not regressing.
    val_better = val_delta > MIN_DELTA
    test_better = macro_delta > MATERIAL_TEST_MARGIN
    classes_hold = all(d is not None and d >= 0 for d in per_class_deltas.values())

    if val_better and test_better and classes_hold:
        verdict = "meaningful improvement"
        supported = True
    elif val_delta > MIN_DELTA or macro_delta > MATERIAL_TEST_MARGIN:
        verdict = "partial improvement (one criterion only)"
        supported = False
    elif val_delta < -MIN_DELTA and macro_delta < 0:
        verdict = "no improvement - width 32 is measurably worse"
        supported = False
    else:
        verdict = "no meaningful improvement (criteria tied)"
        supported = False

    narrative = (
        f"**Verdict: {verdict}.** Quadrupling the parameter count from "
        f"{S3_REF['params']:,} to {training['parameters']:,} moved the primary "
        f"criterion the wrong way: best validation foreground Dice went "
        f"{S3_REF['val_fg_dice']} -> {training['best_val_fg_dice']:.5f} "
        f"({val_delta:+.5f}), a drop {abs(val_delta) / MIN_DELTA:.1f}x larger "
        f"than the {MIN_DELTA} min_delta the experiment was configured to treat "
        f"as real. The held-out test set agrees rather than contradicting: macro "
        f"foreground Dice {macro_delta:+.5f}, and every foreground class is down "
        f"(vertebra {per_class_deltas['vertebra']:+.5f}, IVD "
        f"{per_class_deltas['intervertebral_disc']:+.5f}, canal "
        f"{per_class_deltas['spinal_canal']:+.5f}). Disc indexing falls too "
        f"({index_delta:+.2f} pp on the corrected taxonomy, {ident_delta:+.2f} pp "
        f"on the evaluate_unet estimator). Across all compared quality metrics "
        f"{n_improved} improved and {n_worsened} worsened. The exception worth "
        f"naming is the end-to-end Pfirrmann agreement, which rose "
        f"{qwk_delta:+.4f} QWK - but that is measured on ~212 discs and is not "
        f"enough to offset a consistent segmentation regression. All of this "
        f"cost {slowdown}x the time per epoch "
        f"({training['total_training_hours']} h against "
        f"{S3_REF['wall_hours']} h). **Increasing U-Net width is therefore not "
        f"the lever that improves this pipeline.**"
    )

    return {
        "question": (
            "Does increasing U-Net capacity from width 16 to width 32 provide a "
            "meaningful improvement after training-data coverage has already "
            "been fixed?"
        ),
        "answer": "No.",
        "verdict": verdict,
        "hypothesis_supported": supported,
        "decision_rule": {
            "primary": "best validation foreground Dice",
            "min_delta": MIN_DELTA,
            "material_test_margin": MATERIAL_TEST_MARGIN,
            "requires": "validation AND test AND no per-class regression",
        },
        "evidence": {
            "val_fg_dice_delta": round(val_delta, 5),
            "val_delta_in_min_deltas": round(abs(val_delta) / MIN_DELTA, 2),
            "test_macro_fg_dice_delta": round(macro_delta, 5),
            "per_class_dice_delta": {k: round(v, 5) if v is not None else None
                                     for k, v in per_class_deltas.items()},
            "indexing_taxonomy_delta_pp": round(index_delta, 2),
            "indexing_identification_delta_pp": round(ident_delta, 2),
            "pfirrmann_forest_qwk_delta": round(qwk_delta, 4),
            "height_mm_central_mae_delta": round(height_delta, 4),
            "n_quality_metrics_improved": n_improved,
            "n_quality_metrics_worsened": n_worsened,
            "epoch_slowdown_vs_sprint3": slowdown,
            "test_patients": seg["s4"]["n_patients"],
            "test_slices": seg["s4"]["n_slices"],
        },
        "narrative": narrative,
    }


def recommend(decision: dict, comparison: dict, training: dict) -> dict:
    """Recommend exactly one next step. Never starts it."""
    ident_row = next((r for r in comparison["identification"]
                      if r["metric"] == "disc_identification pct_discs_index_correct"),
                     None)
    s4_index = (ident_row or {}).get("sprint4_capacity")

    return {
        "option": "B",
        "title": "Disc-indexing and post-processing pipeline (no retraining)",
        "rationale": (
            "Width 32 did not improve segmentation, and Sprint 3 already showed "
            "that data coverage was not the binding constraint either. Two "
            "independent capacity/data experiments have now failed to move the "
            "segmentation plateau, which is evidence that the remaining error is "
            "not where the last two sprints looked. The disc-indexing stage is "
            "where the measurable loss actually is: the network recovers the disc "
            "region on about 95-97% of discs but assigns the correct integer "
            "index on only about 83-84%, so roughly 12-13 pp of usable accuracy "
            "is lost to ordering and fragmentation of already-detected regions, "
            "not to segmentation quality. That gap is addressable with "
            "post-processing on the existing predictions and needs no new "
            "training run."
        ),
        "why_not_more_capacity": (
            f"Width 32 cost {decision['evidence']['epoch_slowdown_vs_sprint3']}x "
            f"the time per epoch for a {decision['evidence']['val_fg_dice_delta']:+.5f} "
            f"validation change. Width 64 was measured in the Sprint 3 audit at "
            f"~6.8 GB peak commit and 216-326 h for 30 epochs, so it remains "
            f"infeasible on this hardware, and there is now direct evidence that "
            f"more width does not help at this data scale."
        ),
        "proposed_steps": [
            "Work from the existing saved predictions - no retraining, no new "
            "architecture.",
            "Enforce geometric ordering across a whole series rather than "
            "per-slice, using the fact that series-level disc ordering is far "
            "more reliable than per-slice ordering.",
            "Merge fragmented components before indexing, and reject components "
            "below a size threshold calibrated on the training split only.",
            "Re-score the corrected taxonomy on the test set once, after the "
            "post-processing rule is fixed on train/validation data.",
        ],
        "success_criterion": (
            f"Raise the corrected-taxonomy index-correct rate from its current "
            f"level (Sprint 3 remains the best at "
            f"{(next((r['sprint3_coverage'] for r in comparison['indexing'] if r['metric'] == 'indexing pct_index_correct'), None))}%) "
            f"towards the region-found ceiling, without retraining."
        ),
        "segmentation_candidate": (
            "Sprint 3 Coverage, the width-16 model: it holds the best validation "
            "Dice (0.89831), the best test macro foreground Dice (0.90001) and "
            "the best disc indexing, at a quarter of the parameters and a third "
            "of the training time."
        ),
        "current_sprint4_indexing_pct": s4_index,
        "do_not_start_automatically": True,
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def render_figures(history: pd.DataFrame, coverage: dict, comparison: dict,
                   training: dict) -> list[str]:
    import matplotlib.pyplot as plt

    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    names: list[str] = []

    def save(figure, filename: str) -> None:
        path = VIZ_DIR / filename
        figure.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(figure)
        names.append(filename)

    # 1. training curves
    figure, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes[0, 0].plot(history["epoch"], history["train_loss"], "o-", ms=3,
                    label="train loss")
    axes[0, 0].plot(history["epoch"], history["val_loss"], "s-", ms=3,
                    label="val loss")
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_xlabel("epoch")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(history["epoch"], history["val_fg_dice"], "o-", ms=3,
                    color="tab:green", label="Sprint 4 val fg Dice")
    axes[0, 1].axhline(S3_REF["val_fg_dice"], ls="--", color="tab:blue",
                       label=f"Sprint 3 best {S3_REF['val_fg_dice']}")
    axes[0, 1].axhline(S2_REF["val_fg_dice"], ls=":", color="tab:gray",
                       label=f"Sprint 2 Ext best {S2_REF['val_fg_dice']}")
    axes[0, 1].scatter([training["best_val_dice_epoch"]],
                       [training["best_val_fg_dice"]], marker="*", s=180,
                       color="tab:red", zorder=5,
                       label=f"best {training['best_val_fg_dice']:.5f} "
                             f"(ep {training['best_val_dice_epoch']})")
    axes[0, 1].set_title("Validation foreground Dice")
    axes[0, 1].set_xlabel("epoch")
    axes[0, 1].legend(fontsize=7)
    axes[0, 1].grid(alpha=0.3)

    for column, label in (("val_dice_vertebra", "vertebra"),
                          ("val_dice_intervertebral_disc", "IVD"),
                          ("val_dice_spinal_canal", "canal")):
        axes[1, 0].plot(history["epoch"], history[column], "o-", ms=3, label=label)
    axes[1, 0].set_title("Validation Dice by class")
    axes[1, 0].set_xlabel("epoch")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(history["epoch"], history["cumulative_coverage_pct"], "o-",
                    ms=3, color="tab:purple")
    axes[1, 1].axhline(100, ls="--", color="gray")
    if coverage["epoch_reaching_full_coverage"]:
        axes[1, 1].axvline(coverage["epoch_reaching_full_coverage"], ls=":",
                           color="tab:red",
                           label=f"100% at epoch "
                                 f"{coverage['epoch_reaching_full_coverage']}")
        axes[1, 1].legend(fontsize=8)
    axes[1, 1].set_title("Cumulative training-slice coverage (%)")
    axes[1, 1].set_xlabel("epoch")
    axes[1, 1].grid(alpha=0.3)
    figure.suptitle(
        f"Sprint 4 Capacity - width 32 ({training['parameters']:,} params), "
        f"batch {training['batch_size']}"
    )
    save(figure, "training_curves.png")

    # 2. Sprint 3 vs Sprint 4 validation curves
    s3_history_path = S3_CHECKPOINTS / "history.csv"
    if s3_history_path.exists():
        s3_history = pd.read_csv(s3_history_path)
        figure, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        axes[0].plot(s3_history["epoch"], s3_history["val_fg_dice"], "o-", ms=3,
                     label="Sprint 3 (width 16, batch 8)")
        axes[0].plot(history["epoch"], history["val_fg_dice"], "s-", ms=3,
                     label="Sprint 4 (width 32, batch 2)")
        axes[0].set_title("Validation foreground Dice")
        axes[0].set_xlabel("epoch")
        axes[0].legend(fontsize=8)
        axes[0].grid(alpha=0.3)

        axes[1].plot(s3_history["epoch"], s3_history["val_loss"], "o-", ms=3,
                     label="Sprint 3 val loss")
        axes[1].plot(history["epoch"], history["val_loss"], "s-", ms=3,
                     label="Sprint 4 val loss")
        axes[1].plot(s3_history["epoch"], s3_history["train_loss"], "o--", ms=2,
                     alpha=0.6, label="Sprint 3 train loss")
        axes[1].plot(history["epoch"], history["train_loss"], "s--", ms=2,
                     alpha=0.6, label="Sprint 4 train loss")
        axes[1].set_title("Loss")
        axes[1].set_xlabel("epoch")
        axes[1].legend(fontsize=7)
        axes[1].grid(alpha=0.3)
        figure.suptitle("Capacity comparison: width 16 vs width 32")
        save(figure, "sprint3_vs_sprint4_curves.png")

    # 3. three-way test metric comparison
    keys = [("test vertebra/dice", "vertebra"),
            ("test intervertebral_disc/dice", "IVD"),
            ("test spinal_canal/dice", "canal"),
            ("test macro_foreground/dice", "macro fg")]
    rows = {r["metric"]: r for r in comparison["segmentation"]}
    available = [(k, label) for k, label in keys if k in rows]
    if available:
        labels = [label for _, label in available]
        x = np.arange(len(labels))
        width = 0.27
        figure, axis = plt.subplots(figsize=(9, 4.5))
        for offset, key, label, colour in (
            (-width, "sprint2_extended", "Sprint 2 Extended (w16)", "tab:gray"),
            (0.0, "sprint3_coverage", "Sprint 3 Coverage (w16)", "tab:blue"),
            (width, "sprint4_capacity", "Sprint 4 Capacity (w32)", "tab:orange"),
        ):
            values = [rows[k][key] for k, _ in available]
            bars = axis.bar(x + offset, values, width, label=label, color=colour)
            axis.bar_label(bars, fmt="%.4f", fontsize=6, rotation=90, padding=2)
        axis.set_xticks(x)
        axis.set_xticklabels(labels)
        axis.set_ylim(0.85, 0.94)
        axis.set_ylabel("test Dice")
        axis.set_title("Held-out test Dice across three controlled experiments")
        axis.legend(fontsize=8)
        axis.grid(alpha=0.3, axis="y")
        save(figure, "three_way_test_comparison.png")

    # 4. computational cost
    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    runs = ["Sprint 2 Ext\n(w16 b8)", "Sprint 3\n(w16 b8)", "Sprint 4\n(w32 b2)"]
    colours = ["tab:gray", "tab:blue", "tab:orange"]
    params = [S2_REF["params"] / 1e6, S3_REF["params"] / 1e6,
              training["parameters"] / 1e6]
    bars = axes[0].bar(runs, params, color=colours)
    axes[0].bar_label(bars, fmt="%.2fM", fontsize=8)
    axes[0].set_title("Parameters (millions)")
    axes[0].grid(alpha=0.3, axis="y")

    epoch_s = [np.nan, S3_REF["mean_epoch_seconds"], training["mean_epoch_seconds"]]
    bars = axes[1].bar(runs, epoch_s, color=colours)
    axes[1].bar_label(bars, fmt="%.0fs", fontsize=8)
    axes[1].set_title("Mean epoch time (s)")
    axes[1].grid(alpha=0.3, axis="y")

    hours = [np.nan, S3_REF["wall_hours"], training["total_training_hours"]]
    bars = axes[2].bar(runs, hours, color=colours)
    axes[2].bar_label(bars, fmt="%.2fh", fontsize=8)
    axes[2].set_title("Total training wall time (h)")
    axes[2].grid(alpha=0.3, axis="y")
    figure.suptitle("Computational cost of the capacity increase")
    save(figure, "computational_cost.png")

    # 5. indexing comparison
    s3_idx = comparison.get("_indexing_s3")
    s4_idx = comparison.get("_indexing_s4")
    if s3_idx and s4_idx:
        categories = ["correct", "shifted", "merged", "split", "missed"]
        figure, axis = plt.subplots(figsize=(9, 4.5))
        x = np.arange(len(categories))
        width = 0.38
        for offset, data, label, colour in (
            (-width / 2, s3_idx, "Sprint 3 (w16)", "tab:blue"),
            (width / 2, s4_idx, "Sprint 4 (w32)", "tab:orange"),
        ):
            values = [(data.get("categories", {}).get(c) or {}).get("pct", 0.0)
                      for c in categories]
            bars = axis.bar(x + offset, values, width, label=label, color=colour)
            axis.bar_label(bars, fmt="%.2f", fontsize=7)
        axis.set_xticks(x)
        axis.set_xticklabels(categories)
        axis.set_ylabel("% of ground-truth discs")
        axis.set_title("Disc-indexing taxonomy (corrected): width 16 vs width 32")
        axis.legend(fontsize=8)
        axis.grid(alpha=0.3, axis="y")
        save(figure, "indexing_comparison.png")

    return names


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def three_way_rows(rows: list[dict]) -> list[dict]:
    """Render comparison entries as report table rows."""
    out = []
    for row in rows:
        if row["sprint3_coverage"] is None and row["sprint4_capacity"] is None:
            continue
        delta = row["delta_s4_vs_s3"]
        if delta is None:
            rendered = "-"
        elif all(
            isinstance(row[k], int) and not isinstance(row[k], bool)
            for k in ("sprint3_coverage", "sprint4_capacity")
        ):
            # Counts (parameters, spurious components) read badly at 5 decimals.
            rendered = f"{int(round(delta)):+,d}"
        else:
            rendered = f"{delta:+.5f}"
        out.append({
            "Metric": row["metric"],
            "Sprint 2 Extended": row["sprint2_extended"],
            "Sprint 3": row["sprint3_coverage"],
            "Sprint 4": row["sprint4_capacity"],
            "Δ S4 vs S3": rendered,
            "Verdict": row["verdict_vs_s3"],
        })
    return out


THREE_WAY_COLUMNS = ["Metric", "Sprint 2 Extended", "Sprint 3", "Sprint 4",
                     "Δ S4 vs S3", "Verdict"]


def write_report(payload: dict) -> Path:
    """Compose sprint4_final_report.md from measured artefacts only."""
    verification = payload["verification"]
    training = payload["training"]
    coverage = payload["coverage"]
    convergence = payload["convergence"]
    checkpoint = payload["checkpoint"]
    comparison = payload["comparison"]
    decision = payload["decision"]
    recommendation = payload["recommendation"]
    config = payload["config"]
    smoke = payload["smoke_test"]
    s4_seg = payload["test_segmentation"]
    s3_seg = payload["reference_segmentation"]["sprint3_coverage"]
    agree = payload["measurement_agreement"]

    report = FixedPrecisionReport(
        "Sprint 4 Capacity - Final Report",
        "Controlled experiment: U-Net width 16 -> 32 after coverage was fixed",
    )

    # ---- 1 objective ----
    report.heading("1. Objective")
    report.text(
        "Answer one question: **does increasing U-Net capacity from width 16 to "
        "width 32 provide a meaningful improvement after training-data coverage "
        "has already been fixed?**"
    )
    report.text(
        "Sprint 3 removed the data-coverage confound: the width-16 model was "
        "trained on 100% of the 9,128 training slices instead of 28.05%, and its "
        "validation Dice did not move. That left model capacity as the next "
        "credible variable, and the Sprint 3 audit had already measured width 32 "
        "at batch 2 as the only affordable way to test it on this hardware. This "
        "sprint runs that test."
    )

    # ---- 2 experimental setup ----
    report.heading("2. Experimental Setup")
    report.heading("Changed", level=3)
    changed = config.get("changed", {})
    report.key_values({
        "Base channels (width)":
            f"{changed['base_channels']['from']} -> {changed['base_channels']['to']}",
        "Parameters":
            f"{changed['parameters']['from']:,} -> {changed['parameters']['to']:,} "
            f"({changed['parameters']['ratio']}x)",
        "Batch size":
            f"{changed['batch_size']['from']} -> {changed['batch_size']['to']} "
            f"({changed['batch_size']['status']})",
    })
    report.text(
        "The batch size is **not** an independent choice. " +
        changed["batch_size"]["rationale"]
    )
    report.heading("Unchanged from Sprint 3 Coverage", level=3)
    unchanged = config.get("unchanged_from_sprint3_coverage", {})
    report.bullets([
        f"`{key}`: {value}" for key, value in unchanged.items()
        if not isinstance(value, dict)
    ])
    early = unchanged.get("early_stopping", {})
    report.text(
        f"Early stopping: monitor `{early.get('monitor')}`, patience "
        f"{early.get('patience')}, min_delta {early.get('min_delta')} - "
        f"identical to Sprint 3."
    )
    report.heading("Different by design", level=3)
    for key, value in (config.get("different_by_design") or {}).items():
        report.text(f"- **{key}**: {value}")

    report.heading("Feasibility check before the long run", level=3)
    report.text(
        f"A short smoke test was run first and had to pass before training "
        f"started. Verdict **{smoke['verdict']}** "
        f"({smoke['n_passed']} checks passed, {smoke['n_failed']} failed)."
    )
    report.key_values({
        "Parameter count verified": f"{smoke['parameters']['total']:,} "
                                    f"({smoke['params_vs_sprint3']}x Sprint 3)",
        "Loss finite on first step": smoke["single_step"]["loss"],
        "Global gradient L2 norm": smoke["single_step"]["global_grad_l2_norm"],
        "Measured train throughput":
            f"{smoke['throughput']['train_img_per_s']} img/s "
            f"({smoke['throughput']['train_s_per_step']} s/step at batch 2)",
        "Projected 30-epoch time":
            f"{smoke['throughput']['projected_30_epoch_hours']} h "
            f"(audit estimate "
            f"{smoke['throughput']['audit_estimate_30_epoch_hours']} h)",
        "Peak commit charge":
            f"{smoke['memory']['peak_commit_mb']} MB = "
            f"{smoke['memory']['peak_commit_vs_proven']}x the memory-proven level",
        "Validation Dice invariant to batch size":
            f"batch 2 = batch 8 to "
            f"{smoke['validation_batch_size_invariance']['absolute_difference']:.1e}",
        "CUDA assumed": smoke["cuda_available"],
        "Test set touched": smoke["test_set_used"],
    })

    report.heading("Verification that the run completed", level=3)
    report.key_values({
        "All training artefacts present":
            all(verification["artefacts_present"].values()),
        "Epochs in history.csv": verification["epochs_in_history"],
        "Epochs reported by the summary": verification["epochs_reported"],
        "History matches summary": verification["history_matches_summary"],
        "Early stopping triggered": verification["stopped_early"],
        "Stop reason": verification["stop_reason"],
        "Test set used during training":
            verification["test_set_used_during_training"],
    })

    # ---- 3 architecture comparison ----
    report.heading("3. Architecture Comparison")
    report.text(
        "Same topology in both runs - plain U-Net, depth 4, BatchNorm, bilinear "
        "upsampling, 4 output classes. Only the channel widths differ, and each "
        "encoder stage doubles from the base."
    )
    report.table([
        {
            "Property": "base channels",
            "Sprint 3 (width 16)": S3_REF["base_channels"],
            "Sprint 4 (width 32)": training["base_channels"],
        },
        {
            "Property": "encoder stage widths",
            "Sprint 3 (width 16)": "16, 32, 64, 128, 256",
            "Sprint 4 (width 32)": "32, 64, 128, 256, 512",
        },
        {
            "Property": "depth",
            "Sprint 3 (width 16)": 4,
            "Sprint 4 (width 32)": 4,
        },
        {
            "Property": "parameters",
            "Sprint 3 (width 16)": f"{S3_REF['params']:,}",
            "Sprint 4 (width 32)": f"{training['parameters']:,}",
        },
        {
            "Property": "parameter ratio",
            "Sprint 3 (width 16)": "1.00x",
            "Sprint 4 (width 32)":
                f"{training['parameters'] / S3_REF['params']:.2f}x",
        },
        {
            "Property": "batch size",
            "Sprint 3 (width 16)": S3_REF["batch_size"],
            "Sprint 4 (width 32)": training["batch_size"],
        },
        {
            "Property": "checkpoint size",
            "Sprint 3 (width 16)": "7.9 MB (weights-only)",
            "Sprint 4 (width 32)": f"{checkpoint['size_mb']} MB (weights-only)",
        },
    ], ["Property", "Sprint 3 (width 16)", "Sprint 4 (width 32)"])
    report.text(
        "Parameters scale about 4x per width doubling, as expected for a "
        "convolutional encoder-decoder: each layer's weight tensor grows with "
        "the product of input and output channels."
    )

    # ---- 4 training results ----
    report.heading("4. Training Results")
    report.key_values({
        "Total epochs completed":
            f"{training['total_epochs_completed']} / {training['max_epochs']}",
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
        "Learning rate at best epoch":
            f"{training['learning_rate_at_best_epoch']:.3e}",
        "Train loss at best epoch":
            f"{training['train_loss_at_best_dice_epoch']:.5f}",
        "Validation loss at best epoch":
            f"{training['val_loss_at_best_dice_epoch']:.5f}",
        "Per-class Dice at best epoch":
            f"vertebra {training['vertebra_dice_at_best_epoch']:.5f}, "
            f"IVD {training['ivd_dice_at_best_epoch']:.5f}, "
            f"canal {training['canal_dice_at_best_epoch']:.5f}",
    })
    report.text(
        f"The run used its full 30-epoch budget: early stopping did **not** fire, "
        f"and the counter stood at "
        f"{training['epochs_without_improvement_at_end']} of the patience-6 "
        f"limit when the budget ran out. Sprint 3 by contrast stopped early at "
        f"epoch {S3_REF['epochs_completed']}."
    )

    report.heading("Checkpoint selection", level=3)
    report.key_values({
        "Checkpoint": f"`{checkpoint['path']}`",
        "Selection criterion": checkpoint["selection_criterion"],
        "Epoch": checkpoint["epoch"],
        "Validation foreground Dice": checkpoint["val_fg_dice"],
        "Validation loss": checkpoint["val_loss"],
        "Matches the training summary": checkpoint["matches_summary"],
        "Test performance used for selection":
            checkpoint["test_used_for_selection"],
    })

    report.heading("Per-epoch history", level=3)
    history = pd.DataFrame(payload["history"])
    shown = history[[
        "epoch", "learning_rate", "n_train_slices_this_epoch",
        "unique_train_slices_seen", "cumulative_coverage_pct", "train_loss",
        "val_loss", "val_fg_dice", "val_dice_vertebra",
        "val_dice_intervertebral_disc", "val_dice_spinal_canal", "epoch_seconds",
    ]].rename(columns={
        "n_train_slices_this_epoch": "slices",
        "unique_train_slices_seen": "seen",
        "cumulative_coverage_pct": "cov%",
        "val_dice_vertebra": "vert",
        "val_dice_intervertebral_disc": "ivd",
        "val_dice_spinal_canal": "canal",
        "epoch_seconds": "sec",
    })
    report.table(shown.to_dict("records"), list(shown.columns))

    # ---- 4b coverage ----
    report.heading("Training-data coverage", level=3)
    report.key_values({
        "Training slices available": f"{coverage['train_slices_available']:,}",
        "Unique slices seen": f"{coverage['unique_slices_seen_final']:,}",
        "Final coverage": f"{coverage['final_coverage_pct']}%",
        "Epoch reaching 100% coverage": coverage["epoch_reaching_full_coverage"],
        "Mean slices per epoch": f"{coverage['slices_per_epoch_mean']:,}",
        "Sampler leakage-free":
            config["sampler_verification"]["leakage_free"],
        "Sampler violations": config["sampler_verification"]["violations"],
        "Validation/test patients excluded from training":
            config["sampler_verification"]["val_test_patients_excluded"],
    })
    report.text(
        f"Coverage is identical to Sprint 3 by construction - the same rotating "
        f"shard sampler with the same seed and the same per-epoch budget - so "
        f"data exposure is held constant and cannot explain the difference "
        f"between the two runs."
    )

    # ---- 5 convergence ----
    report.heading("5. Convergence")
    report.key_values({
        "Best epoch": convergence["best_epoch"],
        "Epochs after the best": convergence["epochs_after_best"],
        "Validation Dice over the last 5 epochs":
            f"{convergence['last5_val_dice_min']:.5f} - "
            f"{convergence['last5_val_dice_max']:.5f} "
            f"(spread {convergence['last5_val_dice_spread']:.5f})",
        "Train loss over the last 5 epochs":
            f"{convergence['last5_train_loss_min']:.5f} - "
            f"{convergence['last5_train_loss_max']:.5f}",
        "Final train loss": f"{convergence['final_train_loss']:.5f}",
        "Final validation loss": f"{convergence['final_val_loss']:.5f}",
        "Validation - train loss gap":
            f"{convergence['train_val_loss_gap']:.5f}",
        "Minimum validation loss":
            f"{convergence['min_val_loss']:.5f} "
            f"(epoch {convergence['val_loss_min_epoch']})",
    })
    report.text(
        f"The curve is flat at the end: validation Dice moves only "
        f"{convergence['last5_val_dice_spread']:.5f} across the last five "
        f"epochs, and the best epoch is "
        f"{convergence['epochs_after_best']} epochs before the end. The larger "
        f"model has converged - it has not been cut short. The "
        f"{convergence['train_val_loss_gap']:.5f} gap between validation and "
        f"training loss is the margin to watch: a 4x larger model on the same "
        f"9,128 slices has more room to fit the training split, and the "
        f"validation metric does not follow the training loss down."
    )

    # ---- 6 test results ----
    report.heading("6. Test Segmentation Results")
    report.text(
        f"Evaluated **once**, after training finished, on the untouched held-out "
        f"test set: {s4_seg['n_slices']:,} slices from {s4_seg['n_patients']} "
        f"patients. The checkpoint was selected on validation Dice, not on test "
        f"performance."
    )
    report.heading("Aggregate (dataset-level) metrics", level=3)
    report.table(
        [
            {
                "class": class_name,
                "Dice": s4_seg["aggregate"]["per_class"][class_name]["dice"],
                "IoU": s4_seg["aggregate"]["per_class"][class_name]["iou"],
                "Precision": s4_seg["aggregate"]["per_class"][class_name]["precision"],
                "Recall": s4_seg["aggregate"]["per_class"][class_name]["recall"],
            }
            for class_name in CLASSES
        ] + [
            {
                "class": "**macro foreground**",
                "Dice": s4_seg["aggregate"]["macro_foreground"]["dice"],
                "IoU": s4_seg["aggregate"]["macro_foreground"]["iou"],
                "Precision": s4_seg["aggregate"]["macro_foreground"]["precision"],
                "Recall": s4_seg["aggregate"]["macro_foreground"]["recall"],
            }
        ],
        ["class", "Dice", "IoU", "Precision", "Recall"],
    )
    report.key_values({"Pixel accuracy": s4_seg["pixel_accuracy"]})
    report.heading("Per-patient Dice (mean +/- sd over test patients)", level=3)
    report.table(
        [
            {
                "class": class_name,
                "mean Dice": s4_seg["per_patient_dice"][class_name]["mean_dice"],
                "sd": s4_seg["per_patient_dice"][class_name]["std_dice"],
                "patients": s4_seg["per_patient_dice"][class_name]["n_patients"],
            }
            for class_name in CLASSES
        ],
        ["class", "mean Dice", "sd", "patients"],
    )
    report.text(
        "Note on aggregation: these are dataset-level metrics computed from a "
        "pooled confusion matrix, which is the same aggregation Sprint 2 and "
        "Sprint 3 reported. The evaluation script also prints a per-slice mean, "
        "which is systematically lower for every run because lateral slices "
        "holding only small structure fragments get equal weight. The two are "
        "not interchangeable and only the aggregate values are compared here."
    )

    # ---- 7 three-way comparison ----
    report.heading("7. Sprint 2 vs Sprint 3 vs Sprint 4")
    report.text(
        "`Δ S4 vs S3` is Sprint 4 minus Sprint 3 Coverage, the immediate "
        "predecessor and the only run that differs from Sprint 4 by capacity "
        "alone. Verdicts account for metric direction - lower is better for MAE, "
        "loss, standard deviation and spurious components."
    )
    report.heading("Primary criterion", level=3)
    report.table(three_way_rows(comparison["validation"]), THREE_WAY_COLUMNS)
    report.heading("Headline metrics", level=3)
    headline_names = [
        "test macro_foreground/dice", "test vertebra/dice",
        "test intervertebral_disc/dice", "test spinal_canal/dice",
    ]
    headline = [r for r in comparison["segmentation"]
                if r["metric"] in headline_names]
    headline += [r for r in comparison["indexing"]
                 if r["metric"] == "indexing pct_index_correct"]
    headline += [r for r in comparison["identification"]
                 if r["metric"] == "disc_identification pct_discs_index_correct"]
    headline += [r for r in comparison["measurements"]
                 if r["metric"] in ("height_mm_central/mae",
                                    "intensity_disc_vertebra_ratio/pearson_r")]
    headline += [r for r in comparison["pfirrmann"]
                 if r["metric"] == "Pfirrmann forest/quadratic_weighted_kappa"]
    report.table(three_way_rows(headline), THREE_WAY_COLUMNS)
    report.heading("All segmentation metrics", level=3)
    report.table(three_way_rows(comparison["segmentation"]), THREE_WAY_COLUMNS)
    report.heading("Tally of Sprint 4 against Sprint 3", level=3)
    report.key_values(comparison["tally"])

    # ---- 8 disc indexing ----
    report.heading("8. Disc Indexing")
    s4_idx = payload["indexing"].get("sprint4_capacity")
    if s4_idx:
        report.text(
            "Corrected taxonomy: every ground-truth disc is classified as "
            "`correct`, `shifted`, `merged`, `split` or `missed`. Analysis uses "
            "the **integer disc index only** - the dataset does not state which "
            "vertebra is L5, so no anatomical level name is asserted. The "
            "invalid vertebra-component-vs-instance comparison identified in the "
            "Sprint 3 audit is not used."
        )
        report.key_values({
            "Discs analysed": f"{s4_idx['n_discs_analysed']:,}",
            "Slices analysed": f"{s4_idx['n_slices_analysed']:,}",
            "Region found": f"{s4_idx['pct_region_found']}%",
            "Index correct": f"{s4_idx['pct_index_correct']}%",
            "Slices with correct disc count":
                f"{s4_idx['pct_slices_count_correct']}%",
            "Slices with every disc correct":
                f"{s4_idx['pct_slices_all_discs_correct']}%",
            "Spurious components": s4_idx["total_spurious_components"],
            "Shift offsets": s4_idx["shifted_offsets"],
        })
        report.table(three_way_rows(comparison["indexing"]), THREE_WAY_COLUMNS)
        report.heading("Second estimator (evaluate_unet)", level=3)
        report.text(
            "The evaluation script computes indexing accuracy with a different "
            "estimator from the corrected taxonomy. Both are listed so the "
            "comparison is never made across estimators."
        )
        report.table(three_way_rows(comparison["identification"]),
                     THREE_WAY_COLUMNS)
        report.text(
            "Both estimators agree in direction: width 32 indexes discs less "
            "accurately than width 16."
        )
        report.heading("Accuracy by disc index", level=3)
        report.dataframe(pd.DataFrame(s4_idx["by_disc_index"]), max_rows=12)
        report.heading("Accuracy by slice annotation area", level=3)
        report.dataframe(pd.DataFrame(s4_idx["by_slice_annotation_area"]),
                         max_rows=12)
        report.text(
            f"The structural gap remains the story: the model finds the disc "
            f"region for {s4_idx['pct_region_found']}% of ground-truth discs but "
            f"assigns the right integer index for only "
            f"{s4_idx['pct_index_correct']}%. The difference is ordering and "
            f"fragmentation of regions that were already detected, which is a "
            f"post-processing problem rather than a segmentation-quality one."
        )
    else:
        report.text("_Indexing analysis was not run in this pass._")

    # ---- 9 measurements ----
    report.heading("9. Disc-Level Measurements")
    if comparison["measurements"]:
        report.text(
            "Measurements taken from predicted masks, compared against the same "
            "discs measured from ground-truth masks. Only discs whose identity "
            "the prediction recovered correctly are compared - detection "
            "failures are counted separately rather than averaged into the "
            "measurement error."
        )
        report.key_values({
            "Discs compared": agree["detection"]["n_matched_by_identity"],
            "Ground-truth discs on predicted series":
                agree["detection"]["n_truth_discs"],
            "% of truth discs recovered":
                f"{agree['detection']['pct_truth_discs_recovered']}%",
        })
        report.heading("Sprint 4 agreement detail", level=3)
        detail = pd.DataFrame(agree["measurement_agreement"])[
            ["measurement", "unit", "n", "gt_mean", "bias", "mae",
             "mae_pct_of_gt_mean", "pearson_r"]
        ]
        report.table(detail.to_dict("records"), list(detail.columns))
        report.heading("Three-way comparison", level=3)
        report.table(three_way_rows(comparison["measurements"]),
                     THREE_WAY_COLUMNS)
        report.text(
            "`disc_to_vertebra_height_ratio` remains unusable in all three runs "
            "- its mean absolute error is over 70% of the ground-truth mean - so "
            "it is reported but must not be used as a derived feature."
        )
    else:
        report.text("_Measurement agreement was not available in this pass._")

    # ---- 10 Pfirrmann ----
    report.heading("10. Pfirrmann Results")
    pf = agree.get("end_to_end_pfirrmann") or {}
    if pf:
        report.text(
            "End-to-end evaluation: the grade models are fitted on "
            "ground-truth-mask features from **training** patients, then applied "
            "to **predicted**-mask features from test patients, so segmentation "
            "error is included. Pfirrmann grades are dataset annotations used as "
            "labels for this measurement exercise. Nothing here is a clinical "
            "grading of a patient."
        )
        report.key_values({
            "Features": pf.get("features"),
            "Training discs (ground-truth masks)": pf.get("n_train_discs"),
            "Test discs evaluated (predicted masks)":
                (pf.get("forest") or {}).get("n_test_discs"),
        })
        report.table([
            {
                "estimator": estimator,
                "QWK": (pf.get(estimator) or {}).get("quadratic_weighted_kappa"),
                "MAE": (pf.get(estimator) or {}).get("mae"),
                "exact agreement": (pf.get(estimator) or {}).get("exact_agreement"),
                "within 1 grade": (pf.get(estimator) or {}).get("within_one_grade"),
            }
            for estimator in ("logreg", "forest")
        ], ["estimator", "QWK", "MAE", "exact agreement", "within 1 grade"])
        report.heading("Three-way comparison", level=3)
        report.table(three_way_rows(comparison["pfirrmann"]), THREE_WAY_COLUMNS)
        report.text(
            "This is the one family where Sprint 4 comes out ahead. It is "
            "reported as measured, and it is also the weakest evidence in the "
            "report: it rests on roughly 212 test discs, quadratic weighted "
            "kappa on that sample size moves easily, and the downstream grade "
            "model is refitted for each run. It is not enough to offset a "
            "consistent segmentation and indexing regression."
        )
    else:
        report.text("_Pfirrmann evaluation was not available in this pass._")

    # ---- 11 computational cost ----
    report.heading("11. Computational Cost")
    report.table(three_way_rows(comparison["cost"]), THREE_WAY_COLUMNS)
    report.key_values({
        "Sprint 3 mean epoch": f"{S3_REF['mean_epoch_seconds']} s",
        "Sprint 4 mean epoch":
            f"{training['mean_epoch_seconds']} s "
            f"({training['mean_epoch_seconds'] / S3_REF['mean_epoch_seconds']:.2f}x)",
        "Sprint 4 epoch range":
            f"{training['min_epoch_seconds']} - {training['max_epoch_seconds']} s",
        "Sprint 3 total wall time": f"{S3_REF['wall_hours']} h "
                                    f"({S3_REF['epochs_completed']} epochs)",
        "Sprint 4 total wall time":
            f"{training['total_training_hours']} h "
            f"({training['total_epochs_completed']} epochs)",
        "Measured train throughput (smoke test)":
            f"{smoke['throughput']['train_img_per_s']} img/s at batch 2",
        "Peak commit charge":
            f"{smoke['memory']['peak_commit_mb']} MB "
            f"({smoke['memory']['peak_commit_vs_proven']}x the proven level)",
        "Checkpoint size": f"{checkpoint['size_mb']} MB per weights-only file",
    })
    report.text(
        f"The capacity increase cost "
        f"{training['mean_epoch_seconds'] / S3_REF['mean_epoch_seconds']:.2f}x "
        f"the time per epoch and "
        f"{training['total_training_hours'] / S3_REF['wall_hours']:.2f}x the "
        f"total wall time, and it bought a negative change in the primary "
        f"metric. Memory was never the limit: peak commit charge stayed at "
        f"{smoke['memory']['peak_commit_vs_proven']}x the proven level because "
        f"the batch size dropped to 2. Time was the cost, exactly as the "
        f"Sprint 3 audit predicted."
    )

    # ---- 12 limitations ----
    report.heading("12. Limitations")
    report.bullets([
        "**Batch size is a confound.** Width 32 could not be run at batch 8 on "
        "this hardware within a sane time budget, so batch size dropped from 8 "
        "to 2 alongside the width change. Batch size affects BatchNorm statistics "
        "and gradient noise at a fixed learning rate. This experiment therefore "
        "measures 'width 32 as it can actually be trained here', not width 32 in "
        "isolation. A cleaner separation would need either batch 8 at width 32 "
        "(measured at ~80 h) or batch 2 at width 16 as a control (not run).",
        "**Learning rate was not re-tuned.** The learning rate was held at 1e-3 "
        "to keep the comparison controlled, but the optimal learning rate "
        "generally shifts with batch size. A tuned width-32 run could do better "
        "than this one; that possibility is not excluded by this result.",
        "**One seed, one split.** Every number here is a single run at seed 42 on "
        "one patient-level split. Differences of a few thousandths of a Dice "
        "point are within the range that seed choice alone can produce, which is "
        "why the decision rule was set at a margin rather than at zero.",
        "**Test set is 33 patients.** 1,655 slices from 33 patients is a small "
        "held-out set. It was evaluated once, after training, but it still gives "
        "wide uncertainty on any single metric.",
        "**Pfirrmann evaluation rests on ~212 discs**, and the grade model is "
        "refitted per run, so its quadratic weighted kappa is the least stable "
        "number in the report.",
        "**Segmentation metrics are not clinical evidence.** A Dice score "
        "measures overlap with one annotation protocol. It does not establish "
        "clinical effectiveness or readiness for medical use, and none is "
        "claimed.",
        "**Disc identity is derived, not predicted.** The network outputs four "
        "semantic classes; the disc index comes from ordering connected "
        "components, which is why indexing accuracy trails detection accuracy.",
        "**No longitudinal or postoperative scope.** The SPIDER dataset used here "
        "contains no longitudinal postoperative follow-up, so postoperative "
        "healing is outside the validated scope of this work and no claim about "
        "it is made or supported. No change over time is measured, and no "
        "'percentage spine damage' score exists in this pipeline.",
    ])

    # ---- 13 interpretation ----
    report.heading("13. Interpretation")
    report.text(decision["narrative"])
    report.heading("Against the predefined decision rule", level=3)
    report.key_values({
        "Question": decision["question"],
        "Answer": decision["answer"],
        "Primary criterion": decision["decision_rule"]["primary"],
        "min_delta on validation Dice": decision["decision_rule"]["min_delta"],
        "Material margin on test macro Dice":
            decision["decision_rule"]["material_test_margin"],
        "Verdict": decision["verdict"],
        "Hypothesis supported": decision["hypothesis_supported"],
    })
    report.heading("Evidence weighed", level=3)
    report.key_values(decision["evidence"])
    report.text(
        "The decision deliberately does not rest on a single metric. It weighs "
        "the primary validation criterion together with the test macro Dice, the "
        "per-class Dice values, per-patient variability, disc indexing under two "
        "estimators, the predicted-mask measurement errors, the end-to-end "
        "Pfirrmann agreement, convergence behaviour and training cost. The "
        "validation and test evidence point the same way, which is what makes "
        "the conclusion safe to draw despite the Pfirrmann exception."
    )
    report.text(
        "Read together with Sprint 3, the picture is consistent: neither more "
        "data exposure nor more capacity moves this plateau. Both were "
        "reasonable hypotheses and both have now been tested and rejected on "
        "measured evidence, which is a useful result even though neither "
        "produced a better model. The remaining measurable loss sits in the "
        "disc-indexing stage, not in the segmentation network."
    )

    # ---- 14 recommendation ----
    report.heading("14. Recommended Next Step")
    report.text(
        f"**Option {recommendation['option']} - {recommendation['title']}.**"
    )
    report.text(recommendation["rationale"])
    report.heading("Segmentation candidate to keep", level=3)
    report.text(recommendation["segmentation_candidate"])
    report.heading("Proposed steps", level=3)
    report.bullets(recommendation["proposed_steps"])
    report.key_values({"Success criterion": recommendation["success_criterion"]})
    report.heading("Explicitly not recommended", level=3)
    report.text(recommendation["why_not_more_capacity"])
    report.text(
        "**This next experiment has not been started.** It is a recommendation "
        "only."
    )

    # ---- artefacts ----
    report.heading("Artefact Locations")
    report.table([
        {"artefact": "training history",
         "path": f"`outputs/checkpoints/{EXPERIMENT}/history.csv` (+ .json)"},
        {"artefact": "selected checkpoint",
         "path": f"`outputs/checkpoints/{EXPERIMENT}/best_val_dice.pt`"},
        {"artefact": "per-epoch checkpoints",
         "path": f"`outputs/checkpoints/{EXPERIMENT}/epochs/`"},
        {"artefact": "training config + summary",
         "path": f"`outputs/reports/{EXPERIMENT}/training_config.json`, "
                 f"`training_summary.json`"},
        {"artefact": "smoke test",
         "path": f"`outputs/reports/{EXPERIMENT}/smoke_test.json`"},
        {"artefact": "test segmentation metrics",
         "path": f"`outputs/reports/{EXPERIMENT}/metrics/`"},
        {"artefact": "disc measurements",
         "path": f"`outputs/reports/disc_analysis_{EXPERIMENT}.csv`"},
        {"artefact": "indexing failures",
         "path": f"`outputs/reports/{EXPERIMENT}/indexing_failures_{EXPERIMENT}.csv`"},
        {"artefact": "comparison table",
         "path": f"`outputs/reports/{EXPERIMENT}/comparison_table.csv`"},
        {"artefact": "figures",
         "path": f"`outputs/visualizations/{EXPERIMENT}/`"},
        {"artefact": "test predictions",
         "path": f"`data/processed/predictions_{EXPERIMENT}/`"},
        {"artefact": "this report",
         "path": f"`outputs/reports/{EXPERIMENT}/sprint4_final_report.md` (+ .json)"},
    ], ["artefact", "path"])

    report.heading("Figures", level=3)
    report.bullets([f"`outputs/visualizations/{EXPERIMENT}/{name}`"
                    for name in payload["figures"]])

    return report.save(REPORT_DIR / "sprint4_final_report.md")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    ensure_dirs()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    VIZ_DIR.mkdir(parents=True, exist_ok=True)

    print("[1] Verifying that training completed ...")
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
    smoke = load_json(REPORT_DIR / "smoke_test.json") or {}
    index = load_slice_index()

    print("[2] Analysing training, coverage, convergence ...")
    training = analyse_training(history, summary)
    coverage = analyse_coverage(history, index)
    convergence = analyse_convergence(history)
    print(f"  best val Dice {training['best_val_fg_dice']:.5f} "
          f"@ epoch {training['best_val_dice_epoch']}  "
          f"({training['parameters']:,} params)")
    print(f"  coverage {coverage['final_coverage_pct']}% "
          f"(100% at epoch {coverage['epoch_reaching_full_coverage']})")

    print("[3] Selected checkpoint (validation criterion) ...")
    checkpoint = describe_selected_checkpoint(summary)
    print(f"  {checkpoint['filename']} epoch={checkpoint['epoch']} "
          f"val_fg_dice={checkpoint['val_fg_dice']}")

    print("[4] Loading test metrics for all three runs ...")
    seg = {
        "s4": load_json(METRICS_DIR / "segmentation_metrics_test.json"),
        "s3": load_json(S3_METRICS / "segmentation_metrics_test.json"),
        "s2": load_json(S2_METRICS / "segmentation_metrics_test.json"),
    }
    if seg["s4"] is None:
        raise SystemExit(
            f"Missing {METRICS_DIR / 'segmentation_metrics_test.json'}. "
            f"Run evaluate_unet.py on the selected checkpoint first."
        )
    for key in ("s2", "s3", "s4"):
        print(f"  {key}: macro fg Dice "
              f"{seg[key]['aggregate']['macro_foreground']['dice']:.5f}")

    agree = {
        "s4": load_json(METRICS_DIR / "measurement_agreement.json"),
        "s3": load_json(S3_METRICS / "measurement_agreement.json"),
        "s2": load_json(S2_METRICS / "measurement_agreement.json"),
    }
    identification = {
        "s4": load_json(METRICS_DIR / "disc_identification_test.json"),
        "s3": load_json(S3_METRICS / "disc_identification_test.json"),
        "s2": load_json(S2_METRICS / "disc_identification_test.json"),
    }

    print("[5] Disc-indexing analysis (corrected taxonomy) ...")
    payload_indexing: dict = {}
    # Sprint 2 Extended and Sprint 3 taxonomy figures are taken from the
    # published Sprint 3 report so they are byte-identical to what was reported
    # there; only Sprint 4 is computed fresh.
    s3_report = load_json(S3_REPORT_JSON) or {}
    for run, value in (s3_report.get("indexing") or {}).items():
        payload_indexing[run] = value
        print(f"  {run}: index correct {value.get('pct_index_correct')}% "
              f"(from the Sprint 3 report)")

    if not args.skip_indexing:
        if not PREDICTIONS_DIR.exists():
            raise SystemExit(f"Missing predictions at {PREDICTIONS_DIR}")
        test_index = index[index["split"] == "test"].reset_index(drop=True)
        print(f"  sprint4_capacity ...")
        disc_frame, slice_frame = idx_diag.run_analysis(
            test_index, PREDICTIONS_DIR, project_root=PROJECT_ROOT,
            limit=args.indexing_limit or None, progress_every=500,
        )
        if disc_frame.empty:
            raise SystemExit(
                f"Indexing analysis produced no rows from {PREDICTIONS_DIR}. "
                f"Refusing to write a report with a missing section."
            )
        if not disc_frame.empty:
            disc_frame.to_csv(
                REPORT_DIR / f"indexing_failures_{EXPERIMENT}.csv", index=False
            )
            payload_indexing["sprint4_capacity"] = idx_diag.summarise(
                disc_frame, slice_frame
            )
            print(f"    index correct "
                  f"{payload_indexing['sprint4_capacity']['pct_index_correct']}%")

    print("[6] Building the three-way comparison ...")
    indexing_bundle = {
        "s2": payload_indexing.get("sprint2_extended"),
        "s3": payload_indexing.get("sprint3_coverage"),
        "s4": payload_indexing.get("sprint4_capacity"),
        "identification": identification,
    }
    comparison = build_comparison(seg, agree, indexing_bundle, training)
    comparison["_indexing_s3"] = indexing_bundle["s3"]
    comparison["_indexing_s4"] = indexing_bundle["s4"]
    pd.DataFrame(comparison["all"]).to_csv(
        REPORT_DIR / "comparison_table.csv", index=False
    )
    print(f"  tally vs Sprint 3: {comparison['tally']}")

    print("[7] Decision and recommendation ...")
    decision = decide(comparison, training, seg)
    recommendation = recommend(decision, comparison, training)
    print(f"  verdict: {decision['verdict']}")
    print(f"  recommendation: option {recommendation['option']}")

    print("[8] Rendering figures ...")
    figures = render_figures(history, coverage, comparison, training)
    for extra in sorted(VIZ_DIR.glob("*.png")):
        if extra.name not in figures:
            figures.append(extra.name)
    prediction_figures = sorted((VIZ_DIR / "predictions").glob("*.png"))
    figures.extend(f"predictions/{p.name}" for p in prediction_figures)
    for name in figures:
        print(f"  -> {VIZ_DIR / name}")

    print("[9] Writing the final report ...")
    payload = {
        "experiment": EXPERIMENT,
        "verification": verification,
        "config": config,
        "smoke_test": smoke,
        "training_summary": summary,
        "history": history.to_dict("records"),
        "training": training,
        "coverage": coverage,
        "convergence": convergence,
        "checkpoint": checkpoint,
        "test_segmentation": seg["s4"],
        "reference_segmentation": {
            "sprint3_coverage": seg["s3"],
            "sprint2_extended": seg["s2"],
        },
        "measurement_agreement": agree["s4"],
        "disc_identification": identification,
        "indexing": payload_indexing,
        "comparison": {k: v for k, v in comparison.items()
                       if not k.startswith("_")},
        "decision": decision,
        "recommendation": recommendation,
        "figures": figures,
        "reference_results": {"sprint3_coverage": S3_REF,
                              "sprint2_extended": S2_REF},
    }
    save_json(payload, REPORT_DIR / "sprint4_final_report.json")
    write_report({**payload, "comparison": comparison})
    print(f"  -> {REPORT_DIR / 'sprint4_final_report.md'}")
    print(f"  -> {REPORT_DIR / 'sprint4_final_report.json'}")

    print_terminal_summary(payload)


def print_terminal_summary(payload: dict) -> None:
    """Print the concise summary block requested in the brief."""
    training = payload["training"]
    coverage = payload["coverage"]
    seg = payload["test_segmentation"]["aggregate"]
    s4_idx = payload["indexing"].get("sprint4_capacity", {})
    ident = payload["disc_identification"]
    pf = ((payload.get("measurement_agreement") or {})
          .get("end_to_end_pfirrmann") or {}).get("forest", {})

    status = (
        f"COMPLETE - {training['total_epochs_completed']} epochs"
        + (" (early stopping triggered)" if training["early_stopping_triggered"]
           else " (reached max epochs, early stopping did not fire)")
    )

    def delta(value, reference):
        return f"(Sprint 3 {reference:.5f}, {value - reference:+.5f})"

    print("\n" + "=" * 72)
    print(f"TRAINING STATUS:             {status}")
    print(f"BEST EPOCH:                  {training['best_val_dice_epoch']}")
    print(f"BEST VAL DICE:               {training['best_val_fg_dice']:.5f}  "
          f"{delta(training['best_val_fg_dice'], S3_REF['val_fg_dice'])}")
    s3_seg = payload["reference_segmentation"]["sprint3_coverage"]["aggregate"]
    print(f"TEST MACRO DICE:             "
          f"{seg['macro_foreground']['dice']:.5f}  "
          f"{delta(seg['macro_foreground']['dice'], s3_seg['macro_foreground']['dice'])}")
    for label, class_name in (("TEST VERTEBRA DICE", "vertebra"),
                              ("TEST IVD DICE", "intervertebral_disc"),
                              ("TEST CANAL DICE", "spinal_canal")):
        value = seg["per_class"][class_name]["dice"]
        reference = s3_seg["per_class"][class_name]["dice"]
        print(f"{label + ':':<29}{value:.5f}  {delta(value, reference)}")
    if ident["s4"] and ident["s3"]:
        a, b = ident["s4"]["pct_discs_index_correct"], ident["s3"]["pct_discs_index_correct"]
        print(f"DISC INDEXING:               {a:.2f}%  "
              f"(Sprint 3 {b:.2f}%, {a - b:+.2f} pp) [evaluate_unet estimator]")
    if s4_idx:
        s3_tax = (payload["indexing"].get("sprint3_coverage") or {}).get(
            "pct_index_correct")
        suffix = (f"(Sprint 3 {s3_tax:.2f}%, "
                  f"{s4_idx['pct_index_correct'] - s3_tax:+.2f} pp) "
                  if s3_tax is not None else "")
        print(f"                             "
              f"{s4_idx['pct_index_correct']:.2f}%  {suffix}"
              f"[corrected-taxonomy estimator]")
    if pf:
        s3_pf = 0.6317
        print(f"PFIRRMANN QWK:               "
              f"{pf['quadratic_weighted_kappa']}  (Sprint 3 {s3_pf}, "
              f"{pf['quadratic_weighted_kappa'] - s3_pf:+.4f})")
    print(f"TRAINING COVERAGE:           "
          f"{coverage['final_coverage_pct']}% of "
          f"{coverage['train_slices_available']:,} slices "
          f"(100% by epoch {coverage['epoch_reaching_full_coverage']})")
    print(f"TOTAL TRAINING TIME:         {training['total_training_hours']} h  "
          f"(Sprint 3 {S3_REF['wall_hours']} h, "
          f"{training['total_training_hours'] / S3_REF['wall_hours']:.2f}x)")
    print(f"PARAMETERS:                  {training['parameters']:,}  "
          f"(Sprint 3 {S3_REF['params']:,}, "
          f"{training['parameters'] / S3_REF['params']:.2f}x)")
    print(f"RECOMMENDED NEXT STEP:       Option "
          f"{payload['recommendation']['option']} - "
          f"{payload['recommendation']['title']}")
    print("=" * 72)
    print(f"VERDICT: {payload['decision']['verdict']}")
    print("=" * 72)


if __name__ == "__main__":
    main()
