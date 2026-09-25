"""Sprint 2 - how much does segmentation error change the disc measurements?

Stages C-F build the feature table from **ground-truth** masks, which isolates
the finding-assessment step from segmentation error. This script closes the loop
by measuring the same discs from **predicted** masks and comparing, so the
end-to-end error is quantified rather than left implicit.

Requires:
    outputs/reports/disc_analysis.csv            (ground truth)
    outputs/reports/disc_analysis_predicted.csv  (predictions; produced by
        evaluate_unet.py --save-predictions then
        06_disc_analysis.py --mask-source prediction)

Writes:
    outputs/metrics/measurement_agreement.json
    outputs/metrics/measurement_agreement.csv
    outputs/visualizations/measurement_agreement.png

Usage
-----
    python scripts/09_compare_gt_vs_predicted.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.paths import (  # noqa: E402
    ANALYSIS_REPORTS_DIR,
    OUTPUTS_DIR,
    VISUALIZATIONS_DIR,
    ensure_dirs,
)
from src.utils.reporting import save_json  # noqa: E402

METRICS_DIR = OUTPUTS_DIR / "metrics"

#: Measurements compared, with the unit used for the error figures.
COMPARED = [
    ("height_mm_central", "mm"),
    ("height_mm_anterior", "mm"),
    ("height_mm_posterior", "mm"),
    ("area_mm2", "mm2"),
    ("ap_extent_mm", "mm"),
    ("height_ratio_to_series_median", "ratio"),
    ("disc_to_vertebra_height_ratio", "ratio"),
    ("canal_width_at_disc_mm", "mm"),
    ("intensity_disc_vertebra_ratio", "ratio"),
]


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predicted", type=Path,
                        default=ANALYSIS_REPORTS_DIR / "disc_analysis_predicted.csv",
                        help="Predicted-mask disc table to compare.")
    parser.add_argument("--truth", type=Path,
                        default=ANALYSIS_REPORTS_DIR / "disc_analysis.csv",
                        help="Ground-truth disc table.")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Where to write the results (default outputs/metrics).")
    parser.add_argument("--viz-dir", type=Path, default=None,
                        help="Where to write the figure (default outputs/visualizations).")
    parser.add_argument("--skip-pfirrmann", action="store_true",
                        help="Skip the end-to-end Pfirrmann evaluation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    out_dir = args.out_dir or METRICS_DIR
    viz_dir = args.viz_dir or VISUALIZATIONS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    truth_path, predicted_path = args.truth, args.predicted
    for path in (truth_path, predicted_path):
        if not path.exists():
            raise SystemExit(f"Missing {path}. See the module docstring.")

    truth = pd.read_csv(truth_path)
    predicted = pd.read_csv(predicted_path)
    print(f"truth     : {truth_path}")
    print(f"predicted : {predicted_path}")

    keys = ["patient_id", "series_id", "ivd_label"]
    merged = truth.merge(predicted, on=keys, how="inner", suffixes=("_gt", "_pred"))

    # Restrict to the series that actually have predictions (the test split).
    print(f"Ground-truth disc records : {len(truth):,}")
    print(f"Predicted disc records    : {len(predicted):,}")
    print(f"Matched on {keys}: {len(merged):,}")

    truth_test = truth[truth["series_id"].isin(predicted["series_id"].unique())]
    print(f"\nOn the {predicted['series_id'].nunique()} series with predictions:")
    print(f"  ground-truth discs : {len(truth_test):,}")
    print(f"  predicted discs    : {len(predicted):,}")
    print(f"  matched by identity: {len(merged):,} "
          f"({100 * len(merged) / max(len(truth_test), 1):.1f}% of ground-truth discs)")

    detection = {
        "n_series_with_predictions": int(predicted["series_id"].nunique()),
        "n_truth_discs": int(len(truth_test)),
        "n_predicted_discs": int(len(predicted)),
        "n_matched_by_identity": int(len(merged)),
        "pct_truth_discs_recovered": round(
            100 * len(merged) / max(len(truth_test), 1), 2
        ),
        "n_predicted_not_in_truth": int(len(predicted) - len(merged)),
        "n_truth_not_recovered": int(len(truth_test) - len(merged)),
    }

    # --- per-measurement agreement ---------------------------------------
    rows: list[dict] = []
    for column, unit in COMPARED:
        gt_col, pred_col = f"{column}_gt", f"{column}_pred"
        if gt_col not in merged.columns or pred_col not in merged.columns:
            continue

        pair = merged[[gt_col, pred_col]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(pair) < 5:
            continue

        gt_values = pair[gt_col].to_numpy(dtype=float)
        pred_values = pair[pred_col].to_numpy(dtype=float)
        error = pred_values - gt_values

        rows.append({
            "measurement": column,
            "unit": unit,
            "n": int(len(pair)),
            "gt_mean": round(float(gt_values.mean()), 4),
            "pred_mean": round(float(pred_values.mean()), 4),
            "bias": round(float(error.mean()), 4),
            "mae": round(float(np.abs(error).mean()), 4),
            "rmse": round(float(np.sqrt((error**2).mean())), 4),
            "p90_abs_error": round(float(np.percentile(np.abs(error), 90)), 4),
            "pearson_r": round(float(np.corrcoef(gt_values, pred_values)[0, 1]), 4),
            "spearman_r": round(
                float(pd.Series(gt_values).corr(pd.Series(pred_values), method="spearman")),
                4,
            ),
            # Relative MAE against the ground-truth mean magnitude, so the
            # errors are comparable across measurements with different scales.
            "mae_pct_of_gt_mean": round(
                100 * float(np.abs(error).mean()) / abs(float(gt_values.mean())), 2
            )
            if gt_values.mean() != 0
            else None,
        })

    agreement = pd.DataFrame(rows)
    agreement.to_csv(out_dir / "measurement_agreement.csv", index=False)

    print("\n=== measurement agreement: predicted mask vs ground-truth mask ===")
    print(agreement[["measurement", "unit", "n", "gt_mean", "bias", "mae",
                     "mae_pct_of_gt_mean", "pearson_r"]].to_string(index=False))

    pfirrmann = None
    if not args.skip_pfirrmann:
        pfirrmann = end_to_end_pfirrmann(truth, predicted)

    save_json(
        {
            "detection": detection,
            "measurement_agreement": agreement.to_dict("records"),
            "end_to_end_pfirrmann": pfirrmann,
            "interpretation": (
                "Measurements are compared only for discs whose identity the "
                "prediction recovered correctly. Discs the segmentation missed or "
                "mis-numbered are counted in 'detection' instead - they are a "
                "detection failure, not a measurement error, and averaging them "
                "into the measurement error would conflate the two."
            ),
        },
        out_dir / "measurement_agreement.json",
    )

    render_figure(merged, agreement, viz_dir)
    print(f"\n  -> {out_dir / 'measurement_agreement.json'}")
    print(f"  -> {out_dir / 'measurement_agreement.csv'}")


def end_to_end_pfirrmann(truth: pd.DataFrame, predicted: pd.DataFrame) -> dict | None:
    """Pfirrmann agreement when the features come from PREDICTED masks.

    The Stage E models are fitted on ground-truth-derived features from the
    **training** patients, then applied to **predicted**-mask features from the
    test patients. That is the honest end-to-end number: it includes
    segmentation error, whereas Stage E's own report deliberately excluded it by
    using ground-truth masks on both sides.

    Returns None when the test split has too few usable predicted discs.
    """
    from src.models.baseline_findings import (
        FEATURE_SETS,
        ORDINAL_TARGET,
        OrdinalClassifier,
        evaluate_ordinal,
        prepare_modelling_table,
        split_xy,
    )

    features = FEATURE_SETS["geometric+intensity"]

    # Fit on ground-truth features, training patients only.
    gt_table = prepare_modelling_table(truth, primary_series_only=True)
    gt_table = gt_table[gt_table["modality"].isin(["t2", "t2_SPACE"])]
    gt_data = split_xy(gt_table, features, ORDINAL_TARGET)
    if "train" not in gt_data:
        return None

    # Evaluate on predicted features, test patients only.
    pred_table = prepare_modelling_table(predicted, primary_series_only=True)
    pred_table = pred_table[
        (pred_table["split"] == "test")
        & pred_table["modality"].isin(["t2", "t2_SPACE"])
    ]
    pred_data = split_xy(pred_table, features, ORDINAL_TARGET)
    if "test" not in pred_data or len(pred_data["test"][1]) < 20:
        return None

    results: dict = {
        "note": (
            "Fitted on ground-truth-mask features from training patients, "
            "evaluated on predicted-mask features from test patients. Includes "
            "segmentation error, unlike the Stage E report."
        ),
        "features": "geometric+intensity",
        "n_train_discs": int(len(gt_data["train"][1])),
    }

    for estimator in ["logreg", "forest"]:
        model = OrdinalClassifier(estimator=estimator, seed=42).fit(
            gt_data["train"][0], gt_data["train"][1].astype(int)
        )
        x_test, y_test = pred_data["test"]
        y_test = y_test.astype(int)
        metrics = evaluate_ordinal(y_test, model.predict(x_test))
        results[estimator] = {
            "n_test_discs": metrics["n"],
            "quadratic_weighted_kappa": metrics["quadratic_weighted_kappa"],
            "mae": metrics["mae"],
            "exact_agreement": metrics["exact_agreement"],
            "within_one_grade": metrics["within_one_grade"],
        }
        print(f"\n=== end-to-end Pfirrmann ({estimator}, predicted masks) ===")
        print(f"  n={metrics['n']}  QWK={metrics['quadratic_weighted_kappa']}  "
              f"MAE={metrics['mae']}  within1={metrics['within_one_grade']}")

    return results


def render_figure(
    merged: pd.DataFrame, agreement: pd.DataFrame, viz_dir: Path
) -> None:
    """Scatter plots of predicted vs ground-truth measurements."""
    import matplotlib.pyplot as plt

    columns = [c for c, _ in COMPARED
               if f"{c}_gt" in merged.columns and c in set(agreement["measurement"])][:6]
    if not columns:
        return

    figure, axes = plt.subplots(2, 3, figsize=(15, 8.6))
    axes = axes.ravel()

    for axis, column in zip(axes, columns):
        pair = merged[[f"{column}_gt", f"{column}_pred"]].replace(
            [np.inf, -np.inf], np.nan
        ).dropna()
        gt_values = pair[f"{column}_gt"]
        pred_values = pair[f"{column}_pred"]

        axis.scatter(gt_values, pred_values, s=9, alpha=0.4, color="#4c72b0")
        low = float(min(gt_values.min(), pred_values.min()))
        high = float(max(gt_values.max(), pred_values.max()))
        axis.plot([low, high], [low, high], "k--", linewidth=1, label="identity")

        row = agreement[agreement["measurement"] == column].iloc[0]
        axis.set_xlabel(f"ground-truth mask ({row['unit']})")
        axis.set_ylabel(f"predicted mask ({row['unit']})")
        axis.set_title(
            f"{column}\nMAE={row['mae']} {row['unit']}  "
            f"({row['mae_pct_of_gt_mean']}% of mean)  r={row['pearson_r']}",
            fontsize=9,
        )
        axis.legend(fontsize=8, frameon=False)
        axis.grid(alpha=0.2)

    for axis in axes[len(columns):]:
        axis.axis("off")

    figure.suptitle(
        "Disc measurements from predicted vs ground-truth segmentation "
        "(test patients, correctly identified discs only)",
        fontsize=12,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    path = viz_dir / "measurement_agreement.png"
    figure.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(figure)
    print(f"\n  -> {path}")


if __name__ == "__main__":
    main()
