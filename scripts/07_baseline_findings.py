"""Sprint 2 / Stage E - baseline disc-level radiological finding estimators.

Fits classical baselines on the Stage D table, using the Sprint 1 patient-level
split. Compares two feature families (geometry from the segmentation vs
intensity from the ROI) and their union, so the contribution of each is visible
rather than assumed.

Writes:

    outputs/metrics/baseline_findings.json         all results
    outputs/metrics/baseline_findings_summary.csv  one row per (target, features, model)
    outputs/reports/baseline_findings.md           readable report
    outputs/visualizations/baseline_findings.png
    outputs/visualizations/baseline_pfirrmann_confusion.png

This is a *baseline*, not a final system, and it is not a clinical result.

Usage
-----
    python scripts/07_baseline_findings.py
    python scripts/07_baseline_findings.py --all-series   # keep every series
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.baseline_findings import (  # noqa: E402
    BINARY_TARGETS,
    FEATURE_SETS,
    ORDINAL_TARGET,
    PFIRRMANN_CLASSES,
    check_split_integrity,
    fit_binary_target,
    fit_pfirrmann,
    prepare_modelling_table,
)
from src.utils.paths import (  # noqa: E402
    ANALYSIS_REPORTS_DIR,
    OUTPUTS_DIR,
    RANDOM_SEED,
    VISUALIZATIONS_DIR,
    ensure_dirs,
)
from src.utils.reporting import MarkdownReport, save_json  # noqa: E402

METRICS_DIR = OUTPUTS_DIR / "metrics"
DISC_ANALYSIS_CSV = ANALYSIS_REPORTS_DIR / "disc_analysis.csv"

ESTIMATORS = ["majority", "logreg", "forest"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-series", action="store_true",
                        help="Use every series instead of one per patient. Not "
                             "recommended: it duplicates patient-level gradings.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    if not DISC_ANALYSIS_CSV.exists():
        raise SystemExit(
            f"{DISC_ANALYSIS_CSV} not found. Run scripts/06_disc_analysis.py first."
        )

    analysis = pd.read_csv(DISC_ANALYSIS_CSV)
    table = prepare_modelling_table(
        analysis, primary_series_only=not args.all_series
    )
    integrity = check_split_integrity(table)

    print(f"Modelling table: {len(table):,} disc records, "
          f"{table['patient_id'].nunique()} patients")
    print(f"  rows per split     : {integrity['rows_per_split']}")
    print(f"  patients per split : {integrity['patients_per_split']}")
    print(f"  leakage-free       : {integrity['leakage_free']}")
    print(f"  duplicate (patient, disc) rows: "
          f"{integrity['duplicate_patient_disc_rows']}")

    results: dict = {
        "config": {
            "primary_series_only": not args.all_series,
            "seed": args.seed,
            "n_records": int(len(table)),
            "n_patients": int(table["patient_id"].nunique()),
            "split_integrity": integrity,
            "feature_sets": {k: len(v) for k, v in FEATURE_SETS.items()},
            "estimators": ESTIMATORS,
        },
        "binary": {},
        "ordinal": {},
    }

    # --- binary findings -------------------------------------------------
    print("\n=== Binary findings ===")
    rows: list[dict] = []
    for target in BINARY_TARGETS:
        if target not in table.columns:
            print(f"  {target}: column missing, skipped")
            continue
        results["binary"][target] = {}
        print(f"\n  {target}  (positives: train="
              f"{int(table.loc[table.split == 'train', target].sum())}, "
              f"test={int(table.loc[table.split == 'test', target].sum())})")

        for feature_name, features in FEATURE_SETS.items():
            results["binary"][target][feature_name] = {}
            for estimator in ESTIMATORS:
                outcome = fit_binary_target(
                    table, target, features, estimator=estimator, seed=args.seed
                )
                results["binary"][target][feature_name][estimator] = outcome
                if "error" in outcome:
                    continue
                test = outcome["splits"].get("test", {})
                rows.append({
                    "target": target,
                    "type": "binary",
                    "features": feature_name,
                    "estimator": estimator,
                    "test_n": test.get("n"),
                    "test_n_positive": test.get("n_positive"),
                    "test_pr_auc": test.get("pr_auc"),
                    "test_pr_auc_baseline": test.get("pr_auc_baseline"),
                    "test_roc_auc": test.get("roc_auc"),
                    "test_balanced_accuracy": test.get("balanced_accuracy"),
                    "test_sensitivity": test.get("sensitivity"),
                    "test_specificity": test.get("specificity"),
                })
                if estimator != "majority":
                    print(f"    {feature_name:20s} {estimator:8s} "
                          f"PR-AUC={test.get('pr_auc')} "
                          f"(baseline {test.get('pr_auc_baseline')})  "
                          f"ROC-AUC={test.get('roc_auc')}")

    # --- ordinal Pfirrmann ----------------------------------------------
    print("\n=== Pfirrmann grade (ordinal) ===")
    for feature_name, features in FEATURE_SETS.items():
        results["ordinal"][feature_name] = {}
        for estimator in ["logreg", "forest"]:
            outcome = fit_pfirrmann(
                table, features, estimator=estimator, seed=args.seed, t2_only=True
            )
            results["ordinal"][feature_name][estimator] = outcome
            if "error" in outcome:
                print(f"  {feature_name} {estimator}: {outcome['error']}")
                continue
            test = outcome["splits"].get("test", {})
            rows.append({
                "target": ORDINAL_TARGET,
                "type": "ordinal",
                "features": feature_name,
                "estimator": estimator,
                "test_n": test.get("n"),
                "test_qwk": test.get("quadratic_weighted_kappa"),
                "test_mae": test.get("mae"),
                "test_exact": test.get("exact_agreement"),
                "test_within_one": test.get("within_one_grade"),
                "test_spearman": test.get("spearman_expected_vs_true"),
            })
            print(f"  {feature_name:20s} {estimator:8s} "
                  f"QWK={test.get('quadratic_weighted_kappa')} "
                  f"MAE={test.get('mae')} "
                  f"exact={test.get('exact_agreement')} "
                  f"within1={test.get('within_one_grade')}")

    summary = pd.DataFrame(rows)
    summary.to_csv(METRICS_DIR / "baseline_findings_summary.csv", index=False)
    save_json(results, METRICS_DIR / "baseline_findings.json")

    render_figures(summary, results)
    write_report(table, integrity, results, summary)

    print(f"\n  -> {METRICS_DIR / 'baseline_findings.json'}")
    print(f"  -> {METRICS_DIR / 'baseline_findings_summary.csv'}")
    print(f"  -> {ANALYSIS_REPORTS_DIR / 'baseline_findings.md'}")
    print("\nDone. Baseline only - not a clinical result.")


def render_figures(summary: pd.DataFrame, results: dict) -> None:
    """PR-AUC per binary target and the Pfirrmann confusion matrix."""
    import matplotlib.pyplot as plt

    binary = summary[(summary["type"] == "binary") & (summary["estimator"] != "majority")]
    if not binary.empty:
        figure, axis = plt.subplots(figsize=(13, 5.2))
        targets = list(dict.fromkeys(binary["target"]))
        combos = [
            (f, e) for f in FEATURE_SETS for e in ["logreg", "forest"]
        ]
        width = 0.8 / len(combos)
        x = np.arange(len(targets))

        for offset, (feature_name, estimator) in enumerate(combos):
            values, baselines = [], []
            for target in targets:
                row = binary[
                    (binary["target"] == target)
                    & (binary["features"] == feature_name)
                    & (binary["estimator"] == estimator)
                ]
                values.append(float(row["test_pr_auc"].iloc[0]) if len(row) else np.nan)
                baselines.append(
                    float(row["test_pr_auc_baseline"].iloc[0]) if len(row) else np.nan
                )
            axis.bar(x + (offset - len(combos) / 2 + 0.5) * width, values, width,
                     label=f"{feature_name} / {estimator}")

        # Prevalence baseline: PR-AUC must beat this to be informative.
        axis.plot(x, baselines, "kv", markersize=9,
                  label="prevalence baseline (random ranker)")

        axis.set_xticks(x, targets, rotation=20, ha="right")
        axis.set_ylabel("test PR-AUC")
        axis.set_title(
            "Stage E baseline: test-set PR-AUC per binary finding\n"
            "(held-out patients; a bar must clear the black marker to be informative)",
            fontsize=11,
        )
        axis.legend(fontsize=8, frameon=False, ncol=3)
        axis.grid(alpha=0.2, axis="y")
        figure.tight_layout()
        path = VISUALIZATIONS_DIR / "baseline_findings.png"
        figure.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(figure)
        print(f"\n  -> {path}")

    # --- Pfirrmann confusion matrix for the best feature set -------------
    ordinal = summary[summary["type"] == "ordinal"]
    if ordinal.empty or ordinal["test_qwk"].isna().all():
        return
    best = ordinal.loc[ordinal["test_qwk"].idxmax()]
    entry = results["ordinal"][best["features"]][best["estimator"]]
    matrix = np.array(entry["splits"]["test"]["confusion_matrix"])

    figure, axis = plt.subplots(figsize=(6.4, 5.6))
    image = axis.imshow(matrix, cmap="Blues")
    labels = entry["splits"]["test"]["confusion_matrix_labels"]
    axis.set_xticks(range(len(labels)), labels)
    axis.set_yticks(range(len(labels)), labels)
    axis.set_xlabel("predicted Pfirrmann grade")
    axis.set_ylabel("annotated Pfirrmann grade")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            axis.text(j, i, int(matrix[i, j]), ha="center", va="center",
                      color="white" if matrix[i, j] > matrix.max() / 2 else "black",
                      fontsize=10)
    axis.set_title(
        f"Pfirrmann grade, test patients\n"
        f"{best['features']} / {best['estimator']}  "
        f"QWK={best['test_qwk']}  MAE={best['test_mae']}",
        fontsize=11,
    )
    figure.colorbar(image, ax=axis, shrink=0.8, label="disc records")
    figure.tight_layout()
    path = VISUALIZATIONS_DIR / "baseline_pfirrmann_confusion.png"
    figure.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(figure)
    print(f"  -> {path}")


def write_report(
    table: pd.DataFrame, integrity: dict, results: dict, summary: pd.DataFrame
) -> Path:
    """Compose outputs/reports/baseline_findings.md."""
    report = MarkdownReport(
        "Stage E - Baseline Disc-Level Finding Assessment",
        "Classical baselines on segmentation geometry and ROI intensity features",
    )

    report.text(
        "**This is a baseline, not a finished system, and not a clinical result.** "
        "Its purpose is to establish what the available disc-level data supports "
        "before any larger architecture is considered. Every score below is on "
        "**held-out test patients** from the Sprint 1 patient-level split."
    )

    report.heading("1. Data and leakage control")
    report.key_values(
        {
            "Disc records used": len(table),
            "Patients": table["patient_id"].nunique(),
            "One series per patient": results["config"]["primary_series_only"],
            "Rows per split": integrity["rows_per_split"],
            "Patients per split": integrity["patients_per_split"],
            "Duplicate (patient, disc) rows": integrity["duplicate_patient_disc_rows"],
            "Patient-disjoint splits verified": integrity["leakage_free"],
        }
    )
    report.text(
        "Gradings are a **patient-level** annotation shared by a patient's 1-3 series. "
        "The table is therefore deduplicated to one series per patient (T2 preferred, "
        "because the Pfirrmann grade is defined on T2 signal) before fitting. Keeping "
        "all series would train on the same label two or three times and inflate every "
        "score. Both the deduplication and the patient-disjointness are asserted in "
        "code, not assumed."
    )

    report.heading("2. Feature families")
    report.table(
        [
            {
                "family": "`geometric`",
                "n": len(FEATURE_SETS["geometric"]),
                "source": "segmentation masks, measured in mm at Sprint 1's 1.0 mm/px",
                "examples": "central/anterior/posterior disc height, area, AP extent, "
                            "height relative to series median and to neighbours, "
                            "disc-to-vertebra height ratio, vertebral AP offset, "
                            "canal width",
            },
            {
                "family": "`intensity`",
                "n": len(FEATURE_SETS["intensity"]),
                "source": "preprocessed image inside the disc and its references",
                "examples": "disc mean/median/percentile signal, coefficient of "
                            "variation, nucleus vs annulus, disc-to-vertebra and "
                            "disc-to-canal signal ratios",
            },
            {
                "family": "`geometric+intensity`",
                "n": len(FEATURE_SETS["geometric+intensity"]),
                "source": "both",
                "examples": "union of the two",
            },
        ],
        ["family", "n", "source", "examples"],
    )
    report.text(
        "The two families are reported separately on purpose. The Pfirrmann grade is "
        "defined by nucleus **signal** plus disc height, so intensity features should "
        "matter for it; disc narrowing and spondylolisthesis are **geometric** by "
        "definition. Splitting the comparison shows whether the models behave "
        "consistently with that expectation, which is a useful check that the features "
        "measure what they claim to."
    )

    report.heading("3. Binary findings - test-set results")
    report.text(
        "**PR-AUC is the headline metric**, not ROC-AUC: several findings are rare "
        "(spondylolisthesis is ~2.8% of discs) and ROC-AUC is optimistic under that "
        "imbalance. The `prevalence baseline` column is what a random ranker scores - "
        "a model must clearly beat it to carry information. `majority` is a "
        "predict-the-prior reference."
    )
    binary = summary[summary["type"] == "binary"]
    if not binary.empty:
        shown = binary[binary["estimator"] != "majority"][
            ["target", "features", "estimator", "test_n", "test_n_positive",
             "test_pr_auc", "test_pr_auc_baseline", "test_roc_auc",
             "test_balanced_accuracy", "test_sensitivity", "test_specificity"]
        ]
        report.dataframe(shown, max_rows=100)

    report.heading("4. Pfirrmann grade - test-set results")
    report.text(
        "Modelled as an **ordinal** target with the Frank & Hall decomposition: four "
        "binary models estimate `P(grade > k)` and the per-class probabilities come "
        "from consecutive differences. Quadratic weighted kappa (QWK) is the headline "
        "metric - it is the convention for ordinal radiological grading and penalises "
        "a 1-vs-5 confusion far more than a 4-vs-5 one. Restricted to T2 / T2 SPACE "
        "series."
    )
    ordinal = summary[summary["type"] == "ordinal"]
    if not ordinal.empty:
        report.dataframe(
            ordinal[["features", "estimator", "test_n", "test_qwk", "test_mae",
                     "test_exact", "test_within_one", "test_spearman"]],
            max_rows=20,
        )
    first = next(iter(results["ordinal"].values()), {})
    entry = first.get("logreg", {})
    if "majority_baseline" in entry:
        report.key_values(
            {
                "Majority-class baseline predicted grade":
                    entry["majority_baseline"]["predicted_class"],
                "Majority-class exact agreement":
                    entry["majority_baseline"]["exact_agreement"],
                "Majority-class MAE": entry["majority_baseline"]["mae"],
            }
        )
        report.text(
            "A constant prediction has a quadratic weighted kappa of 0 by "
            "construction, so accuracy is the meaningful comparison against it."
        )

    report.heading("5. How to read these numbers")
    report.bullets(
        [
            "**Test splits are small.** The test set holds 33 patients; for a finding "
            "with ~3% prevalence that is a single-digit number of positive discs. Point "
            "estimates for the rare findings are therefore very uncertain, and the "
            "positive count is printed next to every score for that reason.",
            "**These are the features a baseline can see.** Geometry and summary "
            "intensity describe a disc coarsely. A focal herniation is a local shape "
            "detail that a handful of summary statistics is not expected to capture "
            "well - a low score for herniation is an honest reflection of the feature "
            "set, not evidence the task is impossible.",
            "**No composite severity score is produced.** Each finding is reported on "
            "its own, as the dataset annotates it.",
            "**The segmentation used here is ground truth.** These numbers isolate the "
            "finding-assessment step from segmentation error. Running the same "
            "pipeline on predicted masks is what quantifies the combined error, and "
            "the disc-identification audit in "
            "`outputs/metrics/disc_identification_test.json` shows how much identity "
            "error a predicted mask introduces.",
        ]
    )

    return report.save(ANALYSIS_REPORTS_DIR / "baseline_findings.md")


if __name__ == "__main__":
    main()
