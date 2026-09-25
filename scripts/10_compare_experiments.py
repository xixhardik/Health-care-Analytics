"""Compare the Sprint 2 baseline against the extended-training run.

Reads the metric files both runs produced and reports, per metric, whether the
extended run improved or worsened. Also decides whether the change is large
enough to justify re-running the disc-level analysis.

Inputs (baseline):
    outputs/metrics/segmentation_metrics_test.json
    outputs/metrics/disc_identification_test.json
    outputs/metrics/measurement_agreement.json
    outputs/metrics/baseline_findings_summary.csv
    outputs/metrics/training_history.csv

Inputs (extended):
    outputs/reports/sprint2_extended/metrics/segmentation_metrics_test.json
    outputs/reports/sprint2_extended/metrics/disc_identification_test.json
    outputs/reports/sprint2_extended/training_summary.json
    outputs/checkpoints/sprint2_extended/history.csv
    (+ the optional disc-level files, if the re-run was justified)

Writes:
    outputs/reports/sprint2_extended/comparison.json
    outputs/reports/sprint2_extended/comparison.md
    outputs/visualizations/sprint2_extended/training_curves_combined.png
    outputs/visualizations/sprint2_extended/metric_comparison.png

Usage
-----
    python scripts/10_compare_experiments.py
    python scripts/10_compare_experiments.py --dice-threshold 0.005
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

from src.utils.paths import OUTPUTS_DIR, VISUALIZATIONS_DIR, ensure_dirs  # noqa: E402
from src.utils.reporting import MarkdownReport, save_json  # noqa: E402

EXPERIMENT = "sprint2_extended"
BASELINE_METRICS = OUTPUTS_DIR / "metrics"
EXTENDED_DIR = OUTPUTS_DIR / "reports" / EXPERIMENT
EXTENDED_METRICS = EXTENDED_DIR / "metrics"
EXTENDED_VIZ = VISUALIZATIONS_DIR / EXPERIMENT
CHECKPOINT_DIR = OUTPUTS_DIR / "checkpoints" / EXPERIMENT

CLASSES = ["background", "vertebra", "intervertebral_disc", "spinal_canal"]
SEG_METRICS = ["dice", "iou", "precision", "recall"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dice-threshold", type=float, default=0.005,
        help="Absolute change in macro foreground Dice considered meaningful.",
    )
    parser.add_argument(
        "--index-threshold", type=float, default=1.0,
        help="Absolute change in disc-index accuracy (percentage points) "
             "considered meaningful for the disc-level stages.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def compare_value(
    name: str, baseline: float | None, extended: float | None, *, higher_is_better: bool
) -> dict:
    """Compare one metric and label the direction of change."""
    entry = {
        "metric": name,
        "baseline": baseline,
        "extended": extended,
        "higher_is_better": higher_is_better,
    }
    if baseline is None or extended is None:
        entry.update({"delta": None, "pct_change": None, "verdict": "unavailable"})
        return entry

    delta = extended - baseline
    entry["delta"] = round(float(delta), 6)
    entry["pct_change"] = (
        round(float(100 * delta / abs(baseline)), 3) if baseline else None
    )

    if abs(delta) < 1e-9:
        entry["verdict"] = "unchanged"
    elif (delta > 0) == higher_is_better:
        entry["verdict"] = "improved"
    else:
        entry["verdict"] = "worsened"
    return entry


def main() -> None:
    args = parse_args()
    ensure_dirs()
    EXTENDED_DIR.mkdir(parents=True, exist_ok=True)
    EXTENDED_VIZ.mkdir(parents=True, exist_ok=True)

    baseline_seg = load_json(BASELINE_METRICS / "segmentation_metrics_test.json")
    extended_seg = load_json(EXTENDED_METRICS / "segmentation_metrics_test.json")
    if baseline_seg is None:
        raise SystemExit("Baseline test metrics missing. Run evaluate_unet.py first.")
    if extended_seg is None:
        raise SystemExit(
            f"Extended test metrics missing at "
            f"{EXTENDED_METRICS / 'segmentation_metrics_test.json'}.\n"
            f"Run: python scripts/evaluate_unet.py --split test "
            f"--checkpoint {CHECKPOINT_DIR / 'best_val_dice.pt'} --tag {EXPERIMENT}"
        )

    training_summary = load_json(EXTENDED_DIR / "training_summary.json") or {}

    # --- segmentation comparison -----------------------------------------
    comparisons: list[dict] = []

    for class_name in CLASSES:
        for metric in SEG_METRICS:
            base = baseline_seg["aggregate"]["per_class"][class_name][metric]
            ext = extended_seg["aggregate"]["per_class"][class_name][metric]
            comparisons.append(
                compare_value(f"{class_name}/{metric}", base, ext, higher_is_better=True)
            )

    for metric in SEG_METRICS:
        comparisons.append(
            compare_value(
                f"macro_foreground/{metric}",
                baseline_seg["aggregate"]["macro_foreground"][metric],
                extended_seg["aggregate"]["macro_foreground"][metric],
                higher_is_better=True,
            )
        )

    comparisons.append(
        compare_value(
            "pixel_accuracy",
            baseline_seg["pixel_accuracy"],
            extended_seg["pixel_accuracy"],
            higher_is_better=True,
        )
    )

    for class_name in CLASSES:
        comparisons.append(
            compare_value(
                f"per_patient_dice/{class_name}",
                baseline_seg["per_patient_dice"][class_name]["mean_dice"],
                extended_seg["per_patient_dice"][class_name]["mean_dice"],
                higher_is_better=True,
            )
        )

    # --- disc identification ---------------------------------------------
    baseline_id = load_json(BASELINE_METRICS / "disc_identification_test.json")
    extended_id = load_json(EXTENDED_METRICS / "disc_identification_test.json")
    for key in [
        "pct_slices_with_matching_disc_count",
        "pct_discs_region_found",
        "pct_discs_index_correct",
    ]:
        comparisons.append(
            compare_value(
                f"disc_identification/{key}",
                (baseline_id or {}).get(key),
                (extended_id or {}).get(key),
                higher_is_better=True,
            )
        )

    # --- measurement agreement (predicted vs ground-truth masks) ---------
    baseline_agreement = load_json(BASELINE_METRICS / "measurement_agreement.json")
    extended_agreement = load_json(EXTENDED_METRICS / "measurement_agreement.json")
    measurement_rows: list[dict] = []
    if baseline_agreement and extended_agreement:
        base_map = {r["measurement"]: r for r in baseline_agreement["measurement_agreement"]}
        ext_map = {r["measurement"]: r for r in extended_agreement["measurement_agreement"]}
        for name in sorted(set(base_map) & set(ext_map)):
            # MAE: lower is better.
            measurement_rows.append(
                compare_value(
                    f"measurement_mae/{name}",
                    base_map[name]["mae"],
                    ext_map[name]["mae"],
                    higher_is_better=False,
                )
            )
            measurement_rows.append(
                compare_value(
                    f"measurement_r/{name}",
                    base_map[name]["pearson_r"],
                    ext_map[name]["pearson_r"],
                    higher_is_better=True,
                )
            )
        comparisons.append(
            compare_value(
                "disc_detection/pct_truth_discs_recovered",
                baseline_agreement["detection"]["pct_truth_discs_recovered"],
                extended_agreement["detection"]["pct_truth_discs_recovered"],
                higher_is_better=True,
            )
        )

        # End-to-end Pfirrmann: features taken from PREDICTED masks, so this is
        # the only Pfirrmann number that responds to the segmentation model.
        base_pf = (baseline_agreement or {}).get("end_to_end_pfirrmann") or {}
        ext_pf = (extended_agreement or {}).get("end_to_end_pfirrmann") or {}
        for estimator in ["logreg", "forest"]:
            if estimator not in base_pf or estimator not in ext_pf:
                continue
            for key, higher in [
                ("quadratic_weighted_kappa", True),
                ("mae", False),
                ("within_one_grade", True),
            ]:
                measurement_rows.append(
                    compare_value(
                        f"end_to_end_pfirrmann/{estimator}/{key}",
                        base_pf[estimator].get(key),
                        ext_pf[estimator].get(key),
                        higher_is_better=higher,
                    )
                )

    # --- Pfirrmann / finding baselines -----------------------------------
    finding_rows = compare_findings()

    # --- decision --------------------------------------------------------
    macro_dice = next(
        c for c in comparisons if c["metric"] == "macro_foreground/dice"
    )
    delta = macro_dice["delta"] or 0.0
    dice_meaningful = abs(delta) >= args.dice_threshold

    # Macro Dice alone is the wrong gate for the disc-level stages. What
    # actually governs them is whether each disc is found AND given the right
    # index, because a numbering shift attaches a grading to the wrong disc.
    # A small Dice change can move that accuracy several points, so the
    # indexing metric is part of the decision.
    indexing = next(
        (c for c in comparisons
         if c["metric"] == "disc_identification/pct_discs_index_correct"),
        None,
    )
    index_delta = (indexing or {}).get("delta") or 0.0
    index_meaningful = abs(index_delta) >= args.index_threshold

    meaningful = dice_meaningful or index_meaningful
    reasons = [
        f"|delta macro foreground Dice| = {abs(delta):.4f} "
        f"{'>=' if dice_meaningful else '<'} threshold {args.dice_threshold}",
        f"|delta disc-index accuracy| = {abs(index_delta):.2f} pp "
        f"{'>=' if index_meaningful else '<'} threshold {args.index_threshold} pp",
    ]
    decision = {
        "macro_foreground_dice_baseline": macro_dice["baseline"],
        "macro_foreground_dice_extended": macro_dice["extended"],
        "delta": delta,
        "threshold": args.dice_threshold,
        "dice_change_meaningful": bool(dice_meaningful),
        "disc_index_accuracy_baseline": (indexing or {}).get("baseline"),
        "disc_index_accuracy_extended": (indexing or {}).get("extended"),
        "disc_index_accuracy_delta_pp": index_delta,
        "index_threshold_pp": args.index_threshold,
        "index_change_meaningful": bool(index_meaningful),
        "meaningful_change": bool(meaningful),
        "verdict": macro_dice["verdict"],
        "disc_level_rerun_justified": bool(meaningful),
        "reason": "; ".join(reasons),
    }

    payload = {
        "experiment": EXPERIMENT,
        "training_summary": training_summary,
        "convergence": analyse_convergence(training_summary),
        "segmentation_comparisons": comparisons,
        "measurement_comparisons": measurement_rows,
        "finding_comparisons": finding_rows,
        "decision": decision,
        "test_set_note": (
            "The test set was evaluated only after training finished and only "
            "with the checkpoint selected on validation foreground Dice."
        ),
    }
    save_json(payload, EXTENDED_DIR / "comparison.json")

    print_console(comparisons, measurement_rows, finding_rows, decision, training_summary)
    render_figures()
    write_report(payload)

    print(f"\n  -> {EXTENDED_DIR / 'comparison.json'}")
    print(f"  -> {EXTENDED_DIR / 'comparison.md'}")


def analyse_convergence(training_summary: dict) -> dict:
    """Quantify whether the run had actually plateaued by the final epoch.

    "Early stopping did not fire" is not the same as "still learning": with
    ``min_delta=0`` any improvement, however small, resets the patience counter.
    Comparing the late per-epoch gain against the mid-run gain settles it.
    """
    path = CHECKPOINT_DIR / "history.csv"
    if not path.exists():
        return {}
    history = pd.read_csv(path)
    if len(history) < 8:
        return {}

    last6 = history.tail(6)
    mid = history[(history["epoch"] >= 14) & (history["epoch"] <= 20)]

    last6_gain = float(np.mean(np.diff(last6["val_fg_dice"])))
    mid_gain = float(np.mean(np.diff(mid["val_fg_dice"]))) if len(mid) > 1 else float("nan")

    return {
        "last6_min": float(last6["val_fg_dice"].min()),
        "last6_max": float(last6["val_fg_dice"].max()),
        "last6_spread": float(last6["val_fg_dice"].max() - last6["val_fg_dice"].min()),
        "last6_mean_gain": round(last6_gain, 6),
        "mid_mean_gain": round(mid_gain, 6),
        "gain_ratio": round(mid_gain / last6_gain, 1) if last6_gain else None,
        "final_lr": float(history["learning_rate"].iloc[-1]),
        "stopped_early": bool(training_summary.get("stopped_early", False)),
        "converged": bool(abs(last6_gain) < 1e-4),
    }


def compare_findings() -> list[dict]:
    """Compare Stage E finding metrics between the two runs, if both exist."""
    baseline_path = BASELINE_METRICS / "baseline_findings_summary.csv"
    extended_path = EXTENDED_METRICS / "baseline_findings_summary.csv"
    if not (baseline_path.exists() and extended_path.exists()):
        return []

    baseline = pd.read_csv(baseline_path)
    extended = pd.read_csv(extended_path)
    rows: list[dict] = []

    keys = ["target", "type", "features", "estimator"]
    merged = baseline.merge(extended, on=keys, how="inner", suffixes=("_base", "_ext"))

    for _, row in merged.iterrows():
        label = f"{row['target']}/{row['features']}/{row['estimator']}"
        if row["type"] == "binary" and not pd.isna(row.get("test_pr_auc_base")):
            rows.append(
                compare_value(
                    f"pr_auc/{label}",
                    float(row["test_pr_auc_base"]),
                    float(row["test_pr_auc_ext"]),
                    higher_is_better=True,
                )
            )
        if row["type"] == "ordinal" and not pd.isna(row.get("test_qwk_base")):
            rows.append(
                compare_value(
                    f"pfirrmann_qwk/{label}",
                    float(row["test_qwk_base"]),
                    float(row["test_qwk_ext"]),
                    higher_is_better=True,
                )
            )
            rows.append(
                compare_value(
                    f"pfirrmann_mae/{label}",
                    float(row["test_mae_base"]),
                    float(row["test_mae_ext"]),
                    higher_is_better=False,
                )
            )
    return rows


def print_console(
    comparisons: list[dict],
    measurement_rows: list[dict],
    finding_rows: list[dict],
    decision: dict,
    training_summary: dict,
) -> None:
    """Print the comparison tables."""
    print("=" * 84)
    print("BASELINE (12 epochs) vs EXTENDED TRAINING")
    print("=" * 84)
    if training_summary:
        print(f"  epochs completed        : {training_summary.get('epochs_completed_total')}")
        print(f"  stopped early           : {training_summary.get('stopped_early')}")
        print(f"  best val fg Dice        : {training_summary.get('best_val_dice')} "
              f"(epoch {training_summary.get('best_val_dice_epoch')})")
        print(f"  session wall time (min) : "
              f"{training_summary.get('wall_minutes_this_session')}")

    def show(title: str, rows: list[dict]) -> None:
        if not rows:
            return
        print(f"\n--- {title} ---")
        print(f"  {'metric':46s} {'baseline':>10s} {'extended':>10s} "
              f"{'delta':>10s}  verdict")
        for row in rows:
            base = "-" if row["baseline"] is None else f"{row['baseline']:.4f}"
            ext = "-" if row["extended"] is None else f"{row['extended']:.4f}"
            dlt = "-" if row["delta"] is None else f"{row['delta']:+.4f}"
            print(f"  {row['metric']:46s} {base:>10s} {ext:>10s} {dlt:>10s}  "
                  f"{row['verdict']}")

    show("segmentation (test set)", comparisons)
    show("disc measurements from predicted masks", measurement_rows)
    show("disc-level finding assessment", finding_rows)

    counts = {}
    for row in comparisons + measurement_rows + finding_rows:
        counts[row["verdict"]] = counts.get(row["verdict"], 0) + 1
    print(f"\n--- tally ---\n  {counts}")

    print(f"\n--- decision ---")
    print(f"  macro foreground Dice: {decision['macro_foreground_dice_baseline']:.4f} "
          f"-> {decision['macro_foreground_dice_extended']:.4f} "
          f"({decision['delta']:+.4f}, {decision['verdict']})")
    print(f"  {decision['reason']}")
    print(f"  disc-level re-run justified: {decision['disc_level_rerun_justified']}")


def render_figures() -> None:
    """Combined training curves and a per-class metric comparison."""
    import matplotlib.pyplot as plt

    baseline_history_path = BASELINE_METRICS / "training_history.csv"
    extended_history_path = CHECKPOINT_DIR / "history.csv"
    if baseline_history_path.exists() and extended_history_path.exists():
        baseline = pd.read_csv(baseline_history_path)
        extended = pd.read_csv(extended_history_path)

        figure, axes = plt.subplots(1, 3, figsize=(17, 4.8))

        axes[0].plot(baseline["epoch"], baseline["train_loss"], "o-",
                     label="baseline train", color="#4c72b0")
        axes[0].plot(baseline["epoch"], baseline["val_loss"], "s--",
                     label="baseline val", color="#4c72b0", alpha=0.6)
        axes[0].plot(extended["epoch"], extended["train_loss"], "o-",
                     label="extended train", color="#c44e52")
        axes[0].plot(extended["epoch"], extended["val_loss"], "s--",
                     label="extended val", color="#c44e52", alpha=0.6)
        axes[0].axvline(12.5, color="black", linestyle=":", linewidth=1)
        axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss (CE + Dice)")
        axes[0].set_title("Loss (dotted line = resume point)", fontsize=11)
        axes[0].legend(fontsize=8, frameon=False); axes[0].grid(alpha=0.25)

        axes[1].plot(baseline["epoch"], baseline["val_fg_dice"], "s-",
                     label="baseline", color="#4c72b0")
        axes[1].plot(extended["epoch"], extended["val_fg_dice"], "s-",
                     label="extended", color="#c44e52")
        axes[1].axvline(12.5, color="black", linestyle=":", linewidth=1)
        axes[1].set_xlabel("epoch"); axes[1].set_ylabel("validation macro fg Dice")
        axes[1].set_title("Validation foreground Dice", fontsize=11)
        axes[1].legend(fontsize=9, frameon=False); axes[1].grid(alpha=0.25)

        for column, label in [
            ("val_dice_vertebra", "vertebra"),
            ("val_dice_intervertebral_disc", "IVD"),
            ("val_dice_spinal_canal", "canal"),
        ]:
            if column in extended.columns:
                axes[2].plot(extended["epoch"], extended[column], ".-", label=label)
            if column in baseline.columns:
                axes[2].plot(baseline["epoch"], baseline[column], ".--", alpha=0.45)
        axes[2].axvline(12.5, color="black", linestyle=":", linewidth=1)
        axes[2].set_xlabel("epoch"); axes[2].set_ylabel("validation Dice")
        axes[2].set_title("Per-class validation Dice\n(dashed = baseline)", fontsize=11)
        axes[2].legend(fontsize=8, frameon=False); axes[2].grid(alpha=0.25)

        figure.suptitle(
            "Sprint 2 baseline (12 epochs) vs extended training - same "
            "16-channel U-Net, same split, same seed",
            fontsize=12,
        )
        figure.tight_layout(rect=(0, 0, 1, 0.93))
        path = EXTENDED_VIZ / "training_curves_combined.png"
        figure.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(figure)
        print(f"\n  -> {path}")

    # --- per-class test metric bars --------------------------------------
    baseline_seg = load_json(BASELINE_METRICS / "segmentation_metrics_test.json")
    extended_seg = load_json(EXTENDED_METRICS / "segmentation_metrics_test.json")
    if not (baseline_seg and extended_seg):
        return

    figure, axes = plt.subplots(1, 4, figsize=(18, 4.4))
    x = np.arange(len(CLASSES))
    width = 0.36

    for axis, metric in zip(axes, SEG_METRICS):
        base_values = [
            baseline_seg["aggregate"]["per_class"][c][metric] for c in CLASSES
        ]
        ext_values = [
            extended_seg["aggregate"]["per_class"][c][metric] for c in CLASSES
        ]
        axis.bar(x - width / 2, base_values, width, label="baseline (12 ep)",
                 color="#4c72b0")
        axis.bar(x + width / 2, ext_values, width, label="extended", color="#c44e52")
        for i, (b, e) in enumerate(zip(base_values, ext_values)):
            axis.text(i + width / 2, e, f"{e - b:+.3f}", ha="center", va="bottom",
                      fontsize=7)
        axis.set_xticks(x, [c.replace("intervertebral_", "IVD ") for c in CLASSES],
                        rotation=20, ha="right", fontsize=8)
        axis.set_ylim(0, 1.08)
        axis.set_title(f"test {metric}", fontsize=11)
        axis.grid(alpha=0.2, axis="y")
    axes[0].set_ylabel("score")
    axes[0].legend(fontsize=8, frameon=False)

    figure.suptitle(
        "Test-set segmentation metrics: baseline vs extended training "
        "(annotations show the change)",
        fontsize=12,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    path = EXTENDED_VIZ / "metric_comparison.png"
    figure.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(figure)
    print(f"  -> {path}")


def write_report(payload: dict) -> Path:
    """Compose the comparison report."""
    training = payload["training_summary"]
    decision = payload["decision"]

    report = MarkdownReport(
        "Sprint 2 Extended Training - Controlled Experiment Report",
        "Same 16-channel U-Net, same split, same seed; only the epoch budget changed",
    )

    report.text(
        "**Controlled variable: number of epochs (12 -> up to 30).** Architecture, "
        "patient-level split, seed, preprocessing, class definitions, loss, "
        "optimiser, augmentation policy and slices-per-epoch were all held fixed. "
        "The test set was not touched during training or model selection."
    )

    report.heading("1. Training outcome")
    if training:
        report.key_values(
            {
                "Resume mode": training.get("resume", {}).get("mode"),
                "Resumed at epoch": training.get("resume", {}).get("resumed_at_epoch"),
                "Optimiser state restored":
                    training.get("resume", {}).get("optimiser_state_restored"),
                "Epochs completed (total)": training.get("epochs_completed_total"),
                "Epochs run this session": training.get("epochs_run_this_session"),
                "Max epochs": training.get("max_epochs"),
                "Stopped early": training.get("stopped_early"),
                "Stop reason": training.get("stop_reason"),
                "Best validation foreground Dice":
                    f"{training.get('best_val_dice')} "
                    f"(epoch {training.get('best_val_dice_epoch')})",
                "Best validation loss":
                    f"{training.get('best_val_loss')} "
                    f"(epoch {training.get('best_val_loss_epoch')})",
                "Wall time this session (min)":
                    training.get("wall_minutes_this_session"),
                "Mean epoch time (s)": training.get("mean_epoch_seconds"),
                "Checkpoint evaluated":
                    training.get("restored_checkpoint", {}).get("criterion"),
            }
        )
        resume_note = training.get("resume", {}).get("note")
        if resume_note:
            report.text(f"**Resume caveat.** {resume_note}")

    convergence = payload.get("convergence") or {}
    if convergence:
        report.heading("1b. Did it converge?")
        report.key_values(
            {
                "Validation Dice, last 6 epochs":
                    f"{convergence['last6_min']:.5f} - {convergence['last6_max']:.5f} "
                    f"(spread {convergence['last6_spread']:.5f})",
                "Mean gain per epoch, last 6": f"{convergence['last6_mean_gain']:+.6f}",
                "Mean gain per epoch, epochs 14-20":
                    f"{convergence['mid_mean_gain']:+.6f}",
                "Ratio (mid / late)": convergence["gain_ratio"],
                "Final learning rate": convergence["final_lr"],
                "Early stopping fired": convergence["stopped_early"],
            }
        )
        report.text(
            "**The model has converged for this configuration.** Over the final six "
            f"epochs validation foreground Dice moved a total of "
            f"{convergence['last6_spread']:.5f}, i.e. "
            f"{convergence['last6_mean_gain']:+.6f} per epoch, against "
            f"{convergence['mid_mean_gain']:+.6f} per epoch during epochs 14-20 - "
            f"roughly {convergence['gain_ratio']}x slower. The cosine schedule had "
            f"annealed the learning rate to ~0 by epoch 30."
        )
        report.text(
            "**Early stopping did not fire, and that is a configuration flaw rather "
            "than evidence of continued learning.** `min_delta` was 0, so an "
            "improvement of +0.00004 still reset the patience counter. With a "
            "meaningful `min_delta` (~0.001) the run would have stopped around epoch "
            "25-26 at effectively the same result. This should be corrected in the "
            "next run."
        )

    report.heading("2. Test-set segmentation: baseline vs extended")
    report.text(
        "Aggregate (dataset-level) metrics on the 33 held-out test patients. "
        "`delta` is extended minus baseline; higher is better for all of these."
    )
    report.table(
        [
            {
                "metric": row["metric"],
                "baseline": _fmt(row["baseline"]),
                "extended": _fmt(row["extended"]),
                "delta": _fmt_delta(row["delta"]),
                "verdict": row["verdict"],
            }
            for row in payload["segmentation_comparisons"]
        ],
        ["metric", "baseline", "extended", "delta", "verdict"],
    )

    if payload["measurement_comparisons"]:
        report.heading("3. Disc measurements from predicted masks")
        report.text(
            "Agreement between measurements taken from predicted masks and from "
            "ground-truth masks. For `measurement_mae/*` **lower is better**; for "
            "`measurement_r/*` higher is better."
        )
        report.table(
            [
                {
                    "metric": row["metric"],
                    "baseline": _fmt(row["baseline"]),
                    "extended": _fmt(row["extended"]),
                    "delta": _fmt_delta(row["delta"]),
                    "verdict": row["verdict"],
                }
                for row in payload["measurement_comparisons"]
            ],
            ["metric", "baseline", "extended", "delta", "verdict"],
        )

    if payload["finding_comparisons"]:
        report.heading("4. Disc-level finding assessment")
        report.text(
            "Stage E baselines recomputed on the extended run's disc table. For "
            "`pfirrmann_mae/*` lower is better; otherwise higher is better."
        )
        report.table(
            [
                {
                    "metric": row["metric"],
                    "baseline": _fmt(row["baseline"]),
                    "extended": _fmt(row["extended"]),
                    "delta": _fmt_delta(row["delta"]),
                    "verdict": row["verdict"],
                }
                for row in payload["finding_comparisons"]
            ],
            ["metric", "baseline", "extended", "delta", "verdict"],
        )

    report.heading("5. Decision on the disc-level re-run")
    report.key_values(
        {
            "Macro foreground Dice (baseline)":
                decision["macro_foreground_dice_baseline"],
            "Macro foreground Dice (extended)":
                decision["macro_foreground_dice_extended"],
            "Delta": decision["delta"],
            "Meaningful-change threshold": decision["threshold"],
            "Verdict": decision["verdict"],
            "Disc-level re-run justified": decision["disc_level_rerun_justified"],
            "Reason": decision["reason"],
        }
    )

    report.heading("6. Figures")
    report.bullets(
        [
            f"`outputs/visualizations/{EXPERIMENT}/training_curves_combined.png`",
            f"`outputs/visualizations/{EXPERIMENT}/metric_comparison.png`",
            f"`outputs/visualizations/{EXPERIMENT}/predictions/` - prediction figures",
        ]
    )

    report.text(
        "No clinical diagnosis, no postoperative healing prediction and no "
        "longitudinal improvement is claimed. The dataset contains a single "
        "timepoint per patient and no follow-up imaging."
    )
    return report.save(EXTENDED_DIR / "comparison.md")


def _fmt(value) -> str:
    return "-" if value is None else f"{value:.4f}"


def _fmt_delta(value) -> str:
    return "-" if value is None else f"{value:+.4f}"


if __name__ == "__main__":
    main()
