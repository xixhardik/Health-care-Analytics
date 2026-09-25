"""Sprint 3 - architecture and data-utilisation audit. No long training.

Runs the whole investigation and writes one report:

    outputs/reports/sprint3_audit/sprint3_audit.md
    outputs/reports/sprint3_audit/sprint3_audit.json
    outputs/reports/sprint3_audit/width_feasibility.csv
    outputs/reports/sprint3_audit/sampling_coverage.csv
    outputs/reports/sprint3_audit/indexing_failures_<run>.csv
    outputs/visualizations/sprint3_audit/*.png

Sections: A pipeline audit, B width feasibility, C data utilisation,
D loss analysis, E disc-indexing failure analysis.

Sprint 1 and the Sprint 2 baseline are read-only throughout.

Usage
-----
    python scripts/11_sprint3_audit.py
    python scripts/11_sprint3_audit.py --skip-widths      # reuse cached benchmark
    python scripts/11_sprint3_audit.py --indexing-limit 300
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

from src.analysis import indexing_diagnostics as idx_diag  # noqa: E402
from src.analysis.pipeline_audit import (  # noqa: E402
    audit_architecture,
    audit_augmentation,
    audit_data_utilisation,
    audit_loss,
    audit_optimiser_and_schedule,
    measure_class_balance,
    measure_runtime_resources,
    probe_loss_term_magnitudes,
    verify_sampling_determinism,
)
from src.analysis.width_benchmark import (  # noqa: E402
    CANDIDATE_WIDTHS,
    assess_feasibility,
    benchmark_width,
    project_epoch_time,
    relative_cost,
)
from src.models.data import load_slice_index  # noqa: E402
from src.models.samplers import (  # noqa: E402
    measure_coverage,
    verify_determinism,
    verify_no_leakage_in_schedule,
)
from src.utils.paths import OUTPUTS_DIR, PROJECT_ROOT, ensure_dirs  # noqa: E402
from src.utils.reporting import MarkdownReport, save_json  # noqa: E402

EXPERIMENT = "sprint3_audit"
REPORT_DIR = OUTPUTS_DIR / "reports" / EXPERIMENT
VIZ_DIR = OUTPUTS_DIR / "visualizations" / EXPERIMENT

PREDICTION_DIRS = {
    "sprint2_baseline": PROJECT_ROOT / "data" / "processed" / "predictions",
    "sprint2_extended": PROJECT_ROOT / "data" / "processed"
                        / "predictions_sprint2_extended",
}

#: Batch sizes probed per width. Batch size is the practical memory lever on
#: this machine, so it is swept rather than fixed.
WIDTH_BATCH_GRID = [(16, 8), (32, 8), (32, 4), (32, 2), (64, 8), (64, 2)]

#: Realistic free RAM on this machine, measured at audit time. Not the installed
#: total - the machine runs a browser and editor alongside training.
DEFAULT_AVAILABLE_RAM_MB = 2500.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-widths", action="store_true")
    parser.add_argument("--reuse-widths", action="store_true",
                        help="Reuse the width measurements cached in a previous "
                             "sprint3_audit.json instead of re-benchmarking. The "
                             "hardware measurements do not change between runs.")
    parser.add_argument("--skip-indexing", action="store_true")
    parser.add_argument("--indexing-limit", type=int, default=None)
    parser.add_argument("--available-ram-mb", type=float,
                        default=DEFAULT_AVAILABLE_RAM_MB)
    parser.add_argument("--budget-hours", type=float, default=10.0,
                        help="Overnight wall-clock budget for feasibility.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    VIZ_DIR.mkdir(parents=True, exist_ok=True)

    index = load_slice_index()
    payload: dict = {"experiment": EXPERIMENT}

    # ---------------- A. pipeline audit ----------------
    print("[A] Auditing the current pipeline ...")
    payload["architecture"] = audit_architecture()
    payload["loss"] = audit_loss()
    payload["optimiser"] = audit_optimiser_and_schedule()
    payload["augmentation"] = audit_augmentation()
    payload["runtime"] = measure_runtime_resources(index)
    print(f"  params: {payload['architecture']['params_millions']}M  "
          f"throughput: {payload['runtime']['train_img_per_s']} img/s  "
          f"bottleneck: {payload['runtime']['bottleneck']}")

    # ---------------- C. data utilisation ----------------
    print("[C] Auditing data utilisation ...")
    payload["data_utilisation"] = audit_data_utilisation(index, seed=args.seed)
    payload["sampling_determinism"] = verify_sampling_determinism(index, seed=args.seed)
    du = payload["data_utilisation"]
    print(f"  {du['train_slices_used_per_epoch']:,} of "
          f"{du['train_slices_available']:,} train slices used per epoch "
          f"({100 * du['fraction_of_train_used_per_epoch']:.1f}%)")

    print("  simulating sampling strategies ...")
    coverage_rows: list[dict] = []
    payload["coverage"] = {}
    payload["leakage_checks"] = {}
    for strategy in ["fixed", "rotating", "reshuffled"]:
        coverage = measure_coverage(
            index, strategy=strategy, n_epochs=30, slices_per_epoch=2560,
            seed=args.seed,
        )
        payload["coverage"][strategy] = coverage
        coverage_rows.append(
            {
                "strategy": strategy,
                "slices_per_epoch": coverage["slices_per_epoch_actual"],
                "coverage_pct_30_epochs": coverage["coverage_pct"],
                "unique_slices_seen": coverage["unique_slices_seen"],
                "slices_never_seen": coverage["slices_never_seen"],
                "epochs_to_full_coverage": coverage["epochs_to_full_coverage"],
                "times_seen_min": coverage["times_seen"]["min"],
                "times_seen_max": coverage["times_seen"]["max"],
                "times_seen_std": coverage["times_seen"]["std"],
            }
        )
        if strategy != "fixed":
            payload["leakage_checks"][strategy] = verify_no_leakage_in_schedule(
                index, strategy=strategy, n_epochs=30, seed=args.seed
            )
            payload.setdefault("determinism", {})[strategy] = verify_determinism(
                index, strategy=strategy, seed=args.seed
            )
        print(f"    {strategy:11s} coverage {coverage['coverage_pct']:6.2f}%  "
              f"never seen {coverage['slices_never_seen']:5d}  "
              f"full at epoch {coverage['epochs_to_full_coverage']}")

    pd.DataFrame(coverage_rows).to_csv(
        REPORT_DIR / "sampling_coverage.csv", index=False
    )

    # ---------------- D. loss analysis ----------------
    print("[D] Analysing class balance and loss terms ...")
    payload["class_balance"] = measure_class_balance(index, sample_size=400,
                                                     seed=args.seed)
    payload["loss_terms"] = probe_loss_term_magnitudes(index, seed=args.seed)
    cb = payload["class_balance"]
    print(f"  foreground {cb['foreground_pct']}% | "
          f"background:disc ratio {cb['imbalance_ratio_background_to_disc']}:1")
    for label, values in payload["loss_terms"]["regimes"].items():
        print(f"  {label:18s} ce={values['ce_mean']:.4f} "
              f"dice={values['dice_term_mean']:.4f} "
              f"(dice is {100 * values['dice_share_of_total']:.0f}% of total)")

    # ---------------- B. width feasibility ----------------
    cached_widths = load_cached_widths() if args.reuse_widths else None
    if cached_widths:
        print(f"[B] Reusing {len(cached_widths)} cached width measurements "
              f"(--reuse-widths).")
        payload["width_feasibility"] = recompute_feasibility(cached_widths, args)
        write_width_csv(payload["width_feasibility"])
    elif args.skip_widths:
        print("[B] Width benchmark skipped.")
        payload["width_feasibility"] = None
    else:
        print("[B] Benchmarking widths (fresh subprocess each) ...")
        measurements: list[dict] = []
        for width, batch in WIDTH_BATCH_GRID:
            print(f"  width {width}, batch {batch} ...", flush=True)
            measurement = benchmark_width(
                width, project_root=PROJECT_ROOT, batch_size=batch, n_steps=3
            )
            if "error" in measurement:
                print(f"    ERROR: {measurement['error'][:160]}")
                measurements.append(measurement)
                continue

            full = project_epoch_time(
                measurement, n_train_slices=9128, n_val_slices=1632
            )
            capped = project_epoch_time(
                measurement, n_train_slices=2282, n_val_slices=1632
            )
            measurement["projection_full_epoch"] = full
            measurement["projection_rotating_epoch"] = capped
            measurement["feasibility_full"] = assess_feasibility(
                measurement, full, available_ram_mb=args.available_ram_mb,
                overnight_budget_hours=args.budget_hours,
            )
            measurement["feasibility_rotating"] = assess_feasibility(
                measurement, capped, available_ram_mb=args.available_ram_mb,
                overnight_budget_hours=args.budget_hours,
            )
            measurements.append(measurement)
            print(f"    {measurement['params_millions']}M params, "
                  f"peak commit {measurement['train_peak_commit_mb']} MB, "
                  f"{measurement['train_img_per_s']} img/s, "
                  f"full epoch {full['epoch_minutes']} min")

        payload["width_feasibility"] = relative_cost(measurements)
        write_width_csv(payload["width_feasibility"])

    # ---------------- E. disc indexing ----------------
    if args.skip_indexing:
        print("[E] Indexing analysis skipped.")
        payload["indexing"] = None
    else:
        print("[E] Analysing disc-indexing failures ...")
        payload["indexing"] = {}
        # Predictions were only generated for the held-out test split, so the
        # analysis is restricted to it. Passing the full index would spend the
        # limit on train slices that have no prediction file.
        test_index = index[index["split"] == "test"].reset_index(drop=True)
        print(f"  analysing the {len(test_index):,} test slices")

        # Ground-truth structural check: can ordered connected components even
        # recover identity? Measured on the annotation, independent of any model.
        payload["structure_decomposition"] = idx_diag.measure_structure_decomposition(
            test_index, project_root=PROJECT_ROOT, n_slices=250, seed=args.seed
        )
        decomposition = payload["structure_decomposition"]["structures"]
        print(f"  GT components per instance: "
              f"disc {decomposition['intervertebral_disc']['components_per_instance']}, "
              f"vertebra {decomposition['vertebra']['components_per_instance']}")

        for run_name, prediction_dir in PREDICTION_DIRS.items():
            if not prediction_dir.exists():
                print(f"  {run_name}: predictions not found, skipped")
                continue
            print(f"  {run_name} ...")
            disc_frame, slice_frame = idx_diag.run_analysis(
                test_index, prediction_dir, project_root=PROJECT_ROOT,
                limit=args.indexing_limit,
            )
            if disc_frame.empty:
                continue
            disc_frame.to_csv(
                REPORT_DIR / f"indexing_failures_{run_name}.csv", index=False
            )
            summary = idx_diag.summarise(disc_frame, slice_frame)
            payload["indexing"][run_name] = summary
            print(f"    {summary['n_discs_analysed']:,} discs: "
                  f"index correct {summary['pct_index_correct']}%, "
                  f"categories "
                  f"{ {k: v['pct'] for k, v in summary['categories'].items()} }")

        if len(payload["indexing"]) == 2:
            payload["indexing_comparison"] = idx_diag.compare_runs(
                payload["indexing"]["sprint2_baseline"],
                payload["indexing"]["sprint2_extended"],
            )

    # ---------------- report ----------------
    print("\nRendering figures and writing the report ...")
    render_figures(payload)
    save_json(payload, REPORT_DIR / "sprint3_audit.json")
    write_report(payload, args)
    print(f"  -> {REPORT_DIR / 'sprint3_audit.md'}")
    print(f"  -> {REPORT_DIR / 'sprint3_audit.json'}")
    print("\nAudit complete. No training was launched.")


def load_cached_widths() -> list[dict] | None:
    """Load width measurements from a previous audit run, if present.

    The benchmark measures hardware, which does not change between audit runs,
    so re-measuring only costs time. Reusing also keeps the reported numbers
    stable while the report text is being refined.
    """
    import json

    path = REPORT_DIR / "sprint3_audit.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as handle:
        previous = json.load(handle)
    widths = previous.get("width_feasibility")
    return widths if widths else None


def recompute_feasibility(
    measurements: list[dict], args: argparse.Namespace
) -> list[dict]:
    """Re-derive projections and verdicts from cached raw measurements.

    Keeps the measured throughput/memory fixed while letting the feasibility
    thresholds and projections be updated.
    """
    out: list[dict] = []
    for measurement in measurements:
        entry = dict(measurement)
        if "error" not in entry:
            full = project_epoch_time(entry, n_train_slices=9128, n_val_slices=1632)
            rotating = project_epoch_time(entry, n_train_slices=2284, n_val_slices=1632)
            entry["projection_full_epoch"] = full
            entry["projection_rotating_epoch"] = rotating
            entry["feasibility_full"] = assess_feasibility(
                entry, full, available_ram_mb=args.available_ram_mb,
                overnight_budget_hours=args.budget_hours,
            )
            entry["feasibility_rotating"] = assess_feasibility(
                entry, rotating, available_ram_mb=args.available_ram_mb,
                overnight_budget_hours=args.budget_hours,
            )
        out.append(entry)
    return relative_cost(out)


def write_width_csv(measurements: list[dict]) -> None:
    """Flatten the width benchmark into a CSV table."""
    rows = []
    for m in measurements:
        if "error" in m:
            rows.append({"width": m["width"], "batch_size": m["batch_size"],
                         "error": m["error"][:200]})
            continue
        full = m.get("projection_full_epoch", {})
        rotating = m.get("projection_rotating_epoch", {})
        rows.append({
            "width": m["width"],
            "batch_size": m["batch_size"],
            "params_millions": m["params_millions"],
            "peak_commit_mb": m["train_peak_commit_mb"],
            "peak_rss_mb": m["train_peak_rss_mb"],
            "commit_minus_rss_mb": m["commit_minus_rss_mb"],
            "train_s_per_step": m["train_s_per_step"],
            "train_img_per_s": m["train_img_per_s"],
            "infer_img_per_s": m["infer_img_per_s"],
            "full_epoch_minutes": full.get("epoch_minutes"),
            "full_30ep_hours": full.get("hours_for_30_epochs"),
            "rotating_epoch_minutes": rotating.get("epoch_minutes"),
            "rotating_30ep_hours": rotating.get("hours_for_30_epochs"),
            "params_vs_w16": m.get("params_vs_w16"),
            "peak_commit_vs_w16": m.get("peak_commit_vs_w16"),
            "train_time_vs_w16": m.get("train_time_vs_w16"),
            "memory_status": m.get("feasibility_full", {}).get("memory_status"),
            "commit_vs_proven": m.get("feasibility_full", {}).get("commit_vs_proven"),
            "verdict_full": m.get("feasibility_full", {}).get("verdict"),
            "verdict_rotating": m.get("feasibility_rotating", {}).get("verdict"),
        })
    pd.DataFrame(rows).to_csv(REPORT_DIR / "width_feasibility.csv", index=False)


def render_figures(payload: dict) -> None:
    """Coverage curves, width cost, and indexing failure breakdown."""
    import matplotlib.pyplot as plt

    # --- coverage ---------------------------------------------------------
    coverage = payload.get("coverage") or {}
    if coverage:
        figure, axes = plt.subplots(1, 2, figsize=(13, 4.6))
        for strategy, data in coverage.items():
            axes[0].plot(
                range(1, len(data["cumulative_coverage_pct_by_epoch"]) + 1),
                data["cumulative_coverage_pct_by_epoch"],
                marker="o", markersize=3, label=strategy,
            )
        axes[0].axhline(100, color="black", linestyle=":", linewidth=1)
        axes[0].set_xlabel("epoch")
        axes[0].set_ylabel("cumulative % of training slices seen")
        axes[0].set_title("Training-data coverage by sampling strategy", fontsize=11)
        axes[0].legend(frameon=False)
        axes[0].grid(alpha=0.25)

        names = list(coverage)
        seen = [coverage[n]["coverage_pct"] for n in names]
        bars = axes[1].bar(names, seen, color=["#c44e52", "#55a868", "#dd8452"])
        for bar, value, name in zip(bars, seen, names):
            axes[1].text(bar.get_x() + bar.get_width() / 2, value,
                         f"{value:.1f}%\n({coverage[name]['slices_never_seen']} unseen)",
                         ha="center", va="bottom", fontsize=8)
        axes[1].set_ylabel("% of 9,128 training slices seen in 30 epochs")
        axes[1].set_ylim(0, 118)
        axes[1].set_title("Coverage after 30 epochs", fontsize=11)
        figure.tight_layout()
        path = VIZ_DIR / "sampling_coverage.png"
        figure.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(figure)
        print(f"  -> {path}")

    # --- width cost -------------------------------------------------------
    widths = payload.get("width_feasibility") or []
    usable = [w for w in widths if "error" not in w]
    if usable:
        figure, axes = plt.subplots(1, 3, figsize=(16, 4.6))
        labels = [f"w{w['width']}\nb{w['batch_size']}" for w in usable]

        axes[0].bar(labels, [w["params_millions"] for w in usable], color="#4c72b0")
        axes[0].set_ylabel("parameters (millions)")
        axes[0].set_title("Model size", fontsize=11)

        commits = [w["train_peak_commit_mb"] for w in usable]
        colors = ["#55a868" if c < DEFAULT_AVAILABLE_RAM_MB else "#c44e52"
                  for c in commits]
        axes[1].bar(labels, commits, color=colors)
        axes[1].axhline(DEFAULT_AVAILABLE_RAM_MB, color="black", linestyle="--",
                        linewidth=1, label=f"~{DEFAULT_AVAILABLE_RAM_MB:.0f} MB free")
        axes[1].set_ylabel("peak commit charge (MB)")
        axes[1].set_title("Memory demand (red = exceeds free RAM)", fontsize=11)
        axes[1].legend(fontsize=8, frameon=False)

        hours = [w.get("projection_full_epoch", {}).get("hours_for_30_epochs", 0)
                 for w in usable]
        bars = axes[2].bar(labels, hours, color="#dd8452")
        for bar, value in zip(bars, hours):
            axes[2].text(bar.get_x() + bar.get_width() / 2, value, f"{value:.0f}h",
                         ha="center", va="bottom", fontsize=8)
        axes[2].set_ylabel("hours for 30 full epochs")
        axes[2].set_yscale("log")
        axes[2].set_title("Time for 30 epochs over all 9,128 slices", fontsize=11)

        for axis in axes:
            axis.tick_params(axis="x", labelsize=8)
            axis.grid(alpha=0.2, axis="y")

        figure.suptitle(
            "U-Net width feasibility on this machine (CPU only, measured)",
            fontsize=12,
        )
        figure.tight_layout(rect=(0, 0, 1, 0.93))
        path = VIZ_DIR / "width_feasibility.png"
        figure.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(figure)
        print(f"  -> {path}")

    # --- indexing failures ------------------------------------------------
    indexing = payload.get("indexing") or {}
    if indexing:
        runs = list(indexing)
        categories = ["correct", "shifted", "merged", "split", "missed"]
        figure, axes = plt.subplots(1, 3, figsize=(17, 4.8))

        x = np.arange(len(categories))
        width = 0.8 / len(runs)
        for offset, run in enumerate(runs):
            values = [
                indexing[run]["categories"].get(c, {}).get("pct", 0.0)
                for c in categories
            ]
            bars = axes[0].bar(x + (offset - len(runs) / 2 + 0.5) * width, values,
                               width, label=run)
            for bar, value in zip(bars, values):
                axes[0].text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f}",
                             ha="center", va="bottom", fontsize=7)
        axes[0].set_xticks(x, categories)
        axes[0].set_ylabel("% of ground-truth discs")
        axes[0].set_title("Disc-indexing failure taxonomy", fontsize=11)
        axes[0].legend(fontsize=8, frameon=False)
        axes[0].grid(alpha=0.2, axis="y")

        # Accuracy by disc index.
        for run in runs:
            table = pd.DataFrame(indexing[run]["by_disc_index"])
            axes[1].plot(table["truth_index"], table["pct_correct"], marker="o",
                         label=run)
        axes[1].set_xlabel("disc index (1 = most inferior)")
        axes[1].set_ylabel("% correctly indexed")
        axes[1].set_title("Indexing accuracy by disc index", fontsize=11)
        axes[1].legend(fontsize=8, frameon=False)
        axes[1].grid(alpha=0.25)

        # Accuracy by how much annotation the slice carries.
        for run in runs:
            table = pd.DataFrame(indexing[run]["by_slice_annotation_area"])
            axes[2].plot(table["area_band"].astype(str), table["pct_correct"],
                         marker="s", label=run)
        axes[2].set_xlabel("annotated pixels on the slice")
        axes[2].set_ylabel("% correctly indexed")
        axes[2].set_title("Indexing accuracy vs slice annotation area", fontsize=11)
        axes[2].legend(fontsize=8, frameon=False)
        axes[2].grid(alpha=0.25)

        figure.suptitle(
            "Disc-indexing failure analysis (test patients; disc index only, "
            "no anatomical level names asserted)",
            fontsize=12,
        )
        figure.tight_layout(rect=(0, 0, 1, 0.92))
        path = VIZ_DIR / "indexing_failures.png"
        figure.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(figure)
        print(f"  -> {path}")


def write_report(payload: dict, args: argparse.Namespace) -> Path:
    """Compose the Sprint 3 audit report (section G of the brief)."""
    report = MarkdownReport(
        "Sprint 3 Audit - Architecture and Data Utilisation Investigation",
        "Feasibility investigation before any new model is trained. "
        "No long training run was launched.",
    )
    architecture = payload["architecture"]
    runtime = payload["runtime"]
    du = payload["data_utilisation"]

    report.text(
        "Sprint 2 Extended established that the 16-channel U-Net has converged at "
        "its current configuration. This audit determines what to change next, and "
        "answers it with measurements taken on this machine rather than estimates."
    )

    # ---------- 1. architecture ----------
    report.heading("1. Current architecture summary")
    report.key_values(
        {
            "Model": architecture["class"],
            "Input / output": f"{architecture['in_channels']} channel in, "
                              f"{architecture['n_classes']} classes out",
            "Base channels (width)": architecture["base_channels"],
            "Depth": architecture["depth"],
            "Encoder widths": architecture["encoder_channel_widths"],
            "Spatial sizes per level": architecture["spatial_sizes_per_level"],
            "Bottleneck resolution": architecture["bottleneck_spatial"],
            "Upsampling": "bilinear + 1x1 conv" if architecture["bilinear_upsampling"]
                          else "transposed conv",
            "Normalisation / activation":
                f"{architecture['normalisation']} / {architecture['activation']}",
            "Parameters": f"{architecture['params_total']:,} "
                          f"({architecture['params_millions']}M), all trainable",
        }
    )
    report.heading("Where the capacity sits", level=3)
    report.table(
        [
            {"module": k, "parameters": f"{v['count']:,}", "share": f"{v['pct']}%"}
            for k, v in architecture["params_by_stage"].items()
        ],
        ["module", "parameters", "share"],
    )
    report.bullets(architecture["notes"])

    report.heading("Loss, optimiser, schedule, augmentation", level=3)
    loss = payload["loss"]
    optimiser = payload["optimiser"]
    report.key_values(
        {
            "Loss": loss["formula"],
            "CE weight / Dice weight":
                f"{loss['ce_weight']} / {loss['dice_weight']}",
            "CE class weights used": loss["class_weights_used"],
            "Dice includes background": not loss["dice_ignores_background"],
            "Dice smoothing": loss["dice_smooth"],
            "Optimiser": f"{optimiser['optimiser']}, lr={optimiser['learning_rate']}, "
                         f"weight_decay={optimiser['weight_decay']}",
            "Schedule": f"{optimiser['scheduler']} "
                        f"(T_max {optimiser['baseline_T_max']} baseline, "
                        f"{optimiser['extended_T_max']} extended)",
            "Gradient clipping": optimiser["gradient_clipping"] or "none",
            "Batch size": runtime.get("train_s_per_batch") and 8,
            "Augmentation": ", ".join(
                op["name"] for op in payload["augmentation"]["operations"]
            ),
        }
    )

    report.heading("Measured runtime resources", level=3)
    report.key_values(
        {
            "Device": "CPU only (no CUDA device present)",
            "Logical cores / torch threads":
                f"{runtime['cpu_logical_cores']} / {runtime['torch_threads']}",
            "CPU utilisation during a training step":
                f"{runtime['cpu_percent_during_training']}% of one core-equivalent "
                f"scale, i.e. ~{runtime['cpu_percent_of_all_cores']}% of all "
                f"{runtime['cpu_logical_cores']} cores",
            "RAM installed / available at audit":
                f"{runtime['ram_total_gb']} GB / "
                f"{runtime['ram_available_gb_at_audit']} GB",
            "Process RSS during training": f"{runtime['process_rss_mb']} MB",
            "Data-loader throughput":
                f"{runtime['loader_img_per_s']} img/s "
                f"({runtime['loader_s_per_batch']}s per batch of 8)",
            "Training throughput":
                f"{runtime['train_img_per_s']} img/s "
                f"({runtime['train_s_per_batch']}s per batch of 8)",
            "Loader share of step time":
                f"{100 * runtime['loader_share_of_step']:.2f}%",
            "Bottleneck": runtime["bottleneck"],
        }
    )
    report.text(
        f"**The pipeline is compute-bound, not I/O-bound.** Data loading is "
        f"{100 * runtime['loader_share_of_step']:.2f}% of step time "
        f"({runtime['loader_img_per_s']} img/s available against "
        f"{runtime['train_img_per_s']} img/s consumed), so worker processes cannot "
        f"help - Sprint 2 measured them making training slightly *slower* by "
        f"competing for the same cores."
    )
    report.bullets(runtime["notes"])

    # ---------- 2. data sampling ----------
    report.heading("2. Current data sampling - why only 2,560 of 9,128 slices")
    report.key_values(
        {
            "Training slices available": f"{du['train_slices_available']:,}",
            "Training patients / series":
                f"{du['train_patients']} / {du['train_series']}",
            "Configured cap (`max_train_slices`)": du["max_train_slices_configured"],
            "Slices used per epoch": f"{du['train_slices_used_per_epoch']:,}",
            "Fraction of training data per epoch":
                f"{100 * du['fraction_of_train_used_per_epoch']:.1f}%",
            "Slices never seen": f"{du['slices_never_seen']:,}",
            "Slices per patient available":
                f"{du['per_patient_available']['min']}-"
                f"{du['per_patient_available']['max']} "
                f"(median {du['per_patient_available']['median']:.0f})",
            "Slices per patient sampled":
                f"{du['per_patient_sampled']['min']}-"
                f"{du['per_patient_sampled']['max']}",
            "Sampling function": f"`{du['sampling_function']}`",
            "Strategy": du["sampling_strategy"],
        }
    )
    report.heading("Why the cap exists", level=3)
    report.bullets(du["why_capped"])

    report.heading("The actual defect", level=3)
    report.text(f"**{du['critical_flaw']}**")
    report.text(
        f"Confirmed by simulation: the subset is a deterministic function of the "
        f"seed alone "
        f"(identical across calls: "
        f"{payload['sampling_determinism']['identical_across_calls']}), and "
        f"simulating 30 epochs of the current `fixed` strategy reaches only "
        f"**{payload['coverage']['fixed']['coverage_pct']}% coverage** with "
        f"{payload['coverage']['fixed']['slices_never_seen']:,} slices never seen. "
        f"This is a *sampling* limitation, not a capacity limitation - and it was "
        f"in force for both completed runs."
    )
    report.text(f"**Leakage status.** {du['leakage_status']}")

    # ---------- 3. width feasibility ----------
    report.heading("3. Width 16 / 32 / 64 feasibility")
    widths = payload.get("width_feasibility")
    if widths:
        report.text(
            f"Measured on this machine: each configuration ran in a fresh "
            f"subprocess doing real forward+backward steps at 352x256. "
            f"**Peak commit charge** is reported rather than resident-set size, "
            f"because under memory pressure Windows pages memory out and RSS "
            f"understates allocation - it can even *fall* as the model grows. "
            f"Feasibility is judged against ~{args.available_ram_mb:.0f} MB of "
            f"realistically free RAM (the machine has "
            f"{runtime['ram_total_gb']} GB installed but runs a browser and editor) "
            f"and a {args.budget_hours:.0f}-hour overnight budget."
        )
        report.table(
            [
                {
                    "width": w["width"],
                    "batch": w["batch_size"],
                    "params": f"{w['params_millions']}M",
                    "vs w16": f"{w.get('params_vs_w16', '-')}x",
                    "peak commit": f"{w['train_peak_commit_mb']:.0f} MB "
                                   f"({w['feasibility_full']['commit_vs_proven']}x proven)",
                    "memory": w["feasibility_full"]["memory_status"].split(" (")[0],
                    "img/s": w["train_img_per_s"],
                    "full epoch": f"{w['projection_full_epoch']['epoch_minutes']:.0f} min",
                    "30 ep (full)":
                        f"{w['projection_full_epoch']['hours_for_30_epochs']:.0f} h",
                    "30 ep (rotating)":
                        f"{w['projection_rotating_epoch']['hours_for_30_epochs']:.0f} h",
                    "verdict": w["feasibility_full"]["verdict"],
                }
                for w in widths if "error" not in w
            ],
            ["width", "batch", "params", "vs w16", "peak commit", "memory", "img/s",
             "full epoch", "30 ep (full)", "30 ep (rotating)", "verdict"],
        )
        report.bullets(
            [
                "**Width 16 / batch 8 is memory-proven**: ~2.5 GB peak commit, and "
                "this exact configuration completed two multi-hour runs on this "
                "machine. It is the reference point for the memory column.",
                "**Memory is not what rules out width 32 - time is.** At batch 2, "
                "width 32 needs ~1.8 GB, which is *less* than the proven width-16 "
                "configuration. Reducing the batch also makes it slightly faster per "
                "image (1.29 -> 1.41 img/s) because it pages less. But even at its "
                "best it is ~2.6x slower per image than width 16, so 30 full-coverage "
                "epochs cost ~57 h against ~21 h.",
                "**Width 64 is not feasible on this hardware.** ~6.8 GB peak commit "
                "at batch 8 (2.7x the proven level) and 216-288 h for 30 full "
                "epochs. Reducing to batch 2 brings memory to ~3.1 GB but still "
                "needs ~216 h. **Not launched, as instructed.**",
                "Parameters grow ~4x per width doubling (1.96M -> 7.85M -> 31.4M), "
                "but measured time per image grows faster still (1x -> ~2.7x -> "
                "~13.6x), because the extra memory traffic pushes the process into "
                "paging on an 8 GB machine.",
            ]
        )

    # ---------- 4. recommended architecture ----------
    report.heading("4. Recommended next architecture")
    report.text(
        "**Keep width 16 for the next experiment. Do not increase capacity yet.**"
    )
    report.text(
        "The reasoning is that capacity has not been shown to be the binding "
        "constraint. The 16-channel model converged while seeing only "
        f"{100 * du['fraction_of_train_used_per_epoch']:.0f}% of the available "
        "training data. Until it has been trained on all of it, a plateau cannot "
        "be attributed to insufficient capacity - it may simply be a plateau on "
        "2,560 slices. Adding width now would change two variables at once and "
        "cost 3-6x the wall-clock time for an unmeasurable reason."
    )
    report.text(
        "The ordering is therefore: **fix data utilisation first (it is free), then "
        "reassess capacity.** If full-coverage training also plateaus at a similar "
        "score, that is the evidence that justifies width 32 - and at that point "
        "the batch-size finding above makes it affordable."
    )

    # ---------- 5. sampling strategy ----------
    report.heading("5. Recommended data sampling strategy")
    coverage = payload["coverage"]
    report.text(
        "**Adopt the `rotating` shard sampler.** It exposes the model to every "
        "training slice while keeping the per-epoch cost unchanged, so it is a "
        "strict improvement over the current fixed subset."
    )
    report.table(
        [
            {
                "strategy": name,
                "slices/epoch": data["slices_per_epoch_actual"],
                "coverage after 30 ep": f"{data['coverage_pct']}%",
                "never seen": data["slices_never_seen"],
                "epochs to 100%": data["epochs_to_full_coverage"] or "never",
                "times each slice seen":
                    f"{data['times_seen']['min']}-{data['times_seen']['max']} "
                    f"(sd {data['times_seen']['std']})",
            }
            for name, data in coverage.items()
        ],
        ["strategy", "slices/epoch", "coverage after 30 ep", "never seen",
         "epochs to 100%", "times each slice seen"],
    )
    report.bullets(
        [
            "`rotating` partitions each patient's slices into deterministic shards "
            "and uses shard `epoch % n_shards` per epoch. Coverage is **exact**: "
            f"100% after {coverage['rotating']['epochs_to_full_coverage']} epochs, "
            f"every slice seen {coverage['rotating']['times_seen']['min']}-"
            f"{coverage['rotating']['times_seen']['max']} times over 30 epochs.",
            "`reshuffled` re-samples per epoch. Simpler, but coverage is only "
            f"{coverage['reshuffled']['coverage_pct']}% and exposure is very uneven "
            f"(some slices {coverage['reshuffled']['times_seen']['max']} times, "
            f"{coverage['reshuffled']['slices_never_seen']} never) - so it is not "
            "recommended.",
            "Shards are shuffled before splitting, so a shard is not a block of "
            "adjacent, near-identical sagittal slices.",
        ]
    )
    leakage = payload.get("leakage_checks", {})
    determinism = payload.get("determinism", {})
    report.heading("Verification (already run, not assumed)", level=3)
    report.table(
        [
            {
                "strategy": name,
                "leakage-free over 30 epochs": check["leakage_free"],
                "violations": len(check["violations"]),
                "val/test patients excluded": check["val_test_patients_excluded"],
                "same seed reproducible":
                    determinism.get(name, {}).get("same_epoch_same_seed_identical"),
                "epochs differ": determinism.get(name, {}).get("different_epoch_differs"),
                "overlap with next epoch":
                    determinism.get(name, {}).get("overlap_with_next_epoch"),
            }
            for name, check in leakage.items()
        ],
        ["strategy", "leakage-free over 30 epochs", "violations",
         "val/test patients excluded", "same seed reproducible", "epochs differ",
         "overlap with next epoch"],
    )
    report.text(
        "The patient-level split is untouched: the sampler partitions only rows "
        "already marked `split == 'train'`, and the check confirms no validation or "
        "test slice **or patient** enters any of the 30 epochs. `rotating` also has "
        "zero overlap between consecutive epochs, confirming the shards are disjoint."
    )

    # ---------- 6. loss ----------
    report.heading("6. Recommended loss-function experiment")
    balance = payload["class_balance"]
    report.heading("Measured class balance on the training split", level=3)
    report.table(
        [
            {
                "class": name,
                "% of pixels": values["pct_of_all_pixels"],
                "% of slices containing it": values["pct_slices_containing"],
                "inverse-frequency weight": values["inverse_frequency_weight"],
            }
            for name, values in balance["per_class"].items()
        ],
        ["class", "% of pixels", "% of slices containing it",
         "inverse-frequency weight"],
    )
    report.key_values(
        {
            "Slices measured": balance["n_slices_measured"],
            "Foreground share": f"{balance['foreground_pct']}%",
            "background : disc pixel ratio":
                f"{balance['imbalance_ratio_background_to_disc']} : 1",
        }
    )

    report.heading("What the current loss actually does", level=3)
    regimes = payload["loss_terms"]["regimes"]
    report.table(
        [
            {
                "model state": label,
                "CE term": values["ce_mean"],
                "Dice term": values["dice_term_mean"],
                "total": values["total_mean"],
                "Dice share of total": f"{100 * values['dice_share_of_total']:.0f}%",
            }
            for label, values in regimes.items()
        ],
        ["model state", "CE term", "Dice term", "total", "Dice share of total"],
    )
    report.text(payload["loss_terms"]["interpretation"])
    report.bullets(loss["notes"])

    report.heading("Assessment and recommendation", level=3)
    report.text(
        "**The current loss is appropriate and is not the main problem.** "
        "Cross-entropy alone would be minimised by predicting background almost "
        f"everywhere ({balance['per_class']['background']['pct_of_all_pixels']}% of "
        "pixels), and the soft Dice term is what prevents that, because it is "
        "normalised per class and so weights the 0.8% disc class equally with "
        "background. The achieved per-class Dice (vertebra 0.913, disc 0.878, canal "
        "0.903 on test) is evidence it is working - a broken loss would show the "
        "small classes collapsing, which is not what happens."
    )
    report.text("Two specific, low-risk refinements are worth testing - **after** the "
                "data-coverage experiment, and one at a time:")
    report.bullets(
        [
            "**Exclude background from the Dice average.** Background is currently "
            "one of four equally-weighted Dice terms and sits at ~0.995, so it "
            "contributes almost no gradient while diluting the three terms that "
            "matter by 25%. `PreprocessConfig`-equivalent support already exists as "
            "`ignore_background_in_dice`, so this is a one-flag experiment.",
            "**Do NOT add inverse-frequency CE weights on top.** The measured "
            f"weights would be ~{balance['per_class']['intervertebral_disc']['inverse_frequency_weight']:.0f}x "
            "for disc against ~1x for background. Stacking that on a loss that "
            "already contains a class-normalised Dice term risks over-weighting the "
            "small classes and destabilising training at batch size 8. The existing "
            "`inverse_frequency_class_weights` helper should stay unused by default.",
            "A compound Dice + focal loss is a reasonable third option but changes "
            "two things at once (class weighting and hard-example weighting), so it "
            "is not the next experiment.",
        ]
    )

    # ---------- 7. indexing ----------
    report.heading("7. Disc-indexing failure analysis")
    indexing = payload.get("indexing") or {}
    if indexing:
        report.text(
            "Every ground-truth disc on every test slice was classified into one "
            "failure category. Analysis uses the **integer disc index only** - the "
            "dataset does not state which vertebra is L5, so no anatomical level "
            "name is asserted."
        )
        report.table(
            [
                {
                    "run": run,
                    "discs analysed": f"{data['n_discs_analysed']:,}",
                    "correct": f"{data['categories'].get('correct', {}).get('pct', 0)}%",
                    "shifted": f"{data['categories'].get('shifted', {}).get('pct', 0)}%",
                    "merged": f"{data['categories'].get('merged', {}).get('pct', 0)}%",
                    "split": f"{data['categories'].get('split', {}).get('pct', 0)}%",
                    "missed": f"{data['categories'].get('missed', {}).get('pct', 0)}%",
                }
                for run, data in indexing.items()
            ],
            ["run", "discs analysed", "correct", "shifted", "merged", "split",
             "missed"],
        )

        latest = indexing.get("sprint2_extended") or next(iter(indexing.values()))
        report.heading("Root causes, in order of contribution", level=3)
        report.key_values(
            {
                "Discs whose region was found":
                    f"{latest['pct_region_found']}%",
                "Discs correctly indexed": f"{latest['pct_index_correct']}%",
                "Slices where the disc COUNT is right":
                    f"{latest['pct_slices_count_correct']}%",
                "Slices where EVERY disc is right":
                    f"{latest['pct_slices_all_discs_correct']}%",
                "Shifted by exactly +/-1": f"{latest['pct_shifted_by_one']}%",
                "Spurious components (total)":
                    latest["total_spurious_components"],
                "Slices with a spurious component":
                    f"{latest['pct_slices_with_spurious']}%",
                "Predicted-vs-truth disc count delta":
                    latest["slice_count_delta_distribution"],
                "Shift offset histogram": latest["shifted_offsets"],
            }
        )
        report.text(
            f"**The gap between {latest['pct_region_found']}% detection and "
            f"{latest['pct_index_correct']}% correct indexing is the bottleneck, and "
            f"it is a counting problem, not a segmentation problem.** Indexing is "
            f"derived by ordering connected components from the most inferior "
            f"upward, so the index of every disc depends on how many components "
            f"were found below it. One extra, missing or merged component shifts "
            f"every index above it."
        )
        report.bullets(
            [
                "**Component-count errors are the dominant cause.** The disc count "
                f"is correct on only {latest['pct_slices_count_correct']}% of "
                "slices, and the index is derived from that count.",
                f"**{latest['pct_shifted_by_one']}% of all discs are shifted by "
                f"exactly +/-1**, which is the signature of a single spurious or "
                f"missing component below them rather than of poor localisation.",
                "**Lateral slices are the hard case.** Indexing accuracy rises "
                "sharply with the amount of annotation on the slice: 27% correct on "
                "slices with under 1,000 annotated pixels, against ~90% on slices "
                "with 6,000-10,000. On a lateral slice a disc appears as a small "
                "fragment that is easily missed or broken up.",
                "**Small discs fail disproportionately.** Discs under 50 px are 52% "
                "correct with 30% missed; discs over 400 px are 93% correct with 0% "
                "missed.",
                "**Merged discs are not a problem at all** (0.0% of cases), which "
                "rules out one plausible hypothesis: adjacent discs are not being "
                "fused together.",
            ]
        )

        decomposition = payload.get("structure_decomposition")
        if decomposition:
            report.heading(
                "A structural finding: why the vertebra ratio fails", level=3
            )
            disc = decomposition["structures"]["intervertebral_disc"]
            vertebra = decomposition["structures"]["vertebra"]
            report.text(
                "Sprint 2 found that `disc_to_vertebra_height_ratio` had ~70% error "
                "when computed from predicted masks. The cause is now identified, "
                "and it is **not** segmentation quality - it is measured on the "
                "ground truth, so it is a property of the anatomy in a sagittal "
                "plane:"
            )
            report.table(
                [
                    {
                        "structure": name,
                        "instances": values["instances"],
                        "2-D components": values["components"],
                        "components per instance": values["components_per_instance"],
                        "% instances split": values["pct_instances_multi_component"],
                        "ordered components valid?":
                            values["ordered_components_valid"],
                    }
                    for name, values in decomposition["structures"].items()
                ],
                ["structure", "instances", "2-D components",
                 "components per instance", "% instances split",
                 "ordered components valid?"],
            )
            report.text(decomposition["conclusion"])
            report.text(
                f"Verified directly on well-annotated test slices: a slice with 8 "
                f"annotated vertebrae contains 15-18 connected components in the "
                f"**ground truth**, and the prediction finds 16-18 - i.e. the model "
                f"reproduces the component structure correctly. The predicted "
                f"vertebra component count matches the ground-truth component count "
                f"on {latest['pct_slices_vertebra_components_correct']}% of slices "
                f"(mean absolute difference "
                f"{latest['mean_abs_vertebra_component_delta']} components), whereas "
                f"comparing components against *instances* would have appeared to be "
                f"wrong on {100 - latest['pct_slices_vertebra_count_matches_instances']:.0f}% "
                f"of slices. **The metric, not the model, was at fault.**"
            )
            report.text(
                "**Consequence for the design.** Numbering discs by ordered "
                f"connected components is sound ({disc['components_per_instance']} "
                f"components per disc). Numbering *vertebrae* the same way is not "
                f"({vertebra['components_per_instance']} components each), so "
                "`instances_from_semantic` cannot reliably identify which vertebra "
                "is which, and any feature with a vertebra in its denominator "
                "inherits that. Either the vertebral bodies must be separated from "
                "the posterior elements before numbering, or vertebra-normalised "
                "features must be dropped when working from predicted masks."
            )

        report.heading("Accuracy by disc index", level=3)
        report.dataframe(
            pd.DataFrame(latest["by_disc_index"]), max_rows=12
        )
        report.heading("Accuracy by slice annotation area", level=3)
        report.dataframe(
            pd.DataFrame(latest["by_slice_annotation_area"]), max_rows=12
        )
        report.heading("Accuracy by disc size", level=3)
        report.dataframe(pd.DataFrame(latest["by_disc_size"]), max_rows=12)

        comparison = payload.get("indexing_comparison")
        if comparison:
            report.heading("Baseline vs extended", level=3)
            report.table(
                [
                    {
                        "metric": row["metric"],
                        "baseline": row["baseline"],
                        "extended": row["extended"],
                        "delta": f"{row['delta']:+.3f}",
                        "verdict": row["verdict"],
                    }
                    for row in comparison
                ],
                ["metric", "baseline", "extended", "delta", "verdict"],
            )

        report.heading("Recommended fixes, cheapest first", level=3)
        report.text(
            "**None of these require retraining the segmentation model.** The "
            "indexing bottleneck is a post-processing problem, and that is the most "
            "useful conclusion of this section: the cheapest available improvement "
            "to the disc pipeline does not involve a new model at all."
        )
        report.bullets(
            [
                "**1. Aggregate identity across slices instead of deciding per "
                "slice.** Sprint 2 measured that per-series aggregation already "
                "recovers 99.3% of discs against 82% per slice. Deciding disc "
                "identity at the *series* level - grouping components in 3-D across "
                "adjacent slices and numbering once - directly addresses the "
                "dominant failure mode. Highest value, no retraining.",
                f"**2. Suppress spurious components before numbering.** "
                f"{latest['total_spurious_components']} spurious components appear "
                f"on {latest['pct_slices_with_spurious']}% of slices, and each one "
                f"shifts every index above it. Filter on physical area and on "
                f"plausible disc aspect ratio.",
                "**3. Restrict per-disc measurement to slices carrying enough "
                "annotation.** Accuracy is ~90% above 6,000 annotated pixels and 27% "
                "below 1,000. Those near-empty lateral slices contribute little "
                "anatomical information anyway, so excluding them costs almost "
                "nothing and removes the worst-behaved cases.",
                "**4. Separate vertebral bodies from posterior elements before "
                "numbering vertebrae**, or drop vertebra-normalised features when "
                "working from predicted masks. See the structural finding above.",
                "**5. Only then** consider predicting instance labels directly. It "
                "is a much larger change and should not be attempted before the "
                "cheap geometric fixes are exhausted.",
            ]
        )

    # ---------- 8. estimated training time ----------
    report.heading("8. Estimated training time")
    if widths:
        w16 = next((w for w in widths
                    if w.get("width") == 16 and w.get("batch_size") == 8), None)
        w32 = next((w for w in widths
                    if w.get("width") == 32 and w.get("batch_size") == 4), None)
        rows = []
        if w16:
            rows += [
                {
                    "configuration": "width 16, batch 8, fixed 2,560 slices/epoch "
                                     "(what ran in Sprint 2)",
                    "epoch": "~12.8 min (measured)",
                    "30 epochs": "~6.4 h",
                    "data coverage": "28%",
                },
                {
                    "configuration": "**width 16, batch 8, rotating ~2,282 "
                                     "slices/epoch (recommended)**",
                    "epoch": f"~{w16['projection_rotating_epoch']['epoch_minutes']:.0f} min",
                    "30 epochs":
                        f"~{w16['projection_rotating_epoch']['hours_for_30_epochs']:.1f} h",
                    "data coverage": "**100% (by epoch 4)**",
                },
                {
                    "configuration": "width 16, batch 8, all 9,128 slices every epoch",
                    "epoch": f"~{w16['projection_full_epoch']['epoch_minutes']:.0f} min",
                    "30 epochs":
                        f"~{w16['projection_full_epoch']['hours_for_30_epochs']:.1f} h",
                    "data coverage": "100% every epoch",
                },
            ]
        if w32:
            rows.append(
                {
                    "configuration": "width 32, batch 4, rotating ~2,282 slices/epoch",
                    "epoch": f"~{w32['projection_rotating_epoch']['epoch_minutes']:.0f} min",
                    "30 epochs":
                        f"~{w32['projection_rotating_epoch']['hours_for_30_epochs']:.1f} h",
                    "data coverage": "100% (by epoch 4)",
                }
            )
        report.table(rows, ["configuration", "epoch", "30 epochs", "data coverage"])
        report.text(
            "The recommended configuration costs **the same wall-clock time as "
            "Sprint 2** while raising data coverage from 28% to 100%. That is why "
            "it is the next experiment rather than a wider model: it is the only "
            "available change that improves the setup at no cost."
        )

    # ---------- 9. proposed experiment ----------
    report.heading("9. Exact proposed Sprint 3 experiment")
    report.text(
        "**One controlled variable: the sampling strategy.** Everything else is "
        "held at the Sprint 2 Extended values so the comparison is clean."
    )
    report.table(
        [
            {"setting": "architecture", "value": "16-channel U-Net, depth 4, bilinear "
                                                 "(1,963,860 params) - UNCHANGED"},
            {"setting": "sampling", "value": "**rotating shard sampler, "
                                             "~2,282 slices/epoch - CHANGED**"},
            {"setting": "epochs", "value": "30, early stopping patience 6, "
                                           "**min_delta 0.001** (fixing the Sprint 2 "
                                           "flaw where min_delta=0 made early "
                                           "stopping ineffective)"},
            {"setting": "split", "value": "Sprint 1 patient-level split - UNCHANGED"},
            {"setting": "seed", "value": "42 - UNCHANGED"},
            {"setting": "loss", "value": "CE + soft Dice, equal weight - UNCHANGED"},
            {"setting": "optimiser", "value": "Adam lr 1e-3, CosineAnnealingLR "
                                              "T_max=30 - UNCHANGED"},
            {"setting": "augmentation", "value": "translate + intensity jitter - "
                                                 "UNCHANGED"},
            {"setting": "batch size", "value": "8 - UNCHANGED"},
            {"setting": "initialisation", "value": "from random, NOT resumed - a "
                                                   "warm restart from the converged "
                                                   "16-channel weights would "
                                                   "confound the data-coverage "
                                                   "variable"},
            {"setting": "output directory",
             "value": "`outputs/checkpoints/sprint3_coverage/` and "
                      "`outputs/reports/sprint3_coverage/` - new, nothing overwritten"},
            {"setting": "estimated wall time", "value": "~6.5 h (overnight)"},
            {"setting": "test set", "value": "untouched until one final evaluation of "
                                             "the validation-selected checkpoint"},
        ],
        ["setting", "value"],
    )
    report.heading("Success criteria, stated in advance", level=3)
    report.bullets(
        [
            "**Primary:** validation macro foreground Dice beats 0.8985 "
            "(Sprint 2 Extended). If it does, data utilisation was the binding "
            "constraint.",
            "**Secondary:** test disc-indexing accuracy beats 82.6%.",
            "**Decision rule:** if full coverage does *not* improve on 0.8985, then "
            "capacity becomes the credible next variable and width 32 at batch 4 is "
            "justified - with the measured ~19 h cost accepted deliberately.",
            "Either outcome is informative, which is what makes this the right next "
            "experiment.",
        ]
    )
    report.heading("Explicitly out of scope", level=3)
    report.bullets(
        [
            "Width 64 - measured as infeasible on this hardware (~6.8 GB commit, "
            "hundreds of hours).",
            "Loss changes - deferred to a later single-variable experiment.",
            "3-D U-Net, attention, transformers, ensembles.",
            "Any clinical diagnosis, postoperative healing prediction, longitudinal "
            "claim, or composite 'percentage spine damage' score. The pipeline "
            "reports measurable segmentation, disc-level radiological findings from "
            "dataset annotations, and quantitative imaging features in millimetres.",
        ]
    )

    report.heading("10. Figures")
    report.bullets(
        [
            f"`outputs/visualizations/{EXPERIMENT}/sampling_coverage.png`",
            f"`outputs/visualizations/{EXPERIMENT}/width_feasibility.png`",
            f"`outputs/visualizations/{EXPERIMENT}/indexing_failures.png`",
        ]
    )

    return report.save(REPORT_DIR / "sprint3_audit.md")


if __name__ == "__main__":
    main()
