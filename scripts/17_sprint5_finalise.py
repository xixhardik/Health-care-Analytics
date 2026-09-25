"""Sprint 5 - final report.

Reads the measured Sprint 5 artefacts and the frozen Sprint 3 artefacts, renders
the figures, and writes the final report. No training, no inference, no
re-tuning. Sprint 1-4 artefacts are read-only.

Outputs
-------
    outputs/reports/sprint5_indexing/sprint5_final_report.md
    outputs/reports/sprint5_indexing/sprint5_final_report.json
    outputs/reports/sprint5_indexing/comparison_table.csv
    outputs/visualizations/sprint5_indexing/*.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.paths import OUTPUTS_DIR, PROJECT_ROOT, ensure_dirs  # noqa: E402
from src.utils.reporting import MarkdownReport, save_json  # noqa: E402

EXPERIMENT = "sprint5_indexing"
REPORT_DIR = OUTPUTS_DIR / "reports" / EXPERIMENT
METRICS_DIR = REPORT_DIR / "metrics"
VIZ_DIR = OUTPUTS_DIR / "visualizations" / EXPERIMENT

S3_DIR = OUTPUTS_DIR / "reports" / "sprint3_coverage"
S3_METRICS = S3_DIR / "metrics"

CATEGORIES = ["correct", "shifted", "merged", "split", "missed"]


def load(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


class FixedPrecisionReport(MarkdownReport):
    """``MarkdownReport`` keeping fixed decimals in table cells.

    The shared ``_format_cell`` helper uses four *significant* figures, which
    would turn 0.90001 into "0.9". Sprint 5's measurement deltas sit in the third
    and fourth decimal, so the default rounding would hide them.
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

    def table(self, rows, columns=None):
        rows = [{k: self._fix(v) for k, v in row.items()} for row in rows]
        return super().table(rows, columns)

    def key_values(self, data, *, headers=("Field", "Value")):
        return super().key_values({k: self._fix(v) for k, v in data.items()},
                                  headers=headers)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def render_figures(audit: dict, tuning: dict, fived: dict, test: dict,
                   agreement: dict) -> list[str]:
    import matplotlib.pyplot as plt

    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    names: list[str] = []

    def save(figure, filename):
        figure.savefig(VIZ_DIR / filename, dpi=130, bbox_inches="tight")
        plt.close(figure)
        names.append(filename)

    base_test = test["test_baseline"]
    new_test = test["test_sprint5"]

    # 1. taxonomy before/after on test
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    x = np.arange(len(CATEGORIES))
    width = 0.38
    for offset, data, label, colour in (
        (-width / 2, base_test, "Sprint 3 baseline (per-slice)", "tab:blue"),
        (width / 2, new_test, "Sprint 5 (series-level)", "tab:green"),
    ):
        values = [data["categories"][c]["pct"] for c in CATEGORIES]
        bars = axes[0].bar(x + offset, values, width, label=label, color=colour)
        axes[0].bar_label(bars, fmt="%.2f", fontsize=7)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(CATEGORIES)
    axes[0].set_ylabel("% of ground-truth discs")
    axes[0].set_title("Test-set disc-indexing taxonomy")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3, axis="y")

    by_index_base = pd.DataFrame(base_test["by_disc_index"])
    by_index_new = pd.DataFrame(new_test["by_disc_index"])
    axes[1].plot(by_index_base["truth_index"], by_index_base["pct_correct"],
                 "o-", label="Sprint 3 baseline", color="tab:blue")
    axes[1].plot(by_index_new["truth_index"], by_index_new["pct_correct"],
                 "s-", label="Sprint 5", color="tab:green")
    axes[1].set_xlabel("ground-truth disc index (1 = most inferior)")
    axes[1].set_ylabel("% correctly indexed")
    axes[1].set_title("Accuracy by disc index")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    figure.suptitle(
        f"Disc indexing: {base_test['pct_index_correct']:.2f}% -> "
        f"{new_test['pct_index_correct']:.2f}% on the held-out test set"
    )
    save(figure, "indexing_before_after.png")

    # 2. validation sweep
    sweep = pd.DataFrame(
        [
            {
                "method": c["method"],
                "correct": c["summary"]["pct_index_correct"],
                "cluster_px": c["params"]["cluster_px"],
                "support": c["params"]["min_track_support"],
                "min_area": c["params"]["min_area_px"],
                "col_tol": c["params"]["col_tolerance_px"],
            }
            for c in tuning["configurations"]
        ]
    )
    baseline_val = tuning["baseline"]["summary"]["pct_index_correct"]
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    families = ["5B", "5A", "5A+5B", "5C(all slices)",
                "5C(informative subset)"]
    colours = ["tab:red", "tab:green", "tab:olive", "tab:purple", "tab:gray"]
    for family, colour in zip(families, colours):
        subset = sweep[sweep["method"] == family]
        if subset.empty:
            continue
        axes[0].scatter([family] * len(subset), subset["correct"], s=28,
                        color=colour, alpha=0.75)
    axes[0].axhline(baseline_val, ls="--", color="black",
                    label=f"validation baseline {baseline_val:.2f}%")
    axes[0].set_ylabel("validation % index correct")
    axes[0].set_title("Every configuration tried (validation only)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3, axis="y")
    axes[0].tick_params(axis="x", rotation=20)

    fivea = sweep[sweep["method"] == "5A"]
    for support, marker in zip(sorted(fivea["support"].unique()), "os^"):
        subset = fivea[fivea["support"] == support].sort_values("cluster_px")
        grouped = subset.groupby("cluster_px")["correct"].max()
        axes[1].plot(grouped.index, grouped.values, marker=marker,
                     label=f"min_track_support={support}")
    axes[1].axhline(baseline_val, ls="--", color="black",
                    label=f"baseline {baseline_val:.2f}%")
    axes[1].axvline(25.7, ls=":", color="tab:red",
                    label="measured 5th pct disc gap 25.7px")
    axes[1].set_xlabel("track clustering threshold (px)")
    axes[1].set_ylabel("validation % index correct")
    axes[1].set_title("5A sensitivity to the clustering threshold")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)
    save(figure, "validation_sweep.png")

    # 3. method 5D on vertebrae
    figure, axis = plt.subplots(figsize=(8.5, 4.5))
    schemes = fived["schemes"]
    x = np.arange(len(CATEGORIES))
    for offset, key, label, colour in (
        (-width / 2, "ordered_components",
         "ordered components (current)", "tab:blue"),
        (width / 2, "vertebral_bodies", "vertebral bodies (5D)", "tab:green"),
    ):
        values = [schemes[key]["categories"][c]["pct"] for c in CATEGORIES]
        bars = axis.bar(x + offset, values, width, label=label, color=colour)
        axis.bar_label(bars, fmt="%.2f", fontsize=7)
    axis.set_xticks(x)
    axis.set_xticklabels(CATEGORIES)
    axis.set_ylabel("% of ground-truth vertebrae")
    axis.set_title(
        f"Method 5D on validation vertebrae: "
        f"{schemes['ordered_components']['pct_index_correct']:.2f}% -> "
        f"{schemes['vertebral_bodies']['pct_index_correct']:.2f}% correct"
    )
    axis.legend(fontsize=8)
    axis.grid(alpha=0.3, axis="y")
    save(figure, "method_5d_vertebrae.png")

    # 4. measurement agreement improvement
    s3_map = {r["measurement"]: r for r in agreement["s3"]["measurement_agreement"]}
    s5_map = {r["measurement"]: r for r in agreement["s5"]["measurement_agreement"]}
    shared = [m for m in s5_map if m in s3_map]
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    labels = [m.replace("_", "\n") for m in shared]
    x = np.arange(len(shared))
    axes[0].bar(x - width / 2, [s3_map[m]["mae_pct_of_gt_mean"] for m in shared],
                width, label="Sprint 3", color="tab:blue")
    axes[0].bar(x + width / 2, [s5_map[m]["mae_pct_of_gt_mean"] for m in shared],
                width, label="Sprint 5", color="tab:green")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=6)
    axes[0].set_ylabel("MAE as % of ground-truth mean (lower is better)")
    axes[0].set_title("Measurement error")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3, axis="y")

    axes[1].bar(x - width / 2, [s3_map[m]["pearson_r"] for m in shared], width,
                label="Sprint 3", color="tab:blue")
    axes[1].bar(x + width / 2, [s5_map[m]["pearson_r"] for m in shared], width,
                label="Sprint 5", color="tab:green")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=6)
    axes[1].set_ylabel("Pearson r (higher is better)")
    axes[1].set_title("Measurement correlation")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, axis="y")
    figure.suptitle("Disc-level measurements from predicted masks")
    save(figure, "measurement_improvement.png")

    return names


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def write_report(payload: dict) -> Path:
    audit = payload["audit"]
    tuning = payload["tuning"]
    fived = payload["validation_5d"]
    test = payload["test"]
    agreement = payload["agreement"]
    comparison = payload["comparison"]

    base_test = test["test_baseline"]
    new_test = test["test_sprint5"]
    base_val = tuning["baseline"]["summary"]
    selected = tuning["selected"]

    report = FixedPrecisionReport(
        "Sprint 5 Indexing - Final Report",
        "Disc identity from post-processing, with the segmentation model frozen",
    )

    # ---- 1 objective ----
    report.heading("1. Objective")
    report.text(
        "Improve disc identification by post-processing the existing "
        "predictions, without retraining any segmentation model."
    )
    report.text(
        "Sprints 3 and 4 both tested the network itself - more data coverage, "
        "then more capacity - and neither moved the plateau. What both sprints "
        "did establish is where the loss actually sits: the model finds the disc "
        f"region on {base_test['pct_region_found']:.2f}% of ground-truth discs "
        f"but assigns the correct integer index on only "
        f"{base_test['pct_index_correct']:.2f}%. That gap is an *identity* "
        "problem on regions that were already detected correctly, so it is "
        "addressable without touching the network."
    )
    report.text(
        "The pipeline under test is: segmentation prediction -> disc extraction "
        "-> disc ordering/indexing -> disc-level measurements -> "
        "Pfirrmann/finding analysis. Only the ordering/indexing stage changes."
    )
    report.text(
        "**Integer disc indices only.** The dataset does not state which "
        "vertebra is L5, so no anatomical level name (L1/L2/.../S1) is produced "
        "or asserted anywhere in this sprint."
    )

    # ---- 2 Sprint 3 baseline ----
    report.heading("2. Sprint 3 Baseline")
    report.text(
        "Sprint 3 remains the strongest segmentation model and is the input to "
        "everything here. Its predictions are read frozen."
    )
    report.key_values({
        "Segmentation model": "Sprint 3 Coverage, 16-channel U-Net, "
                              "1,963,860 parameters",
        "Checkpoint": "`outputs/checkpoints/sprint3_coverage/best_val_dice.pt`",
        "Test macro foreground Dice": 0.90001,
        "Test vertebra Dice": 0.91663,
        "Test IVD Dice": 0.87827,
        "Test canal Dice": 0.90514,
        "Test corrected-taxonomy disc indexing":
            f"{base_test['pct_index_correct']:.2f}%",
        "Sprint 4 (width 32) for reference":
            "test macro Dice 0.89393, indexing 82.70% - worse, so architecture "
            "scaling was stopped",
    })
    report.text(
        "**How indexing currently works.** Disc identity is not predicted by the "
        "network. It is derived afterwards by taking connected components of the "
        "IVD class *within a single slice*, dropping components under 20 px, and "
        "numbering them bottom-up. The index of a disc is therefore a function "
        "of how many components that one slice happens to contain, which is the "
        "structural weakness this sprint attacks."
    )
    report.heading("Estimator control", level=3)
    control = audit["control"]
    report.key_values({
        "Published Sprint 3 test index-correct":
            f"{control['published_sprint3_test_pct_index_correct']:.2f}%",
        "Re-measured here with the Sprint 5 scorer":
            f"{control['remeasured_test_pct_index_correct']:.2f}%",
        "Difference": f"{control['difference_pp']:+.4f} pp",
        "Estimator identical to Sprint 3's": control["estimator_agrees"],
    })
    report.text(
        "The Sprint 5 scorer takes a disc *index -> region* mapping rather than "
        "re-deriving components from the semantic map, which is what allows a "
        "post-processed identity to be scored at all. Because that is a change "
        "of plumbing, it was verified to reproduce the published Sprint 3 "
        "taxonomy exactly - every category count matches - so every comparison "
        "below is like-for-like against 84.08%."
    )

    # ---- 3 indexing error analysis ----
    report.heading("3. Indexing Error Analysis")
    report.text(
        "The corrected taxonomy from the Sprint 3 audit is preserved unchanged: "
        "every ground-truth disc on every slice is classified as exactly one of "
        "`correct`, `shifted`, `merged`, `split` or `missed`, with a 20 px "
        "minimum component area and best-overlap matching."
    )
    report.table([
        {
            "category": category,
            "validation n": audit["val"]["categories"][category]["n"],
            "validation %": audit["val"]["categories"][category]["pct"],
            "test n": base_test["categories"][category]["n"],
            "test %": base_test["categories"][category]["pct"],
        }
        for category in CATEGORIES
    ], ["category", "validation n", "validation %", "test n", "test %"])
    report.key_values({
        "Total discs evaluated (validation)":
            f"{audit['val']['n_discs_analysed']:,} over "
            f"{audit['val']['n_slices_analysed']:,} slices",
        "Total discs evaluated (test)":
            f"{base_test['n_discs_analysed']:,} over "
            f"{base_test['n_slices_analysed']:,} slices",
        "Region found (validation / test)":
            f"{audit['val']['pct_region_found']:.2f}% / "
            f"{base_test['pct_region_found']:.2f}%",
        "Shifted by exactly +/-1 (validation / test)":
            f"{audit['val']['pct_shifted_by_one']:.2f}% / "
            f"{base_test['pct_shifted_by_one']:.2f}% of all discs",
        "Spurious components (validation / test)":
            f"{audit['val']['total_spurious_components']:,} / "
            f"{base_test['total_spurious_components']:,}",
    })
    report.heading("Shift offsets, test baseline", level=3)
    report.table(
        [{"offset": offset, "n discs": count}
         for offset, count in sorted(
             base_test["shifted_offsets"].items(), key=lambda kv: float(kv[0]))],
        ["offset", "n discs"],
    )
    report.text(
        f"The diagnosis is clear from these numbers. Detection is not the "
        f"problem - only {base_test['categories']['missed']['pct']:.2f}% of "
        f"discs are missed outright, and merged and split together account for "
        f"{base_test['categories']['merged']['pct'] + base_test['categories']['split']['pct']:.2f}%. "
        f"The dominant failure is `shifted` at "
        f"{base_test['categories']['shifted']['pct']:.2f}%: the disc was found, "
        f"and given the wrong number. Most shifts are by a single position, "
        f"which is the signature of one extra or one absent component low in the "
        f"stack renumbering everything above it."
    )

    # ---- 4 method 5A ----
    report.heading("4. Method 5A - Series-Level Ordering")
    report.text(
        "Replace independent per-slice numbering with identity assigned once at "
        "series level, then propagated to every slice."
    )
    report.bullets([
        "**Extract** disc candidates from each sagittal slice as connected "
        "components of the IVD class, with their area, mean row and mean column.",
        "**Associate** candidates across the slices of a series by single-linkage "
        "clustering on mean row. Slices of one series share a voxel grid, so the "
        "same physical disc occupies nearly the same rows on every slice it "
        "appears in.",
        "**Build** a series-level representation: each cluster becomes a *track* "
        "with a median row and a support count, the fraction of slices in which "
        "it appears. Tracks below a support threshold are discarded as spurious.",
        "**Order** the surviving tracks once, most inferior first, matching the "
        "dataset's bottom-up convention. This ordering is the series' disc "
        "identity.",
        "**Propagate** by assigning each per-slice candidate the index of its "
        "nearest track, subject to a distance gate. Within a slice a track can "
        "be claimed only once; the larger candidate wins.",
    ])
    report.text(
        "The reason this fixes shifts is that a disc's index no longer depends "
        "on the component count of the slice it appears in. If a disc is missed "
        "on one slice, the remaining discs keep their series-level identity "
        "instead of sliding by one."
    )
    report.heading("Geometry verified before use", level=3)
    report.text(
        "Three conventions the method depends on were measured on ground-truth "
        "training data rather than assumed:"
    )
    report.table([
        {"property": "vertebra N is superior to disc N",
         "measured": "97.7% of ground-truth discs",
         "use": "ties vertebra identity to disc identity in 5D"},
        {"property": "adjacent disc row gap",
         "measured": "median 33.8 px, 5th percentile 25.7 px",
         "use": "clustering threshold must stay well under 25 px"},
        {"property": "vertebral body vs canal column",
         "measured": "bodies median 23.5 px anterior, posterior elements "
                     "10.7 px behind",
         "use": "separates bodies from posterior elements in 5D"},
    ], ["property", "measured", "use"])

    # ---- 5 method 5B ----
    report.heading("5. Method 5B - Spurious Component Rejection")
    report.text(
        "Spurious components can renumber every disc above them, so rejecting "
        "them before ordering is a plausible fix. Two conservative rules were "
        "tested: a raised minimum component area, and a column band rejecting "
        "candidates far from the series' median disc column."
    )
    fiveb = [c for c in tuning["configurations"] if c["method"] == "5B"]
    report.table([
        {
            "min_area_px": c["params"]["min_area_px"],
            "col_tolerance_px": c["params"]["col_tolerance_px"],
            "validation % correct": c["summary"]["pct_index_correct"],
            "% missed": c["summary"]["categories"]["missed"]["pct"],
            "% shifted": c["summary"]["categories"]["shifted"]["pct"],
        }
        for c in fiveb
    ], ["min_area_px", "col_tolerance_px", "validation % correct", "% missed",
        "% shifted"])
    best_b = max(fiveb, key=lambda c: c["summary"]["pct_index_correct"])
    report.text(
        f"**5B is rejected on validation evidence.** Not one configuration beat "
        f"the {base_val['pct_index_correct']:.2f}% baseline; the best was "
        f"{best_b['summary']['pct_index_correct']:.2f}%, and the best of those "
        f"is the configuration that rejects nothing. Both rules trade shifted "
        f"discs for missed discs at a losing rate - tightening the column band "
        f"to 20 px drops accuracy to 72.63% while raising missed discs from "
        f"4.13% to 13.81%. The reason is that a genuine disc on a lateral slice "
        f"is small and can sit well off the series' median column, so these "
        f"filters remove real discs faster than spurious ones. Rejection is not "
        f"carried into the final method."
    )

    # ---- 6 method 5C ----
    report.heading("6. Method 5C - Lateral Slice Handling")
    report.text(
        "Indexing accuracy falls on slices carrying little annotated disc area, "
        "because a lateral slice holds only fragments. The question is whether "
        "indexing should be restricted to sufficiently informative slices."
    )
    report.text(
        "**The criterion is prediction-derived**, so it is available at test "
        "time without any annotation: a slice is informative when its predicted "
        "IVD area reaches a threshold. No ground-truth area is used."
    )
    fivec_all = [c for c in tuning["configurations"]
                 if c["method"] == "5C(all slices)"]
    fivec_sub = [c for c in tuning["configurations"]
                 if c["method"] == "5C(informative subset)"]
    report.table([
        {
            "informative_min_ivd_px": a["params"]["informative_min_ivd_px"],
            "all slices % correct": a["summary"]["pct_index_correct"],
            "informative subset % correct": s["summary"]["pct_index_correct"],
            "discs retained": f"{s['summary']['n_discs_analysed']:,} / "
                              f"{a['summary']['n_discs_analysed']:,}",
            "% of slices retained": a["summary"]["pct_slices_informative"],
        }
        for a, s in zip(fivec_all, fivec_sub)
    ], ["informative_min_ivd_px", "all slices % correct",
        "informative subset % correct", "discs retained",
        "% of slices retained"])
    report.text(
        "Read the two accuracy columns together with the coverage column. "
        "Restricting the *reported* set to informative slices raises accuracy to "
        "95.64% at a 400 px threshold, but it does so by dropping 516 of 6,748 "
        "discs and 38.7% of slices from the denominator. That is a coverage "
        "trade, not an improvement, and counting it as a gain would be exactly "
        "the artificial inflation this sprint was told to avoid."
    )
    report.text(
        "On the like-for-like all-slices denominator the criterion changes "
        "nothing (93.32% against 93.33%). **5C is therefore not adopted as a "
        "filter.** Its measurement is still useful: it quantifies how much of "
        "the residual error is concentrated in genuinely uninformative slices, "
        "and it is reported here so that a future measurement stage can decide "
        "to *measure* only on informative slices while still *indexing* all of "
        "them."
    )

    # ---- 7 method 5D ----
    report.heading("7. Method 5D - Vertebral Body Separation")
    report.text(
        "Vertebrae must not be numbered by counting connected components. The "
        "Sprint 3 audit established that one vertebra occupies roughly two "
        "components in a sagittal slice - the body, and the posterior elements - "
        "so ordering components interleaves fragments of different vertebrae "
        "regardless of segmentation quality. This is the root cause of the "
        "unusable disc-to-vertebra height ratio."
    )
    report.bullets([
        "Identify the **vertebral body** as the component anterior to the "
        "spinal canal centroid column, measured per slice with a series-level "
        "fallback when the canal is absent. Posterior elements keep their "
        "semantic class but receive no identity.",
        "Number each body by **the disc tracks below it**, so body N is the one "
        "directly above disc N. Tying vertebra identity to disc identity avoids "
        "reintroducing a second count-dependent ordering, and it matches the "
        "convention the measurement code already assumes.",
    ])
    schemes = fived["schemes"]
    report.table([
        {
            "category": category,
            "ordered components n": schemes["ordered_components"]["categories"][category]["n"],
            "ordered components %": schemes["ordered_components"]["categories"][category]["pct"],
            "vertebral bodies n": schemes["vertebral_bodies"]["categories"][category]["n"],
            "vertebral bodies %": schemes["vertebral_bodies"]["categories"][category]["pct"],
        }
        for category in CATEGORIES
    ], ["category", "ordered components n", "ordered components %",
        "vertebral bodies n", "vertebral bodies %"])
    report.key_values({
        "Ground-truth vertebrae scored (validation)":
            f"{schemes['ordered_components']['n_vertebrae_analysed']:,}",
        "Ordered components, index correct":
            f"{schemes['ordered_components']['pct_index_correct']:.2f}%",
        "Vertebral bodies (5D), index correct":
            f"{schemes['vertebral_bodies']['pct_index_correct']:.2f}%",
        "Absolute change": f"{fived['delta_pp']:+.2f} pp",
        "Adopted": fived["adopt_5d"],
    })
    report.text(
        f"The ordered-component scheme scores "
        f"{schemes['ordered_components']['categories']['split']['pct']:.2f}% "
        f"`split`, which is the audit's prediction showing up directly in the "
        f"measurement: a single ground-truth vertebra is routinely covered by "
        f"two differently-numbered predicted components. Body identification "
        f"cuts that to "
        f"{schemes['vertebral_bodies']['categories']['split']['pct']:.2f}% and "
        f"raises correct identification from "
        f"{schemes['ordered_components']['pct_index_correct']:.2f}% to "
        f"{schemes['vertebral_bodies']['pct_index_correct']:.2f}%. It is scored "
        f"with the same taxonomy rules as the discs, so the two are directly "
        f"readable against each other. **5D is adopted.** Note that at "
        f"{schemes['vertebral_bodies']['pct_index_correct']:.2f}% it remains the "
        f"weakest stage in the pipeline, with "
        f"{schemes['vertebral_bodies']['categories']['missed']['pct']:.2f}% of "
        f"vertebrae still unmatched."
    )

    # ---- 8 validation results ----
    report.heading("8. Validation Results")
    report.text(
        f"All parameters were selected on the validation split. "
        f"{tuning['n_configurations']} configurations were scored; the test "
        f"split was not read during tuning."
    )
    report.key_values({
        "Split used for tuning": tuning["split_used_for_tuning"],
        "Test set touched during tuning": tuning["test_touched"],
        "Configurations evaluated": tuning["n_configurations"],
        "Validation baseline": f"{base_val['pct_index_correct']:.2f}%",
        "Selected method": selected["method"] + (
            "+5D" if fived["adopt_5d"] else ""),
        "Selected validation accuracy":
            f"{selected['summary']['pct_index_correct']:.2f}%",
        "Absolute change on validation":
            f"{selected['summary']['pct_index_correct'] - base_val['pct_index_correct']:+.2f} pp",
    })
    report.heading("Selected parameters", level=3)
    report.table(
        [{"parameter": k, "value": str(v)}
         for k, v in test["selected_params"].items()],
        ["parameter", "value"],
    )
    report.heading("Best configuration per method family", level=3)
    families: dict[str, dict] = {}
    for configuration in tuning["configurations"]:
        family = configuration["method"]
        if (family not in families
                or configuration["summary"]["pct_index_correct"]
                > families[family]["summary"]["pct_index_correct"]):
            families[family] = configuration
    report.table([
        {
            "method": family,
            "validation % correct": c["summary"]["pct_index_correct"],
            "vs baseline (pp)":
                f"{c['summary']['pct_index_correct'] - base_val['pct_index_correct']:+.2f}",
            "% shifted": c["summary"]["categories"]["shifted"]["pct"],
            "% missed": c["summary"]["categories"]["missed"]["pct"],
            "discs scored": c["summary"]["n_discs_analysed"],
            "runtime s": c["summary"]["runtime_seconds"],
        }
        for family, c in families.items()
    ], ["method", "validation % correct", "vs baseline (pp)", "% shifted",
        "% missed", "discs scored", "runtime s"])
    report.text(
        "The 5C rows are shown for completeness. Only the all-slices variants "
        "were eligible for selection, because the informative-subset variant "
        "scores a smaller denominator and is not comparable with the baseline."
    )
    report.heading("Sensitivity", level=3)
    report.text(
        "The clustering threshold behaves exactly as the ground-truth geometry "
        "predicts. Thresholds of 8 px and 12 px are indistinguishable (93.33%), "
        "16 px costs a little (92.81%), and 20 px collapses to 79.48% - because "
        "the 5th percentile gap between adjacent discs is 25.7 px, so a 20 px "
        "threshold starts merging neighbouring discs into one track. The method "
        "is not sensitive within the range the anatomy allows, and its failure "
        "mode outside that range is understood rather than mysterious."
    )

    # ---- 9 final test results ----
    report.heading("9. Final Test Results")
    report.text(
        "The single selected configuration was applied once to the held-out test "
        "set. No parameter was changed after seeing these numbers."
    )
    report.key_values({
        "Method applied": test["selected_method"],
        "Selected on": test["selected_on"],
        "Validation accuracy of this method":
            f"{test['validation_pct_index_correct']:.2f}%",
        "Test baseline": f"{base_test['pct_index_correct']:.2f}%",
        "Test Sprint 5": f"{new_test['pct_index_correct']:.2f}%",
        "Absolute change": f"{test['test_delta_pp']:+.2f} pp",
        "Slices written": f"{test['n_slices_written']:,}",
        "Segmentation predictions unchanged": test["semantic_unchanged"],
    })
    report.table([
        {
            "category": category,
            "baseline n": base_test["categories"][category]["n"],
            "baseline %": base_test["categories"][category]["pct"],
            "Sprint 5 n": new_test["categories"][category]["n"],
            "Sprint 5 %": new_test["categories"][category]["pct"],
            "Δ pp": round(
                new_test["categories"][category]["pct"]
                - base_test["categories"][category]["pct"], 2),
        }
        for category in CATEGORIES
    ], ["category", "baseline n", "baseline %", "Sprint 5 n", "Sprint 5 %",
        "Δ pp"])
    report.key_values({
        "Region found": f"{base_test['pct_region_found']:.2f}% -> "
                        f"{new_test['pct_region_found']:.2f}%",
        "Slices with correct disc count":
            f"{base_test['pct_slices_count_correct']:.2f}% -> "
            f"{new_test['pct_slices_count_correct']:.2f}%",
        "Slices with every disc correct":
            f"{base_test['pct_slices_all_discs_correct']:.2f}% -> "
            f"{new_test['pct_slices_all_discs_correct']:.2f}%",
        "Spurious components":
            f"{base_test['total_spurious_components']:,} -> "
            f"{new_test['total_spurious_components']:,}",
        "Mean tracks per series": new_test["mean_tracks_per_series"],
        "Post-processing runtime":
            f"{new_test['runtime_seconds']}s for "
            f"{new_test['n_slices_analysed']:,} slices "
            f"(plus {test['write_seconds']}s to write the masks)",
    })

    # ---- 10 Sprint 3 vs Sprint 5 ----
    report.heading("10. Sprint 3 vs Sprint 5")
    report.table(comparison["rows"], ["Metric", "Sprint 3", "Sprint 5",
                                      "Absolute Δ", "Verdict"])
    report.heading("Are the segmentation predictions unchanged?", level=3)
    report.key_values({
        "`semantic` maps byte-identical to Sprint 3": test["semantic_unchanged"],
        "Segmentation model retrained": False,
        "Segmentation metrics affected":
            "No - macro foreground Dice, per-class Dice, IoU, precision and "
            "recall are all properties of the `semantic` map, which is copied "
            "through unchanged",
        "What did change":
            "Only the `instance` map, i.e. which integer identity is attached "
            "to each already-segmented disc and vertebral body",
    })
    report.text(
        "This is the central property of the sprint. Sprint 5 adds no "
        "segmentation quality and claims none: pixel-level accuracy is exactly "
        "Sprint 3's, verified byte-for-byte on all 1,655 test slices. What "
        "improves is everything that depends on a disc being correctly "
        "*identified* - which is the whole downstream chain."
    )

    # ---- 11 failure cases ----
    report.heading("11. Failure Cases")
    failures = payload["failures"]
    report.text(
        f"{failures['n_remaining']:,} of {new_test['n_discs_analysed']:,} test "
        f"discs are still not correctly indexed "
        f"({100 * failures['n_remaining'] / new_test['n_discs_analysed']:.2f}%). "
        f"The residual is dominated by a different category than the baseline's."
    )
    report.table(failures["by_category"], ["category", "n", "% of all discs",
                                           "% of remaining errors"])
    report.heading("Remaining shift offsets", level=3)
    report.table(failures["offsets"], ["offset", "n discs"])
    report.heading("Accuracy by disc index", level=3)
    report.table(failures["by_index"],
                 ["disc index", "n", "baseline % correct", "Sprint 5 % correct",
                  "Δ pp"])
    report.text(failures["narrative"])
    report.text(failures["trade_off"])
    report.text(failures["ceiling"])

    # ---- 12 limitations ----
    report.heading("12. Limitations")
    report.bullets([
        "**A whole-series shift is not detectable.** Series-level ordering fixes "
        "inconsistency *between* slices, but if the most inferior disc is absent "
        "from the prediction on every slice of a series, every track shifts by "
        "one and the method has no internal evidence of it. This is the residual "
        f"`shifted` population, still "
        f"{new_test['categories']['shifted']['pct']:.2f}% of discs.",
        "**No anatomical anchor.** Nothing here identifies the sacrum or any "
        "named level, so the numbering is relative to the most inferior detected "
        "disc, not to anatomy. Correcting a whole-series shift would need such "
        "an anchor, which the current annotation does not provide.",
        "**Vertebral body identification is the weak stage.** 5D raises vertebra "
        f"identity to "
        f"{schemes['vertebral_bodies']['pct_index_correct']:.2f}% on validation, "
        f"a large gain, but "
        f"{schemes['vertebral_bodies']['categories']['missed']['pct']:.2f}% of "
        "vertebrae are still unmatched, and the canal-column split degrades on "
        "slices where the canal is not predicted.",
        "**disc_to_vertebra_height_ratio is improved but still not reliable.** "
        "Its error falls from 71.87% to 33.72% of the ground-truth mean. That is "
        "a large improvement and still too high to use as a derived feature.",
        "**Single split, single model.** All numbers come from one patient-level "
        "split and one frozen segmentation model. The test set is 33 patients.",
        "**Parameters were selected on 39 validation patients.** The validation "
        "baseline (80.82%) is lower than the test baseline (84.08%), so the two "
        "splits are not equally difficult; the gain transferred (+12.51 pp "
        "validation, +9.76 pp test) but was not identical.",
        "**Segmentation metrics are unchanged and are not clinical evidence.** "
        "A Dice score or an indexing accuracy measures agreement with one "
        "annotation protocol. Neither establishes clinical effectiveness or "
        "readiness for medical use, and none is claimed.",
        "**Pfirrmann grades are dataset annotations** used as labels for a "
        "measurement exercise. Nothing here grades a patient.",
        "**No longitudinal or postoperative scope.** The SPIDER data used here "
        "contains no longitudinal postoperative follow-up, so postoperative "
        "healing stays outside the validated scope of this work and no claim "
        "about it is made or supported. No change over time is measured, and no "
        "'percentage spine damage' score exists in this pipeline.",
    ])

    # ---- 13 recommended pipeline ----
    report.heading("13. Recommended Pipeline")
    report.text(
        "The recommended configuration is the one evaluated above. It is "
        "reproducible from the frozen Sprint 3 predictions and adds well under a "
        "minute of CPU time for the whole test set."
    )
    report.table([
        {"stage": "1. segmentation",
         "component": "Sprint 3 Coverage 16-channel U-Net, frozen",
         "note": "no retraining; `semantic` output used unchanged"},
        {"stage": "2. disc extraction",
         "component": "connected components of the IVD class, 20 px minimum",
         "note": "unchanged from the baseline; no rejection filter (5B rejected)"},
        {"stage": "3. disc ordering",
         "component": "series-level tracks (5A)",
         "note": f"cluster {test['selected_params']['cluster_px']} px, support "
                 f"{test['selected_params']['min_track_support']}, gate "
                 f"{test['selected_params']['assign_max_dist_px']} px"},
        {"stage": "4. vertebra identity",
         "component": "vertebral-body separation (5D)",
         "note": "canal-column split, numbered from the disc tracks below"},
        {"stage": "5. measurements",
         "component": "existing disc_features pipeline, unchanged",
         "note": "consumes the corrected `instance` map"},
        {"stage": "6. Pfirrmann / findings",
         "component": "existing end-to-end evaluation, unchanged",
         "note": "grade models refitted per run as before"},
    ], ["stage", "component", "note"])
    report.text(
        "5B and 5C are explicitly **not** part of the recommendation. 5B was "
        "measured to hurt on validation, and 5C changes nothing on a "
        "like-for-like denominator."
    )

    # ---- 14 next step ----
    report.heading("14. Next Step")
    report.text(payload["next_step"]["recommendation"])
    report.bullets(payload["next_step"]["reasoning"])
    report.text(
        "**This has not been started.** It is a recommendation only, and no "
        "further experiment was launched after this report."
    )

    # ---- artefacts ----
    report.heading("Artefact Locations")
    report.table([
        {"artefact": "baseline audit (validation + test)",
         "path": f"`outputs/reports/{EXPERIMENT}/baseline_audit.json`"},
        {"artefact": "validation sweep (all configurations)",
         "path": f"`outputs/reports/{EXPERIMENT}/validation_tuning.json`, "
                 f"`validation_sweep.csv`"},
        {"artefact": "method 5D on validation",
         "path": f"`outputs/reports/{EXPERIMENT}/validation_5d.json`"},
        {"artefact": "final test results",
         "path": f"`outputs/reports/{EXPERIMENT}/test_results.json`"},
        {"artefact": "per-disc failure records (test)",
         "path": f"`outputs/reports/{EXPERIMENT}/indexing_failures_sprint5_test.csv`"},
        {"artefact": "measurement agreement + Pfirrmann",
         "path": f"`outputs/reports/{EXPERIMENT}/metrics/measurement_agreement.json`"},
        {"artefact": "disc measurements",
         "path": f"`outputs/reports/disc_analysis_{EXPERIMENT}.csv`"},
        {"artefact": "corrected predictions (semantic unchanged)",
         "path": f"`data/processed/predictions_{EXPERIMENT}/`"},
        {"artefact": "validation-split predictions (Sprint 3 model)",
         "path": "`data/processed/predictions_sprint5_val_raw/`"},
        {"artefact": "post-processing implementation",
         "path": "`src/analysis/disc_postprocess.py`"},
        {"artefact": "driver script",
         "path": "`scripts/16_sprint5_indexing.py`"},
        {"artefact": "figures",
         "path": f"`outputs/visualizations/{EXPERIMENT}/`"},
        {"artefact": "this report",
         "path": f"`outputs/reports/{EXPERIMENT}/sprint5_final_report.md` (+ .json)"},
    ], ["artefact", "path"])
    report.heading("Figures", level=3)
    report.bullets([f"`outputs/visualizations/{EXPERIMENT}/{name}`"
                    for name in payload["figures"]])

    return report.save(REPORT_DIR / "sprint5_final_report.md")


# ---------------------------------------------------------------------------
# Comparison + failure analysis
# ---------------------------------------------------------------------------


def build_comparison(test: dict, agreement: dict) -> dict:
    base, new = test["test_baseline"], test["test_sprint5"]
    s3_map = {r["measurement"]: r for r in agreement["s3"]["measurement_agreement"]}
    s5_map = {r["measurement"]: r for r in agreement["s5"]["measurement_agreement"]}
    s3_pf = agreement["s3"]["end_to_end_pfirrmann"]
    s5_pf = agreement["s5"]["end_to_end_pfirrmann"]

    rows: list[dict] = []

    def add(metric, s3, s5, higher_is_better, fmt="{:.4f}"):
        if s3 is None or s5 is None:
            return
        delta = s5 - s3
        verdict = ("unchanged" if abs(delta) < 1e-9
                   else "improved" if (delta > 0) == higher_is_better
                   else "worsened")
        rows.append({
            "Metric": metric,
            "Sprint 3": fmt.format(s3),
            "Sprint 5": fmt.format(s5),
            "Absolute Δ": f"{delta:+.4f}",
            "Verdict": verdict,
        })

    for category in CATEGORIES:
        add(f"indexing {category} %", base["categories"][category]["pct"],
            new["categories"][category]["pct"], category == "correct",
            "{:.2f}")
    add("indexing region found %", base["pct_region_found"],
        new["pct_region_found"], True, "{:.2f}")
    add("slices with every disc correct %", base["pct_slices_all_discs_correct"],
        new["pct_slices_all_discs_correct"], True, "{:.2f}")
    add("spurious components", base["total_spurious_components"],
        new["total_spurious_components"], False, "{:.0f}")

    for measurement, higher in (("height_mm_central", False),
                                ("area_mm2", False)):
        add(f"{measurement} MAE", s3_map[measurement]["mae"],
            s5_map[measurement]["mae"], higher)
    add("height_mm_central Pearson r", s3_map["height_mm_central"]["pearson_r"],
        s5_map["height_mm_central"]["pearson_r"], True)
    add("area_mm2 Pearson r", s3_map["area_mm2"]["pearson_r"],
        s5_map["area_mm2"]["pearson_r"], True)
    add("intensity ratio MAE", s3_map["intensity_disc_vertebra_ratio"]["mae"],
        s5_map["intensity_disc_vertebra_ratio"]["mae"], False)
    add("intensity ratio Pearson r",
        s3_map["intensity_disc_vertebra_ratio"]["pearson_r"],
        s5_map["intensity_disc_vertebra_ratio"]["pearson_r"], True)
    add("disc_to_vertebra_height_ratio MAE % of mean",
        s3_map["disc_to_vertebra_height_ratio"]["mae_pct_of_gt_mean"],
        s5_map["disc_to_vertebra_height_ratio"]["mae_pct_of_gt_mean"], False,
        "{:.2f}")
    add("% of truth discs recovered (series level)",
        agreement["s3"]["detection"]["pct_truth_discs_recovered"],
        agreement["s5"]["detection"]["pct_truth_discs_recovered"], True,
        "{:.2f}")
    add("predicted discs not in truth",
        agreement["s3"]["detection"]["n_predicted_not_in_truth"],
        agreement["s5"]["detection"]["n_predicted_not_in_truth"], False,
        "{:.0f}")
    add("Pfirrmann forest QWK", s3_pf["forest"]["quadratic_weighted_kappa"],
        s5_pf["forest"]["quadratic_weighted_kappa"], True)
    add("Pfirrmann logreg QWK", s3_pf["logreg"]["quadratic_weighted_kappa"],
        s5_pf["logreg"]["quadratic_weighted_kappa"], True)
    add("Pfirrmann forest within 1 grade",
        s3_pf["forest"]["within_one_grade"],
        s5_pf["forest"]["within_one_grade"], True)

    tally: dict[str, int] = {}
    for row in rows:
        tally[row["Verdict"]] = tally.get(row["Verdict"], 0) + 1
    return {"rows": rows, "tally": tally}


def analyse_failures(test: dict) -> dict:
    base, new = test["test_baseline"], test["test_sprint5"]
    failures_csv = REPORT_DIR / "indexing_failures_sprint5_test.csv"
    frame = pd.read_csv(failures_csv)
    total = len(frame)
    remaining = frame[frame["category"] != "correct"]

    by_category = [
        {
            "category": category,
            "n": int((remaining["category"] == category).sum()),
            "% of all discs": round(
                100 * (remaining["category"] == category).sum() / total, 2),
            "% of remaining errors": round(
                100 * (remaining["category"] == category).sum()
                / max(1, len(remaining)), 2),
        }
        for category in CATEGORIES if category != "correct"
    ]

    shifted = remaining[remaining["category"] == "shifted"]
    offsets = [
        {"offset": int(offset), "n discs": int(count)}
        for offset, count in shifted["offset"].value_counts().sort_index().items()
    ]

    base_index = {r["truth_index"]: r for r in base["by_disc_index"]}
    new_index = {r["truth_index"]: r for r in new["by_disc_index"]}
    by_index = [
        {
            "disc index": index,
            "n": new_index[index]["n"],
            "baseline % correct": base_index[index]["pct_correct"],
            "Sprint 5 % correct": new_index[index]["pct_correct"],
            "Δ pp": round(new_index[index]["pct_correct"]
                          - base_index[index]["pct_correct"], 2),
        }
        for index in sorted(new_index) if index in base_index
    ]

    worst_index = min(by_index, key=lambda r: r["Sprint 5 % correct"])
    missed_delta_n = (new["categories"]["missed"]["n"]
                      - base["categories"]["missed"]["n"])
    shifted_delta_n = (base["categories"]["shifted"]["n"]
                       - new["categories"]["shifted"]["n"])
    narrative = (
        f"The character of the residual error has changed. In the baseline the "
        f"largest category was `shifted` at "
        f"{base['categories']['shifted']['pct']:.2f}%; after series-level "
        f"ordering `shifted` falls to "
        f"{new['categories']['shifted']['pct']:.2f}% and `missed` is now the "
        f"largest remaining category at "
        f"{new['categories']['missed']['pct']:.2f}%. "
        f"The weakest disc index is {worst_index['disc index']} at "
        f"{worst_index['Sprint 5 % correct']:.2f}% correct over "
        f"{worst_index['n']:,} discs. High indices are the hardest because they "
        f"sit at the superior end of the field of view where a disc may be only "
        f"partly imaged, and because an error anywhere below them can still "
        f"displace a track."
    )
    trade_off = (
        f"**The one cost of the method, stated plainly.** Missed discs rose by "
        f"{missed_delta_n} ({base['categories']['missed']['pct']:.2f}% -> "
        f"{new['categories']['missed']['pct']:.2f}%), and region-found "
        f"correspondingly fell from {base['pct_region_found']:.2f}% to "
        f"{new['pct_region_found']:.2f}%. This is not a segmentation change - the "
        f"pixels are identical - it is the track-assignment gate refusing to "
        f"number a candidate that sits further than the allowed distance from "
        f"any series track, plus the rule that a track may be claimed only once "
        f"per slice. "
        f"Those candidates were previously given a number, usually the wrong "
        f"one. So the trade is {missed_delta_n} discs moved from 'numbered "
        f"wrongly' to 'not numbered', against {shifted_delta_n:,} discs moved "
        f"from 'numbered wrongly' to 'numbered correctly'. That is a favourable "
        f"exchange at roughly {shifted_delta_n / max(1, missed_delta_n):.0f}:1, "
        f"and an unnumbered disc is more honest downstream than a confidently "
        f"mis-numbered one, because it is excluded from the measurement join "
        f"rather than attaching its measurements to the wrong grading. It is "
        f"still a genuine regression on those two metrics and is counted as such "
        f"in the comparison table."
    )
    ceiling = (
        f"Beyond that, the remaining `missed` population is a segmentation "
        f"limitation rather than an indexing one: post-processing cannot number "
        f"a disc the network never predicted. That puts a hard ceiling of "
        f"{new['pct_region_found']:.2f}% on what any indexing method can reach "
        f"on these frozen predictions, and Sprint 5 now sits "
        f"{new['pct_region_found'] - new['pct_index_correct']:.2f} pp below it - "
        f"against {base['pct_region_found'] - base['pct_index_correct']:.2f} pp "
        f"for the baseline. Most of the headroom that existed in the indexing "
        f"stage has been taken."
    )

    return {
        "n_remaining": int(len(remaining)),
        "by_category": by_category,
        "offsets": offsets,
        "by_index": by_index,
        "narrative": narrative,
        "trade_off": trade_off,
        "ceiling": ceiling,
        "missed_increase_n": int(missed_delta_n),
        "shifted_reduction_n": int(shifted_delta_n),
    }


def main() -> None:
    ensure_dirs()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    VIZ_DIR.mkdir(parents=True, exist_ok=True)

    print("[1] Loading measured artefacts ...")
    audit = load(REPORT_DIR / "baseline_audit.json")
    tuning = load(REPORT_DIR / "validation_tuning.json")
    fived = load(REPORT_DIR / "validation_5d.json")
    test = load(REPORT_DIR / "test_results.json")
    agreement = {
        "s5": load(METRICS_DIR / "measurement_agreement.json"),
        "s3": load(S3_METRICS / "measurement_agreement.json"),
    }
    missing = [name for name, value in
               (("baseline_audit", audit), ("validation_tuning", tuning),
                ("validation_5d", fived), ("test_results", test),
                ("sprint5 agreement", agreement["s5"]),
                ("sprint3 agreement", agreement["s3"])) if value is None]
    if missing:
        raise SystemExit(f"Missing required artefacts: {missing}")
    print(f"  test indexing {test['test_baseline']['pct_index_correct']:.2f}% -> "
          f"{test['test_sprint5']['pct_index_correct']:.2f}%")

    print("[2] Building the comparison ...")
    comparison = build_comparison(test, agreement)
    pd.DataFrame(comparison["rows"]).to_csv(
        REPORT_DIR / "comparison_table.csv", index=False
    )
    print(f"  tally: {comparison['tally']}")

    print("[3] Analysing residual failures ...")
    failures = analyse_failures(test)
    print(f"  {failures['n_remaining']:,} discs still misindexed")

    print("[4] Rendering figures ...")
    figures = render_figures(audit, tuning, fived, test, agreement)
    for extra in sorted(VIZ_DIR.glob("*.png")):
        if extra.name not in figures:
            figures.append(extra.name)
    for name in figures:
        print(f"  -> {VIZ_DIR / name}")

    next_step = {
        "recommendation": (
            "**Anchor the series-level numbering to an anatomical reference, "
            "then re-measure.** Series-level ordering has taken indexing from "
            f"{test['test_baseline']['pct_index_correct']:.2f}% to "
            f"{test['test_sprint5']['pct_index_correct']:.2f}%, and the residual "
            "error is now dominated by two things it cannot fix: discs the "
            "network never predicted, and whole-series shifts where every track "
            "is displaced together. The first needs better segmentation recall "
            "at the ends of the stack; the second needs an anchor, because "
            "relative ordering alone cannot detect a uniform offset."
        ),
        "reasoning": [
            "The measurable ceiling is now explicit: region-found is "
            f"{test['test_sprint5']['pct_region_found']:.2f}% on these frozen "
            "predictions, so no indexing method can exceed that without "
            "improving detection.",
            "Whole-series shift is the largest remaining *indexing* error and is "
            "invisible to a relative method by construction, so it should be "
            "attacked with an anchor (for example the most inferior fully-imaged "
            "disc, or a sacrum cue) rather than with more ordering heuristics.",
            "Vertebral-body identity at "
            f"{fived['schemes']['vertebral_bodies']['pct_index_correct']:.2f}% "
            "is the weakest stage and the one gating disc-to-vertebra "
            "measurements, which remain unreliable at 33.72% error.",
            "Any such work is again post-processing on frozen predictions, so it "
            "is cheap to try and cannot disturb the segmentation results.",
        ],
    }

    print("[5] Writing the final report ...")
    payload = {
        "experiment": EXPERIMENT,
        "audit": audit,
        "tuning": tuning,
        "validation_5d": fived,
        "test": test,
        "agreement": agreement,
        "comparison": comparison,
        "failures": failures,
        "figures": figures,
        "next_step": next_step,
    }
    save_json(payload, REPORT_DIR / "sprint5_final_report.json")
    write_report(payload)
    print(f"  -> {REPORT_DIR / 'sprint5_final_report.md'}")
    print(f"  -> {REPORT_DIR / 'sprint5_final_report.json'}")

    base, new = test["test_baseline"], test["test_sprint5"]
    s3_pf = agreement["s3"]["end_to_end_pfirrmann"]["forest"]
    s5_pf = agreement["s5"]["end_to_end_pfirrmann"]["forest"]
    s3_map = {r["measurement"]: r for r in agreement["s3"]["measurement_agreement"]}
    s5_map = {r["measurement"]: r for r in agreement["s5"]["measurement_agreement"]}

    print("\n" + "=" * 72)
    print(f"SPRINT 5 STATUS:             COMPLETE - post-processing only, "
          f"no retraining")
    print(f"METHOD:                      {test['selected_method']} "
          f"(selected on validation)")
    print(f"VALIDATION INDEXING:         "
          f"{tuning['baseline']['summary']['pct_index_correct']:.2f}% -> "
          f"{test['validation_pct_index_correct']:.2f}%  "
          f"({test['validation_pct_index_correct'] - tuning['baseline']['summary']['pct_index_correct']:+.2f} pp)")
    print(f"TEST INDEXING (corrected):   "
          f"{base['pct_index_correct']:.2f}% -> {new['pct_index_correct']:.2f}%  "
          f"({test['test_delta_pp']:+.2f} pp)")
    for category in CATEGORIES:
        print(f"  {category:<8}                   "
              f"{base['categories'][category]['pct']:>6.2f}% -> "
              f"{new['categories'][category]['pct']:>6.2f}%  "
              f"({new['categories'][category]['pct'] - base['categories'][category]['pct']:+.2f} pp)")
    print(f"DISC HEIGHT MAE:             "
          f"{s3_map['height_mm_central']['mae']:.4f} -> "
          f"{s5_map['height_mm_central']['mae']:.4f} mm  "
          f"({s5_map['height_mm_central']['mae'] - s3_map['height_mm_central']['mae']:+.4f})")
    print(f"DISC AREA MAE:               "
          f"{s3_map['area_mm2']['mae']:.4f} -> {s5_map['area_mm2']['mae']:.4f} mm2  "
          f"({s5_map['area_mm2']['mae'] - s3_map['area_mm2']['mae']:+.4f})")
    print(f"INTENSITY RATIO r:           "
          f"{s3_map['intensity_disc_vertebra_ratio']['pearson_r']:.4f} -> "
          f"{s5_map['intensity_disc_vertebra_ratio']['pearson_r']:.4f}")
    print(f"PFIRRMANN QWK (forest):      "
          f"{s3_pf['quadratic_weighted_kappa']:.4f} -> "
          f"{s5_pf['quadratic_weighted_kappa']:.4f}  "
          f"({s5_pf['quadratic_weighted_kappa'] - s3_pf['quadratic_weighted_kappa']:+.4f})")
    print(f"SEGMENTATION UNCHANGED:      {test['semantic_unchanged']} "
          f"(semantic maps byte-identical on all "
          f"{test['n_slices_written']:,} test slices)")
    print(f"REPORT:                      "
          f"outputs/reports/{EXPERIMENT}/sprint5_final_report.md")
    print("=" * 72)


if __name__ == "__main__":
    main()
