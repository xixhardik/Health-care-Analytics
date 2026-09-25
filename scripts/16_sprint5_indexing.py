"""Sprint 5 - disc indexing post-processing: audit, tune, select, apply.

No training. The segmentation predictions are read frozen and never rewritten;
only the disc/vertebra *identity* attached to them changes.

Phases
------
``--audit``
    Build the compact caches for the validation and test splits from the frozen
    Sprint 3 predictions, reproduce the Sprint 3 baseline taxonomy as a control,
    and report the current error breakdown.

``--tune``
    Sweep the 5A/5B/5C parameters on the **validation split only** and record
    every configuration. The test cache is not read in this phase.

``--apply``
    Take the single configuration selected on validation, apply it once to the
    test split, and write a new predictions directory with the corrected
    ``instance`` map and a byte-identical ``semantic`` map.

Usage
-----
    python scripts/16_sprint5_indexing.py --audit
    python scripts/16_sprint5_indexing.py --tune
    python scripts/16_sprint5_indexing.py --apply
"""

from __future__ import annotations

import argparse
import itertools
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis import disc_postprocess as pp  # noqa: E402
from src.analysis import indexing_diagnostics as idx_diag  # noqa: E402
from src.models.data import load_slice_index  # noqa: E402
from src.utils.paths import OUTPUTS_DIR, PROJECT_ROOT, ensure_dirs  # noqa: E402
from src.utils.reporting import save_json  # noqa: E402

EXPERIMENT = "sprint5_indexing"
REPORT_DIR = OUTPUTS_DIR / "reports" / EXPERIMENT
CACHE_DIR = REPORT_DIR / "cache"

#: Frozen inputs. Read only, never written.
TEST_PREDICTIONS = PROJECT_ROOT / "data" / "processed" / "predictions_sprint3_coverage"
VAL_PREDICTIONS = PROJECT_ROOT / "data" / "processed" / "predictions_sprint5_val_raw"

#: Output of --apply.
TEST_OUT = PROJECT_ROOT / "data" / "processed" / f"predictions_{EXPERIMENT}"

#: The Sprint 3 corrected-taxonomy baseline on test, as published.
SPRINT3_TEST_INDEX_CORRECT = 84.08


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--tune-5d", action="store_true", dest="tune_5d")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def cache_path(split: str) -> Path:
    return CACHE_DIR / f"{split}_cache.pkl"


def build_cache(split: str, predictions_dir: Path) -> dict:
    """Build the compact per-slice cache for one split."""
    index = load_slice_index()
    rows = index[index["split"] == split].reset_index(drop=True)
    entries: dict[str, dict] = {}
    meta: dict[str, dict] = {}
    started = time.perf_counter()

    for position, (_, row) in enumerate(rows.iterrows(), start=1):
        prediction_path = predictions_dir / f"{row['slice_id']}.npz"
        if not prediction_path.exists():
            continue
        with np.load(PROJECT_ROOT / row["npz_path"]) as bundle:
            truth_instance = bundle["mask_instance"]
        with np.load(prediction_path) as bundle:
            semantic = bundle["semantic"]
            saved_instance = bundle["instance"]

        entry = pp.build_slice_cache(truth_instance, semantic)
        # Keep the production index assignment so the control check can compare
        # against what the pipeline actually saved.
        entry["saved_regions_summary"] = {
            index_value: int((saved_instance == pp.IVD_OFFSET + index_value).sum())
            for index_value in range(1, pp.MAX_DISCS + 1)
            if int((saved_instance == pp.IVD_OFFSET + index_value).sum())
            >= pp.MIN_AREA_PX
        }
        entries[row["slice_id"]] = entry
        meta[row["slice_id"]] = {
            "image_id": row["image_id"],
            "patient_id": int(row["patient_id"]),
            "modality": row["modality"],
            "labelled_pixels": int(row.get("labelled_pixels_processed", 0)),
        }
        if position % 400 == 0:
            print(f"    {position}/{len(rows)} slices cached", flush=True)

    elapsed = time.perf_counter() - started
    print(f"    {len(entries)} slices cached in {elapsed:.1f}s")
    return {"entries": entries, "meta": meta, "split": split,
            "predictions_dir": str(predictions_dir), "build_seconds": elapsed}


def load_cache(split: str, predictions_dir: Path, rebuild: bool = False) -> dict:
    path = cache_path(split)
    if path.exists() and not rebuild:
        with path.open("rb") as handle:
            return pickle.load(handle)
    print(f"  building {split} cache from {predictions_dir.name} ...")
    cache = build_cache(split, predictions_dir)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(cache, handle)
    return cache


# ---------------------------------------------------------------------------
# Scoring a configuration
# ---------------------------------------------------------------------------


def score_configuration(
    cache: dict, params: pp.PostProcessParams, *, baseline: bool = False,
    informative_only: bool = False,
) -> dict:
    """Score one configuration over a whole split. Returns the taxonomy summary."""
    entries, meta = cache["entries"], cache["meta"]
    by_series: dict[str, list[str]] = {}
    for slice_id, info in meta.items():
        by_series.setdefault(info["image_id"], []).append(slice_id)

    disc_records: list[dict] = []
    slice_records: list[dict] = []
    diagnostics = {"n_tracks": [], "reject_reasons": {}, "unassigned_reasons": {},
                   "n_informative_slices": 0, "n_slices": 0}
    started = time.perf_counter()

    for image_id, slice_ids in by_series.items():
        series_entries = {sid: entries[sid] for sid in slice_ids}

        if baseline:
            assignment = {
                sid: pp.assign_baseline(entry, params)
                for sid, entry in series_entries.items()
            }
            info = {"n_tracks": 0, "reject_reasons": {}, "unassigned_reasons": {},
                    "n_informative_slices": len(slice_ids),
                    "informative": {sid: True for sid in slice_ids}}
        else:
            assignment, info = pp.assign_series(series_entries, params)

        diagnostics["n_tracks"].append(info.get("n_tracks", 0))
        diagnostics["n_informative_slices"] += info.get("n_informative_slices", 0)
        diagnostics["n_slices"] += len(slice_ids)
        for key, value in (info.get("reject_reasons") or {}).items():
            diagnostics["reject_reasons"][key] = (
                diagnostics["reject_reasons"].get(key, 0) + value
            )
        for key, value in (info.get("unassigned_reasons") or {}).items():
            diagnostics["unassigned_reasons"][key] = (
                diagnostics["unassigned_reasons"].get(key, 0) + value
            )

        informative_map = info.get("informative", {})
        for slice_id in slice_ids:
            if informative_only and not informative_map.get(slice_id, True):
                continue
            result = pp.classify_from_cache(
                series_entries[slice_id], assignment.get(slice_id, {})
            )
            context = {"slice_id": slice_id, **meta[slice_id]}
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
                    "n_correct": result["n_correct"],
                }
            )

    summary = pp.summarise_records(disc_records, slice_records)
    summary["runtime_seconds"] = round(time.perf_counter() - started, 2)
    summary["mean_tracks_per_series"] = (
        round(float(np.mean(diagnostics["n_tracks"])), 2)
        if diagnostics["n_tracks"] else 0.0
    )
    summary["n_components_rejected"] = int(
        sum(diagnostics["reject_reasons"].values())
    )
    summary["reject_reasons"] = diagnostics["reject_reasons"]
    summary["n_components_unassigned"] = int(
        sum(diagnostics["unassigned_reasons"].values())
    )
    summary["unassigned_reasons"] = diagnostics["unassigned_reasons"]
    summary["n_informative_slices"] = diagnostics["n_informative_slices"]
    summary["pct_slices_informative"] = round(
        100 * diagnostics["n_informative_slices"] / max(1, diagnostics["n_slices"]), 2
    )
    return summary, disc_records


# ---------------------------------------------------------------------------
# Phase: audit
# ---------------------------------------------------------------------------


def run_audit(args: argparse.Namespace) -> dict:
    print("=" * 74)
    print("SPRINT 5 PHASE 1 - AUDIT OF CURRENT INDEXING")
    print("=" * 74)

    val_cache = load_cache("val", VAL_PREDICTIONS, args.rebuild_cache)
    test_cache = load_cache("test", TEST_PREDICTIONS, args.rebuild_cache)

    baseline_params = pp.PostProcessParams(
        use_series_tracks=False, use_vertebral_bodies=False,
        col_tolerance_px=None, informative_min_ivd_px=0,
    )

    print("\nBaseline (per-slice ordered components, the current behaviour):")
    results = {}
    for split, cache in (("val", val_cache), ("test", test_cache)):
        summary, _ = score_configuration(cache, baseline_params, baseline=True)
        results[split] = summary
        cat = summary["categories"]
        print(f"\n  {split.upper()}  ({summary['n_discs_analysed']:,} discs over "
              f"{summary['n_slices_analysed']:,} slices)")
        for name in ("correct", "shifted", "merged", "split", "missed"):
            print(f"    {name:<9} {cat[name]['n']:>6,}  {cat[name]['pct']:>6.2f}%")
        print(f"    region found                {summary['pct_region_found']:>6.2f}%")
        print(f"    slices, count correct       "
              f"{summary['pct_slices_count_correct']:>6.2f}%")
        print(f"    slices, all discs correct   "
              f"{summary['pct_slices_all_discs_correct']:>6.2f}%")
        print(f"    spurious components         "
              f"{summary['total_spurious_components']:>6,}")
        print(f"    shifted by exactly +/-1     "
              f"{summary['pct_shifted_by_one']:>6.2f}% of all discs")
        print(f"    shift offsets: {summary['shifted_offsets']}")

    # --- control: does the cache-based scorer reproduce the published number? --
    published = SPRINT3_TEST_INDEX_CORRECT
    measured = results["test"]["pct_index_correct"]
    print("\n" + "-" * 74)
    print("CONTROL - cache-based scorer vs the published Sprint 3 taxonomy")
    print("-" * 74)
    print(f"  published Sprint 3 test pct_index_correct : {published:.2f}%")
    print(f"  re-measured here                          : {measured:.2f}%")
    print(f"  difference                                : {measured - published:+.2f} pp")
    agrees = abs(measured - published) < 0.05
    print(f"  {'AGREES' if agrees else 'DOES NOT AGREE'} - the estimator is "
          f"{'identical' if agrees else 'NOT identical'} to Sprint 3's")
    if not agrees:
        print("  WARNING: comparisons against 84.08% would not be like-for-like.")

    save_json(
        {
            "baseline_params": baseline_params.to_dict(),
            "val": results["val"],
            "test": results["test"],
            "control": {
                "published_sprint3_test_pct_index_correct": published,
                "remeasured_test_pct_index_correct": measured,
                "difference_pp": round(measured - published, 4),
                "estimator_agrees": bool(agrees),
            },
        },
        REPORT_DIR / "baseline_audit.json",
    )
    print(f"\n  -> {REPORT_DIR / 'baseline_audit.json'}")
    return results


# ---------------------------------------------------------------------------
# Phase: tune (validation only)
# ---------------------------------------------------------------------------


def run_tune(args: argparse.Namespace) -> dict:
    print("=" * 74)
    print("SPRINT 5 PHASE 2 - PARAMETER SWEEP ON VALIDATION ONLY")
    print("=" * 74)
    print("The test cache is not read in this phase.")

    val_cache = load_cache("val", VAL_PREDICTIONS, args.rebuild_cache)

    baseline_params = pp.PostProcessParams(
        use_series_tracks=False, use_vertebral_bodies=False, col_tolerance_px=None
    )
    baseline_summary, _ = score_configuration(
        val_cache, baseline_params, baseline=True
    )
    baseline_correct = baseline_summary["pct_index_correct"]
    print(f"\nValidation baseline: {baseline_correct:.2f}% index correct")

    configurations: list[dict] = []

    # --- 5B alone: area threshold and column band -------------------------
    print("\n[5B] spurious-component rejection (no series ordering)")
    for min_area in (20, 30, 40, 60, 80):
        for col_tol in (None, 40.0, 30.0, 20.0):
            params = pp.PostProcessParams(
                min_area_px=min_area, col_tolerance_px=col_tol,
                use_series_tracks=False, use_vertebral_bodies=False,
            )
            summary, _ = score_configuration(val_cache, params)
            configurations.append(
                {"method": "5B", "params": params.to_dict(), "summary": summary}
            )
            print(f"  min_area={min_area:<3} col_tol={str(col_tol):<5} -> "
                  f"correct {summary['pct_index_correct']:.2f}%  "
                  f"missed {summary['categories']['missed']['pct']:.2f}%  "
                  f"shifted {summary['categories']['shifted']['pct']:.2f}%")

    # --- 5A alone: series-level tracks ------------------------------------
    print("\n[5A] series-level ordering")
    for cluster_px in (8.0, 12.0, 16.0, 20.0):
        for support in (0.1, 0.25, 0.4):
            for gate in (14.0, 18.0, 25.0):
                params = pp.PostProcessParams(
                    use_series_tracks=True, cluster_px=cluster_px,
                    min_track_support=support, assign_max_dist_px=gate,
                    use_vertebral_bodies=False,
                )
                summary, _ = score_configuration(val_cache, params)
                configurations.append(
                    {"method": "5A", "params": params.to_dict(),
                     "summary": summary}
                )
                print(f"  cluster={cluster_px:<5} support={support:<5} "
                      f"gate={gate:<5} -> "
                      f"correct {summary['pct_index_correct']:.2f}%  "
                      f"tracks/series {summary['mean_tracks_per_series']}")

    # --- 5A + 5B combined -------------------------------------------------
    print("\n[5A+5B] series-level ordering with candidate rejection")
    best_a = max(
        (c for c in configurations if c["method"] == "5A"),
        key=lambda c: c["summary"]["pct_index_correct"],
    )
    for min_area in (20, 30, 40, 60):
        for col_tol in (None, 40.0, 30.0):
            params = pp.PostProcessParams(
                **{**best_a["params"], "min_area_px": min_area,
                   "col_tolerance_px": col_tol}
            )
            summary, _ = score_configuration(val_cache, params)
            configurations.append(
                {"method": "5A+5B", "params": params.to_dict(),
                 "summary": summary}
            )
            print(f"  min_area={min_area:<3} col_tol={str(col_tol):<5} -> "
                  f"correct {summary['pct_index_correct']:.2f}%  "
                  f"missed {summary['categories']['missed']['pct']:.2f}%")

    # --- 5C: informative-slice restriction --------------------------------
    print("\n[5C] informative-slice criterion")
    best_ab = max(
        (c for c in configurations if c["method"] in ("5A", "5A+5B")),
        key=lambda c: c["summary"]["pct_index_correct"],
    )
    for min_ivd in (0, 200, 400, 800, 1200):
        params = pp.PostProcessParams(
            **{**best_ab["params"], "informative_min_ivd_px": min_ivd}
        )
        # Scored on ALL slices: the criterion only decides which slices vote for
        # the tracks, so the denominator is unchanged and the comparison stays
        # like-for-like.
        summary_all, _ = score_configuration(val_cache, params)
        # Scored on the informative subset only, reported separately with its
        # coverage so a restricted denominator can never be mistaken for a gain.
        summary_subset, _ = score_configuration(
            val_cache, params, informative_only=True
        )
        configurations.append(
            {"method": "5C(all slices)", "params": params.to_dict(),
             "summary": summary_all}
        )
        configurations.append(
            {"method": "5C(informative subset)", "params": params.to_dict(),
             "summary": summary_subset}
        )
        print(f"  min_ivd_px={min_ivd:<5} -> all slices "
              f"{summary_all['pct_index_correct']:.2f}%  |  informative subset "
              f"{summary_subset['pct_index_correct']:.2f}% "
              f"(covers {summary_subset['n_discs_analysed']:,}/"
              f"{summary_all['n_discs_analysed']:,} discs, "
              f"{summary_all['pct_slices_informative']:.1f}% of slices)")

    # --- selection ---------------------------------------------------------
    # Selected on validation only, on the all-slices denominator so the metric is
    # comparable with the Sprint 3 baseline.
    eligible = [
        c for c in configurations
        if c["method"] in ("5B", "5A", "5A+5B", "5C(all slices)")
    ]
    selected = max(eligible, key=lambda c: c["summary"]["pct_index_correct"])

    print("\n" + "=" * 74)
    print("SELECTED ON VALIDATION")
    print("=" * 74)
    print(f"  method            : {selected['method']}")
    print(f"  validation correct: {selected['summary']['pct_index_correct']:.2f}% "
          f"(baseline {baseline_correct:.2f}%, "
          f"{selected['summary']['pct_index_correct'] - baseline_correct:+.2f} pp)")
    print(f"  parameters        :")
    for key, value in selected["params"].items():
        print(f"      {key} = {value}")

    payload = {
        "split_used_for_tuning": "val",
        "test_touched": False,
        "baseline": {"params": baseline_params.to_dict(),
                     "summary": baseline_summary},
        "n_configurations": len(configurations),
        "configurations": configurations,
        "selected": selected,
    }
    save_json(payload, REPORT_DIR / "validation_tuning.json")
    pd.DataFrame(
        [
            {
                "method": c["method"],
                **{k: v for k, v in c["params"].items()},
                "val_pct_index_correct": c["summary"]["pct_index_correct"],
                "val_pct_region_found": c["summary"]["pct_region_found"],
                "val_pct_correct_cat": c["summary"]["categories"]["correct"]["pct"],
                "val_pct_shifted": c["summary"]["categories"]["shifted"]["pct"],
                "val_pct_merged": c["summary"]["categories"]["merged"]["pct"],
                "val_pct_split": c["summary"]["categories"]["split"]["pct"],
                "val_pct_missed": c["summary"]["categories"]["missed"]["pct"],
                "val_n_discs": c["summary"]["n_discs_analysed"],
                "val_spurious": c["summary"]["total_spurious_components"],
                "mean_tracks_per_series": c["summary"]["mean_tracks_per_series"],
                "runtime_seconds": c["summary"]["runtime_seconds"],
            }
            for c in configurations
        ]
    ).to_csv(REPORT_DIR / "validation_sweep.csv", index=False)
    print(f"\n  -> {REPORT_DIR / 'validation_tuning.json'}")
    print(f"  -> {REPORT_DIR / 'validation_sweep.csv'}")
    return payload


# ---------------------------------------------------------------------------
# Phase: apply to test (once)
# ---------------------------------------------------------------------------


def run_apply(args: argparse.Namespace) -> dict:
    print("=" * 74)
    print("SPRINT 5 PHASE 3 - APPLY THE SELECTED METHOD TO TEST (ONCE)")
    print("=" * 74)

    tuning = json.loads(
        (REPORT_DIR / "validation_tuning.json").read_text(encoding="utf-8")
    )
    selected = tuning["selected"]

    # Method 5D is decided by its own validation measurement, since the disc
    # taxonomy does not score vertebrae.
    fived_path = REPORT_DIR / "validation_5d.json"
    adopt_5d = False
    fived = None
    if fived_path.exists():
        fived = json.loads(fived_path.read_text(encoding="utf-8"))
        adopt_5d = bool(fived["adopt_5d"])

    params = pp.PostProcessParams(
        **{**selected["params"], "use_vertebral_bodies": adopt_5d}
    )
    method_name = selected["method"] + ("+5D" if adopt_5d else "")
    print(f"  method (selected on validation): {method_name}")
    print(f"  validation index correct       : "
          f"{selected['summary']['pct_index_correct']:.2f}%")
    if fived:
        print(f"  5D vertebral bodies            : adopted={adopt_5d} "
              f"(validation {fived['schemes']['vertebral_bodies']['pct_index_correct']}% "
              f"vs {fived['schemes']['ordered_components']['pct_index_correct']}% "
              f"for ordered components)")

    test_cache = load_cache("test", TEST_PREDICTIONS, args.rebuild_cache)

    baseline_params = pp.PostProcessParams(
        use_series_tracks=False, use_vertebral_bodies=False, col_tolerance_px=None
    )
    baseline_summary, baseline_records = score_configuration(
        test_cache, baseline_params, baseline=True
    )
    final_summary, final_records = score_configuration(test_cache, params)

    print(f"\n  test baseline  : {baseline_summary['pct_index_correct']:.2f}%")
    print(f"  test Sprint 5  : {final_summary['pct_index_correct']:.2f}%  "
          f"({final_summary['pct_index_correct'] - baseline_summary['pct_index_correct']:+.2f} pp)")

    # --- write the new predictions (semantic copied unchanged) -------------
    print(f"\n  writing corrected instance maps -> {TEST_OUT}")
    TEST_OUT.mkdir(parents=True, exist_ok=True)
    index = load_slice_index()
    test_rows = index[index["split"] == "test"].reset_index(drop=True)
    by_series: dict[str, list] = {}
    for _, row in test_rows.iterrows():
        by_series.setdefault(row["image_id"], []).append(row)

    n_written = 0
    semantic_identical = True
    started = time.perf_counter()
    for image_id, rows in by_series.items():
        semantic_by_slice: dict[str, np.ndarray] = {}
        for row in rows:
            path = TEST_PREDICTIONS / f"{row['slice_id']}.npz"
            if not path.exists():
                continue
            with np.load(path) as bundle:
                semantic_by_slice[row["slice_id"]] = bundle["semantic"]
        if not semantic_by_slice:
            continue
        result = pp.process_series(semantic_by_slice, params=params)
        for slice_id, instance_map in result["instance_maps"].items():
            source = semantic_by_slice[slice_id]
            np.savez_compressed(
                TEST_OUT / f"{slice_id}.npz",
                semantic=source.astype(np.uint8),
                instance=instance_map.astype(np.uint8),
            )
            # Verify the segmentation itself was not altered.
            with np.load(TEST_OUT / f"{slice_id}.npz") as check:
                if not np.array_equal(check["semantic"], source):
                    semantic_identical = False
            n_written += 1
        if n_written % 400 < len(result["instance_maps"]):
            print(f"    {n_written} slices written", flush=True)

    elapsed = time.perf_counter() - started
    print(f"  {n_written} slices written in {elapsed:.1f}s")
    print(f"  semantic maps byte-identical to the Sprint 3 predictions: "
          f"{semantic_identical}")

    pd.DataFrame(final_records).to_csv(
        REPORT_DIR / "indexing_failures_sprint5_test.csv", index=False
    )

    payload = {
        "selected_method": method_name,
        "selected_params": params.to_dict(),
        "selected_on": "validation",
        "adopt_5d": adopt_5d,
        "validation_5d": fived["schemes"] if fived else None,
        "validation_pct_index_correct": selected["summary"]["pct_index_correct"],
        "test_baseline": baseline_summary,
        "test_sprint5": final_summary,
        "test_delta_pp": round(
            final_summary["pct_index_correct"]
            - baseline_summary["pct_index_correct"], 3
        ),
        "published_sprint3_baseline": SPRINT3_TEST_INDEX_CORRECT,
        "n_slices_written": n_written,
        "semantic_unchanged": bool(semantic_identical),
        "write_seconds": round(elapsed, 1),
        "predictions_dir": str(TEST_OUT.relative_to(PROJECT_ROOT)).replace("\\", "/"),
    }
    save_json(payload, REPORT_DIR / "test_results.json")
    print(f"\n  -> {REPORT_DIR / 'test_results.json'}")
    print(f"  -> {REPORT_DIR / 'indexing_failures_sprint5_test.csv'}")
    return payload


# ---------------------------------------------------------------------------
# Phase: 5D - vertebral bodies, evaluated on validation
# ---------------------------------------------------------------------------


def run_tune_5d(args: argparse.Namespace) -> dict:
    """Compare vertebra numbering schemes on validation.

    The disc taxonomy says nothing about vertebrae, so method 5D needs its own
    measurement. Both schemes are scored with the *same* taxonomy rules against
    the ground-truth vertebra instances:

    ``ordered components``
        the current behaviour - number every vertebra connected component
        bottom-up. The Sprint 3 audit predicted this must fail, because a
        vertebra occupies ~2 components in a sagittal slice.
    ``vertebral bodies (5D)``
        keep only components anterior to the canal, then give each body the
        index of the disc track directly above it.
    """
    print("=" * 74)
    print("SPRINT 5 PHASE 2b - METHOD 5D ON VALIDATION")
    print("=" * 74)

    tuning = json.loads(
        (REPORT_DIR / "validation_tuning.json").read_text(encoding="utf-8")
    )
    disc_params = pp.PostProcessParams(**tuning["selected"]["params"])

    index = load_slice_index()
    rows = index[index["split"] == "val"].reset_index(drop=True)
    by_series: dict[str, list] = {}
    for _, row in rows.iterrows():
        by_series.setdefault(row["image_id"], []).append(row)

    tallies = {
        "ordered_components": [],
        "vertebral_bodies": [],
    }
    started = time.perf_counter()
    n_slices = 0

    for image_id, series_rows in by_series.items():
        semantic_by_slice: dict[str, np.ndarray] = {}
        truth_by_slice: dict[str, np.ndarray] = {}
        for row in series_rows:
            path = VAL_PREDICTIONS / f"{row['slice_id']}.npz"
            if not path.exists():
                continue
            with np.load(path) as bundle:
                semantic_by_slice[row["slice_id"]] = bundle["semantic"]
            with np.load(PROJECT_ROOT / row["npz_path"]) as bundle:
                truth_by_slice[row["slice_id"]] = bundle["mask_instance"]
        if not semantic_by_slice:
            continue

        for scheme, use_bodies in (("ordered_components", False),
                                   ("vertebral_bodies", True)):
            params = pp.PostProcessParams(
                **{**disc_params.to_dict(), "use_vertebral_bodies": use_bodies}
            )
            result = pp.process_series(semantic_by_slice, params=params)
            for slice_id, instance_map in result["instance_maps"].items():
                truth_regions = pp.vertebra_truth_regions(truth_by_slice[slice_id])
                if not truth_regions:
                    continue
                predicted = {
                    value: (instance_map == value)
                    for value in range(1, pp.MAX_DISCS + 1)
                    if int((instance_map == value).sum()) >= pp.MIN_AREA_PX
                }
                tallies[scheme].extend(
                    pp.classify_generic(truth_regions, predicted)
                )
        n_slices += len(semantic_by_slice)

    elapsed = time.perf_counter() - started
    print(f"\n  {n_slices:,} validation slices, {elapsed:.1f}s")

    summaries = {}
    for scheme, records in tallies.items():
        frame = pd.DataFrame(records)
        total = len(frame)
        counts = frame["category"].value_counts().to_dict()
        summaries[scheme] = {
            "n_vertebrae_analysed": int(total),
            "categories": {
                name: {
                    "n": int(counts.get(name, 0)),
                    "pct": round(100 * counts.get(name, 0) / total, 2),
                }
                for name in ("correct", "shifted", "merged", "split", "missed")
            },
            "pct_index_correct": round(100 * counts.get("correct", 0) / total, 2),
        }
        print(f"\n  {scheme}  ({total:,} ground-truth vertebrae)")
        for name in ("correct", "shifted", "merged", "split", "missed"):
            entry = summaries[scheme]["categories"][name]
            print(f"    {name:<9} {entry['n']:>6,}  {entry['pct']:>6.2f}%")

    legacy = summaries["ordered_components"]["pct_index_correct"]
    bodies = summaries["vertebral_bodies"]["pct_index_correct"]
    delta = bodies - legacy
    better = delta > 0
    print("\n" + "-" * 74)
    print(f"  ordered components : {legacy:.2f}% correct")
    print(f"  vertebral bodies   : {bodies:.2f}% correct  ({delta:+.2f} pp)")
    print(f"  -> 5D is {'ADOPTED' if better else 'REJECTED'} on validation evidence")

    payload = {
        "split_used": "val",
        "disc_params": disc_params.to_dict(),
        "schemes": summaries,
        "delta_pp": round(delta, 2),
        "adopt_5d": bool(better),
        "runtime_seconds": round(elapsed, 1),
    }
    save_json(payload, REPORT_DIR / "validation_5d.json")
    print(f"\n  -> {REPORT_DIR / 'validation_5d.json'}")
    return payload


def main() -> None:
    args = parse_args()
    ensure_dirs()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if not (args.audit or args.tune or args.tune_5d or args.apply):
        raise SystemExit(
            "Pick at least one of --audit / --tune / --tune-5d / --apply"
        )
    if args.audit:
        run_audit(args)
    if args.tune:
        run_tune(args)
    if args.tune_5d:
        run_tune_5d(args)
    if args.apply:
        run_apply(args)


if __name__ == "__main__":
    main()
