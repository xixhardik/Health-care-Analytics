"""Sprint 1 / Step 8 - patient-level train/validation/test split.

Splits by ``patient_id``, never by slice, so no patient contributes slices to
more than one split. Writes:

    data/processed/splits.csv                              series -> split
    data/processed/slice_index.csv                         updated with 'split'
    outputs/preprocessing_reports/split_report.md
    outputs/preprocessing_reports/split_report.json
    outputs/visualizations/split_distribution.png

No model is trained here.

Usage
-----
    python scripts/04_split.py
    python scripts/04_split.py --seed 7 --train 0.6 --val 0.2 --test 0.2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import pandas as pd

# Figures are written to disk, never displayed: pick the headless backend
# before pyplot is imported anywhere.
matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing.splits import (  # noqa: E402
    create_dataset_split,
    split_fraction_table,
    summarise_split,
    verify_no_leakage,
)
from src.preprocessing.visualize import visualize_split  # noqa: E402
from src.utils.paths import (  # noqa: E402
    PAIRS_CSV,
    RANDOM_SEED,
    REPORTS_DIR,
    SLICE_INDEX_CSV,
    SPLIT_CSV,
    VISUALIZATIONS_DIR,
    ensure_dirs,
)
from src.utils.reporting import MarkdownReport, save_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--train", type=float, default=0.70)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.15)
    parser.add_argument("--no-stratify", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    pairs = pd.read_csv(PAIRS_CSV)
    fractions = {"train": args.train, "val": args.val, "test": args.test}

    print(f"Splitting {pairs['patient_id'].nunique()} patients "
          f"({len(pairs)} series) with seed={args.seed} and fractions={fractions}")

    split_frame = create_dataset_split(
        pairs, fractions=fractions, seed=args.seed, stratify=not args.no_stratify
    )

    # --- leakage check: this is the whole point of the exercise -----------
    leakage = verify_no_leakage(split_frame)
    print(f"\nLeakage check: ok={leakage['ok']}  "
          f"patients per split={leakage['ids_per_split']}")
    if not leakage["ok"]:
        raise SystemExit(f"Patient leakage detected: {leakage['overlaps']}")

    split_frame.to_csv(SPLIT_CSV, index=False)
    print(f"  -> {SPLIT_CSV}")

    # --- propagate the split onto the per-slice index ---------------------
    slice_index = None
    if SLICE_INDEX_CSV.exists():
        slice_index = pd.read_csv(SLICE_INDEX_CSV)
        mapping = split_frame.set_index("image_id")["split"]
        slice_index["split"] = slice_index["image_id"].map(mapping)

        unassigned = int(slice_index["split"].isna().sum())
        if unassigned:
            raise SystemExit(f"{unassigned} slices could not be assigned a split.")

        slice_index.to_csv(SLICE_INDEX_CSV, index=False)
        print(f"  -> {SLICE_INDEX_CSV} (added 'split' column)")

        slice_leakage = verify_no_leakage(slice_index)
        print(f"Slice-level leakage check: ok={slice_leakage['ok']}")
        if not slice_leakage["ok"]:
            raise SystemExit(f"Slice-level patient leakage: {slice_leakage['overlaps']}")
    else:
        print(f"  (no {SLICE_INDEX_CSV}; run scripts/03_preprocess.py for slice counts)")

    summary = summarise_split(split_frame, slice_index)
    table = split_fraction_table(summary)
    print("\n=== split summary ===")
    print(table.to_string(index=False))

    visualize_split(summary, save_path=VISUALIZATIONS_DIR / "split_distribution.png")
    print(f"\n  -> {VISUALIZATIONS_DIR / 'split_distribution.png'}")

    payload = {
        "seed": args.seed,
        "fractions_requested": fractions,
        "stratified": not args.no_stratify,
        "split_unit": "patient_id",
        "leakage_check": leakage,
        "counts": summary,
        "achieved": table.to_dict("records"),
    }
    write_report(payload, table, split_frame, slice_index)
    save_json(payload, REPORTS_DIR / "split_report.json")
    print(f"  -> {REPORTS_DIR / 'split_report.md'}")
    print("\nDone. Sprint 1 preprocessing is complete; no model has been trained.")


def write_report(
    payload: dict,
    table: pd.DataFrame,
    split_frame: pd.DataFrame,
    slice_index: pd.DataFrame | None,
) -> Path:
    """Compose the split report."""
    report = MarkdownReport(
        "Train / Validation / Test Split Report",
        "Patient-level split - Sprint 1",
    )

    report.heading("1. Splitting strategy")
    report.bullets(
        [
            f"**Split unit: `{payload['split_unit']}`.** Every series and every slice of "
            "a patient lands in exactly one split.",
            f"**Seed: {payload['seed']}** - fixed, so the split is reproducible.",
            f"**Requested proportions of patients:** {payload['fractions_requested']}.",
            f"**Stratified:** {payload['stratified']} (by sex and annotated-vertebra "
            "count, with rare combinations pooled so every stratum can be divided "
            "three ways).",
        ]
    )
    report.text(
        "**Why not split slices at random.** A single series contributes 8-154 "
        "adjacent sagittal slices that are nearly identical to their neighbours, and a "
        "patient contributes 1-3 series of the *same* anatomy. Dataset inspection also "
        "found that the T1 and T2 masks of a patient are frequently byte-identical - "
        "one annotation shared across co-registered series. Random slice splitting "
        "would therefore place near-copies of the same image in both training and test "
        "sets and inflate the eventual Dice/IoU scores. Splitting on the patient "
        "removes that path entirely."
    )

    report.heading("2. Resulting counts")
    report.dataframe(table, max_rows=10)
    if slice_index is None:
        report.text(
            "_Slice counts are zero because the split was generated before "
            "preprocessing; re-run after `scripts/03_preprocess.py`._"
        )

    report.heading("3. Leakage verification")
    leakage = payload["leakage_check"]
    report.key_values(
        {
            "No patient in more than one split": leakage["ok"],
            "Overlapping patients": leakage["overlaps"] or "none",
            "Patients per split": leakage["ids_per_split"],
            "Sum of per-split patient counts": leakage["total_ids_counted"],
            "Distinct patients in the dataset": leakage["total_ids_unique"],
        }
    )
    report.text(
        "The last two rows matching is the arithmetic proof that the split is a true "
        "partition: no patient was counted twice and none was dropped."
    )

    report.heading("4. Composition per split")
    for split in ["train", "val", "test"]:
        rows = split_frame[split_frame["split"] == split]
        report.heading(split, level=3)
        report.key_values(
            {
                "Patients": rows["patient_id"].nunique(),
                "Series": len(rows),
                "Modalities": rows["modality"].value_counts().to_dict(),
                "Sex": rows["sex"].value_counts().to_dict() if "sex" in rows else "-",
                "Mean annotated vertebrae": round(float(rows["num_vertebrae"].mean()), 2)
                if "num_vertebrae" in rows
                else "-",
                "Mean annotated discs": round(float(rows["num_discs"].mean()), 2)
                if "num_discs" in rows
                else "-",
            }
        )

    report.heading("5. Relationship to the dataset's own subset column")
    if "subset" in split_frame.columns:
        crosstab = pd.crosstab(split_frame["subset"], split_frame["split"])
        report.text(
            "`overview.csv` ships a `subset` column, but it only distinguishes "
            "training from validation (no test set) and is defined per *series*. "
            "Sprint 1 therefore derives its own patient-level three-way split. The "
            "overlap between the two is shown below for reference."
        )
        report.dataframe(crosstab.reset_index(), max_rows=10)

    report.heading("6. Files")
    report.bullets(
        [
            "`data/processed/splits.csv` - series-level split assignment",
            "`data/processed/slice_index.csv` - per-slice index including `split`",
            "`outputs/visualizations/split_distribution.png` - split chart",
        ]
    )
    report.text(
        "**No model has been trained.** The test split has not been looked at beyond "
        "counting it, and no Dice or IoU value exists yet."
    )

    return report.save(REPORTS_DIR / "split_report.md")


if __name__ == "__main__":
    main()
