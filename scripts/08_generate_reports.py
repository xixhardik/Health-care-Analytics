"""Sprint 2 / Stages F+G - generate per-disc finding reports and the
longitudinal-ready table.

Writes:

    outputs/reports/finding_assessments/patient_XXX.json   machine-readable
    outputs/reports/finding_assessments/patient_XXX.md     readable
    outputs/reports/finding_assessments/patient_XXX.txt    compact tree layout
    outputs/reports/disc_records_longitudinal.csv          Stage G flat table
    outputs/reports/finding_assessments/INDEX.md           index of reports

Every reported value is tagged with its provenance. No composite severity score
is produced, and no timestamp is invented.

Usage
-----
    python scripts/08_generate_reports.py                 # sample of patients
    python scripts/08_generate_reports.py --all-patients
    python scripts/08_generate_reports.py --patients 1 2 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.report_generator import (  # noqa: E402
    DISCLAIMER,
    RECORD_SCHEMA_VERSION,
    build_longitudinal_ready_table,
    build_patient_report,
    render_patient_text,
    save_patient_report,
)
from src.utils.paths import ANALYSIS_REPORTS_DIR, ensure_dirs  # noqa: E402

REPORTS_DIR = ANALYSIS_REPORTS_DIR / "finding_assessments"
DISC_ANALYSIS_CSV = ANALYSIS_REPORTS_DIR / "disc_analysis.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-patients", action="store_true",
                        help="Generate a report for every patient.")
    parser.add_argument("--patients", type=int, nargs="*", default=None,
                        help="Specific patient ids.")
    parser.add_argument("--n-sample", type=int, default=6,
                        help="How many patients to sample when neither of the "
                             "above is given.")
    parser.add_argument("--primary-series-only", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if not DISC_ANALYSIS_CSV.exists():
        raise SystemExit(
            f"{DISC_ANALYSIS_CSV} not found. Run scripts/06_disc_analysis.py first."
        )
    analysis = pd.read_csv(DISC_ANALYSIS_CSV)

    # --- Stage G: longitudinal-ready flat table --------------------------
    longitudinal = build_longitudinal_ready_table(analysis)
    longitudinal_path = ANALYSIS_REPORTS_DIR / "disc_records_longitudinal.csv"
    longitudinal.to_csv(longitudinal_path, index=False)
    print(f"Stage G: longitudinal-ready table")
    print(f"  {len(longitudinal):,} rows x {len(longitudinal.columns)} columns")
    print(f"  keys preserved: patient_id, series_id, ivd_label")
    print(f"  time columns present but null: study_id, acquisition_timestamp, "
          f"timepoint_label")
    print(f"  -> {longitudinal_path}")

    # --- Stage F: per-patient reports ------------------------------------
    source = analysis[analysis["is_primary_series"]] if args.primary_series_only \
        else analysis

    if args.patients:
        patient_ids = args.patients
    elif args.all_patients:
        patient_ids = sorted(source["patient_id"].unique().tolist())
    else:
        # Spread the sample across splits so the example set is representative.
        patient_ids = []
        for split in ["train", "val", "test"]:
            subset = source[source["split"] == split]["patient_id"].unique()
            patient_ids.extend(sorted(subset)[: max(1, args.n_sample // 3)])
        patient_ids = sorted(set(patient_ids))[: args.n_sample]

    print(f"\nStage F: generating reports for {len(patient_ids)} patient(s) ...")
    written: list[Path] = []
    index_rows: list[dict] = []

    for patient_id in patient_ids:
        rows = source[source["patient_id"] == patient_id]
        if rows.empty:
            print(f"  patient {patient_id}: no disc records, skipped")
            continue

        report = build_patient_report(rows)
        paths = save_patient_report(report, REPORTS_DIR, formats=("json", "md", "txt"))
        written.extend(paths)
        index_rows.append({
            "patient_id": patient_id,
            "split": report.get("split"),
            "n_discs": report["n_discs"],
            "series": ", ".join(f"`{s}`" for s in report["series_ids"]),
            "json": f"[json](patient_{patient_id:03d}.json)",
            "markdown": f"[md](patient_{patient_id:03d}.md)",
            "text": f"[txt](patient_{patient_id:03d}.txt)",
        })
        print(f"  patient {patient_id:>3}: {report['n_discs']} discs -> "
              f"patient_{patient_id:03d}.{{json,md,txt}}")

    write_index(index_rows, len(longitudinal))

    # Show one report inline so the format is visible in the run log.
    if index_rows:
        example_id = index_rows[0]["patient_id"]
        example = build_patient_report(source[source["patient_id"] == example_id])
        print("\n" + "=" * 78)
        print("EXAMPLE REPORT (compact tree layout)")
        print("=" * 78)
        print(render_patient_text(example))
        print("=" * 78)

    print(f"\n  {len(written)} files written to {REPORTS_DIR}")
    print("\nDone. Findings are tagged by provenance; no severity score, no "
          "invented timestamps.")


def write_index(index_rows: list[dict], n_longitudinal_rows: int) -> Path:
    """Write an index of the generated reports."""
    from src.utils.reporting import MarkdownReport

    report = MarkdownReport(
        "Radiological Finding Assessments",
        "Automated MRI Finding Reports - Sprint 2 Stage F",
    )
    report.text(f"> {DISCLAIMER}")
    report.heading("Reports")
    report.table(
        index_rows,
        ["patient_id", "split", "n_discs", "series", "json", "markdown", "text"],
    )
    report.heading("Format")
    report.bullets(
        [
            "`.json` - machine-readable, one record per disc, schema version "
            f"`{RECORD_SCHEMA_VERSION}`. Carries the longitudinal keys.",
            "`.md` - readable tables of findings and measurements per disc level.",
            "`.txt` - the compact indented tree layout.",
        ]
    )
    report.heading("Provenance tags")
    report.table(
        [
            {"tag": "`dataset_annotation`",
             "meaning": "recorded by human graders in radiological_gradings.csv"},
            {"tag": "`measured_image_feature`",
             "meaning": "computed from the segmentation mask, in millimetres"},
            {"tag": "`model_prediction`",
             "meaning": "Stage E baseline estimate, reported with its probability"},
        ],
        ["tag", "meaning"],
    )
    report.heading("Level naming")
    report.text(
        "Disc index is authoritative. Conventional names (`L5-S1`, `L4-L5`, ...) are "
        "**provisional**: they assume the lowest annotated vertebra is L5, which the "
        "dataset documents as usual but not guaranteed - it can also be L4 or L6. "
        "Every record carries `level_name_is_provisional: true`."
    )
    report.heading("Longitudinal readiness")
    report.text(
        f"`outputs/reports/disc_records_longitudinal.csv` holds "
        f"{n_longitudinal_rows:,} rows keyed on `(patient_id, series_id, ivd_label)` "
        f"with `study_id`, `acquisition_timestamp` and `timepoint_label` present but "
        f"**null**. This dataset is cross-sectional, so those fields are deliberately "
        f"empty rather than populated with invented values. See "
        f"`outputs/reports/longitudinal_plan.md`."
    )
    return report.save(REPORTS_DIR / "INDEX.md")


if __name__ == "__main__":
    main()
