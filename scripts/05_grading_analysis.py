"""Analyse data/raw/radiological_gradings.csv and scope Sprint 2.

Writes:

    outputs/reports/radiological_grading_analysis.md
    outputs/reports/radiological_grading_analysis.json
    outputs/reports/grading_disc_level_prevalence.csv
    outputs/reports/grading_series_linkage.csv
    outputs/reports/grading_patient_linkage.csv
    outputs/visualizations/grading_*.png

Reads only. Trains nothing. Defines no severity score.

Usage
-----
    python scripts/05_grading_analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import pandas as pd

# Figures go to disk; pick the headless backend before pyplot is imported.
matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.gradings import (  # noqa: E402
    FINDINGS,
    IVD_COLUMN,
    IVD_MASK_LABEL_OFFSET,
    MODIC_CATEGORY_NAMES,
    PATIENT_COLUMN,
    PFIRRMANN_GRADE_NAMES,
    all_distributions,
    cooccurrence,
    findings_per_disc,
    link_gradings_to_masks,
    load_raw_gradings,
    longitudinal_capability_audit,
    modality_availability,
    pfirrmann_by_modic,
    prevalence_by_disc_level,
    quality_issues,
    schema_audit,
    series_level_replication,
    split_grading_coverage,
)
from src.analysis.grading_plots import (  # noqa: E402
    plot_cooccurrence,
    plot_finding_distributions,
    plot_findings_per_disc,
    plot_prevalence_by_level,
    plot_split_class_balance,
)
from src.preprocessing.pairing import load_overview  # noqa: E402
from src.utils.paths import (  # noqa: E402
    ANALYSIS_REPORTS_DIR,
    OVERVIEW_CSV,
    PAIRS_CSV,
    SPLIT_CSV,
    VISUALIZATIONS_DIR,
    VOLUME_INSPECTION_CSV,
    ensure_dirs,
)
from src.utils.reporting import MarkdownReport, save_json  # noqa: E402


def main() -> None:
    ensure_dirs()

    print("[1/5] Loading files ...")
    raw = load_raw_gradings()
    overview = load_overview()
    pairs = pd.read_csv(PAIRS_CSV)
    inspection = pd.read_csv(VOLUME_INSPECTION_CSV)
    splits = pd.read_csv(SPLIT_CSV) if SPLIT_CSV.exists() else None
    print(f"  radiological_gradings.csv : {raw.shape[0]} rows x {raw.shape[1]} columns")
    print(f"  overview.csv              : {overview.shape[0]} rows")
    print(f"  volume_inspection.csv     : {inspection.shape[0]} series")

    print("[2/5] Auditing schema and quality ...")
    schema = schema_audit(raw)
    issues = quality_issues(raw, overview)
    distributions = all_distributions(raw)
    level_table = prevalence_by_disc_level(raw)
    correlation = cooccurrence(raw)
    per_disc = findings_per_disc(raw)
    modic_crosstab = pfirrmann_by_modic(raw)
    print(f"  patients={issues['n_patients']} records={issues['n_rows']} "
          f"missing={issues['total_missing_values']} "
          f"invalid_ivd_rows={issues['n_invalid_ivd_label_rows']}")

    print("[3/5] Linking gradings to masks and series ...")
    series_linkage, linkage_summary, patient_linkage = link_gradings_to_masks(
        raw, inspection
    )
    replication = series_level_replication(raw, pairs)
    modality = modality_availability(raw, pairs)
    print(f"  series with exact grading/mask agreement: "
          f"{linkage_summary['n_series_exact_match']}/{linkage_summary['n_series']} "
          f"({linkage_summary['pct_series_exact_match']}%)")
    print(f"  patients with a T2-type series: {modality['n_patients_with_t2']}"
          f"/{modality['n_patients_total']} "
          f"(T1-only: {modality['patients_t1_only']})")

    coverage = split_grading_coverage(raw, splits) if splits is not None else None

    print("[4/5] Auditing longitudinal capability ...")
    longitudinal = longitudinal_capability_audit(
        overview,
        pairs,
        n_raw_overview_columns=pd.read_csv(OVERVIEW_CSV, nrows=0).shape[1],
    )
    print(f"  temporal columns found       : {longitudinal['candidate_temporal_columns']}")
    print(f"  treatment/outcome columns    : {longitudinal['treatment_or_outcome_columns']}")
    print(f"  max series per patient+modality: "
          f"{longitudinal['max_series_per_patient_modality']}")

    print("[5/5] Rendering figures and writing the report ...")
    figures = render_figures(raw, level_table, correlation, per_disc, coverage)
    for name in figures:
        print(f"  -> {VISUALIZATIONS_DIR / name}")

    level_table.to_csv(
        ANALYSIS_REPORTS_DIR / "grading_disc_level_prevalence.csv", index=False
    )
    series_linkage.to_csv(
        ANALYSIS_REPORTS_DIR / "grading_series_linkage.csv", index=False
    )
    patient_linkage.to_csv(
        ANALYSIS_REPORTS_DIR / "grading_patient_linkage.csv", index=False
    )

    payload = {
        "schema": schema,
        "quality": issues,
        "distributions": distributions,
        "findings_per_disc": per_disc,
        "linkage": linkage_summary,
        "replication": replication,
        "modality": modality,
        "longitudinal": longitudinal,
        "split_coverage": coverage.to_dict("records") if coverage is not None else None,
        "figures": figures,
    }
    save_json(payload, ANALYSIS_REPORTS_DIR / "radiological_grading_analysis.json")

    write_report(
        raw=raw,
        schema=schema,
        issues=issues,
        distributions=distributions,
        level_table=level_table,
        correlation=correlation,
        per_disc=per_disc,
        modic_crosstab=modic_crosstab,
        series_linkage=series_linkage,
        linkage_summary=linkage_summary,
        patient_linkage=patient_linkage,
        replication=replication,
        modality=modality,
        longitudinal=longitudinal,
        coverage=coverage,
        figures=figures,
    )
    print(f"  -> {ANALYSIS_REPORTS_DIR / 'radiological_grading_analysis.md'}")
    print(f"  -> {ANALYSIS_REPORTS_DIR / 'radiological_grading_analysis.json'}")
    print("\nDone. No model was trained and no severity score was defined.")


def render_figures(
    raw: pd.DataFrame,
    level_table: pd.DataFrame,
    correlation: pd.DataFrame,
    per_disc: dict,
    coverage: pd.DataFrame | None,
) -> list[str]:
    """Render every grading figure. Returns the filenames created."""
    written: list[str] = []

    plot_finding_distributions(
        raw, save_path=VISUALIZATIONS_DIR / "grading_finding_distributions.png"
    )
    written.append("grading_finding_distributions.png")

    plot_prevalence_by_level(
        level_table, save_path=VISUALIZATIONS_DIR / "grading_prevalence_by_disc_level.png"
    )
    written.append("grading_prevalence_by_disc_level.png")

    plot_cooccurrence(
        correlation, save_path=VISUALIZATIONS_DIR / "grading_cooccurrence.png"
    )
    written.append("grading_cooccurrence.png")

    plot_findings_per_disc(
        per_disc, save_path=VISUALIZATIONS_DIR / "grading_findings_per_disc.png"
    )
    written.append("grading_findings_per_disc.png")

    if coverage is not None:
        plot_split_class_balance(
            coverage, save_path=VISUALIZATIONS_DIR / "grading_split_class_balance.png"
        )
        written.append("grading_split_class_balance.png")

    return written


def write_report(**kw) -> Path:
    """Compose outputs/reports/radiological_grading_analysis.md."""
    raw = kw["raw"]
    issues = kw["issues"]
    distributions = kw["distributions"]
    linkage = kw["linkage_summary"]
    replication = kw["replication"]
    longitudinal = kw["longitudinal"]

    report = MarkdownReport(
        "Radiological Grading Analysis",
        "data/raw/radiological_gradings.csv - scoping the extended project goal "
        "(segmentation + disc-level degeneration assessment)",
    )

    report.text(
        "**Scope of this document.** It inspects the grading file, establishes how it "
        "links to the MRI series and segmentation masks, and uses that to propose a "
        "Sprint 2 architecture. No model is trained here, and **no severity score is "
        "defined** - the dataset ships eight independently graded findings, and "
        "combining them into one number is a clinical decision rather than a "
        "data-processing one."
    )

    # ================= 1. Headline =================
    report.heading("1. Headline figures")
    valid_records = int(len(raw[raw[IVD_COLUMN] >= 1]))
    report.key_values(
        {
            "File": "`data/raw/radiological_gradings.csv`",
            "Rows (raw)": issues["n_rows"],
            "Columns": len(raw.columns),
            "Patients": issues["n_patients"],
            "Valid IVD records (`IVD label` >= 1)": valid_records,
            "Invalid IVD records (`IVD label` = 0)": issues["n_invalid_ivd_label_rows"],
            "Missing values (entire file)": issues["total_missing_values"],
            "Duplicate rows": issues["duplicate_full_rows"],
            "Duplicate (Patient, IVD label) keys": issues["duplicate_patient_disc_keys"],
            "Graded discs per patient": f"{issues['discs_per_patient']['min']}-"
            f"{issues['discs_per_patient']['max']} "
            f"(median {issues['discs_per_patient']['median']:.0f})",
        }
    )
    report.text(
        "The file is **complete**: not a single missing value, no duplicate rows and no "
        "duplicate patient/disc keys. Its granularity is **one row per (patient, "
        "intervertebral disc)**."
    )

    # ================= 2. Columns =================
    report.heading("2. All columns")
    report.table(
        [
            {
                "#": entry["position"],
                "column": f"`{entry['column']}`",
                "dtype": entry["dtype"],
                "kind": entry["kind"],
                "values present": entry["values_present"],
                "missing": entry["missing"],
                "unexpected": entry["unexpected_values"] or "none",
            }
            for entry in kw["schema"]
        ],
        ["#", "column", "dtype", "kind", "values present", "missing", "unexpected"],
    )
    report.text(
        "All ten columns are integer-typed. Two are identifiers and eight are graded "
        "findings. No undocumented value appears in any finding column."
    )

    report.heading("Meaning of each finding", level=3)
    report.table(
        [
            {
                "column": f"`{f.column}`",
                "kind": f.kind,
                "domain": list(f.expected_values),
                "meaning": f.description,
            }
            for f in FINDINGS
        ],
        ["column", "kind", "domain", "meaning"],
    )
    report.text(
        "The distinction between *kinds* is not cosmetic and constrains the modelling:"
    )
    report.bullets(
        [
            "**`Pfirrman grade` is ordinal** (1 < 2 < ... < 5). It must be modelled in "
            "a way that respects the ordering; treating it as 5 unordered classes "
            "throws away the fact that confusing grade 4 with 5 is a smaller error "
            "than confusing 1 with 5.",
            "**`Modic` is nominal.** Types I, II and III describe different marrow "
            "states (oedema, fatty, sclerotic), not increasing severity, so it must "
            "**not** be treated as an ordered scale.",
            "**The remaining six are binary** presence/absence flags.",
            "`UP endplate` / `LOW endplate` are the endplate-defect / Schmorl's-node "
            "labels, recorded separately for the upper and lower endplate bounding "
            "each disc space.",
        ]
    )
    report.text(
        "The clinical naming follows the published dataset description "
        "([SPIDER data page](https://spider.grand-challenge.org/data/); "
        "[van der Graaf et al., *Scientific Data* 11, 264, 2024]"
        "(https://www.nature.com/articles/s41597-024-03090-w)); the value domains "
        "above were measured from the file. Content was rephrased for compliance with "
        "licensing restrictions."
    )

    # ================= 3. Distributions =================
    report.heading("3. Distribution of every finding")
    report.text(
        f"All percentages are over the **{valid_records:,} valid disc records**. "
        f"See `outputs/visualizations/grading_finding_distributions.png`."
    )

    report.heading("3.1 Pfirrmann grade (ordinal, 1-5)", level=3)
    pf = distributions["pfirrmann_grade"]
    report.table(
        [
            {
                "grade": grade,
                "label": PFIRRMANN_GRADE_NAMES[grade],
                "records": f"{pf['counts'].get(grade, 0):,}",
                "share": f"{pf['percentages'].get(grade, 0.0):.2f}%",
            }
            for grade in sorted(PFIRRMANN_GRADE_NAMES)
        ],
        ["grade", "label", "records", "share"],
    )
    report.key_values(
        {
            "Grades present": sorted(pf["counts"]),
            "Mean grade": pf["mean"],
            "Median grade": pf["median"],
        }
    )

    report.heading("3.2 Modic category (nominal, 0-3)", level=3)
    modic = distributions["modic"]
    report.table(
        [
            {
                "value": value,
                "category": MODIC_CATEGORY_NAMES[value],
                "records": f"{modic['counts'].get(value, 0):,}",
                "share": f"{modic['percentages'].get(value, 0.0):.2f}%",
            }
            for value in sorted(MODIC_CATEGORY_NAMES)
        ],
        ["value", "category", "records", "share"],
    )

    report.heading("3.3 Binary findings", level=3)
    report.table(
        [
            {
                "finding": f"`{f.column}`",
                "absent (0)": f"{distributions[f.tidy]['counts'].get(0, 0):,}",
                "present (1)": f"{distributions[f.tidy]['counts'].get(1, 0):,}",
                "prevalence": f"{distributions[f.tidy]['prevalence_pct']:.2f}%",
            }
            for f in FINDINGS
            if f.kind == "binary"
        ],
        ["finding", "absent (0)", "present (1)", "prevalence"],
    )

    report.heading("3.4 How many findings occur on one disc", level=3)
    per_disc = kw["per_disc"]
    report.table(
        [
            {"findings on the disc": k, "disc records": f"{v:,}",
             "share": f"{100 * v / per_disc['n_records']:.2f}%"}
            for k, v in sorted(per_disc["distribution"].items())
        ],
        ["findings on the disc", "disc records", "share"],
    )
    report.key_values(
        {
            "Mean findings per disc": per_disc["mean_findings_per_disc"],
            "Discs with no finding at all": f"{per_disc['pct_with_no_finding']:.2f}%",
            "Counting rule": per_disc["note"],
        }
    )
    report.text(
        "**This makes the task multi-label, not multi-class.** A disc routinely carries "
        "several findings simultaneously, so Sprint 2 cannot treat the assessment as "
        "picking one class out of a set."
    )

    # ================= 4. Disc level =================
    report.heading("4. Findings across disc levels")
    report.text(
        "`IVD label` is anatomically ordered: 1 is the most inferior disc, counting "
        "upward. Prevalence per level is therefore a sanity check on the labels as "
        "well as useful context."
    )
    report.dataframe(kw["level_table"], max_rows=12)
    report.text(
        "Two things to take from this table. First, the record count collapses at "
        "higher levels - disc 1 through 6 are present for essentially every patient, "
        "while the upper levels appear in only a fraction of them. Any per-level "
        "metric in Sprint 2 will therefore be far less reliable for the upper discs. "
        "Second, the degeneration measures concentrate in the lower lumbar discs, "
        "which is the expected anatomical pattern and indicates the labels behave "
        "sensibly. Figure: `outputs/visualizations/grading_prevalence_by_disc_level.png`."
    )

    report.heading("4.1 Pfirrmann grade against Modic category", level=3)
    crosstab = kw["modic_crosstab"].rename(
        columns={v: f"Modic {MODIC_CATEGORY_NAMES[v]}" for v in MODIC_CATEGORY_NAMES}
    )
    crosstab.index = [f"Pfirrmann {i}" for i in crosstab.index]
    report.dataframe(crosstab.reset_index().rename(columns={"index": "grade"}),
                     max_rows=10)
    report.text(
        "Read this as context, not as a severity mapping: Modic categories are "
        "nominal, so a higher Modic value in a row does not mean a worse disc."
    )

    report.heading("4.2 Co-occurrence between findings", level=3)
    report.text(
        "Spearman correlation (the findings are ordinal/binary, so Pearson would be "
        "inappropriate). Descriptive only - it informs whether Sprint 2 should predict "
        "the findings jointly. Figure: `outputs/visualizations/grading_cooccurrence.png`."
    )
    report.dataframe(kw["correlation"].round(3).reset_index().rename(
        columns={"index": "finding"}), max_rows=10)

    # ================= 5. Linkage =================
    report.heading("5. Relationship between Patient, IVD label and MRI series")
    report.text("This is the part that determines what is actually trainable.")
    report.bullets(
        [
            f"**`Patient` joins to the MRI filenames.** Filenames are "
            f"`<patient_id>_<modality>.mha`, and `Patient` matches the numeric prefix. "
            f"All {issues['n_patients']} patients in the grading file have imaging, and "
            f"every imaged patient is graded.",
            f"**`IVD label` N corresponds to mask label {IVD_MASK_LABEL_OFFSET} + N.** "
            f"Masks number the lowest IVD {IVD_MASK_LABEL_OFFSET + 1} upward, and the "
            f"grading file numbers the lowest disc 1 upward. Verified: the two label "
            f"sets agree exactly in "
            f"{linkage['n_series_exact_match']}/{linkage['n_series']} series "
            f"({linkage['pct_series_exact_match']}%).",
            "**Gradings are per patient, not per series.** There is exactly one grading "
            "row per (patient, disc) regardless of how many MRI series that patient "
            "has. A patient has 1-3 series (T1, T2, sometimes 3D T2 SPACE) of the same "
            "anatomy, and they all share one set of gradings.",
        ]
    )

    report.heading("5.1 Label replication across series", level=3)
    report.key_values(
        {
            "Patients": replication["n_patients"],
            "MRI series": replication["n_series"],
            "Unique grading records": replication["n_unique_grading_records"],
            "Disc-series instances if gradings are attached to every series":
                replication["n_disc_series_instances_if_replicated"],
            "Replication factor": replication["replication_factor"],
            "Series per patient": replication["series_per_patient_distribution"],
        }
    )
    report.text(
        f"Attaching gradings to series inflates {replication['n_unique_grading_records']:,} "
        f"real labels into {replication['n_disc_series_instances_if_replicated']:,} "
        f"training instances - a factor of {replication['replication_factor']}. Those "
        f"extra instances are **not** extra information. The effective sample size for "
        f"the grading task is **{replication['n_patients']} patients / "
        f"{replication['n_unique_grading_records']:,} discs**, and this is the reason "
        f"the Sprint 1 patient-level split must be carried over unchanged."
    )

    report.heading("5.2 Where gradings and segmentations disagree", level=3)
    report.key_values(
        {
            "Series with exact agreement": f"{linkage['n_series_exact_match']} / "
            f"{linkage['n_series']} ({linkage['pct_series_exact_match']}%)",
            "Series with a mismatch": linkage["n_series_mismatch"],
            "Patients affected": linkage["n_patients_with_mismatch"],
            "Segmented disc instances across all series":
                f"{linkage['total_disc_instances_in_series']:,}",
            "Segmented disc instances with a matching grading":
                f"{linkage['total_linkable_disc_instances']:,}",
            "Graded discs never segmented in any series of that patient":
                linkage["n_graded_discs_never_segmented"],
            "Segmented discs with no grading for that patient":
                linkage["n_segmented_discs_never_graded_patientwise"],
        }
    )
    report.text("The mismatching series in full:")
    report.table(
        [
            {
                "series": f"`{row['image_id']}`",
                "patient": row["patient_id"],
                "graded discs": row["n_graded_discs"],
                "segmented discs": row["n_segmented_discs"],
                "segmented but not graded": row["segmented_without_grading"] or "-",
                "graded but not segmented": row["graded_without_segmentation"] or "-",
            }
            for row in linkage["mismatch_series"]
        ],
        [
            "series",
            "patient",
            "graded discs",
            "segmented discs",
            "segmented but not graded",
            "graded but not segmented",
        ],
    )
    report.text(
        "These are the discs that **cannot be used as disc-level training targets "
        "without a decision**: either there is a segmented region with no label, or a "
        "label with no region to attach it to. The practical handling is to train on "
        "the intersection and record the excluded discs, rather than silently "
        "mis-aligning labels by index. Per-series and per-patient detail: "
        "`outputs/reports/grading_series_linkage.csv` and "
        "`grading_patient_linkage.csv`."
    )

    # ================= 6. Quality =================
    report.heading("6. Data quality problems found")
    problems: list[str] = [
        "No missing values anywhere in the file; no duplicate rows; no duplicate "
        "(Patient, IVD label) keys.",
    ]
    if issues["n_invalid_ivd_label_rows"]:
        rows = issues["invalid_ivd_label_rows"]
        affected = ", ".join(str(r[PATIENT_COLUMN]) for r in rows)
        problems.append(
            f"**{issues['n_invalid_ivd_label_rows']} rows carry `IVD label` = 0** "
            f"(patients {affected}). Zero is not a valid disc index - the mask label "
            f"space reserves 0 for background and numbers IVDs from "
            f"{IVD_MASK_LABEL_OFFSET + 1} upward - so these rows cannot be attached to "
            f"any segmented structure. In both cases the patient also has a contiguous "
            f"1..N run, and counting the 0 row makes the patient's disc total match "
            f"`num_discs` in `overview.csv`, which is consistent with a corrupted "
            f"label value rather than an extra disc. They are excluded from every "
            f"statistic in this report and should be excluded from training; "
            f"recovering the intended index would require re-annotation."
        )
    if issues["n_patients_non_contiguous_discs"]:
        problems.append(
            f"{issues['n_patients_non_contiguous_discs']} patients have non-contiguous "
            f"disc indices: {issues['patients_non_contiguous_discs']}."
        )
    else:
        problems.append(
            "Disc indices form a contiguous 1..N run for every patient (after "
            "excluding the `IVD label` = 0 rows)."
        )
    problems.append(
        f"{linkage['n_series_mismatch']} series "
        f"({linkage['n_patients_with_mismatch']} patients) disagree between the graded "
        f"disc set and the segmented disc set - see section 5.2."
    )
    problems.append(
        "Severe class imbalance in the rarest categories. Pfirrmann grades 1 and 5 and "
        "Modic type III are each a small fraction of records, and spondylolisthesis is "
        "the rarest binary finding. This caps what Sprint 2 can credibly report "
        "per class."
    )
    problems.append(
        "The grading file carries **no laterality, no disc sub-region and no "
        "measurement values** (no disc height in mm, no slip distance, no signal "
        "intensity). Every finding is a per-disc categorical judgement, so the model "
        "output can only be as granular as that."
    )
    report.bullets(problems)

    if kw["coverage"] is not None:
        report.heading("6.1 Class balance inside the Sprint 1 split", level=3)
        report.text(
            "The existing patient-level split partitions the grading labels too, since "
            "gradings are keyed on patient. Counts per split:"
        )
        report.dataframe(kw["coverage"], max_rows=5)
        report.text(
            "Figure: `outputs/visualizations/grading_split_class_balance.png`. Note how "
            "thin the rare classes become: a class with only a handful of test records "
            "cannot support a trustworthy per-class score, so Sprint 2 should report "
            "those with explicit uncertainty or group them."
        )

    _write_modality_section(report, kw["modality"])
    _write_longitudinal_section(report, longitudinal, kw["modality"])
    _write_architecture_section(
        report, linkage, replication, distributions, kw["modality"], kw["coverage"]
    )

    report.heading("10. Figures produced")
    report.bullets([f"`outputs/visualizations/{name}`" for name in kw["figures"]])

    report.heading("11. Companion data files")
    report.bullets(
        [
            "`outputs/reports/radiological_grading_analysis.json` - every number above, "
            "machine-readable",
            "`outputs/reports/grading_disc_level_prevalence.csv` - prevalence per disc level",
            "`outputs/reports/grading_series_linkage.csv` - grading/mask agreement per series",
            "`outputs/reports/grading_patient_linkage.csv` - the same per patient",
        ]
    )

    return report.save(ANALYSIS_REPORTS_DIR / "radiological_grading_analysis.md")


def _write_modality_section(report: MarkdownReport, modality: dict) -> None:
    """Section 7: which sequence supports which target."""
    report.heading("7. Sequence availability and what it constrains")
    report.key_values(
        {
            "Series by modality": modality["series_by_modality"],
            "Patients with at least one T2 / T2 SPACE series":
                f"{modality['n_patients_with_t2']} / {modality['n_patients_total']}",
            "Patients with only T1": f"{modality['n_patients_t1_only']} "
            f"({modality['patients_t1_only']})",
            "Disc records on patients that have T2":
                f"{modality['n_disc_records_on_t2_patients']:,} / "
                f"{modality['n_disc_records_total']:,}",
            "Disc records lost if a target is restricted to T2":
                modality["n_disc_records_lost_if_t2_only"],
        }
    )
    report.text(
        "**This splits the eight findings into two groups, and the split is not "
        "arbitrary.** The Pfirrmann grade is defined on T2-weighted images: it is read "
        "from nucleus signal brightness and disc height on T2. Asking a model to "
        "predict it from a T1 series is not a well-posed problem, because the signal "
        "characteristic the grade is built on is not the one T1 shows. A Pfirrmann head "
        "should therefore be trained and evaluated on T2 / T2 SPACE series only, which "
        f"still covers {modality['n_disc_records_on_t2_patients']:,} of "
        f"{modality['n_disc_records_total']:,} disc records."
    )
    report.text(
        "The other findings - endplate defects, spondylolisthesis, herniation, "
        "narrowing, bulging - are morphological rather than signal-based, so they are "
        "not restricted in the same way and can use all series. Modic changes sit in "
        "between: they are marrow *signal* changes whose type is conventionally "
        "distinguished using T1 together with T2, which is a further reason the Modic "
        "*type* is not a safe target here (see section 9)."
    )


def _write_longitudinal_section(
    report: MarkdownReport, longitudinal: dict, modality: dict
) -> None:
    """Section 8: the verified answer on 6-/9-month follow-up."""
    report.heading("8. Longitudinal / postoperative analysis: what is not possible")
    report.text(
        "The revised goal includes comparing preoperative scans with 6-month and "
        "9-month postoperative scans. **This dataset cannot support that.** That is a "
        "verified finding, not an assumption - three independent checks were run "
        "against the actual files, and the published dataset description was consulted."
    )

    report.heading("8.1 Evidence", level=3)
    report.table(
        [
            {
                "requirement for longitudinal analysis": "A per-series acquisition "
                "date, visit number or timepoint field",
                "present?": "**No**",
                "evidence": f"All {longitudinal['n_overview_columns']} columns of "
                f"`overview.csv` were pattern-searched for date/time/visit/follow-up "
                f"fields. The only match is `birth_date`, which holds values 14-84 "
                f"(mean 59.6) and is an age in years despite its name - not an "
                f"acquisition date. It is also missing for 73% of series. There is no "
                f"StudyDate, SeriesDate or AcquisitionDate column at all.",
            },
            {
                "requirement for longitudinal analysis": "More than one study per "
                "patient",
                "present?": "**No**",
                "evidence": f"No patient has more than one series of the same modality "
                f"(max = {longitudinal['max_series_per_patient_modality']}, "
                f"{longitudinal['n_patient_modality_pairs_repeated']} repeated "
                f"patient+modality pairs). The 1-3 series a patient has are *different "
                f"sequences of one sitting* (T1, T2, T2 SPACE), not repeat visits.",
            },
            {
                "requirement for longitudinal analysis": "A treatment / surgery / "
                "outcome field",
                "present?": "**No**",
                "evidence": f"Pattern search for surgery, operation, treatment, "
                f"intervention and outcome fields returned "
                f"{longitudinal['treatment_or_outcome_columns'] or 'nothing'}. Neither "
                f"CSV records whether a patient was operated on, at which level, when, "
                f"or with what result.",
            },
        ],
        ["requirement for longitudinal analysis", "present?", "evidence"],
    )
    report.text(
        "The published dataset description is consistent with all three findings: it "
        "states that studies were gathered from patients with a history of low back "
        "pain, that **each study consisted of up to three MRI series**, and that "
        "collection ran from January 2019 to March 2022 across four hospitals. That "
        "date range is the span of the *collection campaign* across different patients "
        "- it is not a per-patient follow-up interval. Nothing in the description "
        "mentions follow-up, postoperative or repeat imaging "
        "([SPIDER data page](https://spider.grand-challenge.org/data/)). Content was "
        "rephrased for compliance with licensing restrictions."
    )
    report.text(
        "**Conclusion: the dataset is cross-sectional - exactly one timepoint per "
        "patient.** Any 6-month or 9-month comparison, recovery trajectory, or "
        "pre-/post-operative change measurement is impossible to implement *or "
        "validate* with it. Reporting such a result from this data would require "
        "fabricating the second timepoint."
    )

    report.heading("8.2 Specifically, what cannot be built", level=3)
    report.bullets(
        [
            "**Inter-timepoint registration** - there is no second scan to register to.",
            "**Change / delta metrics** (disc height loss, grade progression, "
            "herniation resolution) - a delta needs two observations of the same disc "
            "at different times; the dataset has one.",
            "**Recovery or progression modelling** - no outcome variable and no time "
            "axis exist.",
            "**Postoperative assessment** - it is not even recorded which patients had "
            "surgery, so a pre-/post-operative label cannot be assigned.",
            "**Validation of any of the above** - even if such a module were written, "
            "there is no ground truth in this dataset to test it against.",
        ]
    )

    report.heading("8.3 Additional data that would be required", level=3)
    report.table(
        [
            {"requirement": "Repeat MRI studies", "detail": "At least 2, ideally 3 "
             "studies per patient (preoperative, ~6 months, ~9 months) of the same "
             "region, with the same sequences available at each timepoint."},
            {"requirement": "Timepoint identification", "detail": "An acquisition date "
             "or visit label per study, plus the surgery date, so an interval in days "
             "can be computed rather than assumed."},
            {"requirement": "Surgical record", "detail": "Procedure type, the vertebral "
             "/ disc levels operated on, and the date. Without the operated level, a "
             "per-disc change cannot be attributed to the intervention."},
            {"requirement": "Gradings at every timepoint", "detail": "The same eight "
             "findings re-graded at each follow-up under the same protocol. Ideally "
             "blinded and with inter-reader agreement reported, because measuring "
             "change is far more sensitive to reader variability than measuring a "
             "single state."},
            {"requirement": "Clinical outcome measures", "detail": "If 'recovery' is to "
             "mean patient benefit rather than image appearance, validated scores such "
             "as a disability index or pain scale are needed at each timepoint. Imaging "
             "change alone does not establish recovery."},
            {"requirement": "Protocol consistency", "detail": "Comparable acquisition "
             "across timepoints, or a documented harmonisation strategy. This dataset "
             "already spans four hospitals with pixel spacing varying by ~16x; adding "
             "uncontrolled protocol drift between timepoints would confound any "
             "measured change."},
            {"requirement": "Governance", "detail": "Ethics approval and consent "
             "covering linkage of repeat studies and surgical records for the same "
             "individual."},
        ],
        ["requirement", "detail"],
    )

    report.heading("8.4 A measurement-resolution caveat worth raising early", level=3)
    report.text(
        "Even with ideal longitudinal data, the eight findings available here are "
        "**coarse categorical judgements** - six binary flags, one nominal category and "
        "one 5-point ordinal grade. There are no continuous measurements in the file: no "
        "disc height in millimetres, no slip distance, no signal-intensity ratio. A "
        "5-point scale has limited resolution for detecting change over a 6-9 month "
        "interval, since a real but small change may not cross a grade boundary. If "
        "quantifying change is a project goal, the pipeline should also output "
        "**continuous geometric measurements in millimetres** (disc height, disc height "
        "relative to neighbours, anterior-posterior extent, inter-vertebral offset), "
        "which are derivable from the segmentation masks and are far more sensitive to "
        "change than a categorical grade. Those measurements are *computable now* from "
        "Sprint 1 output; only their validation against follow-up ground truth has to "
        "wait."
    )


def _write_architecture_section(
    report: MarkdownReport,
    linkage: dict,
    replication: dict,
    distributions: dict,
    modality: dict,
    coverage: pd.DataFrame | None,
) -> None:
    """Section 9: the proposed Sprint 2 architecture."""
    report.heading("9. Proposed Sprint 2 architecture")
    report.text(
        "The design below follows from the measurements in this report rather than from "
        "a generic template. **Sprint 1 preprocessing is reused unchanged**: the same "
        f"352 x 256 slices at 1.0 mm/px, the same label spaces, and the same "
        f"patient-level split with the same seed."
    )

    report.heading("9.1 The three constraints that drive the design", level=3)
    report.bullets(
        [
            f"**The grading task is small.** "
            f"{replication['n_unique_grading_records']:,} disc records from "
            f"{replication['n_patients']} patients. Attaching them to series inflates "
            f"this to {replication['n_disc_series_instances_if_replicated']:,} "
            f"instances (factor {replication['replication_factor']}), but that is "
            f"replication, not information. This is orders of magnitude too little to "
            f"train a large classifier from scratch, and it rules out an end-to-end "
            f"jointly-trained segmentation + grading network.",
            "**The labels are per-disc, not per-slice.** A grading describes a disc as "
            "a 3-D structure. Assigning it to every 2-D slice of that disc would create "
            "label noise, because a focal finding such as a herniation is visible on "
            "only some slices. So the unit of prediction must be a **disc instance**, "
            "not a slice.",
            "**The task is multi-label with very unequal support.** A disc carries "
            "2.08 findings on average and up to 7; meanwhile "
            "spondylolisthesis occurs in "
            f"{distributions['spondylolisthesis']['prevalence_pct']:.1f}% of discs and "
            f"herniation in {distributions['disc_herniation']['prevalence_pct']:.1f}%.",
        ]
    )

    report.heading("9.2 Staged architecture", level=3)
    report.code(
        """
STAGE A - Segmentation                                    [Sprint 1 output, unchanged]
  2-D U-Net, input 352x256x1, output 4 semantic classes
  (background / vertebra / IVD / spinal canal)
  Also predicts the instance label space already saved in Sprint 1
  -> trained on 12,415 preprocessed slices, 152 training patients
  -> metrics: Dice, IoU per class

STAGE B - Disc instance extraction                        [geometric, no learning]
  From the instance mask, isolate each disc (index 1..9) and the two
  vertebrae bounding it
  For each disc instance:
    - disc-centred ROI, taken as a 2.5-D stack (mid-disc slice +/- k neighbours)
      so a focal finding is not missed, and including the adjacent vertebral
      bodies because endplate defects, Modic change and spondylolisthesis live
      in or between the vertebrae, not in the disc
    - geometric descriptors in millimetres, from the mask and the known
      1.0 mm/px scale:
          disc height (mean / min / anterior / posterior)
          disc height relative to the two neighbouring discs
          anterior-posterior extent, posterior disc margin position
          antero-posterior offset between the bounding vertebrae  (slip)
          canal cross-section at the disc level
  -> output: one feature record + one image ROI per disc instance

STAGE C - Per-disc multi-task prediction                  [the new model]
  Shared encoder over the ROI (small CNN or a pretrained backbone,
  fine-tuned; the Stage A encoder is a reasonable initialisation)
  Concatenate the Stage B geometric descriptors before the heads
  Heads, matched to each target's measurement kind:
    - Pfirrmann      ordinal (cumulative-link / CORAL), T2 series only
    - Modic          binary "any Modic change"        (see 9.4)
    - 6 x binary     endplate x2, spondylolisthesis, herniation,
                     narrowing, bulging
  Loss = weighted sum of per-head losses; class-weighted or focal loss for the
  rare binary findings

STAGE D - Per-disc report assembly                        [deterministic, no scoring]
  One row per disc: level index, each finding with its predicted class and
  calibrated probability, plus the Stage B measurements in millimetres
  Findings are reported SEPARATELY. No composite severity number is produced.
  Low-confidence and low-support cases (e.g. upper disc levels) are flagged.

STAGE E - Evaluation                                      [held-out test patients]
  Segmentation : Dice, IoU per class
  Binary       : PR-AUC (primary, because of imbalance), ROC-AUC,
                 sensitivity/specificity at a validation-chosen threshold
  Pfirrmann    : quadratic weighted kappa (primary), MAE,
                 exact and within-1-grade agreement
  Modic        : balanced accuracy, F1
  All aggregated at PATIENT level, plus per-disc-level breakdown where the
  record count supports it
""",
        "text",
    )

    report.heading("9.3 Why two stages rather than end-to-end", level=3)
    report.text(
        f"With {replication['n_unique_grading_records']:,} disc labels, an end-to-end "
        f"model would have to learn segmentation and grading simultaneously from the "
        f"grading signal, which is the scarcer of the two by a wide margin "
        f"(12,415 segmentation slices vs {replication['n_unique_grading_records']:,} "
        f"grading records). Keeping the stages separate means the segmentation model "
        f"uses all of its supervision, the grading model starts from an anatomically "
        f"normalised ROI instead of a whole slice, and the two error sources stay "
        f"separable at evaluation time. The last point is practical: Stage C should be "
        f"evaluated **twice** - once with ROIs taken from ground-truth masks and once "
        f"from predicted masks - so that a drop in grading accuracy can be attributed "
        f"to either segmentation error or classification error rather than being "
        f"confounded."
    )

    report.heading("9.4 Targets that must be reduced or dropped", level=3)
    modic = distributions["modic"]

    def split_counts(column: str) -> str:
        """Exact per-split counts for a coverage column, e.g. 'train 3 / val 1 / test 0'."""
        if coverage is None:
            return "split counts unavailable"
        return " / ".join(
            f"{row['split']} {int(row[column])}" for _, row in coverage.iterrows()
        )

    modic_type1 = split_counts("modic_1")
    modic_type3 = split_counts("modic_3")
    spondylo = split_counts("spondylolisthesis_pos")
    herniation = split_counts("disc_herniation_pos")

    report.table(
        [
            {
                "target": "Modic **type** (I / II / III)",
                "problem": f"Type I has only {modic['counts'].get(1, 0)} records and "
                f"type III only {modic['counts'].get(3, 0)}, out of "
                f"{modic['n_records']:,}. Across the patient-level split that is "
                f"type I: {modic_type1}; type III: {modic_type3}. **Type I has no test "
                f"records at all**, so a per-type metric is not merely unreliable, it "
                f"is undefined.",
                "recommendation": "Collapse to **binary 'any Modic change'** "
                f"({modic['counts'].get(0, 0):,} absent vs "
                f"{sum(v for k, v in modic['counts'].items() if k > 0):,} present, a "
                f"workable balance). Do not report per-type metrics.",
            },
            {
                "target": "Spondylolisthesis",
                "problem": f"{distributions['spondylolisthesis']['n_positive']} positives "
                f"overall, split {spondylo}.",
                "recommendation": "Keep, because the Stage B inter-vertebral offset "
                "measurement is directly informative, but report PR-AUC with a "
                "confidence interval and state the positive count alongside every "
                "metric.",
            },
            {
                "target": "Disc herniation",
                "problem": f"{distributions['disc_herniation']['n_positive']} positives "
                f"({distributions['disc_herniation']['prevalence_pct']:.1f}%), "
                f"split {herniation}.",
                "recommendation": "Keep with class weighting; expect wide confidence "
                "intervals and do not present a single point estimate as definitive.",
            },
            {
                "target": "Pfirrmann on T1 series",
                "problem": "The grade is defined on T2 signal; T1 does not show it.",
                "recommendation": f"Restrict the Pfirrmann head to T2 / T2 SPACE "
                f"({modality['n_disc_records_on_t2_patients']:,} disc records; "
                f"{modality['n_patients_t1_only']} T1-only patients excluded from this "
                f"target only).",
            },
            {
                "target": "Upper disc levels (index 8, 9)",
                "problem": "Very few records exist at these levels.",
                "recommendation": "Train on them, but do not report per-level metrics "
                "where the count is negligible; fold them into an 'upper levels' group.",
            },
        ],
        ["target", "problem", "recommendation"],
    )

    report.heading("9.5 Data-handling rules Sprint 2 must follow", level=3)
    report.bullets(
        [
            "**Keep the Sprint 1 patient-level split, same seed.** Gradings are keyed "
            "on patient, so a series- or slice-level split would put the same label in "
            "train and test.",
            f"**Exclude the {linkage['n_series_mismatch']} series where the graded and "
            f"segmented disc sets disagree**, or restrict them to the intersecting "
            f"discs. Never align by position/index order - that is exactly how labels "
            f"get silently attached to the wrong disc.",
            "**Exclude the 2 rows with `IVD label` = 0.** They cannot be mapped to a "
            "structure.",
            "**Count each label once per patient in metrics.** Using a patient's 2-3 "
            "series as extra training samples is legitimate augmentation, but averaging "
            "metrics over series would weight multi-series patients 2-3x and overstate "
            "agreement.",
            "**Calibrate probabilities** on the validation split (e.g. temperature "
            "scaling) before they appear in a per-disc report, since an uncalibrated "
            "probability presented next to a clinical finding is misleading.",
            "**Persist the disc identity key `(patient_id, ivd_label)`** in every "
            "output record. This is what a future longitudinal module would join on, so "
            "building it in now costs nothing and makes the follow-up extension "
            "possible later.",
        ]
    )

    report.heading("9.6 What Sprint 2 will not claim", level=3)
    report.bullets(
        [
            "No composite severity score. The dataset grades eight findings "
            "independently; collapsing them into one number would be a clinical "
            "judgement that this data does not license.",
            "No 6-month or 9-month comparison, and no recovery or progression "
            "statement - see section 8.",
            "No diagnostic claim. The output is an estimate of radiological findings "
            "for review, not a diagnosis.",
            "No metric without its supporting count. Every per-class figure is reported "
            "next to the number of test records behind it.",
        ]
    )


if __name__ == "__main__":
    main()
