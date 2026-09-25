"""Render the downloadable report.

Deterministic Markdown assembled from the structured result. Every sentence is a
template filled with measured values, so the document cannot assert anything the
result does not contain.
"""

from __future__ import annotations

from datetime import datetime


def _grade(value) -> str:
    return "not reported" if value is None else str(value)


def _flag(value) -> str:
    if value is None:
        return "not reported"
    return "yes" if value else "no"


def render_markdown(result: dict) -> str:
    study = result.get("study", {})
    processing = result.get("processing", {})
    segmentation = result.get("segmentation", {})
    summary = result.get("summary", {})
    metrics = result.get("validated_metrics", {})
    discs = result.get("discs", [])

    lines: list[str] = []
    add = lines.append

    pipeline = result.get("pipeline") or {}

    add("# Lumbar Spine MRI Analysis Report")
    add("")
    add("**AI-assisted research analysis. Not a clinical diagnosis.**")
    add("")
    add("| | |")
    add("| --- | --- |")
    add(f"| Study | {study.get('filename')} |")
    add(f"| Analysis id | `{result.get('analysis_id')}` |")
    add(f"| Pipeline version | **{pipeline.get('version', 'unknown')}** |")
    add(f"| Segmentation | {pipeline.get('segmentation', '-')} |")
    add(f"| Post-processing | {pipeline.get('postprocessing', '-')} |")
    add(f"| Processed | {result.get('completed_at') or 'unknown'} |")
    if result.get("is_sample"):
        add("| Source | Sample research study (bundled demonstration volume) |")
    add("")

    # 1
    add("## 1. Study Information")
    add("")
    add("| Field | Value |")
    add("| --- | --- |")
    add(f"| Filename | {study.get('filename')} |")
    add(f"| Format | {study.get('format')} |")
    modality = study.get("modality") or "not determinable"
    add(f"| Modality | {modality} ({study.get('modality_source')}) |")
    add(f"| Sagittal slices | {study.get('slice_count')} |")
    dims = study.get("dimensions") or []
    add(f"| Native in-plane size | {' x '.join(str(d) for d in dims)} px |")
    spacing = study.get("in_plane_spacing_mm") or []
    add(f"| Native in-plane spacing | "
        f"{' x '.join(f'{s:.4f}' for s in spacing)} mm |")
    add(f"| Slice spacing | {study.get('slice_spacing_mm')} mm |")
    add(f"| Native orientation | {study.get('native_orientation')} |")
    timepoint = result.get("timepoint") or {}
    add(f"| Timepoint | {timepoint.get('label', 'Baseline')} (single timepoint) |")
    add("")
    add(f"Pipeline **{pipeline.get('version', 'unknown')}** — "
        f"preprocessing {pipeline.get('preprocessing', '-')}, segmentation "
        f"{pipeline.get('segmentation', '-')}, post-processing "
        f"{pipeline.get('postprocessing', '-')}.")
    for name, reason in (pipeline.get("excluded") or {}).items():
        add(f"{name} is deliberately not served: {reason}")
    add("")

    # 2
    add("## 2. Processing Information")
    add("")
    add("| Stage | Detail |")
    add("| --- | --- |")
    add(f"| Preprocessing | {processing.get('preprocessing')} |")
    add(f"| Segmentation model | {processing.get('segmentation_model')} |")
    add(f"| Model provenance | {processing.get('segmentation_source')} |")
    add(f"| Post-processing | {processing.get('postprocessing_method')} |")
    add(f"| Compute device | {processing.get('device')} |")
    add(f"| Total duration | {processing.get('duration_seconds')} s |")
    add(f"| Disc tracks found | {processing.get('disc_tracks_found')} |")
    add("")
    durations = processing.get("stage_durations") or {}
    if durations:
        add("Stage timings (seconds):")
        add("")
        for name, value in durations.items():
            add(f"- {name.replace('_', ' ')}: {value}")
        add("")

    # 3
    add("## 3. Segmentation Overview")
    add("")
    add("| Class | Validated test Dice |")
    add("| --- | --- |")
    for entry in segmentation.get("classes", []):
        dice = entry.get("validated_test_dice")
        add(f"| {entry.get('display_name')} | "
            f"{'-' if dice is None else f'{dice:.5f}'} |")
    add("")
    add(f"Slices segmented: {segmentation.get('slice_count')}. "
        f"Slices carrying enough predicted anatomy to measure on: "
        f"{segmentation.get('informative_slice_count')}.")
    add("")
    add(f"_{segmentation.get('note')}_")
    add("")

    # 4
    add("## 4. Disc-Level Analysis")
    add("")
    if not discs:
        add("No intervertebral disc was identified in this study.")
        add("")
    else:
        add("Disc indices are integer positions counting upward from the most "
            "inferior disc. No anatomical level name is asserted, because the "
            "dataset does not state which vertebra is L5.")
        add("")
        add("| Disc | Pfirrmann | Modic (any) | Bulging | Narrowing | "
            "Herniation | Upper endplate | Lower endplate | Identity confidence |")
        add("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for disc in discs:
            confidence = disc.get("identity_confidence")
            add(
                f"| {disc['index']} | {_grade(disc.get('pfirrmann_grade'))} | "
                f"{_flag(disc.get('modic_any'))} | {_flag(disc.get('bulging'))} | "
                f"{_flag(disc.get('narrowing'))} | {_flag(disc.get('herniation'))} | "
                f"{_flag(disc.get('upper_endplate'))} | "
                f"{_flag(disc.get('lower_endplate'))} | "
                f"{'-' if confidence is None else f'{confidence:.2f}'} |"
            )
        add("")

    # 5
    add("## 5. Quantitative Measurements")
    add("")
    if discs:
        add("All values are derived from the predicted segmentation at "
            "1.0 mm/pixel.")
        add("")
        add("| Disc | Central height (mm) | Area (mm2) | AP extent (mm) | "
            "Disc/vertebra signal ratio | Canal width (mm) | Slices |")
        add("| --- | --- | --- | --- | --- | --- | --- |")
        for disc in discs:
            m = disc.get("measurements", {})

            def fmt(key: str, places: int = 3) -> str:
                value = m.get(key)
                return "-" if value is None else f"{value:.{places}f}"

            add(
                f"| {disc['index']} | {fmt('height_mm_central')} | "
                f"{fmt('area_mm2', 1)} | {fmt('ap_extent_mm')} | "
                f"{fmt('intensity_disc_vertebra_ratio', 4)} | "
                f"{fmt('canal_width_at_disc_mm')} | "
                f"{disc.get('slices_present', '-')} |"
            )
        add("")
        mean_height = summary.get("mean_disc_height_mm")
        if mean_height is not None:
            add(f"Mean central disc height across identified discs: "
                f"{mean_height:.3f} mm.")
            add("")
    else:
        add("No measurements are available because no disc was identified.")
        add("")

    # 6
    add("## 6. Model-Derived Findings")
    add("")
    add("Each statement below is generated by rule from the structured result. "
        "No language model is involved.")
    add("")
    for finding in summary.get("findings", []):
        add(f"- **{finding.get('category')}** - {finding.get('text')}")
    add("")
    unsupported = summary.get("unsupported_targets") or {}
    if unsupported:
        add("### Targets deliberately not reported")
        add("")
        for name, reason in unsupported.items():
            add(f"- `{name}`: {reason}")
        add("")

    # 7
    add("## 7. Limitations")
    add("")
    add("- Validated performance is measured on a 33-patient held-out test "
        "split, not on this study. Per-study accuracy is unknown.")
    if metrics:
        add(f"- Segmentation macro foreground Dice "
            f"{metrics.get('segmentation_macro_foreground_dice')}; disc indexing "
            f"{metrics.get('disc_indexing_percent')}%; end-to-end Pfirrmann "
            f"quadratic weighted kappa {metrics.get('pfirrmann_qwk_end_to_end')}.")
        add(f"- Disc height mean absolute error "
            f"{metrics.get('disc_height_mae_mm')} mm and area mean absolute error "
            f"{metrics.get('disc_area_mae_mm2')} mm2 against ground-truth masks.")
    add("- Disc identity is derived from ordered geometry, not predicted "
        "directly. Identity confidence describes how consistently a disc was "
        "tracked across slices; it is not a clinical confidence.")
    add("- The nominal Modic type and spondylolisthesis are not reported: the "
        "former was never modelled and the latter had too few positive cases to "
        "validate.")
    add("- Pfirrmann grading applies to T2 and T2-SPACE series only, because the "
        "grade is defined on T2 signal.")
    add("- This is a single timepoint. No change over time is measured, and no "
        "postoperative or longitudinal assessment is provided, because the data "
        "this system was validated on contains no longitudinal follow-up.")
    add("- A segmentation or indexing metric measures agreement with one "
        "annotation protocol. It does not establish clinical effectiveness.")
    add("")

    # 8
    add("## 8. Research Disclaimer")
    add("")
    add("Generated by an AI-assisted research pipeline.")
    add("")
    add("**Results require review by a qualified radiologist.** This is a "
        "research and educational prototype. It does not provide a clinical "
        "diagnosis, it is not a medical device, and it must not be used to make "
        "care decisions.")
    add("")
    return "\n".join(lines) + "\n"
