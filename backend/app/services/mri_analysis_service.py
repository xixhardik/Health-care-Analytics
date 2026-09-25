"""The analysis service: one uploaded volume in, one structured result out.

All pipeline orchestration lives here. Route handlers call this and do nothing
else, so the HTTP layer stays thin and the pipeline stays testable without a
web server.

The summary text is generated from deterministic templates driven by the
structured result. No language model is involved anywhere, so the prose can never
say something the numbers do not support.
"""

from __future__ import annotations

import logging
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import Settings  # noqa: E402
from backend.app.pipeline_info import (  # noqa: E402
    DISCLAIMER, EVIDENCE_LABELS, PIPELINE, PROVENANCE, SEGMENTATION_DICE,
    VALIDATED_METRICS,
)
from backend.app.services.model_service import CLASS_TABLE, ModelService  # noqa: E402
from backend.app.services.storage import AnalysisStore, utcnow  # noqa: E402
from ml.pipeline import (  # noqa: E402
    PipelineConfig, estimate_findings, index_discs, measure_discs,
    preprocess_volume, segment_slices,
)

logger = logging.getLogger("lumbar.api")

ProgressFn = Callable[[int, str], None]

STAGES = [
    (0, "Upload validated"),
    (10, "Loading MRI"),
    (20, "Preprocessing"),
    (40, "Segmentation"),
    (60, "Disc indexing"),
    (75, "Feature extraction"),
    (90, "Radiological analysis"),
    (100, "Complete"),
]

#: Finding name -> (display label, severity when positive).
FINDING_SEVERITY = {
    "narrowing": ("Disc narrowing", "moderate"),
    "bulging": ("Disc bulging", "low"),
    "any_modic": ("Modic-type endplate change", "moderate"),
    "upper_endplate": ("Upper endplate change", "low"),
    "lower_endplate": ("Lower endplate change", "low"),
    "herniation": ("Disc herniation", "high"),
}


def _clean(value) -> float | None:
    """Convert a measurement to a JSON-safe float, or None when undefined."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return round(number, 4)


@dataclass
class AnalysisOutcome:
    result: dict
    images: np.ndarray
    semantic: np.ndarray
    instance: np.ndarray
    slice_ids: list[str]


class MriAnalysisService:
    """Runs the frozen Sprint 1 / 3 / 5 pipeline for one study."""

    def __init__(
        self, settings: Settings, models: ModelService, store: AnalysisStore
    ) -> None:
        self.settings = settings
        self.models = models
        self.store = store

    @property
    def pipeline_config(self) -> PipelineConfig:
        return PipelineConfig(
            project_root=self.settings.project_root,
            checkpoint=self.settings.checkpoint,
            finding_model_dir=self.settings.finding_model_dir,
            sprint5_params_file=self.settings.sprint5_params_file,
            batch_size=self.settings.batch_size,
        )

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def run(
        self,
        analysis_id: str,
        upload_path: Path,
        study: dict,
        *,
        progress: ProgressFn | None = None,
    ) -> AnalysisOutcome:
        """Execute every stage. Progress is reported from real stage boundaries."""
        def report(value: int, stage: str) -> None:
            if progress:
                progress(value, stage)

        started = time.perf_counter()
        durations: dict[str, float] = {}
        loaded = self.models.require_segmentation()
        config = self.pipeline_config

        report(10, "Loading MRI")
        mark = time.perf_counter()

        # --- Sprint 1 preprocessing ---
        report(20, "Preprocessing")
        preprocessed = preprocess_volume(
            upload_path, series_id=f"study_{analysis_id}", progress=progress
        )
        durations["preprocessing"] = round(time.perf_counter() - mark, 2)

        # --- Sprint 3 segmentation ---
        mark = time.perf_counter()
        report(40, "Segmentation")
        semantic = segment_slices(
            loaded.model,
            preprocessed.images,
            batch_size=self.settings.batch_size,
            device=loaded.device,
            progress=progress,
        )
        durations["segmentation"] = round(time.perf_counter() - mark, 2)

        # --- Sprint 5 post-processing ---
        mark = time.perf_counter()
        report(60, "Disc indexing")
        indexed = index_discs(
            semantic, preprocessed.slice_ids, config=config, progress=progress
        )
        durations["disc_indexing"] = round(time.perf_counter() - mark, 2)

        # --- measurements ---
        mark = time.perf_counter()
        report(75, "Feature extraction")
        per_slice, aggregated = measure_discs(
            preprocessed.images,
            indexed.instance_maps,
            preprocessed.slice_ids,
            preprocessed.slice_indices,
            series_id=f"study_{analysis_id}",
            config=config,
            progress=progress,
        )
        durations["feature_extraction"] = round(time.perf_counter() - mark, 2)

        # --- findings ---
        mark = time.perf_counter()
        report(90, "Radiological analysis")
        findings = estimate_findings(
            aggregated, self.models.findings, modality=study.get("modality")
        )
        durations["radiological_analysis"] = round(time.perf_counter() - mark, 2)

        total = round(time.perf_counter() - started, 2)
        result = self._assemble(
            analysis_id=analysis_id,
            study=study,
            preprocessed=preprocessed,
            semantic=semantic,
            indexed=indexed,
            aggregated=aggregated,
            findings=findings,
            device=loaded.device,
            architecture=loaded.architecture,
            durations=durations,
            total_seconds=total,
        )
        report(100, "Complete")
        return AnalysisOutcome(
            result=result,
            images=preprocessed.images,
            semantic=semantic,
            instance=indexed.instance_maps,
            slice_ids=preprocessed.slice_ids,
        )

    # ------------------------------------------------------------------
    # Result assembly
    # ------------------------------------------------------------------

    def _assemble(
        self, *, analysis_id: str, study: dict, preprocessed, semantic,
        indexed, aggregated, findings: dict, device: str, architecture: str,
        durations: dict, total_seconds: float,
    ) -> dict:
        informative = int(sum(1 for v in indexed.informative.values() if v))
        foreground_per_slice = (semantic > 0).sum(axis=(1, 2))
        measurable = int((foreground_per_slice >= 50).sum())

        discs = self._build_discs(aggregated, indexed, findings)
        summary = self._build_summary(discs)

        unsupported = (self.models.findings.manifest or {}).get("unsupported", {})

        return {
            "analysis_id": analysis_id,
            "status": "completed",
            "created_at": study.get("created_at") or utcnow().isoformat(),
            "completed_at": utcnow().isoformat(),
            "is_sample": bool(study.get("is_sample")),
            "pipeline": PIPELINE,
            "study": {
                "filename": study["filename"],
                "size_bytes": study["size_bytes"],
                "format": study["format"],
                "modality": study.get("modality"),
                "modality_source": study.get("modality_source", "not_determinable"),
                "slice_count": study["slice_count"],
                "dimensions": study["dimensions"],
                "in_plane_spacing_mm": study["in_plane_spacing_mm"],
                "slice_spacing_mm": study["slice_spacing_mm"],
                "native_orientation": study["native_orientation"],
            },
            "timepoint": {
                "id": "baseline", "label": "Baseline",
                "acquired_at": None, "is_baseline": True,
            },
            "segmentation": {
                "classes": [
                    {"id": cid, "name": name, "display_name": display,
                     "validated_test_dice": dice}
                    for cid, name, display, dice in CLASS_TABLE
                ],
                "slice_count": int(len(preprocessed.slice_ids)),
                "informative_slice_count": max(informative, measurable),
                "macro_foreground_dice_validated":
                    SEGMENTATION_DICE["macro_foreground"],
                "note": (
                    "Every sagittal slice of the upload was segmented. Slices "
                    "carrying too little predicted anatomy are excluded from "
                    "measurement but are still viewable."
                ),
            },
            "processing": {
                "preprocessing": "Sprint 1: resample to 1.0 mm/px, centre "
                                 "crop/pad to 352x256, per-volume percentile "
                                 "normalisation, median denoise, CLAHE",
                "segmentation_model": architecture,
                "segmentation_source": "Sprint 3 Coverage (best validated model)",
                "postprocessing_method": f"Sprint 5 {indexed.method}",
                "postprocessing_params": indexed.params,
                "device": device,
                "duration_seconds": total_seconds,
                "stage_durations": durations,
                "disc_tracks_found": indexed.n_tracks,
                "components_rejected": indexed.n_components_rejected,
                "components_unassigned": indexed.n_components_unassigned,
            },
            "discs": discs,
            "summary": {**summary, "unsupported_targets": unsupported},
            "validated_metrics": VALIDATED_METRICS,
            "provenance_labels": PROVENANCE,
            "evidence_labels": EVIDENCE_LABELS,
            "disclaimer": DISCLAIMER,
        }

    def _build_discs(self, aggregated, indexed, findings: dict) -> list[dict]:
        if aggregated is None or len(aggregated) == 0:
            return []

        # Series-level track support doubles as an identity confidence: it is the
        # fraction of slices that voted for this disc's track. Named carefully -
        # it describes identity stability, not clinical certainty.
        support = {
            int(track["index"]): round(float(track.get("support_frac", 0.0)), 4)
            for track in indexed.tracks
        }

        discs: list[dict] = []
        for row in aggregated.itertuples():
            index = int(row.ivd_label)
            entry = findings.get(index, {})
            finding_list = []
            for name, payload in sorted(entry.items()):
                finding_list.append({
                    "name": name,
                    "label": payload.get("label")
                             or FINDING_SEVERITY.get(name, (name, "info"))[0],
                    "value": payload.get("value"),
                    "probability": payload.get("probability"),
                    "expected": payload.get("expected"),
                    "probabilities": payload.get("probabilities"),
                    "strength": payload.get("strength"),
                    "validated_metric": payload.get("validated_metric"),
                    "validated_value": payload.get("validated_value"),
                    "prevalence_baseline": payload.get("prevalence_baseline"),
                    "within_one_grade": payload.get("within_one_grade"),
                    "source": payload.get("source", "model_prediction"),
                    "caveat": payload.get("caveat"),
                    "unavailable_reason": payload.get("unavailable_reason"),
                })

            def value_of(name: str):
                item = entry.get(name) or {}
                return item.get("value")

            discs.append({
                "index": index,
                "structured": self._structured_lines(row, entry),
                "identity_confidence": support.get(index),
                "representative_slice_id": getattr(row, "slice_id", None),
                "representative_slice_index": int(getattr(row, "slice_index", 0))
                if getattr(row, "slice_index", None) is not None else None,
                "slices_present": int(getattr(row, "n_slices_present", 0) or 0),
                "measurements": {
                    "height_mm_central": _clean(getattr(row, "height_mm_central", None)),
                    "height_mm_anterior": _clean(getattr(row, "height_mm_anterior", None)),
                    "height_mm_posterior": _clean(getattr(row, "height_mm_posterior", None)),
                    "area_mm2": _clean(getattr(row, "area_mm2", None)),
                    "ap_extent_mm": _clean(getattr(row, "ap_extent_mm", None)),
                    "intensity_disc_vertebra_ratio":
                        _clean(getattr(row, "intensity_disc_vertebra_ratio", None)),
                    "intensity_nucleus_annulus_ratio":
                        _clean(getattr(row, "intensity_nucleus_annulus_ratio", None)),
                    "canal_width_at_disc_mm":
                        _clean(getattr(row, "canal_width_at_disc_mm", None)),
                    "disc_to_vertebra_height_ratio":
                        _clean(getattr(row, "disc_to_vertebra_height_ratio", None)),
                    "height_ratio_to_series_median":
                        _clean(getattr(row, "height_ratio_to_series_median", None)),
                    "source": "segmentation_derived_measurement",
                },
                "findings": finding_list,
                "pfirrmann_grade": value_of("pfirrmann_grade"),
                "modic_type": None,
                "modic_any": value_of("any_modic"),
                "bulging": value_of("bulging"),
                "narrowing": value_of("narrowing"),
                "herniation": value_of("herniation"),
                "spondylolisthesis": None,
                "upper_endplate": value_of("upper_endplate"),
                "lower_endplate": value_of("lower_endplate"),
            })
        discs.sort(key=lambda d: d["index"])
        return discs

    def _structured_lines(self, row, entry: dict) -> list[dict]:
        """Label/value lines for one disc, each tagged with its provenance.

        Deliberately flat and deterministic: a label, a rendered value, and how
        that value was obtained. A missing value reads "unavailable" - never 0,
        "No" or "Normal", because absence is not a negative finding.
        """
        def measurement(label: str, attribute: str, unit: str, places: int = 2):
            value = _clean(getattr(row, attribute, None))
            return {
                "label": label,
                "value": "unavailable" if value is None
                         else f"{value:.{places}f} {unit}".strip(),
                "provenance": "segmentation_derived",
                "available": value is not None,
            }

        lines: list[dict] = []

        grade = (entry.get("pfirrmann_grade") or {})
        lines.append({
            "label": "Pfirrmann grade",
            "value": ("unavailable" if grade.get("value") is None
                      else str(grade["value"])),
            "provenance": ("unsupported" if grade.get("value") is None
                           and grade.get("source") == "unsupported"
                           else "model_prediction"),
            "available": grade.get("value") is not None,
            "strength": grade.get("strength"),
            "unavailable_reason": grade.get("unavailable_reason"),
        })

        for name, label in (
            ("bulging", "Bulging"),
            ("narrowing", "Narrowing"),
            ("any_modic", "Modic change (any)"),
            ("upper_endplate", "Upper endplate change"),
            ("lower_endplate", "Lower endplate change"),
            ("herniation", "Herniation"),
            ("modic_type", "Modic type"),
            ("spondylolisthesis", "Spondylolisthesis"),
        ):
            item = entry.get(name) or {}
            value = item.get("value")
            if value is None:
                rendered = "unavailable"
            else:
                rendered = "detected" if value else "not detected"
            lines.append({
                "label": label,
                "value": rendered,
                "provenance": item.get("source", "unsupported"),
                "available": value is not None,
                "strength": item.get("strength"),
                "unavailable_reason": item.get("unavailable_reason"),
            })

        lines.append(measurement("Disc height (central)", "height_mm_central", "mm"))
        lines.append(measurement("Disc area", "area_mm2", "mm²", 1))
        lines.append(measurement("AP extent", "ap_extent_mm", "mm"))
        lines.append(
            measurement("Disc / vertebra signal ratio",
                        "intensity_disc_vertebra_ratio", "", 4)
        )
        lines.append(
            measurement("Canal width at disc", "canal_width_at_disc_mm", "mm")
        )
        return lines

    def _build_summary(self, discs: list[dict]) -> dict:
        """Deterministic template summary. No free-form generation."""
        heights = [
            d["measurements"]["height_mm_central"] for d in discs
            if d["measurements"]["height_mm_central"] is not None
        ]
        with_findings = [
            d for d in discs
            if any(
                f.get("value") is True for f in d["findings"]
                if f["name"] in FINDING_SEVERITY
            )
        ]

        observed: list[dict] = []

        if not discs:
            observed.append({
                "category": "Detection",
                "text": "No intervertebral disc was identified in this study. "
                        "This can happen when the field of view does not cover "
                        "the lumbar spine, or when the study differs markedly "
                        "from the data the model was validated on.",
                "disc_indices": [], "severity": "info",
            })
            return {"disc_count": 0, "discs_with_findings_count": 0,
                    "mean_disc_height_mm": None, "findings": observed}

        observed.append({
            "category": "Detection",
            "text": f"{len(discs)} intervertebral disc "
                    f"{'level was' if len(discs) == 1 else 'levels were'} "
                    f"identified and indexed from the most inferior disc upward. "
                    f"Indices are integer positions; no anatomical level name "
                    f"(such as L4-L5) is asserted, because the dataset does not "
                    f"state which vertebra is L5.",
            "disc_indices": [d["index"] for d in discs], "severity": "info",
        })

        # Per-finding grouping, only from positive served findings.
        for name, (label, severity) in FINDING_SEVERITY.items():
            positive = [
                d["index"] for d in discs
                if any(f["name"] == name and f.get("value") is True
                       for f in d["findings"])
            ]
            if not positive:
                continue
            weak = any(
                f.get("strength") == "weak"
                for d in discs for f in d["findings"] if f["name"] == name
            )
            qualifier = (
                " This target validated weakly on few positive cases, so treat it "
                "with particular caution."
                if weak else ""
            )
            observed.append({
                "category": "Disc-level finding",
                "text": f"{label}: model-estimated at disc "
                        f"{', '.join(str(i) for i in positive)}.{qualifier}",
                "disc_indices": positive,
                "severity": severity,
            })

        graded = [
            (d["index"], d["pfirrmann_grade"]) for d in discs
            if d["pfirrmann_grade"] is not None
        ]
        if graded:
            highest = max(g for _, g in graded)
            worst = [i for i, g in graded if g == highest]
            observed.append({
                "category": "Degeneration grade",
                "text": f"Model-estimated Pfirrmann grades range from "
                        f"{min(g for _, g in graded)} to {highest}, with the "
                        f"highest at disc {', '.join(str(i) for i in worst)}. "
                        f"Grades are research estimates from segmentation-derived "
                        f"signal and shape features.",
                "disc_indices": worst,
                "severity": "moderate" if highest >= 4 else "low",
            })
        else:
            observed.append({
                "category": "Degeneration grade",
                "text": "Pfirrmann grading was not reported for this study. The "
                        "estimator was validated on T2 and T2-SPACE series only, "
                        "because the grade is defined on T2 signal.",
                "disc_indices": [], "severity": "info",
            })

        observed.append({
            "category": "Scope",
            "text": "Findings above are model-derived research estimates from a "
                    "single timepoint. No change over time is measured, and no "
                    "postoperative assessment is provided, because the data this "
                    "system was validated on contains no longitudinal follow-up.",
            "disc_indices": [], "severity": "info",
        })

        return {
            "disc_count": len(discs),
            "discs_with_findings_count": len(with_findings),
            "mean_disc_height_mm": round(float(np.mean(heights)), 3) if heights else None,
            "findings": observed,
        }
