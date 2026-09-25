"""Explicit request and response models.

Every endpoint returns a declared model rather than an ad-hoc dictionary, so the
generated OpenAPI schema is the contract the frontend types are derived from.

Schema note on longitudinal support: the current SPIDER data contains no
postoperative follow-up, so nothing here models change over time. The result
carries a ``timepoint`` block with a single baseline entry and a ``series`` list,
which is the seam a future longitudinal module would populate without altering
any existing field.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ErrorDetail(BaseModel):
    code: str = Field(..., description="Stable machine-readable error code.")
    message: str = Field(..., description="Human-readable explanation.")
    details: dict[str, Any] | None = Field(
        None, description="Optional structured context. Never contains a traceback."
    )


class ErrorResponse(BaseModel):
    error: ErrorDetail


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class ModelInfo(BaseModel):
    loaded: bool
    architecture: str | None = None
    parameters: int | None = None
    checkpoint_name: str | None = None
    selection: str | None = None
    epoch: int | None = None
    device: str
    validated_macro_foreground_dice: float | None = None
    source_sprint: str | None = None


class FindingModelInfo(BaseModel):
    supported_targets: list[str]
    unsupported_targets: dict[str, str] = Field(
        default_factory=dict,
        description="Target name mapped to why it is not served.",
    )
    loaded: bool


class PipelineInfo(BaseModel):
    """Identity of the frozen served pipeline."""

    version: str = Field(..., description="e.g. SPIDER-Lumbar-v1")
    preprocessing: str
    segmentation: str
    postprocessing: str
    preprocessing_detail: str
    segmentation_detail: str
    postprocessing_detail: str
    excluded: dict[str, str] = Field(
        default_factory=dict,
        description="Experiments deliberately not served, and why.",
    )
    dataset: str


class MetricRow(BaseModel):
    """One research evaluation figure, for display."""

    label: str
    value: str
    kind: Literal["segmentation", "postprocessing"]
    sprint: str


class ResearchStage(BaseModel):
    """One experiment in the record, including rejected ones."""

    sprint: str
    title: str
    outcome: Literal["selected", "rejected"]
    summary: str
    headline: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    api_version: str
    pipeline: PipelineInfo
    segmentation_model: ModelInfo
    finding_models: FindingModelInfo
    postprocessing_method: str | None = None
    validated_metrics: "ValidatedMetrics"
    metric_table: list[MetricRow] = Field(
        default_factory=list,
        description="Research evaluation figures for the methodology page. "
                    "Served so the UI never hard-codes a metric.",
    )
    research_progress: list[ResearchStage] = Field(default_factory=list)
    provenance_labels: dict[str, str] = Field(default_factory=dict)
    evidence_labels: dict[str, str] = Field(default_factory=dict)
    sample_study_available: bool = Field(
        False,
        description="Whether the validated sample volume is present, so the UI "
                    "can offer 'Load Sample Study' only when it would work.",
    )
    disclaimer: str
    research_notice: str


# ---------------------------------------------------------------------------
# Study / upload
# ---------------------------------------------------------------------------


class StudyInfo(BaseModel):
    filename: str
    size_bytes: int
    format: str
    modality: str | None = None
    modality_source: str = Field(
        "not_determinable",
        description="How the modality was determined. A converted .mha carries no "
                    "DICOM series description, so this is usually a filename "
                    "heuristic.",
    )
    slice_count: int
    dimensions: list[int] = Field(..., description="Native in-plane [rows, cols].")
    in_plane_spacing_mm: list[float]
    slice_spacing_mm: float
    native_orientation: str


class UploadResponse(BaseModel):
    analysis_id: str
    status: Literal["queued"]
    study: StudyInfo
    created_at: datetime
    message: str
    is_sample: bool = Field(
        False, description="True when created by the sample-study endpoint."
    )


# ---------------------------------------------------------------------------
# Job status
# ---------------------------------------------------------------------------


class AnalysisStatus(str, Enum):
    queued = "queued"
    processing = "processing"
    completed = "completed"
    failed = "failed"


class RunResponse(BaseModel):
    analysis_id: str
    status: AnalysisStatus


class StatusResponse(BaseModel):
    analysis_id: str
    status: AnalysisStatus
    progress: int = Field(..., ge=0, le=100)
    stage: str
    stages_completed: list[str] = Field(default_factory=list)
    elapsed_seconds: float
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: ErrorDetail | None = None


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


class SegmentationClass(BaseModel):
    id: int
    name: str
    display_name: str
    validated_test_dice: float | None = None


class SegmentationInfo(BaseModel):
    classes: list[SegmentationClass]
    slice_count: int
    informative_slice_count: int = Field(
        ..., description="Slices carrying enough predicted anatomy to measure on."
    )
    macro_foreground_dice_validated: float | None = None
    note: str


class FindingEstimate(BaseModel):
    """One model-derived finding for one disc.

    ``value`` is ``None`` whenever the finding is not served or could not be
    computed; ``unavailable_reason`` then explains why. Nothing is guessed.
    """

    name: str
    label: str
    value: bool | int | None = None
    probability: float | None = None
    expected: float | None = Field(
        None, description="Continuous expected grade, for ordinal targets only."
    )
    probabilities: dict[str, float] | None = None
    strength: str | None = Field(
        None, description="How well this target validated: strong/moderate/modest/weak."
    )
    validated_metric: str | None = None
    validated_value: float | None = None
    prevalence_baseline: float | None = None
    within_one_grade: float | None = None
    source: str = "model_prediction"
    caveat: str | None = None
    unavailable_reason: str | None = None


class DiscMeasurements(BaseModel):
    height_mm_central: float | None = None
    height_mm_anterior: float | None = None
    height_mm_posterior: float | None = None
    area_mm2: float | None = None
    ap_extent_mm: float | None = None
    intensity_disc_vertebra_ratio: float | None = None
    intensity_nucleus_annulus_ratio: float | None = None
    canal_width_at_disc_mm: float | None = None
    disc_to_vertebra_height_ratio: float | None = None
    height_ratio_to_series_median: float | None = None
    source: str = "segmentation_derived_measurement"


class StructuredLine(BaseModel):
    """One label/value line for a disc, tagged with how the value was obtained.

    ``value`` is already rendered for display. An unavailable quantity reads
    "unavailable" rather than 0, "No" or "Normal", because a missing value is not
    a negative finding.
    """

    label: str
    value: str
    provenance: Literal["segmentation_derived", "model_prediction", "unsupported"]
    available: bool
    strength: str | None = None
    unavailable_reason: str | None = None


class DiscResult(BaseModel):
    index: int = Field(..., description="Integer disc index, 1 = most inferior.")
    structured: list[StructuredLine] = Field(
        default_factory=list,
        description="Deterministic label/value lines for display.",
    )
    identity_confidence: float | None = Field(
        None,
        description="Fraction of the series' slices supporting this disc's "
                    "series-level track. Not a clinical confidence.",
    )
    representative_slice_id: str | None = None
    representative_slice_index: int | None = None
    slices_present: int | None = None
    measurements: DiscMeasurements
    findings: list[FindingEstimate] = Field(default_factory=list)

    # Convenience mirrors of the most-requested findings, so the frontend does
    # not have to search the list for the common case. Always consistent with
    # `findings`; null when unsupported or not computed.
    pfirrmann_grade: int | None = None
    modic_type: int | None = Field(
        None,
        description="Always null. The nominal Modic type was never modelled; only "
                    "the binary 'any Modic change' target was.",
    )
    modic_any: bool | None = None
    bulging: bool | None = None
    narrowing: bool | None = None
    herniation: bool | None = None
    spondylolisthesis: bool | None = Field(
        None, description="Always null. Too few positive cases to validate."
    )
    upper_endplate: bool | None = None
    lower_endplate: bool | None = None


class ObservedFinding(BaseModel):
    """A deterministic, template-generated statement about the result.

    Produced by rule from the structured fields. No language model is involved,
    so the text cannot drift from the numbers.
    """

    category: str
    text: str
    disc_indices: list[int] = Field(default_factory=list)
    severity: Literal["info", "low", "moderate", "high"] = "info"


class AnalysisSummary(BaseModel):
    disc_count: int
    discs_with_findings_count: int = Field(
        ...,
        description="Discs with at least one positive served finding. Not a "
                    "severity score.",
    )
    mean_disc_height_mm: float | None = None
    findings: list[ObservedFinding] = Field(default_factory=list)
    unsupported_targets: dict[str, str] = Field(default_factory=dict)


class ProcessingInfo(BaseModel):
    preprocessing: str
    segmentation_model: str
    segmentation_source: str
    postprocessing_method: str
    postprocessing_params: dict[str, Any] = Field(default_factory=dict)
    device: str
    duration_seconds: float
    stage_durations: dict[str, float] = Field(default_factory=dict)
    disc_tracks_found: int | None = None
    components_rejected: int | None = None
    components_unassigned: int | None = None


class Timepoint(BaseModel):
    """Single-timepoint container, kept so a longitudinal module can extend it.

    The current dataset has no postoperative follow-up, so exactly one baseline
    timepoint exists and no change over time is computed or claimed.
    """

    id: str = "baseline"
    label: str = "Baseline"
    acquired_at: datetime | None = None
    is_baseline: bool = True


class ValidatedMetrics(BaseModel):
    """The published test-split numbers for the served pipeline."""

    segmentation_macro_foreground_dice: float
    segmentation_vertebra_dice: float
    segmentation_disc_dice: float
    segmentation_canal_dice: float
    disc_indexing_percent: float
    disc_indexing_baseline_percent: float
    disc_height_mae_mm: float
    disc_area_mae_mm2: float
    intensity_ratio_pearson_r: float
    pfirrmann_qwk_end_to_end: float
    test_patients: int
    note: str


class FindingOverlayDisc(BaseModel):
    """One disc the findings overlay marks, and why."""

    index: int
    findings: list[str] = Field(
        default_factory=list,
        description="Labels of the supported positive findings on this disc.",
    )


class FindingOverlayInfo(BaseModel):
    """Which disc regions the findings overlay marks.

    Derived on the server from the served findings so the interface renders the
    same discs the slice endpoint tints, without re-implementing the rule. The
    marked region is the validated disc segmentation, not a pathology mask.
    """

    disc_indices: list[int] = Field(default_factory=list)
    discs: list[FindingOverlayDisc] = Field(default_factory=list)
    eligible_findings: list[str] = Field(
        default_factory=list,
        description="Binary targets that may trigger the overlay. Excludes the "
                    "ordinal Pfirrmann grade and every unsupported target.",
    )
    label: str = ""
    note: str = Field(
        "",
        description="Rendered verbatim by the UI: states that the overlay marks "
                    "finding-associated disc regions, not damaged tissue.",
    )


class AnalysisResult(BaseModel):
    analysis_id: str
    status: AnalysisStatus
    created_at: datetime
    completed_at: datetime | None = None
    is_sample: bool = Field(
        False, description="True when created from the bundled sample study."
    )
    pipeline: PipelineInfo
    study: StudyInfo
    timepoint: Timepoint = Field(default_factory=Timepoint)
    segmentation: SegmentationInfo
    processing: ProcessingInfo
    discs: list[DiscResult] = Field(default_factory=list)
    summary: AnalysisSummary
    validated_metrics: ValidatedMetrics
    provenance_labels: dict[str, str] = Field(default_factory=dict)
    evidence_labels: dict[str, str] = Field(default_factory=dict)
    finding_overlay: FindingOverlayInfo = Field(default_factory=FindingOverlayInfo)
    disclaimer: str


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------


class HistoryItem(BaseModel):
    analysis_id: str
    filename: str
    created_at: datetime
    status: AnalysisStatus
    slice_count: int | None = None
    disc_count: int | None = None
    findings_count: int | None = None
    modality: str | None = None
    is_sample: bool = False


class HistoryResponse(BaseModel):
    items: list[HistoryItem]
    total: int
    counts: dict[str, int] = Field(
        default_factory=dict, description="Analyses per status."
    )


class DeleteResponse(BaseModel):
    analysis_id: str
    deleted: bool
