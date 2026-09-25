"""Health endpoint.

Reports what is actually loaded, plus the pipeline identity and the research
evaluation figures. Serving the metrics from here is deliberate: the frontend must
never hard-code an experimental number, so there is exactly one place a figure can
come from.
"""

from __future__ import annotations

from fastapi import APIRouter, Request

from backend.app.pipeline_info import (
    DISCLAIMER,
    EVIDENCE_LABELS,
    METRIC_TABLE,
    PIPELINE,
    PROVENANCE,
    RESEARCH_NOTICE,
    RESEARCH_PROGRESS,
    SEGMENTATION_DICE,
    VALIDATED_METRICS,
)
from backend.app.schemas import (
    FindingModelInfo,
    HealthResponse,
    MetricRow,
    ModelInfo,
    PipelineInfo,
    ResearchStage,
    ValidatedMetrics,
)
from backend.app.services.sample_study import sample_study_path

router = APIRouter(prefix="/api", tags=["system"])


@router.get("/health", response_model=HealthResponse, summary="Service health")
async def health(request: Request) -> HealthResponse:
    models = request.app.state.models
    settings = request.app.state.settings
    loaded = models.segmentation

    if loaded is None:
        segmentation = ModelInfo(loaded=False, device="unknown")
    else:
        segmentation = ModelInfo(
            loaded=True,
            architecture=loaded.architecture,
            parameters=loaded.parameters,
            checkpoint_name=loaded.checkpoint_name,
            selection=loaded.metadata.get("selection"),
            epoch=loaded.metadata.get("epoch"),
            device=loaded.device,
            validated_macro_foreground_dice=SEGMENTATION_DICE["macro_foreground"],
            source_sprint="Sprint 3 Coverage",
        )

    findings = models.findings
    return HealthResponse(
        status="ok" if models.ready else "degraded",
        api_version=settings.api_version,
        pipeline=PipelineInfo(**PIPELINE),
        segmentation_model=segmentation,
        finding_models=FindingModelInfo(
            supported_targets=findings.supported_targets,
            unsupported_targets=(findings.manifest or {}).get("unsupported", {}),
            loaded=bool(findings.supported_targets),
        ),
        postprocessing_method=(
            f"Sprint 5 {models.postprocessing_method}"
            if models.postprocessing_method else None
        ),
        validated_metrics=ValidatedMetrics(**VALIDATED_METRICS),
        metric_table=[MetricRow(**row) for row in METRIC_TABLE],
        research_progress=[ResearchStage(**row) for row in RESEARCH_PROGRESS],
        provenance_labels=PROVENANCE,
        evidence_labels=EVIDENCE_LABELS,
        sample_study_available=sample_study_path(settings) is not None,
        disclaimer=DISCLAIMER,
        research_notice=RESEARCH_NOTICE,
    )
