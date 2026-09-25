"""Analysis endpoints.

Handlers validate input, call a service, and shape a response. No pipeline code
lives here.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, File, Query, Request, UploadFile, status
from fastapi.responses import Response, StreamingResponse

from backend.app.errors import ApiError, not_found
from backend.app.schemas import (
    AnalysisResult, DeleteResponse, HistoryItem, HistoryResponse, RunResponse,
    StatusResponse, StudyInfo, UploadResponse,
)
from backend.app.services.findings_overlay import overlay_discs, overlay_summary
from backend.app.services.longitudinal import (
    ManifestError, load_case, resolve_volume, stage_ids,
)
from backend.app.services.imaging import (
    CLASS_COLOURS, RENDER_MODES, encode_png, render_slice,
)
from backend.app.services.sample_study import (
    SAMPLE_LABEL, SAMPLE_NOTE, sample_study_path,
)
from backend.app.services.storage import new_analysis_id, utcnow
from backend.app.services.volume_export import build_volume_payload
from ml.volume import ValidationError, sanitise_filename, validate_upload

logger = logging.getLogger("lumbar.api")
router = APIRouter(prefix="/api/analysis", tags=["analysis"])

#: Read uploads in chunks so a large file never sits in memory twice.
CHUNK_BYTES = 1024 * 1024


def _store(request: Request):
    return request.app.state.store


def _jobs(request: Request):
    return request.app.state.jobs


def _settings(request: Request):
    return request.app.state.settings


def _require_record(request: Request, analysis_id: str) -> dict:
    record = _store(request).read_record(analysis_id) if _valid(analysis_id) else None
    if record is None:
        raise not_found(analysis_id)
    return record


def _valid(analysis_id: str) -> bool:
    return bool(analysis_id) and all(c in "0123456789abcdef" for c in analysis_id)


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload an MRI volume and create an analysis",
)
async def upload(request: Request, file: UploadFile = File(...)) -> UploadResponse:
    settings = _settings(request)
    store = _store(request)

    original = sanitise_filename(file.filename or "upload")
    analysis_id = new_analysis_id()
    directory = store.directory(analysis_id)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / original

    written = 0
    try:
        with destination.open("wb") as handle:
            while True:
                chunk = await file.read(CHUNK_BYTES)
                if not chunk:
                    break
                written += len(chunk)
                if written > 300 * 1024 * 1024:
                    raise ApiError(
                        "FILE_TOO_LARGE",
                        "The upload exceeds the 300 MB limit.",
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    )
                handle.write(chunk)
    except ApiError:
        store.delete(analysis_id)
        raise
    except Exception as exc:  # noqa: BLE001
        store.delete(analysis_id)
        logger.exception("Upload write failed")
        raise ApiError(
            "UPLOAD_FAILED", "The upload could not be saved.",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        ) from exc
    finally:
        await file.close()

    # Validate only after the bytes are on disk, because readability and geometry
    # can only be checked against the complete volume.
    try:
        info = validate_upload(destination, original)
    except ValidationError as exc:
        store.delete(analysis_id)
        raise ApiError(
            exc.code, exc.message, status_code=status.HTTP_400_BAD_REQUEST,
            details=exc.details,
        ) from exc

    return _register(request, analysis_id, destination, info, is_sample=False)


def _register(
    request, analysis_id, destination, info, *,
    is_sample: bool, demo_stage: dict | None = None,
):
    """Persist a validated upload as a new analysis record."""
    store = _store(request)
    created = utcnow()
    study = {
        "filename": info.filename,
        "size_bytes": info.size_bytes,
        "format": info.suffix,
        "modality": info.modality,
        "modality_source": info.modality_source,
        "slice_count": info.slice_count,
        "dimensions": list(info.in_plane_shape),
        "in_plane_spacing_mm": list(info.in_plane_spacing_mm),
        "slice_spacing_mm": info.slice_spacing_mm,
        "native_orientation": info.native_orientation,
        "created_at": created.isoformat(),
    }
    store.create(analysis_id, {
        "analysis_id": analysis_id,
        "created_at": created.isoformat(),
        "status": "queued",
        "progress": 0,
        "stage": "Upload validated",
        "stages_completed": ["Upload validated"],
        "upload_name": destination.name,
        "is_sample": is_sample,
        "demo_stage": demo_stage,
        "study": study,
    })

    if demo_stage:
        message = (
            f"Simulated longitudinal demo stage '{demo_stage['label']}' loaded and "
            f"validated. This is a real SPIDER study standing in for a timeline "
            f"position, not follow-up imaging. Start the analysis to process it."
        )
    elif is_sample:
        message = (
            f"{SAMPLE_LABEL} loaded and validated. Start the analysis to process it."
        )
    else:
        message = "Upload validated. Start the analysis to process this study."
    return UploadResponse(
        analysis_id=analysis_id,
        status="queued",
        study=StudyInfo(**{k: v for k, v in study.items() if k != "created_at"}),
        created_at=created,
        message=message,
        is_sample=is_sample,
    )


@router.post(
    "/sample",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create an analysis from the bundled sample study",
)
async def load_sample(request: Request) -> UploadResponse:
    """Load the validated sample volume as a real analysis.

    Demonstration convenience only. The volume is copied into a fresh analysis
    and processed by the identical pipeline - nothing is pre-computed, and the
    response is flagged ``is_sample`` so the interface can label it honestly.
    """
    settings = _settings(request)
    store = _store(request)

    source = sample_study_path(settings)
    if source is None:
        raise ApiError(
            "SAMPLE_UNAVAILABLE",
            "The bundled sample study is not present on this machine. Extract "
            "the dataset with scripts/01_extract.py, or upload a volume instead.",
            status_code=status.HTTP_404_NOT_FOUND,
            details={"note": SAMPLE_NOTE},
        )

    analysis_id = new_analysis_id()
    directory = store.directory(analysis_id)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / sanitise_filename(source.name)

    try:
        destination.write_bytes(source.read_bytes())
        info = validate_upload(destination, source.name)
    except ValidationError as exc:
        store.delete(analysis_id)
        raise ApiError(
            exc.code, exc.message, status_code=status.HTTP_400_BAD_REQUEST,
            details=exc.details,
        ) from exc
    except OSError as exc:
        store.delete(analysis_id)
        logger.exception("Sample study copy failed")
        raise ApiError(
            "SAMPLE_UNAVAILABLE",
            "The sample study could not be prepared.",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        ) from exc

    return _register(request, analysis_id, destination, info, is_sample=True)


@router.post(
    "/demo-stage/{case_id}/{stage_id}",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create an analysis from one stage of a simulated longitudinal demo",
)
async def load_demo_stage(
    request: Request, case_id: str, stage_id: str
) -> UploadResponse:
    """Run the study behind one demonstration stage through the real pipeline.

    The stage's source study is a genuine SPIDER volume, processed by the identical
    pipeline as any upload - nothing is pre-computed. What is simulated is the
    *timeline*: the stages of a case are different studies from different patients,
    not follow-up imaging of one person.

    The disclaimer and the stage's provenance are attached to the analysis record,
    so every later read of this analysis carries them and the interface cannot show
    the stage without them.
    """
    settings = _settings(request)
    store = _store(request)

    try:
        case = load_case(settings.project_root, case_id)
    except FileNotFoundError:
        raise ApiError(
            "DEMO_CASE_NOT_FOUND",
            f"No longitudinal demonstration case {case_id!r} is installed.",
            status_code=status.HTTP_404_NOT_FOUND,
        ) from None
    except ManifestError as exc:
        raise ApiError(
            "DEMO_CASE_INVALID", str(exc),
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        ) from None

    stage = next((s for s in case["stages"] if s["stage_id"] == stage_id), None)
    if stage is None:
        raise ApiError(
            "DEMO_STAGE_NOT_FOUND",
            f"Case {case_id!r} has no stage {stage_id!r}.",
            status_code=status.HTTP_404_NOT_FOUND,
            details={"valid_stages": stage_ids(case)},
        )
    if not stage["available"]:
        raise ApiError(
            "DEMO_STAGE_UNAVAILABLE",
            stage["unavailable_reason"] or "This stage's source study is missing.",
            status_code=status.HTTP_404_NOT_FOUND,
        )

    raw = next(
        s for s in _raw_stages(settings.project_root, case_id)
        if s.get("stage_id") == stage_id
    )
    source = resolve_volume(settings.project_root, str(raw.get("volume_path") or ""))
    if source is None:
        raise ApiError(
            "DEMO_STAGE_UNAVAILABLE",
            "This stage's source study could not be resolved on this machine.",
            status_code=status.HTTP_404_NOT_FOUND,
        )

    analysis_id = new_analysis_id()
    directory = store.directory(analysis_id)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / sanitise_filename(source.name)

    try:
        destination.write_bytes(source.read_bytes())
        info = validate_upload(destination, source.name)
    except ValidationError as exc:
        store.delete(analysis_id)
        raise ApiError(
            exc.code, exc.message, status_code=status.HTTP_400_BAD_REQUEST,
            details=exc.details,
        ) from exc
    except OSError as exc:
        store.delete(analysis_id)
        logger.exception("Demo stage copy failed")
        raise ApiError(
            "DEMO_STAGE_UNAVAILABLE",
            "This demonstration stage could not be prepared.",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        ) from exc

    demo_stage = {
        "case_id": case["case_id"],
        "stage_id": stage["stage_id"],
        "label": stage["label"],
        "research_status": stage["research_status"],
        "display_reference": stage["display_reference"],
        "order": stage["order"],
        "stage_note": stage["stage_note"],
        "provenance": stage["provenance"],
        # Carried on the analysis itself, not looked up later, so the statement
        # travels with every read of this study.
        "disclaimer": case["disclaimer"],
        "ui_notice": case["ui_notice"],
        "is_simulated_timeline": True,
    }
    return _register(
        request, analysis_id, destination, info,
        is_sample=True, demo_stage=demo_stage,
    )


def _raw_stages(project_root, case_id: str) -> list[dict]:
    """The manifest's own stage entries, for fields the API view does not expose.

    ``volume_path`` is deliberately absent from the API response - a client has no
    use for a server filesystem path - so it is read back from the manifest here.
    """
    path = project_root / "demo" / "longitudinal_cases" / case_id / "manifest.json"
    with path.open(encoding="utf-8") as handle:
        return json.load(handle).get("stages") or []


# ---------------------------------------------------------------------------
# Run / status / result
# ---------------------------------------------------------------------------


@router.post(
    "/{analysis_id}/run",
    response_model=RunResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start processing (returns immediately)",
)
async def run(request: Request, analysis_id: str) -> RunResponse:
    record = _require_record(request, analysis_id)
    store, jobs = _store(request), _jobs(request)

    if record.get("status") == "completed":
        return RunResponse(analysis_id=analysis_id, status="completed")
    if jobs.is_active(analysis_id):
        state = jobs.get(analysis_id)
        return RunResponse(analysis_id=analysis_id, status=state.status)

    upload_path = store.directory(analysis_id) / record["upload_name"]
    if not upload_path.exists():
        raise ApiError(
            "UPLOAD_MISSING",
            "The uploaded volume for this analysis is no longer available.",
            status_code=status.HTTP_410_GONE,
        )

    # The sample flag lives on the record; the service needs it to label the
    # result, so it travels with the study payload.
    study = {**record["study"], "is_sample": bool(record.get("is_sample"))}
    jobs.submit(analysis_id, upload_path, study)
    return RunResponse(analysis_id=analysis_id, status="queued")


@router.get(
    "/{analysis_id}/status",
    response_model=StatusResponse,
    summary="Poll job progress",
)
async def job_status(request: Request, analysis_id: str) -> StatusResponse:
    _require_record(request, analysis_id)
    state = _jobs(request).get(analysis_id)
    if state is None:
        raise not_found(analysis_id)
    return StatusResponse(**state.to_dict())


@router.get(
    "/{analysis_id}/result",
    response_model=AnalysisResult,
    summary="Fetch the completed analysis result",
)
async def result(request: Request, analysis_id: str) -> AnalysisResult:
    record = _require_record(request, analysis_id)
    payload = _store(request).read_result(analysis_id)
    if payload is None:
        state = _jobs(request).get(analysis_id)
        current = state.status if state else record.get("status", "queued")
        if current == "failed":
            raise ApiError(
                "ANALYSIS_FAILED",
                "This analysis failed, so no result is available.",
                status_code=status.HTTP_409_CONFLICT,
                details=(state.error if state else None),
            )
        raise ApiError(
            "RESULT_NOT_READY",
            f"The analysis is {current}. Poll the status endpoint until it "
            f"reports 'completed'.",
            status_code=status.HTTP_409_CONFLICT,
            details={"status": current},
        )
    # Derived on read rather than stored, so the overlay rule applies to analyses
    # that were completed before it existed and there is nothing to migrate.
    payload = {
        **payload,
        "finding_overlay": overlay_summary(payload),
        # From the record rather than the stored result: the pipeline has no
        # concept of a demonstration stage, and should not acquire one.
        "demo_stage": record.get("demo_stage"),
    }
    return AnalysisResult(**payload)


# ---------------------------------------------------------------------------
# Slice imaging
# ---------------------------------------------------------------------------


@router.get(
    "/{analysis_id}/slice/{slice_id}",
    summary="Render one slice as PNG",
    responses={200: {"content": {"image/png": {}}}},
    response_class=Response,
)
async def slice_image(
    request: Request,
    analysis_id: str,
    slice_id: int,
    mode: str = Query("overlay", pattern="^(original|mask|overlay)$"),
    opacity: float = Query(0.45, ge=0.0, le=1.0),
    classes: str = Query("1,2,3", description="Comma-separated class ids to show."),
    scale: int = Query(1, ge=1, le=3),
    highlight_disc: int | None = Query(
        None, ge=1, le=9,
        description="Draw a marker around this disc index, to keep the viewer in "
                    "step with the disc selected in the results panel.",
    ),
    highlight: str = Query(
        "disc", pattern="^(none|disc|findings)$",
        description="How to interpret the highlight request. 'disc' draws the "
                    "selected-disc marker only. 'findings' additionally tints "
                    "the disc regions associated with a supported, positive, "
                    "model-estimated finding. 'none' suppresses both.",
    ),
    finding_opacity: float = Query(
        0.5, ge=0.0, le=1.0,
        description="Alpha of the findings tint, when highlight=findings.",
    ),
) -> Response:
    """Return a single rendered slice.

    ``slice_id`` is the zero-based slice index within the study. Rendering happens
    server-side so the volume itself never crosses the network, and a hidden class
    is never transmitted.

    Which discs the findings overlay may mark is decided here, on the server, from
    the stored result - see ``services/findings_overlay.py``. The browser asks for
    the overlay; it does not decide what qualifies for it.
    """
    _require_record(request, analysis_id)
    arrays = _store(request).read_arrays(analysis_id)
    if arrays is None:
        raise ApiError(
            "SLICES_NOT_READY",
            "Slice images are not available until the analysis has completed.",
            status_code=status.HTTP_409_CONFLICT,
        )

    total = len(arrays["images"])
    if not (0 <= slice_id < total):
        raise ApiError(
            "SLICE_OUT_OF_RANGE",
            f"Slice {slice_id} does not exist. This study has {total} slices "
            f"(0-{total - 1}).",
            status_code=status.HTTP_404_NOT_FOUND,
            details={"slice_count": total},
        )

    try:
        wanted = tuple(
            int(part) for part in classes.split(",") if part.strip() != ""
        )
    except ValueError as exc:
        raise ApiError(
            "INVALID_CLASSES",
            "The 'classes' parameter must be a comma-separated list of class ids.",
        ) from exc
    wanted = tuple(c for c in wanted if c in CLASS_COLOURS)

    # The overlay's membership rule lives in one module and is applied here, so a
    # client cannot widen it by crafting a request.
    finding_discs: tuple[int, ...] = ()
    if highlight == "findings":
        stored = _store(request).read_result(analysis_id)
        if stored:
            finding_discs = tuple(overlay_discs(stored))

    rgb = render_slice(
        arrays["images"][slice_id],
        arrays["semantic"][slice_id],
        mode=mode,
        opacity=opacity,
        classes=wanted or (1, 2, 3),
        instance=arrays["instance"][slice_id],
        highlight_disc=None if highlight == "none" else highlight_disc,
        finding_discs=finding_discs,
        finding_opacity=finding_opacity,
    )
    png = encode_png(rgb, scale=scale)
    return Response(
        content=png,
        media_type="image/png",
        headers={
            # Immutable: a completed analysis's slices never change, so the
            # browser can cache aggressively and the viewer feels instant.
            "Cache-Control": "public, max-age=86400, immutable",
            "X-Slice-Count": str(total),
            # Which discs the server decided to mark, for auditing a rendered
            # frame without re-deriving the rule.
            "X-Finding-Discs": ",".join(str(i) for i in sorted(finding_discs)),
        },
    )


@router.get(
    "/{analysis_id}/volume",
    summary="Download the analysis volume for browser-side 3D rendering",
    responses={200: {"content": {"application/octet-stream": {}}}},
    response_class=Response,
)
async def volume(request: Request, analysis_id: str) -> Response:
    """Return the whole volume in one response, for a WebGL renderer.

    **This intentionally reverses the previous architecture.** Until Sprint 8 the
    volume never crossed the network: slices were composed server-side and sent as
    PNGs, so a class hidden in the interface was never transmitted. Real
    volumetric rendering needs the voxels, so this endpoint sends the image volume
    and both label maps in full - including labels for classes the user has
    hidden. The slice endpoint is unchanged and still the path the 2D viewer uses.

    Restricted to completed analyses, because the arrays do not exist before then.
    See ``services/volume_export.py`` for the wire format.
    """
    _require_record(request, analysis_id)
    arrays = _store(request).read_arrays(analysis_id)
    if arrays is None:
        raise ApiError(
            "VOLUME_NOT_READY",
            "The volume is available once the analysis has completed.",
            status_code=status.HTTP_409_CONFLICT,
        )

    stored = _store(request).read_result(analysis_id) or {}
    try:
        payload, header = build_volume_payload(
            arrays,
            analysis_id=analysis_id,
            study=stored.get("study"),
            pipeline_version=(stored.get("pipeline") or {}).get("version"),
            finding_discs=tuple(overlay_discs(stored)) if stored else (),
        )
    except ValueError as exc:
        # A malformed stored volume is a server-side problem, but it must not leak
        # a traceback or 500 without a code the client can act on.
        logger.exception("Volume export failed for %s", analysis_id)
        raise ApiError(
            "VOLUME_UNAVAILABLE",
            f"The stored volume for this analysis could not be serialised: {exc}",
            status_code=status.HTTP_409_CONFLICT,
        ) from None

    return Response(
        content=payload,
        media_type="application/octet-stream",
        headers={
            # A completed analysis's volume never changes, so one fetch per
            # analysis is enough and a revisit is free.
            "Cache-Control": "public, max-age=86400, immutable",
            "X-Volume-Format-Version": str(header["format_version"]),
            "X-Volume-Dimensions": (
                f"{header['dimensions']['slices']}x"
                f"{header['dimensions']['rows']}x"
                f"{header['dimensions']['cols']}"
            ),
            "X-Finding-Discs": ",".join(str(i) for i in header["finding_discs"]),
        },
    )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


@router.get(
    "/{analysis_id}/report",
    summary="Structured report payload",
    response_model=AnalysisResult,
)
async def report(request: Request, analysis_id: str) -> AnalysisResult:
    """The report view uses the same validated payload as the result view.

    Keeping one source avoids a second schema that could drift from the first.
    """
    return await result(request, analysis_id)


@router.get(
    "/{analysis_id}/download",
    summary="Download the report as Markdown or JSON",
    response_class=Response,
)
async def download(
    request: Request,
    analysis_id: str,
    fmt: str = Query("md", pattern="^(md|json)$"),
) -> Response:
    payload = _store(request).read_result(analysis_id)
    if payload is None:
        raise ApiError(
            "RESULT_NOT_READY",
            "The report is available once the analysis has completed.",
            status_code=status.HTTP_409_CONFLICT,
        )

    if fmt == "json":
        body = json.dumps(payload, indent=2).encode("utf-8")
        media, suffix = "application/json", "json"
    else:
        from backend.app.services.report_writer import render_markdown

        body = render_markdown(payload).encode("utf-8")
        media, suffix = "text/markdown; charset=utf-8", "md"

    filename = f"lumbar_analysis_{analysis_id}.{suffix}"
    return Response(
        content=body,
        media_type=media,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------


@router.get("", response_model=HistoryResponse, summary="List analyses")
async def history(request: Request, limit: int = Query(50, ge=1, le=200)) -> HistoryResponse:
    records = _store(request).list_records()
    counts: dict[str, int] = {}
    for record in records:
        state = record.get("status", "queued")
        counts[state] = counts.get(state, 0) + 1

    items: list[HistoryItem] = []
    for record in records[:limit]:
        study = record.get("study") or {}
        created = record.get("created_at")
        items.append(HistoryItem(
            analysis_id=record["analysis_id"],
            filename=study.get("filename") or record.get("upload_name") or "unknown",
            created_at=datetime.fromisoformat(created) if created else utcnow(),
            status=record.get("status", "queued"),
            slice_count=study.get("slice_count"),
            disc_count=record.get("disc_count"),
            findings_count=record.get("findings_count"),
            modality=study.get("modality"),
            is_sample=bool(record.get("is_sample")),
        ))
    return HistoryResponse(items=items, total=len(records), counts=counts)


@router.delete(
    "/{analysis_id}", response_model=DeleteResponse, summary="Delete an analysis"
)
async def delete(request: Request, analysis_id: str) -> DeleteResponse:
    _require_record(request, analysis_id)
    deleted = _store(request).delete(analysis_id)
    return DeleteResponse(analysis_id=analysis_id, deleted=deleted)
