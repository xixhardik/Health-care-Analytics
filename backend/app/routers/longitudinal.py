"""Longitudinal demonstration endpoints.

Serves the simulated demonstration cases defined under ``demo/longitudinal_cases/``.
Every response carries the disclaimer and per-stage provenance, so a client
cannot obtain a stage without also obtaining the fact that it is not real
postoperative follow-up.

These endpoints read demonstration manifests. They never touch raw dataset files
and never create an analysis: running a stage through the pipeline goes through
the existing upload/sample architecture.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request, status

from backend.app.errors import ApiError
from backend.app.schemas import (
    LongitudinalCase, LongitudinalCaseList, LongitudinalStage,
)
from backend.app.services.longitudinal import (
    RESEARCH_STATUSES, ManifestError, load_case, load_cases, stage_ids,
)

logger = logging.getLogger("lumbar.api")
router = APIRouter(prefix="/api/longitudinal", tags=["longitudinal"])

LIST_NOTICE = (
    "Simulated longitudinal demonstrations. The SPIDER dataset contains no "
    "postoperative longitudinal follow-up, so each case arranges different real "
    "studies from different patients into a timeline to demonstrate the "
    "workflow. No recovery outcome is claimed."
)


def _project_root(request: Request):
    return request.app.state.settings.project_root


@router.get(
    "/demo-cases",
    response_model=LongitudinalCaseList,
    summary="List simulated longitudinal demonstration cases",
)
async def demo_cases(request: Request) -> LongitudinalCaseList:
    """Every readable demonstration case.

    An empty list is a valid answer - it means no demonstration case is installed
    on this machine - and is not an error.
    """
    cases = load_cases(_project_root(request))
    return LongitudinalCaseList(
        cases=[LongitudinalCase(**case) for case in cases],
        total=len(cases),
        notice=LIST_NOTICE,
        research_statuses=list(RESEARCH_STATUSES),
    )


@router.get(
    "/demo-cases/{case_id}",
    response_model=LongitudinalCase,
    summary="One simulated longitudinal demonstration case",
)
async def demo_case(request: Request, case_id: str) -> LongitudinalCase:
    try:
        case = load_case(_project_root(request), case_id)
    except FileNotFoundError:
        raise ApiError(
            "DEMO_CASE_NOT_FOUND",
            f"No longitudinal demonstration case {case_id!r} is installed.",
            status_code=status.HTTP_404_NOT_FOUND,
        ) from None
    except ManifestError as exc:
        # A manifest that misdescribes itself is refused rather than served.
        raise ApiError(
            "DEMO_CASE_INVALID",
            str(exc),
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        ) from None
    return LongitudinalCase(**case)


@router.get(
    "/demo-cases/{case_id}/stage/{stage_id}",
    response_model=LongitudinalStage,
    summary="One stage of a demonstration case",
)
async def demo_stage(
    request: Request, case_id: str, stage_id: str
) -> LongitudinalStage:
    """One timeline position.

    An unknown stage returns the valid stage ids in ``details``, so a client that
    has drifted from the manifest can correct itself instead of guessing.
    """
    try:
        case = load_case(_project_root(request), case_id)
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

    for stage in case["stages"]:
        if stage["stage_id"] == stage_id:
            return LongitudinalStage(**stage)

    raise ApiError(
        "DEMO_STAGE_NOT_FOUND",
        f"Case {case_id!r} has no stage {stage_id!r}.",
        status_code=status.HTTP_404_NOT_FOUND,
        details={"valid_stages": stage_ids(case)},
    )
