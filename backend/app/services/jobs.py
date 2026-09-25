"""Background analysis jobs.

A single-process thread pool is the right tool here. Inference is CPU-bound and
memory-constrained, the workload is one study at a time on a researcher's laptop,
and introducing Celery plus Redis would add two services to run for no benefit.
Job state is kept in memory for liveness and mirrored into the analysis record so
it survives a restart.

``POST /run`` returns as soon as the job is queued; the browser polls
``GET /status``. Progress values come from real stage boundaries in the pipeline,
never from a timer.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from backend.app.services.mri_analysis_service import MriAnalysisService, STAGES
from backend.app.services.storage import AnalysisStore, utcnow

logger = logging.getLogger("lumbar.api")


@dataclass
class JobState:
    analysis_id: str
    status: str = "queued"
    progress: int = 0
    stage: str = "Queued"
    stages_completed: list[str] = field(default_factory=list)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: dict | None = None

    @property
    def elapsed_seconds(self) -> float:
        if self.started_at is None:
            return 0.0
        end = self.finished_at or utcnow()
        return round((end - self.started_at).total_seconds(), 2)

    def to_dict(self) -> dict:
        return {
            "analysis_id": self.analysis_id,
            "status": self.status,
            "progress": self.progress,
            "stage": self.stage,
            "stages_completed": list(self.stages_completed),
            "elapsed_seconds": self.elapsed_seconds,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "error": self.error,
        }


class JobManager:
    """Owns the executor and the live job table."""

    def __init__(
        self, service: MriAnalysisService, store: AnalysisStore, *,
        max_workers: int = 1, timeout_seconds: int = 1800,
    ) -> None:
        self.service = service
        self.store = store
        self.timeout_seconds = timeout_seconds
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="analysis"
        )
        self._jobs: dict[str, JobState] = {}
        self._lock = threading.Lock()

    # -- state -----------------------------------------------------------

    def get(self, analysis_id: str) -> JobState | None:
        with self._lock:
            live = self._jobs.get(analysis_id)
        if live is not None:
            return live
        # Not in memory: rebuild from the persisted record so a restart does not
        # make a finished analysis look like it never ran.
        record = self.store.read_record(analysis_id)
        if record is None:
            return None
        status = record.get("status", "queued")
        error = record.get("error")
        if status == "processing":
            # The record says processing but no job is running in this process, so
            # the server restarted mid-analysis. Reporting it as still processing
            # would leave the UI polling forever.
            status = "failed"
            error = error or {
                "code": "ANALYSIS_INTERRUPTED",
                "message": "The analysis was interrupted before it finished, most "
                           "likely by a server restart. Start it again.",
                "details": None,
            }
        state = JobState(
            analysis_id=analysis_id,
            status=status,
            progress=int(record.get("progress", 0)),
            stage=record.get("stage", "Queued"),
            stages_completed=list(record.get("stages_completed") or []),
            error=error,
        )
        for key in ("started_at", "finished_at"):
            raw = record.get(key)
            if raw:
                try:
                    setattr(state, key, datetime.fromisoformat(raw))
                except ValueError:
                    pass
        return state

    def is_active(self, analysis_id: str) -> bool:
        """True only if this process is currently running the job.

        Liveness is process-local by definition. A persisted record can say
        'queued' simply because the upload created it, and after a restart a
        record left at 'processing' is stale rather than running - in both cases
        the job is not active and must be submittable, so only the in-memory
        table is consulted.
        """
        with self._lock:
            state = self._jobs.get(analysis_id)
        return state is not None and state.status in ("queued", "processing")

    # -- submission ------------------------------------------------------

    def submit(self, analysis_id: str, upload_path: Path, study: dict) -> JobState:
        with self._lock:
            existing = self._jobs.get(analysis_id)
            if existing and existing.status in ("queued", "processing"):
                return existing
            state = JobState(analysis_id=analysis_id, status="queued", progress=0,
                             stage="Queued")
            self._jobs[analysis_id] = state

        self.store.update_record(
            analysis_id, status="queued", progress=0, stage="Queued", error=None
        )
        self._executor.submit(self._run, analysis_id, upload_path, study)
        return state

    # -- execution -------------------------------------------------------

    def _persist(self, state: JobState) -> None:
        self.store.update_record(
            state.analysis_id,
            status=state.status,
            progress=state.progress,
            stage=state.stage,
            stages_completed=state.stages_completed,
            started_at=state.started_at.isoformat() if state.started_at else None,
            finished_at=state.finished_at.isoformat() if state.finished_at else None,
            error=state.error,
        )

    def _run(self, analysis_id: str, upload_path: Path, study: dict) -> None:
        state = self._jobs[analysis_id]
        state.status = "processing"
        state.started_at = utcnow()
        state.progress = 0
        state.stage = "Upload validated"
        state.stages_completed = ["Upload validated"]
        self._persist(state)
        deadline = time.monotonic() + self.timeout_seconds

        def progress(value: int, stage: str) -> None:
            # Monotonic: a later stage never reports a lower number than an
            # earlier one, so the bar cannot move backwards.
            state.progress = max(state.progress, min(int(value), 100))
            if stage != state.stage:
                if state.stage and state.stage not in state.stages_completed:
                    state.stages_completed.append(state.stage)
                state.stage = stage
            if time.monotonic() > deadline:
                raise TimeoutError("Analysis exceeded the configured time limit")
            self._persist(state)

        try:
            outcome = self.service.run(
                analysis_id, upload_path, study, progress=progress
            )
            self.store.write_result(analysis_id, outcome.result)
            self.store.write_arrays(
                analysis_id,
                images=outcome.images,
                semantic=outcome.semantic,
                instance=outcome.instance,
                slice_ids=outcome.slice_ids,
            )
            state.status = "completed"
            state.progress = 100
            if state.stage not in state.stages_completed:
                state.stages_completed.append(state.stage)
            state.stage = "Complete"
            if "Complete" not in state.stages_completed:
                state.stages_completed.append("Complete")
            state.finished_at = utcnow()
            summary = outcome.result.get("summary", {})
            self.store.update_record(
                analysis_id,
                disc_count=summary.get("disc_count"),
                findings_count=summary.get("discs_with_findings_count"),
            )
            logger.info(
                "Analysis %s completed in %.1fs (%s discs)",
                analysis_id, state.elapsed_seconds, summary.get("disc_count"),
            )
        except TimeoutError as exc:
            state.status = "failed"
            state.finished_at = utcnow()
            state.error = {
                "code": "ANALYSIS_TIMEOUT",
                "message": "Analysis timed out before it could finish.",
                "details": {"limit_seconds": self.timeout_seconds},
            }
            logger.warning("Analysis %s timed out: %s", analysis_id, exc)
        except Exception as exc:  # noqa: BLE001 - a job must never crash the app
            state.status = "failed"
            state.finished_at = utcnow()
            stage = state.stage or "processing"
            # The exception type is safe to surface; the message and traceback are
            # not, so they stay in the server log.
            state.error = {
                "code": "ANALYSIS_FAILED",
                "message": f"Analysis failed during {stage.lower()}.",
                "details": {"stage": stage, "error_type": type(exc).__name__},
            }
            logger.exception("Analysis %s failed during %s", analysis_id, stage)
        finally:
            self._persist(state)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
