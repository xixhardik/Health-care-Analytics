"""FastAPI application entrypoint.

The model is loaded once during the lifespan startup and shared by every request.
Nothing in a request path reads a checkpoint.

Run locally:
    uvicorn backend.app.main:app --reload --port 8000
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.middleware.gzip import GZipMiddleware  # noqa: E402

from backend.app.config import get_settings  # noqa: E402
from backend.app.errors import register_error_handlers  # noqa: E402
from backend.app.routers import analysis as analysis_router  # noqa: E402
from backend.app.routers import health as health_router  # noqa: E402
from backend.app.routers import longitudinal as longitudinal_router  # noqa: E402
from backend.app.services.jobs import JobManager  # noqa: E402
from backend.app.services.model_service import init_model_service  # noqa: E402
from backend.app.services.mri_analysis_service import MriAnalysisService  # noqa: E402
from backend.app.services.storage import AnalysisStore  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
)
logger = logging.getLogger("lumbar.api")

DESCRIPTION = """
REST API for AI-assisted lumbar spine MRI research analysis.

The served pipeline is frozen and was validated across earlier sprints:

* **Preprocessing** - resample to 1.0 mm/px, centre crop or pad to 352x256,
  per-volume percentile normalisation, median denoise, CLAHE
* **Segmentation** - 16-channel U-Net, test macro foreground Dice **0.90001**
* **Disc indexing** - series-level ordering with vertebral-body separation,
  test indexing accuracy **93.84%** (up from 84.08% per-slice)
* **Findings** - ordinal Pfirrmann estimator (end-to-end QWK **0.6543**) plus
  binary finding estimators

Research and educational prototype. Every finding is a model-derived research
estimate requiring review by a qualified radiologist. The API does not provide a
clinical diagnosis.
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.settings = settings
    logger.info("Project root: %s", settings.project_root)
    logger.info("Checkpoint:   %s", settings.checkpoint)

    models = init_model_service(settings)
    store = AnalysisStore(settings.analyses_dir)
    service = MriAnalysisService(settings, models, store)
    jobs = JobManager(
        service, store,
        max_workers=settings.max_workers,
        timeout_seconds=settings.job_timeout_seconds,
    )

    app.state.models = models
    app.state.store = store
    app.state.service = service
    app.state.jobs = jobs

    if not models.ready:
        logger.warning(
            "Segmentation model NOT loaded - /api/health will report 'degraded' "
            "and analyses will be refused."
        )
    yield
    jobs.shutdown()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version=settings.api_version,
        description=DESCRIPTION,
        lifespan=lifespan,
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
    )
    # The volume endpoint sends ~6 MB of uint8 voxels. Label maps in particular
    # are extremely compressible (4 distinct values over a mostly-empty volume),
    # so gzip takes the whole payload to roughly 1 MB. Without this the "one
    # volume request" design still moves 6 MB per analysis. minimum_size skips
    # the small JSON responses where framing overhead would dominate.
    app.add_middleware(GZipMiddleware, minimum_size=1024)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Slice-Count"],
    )
    register_error_handlers(app)
    app.include_router(health_router.router)
    app.include_router(analysis_router.router)
    app.include_router(longitudinal_router.router)
    return app


app = create_app()
