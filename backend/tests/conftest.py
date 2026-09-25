"""Shared test fixtures.

Tests run against the real application, including the real model, because the
point of the suite is to prove the wiring works end to end. Storage is redirected
to a temporary directory so a test run never touches ``outputs/app_data``.

One small representative volume is used throughout rather than the dataset.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

#: The smallest extracted study: 24 sagittal slices, 242x305 in-plane. It
#: exercises both branches of the crop/pad logic, which a 352x256-native volume
#: would not.
SAMPLE_VOLUME = PROJECT_ROOT / "data" / "extracted" / "images" / "33_t2.mha"


@pytest.fixture(scope="session")
def sample_volume() -> Path:
    if not SAMPLE_VOLUME.exists():
        pytest.skip(f"Sample volume not available at {SAMPLE_VOLUME}")
    return SAMPLE_VOLUME


@pytest.fixture(scope="session")
def app_client(tmp_path_factory):
    """A TestClient over the real app, with isolated storage."""
    storage = tmp_path_factory.mktemp("app_data")
    os.environ["STORAGE_DIR"] = str(storage)

    # Settings are cached, so clear the cache after redirecting storage.
    from backend.app.config import get_settings

    get_settings.cache_clear()

    from fastapi.testclient import TestClient

    from backend.app.main import create_app

    with TestClient(create_app()) as client:
        yield client


@pytest.fixture
def uploaded(app_client, sample_volume):
    """Upload the sample volume and return the created analysis payload."""
    with sample_volume.open("rb") as handle:
        response = app_client.post(
            "/api/analysis/upload",
            files={"file": (sample_volume.name, handle, "application/octet-stream")},
        )
    assert response.status_code == 201, response.text
    return response.json()
