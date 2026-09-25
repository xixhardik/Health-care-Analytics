"""Uniform error handling.

Every failure leaves the API as ``{"error": {"code", "message", "details"}}``.
Internal exceptions are logged server-side and replaced with a generic message,
so a Python traceback can never reach the browser.
"""

from __future__ import annotations

import logging
import uuid

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger("lumbar.api")


class ApiError(Exception):
    """An error that is safe to describe to the client."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        status_code: int = status.HTTP_400_BAD_REQUEST,
        details: dict | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or {}


def error_body(code: str, message: str, details: dict | None = None) -> dict:
    return {"error": {"code": code, "message": message, "details": details}}


def not_found(analysis_id: str) -> ApiError:
    return ApiError(
        "ANALYSIS_NOT_FOUND",
        f"No analysis exists with id '{analysis_id}'.",
        status_code=status.HTTP_404_NOT_FOUND,
    )


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=error_body(exc.code, exc.message, exc.details or None),
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(
        _: Request, exc: RequestValidationError
    ) -> JSONResponse:
        # Field names are safe to return; raw input values are not echoed back,
        # because an upload body can contain image data.
        fields = [
            {"field": ".".join(str(p) for p in err.get("loc", [])),
             "problem": err.get("msg", "invalid")}
            for err in exc.errors()
        ]
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=error_body(
                "INVALID_REQUEST",
                "The request could not be processed because it was malformed.",
                {"fields": fields},
            ),
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        codes = {
            404: "NOT_FOUND",
            405: "METHOD_NOT_ALLOWED",
            413: "PAYLOAD_TOO_LARGE",
        }
        return JSONResponse(
            status_code=exc.status_code,
            content=error_body(
                codes.get(exc.status_code, "HTTP_ERROR"),
                str(exc.detail) if exc.detail else "Request failed.",
            ),
        )

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
        # A correlation id lets a developer find the traceback in the server log
        # without any internal detail crossing the network.
        reference = uuid.uuid4().hex[:12]
        logger.exception("Unhandled error [ref=%s]", reference)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=error_body(
                "INTERNAL_ERROR",
                "The server encountered an unexpected error. The incident has "
                "been logged.",
                {"reference": reference},
            ),
        )
