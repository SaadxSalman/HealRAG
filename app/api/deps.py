"""Shared API dependencies: auth, CORS, exception handlers."""

from __future__ import annotations

from fastapi import Depends, Header, HTTPException, Request, status

from app.config import settings


def require_bearer(authorization: str = Header(default="")) -> None:
    """Enforce a bearer token if one is configured in .env."""
    token = settings.api_bearer_token
    if not token:
        return
    expected = f"Bearer {token}"
    if authorization.strip() != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )


AuthDep = Depends(require_bearer)


def cors_middleware(app) -> None:
    """Register permissive CORS for local development from the UI origin."""
    from fastapi.middleware.cors import CORSMiddleware

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def register_exception_handlers(app) -> None:
    """Map domain exceptions to clean JSON error responses."""
    from fastapi.responses import JSONResponse

    from app.core.error_handling import HealRAGError

    @app.exception_handler(HealRAGError)
    async def healrag_error_handler(request: Request, exc: HealRAGError):
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"error": type(exc).__name__, "message": str(exc)},
        )

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception):
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": type(exc).__name__, "message": str(exc)[:300]},
        )