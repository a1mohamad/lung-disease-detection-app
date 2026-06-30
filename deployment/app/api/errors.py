"""FastAPI exception handlers that normalize application errors to JSON."""

from app.utils.errors import (AppError,
                              ArtifactError,
                              AuthError,
                              ImageLoadError,
                              InputError,
                              InferenceError,
                              ModelError,
                              ServiceError)

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


def register_exception_handlers(app: FastAPI) -> None:
    """Register HTTP status mapping for domain-specific application errors.

    Args:
        app: FastAPI application instance that should receive the handlers.

    Notes:
        The project raises typed ``AppError`` subclasses from lower layers.
        Centralizing the HTTP mapping here keeps route handlers focused on
        business flow instead of repeating exception serialization.
    """
    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError):
        """Convert known application exceptions into stable API responses.

        Args:
            request: FastAPI request object supplied by the exception handler.
            exc: Domain-specific exception carrying error type, code, message,
                and optional details.

        Returns:
            JSON response with the project's normalized error envelope.
        """
        status = 500
        # Client and authorization errors receive specific HTTP status codes;
        # model/artifact failures stay as server-side errors.
        if isinstance(exc, (InputError, ImageLoadError)):
            status = 400
        elif isinstance(exc, AuthError):
            status = 403
        elif isinstance(exc, ServiceError):
            status = 503
        elif isinstance(exc, (ArtifactError, InferenceError, ModelError)):
            status = 500

        return JSONResponse(status_code=status, content=exc.to_dict())
    
    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception):
        """Return a safe JSON envelope for unexpected server exceptions.

        Args:
            request: FastAPI request object supplied by the exception handler.
            exc: Unexpected Python exception.

        Returns:
            JSON response using the same envelope shape as known application
            errors, which keeps client parsing simple.
        """
        return JSONResponse(
            status_code=500,
            content={
                "error_type": "server_error",
                "error_code": "UNHANDLED",
                "message": str(exc),
            }
        )
