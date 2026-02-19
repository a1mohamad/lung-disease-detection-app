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
    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError):
        status = 500
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
        return JSONResponse(
            status_code=500,
            content={
                "error_type": "server_error",
                "error_code": "UNHANDLED",
                "message": str(exc),
            }
        )
