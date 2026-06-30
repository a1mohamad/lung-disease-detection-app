"""Domain-specific exception types used across the API and pipeline."""

import json


class AppError(Exception):
    """Base exception that serializes to the API error envelope.

    Attributes:
        error_type: Stable high-level error category.
        error_code: Machine-readable error code.
        message: Human-readable explanation.
        details: Optional structured metadata for debugging.
    """

    def __init__(self, error_type: str, error_code: str, message: str, details=None) -> None:
        """Create an application error with stable type, code, and details.

        Args:
            error_type: Stable high-level error category.
            error_code: Machine-readable error code.
            message: Human-readable explanation.
            details: Optional structured metadata for debugging.
        """
        super().__init__(message)
        self.error_type = error_type
        self.error_code = error_code
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict:
        """Return the JSON-serializable error response body.

        Returns:
            Dictionary matching the public API error envelope.
        """
        # Keep the response envelope stable for API clients, Kafka consumers,
        # and logs; optional details are included only when helpful.
        payload = {
            "error_type": self.error_type,
            "error_code": self.error_code,
            "message": self.message,
        }
        if self.details:
            payload["details"] = self.details
        return payload

    def to_json(self) -> str:
        """Return the error response body encoded as JSON text.

        Returns:
            JSON string representation of ``to_dict()``.
        """
        # ASCII JSON avoids encoding surprises in logs and message brokers.
        return json.dumps(self.to_dict(), ensure_ascii=True)


class InputError(AppError):
    """Raised when client-provided inputs are missing or invalid.

    Typical examples include missing image sources, multiple competing image
    sources, malformed base64 payloads, and invalid pagination values.
    """

    def __init__(self, error_code: str, message: str, details=None) -> None:
        """Create an input validation error for API response serialization.

        Args:
            error_code: Machine-readable input error code.
            message: Human-readable explanation.
            details: Optional structured debugging metadata.
        """
        super().__init__("input_error", error_code, message, details)


class ArtifactError(AppError):
    """Raised when model, metadata, or artifact files are missing or invalid.

    This category usually indicates a broken deployment package rather than a
    client request problem.
    """

    def __init__(self, error_code: str, message: str, details=None) -> None:
        """Create an artifact error for missing or invalid project files.

        Args:
            error_code: Machine-readable artifact error code.
            message: Human-readable explanation.
            details: Optional artifact metadata such as paths.
        """
        super().__init__("artifact_error", error_code, message, details)


class ModelError(AppError):
    """Raised when model loading or runtime configuration fails.

    Model errors include missing ONNX Runtime, failed Keras loading, unsupported
    metadata paths, and runtime session creation failures.
    """

    def __init__(self, error_code: str, message: str, details=None) -> None:
        """Create a model-runtime error for loading or session failures.

        Args:
            error_code: Machine-readable model error code.
            message: Human-readable explanation.
            details: Optional model path or runtime metadata.
        """
        super().__init__("model_error", error_code, message, details)


class PreprocessError(AppError):
    """Raised when preprocessing configuration or transformation fails.

    This separates model-input contract problems from general model inference
    failures, which makes API errors easier to triage.
    """

    def __init__(self, error_code: str, message: str, details=None) -> None:
        """Create a preprocessing error for invalid transform configuration.

        Args:
            error_code: Machine-readable preprocessing error code.
            message: Human-readable explanation.
            details: Optional metadata about the requested transform.
        """
        super().__init__("preprocess_error", error_code, message, details)


class InferenceError(AppError):
    """Raised when a model prediction step fails.

    The pipeline wraps lower-level model exceptions in this type so API clients
    receive a stable error shape instead of backend-specific tracebacks.
    """

    def __init__(self, error_code: str, message: str, details=None) -> None:
        """Create an inference error for failed model prediction stages.

        Args:
            error_code: Machine-readable inference error code.
            message: Human-readable explanation.
            details: Optional backend error metadata.
        """
        super().__init__("inference_error", error_code, message, details)


class AuthError(AppError):
    """Raised when protected API access fails authorization.

    Used by operational endpoints such as prediction logs.
    """

    def __init__(self, error_code: str, message: str, details=None) -> None:
        """Create an authorization error for protected API endpoints.

        Args:
            error_code: Machine-readable auth error code.
            message: Human-readable explanation.
            details: Optional authorization metadata.
        """
        super().__init__("auth_error", error_code, message, details)


class ServiceError(AppError):
    """Raised when an optional integration or service dependency fails.

    Examples include disabled database logging, Supabase storage failures, and
    unavailable integration dependencies.
    """

    def __init__(self, error_code: str, message: str, details=None) -> None:
        """Create a service-integration error for optional dependencies.

        Args:
            error_code: Machine-readable service error code.
            message: Human-readable explanation.
            details: Optional dependency metadata.
        """
        super().__init__("service_error", error_code, message, details)


class ImageLoadError(InputError):
    """Raised when an image cannot be loaded, decoded, or validated.

    This is a specialized input error so image-specific problems still map to
    client-facing 400 responses.
    """

    def __init__(self, error_code: str, message: str, details=None) -> None:
        """Create an image-loading error with the standard input-error type.

        Args:
            error_code: Machine-readable image loading error code.
            message: Human-readable explanation.
            details: Optional source metadata.
        """
        super().__init__(error_code, message, details)
