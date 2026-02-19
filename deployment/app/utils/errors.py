import json


class AppError(Exception):
    def __init__(self, error_type: str, error_code: str, message: str, details=None) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.error_code = error_code
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict:
        payload = {
            "error_type": self.error_type,
            "error_code": self.error_code,
            "message": self.message,
        }
        if self.details:
            payload["details"] = self.details
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=True)


class InputError(AppError):
    def __init__(self, error_code: str, message: str, details=None) -> None:
        super().__init__("input_error", error_code, message, details)


class ArtifactError(AppError):
    def __init__(self, error_code: str, message: str, details=None) -> None:
        super().__init__("artifact_error", error_code, message, details)


class ModelError(AppError):
    def __init__(self, error_code: str, message: str, details=None) -> None:
        super().__init__("model_error", error_code, message, details)


class PreprocessError(AppError):
    def __init__(self, error_code: str, message: str, details=None) -> None:
        super().__init__("preprocess_error", error_code, message, details)


class InferenceError(AppError):
    def __init__(self, error_code: str, message: str, details=None) -> None:
        super().__init__("inference_error", error_code, message, details)


class AuthError(AppError):
    def __init__(self, error_code: str, message: str, details=None) -> None:
        super().__init__("auth_error", error_code, message, details)


class ServiceError(AppError):
    def __init__(self, error_code: str, message: str, details=None) -> None:
        super().__init__("service_error", error_code, message, details)


class ImageLoadError(InputError):
    def __init__(self, error_code: str, message: str, details=None) -> None:
        super().__init__(error_code, message, details)
