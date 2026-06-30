"""Pydantic request schemas for prediction endpoints."""

from pydantic import BaseModel, Field, model_validator
from typing import Optional


class PredictRequest(BaseModel):
    """JSON prediction request containing exactly one image source.

    Attributes:
        image_path: Optional local path available to the API process.
        image_base64: Optional base64-encoded image payload.
        image_url: Optional public URL that the API can fetch.

    Notes:
        Multipart uploads use the separate ``/predict/upload`` endpoint. This
        schema is intentionally JSON-only so validation remains explicit.
    """

    image_path: Optional[str] = Field(
        None, description="Local Path to Image"
    )
    image_base64: Optional[str] = Field(
        None, description="Base64-encoded Image"
    )
    image_url: Optional[str] = Field(
        None, description="Public Image URL"
    )

    @model_validator(mode="after")
    def validate_single_source(self):
        """Ensure clients provide one and only one JSON image source.

        Returns:
            The validated request instance.

        Raises:
            ValueError: If zero or multiple JSON image sources are provided.
        """
        # Keeping the request mutually exclusive prevents accidental precedence
        # bugs where a client sends both a URL and base64 payload.
        provided = [
            v is not None and v != ""
            for v in (self.image_path, self.image_base64, self.image_url)
        ]
        if sum(provided) == 0:
            raise ValueError(
                "No image input provided. Provide exactly one of: image_path, image_base64, image_url."
            )
        if sum(provided) > 1:
            raise ValueError(
                "Multiple image inputs provided. Provide exactly one of: image_path, image_base64, image_url."
            )
        return self
