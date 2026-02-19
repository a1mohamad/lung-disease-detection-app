from pydantic import BaseModel, Field, model_validator
from typing import Optional


class PredictRequest(BaseModel):
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
