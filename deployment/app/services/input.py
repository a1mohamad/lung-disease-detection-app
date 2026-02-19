from __future__ import annotations

from app.preprocessing.image import load_bytes_from_source
from app.utils.errors import InputError

from io import BytesIO
import numpy as np
from PIL import Image
from typing import IO, Optional, Tuple, Union


def _select_image_source(
    *,
    image_path: Optional[str],
    image_url: Optional[str],
    image_base64: Optional[str],
    upload_file: Optional[IO[bytes]],
) -> Tuple[Union[str, IO[bytes], bytes], Optional[bytes]]:
    provided = [
        v is not None and v != ""
        for v in (image_path, image_url, image_base64, upload_file)
    ]

    if sum(provided) == 0:
        raise InputError(
            "NO_IMAGE_PROVIDED",
            "No image input provided. Please provide one of: image_path, image_url, image_base64, upload_file.",
        )
    if sum(provided) > 1:
        raise InputError(
            "MULTIPLE_IMAGE_PROVIDED",
            "Multiple image inputs provided. Please provide only one of: image_path, image_url, image_base64, upload_file.",
        )

    try:
        raw = load_bytes_from_source(
            image_path=image_path,
            image_url=image_url,
            image_base64=image_base64,
            upload_file=upload_file,
        )
    except InputError:
        raise
    except Exception as exc:
        raise InputError(
            "IMAGE_LOAD_FAILED",
            "Failed to load image from the provided input.",
        ) from exc

    if image_path:
        return image_path, raw
    if image_url:
        return image_url, raw
    if image_base64:
        return raw, raw
    if upload_file:
        return upload_file, raw

    raise InputError("INVALID_INPUT", "No valid input found")
    
def _bytes_to_np(data: bytes) -> np.ndarray:
    img = Image.open(BytesIO(data)).convert("RGB")
    return np.array(img)
