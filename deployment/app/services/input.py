"""Input source selection and raw image byte conversion helpers."""

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
    """Resolve the single image source accepted by one prediction request.

    Args:
        image_path: Local filesystem path supplied through JSON.
        image_url: Remote image URL supplied through JSON.
        image_base64: Base64-encoded image body supplied through JSON.
        upload_file: Multipart upload stream supplied by FastAPI.

    Returns:
        A tuple containing the original source identifier and decoded raw image
        bytes. Base64 inputs use the bytes as their source identifier because
        there is no stable external path to report.

    Raises:
        InputError: If no source is provided, multiple sources are provided, or
        the selected source cannot be loaded.
    """
    # Exactly one source keeps request semantics auditable and prevents subtle
    # precedence bugs when callers accidentally send two image fields.
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
    """Decode raw image bytes into an RGB NumPy array for artifact generation.

    Args:
        data: Encoded image bytes loaded from the selected request source.

    Returns:
        RGB NumPy array used when saving source and overlay artifacts.
    """
    img = Image.open(BytesIO(data)).convert("RGB")
    return np.array(img)
