"""Image loading, URL fetching, base64 decoding, and shape normalization."""

from pathlib import Path
from typing import IO, Optional, Tuple, Union
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import base64
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from app.utils.errors import ImageLoadError


def load_image(
    image_source: Union[str, Path, IO[bytes], bytes],
    target_size: Optional[Tuple[int, int]],
) -> np.ndarray:
    """Load, validate, orient, resize, and batch an image for inference.

    Args:
        image_source: Local path, URL, raw bytes, or file-like upload stream.
        target_size: Optional output size in ``(width, height)`` order expected
            by PIL resizing. The project passes ``AppConfig.IMAGE_SIZE``.

    Returns:
        Float32 RGB image batch with shape ``(1, H, W, 3)`` and value range
        ``[0, 255]``.

    Raises:
        ImageLoadError: If the source is missing, target size is invalid, or the
        image cannot be decoded by Pillow.
    """
    path: Optional[Path] = None
    if isinstance(image_source, (str, Path)) and not _is_url(image_source):
        path = Path(image_source)
        if not path.exists():
            raise ImageLoadError("file_not_found", f"Image not found: {path}")

    if target_size is not None:
        if (
            not isinstance(target_size, tuple)
            or len(target_size) != 2
            or not all(isinstance(x, int) and x > 0 for x in target_size)
        ):
            raise ImageLoadError(
                "invalid_target_size",
                "target_size must be a tuple of two positive integers.",
            )

    try:
        img = _open_image(image_source)
        # Respect EXIF orientation so mobile/clinical uploads are not silently
        # rotated compared with the saved artifact or model input.
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
        if target_size is not None:
            img = img.resize(target_size, Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32)
    except ImageLoadError:
        raise
    except (UnidentifiedImageError, OSError) as exc:
        label = path if path is not None else "uploaded image"
        raise ImageLoadError(
            "invalid_image",
            f"Unsupported or corrupted image file: {label}",
        ) from exc
    finally:
        try:
            img.close()
        except Exception:
            pass

    return np.expand_dims(arr, axis=0)


def load_bytes_from_source(
    *,
    image_path: Optional[str] = None,
    image_url: Optional[str] = None,
    image_base64: Optional[str] = None,
    upload_file: Optional[IO[bytes]] = None,
) -> bytes:
    """Load raw image bytes from exactly one supported source type.

    Args:
        image_path: Optional local filesystem path.
        image_url: Optional HTTP/HTTPS image URL.
        image_base64: Optional plain base64 string or data URI.
        upload_file: Optional file-like object from multipart upload.

    Returns:
        Raw encoded image bytes.

    Raises:
        ImageLoadError: If no source is present or the selected source cannot be
        loaded.
    """
    if image_path:
        path = Path(image_path)
        if not path.exists():
            raise ImageLoadError("file_not_found", f"Image not found: {path}")
        return path.read_bytes()

    if image_url:
        return _fetch_url_bytes(image_url)

    if image_base64:
        return _decode_base64_image(image_base64)

    if upload_file:
        return upload_file.read()

    raise ImageLoadError("invalid_source", "No valid image source provided.")


def _open_image(image_source: Union[str, Path, IO[bytes], bytes]) -> Image.Image:
    """Open a PIL image from a local path, URL, raw bytes, or file-like object.

    Args:
        image_source: Supported image source value.

    Returns:
        Open Pillow image object. The caller owns closing it.

    Raises:
        ImageLoadError: If the source type is unsupported or DICOM is supplied.
    """
    if isinstance(image_source, (str, Path)) and _is_url(image_source):
        data = _fetch_url_bytes(str(image_source))
        from io import BytesIO
        return Image.open(BytesIO(data))

    if isinstance(image_source, (str, Path)):
        path = Path(image_source)
        if path.suffix.lower() == ".dcm":
            raise ImageLoadError(
                "dicom_not_supported",
                "DICOM files are not supported for this model. Please upload PNG/JPG/etc.",
            )
        return Image.open(path)

    if isinstance(image_source, (bytes, bytearray)):
        from io import BytesIO
        return Image.open(BytesIO(image_source))

    if hasattr(image_source, "read"):
        return Image.open(image_source)

    raise ImageLoadError(
        "invalid_source",
        "image_source must be a file path, bytes, or a file-like object.",
    )


def _is_url(value: Union[str, Path]) -> bool:
    """Return whether a value is an HTTP or HTTPS URL.

    Args:
        value: Candidate source string or path.

    Returns:
        True when the value has an HTTP(S) scheme and network location.
    """
    text = str(value)
    parsed = urlparse(text)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _fetch_url_bytes(url: str) -> bytes:
    """Fetch image bytes from a remote URL with a short timeout.

    Args:
        url: HTTP or HTTPS image URL.

    Returns:
        Raw response bytes.

    Raises:
        ImageLoadError: If the request fails or times out.
    """
    try:
        req = Request(url, headers={"User-Agent": "LungDetection/1.0"})
        with urlopen(req, timeout=10) as resp:
            return resp.read()
    except Exception as exc:
        raise ImageLoadError(
            "url_fetch_failed",
            f"Failed to fetch image URL: {url}",
        ) from exc


def _decode_base64_image(data: str) -> bytes:
    """Decode a plain or data-URI base64 image payload.

    Args:
        data: Base64 payload, optionally prefixed as a data URI.

    Returns:
        Decoded image bytes.

    Raises:
        ImageLoadError: If the payload is not valid base64.
    """
    # Browser clients often send data URIs; strip the header before decoding so
    # the API accepts both browser-native and raw API-client formats.
    if "," in data and "base64" in data[:50].lower():
        data = data.split(",", 1)[1]
    try:
        return base64.b64decode(data)
    except Exception as exc:
        raise ImageLoadError(
            "invalid_base64",
            "Provided base64 string is not valid.",
        ) from exc
