from pathlib import Path
from typing import IO, Optional, Tuple, Union
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import base64
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
import tensorflow as tf

from app.utils.errors import ImageLoadError


def load_image(
    image_source: Union[str, Path, IO[bytes], bytes],
    target_size: Optional[Tuple[int, int]],
) -> tf.Tensor:
    """
    Load an image from disk with validation and return a 4D tensor.

    Returns:
        tf.Tensor: shape (1, H, W, 3), dtype float32, value range [0, 255]
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

    return tf.expand_dims(tf.convert_to_tensor(arr), axis=0)


def load_bytes_from_source(
    *,
    image_path: Optional[str] = None,
    image_url: Optional[str] = None,
    image_base64: Optional[str] = None,
    upload_file: Optional[IO[bytes]] = None,
) -> bytes:
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
    text = str(value)
    parsed = urlparse(text)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _fetch_url_bytes(url: str) -> bytes:
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
    if "," in data and "base64" in data[:50].lower():
        data = data.split(",", 1)[1]
    try:
        return base64.b64decode(data)
    except Exception as exc:
        raise ImageLoadError(
            "invalid_base64",
            "Provided base64 string is not valid.",
        ) from exc
