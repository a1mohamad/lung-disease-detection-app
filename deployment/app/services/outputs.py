"""Prediction artifact persistence for local and Supabase-backed storage."""

from app.configs.config import AppConfig
from app.utils.visualization import overlay_mask_on_image
from app.utils.errors import ServiceError

import numpy as np
import json
from datetime import datetime, timezone
from pathlib import Path
from shutil import rmtree
from PIL import Image
from uuid import uuid4
from urllib import error, parse, request

def save_output_images(
    *,
    source_image: np.ndarray,
    mask: np.ndarray,
    roi_img: np.ndarray,
    prefix: str | None = None,
) -> dict:
    """Persist the visual artifacts produced by one prediction.

    The function writes four review artifacts: the decoded source image, the
    predicted lung mask, the cropped ROI used for subtype inference, and a mask
    overlay for quick human inspection. Local storage returns static paths and
    URLs; Supabase storage uploads the same files and returns signed URLs.

    Args:
        source_image: Original decoded RGB image.
        mask: Predicted lung mask.
        roi_img: Cropped lung-region image.
        prefix: Optional stable prediction identifier. A UUID is generated when
            omitted.

    Returns:
        Mapping of artifact path and URL fields suitable for API responses and
        prediction logging.

    Raises:
        ServiceError: If the configured storage backend is unsupported or the
        remote storage operation fails.
    """
    uid = prefix or uuid4().hex
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    rel_dir = f"{day}/{uid}"
    # Use UTC day folders plus one prediction folder to keep artifacts readable
    # in object storage and avoid a single unbounded flat directory.
    out_dir = AppConfig.PREDICTION_DIR / day / uid
    out_dir.mkdir(parents=True, exist_ok=True)

    source_path = out_dir / "source.png"
    mask_path = out_dir / "mask.png"
    roi_path = out_dir / "roi.png"
    overlay_path = out_dir / "overlay.png"

    _save_source_image(source_image, source_path)
    _save_mask(mask, mask_path)
    _save_roi(roi_img, roi_path)
    _save_overlay_image_mask(source_image, mask, overlay_path)

    artifacts = {
        "source_path": _artifact_path(rel_dir, source_path),
        "mask_path": _artifact_path(rel_dir, mask_path),
        "roi_path": _artifact_path(rel_dir, roi_path),
        "overlay_path": _artifact_path(rel_dir, overlay_path),
        "source_url": _artifact_url(rel_dir, source_path),
        "mask_url": _artifact_url(rel_dir, mask_path),
        "roi_url": _artifact_url(rel_dir, roi_path),
        "overlay_url": _artifact_url(rel_dir, overlay_path),
    }

    if AppConfig.PREDICTION_STORAGE_BACKEND == "supabase":
        return _upload_prediction_files_to_supabase(
            rel_dir=rel_dir,
            files={
                "source_url": source_path,
                "mask_url": mask_path,
                "roi_url": roi_path,
                "overlay_url": overlay_path,
            },
            paths={
                "source_path": artifacts["source_path"],
                "mask_path": artifacts["mask_path"],
                "roi_path": artifacts["roi_path"],
                "overlay_path": artifacts["overlay_path"],
            },
        )

    if AppConfig.PREDICTION_STORAGE_BACKEND != "local":
        raise ServiceError(
            "UNSUPPORTED_STORAGE_BACKEND",
            f"Unsupported prediction storage backend: {AppConfig.PREDICTION_STORAGE_BACKEND}",
        )

    return artifacts


def _artifact_path(rel_dir: str, path: Path) -> str:
    """Return the storage-relative artifact path exposed in API responses.

    Args:
        rel_dir: Prediction-relative directory, usually ``YYYY-MM-DD/<uuid>``.
        path: Local artifact file path.

    Returns:
        Stable storage path used by logs and review tooling.
    """
    return f"predictions/{rel_dir}/{path.name}"


def _artifact_url(rel_dir: str, path: Path) -> str:
    """Return the public URL for a locally served prediction artifact.

    Args:
        rel_dir: Prediction-relative directory, usually ``YYYY-MM-DD/<uuid>``.
        path: Local artifact file path.

    Returns:
        URL served by the FastAPI static files mount.
    """
    return f"{AppConfig.PREDICTION_PUBLIC_BASE_URL}/{rel_dir}/{path.name}"


def _upload_prediction_files_to_supabase(
    *,
    rel_dir: str,
    files: dict[str, Path],
    paths: dict[str, str],
) -> dict[str, str]:
    """Upload generated artifacts to Supabase and return signed URLs.

    Local files are treated as temporary staging artifacts. They are removed
    only after all uploads succeed, so a failed upload leaves enough local
    evidence for debugging while avoiding partial remote records.
    """
    if not AppConfig.SUPABASE_URL or not AppConfig.SUPABASE_SERVICE_ROLE_KEY:
        raise ServiceError(
            "SUPABASE_NOT_CONFIGURED",
            "Supabase storage is enabled but SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY is missing.",
        )

    artifacts = dict(paths)
    uploaded_paths = []
    for response_key, file_path in files.items():
        object_path = f"predictions/{rel_dir}/{file_path.name}"
        try:
            _supabase_upload(object_path=object_path, file_path=file_path)
            uploaded_paths.append(object_path)
            artifacts[response_key] = _supabase_signed_url(object_path=object_path)
        except Exception:
            _cleanup_supabase_objects(uploaded_paths)
            raise

    _cleanup_local_prediction_dir(files.values())
    return artifacts


def _supabase_upload(*, object_path: str, file_path: Path) -> None:
    """Upload one local artifact file to Supabase Storage.

    Args:
        object_path: Destination object path inside the configured bucket.
        file_path: Local artifact file to upload.

    Raises:
        ServiceError: If Supabase rejects the upload or transport fails.
    """
    encoded_path = "/".join(parse.quote(part) for part in object_path.split("/"))
    url = (
        f"{AppConfig.SUPABASE_URL}/storage/v1/object/"
        f"{AppConfig.SUPABASE_STORAGE_BUCKET}/{encoded_path}"
    )
    headers = _supabase_headers(content_type=_content_type(file_path))
    headers["x-upsert"] = "true"

    req = request.Request(
        url,
        data=file_path.read_bytes(),
        headers=headers,
        method="POST",
    )
    _open_supabase_request(req, "upload prediction artifact")


def _supabase_signed_url(*, object_path: str) -> str:
    """Create a signed public URL for one Supabase object.

    Args:
        object_path: Bucket object path that should be exposed temporarily.

    Returns:
        Absolute signed URL suitable for API responses.

    Raises:
        ServiceError: If Supabase does not return a signed URL.
    """
    encoded_path = "/".join(parse.quote(part) for part in object_path.split("/"))
    url = (
        f"{AppConfig.SUPABASE_URL}/storage/v1/object/sign/"
        f"{AppConfig.SUPABASE_STORAGE_BUCKET}/{encoded_path}"
    )
    body = json.dumps(
        {"expiresIn": AppConfig.SUPABASE_SIGNED_URL_EXPIRES_SECONDS}
    ).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        headers=_supabase_headers(content_type="application/json"),
        method="POST",
    )
    payload = _open_supabase_request(req, "create signed prediction URL")
    signed_url = payload.get("signedURL") or payload.get("signedUrl")
    if not signed_url:
        raise ServiceError(
            "SUPABASE_SIGNED_URL_FAILED",
            "Supabase did not return a signed URL.",
            details={"object_path": object_path},
        )
    if signed_url.startswith("http"):
        return signed_url
    if signed_url.startswith("/object/"):
        return f"{AppConfig.SUPABASE_URL}/storage/v1{signed_url}"
    return f"{AppConfig.SUPABASE_URL}{signed_url}"


def _cleanup_supabase_objects(object_paths: list[str]) -> None:
    """Best-effort cleanup for partially uploaded Supabase artifacts.

    Args:
        object_paths: Object paths uploaded before a later operation failed.

    Notes:
        Cleanup errors are intentionally swallowed so the original upload error
        remains the visible failure for callers.
    """
    for object_path in object_paths:
        try:
            _supabase_delete(object_path=object_path)
        except ServiceError:
            pass


def _supabase_delete(*, object_path: str) -> None:
    """Delete one Supabase object path.

    Args:
        object_path: Bucket object path to remove.

    Raises:
        ServiceError: If Supabase rejects the delete request.
    """
    url = (
        f"{AppConfig.SUPABASE_URL}/storage/v1/object/"
        f"{AppConfig.SUPABASE_STORAGE_BUCKET}"
    )
    body = json.dumps({"prefixes": [object_path]}).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        headers=_supabase_headers(content_type="application/json"),
        method="DELETE",
    )
    _open_supabase_request(req, "delete partial prediction artifact")


def _cleanup_local_prediction_dir(paths: object) -> None:
    """Remove local temporary artifact directories after remote upload.

    Args:
        paths: Iterable of local artifact paths that were uploaded remotely.
    """
    parents = {path.parent for path in paths}
    for parent in parents:
        if _is_prediction_output_dir(parent):
            rmtree(parent, ignore_errors=True)


def _is_prediction_output_dir(path: Path) -> bool:
    """Return whether a path is a removable prediction output subdirectory.

    Args:
        path: Candidate directory path.

    Returns:
        True only for directories under ``AppConfig.PREDICTION_DIR`` that are
        not the prediction root itself.
    """
    try:
        path.relative_to(AppConfig.PREDICTION_DIR)
    except ValueError:
        return False
    return path != AppConfig.PREDICTION_DIR


def _supabase_headers(*, content_type: str) -> dict[str, str]:
    """Build authenticated Supabase Storage request headers.

    Args:
        content_type: MIME type for the upload or JSON request body.

    Returns:
        Header dictionary containing bearer auth, API key, and content type.
    """
    return {
        "Authorization": f"Bearer {AppConfig.SUPABASE_SERVICE_ROLE_KEY}",
        "apikey": AppConfig.SUPABASE_SERVICE_ROLE_KEY,
        "Content-Type": content_type,
    }


def _open_supabase_request(req: request.Request, action: str) -> dict:
    """Execute a Supabase request and normalize transport errors.

    Args:
        req: Fully prepared urllib request.
        action: Human-readable action used in error messages.

    Returns:
        Parsed JSON response body, or an empty dictionary for empty responses.

    Raises:
        ServiceError: If Supabase returns an HTTP error or the request cannot be
        completed.
    """
    try:
        with request.urlopen(req, timeout=30) as response:
            data = response.read()
    except error.HTTPError as exc:
        details = {"status": exc.code, "body": exc.read().decode("utf-8", "ignore")}
        raise ServiceError("SUPABASE_STORAGE_ERROR", f"Failed to {action}.", details) from exc
    except error.URLError as exc:
        raise ServiceError(
            "SUPABASE_STORAGE_ERROR",
            f"Failed to {action}: {exc.reason}",
        ) from exc

    if not data:
        return {}
    return json.loads(data.decode("utf-8"))


def _content_type(path: Path) -> str:
    """Infer a storage content type from an image file extension.

    Args:
        path: Local image artifact path.

    Returns:
        MIME type string understood by Supabase Storage.
    """
    if path.suffix.lower() in (".jpg", ".jpeg"):
        return "image/jpeg"
    if path.suffix.lower() == ".webp":
        return "image/webp"
    return "image/png"

def _save_mask(mask: np.ndarray, path: Path) -> None:
    """Save a predicted mask as an 8-bit grayscale PNG.

    Args:
        mask: Predicted mask array, possibly batched or floating point.
        path: Destination PNG path.
    """
    mask_2d = np.squeeze(mask)
    if mask_2d.dtype != np.uint8:
        mask_2d = (mask_2d > 0).astype("uint8") * 255
    Image.fromarray(mask_2d, mode="L").save(path)

def _save_roi(roi_img: np.ndarray, path: Path) -> None:
    """Save a cropped lung ROI image.

    Args:
        roi_img: Cropped lung-region image, possibly batched.
        path: Destination image path.
    """
    roi_img = np.squeeze(roi_img)
    if roi_img.dtype != np.uint8:
        roi_img = roi_img.astype("uint8")
    Image.fromarray(roi_img).save(path)

def _save_source_image(img: np.ndarray, path: Path) -> None:
    """Save the original decoded image used for inference.

    Args:
        img: RGB source image array.
        path: Destination image path.
    """
    arr = img.astype("uint8")
    Image.fromarray(arr).save(path)

def _save_overlay_image_mask(
    img: np.ndarray, 
    mask: np.ndarray,
    path: Path) -> None:
    """Save a source image with the predicted mask overlaid.

    Args:
        img: RGB source image array.
        mask: Predicted lung mask.
        path: Destination overlay image path.
    """

    overlay = overlay_mask_on_image(img, mask)
    overlay.save(path)

