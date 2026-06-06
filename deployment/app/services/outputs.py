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
    """
    Save source, mask, roi, and overlay images.

    Files are grouped one folder per prediction, under a UTC day folder, e.g.
    ``predictions/2026-06-02/<uid>/source.png`` so the output stays ordered
    instead of a flat dump. Returns dict with the matching static URLs.
    """
    uid = prefix or uuid4().hex
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    rel_dir = f"{day}/{uid}"
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

    links = {
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
        )

    if AppConfig.PREDICTION_STORAGE_BACKEND != "local":
        raise ServiceError(
            "UNSUPPORTED_STORAGE_BACKEND",
            f"Unsupported prediction storage backend: {AppConfig.PREDICTION_STORAGE_BACKEND}",
        )

    return links


def _artifact_url(rel_dir: str, path: Path) -> str:
    return f"{AppConfig.PREDICTION_PUBLIC_BASE_URL}/{rel_dir}/{path.name}"


def _upload_prediction_files_to_supabase(
    *,
    rel_dir: str,
    files: dict[str, Path],
) -> dict[str, str]:
    if not AppConfig.SUPABASE_URL or not AppConfig.SUPABASE_SERVICE_ROLE_KEY:
        raise ServiceError(
            "SUPABASE_NOT_CONFIGURED",
            "Supabase storage is enabled but SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY is missing.",
        )

    links = {}
    uploaded_paths = []
    for response_key, file_path in files.items():
        object_path = f"predictions/{rel_dir}/{file_path.name}"
        try:
            _supabase_upload(object_path=object_path, file_path=file_path)
            uploaded_paths.append(object_path)
            links[response_key] = _supabase_signed_url(object_path=object_path)
        except Exception:
            _cleanup_supabase_objects(uploaded_paths)
            raise

    _cleanup_local_prediction_dir(files.values())
    return links


def _supabase_upload(*, object_path: str, file_path: Path) -> None:
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
    for object_path in object_paths:
        try:
            _supabase_delete(object_path=object_path)
        except ServiceError:
            pass


def _supabase_delete(*, object_path: str) -> None:
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
    parents = {path.parent for path in paths}
    for parent in parents:
        if _is_prediction_output_dir(parent):
            rmtree(parent, ignore_errors=True)


def _is_prediction_output_dir(path: Path) -> bool:
    try:
        path.relative_to(AppConfig.PREDICTION_DIR)
    except ValueError:
        return False
    return path != AppConfig.PREDICTION_DIR


def _supabase_headers(*, content_type: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {AppConfig.SUPABASE_SERVICE_ROLE_KEY}",
        "apikey": AppConfig.SUPABASE_SERVICE_ROLE_KEY,
        "Content-Type": content_type,
    }


def _open_supabase_request(req: request.Request, action: str) -> dict:
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
    if path.suffix.lower() in (".jpg", ".jpeg"):
        return "image/jpeg"
    if path.suffix.lower() == ".webp":
        return "image/webp"
    return "image/png"

def _save_mask(mask: np.ndarray, path: Path) -> None:
    mask_2d = np.squeeze(mask)
    if mask_2d.dtype != np.uint8:
        mask_2d = (mask_2d > 0).astype("uint8") * 255
    Image.fromarray(mask_2d, mode="L").save(path)

def _save_roi(roi_img: np.ndarray, path: Path) -> None:
    roi_img = np.squeeze(roi_img)
    if roi_img.dtype != np.uint8:
        roi_img = roi_img.astype("uint8")
    Image.fromarray(roi_img).save(path)

def _save_source_image(img: np.ndarray, path: Path) -> None:
    arr = img.astype("uint8")
    Image.fromarray(arr).save(path)

def _save_overlay_image_mask(
    img: np.ndarray, 
    mask: np.ndarray,
    path: Path) -> None:

    overlay = overlay_mask_on_image(img, mask)
    overlay.save(path)

