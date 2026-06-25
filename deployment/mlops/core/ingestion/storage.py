from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from urllib import error, parse, request

from mlops.config.settings import MLOpsSettings


class ReviewedDataStorage(ABC):
    @abstractmethod
    def list_manifest_keys(self) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def read_bytes(self, key: str) -> bytes:
        raise NotImplementedError


class LocalReviewedDataStorage(ReviewedDataStorage):
    def __init__(self, root: Path, manifest_index: str) -> None:
        self.root = root.resolve()
        self.manifest_index = manifest_index

    def list_manifest_keys(self) -> list[str]:
        index_path = self._resolve(self.manifest_index)
        if index_path.exists():
            return _manifest_keys_from_index(index_path.read_bytes(), str(index_path))
        return sorted(
            path.relative_to(self.root).as_posix()
            for path in self.root.rglob("manifest.json")
        )

    def read_bytes(self, key: str) -> bytes:
        path = self._resolve(key)
        if not path.is_file():
            raise FileNotFoundError(f"Reviewed-data object not found: {path}")
        return path.read_bytes()

    def _resolve(self, key: str) -> Path:
        candidate = (self.root / key.replace("\\", "/")).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise ValueError(f"Reviewed-data key escapes local root: {key}") from exc
        return candidate


class SupabaseReviewedDataStorage(ReviewedDataStorage):
    def __init__(
        self,
        *,
        base_url: str,
        service_role_key: str,
        bucket: str,
        prefix: str,
        manifest_index: str,
    ) -> None:
        if not base_url or not service_role_key:
            raise ValueError(
                "Supabase reviewed-data backend requires URL and service-role key."
            )
        self.base_url = base_url.rstrip("/")
        self.service_role_key = service_role_key
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.manifest_index = manifest_index.strip("/")

    def list_manifest_keys(self) -> list[str]:
        key = self._prefixed(self.manifest_index)
        raw = self._download(key)
        return _manifest_keys_from_index(raw, key)

    def read_bytes(self, key: str) -> bytes:
        return self._download(self._prefixed(key))

    def _prefixed(self, key: str) -> str:
        clean = key.replace("\\", "/").strip("/")
        if not self.prefix:
            return clean
        return f"{self.prefix}/{clean}"

    def _download(self, key: str) -> bytes:
        encoded_key = "/".join(parse.quote(part) for part in key.split("/"))
        encoded_bucket = parse.quote(self.bucket, safe="")
        url = (
            f"{self.base_url}/storage/v1/object/authenticated/"
            f"{encoded_bucket}/{encoded_key}"
        )
        req = request.Request(
            url,
            headers={
                "Authorization": f"Bearer {self.service_role_key}",
                "apikey": self.service_role_key,
            },
            method="GET",
        )
        try:
            with request.urlopen(req, timeout=60) as response:
                return response.read()
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", "ignore")
            raise RuntimeError(
                f"Supabase reviewed-data download failed for {key}: "
                f"HTTP {exc.code} {body}"
            ) from exc
        except error.URLError as exc:
            raise RuntimeError(
                f"Supabase reviewed-data download failed for {key}: {exc.reason}"
            ) from exc


def build_reviewed_data_storage() -> ReviewedDataStorage:
    backend = MLOpsSettings.REVIEWED_DATA_BACKEND
    if backend == "local":
        return LocalReviewedDataStorage(
            MLOpsSettings.REVIEWED_DATA_LOCAL_ROOT,
            MLOpsSettings.REVIEWED_DATA_MANIFEST_INDEX,
        )
    if backend == "supabase":
        return SupabaseReviewedDataStorage(
            base_url=MLOpsSettings.REVIEWED_DATA_SUPABASE_URL,
            service_role_key=MLOpsSettings.REVIEWED_DATA_SUPABASE_SERVICE_ROLE_KEY,
            bucket=MLOpsSettings.REVIEWED_DATA_SUPABASE_BUCKET,
            prefix=MLOpsSettings.REVIEWED_DATA_SUPABASE_PREFIX,
            manifest_index=MLOpsSettings.REVIEWED_DATA_MANIFEST_INDEX,
        )
    raise ValueError(f"Unsupported REVIEWED_DATA_BACKEND: {backend}")


def _manifest_keys_from_index(raw: bytes, source: str) -> list[str]:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid reviewed-data manifest index: {source}") from exc

    if payload.get("schema_version") != "1.0":
        raise ValueError(f"Unsupported manifest index schema_version: {source}")
    manifests = payload.get("manifests")
    if not isinstance(manifests, list):
        raise ValueError(f"Manifest index must contain a manifests list: {source}")

    keys = []
    for item in manifests:
        key = item.get("key") if isinstance(item, dict) else item
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"Manifest index contains an invalid key: {source}")
        clean = key.replace("\\", "/").strip("/")
        if ".." in clean.split("/"):
            raise ValueError(f"Manifest index contains an unsafe key: {source}")
        keys.append(clean)
    return sorted(set(keys))
