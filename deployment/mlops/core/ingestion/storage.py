"""Reviewed-data storage adapters for local files and Supabase Storage."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from urllib import error, parse, request

from mlops.config.settings import MLOpsSettings


class ReviewedDataStorage(ABC):
    """Abstract storage interface for reviewed-data manifests and objects.

    Storage adapters expose the same two operations regardless of whether
    reviewed data lives on disk or in Supabase Storage. Snapshot creation can
    therefore stay focused on validation and TFRecord writing.
    """

    @abstractmethod
    def list_manifest_keys(self) -> list[str]:
        """Return manifest keys available for ingestion.

        Returns:
            List of storage keys pointing to reviewed-data manifests.
        """
        raise NotImplementedError

    @abstractmethod
    def read_bytes(self, key: str) -> bytes:
        """Read raw bytes for an object key.

        Args:
            key: Storage object key.

        Returns:
            Raw object bytes.
        """
        raise NotImplementedError


class LocalReviewedDataStorage(ReviewedDataStorage):
    """Reviewed-data storage backed by a local filesystem directory.

    The local adapter is useful for development, tests, and offline retraining.
    It still enforces root-relative paths so test manifests cannot accidentally
    read files outside the reviewed-data folder.
    """

    def __init__(self, root: Path, manifest_index: str) -> None:
        """Store the local root and optional manifest index path.

        Args:
            root: Filesystem directory containing reviewed-data objects.
            manifest_index: Optional index JSON key under ``root``.
        """
        self.root = root.resolve()
        self.manifest_index = manifest_index

    def list_manifest_keys(self) -> list[str]:
        """List manifest keys from an index file or recursive discovery.

        Returns:
            Sorted manifest keys relative to the local root.
        """
        index_path = self._resolve(self.manifest_index)
        if index_path.exists():
            # A manifest index makes production snapshots deterministic by
            # explicitly listing which reviewed batches should be ingested.
            return _manifest_keys_from_index(index_path.read_bytes(), str(index_path))
        # Local development can omit the index and rely on recursive discovery
        # of manifest.json files under the reviewed-data root.
        return sorted(
            path.relative_to(self.root).as_posix()
            for path in self.root.rglob("manifest.json")
        )

    def read_bytes(self, key: str) -> bytes:
        """Read a reviewed-data object from the local root.

        Args:
            key: Root-relative object key.

        Returns:
            Raw file bytes.

        Raises:
            FileNotFoundError: If the resolved object does not exist.
        """
        path = self._resolve(key)
        if not path.is_file():
            raise FileNotFoundError(f"Reviewed-data object not found: {path}")
        # The caller validates content hashes after reading, so this adapter
        # returns raw bytes without attempting image/manifest interpretation.
        return path.read_bytes()

    def _resolve(self, key: str) -> Path:
        """Resolve a key under the local root and reject path traversal.

        Args:
            key: Root-relative reviewed-data key.

        Returns:
            Absolute resolved path.

        Raises:
            ValueError: If the key escapes the configured root.
        """
        candidate = (self.root / key.replace("\\", "/")).resolve()
        try:
            # Path traversal protection is important because reviewed manifests
            # may reference image and mask object keys supplied by external tools.
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise ValueError(f"Reviewed-data key escapes local root: {key}") from exc
        return candidate


class SupabaseReviewedDataStorage(ReviewedDataStorage):
    """Reviewed-data storage backed by Supabase authenticated object storage.

    Supabase support lets reviewed production samples be ingested without
    copying them into the repository. The service-role key is used only from
    trusted MLOps jobs, never from the public API frontend.
    """

    def __init__(
        self,
        *,
        base_url: str,
        service_role_key: str,
        bucket: str,
        prefix: str,
        manifest_index: str,
    ) -> None:
        """Configure Supabase endpoint, credentials, bucket, and prefix.

        Args:
            base_url: Supabase project URL.
            service_role_key: Service-role key for authenticated object reads.
            bucket: Storage bucket name.
            prefix: Optional object prefix under the bucket.
            manifest_index: Index object listing manifest keys.

        Raises:
            ValueError: If required Supabase credentials are missing.
        """
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
        """Load manifest keys from the configured Supabase index object.

        Returns:
            Sorted manifest keys parsed from the index object.
        """
        key = self._prefixed(self.manifest_index)
        raw = self._download(key)
        return _manifest_keys_from_index(raw, key)

    def read_bytes(self, key: str) -> bytes:
        """Download a reviewed-data object from Supabase.

        Args:
            key: Reviewed-data object key relative to the configured prefix.

        Returns:
            Raw object bytes.
        """
        return self._download(self._prefixed(key))

    def _prefixed(self, key: str) -> str:
        """Apply the configured object prefix to a storage key.

        Args:
            key: Prefix-relative object key.

        Returns:
            Bucket-relative object key.
        """
        clean = key.replace("\\", "/").strip("/")
        if not self.prefix:
            return clean
        # Prefixing lets multiple environments share one bucket while keeping
        # staging and production reviewed-data sets isolated.
        return f"{self.prefix}/{clean}"

    def _download(self, key: str) -> bytes:
        """Download one object through the Supabase Storage API.

        Args:
            key: Bucket-relative object key.

        Returns:
            Raw object bytes.

        Raises:
            RuntimeError: If Supabase returns an HTTP or network error.
        """
        encoded_key = "/".join(parse.quote(part) for part in key.split("/"))
        encoded_bucket = parse.quote(self.bucket, safe="")
        # Use the authenticated object endpoint so private reviewed images stay
        # unavailable to anonymous clients.
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
            # Keep timeout finite so Airflow tasks fail and retry instead of
            # hanging forever on a network stall.
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
    """Create the reviewed-data storage adapter selected by environment.

    Returns:
        Local or Supabase reviewed-data storage adapter.

    Raises:
        ValueError: If ``REVIEWED_DATA_BACKEND`` is unsupported.
    """
    backend = MLOpsSettings.REVIEWED_DATA_BACKEND
    if backend == "local":
        # Local storage is the default because it works for tests, notebooks,
        # and offline retraining without cloud credentials.
        return LocalReviewedDataStorage(
            MLOpsSettings.REVIEWED_DATA_LOCAL_ROOT,
            MLOpsSettings.REVIEWED_DATA_MANIFEST_INDEX,
        )
    if backend == "supabase":
        # Supabase storage is intended for trusted scheduled jobs that ingest
        # reviewed production samples into retraining snapshots.
        return SupabaseReviewedDataStorage(
            base_url=MLOpsSettings.REVIEWED_DATA_SUPABASE_URL,
            service_role_key=MLOpsSettings.REVIEWED_DATA_SUPABASE_SERVICE_ROLE_KEY,
            bucket=MLOpsSettings.REVIEWED_DATA_SUPABASE_BUCKET,
            prefix=MLOpsSettings.REVIEWED_DATA_SUPABASE_PREFIX,
            manifest_index=MLOpsSettings.REVIEWED_DATA_MANIFEST_INDEX,
        )
    raise ValueError(f"Unsupported REVIEWED_DATA_BACKEND: {backend}")


def _manifest_keys_from_index(raw: bytes, source: str) -> list[str]:
    """Parse and validate a manifest index JSON document.

    Args:
        raw: Raw JSON index bytes.
        source: Source key/path used in validation errors.

    Returns:
        Sorted unique manifest keys.

    Raises:
        ValueError: If the index schema, list, or keys are invalid.
    """
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
    # Sort and deduplicate so repeated index entries do not duplicate training
    # examples or make monthly snapshots nondeterministic.
    return sorted(set(keys))
