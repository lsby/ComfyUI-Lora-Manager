"""Durable snapshots of model-site metadata and media beside model files."""

from __future__ import annotations

import asyncio
import hashlib
import html
import json
import logging
import os
import shutil
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Iterable, Optional
from urllib.parse import urlparse

from .downloader import get_downloader
from .settings_manager import get_settings_manager
from ..utils.example_images_paths import get_model_folder


logger = logging.getLogger(__name__)

ARCHIVE_SCHEMA_VERSION = 1
_MEDIA_EXTENSIONS = {
    ".apng", ".avif", ".gif", ".jpeg", ".jpg", ".m4v", ".mov",
    ".mp4", ".png", ".webm", ".webp",
}
_REMOTE_FAILURE_THRESHOLD = 3

CancelCheck = Callable[[], bool]
ArchiveProgressCallback = Callable[[Dict[str, Any]], Awaitable[None]]


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
    os.replace(temp_path, path)


def _safe_extension(url: str, media_type: str) -> str:
    extension = Path(urlparse(url).path).suffix.lower()
    if extension in _MEDIA_EXTENSIONS:
        return extension
    return ".mp4" if media_type.lower() == "video" else ".webp"


class ModelArchiveService:
    """Create append-only metadata snapshots and repair their local media."""

    def __init__(self, settings_manager=None) -> None:
        self._settings = settings_manager or get_settings_manager()
        self._locks: Dict[str, asyncio.Lock] = {}

    @property
    def enabled(self) -> bool:
        return bool(self._settings.get("model_archive_enabled", False))

    @staticmethod
    def _source_payload(metadata: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "civitai": metadata.get("civitai") or {},
            "modelDescription": metadata.get("modelDescription") or "",
            "tags": metadata.get("tags") or [],
            "usage_tips": metadata.get("usage_tips") or "{}",
            "source_platform": metadata.get("source_platform"),
            "source_url": metadata.get("source_url"),
        }

    @classmethod
    def _content_digest(cls, metadata: Dict[str, Any]) -> str:
        encoded = json.dumps(
            cls._source_payload(metadata),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _archive_key(metadata: Dict[str, Any], file_path: str) -> str:
        sha256 = str(metadata.get("sha256") or "").strip().lower()
        if len(sha256) == 64 and all(char in "0123456789abcdef" for char in sha256):
            return sha256
        civitai = metadata.get("civitai") or {}
        version_id = civitai.get("id") if isinstance(civitai, dict) else None
        if version_id is not None:
            return f"version-{version_id}"
        normalized = os.path.abspath(file_path).replace("\\", "/").lower()
        return f"path-{hashlib.sha256(normalized.encode('utf-8')).hexdigest()}"

    def _model_dir(self, metadata: Dict[str, Any], file_path: str) -> Path:
        return get_model_archive_dir(file_path)

    @staticmethod
    def _source_url(metadata: Dict[str, Any]) -> str:
        explicit = metadata.get("source_url")
        if isinstance(explicit, str) and explicit.strip():
            return explicit.strip()
        civitai = metadata.get("civitai") or {}
        if not isinstance(civitai, dict):
            return ""
        model_info = civitai.get("model") or {}
        model_id = civitai.get("modelId") or (
            model_info.get("id") if isinstance(model_info, dict) else None
        )
        version_id = civitai.get("id")
        if not model_id:
            return ""
        url = f"https://civitai.com/models/{model_id}"
        return f"{url}?modelVersionId={version_id}" if version_id else url

    async def archive_model(
        self,
        file_path: str,
        metadata: Dict[str, Any],
        *,
        download_remote_media: bool = False,
        cancel_check: Optional[CancelCheck] = None,
    ) -> Dict[str, Any]:
        """Archive one model without allowing archive failures to alter its sidecar."""

        if not self.enabled or not isinstance(metadata, dict):
            return {"archived": False, "reason": "disabled"}
        if not file_path:
            return {"archived": False, "reason": "missing_file_path"}
        source_payload = self._source_payload(metadata)
        if not source_payload.get("civitai") and not source_payload.get("modelDescription"):
            return {"archived": False, "reason": "no_source_metadata"}

        key = self._archive_key(metadata, file_path)
        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            try:
                return await self._archive_locked(
                    file_path,
                    metadata,
                    download_remote_media=download_remote_media,
                    cancel_check=cancel_check,
                )
            except Exception as exc:  # pragma: no cover - defensive isolation
                logger.warning("Failed to archive metadata for %s: %s", file_path, exc, exc_info=True)
                return {"archived": False, "reason": "error", "error": str(exc)}

    async def _archive_locked(
        self,
        file_path: str,
        metadata: Dict[str, Any],
        *,
        download_remote_media: bool,
        cancel_check: Optional[CancelCheck],
    ) -> Dict[str, Any]:
        model_dir = self._model_dir(metadata, file_path)
        snapshots_dir = model_dir / "snapshots"
        media_dir = model_dir / "media"
        snapshots_dir.mkdir(parents=True, exist_ok=True)
        media_dir.mkdir(parents=True, exist_ok=True)

        manifest_path = model_dir / "manifest.json"
        manifest = self._load_manifest(manifest_path)
        now = datetime.now(timezone.utc).isoformat()
        digest = self._content_digest(metadata)
        snapshots = manifest.setdefault("snapshots", [])
        existing = next(
            (item for item in snapshots if isinstance(item, dict) and item.get("digest") == digest),
            None,
        )
        snapshot_created = False
        if existing is None:
            snapshot_name = f"{_utc_timestamp()}-{digest[:12]}.json"
            snapshot_payload = {
                "schema_version": ARCHIVE_SCHEMA_VERSION,
                "archived_at": now,
                "source_url": self._source_url(metadata),
                "model_file": {
                    "file_name": metadata.get("file_name") or Path(file_path).stem,
                    "file_path": str(file_path).replace("\\", "/"),
                    "sha256": metadata.get("sha256") or "",
                },
                "metadata": metadata,
            }
            _atomic_write_json(snapshots_dir / snapshot_name, snapshot_payload)
            existing = {
                "digest": digest,
                "archived_at": now,
                "path": f"snapshots/{snapshot_name}",
            }
            snapshots.append(existing)
            snapshot_created = True

        civitai = metadata.get("civitai") or {}
        manifest.update(
            {
                "schema_version": ARCHIVE_SCHEMA_VERSION,
                "archive_key": self._archive_key(metadata, file_path),
                "model_name": metadata.get("model_name") or metadata.get("file_name") or Path(file_path).stem,
                "sha256": metadata.get("sha256") or "",
                "model_id": civitai.get("modelId") if isinstance(civitai, dict) else None,
                "model_version_id": civitai.get("id") if isinstance(civitai, dict) else None,
                "source_url": self._source_url(metadata),
                "first_archived_at": manifest.get("first_archived_at") or now,
                "last_archived_at": now,
                "latest_snapshot": existing["path"],
            }
        )

        self._write_description(model_dir / "description.html", metadata)
        copied = self._copy_local_media(metadata, media_dir, manifest)
        downloaded = 0
        failed = 0
        cancelled = False
        network_failed = False
        if download_remote_media and self._settings.get("model_archive_download_media", True):
            downloaded, failed, cancelled, network_failed = await self._download_remote_media(
                metadata,
                media_dir,
                manifest,
                cancel_check=cancel_check,
            )
        _atomic_write_json(manifest_path, manifest)
        return {
            "archived": True,
            "archive_dir": str(model_dir),
            "snapshot_created": snapshot_created,
            "local_media_copied": copied,
            "remote_media_downloaded": downloaded,
            "remote_media_failed": failed,
            "remote_media_cancelled": cancelled,
            "remote_media_network_failed": network_failed,
        }

    @staticmethod
    def _load_manifest(path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {"schema_version": ARCHIVE_SCHEMA_VERSION, "snapshots": [], "media": []}
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, dict):
                payload.setdefault("snapshots", [])
                payload.setdefault("media", [])
                return payload
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Rebuilding invalid model archive manifest %s: %s", path, exc)
        return {"schema_version": ARCHIVE_SCHEMA_VERSION, "snapshots": [], "media": []}

    @staticmethod
    def _write_description(path: Path, metadata: Dict[str, Any]) -> None:
        civitai = metadata.get("civitai") or {}
        model_description = metadata.get("modelDescription") or ""
        version_description = (
            civitai.get("description") or "" if isinstance(civitai, dict) else ""
        )
        title = metadata.get("model_name") or metadata.get("file_name") or "Model archive"
        safe_title = html.escape(str(title))
        document = (
            "<!doctype html><html><head><meta charset=\"utf-8\">"
            "<meta http-equiv=\"Content-Security-Policy\" "
            "content=\"default-src 'none'; img-src data:; style-src 'unsafe-inline'\">"
            f"<title>{safe_title}</title><style>body{{font:16px/1.55 system-ui;max-width:960px;"
            "margin:2rem auto;padding:0 1rem;color:#ddd;background:#181818}"
            "a{color:#72b7ff}img{max-width:100%}hr{border-color:#444}</style></head><body>"
            f"<h1>{safe_title}</h1><section>{model_description}</section>"
            f"<hr><section>{version_description}</section></body></html>"
        )
        temp_path = path.with_name(f"{path.name}.tmp")
        temp_path.write_text(document, encoding="utf-8")
        os.replace(temp_path, path)

    @staticmethod
    def _record_local_media(manifest: Dict[str, Any], *, kind: str, source: str, target: Path) -> None:
        media = manifest.setdefault("media", [])
        relative = f"media/{target.name}"
        key = (kind, source)
        for item in media:
            if isinstance(item, dict) and (item.get("kind"), item.get("source")) == key:
                item.update({"path": relative, "status": "available"})
                return
        media.append({"kind": kind, "source": source, "path": relative, "status": "available"})

    def _copy_local_media(
        self,
        metadata: Dict[str, Any],
        media_dir: Path,
        manifest: Dict[str, Any],
    ) -> int:
        copied = 0
        preview = str(metadata.get("preview_url") or "")
        if preview and os.path.isfile(preview):
            extension = Path(preview).suffix.lower() or ".webp"
            target = media_dir / f"preview{extension}"
            if not target.exists():
                shutil.copy2(preview, target)
                copied += 1
            self._record_local_media(manifest, kind="preview", source=preview, target=target)

        model_hash = str(metadata.get("sha256") or "").strip().lower()
        example_dir = get_model_folder(model_hash) if model_hash else ""
        if example_dir and os.path.isdir(example_dir):
            for source_path in Path(example_dir).iterdir():
                if not source_path.is_file() or source_path.suffix.lower() not in _MEDIA_EXTENSIONS:
                    continue
                target = media_dir / f"example-{source_path.name}"
                if not target.exists():
                    shutil.copy2(source_path, target)
                    copied += 1
                self._record_local_media(
                    manifest,
                    kind="example",
                    source=str(source_path),
                    target=target,
                )
        return copied

    async def _download_remote_media(
        self,
        metadata: Dict[str, Any],
        media_dir: Path,
        manifest: Dict[str, Any],
        *,
        cancel_check: Optional[CancelCheck] = None,
    ) -> tuple[int, int, bool, bool]:
        civitai = metadata.get("civitai") or {}
        images = civitai.get("images") or [] if isinstance(civitai, dict) else []
        if not isinstance(images, list):
            return 0, 0, False, False
        media_records = manifest.setdefault("media", [])
        downloader = await get_downloader()
        downloaded = 0
        failed = 0
        cancelled = False
        network_failed = False
        for index, image in enumerate(images):
            if cancel_check is not None and cancel_check():
                cancelled = True
                break
            if not isinstance(image, dict):
                continue
            url = str(image.get("url") or "").strip()
            if not url:
                continue
            existing = next(
                (item for item in media_records if isinstance(item, dict) and item.get("remote_url") == url),
                None,
            )
            if existing and existing.get("status") == "downloaded":
                local_path = media_dir.parent / str(existing.get("path") or "")
                if local_path.is_file():
                    continue
            media_type = str(image.get("type") or "image")
            identifier = str(image.get("id") or hashlib.sha256(url.encode("utf-8")).hexdigest()[:12])
            safe_identifier = "".join(
                char for char in identifier if char.isalnum() or char in "-_"
            )[:64]
            extension = _safe_extension(url, media_type)
            target = media_dir / f"remote-{index:03d}-{safe_identifier or index}{extension}"
            download_task = asyncio.create_task(
                downloader.download_file(
                    url,
                    str(target),
                    use_auth=False,
                    max_retries=0,
                    connect_timeout_seconds=10,
                    use_connectivity_guard=True,
                )
            )
            while not download_task.done():
                await asyncio.wait({download_task}, timeout=0.25)
                if cancel_check is not None and cancel_check():
                    download_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await download_task
                    cancelled = True
                    break
            if cancelled:
                break
            success, result = await download_task
            record = existing or {"kind": "remote_example", "remote_url": url}
            record.update(
                {
                    "path": f"media/{target.name}",
                    "status": "downloaded" if success else "failed",
                    "media_type": media_type,
                }
            )
            if success:
                record.pop("error", None)
                downloaded += 1
            else:
                record["error"] = str(result)
                failed += 1
                error_text = str(result).lower()
                network_failed = any(
                    marker in error_text
                    for marker in (
                        "network error",
                        "cannot connect",
                        "connection refused",
                        "timed out",
                        "timeout",
                        "offline_cooldown",
                        "network offline",
                    )
                )
                if target.exists():
                    target.unlink()
            if existing is None:
                media_records.append(record)
            if network_failed:
                # A broken image host would otherwise repeat the same timeout
                # for every URL on this model. The next model provides the
                # next circuit-breaker sample.
                break
        return downloaded, failed, cancelled, network_failed

    async def archive_existing(
        self,
        models: Iterable[Dict[str, Any]],
        *,
        download_remote_media: bool = False,
        cancel_check: Optional[CancelCheck] = None,
        progress_callback: Optional[ArchiveProgressCallback] = None,
    ) -> Dict[str, Any]:
        """Repair archives from existing metadata, optionally fetching media."""

        model_list = list(models)
        total = len(model_list)
        archived = 0
        skipped = 0
        processed = 0
        media_downloaded = 0
        media_failed = 0
        consecutive_network_failures = 0
        remote_media_suspended = False
        cancelled = False
        for model in model_list:
            if cancel_check is not None and cancel_check():
                cancelled = True
                break
            file_path = str(model.get("file_path") or "")
            result = await self.archive_model(
                file_path,
                dict(model),
                download_remote_media=download_remote_media and not remote_media_suspended,
                cancel_check=cancel_check,
            )
            processed += 1
            if result.get("archived"):
                archived += 1
            else:
                skipped += 1
            media_downloaded += int(result.get("remote_media_downloaded") or 0)
            media_failed += int(result.get("remote_media_failed") or 0)
            if result.get("remote_media_network_failed"):
                consecutive_network_failures += 1
            else:
                consecutive_network_failures = 0
            if consecutive_network_failures >= _REMOTE_FAILURE_THRESHOLD:
                remote_media_suspended = True
            if result.get("remote_media_cancelled"):
                cancelled = True

            if progress_callback is not None:
                await progress_callback(
                    {
                        "processed": processed,
                        "total": total,
                        "archived": archived,
                        "skipped": skipped,
                        "media_downloaded": media_downloaded,
                        "media_failed": media_failed,
                        "remote_media_suspended": remote_media_suspended,
                        "current_name": model.get("model_name")
                        or model.get("file_name")
                        or Path(file_path).stem,
                    }
                )
            if cancelled:
                break
        return {
            "archived": archived,
            "skipped": skipped,
            "processed": processed,
            "total": total,
            "media_downloaded": media_downloaded,
            "media_failed": media_failed,
            "remote_media_suspended": remote_media_suspended,
            "cancelled": cancelled,
        }


_MODEL_ARCHIVE_SERVICE: Optional[ModelArchiveService] = None


def get_model_archive_dir(file_path: str) -> Path:
    """Return the archive directory stored beside a model file."""

    model_path = Path(file_path).absolute()
    return model_path.with_name(f"{model_path.stem}.archive")


def get_model_archive_service() -> ModelArchiveService:
    global _MODEL_ARCHIVE_SERVICE
    if _MODEL_ARCHIVE_SERVICE is None:
        _MODEL_ARCHIVE_SERVICE = ModelArchiveService()
    return _MODEL_ARCHIVE_SERVICE
