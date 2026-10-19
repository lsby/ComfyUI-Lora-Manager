"""Use case encapsulating the bulk metadata refresh orchestration."""

from __future__ import annotations

import inspect
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Protocol, Sequence

from ..metadata_sync_service import MetadataSyncService
from ..model_scanner import _normalized_root_prefix, _path_matches_prefix
from ..model_sources import has_external_source
from ..model_archive_service import get_model_archive_service
from ...utils.metadata_manager import MetadataManager


class MetadataRefreshProgressReporter(Protocol):
    """Protocol for progress reporters used during metadata refresh."""

    async def on_progress(self, payload: Dict[str, Any]) -> None:
        """Handle a metadata refresh progress update."""


class BulkMetadataRefreshUseCase:
    """Coordinate bulk metadata refreshes with progress emission."""

    def __init__(
        self,
        *,
        service,
        metadata_sync: MetadataSyncService,
        settings_service,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._service = service
        self._metadata_sync = metadata_sync
        self._settings = settings_service
        self._logger = logger or logging.getLogger(__name__)

    async def execute(
        self,
        *,
        progress_callback: Optional[MetadataRefreshProgressReporter] = None,
        file_paths: Optional[Sequence[str]] = None,
        force: bool = False,
    ) -> Dict[str, Any]:
        """Refresh metadata for all or a caller-selected set of models."""

        cache = await self._service.scanner.get_cached_data()
        requested_paths = list(dict.fromkeys(file_paths or []))
        if file_paths is None:
            scope_models = list(cache.raw_data)
            scope = "all"
        else:
            requested = {self._normalize_path(path) for path in requested_paths}
            scope_models = [
                model
                for model in cache.raw_data
                if self._normalize_path(model.get("file_path", "")) in requested
            ]
            scope = "selected"
        total_models = len(scope_models)

        enable_metadata_archive_db = self._settings.get("enable_metadata_archive_db", False)
        skip_paths = self._settings.get("metadata_refresh_skip_paths", [])
        eligible: List[Dict[str, Any]] = []
        report_items: List[Dict[str, Any]] = []

        for model in scope_models:
            reason = self._skip_reason(
                model,
                skip_paths=skip_paths,
                enable_metadata_archive_db=enable_metadata_archive_db,
                force=force,
            )
            if reason:
                report_items.append(self._report_item(model, "skipped", reason))
            else:
                eligible.append(model)

        # Models on a drive that cannot be read right now stay in the cache
        # (scoped scan keeps them), but fetching their metadata would only
        # spend CivitAI rate limit on sidecar/preview writes that cannot land.
        offline_roots = self._offline_root_details()
        if offline_roots:
            to_process = [
                model
                for model in eligible
                if not self._is_under_offline_root(model.get("file_path", ""), offline_roots)
            ]
        else:
            to_process = list(eligible)
        offline_skipped = len(eligible) - len(to_process)
        if offline_skipped:
            queued_paths = {model.get("file_path") for model in to_process}
            for model in eligible:
                if model.get("file_path") not in queued_paths:
                    report_items.append(
                        self._report_item(model, "skipped", "Model drive is offline")
                    )
        if offline_skipped:
            self._logger.info(
                "Bulk metadata refresh: %d model(s) skipped, drive offline: %s",
                offline_skipped,
                ", ".join(root["label"] for root in offline_roots),
            )

        total_to_process = len(to_process)
        initial_skipped = total_models - total_to_process
        processed = 0
        success = 0
        updated_count = 0
        invalid_count = 0
        skipped_count = initial_skipped
        handled_count = initial_skipped
        needs_resort = False
        start_time = time.monotonic()
        failures: List[Dict[str, str]] = []

        self._service.scanner.reset_cancellation()

        async def emit(status: str, **extra: Any) -> None:
            if progress_callback is None:
                return
            payload: Dict[str, Any] = {
                "status": status,
                "total": total_models,
                "processed": processed,
                "success": success,
                "updated_count": updated_count,
                "invalid_count": invalid_count,
                "failure_count": len(failures),
                "skipped_count": skipped_count,
                "offline_skipped": offline_skipped,
                "handled": handled_count,
                "elapsed_seconds": int(time.monotonic() - start_time),
            }
            # Only include full failure details in terminal emits (completed,
            # cancelled, rate_limited) to avoid serializing the list on every
            # per-model progress update.
            if status in ("completed", "cancelled", "rate_limited"):
                payload["failures"] = failures
                payload["report"] = build_report(status)
                if offline_roots:
                    payload["offline_roots"] = [
                        {"path": root["path"], "label": root["label"]}
                        for root in offline_roots
                    ]
            payload.update(extra)
            await progress_callback.on_progress(payload)

        def build_report(status: str) -> Dict[str, Any]:
            return {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "model_type": self._service.model_type,
                "scope": scope,
                "force": force,
                "status": status,
                "requested": len(requested_paths) if file_paths is not None else total_models,
                "matched": total_models,
                "attempted": processed,
                "successful": success,
                "updated": updated_count,
                "unchanged": max(0, success - updated_count),
                "invalid": invalid_count,
                "failed": len(failures),
                "skipped": skipped_count,
                "elapsed_seconds": int(time.monotonic() - start_time),
                "items": report_items,
            }

        def result_payload(status: str, message: str, ok: bool, **extra: Any) -> Dict[str, Any]:
            payload: Dict[str, Any] = {
                "success": ok,
                "message": message,
                "processed": processed,
                # Preserve the legacy response contract: ``updated`` meant a
                # successful fetch. Detailed reports use ``updated_count`` for
                # models whose metadata actually changed.
                "updated": success,
                "successful_count": success,
                "updated_count": updated_count,
                "invalid_count": invalid_count,
                "total": total_models,
                "failures": failures,
                "failure_count": len(failures),
                "skipped_count": skipped_count,
                "offline_skipped": offline_skipped,
                "elapsed_seconds": int(time.monotonic() - start_time),
                "report": build_report(status),
            }
            payload.update(extra)
            return payload

        await emit(
            "started",
            phase="metadata",
            fetch_total=total_to_process,
        )

        RATE_LIMIT_ABORT_THRESHOLD = 3
        consecutive_rate_limits = 0

        for model in to_process:
            if self._service.scanner.is_cancelled():
                self._logger.info("Bulk metadata refresh cancelled by user")
                await emit("cancelled", processed=processed, success=success)
                return result_payload("cancelled", "Operation cancelled", False)
            try:
                original_name = model.get("model_name")
                await emit(
                    "model_started",
                    phase="metadata",
                    fetch_index=processed + 1,
                    fetch_total=total_to_process,
                    current_name=model.get("model_name")
                    or model.get("file_name")
                    or model.get("file_path")
                    or "Unknown",
                )

                # Handle lazy hash calculation for models with pending hash status
                sha256 = model.get("sha256", "")
                hash_status = model.get("hash_status", "completed")
                file_path = model.get("file_path")

                if not sha256 and hash_status == "pending" and file_path:
                    self._logger.info(f"Calculating pending hash for {file_path}")
                    # Check if scanner has calculate_hash_for_model method (CheckpointScanner)
                    calculate_hash_method = getattr(self._service.scanner, "calculate_hash_for_model", None)
                    if calculate_hash_method:
                        sha256 = await calculate_hash_method(file_path)
                        if sha256:
                            model["sha256"] = sha256
                            model["hash_status"] = "completed"
                            hash_status = "completed"
                        else:
                            self._logger.error(f"Failed to calculate hash for {file_path}")
                            error = "Failed to calculate hash"
                            failures.append({"name": model.get("model_name", file_path or "Unknown"), "error": error})
                            report_items.append(self._report_item(model, "failed", error))
                            await self._persist_metadata_fetch_status(model, "failed", error)
                            processed += 1
                            handled_count += 1
                            continue
                    else:
                        self._logger.warning(f"Scanner does not support lazy hash calculation for {file_path}")
                        report_items.append(self._report_item(model, "skipped", "Hash calculation is not supported"))
                        await self._persist_metadata_fetch_status(
                            model, "failed", "Hash calculation is not supported"
                        )
                        skipped_count += 1
                        processed += 1
                        handled_count += 1
                        continue

                # Skip models without valid hash
                if not model.get("sha256"):
                    self._logger.warning(f"Skipping model without hash: {file_path}")
                    report_items.append(self._report_item(model, "skipped", "Model has no SHA-256 hash"))
                    await self._persist_metadata_fetch_status(
                        model, "failed", "Model has no SHA-256 hash"
                    )
                    skipped_count += 1
                    processed += 1
                    handled_count += 1
                    continue

                await MetadataManager.hydrate_model_data(model)

                # hydrate_model_data replaces model with .metadata.json content,
                # which may lack sha256. Restore from cache and persist the fix.
                if not model.get("sha256"):
                    model["sha256"] = sha256
                    model["hash_status"] = model.get("hash_status", hash_status)
                    data_to_save = model.copy()
                    data_to_save.pop("folder", None)
                    await MetadataManager.save_metadata(file_path, data_to_save)

                before_snapshot = self._metadata_snapshot(model)

                result, error_msg = await self._metadata_sync.fetch_and_update_model(
                    sha256=model["sha256"],
                    file_path=model["file_path"],
                    model_data=model,
                    update_cache_func=self._service.scanner.update_single_model_cache,
                    archive_remote_media=False,
                    cancel_check=self._service.scanner.is_cancelled,
                    preview_max_retries=0,
                    preview_connect_timeout_seconds=10,
                    force_retry=force,
                )

                if not result and error_msg and "Rate limited" in error_msg:
                    consecutive_rate_limits += 1
                else:
                    consecutive_rate_limits = 0

                if not result:
                    current_name = model.get("model_name", file_path or "Unknown")
                    error = error_msg or "Unknown error"
                    if self._is_invalid_model(model, error):
                        invalid_count += 1
                        report_items.append(self._report_item(model, "invalid", error))
                        await self._persist_metadata_fetch_status(model, "invalid", error)
                    else:
                        failures.append({"name": current_name, "error": error})
                        report_items.append(self._report_item(model, "failed", error))
                        await self._persist_metadata_fetch_status(model, "failed", error)
                    self._logger.warning("Failed to fetch metadata for %s: %s", current_name, error_msg)

                if consecutive_rate_limits >= RATE_LIMIT_ABORT_THRESHOLD:
                    # The current model was attempted and failed due to rate limiting;
                    # count it before aborting so the summary is consistent.
                    processed += 1
                    handled_count += 1
                    self._logger.warning(
                        "Bulk metadata refresh aborted: %d consecutive rate limits detected. "
                        "Processed %d/%d models.",
                        consecutive_rate_limits,
                        processed,
                        total_to_process,
                    )
                    await emit(
                        "rate_limited",
                    )
                    return result_payload(
                        "rate_limited",
                        f"Rate limit detected; {total_to_process - processed} models skipped",
                        False,
                    )

                if result:
                    success += 1
                    changed = before_snapshot != self._metadata_snapshot(model)
                    if changed:
                        updated_count += 1
                    report_items.append(
                        self._report_item(model, "updated" if changed else "unchanged")
                    )
                    await self._persist_metadata_fetch_status(model, "valid")
                    if original_name != model.get("model_name"):
                        needs_resort = True
                processed += 1
                handled_count += 1
                await emit(
                    "processing",
                    phase="metadata",
                    processed=processed,
                    success=success,
                    fetch_index=processed,
                    fetch_total=total_to_process,
                    current_name=model.get("model_name", "Unknown"),
                )
            except Exception as exc:  # pragma: no cover - logging path
                processed += 1
                handled_count += 1
                current_name = model.get("model_name", model.get("file_path", "Unknown"))
                failures.append({"name": current_name, "error": str(exc)})
                report_items.append(self._report_item(model, "failed", str(exc)))
                await self._persist_metadata_fetch_status(model, "failed", str(exc))
                self._logger.error(
                    "Error fetching CivitAI data for %s: %s",
                    model.get("file_path"),
                    exc,
                )

        if needs_resort:
            await cache.resort()

        archive_summary: Dict[str, Any] = {
            "archived": 0,
            "skipped": 0,
            "processed": 0,
            "total": 0,
            "media_downloaded": 0,
            "media_failed": 0,
            "remote_media_suspended": False,
            "cancelled": False,
        }
        archive_service = get_model_archive_service()
        if archive_service.enabled:
            archive_total = len(scope_models)
            await emit(
                "archive_started",
                phase="archive",
                archive_processed=0,
                archive_total=archive_total,
                archive_success=0,
                archive_skipped=0,
                media_downloaded=0,
                media_failed=0,
            )

            async def report_archive_progress(payload: Dict[str, Any]) -> None:
                await emit(
                    "archive_processing",
                    phase="archive",
                    archive_processed=payload["processed"],
                    archive_total=payload["total"],
                    archive_success=payload["archived"],
                    archive_skipped=payload["skipped"],
                    media_downloaded=payload["media_downloaded"],
                    media_failed=payload["media_failed"],
                    remote_media_suspended=payload["remote_media_suspended"],
                    current_name=payload["current_name"],
                )

            # Repair archives only inside the requested scope.
            archive_summary = await archive_service.archive_existing(
                scope_models,
                download_remote_media=True,
                cancel_check=self._service.scanner.is_cancelled,
                progress_callback=report_archive_progress,
            )
            if archive_summary["cancelled"]:
                await emit(
                    "cancelled",
                    phase="archive",
                    archive_processed=archive_summary["processed"],
                    archive_total=archive_summary["total"],
                    archive_success=archive_summary["archived"],
                    archive_skipped=archive_summary["skipped"],
                    media_downloaded=archive_summary["media_downloaded"],
                    media_failed=archive_summary["media_failed"],
                )
                return result_payload(
                    "cancelled",
                    "Operation cancelled during model archive repair",
                    False,
                    archive=archive_summary,
                )

        await emit(
            "completed",
            phase="complete",
            processed=processed,
            success=success,
            updated_count=updated_count,
            invalid_count=invalid_count,
            archive_processed=archive_summary["processed"],
            archive_total=archive_summary["total"],
            archive_success=archive_summary["archived"],
            archive_skipped=archive_summary["skipped"],
            media_downloaded=archive_summary["media_downloaded"],
            media_failed=archive_summary["media_failed"],
            remote_media_suspended=archive_summary["remote_media_suspended"],
        )

        message = (
            "Successfully updated "
            f"{success} of {processed} processed {self._service.model_type}s "
            f"({updated_count} changed, {invalid_count} invalid, total: {total_models})"
        )
        if offline_skipped:
            message += (
                f"; {offline_skipped} skipped, drive offline ("
                + ", ".join(root["label"] for root in offline_roots)
                + ")"
            )

        return result_payload("completed", message, True, archive=archive_summary)

    @staticmethod
    def _normalize_path(path: str) -> str:
        if not path:
            return ""
        return os.path.normcase(os.path.abspath(path))

    @staticmethod
    def _report_item(
        model: Dict[str, Any], status: str, detail: str = ""
    ) -> Dict[str, Any]:
        return {
            "name": model.get("model_name")
            or model.get("file_name")
            or model.get("file_path")
            or "Unknown",
            "file_path": model.get("file_path", ""),
            "status": status,
            "detail": detail,
        }

    @staticmethod
    def _metadata_snapshot(model: Dict[str, Any]) -> str:
        fields = {
            key: model.get(key)
            for key in (
                "model_name",
                "modelDescription",
                "tags",
                "base_model",
                "civitai",
                "from_civitai",
                "civitai_deleted",
                "metadata_source",
                "preview_url",
            )
        }
        return json.dumps(fields, sort_keys=True, ensure_ascii=False, default=str)

    async def _persist_metadata_fetch_status(
        self,
        model: Dict[str, Any],
        status: str,
        detail: str = "",
    ) -> None:
        """Persist a stable fetch classification to the owned sidecar and cache."""
        file_path = model.get("file_path", "")
        if not file_path:
            return

        model["metadata_fetch_status"] = status
        model["metadata_fetch_status_detail"] = detail
        model["metadata_fetch_status_at"] = datetime.now(timezone.utc).isoformat()
        payload = model.copy()
        payload.pop("folder", None)

        if os.path.isfile(file_path):
            saved = await MetadataManager.save_metadata(file_path, payload)
            if not saved:
                self._logger.warning(
                    "Failed to persist metadata fetch status to sidecar for %s",
                    file_path,
                )

        try:
            scanner = self._service.scanner
            sync_cache = getattr(scanner, "sync_cache_from_metadata", None)
            if callable(sync_cache):
                result = sync_cache(file_path, payload)
                if inspect.isawaitable(result):
                    await result
                return

            update_cache = getattr(scanner, "update_single_model_cache", None)
            if callable(update_cache):
                result = update_cache(file_path, file_path, payload)
                if inspect.isawaitable(result):
                    await result
        except Exception as exc:  # pragma: no cover - best-effort cache sync
            self._logger.warning(
                "Failed to persist metadata fetch status in cache for %s: %s",
                file_path,
                exc,
            )

    @staticmethod
    def _is_invalid_model(model: Dict[str, Any], error: str) -> bool:
        lowered = error.lower()
        return model.get("civitai_deleted") is True or any(
            marker in lowered
            for marker in ("model not found", "model is deleted", "version not found")
        )

    def _skip_reason(
        self,
        model: Dict[str, Any],
        *,
        skip_paths: List[str],
        enable_metadata_archive_db: bool,
        force: bool,
    ) -> str:
        if model.get("skip_metadata_refresh", False):
            return "Model is configured to skip metadata refresh"
        if self._is_in_skip_path(model.get("folder", ""), skip_paths):
            return "Model folder is configured to skip metadata refresh"
        if has_external_source(model):
            return "Model uses a non-CivitAI source"
        if force:
            return ""
        if model.get("civitai") and model["civitai"].get("id"):
            return "Metadata already exists"
        if (
            model.get("from_civitai") is False
            and model.get("civitai_deleted") is True
            and (not enable_metadata_archive_db or model.get("db_checked", False))
        ):
            return "Model was previously confirmed unavailable"
        return ""

    def _offline_root_details(self) -> List[Dict[str, str]]:
        """Configured roots whose directory cannot be read right now.

        Uses the scanner's root description (live ``os.path.exists`` per root),
        so both a drive switched off at startup and one unplugged mid-session
        are covered. Root probing must never break a fetch: any failure yields
        an empty list, which disables the offline filter.
        """
        describe = getattr(self._service.scanner, "describe_model_roots", None)
        if describe is None:
            return []
        try:
            details = describe()
        except Exception as exc:  # pragma: no cover - defensive logging
            self._logger.debug("Failed to describe model roots: %s", exc)
            return []

        offline: List[Dict[str, str]] = []
        for detail in details:
            if not isinstance(detail, dict) or detail.get("reachable", True):
                continue
            path = detail.get("path") or ""
            if not path:
                continue
            offline.append(
                {
                    "path": path,
                    "label": detail.get("label") or path,
                    "prefix": _normalized_root_prefix(path),
                }
            )
        return offline

    @staticmethod
    def _is_under_offline_root(
        file_path: str, offline_roots: Sequence[Dict[str, str]]
    ) -> bool:
        """Match a cached business path against offline root prefixes."""
        if not file_path:
            return False
        return any(
            _path_matches_prefix(file_path, root["prefix"]) for root in offline_roots
        )

    @staticmethod
    def _is_in_skip_path(folder: str, skip_paths: List[str]) -> bool:
        if not skip_paths or not folder:
            return False
        normalized = folder.replace("\\", "/").strip("/")
        if not normalized:
            return False
        for sp in skip_paths:
            nsp = sp.replace("\\", "/").strip("/")
            if not nsp:
                continue
            if normalized == nsp or normalized.startswith(nsp + "/"):
                return True
        return False

    async def execute_with_error_handling(
        self,
        *,
        progress_callback: Optional[MetadataRefreshProgressReporter] = None,
        file_paths: Optional[Sequence[str]] = None,
        force: bool = False,
    ) -> Dict[str, Any]:
        """Wrapper providing progress notification on unexpected failures."""

        try:
            return await self.execute(
                progress_callback=progress_callback,
                file_paths=file_paths,
                force=force,
            )
        except Exception as exc:
            if progress_callback is not None:
                await progress_callback.on_progress({"status": "error", "error": str(exc)})
            raise
