from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from py.services import model_archive_service as archive_module
from py.services.model_archive_service import ModelArchiveService


class _Settings:
    def __init__(self, root: Path, **overrides):
        self.settings_file = str(root / "settings.json")
        self.values = {
            "model_archive_enabled": True,
            "model_archive_download_media": True,
        }
        self.values.update(overrides)

    def get(self, key, default=None):
        return self.values.get(key, default)


class _Downloader:
    def __init__(self):
        self.calls = []

    async def download_file(self, url, save_path, **_kwargs):
        self.calls.append(url)
        Path(save_path).write_bytes(b"remote-media")
        return True, save_path


def _metadata(model_path: Path, preview_path: Path) -> dict:
    return {
        "file_name": model_path.stem,
        "model_name": "Archived <Model>",
        "file_path": str(model_path),
        "sha256": "a" * 64,
        "preview_url": str(preview_path),
        "tags": ["style"],
        "modelDescription": "<p>Model instructions</p>",
        "usage_tips": '{"strength": 0.7}',
        "civitai": {
            "id": 456,
            "modelId": 123,
            "name": "v1",
            "description": "<p>Version notes</p>",
            "trainedWords": ["trigger"],
            "images": [
                {
                    "id": 789,
                    "type": "image",
                    "url": "https://image.civitai.com/example.png",
                    "meta": {"prompt": "archived prompt"},
                }
            ],
        },
    }


@pytest.mark.asyncio
async def test_archive_is_beside_model_append_only_and_downloads_media(tmp_path, monkeypatch):
    model_path = tmp_path / "model.safetensors"
    preview_path = tmp_path / "model.preview.webp"
    examples_path = tmp_path / "examples"
    model_path.write_bytes(b"model")
    preview_path.write_bytes(b"preview")
    examples_path.mkdir()
    (examples_path / "image_0.webp").write_bytes(b"example")

    downloader = _Downloader()

    async def _get_downloader():
        return downloader

    monkeypatch.setattr(archive_module, "get_downloader", _get_downloader)
    monkeypatch.setattr(archive_module, "get_model_folder", lambda _hash: str(examples_path))

    service = ModelArchiveService(_Settings(tmp_path))
    metadata = _metadata(model_path, preview_path)
    first = await service.archive_model(
        str(model_path), metadata, download_remote_media=True
    )
    second = await service.archive_model(
        str(model_path), metadata, download_remote_media=True
    )

    assert first["snapshot_created"] is True
    assert second["snapshot_created"] is False
    assert downloader.calls == ["https://image.civitai.com/example.png"]

    model_dir = Path(first["archive_dir"])
    assert model_dir == tmp_path / "model.archive"
    manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["model_id"] == 123
    assert manifest["model_version_id"] == 456
    assert len(manifest["snapshots"]) == 1
    assert (model_dir / manifest["latest_snapshot"]).is_file()
    assert (model_dir / "media" / "preview.webp").read_bytes() == b"preview"
    assert (model_dir / "media" / "example-image_0.webp").read_bytes() == b"example"
    assert any(item.get("status") == "downloaded" for item in manifest["media"])
    description = (model_dir / "description.html").read_text(encoding="utf-8")
    assert "Model instructions" in description
    assert "Version notes" in description
    assert "Archived &lt;Model&gt;" in description


@pytest.mark.asyncio
async def test_archive_retries_failed_remote_media(tmp_path, monkeypatch):
    model_path = tmp_path / "model.safetensors"
    preview_path = tmp_path / "preview.webp"
    model_path.write_bytes(b"model")
    preview_path.write_bytes(b"preview")
    attempts = 0

    class _RetryDownloader:
        async def download_file(self, _url, save_path, **_kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                return False, "gone for now"
            Path(save_path).write_bytes(b"recovered")
            return True, save_path

    async def _get_downloader():
        return _RetryDownloader()

    monkeypatch.setattr(archive_module, "get_downloader", _get_downloader)
    monkeypatch.setattr(archive_module, "get_model_folder", lambda _hash: "")

    service = ModelArchiveService(_Settings(tmp_path))
    metadata = _metadata(model_path, preview_path)
    await service.archive_model(str(model_path), metadata, download_remote_media=True)
    result = await service.archive_model(str(model_path), metadata, download_remote_media=True)

    assert attempts == 2
    assert result["remote_media_downloaded"] == 1
    manifest = json.loads(
        (Path(result["archive_dir"]) / "manifest.json").read_text(encoding="utf-8")
    )
    remote = next(item for item in manifest["media"] if item.get("remote_url"))
    assert remote["status"] == "downloaded"
    assert "error" not in remote


@pytest.mark.asyncio
async def test_archive_existing_never_downloads_remote_media(tmp_path, monkeypatch):
    model_path = tmp_path / "model.safetensors"
    preview_path = tmp_path / "preview.webp"
    model_path.write_bytes(b"model")
    preview_path.write_bytes(b"preview")

    async def _unexpected_downloader():
        raise AssertionError("cache rebuild archive repair must remain local-only")

    monkeypatch.setattr(archive_module, "get_downloader", _unexpected_downloader)
    monkeypatch.setattr(archive_module, "get_model_folder", lambda _hash: "")

    service = ModelArchiveService(_Settings(tmp_path))
    result = await service.archive_existing([_metadata(model_path, preview_path)])

    assert result["archived"] == 1
    assert result["skipped"] == 0
    assert result["processed"] == 1
    assert result["cancelled"] is False


@pytest.mark.asyncio
async def test_archive_existing_cancels_an_in_flight_media_download(tmp_path, monkeypatch):
    model_path = tmp_path / "model.safetensors"
    preview_path = tmp_path / "preview.webp"
    model_path.write_bytes(b"model")
    preview_path.write_bytes(b"preview")
    cancel_requested = False

    class _HangingDownloader:
        async def download_file(self, _url, _save_path, **kwargs):
            assert kwargs["max_retries"] == 0
            await asyncio.Event().wait()

    async def _get_downloader():
        return _HangingDownloader()

    monkeypatch.setattr(archive_module, "get_downloader", _get_downloader)
    monkeypatch.setattr(archive_module, "get_model_folder", lambda _hash: "")

    service = ModelArchiveService(_Settings(tmp_path))
    task = asyncio.create_task(
        service.archive_existing(
            [_metadata(model_path, preview_path)],
            download_remote_media=True,
            cancel_check=lambda: cancel_requested,
        )
    )
    await asyncio.sleep(0.05)
    cancel_requested = True
    result = await asyncio.wait_for(task, timeout=1)

    assert result["cancelled"] is True
    assert result["processed"] == 1


@pytest.mark.asyncio
async def test_archive_existing_suspends_remote_media_after_network_failures(
    tmp_path, monkeypatch
):
    calls = 0

    class _OfflineDownloader:
        async def download_file(self, _url, _save_path, **kwargs):
            nonlocal calls
            calls += 1
            assert kwargs["max_retries"] == 0
            return False, "Network error after 1 attempts: timed out"

    async def _get_downloader():
        return _OfflineDownloader()

    monkeypatch.setattr(archive_module, "get_downloader", _get_downloader)
    monkeypatch.setattr(archive_module, "get_model_folder", lambda _hash: "")

    models = []
    for index in range(5):
        model_path = tmp_path / f"model-{index}.safetensors"
        preview_path = tmp_path / f"model-{index}.webp"
        model_path.write_bytes(b"model")
        preview_path.write_bytes(b"preview")
        metadata = _metadata(model_path, preview_path)
        metadata["sha256"] = f"{index:064x}"
        models.append(metadata)

    progress = []

    async def _progress(payload):
        progress.append(payload)

    service = ModelArchiveService(_Settings(tmp_path))
    result = await service.archive_existing(
        models,
        download_remote_media=True,
        progress_callback=_progress,
    )

    assert calls == 3
    assert result["archived"] == 5
    assert result["media_failed"] == 3
    assert result["remote_media_suspended"] is True
    assert progress[-1]["processed"] == 5
