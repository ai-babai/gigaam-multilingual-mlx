from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from openai import AsyncOpenAI

from gigaam_multilingual_mlx.audio import AudioDecodeError
from gigaam_multilingual_mlx.server import ServerConfig, create_app


RESULT = {
    "text": "привет мир",
    "duration_seconds": 2.0,
    "words": [
        {"text": "привет", "start": 0.0, "end": 0.8},
        {"text": "мир", "start": 1.0, "end": 1.8},
    ],
    "model": {"variant": "int8", "revision": "v0.1.0"},
}


class FakeWorker:
    variant = "int8"
    reference = None

    def __init__(self) -> None:
        self.started = False
        self.closed = False
        self.seen_paths: list[Path] = []

    async def start(self) -> None:
        self.started = True

    async def transcribe(self, audio_path: Path) -> dict[str, Any]:
        assert audio_path.is_file()
        self.seen_paths.append(audio_path)
        return RESULT

    async def close(self) -> None:
        self.closed = True


class FailingWorker(FakeWorker):
    async def transcribe(self, audio_path: Path) -> dict[str, Any]:
        assert audio_path.is_file()
        self.seen_paths.append(audio_path)
        raise RuntimeError("private implementation detail")


class FailingStartWorker(FakeWorker):
    async def start(self) -> None:
        raise RuntimeError("model failed to load")


class DecodeFailingWorker(FakeWorker):
    async def transcribe(self, audio_path: Path) -> dict[str, Any]:
        assert audio_path.is_file()
        self.seen_paths.append(audio_path)
        raise AudioDecodeError(f"decoder mentioned private path: {audio_path}")


class BlockingWorker(FakeWorker):
    def __init__(self) -> None:
        super().__init__()
        self.request_started = asyncio.Event()
        self.release_request = asyncio.Event()

    async def transcribe(self, audio_path: Path) -> dict[str, Any]:
        assert audio_path.is_file()
        self.request_started.set()
        await self.release_request.wait()
        return RESULT


def _files(data: bytes = b"fake audio") -> dict[str, tuple[str, bytes, str]]:
    return {"file": ("../../private-name.wav", data, "audio/wav")}


def test_health_models_and_json_transcription_cleanup(tmp_path: Path) -> None:
    worker = FakeWorker()
    app = create_app(ServerConfig(temp_dir=tmp_path), worker=worker)
    with TestClient(app) as client:
        assert client.get("/healthz").json() == {"status": "ok"}
        assert client.get("/readyz").json() == {"status": "ready"}
        models = client.get("/v1/models").json()["data"]
        assert [item["id"] for item in models] == [
            "gigaam-multilingual-mlx-int8-g64",
            "whisper-1",
        ]
        assert models[1]["metadata"]["canonical_model"] == models[0]["id"]
        response = client.post(
            "/v1/audio/transcriptions",
            data={"model": "whisper-1"},
            files=_files(),
        )
        assert response.status_code == 200
        assert response.json() == {"text": "привет мир"}
        assert response.headers["x-gigaam-model"] == "gigaam-multilingual-mlx-int8-g64"
        assert response.headers["x-request-id"].startswith("req_")
        assert not list(tmp_path.iterdir())
    assert worker.started and worker.closed
    assert worker.seen_paths[0].name.startswith("gigaam-upload-")
    assert "private-name" not in worker.seen_paths[0].name


def test_all_response_formats(tmp_path: Path) -> None:
    app = create_app(ServerConfig(temp_dir=tmp_path), worker=FakeWorker())
    with TestClient(app) as client:
        verbose = client.post(
            "/v1/audio/transcriptions",
            data={
                "model": "whisper-1",
                "language": "ru",
                "response_format": "verbose_json",
                "timestamp_granularities[]": ["word", "segment"],
            },
            files=_files(),
        )
        assert verbose.status_code == 200
        assert verbose.json()["words"][0]["word"] == "привет"
        for response_format, content_type, needle in (
            ("text", "text/plain", "привет мир"),
            ("srt", "application/x-subrip", "00:00:00,000"),
            ("vtt", "text/vtt", "WEBVTT"),
        ):
            response = client.post(
                "/v1/audio/transcriptions",
                data={"model": "whisper-1", "response_format": response_format},
                files=_files(),
            )
            assert response.status_code == 200
            assert response.headers["content-type"].startswith(content_type)
            assert needle in response.text


def test_official_openai_python_sdk_contract(tmp_path: Path) -> None:
    async def exercise() -> None:
        app = create_app(ServerConfig(temp_dir=tmp_path), worker=FakeWorker())
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://local.test",
            ) as http_client:
                client = AsyncOpenAI(
                    base_url="http://local.test/v1",
                    api_key="local",
                    http_client=http_client,
                )
                models = await client.models.list()
                assert [model.id for model in models.data] == [
                    "gigaam-multilingual-mlx-int8-g64",
                    "whisper-1",
                ]
                transcription = await client.audio.transcriptions.create(
                    model="whisper-1",
                    file=("sample.wav", b"fake audio", "audio/wav"),
                )
                assert transcription.text == "привет мир"
                verbose = await client.audio.transcriptions.create(
                    model="whisper-1",
                    file=("sample.wav", b"fake audio", "audio/wav"),
                    response_format="verbose_json",
                    timestamp_granularities=["word", "segment"],
                    language="ru",
                )
                assert verbose.text == "привет мир"
                assert verbose.language == "ru"
                assert verbose.words and verbose.words[0].word == "привет"
                assert verbose.segments and verbose.segments[0].end <= RESULT["duration_seconds"]

    asyncio.run(exercise())


def test_authentication_and_non_loopback_guard(tmp_path: Path) -> None:
    app = create_app(ServerConfig(temp_dir=tmp_path, api_key="test-key"), worker=FakeWorker())
    with TestClient(app) as client:
        unauthorized = client.post(
            "/v1/audio/transcriptions",
            data={"model": "whisper-1"},
            files=_files(),
        )
        assert unauthorized.status_code == 401
        assert unauthorized.json()["error"]["code"] == "invalid_api_key"
        authorized = client.post(
            "/v1/audio/transcriptions",
            headers={"Authorization": "Bearer test-key"},
            data={"model": "whisper-1"},
            files=_files(),
        )
        assert authorized.status_code == 200

    try:
        ServerConfig(host="0.0.0.0").validate()
    except ValueError as error:
        assert "bearer token" in str(error)
    else:
        raise AssertionError("non-loopback server started without authentication")


def test_invalid_fields_upload_limit_and_error_cleanup(tmp_path: Path) -> None:
    app = create_app(
        ServerConfig(temp_dir=tmp_path, max_upload_bytes=4),
        worker=FakeWorker(),
    )
    with TestClient(app) as client:
        unsupported = client.post(
            "/v1/audio/transcriptions",
            data={"model": "whisper-1", "diarize": "true"},
            files=_files(b"abc"),
        )
        assert unsupported.status_code == 400
        assert unsupported.json()["error"]["param"] == "diarize"
        too_large = client.post(
            "/v1/audio/transcriptions",
            data={"model": "whisper-1"},
            files=_files(b"abcde"),
        )
        assert too_large.status_code == 413
        assert too_large.json()["error"]["code"] == "file_too_large"
        body_too_large = client.post(
            "/v1/audio/transcriptions",
            data={"model": "whisper-1"},
            files=_files(b"x" * (2 * 1024 * 1024)),
        )
        assert body_too_large.status_code == 413
        assert body_too_large.json()["error"]["code"] == "file_too_large"
        duplicate_files = client.post(
            "/v1/audio/transcriptions",
            data={"model": "whisper-1"},
            files=[
                ("file", ("one.wav", b"a", "audio/wav")),
                ("file", ("two.wav", b"b", "audio/wav")),
            ],
        )
        assert duplicate_files.status_code == 422
        assert duplicate_files.json()["error"]["param"] == "file"
        assert not list(tmp_path.iterdir())

    failing_app = create_app(ServerConfig(temp_dir=tmp_path), worker=FailingWorker())
    with TestClient(failing_app, raise_server_exceptions=False) as client:
        failed = client.post(
            "/v1/audio/transcriptions",
            data={"model": "whisper-1"},
            files=_files(),
        )
        assert failed.status_code == 500
        assert failed.json()["error"]["code"] == "internal_error"
        assert "private implementation detail" not in failed.text
        assert not list(tmp_path.iterdir())

    decode_app = create_app(ServerConfig(temp_dir=tmp_path), worker=DecodeFailingWorker())
    with TestClient(decode_app, raise_server_exceptions=False) as client:
        failed = client.post(
            "/v1/audio/transcriptions",
            data={"model": "whisper-1"},
            files=_files(),
        )
        assert failed.status_code == 422
        assert failed.json()["error"]["code"] == "invalid_audio"
        assert str(tmp_path) not in failed.text
        assert not list(tmp_path.iterdir())


def test_bounded_queue_returns_429(tmp_path: Path) -> None:
    async def exercise() -> None:
        worker = BlockingWorker()
        app = create_app(
            ServerConfig(temp_dir=tmp_path, max_queue=1),
            worker=worker,
        )
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://local.test",
            ) as client:
                first = asyncio.create_task(
                    client.post(
                        "/v1/audio/transcriptions",
                        data={"model": "whisper-1"},
                        files=_files(),
                    )
                )
                await worker.request_started.wait()
                second = await client.post(
                    "/v1/audio/transcriptions",
                    data={"model": "whisper-1"},
                    files=_files(),
                )
                assert second.status_code == 429
                assert second.json()["error"]["code"] == "queue_full"
                worker.release_request.set()
                assert (await first).status_code == 200
        assert not list(tmp_path.iterdir())

    asyncio.run(exercise())


def test_startup_failure_closes_worker(tmp_path: Path) -> None:
    async def exercise() -> None:
        worker = FailingStartWorker()
        app = create_app(ServerConfig(temp_dir=tmp_path), worker=worker)
        with pytest.raises(RuntimeError, match="model failed to load"):
            async with app.router.lifespan_context(app):
                pass
        assert worker.closed

    asyncio.run(exercise())


def test_cancelled_request_cleans_up_and_releases_queue(tmp_path: Path) -> None:
    async def exercise() -> None:
        worker = BlockingWorker()
        app = create_app(
            ServerConfig(temp_dir=tmp_path, max_queue=1),
            worker=worker,
        )
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://local.test",
            ) as client:
                request = asyncio.create_task(
                    client.post(
                        "/v1/audio/transcriptions",
                        data={"model": "whisper-1"},
                        files=_files(),
                    )
                )
                await worker.request_started.wait()
                request.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await request
                assert not list(tmp_path.iterdir())

                worker.release_request.set()
                follow_up = await client.post(
                    "/v1/audio/transcriptions",
                    data={"model": "whisper-1"},
                    files=_files(),
                )
                assert follow_up.status_code == 200
        assert not list(tmp_path.iterdir())

    asyncio.run(exercise())
