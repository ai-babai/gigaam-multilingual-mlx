from __future__ import annotations

import asyncio
import hmac
import ipaddress
import logging
import os
import tempfile
import time
import uuid
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Protocol

import mlx.core as mx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from starlette.datastructures import FormData, UploadFile
from starlette.exceptions import HTTPException
from starlette.formparsers import MultiPartException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from ._version import __version__
from .artifacts import DEFAULT_VARIANT, VARIANTS, ArtifactReference, resolve_model_source
from .audio import AudioDecodeError
from .model import load_model
from .openai_schema import (
    OpenAIAPIError,
    accepted_model_ids,
    canonical_model_id,
    model_object,
    render_openai_response,
    validate_transcription_parameters,
)
from .service import transcribe_file


LOGGER = logging.getLogger("gigaam_multilingual_mlx.server")
CHUNK_BYTES = 1024 * 1024
MULTIPART_OVERHEAD_BYTES = 1024 * 1024
ALLOWED_FORM_FIELDS = {
    "file",
    "model",
    "language",
    "response_format",
    "timestamp_granularities",
    "timestamp_granularities[]",
    "temperature",
    "prompt",
    "stream",
}


def _is_loopback_host(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


@dataclass(frozen=True)
class ServerConfig:
    variant: str = DEFAULT_VARIANT
    source: str | Path | None = None
    revision: str | None = None
    cache_dir: str | Path | None = None
    offline: bool = False
    dtype: str | None = None
    chunk_seconds: float = 20.0
    overlap_seconds: float = 2.0
    compile: bool = False
    host: str = "127.0.0.1"
    port: int = 8000
    max_upload_bytes: int = 1024 * 1024 * 1024
    max_queue: int = 8
    temp_dir: str | Path | None = None
    api_key: str | None = field(default=None, repr=False)
    allow_unauthenticated: bool = False

    def validate(self) -> None:
        if self.variant not in VARIANTS:
            raise ValueError(f"Unknown variant {self.variant!r}; choose from {', '.join(VARIANTS)}")
        if not 1 <= self.port <= 65535:
            raise ValueError("port must be between 1 and 65535")
        if self.max_upload_bytes <= 0:
            raise ValueError("max_upload_bytes must be positive")
        if self.max_queue <= 0:
            raise ValueError("max_queue must be positive")
        if self.chunk_seconds <= self.overlap_seconds or self.overlap_seconds < 0:
            raise ValueError("Require chunk_seconds > overlap_seconds >= 0")
        if not _is_loopback_host(self.host) and not (self.api_key or self.allow_unauthenticated):
            raise ValueError(
                "A bearer token is required for non-loopback bind; set "
                "GIGAAM_STT_API_KEY or --api-key-file, or explicitly use "
                "--allow-unauthenticated"
            )


class WorkerProtocol(Protocol):
    variant: str
    reference: ArtifactReference | None

    async def start(self) -> None: ...

    async def transcribe(self, audio_path: Path) -> dict[str, Any]: ...

    async def close(self) -> None: ...


class ModelWorker:
    def __init__(self, config: ServerConfig) -> None:
        self.config = config
        self.variant = config.variant
        self.reference: ArtifactReference | None = None
        self._model = None
        self._runner = None
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="gigaam-mlx-inference",
        )
        self._started = False
        self._closed = False

    def _load(self) -> None:
        self.reference = resolve_model_source(
            self.config.source,
            variant=self.config.variant,
            revision=self.config.revision,
            cache_dir=self.config.cache_dir,
            local_files_only=self.config.offline,
        )
        mx.reset_peak_memory()
        self._model = load_model(self.reference.path, self.config.dtype)
        if self.config.compile:
            self._runner = mx.compile(self._model.__call__, inputs=self._model.state)

    async def start(self) -> None:
        if self._started:
            return
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, self._load)
        self._started = True

    async def transcribe(self, audio_path: Path) -> dict[str, Any]:
        if not self._started or self._model is None or self.reference is None:
            raise RuntimeError("Model worker is not ready")
        function = partial(
            transcribe_file,
            self._model,
            audio_path,
            self.config.chunk_seconds,
            self.config.overlap_seconds,
            runner=self._runner,
            artifact=self.reference,
        )
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(self._executor, function)
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            try:
                await future
            except Exception:
                pass
            raise

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await asyncio.to_thread(self._executor.shutdown, wait=True, cancel_futures=True)


class AdmissionController:
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self._active = 0
        self._lock = asyncio.Lock()

    async def try_acquire(self) -> bool:
        async with self._lock:
            if self._active >= self.limit:
                return False
            self._active += 1
            return True

    async def release(self) -> None:
        async with self._lock:
            self._active = max(0, self._active - 1)


class RequestTooLarge(Exception):
    pass


class RequestBodyLimitMiddleware:
    def __init__(self, app: ASGIApp, max_body_bytes: int) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("method") != "POST":
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers", []))
        content_length = headers.get(b"content-length")
        if content_length is not None:
            try:
                declared_bytes = int(content_length)
            except ValueError:
                declared_bytes = 0
            if declared_bytes > self.max_body_bytes:
                await self._reject(scope, receive, send)
                return
        total = 0

        async def limited_receive() -> Message:
            nonlocal total
            message = await receive()
            if message["type"] == "http.request":
                total += len(message.get("body", b""))
                if total > self.max_body_bytes:
                    scope["gigaam.request_too_large"] = True
                    raise MultiPartException("Request body exceeded the configured upload limit")
            return message

        try:
            await self.app(scope, limited_receive, send)
        except RequestTooLarge:
            await self._reject(scope, receive, send)

    @staticmethod
    async def _reject(scope: Scope, receive: Receive, send: Send) -> None:
        response = JSONResponse(
            OpenAIAPIError(
                "Request body exceeds the configured upload limit",
                status_code=413,
                param="file",
                code="file_too_large",
            ).body(),
            status_code=413,
        )
        await response(scope, receive, send)


def _request_id() -> str:
    return f"req_{uuid.uuid4().hex}"


def _require_authorization(request: Request, config: ServerConfig) -> None:
    if not config.api_key:
        return
    authorization = request.headers.get("authorization", "")
    scheme, _, provided = authorization.partition(" ")
    if (
        scheme.lower() != "bearer"
        or not provided
        or not hmac.compare_digest(
            provided,
            config.api_key,
        )
    ):
        raise OpenAIAPIError(
            "Invalid bearer token",
            status_code=401,
            error_type="authentication_error",
            code="invalid_api_key",
        )


def _string_field(form: FormData, name: str) -> str | None:
    value = form.get(name)
    if value is None:
        return None
    if isinstance(value, UploadFile):
        raise OpenAIAPIError(
            f"{name} must be a text field",
            status_code=422,
            param=name,
            code="invalid_type",
        )
    return str(value)


def _timestamp_fields(form: FormData) -> list[str]:
    values = [*form.getlist("timestamp_granularities")]
    values.extend(form.getlist("timestamp_granularities[]"))
    if any(isinstance(value, UploadFile) for value in values):
        raise OpenAIAPIError(
            "timestamp_granularities must be text fields",
            status_code=422,
            param="timestamp_granularities",
            code="invalid_type",
        )
    return [str(value) for value in values]


def _safe_suffix(filename: str | None) -> str:
    suffix = Path(filename or "").suffix.lower()
    if 1 < len(suffix) <= 10 and suffix[1:].isalnum():
        return suffix
    return ".bin"


async def _save_upload(upload: UploadFile, config: ServerConfig) -> tuple[Path, int]:
    directory = Path(config.temp_dir).expanduser() if config.temp_dir else None
    if directory is not None:
        directory.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="wb",
        prefix="gigaam-upload-",
        suffix=_safe_suffix(upload.filename),
        dir=directory,
        delete=False,
    )
    path = Path(handle.name)
    total = 0
    try:
        while True:
            chunk = await upload.read(CHUNK_BYTES)
            if not chunk:
                break
            total += len(chunk)
            if total > config.max_upload_bytes:
                raise OpenAIAPIError(
                    "Uploaded file exceeds the configured limit",
                    status_code=413,
                    param="file",
                    code="file_too_large",
                )
            handle.write(chunk)
        handle.flush()
    except BaseException:
        handle.close()
        path.unlink(missing_ok=True)
        raise
    handle.close()
    return path, total


async def _read_form(request: Request) -> FormData:
    content_type = request.headers.get("content-type", "")
    if not content_type.lower().startswith("multipart/form-data"):
        raise OpenAIAPIError(
            "Content-Type must be multipart/form-data",
            status_code=422,
            param="file",
            code="invalid_content_type",
        )
    try:
        return await request.form()
    except Exception:
        if request.scope.get("gigaam.request_too_large"):
            raise RequestTooLarge from None
        raise OpenAIAPIError(
            "Invalid multipart form data",
            status_code=422,
            param="file",
            code="invalid_multipart",
        ) from None


def create_app(
    config: ServerConfig | None = None,
    *,
    worker: WorkerProtocol | None = None,
) -> FastAPI:
    settings = config or ServerConfig()
    settings.validate()
    model_worker = worker or ModelWorker(settings)
    admission = AdmissionController(settings.max_queue)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.ready = False
        try:
            await model_worker.start()
            app.state.ready = True
            yield
        finally:
            app.state.ready = False
            await model_worker.close()

    app = FastAPI(
        title="GigaAM Multilingual MLX transcription server",
        version=__version__,
        lifespan=lifespan,
    )
    app.add_middleware(
        RequestBodyLimitMiddleware,
        max_body_bytes=settings.max_upload_bytes + MULTIPART_OVERHEAD_BYTES,
    )

    @app.exception_handler(OpenAIAPIError)
    async def openai_error_handler(request: Request, error: OpenAIAPIError) -> JSONResponse:
        response = JSONResponse(error.body(), status_code=error.status_code)
        if request_id := getattr(request.state, "request_id", None):
            response.headers["X-Request-ID"] = request_id
        return response

    @app.exception_handler(HTTPException)
    async def http_error_handler(request: Request, error: HTTPException) -> JSONResponse:
        converted = OpenAIAPIError(
            str(error.detail),
            status_code=error.status_code,
            error_type="not_found_error" if error.status_code == 404 else "invalid_request_error",
            code="not_found" if error.status_code == 404 else "http_error",
        )
        return JSONResponse(converted.body(), status_code=converted.status_code)

    @app.exception_handler(Exception)
    async def internal_error_handler(request: Request, error: Exception) -> JSONResponse:
        LOGGER.error(
            "request_id=%s status=500 error_type=%s",
            getattr(request.state, "request_id", "unknown"),
            type(error).__name__,
        )
        converted = OpenAIAPIError(
            "Internal server error",
            status_code=500,
            error_type="server_error",
            code="internal_error",
        )
        return JSONResponse(converted.body(), status_code=500)

    @app.get("/healthz")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def ready(request: Request) -> JSONResponse:
        is_ready = bool(getattr(request.app.state, "ready", False))
        return JSONResponse(
            {"status": "ready" if is_ready else "starting"},
            status_code=200 if is_ready else 503,
        )

    @app.get("/v1/models")
    async def models(request: Request) -> dict[str, Any]:
        _require_authorization(request, settings)
        canonical, alias = accepted_model_ids(model_worker.variant, model_worker.reference)
        return {
            "object": "list",
            "data": [
                model_object(
                    model_id,
                    variant=model_worker.variant,
                    reference=model_worker.reference,
                )
                for model_id in (canonical, alias)
            ],
        }

    @app.get("/v1/models/{model_id}")
    async def retrieve_model(model_id: str, request: Request) -> dict[str, Any]:
        _require_authorization(request, settings)
        if model_id not in accepted_model_ids(model_worker.variant, model_worker.reference):
            raise OpenAIAPIError(
                f"Model {model_id!r} is not served by this process",
                status_code=404,
                error_type="not_found_error",
                param="model",
                code="model_not_found",
            )
        return model_object(
            model_id,
            variant=model_worker.variant,
            reference=model_worker.reference,
        )

    @app.post("/v1/audio/transcriptions")
    async def create_transcription(request: Request) -> Response:
        request.state.request_id = _request_id()
        _require_authorization(request, settings)
        if not await admission.try_acquire():
            raise OpenAIAPIError(
                "The local transcription queue is full; retry later",
                status_code=429,
                error_type="rate_limit_error",
                code="queue_full",
            )
        form: FormData | None = None
        temp_path: Path | None = None
        total_bytes = 0
        started = time.perf_counter()
        try:
            form = await _read_form(request)
            unknown_fields = sorted(set(form.keys()) - ALLOWED_FORM_FIELDS)
            if unknown_fields:
                raise OpenAIAPIError(
                    f"Unsupported request field {unknown_fields[0]!r}",
                    param=unknown_fields[0],
                    code="unsupported_parameter",
                )
            file_values = form.getlist("file")
            if len(file_values) != 1 or not isinstance(file_values[0], UploadFile):
                raise OpenAIAPIError(
                    "Exactly one file field is required and it must be an upload",
                    status_code=422,
                    param="file",
                    code="missing_required_parameter",
                )
            file_value = file_values[0]
            parameters = validate_transcription_parameters(
                model=_string_field(form, "model"),
                variant=model_worker.variant,
                reference=model_worker.reference,
                response_format=_string_field(form, "response_format"),
                language=_string_field(form, "language"),
                timestamp_granularities=_timestamp_fields(form),
                temperature=_string_field(form, "temperature"),
                prompt=_string_field(form, "prompt"),
                stream=_string_field(form, "stream"),
            )
            temp_path, total_bytes = await _save_upload(file_value, settings)
            try:
                result = await model_worker.transcribe(temp_path)
            except AudioDecodeError as error:
                raise OpenAIAPIError(
                    "Uploaded file could not be decoded as supported audio or video",
                    status_code=422,
                    param="file",
                    code="invalid_audio",
                ) from error
            rendered = render_openai_response(result, parameters)
            if isinstance(rendered.content, dict):
                response: Response = JSONResponse(rendered.content)
            else:
                response = Response(rendered.content, media_type=rendered.media_type)
            response.headers["X-Request-ID"] = request.state.request_id
            response.headers["X-GigaAM-Model"] = canonical_model_id(
                model_worker.variant,
                model_worker.reference,
            )
            LOGGER.info(
                "request_id=%s status=200 bytes=%d variant=%s elapsed_seconds=%.3f",
                request.state.request_id,
                total_bytes,
                model_worker.variant,
                time.perf_counter() - started,
            )
            return response
        finally:
            try:
                if form is not None:
                    await form.close()
            finally:
                try:
                    if temp_path is not None:
                        temp_path.unlink(missing_ok=True)
                finally:
                    await admission.release()

    return app


def read_api_key(api_key_file: str | Path | None = None) -> str | None:
    if api_key_file is not None:
        value = Path(api_key_file).expanduser().read_text().strip()
        if not value:
            raise ValueError("API key file is empty")
        return value
    value = os.environ.get("GIGAAM_STT_API_KEY")
    return value.strip() if value and value.strip() else None


def run_server(config: ServerConfig) -> None:
    config.validate()
    import uvicorn

    uvicorn.run(
        create_app(config),
        host=config.host,
        port=config.port,
        workers=1,
    )
