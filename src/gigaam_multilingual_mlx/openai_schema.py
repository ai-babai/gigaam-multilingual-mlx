from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .artifacts import VARIANTS, ArtifactReference
from .outputs import render_transcription, transcription_cues


COMPATIBILITY_ALIAS = "whisper-1"
OPENAI_RESPONSE_FORMATS = ("json", "text", "verbose_json", "srt", "vtt")
SUPPORTED_LANGUAGES = ("ru", "kk", "ky", "uz", "en")
SUPPORTED_TIMESTAMP_GRANULARITIES = ("word", "segment")


class OpenAIAPIError(Exception):
    def __init__(
        self,
        message: str,
        *,
        status_code: int = 400,
        error_type: str = "invalid_request_error",
        param: str | None = None,
        code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_type = error_type
        self.param = param
        self.code = code

    def body(self) -> dict[str, dict[str, str | None]]:
        return {
            "error": {
                "message": self.message,
                "type": self.error_type,
                "param": self.param,
                "code": self.code,
            }
        }


@dataclass(frozen=True)
class TranscriptionParameters:
    model: str
    response_format: str = "json"
    language: str | None = None
    timestamp_granularities: tuple[str, ...] = ()


@dataclass(frozen=True)
class RenderedOpenAIResponse:
    content: dict[str, Any] | str
    media_type: str


def canonical_model_id(variant: str, reference: ArtifactReference | None = None) -> str:
    if reference is not None and reference.repo_id:
        return reference.repo_id.rsplit("/", 1)[-1]
    metadata = VARIANTS.get(variant)
    if metadata:
        return str(metadata["repo_id"]).rsplit("/", 1)[-1]
    return f"gigaam-multilingual-mlx-{variant}"


def accepted_model_ids(variant: str, reference: ArtifactReference | None = None) -> tuple[str, str]:
    return canonical_model_id(variant, reference), COMPATIBILITY_ALIAS


def model_object(
    model_id: str,
    *,
    variant: str,
    reference: ArtifactReference | None = None,
) -> dict[str, Any]:
    canonical = canonical_model_id(variant, reference)
    repo_id = (
        reference.repo_id if reference is not None else VARIANTS.get(variant, {}).get("repo_id")
    )
    revision = (
        reference.revision if reference is not None else VARIANTS.get(variant, {}).get("revision")
    )
    return {
        "id": model_id,
        "object": "model",
        "created": 0,
        "owned_by": "ai-babai",
        "metadata": {
            "canonical_model": canonical,
            "runtime": "gigaam-multilingual-mlx",
            "variant": variant,
            "repo_id": repo_id,
            "revision": revision,
            "compatibility_alias": model_id == COMPATIBILITY_ALIAS,
        },
    }


def _parse_zero_temperature(value: str | float | int | None) -> None:
    if value in (None, ""):
        return
    try:
        temperature = float(value)
    except (TypeError, ValueError) as error:
        raise OpenAIAPIError(
            "temperature must be a number and only 0 is supported",
            param="temperature",
            code="invalid_value",
        ) from error
    if temperature != 0:
        raise OpenAIAPIError(
            "Only temperature=0 is supported by the deterministic GigaAM CTC decoder",
            param="temperature",
            code="unsupported_value",
        )


def _parse_false_stream(value: str | bool | None) -> None:
    if value in (None, "", False, "false", "False", "0", 0):
        return
    if value in (True, "true", "True", "1", 1):
        raise OpenAIAPIError(
            "stream=true is not supported; use one bounded request per file",
            param="stream",
            code="unsupported_value",
        )
    raise OpenAIAPIError(
        "stream must be true or false",
        param="stream",
        code="invalid_value",
    )


def validate_transcription_parameters(
    *,
    model: str | None,
    variant: str,
    reference: ArtifactReference | None = None,
    response_format: str | None = None,
    language: str | None = None,
    timestamp_granularities: list[str] | tuple[str, ...] | None = None,
    temperature: str | float | int | None = None,
    prompt: str | None = None,
    stream: str | bool | None = None,
) -> TranscriptionParameters:
    if not model:
        raise OpenAIAPIError(
            "The model field is required",
            status_code=422,
            param="model",
            code="missing_required_parameter",
        )
    accepted = accepted_model_ids(variant, reference)
    if model not in accepted:
        raise OpenAIAPIError(
            f"Model {model!r} is not served by this process; choose {accepted[0]!r} or "
            f"{COMPATIBILITY_ALIAS!r}",
            status_code=404,
            error_type="not_found_error",
            param="model",
            code="model_not_found",
        )
    selected_format = response_format or "json"
    if selected_format not in OPENAI_RESPONSE_FORMATS:
        raise OpenAIAPIError(
            f"Unsupported response_format {selected_format!r}; choose from "
            f"{', '.join(OPENAI_RESPONSE_FORMATS)}",
            param="response_format",
            code="unsupported_value",
        )
    if language and language not in SUPPORTED_LANGUAGES:
        raise OpenAIAPIError(
            f"Unsupported language {language!r}; choose from {', '.join(SUPPORTED_LANGUAGES)}",
            param="language",
            code="unsupported_value",
        )
    granularities = tuple(timestamp_granularities or ())
    invalid = [item for item in granularities if item not in SUPPORTED_TIMESTAMP_GRANULARITIES]
    if invalid:
        raise OpenAIAPIError(
            f"Unsupported timestamp granularity {invalid[0]!r}; choose word or segment",
            param="timestamp_granularities",
            code="unsupported_value",
        )
    if granularities and selected_format != "verbose_json":
        raise OpenAIAPIError(
            "timestamp_granularities requires response_format=verbose_json",
            param="timestamp_granularities",
            code="invalid_parameter_combination",
        )
    if prompt and prompt.strip():
        raise OpenAIAPIError(
            "Non-empty prompt conditioning is not supported by GigaAM CTC",
            param="prompt",
            code="unsupported_value",
        )
    _parse_zero_temperature(temperature)
    _parse_false_stream(stream)
    return TranscriptionParameters(
        model=model,
        response_format=selected_format,
        language=language,
        timestamp_granularities=granularities,
    )


def render_openai_response(
    result: dict[str, Any], parameters: TranscriptionParameters
) -> RenderedOpenAIResponse:
    response_format = parameters.response_format
    if response_format == "json":
        return RenderedOpenAIResponse({"text": str(result["text"])}, "application/json")
    if response_format in {"text", "srt", "vtt"}:
        internal_format = "txt" if response_format == "text" else response_format
        media_type = {
            "text": "text/plain; charset=utf-8",
            "srt": "application/x-subrip; charset=utf-8",
            "vtt": "text/vtt; charset=utf-8",
        }[response_format]
        return RenderedOpenAIResponse(
            render_transcription(result, internal_format),
            media_type,
        )

    duration = float(result["duration_seconds"])
    words = [
        {
            "word": str(word["text"]),
            "start": float(word["start"]),
            "end": float(word["end"]),
        }
        for word in result.get("words", [])
    ]
    segments = [
        {
            "id": index,
            "start": float(cue["start"]),
            "end": float(cue["end"]),
            "text": str(cue["text"]),
        }
        for index, cue in enumerate(transcription_cues(result.get("words", []), duration))
    ]
    if not segments and str(result.get("text", "")).strip():
        segments = [
            {
                "id": 0,
                "start": 0.0,
                "end": duration,
                "text": str(result["text"]).strip(),
            }
        ]
    return RenderedOpenAIResponse(
        {
            "task": "transcribe",
            "language": parameters.language or "und",
            "duration": duration,
            "text": str(result["text"]),
            "words": words,
            "segments": segments,
        },
        "application/json",
    )
