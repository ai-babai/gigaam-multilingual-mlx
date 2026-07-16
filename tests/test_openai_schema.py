from __future__ import annotations

import pytest

from gigaam_multilingual_mlx.openai_schema import (
    OpenAIAPIError,
    canonical_model_id,
    render_openai_response,
    validate_transcription_parameters,
)


RESULT = {
    "text": "привет мир",
    "duration_seconds": 2.0,
    "words": [
        {"text": "привет", "start": 0.0, "end": 0.8},
        {"text": "мир", "start": 1.0, "end": 1.8},
    ],
}


def test_canonical_model_id_matches_public_artifact() -> None:
    assert canonical_model_id("fp16") == "gigaam-multilingual-mlx"
    assert canonical_model_id("int8") == "gigaam-multilingual-mlx-int8-g64"


def test_whisper_alias_accepts_supported_compatibility_fields() -> None:
    parameters = validate_transcription_parameters(
        model="whisper-1",
        variant="int8",
        response_format="verbose_json",
        language="ru",
        timestamp_granularities=["word", "segment"],
        temperature="0",
        prompt="",
        stream="false",
    )
    assert parameters.model == "whisper-1"
    assert parameters.timestamp_granularities == ("word", "segment")


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("model", "gpt-4o-transcribe", "model_not_found"),
        ("language", "de", "unsupported_value"),
        ("temperature", "0.5", "unsupported_value"),
        ("prompt", "spell names carefully", "unsupported_value"),
        ("stream", "true", "unsupported_value"),
    ],
)
def test_unsupported_parameters_fail_explicitly(field: str, value: str, code: str) -> None:
    arguments = {"model": "whisper-1", "variant": "int8", field: value}
    with pytest.raises(OpenAIAPIError) as caught:
        validate_transcription_parameters(**arguments)
    assert caught.value.param == field
    assert caught.value.code == code


def test_timestamp_granularity_requires_verbose_json() -> None:
    with pytest.raises(OpenAIAPIError, match="requires response_format=verbose_json"):
        validate_transcription_parameters(
            model="whisper-1",
            variant="int8",
            timestamp_granularities=["word"],
        )


def test_verbose_json_uses_openai_word_shape_and_monotonic_segments() -> None:
    parameters = validate_transcription_parameters(
        model="whisper-1",
        variant="int8",
        response_format="verbose_json",
        language="ru",
    )
    rendered = render_openai_response(RESULT, parameters)
    assert isinstance(rendered.content, dict)
    assert rendered.content["words"][0] == {"word": "привет", "start": 0.0, "end": 0.8}
    assert rendered.content["segments"][0]["start"] == 0.0
    assert rendered.content["segments"][0]["end"] <= RESULT["duration_seconds"]


@pytest.mark.parametrize(
    ("response_format", "media_type", "needle"),
    [
        ("text", "text/plain", "привет мир"),
        ("srt", "application/x-subrip", "00:00:00,000"),
        ("vtt", "text/vtt", "WEBVTT"),
    ],
)
def test_text_response_formats(response_format: str, media_type: str, needle: str) -> None:
    parameters = validate_transcription_parameters(
        model="whisper-1",
        variant="int8",
        response_format=response_format,
    )
    rendered = render_openai_response(RESULT, parameters)
    assert rendered.media_type.startswith(media_type)
    assert needle in rendered.content
