from __future__ import annotations

import json

from gigaam_mlx.outputs import render_transcription


RESULT = {
    "text": "привет мир",
    "duration_seconds": 2.0,
    "words": [
        {"text": "привет", "start": 0.0, "end": 0.8},
        {"text": "мир", "start": 1.0, "end": 1.8},
    ],
    "model": {"variant": "int8", "revision": "v0.1.0"},
}


def test_txt_output() -> None:
    assert render_transcription(RESULT, "txt") == "привет мир\n"


def test_json_output_preserves_model_metadata() -> None:
    parsed = json.loads(render_transcription(RESULT, "json"))
    assert parsed["model"] == {"variant": "int8", "revision": "v0.1.0"}
    assert parsed["words"][1]["end"] <= parsed["duration_seconds"]


def test_srt_output_is_valid_and_monotonic() -> None:
    rendered = render_transcription(RESULT, "srt")
    assert rendered.startswith("1\n00:00:00,000 --> 00:00:01,800\n")
    assert rendered.endswith("привет мир\n")


def test_vtt_output_has_header() -> None:
    rendered = render_transcription(RESULT, "vtt")
    assert rendered.startswith("WEBVTT\n\n00:00:00.000 --> 00:00:01.800\n")


def test_empty_words_fall_back_to_full_duration_cue() -> None:
    result = {**RESULT, "words": []}
    rendered = render_transcription(result, "srt")
    assert "00:00:00,000 --> 00:00:02,000" in rendered
