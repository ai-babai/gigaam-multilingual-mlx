from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any


OUTPUT_FORMATS = ("txt", "json", "srt", "vtt")


def _timestamp(seconds: float, *, separator: str) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}{separator}{milliseconds:03d}"


def transcription_cues(words: Iterable[dict[str, Any]], duration: float) -> list[dict[str, Any]]:
    rows = list(words)
    if not rows:
        return []
    cues: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    for word in rows:
        current.append(word)
        span = float(current[-1]["end"]) - float(current[0]["start"])
        ends_sentence = str(word["text"]).rstrip().endswith((".", "!", "?"))
        if len(current) >= 8 or span >= 5.0 or ends_sentence:
            cues.append(
                {
                    "start": max(0.0, float(current[0]["start"])),
                    "end": min(duration, float(current[-1]["end"])),
                    "text": " ".join(str(item["text"]) for item in current),
                }
            )
            current = []
    if current:
        cues.append(
            {
                "start": max(0.0, float(current[0]["start"])),
                "end": min(duration, float(current[-1]["end"])),
                "text": " ".join(str(item["text"]) for item in current),
            }
        )
    return cues


def render_transcription(result: dict[str, Any], output_format: str) -> str:
    if output_format not in OUTPUT_FORMATS:
        raise ValueError(f"Unsupported output format: {output_format}")
    if output_format == "txt":
        return str(result["text"]).strip() + "\n"
    if output_format == "json":
        return json.dumps(result, ensure_ascii=False, indent=2) + "\n"

    cues = transcription_cues(result.get("words", []), float(result["duration_seconds"]))
    if not cues and str(result.get("text", "")).strip():
        cues = [
            {
                "start": 0.0,
                "end": float(result["duration_seconds"]),
                "text": str(result["text"]).strip(),
            }
        ]
    lines = ["WEBVTT", ""] if output_format == "vtt" else []
    separator = "." if output_format == "vtt" else ","
    for index, cue in enumerate(cues, start=1):
        if output_format == "srt":
            lines.append(str(index))
        lines.append(
            f"{_timestamp(float(cue['start']), separator=separator)} --> "
            f"{_timestamp(float(cue['end']), separator=separator)}"
        )
        lines.extend((str(cue["text"]), ""))
    return "\n".join(lines).rstrip() + "\n"
