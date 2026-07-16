from __future__ import annotations

import json
import time
import unicodedata
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from typing import Any, Callable

from jiwer import process_characters, process_words


NORMALIZATION_VERSION = "gigaam-multilingual-mlx-text-v2"


def normalize_multilingual(text: str, language: str) -> str:
    text = unicodedata.normalize("NFC", text).lower()
    if language == "ru_ru":
        text = text.replace("ё", "е")
    for quote in ("’", "‘", "ʻ", "ʼ", "`"):
        text = text.replace(quote, "'")
    output: list[str] = []
    for index, character in enumerate(text):
        if character == "'":
            if (
                0 < index < len(text) - 1
                and text[index - 1].isalnum()
                and text[index + 1].isalnum()
            ):
                output.append(character)
        elif character.isalnum() or character.isspace():
            output.append(character)
        elif unicodedata.category(character) == "Pd":
            output.append(" ")
    return " ".join("".join(output).split())


def manifest_rows(manifest: str | Path, limit: int | None = None) -> tuple[Path, list[dict]]:
    path = Path(manifest).resolve()
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return path.parent, rows[:limit]


def quality_report(
    rows: list[dict],
    hypotheses: list[str],
    *,
    language: str,
    elapsed: float,
    backend: dict[str, Any],
) -> dict[str, Any]:
    references = [normalize_multilingual(row["reference_text"], language) for row in rows]
    normalized_hypotheses = [normalize_multilingual(text, language) for text in hypotheses]
    aggregate = process_words(references, normalized_hypotheses)
    aggregate_characters = process_characters(references, normalized_hypotheses)
    items = []
    for row, reference, hypothesis, raw_hypothesis in zip(
        rows, references, normalized_hypotheses, hypotheses, strict=True
    ):
        word_score = process_words(reference, hypothesis)
        character_score = process_characters(reference, hypothesis)
        items.append(
            {
                "id": row["id"],
                "duration_seconds": row["duration_seconds"],
                "reference": reference,
                "hypothesis": hypothesis,
                "raw_hypothesis": raw_hypothesis,
                "wer": word_score.wer,
                "cer": character_score.cer,
                "substitutions": word_score.substitutions,
                "deletions": word_score.deletions,
                "insertions": word_score.insertions,
                "hits": word_score.hits,
                "exact_match": reference == hypothesis,
            }
        )
    duration = sum(float(row["duration_seconds"]) for row in rows)
    return {
        "schema_version": 1,
        "benchmark_suite_version": "public-asr-multilingual-v1",
        "normalization_version": NORMALIZATION_VERSION,
        "language": language,
        "backend": backend,
        "summary": {
            "files": len(rows),
            "audio_seconds": duration,
            "wall_seconds": elapsed,
            "audio_seconds_per_second": duration / elapsed,
            "wer": aggregate.wer,
            "wer_percent": aggregate.wer * 100,
            "cer": aggregate_characters.cer,
            "cer_percent": aggregate_characters.cer * 100,
            "substitutions": aggregate.substitutions,
            "deletions": aggregate.deletions,
            "insertions": aggregate.insertions,
            "hits": aggregate.hits,
            "empty_hypotheses": sum(not hypothesis for hypothesis in normalized_hypotheses),
            "exact_matches": sum(item["exact_match"] for item in items),
        },
        "items": items,
    }


def evaluate_existing(
    evaluator: Callable[..., dict],
    model_path: str | Path,
    manifest: str | Path,
    *,
    language: str,
    limit: int | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    _, rows = manifest_rows(manifest, limit)
    started = time.perf_counter()
    raw = evaluator(model_path, manifest, limit=limit, **kwargs)
    elapsed = time.perf_counter() - started
    hypotheses = [item["raw_hypothesis"] for item in raw["items"]]
    backend = {**raw["backend"], "source_normalization": raw["normalization_version"]}
    return quality_report(rows, hypotheses, language=language, elapsed=elapsed, backend=backend)


def evaluate_whisper(
    model: str,
    manifest: str | Path,
    *,
    language: str,
    whisper_language: str | None,
    limit: int | None = None,
) -> dict[str, Any]:
    import mlx_whisper

    root, rows = manifest_rows(manifest, limit)
    hypotheses: list[str] = []
    detected_languages: Counter[str] = Counter()
    started = time.perf_counter()
    for row in rows:
        options: dict[str, Any] = {
            "path_or_hf_repo": model,
            "task": "transcribe",
            "temperature": 0.0,
            "beam_size": None,
            "best_of": None,
            "fp16": True,
            "without_timestamps": True,
            "word_timestamps": False,
            "condition_on_previous_text": False,
            "compression_ratio_threshold": None,
            "logprob_threshold": None,
            "no_speech_threshold": None,
            "verbose": None,
        }
        if whisper_language is not None:
            options["language"] = whisper_language
        result = mlx_whisper.transcribe(str(root / row["local_path"]), **options)
        hypotheses.append(result["text"])
        detected_languages[str(result["language"])] += 1
    elapsed = time.perf_counter() - started
    return quality_report(
        rows,
        hypotheses,
        language=language,
        elapsed=elapsed,
        backend={
            "runtime": "mlx-whisper",
            "model": model,
            "whisper_language": whisper_language or "auto",
            "detected_languages": dict(sorted(detected_languages.items())),
            "task": "transcribe",
            "decoding": "greedy-temperature-0",
            "timestamps": False,
            "mlx": version("mlx"),
            "mlx_whisper": version("mlx-whisper"),
        },
    )


def evaluate_parakeet(
    model: str,
    manifest: str | Path,
    *,
    language: str,
    model_name: str,
    model_revision: str,
    limit: int | None = None,
) -> dict[str, Any]:
    import mlx.core as mx
    from parakeet_mlx import from_pretrained

    root, rows = manifest_rows(manifest, limit)
    loaded = from_pretrained(model, dtype=mx.bfloat16)
    mx.eval(loaded.parameters())
    hypotheses: list[str] = []
    started = time.perf_counter()
    for row in rows:
        result = loaded.transcribe(
            root / row["local_path"],
            dtype=mx.bfloat16,
            chunk_duration=None,
        )
        hypotheses.append(result.text)
    elapsed = time.perf_counter() - started
    return quality_report(
        rows,
        hypotheses,
        language=language,
        elapsed=elapsed,
        backend={
            "runtime": "parakeet-mlx",
            "model": model_name,
            "model_revision": model_revision,
            "language": "auto",
            "decoding": "greedy-tdt",
            "dtype": "bfloat16",
            "chunk_duration_seconds": None,
            "timestamps": True,
            "mlx": version("mlx"),
            "parakeet_mlx": version("parakeet-mlx"),
        },
    )
