from __future__ import annotations

import importlib.metadata
import platform
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import psutil

from .artifacts import ArtifactReference
from .audio import SAMPLE_RATE, fixed_chunks, load_audio
from .model import GigaAMCTC


def _word_timestamps(
    model: GigaAMCTC, decoded: dict[str, object], audio_samples: int, encoded_length: int
) -> list[dict[str, object]]:
    ids = decoded["token_ids"]
    frames = decoded["token_frames"]
    assert isinstance(ids, list) and isinstance(frames, list)
    if encoded_length <= 0:
        return []
    shift = audio_samples / SAMPLE_RATE / encoded_length
    words: list[dict[str, object]] = []
    chars: list[str] = []
    word_frames: list[int] = []

    def commit() -> None:
        text = "".join(chars).strip()
        if text and word_frames:
            words.append(
                {
                    "text": text,
                    "start": word_frames[0] * shift,
                    "end": (word_frames[-1] + 1) * shift,
                }
            )
        chars.clear()
        word_frames.clear()

    for token, frame in zip(ids, frames, strict=True):
        char = model.config.vocabulary[token]
        if char == " ":
            commit()
        else:
            chars.append(char)
            word_frames.append(frame)
    commit()
    return words


def _monotonic_words(words: list[dict[str, object]], duration: float) -> list[dict[str, object]]:
    normalized: list[dict[str, object]] = []
    previous_end = 0.0
    for word in words:
        start = min(duration, max(previous_end, float(word["start"])))
        end = min(duration, max(start, float(word["end"])))
        normalized.append({"text": str(word["text"]), "start": start, "end": end})
        previous_end = end
    return normalized


def transcribe_file(
    model: GigaAMCTC,
    audio_path: str | Path,
    chunk_seconds: float = 20.0,
    overlap_seconds: float = 2.0,
    runner=None,
    artifact: ArtifactReference | None = None,
) -> dict[str, Any]:
    audio_path = Path(audio_path).expanduser()
    audio = load_audio(audio_path, model.config.sample_rate)
    started = time.perf_counter()
    chunks: list[dict[str, Any]] = []
    kept_words: list[dict[str, object]] = []
    chunk_ranges = list(fixed_chunks(audio, chunk_seconds, overlap_seconds))
    for index, (start, end, samples) in enumerate(chunk_ranges):
        inference_started = time.perf_counter()
        infer = runner or model
        logits, lengths = infer(mx.array(samples)[None, :], mx.array([len(samples)]))
        mx.eval(logits, lengths)
        inference_seconds = time.perf_counter() - inference_started
        decoded = model.greedy_decode(logits, lengths)[0]
        encoded_length = int(np.asarray(lengths)[0])
        words = _word_timestamps(model, decoded, len(samples), encoded_length)
        start_seconds = start / SAMPLE_RATE
        end_seconds = end / SAMPLE_RATE
        keep_start = start_seconds if index == 0 else start_seconds + overlap_seconds / 2
        keep_end = (
            end_seconds if index == len(chunk_ranges) - 1 else end_seconds - overlap_seconds / 2
        )
        global_words = []
        for word in words:
            item = {
                "text": word["text"],
                "start": start_seconds + float(word["start"]),
                "end": start_seconds + float(word["end"]),
            }
            midpoint = (float(item["start"]) + float(item["end"])) / 2
            item["kept"] = keep_start <= midpoint < keep_end or (
                index == len(chunk_ranges) - 1 and midpoint <= keep_end
            )
            global_words.append(item)
            if item["kept"]:
                kept_words.append({key: item[key] for key in ("text", "start", "end")})
        chunks.append(
            {
                "index": index,
                "start": start_seconds,
                "end": end_seconds,
                "inference_seconds": inference_seconds,
                "text": decoded["text"],
                "words": global_words,
                "memory": {
                    "rss_bytes": psutil.Process().memory_info().rss,
                    "metal_active_bytes": mx.get_active_memory(),
                    "metal_cache_bytes": mx.get_cache_memory(),
                },
            }
        )
        del logits, lengths
    elapsed = time.perf_counter() - started
    duration = len(audio) / SAMPLE_RATE
    kept_words = _monotonic_words(kept_words, duration)
    text = " ".join(str(word["text"]) for word in kept_words)
    if not kept_words:
        text = " ".join(str(chunk["text"]).strip() for chunk in chunks).strip()
    artifact_info = artifact or getattr(model, "artifact_reference", None)
    if isinstance(artifact_info, ArtifactReference):
        model_info = {
            "variant": artifact_info.variant,
            "repo_id": artifact_info.repo_id,
            "revision": artifact_info.revision,
            "source": artifact_info.source,
        }
    elif isinstance(artifact_info, dict):
        model_info = {
            key: artifact_info.get(key) for key in ("variant", "repo_id", "revision", "source")
        }
    else:
        model_info = {"variant": None, "repo_id": None, "revision": None, "source": "local"}
    return {
        "schema_version": 1,
        "audio": audio_path.name,
        "duration_seconds": duration,
        "text": text,
        "words": kept_words,
        "chunks": chunks,
        "model": model_info,
        "metrics": {
            "wall_seconds": elapsed,
            "rtf": elapsed / duration if duration else None,
            "audio_seconds_per_second": duration / elapsed if elapsed else None,
            "rss_bytes": psutil.Process().memory_info().rss,
            "metal_active_bytes": mx.get_active_memory(),
            "metal_cache_bytes": mx.get_cache_memory(),
            "metal_peak_bytes": mx.get_peak_memory(),
        },
        "environment": {
            "python": platform.python_version(),
            "mlx": importlib.metadata.version("mlx"),
            "platform": platform.platform(),
        },
    }
