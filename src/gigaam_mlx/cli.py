from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import psutil

from ._version import __version__
from .artifacts import (
    DEFAULT_VARIANT,
    VARIANTS,
    ArtifactReference,
    resolve_model_source,
    variant_rows,
)
from .audio import SAMPLE_RATE, fixed_chunks, load_audio
from .model import GigaAMCTC, load_model
from .outputs import OUTPUT_FORMATS, render_transcription


def _ensure_supported_platform() -> None:
    machine = platform.machine().lower()
    if sys.platform != "darwin" or machine not in {"arm64", "aarch64"}:
        raise RuntimeError("gigaam-mlx requires macOS on Apple Silicon (arm64)")
    version = platform.mac_ver()[0]
    if version and int(version.split(".", 1)[0]) < 14:
        raise RuntimeError("gigaam-mlx requires macOS 14 or newer")


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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gigaam-mlx",
        description="Transcribe audio locally with GigaAM-Multilingual on Apple Silicon.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    transcribe = sub.add_parser("transcribe", help="transcribe an audio or video file")
    transcribe.add_argument("audio", help="WAV, FLAC, MP3, M4A, or video input")
    transcribe.add_argument(
        "--variant", choices=tuple(VARIANTS), default=DEFAULT_VARIANT, help="model variant"
    )
    transcribe.add_argument("--model", help="local model directory or Hugging Face repo ID")
    transcribe.add_argument("--revision", help="Hugging Face revision or immutable tag")
    transcribe.add_argument("--cache-dir", help="custom Hugging Face cache directory")
    transcribe.add_argument("--offline", action="store_true", help="use cached/local files only")
    transcribe.add_argument("--dtype", choices=("float32", "float16", "bfloat16"))
    transcribe.add_argument("--chunk-seconds", type=float, default=20.0)
    transcribe.add_argument("--overlap-seconds", type=float, default=2.0)
    transcribe.add_argument("--compile", action="store_true")
    transcribe.add_argument(
        "--format",
        choices=OUTPUT_FORMATS,
        help="output format; inferred from --output suffix, otherwise txt",
    )
    output = transcribe.add_mutually_exclusive_group()
    output.add_argument("--output", help="exact output file path")
    output.add_argument("--output-dir", help="directory for INPUT_STEM.{txt,json,srt,vtt}")

    models = sub.add_parser("models", help="list supported model variants")
    models.add_argument("--json", action="store_true", help="print machine-readable JSON")
    return parser


def _selected_format(args: argparse.Namespace) -> str:
    if args.format:
        return str(args.format)
    if args.output:
        suffix = Path(args.output).suffix.lstrip(".").lower()
        if suffix in OUTPUT_FORMATS:
            return suffix
    return "txt"


def _output_path(args: argparse.Namespace, output_format: str) -> Path | None:
    if args.output:
        path = Path(args.output).expanduser()
        suffix = path.suffix.lstrip(".").lower()
        if suffix in OUTPUT_FORMATS and suffix != output_format:
            raise ValueError(
                f"Output suffix .{suffix} conflicts with --format {output_format}; "
                "choose matching values"
            )
        return path
    if args.output_dir:
        directory = Path(args.output_dir).expanduser()
        return directory / f"{Path(args.audio).stem}.{output_format}"
    return None


def _models_output(as_json: bool) -> str:
    rows = variant_rows()
    if as_json:
        return json.dumps(rows, ensure_ascii=False, indent=2) + "\n"
    lines = ["VARIANT  ROLE                         SIZE       REPOSITORY"]
    for row in rows:
        size = f"{int(row['expected_bytes']) / 1_000_000:.0f} MB"
        lines.append(
            f"{str(row['variant']):<8} {str(row['label']):<28} {size:<10} {row['repo_id']}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    try:
        if args.command == "models":
            print(_models_output(args.json), end="")
            return
        _ensure_supported_platform()
        reference = resolve_model_source(
            args.model,
            variant=args.variant,
            revision=args.revision,
            cache_dir=args.cache_dir,
            local_files_only=args.offline,
        )
        location = (
            f"{reference.repo_id}@{reference.revision}"
            if reference.repo_id
            else str(reference.path)
        )
        print(f"Using {reference.variant} model from {location}", file=sys.stderr)
        print(f"Model files: {reference.path}", file=sys.stderr)
        mx.reset_peak_memory()
        model = load_model(reference.path, args.dtype)
        runner = mx.compile(model.__call__, inputs=model.state) if args.compile else None
        result = transcribe_file(
            model,
            args.audio,
            args.chunk_seconds,
            args.overlap_seconds,
            runner=runner,
            artifact=reference,
        )
        output_format = _selected_format(args)
        rendered = render_transcription(result, output_format)
        output_path = _output_path(args, output_format)
        if output_path is None:
            print(rendered, end="")
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(rendered)
            print(f"Wrote {output_format.upper()} to {output_path}", file=sys.stderr)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        parser.exit(2, f"gigaam-mlx: error: {error}\n")


if __name__ == "__main__":
    main()
