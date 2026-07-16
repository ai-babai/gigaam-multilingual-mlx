#!/usr/bin/env python3
"""Build one deterministic 5-minute performance WAV per FLEURS language."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import wave
from pathlib import Path


SAMPLE_RATE = 16_000
SAMPLE_WIDTH = 2
CHANNELS = 1
SILENCE_SECONDS = 0.25
TARGET_SECONDS = 300
ALGORITHM = "audio-sha256-prefix-with-250ms-silence-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _decode(path: Path, ffmpeg: str) -> bytes:
    result = subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-nostdin",
            "-i",
            str(path),
            "-f",
            "s16le",
            "-ac",
            "1",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(SAMPLE_RATE),
            "-",
        ],
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace"))
    return result.stdout


def build(manifest: Path, language: str, output: Path) -> dict:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required")
    manifest = manifest.resolve()
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    rows.sort(key=lambda row: (row["audio_sha256"], row["id"]))
    target_bytes = TARGET_SECONDS * SAMPLE_RATE * SAMPLE_WIDTH
    silence = b"\x00" * round(SILENCE_SECONDS * SAMPLE_RATE) * SAMPLE_WIDTH
    parts: list[bytes] = []
    selected = []
    length = 0
    for row in rows:
        source = manifest.parent / row["local_path"]
        if _sha256(source) != row["audio_sha256"]:
            raise ValueError(f"Audio hash mismatch for {row['id']}")
        pcm = _decode(source, ffmpeg)
        if parts:
            parts.append(silence)
            length += len(silence)
        remaining = target_bytes - length
        used = pcm[:remaining]
        parts.append(used)
        length += len(used)
        selected.append(
            {
                "sample_id": row["id"],
                "audio_sha256": row["audio_sha256"],
                "used_samples": len(used) // SAMPLE_WIDTH,
            }
        )
        if length >= target_bytes:
            break
    if length != target_bytes:
        raise ValueError(f"Not enough audio: {length} != {target_bytes}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output), "wb") as stream:
        stream.setnchannels(CHANNELS)
        stream.setsampwidth(SAMPLE_WIDTH)
        stream.setframerate(SAMPLE_RATE)
        stream.writeframes(b"".join(parts))
    return {
        "schema_version": 1,
        "benchmark_suite_version": "public-asr-multilingual-v1",
        "language": language,
        "selection_algorithm": ALGORITHM,
        "source_revision": rows[0]["source_revision"],
        "source_manifest_sha256": _sha256(manifest),
        "duration_seconds": TARGET_SECONDS,
        "sample_rate": SAMPLE_RATE,
        "channels": CHANNELS,
        "sample_width_bytes": SAMPLE_WIDTH,
        "silence_seconds_between_items": SILENCE_SECONDS,
        "wav_file": output.name,
        "wav_sha256": _sha256(output),
        "items": selected,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--language", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    args = parser.parse_args()
    recipe = build(args.manifest, args.language, args.output.resolve())
    args.recipe.parent.mkdir(parents=True, exist_ok=True)
    args.recipe.write_text(json.dumps(recipe, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: recipe[key] for key in ("language", "wav_sha256")}, indent=2))


if __name__ == "__main__":
    main()
