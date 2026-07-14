#!/usr/bin/env python3
"""Build deterministic public-only short/medium/long performance WAV files."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import wave
from pathlib import Path
from typing import Any


SAMPLE_RATE = 16_000
SAMPLE_WIDTH = 2
CHANNELS = 1
SILENCE_SECONDS = 0.25
TARGETS = {"short": 30, "medium": 300, "long": 1200}
ALGORITHM = "sha256-merge-prefix-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest(value: str) -> tuple[str, Path, list[dict[str, Any]]]:
    if "=" not in value:
        raise ValueError("--manifest must be DATASET_ID=/path/to/manifest.jsonl")
    dataset_id, raw_path = value.split("=", 1)
    path = Path(raw_path).resolve()
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for row in rows:
        row["dataset_id"] = dataset_id
        row["manifest_root"] = str(path.parent)
    return dataset_id, path, rows


def _decode_s16le(path: Path, ffmpeg: str) -> bytes:
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


def _write_wav(path: Path, pcm: bytes) -> None:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(CHANNELS)
        output.setsampwidth(SAMPLE_WIDTH)
        output.setframerate(SAMPLE_RATE)
        output.writeframes(pcm)


def build(manifests: list[str], output_root: Path) -> dict[str, Any]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required")
    parsed = [_manifest(item) for item in manifests]
    rows = [row for _, _, dataset_rows in parsed for row in dataset_rows]
    rows.sort(key=lambda row: (row["audio_sha256"], row["dataset_id"], row["id"]))
    output_root.mkdir(parents=True, exist_ok=True)
    silence = b"\x00" * round(SILENCE_SECONDS * SAMPLE_RATE) * SAMPLE_WIDTH
    profiles: dict[str, Any] = {}
    for profile, target_seconds in TARGETS.items():
        target_bytes = target_seconds * SAMPLE_RATE * SAMPLE_WIDTH
        parts: list[bytes] = []
        selected = []
        length = 0
        for row in rows:
            source = Path(row["manifest_root"]) / row["local_path"]
            actual_hash = _sha256(source)
            if actual_hash != row["audio_sha256"]:
                raise ValueError(f"Audio hash mismatch for {row['id']}")
            pcm = _decode_s16le(source, ffmpeg)
            if parts:
                parts.append(silence)
                length += len(silence)
            remaining = target_bytes - length
            used = pcm[:remaining]
            parts.append(used)
            length += len(used)
            selected.append(
                {
                    "dataset_id": row["dataset_id"],
                    "source_revision": row["source_revision"],
                    "sample_id": row["id"],
                    "audio_sha256": row["audio_sha256"],
                    "used_samples": len(used) // SAMPLE_WIDTH,
                }
            )
            if length >= target_bytes:
                break
        if length != target_bytes:
            raise ValueError(f"Not enough source audio for {profile}: {length} != {target_bytes}")
        output = output_root / f"public-asr-perf-v1-{profile}.wav"
        _write_wav(output, b"".join(parts))
        profiles[profile] = {
            "duration_seconds": target_seconds,
            "wav_sha256": _sha256(output),
            "sample_rate": SAMPLE_RATE,
            "channels": CHANNELS,
            "sample_width_bytes": SAMPLE_WIDTH,
            "silence_seconds_between_items": SILENCE_SECONDS,
            "items": selected,
        }
    return {
        "schema_version": 1,
        "benchmark_suite_version": "public-asr-perf-v1",
        "selection_algorithm": ALGORITHM,
        "datasets": [
            {
                "dataset_id": dataset_id,
                "source_revision": sorted({row["source_revision"] for row in rows})[0],
                "manifest_sha256": _sha256(path),
            }
            for dataset_id, path, rows in parsed
        ],
        "profiles": profiles,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    args = parser.parse_args()
    recipe = build(args.manifest, args.output_root.resolve())
    args.recipe.parent.mkdir(parents=True, exist_ok=True)
    args.recipe.write_text(json.dumps(recipe, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
