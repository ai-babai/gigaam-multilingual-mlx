#!/usr/bin/env python3
"""Prepare pinned FLEURS test manifests for the multilingual benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


DATASET_ID = "google/fleurs"
LICENSE = "CC BY 4.0"
SAMPLE_RATE = 16_000
MAX_DURATION_SECONDS = 30.0
SELECTION_RULE = "test split; exclude audio >30s and references containing digits"
SCHEMA = "gigaam-mlx-fleurs-multilingual/v1"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _reference(value: str) -> str:
    return unicodedata.normalize("NFC", value).strip()


def prepare(parquet_path: Path, output_root: Path, language: str, revision: str) -> dict[str, Any]:
    parquet_path = parquet_path.resolve()
    output_root = output_root.resolve()
    audio_root = output_root / "audio"
    audio_root.mkdir(parents=True, exist_ok=True)

    included: list[dict[str, Any]] = []
    exclusions = {"duration_gt_30_seconds": 0, "reference_contains_digits": 0}
    source_items = 0
    source_duration = 0.0
    parquet = pq.ParquetFile(parquet_path)
    columns = ["id", "num_samples", "audio", "transcription", "raw_transcription", "language"]

    for batch in parquet.iter_batches(batch_size=16, columns=columns):
        for item in batch.to_pylist():
            source_row = source_items
            source_items += 1
            duration = int(item["num_samples"]) / SAMPLE_RATE
            source_duration += duration
            reference = _reference(str(item["transcription"]))
            if duration > MAX_DURATION_SECONDS:
                exclusions["duration_gt_30_seconds"] += 1
                continue
            if any(character.isdigit() for character in reference):
                exclusions["reference_contains_digits"] += 1
                continue

            audio_bytes = item["audio"]["bytes"]
            sample_id = f"fleurs-{language}-test-row-{source_row:05d}"
            audio_path = audio_root / f"{sample_id}.wav"
            if not audio_path.exists() or _sha256_file(audio_path) != _sha256_bytes(audio_bytes):
                audio_path.write_bytes(audio_bytes)
            included.append(
                {
                    "id": sample_id,
                    "source": DATASET_ID,
                    "source_revision": revision,
                    "source_config": language,
                    "source_row": source_row,
                    "source_id": int(item["id"]),
                    "split": "test",
                    "language": language,
                    "audio_sha256": _sha256_bytes(audio_bytes),
                    "duration_seconds": duration,
                    "sample_rate": SAMPLE_RATE,
                    "channels": 1,
                    "reference_text": reference,
                    "raw_reference_text": _reference(str(item["raw_transcription"])),
                    "license": LICENSE,
                    "selection_rule": SELECTION_RULE,
                    "local_path": f"audio/{audio_path.name}",
                }
            )

    manifest = output_root / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in included),
        encoding="utf-8",
    )
    summary = {
        "schema": SCHEMA,
        "dataset_id": DATASET_ID,
        "source_revision": revision,
        "source_config": language,
        "split": "test",
        "license": LICENSE,
        "selection_rule": SELECTION_RULE,
        "source_parquet": parquet_path.name,
        "source_parquet_sha256": _sha256_file(parquet_path),
        "source_items": source_items,
        "source_duration_seconds": source_duration,
        "included_items": len(included),
        "included_duration_seconds": sum(row["duration_seconds"] for row in included),
        "excluded_items": source_items - len(included),
        "exclusions": exclusions,
        "manifest": manifest.name,
        "manifest_sha256": _sha256_file(manifest),
    }
    (output_root / "selection.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--language", required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    summary = prepare(args.parquet, args.output_root, args.language, args.revision)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
