#!/usr/bin/env python3
"""Export path-free FLEURS manifests for public-asr-multilingual-v1."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


FIELDS = (
    "id",
    "source",
    "source_revision",
    "source_config",
    "source_row",
    "source_id",
    "split",
    "language",
    "audio_sha256",
    "duration_seconds",
    "sample_rate",
    "channels",
    "reference_text",
    "raw_reference_text",
    "license",
    "selection_rule",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _slice(value: str) -> tuple[str, Path]:
    language, separator, raw_path = value.partition("=")
    if not separator:
        raise ValueError("--slice must be LANGUAGE=MANIFEST")
    return language, Path(raw_path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slice", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for language, source in map(_slice, args.slice):
        rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
        portable = [{key: row[key] for key in FIELDS if key in row} for row in rows]
        output = args.output_dir / f"google--fleurs-{language}.jsonl"
        output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in portable))
        summary.append(
            {
                "dataset_id": "google/fleurs",
                "config": language,
                "split": "test",
                "source_revision": rows[0]["source_revision"],
                "items": len(rows),
                "audio_seconds": sum(float(row["duration_seconds"]) for row in rows),
                "manifest": output.name,
                "manifest_sha256": _sha256(output),
                "license": sorted({row["license"] for row in rows}),
                "selection_rule": rows[0]["selection_rule"],
            }
        )
    (args.output_dir / "multilingual-manifests.json").write_text(
        json.dumps({"schema_version": 1, "datasets": summary}, ensure_ascii=False, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
