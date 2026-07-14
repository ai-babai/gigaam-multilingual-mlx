#!/usr/bin/env python3
"""Export portable public quality manifests without machine-local paths."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


FIELDS = (
    "id",
    "source",
    "source_revision",
    "source_row",
    "split",
    "audio_sha256",
    "duration_seconds",
    "sample_rate",
    "channels",
    "reference_text",
    "license",
    "selection_bucket",
    "selection_rule",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(dataset: str, source: Path, output: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    portable = [{key: row[key] for key in FIELDS if key in row} for row in rows]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in portable))
    return {
        "dataset_id": dataset,
        "source_revision": sorted({row["source_revision"] for row in rows})[0],
        "items": len(rows),
        "audio_seconds": sum(float(row["duration_seconds"]) for row in rows),
        "manifest": output.name,
        "manifest_sha256": _sha256(output),
        "licenses": sorted({row["license"] for row in rows}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = []
    for value in args.dataset:
        if "=" not in value:
            raise ValueError("--dataset must be DATASET_ID=/path/to/manifest.jsonl")
        dataset, raw_path = value.split("=", 1)
        safe_name = dataset.replace("/", "--") + ".jsonl"
        summary.append(
            export(dataset, Path(raw_path).resolve(), args.output_dir.resolve() / safe_name)
        )
    (args.output_dir.resolve() / "quality-manifests.json").write_text(
        json.dumps({"schema_version": 1, "datasets": summary}, ensure_ascii=False, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
