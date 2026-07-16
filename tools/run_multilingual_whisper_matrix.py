#!/usr/bin/env python3
"""Run several language slices while reusing one loaded MLX Whisper model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gigaam_multilingual_mlx.multilingual_benchmark import evaluate_whisper


def _slice(value: str) -> tuple[str, str, Path]:
    parts = value.split("=", 2)
    if len(parts) != 3:
        raise ValueError("--slice must be LANGUAGE=WHISPER_LANGUAGE=MANIFEST")
    return parts[0], parts[1], Path(parts[2]).resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--slice", action="append", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for language, whisper_language, manifest in map(_slice, args.slice):
        report = evaluate_whisper(
            args.model,
            manifest,
            language=language,
            whisper_language=None if whisper_language == "auto" else whisper_language,
            limit=args.limit,
        )
        output = args.output_dir / f"{args.model_name}-{language}.json"
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(
            json.dumps(
                {"model": args.model_name, "language": language, **report["summary"]},
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
