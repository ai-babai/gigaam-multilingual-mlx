#!/usr/bin/env python3
"""Run supported-language Parakeet quality slices with one loaded model per slice."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gigaam_multilingual_mlx.multilingual_benchmark import evaluate_parakeet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-name", default="mlx-community/parakeet-tdt-0.6b-v3")
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--language", choices=("ru_ru", "en_us"), required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_parakeet(
        args.model,
        args.manifest,
        language=args.language,
        model_name=args.model_name,
        model_revision=args.model_revision,
        limit=args.limit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"language": args.language, **report["summary"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
