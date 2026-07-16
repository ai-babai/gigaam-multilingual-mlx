#!/usr/bin/env python3
"""Run one implementation/language slice of public-asr-multilingual-v1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gigaam_multilingual_mlx.evaluate import evaluate_mlx, evaluate_pytorch
from gigaam_multilingual_mlx.multilingual_benchmark import (
    evaluate_existing,
    evaluate_whisper,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation", choices=("mlx", "pytorch", "whisper"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--language", required=True)
    parser.add_argument("--whisper-language")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--source-device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.implementation == "whisper":
        if args.whisper_language is None:
            parser.error("--whisper-language is required for Whisper")
        report = evaluate_whisper(
            args.model,
            args.manifest,
            language=args.language,
            whisper_language=None if args.whisper_language == "auto" else args.whisper_language,
            limit=args.limit,
        )
    elif args.implementation == "mlx":
        report = evaluate_existing(
            evaluate_mlx,
            args.model,
            args.manifest,
            language=args.language,
            batch_size=args.batch_size,
            limit=args.limit,
        )
    else:
        report = evaluate_existing(
            evaluate_pytorch,
            args.model,
            args.manifest,
            language=args.language,
            batch_size=args.batch_size,
            device=args.source_device,
            limit=args.limit,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
