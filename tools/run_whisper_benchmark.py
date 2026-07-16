#!/usr/bin/env python3
"""Run an isolated MLX Whisper performance measurement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gigaam_multilingual_mlx.whisper_benchmark import benchmark_whisper


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--language", required=True)
    parser.add_argument("--warm-runs", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = benchmark_whisper(
        args.model,
        args.audio,
        language=None if args.language == "auto" else args.language,
        warm_runs=args.warm_runs,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {"warm_summary": report["warm_summary"], "peak_rss_bytes": report["peak_rss_bytes"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
