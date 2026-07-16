#!/usr/bin/env python3
"""Run an isolated Parakeet MLX performance measurement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gigaam_multilingual_mlx.parakeet_benchmark import benchmark_parakeet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-name", default="mlx-community/parakeet-tdt-0.6b-v3")
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--warm-runs", type=int, default=5)
    parser.add_argument("--chunk-duration", type=float, default=120.0)
    parser.add_argument("--overlap-duration", type=float, default=15.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = benchmark_parakeet(
        args.model,
        args.audio,
        model_name=args.model_name,
        model_revision=args.model_revision,
        warm_runs=args.warm_runs,
        chunk_duration=args.chunk_duration,
        overlap_duration=args.overlap_duration,
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
