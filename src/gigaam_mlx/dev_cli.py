"""Developer-only conversion, validation, and benchmark commands.

Install the matching optional extras and run with ``python -m gigaam_mlx.dev_cli``.
These commands are intentionally not exposed by the production ``gigaam-mlx`` CLI.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import statistics
import threading
import time
from pathlib import Path

import mlx.core as mx
import psutil

from .config import sha256_file


def _implementation_id(manifest: dict) -> str:
    profile = manifest.get("quantization", {}).get("profile")
    if profile:
        return f"mlx-{profile}"
    dtype = str(manifest.get("dtype", "unknown")).replace("float", "fp")
    return f"mlx-{dtype}"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m gigaam_mlx.dev_cli")
    sub = parser.add_subparsers(dest="command", required=True)

    convert = sub.add_parser("convert")
    convert.add_argument("--source", required=True)
    convert.add_argument("--output", required=True)
    convert.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")

    quantize = sub.add_parser("quantize")
    quantize.add_argument("--source", required=True)
    quantize.add_argument("--output", required=True)
    quantize.add_argument("--profile", choices=("int8-g64", "int6-g64", "int4-g64"), required=True)

    parity = sub.add_parser("parity")
    parity.add_argument("--source", required=True)
    parity.add_argument("--model", required=True)
    parity.add_argument("--audio", action="append", required=True)
    parity.add_argument("--source-device", choices=("cpu", "mps"), default="mps")
    parity.add_argument("--output", required=True)

    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--manifest", required=True)
    evaluate.add_argument("--model")
    evaluate.add_argument("--source")
    evaluate.add_argument("--batch-size", type=int, default=8)
    evaluate.add_argument("--limit", type=int)
    evaluate.add_argument("--source-device", choices=("cpu", "mps"), default="mps")
    evaluate.add_argument("--output", required=True)

    benchmark = sub.add_parser("benchmark")
    benchmark.add_argument("audio")
    benchmark.add_argument("--model", required=True)
    benchmark.add_argument("--chunk-seconds", type=float, default=20.0)
    benchmark.add_argument("--overlap-seconds", type=float, default=2.0)
    benchmark.add_argument("--compile", action="store_true")
    benchmark.add_argument("--warm-runs", type=int, default=5)
    benchmark.add_argument("--output", required=True)

    pytorch_benchmark = sub.add_parser("benchmark-pytorch")
    pytorch_benchmark.add_argument("audio")
    pytorch_benchmark.add_argument("--source", required=True)
    pytorch_benchmark.add_argument("--source-device", choices=("cpu", "mps"), default="mps")
    pytorch_benchmark.add_argument("--chunk-seconds", type=float, default=20.0)
    pytorch_benchmark.add_argument("--overlap-seconds", type=float, default=2.0)
    pytorch_benchmark.add_argument("--warm-runs", type=int, default=5)
    pytorch_benchmark.add_argument("--output", required=True)

    compare_quality = sub.add_parser("compare-quality")
    compare_quality.add_argument("--report", action="append", required=True)
    compare_quality.add_argument("--output", required=True)
    return parser


def _benchmark(args: argparse.Namespace) -> dict:
    from .cli import transcribe_file
    from .model import load_model

    process = psutil.Process()
    samples: list[dict[str, int | float]] = []
    stop = threading.Event()

    def monitor() -> None:
        while not stop.wait(0.05):
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            samples.append(
                {
                    "time": time.perf_counter(),
                    "rss": process.memory_info().rss,
                    "available": memory.available,
                    "swap_used": swap.used,
                }
            )

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    load_started = time.perf_counter()
    model = load_model(args.model)
    mx.eval(model.parameters())
    load_seconds = time.perf_counter() - load_started
    runner = mx.compile(model.__call__, inputs=model.state) if args.compile else None
    runs = []
    labels = ["cold"] + [f"warm-{index}" for index in range(1, args.warm_runs + 1)]
    try:
        for label in labels:
            mx.reset_peak_memory()
            started = time.perf_counter()
            run = transcribe_file(
                model,
                args.audio,
                args.chunk_seconds,
                args.overlap_seconds,
                runner=runner,
            )
            run["end_to_end_wall_seconds"] = time.perf_counter() - started
            run["label"] = label
            run["metal"] = {
                "peak_bytes": mx.get_peak_memory(),
                "active_bytes": mx.get_active_memory(),
                "cache_bytes": mx.get_cache_memory(),
            }
            runs.append(run)
    finally:
        stop.set()
        thread.join(timeout=1)

    model_dir = Path(args.model)
    manifest = json.loads((model_dir / "manifest.json").read_text())
    artifact_bytes = sum(path.stat().st_size for path in model_dir.iterdir() if path.is_file())
    warm_times = [
        float(run["end_to_end_wall_seconds"])
        for run in runs
        if str(run["label"]).startswith("warm-")
    ]
    sorted_warm = sorted(warm_times)
    p95_index = max(0, min(len(sorted_warm) - 1, int(len(sorted_warm) * 0.95 + 0.999) - 1))
    return {
        "schema_version": 1,
        "implementation": _implementation_id(manifest),
        "audio": {
            "file": Path(args.audio).name,
            "sha256": sha256_file(args.audio),
            "duration_seconds": runs[0]["duration_seconds"] if runs else None,
        },
        "artifact": {
            "repository": "ai-babai/gigaam-multilingual-mlx"
            if not manifest.get("quantization")
            else f"ai-babai/gigaam-multilingual-mlx-{manifest['quantization']['profile']}",
            "revision": "v0.1.0",
            "model_name": manifest.get("model_name"),
            "dtype": manifest.get("dtype"),
            "quantization": manifest.get("quantization"),
            "source_repository": "ai-sage/GigaAM-Multilingual",
            "source_revision": manifest.get("source", {}).get("revision"),
            "weights_sha256": manifest["weights"]["sha256"],
            "bytes": artifact_bytes,
        },
        "compiled": args.compile,
        "decoding": {
            "strategy": "greedy_ctc",
            "timestamps": "word",
            "output_mode": "full_transcription",
        },
        "chunk_seconds": args.chunk_seconds,
        "overlap_seconds": args.overlap_seconds,
        "environment": {
            "python": platform.python_version(),
            "gigaam_mlx": importlib.metadata.version("gigaam-mlx"),
            "mlx": importlib.metadata.version("mlx"),
            "platform": platform.platform(),
            "device": mx.device_info(),
        },
        "load_seconds": load_seconds,
        "peak_rss_bytes": max(
            (sample["rss"] for sample in samples), default=process.memory_info().rss
        ),
        "peak_device_bytes": max((int(run["metal"]["peak_bytes"]) for run in runs), default=None),
        "peak_active_device_bytes": max(
            (int(run["metal"]["active_bytes"]) for run in runs), default=None
        ),
        "peak_cache_device_bytes": max(
            (int(run["metal"]["cache_bytes"]) for run in runs), default=None
        ),
        "minimum_available_memory_bytes": min(
            (sample["available"] for sample in samples), default=psutil.virtual_memory().available
        ),
        "swap_used_start_bytes": samples[0]["swap_used"] if samples else psutil.swap_memory().used,
        "swap_used_end_bytes": samples[-1]["swap_used"] if samples else psutil.swap_memory().used,
        "runs": runs,
        "warm_summary": {
            "runs": len(warm_times),
            "median_end_to_end_seconds": statistics.median(warm_times),
            "p95_end_to_end_seconds": sorted_warm[p95_index],
        },
    }


def main() -> None:
    args = _parser().parse_args()
    if args.command == "convert":
        from .convert import convert_checkpoint

        result = convert_checkpoint(args.source, args.output, args.dtype)
    elif args.command == "quantize":
        from .quantization import quantize_artifact

        result = quantize_artifact(args.source, args.output, args.profile)
    elif args.command == "parity":
        from .parity import run_parity

        result = run_parity(args.source, args.model, args.audio, args.source_device)
    elif args.command == "evaluate":
        from .evaluate import evaluate_mlx, evaluate_pytorch

        if bool(args.model) == bool(args.source):
            raise SystemExit("evaluate requires exactly one of --model or --source")
        result = (
            evaluate_mlx(args.model, args.manifest, args.batch_size, args.limit)
            if args.model
            else evaluate_pytorch(
                args.source, args.manifest, args.batch_size, args.limit, args.source_device
            )
        )
    elif args.command == "benchmark":
        result = _benchmark(args)
    elif args.command == "benchmark-pytorch":
        from .pytorch_benchmark import benchmark_pytorch

        result = benchmark_pytorch(
            args.source,
            args.audio,
            device=args.source_device,
            chunk_seconds=args.chunk_seconds,
            overlap_seconds=args.overlap_seconds,
            warm_runs=args.warm_runs,
        )
    else:
        from .reports import compare_quality_reports

        result = compare_quality_reports(args.report)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
