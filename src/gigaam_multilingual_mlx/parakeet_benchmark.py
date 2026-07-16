from __future__ import annotations

import importlib.metadata
import platform
import statistics
import threading
import time
import wave
from pathlib import Path
from typing import Any

import mlx.core as mx
import psutil

from .config import sha256_file


def _duration(path: Path) -> float:
    with wave.open(str(path), "rb") as stream:
        return stream.getnframes() / stream.getframerate()


def _artifact(model_dir: Path) -> dict[str, Any]:
    weights = model_dir / "model.safetensors"
    return {
        "directory": model_dir.name,
        "weights_file": weights.name,
        "weights_sha256": sha256_file(weights),
        "weights_bytes": weights.stat().st_size,
        "artifact_bytes": sum(
            path.stat().st_size for path in model_dir.iterdir() if path.is_file()
        ),
    }


def benchmark_parakeet(
    model_dir: str | Path,
    audio_path: str | Path,
    *,
    model_name: str,
    model_revision: str,
    warm_runs: int = 5,
    chunk_duration: float = 120.0,
    overlap_duration: float = 15.0,
) -> dict[str, Any]:
    from parakeet_mlx import from_pretrained

    model_dir = Path(model_dir).resolve()
    audio_path = Path(audio_path).resolve()
    duration = _duration(audio_path)
    process = psutil.Process()
    samples: list[dict[str, int | float]] = []
    stop = threading.Event()

    def monitor() -> None:
        while not stop.wait(0.05):
            samples.append(
                {
                    "rss": process.memory_info().rss,
                    "available": psutil.virtual_memory().available,
                    "swap_used": psutil.swap_memory().used,
                }
            )

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    load_started = time.perf_counter()
    model = from_pretrained(str(model_dir), dtype=mx.bfloat16)
    mx.eval(model.parameters())
    load_seconds = time.perf_counter() - load_started

    runs = []
    labels = ["cold"] + [f"warm-{index}" for index in range(1, warm_runs + 1)]
    try:
        for label in labels:
            mx.reset_peak_memory()
            started = time.perf_counter()
            result = model.transcribe(
                audio_path,
                dtype=mx.bfloat16,
                chunk_duration=chunk_duration,
                overlap_duration=overlap_duration,
            )
            wall = time.perf_counter() - started
            runs.append(
                {
                    "label": label,
                    "wall_seconds": wall,
                    "rtf": wall / duration,
                    "audio_seconds_per_second": duration / wall,
                    "text": result.text,
                    "metal": {
                        "peak_bytes": mx.get_peak_memory(),
                        "active_bytes": mx.get_active_memory(),
                        "cache_bytes": mx.get_cache_memory(),
                    },
                }
            )
    finally:
        stop.set()
        thread.join(timeout=1)

    warm_times = [float(run["wall_seconds"]) for run in runs if run["label"].startswith("warm-")]
    sorted_warm = sorted(warm_times)
    p95_index = max(0, min(len(sorted_warm) - 1, int(len(sorted_warm) * 0.95 + 0.999) - 1))
    return {
        "schema_version": 1,
        "benchmark_suite_version": "public-asr-multilingual-v1",
        "implementation": "parakeet-mlx",
        "audio": {
            "file": audio_path.name,
            "sha256": sha256_file(audio_path),
            "duration_seconds": duration,
        },
        "artifact": _artifact(model_dir),
        "decoding": {
            "strategy": "greedy-tdt",
            "language": "auto",
            "dtype": "bfloat16",
            "chunk_duration_seconds": chunk_duration,
            "overlap_duration_seconds": overlap_duration,
            "timestamps": True,
        },
        "model": {"repository": model_name, "revision": model_revision},
        "environment": {
            "python": platform.python_version(),
            "mlx": importlib.metadata.version("mlx"),
            "parakeet_mlx": importlib.metadata.version("parakeet-mlx"),
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
            "median_rtf": statistics.median(warm_times) / duration,
            "median_audio_seconds_per_second": duration / statistics.median(warm_times),
        },
    }
