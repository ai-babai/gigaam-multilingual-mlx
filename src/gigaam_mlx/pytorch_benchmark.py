from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import platform
import statistics
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil

from .audio import SAMPLE_RATE, fixed_chunks, load_audio
from .config import sha256_file


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location("modeling_gigaam", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import source model module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_source_model(source_dir: Path, device: str):
    import omegaconf
    import torch

    raw = json.loads((source_dir / "config.json").read_text())
    module = _load_module(source_dir / "modeling_gigaam.py")
    model = module.GigaAMModel(module.GigaAMConfig(cfg=omegaconf.OmegaConf.create(raw["cfg"])))
    state = torch.load(source_dir / "pytorch_model.bin", map_location="cpu", weights_only=True)
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(f"Non-strict source checkpoint: {incompatible}")
    return model.eval().to(device)


def _decoded_words(
    vocabulary: list[str], token_ids: list[int], token_frames: list[int], shift: float
) -> list[dict[str, Any]]:
    words: list[dict[str, Any]] = []
    chars: list[str] = []
    frames: list[int] = []

    def commit() -> None:
        text = "".join(chars).strip()
        if text and frames:
            words.append(
                {
                    "text": text,
                    "start": frames[0] * shift,
                    "end": (frames[-1] + 1) * shift,
                }
            )
        chars.clear()
        frames.clear()

    for token, frame in zip(token_ids, token_frames, strict=True):
        character = vocabulary[token]
        if character == " ":
            commit()
        else:
            chars.append(character)
            frames.append(frame)
    commit()
    return words


def _transcribe_run(
    model,
    audio: np.ndarray,
    *,
    device: str,
    chunk_seconds: float,
    overlap_seconds: float,
) -> dict[str, Any]:
    import torch

    chunks = list(fixed_chunks(audio, chunk_seconds, overlap_seconds))
    kept_words: list[dict[str, Any]] = []
    rows = []
    vocabulary = list(model.model.decoding.tokenizer.vocab)
    started = time.perf_counter()
    for index, (start, end, samples) in enumerate(chunks):
        tensor = torch.from_numpy(samples)[None, :].to(device)
        lengths = torch.tensor([len(samples)], device=device)
        inference_started = time.perf_counter()
        with torch.inference_mode():
            encoded, encoded_lengths = model.model(tensor, lengths)
            decoded = model.model.decoding.decode(model.model.head, encoded, encoded_lengths)[0]
        if device == "mps":
            torch.mps.synchronize()
        inference_seconds = time.perf_counter() - inference_started
        encoded_length = int(encoded_lengths[0].item())
        shift = len(samples) / SAMPLE_RATE / encoded_length if encoded_length else 0.0
        words = _decoded_words(vocabulary, decoded[1], decoded[2], shift)
        start_seconds = start / SAMPLE_RATE
        end_seconds = end / SAMPLE_RATE
        keep_start = start_seconds if index == 0 else start_seconds + overlap_seconds / 2
        keep_end = end_seconds if index == len(chunks) - 1 else end_seconds - overlap_seconds / 2
        for word in words:
            item = {
                "text": word["text"],
                "start": start_seconds + float(word["start"]),
                "end": start_seconds + float(word["end"]),
            }
            midpoint = (item["start"] + item["end"]) / 2
            if keep_start <= midpoint < keep_end or (
                index == len(chunks) - 1 and midpoint <= keep_end
            ):
                kept_words.append(item)
        rows.append(
            {
                "index": index,
                "start": start_seconds,
                "end": end_seconds,
                "inference_seconds": inference_seconds,
                "text": decoded[0],
                "rss_bytes": psutil.Process().memory_info().rss,
                "mps_driver_allocated_bytes": torch.mps.driver_allocated_memory()
                if device == "mps"
                else None,
            }
        )
        del encoded, encoded_lengths, tensor, lengths
    wall = time.perf_counter() - started
    duration = len(audio) / SAMPLE_RATE
    return {
        "duration_seconds": duration,
        "wall_seconds": wall,
        "rtf": wall / duration,
        "audio_seconds_per_second": duration / wall,
        "text": " ".join(str(word["text"]) for word in kept_words),
        "word_count": len(kept_words),
        "chunks": rows,
    }


def benchmark_pytorch(
    source_dir: str | Path,
    audio_path: str | Path,
    *,
    device: str = "mps",
    chunk_seconds: float = 20.0,
    overlap_seconds: float = 2.0,
    warm_runs: int = 5,
) -> dict[str, Any]:
    import torch

    source_dir = Path(source_dir).resolve()
    audio_path = Path(audio_path).resolve()
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
    model = _load_source_model(source_dir, device)
    if device == "mps":
        torch.mps.synchronize()
    load_seconds = time.perf_counter() - load_started
    audio = load_audio(audio_path)
    runs = []
    try:
        for label in ["cold"] + [f"warm-{index}" for index in range(1, warm_runs + 1)]:
            run = _transcribe_run(
                model,
                audio,
                device=device,
                chunk_seconds=chunk_seconds,
                overlap_seconds=overlap_seconds,
            )
            run["label"] = label
            runs.append(run)
    finally:
        stop.set()
        thread.join(timeout=1)
    warm_times = [float(run["wall_seconds"]) for run in runs if run["label"].startswith("warm-")]
    sorted_warm = sorted(warm_times)
    p95_index = max(0, min(len(sorted_warm) - 1, int(len(sorted_warm) * 0.95 + 0.999) - 1))
    artifact_bytes = sum(path.stat().st_size for path in source_dir.iterdir() if path.is_file())
    return {
        "schema_version": 1,
        "implementation": "original-pytorch-mps" if device == "mps" else "original-pytorch-cpu",
        "audio": {
            "file": audio_path.name,
            "sha256": sha256_file(audio_path),
            "duration_seconds": len(audio) / SAMPLE_RATE,
        },
        "artifact": {
            "source_repository": "ai-sage/GigaAM-Multilingual",
            "source_revision": source_dir.name,
            "weights_sha256": sha256_file(source_dir / "pytorch_model.bin"),
            "bytes": artifact_bytes,
        },
        "decoding": {
            "strategy": "greedy_ctc",
            "timestamps": "word",
            "output_mode": "full_transcription",
        },
        "chunk_seconds": chunk_seconds,
        "overlap_seconds": overlap_seconds,
        "environment": {
            "python": platform.python_version(),
            "gigaam_mlx": importlib.metadata.version("gigaam-mlx"),
            "torch": importlib.metadata.version("torch"),
            "platform": platform.platform(),
            "device": device,
        },
        "load_seconds": load_seconds,
        "peak_rss_bytes": max(
            (int(sample["rss"]) for sample in samples), default=process.memory_info().rss
        ),
        "peak_device_bytes": max(
            (
                int(chunk["mps_driver_allocated_bytes"])
                for run in runs
                for chunk in run["chunks"]
                if chunk["mps_driver_allocated_bytes"] is not None
            ),
            default=None,
        ),
        "minimum_available_memory_bytes": min(
            (int(sample["available"]) for sample in samples),
            default=psutil.virtual_memory().available,
        ),
        "swap_used_start_bytes": int(samples[0]["swap_used"])
        if samples
        else psutil.swap_memory().used,
        "swap_used_end_bytes": int(samples[-1]["swap_used"])
        if samples
        else psutil.swap_memory().used,
        "runs": runs,
        "warm_summary": {
            "runs": len(warm_times),
            "median_end_to_end_seconds": statistics.median(warm_times),
            "p95_end_to_end_seconds": sorted_warm[p95_index],
        },
    }
