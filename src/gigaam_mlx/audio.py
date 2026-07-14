from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np


SAMPLE_RATE = 16_000


def load_audio(path: str | Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Audio or video input does not exist: {path}")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "ffmpeg is required to decode audio and video inputs; install it with "
            "`brew install ffmpeg`"
        )
    command = [
        ffmpeg,
        "-v",
        "error",
        "-nostdin",
        "-threads",
        "0",
        "-i",
        str(path),
        "-f",
        "s16le",
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        "-",
    ]
    result = subprocess.run(command, capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace").strip())
    return np.frombuffer(result.stdout, dtype="<i2").astype(np.float32) / 32768.0


def fixed_chunks(audio: np.ndarray, chunk_seconds: float, overlap_seconds: float):
    if chunk_seconds <= 0 or overlap_seconds < 0 or overlap_seconds >= chunk_seconds:
        raise ValueError("Require chunk_seconds > overlap_seconds >= 0")
    size = round(chunk_seconds * SAMPLE_RATE)
    overlap = round(overlap_seconds * SAMPLE_RATE)
    step = size - overlap
    for start in range(0, len(audio), step):
        end = min(start + size, len(audio))
        yield start, end, audio[start:end]
        if end == len(audio):
            break
