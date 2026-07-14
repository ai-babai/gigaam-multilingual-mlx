from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from jiwer import process_characters, process_words

from .audio import load_audio
from .model import load_model
from .normalization import NORMALIZATION_VERSION, normalize_text


def _rows(manifest: str | Path, limit: int | None = None) -> tuple[Path, list[dict]]:
    manifest = Path(manifest).resolve()
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    return manifest.parent, rows[:limit]


def _report(rows: list[dict], hypotheses: list[str], elapsed: float, backend: dict) -> dict:
    references = [normalize_text(row["reference_text"]) for row in rows]
    normalized_hypotheses = [normalize_text(text) for text in hypotheses]
    aggregate = process_words(references, normalized_hypotheses)
    aggregate_characters = process_characters(references, normalized_hypotheses)
    items = []
    for row, reference, hypothesis, raw_hypothesis in zip(
        rows, references, normalized_hypotheses, hypotheses, strict=True
    ):
        score = process_words(reference, hypothesis)
        character_score = process_characters(reference, hypothesis)
        items.append(
            {
                "id": row["id"],
                "audio": row["local_path"],
                "duration_seconds": row["duration_seconds"],
                "reference": reference,
                "hypothesis": hypothesis,
                "raw_hypothesis": raw_hypothesis,
                "wer": score.wer,
                "cer": character_score.cer,
                "substitutions": score.substitutions,
                "deletions": score.deletions,
                "insertions": score.insertions,
                "hits": score.hits,
                "exact_match": reference == hypothesis,
            }
        )
    duration = sum(float(row["duration_seconds"]) for row in rows)
    return {
        "schema_version": 1,
        "normalization_version": NORMALIZATION_VERSION,
        "backend": backend,
        "summary": {
            "files": len(rows),
            "audio_seconds": duration,
            "wall_seconds": elapsed,
            "audio_seconds_per_second": duration / elapsed,
            "wer": aggregate.wer,
            "wer_percent": aggregate.wer * 100,
            "cer": aggregate_characters.cer,
            "cer_percent": aggregate_characters.cer * 100,
            "substitutions": aggregate.substitutions,
            "deletions": aggregate.deletions,
            "insertions": aggregate.insertions,
            "hits": aggregate.hits,
            "empty_hypotheses": sum(not hypothesis for hypothesis in normalized_hypotheses),
            "exact_matches": sum(item["exact_match"] for item in items),
        },
        "items": items,
    }


def evaluate_mlx(
    model_dir: str | Path, manifest: str | Path, batch_size: int = 8, limit: int | None = None
) -> dict:
    root, rows = _rows(manifest, limit)
    model = load_model(model_dir)
    artifact_manifest = json.loads((Path(model_dir) / "manifest.json").read_text())
    mx.eval(model.parameters())
    hypotheses = []
    started = time.perf_counter()
    for offset in range(0, len(rows), batch_size):
        batch_rows = rows[offset : offset + batch_size]
        audios = [load_audio(root / row["local_path"]) for row in batch_rows]
        lengths = np.array([len(audio) for audio in audios], dtype=np.int32)
        padded = np.zeros((len(audios), int(lengths.max())), dtype=np.float32)
        for index, audio in enumerate(audios):
            padded[index, : len(audio)] = audio
        logits, encoded_lengths = model(mx.array(padded), mx.array(lengths))
        mx.eval(logits, encoded_lengths)
        hypotheses.extend(item["text"] for item in model.greedy_decode(logits, encoded_lengths))
    elapsed = time.perf_counter() - started
    return _report(
        rows,
        hypotheses,
        elapsed,
        {
            "runtime": "mlx",
            "model_dir": str(Path(model_dir).resolve()),
            "dtype": model.config.dtype,
            "revision": model.config.source_revision,
            "batch_size": batch_size,
            "quantization_profile": artifact_manifest.get("quantization", {}).get("profile"),
        },
    )


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location("modeling_gigaam", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def evaluate_pytorch(
    source_dir: str | Path,
    manifest: str | Path,
    batch_size: int = 8,
    limit: int | None = None,
    device: str = "mps",
) -> dict:
    import omegaconf
    import torch

    source_dir = Path(source_dir).resolve()
    raw = json.loads((source_dir / "config.json").read_text())
    module = _load_module(source_dir / "modeling_gigaam.py")
    model = module.GigaAMModel(module.GigaAMConfig(cfg=omegaconf.OmegaConf.create(raw["cfg"])))
    state = torch.load(source_dir / "pytorch_model.bin", map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model = model.eval().to(device)
    root, rows = _rows(manifest, limit)
    hypotheses = []
    started = time.perf_counter()
    for offset in range(0, len(rows), batch_size):
        batch_rows = rows[offset : offset + batch_size]
        audios = [load_audio(root / row["local_path"]) for row in batch_rows]
        lengths = np.array([len(audio) for audio in audios], dtype=np.int64)
        padded = np.zeros((len(audios), int(lengths.max())), dtype=np.float32)
        for index, audio in enumerate(audios):
            padded[index, : len(audio)] = audio
        with torch.inference_mode():
            encoded, encoded_lengths = model.model(
                torch.from_numpy(padded).to(device), torch.from_numpy(lengths).to(device)
            )
            decoded = model.model.decoding.decode(model.model.head, encoded, encoded_lengths)
        if device == "mps":
            torch.mps.synchronize()
        hypotheses.extend(item[0] for item in decoded)
    elapsed = time.perf_counter() - started
    return _report(
        rows,
        hypotheses,
        elapsed,
        {
            "runtime": "pytorch",
            "source_dir": str(source_dir),
            "dtype": "mps_autocast_float16" if device == "mps" else "float32",
            "revision": source_dir.name,
            "batch_size": batch_size,
            "device": device,
            "torch": torch.__version__,
        },
    )
