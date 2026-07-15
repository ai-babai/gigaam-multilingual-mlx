from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

from .audio import load_audio
from .model import load_model


def _source_module(path: Path):
    spec = importlib.util.spec_from_file_location("modeling_gigaam", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _metrics(reference: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    reference = reference.astype(np.float64).ravel()
    actual = actual.astype(np.float64).ravel()
    delta = np.abs(reference - actual)
    denominator = max(float(np.mean(np.abs(reference))), 1e-12)
    cosine = float(np.dot(reference, actual) / (np.linalg.norm(reference) * np.linalg.norm(actual)))
    return {
        "cosine": cosine,
        "mae": float(delta.mean()),
        "normalized_mae": float(delta.mean() / denominator),
        "max_absolute_error": float(delta.max(initial=0.0)),
    }


def run_parity(
    source_dir: str | Path,
    mlx_dir: str | Path,
    audio_paths: list[str | Path],
    source_device: str = "mps",
) -> dict:
    import omegaconf
    import torch

    source_dir = Path(source_dir)
    raw_config = json.loads((source_dir / "config.json").read_text())
    module = _source_module(source_dir / "modeling_gigaam.py")
    source = module.GigaAMModel(
        module.GigaAMConfig(cfg=omegaconf.OmegaConf.create(raw_config["cfg"]))
    )
    state = torch.load(source_dir / "pytorch_model.bin", map_location="cpu", weights_only=True)
    source.load_state_dict(state, strict=True)
    source = source.eval().to(source_device)
    for layer in source.model.encoder.layers:
        layer.self_attn.torch_sdpa_attn = False
    target = load_model(mlx_dir, dtype="float32")
    mx.eval(target.parameters())

    rows = []
    for audio_path in audio_paths:
        audio = load_audio(audio_path)
        torch_audio = torch.from_numpy(audio).to(source_device)[None, :]
        torch_lengths = torch.tensor([len(audio)], device=source_device)
        with torch.inference_mode():
            reference_features, reference_feature_lengths = source.model.preprocessor(
                torch_audio, torch_lengths
            )
            reference_encoded, reference_encoded_lengths = source.model.encoder(
                reference_features, reference_feature_lengths
            )
            reference_logits = source.model.head(reference_encoded)
        if source_device == "mps":
            torch.mps.synchronize()

        mlx_audio = mx.array(audio)[None, :]
        mlx_lengths = mx.array([len(audio)], dtype=mx.int32)
        actual_features, actual_feature_lengths = target.preprocessor(mlx_audio, mlx_lengths)
        actual_encoded, actual_encoded_lengths = target.encoder(
            actual_features, actual_feature_lengths
        )
        actual_logits = target.head(actual_encoded)
        mx.eval(
            actual_features,
            actual_feature_lengths,
            actual_encoded,
            actual_encoded_lengths,
            actual_logits,
        )

        ref_features = reference_features.transpose(1, 2).float().cpu().numpy()
        ref_encoded = reference_encoded.transpose(1, 2).float().cpu().numpy()
        ref_logits = reference_logits.float().cpu().numpy()
        ml_features = np.asarray(actual_features)
        ml_encoded = np.asarray(actual_encoded)
        ml_logits = np.asarray(actual_logits)
        ref_decoded = source.model.decoding.decode(
            source.model.head, reference_encoded, reference_encoded_lengths
        )[0]
        ml_decoded = target.greedy_decode(actual_logits, actual_encoded_lengths)[0]
        rows.append(
            {
                "audio": str(Path(audio_path).resolve()),
                "samples": len(audio),
                "shapes": {
                    "features": list(ml_features.shape),
                    "encoded": list(ml_encoded.shape),
                    "logits": list(ml_logits.shape),
                },
                "lengths": {
                    "feature_reference": int(reference_feature_lengths[0].item()),
                    "feature_mlx": int(np.asarray(actual_feature_lengths)[0]),
                    "encoded_reference": int(reference_encoded_lengths[0].item()),
                    "encoded_mlx": int(np.asarray(actual_encoded_lengths)[0]),
                },
                "finite": {
                    "features": bool(np.isfinite(ml_features).all()),
                    "encoded": bool(np.isfinite(ml_encoded).all()),
                    "logits": bool(np.isfinite(ml_logits).all()),
                },
                "features": _metrics(ref_features, ml_features),
                "encoded": _metrics(ref_encoded, ml_encoded),
                "logits": _metrics(ref_logits, ml_logits),
                "reference_text": ref_decoded[0],
                "mlx_text": ml_decoded["text"],
                "reference_token_ids": ref_decoded[1],
                "mlx_token_ids": ml_decoded["token_ids"],
                "token_sequence_match": ref_decoded[1] == ml_decoded["token_ids"],
            }
        )
    matches = sum(bool(row["token_sequence_match"]) for row in rows)
    return {
        "schema_version": 1,
        "source_revision": source_dir.name,
        "source_device": source_device,
        "mlx_dtype": "float32",
        "thresholds": {"logits_cosine_min": 0.9999, "logits_normalized_mae_max": 1e-3},
        "summary": {
            "files": len(rows),
            "token_sequence_matches": matches,
            "token_sequence_match_rate": matches / len(rows) if rows else None,
            "minimum_logits_cosine": min((row["logits"]["cosine"] for row in rows), default=None),
            "maximum_logits_normalized_mae": max(
                (row["logits"]["normalized_mae"] for row in rows), default=None
            ),
        },
        "files": rows,
    }
