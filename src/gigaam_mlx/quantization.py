from __future__ import annotations

import importlib.metadata
import json
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from .config import sha256_file


PROFILES = {
    "int8-g64": {
        "method": "weight_only_affine",
        "group_size": 64,
        "rules": [{"selector": "all_linear", "bits": 8}],
    },
    "int6-g64": {
        "method": "weight_only_affine",
        "group_size": 64,
        "rules": [{"selector": "all_linear", "bits": 6}],
    },
    "int4-g64": {
        "method": "weight_only_affine",
        "group_size": 64,
        "rules": [{"selector": "all_linear", "bits": 4}],
    },
    "hybrid-ffn4-attn8-g64": {
        "method": "weight_only_affine",
        "group_size": 64,
        "rules": [
            {"selector": "feed_forward_linear", "bits": 4},
            {"selector": "attention_linear", "bits": 8},
        ],
    },
}


def quantization_spec(profile: str) -> dict:
    if profile not in PROFILES:
        raise ValueError(
            f"Unknown quantization profile {profile!r}; choose from {sorted(PROFILES)}"
        )
    return {"profile": profile, **PROFILES[profile]}


def apply_quantization(model: nn.Module, spec: dict) -> dict[str, int]:
    if spec.get("method") != "weight_only_affine":
        raise ValueError(f"Unsupported quantization method: {spec.get('method')}")
    group_size = int(spec["group_size"])
    rules = spec["rules"]
    counts: dict[str, int] = {str(rule["selector"]): 0 for rule in rules}

    def predicate(path: str, module: nn.Module):
        if not isinstance(module, nn.Linear):
            return False
        for rule in rules:
            selector = rule["selector"]
            selected = (
                selector == "all_linear"
                or (selector == "feed_forward_linear" and ".feed_forward" in path)
                or (selector == "attention_linear" and ".self_attn." in path)
            )
            if selected:
                counts[selector] += 1
                return {
                    "group_size": group_size,
                    "bits": int(rule["bits"]),
                    "mode": "affine",
                }
        return False

    nn.quantize(model, class_predicate=predicate)
    return counts


def quantize_artifact(source_dir: str | Path, output_dir: str | Path, profile: str) -> dict:
    # Import here to avoid a model -> quantization -> model cycle at module import time.
    from .model import load_model

    source_dir = Path(source_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    source_manifest = json.loads((source_dir / "manifest.json").read_text())
    if source_manifest.get("quantization"):
        raise ValueError("Quantization must start from a non-quantized artifact")
    if source_manifest["dtype"] != "float16":
        raise ValueError("Quantization profiles are calibrated from the float16 artifact")

    spec = quantization_spec(profile)
    model = load_model(source_dir)
    counts = apply_quantization(model, spec)
    mx.eval(model.parameters())
    arrays = dict(tree_flatten(model.parameters()))

    output_dir.mkdir(parents=True, exist_ok=True)
    config_source = source_dir / source_manifest["config"]["file"]
    config_path = output_dir / "config.json"
    shutil.copyfile(config_source, config_path)
    weights_path = output_dir / "model.safetensors"
    mx.save_safetensors(
        weights_path,
        arrays,
        metadata={
            "format": "gigaam-mlx",
            "source_revision": source_manifest["source"]["revision"],
            "dtype": "float16",
            "quantization_profile": profile,
        },
    )
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "converter": {"package": "gigaam-mlx", "version": "0.1.0"},
        "runtime": {"package": "gigaam-mlx", "requires": ">=0.1,<0.2"},
        "architecture": source_manifest["architecture"],
        "model_name": source_manifest["model_name"],
        "dtype": "float16",
        "quantization": spec,
        "config": {"file": config_path.name, "sha256": sha256_file(config_path)},
        "source": source_manifest["source"],
        "parent_artifact": {
            "repository": "ai-babai/gigaam-multilingual-mlx",
            "revision": "v0.1.0",
            "weights_sha256": source_manifest["weights"]["sha256"],
            "manifest_sha256": sha256_file(source_dir / "manifest.json"),
        },
        "weights": {
            "file": weights_path.name,
            "sha256": sha256_file(weights_path),
            "bytes": weights_path.stat().st_size,
            "tensor_count": len(arrays),
            "logical_parameter_count": source_manifest["weights"]["parameter_count"],
            "stored_parameter_bytes": sum(array.nbytes for array in arrays.values()),
        },
        "validation": {
            "quantized_module_counts": counts,
            "strict_quantized_load": False,
        },
        "environment": {
            "python": platform.python_version(),
            "mlx": importlib.metadata.version("mlx"),
            "platform": platform.platform(),
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")

    verification_model = load_model(output_dir)
    mx.eval(verification_model.parameters())
    manifest["validation"]["strict_quantized_load"] = True
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest
