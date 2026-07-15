from __future__ import annotations

import importlib.util
import importlib.metadata
import json
import platform
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
import mlx.utils
import numpy as np

from .config import ModelConfig, sha256_file
from .model import GigaAMCTC


def _load_source_module(path: Path):
    spec = importlib.util.spec_from_file_location("modeling_gigaam", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _target_name(source_name: str) -> str:
    if not source_name.startswith("model."):
        raise ValueError(f"Unexpected source prefix: {source_name}")
    name = source_name.removeprefix("model.")
    replacements = {
        "preprocessor.featurizer.0.spectrogram.window": "preprocessor.window",
        "preprocessor.featurizer.0.mel_scale.fb": "preprocessor.mel_filter",
    }
    if name in replacements:
        return replacements[name]
    name = name.replace("encoder.pre_encode.conv.0.", "encoder.pre_encode.conv1.")
    name = name.replace("encoder.pre_encode.conv.2.", "encoder.pre_encode.conv2.")
    name = name.replace("head.decoder_layers.0.", "head.decoder.")
    return name


def convert_checkpoint(source: str | Path, output: str | Path, dtype: str = "float32") -> dict:
    import omegaconf
    import torch

    source = Path(source).resolve()
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output}")
    config_path = source / "config.json"
    weights_path = source / "pytorch_model.bin"
    modeling_path = source / "modeling_gigaam.py"
    for required in (config_path, weights_path, modeling_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    source_raw = json.loads(config_path.read_text())
    revision = source.name
    config = ModelConfig.from_source(source_raw, revision)
    if dtype not in {"float32", "float16", "bfloat16"}:
        raise ValueError("dtype must be float32, float16, or bfloat16")
    config = replace(config, dtype=dtype)

    source_module = _load_source_module(modeling_path)
    hf_config = source_module.GigaAMConfig(cfg=omegaconf.OmegaConf.create(source_raw["cfg"]))
    source_model = source_module.GigaAMModel(hf_config)
    raw_state = torch.load(weights_path, map_location="cpu", weights_only=True)
    incompatible = source_model.load_state_dict(raw_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(f"Non-strict source checkpoint: {incompatible}")

    module_by_name = dict(source_model.named_modules())
    target_model = GigaAMCTC(config)
    target_shapes = {
        key: tuple(value.shape) for key, value in mlx.utils.tree_flatten(target_model.parameters())
    }
    converted: dict[str, mx.array] = {}
    consumed: set[str] = set()
    transpose_count = 0
    target_dtype = {"float32": mx.float32, "float16": mx.float16, "bfloat16": mx.bfloat16}[dtype]

    for source_name, tensor in source_model.state_dict().items():
        target_name = _target_name(source_name)
        owner_name, leaf = source_name.rsplit(".", 1)
        array = tensor.detach().cpu().numpy()
        if isinstance(module_by_name[owner_name], torch.nn.Conv1d) and leaf == "weight":
            array = np.transpose(array, (0, 2, 1))
            transpose_count += 1
        if target_name in converted:
            raise ValueError(f"Duplicate target tensor: {target_name}")
        if target_name not in target_shapes:
            raise ValueError(f"Unexpected target tensor: {source_name} -> {target_name}")
        if tuple(array.shape) != target_shapes[target_name]:
            raise ValueError(
                f"Shape mismatch {source_name} -> {target_name}: "
                f"{array.shape} != {target_shapes[target_name]}"
            )
        tensor_dtype = mx.float32 if target_name.startswith("preprocessor.") else target_dtype
        converted[target_name] = mx.array(array).astype(tensor_dtype)
        consumed.add(source_name)

    missing_source = sorted(set(raw_state) - consumed)
    missing_target = sorted(set(target_shapes) - set(converted))
    unexpected_target = sorted(set(converted) - set(target_shapes))
    if missing_source or missing_target or unexpected_target or len(converted) != len(raw_state):
        raise ValueError(
            f"Conversion is not bijective: missing_source={missing_source}, "
            f"missing_target={missing_target}, unexpected_target={unexpected_target}"
        )

    output.mkdir(parents=True, exist_ok=True)
    converted_path = output / "model.safetensors"
    mx.save_safetensors(
        converted_path,
        converted,
        metadata={
            "format": "gigaam-multilingual-mlx",
            "source_revision": revision,
            "dtype": dtype,
        },
    )
    config.save(output / "config.json")
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "converter": {"package": "gigaam-multilingual-mlx", "version": "0.1.0"},
        "runtime": {"package": "gigaam-multilingual-mlx", "requires": ">=0.1,<0.2"},
        "architecture": "gigaam_multilingual_ctc_conv1d_rotary_conformer",
        "model_name": config.model_name,
        "dtype": dtype,
        "config": {
            "file": "config.json",
            "sha256": sha256_file(output / "config.json"),
        },
        "source": {
            "repository": "ai-sage/GigaAM-Multilingual",
            "revision": revision,
            "config_file": "config.json",
            "config_sha256": sha256_file(config_path),
            "weights_file": "pytorch_model.bin",
            "weights_sha256": sha256_file(weights_path),
        },
        "weights": {
            "file": converted_path.name,
            "sha256": sha256_file(converted_path),
            "bytes": converted_path.stat().st_size,
            "tensor_count": len(converted),
            "parameter_count": sum(int(np.prod(shape)) for shape in target_shapes.values()),
        },
        "validation": {
            "source_load_strict": True,
            "missing_source": [],
            "missing_target": [],
            "unexpected_target": [],
            "conv1d_weights_transposed": transpose_count,
        },
        "environment": {
            "python": platform.python_version(),
            "mlx": importlib.metadata.version("mlx"),
            "torch": torch.__version__,
            "platform": platform.platform(),
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest
