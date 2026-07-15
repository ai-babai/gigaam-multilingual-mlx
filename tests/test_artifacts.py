from __future__ import annotations

import json
from pathlib import Path

import pytest

from gigaam_multilingual_mlx.artifacts import (
    DEFAULT_VARIANT,
    VARIANTS,
    resolve_model_source,
    validate_artifact_dir,
)
from gigaam_multilingual_mlx.config import sha256_file


def _artifact(tmp_path: Path, *, absolute_parent: bool = False) -> Path:
    config = tmp_path / "config.json"
    weights = tmp_path / "model.safetensors"
    config.write_text("{}\n")
    weights.write_bytes(b"portable-test-weights")
    manifest = {
        "schema_version": 1,
        "dtype": "float16",
        "config": {"file": config.name, "sha256": sha256_file(config)},
        "weights": {"file": weights.name, "sha256": sha256_file(weights)},
    }
    if absolute_parent:
        manifest["parent_artifact"] = {"directory": "/private/local/model"}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return tmp_path


def test_default_variant_is_pinned_int8() -> None:
    assert DEFAULT_VARIANT == "int8"
    assert VARIANTS[DEFAULT_VARIANT]["repo_id"] == "ai-babai/gigaam-multilingual-mlx-int8-g64"
    assert VARIANTS[DEFAULT_VARIANT]["revision"] == "v0.1.0"


def test_local_artifact_resolution_keeps_backwards_compatible_path(tmp_path: Path) -> None:
    reference = resolve_model_source(_artifact(tmp_path))
    assert reference.source == "local"
    assert reference.variant == "fp16"
    assert reference.repo_id is None
    assert reference.path == tmp_path.resolve()


def test_artifact_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path)
    (artifact / "model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_artifact_dir(artifact)


def test_publish_validation_rejects_absolute_paths(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path, absolute_parent=True)
    with pytest.raises(ValueError, match="absolute local path"):
        validate_artifact_dir(artifact, require_portable=True)


def test_missing_explicit_local_path_is_not_treated_as_repo() -> None:
    with pytest.raises(FileNotFoundError, match="Local model directory"):
        resolve_model_source("/missing/local/model")
