from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import sha256_file


RELEASE_REVISION = "v0.1.0"
DEFAULT_VARIANT = "int8"
REQUIRED_FILES = ("config.json", "manifest.json", "model.safetensors")

VARIANTS: dict[str, dict[str, Any]] = {
    "fp16": {
        "repo_id": "ai-babai/gigaam-multilingual-mlx",
        "revision": RELEASE_REVISION,
        "label": "FP16 reference",
        "expected_bytes": 1_170_775_344,
    },
    "int8": {
        "repo_id": "ai-babai/gigaam-multilingual-mlx-int8-g64",
        "revision": RELEASE_REVISION,
        "label": "INT8 g64 (default)",
        "expected_bytes": 698_960_933,
    },
    "int6": {
        "repo_id": "ai-babai/gigaam-multilingual-mlx-int6-g64",
        "revision": RELEASE_REVISION,
        "label": "INT6 g64 (compact balanced)",
        "expected_bytes": 573_131_675,
    },
    "int4": {
        "repo_id": "ai-babai/gigaam-multilingual-mlx-int4-g64",
        "revision": RELEASE_REVISION,
        "label": "INT4 g64 (smallest)",
        "expected_bytes": 447_302_415,
    },
}


@dataclass(frozen=True)
class ArtifactReference:
    path: Path
    variant: str
    repo_id: str | None
    revision: str | None
    source: str
    manifest: dict[str, Any]


def _variant_from_manifest(manifest: dict[str, Any]) -> str:
    profile = manifest.get("quantization", {}).get("profile")
    if profile:
        return str(profile).split("-", 1)[0]
    if manifest.get("dtype") == "float16":
        return "fp16"
    return str(manifest.get("dtype", "custom"))


def _contains_absolute_path(value: Any) -> bool:
    if isinstance(value, dict):
        return any(_contains_absolute_path(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_absolute_path(item) for item in value)
    if isinstance(value, str):
        return Path(value).is_absolute()
    return False


def validate_artifact_dir(
    model_dir: str | Path,
    *,
    expected_variant: str | None = None,
    require_portable: bool = False,
) -> dict[str, Any]:
    path = Path(model_dir).expanduser()
    if not path.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {path}")
    missing = [name for name in REQUIRED_FILES if not (path / name).is_file()]
    if missing:
        raise ValueError(f"Model artifact is missing required files: {', '.join(missing)}")
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError(f"Unsupported manifest schema: {manifest.get('schema_version')!r}")
    if require_portable and _contains_absolute_path(manifest):
        raise ValueError("Publish manifest contains an absolute local path")
    config = manifest.get("config", {})
    weights = manifest.get("weights", {})
    for section, item in (("config", config), ("weights", weights)):
        filename = item.get("file")
        expected_hash = item.get("sha256")
        if not filename or not expected_hash:
            raise ValueError(f"Manifest {section} entry is incomplete")
        target = path / filename
        if not target.is_file():
            raise ValueError(f"Manifest references a missing file: {filename}")
        if sha256_file(target) != expected_hash:
            raise ValueError(f"Manifest hash mismatch for {filename}")
    actual_variant = _variant_from_manifest(manifest)
    if expected_variant and actual_variant != expected_variant:
        raise ValueError(
            f"Artifact variant mismatch: expected {expected_variant}, found {actual_variant}"
        )
    return manifest


def resolve_model_source(
    source: str | Path | None = None,
    *,
    variant: str = DEFAULT_VARIANT,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
    token: str | bool | None = None,
) -> ArtifactReference:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; choose from {', '.join(VARIANTS)}")

    repo_id: str | None = None
    resolved_revision: str | None = revision
    if source is not None:
        candidate = Path(source).expanduser()
        if candidate.is_dir():
            manifest = validate_artifact_dir(candidate)
            return ArtifactReference(
                path=candidate.resolve(),
                variant=_variant_from_manifest(manifest),
                repo_id=None,
                revision=None,
                source="local",
                manifest=manifest,
            )
        source_text = str(source)
        if candidate.is_absolute() or source_text.startswith((".", "~")):
            raise FileNotFoundError(f"Local model directory does not exist: {candidate}")
        repo_id = source_text
    else:
        repo_id = str(VARIANTS[variant]["repo_id"])
    resolved_revision = resolved_revision or str(VARIANTS[variant]["revision"])

    from huggingface_hub import snapshot_download

    downloaded = snapshot_download(
        repo_id=repo_id,
        revision=resolved_revision,
        cache_dir=str(cache_dir) if cache_dir else None,
        local_files_only=local_files_only,
        token=token,
        allow_patterns=list(REQUIRED_FILES) + ["LICENSE", "LICENSE.*"],
    )
    path = Path(downloaded)
    manifest = validate_artifact_dir(path, expected_variant=variant if source is None else None)
    return ArtifactReference(
        path=path,
        variant=_variant_from_manifest(manifest),
        repo_id=repo_id,
        revision=resolved_revision,
        source="huggingface",
        manifest=manifest,
    )


def variant_rows() -> list[dict[str, Any]]:
    return [
        {"variant": name, **metadata, "default": name == DEFAULT_VARIANT}
        for name, metadata in VARIANTS.items()
    ]
