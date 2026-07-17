#!/usr/bin/env python3
"""Prepare portable, validated Hugging Face model repositories."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import mlx.core as mx

from gigaam_multilingual_mlx.artifacts import validate_artifact_dir
from gigaam_multilingual_mlx.model import load_model


VARIANTS = {
    "fp16": {
        "source_dir": "float16",
        "repo_id": "ai-babai/gigaam-multilingual-mlx",
        "title": "GigaAM-Multilingual MLX FP16",
        "role": "reference MLX artifact",
    },
    "int8": {
        "source_dir": "int8-g64",
        "repo_id": "ai-babai/gigaam-multilingual-mlx-int8-g64",
        "title": "GigaAM-Multilingual MLX INT8 g64",
        "role": "default balanced artifact",
    },
    "int6": {
        "source_dir": "int6-g64",
        "repo_id": "ai-babai/gigaam-multilingual-mlx-int6-g64",
        "title": "GigaAM-Multilingual MLX INT6 g64",
        "role": "compact opt-in artifact",
    },
    "int4": {
        "source_dir": "int4-g64",
        "repo_id": "ai-babai/gigaam-multilingual-mlx-int4-g64",
        "title": "GigaAM-Multilingual MLX INT4 g64",
        "role": "smallest opt-in artifact",
    },
}

UPSTREAM_REPOSITORY = "https://github.com/salute-developers/GigaAM"
UPSTREAM_MODEL = "https://huggingface.co/ai-sage/GigaAM-Multilingual"
UPSTREAM_REVISION = "3905cd51c3ed4e88c8edf33f3302969ba480a327"
RELEASE_REVISION = "v0.1.0"
PACKAGE_VERSION = "0.2.0"
GITHUB = "https://github.com/ai-babai/gigaam-multilingual-mlx"
PYPI = "https://pypi.org/project/gigaam-multilingual-mlx/"
REPORT = f"{GITHUB}/blob/{RELEASE_REVISION}/docs/benchmark-v0.1.0.md"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _format_size(size_bytes: int) -> str:
    if size_bytes >= 1_000_000_000:
        return f"{size_bytes / 1_000_000_000:.2f} GB"
    return f"{size_bytes / 1_000_000:.0f} MB"


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _portable_manifest(
    variant: str,
    source: dict[str, Any],
    fp16_manifest_sha256: str | None = None,
    fp16_weights: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = VARIANTS[variant]
    result = {
        key: value for key, value in source.items() if key not in {"parent_artifact", "environment"}
    }
    result["converter"] = {
        **source.get("converter", {}),
        "package": "gigaam-multilingual-mlx",
    }
    result["source"] = {
        **source["source"],
        "repository": "ai-sage/GigaAM-Multilingual",
        "revision": UPSTREAM_REVISION,
    }
    result["artifact"] = {
        "repository": metadata["repo_id"],
        "revision": RELEASE_REVISION,
        "variant": variant,
        "role": metadata["role"],
    }
    result["compatibility"] = {
        "package": "gigaam-multilingual-mlx",
        "package_version": ">=0.1.0,<0.3.0",
        "python": ">=3.12,<3.14",
        "mlx": ">=0.32,<0.33",
        "platform": "macOS 14+ on Apple Silicon",
    }
    result["conversion_environment"] = source.get("environment", {})
    if variant != "fp16":
        if fp16_manifest_sha256 is None or fp16_weights is None:
            raise ValueError("Quantized manifests require the sanitized FP16 parent")
        result["parent_artifact"] = {
            "repository": VARIANTS["fp16"]["repo_id"],
            "revision": RELEASE_REVISION,
            "manifest_sha256": fp16_manifest_sha256,
            "weights_sha256": fp16_weights["sha256"],
            "weights_bytes": fp16_weights["bytes"],
        }
    return result


def _variant_table(report: dict[str, Any]) -> list[str]:
    quality = report["quality_results"]
    performance = report["performance_results"]
    artifacts = {item["implementation"]: item for item in report["artifacts"]}
    implementations = {item["id"]: item for item in report["implementations"]}
    lines = [
        "| Implementation | Disk | FLEURS WER | Ru Libri WER | SOVA WER | 5 min warm | Speed vs Original | Peak RSS | Device peak |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for implementation in ("original", "fp16", "int8", "int6", "int4"):
        q = {item["dataset"]: item for item in quality if item["implementation"] == implementation}
        p = next(
            item
            for item in performance
            if item["implementation"] == implementation and item["profile"] == "medium"
        )
        artifact = artifacts[implementation]
        lines.append(
            f"| {implementations[implementation]['label']} | "
            f"{artifact['artifact_bytes'] / 1_000_000_000:.3f} GB | "
            f"{q['google/fleurs (ru_ru)']['wer_percent']:.3f}% | "
            f"{q['istupakov/russian_librispeech']['wer_percent']:.3f}% | "
            f"{q['bond005/sova_rudevices']['wer_percent']:.3f}% | "
            f"{p['warm_median_seconds']:.3f}s | {p['warm_speedup_vs_original']:.2f}× | "
            f"{p['peak_rss_bytes'] / 1_000_000_000:.3f} GB | "
            f"{p['peak_device_bytes'] / 1_000_000_000:.3f} GB |"
        )
    return lines


def _card(
    variant: str,
    manifest: dict[str, Any],
    report: dict[str, Any],
    collection_url: str,
) -> str:
    metadata = VARIANTS[variant]
    relation = "\nbase_model_relation: quantized" if variant != "fp16" else ""
    base_model = VARIANTS["fp16"]["repo_id"] if variant != "fp16" else "ai-sage/GigaAM-Multilingual"
    quantization = manifest.get("quantization")
    precision = quantization["profile"] if quantization else "float16"
    default_note = (
        "This is the default variant used by `gigaam-multilingual-mlx`."
        if variant == "int8"
        else "Select this variant explicitly with `--variant`."
    )
    yaml_tags = [
        "mlx",
        "apple-silicon",
        "gigaam",
        "ctc",
        "automatic-speech-recognition",
        "speech-recognition",
        "transcription",
        "offline-asr",
        "speech-to-text",
        "macos",
    ]
    if quantization:
        yaml_tags.extend(["quantized", precision])
    lines = [
        "---",
        "license: mit",
        "language:",
        "- ru",
        "- en",
        "- kk",
        "- ky",
        "- uz",
        "library_name: gigaam-multilingual-mlx",
        "pipeline_tag: automatic-speech-recognition",
        "inference: false",
        f"base_model: {base_model}{relation}",
        "tags:",
        *(f"- {tag}" for tag in yaml_tags),
        "---",
        "",
        f"# {metadata['title']}",
        "",
        "Native offline speech recognition and audio transcription for Russian, Kazakh, "
        "Kyrgyz, and Uzbek on Apple Silicon with MLX. Core speech-to-text languages: "
        "Russian (русский), Kazakh (қазақша), Kyrgyz (кыргызча), and Uzbek (o‘zbekcha).",
        "",
        "This is an independent Apple MLX port of the official "
        f"[GigaAM repository]({UPSTREAM_REPOSITORY}) and the "
        f"[GigaAM-Multilingual source model]({UPSTREAM_MODEL}). It is not an official "
        "GigaChat/GigaAM release.",
        "",
        f"This repository contains the **{precision}** {metadata['role']}. {default_note}",
        "",
        "- Code and CLI: " + f"[{GITHUB}]({GITHUB})",
        "- PyPI package: " + f"[`gigaam-multilingual-mlx`]({PYPI})",
        "- All variants: " + f"[GigaAM Multilingual MLX Collection]({collection_url})",
        "- Full reproducible benchmark: " + f"[v0.1.0 report]({REPORT})",
        "",
        "## Install and transcribe",
        "",
        "Requires macOS 14 or newer on Apple Silicon and Python 3.12 or 3.13.",
        "",
        "```bash",
        "brew install uv ffmpeg",
        f"uv tool install gigaam-multilingual-mlx=={PACKAGE_VERSION}",
        f"gigaam-stt audio.wav --variant {variant} --format json --output transcript.json",
        "```",
        "",
        "For the Python API inside a uv-managed project, run "
        f"`uv add gigaam-multilingual-mlx=={PACKAGE_VERSION}`. A regular "
        f"`python -m pip install gigaam-multilingual-mlx=={PACKAGE_VERSION}` remains supported.",
        "",
        "The package downloads the pinned `v0.1.0` snapshot into the standard Hugging Face "
        "cache. Input may be WAV, FLAC, MP3, M4A, or a video readable by `ffmpeg`; output may "
        "be TXT, JSON, SRT, or VTT.",
        "",
        "```python",
        "from gigaam_multilingual_mlx import load_model",
        "from gigaam_multilingual_mlx.service import transcribe_file",
        "",
        f'model = load_model(variant="{variant}")',
        'result = transcribe_file(model, "audio.wav")',
        'print(result["text"])',
        "```",
        "",
        "## Artifact",
        "",
        f"- Hub revision: `{RELEASE_REVISION}`",
        f"- Source revision: `{UPSTREAM_REVISION}`",
        f"- Model size: `{manifest['weights']['file']}` — "
        f"{_format_size(manifest['weights']['bytes'])}",
        f"- Weights SHA-256: `{manifest['weights']['sha256']}`",
        f"- Config SHA-256: `{manifest['config']['sha256']}`",
        "- Runtime compatibility: `gigaam-multilingual-mlx>=0.1.0,<0.3.0`, `mlx>=0.32,<0.33`",
        "",
        "`manifest.json` records source and parent revisions, hashes, conversion metadata, "
        "quantization rules, strict-load validation, and compatibility without machine-local paths.",
        "",
        "## Quality, speed, and memory",
        "",
        "Measured on one MacBook Pro with Apple M4 Pro and 48 GB unified memory. Quality uses "
        "the public FLEURS `ru_ru`, Russian LibriSpeech, and SOVA subsets. Performance below is "
        "the five-run warm median on the deterministic 5-minute public corpus, with greedy CTC, "
        "20-second chunks and 2-second overlap. Peak RSS is resident process memory; device memory "
        "is shown separately and the two must not be added.",
        "",
        *_variant_table(report),
        "",
        "INT8 is the default for its quality/size/memory balance, not because it is always the "
        "fastest. See the full report for cold/load timings, p95, 30-second and 20-minute profiles, "
        "CER, deltas, hashes, commands, and limitations.",
        "",
        "## Local transcription server",
        "",
        "Version 0.2.0 adds an optional OpenAI-compatible transcription endpoint:",
        "",
        "```bash",
        f"uv tool install 'gigaam-multilingual-mlx[server]=={PACKAGE_VERSION}'",
        f"gigaam-stt serve --variant {variant}",
        "```",
        "",
        f"See the [server guide]({GITHUB}/blob/v{PACKAGE_VERSION}/docs/server.md) for curl and "
        "OpenAI Python client examples, supported formats, network access, and limitations. "
        "The `whisper-1` model name is a compatibility alias; inference still uses GigaAM "
        "Multilingual MLX.",
        "",
        "## Intended use and limitations",
        "",
        "Intended for local, offline automatic speech recognition in Russian, English, Kazakh, "
        "Kyrgyz, and Uzbek on Apple Silicon. It is not a diarization system, realtime streaming "
        "service, forced aligner, or cloud service. Word timestamps are approximate greedy-CTC "
        "frame timings. Accuracy can degrade with noise, far-field speech, accents, code-switching, "
        "music, overlapping speakers, or domains unlike the public evaluation subsets.",
        "",
        "Only Apple M4 Pro was benchmarked for this release. M1-M5 machines are expected to be "
        "runtime-compatible within the documented macOS/MLX range, but they are not claimed to "
        "have the same speed or memory figures.",
        "",
        "## License and provenance",
        "",
        "The MLX port and converted artifacts are released under MIT, matching the upstream "
        f"license. See [`LICENSE`](LICENSE) and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). "
        f"The exact upstream repository is [{UPSTREAM_REPOSITORY}]({UPSTREAM_REPOSITORY}); the "
        f"source model card is [{UPSTREAM_MODEL}]({UPSTREAM_MODEL}).",
        "",
        "## Citation",
        "",
        "Please cite both this MLX software release and the original GigaAM-Multilingual work:",
        "",
        "```bibtex",
        "@software{popkov2026gigaammlx,",
        "  author = {Maksim Popkov},",
        "  title = {GigaAM-Multilingual MLX},",
        "  year = {2026},",
        f"  version = {{{PACKAGE_VERSION}}},",
        f"  url = {{{GITHUB}}}",
        "}",
        "",
        "@misc{gigaam_multilingual,",
        "  title = {GigaAM Multilingual: Foundation Model for Underrepresented Languages},",
        "  author = {Andrei Kuzmenko and Alexandr Maximenko and Aleksandr Kutsakov and "
        "Georgii Gospodinov and Dmitrii Bolotov and Oleg Kutuzov and Pavel Bogomolov and "
        "Fyodor Minkin},",
        "  year = {2026},",
        "  eprint = {2607.10371},",
        "  archivePrefix = {arXiv},",
        "  primaryClass = {eess.AS},",
        "  url = {https://arxiv.org/abs/2607.10371}",
        "}",
        "```",
    ]
    return "\n".join(lines) + "\n"


def _link_or_copy(source: Path, destination: Path) -> str:
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError(f"Output root must be empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    report = json.loads(args.benchmark_json.resolve().read_text())
    release: dict[str, Any] = {
        "schema_version": 1,
        "release": "0.1.0",
        "revision": RELEASE_REVISION,
        "collection_url": args.collection_url,
        "artifacts": [],
    }

    sanitized: dict[str, dict[str, Any]] = {}
    fp16_manifest_hash = None
    fp16_weights = None
    for variant in ("fp16", "int8", "int6", "int4"):
        metadata = VARIANTS[variant]
        source_dir = source_root / metadata["source_dir"]
        destination = output_root / metadata["repo_id"].split("/", 1)[1]
        destination.mkdir()
        shutil.copy2(source_dir / "config.json", destination / "config.json")
        link_mode = _link_or_copy(
            source_dir / "model.safetensors", destination / "model.safetensors"
        )
        source_manifest = json.loads((source_dir / "manifest.json").read_text())
        manifest = _portable_manifest(
            variant,
            source_manifest,
            fp16_manifest_sha256=fp16_manifest_hash,
            fp16_weights=fp16_weights,
        )
        if _sha256(destination / "config.json") != manifest["config"]["sha256"]:
            raise ValueError(f"Config hash mismatch for {variant}")
        if _sha256(destination / "model.safetensors") != manifest["weights"]["sha256"]:
            raise ValueError(f"Weights hash mismatch for {variant}")
        _write_json(destination / "manifest.json", manifest)
        if variant == "fp16":
            fp16_manifest_hash = _sha256(destination / "manifest.json")
            fp16_weights = manifest["weights"]
        sanitized[variant] = manifest
        shutil.copy2(args.license.resolve(), destination / "LICENSE")
        shutil.copy2(args.notices.resolve(), destination / "THIRD_PARTY_NOTICES.md")
        (destination / "README.md").write_text(
            _card(variant, manifest, report, args.collection_url)
        )
        (destination / ".gitattributes").write_text(
            "*.safetensors filter=lfs diff=lfs merge=lfs -text\n"
        )
        published_files = {
            path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in destination.iterdir()
            if path.is_file()
        }
        release["artifacts"].append(
            {
                "variant": variant,
                "repo_id": metadata["repo_id"],
                "revision": RELEASE_REVISION,
                "staging_directory": destination.name,
                "weight_materialization": link_mode,
                "model_artifact_bytes": sum(
                    published_files[name]["bytes"]
                    for name in ("config.json", "manifest.json", "model.safetensors")
                ),
                "files": published_files,
            }
        )

    for item in release["artifacts"]:
        variant = item["variant"]
        destination = output_root / item["staging_directory"]
        validate_artifact_dir(destination, expected_variant=variant, require_portable=True)
        model = load_model(destination)
        mx.eval(model.parameters())
        item["strict_runtime_load"] = True
        del model
        gc.collect()
        mx.clear_cache()

    (output_root / "release-manifest.json").write_text(
        json.dumps(release, ensure_ascii=False, indent=2) + "\n"
    )
    return release


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--benchmark-json", type=Path, required=True)
    parser.add_argument("--license", type=Path, required=True)
    parser.add_argument("--notices", type=Path, required=True)
    parser.add_argument("--collection-url", required=True)
    args = parser.parse_args()
    result = prepare(args)
    print(
        json.dumps(
            {
                "revision": result["revision"],
                "artifacts": [
                    {
                        "variant": item["variant"],
                        "repo_id": item["repo_id"],
                        "bytes": item["model_artifact_bytes"],
                        "strict_runtime_load": item["strict_runtime_load"],
                    }
                    for item in result["artifacts"]
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
