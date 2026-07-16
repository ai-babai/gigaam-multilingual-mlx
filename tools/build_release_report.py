#!/usr/bin/env python3
"""Build the compact, path-free v0.1.0 public benchmark report."""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


IMPLEMENTATIONS = {
    "original": {
        "id": "original",
        "label": "Original PyTorch/MPS",
        "variant": "large_ctc",
        "repository": "ai-sage/GigaAM-Multilingual",
        "revision": "3905cd51c3ed4e88c8edf33f3302969ba480a327",
        "published_artifact": False,
        "default": False,
    },
    "fp16": {
        "id": "fp16",
        "label": "MLX FP16",
        "variant": "fp16",
        "repository": "ai-babai/gigaam-multilingual-mlx",
        "revision": "v0.1.0",
        "published_artifact": True,
        "default": False,
    },
    "int8": {
        "id": "int8",
        "label": "MLX INT8 g64 (default)",
        "variant": "int8-g64",
        "repository": "ai-babai/gigaam-multilingual-mlx-int8-g64",
        "revision": "v0.1.0",
        "published_artifact": True,
        "default": True,
    },
    "int6": {
        "id": "int6",
        "label": "MLX INT6 g64",
        "variant": "int6-g64",
        "repository": "ai-babai/gigaam-multilingual-mlx-int6-g64",
        "revision": "v0.1.0",
        "published_artifact": True,
        "default": False,
    },
    "int4": {
        "id": "int4",
        "label": "MLX INT4 g64",
        "variant": "int4-g64",
        "repository": "ai-babai/gigaam-multilingual-mlx-int4-g64",
        "revision": "v0.1.0",
        "published_artifact": True,
        "default": False,
    },
}

DATASET_NAMES = {
    "fleurs": "google/fleurs (ru_ru)",
    "russian-librispeech": "istupakov/russian_librispeech",
    "sova": "bond005/sova_rudevices",
}

WEIGHT_DIRS = {
    "fp16": "float16",
    "int8": "int8-g64",
    "int6": "int6-g64",
    "int4": "int4-g64",
}


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _round(value: float | None, digits: int = 6) -> float | None:
    return round(value, digits) if value is not None else None


def _word_count(run: dict[str, Any]) -> int:
    return int(run.get("word_count", len(run.get("words", []))))


def _quality_rows(root: Path) -> list[dict[str, Any]]:
    reports = {
        (implementation, dataset): _read(root / implementation / f"{dataset}.json")
        for implementation in IMPLEMENTATIONS
        for dataset in DATASET_NAMES
    }
    rows = []
    for dataset, dataset_name in DATASET_NAMES.items():
        original = reports["original", dataset]["summary"]
        fp16 = reports["fp16", dataset]["summary"]
        original_ids = [item["id"] for item in reports["original", dataset]["items"]]
        original_refs = [item["reference"] for item in reports["original", dataset]["items"]]
        for implementation in IMPLEMENTATIONS:
            report = reports[implementation, dataset]
            summary = report["summary"]
            if report["normalization_version"] not in {
                "gigaam-multilingual-mlx-text-v1",
                "gigaam-mlx-text-v1",  # pre-release report identifier
            }:
                raise ValueError(f"Unexpected normalization for {implementation}/{dataset}")
            if [item["id"] for item in report["items"]] != original_ids:
                raise ValueError(f"Item mismatch for {implementation}/{dataset}")
            if [item["reference"] for item in report["items"]] != original_refs:
                raise ValueError(f"Reference mismatch for {implementation}/{dataset}")
            rows.append(
                {
                    "dataset": dataset_name,
                    "implementation": implementation,
                    "files": summary["files"],
                    "audio_seconds": _round(summary["audio_seconds"]),
                    "wer": _round(summary["wer"]),
                    "wer_percent": _round(summary["wer_percent"]),
                    "cer": _round(summary["cer"]),
                    "cer_percent": _round(summary["cer_percent"]),
                    "wer_delta_pp_vs_original": _round(
                        summary["wer_percent"] - original["wer_percent"]
                    ),
                    "wer_delta_pp_vs_fp16": _round(summary["wer_percent"] - fp16["wer_percent"]),
                    "substitutions": summary["substitutions"],
                    "deletions": summary["deletions"],
                    "insertions": summary["insertions"],
                    "hits": summary["hits"],
                    "empty_hypotheses": summary["empty_hypotheses"],
                    "exact_matches": summary["exact_matches"],
                }
            )
    return rows


def _performance_rows(root: Path, recipe: dict[str, Any]) -> list[dict[str, Any]]:
    reports = {
        (implementation, profile): _read(root / implementation / f"{profile}.json")
        for implementation in IMPLEMENTATIONS
        for profile in recipe["profiles"]
    }
    rows = []
    for profile, expected in recipe["profiles"].items():
        original_warm = reports["original", profile]["warm_summary"]["median_end_to_end_seconds"]
        for implementation in IMPLEMENTATIONS:
            report = reports[implementation, profile]
            if report["audio"]["sha256"] != expected["wav_sha256"]:
                raise ValueError(f"Audio hash mismatch for {implementation}/{profile}")
            if report["chunk_seconds"] != 20 or report["overlap_seconds"] != 2:
                raise ValueError(f"Chunking mismatch for {implementation}/{profile}")
            expected_runs = 3 if profile == "long" else 5
            if report["warm_summary"]["runs"] != expected_runs:
                raise ValueError(f"Warm-run mismatch for {implementation}/{profile}")
            cold = report["runs"][0]
            cold_wall = float(cold.get("wall_seconds", cold.get("end_to_end_wall_seconds")))
            warm = float(report["warm_summary"]["median_end_to_end_seconds"])
            warm_p95 = float(report["warm_summary"]["p95_end_to_end_seconds"])
            duration = float(report["audio"]["duration_seconds"])
            swap_delta = int(report["swap_used_end_bytes"] - report["swap_used_start_bytes"])
            warm_counts = [_word_count(run) for run in report["runs"][1:]]
            rows.append(
                {
                    "profile": profile,
                    "audio_seconds": duration,
                    "audio_sha256": report["audio"]["sha256"],
                    "implementation": implementation,
                    "model_load_seconds": _round(float(report["load_seconds"])),
                    "cold_wall_seconds": _round(cold_wall),
                    "cold_rtf": _round(cold_wall / duration),
                    "warm_runs": expected_runs,
                    "warm_median_seconds": _round(warm),
                    "warm_p95_seconds": _round(warm_p95),
                    "warm_median_rtf": _round(warm / duration),
                    "warm_median_audio_seconds_per_second": _round(duration / warm),
                    "warm_speedup_vs_original": _round(original_warm / warm),
                    "peak_rss_bytes": int(report["peak_rss_bytes"]),
                    "peak_device_bytes": report.get("peak_device_bytes"),
                    "peak_active_device_bytes": report.get("peak_active_device_bytes"),
                    "peak_cache_device_bytes": report.get("peak_cache_device_bytes"),
                    "swap_delta_bytes": swap_delta,
                    "swap_increase_bytes": max(0, swap_delta),
                    "output_word_count_median": int(statistics.median(warm_counts)),
                    "output_word_count_min": min(warm_counts),
                    "output_word_count_max": max(warm_counts),
                    "compiled": bool(report.get("compiled", False)),
                }
            )
    return rows


def _artifact_rows(performance_root: Path, weights_root: Path) -> list[dict[str, Any]]:
    rows = []
    original = _read(performance_root / "original" / "medium.json")["artifact"]
    rows.append(
        {
            "implementation": "original",
            "repository": IMPLEMENTATIONS["original"]["repository"],
            "revision": IMPLEMENTATIONS["original"]["revision"],
            "weights_bytes": (
                weights_root.parent.parent
                / "source"
                / "large_ctc"
                / IMPLEMENTATIONS["original"]["revision"]
                / "pytorch_model.bin"
            )
            .stat()
            .st_size,
            "artifact_bytes": original["bytes"],
            "weights_sha256": original["weights_sha256"],
            "published": False,
        }
    )
    for implementation, directory in WEIGHT_DIRS.items():
        manifest = _read(weights_root / directory / "manifest.json")
        report = _read(performance_root / implementation / "medium.json")
        rows.append(
            {
                "implementation": implementation,
                "repository": IMPLEMENTATIONS[implementation]["repository"],
                "revision": "v0.1.0",
                "weights_bytes": manifest["weights"]["bytes"],
                "artifact_bytes": report["artifact"]["bytes"],
                "weights_sha256": manifest["weights"]["sha256"],
                "published": True,
            }
        )
    return rows


def _commands() -> list[dict[str, str]]:
    rows = []
    for implementation in IMPLEMENTATIONS:
        if implementation == "original":
            quality = (
                "python -m gigaam_multilingual_mlx.dev_cli evaluate --manifest <DATASET_MANIFEST> "
                "--source <UPSTREAM_SNAPSHOT> --source-device mps --batch-size 8 "
                "--output <RAW_REPORT>"
            )
            performance = (
                "python -m gigaam_multilingual_mlx.dev_cli benchmark-pytorch <PUBLIC_PROFILE_WAV> "
                "--source <UPSTREAM_SNAPSHOT> --source-device mps --chunk-seconds 20 "
                "--overlap-seconds 2 --warm-runs <5_SHORT_MEDIUM_OR_3_LONG> "
                "--output <RAW_REPORT>"
            )
        else:
            quality = (
                "python -m gigaam_multilingual_mlx.dev_cli evaluate --manifest <DATASET_MANIFEST> "
                "--model <LOCAL_RELEASE_ARTIFACT> --batch-size 8 --output <RAW_REPORT>"
            )
            performance = (
                "python -m gigaam_multilingual_mlx.dev_cli benchmark <PUBLIC_PROFILE_WAV> "
                "--model <LOCAL_RELEASE_ARTIFACT> --compile --chunk-seconds 20 "
                "--overlap-seconds 2 --warm-runs <5_SHORT_MEDIUM_OR_3_LONG> "
                "--output <RAW_REPORT>"
            )
        rows.append(
            {
                "implementation": implementation,
                "quality": quality,
                "performance": performance,
            }
        )
    return rows


def _datasets(summary: dict[str, Any]) -> list[dict[str, Any]]:
    urls = {
        "google/fleurs": "https://huggingface.co/datasets/google/fleurs",
        "istupakov/russian_librispeech": "https://huggingface.co/datasets/istupakov/russian_librispeech",
        "bond005/sova_rudevices": "https://huggingface.co/datasets/bond005/sova_rudevices",
    }
    return [
        {
            **item,
            "source_url": urls[item["dataset_id"]],
            "manifest_path": f"benchmarks/manifests/{item['manifest']}",
        }
        for item in summary["datasets"]
    ]


def build(args: argparse.Namespace) -> dict[str, Any]:
    quality_root = args.quality_root.resolve()
    performance_root = args.performance_root.resolve()
    recipe = _read(args.performance_recipe.resolve())
    manifest_summary = _read(args.quality_manifest_summary.resolve())
    sample = _read(performance_root / "fp16" / "medium.json")
    original_sample = _read(performance_root / "original" / "medium.json")
    artifacts = _artifact_rows(performance_root, args.weights_root.resolve())
    if args.staging_release_manifest:
        staging = _read(args.staging_release_manifest.resolve())
        staged_bytes = {
            item["variant"]: item["model_artifact_bytes"] for item in staging["artifacts"]
        }
        for artifact in artifacts:
            if artifact["published"]:
                artifact["artifact_bytes"] = staged_bytes[artifact["implementation"]]
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "benchmark_suite_version": {
            "quality": "public-asr-quality-v1",
            "performance": recipe["benchmark_suite_version"],
            "normalization": "gigaam-multilingual-mlx-text-v1",
        },
        "code_revision": args.code_revision,
        "environment": {
            "machine": args.machine,
            "chip": args.chip,
            "memory_bytes": args.memory_bytes,
            "power_state": args.power_state,
            "platform": sample["environment"]["platform"],
            "python": sample["environment"]["python"],
            "gigaam_multilingual_mlx": sample["environment"].get(
                "gigaam_multilingual_mlx",
                sample["environment"].get("gigaam_mlx"),  # pre-release report field
            ),
            "mlx": sample["environment"]["mlx"],
            "torch": original_sample["environment"]["torch"],
            "mlx_device": sample["environment"]["device"],
        },
        "implementations": list(IMPLEMENTATIONS.values()),
        "artifacts": artifacts,
        "published_model_artifact_bytes_total": sum(
            item["artifact_bytes"] for item in artifacts if item["published"]
        ),
        "datasets": _datasets(manifest_summary),
        "performance_corpus": recipe,
        "commands": _commands(),
        "quality_results": _quality_rows(quality_root),
        "performance_results": _performance_rows(performance_root, recipe),
        "limitations": [
            "Measurements are from one MacBook Pro with Apple M4 Pro and 48 GB unified memory; M1-M5 support is not a claim of identical performance.",
            "Short/medium use five measured warm runs after one cold run; long uses three measured warm runs after one cold run.",
            "Peak RSS is peak resident process memory. Device/Metal memory is reported separately and must not be added to RSS because Apple Silicon uses unified memory.",
            "Swap counters are system-wide; every measured run had zero positive swap delta, while negative deltas mean the OS reclaimed existing swap.",
            "Weight-only quantization reduces artifact and memory size but does not guarantee monotonic speedups for every sequence length.",
            "Word timestamps are approximate greedy-CTC frame alignments, not forced alignment.",
            "PyTorch 2.10 emitted a non-fatal STFT output-resize deprecation warning during batched quality evaluation.",
            "The quality suite uses deterministic public subsets and is not presented as a full-dataset leaderboard result.",
        ],
        "private_inputs_excluded": True,
        "raw_reports_published": False,
    }


def _gb(value: int | None) -> str:
    return "n/a" if value is None else f"{value / 1_000_000_000:.3f}"


def _quality_table(report: dict[str, Any]) -> list[str]:
    rows = report["quality_results"]
    lines = [
        "| Implementation | FLEURS WER/CER | Russian LibriSpeech WER/CER | SOVA WER/CER | Max abs. Δ WER vs Original | Max abs. Δ WER vs FP16 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for implementation, metadata in IMPLEMENTATIONS.items():
        values = [row for row in rows if row["implementation"] == implementation]
        by_dataset = {row["dataset"]: row for row in values}
        cells = []
        for dataset in DATASET_NAMES.values():
            item = by_dataset[dataset]
            cells.append(f"{item['wer_percent']:.3f}% / {item['cer_percent']:.3f}%")
        original_delta = max(abs(item["wer_delta_pp_vs_original"]) for item in values)
        fp16_delta = max(abs(item["wer_delta_pp_vs_fp16"]) for item in values)
        lines.append(
            f"| {metadata['label']} | {' | '.join(cells)} | {original_delta:.3f} pp | "
            f"{fp16_delta:.3f} pp |"
        )
    return lines


def _performance_table(report: dict[str, Any], profile: str) -> list[str]:
    rows = [row for row in report["performance_results"] if row["profile"] == profile]
    lines = [
        "| Implementation | Load | Cold | Warm median / p95 | Warm RTF | Speed vs Original | Peak RSS | Device peak | Swap increase | Words |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for implementation, metadata in IMPLEMENTATIONS.items():
        row = next(item for item in rows if item["implementation"] == implementation)
        lines.append(
            f"| {metadata['label']} | {row['model_load_seconds']:.3f}s | "
            f"{row['cold_wall_seconds']:.3f}s | {row['warm_median_seconds']:.3f}s / "
            f"{row['warm_p95_seconds']:.3f}s | {row['warm_median_rtf']:.4f} | "
            f"{row['warm_speedup_vs_original']:.2f}× | {_gb(row['peak_rss_bytes'])} GB | "
            f"{_gb(row['peak_device_bytes'])} GB | {row['swap_increase_bytes'] / 1_000_000:.0f} MB | "
            f"{row['output_word_count_median']} |"
        )
    return lines


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# GigaAM-Multilingual MLX v0.1.0 benchmark",
        "",
        "Apple M4 Pro (48 GB), macOS 15.7.7, public-only suites, greedy CTC, "
        "20-second chunks with 2-second overlap. Cold and warm timings are separated; "
        "Original PyTorch/MPS is the baseline.",
        "",
        "## Artifacts",
        "",
        "| Implementation | Repository / revision | Model size | Model artifact | SHA-256 |",
        "|---|---|---:|---:|---|",
    ]
    for item in report["artifacts"]:
        metadata = IMPLEMENTATIONS[item["implementation"]]
        lines.append(
            f"| {metadata['label']} | `{item['repository']}@{item['revision']}` | "
            f"{_gb(item['weights_bytes'])} GB | {_gb(item['artifact_bytes'])} GB | "
            f"`{item['weights_sha256']}` |"
        )
    lines.extend(
        [
            "",
            f"The four published MLX model artifacts total "
            f"{report['published_model_artifact_bytes_total'] / 1_000_000_000:.3f} GB. "
            "INT8 g64 is the default because it preserves measured quality while substantially "
            "reducing disk and resident-memory requirements; it is not claimed to be the fastest "
            "variant for every duration.",
            "",
            "## Quality: `public-asr-quality-v1`",
            "",
            "WER/CER are corpus metrics after `gigaam-multilingual-mlx-text-v1` normalization. All rows use "
            "the same 1,533 public items and batch size 8.",
            "",
            *_quality_table(report),
            "",
            "Full operation counts, exact-match counts, dataset revisions, licenses, manifests, "
            "and deltas versus both Original and FP16 are in "
            "[`benchmark-v0.1.0.json`](benchmark-v0.1.0.json).",
            "",
            "## Performance: `public-asr-perf-v1`",
            "",
            "MLX runs use `mx.compile`; Original uses upstream PyTorch/MPS. Short and medium "
            "have one cold plus five measured warm runs; long has one cold plus three measured "
            "warm runs. Download time is excluded from model load and inference.",
        ]
    )
    durations = {"short": "30 seconds", "medium": "5 minutes", "long": "20 minutes"}
    for profile in ("short", "medium", "long"):
        lines.extend(["", f"### {durations[profile]}", "", *_performance_table(report, profile)])
    lines.extend(
        [
            "",
            "Peak RSS means peak resident memory of the process, not model-only memory. Metal/MPS "
            "device memory is a separate unified-memory observation and is not added to RSS. All "
            "runs had 0 MB positive swap growth.",
            "",
            "## Reproduction contract",
            "",
            "- Quality manifests: [`../benchmarks/manifests/`](../benchmarks/manifests/)",
            "- Performance recipe: "
            "[`../benchmarks/public-asr-perf-v1.json`](../benchmarks/public-asr-perf-v1.json)",
            "- Methodology: [`../benchmarks/README.md`](../benchmarks/README.md)",
            f"- Code revision: `{report['code_revision']}`",
            "- Upstream model: "
            "[`ai-sage/GigaAM-Multilingual`](https://huggingface.co/ai-sage/GigaAM-Multilingual)",
            "- Upstream repository: "
            "[`salute-developers/GigaAM`](https://github.com/salute-developers/GigaAM)",
            "",
            "Exact path-independent command templates are recorded in the JSON report. Raw "
            "hypotheses and generated WAV files remain outside Git under the machine-local ML "
            "data root.",
            "",
            "## Limitations",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in report["limitations"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quality-root", type=Path, required=True)
    parser.add_argument("--performance-root", type=Path, required=True)
    parser.add_argument("--weights-root", type=Path, required=True)
    parser.add_argument("--quality-manifest-summary", type=Path, required=True)
    parser.add_argument("--performance-recipe", type=Path, required=True)
    parser.add_argument("--staging-release-manifest", type=Path)
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--machine", required=True)
    parser.add_argument("--chip", required=True)
    parser.add_argument("--memory-bytes", type=int, required=True)
    parser.add_argument("--power-state", required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args()
    report = build(args)
    args.json_output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    args.markdown_output.write_text(markdown(report))


if __name__ == "__main__":
    main()
