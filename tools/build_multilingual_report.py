#!/usr/bin/env python3
"""Build compact multilingual benchmark JSON and Markdown from raw external reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


LANGUAGES = {
    "ru_ru": "Russian",
    "kk_kz": "Kazakh",
    "ky_kg": "Kyrgyz",
    "uz_uz": "Uzbek",
    "en_us": "English appendix",
}
IMPLEMENTATIONS = {
    "original": "Original GigaAM PyTorch/MPS",
    "fp16": "GigaAM MLX FP16",
    "int8": "GigaAM MLX INT8 g64",
    "int6": "GigaAM MLX INT6 g64",
    "int4": "GigaAM MLX INT4 g64",
    "whisper-large-v2": "MLX Whisper large-v2",
    "whisper-large-v3": "MLX Whisper large-v3",
    "whisper-turbo": "MLX Whisper large-v3-turbo",
    "parakeet-v3": "MLX Parakeet TDT 0.6B v3",
}
HEADLINE = (
    "int8",
    "parakeet-v3",
    "whisper-large-v2",
    "whisper-large-v3",
    "whisper-turbo",
)
WHISPER_BASELINES = ("whisper-large-v2", "whisper-large-v3", "whisper-turbo")
PARAKEET_BASELINE = "parakeet-v3"
PARAKEET_QUALITY_LANGUAGES = ("ru_ru", "en_us")
PARAKEET_PERFORMANCE_LANGUAGES = ("ru_ru",)
CORE_LANGUAGES = ("ru_ru", "kk_kz", "ky_kg", "uz_uz")
DECISION_ORDER = (
    "original",
    "fp16",
    "int8",
    "int6",
    "int4",
    "whisper-large-v2",
    "whisper-large-v3",
    "whisper-turbo",
    "parakeet-v3",
)
BOOTSTRAP_SEED = 20260716
BOOTSTRAP_ITERATIONS = 10_000


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _quality_path(root: Path, implementation: str, language: str) -> Path:
    direct = root / "quality" / implementation / f"{language}.json"
    if direct.exists():
        return direct
    return root / "quality" / implementation / f"{implementation}-{language}.json"


def _performance_path(root: Path, implementation: str, language: str) -> Path:
    return root / "performance" / implementation / f"{language}.json"


def _item_arrays(report: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    errors = np.array(
        [
            item["substitutions"] + item["deletions"] + item["insertions"]
            for item in report["items"]
        ],
        dtype=np.float64,
    )
    words = np.array(
        [item["substitutions"] + item["deletions"] + item["hits"] for item in report["items"]],
        dtype=np.float64,
    )
    return errors, words, [item["id"] for item in report["items"]]


def _bootstrap(
    reports: dict[str, dict[str, Any]], language: str
) -> tuple[dict[str, list[float]], dict[str, dict[str, list[float]]]]:
    rng = np.random.default_rng(BOOTSTRAP_SEED + list(LANGUAGES).index(language))
    arrays = {name: _item_arrays(report) for name, report in reports.items()}
    ids = next(iter(arrays.values()))[2]
    if any(values[2] != ids for values in arrays.values()):
        raise ValueError(f"Item mismatch for {language}")
    samples = {name: [] for name in arrays}
    for _ in range(0, BOOTSTRAP_ITERATIONS, 250):
        count = min(
            250,
            BOOTSTRAP_ITERATIONS - sum(len(values) for values in samples.values()) // len(samples),
        )
        indices = rng.integers(0, len(ids), size=(count, len(ids)))
        for name, (errors, words, _) in arrays.items():
            samples[name].extend(
                (errors[indices].sum(axis=1) / words[indices].sum(axis=1)).tolist()
            )
    ci = {
        name: [float(value) for value in np.quantile(values, [0.025, 0.975])]
        for name, values in samples.items()
    }
    comparisons: dict[str, dict[str, list[float]]] = {}
    for candidate in ("int8",):
        comparisons[candidate] = {}
        baselines = (*WHISPER_BASELINES, PARAKEET_BASELINE)
        for baseline in baselines:
            if baseline not in samples:
                continue
            candidate_values = np.array(samples[candidate])
            baseline_values = np.array(samples[baseline])
            delta = (candidate_values - baseline_values) * 100
            reduction = (baseline_values - candidate_values) / baseline_values
            comparisons[candidate][baseline] = {
                "wer_delta_pp_ci95": [float(value) for value in np.quantile(delta, [0.025, 0.975])],
                "relative_error_reduction_ci95": [
                    float(value) for value in np.quantile(reduction, [0.025, 0.975])
                ],
            }
    return ci, comparisons


def _model_selection(
    quality: list[dict[str, Any]], performance: list[dict[str, Any]]
) -> dict[str, Any]:
    quality_lookup = {
        (row["implementation"], row["language"]): row["wer_percent"] for row in quality
    }
    performance_lookup = {
        row["implementation"]: row for row in performance if row["language"] == "ru_ru"
    }
    rows = []
    for implementation in DECISION_ORDER:
        wers = {language: quality_lookup.get((implementation, language)) for language in LANGUAGES}
        core_values = [wers[language] for language in CORE_LANGUAGES]
        performance_row = performance_lookup[implementation]
        rows.append(
            {
                "implementation": implementation,
                "label": IMPLEMENTATIONS[implementation],
                "wer_percent": wers,
                "core_macro_wer_percent": (
                    sum(core_values) / len(core_values)
                    if all(value is not None for value in core_values)
                    else None
                ),
                "five_minute_wav_seconds": performance_row["warm_median_seconds"],
                "peak_rss_bytes": performance_row["peak_rss_bytes"],
                "model_size_bytes": performance_row["artifact_bytes"],
            }
        )

    eligible = [
        row
        for row in rows
        if row["implementation"] != "original" and row["core_macro_wer_percent"] is not None
    ]

    def vector(row: dict[str, Any]) -> tuple[float, float, int, int]:
        return (
            row["core_macro_wer_percent"],
            row["five_minute_wav_seconds"],
            row["peak_rss_bytes"],
            row["model_size_bytes"],
        )

    pareto = []
    for candidate in eligible:
        candidate_vector = vector(candidate)
        dominated = any(
            all(left <= right for left, right in zip(vector(other), candidate_vector, strict=True))
            and any(
                left < right for left, right in zip(vector(other), candidate_vector, strict=True)
            )
            for other in eligible
            if other is not candidate
        )
        if not dominated:
            pareto.append(candidate["implementation"])

    metric_values = {
        "ru_wer_percent": {row["implementation"]: row["wer_percent"]["ru_ru"] for row in rows},
        "kk_wer_percent": {row["implementation"]: row["wer_percent"]["kk_kz"] for row in rows},
        "ky_wer_percent": {row["implementation"]: row["wer_percent"]["ky_kg"] for row in rows},
        "uz_wer_percent": {row["implementation"]: row["wer_percent"]["uz_uz"] for row in rows},
        "en_wer_percent": {row["implementation"]: row["wer_percent"]["en_us"] for row in rows},
        "five_minute_wav_seconds": {
            row["implementation"]: row["five_minute_wav_seconds"] for row in rows
        },
        "peak_rss_bytes": {row["implementation"]: row["peak_rss_bytes"] for row in rows},
        "model_size_bytes": {row["implementation"]: row["model_size_bytes"] for row in rows},
    }
    leaders = {}
    for metric, values in metric_values.items():
        observed = {name: value for name, value in values.items() if value is not None}
        minimum = min(observed.values())
        leaders[metric] = [
            name
            for name, value in observed.items()
            if np.isclose(value, minimum, rtol=0, atol=1e-9)
        ]

    return {
        "pareto": {
            "definition": (
                "Non-dominated MLX candidates by equal-weight macro WER across "
                "ru_ru/kk_kz/ky_kg/uz_uz, Russian 5-minute warm median, peak RSS, "
                "and model weight bytes; Original is a reference baseline, English "
                "is an appendix, and models missing a core language are ineligible."
            ),
            "implementations": pareto,
        },
        "leaders": leaders,
        "rows": rows,
    }


def build(
    runs_root: Path, manifests: dict[str, Any], run_context: dict[str, Any]
) -> dict[str, Any]:
    quality: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    performance: list[dict[str, Any]] = []
    for language in LANGUAGES:
        implementations = tuple(
            implementation
            for implementation in IMPLEMENTATIONS
            if implementation != PARAKEET_BASELINE or language in PARAKEET_QUALITY_LANGUAGES
        )
        reports = {
            implementation: _read(_quality_path(runs_root, implementation, language))
            for implementation in implementations
        }
        ci, bootstrap_comparisons = _bootstrap(reports, language)
        for implementation, report in reports.items():
            summary = report["summary"]
            quality.append(
                {
                    "language": language,
                    "implementation": implementation,
                    "files": summary["files"],
                    "audio_seconds": summary["audio_seconds"],
                    "wer_percent": summary["wer_percent"],
                    "wer_ci95_percent": [value * 100 for value in ci[implementation]],
                    "cer_percent": summary["cer_percent"],
                    "substitutions": summary["substitutions"],
                    "deletions": summary["deletions"],
                    "insertions": summary["insertions"],
                    "hits": summary["hits"],
                    "reference_words": sum(
                        len(item["reference"].split()) for item in report["items"]
                    ),
                    "reference_characters_without_spaces": sum(
                        len(item["reference"].replace(" ", "")) for item in report["items"]
                    ),
                    "hypothesis_words": sum(
                        len(item["hypothesis"].split()) for item in report["items"]
                    ),
                    "hypothesis_characters_without_spaces": sum(
                        len(item["hypothesis"].replace(" ", "")) for item in report["items"]
                    ),
                    "empty_hypotheses": summary["empty_hypotheses"],
                    "exact_matches": summary["exact_matches"],
                    "detected_languages": report["backend"].get("detected_languages"),
                }
            )
        int8_wer = reports["int8"]["summary"]["wer"]
        for baseline in (*WHISPER_BASELINES, PARAKEET_BASELINE):
            if baseline not in reports:
                continue
            baseline_wer = reports[baseline]["summary"]["wer"]
            comparisons.append(
                {
                    "language": language,
                    "candidate": "int8",
                    "baseline": baseline,
                    "wer_delta_pp": (int8_wer - baseline_wer) * 100,
                    "relative_error_reduction": (baseline_wer - int8_wer) / baseline_wer,
                    **bootstrap_comparisons["int8"][baseline],
                }
            )
        if language != "en_us":
            performance_implementations = tuple(
                implementation
                for implementation in IMPLEMENTATIONS
                if implementation != PARAKEET_BASELINE or language in PARAKEET_PERFORMANCE_LANGUAGES
            )
            for implementation in performance_implementations:
                report = _read(_performance_path(runs_root, implementation, language))
                warm = report["warm_summary"]
                duration = report["audio"]["duration_seconds"]
                cold_seconds = report["runs"][0].get("wall_seconds") or report["runs"][0].get(
                    "end_to_end_wall_seconds"
                )
                warm_median_seconds = warm["median_end_to_end_seconds"]
                warm_p95_seconds = warm["p95_end_to_end_seconds"]
                performance.append(
                    {
                        "language": language,
                        "implementation": implementation,
                        "artifact_bytes": report["artifact"].get("weights_bytes")
                        or report["artifact"].get("bytes"),
                        "load_seconds": report["load_seconds"],
                        "cold_seconds": cold_seconds,
                        "cold_rtf": cold_seconds / duration,
                        "warm_runs": warm["runs"],
                        "warm_median_seconds": warm_median_seconds,
                        "warm_p95_seconds": warm_p95_seconds,
                        "warm_p95_rtf": warm_p95_seconds / duration,
                        "warm_median_rtf": warm.get("median_rtf") or warm_median_seconds / duration,
                        "warm_median_audio_seconds_per_second": duration / warm_median_seconds,
                        "peak_rss_bytes": report["peak_rss_bytes"],
                        "peak_device_bytes": report.get("peak_device_bytes"),
                        "peak_active_device_bytes": report.get("peak_active_device_bytes"),
                        "peak_cache_device_bytes": report.get("peak_cache_device_bytes"),
                        "minimum_available_memory_bytes": report.get(
                            "minimum_available_memory_bytes"
                        ),
                        "swap_used_start_bytes": report["swap_used_start_bytes"],
                        "swap_used_end_bytes": report["swap_used_end_bytes"],
                        "swap_delta_bytes": max(
                            0, report["swap_used_end_bytes"] - report["swap_used_start_bytes"]
                        ),
                    }
                )
    reference_reports = {
        implementation: _read(_performance_path(runs_root, implementation, "ru_ru"))
        for implementation in IMPLEMENTATIONS
    }
    artifacts = [
        {
            "implementation": implementation,
            "label": IMPLEMENTATIONS[implementation],
            **report["artifact"],
        }
        for implementation, report in reference_reports.items()
    ]
    environments = [
        {
            "implementation": implementation,
            **report["environment"],
        }
        for implementation, report in reference_reports.items()
    ]
    performance_corpora = []
    for language in ("ru_ru", "kk_kz", "ky_kg", "uz_uz"):
        report = _read(_performance_path(runs_root, "original", language))
        performance_corpora.append({"language": language, **report["audio"]})
    performance_paths = [
        _performance_path(runs_root, implementation, language)
        for implementation in IMPLEMENTATIONS
        for language in ("ru_ru", "kk_kz", "ky_kg", "uz_uz")
        if implementation != PARAKEET_BASELINE or language in PARAKEET_PERFORMANCE_LANGUAGES
    ]
    performance_run_order = [
        {
            "implementation": path.parent.name,
            "language": path.stem,
        }
        for path in sorted(performance_paths, key=lambda path: path.stat().st_mtime_ns)
    ]
    performance_comparisons = []
    for language in ("ru_ru", "kk_kz", "ky_kg", "uz_uz"):
        candidate = next(
            row
            for row in performance
            if row["language"] == language and row["implementation"] == "int8"
        )
        baselines = (*WHISPER_BASELINES,)
        if language in PARAKEET_PERFORMANCE_LANGUAGES:
            baselines += (PARAKEET_BASELINE,)
        for baseline in baselines:
            reference = next(
                row
                for row in performance
                if row["language"] == language and row["implementation"] == baseline
            )
            performance_comparisons.append(
                {
                    "language": language,
                    "candidate": "int8",
                    "baseline": baseline,
                    "warm_speedup": reference["warm_median_seconds"]
                    / candidate["warm_median_seconds"],
                    "peak_rss_reduction": 1
                    - candidate["peak_rss_bytes"] / reference["peak_rss_bytes"],
                    "weights_reduction": 1
                    - candidate["artifact_bytes"] / reference["artifact_bytes"],
                }
            )
    model_selection = _model_selection(quality, performance)
    return {
        "schema_version": 1,
        "benchmark_suite_version": "public-asr-multilingual-v1",
        "status": "draft-not-published",
        "bootstrap": {
            "seed": BOOTSTRAP_SEED,
            "iterations": BOOTSTRAP_ITERATIONS,
            "unit": "utterance",
            "interval": "percentile-95",
        },
        "datasets": manifests["datasets"],
        "run_context": run_context,
        "performance_corpora": performance_corpora,
        "artifacts": artifacts,
        "environments": environments,
        "quality": quality,
        "comparisons": comparisons,
        "performance": performance,
        "performance_comparisons": performance_comparisons,
        "performance_run_order": performance_run_order,
        "model_selection": model_selection,
        "support": {
            "parakeet-v3": {
                "quality_languages": list(PARAKEET_QUALITY_LANGUAGES),
                "performance_languages": list(PARAKEET_PERFORMANCE_LANGUAGES),
                "unsupported_languages": ["kk_kz", "ky_kg", "uz_uz"],
            }
        },
        "command_templates": {
            "mlx_quality": "python tools/run_multilingual_quality.py --implementation mlx --model MODEL --manifest MANIFEST --language LANGUAGE --batch-size 8 --output REPORT",
            "pytorch_quality": "python tools/run_multilingual_quality.py --implementation pytorch --model MODEL --manifest MANIFEST --language LANGUAGE --batch-size 8 --source-device mps --output REPORT",
            "whisper_quality": "python tools/run_multilingual_whisper_matrix.py --model MODEL --model-name NAME --slice LANGUAGE=WHISPER_LANGUAGE=MANIFEST --output-dir REPORT_DIR",
            "mlx_performance": "python -m gigaam_multilingual_mlx.dev_cli benchmark AUDIO --model MODEL --compile --warm-runs 5 --output REPORT",
            "pytorch_performance": "python -m gigaam_multilingual_mlx.dev_cli benchmark-pytorch AUDIO --source MODEL --source-device mps --warm-runs 5 --output REPORT",
            "whisper_performance": "python tools/run_whisper_benchmark.py AUDIO --model MODEL --language LANGUAGE_OR_AUTO --warm-runs 5 --output REPORT",
            "parakeet_quality": "python tools/run_parakeet_quality.py --model MODEL --model-revision REVISION --manifest MANIFEST --language LANGUAGE --output REPORT",
            "parakeet_performance": "python tools/run_parakeet_benchmark.py AUDIO --model MODEL --model-revision REVISION --warm-runs 5 --output REPORT",
        },
        "limitations": [
            "Only Apple M4 Pro / 48 GB was benchmarked.",
            "Whisper has no Kyrgyz language token; Kyrgyz uses automatic language detection.",
            "Whisper large-v2 is included as a practical production reference, but this suite uses the common short-form benchmark decoding contract.",
            "English is an appendix language and not part of the core GigaAM claim.",
            "Parakeet TDT 0.6B v3 officially supports Russian and English, but not Kazakh, Kyrgyz, or Uzbek; unsupported cells are N/A and were not run.",
            "All results use deterministic public FLEURS selections, not a universal ASR leaderboard.",
        ],
    }


def _quality(result: dict[str, Any], language: str, implementation: str) -> dict[str, Any]:
    return next(
        row
        for row in result["quality"]
        if row["language"] == language and row["implementation"] == implementation
    )


def _quality_or_none(
    result: dict[str, Any], language: str, implementation: str
) -> dict[str, Any] | None:
    return next(
        (
            row
            for row in result["quality"]
            if row["language"] == language and row["implementation"] == implementation
        ),
        None,
    )


def _selection_row(result: dict[str, Any], implementation: str) -> dict[str, Any]:
    return next(
        row for row in result["model_selection"]["rows"] if row["implementation"] == implementation
    )


def _selection_label(result: dict[str, Any], implementation: str) -> str:
    markers = []
    if implementation in result["model_selection"]["pareto"]["implementations"]:
        markers.append("◇")
    if implementation == "int8":
        markers.append("★")
    return " ".join((*markers, IMPLEMENTATIONS[implementation])).strip()


def _marked_value(result: dict[str, Any], implementation: str, metric: str, value: str) -> str:
    if implementation in result["model_selection"]["leaders"][metric]:
        return f"**✓ {value}**"
    return value


def _model_selection_lines(result: dict[str, Any]) -> list[str]:
    selection = result["model_selection"]
    roles = {
        "fp16": "fastest; reference-port fidelity",
        "int8": "recommended quality / speed / footprint balance",
        "int6": "smaller footprint with near-INT8 quality",
        "int4": "minimum model size and peak RAM",
    }
    lines = [
        "",
        "## Model selection matrix",
        "",
        "Filled pill: column leader · outlined pill: column runner-up · `◇` Pareto frontier ·",
        "`★` recommended default. Lower is better.",
        "English is an appendix. `5-min WAV` is the Russian five-minute warm median after",
        "model load; Peak RAM is whole-process peak RSS; model size is the weight file.",
        "",
        "### ◇ Pareto frontier — focus models",
        "",
        "The frontier uses equal-weight macro WER over Russian, Kazakh, Kyrgyz, and Uzbek",
        "plus 5-min WAV time, Peak RAM, and model size among MLX candidates. Original is a",
        "reference baseline; English is excluded as an appendix; models missing a core",
        "language are ineligible.",
        "",
        "| Pareto MLX variant | Core macro WER | 5-min WAV | Peak RAM | Model size | Best fit |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for implementation in selection["pareto"]["implementations"]:
        row = _selection_row(result, implementation)
        label = _selection_label(result, implementation)
        values = [
            label,
            f"{row['core_macro_wer_percent']:.3f}%",
            _marked_value(
                result,
                implementation,
                "five_minute_wav_seconds",
                f"{row['five_minute_wav_seconds']:.3f}s",
            ),
            _marked_value(
                result,
                implementation,
                "peak_rss_bytes",
                f"{row['peak_rss_bytes'] / 1e9:.3f} GB",
            ),
            _marked_value(
                result,
                implementation,
                "model_size_bytes",
                f"{row['model_size_bytes'] / 1e9:.3f} GB",
            ),
            roles[implementation],
        ]
        if implementation == "int8":
            values = [f"**{value}**" for value in values]
        lines.append("| " + " | ".join(values) + " |")

    lines.extend(
        [
            "",
            "### All compared models",
            "",
            "| Model / variant | RU WER | KZ WER | KY WER | UZ WER | EN WER | 5-min WAV | Peak RAM | Model size |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    metric_by_language = {
        "ru_ru": "ru_wer_percent",
        "kk_kz": "kk_wer_percent",
        "ky_kg": "ky_wer_percent",
        "uz_uz": "uz_wer_percent",
        "en_us": "en_wer_percent",
    }
    for implementation in DECISION_ORDER:
        row = _selection_row(result, implementation)
        values = [_selection_label(result, implementation)]
        for language in LANGUAGES:
            wer = row["wer_percent"][language]
            formatted = "N/A" if wer is None else f"{wer:.3f}%"
            values.append(
                _marked_value(
                    result,
                    implementation,
                    metric_by_language[language],
                    formatted,
                )
                if wer is not None
                else formatted
            )
        values.extend(
            [
                _marked_value(
                    result,
                    implementation,
                    "five_minute_wav_seconds",
                    f"{row['five_minute_wav_seconds']:.3f}s",
                ),
                _marked_value(
                    result,
                    implementation,
                    "peak_rss_bytes",
                    f"{row['peak_rss_bytes'] / 1e9:.3f} GB",
                ),
                _marked_value(
                    result,
                    implementation,
                    "model_size_bytes",
                    f"{row['model_size_bytes'] / 1e9:.3f} GB",
                ),
            ]
        )
        lines.append("| " + " | ".join(values) + " |")
    return lines


def markdown(result: dict[str, Any]) -> str:
    lines = [
        "# GigaAM Multilingual MLX vs MLX Whisper and Parakeet",
        "",
        "![ASR model decision matrix with Pareto frontier](benchmark-multilingual-v1.png)",
        "",
        "Public benchmark. Lower is better for WER, time, memory, and size.",
        "",
        "## Headline quality",
        "",
        "| Language | "
        + " | ".join(f"{IMPLEMENTATIONS[name]} WER (95% CI)" for name in HEADLINE)
        + " |",
        "|---|" + "---:|" * len(HEADLINE),
    ]
    for language in ("ru_ru", "kk_kz", "ky_kg", "uz_uz"):
        values = [_quality_or_none(result, language, implementation) for implementation in HEADLINE]
        formatted = []
        for value in values:
            if value is None:
                formatted.append("N/A")
            else:
                formatted.append(
                    f"{value['wer_percent']:.3f}% ({value['wer_ci95_percent'][0]:.3f}–"
                    f"{value['wer_ci95_percent'][1]:.3f}%)"
                )
        lines.append(f"| {LANGUAGES[language]} | " + " | ".join(formatted) + " |")
    lines.extend(
        [
            "",
            "## English appendix",
            "",
            "| Implementation | WER | CER |",
            "|---|---:|---:|",
        ]
    )
    for implementation in HEADLINE:
        value = _quality(result, "en_us", implementation)
        lines.append(
            f"| {IMPLEMENTATIONS[implementation]} | {value['wer_percent']:.3f}% | "
            f"{value['cer_percent']:.3f}% |"
        )
    lines.extend(_model_selection_lines(result))
    lines.extend(
        [
            "",
            "## Full quality matrix",
            "",
            "| Language | Implementation | WER | CER | Empty | Exact matches |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for language in LANGUAGES:
        for implementation in IMPLEMENTATIONS:
            value = _quality_or_none(result, language, implementation)
            if value is None:
                lines.append(
                    f"| {LANGUAGES[language]} | {IMPLEMENTATIONS[implementation]} | "
                    "N/A | N/A | N/A | N/A |"
                )
                continue
            lines.append(
                f"| {LANGUAGES[language]} | {IMPLEMENTATIONS[implementation]} | "
                f"{value['wer_percent']:.3f}% | {value['cer_percent']:.3f}% | "
                f"{value['empty_hypotheses']} | {value['exact_matches']} |"
            )
    lines.extend(
        [
            "",
            "## GigaAM INT8 compared with MLX baselines",
            "",
            "Positive relative error reduction means lower WER for GigaAM INT8.",
            "",
            "| Language | Baseline | WER delta | Relative error reduction (95% CI) |",
            "|---|---|---:|---:|",
        ]
    )
    for comparison in result["comparisons"]:
        interval = comparison["relative_error_reduction_ci95"]
        lines.append(
            f"| {LANGUAGES[comparison['language']]} | "
            f"{IMPLEMENTATIONS[comparison['baseline']]} | "
            f"{comparison['wer_delta_pp']:+.3f} pp | "
            f"{comparison['relative_error_reduction'] * 100:+.1f}% "
            f"({interval[0] * 100:+.1f}%–{interval[1] * 100:+.1f}%) |"
        )
    lines.extend(
        [
            "",
            "## GigaAM INT8 resource advantage over MLX baselines",
            "",
            "| Language | Baseline | Warm speedup | Peak RSS reduction | Model size reduction |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for comparison in result["performance_comparisons"]:
        lines.append(
            f"| {LANGUAGES[comparison['language']]} | "
            f"{IMPLEMENTATIONS[comparison['baseline']]} | "
            f"{comparison['warm_speedup']:.2f}× | "
            f"{comparison['peak_rss_reduction'] * 100:.1f}% | "
            f"{comparison['weights_reduction'] * 100:.1f}% |"
        )
    lines.extend(["", "## Performance by language", ""])
    for language in ("ru_ru", "kk_kz", "ky_kg", "uz_uz"):
        lines.extend(
            [
                f"### {LANGUAGES[language]}",
                "",
                "| Implementation | Model size | Load | 5-min warm | Peak RSS |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for implementation in IMPLEMENTATIONS:
            row = next(
                (
                    item
                    for item in result["performance"]
                    if item["language"] == language and item["implementation"] == implementation
                ),
                None,
            )
            if row is None:
                lines.append(f"| {IMPLEMENTATIONS[implementation]} | N/A | N/A | N/A | N/A |")
                continue
            lines.append(
                f"| {IMPLEMENTATIONS[implementation]} | {row['artifact_bytes'] / 1e9:.3f} GB | "
                f"{row['load_seconds']:.3f}s | {row['warm_median_seconds']:.3f}s | "
                f"{row['peak_rss_bytes'] / 1e9:.3f} GB |"
            )
        lines.append("")
    lines.extend(
        [
            "## Methodology notes",
            "",
            "- FLEURS test at pinned revision; audio >30s and references with digits excluded.",
            "- GigaAM uses greedy CTC; Whisper uses greedy temperature-0 transcription.",
            "- Parakeet uses its native BF16 greedy TDT decoder with auto language detection; quality is full-attention and the 5-minute performance run uses 120s chunks with 15s overlap.",
            "- Parakeet does not officially support Kazakh, Kyrgyz, or Uzbek; those cells are N/A and were not measured.",
            "- Kyrgyz Whisper uses auto-detect because Whisper has no `ky` language token.",
            "- Performance runs are isolated processes with one cold and five warm runs.",
            "- Full manifests, confidence intervals, revisions, hashes, CER, and limitations are in the JSON report.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--manifests", type=Path, required=True)
    parser.add_argument("--run-context", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    result = build(
        args.runs_root.resolve(),
        _read(args.manifests.resolve()),
        _read(args.run_context.resolve()),
    )
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    args.output_md.write_text(markdown(result))


if __name__ == "__main__":
    main()
