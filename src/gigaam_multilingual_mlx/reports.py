from __future__ import annotations

import json
from pathlib import Path


def compare_quality_reports(paths: list[str | Path]) -> dict:
    reports = [json.loads(Path(path).read_text()) for path in paths]
    names = [
        (
            f"{report['backend']['runtime']}-{report['backend']['quantization_profile']}"
            if report["backend"].get("quantization_profile")
            else f"{report['backend']['runtime']}-{report['backend']['dtype']}"
        )
        for report in reports
    ]
    if len(set(names)) != len(names):
        raise ValueError(f"Quality report names are not unique: {names}")
    items_by_report = [{item["id"]: item for item in report["items"]} for report in reports]
    ids = list(items_by_report[0])
    if any(set(items) != set(ids) for items in items_by_report[1:]):
        raise ValueError("Quality reports contain different item sets")
    disagreements = []
    for item_id in ids:
        hypotheses = {
            name: items[item_id]["hypothesis"]
            for name, items in zip(names, items_by_report, strict=True)
        }
        if len(set(hypotheses.values())) > 1:
            disagreements.append(
                {
                    "id": item_id,
                    "reference": items_by_report[0][item_id]["reference"],
                    "hypotheses": hypotheses,
                }
            )
    fp32 = next(
        report
        for report in reports
        if report["backend"]["runtime"] == "mlx" and report["backend"]["dtype"] == "float32"
    )
    fp16 = next(
        report
        for report in reports
        if report["backend"]["runtime"] == "mlx"
        and report["backend"]["dtype"] == "float16"
        and not report["backend"].get("quantization_profile")
    )
    return {
        "schema_version": 1,
        "reports": {
            name: {
                "wer_percent": report["summary"]["wer_percent"],
                "delta_vs_mlx_fp32_percentage_points": report["summary"]["wer_percent"]
                - fp32["summary"]["wer_percent"],
                "delta_vs_mlx_float16_percentage_points": report["summary"]["wer_percent"]
                - fp16["summary"]["wer_percent"],
                "empty_hypotheses": report["summary"]["empty_hypotheses"],
                "audio_seconds_per_second": report["summary"]["audio_seconds_per_second"],
            }
            for name, report in zip(names, reports, strict=True)
        },
        "disagreement_count": len(disagreements),
        "disagreements": disagreements,
    }
