from __future__ import annotations

import json
from pathlib import Path

import pytest

from gigaam_multilingual_mlx import __version__
from gigaam_multilingual_mlx.cli import (
    _ensure_supported_platform,
    _models_output,
    _output_path,
    _parser,
    _selected_format,
    _short_cli_args,
)


def test_public_cli_exposes_only_user_commands() -> None:
    parser = _parser()
    subparsers = next(action for action in parser._actions if action.dest == "command")
    assert set(subparsers.choices) == {"transcribe", "models"}
    help_text = parser.format_help()
    assert "film" not in help_text.lower()
    assert "convert" not in help_text.lower()
    assert "benchmark" not in help_text.lower()


def test_version_is_release_version() -> None:
    assert __version__ == "0.1.1"


def test_short_cli_accepts_audio_without_transcribe_subcommand() -> None:
    parser = _parser("gigaam-stt")
    args = parser.parse_args(_short_cli_args(["meeting.m4a", "--variant", "int6"]))
    assert args.command == "transcribe"
    assert args.audio == "meeting.m4a"
    assert args.variant == "int6"
    assert "gigaam-stt AUDIO" in parser.format_help()


def test_short_cli_keeps_explicit_commands() -> None:
    assert _short_cli_args(["models", "--json"]) == ["models", "--json"]
    assert _short_cli_args(["transcribe", "meeting.wav"]) == ["transcribe", "meeting.wav"]


def test_models_json_marks_int8_default() -> None:
    rows = json.loads(_models_output(True))
    assert [row["variant"] for row in rows if row["default"]] == ["int8"]


def test_output_dir_is_predictable(tmp_path: Path) -> None:
    args = _parser().parse_args(
        ["transcribe", "meeting.m4a", "--format", "vtt", "--output-dir", str(tmp_path)]
    )
    assert _output_path(args, _selected_format(args)) == tmp_path / "meeting.vtt"


def test_output_format_is_inferred_from_suffix(tmp_path: Path) -> None:
    args = _parser().parse_args(["transcribe", "meeting.wav", "--output", str(tmp_path / "x.json")])
    assert _selected_format(args) == "json"


def test_conflicting_output_suffix_is_rejected(tmp_path: Path) -> None:
    args = _parser().parse_args(
        ["transcribe", "meeting.wav", "--format", "json", "--output", str(tmp_path / "x.srt")]
    )
    with pytest.raises(ValueError, match="conflicts"):
        _output_path(args, _selected_format(args))


def test_unsupported_platform_message_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.platform", "linux")
    monkeypatch.setattr("platform.machine", lambda: "x86_64")
    with pytest.raises(RuntimeError, match="Apple Silicon"):
        _ensure_supported_platform()
