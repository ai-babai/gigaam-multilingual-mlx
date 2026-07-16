from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import mlx.core as mx

from ._version import __version__
from .artifacts import (
    DEFAULT_VARIANT,
    VARIANTS,
    resolve_model_source,
    variant_rows,
)
from .model import load_model
from .outputs import OUTPUT_FORMATS, render_transcription
from .service import transcribe_file


def _ensure_supported_platform() -> None:
    machine = platform.machine().lower()
    if sys.platform != "darwin" or machine not in {"arm64", "aarch64"}:
        raise RuntimeError("gigaam-multilingual-mlx requires macOS on Apple Silicon (arm64)")
    version = platform.mac_ver()[0]
    if version and int(version.split(".", 1)[0]) < 14:
        raise RuntimeError("gigaam-multilingual-mlx requires macOS 14 or newer")


def _parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Transcribe audio locally with GigaAM-Multilingual on Apple Silicon.",
        epilog=(
            "Short form: gigaam-stt AUDIO [OPTIONS] is equivalent to "
            "gigaam-stt transcribe AUDIO [OPTIONS]."
            if prog == "gigaam-stt"
            else None
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    transcribe = sub.add_parser("transcribe", help="transcribe an audio or video file")
    transcribe.add_argument("audio", help="WAV, FLAC, MP3, M4A, or video input")
    transcribe.add_argument(
        "--variant", choices=tuple(VARIANTS), default=DEFAULT_VARIANT, help="model variant"
    )
    transcribe.add_argument("--model", help="local model directory or Hugging Face repo ID")
    transcribe.add_argument("--revision", help="Hugging Face revision or immutable tag")
    transcribe.add_argument("--cache-dir", help="custom Hugging Face cache directory")
    transcribe.add_argument("--offline", action="store_true", help="use cached/local files only")
    transcribe.add_argument("--dtype", choices=("float32", "float16", "bfloat16"))
    transcribe.add_argument("--chunk-seconds", type=float, default=20.0)
    transcribe.add_argument("--overlap-seconds", type=float, default=2.0)
    transcribe.add_argument("--compile", action="store_true")
    transcribe.add_argument(
        "--format",
        choices=OUTPUT_FORMATS,
        help="output format; inferred from --output suffix, otherwise txt",
    )
    output = transcribe.add_mutually_exclusive_group()
    output.add_argument("--output", help="exact output file path")
    output.add_argument("--output-dir", help="directory for INPUT_STEM.{txt,json,srt,vtt}")

    models = sub.add_parser("models", help="list supported model variants")
    models.add_argument("--json", action="store_true", help="print machine-readable JSON")

    serve = sub.add_parser("serve", help="run the local OpenAI-compatible transcription API")
    serve.add_argument(
        "--variant", choices=tuple(VARIANTS), default=DEFAULT_VARIANT, help="model variant"
    )
    serve.add_argument("--model", help="local model directory or Hugging Face repo ID")
    serve.add_argument("--revision", help="Hugging Face revision or immutable tag")
    serve.add_argument("--cache-dir", help="custom Hugging Face cache directory")
    serve.add_argument("--offline", action="store_true", help="use cached/local files only")
    serve.add_argument("--dtype", choices=("float32", "float16", "bfloat16"))
    serve.add_argument("--chunk-seconds", type=float, default=20.0)
    serve.add_argument("--overlap-seconds", type=float, default=2.0)
    serve.add_argument("--compile", action="store_true")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--max-upload-mb", type=int, default=1024)
    serve.add_argument("--max-queue", type=int, default=8)
    serve.add_argument("--temp-dir", help="temporary upload directory")
    serve.add_argument(
        "--api-key-file",
        help="file containing the bearer token; alternatively set GIGAAM_STT_API_KEY",
    )
    serve.add_argument(
        "--allow-unauthenticated",
        action="store_true",
        help="allow an unauthenticated non-loopback bind (unsafe)",
    )
    return parser


def _selected_format(args: argparse.Namespace) -> str:
    if args.format:
        return str(args.format)
    if args.output:
        suffix = Path(args.output).suffix.lstrip(".").lower()
        if suffix in OUTPUT_FORMATS:
            return suffix
    return "txt"


def _output_path(args: argparse.Namespace, output_format: str) -> Path | None:
    if args.output:
        path = Path(args.output).expanduser()
        suffix = path.suffix.lstrip(".").lower()
        if suffix in OUTPUT_FORMATS and suffix != output_format:
            raise ValueError(
                f"Output suffix .{suffix} conflicts with --format {output_format}; "
                "choose matching values"
            )
        return path
    if args.output_dir:
        directory = Path(args.output_dir).expanduser()
        return directory / f"{Path(args.audio).stem}.{output_format}"
    return None


def _models_output(as_json: bool) -> str:
    rows = variant_rows()
    if as_json:
        return json.dumps(rows, ensure_ascii=False, indent=2) + "\n"
    lines = ["VARIANT  ROLE                         SIZE       REPOSITORY"]
    for row in rows:
        size = f"{int(row['expected_bytes']) / 1_000_000:.0f} MB"
        lines.append(
            f"{str(row['variant']):<8} {str(row['label']):<28} {size:<10} {row['repo_id']}"
        )
    return "\n".join(lines) + "\n"


def _short_cli_args(argv: list[str]) -> list[str]:
    if argv and argv[0] not in {"transcribe", "models", "serve"} and not argv[0].startswith("-"):
        return ["transcribe", *argv]
    return argv


def _run_server(args: argparse.Namespace) -> None:
    try:
        from .server import ServerConfig, read_api_key, run_server
    except ModuleNotFoundError as error:
        if error.name in {"fastapi", "multipart", "starlette", "uvicorn"}:
            raise RuntimeError(
                "Server dependencies are not installed; run "
                "`uv tool install 'gigaam-multilingual-mlx[server]'` or "
                "`python -m pip install 'gigaam-multilingual-mlx[server]'`"
            ) from None
        raise
    api_key = read_api_key(args.api_key_file)
    config = ServerConfig(
        variant=args.variant,
        source=args.model,
        revision=args.revision,
        cache_dir=args.cache_dir,
        offline=args.offline,
        dtype=args.dtype,
        chunk_seconds=args.chunk_seconds,
        overlap_seconds=args.overlap_seconds,
        compile=args.compile,
        host=args.host,
        port=args.port,
        max_upload_bytes=args.max_upload_mb * 1024 * 1024,
        max_queue=args.max_queue,
        temp_dir=args.temp_dir,
        api_key=api_key,
        allow_unauthenticated=args.allow_unauthenticated,
    )
    run_server(config)


def _run(
    argv: list[str] | None = None,
    *,
    prog: str | None = None,
    implicit_transcribe: bool = False,
) -> None:
    parser = _parser(prog)
    values = list(sys.argv[1:] if argv is None else argv)
    if implicit_transcribe:
        values = _short_cli_args(values)
    args = parser.parse_args(values)
    try:
        if args.command == "models":
            print(_models_output(args.json), end="")
            return
        _ensure_supported_platform()
        if args.command == "serve":
            _run_server(args)
            return
        reference = resolve_model_source(
            args.model,
            variant=args.variant,
            revision=args.revision,
            cache_dir=args.cache_dir,
            local_files_only=args.offline,
        )
        location = (
            f"{reference.repo_id}@{reference.revision}"
            if reference.repo_id
            else str(reference.path)
        )
        print(f"Using {reference.variant} model from {location}", file=sys.stderr)
        print(f"Model files: {reference.path}", file=sys.stderr)
        mx.reset_peak_memory()
        model = load_model(reference.path, args.dtype)
        runner = mx.compile(model.__call__, inputs=model.state) if args.compile else None
        result = transcribe_file(
            model,
            args.audio,
            args.chunk_seconds,
            args.overlap_seconds,
            runner=runner,
            artifact=reference,
        )
        output_format = _selected_format(args)
        rendered = render_transcription(result, output_format)
        output_path = _output_path(args, output_format)
        if output_path is None:
            print(rendered, end="")
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(rendered)
            print(f"Wrote {output_format.upper()} to {output_path}", file=sys.stderr)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        parser.exit(2, f"{parser.prog}: error: {error}\n")


def main() -> None:
    _run()


def short_main() -> None:
    _run(prog="gigaam-stt", implicit_transcribe=True)


if __name__ == "__main__":
    main()
