# Changelog

All notable changes follow [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and releases use [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.3] - 2026-07-16

### Changed

- Reworked the English and Russian landing documentation around a compact,
  mobile-friendly benchmark graphic near the top.
- Removed duplicate full benchmark tables from the README while retaining one
  accessible text summary and links to the reproducible report.

## [0.1.2] - 2026-07-16

### Added

- Reproducible multilingual FLEURS benchmark for Russian, Kazakh, Kyrgyz,
  Uzbek, and English with MLX Whisper and MLX Parakeet references.
- English and Russian documentation with a Pareto model-selection matrix,
  confidence intervals, speed, peak memory, and weight-size comparisons.

### Changed

- PyPI metadata now surfaces offline ASR, Apple Silicon, supported languages,
  transcription, and quantization keywords.
- INT8 is presented as the recommended quality, speed, memory, and size balance.
- Developer report commands now create missing output directories.

## [0.1.1] - 2026-07-16

### Added

- Short `gigaam-stt` CLI alias with direct `gigaam-stt AUDIO [OPTIONS]` transcription.

### Changed

- User-facing examples now prefer the short command while preserving
  `gigaam-multilingual-mlx` and explicit `transcribe` compatibility.

## [0.1.0] - 2026-07-15

### Added

- Native MLX inference for the GigaAM-Multilingual CTC architecture.
- FP16, INT8 g64, INT6 g64, and INT4 g64 artifact support.
- Zero-configuration INT8 Hub download with immutable release revisions and cache reuse.
- TXT, JSON, SRT, and WebVTT output for audio and video inputs decoded by `ffmpeg`.
- Local model, explicit Hub repository/revision, and offline workflows.
- Developer-only conversion, parity, quality, and benchmark interfaces.

[0.1.3]: https://github.com/ai-babai/gigaam-multilingual-mlx/releases/tag/v0.1.3
[0.1.2]: https://github.com/ai-babai/gigaam-multilingual-mlx/releases/tag/v0.1.2
[0.1.1]: https://github.com/ai-babai/gigaam-multilingual-mlx/releases/tag/v0.1.1
[0.1.0]: https://github.com/ai-babai/gigaam-multilingual-mlx/releases/tag/v0.1.0
