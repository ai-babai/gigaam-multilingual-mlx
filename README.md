# GigaAM-Multilingual MLX

**English** · [Русский](README.ru.md)

Fast, offline speech-to-text for Russian, Kazakh, Kyrgyz, and Uzbek on Apple
Silicon. This is an independent native [MLX](https://github.com/ml-explore/mlx)
port of [GigaAM](https://github.com/salute-developers/GigaAM), based on the
official [GigaAM-Multilingual model](https://huggingface.co/ai-sage/GigaAM-Multilingual).

[PyPI](https://pypi.org/project/gigaam-multilingual-mlx/) ·
[Hugging Face models](https://huggingface.co/collections/ai-babai/gigaam-multilingual-mlx-6a57ef36e44d1a0c4eb49276) ·
[Latest release](https://github.com/ai-babai/gigaam-multilingual-mlx/releases/latest) ·
[Full benchmark](docs/benchmark-multilingual-v1.md)

![GigaAM MLX, Whisper, and Parakeet model decision matrix](https://raw.githubusercontent.com/ai-babai/gigaam-multilingual-mlx/main/docs/benchmark-multilingual-v1.png)

The chart compares WER, five-minute transcription time, peak process memory, and
model size. `✓` marks a column leader, `◇` the Pareto frontier, and `★` the
recommended default. Lower is better.

## Quick start

Requires an Apple Silicon Mac, macOS 14+, Python 3.12 or 3.13,
[`uv`](https://docs.astral.sh/uv/getting-started/installation/), and
[`ffmpeg`](https://ffmpeg.org/).

```bash
brew install uv ffmpeg
uv tool install gigaam-multilingual-mlx
gigaam-stt meeting.m4a --output transcript.txt
```

For a uv-managed Python project:

```bash
uv add gigaam-multilingual-mlx
```

`pip install gigaam-multilingual-mlx` and the long
`gigaam-multilingual-mlx transcribe ...` command remain supported.

## Why this port?

- Native MLX inference without PyTorch, ONNX Runtime, Core ML, or a cloud API.
- Strong measured quality for the four core GigaAM languages, especially
  Kazakh, Kyrgyz, and Uzbek.
- A 699 MB default model that used 0.877 GB peak process memory in the published
  five-minute benchmark.
- Local WAV, FLAC, MP3, M4A, and video transcription to TXT, JSON, SRT, or VTT.

On the tested Russian five-minute input, INT8 was 3.30× faster than Whisper v3
Turbo, 7.02× faster than Whisper large-v2, and 8.94× faster than Whisper
large-v3. Whisper and Parakeet were better on the English appendix. These are
single-machine public-corpus results, not a universal ASR leaderboard; see the
[full report](docs/benchmark-multilingual-v1.md) for confidence intervals,
commands, hashes, and limitations.

## Model variants

| Variant | Recommended use | Weights | Hugging Face |
|---|---|---:|---|
| **INT8 g64** | **default balance** | **699 MB** | [`ai-babai/...-int8-g64`](https://huggingface.co/ai-babai/gigaam-multilingual-mlx-int8-g64) |
| FP16 | fastest measured, reference port | 1.17 GB | [`ai-babai/gigaam-multilingual-mlx`](https://huggingface.co/ai-babai/gigaam-multilingual-mlx) |
| INT6 g64 | smaller, near-INT8 quality | 573 MB | [`ai-babai/...-int6-g64`](https://huggingface.co/ai-babai/gigaam-multilingual-mlx-int6-g64) |
| INT4 g64 | minimum memory and disk | 447 MB | [`ai-babai/...-int4-g64`](https://huggingface.co/ai-babai/gigaam-multilingual-mlx-int4-g64) |

Choose a variant without remembering its repository name:

```bash
gigaam-stt speech.wav --variant fp16
gigaam-stt speech.wav --variant int6 --format json
gigaam-stt models
```

The runtime downloads immutable model revision `v0.1.0` on first use and then
reuses the standard Hugging Face cache.

## Common commands

```bash
# Subtitles
gigaam-stt interview.mp3 --format srt --output interview.srt

# Output directory; creates transcripts/recording.vtt
gigaam-stt recording.mov --format vtt --output-dir transcripts/

# Machine-readable text, timestamps, revision, and metrics
gigaam-stt sample.flac --format json --output sample.json

# Reuse a downloaded snapshot without network access
gigaam-stt audio.wav --offline

# Move the Hugging Face cache
gigaam-stt audio.wav --cache-dir /Volumes/ML/huggingface
```

Long inputs are processed in deterministic overlapping chunks. CTC word
timestamps are approximate emission times, suitable for navigation and
subtitles but not forced alignment.

## Python API

```python
import mlx.core as mx
from gigaam_multilingual_mlx import load_model

model = load_model(variant="int8")
audio = mx.zeros((1, 16_000), dtype=mx.float32)
log_probs, lengths = model(audio)
text = model.greedy_decode(log_probs, lengths)[0]["text"]
```

Local portable model directories are also supported:
`load_model("/path/to/model")`.

## Compatibility and limitations

- Supported: Apple Silicon, native ARM Python, macOS 14+.
- M1–M5 are expected to be runtime-compatible; published performance numbers
  come from one 14-inch MacBook Pro with Apple M4 Pro and 48 GB memory.
- Not supported: Intel Mac, Linux, Windows, or iOS.
- This release uses greedy CTC decoding. It does not provide diarization,
  training, microphone streaming, or a local HTTP server.
- Accuracy can degrade with noise, far-field speech, overlapping speakers,
  code-switching, or domains unlike the public evaluation data.

## Reproducibility and development

The repository contains pinned public FLEURS manifests, compact benchmark
results, and commands for GigaAM, MLX Whisper, and MLX Parakeet. No private audio,
model weights, datasets, caches, or raw large benchmark outputs are stored in
Git.

Developer-only conversion and evaluation tools use optional dependencies:

```bash
python -m pip install 'gigaam-multilingual-mlx[convert,quality]'
python -m gigaam_multilingual_mlx.dev_cli --help
```

Start with [`benchmarks/multilingual-v1/README.md`](benchmarks/multilingual-v1/README.md)
for the protocol and [`docs/benchmark-multilingual-v1.md`](docs/benchmark-multilingual-v1.md)
for the results.

### Key numbers in text

This compact table is the text alternative to the GigaAM part of the image.
Core macro WER averages Russian, Kazakh, Kyrgyz, and Uzbek; lower is better.

| GigaAM MLX variant | Core macro WER | 5-min WAV | Peak RAM | Weights |
|---|---:|---:|---:|---:|
| FP16 | 5.066% | **1.952s** | 1.350 GB | 1.171 GB |
| **INT8 g64 (default)** | **5.070%** | **2.036s** | **0.877 GB** | **0.699 GB** |
| INT6 g64 | 5.069% | 2.195s | 0.755 GB | 0.573 GB |
| INT4 g64 | 5.219% | 2.563s | **0.626 GB** | **0.447 GB** |

## License and attribution

The MLX port is MIT-licensed. Upstream and dataset provenance is recorded in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). Please cite the original
GigaAM-Multilingual work from the upstream model card and this project using
[`CITATION.cff`](CITATION.cff).

This project is not an official release of the GigaAM authors. Security reports
and contributions are covered by [`SECURITY.md`](SECURITY.md) and
[`CONTRIBUTING.md`](CONTRIBUTING.md).
