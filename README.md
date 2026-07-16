# GigaAM-Multilingual MLX

**English** · [Русский](README.ru.md)

Offline speech-to-text for Russian, Kazakh, Kyrgyz, and Uzbek on Apple Silicon.
This is an independent native [MLX](https://github.com/ml-explore/mlx) port of
[GigaAM](https://github.com/salute-developers/GigaAM) for Apple Silicon, based on the
official [GigaAM-Multilingual model](https://huggingface.co/ai-sage/GigaAM-Multilingual).
This project is not an official release of the GigaAM authors.

[PyPI package](https://pypi.org/project/gigaam-multilingual-mlx/) ·
[Model Collection](https://huggingface.co/collections/ai-babai/gigaam-multilingual-mlx-6a57ef36e44d1a0c4eb49276) ·
[latest release](https://github.com/ai-babai/gigaam-multilingual-mlx/releases/latest)

`gigaam-multilingual-mlx` runs local multilingual speech recognition without PyTorch, ONNX
Runtime, Core ML, or a cloud ASR service. Model weights are downloaded separately
from Hugging Face and reused from the standard Hub cache.

## Install and transcribe in 60 seconds

Requirements: an Apple Silicon Mac, macOS 14 or newer, Python 3.12 or 3.13,
[`uv`](https://docs.astral.sh/uv/getting-started/installation/), and
[`ffmpeg`](https://ffmpeg.org/).

```bash
brew install uv ffmpeg
uv tool install gigaam-multilingual-mlx
gigaam-stt meeting.m4a --output transcript.txt
```

For the Python API inside a uv-managed project, use
`uv add gigaam-multilingual-mlx`. A regular
`python -m pip install gigaam-multilingual-mlx` remains supported.
The canonical `gigaam-multilingual-mlx transcribe ...` command and the explicit
`gigaam-stt transcribe ...` form remain available for compatibility.

The first run downloads the default INT8 model at the immutable `v0.1.0` release
tag. Later runs reuse the cached files.

## Why GigaAM STT?

- Strong measured transcription quality for the four core GigaAM languages,
  especially Kazakh, Kyrgyz, and Uzbek.
- Native MLX inference on Apple Silicon without PyTorch, ONNX Runtime, Core ML,
  or a cloud ASR service.
- A 699 MB default model with substantially lower process memory than the tested
  MLX Whisper checkpoints.
- Fully local operation after the first model download, with a simple
  `gigaam-stt AUDIO` CLI and TXT, JSON, SRT, or VTT output.

### Model selection: GigaAM MLX, MLX Whisper, and MLX Parakeet

Corpus WER uses pinned FLEURS test selections. Resource columns use one Russian
five-minute WAV on a 14-inch MacBook Pro with Apple M4 Pro and 48 GB unified
memory. `5-min WAV` is the median of five runs after model load; Peak RAM is
whole-process peak RSS; model size is the weight file on disk.

`✓` column leader · `◇` Pareto frontier · `★` recommended default. Lower is
better. English is an appendix. Parakeet does not officially support Kazakh,
Kyrgyz, or Uzbek, so those cells are `N/A`.

#### ◇ Pareto frontier — start here

The frontier is computed among MLX candidates using equal-weight macro WER over
Russian, Kazakh, Kyrgyz, and Uzbek plus 5-min WAV time, Peak RAM, and model size.
Original GigaAM is a PyTorch/MPS reference baseline; English is excluded as an
appendix; models missing a core language are ineligible.

| Pareto MLX variant | Core macro WER | 5-min WAV | Peak RAM | Model size | Best fit |
|---|---:|---:|---:|---:|---|
| ◇ GigaAM MLX FP16 | 5.066% | **✓ 1.952s** | 1.350 GB | 1.171 GB | fastest; reference-port fidelity |
| **◇ ★ GigaAM MLX INT8** | **5.070%** | **2.036s** | **0.877 GB** | **0.699 GB** | **recommended quality / speed / footprint balance** |
| ◇ GigaAM MLX INT6 | 5.069% | 2.195s | 0.755 GB | 0.573 GB | smaller footprint with near-INT8 quality |
| ◇ GigaAM MLX INT4 | 5.219% | 2.563s | **✓ 0.626 GB** | **✓ 0.447 GB** | minimum model size and Peak RAM |

#### All compared models

| Model / variant | RU WER | KZ WER | KY WER | UZ WER | EN WER | 5-min WAV | Peak RAM | Model size |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Original GigaAM PyTorch/MPS | **✓ 2.995%** | **✓ 4.325%** | **✓ 5.553%** | **✓ 7.310%** | 9.717% | 2.789s | 5.059 GB | 2.342 GB |
| ◇ GigaAM MLX FP16 | **✓ 2.995%** | 4.342% | 5.560% | 7.367% | 9.726% | **✓ 1.952s** | 1.350 GB | 1.171 GB |
| **◇ ★ GigaAM MLX INT8** | 3.013% | 4.351% | 5.582% | 7.334% | 9.734% | 2.036s | 0.877 GB | 0.699 GB |
| ◇ GigaAM MLX INT6 | 3.013% | 4.385% | 5.568% | **✓ 7.310%** | 9.848% | 2.195s | 0.755 GB | 0.573 GB |
| ◇ GigaAM MLX INT4 | 3.234% | 4.377% | 5.768% | 7.497% | 9.883% | 2.563s | **✓ 0.626 GB** | **✓ 0.447 GB** |
| MLX Whisper large-v2 | 3.855% | 39.468% | 92.582% | 95.551% | 4.194% | 14.287s | 3.733 GB | 3.083 GB |
| MLX Whisper large-v3 | 3.123% | 32.778% | 86.680% | 87.981% | **✓ 4.098%** | 18.214s | 3.765 GB | 3.084 GB |
| MLX Whisper large-v3-turbo | 3.549% | 20.468% | 84.078% | 108.931% | 4.579% | 6.722s | 1.898 GB | 1.614 GB |
| MLX Parakeet TDT 0.6B v3 | 4.961% | N/A | N/A | N/A | 4.928% | 3.843s | 1.085 GB | 2.508 GB |

For Kazakh, Kyrgyz, and Uzbek, paired bootstrap 95% confidence intervals support
lower WER for INT8 than all three Whisper references. Russian is close: the
measured INT8 advantage is supported against large-v2 and v3 Turbo, but not against
large-v3. INT8 also has 39.3% lower Russian WER than Parakeet in this suite
(paired-bootstrap 95% CI: 32.7–45.6%). Whisper and Parakeet are better on the
English appendix; Whisper supports many more languages and speech-to-English
translation. Parakeet does not officially support Kazakh, Kyrgyz, or Uzbek, so
those cells are `N/A` rather than out-of-scope measurements.

INT8 was 3.30× faster than Whisper v3 Turbo, 7.02× faster than large-v2, and
8.94× faster than large-v3 on this input, with 54–77% lower peak RSS. It was also
1.89× faster than Parakeet, with 19.2% lower peak RSS and 72.1% smaller weights. See the
[multilingual benchmark](https://github.com/ai-babai/gigaam-multilingual-mlx/blob/main/docs/benchmark-multilingual-v1.md) for all languages,
FP16/INT6/INT4, CER, p95, confidence intervals, hashes, and limitations.

## Model variants

> **Recommended default: INT8 g64.** It preserves near-FP16 quality while giving
> the strongest measured balance of download size, process memory, and speed.

| Variant | Role | Model size | Hugging Face repository |
|---|---|---:|---|
| **INT8 g64** | **recommended default, lowest-risk quantized** | **699 MB** | **[`ai-babai/gigaam-multilingual-mlx-int8-g64`](https://huggingface.co/ai-babai/gigaam-multilingual-mlx-int8-g64)** |
| FP16 | reference MLX artifact | 1.17 GB | [`ai-babai/gigaam-multilingual-mlx`](https://huggingface.co/ai-babai/gigaam-multilingual-mlx) |
| INT6 g64 | compact balanced | 573 MB | [`ai-babai/gigaam-multilingual-mlx-int6-g64`](https://huggingface.co/ai-babai/gigaam-multilingual-mlx-int6-g64) |
| INT4 g64 | smallest opt-in | 447 MB | [`ai-babai/gigaam-multilingual-mlx-int4-g64`](https://huggingface.co/ai-babai/gigaam-multilingual-mlx-int4-g64) |

All variants are grouped in the
[GigaAM Multilingual MLX Collection](https://huggingface.co/collections/ai-babai/gigaam-multilingual-mlx-6a57ef36e44d1a0c4eb49276).
The technical upstream branch name `large_ctc` remains artifact metadata; it is
not part of the public project name.

Select another variant without knowing its repository name:

```bash
gigaam-stt speech.wav --variant fp16
gigaam-stt speech.wav --variant int6 --format json
gigaam-stt models
```

## Inputs and outputs

`ffmpeg` provides decoding for WAV, FLAC, MP3, M4A, and audio tracks in common
video containers. Inputs are converted to mono 16 kHz PCM before inference.

```bash
# Write next to an explicit path
gigaam-stt interview.mp3 --format srt --output interview.srt

# Generate OUTPUT_DIR/INPUT_STEM.vtt
gigaam-stt recording.mov --format vtt --output-dir transcripts/

# Machine-readable output with text, word timestamps, model revision, and metrics
gigaam-stt sample.flac --format json --output sample.json
```

Supported output formats are `txt`, `json`, `srt`, and `vtt`. CTC word timestamps
are estimates derived from emission frames. They are monotonic and useful for
navigation/subtitles, but they are not forced-alignment timestamps.

Long inputs are processed in deterministic overlapping chunks. The inference
graph and tensors are reused between chunks instead of accumulating model state.

## Cache, local models, and offline use

By default, Hugging Face stores model snapshots under `~/.cache/huggingface/hub`.
Use the normal Hub variables such as `HF_HOME` to move that cache, or pass an
explicit directory:

```bash
gigaam-stt audio.wav --cache-dir /Volumes/ML/huggingface
gigaam-stt audio.wav --offline
gigaam-stt audio.wav --model /path/to/portable/model
gigaam-stt audio.wav \
  --model ai-babai/gigaam-multilingual-mlx-int8-g64 \
  --revision v0.1.0
```

`--offline` never contacts the Hub and fails clearly if the requested snapshot is
not already cached. To remove only downloaded model data, use the Hugging Face CLI
cache tools or delete the corresponding `models--ai-babai--gigaam-*` cache folders;
do not remove unrelated Hub cache entries.

## Python API

Existing local-directory loading remains supported, and Hub variants use the same
function:

```python
import mlx.core as mx
from gigaam_multilingual_mlx import load_model

model = load_model(variant="int8")  # downloads/reuses ai-babai/...-int8-g64@v0.1.0
audio = mx.zeros((1, 16_000), dtype=mx.float32)
log_probs, lengths = model(audio)
text = model.greedy_decode(log_probs, lengths)[0]["text"]

local_model = load_model("/path/to/portable/model")
```

## Support matrix

| Scope | Status |
|---|---|
| Apple Silicon supported by MLX, macOS 14+, native ARM Python | supported |
| M1–M5 generations | expected compatible; not benchmarked individually |
| Apple M4 Pro, 48 GB, macOS 15.7.7 | release benchmark machine |
| Intel Mac, Linux, Windows, iOS | not supported in `v0.1.0` |

The package reports an actionable error on unsupported platforms or when `ffmpeg`
is missing.

## Validation and benchmark

The accepted model artifacts use source revision
`3905cd51c3ed4e88c8edf33f3302969ba480a327`. Strict FP32 conversion reached 100%
greedy-token agreement on the frozen parity set. Every production artifact reloads
strictly with its recorded SHA-256.

The `v0.1.0` release quality suite uses pinned subsets of FLEURS `ru_ru`, Russian
LibriSpeech, and SOVA RuDevices with one normalization rule and identical greedy
decoding. The release report publishes Original PyTorch/MPS, FP16, INT8, INT6,
and INT4 side by side, including WER/CER, load/cold/warm timing, RTF, peak resident
memory, MLX memory, and swap. It contains no private/user audio.

### Results at a glance

Apple M4 Pro (48 GB), macOS 15.7.7. WER columns are FLEURS / Russian
LibriSpeech / SOVA corpus WER. Runtime is the warm median for the same public
5-minute input; download time is excluded. Lower is better except speedup.

| Implementation | Weights | WER: FLEURS / RuLibri / SOVA | Load | 5-min warm | Speed vs Original | Peak RSS |
|---|---:|---:|---:|---:|---:|---:|
| Original PyTorch/MPS | 2.342 GB | 6.271% / 5.911% / 12.448% | 6.066s | 3.023s | 1.00× | 5.019 GB |
| MLX FP16 | 1.171 GB | 6.271% / 5.929% / 12.431% | 0.947s | 2.089s | 1.45× | 1.348 GB |
| **MLX INT8 g64 (default)** | **0.699 GB** | **6.292% / 5.911% / 12.481%** | **0.578s** | **2.332s** | **1.30×** | **0.879 GB** |
| MLX INT6 g64 | 0.573 GB | 6.249% / 6.022% / 12.465% | 0.478s | 3.142s | 0.96× | 0.755 GB |
| MLX INT4 g64 | 0.447 GB | 6.358% / 6.003% / 12.581% | 0.378s | 2.750s | 1.10× | 0.628 GB |

INT8 is the default for its measured quality, size, and memory balance. FP16 is
the fastest measured variant on the representative 5-minute input; quantization
does not guarantee a speedup for every duration. Peak RSS is whole-process
resident memory, not model-only or additional GPU memory.

See [`benchmarks/README.md`](benchmarks/README.md) for the reproducible public-only
methodology and [`docs/benchmark-v0.1.0.md`](docs/benchmark-v0.1.0.md) for release
results.

## Limitations

- The `v0.1.0` release exposes greedy CTC decoding; it does not add beam search.
- Timestamps are approximate CTC emission times.
- Training, fine-tuning, diarization, microphone streaming, and a local HTTP server
  are outside this release.
- Quality is strongest for Russian, Kazakh, Kyrgyz, and Uzbek; upstream reports
  moderate English quality. See the upstream model card for the authoritative scope.
- Model and dataset biases from the upstream training/evaluation material remain.

## Developer tools

Conversion, quantization, parity, quality evaluation, and benchmark commands are
kept out of the production CLI and use optional dependencies:

```bash
python -m pip install 'gigaam-multilingual-mlx[convert,quality]'
python -m gigaam_multilingual_mlx.dev_cli --help
```

Weights, datasets, generated audio, and raw benchmark outputs must stay outside the
Git repository.

## License, attribution, and citation

The code in this repository is MIT-licensed. GigaAM code/model provenance and
dataset notices are recorded in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
When using the model, cite the original GigaAM-Multilingual work from the upstream
model card; project citation metadata is available in [`CITATION.cff`](CITATION.cff).

Security reports and contributions are described in [`SECURITY.md`](SECURITY.md)
and [`CONTRIBUTING.md`](CONTRIBUTING.md).
