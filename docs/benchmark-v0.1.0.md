# GigaAM-Multilingual MLX v0.1.0 benchmark

Apple M4 Pro (48 GB), macOS 15.7.7, public-only suites, greedy CTC, 20-second chunks with 2-second overlap. Cold and warm timings are separated; Original PyTorch/MPS is the baseline.

## Artifacts

| Implementation | Repository / revision | Weights | Model artifact | SHA-256 |
|---|---|---:|---:|---|
| Original PyTorch/MPS | `ai-sage/GigaAM-Multilingual@3905cd51c3ed4e88c8edf33f3302969ba480a327` | 2.342 GB | 2.342 GB | `c3fabefb50b41f08f4d7ad44e02c26c37d242882704cdcca2ebd98e45eff73d1` |
| MLX FP16 | `ai-babai/gigaam-multilingual-mlx@v0.1.0` | 1.171 GB | 1.171 GB | `d6d7bcef5e1b77700c84d669635a88021724fd3b1ae0c408890bccdf3c8d2269` |
| MLX INT8 g64 (default) | `ai-babai/gigaam-multilingual-mlx-int8-g64@v0.1.0` | 0.699 GB | 0.699 GB | `efd4103be5875ba556e6645ce34dd7def2339c7377209c86b7dd5787fb443729` |
| MLX INT6 g64 | `ai-babai/gigaam-multilingual-mlx-int6-g64@v0.1.0` | 0.573 GB | 0.573 GB | `83dad98e9bbb655c2bb49bd1ee65c575d0ccd77d5cad9044f5af8f8cca2987ae` |
| MLX INT4 g64 | `ai-babai/gigaam-multilingual-mlx-int4-g64@v0.1.0` | 0.447 GB | 0.447 GB | `25a7912af016dc24526e0a06efbc488a979a4f0c874225d5ab4966421b1211da` |

The four published MLX model artifacts total 2.890 GB. INT8 g64 is the default because it preserves measured quality while substantially reducing disk and resident-memory requirements; it is not claimed to be the fastest variant for every duration.

## Quality: `public-asr-quality-v1`

WER/CER are corpus metrics after `gigaam-mlx-text-v1` normalization. All rows use the same 1,533 public items and batch size 8.

| Implementation | FLEURS WER/CER | Russian LibriSpeech WER/CER | SOVA WER/CER | Max abs. Δ WER vs Original | Max abs. Δ WER vs FP16 |
|---|---:|---:|---:|---:|---:|
| Original PyTorch/MPS | 6.271% / 3.816% | 5.911% / 1.156% | 12.448% / 4.845% | 0.000 pp | 0.019 pp |
| MLX FP16 | 6.271% / 3.816% | 5.929% / 1.156% | 12.431% / 4.839% | 0.019 pp | 0.000 pp |
| MLX INT8 g64 (default) | 6.292% / 3.819% | 5.911% / 1.149% | 12.481% / 4.836% | 0.033 pp | 0.050 pp |
| MLX INT6 g64 | 6.249% / 3.809% | 6.022% / 1.165% | 12.465% / 4.839% | 0.111 pp | 0.093 pp |
| MLX INT4 g64 | 6.358% / 3.816% | 6.003% / 1.168% | 12.581% / 4.851% | 0.133 pp | 0.150 pp |

Full operation counts, exact-match counts, dataset revisions, licenses, manifests, and deltas versus both Original and FP16 are in [`benchmark-v0.1.0.json`](benchmark-v0.1.0.json).

## Performance: `public-asr-perf-v1`

MLX runs use `mx.compile`; Original uses upstream PyTorch/MPS. Short and medium have one cold plus five measured warm runs; long has one cold plus three measured warm runs. Download time is excluded from model load and inference.

### 30 seconds

| Implementation | Load | Cold | Warm median / p95 | Warm RTF | Speed vs Original | Peak RSS | Device peak | Swap increase | Words |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Original PyTorch/MPS | 6.671s | 0.606s | 0.298s / 0.301s | 0.0099 | 1.00× | 5.019 GB | 4.064 GB | 0 MB | 55 |
| MLX FP16 | 0.978s | 0.284s | 0.225s / 0.226s | 0.0075 | 1.32× | 1.294 GB | 1.918 GB | 0 MB | 55 |
| MLX INT8 g64 (default) | 0.604s | 0.279s | 0.223s / 0.227s | 0.0074 | 1.34× | 0.828 GB | 1.215 GB | 0 MB | 55 |
| MLX INT6 g64 | 0.496s | 0.275s | 0.246s / 0.277s | 0.0082 | 1.21× | 0.692 GB | 1.090 GB | 0 MB | 56 |
| MLX INT4 g64 | 0.394s | 0.273s | 0.225s / 0.236s | 0.0075 | 1.32× | 0.580 GB | 0.964 GB | 0 MB | 54 |

### 5 minutes

| Implementation | Load | Cold | Warm median / p95 | Warm RTF | Speed vs Original | Peak RSS | Device peak | Swap increase | Words |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Original PyTorch/MPS | 6.066s | 3.374s | 3.023s / 3.037s | 0.0101 | 1.00× | 5.019 GB | 4.064 GB | 0 MB | 554 |
| MLX FP16 | 0.947s | 2.154s | 2.089s / 2.098s | 0.0070 | 1.45× | 1.348 GB | 1.918 GB | 0 MB | 554 |
| MLX INT8 g64 (default) | 0.578s | 2.151s | 2.332s / 2.520s | 0.0078 | 1.30× | 0.879 GB | 1.215 GB | 0 MB | 554 |
| MLX INT6 g64 | 0.478s | 3.684s | 3.142s / 3.238s | 0.0105 | 0.96× | 0.755 GB | 1.090 GB | 0 MB | 554 |
| MLX INT4 g64 | 0.378s | 2.983s | 2.750s / 2.847s | 0.0092 | 1.10× | 0.628 GB | 0.964 GB | 0 MB | 551 |

### 20 minutes

| Implementation | Load | Cold | Warm median / p95 | Warm RTF | Speed vs Original | Peak RSS | Device peak | Swap increase | Words |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Original PyTorch/MPS | 6.115s | 12.794s | 12.130s / 12.162s | 0.0101 | 1.00× | 5.020 GB | 4.064 GB | 0 MB | 2194 |
| MLX FP16 | 0.944s | 8.358s | 8.619s / 9.104s | 0.0072 | 1.41× | 1.569 GB | 1.918 GB | 0 MB | 2194 |
| MLX INT8 g64 (default) | 0.593s | 10.396s | 11.405s / 11.951s | 0.0095 | 1.06× | 1.093 GB | 1.215 GB | 0 MB | 2194 |
| MLX INT6 g64 | 0.477s | 12.197s | 11.543s / 11.806s | 0.0096 | 1.05× | 0.968 GB | 1.090 GB | 0 MB | 2194 |
| MLX INT4 g64 | 0.379s | 10.501s | 10.891s / 11.761s | 0.0091 | 1.11× | 0.851 GB | 0.964 GB | 0 MB | 2191 |

Peak RSS means peak resident memory of the process, not model-only memory. Metal/MPS device memory is a separate unified-memory observation and is not added to RSS. All runs had 0 MB positive swap growth.

## Reproduction contract

- Quality manifests: [`../benchmarks/manifests/`](../benchmarks/manifests/)
- Performance recipe: [`../benchmarks/public-asr-perf-v1.json`](../benchmarks/public-asr-perf-v1.json)
- Methodology: [`../benchmarks/README.md`](../benchmarks/README.md)
- Code revision: `pending-v0.1.0-release-commit`
- Upstream model: [`ai-sage/GigaAM-Multilingual`](https://huggingface.co/ai-sage/GigaAM-Multilingual)
- Upstream repository: [`salute-developers/GigaAM`](https://github.com/salute-developers/GigaAM)

Exact path-independent command templates are recorded in the JSON report. Raw hypotheses and generated WAV files remain outside Git under the machine-local ML data root.

## Limitations

- Measurements are from one MacBook Pro with Apple M4 Pro and 48 GB unified memory; M1-M5 support is not a claim of identical performance.
- Short/medium use five measured warm runs after one cold run; long uses three measured warm runs after one cold run.
- Peak RSS is peak resident process memory. Device/Metal memory is reported separately and must not be added to RSS because Apple Silicon uses unified memory.
- Swap counters are system-wide; every measured run had zero positive swap delta, while negative deltas mean the OS reclaimed existing swap.
- Weight-only quantization reduces artifact and memory size but does not guarantee monotonic speedups for every sequence length.
- Word timestamps are approximate greedy-CTC frame alignments, not forced alignment.
- PyTorch 2.10 emitted a non-fatal STFT output-resize deprecation warning during batched quality evaluation.
- The quality suite uses deterministic public subsets and is not presented as a full-dataset leaderboard result.
