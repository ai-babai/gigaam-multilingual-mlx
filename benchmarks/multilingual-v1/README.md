# Public multilingual benchmark v1

`public-asr-multilingual-v1` compares GigaAM-Multilingual with three MLX Whisper
references and MLX Parakeet TDT 0.6B v3 on identical public audio. Raw audio,
model weights, hypotheses, and
process reports stay outside Git. This directory contains portable manifests,
5-minute corpus recipes, and the recorded run context.

## Scope

Headline languages:

- Russian: FLEURS `ru_ru`;
- Kazakh: FLEURS `kk_kz`;
- Kyrgyz: FLEURS `ky_kg`;
- Uzbek: FLEURS `uz_uz`.

English `en_us` is reported as an appendix, not as a core GigaAM claim.

All data comes from `google/fleurs` at revision
`70bb2e84b976b7e960aa89f1c648e09c59f894dd`, license CC BY 4.0. The frozen rule
uses the full test split after excluding audio longer than 30 seconds and
references containing digits. `manifests/` records source rows, IDs, durations,
references, audio SHA-256 hashes, license, revision, and selection rule without
machine-local paths.

## Implementations

- Original GigaAM-Multilingual `large_ctc`, PyTorch/MPS, revision
  `3905cd51c3ed4e88c8edf33f3302969ba480a327`;
- GigaAM MLX FP16, INT8 g64, INT6 g64, and INT4 g64 from the same source revision;
- `mlx-community/whisper-large-v2-mlx` at
  `cce86229e2765266197fef869ce9f7e2550067ab`;
- `mlx-community/whisper-large-v3-mlx` at
  `49e6aa286ad60c14352c404340ded53710378a11`;
- `mlx-community/whisper-large-v3-turbo` at
  `a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb`;
- [`mlx-community/parakeet-tdt-0.6b-v3`](https://huggingface.co/mlx-community/parakeet-tdt-0.6b-v3) at
  `ed2b7e8c15f9aaa0b5772e2efb986255eaef7e15`.

Whisper runs with `mlx-whisper==0.4.3` and `mlx==0.32.0`. Mutable model revisions
are not used. Parakeet runs with `parakeet-mlx==0.5.2` and `mlx==0.32.0`.
Its official language set includes Russian and English, but not Kazakh, Kyrgyz,
or Uzbek; unsupported cells are reported as `N/A` and were not run. Model scope
comes from the official
[`nvidia/parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
card; the MLX runtime is
[`senstella/parakeet-mlx`](https://github.com/senstella/parakeet-mlx).

## Quality protocol

Every implementation receives the same audio bytes and normalized reference for
an item. GigaAM uses greedy CTC. Whisper uses:

- `task=transcribe`;
- temperature 0 with no beam, best-of, or fallback;
- timestamps and word timestamps disabled;
- `condition_on_previous_text=false`;
- explicit language `ru`, `kk`, `uz`, or `en`.

The official Whisper tokenizer has no Kyrgyz `ky` token. Kyrgyz therefore uses
standard automatic language detection, and every report records the detected
language distribution. No transliteration or manual output correction is applied.

Parakeet quality uses native BF16 greedy TDT, built-in language detection, and
full attention (`chunk_duration=None`) on the same Russian and English utterances.
Its timestamps are model-native but do not affect WER normalization.

Normalization is Unicode NFC, lowercase, whitespace collapse, Unicode-aware
letters/numbers, and preserved in-word apostrophes. Russian additionally maps
`ё` to `е`. Punctuation is removed without transliterating scripts.

Reported metrics include corpus WER/CER, substitutions, deletions, insertions,
hits, empty hypotheses, exact matches, and normalized word/character counts.
Headline WER uses paired utterance-level percentile bootstrap confidence intervals:
10,000 iterations, seed `20260716` plus the language index.

## Performance protocol

`performance/*.json` records deterministic recipes for four 300-second mono 16 kHz
signed 16-bit PCM WAV files. Audio bytes remain outside Git.

Each implementation/language pair runs in a new process:

1. load the model;
2. transcribe once as the cold run;
3. transcribe five times as warm runs;
4. write the report and exit before the next pair.

Reports contain load/cold/warm wall time, median, p95, RTF, throughput, artifact
bytes and hashes, peak RSS, device/MLX active/cache/peak memory, available memory,
and swap start/end/delta. Runs are kept separate by language because autoregressive
latency depends on generated text and language detection. Parakeet performance is
measured only on the supported Russian corpus with practical 120-second chunks
and 15-second overlap; quality remains unchunked.

The recorded machine and run conditions are in `run-context.json`. No positive
swap delta or macOS thermal/performance warning was observed.

## Model selection presentation

The compact report and bilingual READMEs include one decision matrix with WER for
Russian, Kazakh, Kyrgyz, Uzbek, and the English appendix, plus Russian 5-minute
warm median, whole-process peak RSS, and model weight size. Symbols are computed
from report data rather than assigned manually:

- `✓` marks the minimum observed value in a column; exact ties are all marked;
- `◇` marks the Pareto frontier among MLX candidates by equal-weight macro WER
  over the four core languages, 5-minute time, peak RSS, and model size;
- `★` marks INT8 as the recommended default balance, not a per-column winner.

Original GigaAM is excluded from the Pareto candidate set because it is the
upstream PyTorch/MPS reference baseline. English is excluded because it is an
appendix. Parakeet is ineligible because it does not support three core languages.
The resulting frontier is GigaAM MLX FP16, INT8, INT6, and INT4. The compact JSON
records the definition, rows, frontier, and column leaders under
`model_selection`.

## Reproduce

Heavy inputs belong in an external ML data root. Adapt these templates to that
root; do not place weights, audio, or raw reports in Git.

```bash
python tools/prepare_fleurs_multilingual.py --help
python tools/export_multilingual_manifests.py --help
python tools/build_multilingual_perf_corpus.py --help
python tools/run_multilingual_quality.py --help
python tools/run_multilingual_whisper_matrix.py --help
python tools/run_whisper_benchmark.py --help
python tools/run_parakeet_quality.py --help
python tools/run_parakeet_benchmark.py --help
python tools/build_multilingual_report.py --help
```

Compact results are in [`docs/benchmark-multilingual-v1.md`](../../docs/benchmark-multilingual-v1.md)
and [`docs/benchmark-multilingual-v1.json`](../../docs/benchmark-multilingual-v1.json).
