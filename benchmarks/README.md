# Public release benchmark

The `v0.1.0` benchmark uses public, pinned inputs only. It never uses private/user
audio or a private pseudo-reference.

The newer [`public-asr-multilingual-v1`](multilingual-v1/README.md) suite adds
same-audio quality, speed, disk, and memory comparisons with MLX Whisper large-v2,
large-v3, Turbo, and MLX Parakeet TDT 0.6B v3. Parakeet is reported only for its
supported Russian and English slices; Kazakh, Kyrgyz, and Uzbek are `N/A`. Its
compact results are in
[`docs/benchmark-multilingual-v1.md`](../docs/benchmark-multilingual-v1.md).

## Quality suite: `public-asr-quality-v1`

| Dataset | Config/split | Revision | License recorded by source card |
|---|---|---|---|
| `google/fleurs` | `ru_ru`, test subset | `70bb2e84b976b7e960aa89f1c648e09c59f894dd` | CC BY 4.0 |
| `istupakov/russian_librispeech` | test subset | `a519c986bb3342cc8136d3d14e5ad8a4f1e1a2bd` | Public Domain in the USA; HF conversion metadata also records CC BY 4.0 |
| `bond005/sova_rudevices` | test subset | `d9197eacfb0afff29d90a2d4e7d0d98a5dfb54bc` | CC BY 4.0 |

Portable selection manifests under `manifests/` contain source IDs/rows, revisions,
audio SHA-256, references, durations, licenses, and selection rules. They deliberately
exclude local paths and audio bytes.

All implementations use greedy CTC decoding, identical audio bytes, batch size 8,
and normalization rule `gigaam-multilingual-mlx-text-v1`:

1. Unicode text is lowercased.
2. `ё` is mapped to `е`.
3. Curly apostrophes become `'`; an apostrophe is retained only inside a word.
4. Letters, digits, and whitespace are retained.
5. Unicode dash punctuation becomes a space; other punctuation is removed.
6. Whitespace is collapsed; empty references remain explicit and are not silently
   removed.

The primary metric is corpus WER from total substitutions/deletions/insertions.
Corpus CER, empty hypotheses, exact matches, item/word counts, and duration are also
reported. Dataset rows are shown separately; pooled WER is supplementary.

## Performance suite: `public-asr-perf-v1`

`tools/build_public_perf_corpus.py` deterministically merges FLEURS and Russian
LibriSpeech items sorted by audio SHA-256. It inserts 250 ms of silence between
items, decodes/resamples every source through `ffmpeg` to mono 16 kHz signed 16-bit
PCM, and trims the final item to exact profiles:

- short: 30 seconds;
- medium: 5 minutes;
- long: 20 minutes.

The generated WAV files stay outside Git. `public-asr-perf-v1.json` records source
IDs/revisions/hashes, item order, used samples, silence, PCM rules, and final WAV
hashes so the inputs can be reproduced.

For every implementation, the release run records model load separately from cold
and warm inference, five measured warm runs for short/medium, at least three long
runs (or an explicit limitation), RTF/throughput, peak resident process memory,
device/MLX memory, and swap delta. Original PyTorch/MPS and all MLX variants run on
the same machine with 20-second chunks and two-second overlap.

## Developer commands

The exact absolute paths depend on the local external data root; none belong in a
published report.

```bash
python tools/export_quality_manifests.py --help
python tools/build_public_perf_corpus.py --help
python -m gigaam_multilingual_mlx.dev_cli evaluate --help
python -m gigaam_multilingual_mlx.dev_cli benchmark --help
python -m gigaam_multilingual_mlx.dev_cli benchmark-pytorch --help
```

Raw hypotheses and generated audio remain outside the repository. The compact final
results are published in `docs/benchmark-v0.1.0.md` and
`docs/benchmark-v0.1.0.json`.
