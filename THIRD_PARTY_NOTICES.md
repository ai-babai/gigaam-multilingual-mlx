# Third-party notices

## GigaAM and GigaAM-Multilingual

This project is an independent MLX port derived from the architecture and model
artifacts published by the GigaAM authors:

- official code: <https://github.com/salute-developers/GigaAM>
- source model: <https://huggingface.co/ai-sage/GigaAM-Multilingual>
- pinned source revision: `3905cd51c3ed4e88c8edf33f3302969ba480a327`

The upstream code and model card declare the MIT License.

```text
MIT License

Copyright (c) 2024 GigaChat Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

Converted weights remain derived model artifacts and are published separately on
Hugging Face. They are not included in the Python package or GitHub repository.

## Public evaluation datasets

Only pinned IDs, hashes, selection rules, and compact aggregate results are stored
in this repository. Dataset audio remains outside Git.

| Dataset | Source | License recorded by source card |
|---|---|---|
| FLEURS | <https://huggingface.co/datasets/google/fleurs> | CC BY 4.0 |
| Russian LibriSpeech (OpenSLR SLR96) | <https://openslr.org/96/> | Public Domain in the USA; see source card |
| SOVA RuDevices | <https://huggingface.co/datasets/bond005/sova_rudevices> | CC BY 4.0 |

Golos and unofficial Common Voice mirrors are not part of the headline `v0.1.0`
benchmark.
