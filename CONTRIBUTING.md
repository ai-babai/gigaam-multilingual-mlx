# Contributing

Thank you for helping improve this independent GigaAM-Multilingual MLX port.

## Development setup

Use Python 3.12 or 3.13 on an Apple Silicon Mac. Keep model weights, datasets,
virtual environments, and generated benchmark outputs outside the repository.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,convert,quality]'
pytest
ruff check .
```

The production `gigaam-multilingual-mlx` CLI contains user inference commands only. Conversion,
quantization, evaluation, parity, and benchmark work belongs in
`python -m gigaam_multilingual_mlx.dev_cli` and must use lazy optional imports.

## Pull requests

- Describe observable behavior and user impact.
- Add or update behavior tests for changed public behavior.
- Do not commit weights, datasets, generated audio, raw hypotheses, secrets, local
  paths, or private/user media.
- Preserve upstream attribution and immutable revisions.
- If tensor bytes change, rerun strict load, parity, and public quality gates.

Security vulnerabilities should follow [`SECURITY.md`](SECURITY.md), not a public
issue.
