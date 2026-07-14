# Security policy

## Supported versions

| Version | Supported |
|---|---|
| 0.1.x | yes |
| older/unreleased snapshots | no |

## Reporting a vulnerability

Use GitHub private vulnerability reporting for
`ai-babai/gigaam-multilingual-mlx`. Do not open a public issue containing an
unpatched vulnerability, credentials, private audio, or local filesystem data.

Include the affected version, platform, reproduction steps, and impact. Reports are
acknowledged as soon as practical; a disclosure timeline is agreed after validation.

The project never requests model-Hub or package-registry tokens in an issue. Release
workflows use GitHub OIDC for PyPI and narrowly scoped credentials for Hugging Face.
