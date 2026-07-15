from __future__ import annotations

from gigaam_multilingual_mlx.normalization import NORMALIZATION_VERSION, normalize_text


def test_public_quality_normalization_contract() -> None:
    assert NORMALIZATION_VERSION == "gigaam-multilingual-mlx-text-v1"
    assert normalize_text("  Ёж — это ‘тест’, rock’n’roll!  ") == "еж это тест rock'n'roll"


def test_public_quality_normalization_keeps_letters_and_digits() -> None:
    assert normalize_text("Русский English 2026; қазақша.") == "русский english 2026 қазақша"
