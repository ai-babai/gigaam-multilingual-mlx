from __future__ import annotations

import unicodedata


NORMALIZATION_VERSION = "gigaam-multilingual-mlx-text-v1"


def normalize_text(text: str) -> str:
    text = text.replace("ё", "е").replace("Ё", "Е").lower()
    for quote in ("’", "‘", "ʻ", "ʼ"):
        text = text.replace(quote, "'")
    output = []
    for index, char in enumerate(text):
        if char == "'":
            if (
                0 < index < len(text) - 1
                and text[index - 1].isalnum()
                and text[index + 1].isalnum()
            ):
                output.append(char)
        elif char.isalnum() or char.isspace():
            output.append(char)
        elif unicodedata.category(char) == "Pd":
            output.append(" ")
    return " ".join("".join(output).split())
