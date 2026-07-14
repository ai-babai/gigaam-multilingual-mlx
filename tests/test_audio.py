from __future__ import annotations

import numpy as np
import pytest

from gigaam_mlx.audio import SAMPLE_RATE, fixed_chunks


def test_fixed_chunks_cover_audio_with_deterministic_overlap() -> None:
    audio = np.zeros(45 * SAMPLE_RATE, dtype=np.float32)
    chunks = list(fixed_chunks(audio, chunk_seconds=20, overlap_seconds=2))
    assert [(start, end) for start, end, _ in chunks] == [
        (0, 20 * SAMPLE_RATE),
        (18 * SAMPLE_RATE, 38 * SAMPLE_RATE),
        (36 * SAMPLE_RATE, 45 * SAMPLE_RATE),
    ]


@pytest.mark.parametrize(
    ("chunk_seconds", "overlap_seconds"),
    [(0, 0), (20, -1), (20, 20), (20, 21)],
)
def test_fixed_chunks_reject_invalid_geometry(chunk_seconds: float, overlap_seconds: float) -> None:
    with pytest.raises(ValueError, match="chunk_seconds > overlap_seconds"):
        list(fixed_chunks(np.zeros(SAMPLE_RATE), chunk_seconds, overlap_seconds))
