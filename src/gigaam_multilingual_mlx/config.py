from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    model_name: str
    source_revision: str
    sample_rate: int
    features: int
    win_length: int
    hop_length: int
    n_fft: int
    center: bool
    n_layers: int
    d_model: int
    subs_kernel_size: int
    subsampling_factor: int
    ff_expansion_factor: int
    n_heads: int
    pos_emb_base: int
    conv_kernel_size: int
    num_classes: int
    vocabulary: list[str]
    dtype: str = "float32"

    @classmethod
    def from_source(cls, raw: dict[str, Any], revision: str) -> "ModelConfig":
        cfg = raw["cfg"]["model"]["cfg"]
        pre = cfg["preprocessor"]
        enc = cfg["encoder"]
        head = cfg["head"]
        if cfg["model_class"] != "ctc" or enc["subsampling"] != "conv1d":
            raise ValueError("Only the GigaAM CTC/conv1d architecture is supported")
        if enc["self_attention_model"] != "rotary" or enc["conv_norm_type"] != "layer_norm":
            raise ValueError("Only rotary attention with convolution LayerNorm is supported")
        return cls(
            model_name=cfg["model_name"],
            source_revision=revision,
            sample_rate=int(cfg["sample_rate"]),
            features=int(pre["features"]),
            win_length=int(pre["win_length"]),
            hop_length=int(pre["hop_length"]),
            n_fft=int(pre["n_fft"]),
            center=bool(pre["center"]),
            n_layers=int(enc["n_layers"]),
            d_model=int(enc["d_model"]),
            subs_kernel_size=int(enc["subs_kernel_size"]),
            subsampling_factor=int(enc["subsampling_factor"]),
            ff_expansion_factor=int(enc["ff_expansion_factor"]),
            n_heads=int(enc["n_heads"]),
            pos_emb_base=int(enc["pos_emb_max_len"]),
            conv_kernel_size=int(enc["conv_kernel_size"]),
            num_classes=int(head["num_classes"]),
            vocabulary=list(cfg["decoding"]["vocabulary"]),
        )

    @classmethod
    def load(cls, path: str | Path) -> "ModelConfig":
        return cls(**json.loads(Path(path).read_text()))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2) + "\n")


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
