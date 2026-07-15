from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .config import ModelConfig, sha256_file


def _length_after_conv(lengths: mx.array, kernel: int, stride: int, padding: int) -> mx.array:
    return mx.floor((lengths.astype(mx.float32) + 2 * padding - kernel) / stride + 1).astype(
        mx.int32
    )


def _time_mask(x: mx.array, lengths: mx.array) -> mx.array:
    valid = mx.arange(x.shape[1])[None, :] < lengths[:, None]
    return mx.where(valid[:, :, None], x, mx.zeros_like(x))


class LogMelFrontend(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.win_length = config.win_length
        self.hop_length = config.hop_length
        self.n_fft = config.n_fft
        self.center = config.center
        self.window = mx.zeros((config.win_length,), dtype=mx.float32)
        self.mel_filter = mx.zeros((config.n_fft // 2 + 1, config.features), dtype=mx.float32)

    def __call__(self, audio: mx.array, lengths: mx.array) -> tuple[mx.array, mx.array]:
        if self.center:
            raise ValueError("The supported checkpoints use center=False")
        if audio.ndim == 1:
            audio = audio[None, :]
        if audio.shape[1] < self.win_length:
            audio = mx.pad(audio, ((0, 0), (0, self.win_length - audio.shape[1])))
        frames_count = 1 + (audio.shape[1] - self.win_length) // self.hop_length
        frames = mx.as_strided(
            audio,
            shape=(audio.shape[0], frames_count, self.win_length),
            strides=(audio.shape[1], self.hop_length, 1),
        )
        spectrum = mx.fft.rfft(frames * self.window, n=self.n_fft, axis=-1)
        power = mx.abs(spectrum).square()
        features = mx.log(mx.clip(power @ self.mel_filter, 1e-9, 1e9))
        output_lengths = mx.floor((lengths - self.win_length) / self.hop_length + 1).astype(
            mx.int32
        )
        return features, mx.maximum(output_lengths, 0)


class StridingSubsampling(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        padding = (config.subs_kernel_size - 1) // 2
        self.kernel = config.subs_kernel_size
        self.stride = 2
        self.padding = padding
        self.conv1 = nn.Conv1d(
            config.features, config.d_model, self.kernel, stride=2, padding=padding
        )
        self.conv2 = nn.Conv1d(
            config.d_model, config.d_model, self.kernel, stride=2, padding=padding
        )

    def __call__(self, x: mx.array, lengths: mx.array) -> tuple[mx.array, mx.array]:
        x = _time_mask(x, lengths)
        lengths1 = _length_after_conv(lengths, self.kernel, self.stride, self.padding)
        x = _time_mask(nn.relu(self.conv1(x)), lengths1)
        lengths2 = _length_after_conv(lengths1, self.kernel, self.stride, self.padding)
        x = _time_mask(nn.relu(self.conv2(x)), lengths2)
        return x, lengths2


class FeedForward(nn.Module):
    def __init__(self, d_model: int, expansion: int):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_model * expansion)
        self.linear2 = nn.Linear(d_model * expansion, d_model)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear2(nn.silu(self.linear1(x)))


def _rotate_half(x: mx.array) -> mx.array:
    half = x.shape[-1] // 2
    return mx.concatenate((-x[..., half:], x[..., :half]), axis=-1)


class RotaryAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)

    def __call__(
        self, x: mx.array, cos: mx.array, sin: mx.array, attention_mask: mx.array | None
    ) -> mx.array:
        batch, time, width = x.shape
        shaped = x.reshape(batch, time, self.n_heads, self.head_dim)
        rotated = shaped * cos[None, :, None, :] + _rotate_half(shaped) * sin[None, :, None, :]
        flat = rotated.reshape(batch, time, width)
        q = (
            self.linear_q(flat)
            .reshape(batch, time, self.n_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        k = (
            self.linear_k(flat)
            .reshape(batch, time, self.n_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        v = self.linear_v(x).reshape(batch, time, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        scores = (q @ k.transpose(0, 1, 3, 2)) * (self.head_dim**-0.5)
        if attention_mask is not None:
            mask = attention_mask[:, None, :, :]
            scores = mx.where(mask, mx.array(-10000.0, scores.dtype), scores)
            probabilities = mx.where(mask, mx.zeros_like(scores), mx.softmax(scores, axis=-1))
        else:
            probabilities = mx.softmax(scores, axis=-1)
        output = (probabilities @ v).transpose(0, 2, 1, 3).reshape(batch, time, width)
        return self.linear_out(output)


class ConformerConvolution(nn.Module):
    def __init__(self, d_model: int, kernel: int):
        super().__init__()
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, 1)
        self.depthwise_conv = nn.Conv1d(
            d_model, d_model, kernel, padding=(kernel - 1) // 2, groups=d_model
        )
        # The official model calls this field batch_norm even when it is LayerNorm.
        self.batch_norm = nn.LayerNorm(d_model)
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, 1)

    def __call__(self, x: mx.array, padding_mask: mx.array | None) -> mx.array:
        x = self.pointwise_conv1(x)
        left, right = mx.split(x, 2, axis=-1)
        x = left * mx.sigmoid(right)
        if padding_mask is not None:
            x = mx.where(padding_mask[:, :, None], mx.zeros_like(x), x)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        return self.pointwise_conv2(nn.silu(x))


class ConformerLayer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        d = config.d_model
        self.norm_feed_forward1 = nn.LayerNorm(d)
        self.feed_forward1 = FeedForward(d, config.ff_expansion_factor)
        self.norm_conv = nn.LayerNorm(d)
        self.conv = ConformerConvolution(d, config.conv_kernel_size)
        self.norm_self_att = nn.LayerNorm(d)
        self.self_attn = RotaryAttention(d, config.n_heads)
        self.norm_feed_forward2 = nn.LayerNorm(d)
        self.feed_forward2 = FeedForward(d, config.ff_expansion_factor)
        self.norm_out = nn.LayerNorm(d)

    def __call__(
        self,
        x: mx.array,
        cos: mx.array,
        sin: mx.array,
        attention_mask: mx.array | None,
        padding_mask: mx.array | None,
    ) -> mx.array:
        residual = x + 0.5 * self.feed_forward1(self.norm_feed_forward1(x))
        residual = residual + self.self_attn(self.norm_self_att(residual), cos, sin, attention_mask)
        residual = residual + self.conv(self.norm_conv(residual), padding_mask)
        residual = residual + 0.5 * self.feed_forward2(self.norm_feed_forward2(residual))
        return self.norm_out(residual)


class ConformerEncoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.pre_encode = StridingSubsampling(config)
        self.layers = [ConformerLayer(config) for _ in range(config.n_layers)]

    def _rotary(self, length: int, dtype: mx.Dtype) -> tuple[mx.array, mx.array]:
        dim = self.config.d_model // self.config.n_heads
        inv_freq = 1.0 / (
            self.config.pos_emb_base ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim)
        )
        freqs = mx.arange(length, dtype=mx.float32)[:, None] * inv_freq[None, :]
        embedding = mx.concatenate((freqs, freqs), axis=-1).astype(dtype)
        return mx.cos(embedding), mx.sin(embedding)

    def __call__(self, features: mx.array, lengths: mx.array) -> tuple[mx.array, mx.array]:
        x, lengths = self.pre_encode(features, lengths)
        cos, sin = self._rotary(x.shape[1], x.dtype)
        valid = mx.arange(x.shape[1])[None, :] < lengths[:, None]
        padding_mask = ~valid
        attention_mask = None
        if x.shape[0] > 1:
            attention_mask = ~(valid[:, :, None] & valid[:, None, :])
        for layer in self.layers:
            x = layer(x, cos, sin, attention_mask, padding_mask)
        return x, lengths


class CTCHead(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.decoder = nn.Conv1d(config.d_model, config.num_classes, 1)

    def __call__(self, x: mx.array) -> mx.array:
        logits = self.decoder(x)
        return logits - mx.logsumexp(logits, axis=-1, keepdims=True)


class GigaAMCTC(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.preprocessor = LogMelFrontend(config)
        self.encoder = ConformerEncoder(config)
        self.head = CTCHead(config)

    def __call__(
        self, audio: mx.array, lengths: mx.array | None = None
    ) -> tuple[mx.array, mx.array]:
        if audio.ndim == 1:
            audio = audio[None, :]
        if lengths is None:
            lengths = mx.array([audio.shape[1]] * audio.shape[0], dtype=mx.int32)
        features, feature_lengths = self.preprocessor(audio, lengths)
        compute_dtype = {
            "float32": mx.float32,
            "float16": mx.float16,
            "bfloat16": mx.bfloat16,
        }[self.config.dtype]
        features = features.astype(compute_dtype)
        encoded, encoded_lengths = self.encoder(features, feature_lengths)
        return self.head(encoded), encoded_lengths

    def greedy_decode(self, logits: mx.array, lengths: mx.array) -> list[dict[str, object]]:
        labels = np.asarray(mx.argmax(logits, axis=-1))
        lens = np.asarray(lengths).tolist()
        blank = len(self.config.vocabulary)
        results = []
        for row, length in zip(labels, lens, strict=True):
            token_ids: list[int] = []
            frames: list[int] = []
            previous = None
            for frame, token in enumerate(row[: int(length)].tolist()):
                if token != blank and token != previous:
                    token_ids.append(token)
                    frames.append(frame)
                previous = token
            results.append(
                {
                    "text": "".join(self.config.vocabulary[token] for token in token_ids),
                    "token_ids": token_ids,
                    "token_frames": frames,
                }
            )
        return results


def load_model(
    model_dir: str | Path | None = None,
    dtype: str | None = None,
    *,
    variant: str = "int8",
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
    token: str | bool | None = None,
) -> GigaAMCTC:
    from .artifacts import resolve_model_source

    reference = resolve_model_source(
        model_dir,
        variant=variant,
        revision=revision,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        token=token,
    )
    model_dir = reference.path
    manifest = reference.manifest
    config_path = model_dir / manifest["config"]["file"]
    if sha256_file(config_path) != manifest["config"]["sha256"]:
        raise ValueError("Converted config hash does not match manifest")
    config = ModelConfig.load(config_path)
    weights_path = model_dir / manifest["weights"]["file"]
    if sha256_file(weights_path) != manifest["weights"]["sha256"]:
        raise ValueError("Converted weight hash does not match manifest")
    if dtype is not None:
        if manifest.get("quantization") and dtype != config.dtype:
            raise ValueError("A quantized artifact cannot be loaded with a different dtype")
        config = replace(config, dtype=dtype)
    model = GigaAMCTC(config)
    if quantization := manifest.get("quantization"):
        from .quantization import apply_quantization

        apply_quantization(model, quantization)
    model.load_weights(str(weights_path), strict=True)
    model.eval()
    model.artifact_reference = {
        "variant": reference.variant,
        "repo_id": reference.repo_id,
        "revision": reference.revision,
        "source": reference.source,
        "path": str(reference.path),
    }
    return model
