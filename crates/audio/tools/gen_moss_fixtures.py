#!/usr/bin/env python3
"""Regenerate the MOSS audio tokenizer Rust parity fixtures.

Normative reference: mlx-audio at commit
e1b19b9054bf163f5d812221a54fcc346f1890e9
(mlX_audio/codec/models/moss_audio_tokenizer, mlx 0.32.3).

Run from inside the mlx-audio checkout root so `mlx_audio` imports from
the pinned source, with the project venv interpreter:

    cd ../mlx-audio
    ../.venv-mlxaudio/bin/python \
        ../turbospark/crates/audio/tools/gen_moss_fixtures.py

Outputs into crates/audio/testdata/moss_audio_tokenizer/:

- tiny_config.json + tiny_weights.safetensors
    Seeded tiny tokenizer: stereo input with channel interleave, a
    patch downsample, windowed causal transformers with LayerScale and
    interleaved RoPE, and a 3-book residual LFQ.
- traces.json + npy files
    Quantized latents, codes (with lookup margins), code length, and
    the decoded waveform.
- manifest.json
    File list with SHA-256 digests and generator metadata.

The script is deterministic: fixed seeds, no network access. The
streaming step path is not part of the batch contract and is not
exercised.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from mlx_audio.codec.models.moss_audio_tokenizer.moss_audio_tokenizer import (
    AudioTokenizerConfig,
    MossAudioTokenizer,
)

OUT = Path("../turbospark/crates/audio/testdata/moss_audio_tokenizer")
REFERENCE_COMMIT = "e1b19b9054bf163f5d812221a54fcc346f1890e9"

TINY = {
    "sample_rate": 8000,
    "downsample_rate": 8,
    "number_channels": 2,
    "enable_channel_interleave": True,
    "causal_transformer_context_duration": 10.0,
    "encoder_kwargs": [
        {"module_type": "PatchedPretransform", "patch_size": 4},
        {
            "module_type": "Transformer",
            "input_dimension": 4,
            "output_dimension": 32,
            "d_model": 32,
            "num_heads": 4,
            "num_layers": 2,
            "dim_feedforward": 64,
            "causal": True,
            "context_duration": 0.001,
            "conv_layout": False,
            "positional_embedding": "rope",
            "max_period": 10000.0,
            "layer_scale": 0.1,
        },
    ],
    "decoder_kwargs": [
        {
            "module_type": "Transformer",
            "input_dimension": 32,
            "output_dimension": 4,
            "d_model": 32,
            "num_heads": 4,
            "num_layers": 2,
            "dim_feedforward": 64,
            "causal": True,
            "context_duration": 0.001,
            "conv_layout": False,
            "positional_embedding": "rope",
            "max_period": 10000.0,
            "layer_scale": 0.1,
        },
        {"module_type": "PatchedPretransform", "patch_size": 4},
    ],
    "quantizer_type": "rlfq",
    "quantizer_kwargs": {
        "input_dim": 32,
        "rvq_dim": 32,
        "output_dim": 32,
        "num_quantizers": 3,
        "codebook_size": 16,
        "codebook_dim": 4,
    },
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_npy(name: str, arr: np.ndarray) -> None:
    np.save(OUT / name, np.ascontiguousarray(arr, dtype=np.float32))


def lfq_margins(model: MossAudioTokenizer, hidden: np.ndarray, length: int) -> float:
    """Smallest best-vs-second dot margin across books (normalized).

    `hidden` is one unbatched latent (rvq_dim, frames); the residual
    update mirrors ResidualLFQ.__call__ with the frame mask applied.
    """
    quantizer = model.quantizer
    z = np.asarray(quantizer.input_proj(mx.array(hidden)[None]), dtype=np.float32)[0]
    z[:, length:] = 0.0
    residual = z.copy()
    min_margin = np.inf
    for book in quantizer.quantizers:
        w = np.asarray(book.codebook.weight, dtype=np.float32)
        w = w / np.maximum(np.linalg.norm(w, axis=-1, keepdims=True), 1e-12)
        z_e = np.asarray(book.in_proj(mx.array(residual)[None]), dtype=np.float32)[0]
        r = np.ascontiguousarray(z_e.T)
        r = r / np.maximum(np.linalg.norm(r, axis=-1, keepdims=True), 1e-12)
        dots = r @ w.T  # (T, size)
        order = np.argsort(-dots, axis=-1, kind="stable")
        best = np.take_along_axis(dots, order[:, :1], axis=-1)
        second = np.take_along_axis(dots, order[:, 1:2], axis=-1)
        min_margin = min(min_margin, float((best - second).min()))
        indices = np.argmax(dots, axis=-1)
        raw = w[indices]  # (T, dim) raw rows
        # moss WNConv1d takes (B, C, T) and transposes internally.
        zq = np.asarray(
            book.out_proj(mx.array(np.ascontiguousarray(raw.T))[None]),
            dtype=np.float32,
        )[0]  # (rvq_dim, T)
        residual = residual - zq
    return min_margin


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    traces: dict = {}
    trace_files: list[str] = []

    config = AudioTokenizerConfig.from_dict(TINY)
    mx.random.seed(501)
    weights = mx.load("../mlx-audio/noop.safetensors") if False else None
    model = MossAudioTokenizer(config)
    flat = dict(tree_flatten(model.parameters()))
    rng = np.random.default_rng(503)
    seeded = {
        name: mx.array((rng.standard_normal(v.shape) * 0.3).astype(np.float32))
        for name, v in flat.items()
    }
    model.load_weights(list(seeded.items()))
    mx.eval(model)
    mx.save_safetensors(str(OUT / "tiny_weights.safetensors"), seeded)
    (OUT / "tiny_config.json").write_text(json.dumps(TINY, indent=2))
    (OUT / "weights_manifest.json").write_text(
        json.dumps({"tiny": [{"name": n, "shape": list(v.shape)} for n, v in sorted(flat.items())]}, indent=2)
    )

    # Stereo-capable mono input: 40 samples per channel.
    rng = np.random.default_rng(507)
    wave = (rng.standard_normal(40) * 0.2).astype(np.float32)
    save_npy("wave.npy", wave)
    trace_files.append("wave.npy")

    codes = model.encode_audio(mx.array(wave), sample_rate=8000)
    codes_np = np.asarray(codes, dtype=np.int32)  # (frames, nq)
    np.save(OUT / "codes.npy", np.ascontiguousarray(codes_np))
    traces["codes_shape"] = list(codes_np.shape)
    trace_files.append("codes.npy")

    # Quantized latents + margins through the internal paths.
    input_values = mx.array(np.stack([wave, wave]))[None]  # (1, 2, 40)
    audio_codes, code_lengths, hidden = model._encode_frame(input_values, None, None)
    hidden = np.asarray(hidden)[0]
    save_npy("quantized.npy", np.asarray(model.quantizer(mx.array(hidden)[None], mx.array([20]))[0])[0])
    traces["code_len"] = int(np.asarray(code_lengths)[0])
    traces["hidden_shape"] = list(hidden.shape)
    traces["lfq_min_margin"] = lfq_margins(model, hidden, traces["code_len"])
    trace_files.append("quantized.npy")

    audio = model.decode_audio_codes(codes_np)
    audio_np = np.asarray(audio, dtype=np.float32)  # (samples, channels)
    save_npy("decoded.npy", audio_np)
    traces["decoded_shape"] = list(audio_np.shape)
    trace_files.append("decoded.npy")

    (OUT / "traces.json").write_text(json.dumps(traces, indent=2))

    files = [
        "tiny_config.json",
        "tiny_weights.safetensors",
        "weights_manifest.json",
        "traces.json",
        *trace_files,
    ]
    manifest = {
        "generator": "tools/gen_moss_fixtures.py",
        "reference_commit": REFERENCE_COMMIT,
        "mlx_version": mx.__version__,
        "files": {name: digest(OUT / name) for name in files},
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(files) + 1} fixture files to {OUT}")


if __name__ == "__main__":
    main()
