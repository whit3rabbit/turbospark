#!/usr/bin/env python3
"""Regenerate the MiMo audio tokenizer Rust parity fixtures.

Normative reference: mlx-audio at commit
e1b19b9054bf163f5d812221a54fcc346f1890e9
(mlX_audio/codec/models/mimo_audio_tokenizer, mlx 0.32.3).

Run from inside the mlx-audio checkout root so `mlx_audio` imports from
the pinned source, with the project venv interpreter:

    cd ../mlx-audio
    ../.venv-mlxaudio/bin/python \
        ../turbospark/crates/audio/tools/gen_mimo_fixtures.py

Outputs into crates/audio/testdata/mimo_audio_tokenizer/:

- tiny_config.json + tiny_weights.safetensors
    Seeded tiny tokenizer (the dataclass is config-driven, so the tiny
    geometry is a plain config; weights_format=mlx keeps tree keys).
- traces.json + npy files
    Log-mel front end golden, encoder hidden states, RVQ codes with
    margins, and a full codes-to-waveform decode.
- manifest.json
    File list with SHA-256 digests and generator metadata.

The script is deterministic: fixed seeds, no network access.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import fields
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from mlx_audio.codec.models.mimo_audio_tokenizer.config import ModelConfig
from mlx_audio.codec.models.mimo_audio_tokenizer.model import MiMoAudioTokenizer

OUT = Path("../turbospark/crates/audio/testdata/mimo_audio_tokenizer")
REFERENCE_COMMIT = "e1b19b9054bf163f5d812221a54fcc346f1890e9"

TINY = dict(
    model_type="mimo_audio_tokenizer",
    d_model=32,
    n_mels=8,
    sampling_rate=24000,
    nfft=40,
    hop_length=10,
    window_size=40,
    fmin=0,
    fmax=None,
    kernel_size=3,
    stride_size=2,
    avg_pooler=2,
    encoder_layers=3,
    encoder_skip_layer_id=2,
    encoder_attention_heads=4,
    encoder_ffn_dim=64,
    encoder_causal=False,
    encoder_attn_window_size=[-1, -1],
    decoder_layers=2,
    decoder_attention_heads=4,
    decoder_ffn_dim=64,
    decoder_kernel_size=3,
    decoder_stride_size=2,
    decoder_causal=True,
    decoder_attn_window_size=[-1, -1],
    vocoder_dim=16,
    vocoder_intermediate_dim=32,
    vocoder_num_layers=2,
    vocoder_attention_heads=2,
    vocoder_attn_window_size=[2, 1],
    vocoder_padding="same",
    num_quantizers=3,
    codebook_size=[8, 6, 5],
    rope_theta=10000,
    rope_type="default",
    ln_type="LayerNorm",
    activation_function="gelu",
    weights_format="mlx",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_npy(name: str, arr: np.ndarray) -> None:
    np.save(OUT / name, np.ascontiguousarray(arr, dtype=np.float32))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    traces: dict = {}
    trace_files: list[str] = []

    cfg = ModelConfig(**TINY)
    mx.random.seed(401)
    model = MiMoAudioTokenizer(cfg)
    flat = dict(tree_flatten(model.parameters()))
    rng = np.random.default_rng(403)
    seeded = {
        name: mx.array((rng.standard_normal(v.shape) * 0.2).astype(np.float32))
        for name, v in flat.items()
    }
    model.load_weights(list(seeded.items()))
    mx.eval(model)
    mx.save_safetensors(str(OUT / "tiny_weights.safetensors"), seeded)
    (OUT / "tiny_config.json").write_text(
        json.dumps({f.name: getattr(cfg, f.name) for f in fields(cfg)}, indent=2)
    )
    (OUT / "weights_manifest.json").write_text(
        json.dumps({"tiny": [{"name": n, "shape": list(v.shape)} for n, v in sorted(flat.items())]}, indent=2)
    )

    # Log-mel front end golden: 960 samples at 24 kHz -> 5 frames.
    wave_rng = np.random.default_rng(407)
    wave = (wave_rng.standard_normal(960) * 0.1).astype(np.float32)
    save_npy("wave.npy", wave)
    trace_files.append("wave.npy")
    mels = np.asarray(model.encoder_input(
        mx.array(wave)
    )) if hasattr(model, "encoder_input") else None
    from mlx_audio.codec.models.mimo_audio_tokenizer.audio import log_mel_spectrogram
    mel = np.asarray(log_mel_spectrogram(mx.array(wave), cfg))
    save_npy("mel_golden.npy", mel)
    traces["mel_shape"] = list(mel.shape)
    trace_files.append("mel_golden.npy")

    # Encode: mel rows + valid length -> hidden + codes.
    mels_in = mel
    length = mels_in.shape[0]
    hidden = np.asarray(model.encoder(mx.array(mels_in)[None], length=length))[0]
    save_npy("hidden.npy", hidden)
    traces["hidden_shape"] = list(hidden.shape)
    trace_files.append("hidden.npy")
    codes = np.asarray(model.encoder.quantizer.encode(mx.array(hidden), None))
    np.save(OUT / "codes.npy", np.ascontiguousarray(codes.astype(np.int32)))
    traces["codes_shape"] = list(codes.shape)
    trace_files.append("codes.npy")
    # Margins per book (with the x^2 term restored for the margin only).
    margins = []
    residual = hidden.astype(np.float32)
    for book in model.encoder.quantizer.codebooks:
        w = np.asarray(book.weight, dtype=np.float32)
        d = (residual**2).sum(-1, keepdims=True) - 2.0 * residual @ w.T + (w**2).sum(-1)[None, :]
        order = np.argsort(d, axis=-1, kind="stable")
        best = np.take_along_axis(d, order[:, :1], axis=-1)
        second = np.take_along_axis(d, order[:, 1:2], axis=-1)
        margins.append(float((second - best).min()))
        idx = np.argmin(d, axis=-1)
        residual = residual - w[idx]
    traces["rvq_min_margin"] = min(margins)

    # Decode the codes back to a waveform.
    audio = np.asarray(model.decode(codes))
    save_npy("decoded.npy", audio)
    traces["decoded_len"] = int(audio.shape[0])
    traces["downsample_rate"] = int(cfg.downsample_rate)
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
        "generator": "tools/gen_mimo_fixtures.py",
        "reference_commit": REFERENCE_COMMIT,
        "mlx_version": mx.__version__,
        "files": {name: digest(OUT / name) for name in files},
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(files) + 1} fixture files to {OUT}")


if __name__ == "__main__":
    main()
