#!/usr/bin/env python3
"""Regenerate the DialogueSidon Rust parity fixtures.

Normative reference: mlx-audio at commit
e1b19b9054bf163f5d812221a54fcc346f1890e9
(mlX_audio/sts/models/dialogue_sidon, mlx 0.32.3).

Run from inside the mlx-audio checkout root so `mlx_audio` imports from
the pinned source, with the project venv interpreter:

    cd ../mlx-audio
    ../.venv-mlxaudio/bin/python \
        ../turbospark/crates/audio/tools/gen_dialogue_sidon_fixtures.py

Outputs into crates/audio/testdata/dialogue_sidon/:

- config.json
    Tiny seeded architecture (2 encoder layers at hidden 32, 2
    diffusion blocks at hidden 32, small DAC decoder). The real preset
    scales layer counts/dims only (13 encoder layers at 1024, 8
    diffusion blocks at 768, decoder 1536 with rates [8, 5, 4, 3]).
- tiny_weights.safetensors
    Seeded F32 parameters for the Model tree (encoder, linear1/2,
    diffusion head, DAC decoder at "decoder.model.*" keys).
- tiny_dac.safetensors
    The decoder subtree plus minimal dummy encoder/quantizer tensors
    so the crate Dac loader (which expects a full DAC tree) can load
    it; only the decoder weights are meaningful.
- wave.npy + stage traces
    A deterministic mono wave with fbank features and mask, encoder
    outputs, conditioning, one diffusion-head call, the sampled
    latents from seeded noise, and the decoded two-speaker waveform.
- manifest.json
    File list with SHA-256 digests and generator metadata.

The script is deterministic: fixed seeds, no network access.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

from mlx_audio.codec.models.descript.dac import Decoder
from mlx_audio.sts.models.dialogue_sidon.config import (
    DiffusionConfig,
    EncoderConfig,
    ModelConfig,
)
from mlx_audio.sts.models.dialogue_sidon.diffusion import DiffusionHead
from mlx_audio.sts.models.dialogue_sidon.encoder import Encoder
from mlx_audio.sts.models.dialogue_sidon.frontend import extract_features
from mlx_audio.sts.models.dialogue_sidon.model import Model

OUT = Path("../turbospark/crates/audio/testdata/dialogue_sidon")

CONFIG = {
    "sample_rate": 24000,
    "latent_dim": 4,
    "encoder": {
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "feature_projection_input_dim": 160,
        "conv_depthwise_kernel_size": 5,
        "left_max_position_embeddings": 64,
        "right_max_position_embeddings": 8,
        "layer_norm_eps": 1e-5,
    },
    "diffusion": {
        "hidden_size": 32,
        "num_layers": 2,
        "num_heads": 2,
        "ffn_ratio": 4.0,
        "frequency_embedding_size": 32,
        "num_train_timesteps": 1000,
        "beta_start": 0.0001,
        "beta_end": 0.02,
        "prediction_type": "v_prediction",
    },
    "decoder_channels": 64,
    "decoder_rates": [2, 3],
    "latent_norm_initialized": False,
    "latent_norm_mean": [],
    "latent_norm_std": [],
}

DAC_DUMMY = dict(
    encoder_dim=8,
    encoder_rates=[2, 4],
    latent_dim=16,
    decoder_dim=72,
    decoder_rates=[4, 2],
    n_codebooks=4,
    codebook_size=64,
    codebook_dim=8,
    sample_rate=24000,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_npy(name: str, arr: np.ndarray) -> None:
    np.save(OUT / name, np.ascontiguousarray(arr, dtype=np.float32))


def seed_parameters(model: nn.Module, seed: int, scale: float = 0.05) -> None:
    rng = np.random.default_rng(seed)
    flat = dict(tree_flatten(model.parameters()))
    updates = {
        name: mx.array((scale * rng.standard_normal(v.shape)).astype(np.float32))
        for name, v in flat.items()
    }
    model.update(tree_unflatten(updates))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    trace_files: list[str] = []
    traces: dict = {}

    (OUT / "config.json").write_text(json.dumps(CONFIG, indent=2) + "\n")
    trace_files.append("config.json")

    config = ModelConfig.from_dict(CONFIG)
    model = Model(config)
    model.eval()
    seed_parameters(model, 314)

    flat = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(str(OUT / "tiny_weights.safetensors"), flat)
    trace_files.append("tiny_weights.safetensors")

    # Standalone DAC tree for the crate loader: build a tiny real DAC,
    # seed it, and overwrite its decoder subtree with the sidon decoder
    # weights so only the decoder weights are meaningful.
    from mlx_audio.codec.models.descript.dac import DAC

    # encoder_dim * prod(rates) must equal the sidon latent dim, which
    # the crate DacConfig validates and the decoder pre-conv consumes.
    sidon_latent = CONFIG["latent_dim"]
    dac_cfg = dict(
        encoder_dim=sidon_latent,
        encoder_rates=[],
        latent_dim=sidon_latent,
        decoder_dim=CONFIG["decoder_channels"],
        decoder_rates=CONFIG["decoder_rates"],
        n_codebooks=4,
        codebook_size=64,
        codebook_dim=sidon_latent,
        sample_rate=24000,
    )
    dac = DAC(**dac_cfg)
    seed_parameters(dac, 4321)
    dac_flat = dict(tree_flatten(dac.parameters()))
    dec_flat = {f"decoder.{k}": v for k, v in flat.items() if k.startswith("decoder.")}
    merged = dict(dac_flat)
    merged.update(dec_flat)
    mx.save_safetensors(str(OUT / "tiny_dac.safetensors"), merged)
    (OUT / "tiny_dac_config.json").write_text(json.dumps(dac_cfg, indent=2) + "\n")
    trace_files.extend(["tiny_dac.safetensors", "tiny_dac_config.json"])

    # Deterministic mono wave: 1.0 s at 16 kHz after frontend resample.
    rng = np.random.default_rng(2718)
    t = np.arange(12000) / 16000.0
    wave = (
        0.3 * np.sin(2 * np.pi * 210.0 * t)
        + 0.15 * np.sin(2 * np.pi * 950.0 * t)
        + 0.05 * rng.standard_normal(12000)
    ).astype(np.float32)
    save_npy("wave.npy", wave)
    trace_files.append("wave.npy")

    # Stage: frontend features and mask.
    from mlx_audio.sts.models.dialogue_sidon.frontend import normalize_chunk

    features, mask = extract_features(normalize_chunk(mx.array(wave)))
    save_npy("features.npy", np.array(features[0]))
    np.save(OUT / "mask.npy", np.array(mask[0]).astype(np.float32))
    trace_files.extend(["features.npy", "mask.npy"])
    traces["features_shape"] = [int(d) for d in features.shape]

    # Stage: encoder + projections + conditioning.
    enc_out, first, second = model.encode(features, mask)
    save_npy("encoder_out.npy", np.array(enc_out[0]))
    save_npy("first.npy", np.array(first[0]))
    save_npy("second.npy", np.array(second[0]))
    trace_files.extend(["encoder_out.npy", "first.npy", "second.npy"])

    predicted = mx.concatenate((first, second), axis=-1)
    conditioning = mx.concatenate((model.normalize_latents(predicted), enc_out), axis=-1)
    save_npy("conditioning.npy", np.array(conditioning[0]))
    trace_files.append("conditioning.npy")
    traces["conditioning_shape"] = [int(d) for d in conditioning.shape]

    # Stage: one diffusion-head evaluation at timestep 500.
    rng2 = np.random.default_rng(1618)
    noise = rng2.standard_normal(conditioning.shape[:2] + (config.latent_dim * 2,)).astype(np.float32)
    save_npy("noise.npy", noise)
    trace_files.append("noise.npy")
    prediction = model.diffusion_head(mx.array(noise), mx.full((1,), 500.0, dtype=mx.float32), conditioning)
    save_npy("prediction.npy", np.array(prediction[0]))
    trace_files.append("prediction.npy")
    traces["prediction_shape"] = [int(d) for d in prediction.shape]

    # End-to-end: sample latents from seeded noise and decode.
    latents = model.sample_latents(conditioning, num_steps=8, initial_noise=mx.array(noise))
    save_npy("latents.npy", np.array(latents[0]))
    trace_files.append("latents.npy")
    speakers = model.decode_latents(latents)
    save_npy("speakers.npy", np.array(speakers[0]))
    trace_files.append("speakers.npy")
    traces["speakers_shape"] = [int(d) for d in speakers[0].shape]

    manifest = {
        "generator": "tools/gen_dialogue_sidon_fixtures.py",
        "reference_commit": "e1b19b9054bf163f5d812221a54fcc346f1890e9",
        "mlx_version": mx.__version__,
        "files": {name: digest(OUT / name) for name in sorted(set(trace_files))},
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(manifest['files']) + 1} fixture files to {OUT}")


if __name__ == "__main__":
    main()
