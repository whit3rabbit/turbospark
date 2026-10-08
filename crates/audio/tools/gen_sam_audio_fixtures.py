#!/usr/bin/env python3
"""Regenerate the SAM-Audio Rust parity fixtures.

Normative reference: mlx-audio at commit
e1b19b9054bf163f5d812221a54fcc346f1890e9
(mlX_audio/sts/models/sam_audio, mlx 0.32.3).

Run from inside the mlx-audio checkout root so `mlx_audio` imports from
the pinned source, with the project venv interpreter:

    cd ../mlx-audio
    ../.venv-mlxaudio/bin/python \
        ../turbospark/crates/audio/tools/gen_sam_audio_fixtures.py

Outputs into crates/audio/testdata/sam_audio/:

- config.json
    Tiny seeded architecture: a small DACVAE (codebook_dim 8, so
    in_channels 48 / out_channels 16), a 2-layer T5 encoder, and a
    2-layer DiT. The real presets scale dims/layers only
    (sam-audio-large: DiT dim 2816, 22 heads, 22 layers, t5-base).
- tiny_weights.safetensors
    Seeded F32 parameters for the full SAMAudio tree (DACVAE codec,
    T5, DiT, projections, anchor embedding).
- tokens.npy / token_mask.npy, wave.npy
    Deterministic inputs: T5 token ids with a padded row, and a mono
    waveform for the codec.
- stage traces
    text_features (T5 out), audio_features (codec codebook means,
    doubled for target+residual), velocity (one DiT call at t=0.25),
    velocity_with_anchors (EmbedAnchors path), noisy_after_step (one
    midpoint ODE step), and the full separate() output with seeded
    noise: target.npy / residual.npy.
- ops_trace.json + ops_*.npy
    T5 relative-position buckets, DiT RoPE cos/sin, RMSNorm.
- manifest.json
    File list with SHA-256 digests and generator metadata.

The script is deterministic: fixed seeds, no network access. The T5
tokenizer is not exercised (the Rust port takes token ids; tokenization
is a runtime/catalog concern, matching the reference where the
checkpoint ships no T5 weights and t5-base comes from HuggingFace).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

from mlx_audio.codec.models.dacvae.codec import DACVAEConfig
from mlx_audio.sts.models.sam_audio.config import (
    SAMAudioConfig,
    T5EncoderConfig,
    TransformerConfig,
)
from mlx_audio.sts.models.sam_audio.model import SAMAudio
from mlx_audio.sts.models.sam_audio.rope import RotaryEmbedding
from mlx_audio.sts.models.sam_audio.text_encoder import (
    T5Attention,
    T5Config,
    T5Encoder,
)
from mlx_audio.sts.models.sam_audio.transformer import RMSNorm

OUT = Path("../turbospark/crates/audio/testdata/sam_audio")

TINY_DACVAE = dict(
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

CONFIG = {
    "in_channels": 48,
    "audio_codec": TINY_DACVAE,
    "text_encoder": {"name": "t5-tiny-fixture", "max_length": 64, "pad_mode": "longest", "dim": 32},
    # Fixture-only T5 architecture fields; the reference ignores this
    # key (real checkpoints use t5-base defaults from HuggingFace).
    "t5_arch": {
        "vocab_size": 64, "d_model": 32, "d_kv": 16, "d_ff": 64,
        "num_layers": 2, "num_heads": 2,
        "relative_attention_num_buckets": 32, "relative_attention_max_distance": 128,
        "is_gated_act": True, "layer_norm_epsilon": 1e-6
    },
    "transformer": {
        "dim": 64,
        "n_heads": 4,
        "n_layers": 2,
        "dropout": 0.1,
        "norm_eps": 1e-5,
        "qk_norm": True,
        "fc_bias": False,
        "ffn_exp": 4,
        "ffn_dim_multiplier": 1,
        "multiple_of": 8,
        "non_linearity": "swiglu",
        "use_rope": True,
        "max_positions": 1024,
        "frequency_embedding_dim": 32,
        "timestep_non_linearity": "swiglu",
        "t_block_non_linearity": "silu",
        "t_block_bias": True,
        "context_dim": 64,
        "context_non_linearity": "swiglu",
        "context_embedder_dropout": 0.0,
        "context_norm": False,
        "out_channels": 16,
        "in_channels": None,
    },
    "num_anchors": 3,
    "anchor_embedding_dim": 8,
}

T5_TINY = dict(
    vocab_size=64,
    d_model=32,
    d_kv=16,
    d_ff=64,
    num_layers=2,
    num_heads=2,
    relative_attention_num_buckets=32,
    relative_attention_max_distance=128,
    dropout_rate=0.1,
    layer_norm_epsilon=1e-6,
    is_gated_act=True,
    dense_act_fn="gelu_new",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_npy(name: str, arr: np.ndarray) -> None:
    np.save(OUT / name, np.ascontiguousarray(arr, dtype=np.float32))


def seed_parameters(model: nn.Module, seed: int, scale: float = 0.05) -> None:
    rng = np.random.default_rng(seed)
    flat = dict(tree_flatten(model.parameters()))
    updates = {}
    for name, value in flat.items():
        if name.endswith("gate") and value.ndim == 0:
            # EmbedAnchors gate starts at 0; make it nonzero for parity.
            updates[name] = mx.array(np.array(0.3, dtype=np.float32))
        elif name.endswith("_inv_freq"):
            continue
        else:
            updates[name] = mx.array(
                (scale * rng.standard_normal(value.shape)).astype(np.float32)
            )
    model.update(tree_unflatten(updates))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    trace_files: list[str] = []
    traces: dict = {}

    (OUT / "config.json").write_text(json.dumps(CONFIG, indent=2) + "\n")
    trace_files.append("config.json")

    config = SAMAudioConfig.from_dict(CONFIG)
    model = SAMAudio(config)
    # T5TextEncoder is a plain Python class, so its model is NOT part of
    # the SAMAudio parameter tree; seed and save it separately under the
    # text_encoder.model.* key prefix.
    t5 = T5Encoder(T5Config(**T5_TINY))
    seed_parameters(t5, 555)
    t5.eval()
    model.text_encoder.model = t5
    seed_parameters(model, 99)
    model.eval()

    flat = dict(tree_flatten(model.parameters()))
    saved = {
        k: v
        for k, v in flat.items()
        if not k.endswith("_inv_freq") and not k.startswith("audio_codec.")
    }
    for k, v in tree_flatten(t5.parameters()):
        saved[f"text_encoder.model.{k}"] = v
    mx.save_safetensors(str(OUT / "tiny_weights.safetensors"), saved)
    trace_files.append("tiny_weights.safetensors")
    # The DACVAE codec subtree ships separately with bare keys so the
    # existing crate Dacvae loader consumes it directly. The real
    # mlx-community checkpoint keeps one file with an audio_codec.
    # prefix; splitting that out is a runtime-layer concern.
    codec_saved = {
        k[len("audio_codec.") :]: v
        for k, v in flat.items()
        if k.startswith("audio_codec.")
    }
    mx.save_safetensors(str(OUT / "tiny_codec.safetensors"), codec_saved)
    trace_files.append("tiny_codec.safetensors")

    # Inputs: T5 tokens with a padded row, mono wave.
    rng = np.random.default_rng(1234)
    # Batch 1: the audio batch is 1 and the reference requires matching
    # text/audio batch sizes.
    tokens = rng.integers(4, 60, size=(1, 8)).astype(np.int32)
    mask = np.ones((1, 8), dtype=np.float32)
    np.save(OUT / "tokens.npy", tokens)
    np.save(OUT / "token_mask.npy", mask)
    trace_files.extend(["tokens.npy", "token_mask.npy"])

    wave = (
        0.4 * np.sin(2 * np.pi * 300.0 * np.arange(4000) / 24000.0)
        + 0.05 * rng.standard_normal(4000)
    ).astype(np.float32)
    save_npy("wave.npy", wave)
    trace_files.append("wave.npy")

    # Stage: T5 text encoding.
    text_features = model.text_encoder.model(
        input_ids=mx.array(tokens), attention_mask=mx.array(mask)
    )
    save_npy("text_features.npy", np.array(text_features))
    trace_files.append("text_features.npy")
    traces["text_features_shape"] = [int(d) for d in text_features.shape]

    # Stage: codec features (codebook means, doubled).
    audios = mx.array(wave[None, None])  # [1, 1, T]
    audio_features = model._get_audio_features(audios)
    save_npy("audio_features.npy", np.array(audio_features))
    trace_files.append("audio_features.npy")
    traces["audio_features_shape"] = [int(d) for d in audio_features.shape]

    # Stage: one DiT velocity evaluation at t=0.25.
    t_frames = audio_features.shape[1]
    rng2 = np.random.default_rng(77)
    noise = rng2.standard_normal(audio_features.shape).astype(np.float32)
    save_npy("noise.npy", noise)
    trace_files.append("noise.npy")
    noisy = mx.array(noise)
    velocity = model(
        noisy_audio=noisy,
        audio_features=audio_features,
        text_features=text_features,
        time=mx.full((1,), 0.25, dtype=mx.float32),
        text_mask=mx.array(mask).astype(mx.bool_),
    )
    save_npy("velocity.npy", np.array(velocity))
    trace_files.append("velocity.npy")
    traces["velocity_shape"] = [int(d) for d in velocity.shape]

    # Finer traces for bisection: aligned/proj inputs and memory.
    aligned = model.align_inputs(noisy, audio_features)
    save_npy("aligned.npy", np.array(aligned))
    trace_files.append("aligned.npy")
    timestep_emb = model.timestep_emb(mx.full((1,), 0.25, dtype=mx.float32), pos=mx.full((1,), 0.25, dtype=mx.float32))
    memory = model.memory_proj(text_features) + mx.expand_dims(timestep_emb, 1)
    save_npy("memory.npy", np.array(memory))
    trace_files.append("memory.npy")
    dit = model.transformer
    t_freq_in = dit.t_embedder._timestep_embedding(mx.array(np.array([0.25], dtype=np.float32)), 32)
    t_e = dit.t_embedder.projection(t_freq_in)
    t0 = dit.t_block(nn.silu(t_e))
    save_npy("t0.npy", np.array(t0))
    trace_files.append("t0.npy")

    # Stage: velocity with temporal anchors.
    anchor_ids = mx.array(np.array([[1, 0, 2]], dtype=np.int32))
    alignment = np.zeros((1, t_frames), dtype=np.int32)
    alignment[0, t_frames // 2 :] = 2
    alignment[0, : t_frames // 4] = 1
    anchor_alignment = mx.array(alignment)
    velocity_anchors = model(
        noisy_audio=noisy,
        audio_features=audio_features,
        text_features=text_features,
        time=mx.full((1,), 0.25, dtype=mx.float32),
        text_mask=mx.array(mask).astype(mx.bool_),
        anchor_ids=anchor_ids,
        anchor_alignment=anchor_alignment,
    )
    save_npy("velocity_with_anchors.npy", np.array(velocity_anchors))
    trace_files.append("velocity_with_anchors.npy")
    save_npy("anchor_alignment.npy", np.array(anchor_alignment).astype(np.float32))
    trace_files.append("anchor_alignment.npy")
    np.save(OUT / "anchor_ids.npy", np.array(anchor_ids))
    trace_files.append("anchor_ids.npy")

    # Stage: one midpoint ODE step.
    after_step = model._ode_step_midpoint(
        t=0.0,
        dt=0.125,
        noisy_audio=noisy,
        audio_features=audio_features,
        text_features=text_features,
        text_mask=mx.array(mask).astype(mx.bool_),
        anchor_ids=None,
        anchor_alignment=None,
        audio_pad_mask=None,
    )
    save_npy("noisy_after_step.npy", np.array(after_step))
    trace_files.append("noisy_after_step.npy")

    # End-to-end separate with seeded noise and deterministic ODE.
    result = model.separate(
        audios=audios,
        descriptions=["ignored; tokens come from the fixture"],
        noise=noisy,
        ode_opt={"method": "midpoint", "step_size": 2 / 16},
        _text_features=text_features,
        _text_mask=mx.array(mask).astype(mx.bool_),
    )
    target = np.array(result.target[0]).reshape(-1)
    residual = np.array(result.residual[0]).reshape(-1)
    save_npy("target.npy", target)
    save_npy("residual.npy", residual)
    trace_files.extend(["target.npy", "residual.npy"])
    traces["target_shape"] = [int(d) for d in target.shape]

    # Op traces.
    t5cfg = T5Config(**T5_TINY)
    attn = T5Attention(t5cfg, has_relative_attention_bias=True)
    attn.relative_attention_bias.weight = mx.array(
        (0.05 * np.random.default_rng(5).standard_normal((32, 2))).astype(np.float32)
    )
    bias = attn.compute_bias(8, 8)
    save_npy("ops_position_bias.npy", np.array(bias[0]))
    trace_files.append("ops_position_bias.npy")

    rope = RotaryEmbedding(theta=20000.0, head_dim=16, max_seqlen=16)
    x = mx.array(np.random.default_rng(6).standard_normal((1, 16, 2, 16)).astype(np.float32))
    roped = rope(x)
    save_npy("ops_rope.npy", np.array(roped))
    trace_files.append("ops_rope.npy")

    norm = RMSNorm(8)
    norm.weight = mx.array(
        (1.0 + 0.1 * np.random.default_rng(7).standard_normal(8)).astype(np.float32)
    )
    xn = mx.array(np.random.default_rng(8).standard_normal((2, 5, 8)).astype(np.float32))
    save_npy("ops_rmsnorm.npy", np.array(norm(xn)))
    trace_files.append("ops_rmsnorm.npy")

    (OUT / "traces.json").write_text(json.dumps(traces, indent=2))
    trace_files.append("traces.json")

    manifest = {
        "generator": "tools/gen_sam_audio_fixtures.py",
        "reference_commit": "e1b19b9054bf163f5d812221a54fcc346f1890e9",
        "mlx_version": mx.__version__,
        "files": {name: digest(OUT / name) for name in sorted(set(trace_files))},
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(manifest['files']) + 1} fixture files to {OUT}")


if __name__ == "__main__":
    main()
