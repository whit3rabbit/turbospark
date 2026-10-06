#!/usr/bin/env python3
"""Regenerate the Mimi codec Rust parity fixtures.

Normative reference: mlx-audio at commit
e1b19b9054bf163f5d812221a54fcc346f1890e9
(mlx_audio/codec/models/mimi, mlx 0.32.3).

Run from inside the mlx-audio checkout root so `mlx_audio` imports from
the pinned source, with the project venv interpreter:

    cd ../mlx-audio
    ../.venv-mlxaudio/bin/python \
        ../turbospark/crates/audio/tools/gen_mimi_fixtures.py

Outputs into crates/audio/testdata/mimi/:

- tiny_config.json
    A tiny mimi_202407-style config (batch paths only).
- tiny_weights.safetensors
    Seeded parameters under the post-mapping MLX names (the names
    load_pytorch_weights produces from a PyTorch checkpoint).
- traces.json + npy files
    Encode codes (exact-index gates with half-norm distance margins
    for the first codebook), quantized latents, and decode waveforms
    for two input lengths, plus an isolated transformer trace.
- manifest.json
    File list with SHA-256 digests and generator metadata.

The script is deterministic: fixed seeds, no network access. Mimi's
batch encode/decode reset the streaming state, so the batch parity
here is exactly the reset-state streaming contract.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from mlx_audio.codec.models.mimi.mimi import Mimi, mimi_202407
from mlx_audio.codec.models.mimi.modules.conv import ConvTranspose1d
from mlx_audio.codec.models.mimi.modules.quantization import EuclideanCodebook

OUT = Path("../turbospark/crates/audio/testdata/mimi")


def load_and_refresh(model: Mimi, weights: dict) -> None:
    """load_weights plus the derived-state refresh load_pytorch_weights
    applies (EuclideanCodebook._embedding and the depthwise ConvTranspose
    expanded weight)."""
    model.load_weights(list(weights.items()))
    # Refresh ALL the derived state load_pytorch_weights refreshes: the
    # split-RVQ codebooks (fixed paths; the layers lists are not
    # nn.Module attributes) and EVERY ConvTranspose1d (each caches an
    # expanded weight at construction; stale ones decode with the
    # random init).
    for split in ("rvq_first", "rvq_rest"):
        vq = getattr(model.quantizer, split).vq
        for layer in vq.layers:
            layer.codebook.update_in_place()
    model.upsample.convtr.convtr.convtr.update_in_place()
    for layer in model.decoder.layers:
        layer.upsample.convtr.convtr.update_in_place()


def tiny_cfg():
    cfg = mimi_202407(8)
    cfg.seanet.dimension = 16
    cfg.seanet.nfilters = 8
    cfg.transformer.d_model = 16
    cfg.transformer.num_heads = 4
    cfg.transformer.dim_feedforward = 32
    cfg.transformer.context = 250
    cfg.quantizer_dim = 8
    return cfg


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_npy(name: str, arr: np.ndarray) -> None:
    np.save(OUT / name, np.ascontiguousarray(arr, dtype=np.float32))


def half_norm_margins(codebook: np.ndarray, encodings: np.ndarray) -> dict:
    """Best vs second-best (c2 - dot) per frame; c2 = ||emb||^2 / 2."""
    dots = encodings @ codebook.T  # (T, size)
    c2 = (codebook**2).sum(axis=1) / 2.0  # (size,)
    dist = c2[None, :] - dots
    best = []
    second = []
    for row in dist:
        order = np.argsort(row, kind="stable")
        best.append(float(row[order[0]]))
        second.append(float(row[order[1]]))
    return {"best": best, "second": second}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    traces: dict = {}
    trace_files = []

    mx.random.seed(46)
    cfg = tiny_cfg()
    model = Mimi(cfg)
    flat = dict(tree_flatten(model.parameters()))
    # Seed every parameter deterministically (the constructor drew from
    # the global chain already; redraw with a fixed generator so the
    # fixture is independent of mlx's RNG stream).
    rng = np.random.default_rng(41)
    seeded = {}
    for name, value in flat.items():
        seeded[name] = mx.array(
            (rng.standard_normal(value.shape) * 0.4).astype(np.float32)
        )
    load_and_refresh(model, seeded)
    mx.eval(model)

    mx.save_safetensors(str(OUT / "tiny_weights.safetensors"), seeded)
    cfg_json = {
        "channels": cfg.channels,
        "sample_rate": cfg.sample_rate,
        "frame_rate": cfg.frame_rate,
        "renormalize": cfg.renormalize,
        "seanet": {
            "dimension": cfg.seanet.dimension,
            "channels": cfg.seanet.channels,
            "causal": cfg.seanet.causal,
            "nfilters": cfg.seanet.nfilters,
            "nresidual_layers": cfg.seanet.nresidual_layers,
            "ratios": cfg.seanet.ratios,
            "ksize": cfg.seanet.ksize,
            "residual_ksize": cfg.seanet.residual_ksize,
            "last_ksize": cfg.seanet.last_ksize,
            "dilation_base": cfg.seanet.dilation_base,
            "pad_mode": cfg.seanet.pad_mode,
            "true_skip": cfg.seanet.true_skip,
            "compress": cfg.seanet.compress,
        },
        "transformer": {
            "d_model": cfg.transformer.d_model,
            "num_heads": cfg.transformer.num_heads,
            "num_layers": cfg.transformer.num_layers,
            "causal": cfg.transformer.causal,
            "norm_first": cfg.transformer.norm_first,
            "bias_ff": cfg.transformer.bias_ff,
            "bias_attn": cfg.transformer.bias_attn,
            "layer_scale": cfg.transformer.layer_scale,
            "positional_embedding": cfg.transformer.positional_embedding,
            "use_conv_bias": cfg.transformer.use_conv_bias,
            "gating": cfg.transformer.gating,
            "norm": cfg.transformer.norm,
            "context": cfg.transformer.context,
            "max_period": cfg.transformer.max_period,
            "max_seq_len": cfg.transformer.max_seq_len,
            "kv_repeat": cfg.transformer.kv_repeat,
            "dim_feedforward": cfg.transformer.dim_feedforward,
            "conv_layout": cfg.transformer.conv_layout,
            "use_conv_block": cfg.transformer.use_conv_block,
            "cross_attention": cfg.transformer.cross_attention,
            "conv_kernel_size": cfg.transformer.conv_kernel_size,
        },
        "quantizer_nq": cfg.quantizer_nq,
        "quantizer_bins": cfg.quantizer_bins,
        "quantizer_dim": cfg.quantizer_dim,
    }
    (OUT / "tiny_config.json").write_text(json.dumps(cfg_json, indent=2))

    rng = np.random.default_rng(47)
    # 1920 samples = one latent frame (hop 960 at frame_rate 2 in the
    # tiny config); use several frames so the gates are meaningful.
    cases = {
        "a": rng.standard_normal(1920 * 8).astype(np.float32),
        "b": rng.standard_normal(4000).astype(np.float32),
    }
    for tag, wave in cases.items():
        save_npy(f"{tag}_wave.npy", wave)
        trace_files.append(f"{tag}_wave.npy")
        audio = mx.array(wave)[None, None, :]  # (B, C, T) NCL contract
        codes = model.encode(audio)
        codes_np = np.asarray(codes)  # (B, nq, T)
        save_npy(f"{tag}_codes.npy", codes_np)
        traces[f"{tag}_codes_shape"] = list(codes_np.shape)
        # First-codebook margins against its codebook, over the same
        # quantizer input (encoder -> transformer -> downsample).
        emb = np.asarray(model.quantizer.rvq_first.vq.layers[0].codebook._embedding)
        model.encoder.reset_state()
        for c in model.encoder_cache:
            c.keys = None
            c.values = None
            c.offset = 0
        q_in = model.encoder(audio)
        q_in = model.encoder_transformer(q_in, cache=model.encoder_cache)[0]
        q_in = model.downsample(q_in)
        z = np.asarray(model.quantizer.rvq_first.input_proj(q_in))  # (B, T, dim)
        z = z.reshape(-1, emb.shape[-1])
        traces[f"{tag}_vq_margins"] = half_norm_margins(emb, z)
        gaps = [
            sec - best
            for best, sec in zip(
                traces[f"{tag}_vq_margins"]["best"],
                traces[f"{tag}_vq_margins"]["second"],
            )
        ]
        finite = [g for g in gaps if g == g]
        assert finite and min(finite) > 1e-6, (
            f"{tag} vq margin gap {min(gaps) if gaps else 'none'} "
            f"(nan count {sum(1 for g in gaps if g != g)}, z std "
            f"{float(np.std(z))})"
        )
        audio_out = model.decode(codes)
        save_npy(f"{tag}_audio.npy", np.asarray(audio_out))
        traces[f"{tag}_audio_len"] = int(audio_out.shape[-1])
        trace_files.extend([f"{tag}_codes.npy", f"{tag}_audio.npy"])

    # Isolated transformer trace (encoder transformer, no cache).
    cfg_t = tiny_cfg()
    model_t = Mimi(cfg_t)
    load_and_refresh(model_t, seeded)
    mx.eval(model_t)
    z = mx.array(rng.standard_normal((1, cfg_t.seanet.dimension, 24)).astype(np.float32))
    tf_out = model_t.encoder_transformer(z, cache=model_t.encoder_cache)[0]
    save_npy("tf_in.npy", np.asarray(z))
    save_npy("tf_out.npy", np.asarray(tf_out))
    trace_files.extend(["tf_in.npy", "tf_out.npy"])

    (OUT / "traces.json").write_text(json.dumps(traces))

    files = [
        "tiny_config.json",
        "tiny_weights.safetensors",
        "traces.json",
        *trace_files,
    ]
    manifest = {
        "generator": "tools/gen_mimi_fixtures.py",
        "reference_commit": "e1b19b9054bf163f5d812221a54fcc346f1890e9",
        "mlx_version": mx.__version__,
        "files": {name: digest(OUT / name) for name in files},
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(files) + 2} fixture files to {OUT}")


if __name__ == "__main__":
    main()
