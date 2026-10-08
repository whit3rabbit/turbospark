#!/usr/bin/env python3
"""Regenerate the MossFormer2 SE Rust parity fixtures.

Normative reference: mlx-audio at commit
e1b19b9054bf163f5d812221a54fcc346f1890e9
(mlX_audio/sts/models/mossformer2_se, mlx 0.32.3).

Run from inside the mlx-audio checkout root so `mlx_audio` imports from
the pinned source, with the project venv interpreter:

    cd ../mlx-audio
    ../.venv-mlxaudio/bin/python \
        ../turbospark/crates/audio/tools/gen_mossformer2_se_fixtures.py

Outputs into crates/audio/testdata/mossformer2_se/:

- config.json
    The real-model processing config (defaults of
    MossFormer2SEConfig) plus the tiny network geometry used for the
    stored weights. The real checkpoint keeps out_channels=512 and
    num_blocks=24; only out_channels and num_blocks shrink here. The
    attention geometry (group_size=256, query_key_dim=128) and the
    Gated FSMN geometry (inner_channels=256, lorder=20) are fixed by
    the reference constructors and match the real checkpoint.
- tiny_weights.safetensors
    Seeded F32 parameters named after the checkpoint tree minus the
    leading "model." component (the Rust loader strips that prefix
    from real checkpoints).
- wave.npy + stage traces
    A deterministic 1 s wave; the Kaldi fbank features (dither=0 --
    the reference default dither=1.0 is stochastic and is a documented
    divergence), delta and delta-delta features, the concatenated
    network input, per-stage network intermediates, the predicted
    mask, the STFT of the scaled wave, the masked spectrum, and the
    reconstructed enhanced audio.
- ops_trace.json + ops_*.npy
    Tiny op-level goldens (GLN, pytorch-compatible GroupNorm,
    CLayerNorm, ScaleNorm, ScaledSinuEmbedding, OffsetScale, MLX
    RoPE, FSMN depthwise conv, ConvModule depthwise conv, Kaldi
    deltas, ReLU^2 group attention).
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

from mlx_audio.dsp import (
    ISTFTCache,
    compute_deltas_kaldi,
    compute_fbank_kaldi,
    hamming,
    stft,
)
from mlx_audio.sts.models.mossformer2_se.gated_fsmn import Gated_FSMN
from mlx_audio.sts.models.mossformer2_se.gated_fsmn_block import (
    CLayerNorm,
    Gated_FSMN_Block,
)
from mlx_audio.sts.models.mossformer2_se.globallayernorm import GlobalLayerNorm
from mlx_audio.sts.models.mossformer2_se.mossformer2_se_wrapper import MossFormer2SE
from mlx_audio.sts.models.mossformer2_se.mossformer_masknet import MossFormer_MaskNet
from mlx_audio.sts.models.mossformer2_se.offsetscale import OffsetScale
from mlx_audio.sts.models.mossformer2_se.scalenorm import ScaleNorm
from mlx_audio.sts.models.mossformer2_se.scaledsinuembedding import (
    ScaledSinuEmbedding,
)
from mlx_audio.sts.models.mossformer2_se.unideepfsmn import UniDeepFsmn

OUT = Path("../turbospark/crates/audio/testdata/mossformer2_se")

CONFIG = {
    "sample_rate": 48000,
    "win_len": 1920,
    "win_inc": 384,
    "fft_len": 1920,
    "win_type": "hamming",
    "num_mels": 60,
    "preemphasis": 0.97,
    "one_time_decode_length": 20,
    "decode_window": 4,
    "chunk_seconds": 4.0,
    "chunk_overlap": 0.25,
    "auto_chunk_threshold": 60.0,
    "in_channels": 180,
    "out_channels": 64,
    "out_channels_final": 961,
    "num_blocks": 2,
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_npy(name: str, arr: np.ndarray) -> None:
    np.save(OUT / name, np.ascontiguousarray(arr, dtype=np.float32))


def seed_parameters(model: nn.Module, seed: int) -> None:
    """Overwrite every parameter with small seeded non-trivial values."""
    rng = np.random.default_rng(seed)
    flat = dict(tree_flatten(model.parameters()))
    updates = {}
    for name, value in flat.items():
        leaf = name.rsplit(".", 1)[-1]
        parent = name.rsplit(".", 1)[0].rsplit(".", 1)[-1]
        shape = value.shape
        if leaf == "inv_freq":
            continue
        if leaf == "scale":
            updates[name] = mx.array(np.full(shape, 0.7, dtype=np.float32))
        elif parent == "prelu":
            updates[name] = mx.array(np.full(shape, 0.21, dtype=np.float32))
        elif leaf == "gamma":
            updates[name] = mx.array(
                1.0 + 0.02 * rng.standard_normal(shape), dtype=mx.float32
            )
        elif leaf == "beta":
            updates[name] = mx.array(
                0.02 * rng.standard_normal(shape), dtype=mx.float32
            )
        elif leaf == "g":
            updates[name] = mx.array(np.full(shape, 1.3, dtype=np.float32))
        elif leaf == "weight" and parent.endswith("norm") or parent in (
            "norm1",
            "norm2",
            "intra_norm",
        ):
            updates[name] = mx.array(
                1.0 + 0.1 * rng.standard_normal(shape), dtype=mx.float32
            )
        elif leaf == "bias" and parent.endswith("norm") or parent in (
            "norm1",
            "norm2",
            "intra_norm",
        ):
            updates[name] = mx.array(
                0.05 * rng.standard_normal(shape), dtype=mx.float32
            )
        elif leaf == "weight":
            updates[name] = mx.array(
                0.05 * rng.standard_normal(shape), dtype=mx.float32
            )
        else:
            updates[name] = mx.array(0.05 * rng.standard_normal(shape), dtype=mx.float32)
    model.update(tree_unflatten(updates))


def masknet_forward_trace(masknet: MossFormer_MaskNet, features: np.ndarray) -> dict:
    """Step-by-step replay of MossFormer_MaskNet.__call__ (gln branch)."""
    out = {}
    x = mx.array(features)[None]  # [1, T, N]
    x = mx.transpose(x, (0, 2, 1))  # [1, N, T]
    x = masknet.norm(x)
    out["normed"] = np.array(x[0].T)
    x = mx.transpose(x, (0, 2, 1))
    x = masknet.conv1d_encoder(x)
    x = mx.transpose(x, (0, 2, 1))  # [1, C, T]
    out["encoded"] = np.array(x[0].T)

    base = x
    xt = mx.transpose(x, (0, 2, 1))  # [1, T, C]
    emb = masknet.pos_enc(xt)
    if len(emb.shape) == 2:
        emb = mx.broadcast_to(
            mx.expand_dims(emb, axis=0), (xt.shape[0], emb.shape[0], emb.shape[1])
        )
    emb = mx.transpose(emb, (0, 2, 1))  # [1, C, T]
    x = base + emb
    out["pos_added"] = np.array(x[0].T)

    intra = mx.transpose(x, (0, 2, 1))  # [1, T, C]
    intra_mdl = masknet.mdl.intra_mdl
    block = intra_mdl.mossformerM
    for i, (layer, fsmn) in enumerate(zip(block.layers, block.fsmn)):
        intra = layer(intra)
        if i == 0:
            out["flash0"] = np.array(intra[0])
        intra = fsmn(intra)
        if i == 0:
            out["gfsmn0"] = np.array(intra[0])
    intra = intra_mdl.norm(intra)
    out["intramdl_normed"] = np.array(intra[0])
    # Computation_Block: intra_norm in NLC then back, then skip add.
    intra = masknet.mdl.intra_norm(intra)
    intra = mx.transpose(intra, (0, 2, 1))
    intra = intra + x  # skip_around_intra: + block input [1, C, T]
    out["compblock_out"] = np.array(intra[0].T)

    x = intra
    x = masknet.prelu(x)
    out["prelu"] = np.array(x[0].T)
    x = mx.transpose(x, (0, 2, 1))
    x = masknet.conv1d_out(x)
    x = mx.transpose(x, (0, 2, 1))
    B, _, S = x.shape
    x = mx.reshape(x, (B * masknet.num_spks, -1, S))
    x = mx.transpose(x, (0, 2, 1))
    output_val = mx.tanh(masknet.output(x))
    gate_val = mx.sigmoid(masknet.output_gate(x))
    x = output_val * gate_val
    x = masknet.conv1_decoder(x)
    x = mx.transpose(x, (0, 2, 1))
    _, N, L = x.shape
    x = mx.reshape(x, (B, masknet.num_spks, N, L))
    x = nn.ReLU()(x)
    x = mx.transpose(x, (1, 0, 2, 3))
    result = mx.transpose(x[0], (0, 2, 1))  # [B, L, out_final]
    out["mask"] = np.array(result[0])
    return out


def op_traces() -> tuple[dict, list[str]]:
    """Tiny op-level goldens against the reference classes."""
    rng = np.random.default_rng(77)
    traces = {}
    files = []

    def record(tag: str, arr: np.ndarray) -> None:
        name = f"ops_{tag}.npy"
        save_npy(name, arr)
        files.append(name)
        traces[f"{tag}_shape"] = [int(d) for d in arr.shape]

    # GlobalLayerNorm over [B, C, L] with (dim, 1) affine, as stored.
    x = rng.standard_normal((2, 8, 6)).astype(np.float32)
    gln = GlobalLayerNorm(8, 3)
    gln.weight = mx.array(1.0 + 0.1 * rng.standard_normal((8, 1)), dtype=mx.float32)
    gln.bias = mx.array(0.05 * rng.standard_normal((8, 1)), dtype=mx.float32)
    record("gln", np.array(gln(mx.array(x))[0]))

    # PyTorch-compatible GroupNorm(1, C) on [B, T, C].
    x = rng.standard_normal((2, 5, 7)).astype(np.float32)
    gn = nn.GroupNorm(1, 7, eps=1e-8, affine=True, pytorch_compatible=True)
    gn.weight = mx.array(1.0 + 0.1 * rng.standard_normal(7), dtype=mx.float32)
    gn.bias = mx.array(0.05 * rng.standard_normal(7), dtype=mx.float32)
    record("groupnorm", np.array(gn(mx.array(x))))

    # CLayerNorm on [B, T, C].
    x = rng.standard_normal((2, 5, 7)).astype(np.float32)
    cln = CLayerNorm(7)
    cln.weight = mx.array(1.0 + 0.1 * rng.standard_normal(7), dtype=mx.float32)
    cln.bias = mx.array(0.05 * rng.standard_normal(7), dtype=mx.float32)
    record("clayernorm", np.array(cln(mx.array(x))))

    # ScaleNorm.
    x = rng.standard_normal((2, 5, 7)).astype(np.float32)
    sn = ScaleNorm(7)
    sn.g = mx.array(np.full((1,), 1.3, dtype=np.float32))
    record("scalenorm", np.array(sn(mx.array(x))))

    # ScaledSinuEmbedding.
    se = ScaledSinuEmbedding(8)
    se.scale = mx.array(np.full((1,), 0.7, dtype=np.float32))
    record("sinu", np.array(se(mx.zeros((1, 10, 8)))))

    # OffsetScale with three heads.
    x = rng.standard_normal((2, 4, 6)).astype(np.float32)
    os_ = OffsetScale(6, heads=3)
    os_.gamma = mx.array(1.0 + 0.02 * rng.standard_normal((3, 6)), dtype=mx.float32)
    os_.beta = mx.array(0.02 * rng.standard_normal((3, 6)), dtype=mx.float32)
    for h, head_out in enumerate(os_(mx.array(x))):
        record(f"offsetscale_h{h}", np.array(head_out))

    # MLX RoPE (half-split, positions along axis -2, first dims rotated).
    x = rng.standard_normal((1, 7, 12)).astype(np.float32)
    roped = mx.fast.rope(
        mx.array(x), 4, traditional=False, base=10000, scale=1.0, offset=0
    )
    record("rope", np.array(roped))

    # UniDeepFsmn depthwise conv over time (conv output before residual).
    fsmn = UniDeepFsmn(3, 3, lorder=3, hidden_size=4)
    fsmn.linear.weight = mx.array(0.05 * rng.standard_normal((4, 3)), dtype=mx.float32)
    fsmn.linear.bias = mx.array(0.05 * rng.standard_normal(4), dtype=mx.float32)
    fsmn.project.weight = mx.array(0.05 * rng.standard_normal((3, 4)), dtype=mx.float32)
    fsmn.conv1.weight = mx.array(0.05 * rng.standard_normal((3, 5, 1, 1)), dtype=mx.float32)
    x = rng.standard_normal((1, 10, 3)).astype(np.float32)
    f1 = mx.maximum(fsmn.linear(mx.array(x)), 0)
    p1 = fsmn.project(f1)
    p1w = mx.expand_dims(p1, axis=2)  # [B, T, 1, C]
    padded = mx.pad(p1w, [(0, 0), (2, 2), (0, 0), (0, 0)])
    conv_out = fsmn.conv1(padded)
    record("fsmn_conv", np.array(conv_out[0]))
    record("fsmn_module", np.array(fsmn(mx.array(x))[0]))

    # ConvModule depthwise conv1d (conv output before residual add).
    x = rng.standard_normal((1, 9, 3)).astype(np.float32)
    weight = mx.array(0.05 * rng.standard_normal((3, 17, 1)), dtype=mx.float32)
    conv_out = mx.conv1d(mx.array(x), weight, stride=1, padding=8, groups=3)
    record("dw_conv1d", np.array(conv_out[0]))

    # Kaldi deltas.
    x = rng.standard_normal((3, 9)).astype(np.float32)
    record("kaldi_delta", np.array(compute_deltas_kaldi(mx.array(x), win_length=5)))

    # ReLU^2 group attention (FlashAttentionImplementations.standard).
    quad_q = mx.array(rng.standard_normal((1, 2, 4, 8)).astype(np.float32))
    quad_k = mx.array(rng.standard_normal((1, 2, 4, 8)).astype(np.float32))
    v = mx.array(rng.standard_normal((1, 2, 4, 5)).astype(np.float32))
    g = 4
    sim = mx.matmul(quad_q, mx.transpose(mx.array(quad_k), [0, 1, 3, 2])) * (1.0 / g)
    attn = mx.maximum(sim, 0)
    attn = attn * attn
    quad_out = mx.matmul(attn, mx.array(v))
    record("relu2_attention", np.array(quad_out[0]))

    return traces, files


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    trace_files: list[str] = []
    traces: dict = {}

    (OUT / "config.json").write_text(json.dumps(CONFIG, indent=2) + "\n")
    trace_files.append("config.json")

    model = MossFormer_MaskNet(
        in_channels=CONFIG["in_channels"],
        out_channels=CONFIG["out_channels"],
        out_channels_final=CONFIG["out_channels_final"],
        num_blocks=CONFIG["num_blocks"],
    )
    masknet = model
    seed_parameters(masknet, 1234)

    flat = dict(tree_flatten(masknet.parameters()))
    saved = {}
    for name, value in flat.items():
        # Skip the computed inv_freq buffer; the Rust port recomputes it.
        if name.endswith("pos_enc.inv_freq"):
            continue
        # Prefix with the MaskNet attribute name so keys mirror the real
        # checkpoint tree minus its leading "model." component.
        saved[f"mossformer.{name}"] = value
    mx.save_safetensors(str(OUT / "tiny_weights.safetensors"), saved)
    trace_files.append("tiny_weights.safetensors")

    # Deterministic wave: two sines plus seeded noise, 1 s at 48 kHz.
    rng = np.random.default_rng(2024)
    wave = (
        0.4 * np.sin(2 * np.pi * 220.0 * np.arange(48000) / 48000.0)
        + 0.2 * np.sin(2 * np.pi * 1400.0 * np.arange(48000) / 48000.0)
        + 0.05 * rng.standard_normal(48000)
    ).astype(np.float32)
    save_npy("wave.npy", wave)
    trace_files.append("wave.npy")
    traces["wave_len"] = int(wave.shape[0])

    scaled = wave * 32768.0

    fbanks = np.array(
        compute_fbank_kaldi(
            mx.array(scaled),
            sample_rate=CONFIG["sample_rate"],
            win_len=CONFIG["win_len"],
            win_inc=CONFIG["win_inc"],
            num_mels=CONFIG["num_mels"],
            win_type=CONFIG["win_type"],
            preemphasis=CONFIG["preemphasis"],
            dither=0.0,
        )
    )
    save_npy("fbank.npy", fbanks)
    trace_files.append("fbank.npy")
    traces["fbank_shape"] = list(fbanks.shape)

    fbank_t = mx.transpose(mx.array(fbanks), [1, 0])
    delta = np.array(compute_deltas_kaldi(fbank_t, win_length=5)).T
    ddelta = np.array(compute_deltas_kaldi(mx.transpose(mx.array(delta), [1, 0]), win_length=5)).T
    save_npy("delta.npy", delta)
    save_npy("ddelta.npy", ddelta)
    trace_files.extend(["delta.npy", "ddelta.npy"])

    features = np.concatenate([fbanks, delta, ddelta], axis=1)
    save_npy("features.npy", features)
    trace_files.append("features.npy")
    traces["features_shape"] = list(features.shape)

    stages = masknet_forward_trace(masknet, features)
    for tag, arr in stages.items():
        save_npy(f"net_{tag}.npy", arr)
        trace_files.append(f"net_{tag}.npy")
        traces[f"net_{tag}_shape"] = list(arr.shape)

    window = hamming(CONFIG["win_len"], periodic=False)
    stft_complex = stft(
        mx.array(scaled),
        CONFIG["fft_len"],
        CONFIG["win_inc"],
        CONFIG["win_len"],
        window,
        center=False,
    )
    stft_real = np.array(mx.real(stft_complex))
    stft_imag = np.array(mx.imag(stft_complex))
    save_npy("stft_real.npy", stft_real)
    save_npy("stft_imag.npy", stft_imag)
    trace_files.extend(["stft_real.npy", "stft_imag.npy"])
    traces["stft_shape"] = list(stft_real.shape)

    mask = stages["mask"]  # [T, out_final]
    real_part = stft_real.T  # [freq, T]
    imag_part = stft_imag.T
    pred_mask = mask.T[:, :, None]  # [freq, T, 1]
    spectrum_real = real_part * pred_mask[:, :, 0]
    spectrum_imag = imag_part * pred_mask[:, :, 0]
    save_npy("masked_real.npy", spectrum_real.T)
    save_npy("masked_imag.npy", spectrum_imag.T)
    trace_files.extend(["masked_real.npy", "masked_imag.npy"])

    output = ISTFTCache().istft(
        mx.array(spectrum_real[None]),
        mx.array(spectrum_imag[None]),
        CONFIG["fft_len"],
        CONFIG["win_inc"],
        CONFIG["win_len"],
        window,
        center=False,
        audio_length=int(scaled.shape[0]),
    )
    enhanced = np.array(output)[0]
    save_npy("enhanced.npy", enhanced)
    trace_files.append("enhanced.npy")
    traces["enhanced_len"] = int(enhanced.shape[0])

    ops, op_files = op_traces()
    traces.update(ops)
    trace_files.extend(op_files)

    (OUT / "traces.json").write_text(json.dumps(traces, indent=2))
    trace_files.append("traces.json")

    manifest = {
        "generator": "tools/gen_mossformer2_se_fixtures.py",
        "reference_commit": "e1b19b9054bf163f5d812221a54fcc346f1890e9",
        "mlx_version": mx.__version__,
        "dither": "fbank features generated with dither=0; the reference "
        "default dither=1.0 is stochastic and is a documented divergence",
        "files": {name: digest(OUT / name) for name in sorted(set(trace_files))},
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(manifest['files']) + 1} fixture files to {OUT}")


if __name__ == "__main__":
    main()
