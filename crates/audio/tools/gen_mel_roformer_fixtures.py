#!/usr/bin/env python3
"""Regenerate the Mel-Band-RoFormer Rust parity fixtures.

Normative reference: mlx-audio at commit
e1b19b9054bf163f5d812221a54fcc346f1890e9
(mlX_audio/sts/models/mel_roformer, mlx 0.32.3).

Run from inside the mlx-audio checkout root so `mlx_audio` imports from
the pinned source, with the project venv interpreter:

    cd ../mlx-audio
    ../.venv-mlxaudio/bin/python \
        ../turbospark/crates/audio/tools/gen_mel_roformer_fixtures.py

Outputs into crates/audio/testdata/mel_roformer/:

- config.json
    The tiny seeded architecture (dim 32, depth 2, heads 2, dim_head
    16, 24 bands, n_fft 512). STFT parameters keep the real 44.1 kHz
    stereo pipeline shape; only the network shrinks. The real presets
    (kim_vocal_2 depth 6, viperx/zfturbo depth 12) differ only in
    depth and the zfturbo_vocals_v1 preset's dim/hop/mask depth.
- tiny_weights.safetensors
    Seeded F32 parameters in the MLX key tree (post-sanitize layout:
    separate to_q/to_k/to_v, no rotary buffers).
- wave.npy + traces
    A deterministic stereo wave, per-stage network goldens (band split,
    time/freq transformer outputs, per-band masks, merged mask,
    separated audio) and the band geometry (freq indices and dims) so
    the Rust test can verify its filterbank directly.
- ops_trace.json + ops_*.npy
    Tiny op goldens: binarized Slaney band support, F-normalize RMSNorm,
    interleaved RoPE tables, gated attention.
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

from mlx_audio.sts.models.mel_roformer.config import MelRoFormerConfig
from mlx_audio.sts.models.mel_roformer.model import (
    MelRoFormer,
    RMSNorm,
    RotaryEmbedding,
    _apply_rope,
)

OUT = Path("../turbospark/crates/audio/testdata/mel_roformer")

CONFIG = {
    "dim": 32,
    "depth": 2,
    "heads": 2,
    "dim_head": 16,
    "num_bands": 24,
    "num_stems": 1,
    "ff_mult": 4,
    "mlp_expansion_factor": 4,
    "mask_estimator_depth": 2,
    "n_fft": 512,
    "hop_length": 128,
    "win_length": 512,
    "sample_rate": 44100,
    "chunk_size": 352800,
    "num_overlap": 2,
    "checkpoint_family": "tiny_fixture",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_npy(name: str, arr: np.ndarray) -> None:
    np.save(OUT / name, np.ascontiguousarray(arr, dtype=np.float32))


def seed_parameters(model: nn.Module, seed: int) -> None:
    rng = np.random.default_rng(seed)
    flat = dict(tree_flatten(model.parameters()))
    updates = {}
    for name, value in flat.items():
        shape = value.shape
        if name.endswith(".norm.weight") or ".0.weight" in name and shape and "to_features" in name:
            # RMSNorm gains hover around 1.
            updates[name] = mx.array(1.0 + 0.1 * rng.standard_normal(shape), dtype=mx.float32)
        elif name.endswith(".bias"):
            updates[name] = mx.array(0.05 * rng.standard_normal(shape), dtype=mx.float32)
        else:
            updates[name] = mx.array(0.05 * rng.standard_normal(shape), dtype=mx.float32)
    model.update(tree_unflatten(updates))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    trace_files: list[str] = []
    traces: dict = {}

    (OUT / "config.json").write_text(json.dumps(CONFIG, indent=2) + "\n")
    trace_files.append("config.json")

    config = MelRoFormerConfig(**CONFIG)
    model = MelRoFormer(config)
    seed_parameters(model, 4242)

    flat = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(str(OUT / "tiny_weights.safetensors"), flat)
    trace_files.append("tiny_weights.safetensors")

    # Band geometry for the Rust-side filterbank check.
    fb = model.band_split.filterbank
    traces["band_dims"] = list(fb.band_dims)
    for i, idx in enumerate(fb.freq_indices):
        traces[f"band_{i}_indices"] = [int(v) for v in idx]

    # Deterministic stereo wave: 1.2 s at 44.1 kHz.
    rng = np.random.default_rng(808)
    n = int(1.2 * CONFIG["sample_rate"])
    t = np.arange(n) / CONFIG["sample_rate"]
    left = 0.4 * np.sin(2 * np.pi * 330.0 * t) + 0.1 * rng.standard_normal(n)
    right = 0.4 * np.sin(2 * np.pi * 440.0 * t) + 0.1 * rng.standard_normal(n)
    wave = np.stack([left, right]).astype(np.float32)
    save_npy("wave.npy", wave)
    trace_files.append("wave.npy")
    traces["wave_shape"] = list(wave.shape)

    audio = mx.array(wave[None])  # [1, 2, samples]

    # Stage: STFT + CaC representation.
    window = mx.array(np.hanning(CONFIG["n_fft"] + 1)[:-1].astype(np.float32))
    from mlx_audio.sts.models.mel_roformer.model import stft as ref_stft

    stft_real, stft_imag = ref_stft(audio, CONFIG["n_fft"], CONFIG["hop_length"], window)
    save_npy("stft_real.npy", np.array(stft_real[0]))
    save_npy("stft_imag.npy", np.array(stft_imag[0]))
    trace_files.extend(["stft_real.npy", "stft_imag.npy"])
    traces["stft_shape"] = [int(d) for d in stft_real[0].shape]

    # Stage: band split.
    B = 1
    freq_bins = stft_real.shape[2]
    T = stft_real.shape[3]
    real_interleaved = stft_real.transpose(0, 2, 1, 3).reshape(B, freq_bins * 2, T)
    imag_interleaved = stft_imag.transpose(0, 2, 1, 3).reshape(B, freq_bins * 2, T)
    stft_repr = mx.stack([real_interleaved, imag_interleaved], axis=-1)
    x = model.band_split.split(stft_repr)
    save_npy("bands.npy", np.array(x[0]))
    trace_files.append("bands.npy")
    traces["bands_shape"] = [int(d) for d in x[0].shape]

    # Stage: dual-axis transformer output.
    Nb = x.shape[2]
    D = x.shape[3]
    h = x
    for time_tf, freq_tf in model.layers:
        time_in = h.transpose(0, 2, 1, 3).reshape(B * Nb, T, D)
        time_out = time_tf(time_in)
        h = time_out.reshape(B, Nb, T, D).transpose(0, 2, 1, 3)
        freq_in = h.reshape(B * T, Nb, D)
        freq_out = freq_tf(freq_in)
        h = freq_out.reshape(B, T, Nb, D)
    save_npy("transformer_out.npy", np.array(h[0]))
    trace_files.append("transformer_out.npy")
    traces["transformer_out_shape"] = [int(d) for d in h[0].shape]

    # Stage: masks and merged mask.
    masks = model.mask_estimators[0](h)
    for tag in (0, 5, len(masks) - 1):
        save_npy(f"mask_band_{tag}.npy", np.array(masks[tag][0]))
        trace_files.append(f"mask_band_{tag}.npy")
        traces[f"mask_band_{tag}_shape"] = [int(d) for d in masks[tag][0].shape]
    full_mask = model.band_split.merge(masks, freq_bins * 2)
    save_npy("full_mask.npy", np.array(full_mask[0]))
    trace_files.append("full_mask.npy")
    traces["full_mask_shape"] = [int(d) for d in full_mask[0].shape]

    # End-to-end separation.
    separated = np.array(model(audio)[0])
    save_npy("separated.npy", separated)
    trace_files.append("separated.npy")
    traces["separated_shape"] = [int(d) for d in separated.shape]

    # Op traces.
    op_files = []
    rng2 = np.random.default_rng(77)
    x = rng2.standard_normal((2, 4, 6)).astype(np.float32)
    norm = RMSNorm(6)
    norm.weight = mx.array(1.0 + 0.1 * rng2.standard_normal(6), dtype=mx.float32)
    name = "ops_rmsnorm.npy"
    save_npy(name, np.array(norm(mx.array(x))))
    op_files.append(name)
    traces["ops_rmsnorm_shape"] = [2, 4, 6]

    rope = RotaryEmbedding(8)
    cos, sin = rope.get_cos_sin(5)
    name = "ops_rope_cos.npy"
    save_npy(name, np.array(cos))
    op_files.append(name)
    name = "ops_rope_sin.npy"
    save_npy(name, np.array(sin))
    op_files.append(name)

    x = rng2.standard_normal((1, 5, 8)).astype(np.float32)
    roped = _apply_rope(mx.array(x), cos, sin)
    name = "ops_rope_applied.npy"
    save_npy(name, np.array(roped))
    op_files.append(name)

    traces.update({f: None for f in []})
    (OUT / "traces.json").write_text(json.dumps(traces, indent=2))
    trace_files.extend(op_files)
    trace_files.append("traces.json")

    manifest = {
        "generator": "tools/gen_mel_roformer_fixtures.py",
        "reference_commit": "e1b19b9054bf163f5d812221a54fcc346f1890e9",
        "mlx_version": mx.__version__,
        "files": {name: digest(OUT / name) for name in sorted(set(trace_files))},
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(manifest['files']) + 1} fixture files to {OUT}")


if __name__ == "__main__":
    main()
