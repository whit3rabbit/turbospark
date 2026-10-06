#!/usr/bin/env python3
"""Regenerate the Nemotron VoiceChat codec Rust parity fixtures.

Normative reference: mlx-audio at commit
e1b19b9054bf163f5d812221a54fcc346f1890e9
(mlX_audio/codec/models/nemotron_voicechat, mlx 0.32.3).

Run from inside the mlx-audio checkout root so `mlx_audio` imports from
the pinned source, with the project venv interpreter:

    cd ../mlx-audio
    ../.venv-mlxaudio/bin/python \
        ../turbospark/crates/audio/tools/gen_nemotron_voicechat_fixtures.py

Outputs into crates/audio/testdata/nemotron_voicechat/:

- tiny_config.json + tiny_weights.safetensors
    Seeded tiny codec (encoder, decoder, PRVQ means and variances).
- traces.json + npy files
    Codes (with quantization margins), latents, and decoded waveforms
    for two input lengths plus a random-codes decode.
- manifest.json
    File list with SHA-256 digests and generator metadata.

The script is deterministic: fixed seeds, no network access. The
streaming cache path is not part of the batch contract and is not
exercised.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import fields
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from mlx_audio.codec.models.nemotron_voicechat.codec import NemotronVoiceChatCodec
from mlx_audio.codec.models.nemotron_voicechat.config import (
    NemotronVoiceChatCodecConfig,
)

OUT = Path("../turbospark/crates/audio/testdata/nemotron_voicechat")
REFERENCE_COMMIT = "e1b19b9054bf163f5d812221a54fcc346f1890e9"

TINY = dict(
    sample_rate=22050,
    base_channels=8,
    channel_multipliers=(1, 2),
    downsample_rates=(4, 5),
    blocks_per_stage=2,
    block_kernel_size=5,
    latent_dim=16,
    n_fft=16,
    hop_length=4,
    num_quantizers=5,
    codebook_size=32,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_npy(name: str, arr: np.ndarray) -> None:
    np.save(OUT / name, np.ascontiguousarray(arr, dtype=np.float32))


def prvq_margins(codec: NemotronVoiceChatCodec, latents: np.ndarray) -> float:
    """Smallest best-vs-second distance margin across quantizers."""
    residual = latents.copy()
    min_margin = np.inf
    for means in codec.prvq.mus_list:
        m = np.asarray(means, dtype=np.float32)
        d = (
            (residual**2).sum(axis=-1, keepdims=True)
            - 2.0 * residual @ m.T
            + (m**2).sum(axis=-1)[None, None, :]
        )
        order = np.argsort(d, axis=-1, kind="stable")
        best = np.take_along_axis(d, order[..., :1], axis=-1)
        second = np.take_along_axis(d, order[..., 1:2], axis=-1)
        min_margin = min(min_margin, float((second - best).min()))
        idx = np.argmin(d, axis=-1)
        residual = residual - m[idx]
    return min_margin


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    traces: dict = {}
    trace_files: list[str] = []

    cfg = NemotronVoiceChatCodecConfig(**TINY)
    mx.random.seed(211)
    codec = NemotronVoiceChatCodec(cfg)
    flat = dict(tree_flatten(codec.parameters()))
    rng = np.random.default_rng(213)
    seeded = {
        name: mx.array((rng.standard_normal(v.shape) * 0.3).astype(np.float32))
        for name, v in flat.items()
    }
    codec.load_weights(list(seeded.items()))
    mx.eval(codec)
    mx.save_safetensors(str(OUT / "tiny_weights.safetensors"), seeded)
    (OUT / "tiny_config.json").write_text(
        json.dumps({f.name: list(getattr(cfg, f.name)) if isinstance(getattr(cfg, f.name), tuple) else getattr(cfg, f.name) for f in fields(cfg)}, indent=2)
    )
    (OUT / "weights_manifest.json").write_text(
        json.dumps(
            {"tiny": [{"name": n, "shape": list(v.shape)} for n, v in sorted(flat.items())]},
            indent=2,
        )
    )

    wave_rng = np.random.default_rng(217)
    for tag, n in (("a", 240), ("b", 200)):
        wave = wave_rng.standard_normal(n).astype(np.float32) * 0.2
        save_npy(f"{tag}_wave.npy", wave)
        trace_files.append(f"{tag}_wave.npy")
        latents = np.asarray(codec.encode_latents(mx.array(wave)[None]))[0]
        save_npy(f"{tag}_latents.npy", latents)
        trace_files.append(f"{tag}_latents.npy")
        codes = np.asarray(codec.encode(mx.array(wave)[None]))[0].astype(np.int64)
        np.save(OUT / f"{tag}_codes.npy", np.ascontiguousarray(codes, dtype=np.int32))
        traces[f"{tag}_input_len"] = n
        traces[f"{tag}_codes_shape"] = list(codes.shape)
        traces[f"{tag}_min_margin"] = prvq_margins(codec, latents[None])
        decoded = np.asarray(codec.decode(mx.array(codes)[None].astype(mx.int32)))[0, 0]
        save_npy(f"{tag}_decoded.npy", decoded)
        trace_files.extend([f"{tag}_codes.npy", f"{tag}_decoded.npy"])

    # Random-codes decode: exercises the decoder without the encoder.
    rng = np.random.default_rng(219)
    codes = rng.integers(0, TINY["codebook_size"], size=(TINY["num_quantizers"], 3)).astype(np.int32)
    np.save(OUT / "rand_codes.npy", np.ascontiguousarray(codes))
    decoded = np.asarray(codec.decode(mx.array(codes)[None]))[0, 0]
    save_npy("rand_decoded.npy", decoded)
    trace_files.extend(["rand_codes.npy", "rand_decoded.npy"])
    traces["rand_codes_shape"] = list(codes.shape)
    traces["rand_decoded_len"] = int(decoded.shape[0])

    (OUT / "traces.json").write_text(json.dumps(traces, indent=2))

    files = [
        "tiny_config.json",
        "tiny_weights.safetensors",
        "weights_manifest.json",
        "traces.json",
        *trace_files,
    ]
    manifest = {
        "generator": "tools/gen_nemotron_voicechat_fixtures.py",
        "reference_commit": REFERENCE_COMMIT,
        "mlx_version": mx.__version__,
        "files": {name: digest(OUT / name) for name in files},
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(files) + 1} fixture files to {OUT}")


if __name__ == "__main__":
    main()
