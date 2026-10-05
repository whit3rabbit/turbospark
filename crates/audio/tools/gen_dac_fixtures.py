#!/usr/bin/env python3
"""Regenerate the Descript Audio Codec Rust parity fixtures.

Normative reference: mlx-audio at commit
e1b19b9054bf163f5d812221a54fcc346f1890e9
(mlx_audio/codec/models/descript, mlx 0.32.3).

Run from inside the mlx-audio checkout root so `mlx_audio` imports from
the pinned source, with the project venv interpreter:

    cd ../mlx-audio
    ../.venv-mlxaudio/bin/python \
        ../turbospark/crates/audio/tools/gen_dac_fixtures.py

Outputs into crates/audio/testdata/descript/:

- tiny_config.json + tiny_weights.safetensors
    Seeded tiny DAC (dense convs, three 64x4 codebooks, hop 8).
- traces.json
    wave_a (one pad block) and wave_b (shorter, exercises the
    preprocess right-pad path) with encoder z, quantizer z_q and codes
    (exact-index gates with level-0 distance margins), decoder audio,
    the from_codes decode path, the compress/decompress gain round
    trip, and the recorded model delay.
- weights_manifest.json
    Flattened parameter names and shapes.
- manifest.json
    File list with SHA-256 digests and generator metadata.

The script is deterministic: fixed seeds, no network access.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from mlx_audio.codec.models.descript.dac import DAC

OUT = Path("../turbospark/crates/audio/testdata/descript")
TINY = dict(
    encoder_dim=8,
    encoder_rates=[2, 4],
    latent_dim=None,
    decoder_dim=16,
    decoder_rates=[4, 2],
    n_codebooks=3,
    codebook_size=64,
    codebook_dim=4,
    sample_rate=44100,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_npy(name: str, arr: np.ndarray) -> None:
    np.save(OUT / name, np.ascontiguousarray(arr, dtype=np.float32))


def floats(arr) -> list:
    return np.asarray(arr, dtype=np.float32).ravel().tolist()


def vq_margins(dac: DAC, z: mx.array) -> dict:
    """Level-0 best vs second-best squared distances per frame."""
    quantizer = dac.quantizer.quantizers[0]
    z_e = quantizer.in_proj(z.moveaxis(1, 2)).moveaxis(1, 2)
    b, d, t = z_e.shape
    encodings = np.asarray(z_e.transpose(0, 2, 1).reshape(b * t, d))
    codebook = np.asarray(quantizer.codebook.weight)

    def sqnorm(rows):
        return rows / np.maximum(
            np.sqrt((np.abs(rows) ** 2).sum(axis=1, keepdims=True)), 1e-12
        )

    encodings = sqnorm(encodings)
    codebook = sqnorm(codebook)
    dist = (
        (encodings**2).sum(axis=1, keepdims=True)
        - 2.0 * encodings @ codebook.T
        + (codebook**2).sum(axis=1)[None, :]
    )
    best = []
    second = []
    for row in dist:
        order = np.argsort(row, kind="stable")
        best.append(float(row[order[0]]))
        second.append(float(row[order[1]]))
    return {"best": best, "second": second}


def build_case(dac: DAC, tag: str, wave: np.ndarray, traces: dict) -> None:
    audio = mx.array(wave[None, None, :])
    traces[f"{tag}_wave"] = floats(wave)
    traces[f"{tag}_input_len"] = int(wave.shape[0])
    traces[f"{tag}_padded_len"] = int(dac.preprocess(audio, None).shape[-1])

    z, codes, latents, _, _ = dac.encode(dac.preprocess(audio, None))
    # codes is (n_codebooks, T) or (B, n_codebooks, T); keep the
    # original array for from_codes and record one flat list per level.
    codes_np = np.asarray(codes)
    if codes_np.ndim == 3:
        codes_np = codes_np[0]
    traces[f"{tag}_codes"] = [
        codes_np[i].ravel().tolist() for i in range(codes_np.shape[0])
    ]
    save_npy(f"{tag}_zq.npy", np.asarray(z))
    traces[f"{tag}_vq_margins"] = vq_margins(dac, z)

    audio_hat = dac.decode(z)
    save_npy(f"{tag}_audio_hat.npy", np.asarray(audio_hat))

    z_fc, _, _ = dac.quantizer.from_codes(codes)
    save_npy(f"{tag}_z_from_codes.npy", np.asarray(z_fc))
    audio_fc = dac.decode(z_fc)
    save_npy(f"{tag}_audio_from_codes.npy", np.asarray(audio_fc))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    traces: dict = {}
    trace_files = [
        "a_zq.npy", "a_audio_hat.npy", "a_z_from_codes.npy",
        "a_audio_from_codes.npy", "b_zq.npy", "b_audio_hat.npy",
        "b_z_from_codes.npy", "b_audio_from_codes.npy",
    ]

    rng = np.random.default_rng(17)
    wave_a = rng.standard_normal(96).astype(np.float32)
    wave_b = rng.standard_normal(50).astype(np.float32)

    mx.random.seed(42)
    dac = DAC(**TINY)
    flat = tree_flatten(dac.parameters())
    mx.save_safetensors(
        str(OUT / "tiny_weights.safetensors"),
        {name: value for name, value in flat},
    )
    (OUT / "tiny_config.json").write_text(json.dumps(TINY, indent=2))
    (OUT / "weights_manifest.json").write_text(
        json.dumps(
            {
                "tiny": [
                    {"name": name, "shape": list(value.shape)}
                    for name, value in flat
                ]
            },
            indent=2,
        )
    )
    traces["delay"] = int(dac.delay)
    traces["hop_length"] = int(dac.hop_length)

    build_case(dac, "a", wave_a, traces)
    build_case(dac, "b", wave_b, traces)

    # Compress / decompress gain round trip on wave_a.
    audio = mx.array(wave_a[None, None, :])
    rms = mx.sqrt(mx.mean(mx.power(audio, 2), axis=-1) + 1e-12)
    input_db = 20 * mx.log10(rms / 1.0 + 1e-12)
    normalize_db = -16
    audio_norm = audio * mx.power(10, (normalize_db - input_db) / 20)
    padded = dac.preprocess(audio_norm, None)
    z, codes, _, _, _ = dac.encode(padded)
    recons = dac.decode(z)
    recons = recons * mx.power(10, (input_db - normalize_db) / 20)
    traces["compress_input_db"] = float(input_db.item())
    save_npy("compress_wave.npy", np.asarray(wave_a))
    save_npy("compress_recons.npy", np.asarray(recons))
    trace_files.extend(["compress_wave.npy", "compress_recons.npy"])

    (OUT / "traces.json").write_text(json.dumps(traces))

    files = [
        "tiny_config.json",
        "tiny_weights.safetensors",
        "traces.json",
        "weights_manifest.json",
        *trace_files,
    ]
    manifest = {
        "generator": "tools/gen_dac_fixtures.py",
        "reference_commit": "e1b19b9054bf163f5d812221a54fcc346f1890e9",
        "mlx_version": mx.__version__,
        "files": {name: digest(OUT / name) for name in files},
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(files) + 2} fixture files to {OUT}")


if __name__ == "__main__":
    main()
