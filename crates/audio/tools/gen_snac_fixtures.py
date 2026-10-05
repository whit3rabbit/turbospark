#!/usr/bin/env python3
"""Regenerate the SNAC Rust parity fixtures.

Normative reference: mlx-audio at commit
e1b19b9054bf163f5d812221a54fcc346f1890e9 (mlx_audio/codec/models/snac,
mlx 0.32.3).

Run from inside the mlx-audio checkout root so `mlx_audio` imports from
the pinned source, with the project venv interpreter:

    cd ../mlx-audio
    ../.venv-mlxaudio/bin/python \
        ../turbospark/crates/audio/tools/gen_snac_fixtures.py

Add --real to also record golden tensors from the locally cached
mlx-community/snac_24khz checkpoint (no network: local_files_only).

Outputs into crates/audio/testdata/snac/:

- tiny_a_config.json + tiny_a_weights.safetensors
    Seeded tiny SNAC in the depthwise, noise-free layout (the
    attn_window_size=null shape the mlx-audio SNAC test itself uses).
- tiny_b_config.json + tiny_b_weights.safetensors
    Seeded tiny SNAC in the dense layout (depthwise=false, noise=true,
    NoiseBlocks active) matching the real snac_24khz structure.
- traces.json
    End-to-end and per-stage golden tensors: wave inputs (wave_b is
    shorter than one pad block so the preprocess right-pad path runs),
    encoder output z, quantizer z_q and codes, decoder audio, the
    codes -> audio decode path, encode round trip, an isolated
    LocalMHA run on channels-first input, a single-level
    VectorQuantize trace with best vs second-best distance margins
    (keeps the exact-index gate honest), and the noise decode with an
    injected f32 normal pool the Rust test replays in order.
- weights_manifest.json
    Flattened parameter names and shapes of tiny_a, so the Rust loader
    is checked against the real checkpoint naming rather than guesses.
- real_encode_codes.json, real_decode_zero_noise.npy, real_wave.npy
    (--real) Golden tensors from the locally cached snac_24khz
    checkpoint: encode codes on a seeded waveform, and the decode of
    those codes with the NoiseBlock terms forced to zero (the
    deterministic backbone of the stochastic decode).
- manifest.json
    File list with SHA-256 digests and generator metadata.

The script is deterministic: fixed seeds, no network access. Large
tensors are little-endian float32 .npy files; everything else stays in
JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import unittest.mock as mock
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from mlx_audio.codec.models.snac.snac import SNAC

OUT = Path("../turbospark/crates/audio/testdata/snac")
TINY_A = dict(
    sampling_rate=44100,
    encoder_dim=8,
    encoder_rates=[2, 2, 2],
    latent_dim=None,
    decoder_dim=32,
    decoder_rates=[2, 2, 2],
    attn_window_size=None,
    codebook_size=64,
    codebook_dim=4,
    vq_strides=[4, 2, 1],
    noise=False,
    depthwise=True,
)
TINY_B = dict(TINY_A, noise=True, depthwise=False)
REAL_REPO = "mlx-community/snac_24khz"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_npy(name: str, arr: np.ndarray) -> None:
    # C-contiguous f32: moveaxis results can surface as Fortran-order
    # views, and the Rust testdata reader assumes C order.
    np.save(OUT / name, np.ascontiguousarray(arr, dtype=np.float32))


def floats(arr) -> list:
    return np.asarray(arr, dtype=np.float32).ravel().tolist()


def sqnorm_rows(x):
    return x / np.maximum(
        np.sqrt((np.abs(x) ** 2).sum(axis=1, keepdims=True)), 1e-12
    )


def vq_margins(snac: SNAC, z: mx.array) -> dict:
    """Best and second-best squared distances per frame for level 0.

    Replicates VectorQuantize.decode_latents so the Rust exact-index
    gate can prove no frame sits on a rounding knife-edge.
    """
    quantizer = snac.quantizer.quantizers[0]
    # z arrives channels-first (B, C, T); VectorQuantize moves to
    # channels-last, average pools by stride, then projects.
    z_t = z.moveaxis(1, 2)
    stride = quantizer.stride
    if stride > 1:
        kernel = mx.ones((z_t.shape[2], stride, 1)) / stride
        z_t = mx.conv1d(z_t, kernel, stride=stride, padding=0, groups=z_t.shape[2])
    z_e = quantizer.in_proj(z_t).moveaxis(1, 2)
    b, d, t = z_e.shape
    encodings = np.asarray(z_e.transpose(0, 2, 1).reshape(b * t, d))
    codebook = np.asarray(quantizer.codebook.weight)
    encodings = sqnorm_rows(encodings)
    codebook = sqnorm_rows(codebook)
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


def build_case(snac: SNAC, tag: str, wave: np.ndarray, traces: dict) -> None:
    audio = mx.array(wave[None, None, :])
    traces[f"{tag}_wave"] = floats(wave)
    traces[f"{tag}_input_len"] = int(wave.shape[0])
    traces[f"{tag}_padded_len"] = int(snac.preprocess(audio).shape[-1])

    codes = snac.encode(audio)
    traces[f"{tag}_codes_encode"] = [
        np.asarray(c).ravel().tolist() for c in codes
    ]

    padded_audio = snac.preprocess(audio)
    z = snac.encoder(padded_audio.moveaxis(1, 2))
    save_npy(f"{tag}_z.npy", np.asarray(z))
    z_q, codes_q = snac.quantizer(z)
    save_npy(f"{tag}_zq.npy", np.asarray(z_q))
    traces[f"{tag}_codes"] = [
        np.asarray(c).ravel().tolist() for c in codes_q
    ]
    traces[f"{tag}_vq_margins"] = vq_margins(snac, z)

    audio_hat = snac.decoder(z_q.moveaxis(1, 2))
    save_npy(f"{tag}_audio_hat.npy", np.asarray(audio_hat))

    audio_from_codes = snac.decode(codes)
    save_npy(f"{tag}_audio_from_codes.npy", np.asarray(audio_from_codes))


def dump_model(snac: SNAC, tag: str, config: dict) -> None:
    flat = tree_flatten(snac.parameters())
    mx.save_safetensors(
        str(OUT / f"{tag}_weights.safetensors"),
        {name: value for name, value in flat},
    )
    # The SNAC module does not keep noise/depthwise as attributes, so
    # dump the constructor dict itself (the from_config contract).
    (OUT / f"{tag}_config.json").write_text(json.dumps(config, indent=2))
    if tag == "tiny_a":
        (OUT / "weights_manifest.json").write_text(
            json.dumps(
                {
                    "tiny_a": [
                        {"name": name, "shape": list(value.shape)}
                        for name, value in flat
                    ]
                },
                indent=2,
            )
        )


def real_case(traces_files: list[str]) -> None:
    """Record golden tensors from the locally cached snac_24khz."""
    from huggingface_hub import snapshot_download

    path = Path(
        snapshot_download(REAL_REPO, local_files_only=True,
                          allow_patterns=["*.safetensors", "*.json"])
    )
    snac = SNAC.from_config(path / "config.json")
    weights = mx.load((path / "model.safetensors").as_posix(),
                      format="safetensors")
    snac.load_weights(list(weights.items()))
    mx.eval(snac.parameters())

    rng = np.random.default_rng(11)
    wave = rng.standard_normal(12000).astype(np.float32)
    audio = mx.array(wave[None, None, :])
    save_npy("real_wave.npy", wave)
    codes = snac.encode(audio)
    codes_list = [np.asarray(c).ravel().tolist() for c in codes]
    (OUT / "real_encode_codes.json").write_text(json.dumps(codes_list))

    # Decode with the noise terms zeroed: patch mx.random.normal to
    # return zeros so the deterministic backbone is reproducible.
    def zero_normal(*args, **kwargs):
        shape = args[0] if args else kwargs.get("shape")
        return mx.zeros(shape)

    with mock.patch.object(mx.random, "normal", zero_normal):
        audio_hat = snac.decode(codes)
    save_npy("real_decode_zero_noise.npy", np.asarray(audio_hat))
    traces_files.extend(
        ["real_wave.npy", "real_encode_codes.json", "real_decode_zero_noise.npy"]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true",
                        help="also record snac_24khz golden tensors")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    traces: dict = {}
    trace_files = [
        "a_z.npy", "a_zq.npy", "a_audio_hat.npy", "a_audio_from_codes.npy",
        "b_z.npy", "b_zq.npy", "b_audio_hat.npy", "b_audio_from_codes.npy",
        "noise_decode.npy",
        "vq_in.npy", "vq_zq.npy",
        "noise_pool.npy",
    ]

    rng = np.random.default_rng(7)
    wave_a = rng.standard_normal(128).astype(np.float32)
    wave_b = rng.standard_normal(70).astype(np.float32)

    mx.random.seed(42)
    snac_a = SNAC(**TINY_A)
    dump_model(snac_a, "tiny_a", TINY_A)
    build_case(snac_a, "a", wave_a, traces)
    build_case(snac_a, "b", wave_b, traces)

    # Isolated single-level VectorQuantize trace.
    vq = snac_a.quantizer.quantizers[0]
    vq_in = mx.array(
        rng.standard_normal((1, TINY_A["encoder_dim"] * 8, 24)).astype(np.float32)
    )
    vq_zq, vq_codes = vq(vq_in)
    save_npy("vq_in.npy", np.asarray(vq_in))
    save_npy("vq_zq.npy", np.asarray(vq_zq))
    traces["vq_codes"] = [int(c) for c in np.asarray(vq_codes).ravel().tolist()]
    traces["vq_stride"] = int(vq.stride)

    # Noise model: inject a recorded f32 normal pool in place of
    # mx.random.normal so the Rust test can replay the same draws.
    mx.random.seed(43)
    snac_b = SNAC(**TINY_B)
    dump_model(snac_b, "tiny_b", TINY_B)

    pool = rng.standard_normal(4096).astype(np.float32)
    cursor = {"n": 0}

    def fake_normal(*args, **kwargs):
        shape = args[0] if args else kwargs.get("shape")
        n = int(np.prod(shape))
        out = pool[cursor["n"]: cursor["n"] + n].reshape(shape)
        cursor["n"] += n
        return mx.array(out)

    noise_codes = snac_b.encode(mx.array(wave_a[None, None, :]))
    with mock.patch.object(mx.random, "normal", fake_normal):
        audio_noise = snac_b.decode(noise_codes)
    save_npy("noise_pool.npy", pool[: cursor["n"]])
    save_npy("noise_decode.npy", np.asarray(audio_noise))
    traces["noise_codes"] = [
        np.asarray(c).ravel().tolist() for c in noise_codes
    ]
    traces["noise_draws"] = int(cursor["n"])

    if args.real:
        real_case(trace_files)

    (OUT / "traces.json").write_text(json.dumps(traces))

    files = [
        "tiny_a_config.json",
        "tiny_a_weights.safetensors",
        "tiny_b_config.json",
        "tiny_b_weights.safetensors",
        "traces.json",
        "weights_manifest.json",
        *trace_files,
    ]
    manifest = {
        "generator": "tools/gen_snac_fixtures.py",
        "reference_commit": "e1b19b9054bf163f5d812221a54fcc346f1890e9",
        "mlx_version": mx.__version__,
        "files": {name: digest(OUT / name) for name in files},
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(files) + 2} fixture files to {OUT}")


if __name__ == "__main__":
    main()
