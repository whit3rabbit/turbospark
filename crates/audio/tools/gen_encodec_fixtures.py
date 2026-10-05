#!/usr/bin/env python3
"""Regenerate the EnCodec Rust parity fixtures.

Normative reference: mlx-audio at commit
e1b19b9054bf163f5d812221a54fcc346f1890e9
(mlx_audio/codec/models/encodec, mlx 0.32.3).

Run from inside the mlx-audio checkout root so `mlx_audio` imports from
the pinned source, with the project venv interpreter:

    cd ../mlx-audio
    ../.venv-mlxaudio/bin/python \
        ../turbospark/crates/audio/tools/gen_encodec_fixtures.py

Outputs into crates/audio/testdata/encodec/:

- tiny_config.json + tiny_weights.safetensors
    Seeded tiny EnCodec (causal reflect convs, one resnet layer per
    stage, two LSTM layers, [2, 4] upsampling ratios, two bandwidths).
    The reference LSTM constructor leaves Wx/Wh/bias at zero, so the
    generator draws seeded normals for them before saving; without that
    the LSTM path would be degenerate and parity-meaningless.
- traces.json
    wave_a / wave_b (arbitrary lengths: EnCodec pads internally in the
    conv stack, no caller-side preprocess), codes at the default and
    second bandwidth (exact-index gates with Euclidean distance
    margins), quantized latents, decoder audio, and an isolated LSTM
    trace.
- weights_manifest.json
    Flattened parameter names and shapes.
- manifest.json
    File list with SHA-256 digests and generator metadata.

The script is deterministic: fixed seeds, no network access.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from mlx_audio.codec.models.encodec.encodec import Encodec, EncodecConfig

OUT = Path("../turbospark/crates/audio/testdata/encodec")
TINY = dict(
    audio_channels=1,
    num_filters=8,
    kernel_size=7,
    num_residual_layers=1,
    dilation_growth_rate=2,
    codebook_size=64,
    codebook_dim=8,
    hidden_size=8,
    num_lstm_layers=2,
    residual_kernel_size=3,
    use_causal_conv=True,
    normalize=False,
    pad_mode="reflect",
    norm_type="weight_norm",
    last_kernel_size=7,
    trim_right_ratio=1.0,
    compress=2,
    upsampling_ratios=[2, 4],
    target_bandwidths=[1.5, 3.0],
    sampling_rate=240,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_npy(name: str, arr: np.ndarray) -> None:
    np.save(OUT / name, np.ascontiguousarray(arr, dtype=np.float32))


def floats(arr) -> list:
    return np.asarray(arr, dtype=np.float32).ravel().tolist()


def euclid_margins(codebook: np.ndarray, encodings: np.ndarray) -> dict:
    """Best vs second-best squared distances per frame.

    codebook is (size, dim); encodings is (frames, dim) channels-last.
    """
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


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    traces: dict = {}
    trace_files = [
        "a_zq.npy", "a_audio.npy", "b_zq.npy", "b_audio.npy",
        "lstm_in.npy", "lstm_out.npy",
    ]

    config = EncodecConfig(**TINY)
    mx.random.seed(42)
    model = Encodec(config)

    # The reference LSTM constructor leaves Wx/Wh/bias at zero and the
    # codebooks start at zero; draw seeded normals for both so the
    # recurrence and the nearest-code lookups are exercised (zero
    # codebooks make every code a degenerate index-0 tie).
    rng = np.random.default_rng(3)
    flat = dict(tree_flatten(model.parameters()))
    for name in list(flat):
        if (
            name.endswith(".Wx")
            or name.endswith(".Wh")
            or name.endswith(".bias")
            or name.endswith(".embed")
        ):
            shape = flat[name].shape
            flat[name] = mx.array(
                (rng.standard_normal(shape) * 0.5).astype(np.float32)
            )
    model.load_weights(list(flat.items()))
    mx.eval(model)

    mx.save_safetensors(
        str(OUT / "tiny_weights.safetensors"),
        {name: value for name, value in flat.items()},
    )
    (OUT / "tiny_config.json").write_text(
        json.dumps({k: getattr(config, k) for k in
                    ("audio_channels", "num_filters", "kernel_size",
                     "num_residual_layers", "dilation_growth_rate",
                     "codebook_size", "codebook_dim", "hidden_size",
                     "num_lstm_layers", "residual_kernel_size",
                     "use_causal_conv", "normalize", "pad_mode",
                     "norm_type", "last_kernel_size", "trim_right_ratio",
                     "compress", "upsampling_ratios", "target_bandwidths",
                     "sampling_rate")}, indent=2)
    )
    (OUT / "weights_manifest.json").write_text(
        json.dumps(
            {
                "tiny": [
                    {"name": name, "shape": list(value.shape)}
                    for name, value in flat.items()
                ]
            },
            indent=2,
        )
    )
    traces["frame_rate"] = int(model.quantizer.frame_rate)
    traces["num_quantizers"] = int(model.quantizer.num_quantizers)

    rng = np.random.default_rng(23)
    wave_a = rng.standard_normal(64).astype(np.float32)
    wave_b = rng.standard_normal(50).astype(np.float32)

    for tag, wave in (("a", wave_a), ("b", wave_b)):
        traces[f"{tag}_wave"] = floats(wave)
        # The implementation consumes channels-last (B, T, C) input.
        audio = mx.array(wave)[None, :, None]
        # Default bandwidth (target_bandwidths[0]).
        frames, scales = model.encode(audio, None, None)
        codes = frames[0]
        traces[f"{tag}_codes"] = [
            np.asarray(codes[:, i, :]).ravel().tolist()
            for i in range(codes.shape[1])
        ]
        # Second bandwidth exercises a deeper residual stack.
        bw = TINY["target_bandwidths"][1]
        frames_bw, _ = model.encode(audio, None, bw)
        codes_bw = frames_bw[0]
        traces[f"{tag}_codes_bw"] = [
            np.asarray(codes_bw[:, i, :]).ravel().tolist()
            for i in range(codes_bw.shape[1])
        ]
        quantized = model.quantizer.decode(codes)
        save_npy(f"{tag}_zq.npy", np.asarray(quantized))
        # Euclidean margins at the first codebook for the default codes.
        embed = np.asarray(model.quantizer.layers[0].codebook.embed)
        z = np.asarray(quantized)[0]
        traces[f"{tag}_vq_margins"] = euclid_margins(embed, z)
        # _decode_frame is the working math path (quantizer decode +
        # decoder); the public decode wrapper's single-frame shape
        # check does not accept (B, num_q, T) codes at this commit.
        decoded = model._decode_frame(codes, None)
        save_npy(f"{tag}_audio.npy", np.asarray(decoded))

    # Isolated LSTM trace (encoder LSTM over its residual add). The
    # encoder LSTM runs at scaling * num_filters after the doublings.
    lstm_dim = TINY["num_filters"] * (2 ** len(TINY["upsampling_ratios"]))
    steps = 9
    lstm_in = rng.standard_normal((1, steps, lstm_dim)).astype(np.float32)
    enc_lstm = model.encoder.layers[-3]
    h = mx.array(lstm_in)
    out = enc_lstm(h)
    save_npy("lstm_in.npy", lstm_in)
    save_npy("lstm_out.npy", np.asarray(out))

    (OUT / "traces.json").write_text(json.dumps(traces))

    files = [
        "tiny_config.json",
        "tiny_weights.safetensors",
        "traces.json",
        "weights_manifest.json",
        *trace_files,
    ]
    manifest = {
        "generator": "tools/gen_encodec_fixtures.py",
        "reference_commit": "e1b19b9054bf163f5d812221a54fcc346f1890e9",
        "mlx_version": mx.__version__,
        "files": {name: digest(OUT / name) for name in files},
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(files) + 2} fixture files to {OUT}")


if __name__ == "__main__":
    main()
