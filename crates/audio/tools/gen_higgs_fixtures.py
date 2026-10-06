#!/usr/bin/env python3
"""Regenerate the Higgs Audio tokenizer Rust parity fixtures.

Normative reference: mlx-audio at commit
e1b19b9054bf163f5d812221a54fcc346f1890e9
(mlX_audio/codec/models/higgs_audio, mlx 0.32.3).

Run from inside the mlx-audio checkout root so `mlx_audio` imports from
the pinned source, with the project venv interpreter:

    cd ../mlx-audio
    ../.venv-mlxaudio/bin/python \
        ../turbospark/crates/audio/tools/gen_higgs_fixtures.py

Outputs into crates/audio/testdata/higgs_audio/:

- tiny_config.json + tiny_weights.safetensors
    Seeded tiny tokenizer. The reference hardcodes the acoustic
    geometry as class constants (_STRIDES/_CHANNELS) and constructor
    defaults (latent 1024, decoder input 256), so the tiny fixture
    mirrors those __init__ bodies at shrunk dims while keeping the
    attribute tree (and therefore every weight key) identical.
- traces.json + npy files
    Acoustic features, RVQ codes (with margins), RVQ reconstructions,
    and full token decodes for two waveforms plus a synthetic
    embedding round trip.
- manifest.json
    File list with SHA-256 digests and generator metadata.

The script is deterministic: fixed seeds, no network access. The
semantic encode path (wav2vec2 fusion) is not exercised; it lives in
mlx_audio/stt and is out of the codec contract.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from mlx_audio.codec.models.higgs_audio.dac import (
    AcousticDecoderBlock,
    AcousticEncoderBlock,
    ResidualVectorQuantizer,
    Snake1d,
    VectorQuantizer,
)
from mlx_audio.codec.models.higgs_audio.higgs_audio import AcousticDecoder, AcousticEncoder
from mlx_audio.codec.models.dacvae.codec import WNConv1d
import mlx.nn as nn

OUT = Path("../turbospark/crates/audio/testdata/higgs_audio")
REFERENCE_COMMIT = "e1b19b9054bf163f5d812221a54fcc346f1890e9"


class TinyAcousticEncoder(AcousticEncoder):
    """AcousticEncoder.__init__ at encoder_hidden=4, strides [8,5,4,2,3]."""

    _STRIDES = [8, 5, 4, 2, 3]
    _CHANNELS = [4, 8, 16, 32, 64, 128]

    def __init__(self):
        c = self._CHANNELS
        self.conv1 = WNConv1d(1, c[0], kernel_size=7, padding=3, norm="none")
        self.block = [
            AcousticEncoderBlock(c[i], c[i + 1], self._STRIDES[i])
            for i in range(len(self._STRIDES))
        ]
        self.snake1 = Snake1d(c[-1])
        self.conv2 = WNConv1d(c[-1], 6, kernel_size=3, padding=1, norm="none")


class TinyAcousticDecoder(AcousticDecoder):
    """AcousticDecoder.__init__ at decoder_hidden=32, decoder input 6."""

    _STRIDES = [8, 5, 4, 2, 3]
    _IN_CHANNELS = [32, 16, 8, 4, 2]
    _OUT_CHANNELS = [16, 8, 4, 2, 1]

    def __init__(self):
        self.conv1 = WNConv1d(6, self._IN_CHANNELS[0], kernel_size=7, padding=3, norm="none")
        self.block = [
            AcousticDecoderBlock(self._IN_CHANNELS[i], self._OUT_CHANNELS[i], self._STRIDES[i])
            for i in range(len(self._STRIDES))
        ]
        self.snake1 = Snake1d(self._OUT_CHANNELS[-1])
        self.conv2 = WNConv1d(self._OUT_CHANNELS[-1], 1, kernel_size=7, padding=3, norm="none")


class TinyVq(VectorQuantizer):
    def __init__(self):
        super().__init__(latent_dim=8, codebook_size=16, codebook_dim=4)


class TinyRvq(ResidualVectorQuantizer):
    def __init__(self):
        self.quantizers = [TinyVq() for _ in range(3)]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_npy(name: str, arr: np.ndarray) -> None:
    np.save(OUT / name, np.ascontiguousarray(arr, dtype=np.float32))


def seed(model, seed: int) -> dict:
    flat = dict(tree_flatten(model.parameters()))
    rng = np.random.default_rng(seed)
    seeded = {
        name: mx.array((rng.standard_normal(v.shape) * 0.3).astype(np.float32))
        for name, v in flat.items()
    }
    model.load_weights(list(seeded.items()))
    mx.eval(model)
    return seeded


def rvq_margins(rvq: TinyRvq, z: np.ndarray) -> float:
    """Smallest best-vs-second margin across books for these rows."""
    residual = z.copy()
    min_margin = np.inf
    for vq in rvq.quantizers:
        zq = np.asarray(vq.project_in(mx.array(residual)), dtype=np.float32)
        w = np.asarray(vq.codebook.weight, dtype=np.float32)
        d = (zq**2).sum(-1, keepdims=True) - 2.0 * zq @ w.T + (w**2).sum(-1)[None, :]
        order = np.argsort(d, axis=-1, kind="stable")
        best = np.take_along_axis(d, order[:, :1], axis=-1)
        second = np.take_along_axis(d, order[:, 1:2], axis=-1)
        min_margin = min(min_margin, float((second - best).min()))
        idx = np.argmin(d, axis=-1)
        recon = np.asarray(vq.decode_codes(mx.array(idx)), dtype=np.float32)
        residual = residual - recon
    return min_margin


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    traces: dict = {}
    trace_files: list[str] = []

    enc = TinyAcousticEncoder()
    dec = TinyAcousticDecoder()
    rvq = TinyRvq()
    fc2 = nn.Linear(8, 6, bias=True)

    seeded = {}
    seeded.update({f"acoustic_encoder.{k}": v for k, v in seed(enc, 301).items()})
    seeded.update({f"acoustic_decoder.{k}": v for k, v in seed(dec, 303).items()})
    seeded.update({f"quantizer.{k}": v for k, v in seed(rvq, 305).items()})
    flat_fc2 = dict(tree_flatten(fc2.parameters()))
    rng = np.random.default_rng(307)
    for name, v in flat_fc2.items():
        seeded[f"fc2.{name}"] = mx.array((rng.standard_normal(v.shape) * 0.3).astype(np.float32))
    fc2.load_weights([(n[len("fc2."):], v) for n, v in seeded.items() if n.startswith("fc2.")])
    mx.eval(fc2)

    mx.save_safetensors(str(OUT / "tiny_weights.safetensors"), seeded)
    config = {
        "sample_rate": 24000,
        "codebook_size": 16,
        "codebook_dim": 4,
        "downsample_factor": 320,
        "dac_num_codebooks": 3,
        "dac_encoder_ratios": [8, 5, 4, 2, 3],
        "dac_encoder_hidden": 4,
        "dac_decoder_hidden": 32,
        "semantic_sample_rate": 16000,
        "quantizer_latent_dim": 8,
        "decoder_input_dim": 6,
    }
    (OUT / "tiny_config.json").write_text(json.dumps(config, indent=2))
    (OUT / "weights_manifest.json").write_text(
        json.dumps({"tiny": [{"name": n, "shape": list(v.shape)} for n, v in sorted(seeded.items())]}, indent=2)
    )

    wave_rng = np.random.default_rng(311)
    for tag, n in (("a", 1920), ("b", 1440)):
        wave = wave_rng.standard_normal(n).astype(np.float32) * 0.2
        save_npy(f"{tag}_wave.npy", wave)
        trace_files.append(f"{tag}_wave.npy")
        feats = np.asarray(enc(mx.array(wave)[None, :, None]))[0]
        save_npy(f"{tag}_acoustic.npy", feats)
        traces[f"{tag}_input_len"] = n
        traces[f"{tag}_acoustic_shape"] = list(feats.shape)
        trace_files.append(f"{tag}_acoustic.npy")

    # Synthetic embedding round trip through the quantizer.
    rng = np.random.default_rng(313)
    frames = 5
    z = (rng.standard_normal((frames, 8)) * 0.5).astype(np.float32)
    save_npy("z_rows.npy", z)
    codes = np.asarray(rvq.encode(mx.array(z)[None]), dtype=np.int32)[0]  # (T, books)
    np.save(OUT / "codes.npy", np.ascontiguousarray(codes))
    recon = np.asarray(rvq.decode(mx.array(codes)[None]))[0]
    save_npy("z_recon.npy", recon)
    traces["z_frames"] = frames
    traces["codes_shape"] = list(codes.shape)
    traces["rvq_min_margin"] = rvq_margins(rvq, z)
    trace_files.extend(["z_rows.npy", "codes.npy", "z_recon.npy"])

    # Full token decode: fc2 + acoustic decoder.
    wav = np.asarray(dec(fc2(mx.array(recon)[None])))[0, :, 0]
    save_npy("decoded.npy", wav)
    traces["decoded_len"] = int(wav.shape[0])
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
        "generator": "tools/gen_higgs_fixtures.py",
        "reference_commit": REFERENCE_COMMIT,
        "mlx_version": mx.__version__,
        "files": {name: digest(OUT / name) for name in files},
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(files) + 1} fixture files to {OUT}")


if __name__ == "__main__":
    main()
