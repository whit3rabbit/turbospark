#!/usr/bin/env python3
"""Regenerate the BigVGAN Rust parity fixtures.

Normative reference: mlx-audio at commit
e1b19b9054bf163f5d812221a54fcc346f1890e9
(mlx_audio/codec/models/bigvgan, mlx 0.32.3).

Run from inside the mlx-audio checkout root so `mlx_audio` imports from
the pinned source, with the project venv interpreter:

    cd ../mlx-audio
    ../.venv-mlxaudio/bin/python \
        ../turbospark/crates/audio/tools/gen_bigvgan_fixtures.py

Outputs into crates/audio/testdata/bigvgan/:

- tiny_config.json + tiny_weights.safetensors
    Seeded tiny BigVGAN (snakebeta activations, resblock "1", two
    upsample stages, Activation1d kaiser resampling active).
- traces.json + npy files
    Mel input -> waveform for two lengths (the Activation1d
    up/downsample path and the kaiser filters are exercised on both),
    plus the intermediate pre-activation trace.
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

from mlx_audio.codec.models.bigvgan.bigvgan import BigVGAN, BigVGANConfig

OUT = Path("../turbospark/crates/audio/testdata/bigvgan")
TINY = dict(
    num_mels=8,
    upsample_rates=[4, 4],
    upsample_kernel_sizes=[8, 8],
    upsample_initial_channel=16,
    resblock="1",
    resblock_kernel_sizes=[3],
    resblock_dilation_sizes=[[1, 3]],
    activation="snakebeta",
    snake_logscale=True,
    use_bias_at_final=True,
    use_tanh_at_final=True,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_npy(name: str, arr: np.ndarray) -> None:
    np.save(OUT / name, np.ascontiguousarray(arr, dtype=np.float32))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    traces: dict = {}
    trace_files = []

    mx.random.seed(45)
    config = BigVGANConfig(**TINY)
    model = BigVGAN(config)
    flat = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(str(OUT / "tiny_weights.safetensors"), flat)
    (OUT / "tiny_config.json").write_text(json.dumps(TINY, indent=2))

    rng = np.random.default_rng(37)
    for tag, frames in (("a", 20), ("b", 33)):
        # Reference input is (batch, num_mels, seq).
        mel = rng.standard_normal((1, TINY["num_mels"], frames)).astype(np.float32)
        audio = model(mx.array(mel))
        save_npy(f"{tag}_mel.npy", mel)
        save_npy(f"{tag}_audio.npy", np.asarray(audio))
        traces[f"{tag}_frames"] = frames
        traces[f"{tag}_audio_len"] = int(audio.shape[-1])
        trace_files.extend([f"{tag}_mel.npy", f"{tag}_audio.npy"])

    (OUT / "traces.json").write_text(json.dumps(traces))

    files = [
        "tiny_config.json",
        "tiny_weights.safetensors",
        "traces.json",
        *trace_files,
    ]
    manifest = {
        "generator": "tools/gen_bigvgan_fixtures.py",
        "reference_commit": "e1b19b9054bf163f5d812221a54fcc346f1890e9",
        "mlx_version": mx.__version__,
        "files": {name: digest(OUT / name) for name in files},
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(files) + 2} fixture files to {OUT}")


if __name__ == "__main__":
    main()
