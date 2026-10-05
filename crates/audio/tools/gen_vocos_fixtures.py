#!/usr/bin/env python3
"""Regenerate the Vocos Rust parity fixtures.

Normative reference: mlx-audio at commit
e1b19b9054bf163f5d812221a54fcc346f1890e9
(mlx_audio/codec/models/vocos, mlx 0.32.3).

Run from inside the mlx-audio checkout root so `mlx_audio` imports from
the pinned source, with the project venv interpreter:

    cd ../mlx-audio
    ../.venv-mlxaudio/bin/python \
        ../turbospark/crates/audio/tools/gen_vocos_fixtures.py

Outputs into crates/audio/testdata/vocos/:

- config.yaml
    The real from_hparams structure (MelSpectrogramFeatures front end,
    ConvNeXt backbone, ISTFT head) for a tiny seeded model. n_fft is a
    power of two because the Rust FFT is radix-2.
- tiny_weights.safetensors
    Seeded parameters with backbone.embed and dwconv weights
    transposed to the PyTorch [out, in, K] layout the real vocos
    checkpoints store (from_pretrained transposes on load).
- traces.json + golden npy files
    wave -> log-mel features (mel_*.npy), decoded audio (audio_*.npy)
    for two input lengths, and the recorded model delay equivalent
    (head output length math).
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

from mlx_audio.codec.models.vocos.vocos import Vocos

OUT = Path("../turbospark/crates/audio/testdata/vocos")
CONFIG = {
    "feature_extractor": {
        "class_path": "vocos.feature_extractors.MelSpectrogramFeatures",
        "init_args": {
            "sample_rate": 24000,
            "n_fft": 256,
            "hop_length": 64,
            "n_mels": 20,
            "padding": "center",
        },
    },
    "backbone": {
        "class_path": "vocos.models.VocosBackbone",
        "init_args": {
            "input_channels": 20,
            "dim": 32,
            "intermediate_dim": 64,
            "num_layers": 4,
            "layer_scale_init_value": None,
            "adanorm_num_embeddings": None,
            "bias": True,
            "input_kernel_size": 7,
            "dw_kernel_size": 7,
        },
    },
    "head": {
        "class_path": "vocos heads.ISTFTHead",
        "init_args": {"dim": 32, "n_fft": 256, "hop_length": 64, "padding": "center"},
    },
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_npy(name: str, arr: np.ndarray) -> None:
    np.save(OUT / name, np.ascontiguousarray(arr, dtype=np.float32))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    traces: dict = {}
    trace_files = []

    mx.random.seed(42)
    model = Vocos.from_hparams(CONFIG)

    flat = dict(tree_flatten(model.parameters()))
    # Store embed/dwconv weights in the PyTorch layout real vocos
    # checkpoints use (from_pretrained applies moveaxis(1, 2)).
    saved = {}
    for name, value in flat.items():
        basename, pname = name.rsplit(".", 1)
        if pname == "weight" and ("backbone.embed" in basename or "dwconv" in basename):
            saved[name] = value.moveaxis(1, 2)
        else:
            saved[name] = value
    mx.save_safetensors(str(OUT / "tiny_weights.safetensors"), saved)

    yaml_text = ["feature_extractor:"]
    fe = CONFIG["feature_extractor"]
    yaml_text.append(f"  class_path: {fe['class_path']}")
    yaml_text.append("  init_args:")
    for k, v in fe["init_args"].items():
        yaml_text.append(f"    {k}: {v}")
    yaml_text.append("backbone:")
    bb = CONFIG["backbone"]
    yaml_text.append(f"  class_path: {bb['class_path']}")
    yaml_text.append("  init_args:")
    for k, v in bb["init_args"].items():
        yaml_text.append(
            f"    {k}: {str(v).lower() if isinstance(v, bool) else v}"
        )
    yaml_text.append("head:")
    hd = CONFIG["head"]
    yaml_text.append(f"  class_path: {hd['class_path']}")
    yaml_text.append("  init_args:")
    for k, v in hd["init_args"].items():
        yaml_text.append(f"    {k}: {v}")
    (OUT / "config.yaml").write_text("\n".join(yaml_text) + "\n")

    rng = np.random.default_rng(29)
    cases = {
        "a": rng.standard_normal(2400).astype(np.float32),
        "b": rng.standard_normal(701).astype(np.float32),
    }
    for tag, wave in cases.items():
        save_npy(f"{tag}_wave.npy", wave)
        trace_files.append(f"{tag}_wave.npy")
        features = model.feature_extractor(mx.array(wave))
        save_npy(f"{tag}_mel.npy", np.asarray(features))
        audio = model.decode(features)
        save_npy(f"{tag}_audio.npy", np.asarray(audio))
        traces[f"{tag}_input_len"] = int(wave.shape[0])
        traces[f"{tag}_mel_shape"] = [int(d) for d in features.shape]
        traces[f"{tag}_audio_len"] = int(audio.shape[-1])
        trace_files.extend([f"{tag}_mel.npy", f"{tag}_audio.npy"])

    (OUT / "traces.json").write_text(json.dumps(traces))

    files = [
        "config.yaml",
        "tiny_weights.safetensors",
        "traces.json",
        *trace_files,
    ]
    manifest = {
        "generator": "tools/gen_vocos_fixtures.py",
        "reference_commit": "e1b19b9054bf163f5d812221a54fcc346f1890e9",
        "mlx_version": mx.__version__,
        "files": {name: digest(OUT / name) for name in files},
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(files) + 2} fixture files to {OUT}")


if __name__ == "__main__":
    main()
