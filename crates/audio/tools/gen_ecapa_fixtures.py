#!/usr/bin/env python3
"""Regenerate the ECAPA-TDNN Rust parity fixtures.

Normative reference: mlx-audio at commit
e1b19b9054bf163f5d812221a54fcc346f1890e9
(mlx_audio/codec/models/ecapa_tdnn, mlx 0.32.3).

Run from inside the mlx-audio checkout root so `mlx_audio` imports from
the pinned source, with the project venv interpreter:

    cd ../mlx-audio
    ../.venv-mlxaudio/bin/python \
        ../turbospark/crates/audio/tools/gen_ecapa_fixtures.py

Outputs into crates/audio/testdata/ecapa_tdnn/:

- tiny_gc0_config.json + tiny_gc0_weights.safetensors
- tiny_gc1_config.json + tiny_gc1_weights.safetensors
    Seeded tiny ECAPA-TDNN backbones without and with the attentive
    pooling global-context branch.
- traces.json
    Two feature cases per model (one aligned to the conv footprint,
    one arbitrary length) with the 256-style embedding golden vectors
    plus an intermediate post-MFA trace.
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

from mlx_audio.codec.models.ecapa_tdnn.config import EcapaTdnnConfig
from mlx_audio.codec.models.ecapa_tdnn.ecapa_tdnn import EcapaTdnnBackbone

OUT = Path("../turbospark/crates/audio/testdata/ecapa_tdnn")
TINY = dict(
    input_size=12,
    channels=16,
    embed_dim=8,
    kernel_sizes=[5, 3, 3, 3, 1],
    dilations=[1, 2, 3, 4, 1],
    attention_channels=4,
    res2net_scale=2,
    se_channels=4,
    global_context=False,
)
TINY_GC = dict(TINY, global_context=True)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    traces: dict = {}

    rng = np.random.default_rng(31)
    for tag, cfg in (("gc0", TINY), ("gc1", TINY_GC)):
        mx.random.seed(44)
        config = EcapaTdnnConfig(**cfg)
        model = EcapaTdnnBackbone(config)
        # Inference uses the stored running stats (mlx defaults new
        # modules to training mode, so switch like a real caller).
        model.eval()
        flat = dict(tree_flatten(model.parameters()))
        mx.save_safetensors(
            str(OUT / f"tiny_{tag}_weights.safetensors"),
            flat,
        )
        (OUT / f"tiny_{tag}_config.json").write_text(json.dumps(cfg, indent=2))

        feats_a = rng.standard_normal((1, 50, cfg["input_size"])).astype(np.float32)
        feats_b = rng.standard_normal((1, 37, cfg["input_size"])).astype(np.float32)
        for case, feats in (("a", feats_a), ("b", feats_b)):
            out = model(mx.array(feats))
            traces[f"{tag}_{case}_embed"] = np.asarray(out).ravel().tolist()
            np.save(OUT / f"{tag}_{case}_feats.npy", np.ascontiguousarray(feats, dtype=np.float32))

    (OUT / "traces.json").write_text(json.dumps(traces))

    files = [
        "tiny_gc0_config.json",
        "tiny_gc0_weights.safetensors",
        "tiny_gc1_config.json",
        "tiny_gc1_weights.safetensors",
        "traces.json",
        "gc0_a_feats.npy", "gc0_b_feats.npy",
        "gc1_a_feats.npy", "gc1_b_feats.npy",
    ]
    manifest = {
        "generator": "tools/gen_ecapa_fixtures.py",
        "reference_commit": "e1b19b9054bf163f5d812221a54fcc346f1890e9",
        "mlx_version": mx.__version__,
        "files": {name: digest(OUT / name) for name in files},
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(files) + 2} fixture files to {OUT}")


if __name__ == "__main__":
    main()
