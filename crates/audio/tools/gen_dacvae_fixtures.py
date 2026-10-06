#!/usr/bin/env python3
"""Regenerate the DACVAE Rust parity fixtures.

Normative reference: mlx-audio at commit
e1b19b9054bf163f5d812221a54fcc346f1890e9
(mlx_audio/codec/models/dacvae, mlx 0.32.3).

Run from inside the mlx-audio checkout root so `mlx_audio` imports from
the pinned source, with the project venv interpreter:

    cd ../mlx-audio
    ../.venv-mlxaudio/bin/python \
        ../turbospark/crates/audio/tools/gen_dacvae_fixtures.py

Outputs into crates/audio/testdata/dacvae/:

- tiny_config.json + tiny_weights.safetensors
    Seeded tiny DACVAE (the non-watermark batch encode/decode paths).
- traces.json + npy files
    Encode latents (the VAE mean half), decoded waveforms, and the
    decoder block internals for two input lengths.
- manifest.json
    File list with SHA-256 digests and generator metadata.

The script is deterministic: fixed seeds, no network access. The
watermark path is not part of the batch encode/decode contract and is
not exercised.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import fields
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from mlx_audio.codec.models.dacvae.codec import DACVAE, DACVAEConfig

OUT = Path("../turbospark/crates/audio/testdata/dacvae")
TINY = dict(
    encoder_dim=8,
    encoder_rates=[2, 4],
    latent_dim=16,
    # 72 keeps every watermark-path width (out_dim / 3, twice) nonzero.
    decoder_dim=72,
    decoder_rates=[4, 2],
    n_codebooks=4,
    codebook_size=64,
    codebook_dim=8,
    sample_rate=24000,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_npy(name: str, arr: np.ndarray) -> None:
    np.save(OUT / name, np.ascontiguousarray(arr, dtype=np.float32))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    traces: dict = {}
    trace_files = []

    cfg = DACVAEConfig(**TINY)
    mx.random.seed(48)
    model = DACVAE(cfg)
    flat = dict(tree_flatten(model.parameters()))
    rng = np.random.default_rng(53)
    seeded = {
        name: mx.array((rng.standard_normal(v.shape) * 0.3).astype(np.float32))
        for name, v in flat.items()
    }
    model.load_weights(list(seeded.items()))
    mx.eval(model)
    mx.save_safetensors(str(OUT / "tiny_weights.safetensors"), seeded)
    (OUT / "tiny_config.json").write_text(
        json.dumps({f.name: getattr(cfg, f.name) for f in fields(DACVAEConfig)}, indent=2)
    )
    (OUT / "weights_manifest.json").write_text(
        json.dumps(
            {
                "tiny": [
                    {"name": name, "shape": list(v.shape)} for name, v in flat.items()
                ]
            },
            indent=2,
        )
    )

    rng = np.random.default_rng(59)
    for tag, n in (("a", 1920), ("b", 700)):
        wave = rng.standard_normal(n).astype(np.float32)
        save_npy(f"{tag}_wave.npy", wave)
        trace_files.append(f"{tag}_wave.npy")
        # (batch, length, 1) channels-last contract.
        latents = model.encode(mx.array(wave)[None, :, None])
        save_npy(f"{tag}_latents.npy", np.asarray(latents))
        out = model.decode(latents)
        save_npy(f"{tag}_audio.npy", np.asarray(out))
        traces[f"{tag}_input_len"] = n
        traces[f"{tag}_latents_shape"] = [int(d) for d in latents.shape]
        traces[f"{tag}_audio_len"] = int(out.shape[1])
        trace_files.extend([f"{tag}_latents.npy", f"{tag}_audio.npy"])

    (OUT / "traces.json").write_text(json.dumps(traces))

    files = [
        "tiny_config.json",
        "tiny_weights.safetensors",
        "weights_manifest.json",
        "traces.json",
        *trace_files,
    ]
    manifest = {
        "generator": "tools/gen_dacvae_fixtures.py",
        "reference_commit": "e1b19b9054bf163f5d812221a54fcc346f1890e9",
        "mlx_version": mx.__version__,
        "files": {name: digest(OUT / name) for name in files},
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(files) + 2} fixture files to {OUT}")


if __name__ == "__main__":
    main()
