#!/usr/bin/env python3
"""Regenerate the Fish S1 DAC Rust parity fixtures.

Normative reference: mlx-audio at commit
e1b19b9054bf163f5d812221a54fcc346f1890e9
(mlX_audio/codec/models/fish_s1_dac, mlx 0.32.3).

Run from inside the mlx-audio checkout root so `mlx_audio` imports from
the pinned source, with the project venv interpreter:

    cd ../mlx-audio
    ../.venv-mlxaudio/bin/python \
        ../turbospark/crates/audio/tools/gen_fish_s1_dac_fixtures.py

The reference geometry (`build_ae()`) is hardcoded, so the tiny fixture
mirrors the `__init__` bodies at shrunk dims while keeping every
attribute path (and therefore weight key) identical. The transformers'
`freqs_cis` tables and causal masks are derived state: they are excluded
from the seeded weights and keep their true init values.

Outputs into crates/audio/testdata/fish_s1_dac/:

- tiny_config.json + tiny_weights.safetensors
- traces.json + npy files (codes with margins, decoded waveform)
- manifest.json
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from mlx_audio.codec.models.fish_s1_dac import fish_s1_dac as F

OUT = Path("../turbospark/crates/audio/testdata/fish_s1_dac")
REFERENCE_COMMIT = "e1b19b9054bf163f5d812221a54fcc346f1890e9"


class TinyArgs(F.ModelArgs):
    def __post_init__(self):
        if self.n_local_heads == -1:
            self.n_local_heads = self.n_head


def make_tiny_dac() -> F.DAC:
    """build_ae() at encoder_dim 8, hop 8, latent 32, 3 acoustic books."""
    q_config = TinyArgs(
        block_size=256,
        n_layer=2,
        n_head=4,
        dim=32,
        intermediate_size=48,
        head_dim=8,
        norm_eps=1e-5,
        dropout_rate=0.1,
        attn_dropout_rate=0.1,
        channels_first=True,
    )

    def make_transformer():
        return F.WindowLimitedTransformer(
            causal=True,
            window_size=4,
            input_dim=32,
            config=q_config,
        )

    def transformer_general_config(**kw):
        dim = kw.get("dim", 32)
        # The reference derives n_head = dim // 64 with head_dim 64;
        # at tiny dims keep four 8-dim heads instead.
        return TinyArgs(
            block_size=kw.get("block_size", 256),
            n_layer=kw.get("n_layer", 2),
            n_head=4 if dim <= 64 else kw.get("n_head", 4),
            dim=dim,
            intermediate_size=kw.get("intermediate_size", 48),
            n_local_heads=kw.get("n_local_heads", -1),
            head_dim=8 if dim <= 64 else kw.get("head_dim", 8),
            rope_base=kw.get("rope_base", 10000),
            norm_eps=kw.get("norm_eps", 1e-5),
            dropout_rate=kw.get("dropout_rate", 0.1),
            attn_dropout_rate=kw.get("attn_dropout_rate", 0.1),
            channels_first=True,
        )

    quantizer = F.DownsampleResidualVectorQuantize(
        input_dim=32,
        n_codebooks=3,
        codebook_size=12,
        codebook_dim=4,
        quantizer_dropout=0.5,
        downsample_factor=(2, 2),
        semantic_codebook_size=16,
        pre_module=make_transformer(),
        post_module=make_transformer(),
    )
    return F.DAC(
        encoder_dim=8,
        encoder_rates=[2, 4],
        latent_dim=32,
        decoder_dim=16,
        decoder_rates=[4, 2],
        quantizer=quantizer,
        sample_rate=44100,
        causal=True,
        encoder_transformer_layers=[0, 2],
        decoder_transformer_layers=[0, 0],
        transformer_general_config=transformer_general_config,
    )


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_npy(name: str, arr: np.ndarray) -> None:
    np.save(OUT / name, np.ascontiguousarray(arr, dtype=np.float32))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    traces: dict = {}
    trace_files: list[str] = []

    dac = make_tiny_dac()
    flat = dict(tree_flatten(dac.parameters()))
    derived = [k for k in flat if "freqs_cis" in k or "causal_mask" in k]
    trainable = {k: v for k, v in flat.items() if k not in derived}
    rng = np.random.default_rng(601)
    seeded = {
        name: mx.array((rng.standard_normal(v.shape) * 0.3).astype(np.float32))
        for name, v in trainable.items()
    }
    dac.load_weights(list(seeded.items()), strict=False)
    mx.eval(dac)
    mx.save_safetensors(str(OUT / "tiny_weights.safetensors"), seeded)
    config = {
        "sample_rate": 44100,
        "encoder_dim": 8,
        "encoder_rates": [2, 4],
        "latent_dim": 32,
        "decoder_dim": 16,
        "decoder_rates": [4, 2],
        "encoder_transformer_layers": [0, 2],
        "encoder_transformer": {
            "window_size": 512,
            "n_layer": 2,
            "n_head": 4,
            "dim": 32,
            "intermediate_size": 96,
            "head_dim": 8,
            "rope_base": 10000.0,
            "norm_eps": 1e-5,
        },
        "quantizer": {
            "input_dim": 32,
            "n_codebooks": 3,
            "codebook_size": 12,
            "semantic_codebook_size": 16,
            "codebook_dim": 4,
            "downsample_factor": [2, 2],
            "pre": {
                "window_size": 4,
                "n_layer": 2,
                "n_head": 4,
                "dim": 32,
                "intermediate_size": 48,
                "head_dim": 8,
                "rope_base": 10000.0,
                "norm_eps": 1e-5,
            },
            "post": {
                "window_size": 4,
                "n_layer": 2,
                "n_head": 4,
                "dim": 32,
                "intermediate_size": 48,
                "head_dim": 8,
                "rope_base": 10000.0,
                "norm_eps": 1e-5,
            },
        },
    }
    (OUT / "tiny_config.json").write_text(json.dumps(config, indent=2))
    (OUT / "weights_manifest.json").write_text(
        json.dumps({"tiny": [{"name": n, "shape": list(v.shape)} for n, v in sorted(seeded.items())]}, indent=2)
    )

    # One frame-length input (frame_length = hop 8 * downsample 4 = 32)
    # plus a two-frame one.
    rng = np.random.default_rng(603)
    for tag, n in (("a", 32), ("b", 64)):
        wave = (rng.standard_normal(n) * 0.2).astype(np.float32)
        save_npy(f"{tag}_wave.npy", wave)
        trace_files.append(f"{tag}_wave.npy")
        indices, _lens = dac.encode(mx.array(wave)[None])
        codes = np.asarray(indices)[0].astype(np.int32)  # (books, T)
        np.save(OUT / f"{tag}_codes.npy", np.ascontiguousarray(codes))
        traces[f"{tag}_codes_shape"] = list(codes.shape)
        trace_files.append(f"{tag}_codes.npy")
        wav_out, _al = dac.decode(mx.array(codes)[None], mx.array([codes.shape[-1]]))
        decoded = np.asarray(wav_out)[0, 0]
        save_npy(f"{tag}_decoded.npy", decoded)
        traces[f"{tag}_decoded_len"] = int(decoded.shape[0])
        trace_files.append(f"{tag}_decoded.npy")

    # Random-codes decode (books x frames).
    rng = np.random.default_rng(607)
    codes = np.stack([
        rng.integers(0, 16, size=2),  # semantic
        rng.integers(0, 12, size=2),
        rng.integers(0, 12, size=2),
        rng.integers(0, 12, size=2),
    ]).astype(np.int32)
    np.save(OUT / "rand_codes.npy", np.ascontiguousarray(codes))
    rand_wav, _ = dac.decode(mx.array(codes)[None], mx.array([2]))
    decoded = np.asarray(rand_wav)[0, 0]
    save_npy("rand_decoded.npy", decoded)
    traces["rand_codes_shape"] = list(codes.shape)
    traces["rand_decoded_len"] = int(decoded.shape[0])
    trace_files.extend(["rand_codes.npy", "rand_decoded.npy"])

    (OUT / "traces.json").write_text(json.dumps(traces, indent=2))

    files = [
        "tiny_config.json",
        "tiny_weights.safetensors",
        "weights_manifest.json",
        "traces.json",
        *trace_files,
    ]
    manifest = {
        "generator": "tools/gen_fish_s1_dac_fixtures.py",
        "reference_commit": REFERENCE_COMMIT,
        "mlx_version": mx.__version__,
        "files": {name: digest(OUT / name) for name in files},
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(files) + 1} fixture files to {OUT}")


if __name__ == "__main__":
    main()
