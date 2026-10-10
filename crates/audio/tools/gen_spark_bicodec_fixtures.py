#!/usr/bin/env python3
"""Regenerate the Spark BiCodec decode-direction Rust parity fixtures.

Normative reference: mlx-audio at commit
e1b19b9054bf163f5d812221a54fcc346f1890e9 (mlx_audio/tts/models/spark,
mlx 0.32.3). The reference BiCodec modules are built directly at tiny
seeded dims and driven through the exact `detokenize` call the TTS
family makes: quantizer.detokenize code lookup, FSQ global-token
speaker path, conditioned Vocos prenet, and the DAC-style WaveGenerator.
The Wav2Vec2 feature extractor, the BiCodec encoder, and the ECAPA
speaker encoder belong to the unported encode direction and are built
only because the reference module tree requires them.

Run from inside the mlx-audio checkout root so `mlx_audio` imports from
the pinned source, with the project venv interpreter:

    cd ../mlx-audio
    ../.venv-mlxaudio/bin/python \
        ../turbospark/crates/audio/tools/gen_spark_bicodec_fixtures.py

Outputs into crates/audio/testdata/spark_bicodec/:

- detokenize/config.yaml + model.safetensors
    The seeded tiny BiCodec (quantizer input 16 over a 64-entry x
    4-dim codebook, 4-dim FSQ levels [3, 3, 3, 3] with 4 global
    tokens, 2-layer Vocos prenet at width 12 conditioned on a
    12-dim speaker vector, WaveGenerator channels 8 with two stride-2
    blocks for a 4x upsample). Weights are stored F32 so the gate pins
    the arithmetic rather than weight rounding.
- detokenize/traces.json
    Detokenize cases over seeded token sets plus the reference
    empty-semantic placeholder branch, with recorded waveforms.
- manifest.json
    File list with SHA-256 digests and generator metadata.

The script is deterministic: fixed seeds, no network access.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten, tree_map, tree_unflatten

# Prefer the checkout the script is run from over any installed copy.
sys.path.insert(0, str(Path.cwd()))

from mlx_audio.tts.models.spark.bicodec import BiCodec
from mlx_audio.tts.models.spark.modules.encoder_decoder.feat_decoder import Decoder
from mlx_audio.tts.models.spark.modules.encoder_decoder.feat_encoder import Encoder
from mlx_audio.tts.models.spark.modules.encoder_decoder.wave_generator import (
    WaveGenerator,
)
from mlx_audio.tts.models.spark.modules.residual import FactorizedVectorQuantize
from mlx_audio.tts.models.spark.modules.speaker.speaker_encoder import SpeakerEncoder

OUT = Path("../turbospark/crates/audio/testdata/spark_bicodec")
REFERENCE_COMMIT = "e1b19b9054bf163f5d812221a54fcc346f1890e9"

SEED = 11
TOKEN_SEED = 23
TOLERANCE = 2e-4

CONFIG = {
    "mel_params": {
        "sample_rate": 16000,
        "n_fft": 64,
        "win_length": 64,
        "hop_length": 32,
        "mel_fmin": 0.0,
        "mel_fmax": None,
        "num_mels": 8,
    },
    "encoder": {
        "input_channels": 4,
        "vocos_dim": 12,
        "vocos_intermediate_dim": 24,
        "vocos_num_layers": 1,
        "out_channels": 16,
        "sample_ratios": [1, 1],
    },
    "quantizer": {
        "input_dim": 16,
        "codebook_size": 64,
        "codebook_dim": 4,
        "commitment": 0.02,
        "codebook_loss_weight": 2.0,
    },
    "decoder": {
        "input_channel": 12,
        "channels": 8,
        "rates": [2, 2],
        "kernel_sizes": [4, 4],
        "d_out": 1,
    },
    "speaker_encoder": {
        "input_dim": 8,
        "out_dim": 12,
        "latent_dim": 6,
        "token_num": 4,
        "fsq_levels": [3, 3, 3, 3],
        "fsq_num_quantizers": 1,
    },
    "prenet": {
        "input_channels": 16,
        "vocos_dim": 12,
        "vocos_intermediate_dim": 24,
        "vocos_num_layers": 2,
        "out_channels": 12,
        "condition_dim": 12,
        "sample_ratios": [1, 1],
        "use_tanh_at_final": False,
    },
    "postnet": {
        "input_channels": 12,
        "vocos_dim": 12,
        "vocos_intermediate_dim": 24,
        "vocos_num_layers": 1,
        "out_channels": 12,
        "condition_dim": 12,
        "sample_ratios": [1, 1],
        "use_tanh_at_final": False,
    },
}

CODEBOOK_SIZE = CONFIG["quantizer"]["codebook_size"]
TOKEN_NUM = CONFIG["speaker_encoder"]["token_num"]
FSQ_CARDINALITY = int(np.prod(CONFIG["speaker_encoder"]["fsq_levels"]))
FRAMES = 6


def floats(arr) -> list:
    return np.asarray(arr, dtype=np.float32).ravel().tolist()


def build_model(seed: int):
    """The `load_from_checkpoint` construction at tiny dims with seeded
    weights. The module tree holds the post-sanitize layouts, so the
    safetensors export matches what a converted checkpoint stores."""
    mx.random.seed(seed)
    config = CONFIG
    encoder = Encoder(**config["encoder"])
    quantizer = FactorizedVectorQuantize(**config["quantizer"])
    prenet = Decoder(**config["prenet"])
    postnet = Decoder(**config["postnet"])
    decoder = WaveGenerator(**config["decoder"])
    speaker_encoder = SpeakerEncoder(**config["speaker_encoder"])

    model = BiCodec(
        mel_params=config["mel_params"],
        encoder=encoder,
        decoder=decoder,
        quantizer=quantizer,
        speaker_encoder=speaker_encoder,
        prenet=prenet,
        postnet=postnet,
    )

    def fill(weight):
        return mx.random.uniform(low=-0.35, high=0.35, shape=weight.shape)

    flat = dict(tree_flatten(tree_map(fill, model.parameters())))
    for name, weight in list(flat.items()):
        if name.endswith(".alpha"):
            flat[name] = mx.random.uniform(low=0.5, high=1.5, shape=weight.shape)
    # F32 storage: the fixture pins the arithmetic, not f16 weight
    # rounding (the tiny checkpoint stays small; real checkpoints are
    # a real-hardware gate).
    stored = {name: value.astype(mx.float32) for name, value in flat.items()}
    model.load_weights(list(stored.items()), strict=True)
    model.update(tree_unflatten({k: v.astype(mx.float32) for k, v in stored.items()}))
    mx.eval(model)
    return model, stored


def detokenize(model, semantic, global_tokens):
    """The family call shape: semantic [1, T], global [1, 1, token_num]."""
    semantic_arr = mx.array(semantic, dtype=mx.int32)[None, ...]
    global_arr = mx.array(global_tokens, dtype=mx.int32)[None, None, :]
    return np.asarray(model.detokenize(semantic_arr, global_arr), dtype=np.float32).ravel()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    model, stored = build_model(SEED)

    rng = np.random.default_rng(TOKEN_SEED)
    cases = []
    for index in range(3):
        semantic = [int(v) for v in rng.integers(0, CODEBOOK_SIZE, size=FRAMES)]
        global_tokens = [int(v) for v in rng.integers(0, FSQ_CARDINALITY, size=TOKEN_NUM)]
        waveform = detokenize(model, semantic, global_tokens)
        assert waveform.shape[0] == FRAMES * 4, waveform.shape
        cases.append(
            {
                "name": f"case_{index}",
                "semantic_tokens": semantic,
                "global_tokens": global_tokens,
                "waveform": floats(waveform),
            }
        )

    # The reference empty-semantic placeholder branch crashes upstream
    # (its (1, input_dim, 1) shape never survives the prenet transpose),
    # so there is no golden to pin; the Rust port rejects empty input.

    case_dir = OUT / "detokenize"
    case_dir.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(case_dir / "model.safetensors"), stored)

    yaml_lines = ["audio_tokenizer:"]

    def emit(section, values):
        yaml_lines.append(f"  {section}:")
        for key, value in values.items():
            if isinstance(value, list):
                rendered = "[" + ", ".join(repr(v) for v in value) + "]"
            elif value is None:
                rendered = "null"
            elif isinstance(value, bool):
                rendered = "True" if value else "False"
            else:
                rendered = repr(value)
            yaml_lines.append(f"    {key}: {rendered}")

    for section, values in CONFIG.items():
        emit(section, values)
    (case_dir / "config.yaml").write_text("\n".join(yaml_lines) + "\n")

    traces = {
        "reference_commit": REFERENCE_COMMIT,
        "tolerance": TOLERANCE,
        "upsample_rate": 4,
        "cases": cases,
    }
    (case_dir / "traces.json").write_text(json.dumps(traces))

    files = {}
    for entry in sorted(OUT.rglob("*")):
        if entry.is_file() and entry.name != "manifest.json":
            files[str(entry.relative_to(OUT))] = hashlib.sha256(entry.read_bytes()).hexdigest()
    manifest = {
        "reference_commit": REFERENCE_COMMIT,
        "mlx_version": mx.__version__,
        "generator": "crates/audio/tools/gen_spark_bicodec_fixtures.py",
        "seed": SEED,
        "upsample_rate": 4,
        "files": files,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote fixtures to {OUT}")


if __name__ == "__main__":
    main()
