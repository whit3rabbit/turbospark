#!/usr/bin/env python3
"""Generate numerical reference samples from the pinned mlx-audio FireRedASR2 model."""

import json
import os
from pathlib import Path

import mlx.core as mx
import numpy as np
from scipy.io import wavfile

from mlx_audio.stt.utils import load_model


MODEL_DIR = Path(os.environ["TURBOSPARK_FIREREDASR2_DIR"])
ROOT = Path(__file__).resolve().parents[1]
AUDIO = ROOT / "testdata/qwen3_forced_aligner_reference.wav"
OUTPUT = ROOT / "testdata/fireredasr2_reference.json"


def sampled(values):
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    count = min(16, values.size)
    indices = np.linspace(0, values.size - 1, count, dtype=np.int64)
    return indices.tolist(), values[indices].tolist()


def stage(values):
    indices, samples = sampled(values)
    return {
        "shape": list(values.shape),
        "indices": indices,
        "values": samples,
    }


def main():
    # The checkpoint omits both deterministic positional-encoding buffers.
    model = load_model(MODEL_DIR, strict=False)

    sample_rate, audio = wavfile.read(AUDIO)
    if sample_rate != 16000 or audio.dtype != np.float32:
        raise ValueError("reference input must be mono float32 PCM at 16 kHz")
    audio = mx.array(audio, dtype=mx.float32)
    raw_features = model._extract_fbank(audio)
    means, inverse_std = model._cmvn
    features = (raw_features - means) * inverse_std
    mx.eval(features)

    x = mx.expand_dims(features, axis=0)
    right = mx.zeros((x.shape[0], model.encoder.input_preprocessor.context - 1, x.shape[2]))
    x = mx.concatenate([x, right], axis=1)
    subsampled = model.encoder.input_preprocessor(x)
    positions = model.encoder.positional_encoding(subsampled)
    block = model.encoder.layer_stack[0]
    ffn1_norm = block.ffn1.net_0(subsampled)
    ffn1_expand = block.ffn1.net_1(ffn1_norm)
    ffn1_silu = ffn1_expand * mx.sigmoid(ffn1_expand)
    ffn1_project = block.ffn1.net_4(ffn1_silu)
    ffn1_residual = subsampled + ffn1_project
    ffn1 = 0.5 * subsampled + 0.5 * ffn1_residual
    mhsa = block.mhsa(ffn1, ffn1, ffn1, positions)
    conv = block.conv(mhsa)
    ffn2 = 0.5 * conv + 0.5 * block.ffn2(conv)
    first = block.layer_norm(ffn2)
    encoded = first
    for layer in model.encoder.layer_stack[1:]:
        encoded = layer(encoded, positions)
    mx.eval(
        subsampled,
        ffn1_norm,
        ffn1_expand,
        ffn1_silu,
        ffn1_project,
        ffn1_residual,
        ffn1,
        mhsa,
        conv,
        ffn2,
        first,
        encoded,
    )

    output = model.generate(audio).text
    feature_stage = stage(features)
    feature_stage["all_values"] = np.asarray(features, dtype=np.float32).reshape(-1).tolist()
    feature_stage["raw_values"] = np.asarray(raw_features, dtype=np.float32).reshape(-1).tolist()
    data = {
        "repository": "mlx-community/FireRedASR2-AED-mlx",
        "revision": "f3212eacfa49b851130b97c63653c8e06ee09bdb",
        "mlx_audio_revision": "e1b19b9054bf163f5d812221a54fcc346f1890e9",
        "audio_sha256": "ea38d350100b5d532fbc3e3517499dbd9915b9b0890d9e819bae384fa1384bcb",
        "transcript": output,
        "stages": {
            "features": feature_stage,
            "subsampled": stage(subsampled),
            "first_block": stage(first),
            "first_block_components": {
                "ffn1_norm": stage(ffn1_norm),
                "ffn1_expand": stage(ffn1_expand),
                "ffn1_silu": stage(ffn1_silu),
                "ffn1_project": stage(ffn1_project),
                "ffn1_residual": stage(ffn1_residual),
                "ffn1": stage(ffn1),
                "mhsa": stage(mhsa),
                "conv": stage(conv),
                "ffn2": stage(ffn2),
                "layer_norm": stage(first),
            },
            "final_block": stage(encoded),
        },
    }
    OUTPUT.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"transcript": output, "fixture": str(OUTPUT)}, indent=2))


if __name__ == "__main__":
    main()
