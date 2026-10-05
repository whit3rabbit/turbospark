#!/usr/bin/env python3
"""Generate a small Wav2Vec2/MMS inference witness from the pinned MLX model."""

import argparse
import hashlib
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_audio.stt.utils import load, load_audio


def selected_values(value):
    mx.eval(value)
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 3:
        array = array[0]
    rows = sorted(
        {0, min(1, array.shape[0] - 1), array.shape[0] // 2, array.shape[0] - 1}
    )
    columns = sorted(
        {
            0,
            min(1, array.shape[1] - 1),
            min(7, array.shape[1] - 1),
            min(31, array.shape[1] - 1),
            min(127, array.shape[1] - 1),
            array.shape[1] // 2,
            array.shape[1] - 1,
        }
    )
    return {
        "shape": list(array.shape),
        "rows": rows,
        "columns": columns,
        "values": [[float(array[row, column]) for column in columns] for row in rows],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    # The HF config says "wav2vec2", which the STT registry routes to a
    # nonexistent standalone module. The upstream MMS class is selected by
    # its explicit family override.
    model = load(str(args.model_dir), model_type="mms")
    audio = load_audio(str(args.audio), sr=16_000, dtype=mx.float32)
    audio = mx.expand_dims(audio, axis=0)
    stages = {"audio_input": selected_values(audio)}
    audio = (audio - mx.mean(audio, axis=-1, keepdims=True)) / (
        mx.std(audio, axis=-1, keepdims=True) + 1e-7
    )
    stages["normalized_audio"] = selected_values(audio)

    wav2vec = model.wav2vec2
    features = mx.expand_dims(audio, axis=1)
    for index, layer in enumerate(wav2vec.feature_extractor.conv_layers):
        convolved = layer.conv(features.swapaxes(-2, -1))
        normalized = layer.layer_norm(convolved)
        activated = layer.activation(normalized)
        stages[f"feature_conv_{index}_raw"] = selected_values(
            convolved
        )
        stages[f"feature_conv_{index}_norm"] = selected_values(
            normalized
        )
        features = activated.swapaxes(-2, -1)
        stages[f"feature_conv_{index}"] = selected_values(features.transpose(0, 2, 1))
    feature_rows = features.transpose(0, 2, 1)
    projected, _ = wav2vec.feature_projection(feature_rows)
    stages["feature_extractor"] = selected_values(feature_rows)
    stages["feature_projection"] = selected_values(projected)
    positional = wav2vec.encoder.pos_conv_embed(projected)
    stages["positional_conv"] = selected_values(positional)
    hidden = projected + positional
    capture_layers = {
        0,
        len(wav2vec.encoder.layers) // 2 - 1,
        len(wav2vec.encoder.layers) - 1,
    }
    for index, layer in enumerate(wav2vec.encoder.layers):
        hidden = layer(hidden, attention_mask=None)[0]
        if index in capture_layers:
            stages[f"encoder_layer_{index}"] = selected_values(hidden)

    hidden = wav2vec.encoder.layer_norm(hidden)
    stages["encoder_final_norm"] = selected_values(hidden)
    logits = model.lm_head(hidden)
    stages["ctc_logits"] = selected_values(logits)
    decoded = model._ctc_decode(logits)
    text = model._tokens_to_text(decoded[0]).strip()

    fixture = {
        "repository": "facebook/mms-1b-fl102",
        "revision": "d483345545bea550895b1aa0c6ba40236b9f1e22",
        "language_adapter": "adapter.eng.safetensors",
        "audio_sha256": hashlib.sha256(args.audio.read_bytes()).hexdigest(),
        "transcript": text,
        "stages": stages,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, indent=2) + "\n")
    print(
        json.dumps(
            {"output": str(args.output), "transcript": text, "stage_count": len(stages)}
        )
    )


if __name__ == "__main__":
    main()
