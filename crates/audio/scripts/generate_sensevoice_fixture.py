#!/usr/bin/env python3
"""Generate a small inference witness from pinned MLX SenseVoice weights."""

import argparse
import hashlib
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import soundfile as sf

from mlx_audio.stt import load
from mlx_audio.stt.models.sensevoice.sensevoice import _compute_fbank


def selected_rows(value):
    mx.eval(value)
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 3:
        array = array[0]
    rows = sorted({0, min(1, array.shape[0] - 1), array.shape[0] // 2, array.shape[0] - 1})
    columns = sorted({0, min(1, array.shape[1] - 1), array.shape[1] // 2, array.shape[1] - 1})
    return {
        "shape": list(array.shape),
        "rows": rows,
        "columns": columns,
        "values": [[float(array[row, col]) for col in columns] for row in rows],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    model = load(str(args.model_dir))
    audio, sample_rate = sf.read(args.audio, dtype="float32", always_2d=False)
    if sample_rate != model.config.frontend_conf.fs:
        raise ValueError(f"expected {model.config.frontend_conf.fs} Hz audio, got {sample_rate} Hz")
    if audio.ndim == 2:
        audio = audio.mean(axis=1, dtype=np.float32)
    audio = mx.array(audio, dtype=mx.float32)
    fbank = _compute_fbank(
        audio,
        sample_rate=model.config.frontend_conf.fs,
        n_mels=model.config.frontend_conf.n_mels,
        frame_length_ms=model.config.frontend_conf.frame_length,
        frame_shift_ms=model.config.frontend_conf.frame_shift,
        window=model.config.frontend_conf.window,
    )
    features = model._extract_features(audio)
    features = features[None, :, :]

    textnorm_query, input_query = model._build_query(1, language="auto", use_itn=False)
    speech = mx.concatenate([textnorm_query, features], axis=1)
    speech = mx.concatenate([input_query, speech], axis=1)
    encoder_input = speech * (model.config.encoder_conf.output_size**0.5)
    encoder_input = model.encoder.embed(encoder_input)
    mx.eval(encoder_input)

    hidden = encoder_input
    hidden = model.encoder.encoders0[0](hidden)
    first_block = hidden
    for layer in model.encoder.encoders:
        hidden = layer(hidden)
    hidden = model.encoder.after_norm(hidden)
    encoder_stack = hidden
    for layer in model.encoder.tp_encoders:
        hidden = layer(hidden)
    hidden = model.encoder.tp_norm(hidden)
    encoder_output = hidden
    log_probs = model.ctc_lo(encoder_output)
    mx.eval(log_probs)

    log_probs = log_probs[0]
    rich = model._extract_rich_info(log_probs[:4])
    token_ids, text = model._greedy_ctc_decode(log_probs[4:])

    fixture = {
        "repository": "mlx-community/SenseVoiceSmall",
        "revision": "8ddd966bd96243cff196422f81f0c5d955814792",
        "mlx_audio_version": "0.5.7",
        "audio_input": "soundfile float32 passed directly to the MLX model",
        "audio_sha256": hashlib.sha256(args.audio.read_bytes()).hexdigest(),
        "transcript": text,
        "language": rich.get("language"),
        "emotion": rich.get("emotion"),
        "event": rich.get("event"),
        "token_ids": token_ids,
        "ctc_argmax": mx.argmax(log_probs[4:], axis=-1).tolist(),
        "cmvn_means": np.asarray(model._cmvn_means, dtype=np.float32).tolist(),
        "cmvn_istd": np.asarray(model._cmvn_istd, dtype=np.float32).tolist(),
        "fbank": np.asarray(fbank, dtype=np.float32).tolist(),
        "frontend_features": np.asarray(features[0], dtype=np.float32).tolist(),
        "positioned_encoder_input": selected_rows(encoder_input),
        "first_block": selected_rows(first_block),
        "encoder_stack": selected_rows(encoder_stack),
        "encoder_output": selected_rows(encoder_output),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, separators=(",", ":")) + "\n")
    print(json.dumps({"output": str(args.output), "transcript": text, "feature_shape": list(features.shape)}))


if __name__ == "__main__":
    main()
