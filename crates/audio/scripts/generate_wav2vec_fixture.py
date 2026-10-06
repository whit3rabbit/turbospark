#!/usr/bin/env python3
"""Generate a Wav2Vec2 base inference witness from the pinned checkpoint.

The pinned mlx-audio 0.5.7 wav2vec source exposes a bare Wav2Vec2Model
backbone (its sanitize drops lm_head), so this generator applies the
checkpoint's own CTC head on top of the backbone hidden states, exactly like
transformers Wav2Vec2ForCTC does, and records stage tensors plus the greedy
CTC transcript for the Rust parity tests.
"""

import argparse
import hashlib
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import soundfile as sf

from mlx_audio.stt.models.wav2vec.wav2vec import Wav2Vec2Model as Model
from mlx_audio.stt.models.wav2vec.wav2vec import ModelConfig

SOURCE_COMMIT = "e1b19b9054bf163f5d812221a54fcc346f1890e9"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_spots(array: np.ndarray) -> dict:
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


def spots_of(value) -> dict:
    mx.eval(value)
    return selected_spots(np.asarray(value, dtype=np.float32))


def ctc_greedy(logits: np.ndarray, vocab: dict[str, int]) -> tuple[list[int], str]:
    id_to_token = {index: token for token, index in vocab.items()}
    blank = vocab["<pad>"]
    tokens: list[int] = []
    previous = None
    for row in logits:
        token = int(np.argmax(row))
        if token != previous and token != blank:
            tokens.append(token)
        previous = token
    text = "".join(id_to_token[token] for token in tokens).replace("|", " ").strip()
    return tokens, text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()

    audio, sample_rate = sf.read(args.audio, dtype="float32", always_2d=False)
    if sample_rate != 16000:
        raise ValueError(f"expected 16 kHz audio, got {sample_rate} Hz")
    if audio.ndim == 2:
        audio = audio.mean(axis=1, dtype=np.float32)

    # HF Wav2Vec2 feature extractor convention: zero mean, unit variance with
    # the epsilon inside the square root (feature_extractor.py line 165).
    mean = float(audio.mean())
    variance = float(audio.var())
    normalized = (audio - mean) / np.sqrt(variance + 1e-7)

    config_path = args.model_dir / "config.json"
    config = ModelConfig.from_dict(json.loads(config_path.read_text()))
    model = Model(config)

    weights = mx.load(str(args.model_dir / "model.safetensors"), format="safetensors")
    weights = model.sanitize(weights)
    model.load_weights(list(weights.items()))
    model.eval()
    mx.eval(model.parameters())

    waveform = mx.array(normalized, dtype=mx.float32)
    batched = waveform[None]

    conv0 = model.feature_extractor.conv_layers[0](batched[:, None])
    features = model.feature_extractor(batched)
    features_rows = features.transpose(0, 2, 1)
    projection, norm_features = model.feature_projection(features_rows)
    encoder_hidden = model.encoder(projection, output_hidden_states=True)
    hidden_states = []

    for index, layer_hidden in enumerate(encoder_hidden.hidden_states):
        if index in (0, len(model.encoder.layers) // 2, len(model.encoder.layers) - 1):
            hidden_states.append((f"encoder_layer_{index}", layer_hidden))
    hidden_states.append(("encoder_final", encoder_hidden.last_hidden_state))

    for index, layer_hidden in enumerate(encoder_hidden.hidden_states):
        if index in (0, len(model.encoder.layers) // 2, len(model.encoder.layers) - 1):
            hidden_states.append((f"encoder_layer_{index}", layer_hidden))
    hidden_states.append(("encoder_final", encoder_hidden.last_hidden_state))

    lm = mx.load(str(args.model_dir / "model.safetensors"), format="safetensors")
    lm_weight = lm["lm_head.weight"]
    lm_bias = lm["lm_head.bias"]
    logits = encoder_hidden.last_hidden_state @ lm_weight.T + lm_bias
    mx.eval(logits)

    logits_np = np.asarray(logits[0], dtype=np.float32)
    tokens, text = ctc_greedy(logits_np, json.loads((args.model_dir / "vocab.json").read_text()))

    document = {
        "provenance": {
            "source": "mlx-audio wav2vec at commit " + SOURCE_COMMIT,
            "repository": "facebook/wav2vec2-base-960h",
            "revision": args.revision,
            "audio": args.audio.name,
            "audio_sha256": sha256_file(args.audio),
            "normalization": "(x - mean) / sqrt(var + 1e-7)",
            "head": "checkpoint lm_head applied to backbone hidden states",
        },
        "normalized_audio": {
            "shape": [1, int(normalized.shape[0])],
            "first": [float(value) for value in normalized[:4]],
            "last": [float(value) for value in normalized[-4:]],
        },
        "conv0_group_norm": spots_of(conv0),
        "feature_extractor": spots_of(features_rows),
        "feature_projection": spots_of(projection),
        "hidden_states": [
            {"name": name, "spots": spots_of(value)} for name, value in hidden_states
        ],
        "ctc_logits": spots_of(logits[0]),
        "greedy_token_ids": tokens,
        "transcript": text,
    }
    args.output.write_text(json.dumps(document, indent=1))
    print(f"wrote {args.output}")
    print(f"transcript: {text!r}")
    print(f"tokens: {tokens}")


if __name__ == "__main__":
    main()
