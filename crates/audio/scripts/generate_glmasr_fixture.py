#!/usr/bin/env python3
"""Generate GLM-ASR stage samples from the pinned mlx-audio reference."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from huggingface_hub import snapshot_download
from mlx_audio.audio_io import read
from mlx_audio.stt.utils import load


REPOSITORY = "mlx-community/GLM-ASR-Nano-2512-4bit"
REVISION = "35553fa5bebfcc3ece3ce7d47b98827cb0ac9eef"
SOURCE_COMMIT = "e1b19b9054bf163f5d812221a54fcc346f1890e9"
DEFAULT_AUDIO = Path(__file__).resolve().parents[1] / "testdata/qwen3_forced_aligner_reference.wav"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "testdata/glmasr_reference.json"


def snapshot(value):
    mx.eval(value)
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    count = min(16, array.size)
    indices = np.linspace(0, array.size - 1, count, dtype=int).tolist()
    return {
        "shape": list(value.shape),
        "indices": indices,
        "values": [float(array[index]) for index in indices],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=os.environ.get("TURBOSPARK_GLMASR_DIR"))
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.model_dir is None:
        parser.error("pass --model-dir or set TURBOSPARK_GLMASR_DIR to the pinned snapshot")

    model_dir = args.model_dir
    if not model_dir.exists():
        model_dir = Path(
            snapshot_download(
                repo_id=REPOSITORY,
                revision=REVISION,
                cache_dir=str(model_dir),
            )
        )
    model = load(str(model_dir))
    audio, sample_rate = read(
        str(args.audio), dtype="float32", sample_rate=16_000, nchannels=1
    )
    if sample_rate != 16_000:
        raise RuntimeError(f"expected 16000 Hz audio, got {sample_rate}")

    mel = model._preprocess_audio(audio)
    encoder = model.audio_encoder.whisper
    hidden = nn.gelu(encoder.conv1(mel))
    conv1 = snapshot(hidden)
    hidden = nn.gelu(encoder.conv2(hidden))
    conv2 = snapshot(hidden)
    first_layer = None
    for index, layer in enumerate(encoder.layers):
        hidden = layer(hidden)
        if index == 0:
            first_layer = snapshot(hidden)
    encoder_last = snapshot(hidden)

    normalized = model.audio_encoder.layer_norm(hidden)
    merge_factor = model.config.merge_factor
    batch, sequence, width = normalized.shape
    merged_rows = (sequence - merge_factor) // merge_factor + 1
    max_rows = model.config.max_whisper_length // merge_factor
    merged_rows = min(merged_rows, max_rows)
    chunks = []
    for index in range(merged_rows):
        start = index * merge_factor
        chunk = normalized[:, start : start + merge_factor, :]
        chunks.append(chunk.reshape(batch, -1))
    merged = mx.stack(chunks, axis=1)
    adapted = model.audio_encoder.adapting(merged)

    output = model.generate(audio, max_tokens=128, temperature=0.0)
    payload = {
        "repository": REPOSITORY,
        "revision": REVISION,
        "mlx_audio_version": "0.5.7",
        "mlx_audio_commit": SOURCE_COMMIT,
        "audio_sha256": hashlib.sha256(args.audio.read_bytes()).hexdigest(),
        "transcript": output.text,
        "stages": {
            "mel": snapshot(mel),
            "conv1": conv1,
            "conv2": conv2,
            "encoder_first": first_layer,
            "encoder_last": encoder_last,
            "normalized": snapshot(normalized),
            "merged": snapshot(merged),
            "adapted": snapshot(adapted),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    print(f"transcript: {output.text}")


if __name__ == "__main__":
    main()
