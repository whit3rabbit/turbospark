#!/usr/bin/env python3
"""Generate a tiny fixture from the pinned mlx-audio Fun-ASR-Nano profile."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import soundfile as sf

from mlx_audio.stt.utils import load
from mlx_audio.stt.models.fun_asr_nano.audio import prepare_audio


def sample_stage(value: mx.array) -> dict:
    matrix = np.asarray(value[0].astype(mx.float32))
    rows, columns = matrix.shape
    row_ids = sorted({0, min(1, rows - 1), rows // 2, rows - 1})
    column_ids = sorted({0, min(1, columns - 1), columns // 2, columns - 1})
    return {
        "shape": [rows, columns],
        "rows": row_ids,
        "columns": column_ids,
        "values": [[float(matrix[row, col]) for col in column_ids] for row in row_ids],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    waveform, sample_rate = sf.read(args.audio, dtype="float32")
    if sample_rate != 16000 or waveform.ndim != 1:
        raise SystemExit("fixture audio must be mono 16 kHz PCM")
    model = load(str(args.model_dir), lazy=False, strict=False)
    feats, speech_lengths, fake_token_len = prepare_audio(
        np.asarray(waveform), model.config.frontend_conf
    )
    encoder, encoder_lengths = model.audio_encoder(feats, speech_lengths)
    adaptor, adaptor_lengths = model.audio_adaptor(encoder, encoder_lengths)
    result = model.generate(np.asarray(waveform), max_tokens=32, temperature=0.0)
    fixture = {
        "profile": "mlx-community/Fun-ASR-Nano-2512",
        "revision": "a7bc96fceaafce39ed6748e0c0fa9a9508b67f86",
        "source_commit": "e1b19b9054bf163f5d812221a54fcc346f1890e9",
        "sample_rate": int(sample_rate),
        "transcript": result.text,
        "feature_frames": int(feats.shape[1]),
        "fake_audio_tokens": int(fake_token_len),
        "encoder_frames": int(encoder_lengths[0].item()),
        "adaptor_frames": int(adaptor_lengths[0].item()),
        "lfr_features": sample_stage(feats),
        "audio_encoder": sample_stage(encoder),
        "audio_adaptor": sample_stage(adaptor),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8")
    print(f"transcript: {result.text}")
    print(f"saved fixture: {args.output}")


if __name__ == "__main__":
    main()
