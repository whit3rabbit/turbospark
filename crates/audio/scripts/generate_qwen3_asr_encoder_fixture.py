#!/usr/bin/env python3
"""Generate Qwen3-ASR audio-tower outputs from the pinned MLX checkpoint."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_audio.stt import load


SAMPLE_RATE = 16_000
SAMPLE_COUNT = 44_720
SELECTED_ROWS = [0, 1, 10, 35]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        default=os.environ.get("TURBOSPARK_QWEN3_ASR_DIR"),
        required=os.environ.get("TURBOSPARK_QWEN3_ASR_DIR") is None,
    )
    args = parser.parse_args()
    model_dir = Path(args.model_dir)
    tau = 2.0 * np.pi
    seconds = np.arange(SAMPLE_COUNT, dtype=np.float64) / SAMPLE_RATE
    samples = (
        0.1 * np.sin(tau * 440.0 * seconds)
        + 0.03 * np.sin(tau * 997.0 * seconds)
    ).astype(np.float32)

    model = load(str(model_dir))
    input_features, attention_mask, audio_tokens = model._preprocess_audio(samples)
    encoded = model.get_audio_features(input_features, attention_mask)
    mx.eval(encoded)
    encoded_np = np.asarray(encoded, dtype=np.float32)
    if encoded_np.ndim != 2 or encoded_np.shape[1] != 1024:
        raise ValueError(f"unexpected audio tower output shape: {encoded_np.shape}")
    if max(SELECTED_ROWS) >= encoded_np.shape[0]:
        raise ValueError(f"audio tower output is too short: {encoded_np.shape}")

    fixture = {
        "reference": "mlx-audio 0.5.7 Qwen3ASRModel.get_audio_features",
        "mlx_audio_commit": "e1b19b9054bf163f5d812221a54fcc346f1890e9",
        "model_repo": "mlx-community/Qwen3-ASR-0.6B-8bit",
        "model_revision": "89e96d92ba34aca20b3e29fb10cc284097d1219f",
        "sample_rate": SAMPLE_RATE,
        "sample_count": SAMPLE_COUNT,
        "feature_frames": int(input_features.shape[-1]),
        "attention_mask_sum": int(np.asarray(attention_mask).sum()),
        "audio_token_count": int(audio_tokens),
        "output_shape": list(encoded_np.shape),
        "selected_rows": SELECTED_ROWS,
        "selected_values": [encoded_np[index].tolist() for index in SELECTED_ROWS],
    }
    target = Path(__file__).resolve().parents[1] / "testdata/qwen3_asr_encoder.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(fixture, indent=2) + "\n", encoding="ascii")
    print(f"wrote {target}")
    print(f"sha256 {hashlib.sha256(target.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
