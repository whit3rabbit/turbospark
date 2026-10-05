#!/usr/bin/env python3
"""Generate the pinned 128-band Qwen3-ASR feature frontend fixture."""

import hashlib
import json
from pathlib import Path

import numpy as np
from transformers import WhisperFeatureExtractor


SAMPLE_RATE = 16_000
SAMPLE_COUNT = 44_720
SELECTED_FRAMES = [0, 1, 10, 50, 139, 278]


def main() -> None:
    indices = np.arange(SAMPLE_COUNT, dtype=np.float64)
    seconds = indices / SAMPLE_RATE
    tau = 2.0 * np.pi
    samples = (
        0.1 * np.sin(tau * 440.0 * seconds)
        + 0.03 * np.sin(tau * 997.0 * seconds)
    ).astype(np.float32)
    extractor = WhisperFeatureExtractor(feature_size=128, sampling_rate=SAMPLE_RATE)
    result = extractor(
        samples,
        sampling_rate=SAMPLE_RATE,
        return_attention_mask=True,
        truncation=False,
        padding=True,
        return_tensors="np",
    )
    features = result["input_features"][0]
    fixture = {
        "reference": "Transformers WhisperFeatureExtractor via mlx-audio 0.5.7",
        "mlx_audio_commit": "e1b19b9054bf163f5d812221a54fcc346f1890e9",
        "model_repo": "mlx-community/Qwen3-ASR-0.6B-8bit",
        "model_revision": "89e96d92ba34aca20b3e29fb10cc284097d1219f",
        "sample_rate": SAMPLE_RATE,
        "sample_count": SAMPLE_COUNT,
        "feature_shape": list(features.shape),
        "attention_mask_sum": int(result["attention_mask"][0].sum()),
        "selected_frames": SELECTED_FRAMES,
        "selected_values": [features[:, frame].tolist() for frame in SELECTED_FRAMES],
    }
    target = Path(__file__).resolve().parents[1] / "testdata/qwen3_asr_features.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(fixture, indent=2) + "\n", encoding="ascii")
    print(f"wrote {target}")
    print(f"sha256 {hashlib.sha256(target.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
