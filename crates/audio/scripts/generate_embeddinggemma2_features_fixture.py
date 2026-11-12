#!/usr/bin/env python3
"""Write the small EmbeddingGemma 2 frontend golden from the pinned
transformers Gemma4AudioFeatureExtractor.

Pinned reference: huggingface/transformers @
28034545c35acac05793695bd5c878f66e6b544e
(src/transformers/models/gemma4/feature_extraction_gemma4.py), configured
from google/embeddinggemma-2 preprocessor_config.json. Install with
`pip install numpy "transformers @ git+https://github.com/huggingface/transformers@28034545c35acac05793695bd5c878f66e6b544e"`.

The 4000-sample input is not a multiple of the extractor's default
`pad_to_multiple_of=128`, so the golden also pins the right-padding and the
frame-aware mask (the last frame is masked and zeroed).
"""

import json
import math
from pathlib import Path

import numpy as np
from transformers.models.gemma4.feature_extraction_gemma4 import Gemma4AudioFeatureExtractor


def signal() -> np.ndarray:
    values = []
    for i in range(4000):
        tone = 0.4 * math.sin(2.0 * math.pi * 440.0 * i / 16000.0)
        chirp = 0.2 * math.sin(2.0 * math.pi * (200.0 + 0.5 * i) * i / 16000.0)
        ramp = ((i * 7 % 31) - 15) / 160.0
        values.append(tone + chirp + ramp)
    return np.asarray(values, dtype=np.float32)


def main() -> None:
    extractor = Gemma4AudioFeatureExtractor(
        feature_size=128,
        sampling_rate=16000,
        frame_length_ms=20.0,
        hop_length_ms=10.0,
        min_frequency=0.0,
        max_frequency=8000.0,
        preemphasis=0.0,
        fft_overdrive=False,
        dither=0.0,
        input_scale_factor=1.0,
        mel_floor=1e-3,
    )
    assert extractor.frame_length == 320 and extractor.hop_length == 160
    assert extractor.fft_length == 512
    # Pass a one-item batch: at the pinned commit an unbatched 1-D numpy
    # array reaches `_extract_spectrogram` transposed and fails to squeeze.
    batch = extractor([signal()], return_tensors="np")
    features = batch["input_features"][0]
    mask = batch["input_features_mask"][0]
    output = Path(__file__).resolve().parents[1] / "testdata" / "embeddinggemma2_features.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "samples": 4000,
        "frames": int(features.shape[0]),
        "mask": [bool(value) for value in mask],
        "features": [[float(value) for value in row] for row in features],
    }
    output.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    print(f"wrote {features.shape} features, {int(mask.sum())} valid, to {output}")


if __name__ == "__main__":
    main()
