#!/usr/bin/env python3
"""Write the small Granite5 frontend golden from pinned mlx-audio source."""

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_audio.stt.models.granite_speech5_ctc.granite_speech5 import compute_features


def main() -> None:
    samples = np.asarray([((i * 7 % 31) - 15) / 32.0 for i in range(1280)], dtype=np.float32)
    features = compute_features(mx.array(samples, dtype=mx.float32), num_mel_bins=80)
    mx.eval(features)
    output = Path(__file__).resolve().parents[1] / "testdata" / "granite5_features.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(features.tolist(), separators=(",", ":")) + "\n")
    print(f"wrote {features.shape} features to {output}")


if __name__ == "__main__":
    main()
