#!/usr/bin/env python3
"""Generate a small alignment fixture from the pinned MLX checkpoint."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import soundfile as sf
from mlx_audio.stt import load


REPOSITORY = "mlx-community/Qwen3-ForcedAligner-0.6B-8bit"
REVISION = "0e1a68e91d815300c7c9754b2a7639378b23db15"
TRANSCRIPT = "The quick brown fox jumps over the lazy dog."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        default=os.environ.get("TURBOSPARK_QWEN3_FORCED_ALIGNER_DIR"),
        required=os.environ.get("TURBOSPARK_QWEN3_FORCED_ALIGNER_DIR") is None,
    )
    parser.add_argument(
        "--audio-path",
        default=os.environ.get("TURBOSPARK_QWEN3_ASR_WAV"),
        required=os.environ.get("TURBOSPARK_QWEN3_ASR_WAV") is None,
    )
    args = parser.parse_args()
    audio_path = Path(args.audio_path)
    samples, sample_rate = sf.read(audio_path, dtype="float32")
    if sample_rate != 16_000 or samples.ndim != 1:
        raise ValueError("fixture audio must be mono 16 kHz PCM")

    model = load(str(args.model_dir), model_type="qwen3_forced_aligner")
    result = model.generate(audio=samples, text=TRANSCRIPT, language="English")
    fixture = {
        "reference": "mlx-audio 0.5.7 Qwen3-ForcedAligner.generate",
        "mlx_audio_commit": "e1b19b9054bf163f5d812221a54fcc346f1890e9",
        "model_repo": REPOSITORY,
        "model_revision": REVISION,
        "sample_rate": sample_rate,
        "sample_count": len(samples),
        "audio_sha256": hashlib.sha256(audio_path.read_bytes()).hexdigest(),
        "transcript": TRANSCRIPT,
        "language": "English",
        "items": [
            {
                "text": item.text,
                "start_time": item.start_time,
                "end_time": item.end_time,
            }
            for item in result
        ],
    }
    target = Path(__file__).resolve().parents[1] / "testdata/qwen3_forced_aligner.json"
    target.write_text(json.dumps(fixture, indent=2) + "\n", encoding="ascii")
    print(f"wrote {target}")
    print(f"sha256 {hashlib.sha256(target.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
