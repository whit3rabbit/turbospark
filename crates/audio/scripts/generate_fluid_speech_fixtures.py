#!/usr/bin/env python3
"""Write the Fluid Speech Pro stage goldens from the publisher's MLX loader.

Runs `fluid_speech_mlx.py` (FluidVoice-Labs/fluid-speech-pro-mlx at
aeb1d4e43487586b15d86c0a1dd4511e1cf6b7a0, on parakeet-mlx 0.7.0 and mlx
0.32.5) with the 8-bit weights in float32 compute, on a committed clip with
fillers and repeated words so the clean and verbatim styles differ. It saves
the NeMo mel, the FrameCleaner input and output, the encoder output for both
styles, and both decoded token sequences under
crates/audio/testdata/parakeet_fluid/ for the Rust parity tests.

The reference loader is not vendored: pass the directory that holds a copy of
the pinned `fluid_speech_mlx.py` (read it before running). Tests must not
download assets; this regeneration script owns that step.

Usage (from a directory outside the model folder, isolated interpreter):
  python -I generate_fluid_speech_fixtures.py --reference-dir DIR \
      [--model-dir ~/models/fluid-speech-pro-mlx] [--audio WAV|NPY]
"""

import argparse
import hashlib
import json
import sys
import wave
from pathlib import Path

import numpy as np

REVISION = "aeb1d4e43487586b15d86c0a1dd4511e1cf6b7a0"
REFERENCE = (
    "fluid_speech_mlx.py from FluidVoice-Labs/fluid-speech-pro-mlx at "
    f"{REVISION}, parakeet-mlx 0.7.0, mlx 0.32.5, 8-bit weights, float32 compute"
)


def load_samples(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        samples = np.load(path).astype(np.float32)
    else:
        with wave.open(str(path)) as handle:
            assert handle.getnchannels() == 1, "mono WAV required"
            assert handle.getsampwidth() == 2, "16-bit WAV required"
            assert handle.getframerate() == 16000, "16 kHz WAV required"
            raw = np.frombuffer(handle.readframes(handle.getnframes()), dtype=np.int16)
        samples = raw.astype(np.float32) / 32768.0
    assert samples.size > 0 and np.isfinite(samples).all(), "empty or non-finite audio"
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument(
        "--model-dir", type=Path, default=Path.home() / "models" / "fluid-speech-pro-mlx"
    )
    parser.add_argument("--audio", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.reference_dir))
    import mlx.core as mx  # noqa: E402
    import fluid_speech_mlx as fluid  # noqa: E402

    samples = load_samples(args.audio)
    results = {}
    for style in ("verbatim", "clean"):
        model = fluid.load(
            str(args.model_dir), bits=8, clean_mode=style == "clean", dtype=mx.float32
        )
        pre = model.encoder.pre_encode
        block = pre.fv_insert
        captured = {}

        class Capture:
            def __call__(self, x):
                captured["in"] = np.asarray(x[0], dtype=np.float32).copy()
                y = block(x)
                captured["out"] = np.asarray(y[0], dtype=np.float32).copy()
                return y

        pre.fv_insert = Capture()
        mel = fluid.log_mel(samples, model.preprocessor_config)
        lengths = mx.full((1,), mel.shape[1] - 1, dtype=mx.int64)
        features, out_lengths = model.encoder(mel, lengths)
        mx.eval(features, out_lengths)
        pre.fv_insert = block
        result = model.generate(mel)[0]
        tokens = [t.id for sentence in result.sentences for t in sentence.tokens]
        results[style] = {
            "mel": np.asarray(mel[0], dtype=np.float32),
            "cleaner_in": captured["in"],
            "cleaner_out": captured["out"],
            "encoder": np.asarray(features[0], dtype=np.float32),
            "frames": int(out_lengths[0]),
            "text": result.text,
            "tokens": tokens,
        }
        print(style, repr(result.text))

    verbatim, clean = results["verbatim"], results["clean"]
    assert np.array_equal(verbatim["mel"], clean["mel"]), "mel must not depend on style"
    assert np.array_equal(verbatim["cleaner_in"], clean["cleaner_in"]), (
        "clean mode does not touch subsampling"
    )
    assert verbatim["text"] != clean["text"], "pick a clip where the styles differ"

    output = Path(__file__).resolve().parents[1] / "testdata" / "parakeet_fluid"
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((args.model_dir / "config.json").read_text())
    (output / "config.json").write_text(json.dumps(config, separators=(",", ":")) + "\n")
    arrays = {
        "samples.npy": samples,
        "mel.npy": verbatim["mel"],
        "cleaner_in.npy": verbatim["cleaner_in"],
        "cleaner_out.npy": verbatim["cleaner_out"],
        "encoder_verbatim.npy": verbatim["encoder"],
        "encoder_clean.npy": clean["encoder"],
    }
    manifest = {
        "reference": REFERENCE,
        "model_revision": REVISION,
        "weights": "model.8bit.safetensors",
        "dtype": "float32",
        "audio_source": args.audio.name,
        "frames": verbatim["frames"],
        "verbatim": {"text": verbatim["text"], "tokens": verbatim["tokens"]},
        "clean": {"text": clean["text"], "tokens": clean["tokens"]},
        "files": {},
    }
    for name, array in arrays.items():
        array = np.ascontiguousarray(array, dtype=np.float32)
        np.save(output / name, array)
        manifest["files"][name] = {
            "shape": list(array.shape),
            "dtype": "float32",
            "sha256": hashlib.sha256((output / name).read_bytes()).hexdigest(),
        }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("wrote", output)


if __name__ == "__main__":
    main()
