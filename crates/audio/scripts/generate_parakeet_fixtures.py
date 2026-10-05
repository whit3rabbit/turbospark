#!/usr/bin/env python3
"""Write the Parakeet TDT stage goldens from the pinned mlx-audio reference.

Runs mlx-audio 0.5.7 ParakeetTDT in float32 on a committed speech excerpt,
captures the mel frontend, encoder output, per-step predictor/joint tensors,
and the greedy TDT decode decisions, and saves them under
crates/speech/testdata/parakeet/ for the Rust parity tests.

The weights come from the local checkpoint directory (default
~/models/parakeet-tdt-0.6b-v2, pinned revision 8ae155301e23d820d82aa60d24817c900e69e487).
Tests must not download assets; this regeneration script owns that step.

Usage:
  python3 generate_parakeet_fixtures.py [--model-dir DIR] [--audio WAV|NPY]
"""

import argparse
import hashlib
import json
import wave
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten
from mlx_audio.stt.models.parakeet.audio import log_mel_spectrogram
from mlx_audio.stt.models.parakeet.parakeet import Model

REVISION = "8ae155301e23d820d82aa60d24817c900e69e487"


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


def is_special_piece(piece: str) -> bool:
    return (piece.startswith("<|") and piece.endswith("|>")) or piece in ("<unk>", "<pad>")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path.home() / "models" / "parakeet-tdt-0.6b-v2",
    )
    parser.add_argument("--audio", type=Path, default=Path("/tmp/speech16.wav"))
    args = parser.parse_args()

    config = json.loads((args.model_dir / "config.json").read_text())
    model = Model.from_config(config)
    weights = mx.load(str(args.model_dir / "model.safetensors"))
    model.load_weights(list(tree_flatten(weights)))
    params = dict(tree_flatten(model.parameters()))
    model.update(tree_unflatten([(k, v.astype(mx.float32)) for k, v in params.items()]))

    samples = load_samples(args.audio)
    mel = log_mel_spectrogram(mx.array(samples), model.preprocessor_config)
    features, lengths = model.encoder(mel)
    mx.eval(mel, features, lengths)
    mel_out = np.asarray(mel[0], dtype=np.float32)
    features_out = np.asarray(features[0], dtype=np.float32)
    frame_count = int(lengths[0])
    assert features_out.shape[0] == frame_count, "encoder length mismatch"

    blank = model.blank_id
    durations = model.durations
    max_symbols = model.max_symbols
    vocabulary = model.vocabulary
    hidden, cell = model._make_initial_decoder_state(1, mx.float32)
    last_token = blank
    time = 0
    new_symbols = 0
    steps = []
    emitted = []
    captured_steps = []
    while time < frame_count:
        feature = features[:, time : time + 1]
        current_token = mx.array([[last_token]], dtype=mx.int32)
        embedded = model.decoder.prediction["embed"](current_token)
        blank_mask = mx.expand_dims(current_token == blank, axis=-1)
        embedded = mx.where(blank_mask, mx.zeros_like(embedded), embedded)
        decoder_output, (hidden_out, cell_out) = model.decoder.prediction["dec_rnn"](
            embedded, (hidden, cell)
        )
        decoder_output = decoder_output.astype(mx.float32)
        joint_output = model.joint(feature, decoder_output)
        mx.eval(decoder_output, joint_output)
        pred_token = int(mx.argmax(joint_output[0, 0, :, : blank + 1]))
        decision = int(mx.argmax(joint_output[0, 0, :, blank + 1 :]))
        duration = durations[decision]
        steps.append(
            {
                "time": time,
                "last_token": last_token,
                "pred_token": pred_token,
                "decision": decision,
                "duration": duration,
            }
        )
        if len(captured_steps) < 3:
            captured_steps.append(
                {
                    "decoder_output": np.asarray(
                        decoder_output[0], dtype=np.float32
                    ).copy(),
                    "joint_output": np.asarray(joint_output[0, 0], dtype=np.float32).copy(),
                }
            )
        if pred_token != blank:
            last_token = pred_token
            hidden, cell = hidden_out, cell_out
            emitted.append(pred_token)
        time += duration
        new_symbols += 1
        if duration != 0:
            new_symbols = 0
        elif max_symbols is not None and max_symbols <= new_symbols:
            time += 1
            new_symbols = 0

    pieces = [
        vocabulary[token].replace("▁", " ")
        for token in emitted
        if not is_special_piece(vocabulary[token])
    ]
    transcript = "".join(pieces).strip()

    output = Path(__file__).resolve().parents[1] / "testdata" / "parakeet"
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(
        json.dumps(config, separators=(",", ":")) + "\n"
    )

    arrays = {
        "speech_samples.npy": samples,
        "speech_mel.npy": mel_out,
        "speech_encoder.npy": features_out,
        "speech_steps_joint.npy": np.stack(
            [step["joint_output"] for step in captured_steps]
        ),
        "speech_steps_pred.npy": np.stack(
            [step["decoder_output"][0] for step in captured_steps]
        ),
    }
    manifest = {
        "reference": "mlx-audio 0.5.7 parakeet at commit e1b19b9054bf163f5d812221a54fcc346f1890e9",
        "model_revision": REVISION,
        "dtype": "float32",
        "audio_source": str(args.audio),
        "files": {},
        "steps": steps,
        "emitted_tokens": emitted,
        "transcript": transcript,
        "n_steps": len(steps),
        "n_blank_steps": sum(1 for s in steps if s["pred_token"] == blank),
        "captured_steps": len(captured_steps),
    }
    for name, array in arrays.items():
        np.save(output / name, array, allow_pickle=False)
        digest = hashlib.sha256((output / name).read_bytes()).hexdigest()
        manifest["files"][name] = {
            "shape": list(array.shape),
            "dtype": "float32",
            "sha256": digest,
        }

    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, separators=(",", ":")) + "\n"
    )
    print(f"wrote {len(arrays) + 2} files to {output}")
    print(f"mel {mel.shape} encoder {features.shape} steps {len(steps)}")
    print(f"transcript: {transcript!r}")


if __name__ == "__main__":
    main()
