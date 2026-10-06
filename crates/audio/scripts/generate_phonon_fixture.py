#!/usr/bin/env python3
"""Generate the Phonon-1 parity fixture from the pinned reference stack.

The script is a regeneration tool, never part of the tests. It runs the
mlx-audio 0.5.7 reference (source commit e1b19b9054bf163f5d812221a54fcc346f1890e9)
against the pinned checkpoint FermionResearch/Phonon-1 at revision
0428da04625c51b6f069a9829c7060e6b167b92a, already materialized by
mlx_audio.stt.models.phonon.transport.prepare_model_path, on the shared smoke
clip.

Outputs crates/audio/testdata/phonon_reference.json with the exact transcript,
greedy token ids, prompt token ids, log-mel and audio-tower spot tensors, the
decoder prefill last hidden row, the prefill logits top 8, the packed manifest
verbatim, and quint5 unpack goldens (packed byte rows, unpacked 2-bit plane
words, slim metadata values, and effective weight spots) for three
representative decoder linears.

The reference computes the packed decoder as two native MLX 2-bit quantized
matmuls summed at the output; the Rust port materializes one fused f32 weight
(base + residual) per linear, so decoder-stage comparisons use the honest
relative gates documented in the family README, while the unpack goldens
themselves are exact integer checks.
"""

import argparse
import base64
import hashlib
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_audio.stt.utils import load_model

MAX_TOKENS = 96

# One attention projection, one middle-stack MLP up projection, and one
# final-layer MLP down projection, covering both packing widths in use.
SELECTED_MODULES = [
    "model.layers.0.self_attn.q_proj",
    "model.layers.14.mlp.gate_proj",
    "model.layers.27.mlp.down_proj",
]


def to_f32(array):
    """Materialize an MLX array as float32 numpy (bf16 tensors upcast)."""
    mx.eval(array)
    return np.asarray(mx.astype(array, mx.float32) if array.dtype != mx.float32 else array)


def selected_rows(array):
    mx.eval(array)
    values = to_f32(array)
    if values.ndim == 3:
        values = values[0]
    rows = sorted({0, min(1, values.shape[0] - 1), values.shape[0] // 2, values.shape[0] - 1})
    columns = sorted({0, min(1, values.shape[1] - 1), values.shape[1] // 2, values.shape[1] - 1})
    return {
        "shape": list(values.shape),
        "rows": rows,
        "columns": columns,
        "values": [[float(values[r, c]) for c in columns] for r in rows],
    }


def resolve_module(root, dotted_name):
    current = root
    for part in dotted_name.split("."):
        current = current[int(part)] if part.isdigit() else getattr(current, part)
    return current


def codes_from_words(words: np.ndarray) -> np.ndarray:
    """Expand packed 2-bit plane words [out, in/16] to codes [out, in]."""
    out_features, words_per_row = words.shape
    codes = np.zeros((out_features, words_per_row * 16), dtype=np.uint32)
    for column in range(codes.shape[1]):
        word = words[:, column // 16].astype(np.uint32)
        codes[:, column] = (word >> (2 * (column % 16))) & np.uint32(3)
    return codes


def quint5_fixture(model, weights_path: Path, manifest: dict) -> dict:
    """Record exact unpack goldens for the selected packed decoder linears."""
    asr = model._model if hasattr(model, "_model") else model
    weights = mx.load(str(weights_path))
    modules = {row["name"]: row for row in manifest["modules"]}
    selected = []
    for name in SELECTED_MODULES:
        row = modules[name]
        in_features = int(row["in_features"])
        out_features = int(row["out_features"])
        module = resolve_module(asr, name)
        packed = np.asarray(weights[name + ".quint5_q"], dtype=np.uint8)
        expected_bytes = ((in_features + 9) // 10) * 3
        if packed.shape != (out_features, expected_bytes):
            raise ValueError(f"unexpected quint5 shape for {name}: {packed.shape}")
        base_q = np.asarray(module.base_q, dtype=np.uint32)
        residual_q = np.asarray(module.residual_q, dtype=np.uint32)
        if base_q.shape != (out_features, in_features // 16):
            raise ValueError(f"unpacked word shape mismatch for {name}: {base_q.shape}")

        # Effective fused weight from the module's own runtime metadata, the
        # exact arithmetic the Rust port repeats in f32. The metadata is
        # [out, groups]; each group's scale/bias covers group_size columns.
        group_size = int(manifest["group_size"])

        def expand(metadata: np.ndarray) -> np.ndarray:
            return np.repeat(metadata, group_size, axis=1)

        base_scales = expand(to_f32(module._runtime_base_scales))
        base_biases = expand(to_f32(module._runtime_base_biases))
        residual_scales = expand(to_f32(module._runtime_residual_scales))
        residual_biases = expand(to_f32(module._runtime_residual_biases))
        base_codes = codes_from_words(base_q).astype(np.float32)
        residual_codes = codes_from_words(residual_q).astype(np.float32)
        weight = (
            base_codes * base_scales + base_biases
            + residual_codes * residual_scales + residual_biases
        )

        rows = sorted({0, out_features // 2, out_features - 1})
        columns = sorted({0, 1, 63, 127, 128, in_features // 2, in_features - 1})
        alpha = to_f32(module.base_alpha)
        residual_scale = to_f32(module.residual_scale)
        selected.append(
            {
                "name": name,
                "in_features": in_features,
                "out_features": out_features,
                "group_size": int(manifest["group_size"]),
                "packed_bytes_per_row": expected_bytes,
                "rows": rows,
                "packed_rows_base64": [
                    base64.b64encode(packed[r].tobytes()).decode("ascii") for r in rows
                ],
                "base_words_rows": [
                    [int(w) for w in base_q[r]] for r in rows
                ],
                "residual_words_rows": [
                    [int(w) for w in residual_q[r]] for r in rows
                ],
                "base_alpha_values": [float(alpha[r]) for r in rows],
                "residual_scale_value": float(residual_scale[0]),
                "weight_columns": columns,
                "weight_rows": rows,
                "weight_values": [
                    [float(weight[r, c]) for c in columns] for r in rows
                ],
            }
        )
    return {"selected": selected}


def backbone_fixture(model, samples: np.ndarray) -> dict:
    asr = model._model if hasattr(model, "_model") else model
    input_features, feature_mask, num_audio_tokens = asr._preprocess_audio(samples)
    valid = int(np.asarray(feature_mask).sum(axis=-1)[0])
    features = np.asarray(input_features)[0, :, :valid]

    audio_features = asr.get_audio_features(input_features, feature_mask)
    audio_features_f32 = mx.astype(audio_features, mx.float32)
    audio_rows = int(audio_features.shape[0])

    input_ids = asr._build_prompt(num_audio_tokens, None, None)
    input_ids = np.asarray(input_ids)[0].tolist()

    embedding_dtype = asr.model.embed_tokens(mx.array([input_ids])).dtype
    embeds = asr._build_inputs_embeds(
        mx.array([input_ids]), mx.astype(audio_features_f32, embedding_dtype)
    )
    mx.eval(embeds)
    cache = asr.make_cache()
    hidden = asr.model(inputs_embeds=embeds, cache=cache)
    last_hidden = to_f32(hidden)[0, -1, :]

    logits_mx = asr.model.embed_tokens.as_linear(hidden[:, -1:, :])
    logits = to_f32(logits_mx)[0, 0, :]
    top = np.argsort(logits)[::-1][:8]
    first_logits = {
        "argmax": int(top[0]),
        "argmax_value": float(logits[top[0]]),
        "top8": [[int(i), float(logits[i])] for i in top],
    }

    eos = asr._eos_token_ids()
    generated = []
    y = mx.argmax(logits_mx[:, 0], axis=-1)
    for _ in range(MAX_TOKENS):
        token = int(y[0])
        if token in eos:
            break
        generated.append(token)
        embed = asr.model.embed_tokens(y[:, None])
        hidden = asr._forward_with_embeds(embed, cache)
        y = mx.argmax(hidden[:, -1, :], axis=-1)
        mx.eval(y)

    decoded = asr._tokenizer.decode(generated, skip_special_tokens=True)
    _, text = asr.extract_language(decoded)
    return {
        "audio_rows": audio_rows,
        "input_features": selected_rows(mx.array(features)),
        "audio_embeddings": selected_rows(audio_features),
        "prompt_token_ids": input_ids,
        "prefill_last_hidden": [float(v) for v in last_hidden],
        "first_logits": first_logits,
        "generated_token_ids": generated,
        "transcript": text,
    }


def file_fingerprint(path: Path) -> dict:
    data = path.read_bytes()
    return {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def read_wav_f32(path: Path) -> tuple[np.ndarray, int]:
    """Reads a 32-bit float WAV's data chunk directly.

    mlx_audio.audio_io decodes through miniaudio, which quantizes float32
    WAV input to 16 bits before converting back to float. The Rust reader
    consumes the true float32 samples, so the generator reads the data chunk
    itself to give both sides bit-identical inputs.
    """
    data = path.read_bytes()
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("not a RIFF/WAVE file")
    position = 12
    fmt = None
    payload = None
    while position + 8 <= len(data):
        chunk = data[position : position + 4]
        size = int.from_bytes(data[position + 4 : position + 8], "little")
        body = data[position + 8 : position + 8 + size]
        if chunk == b"fmt ":
            fmt = body
        elif chunk == b"data":
            payload = body
        position += 8 + size + (size & 1)
    if fmt is None or payload is None:
        raise ValueError("missing fmt or data chunk")
    tag, channels, rate = (
        int.from_bytes(fmt[0:2], "little"),
        int.from_bytes(fmt[2:4], "little"),
        int.from_bytes(fmt[4:8], "little"),
    )
    bits = int.from_bytes(fmt[14:16], "little")
    if tag != 3 or bits != 32 or channels != 1:
        raise ValueError(f"expected mono 32-bit float WAV, got tag={tag} bits={bits}")
    return np.frombuffer(payload, dtype="<f4").astype(np.float32), rate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--source-commit", default="e1b19b9054bf163f5d812221a54fcc346f1890e9")
    args = parser.parse_args()

    samples, sample_rate = read_wav_f32(args.audio)
    if sample_rate != 16000:
        raise ValueError(f"expected 16 kHz audio, got {sample_rate}")

    model = load_model(str(args.model_dir), model_type="phonon")
    config_json = json.loads((args.model_dir / "config.json").read_text())
    manifest = json.loads((args.model_dir / "packed_manifest.json").read_text())
    marker = json.loads((args.model_dir / ".mlx_audio_phonon.json").read_text())
    shard = manifest["shards"][0]
    if len(manifest["shards"]) != 1:
        raise ValueError("the pinned profile must be a single shard")
    shard_path = args.model_dir / shard["name"]
    actual_sha = hashlib.sha256(shard_path.read_bytes()).hexdigest()
    if actual_sha != shard["sha256"]:
        raise ValueError("materialized shard digest mismatch")

    backbone = backbone_fixture(model, samples)
    packed = quint5_fixture(model, shard_path, manifest)

    fixture = {
        "schema": "turbospark.phonon.reference/1",
        "provenance": {
            "repository": "FermionResearch/Phonon-1",
            "revision": args.revision,
            "source_commit": args.source_commit,
            "mlx_audio_version": "0.5.7",
            "archive_sha256": marker["archive_sha256"],
            "archive_fingerprint": file_fingerprint(args.archive),
            "profile": marker["profile"],
            "materialized_shard": {
                "name": shard["name"],
                "size": shard["bytes"],
                "sha256": shard["sha256"],
            },
            "audio_path": "crates/audio/testdata/qwen3_forced_aligner_reference.wav",
            "audio_sha256": file_fingerprint(args.audio),
            "audio_samples": int(samples.shape[0]),
            "audio_sample_rate": 16000,
        },
        "config_json": config_json,
        "packed_manifest": manifest,
        "backbone": backbone,
        "quint5": packed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, separators=(",", ":")) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "transcript": backbone["transcript"],
                "audio_rows": backbone["audio_rows"],
                "generated_tokens": len(backbone["generated_token_ids"]),
                "first_token": backbone["first_logits"]["argmax"],
                "quint5_modules": len(packed["selected"]),
            }
        )
    )


if __name__ == "__main__":
    main()
