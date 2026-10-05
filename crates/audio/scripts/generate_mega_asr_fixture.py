#!/usr/bin/env python3
"""Generate the Mega-ASR parity fixture from the pinned reference stack.

The script is a regeneration tool, never part of the tests. It runs the
mlx-audio 0.5.7 reference (source commit e1b19b9054bf163f5d812221a54fcc346f1890e9)
against:

- the pinned always-on-robust checkpoint mlx-community/Mega-ASR-8bit at
  revision b9c3c7020f94944205df7f7b5d5d1ce96678d74f (Qwen3-ASR-1.7B backbone
  with the robustness LoRA folded into the weights before quantization, so
  the distribution carries no router or LoRA files), and
- the trained router and LoRA factors published with the dynamic
  mlx-community/Mega-ASR-bf16 distribution (the 8-bit repository omits the
  extras directory, and the router is backbone independent).

Outputs crates/audio/testdata/mega_asr_reference.json with the exact
transcript, greedy token ids, sparse backbone stage tensors, router weights
with per-stage tensors and routing decisions, and the LoRA module inventory
with selected factor tensors and materialized delta spots.
"""

import argparse
import base64
import hashlib
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_audio.stt.models.mega_asr.convert_lora import load_lora_factors
from mlx_audio.stt.models.mega_asr.lora import materialize_delta
from mlx_audio.stt.models.mega_asr.router import AudioQualityRouter

MAX_TOKENS = 96
DEGRADED_AMPLITUDE = 0.2
DEGRADED_PERIOD = 97


def to_f32(array):
    """Materialize an MLX array as float32 numpy (bf16 checkpoints upcast)."""
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


def to_base64(array):
    mx.eval(array)
    data = np.asarray(array, dtype=np.float32).astype("<f4", copy=False).tobytes()
    return base64.b64encode(data).decode("ascii")


def degraded_waveform(clean: np.ndarray) -> np.ndarray:
    """Deterministic f32 ripple both Python and Rust reproduce bit-exactly."""
    index = np.arange(clean.shape[0], dtype=np.float32)
    ripple = (index % np.float32(DEGRADED_PERIOD)) / np.float32(DEGRADED_PERIOD)
    ripple = ripple - np.float32(0.5)
    return (clean + np.float32(DEGRADED_AMPLITUDE) * ripple).astype(np.float32)


def load_backbone(model_dir: Path):
    """Load through the reference dispatcher with the mega_asr family forced.

    The pinned distribution carries model_type qwen3_asr because its LoRA was
    folded in before quantization; the dynamic mega_asr wrapper still accepts
    the config, and with no extras present the router stays untrained and the
    LoRA table empty, so the backbone stage tensors below are the reference
    decode path for this artifact.
    """
    from mlx_audio.stt.utils import load_model

    model = load_model(str(model_dir), model_type="mega_asr")
    if len(getattr(model, "_deltas", {})) != 0:
        raise ValueError("the pinned distribution must not carry LoRA factors")
    return model, json.loads((model_dir / "config.json").read_text())


def backbone_fixture(model, samples: np.ndarray) -> dict:
    asr = model._asr
    input_features, feature_mask, num_audio_tokens = asr._preprocess_audio(samples)
    valid = int(np.asarray(feature_mask).sum(axis=-1)[0])
    features = np.asarray(input_features)[0, :, :valid]

    audio_features = asr.get_audio_features(input_features, feature_mask)
    audio_features_f32 = mx.astype(audio_features, mx.float32)
    audio_rows = int(audio_features.shape[0])

    input_ids = asr._build_prompt(num_audio_tokens, None, None)
    input_ids = np.asarray(input_ids)[0].tolist()

    # The reference casts audio features to the embedding output dtype; the
    # packed quantized embedding weight itself is U32, so read the dtype off
    # the dequantized embedding output instead.
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


def router_input_fixture(router: AudioQualityRouter, name: str, samples: np.ndarray) -> dict:
    waveform = mx.array(samples, dtype=mx.float32)
    logmel = router.logmel(waveform)
    hidden = mx.expand_dims(logmel, axis=0)
    conv_hidden = router.frontend(hidden)
    positioned = router.pos_encoder(conv_hidden)
    encoded = router.transformer(positioned)
    pooled = router.pooling(encoded)
    logits = router.classifier(pooled)
    logits = mx.squeeze(logits, axis=0)
    mx.eval(logits)
    probabilities = mx.softmax(logits, axis=-1)
    degraded_prob = float(probabilities[1])
    return {
        "name": name,
        "logmel": selected_rows(logmel),
        "frontend_hidden": selected_rows(conv_hidden),
        "transformer_hidden": selected_rows(encoded),
        "pooled": [float(v) for v in np.asarray(pooled)[0]],
        "logits": [float(v) for v in logits],
        "degraded_prob": degraded_prob,
        "use_lora": bool(degraded_prob >= 0.5),
    }


def router_fixture(router_weights: Path, clean: np.ndarray, degraded: np.ndarray) -> dict:
    weights = dict(mx.load(str(router_weights)))
    router = AudioQualityRouter.from_converted(weights)
    router.eval()

    names = sorted(weights.keys())
    shapes = {name: list(weights[name].shape) for name in names}
    packed = b"".join(
        np.asarray(weights[name], dtype=np.float32).astype("<f4", copy=False).tobytes()
        for name in names
    )
    return {
        "weight_names": names,
        "weight_shapes": shapes,
        "weights_base64": base64.b64encode(packed).decode("ascii"),
        "inputs": [
            router_input_fixture(router, "smoke_clean", clean),
            router_input_fixture(router, "smoke_degraded", degraded),
        ],
    }


def lora_fixture(lora_weights: Path) -> dict:
    factors = load_lora_factors(lora_weights)
    names = sorted(factors.keys())
    modules = []
    for name in names:
        a = factors[name]["A"]
        b = factors[name]["B"]
        modules.append([name, int(b.shape[0]), int(a.shape[1]), int(a.shape[0])])

    selected_names = [
        "model.layers.0.self_attn.q_proj",
        "model.layers.27.mlp.down_proj",
        "audio_tower.layers.0.fc1",
    ]
    selected = []
    for name in selected_names:
        module = factors[name]
        delta = materialize_delta(module)
        mx.eval(delta)
        values = np.asarray(delta, dtype=np.float32)
        rows = sorted({0, values.shape[0] // 2, values.shape[0] - 1})
        columns = sorted({0, values.shape[1] // 4, values.shape[1] // 2, values.shape[1] - 1})
        selected.append(
            {
                "name": name,
                "rank": int(module["A"].shape[0]),
                "a_base64": to_base64(module["A"]),
                "b_base64": to_base64(module["B"]),
                "delta_rows": rows,
                "delta_columns": columns,
                "delta_values": [[float(values[r, c]) for c in columns] for r in rows],
                "delta_max_abs": float(np.abs(values).max()),
                "delta_sum_abs": float(np.abs(values).sum(dtype=np.float64)),
            }
        )
    return {
        "scaling": float(factors[names[0]]["scaling"]),
        "module_count": len(modules),
        "modules": modules,
        "selected": selected,
    }


def file_fingerprint(path: Path) -> dict:
    data = path.read_bytes()
    return {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def read_wav_f32(path: Path) -> tuple[np.ndarray, int]:
    """Reads a 32-bit float WAV's data chunk directly.

    mlx_audio.audio_io decodes through miniaudio, which quantizes float32
    WAV input to 16 bits before converting back to float. The Rust reader
    consumes the true float32 samples, so the generator reads the data chunk
    itself to give both sides bit-identical inputs. The reference transcript
    was additionally confirmed through the stock load_audio path, where the
    greedy decode is robust to that quantization.
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
    parser.add_argument("--router-weights", required=True, type=Path)
    parser.add_argument("--lora-weights", required=True, type=Path)
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--extras-revision", required=True)
    parser.add_argument("--source-commit", default="e1b19b9054bf163f5d812221a54fcc346f1890e9")
    args = parser.parse_args()

    clean, sample_rate = read_wav_f32(args.audio)
    if sample_rate != 16000:
        raise ValueError(f"expected 16 kHz audio, got {sample_rate}")
    degraded = degraded_waveform(clean)

    model, config_json = load_backbone(args.model_dir)
    backbone = backbone_fixture(model, clean)

    fixture = {
        "schema": "turbospark.mega_asr.reference/1",
        "provenance": {
            "repository": "mlx-community/Mega-ASR-8bit",
            "revision": args.revision,
            "source_commit": args.source_commit,
            "mlx_audio_version": "0.5.7",
            "router_lora_repository": "mlx-community/Mega-ASR-bf16",
            "router_lora_revision": args.extras_revision,
            "router_weights_sha256": file_fingerprint(args.router_weights),
            "lora_weights_sha256": file_fingerprint(args.lora_weights),
            "audio_path": "crates/audio/testdata/qwen3_forced_aligner_reference.wav",
            "audio_sha256": file_fingerprint(args.audio),
            "audio_samples": int(clean.shape[0]),
            "audio_sample_rate": 16000,
            "degraded_derivation": (
                "degraded[i] = clean[i] + 0.2f32 * (f32(i % 97) / 97.0f32 - 0.5f32), "
                "each step a single round-to-nearest f32 operation"
            ),
        },
        "config_json": config_json,
        "backbone": backbone,
        "router": router_fixture(args.router_weights, clean, degraded),
        "lora": lora_fixture(args.lora_weights),
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
                "router_decisions": [
                    [entry["name"], entry["degraded_prob"], entry["use_lora"]]
                    for entry in fixture["router"]["inputs"]
                ],
                "lora_modules": fixture["lora"]["module_count"],
            }
        )
    )


if __name__ == "__main__":
    main()
