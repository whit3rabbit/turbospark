#!/usr/bin/env python3
"""Generate the MOSS-Transcribe-Diarize parity fixture from the reference stack.

The script is a regeneration tool, never part of the tests. It runs the
mlx-audio 0.5.7 reference (source commit e1b19b9054bf163f5d812221a54fcc346f1890e9)
against the pinned checkpoint vanch007/mlx-MOSS-Transcribe-Diarize-4bit at
revision d42a296ee807e933ddd7588e2041dbbc84aff85d on the shared smoke clip.

Outputs crates/audio/testdata/moss_transcribe_diarize_reference.json with the
exact transcript (speaker and timestamp markers included), parsed segments,
greedy token ids, prompt token ids, the chat-template-rendered prompt string,
log-mel, whisper encoder, and VQ adaptor spot tensors, the decoder prefill
last hidden row, and the prefill logits top 8.

The reference stores the encoder, adaptor, and text backbone in bfloat16 and
runs 4-bit affine quantized matmuls in the backbone; this port computes the
whole pipeline in f32. Numeric-stage comparisons therefore use the relative
gates documented in the family README, while the token-level contract
(prompt ids, greedy ids, transcript) must match exactly.
"""

import argparse
import hashlib
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_audio.stt.utils import load_model

MAX_TOKENS = 256


def to_f32(array):
    """Materialize an MLX array as float32 numpy (bf16 tensors upcast)."""
    mx.eval(array)
    return np.asarray(mx.astype(array, mx.float32) if array.dtype != mx.float32 else array)


def selected_rows(array):
    """Row/column witness: corner, near-corner, middle, and last spots."""
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


def backbone_fixture(model, samples: np.ndarray) -> dict:
    asr = model
    input_features, audio_lengths, chunk_mapping, feature_lengths, duration = (
        asr._preprocess_audio(samples)
    )
    audio_rows = int(sum(feature_lengths))

    encoder_out = asr.model.whisper_encoder(input_features)
    adapted = asr.model.get_audio_features(
        input_features, audio_lengths, chunk_mapping
    )
    audio_features = adapted[0]
    mx.eval(encoder_out, audio_features)

    input_ids = asr._build_prompt(audio_rows, None)
    input_ids_list = np.asarray(input_ids)[0].tolist()

    # Record the literal chat-template render the Rust port must rebuild:
    # _build_prompt applies this exact template to the default prompt.
    from mlx_audio.stt.models.moss_transcribe_diarize.moss_transcribe_diarize import (
        DEFAULT_PROMPT,
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": ""},
                {"type": "text", "text": DEFAULT_PROMPT},
            ],
        }
    ]
    rendered_prompt = asr._tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs_embeds = asr._build_inputs_embeds(
        input_ids=input_ids,
        input_features=input_features,
        audio_feature_lengths=audio_lengths,
        audio_chunk_mapping=chunk_mapping,
    )
    mx.eval(inputs_embeds)

    cache = asr.make_cache()
    # The backbone returns decoder hidden states; Model.__call__ would apply
    # the tied LM head on top.
    hidden = asr.model(input_ids, inputs_embeds=inputs_embeds, cache=cache)
    last_hidden = to_f32(hidden)[0, -1, :]

    logits = to_f32(asr.model.language_model.embed_tokens.as_linear(hidden[:, -1:, :]))[0, 0, :]
    top = np.argsort(logits)[::-1][:8]
    first_logits = {
        "argmax": int(top[0]),
        "argmax_value": float(logits[top[0]]),
        "top8": [[int(i), float(logits[i])] for i in top],
    }

    eos = asr._eos_token_ids()
    # Mirror Model.generate exactly: the reference decode loop runs
    # mlx_audio.lm.generate.generate_step over the prepared ids and
    # embeddings with a greedy sampler. Calling the language model with
    # freshly built int32 id arrays skips the reference embedding path and
    # misdecodes, so the generator below is the single source of truth.
    from mlx_audio.lm.generate import generate_step
    from mlx_audio.lm.sample_utils import make_sampler

    sampler = make_sampler(0.0, top_p=1.0, min_p=0.0, top_k=0)
    generated = []
    for token, _ in generate_step(
        prompt=input_ids[0],
        input_embeddings=inputs_embeds[0],
        model=asr,
        max_tokens=MAX_TOKENS,
        sampler=sampler,
        prefill_step_size=4096,
    ):
        if int(token) in eos:
            break
        generated.append(int(token))

    text = asr._tokenizer.decode(generated, skip_special_tokens=True).strip()
    segments = asr._parse_segments(text, duration)
    digit_ids = {
        digit: int(asr._tokenizer.encode(digit, add_special_tokens=False)[0])
        for digit in "0123456789"
    }
    return {
        "audio_rows": audio_rows,
        "audio_token_count": audio_rows,
        "feature_lengths": [int(v) for v in feature_lengths],
        "duration_seconds": float(duration),
        "audio_tokens_per_second": float(asr.audio_tokens_per_second),
        "time_marker_every_seconds": int(asr.time_marker_every_seconds),
        "enable_time_marker": bool(asr.enable_time_marker),
        "digit_token_ids": digit_ids,
        "eos_token_ids": sorted(int(v) for v in eos),
        "rendered_prompt": rendered_prompt,
        "input_features": selected_rows(input_features),
        "encoder_output": selected_rows(encoder_out),
        "audio_embeddings": selected_rows(audio_features),
        "prompt_token_ids": input_ids_list,
        "prefill_last_hidden": [float(v) for v in last_hidden],
        "first_logits": first_logits,
        "generated_token_ids": generated,
        "transcript": text,
        "segments": segments,
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
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--repository", default="vanch007/mlx-MOSS-Transcribe-Diarize-4bit")
    parser.add_argument(
        "--source-commit", default="e1b19b9054bf163f5d812221a54fcc346f1890e9"
    )
    args = parser.parse_args()

    samples, sample_rate = read_wav_f32(args.audio)
    if sample_rate != 16000:
        raise ValueError(f"expected 16 kHz audio, got {sample_rate}")

    model = load_model(str(args.model_dir), model_type="moss_transcribe_diarize")
    config_json = json.loads((args.model_dir / "config.json").read_text())
    processor_json = json.loads((args.model_dir / "processor_config.json").read_text())
    preprocessor_json = json.loads(
        (args.model_dir / "preprocessor_config.json").read_text()
    )

    backbone = backbone_fixture(model, samples)

    fixture = {
        "schema": "turbospark.moss_transcribe_diarize.reference/1",
        "provenance": {
            "repository": args.repository,
            "revision": args.revision,
            "source_commit": args.source_commit,
            "mlx_audio_version": "0.5.7",
            "audio_path": "crates/audio/testdata/qwen3_forced_aligner_reference.wav",
            "audio_sha256": file_fingerprint(args.audio),
            "audio_samples": int(samples.shape[0]),
            "audio_sample_rate": 16000,
        },
        "config_json": config_json,
        "processor_config": processor_json,
        "preprocessor_config": preprocessor_json,
        "backbone": backbone,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, separators=(",", ":")) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "transcript": backbone["transcript"],
                "segments": backbone["segments"],
                "audio_rows": backbone["audio_rows"],
                "prompt_tokens": len(backbone["prompt_token_ids"]),
                "generated_tokens": len(backbone["generated_token_ids"]),
                "first_token": backbone["first_logits"]["argmax"],
            }
        )
    )


if __name__ == "__main__":
    main()
