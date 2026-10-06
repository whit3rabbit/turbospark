#!/usr/bin/env python3
"""Generate the Granite Speech 1B ASR parity fixture from the reference stack.

The script is a regeneration tool, never part of the tests. It runs the
mlx-audio 0.5.7 reference (source commit e1b19b9054bf163f5d812221a54fcc346f1890e9)
against the pinned checkpoint ibm-granite/granite-4.0-1b-speech at revision
bd87ab862416353633ea431fe49b1614003623c5 on the shared smoke clip with the
default ASR prompt.

Outputs crates/audio/testdata/granite_speech_reference.json with the exact
transcript, greedy token ids, the expanded chat-template prompt string, the
prompt prefix/suffix/full token ids, the frontend, encoder, and projector
spot tensors, the prefill last hidden row, the prefill logits top 8, and the
final decode-step logits evidence.

The reference computes the audio path and the language model in bfloat16
(this pin ships bf16 shards); the Rust port computes f32 with weights
upcast at load. Numeric-stage comparisons therefore use relative gates
documented in the family README, while the token-level contract (prompt
ids, greedy ids, transcript) must match exactly.
"""

import argparse
import hashlib
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_audio.stt.utils import load_model
from mlx_audio.lm.models.base import create_attention_mask

MAX_TOKENS = 256
DEFAULT_ASR_PROMPT = "can you transcribe the speech into a written format?"


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
    parser.add_argument("--repository", default="ibm-granite/granite-4.0-1b-speech")
    parser.add_argument(
        "--source-commit", default="e1b19b9054bf163f5d812221a54fcc346f1890e9"
    )
    args = parser.parse_args()

    samples, sample_rate = read_wav_f32(args.audio)
    if sample_rate != 16000:
        raise ValueError(f"expected 16 kHz audio, got {sample_rate}")

    model = load_model(str(args.model_dir))
    tok = model._tokenizer
    lm = model.language_model
    inner = lm.model

    def enc(text):
        return [int(v) for v in tok.encode(text, add_special_tokens=False)]

    # Reference _extract_features on the raw float32 samples.
    wav = mx.array(np.asarray(samples, dtype=np.float32).reshape(-1), dtype=mx.float32)
    input_features, num_audio_tokens = model._extract_features(wav)
    mx.eval(input_features)
    encoder_output = model.encoder(input_features)
    mx.eval(encoder_output)
    audio_features = model.projector(encoder_output)
    mx.eval(audio_features)
    assert int(audio_features.shape[1]) == int(num_audio_tokens), (
        "projector rows must equal the declared audio token count"
    )

    # Reference _build_prompt: expand the chat template for one user turn,
    # then encode the expanded string. The pinned tokenizer (GPT2 BPE, no
    # post-processor, no BOS) adds no special tokens, so the whole-string
    # encode must equal the piecewise encode around the <|audio|> spans.
    # The Rust port rebuilds the prompt from exactly these segments.
    user_prompt = DEFAULT_ASR_PROMPT
    chat = [{"role": "user", "content": "<|audio|>" * int(num_audio_tokens) + user_prompt}]
    prompt_string = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    prompt_ids = [int(v) for v in tok.encode(prompt_string)]

    audio_token_id = int(model.audio_token_id)
    prefix_ids = enc("USER: ")
    suffix_ids = enc(user_prompt + "\n ASSISTANT:")
    piecewise = prefix_ids + [audio_token_id] * int(num_audio_tokens) + suffix_ids
    assert piecewise == prompt_ids, (
        "piecewise encode diverges from the whole-string encode: "
        f"{piecewise} vs {prompt_ids}"
    )

    # Reference _build_inputs_embeds: embed with audio ids zeroed, then
    # overwrite the audio positions with the projected features.
    inputs_embeds = model._build_inputs_embeds(mx.array(prompt_ids), audio_features)
    mx.eval(inputs_embeds)
    prompt_len = int(inputs_embeds.shape[1])
    assert prompt_len == len(prompt_ids)

    # Prefill over the injected embeddings: the hidden after the final RMS
    # norm, then the scaled head logits at the last prompt position. This is
    # the same body the reference generate_step drives through Model.__call__.
    cache = model.make_cache()
    h = inputs_embeds * inner.embedding_multiplier
    mask = create_attention_mask(h, cache[0])
    for layer, c in zip(inner.layers, cache):
        h = layer(h, mask, cache=c)
    h = inner.norm(h)
    mx.eval(h)
    last_hidden = to_f32(h)[0, -1, :]
    prefill_logits = to_f32(lm.lm_head(h[:, -1:, :]) / lm.logits_scaling)[0, 0, :]
    top = np.argsort(prefill_logits)[::-1][:8]
    first_logits = {
        "argmax": int(top[0]),
        "argmax_value": float(prefill_logits[top[0]]),
        "top8": [[int(i), float(prefill_logits[i])] for i in top],
    }

    # Mirror Model.generate exactly: generate_step over the prepared
    # embeddings with a greedy sampler, stopping at the tokenizer eos id.
    from mlx_audio.lm.generate import generate_step
    from mlx_audio.lm.sample_utils import make_sampler

    sampler = make_sampler(0.0, top_p=1.0, min_p=0.0, top_k=0)
    eos_token_id = int(tok.eos_token_id)
    generated = []
    for token, _ in generate_step(
        prompt=mx.array([], dtype=mx.int32),
        input_embeddings=inputs_embeds.squeeze(0),
        model=model,
        max_tokens=MAX_TOKENS,
        sampler=sampler,
        prefill_step_size=2048,
    ):
        if int(token) == eos_token_id:
            break
        generated.append(int(token))

    raw_text = tok.decode(generated, skip_special_tokens=False)
    text = tok.decode(generated, skip_special_tokens=True)

    # Per-step logits from an equivalent single-stream decode, so the Rust
    # test can compare decision margins at the final step.
    cache2 = model.make_cache()
    h2 = inputs_embeds * inner.embedding_multiplier
    mask2 = create_attention_mask(h2, cache2[0])
    for layer, c in zip(inner.layers, cache2):
        h2 = layer(h2, mask2, cache=c)
    h2 = inner.norm(h2)
    mx.eval(h2)
    prev = h2[:, -1:, :]
    final_step_logits = None
    for index, tid in enumerate(generated):
        step_logits = to_f32(lm.lm_head(prev) / lm.logits_scaling)[0, 0, :]
        if index == len(generated) - 1:
            top = np.argsort(step_logits)[::-1][:8]
            final_step_logits = {
                "argmax": int(top[0]),
                "argmax_value": float(step_logits[top[0]]),
                "top8": [[int(i), float(step_logits[i])] for i in top],
                "token_values": {
                    str(eos_token_id): float(step_logits[eos_token_id]),
                },
            }
        emb = inner.embed_tokens(mx.array([tid])) * inner.embedding_multiplier
        prev = emb[None]
        for layer, c in zip(inner.layers, cache2):
            prev = layer(prev, None, cache=c)
        prev = inner.norm(prev)
        mx.eval(prev)

    input_features_np = to_f32(input_features)
    encoder_rows = int(input_features_np.shape[1])
    # The centered STFT frame count before pair stacking: 1 + len // hop.
    stft_frames = 1 + int(samples.shape[0]) // 160
    fixture = {
        "schema": "turbospark.granite_speech.reference/1",
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
        "config_json": json.loads((args.model_dir / "config.json").read_text()),
        "frontend": {
            "stft_frames": stft_frames,
            "encoder_rows": encoder_rows,
            "input_dim": int(input_features_np.shape[2]),
            "input_features": selected_rows(input_features.transpose(0, 2, 1)),
        },
        "encoder": {
            "output": selected_rows(encoder_output),
        },
        "projector": {
            "num_audio_tokens": int(num_audio_tokens),
            "audio_embeddings": selected_rows(audio_features),
        },
        "backbone": {
            "prompt": user_prompt,
            "prompt_string": prompt_string,
            "prompt_prefix_ids": prefix_ids,
            "prompt_suffix_ids": suffix_ids,
            "prompt_token_ids": prompt_ids,
            "prompt_tokens": prompt_len,
            "audio_token_id": audio_token_id,
            "eos_token_id": eos_token_id,
            "prefill_last_hidden": [float(v) for v in last_hidden],
            "first_logits": first_logits,
            "final_step_logits": final_step_logits,
            "generated_token_ids": generated,
            "raw_decoded_text": raw_text,
            "transcript": text,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, separators=(",", ":")) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "transcript": text,
                "raw_decoded_text": raw_text,
                "stft_frames": stft_frames,
                "encoder_rows": encoder_rows,
                "audio_rows": int(num_audio_tokens),
                "prompt_tokens": prompt_len,
                "generated_tokens": len(generated),
                "first_token": first_logits["argmax"],
                "eos_token_id": eos_token_id,
            }
        )
    )


if __name__ == "__main__":
    main()
