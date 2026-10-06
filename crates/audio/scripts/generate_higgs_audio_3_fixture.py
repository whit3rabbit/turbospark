#!/usr/bin/env python3
"""Generate the Higgs Audio v3 STT parity fixture from the reference stack.

The script is a regeneration tool, never part of the tests. It runs the
mlx-audio 0.5.7 reference (source commit e1b19b9054bf163f5d812221a54fcc346f1890e9)
against the pinned checkpoint bosonai/higgs-audio-v3-stt at revision
2ffd1aa39f5a1266931e405cba12e404a9f994b2 on the shared smoke clip.

Outputs crates/audio/testdata/higgs_audio_3_reference.json with the exact
transcript (lowercase, no punctuation, per the DEFAULT_PROMPT), greedy token
ids, the prompt prefix/suffix/full token ids, log-mel, audio-tower, and
projector spot tensors, the decoder prefill last hidden row, and the prefill
logits top 8.

The reference computes the whole pipeline in bfloat16; this port computes f32.
Numeric-stage comparisons therefore use the relative gates documented in the
family README, while the token-level contract (prompt ids, greedy ids,
transcript) must match exactly.
"""

import argparse
import hashlib
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_audio.stt.utils import load_model
from mlx_audio.stt.models.higgs_audio_3.higgs_audio_3 import DEFAULT_PROMPT
from mlx_audio.stt.models.higgs_audio_3.vad import vad_chunk_ranges

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
    parser.add_argument("--repository", default="bosonai/higgs-audio-v3-stt")
    parser.add_argument(
        "--source-commit", default="e1b19b9054bf163f5d812221a54fcc346f1890e9"
    )
    args = parser.parse_args()

    samples, sample_rate = read_wav_f32(args.audio)
    if sample_rate != 16000:
        raise ValueError(f"expected 16 kHz audio, got {sample_rate}")

    model = load_model(str(args.model_dir), model_type="higgs_audio_3")
    tok = model._tokenizer

    def enc(text):
        return [int(v) for v in tok.encode(text, add_special_tokens=False)]

    # Reference _chunk_waveform, with the backend outcome recorded. A failed
    # or unavailable Silero VAD falls back to the empty-cuts uniform split.
    wav = np.asarray(samples, dtype=np.float32).reshape(-1)
    chunk_samples = int(model.config.chunk_size_seconds * model.sample_rate)
    try:
        backend = model._get_vad_backend()
        cuts = [(int(s), int(e)) for s, e in backend.speech_ranges(wav)]
        vad_mode = "silero"
    except Exception as error:  # the reference swallows backend errors
        cuts = []
        vad_mode = f"fallback ({type(error).__name__})"
    if cuts:
        ranges = _chunk_ranges_with_cuts(int(len(wav)), chunk_samples, cuts, model.config.split_vads)
    else:
        ranges = vad_chunk_ranges(wav, chunk_samples, backend=None, split_vads=model.config.split_vads)

    chunks = [wav[s:e] for s, e in ranges]
    max_len = max(len(c) for c in chunks)
    feature_list = []
    mel0 = None
    encoded0 = None
    for index, c in enumerate(chunks):
        if len(c) < max_len:
            c = np.pad(c, (0, max_len - len(c)))
        mel = model._extract_features(mx.array(c, dtype=mx.float32))
        if index == 0:
            mel0 = mel
            encoded0 = model.audio_tower(mel)
            mx.eval(encoded0)
        feature_list.append(model.get_audio_features(mel))

    prefix = enc("<|im_start|>user\n") + enc(DEFAULT_PROMPT) + enc("<|audio_bos|>")
    suffix = (
        enc("<|audio_eos|>") + enc("<|im_end|>\n") + enc("<|im_start|>assistant\n")
    )
    audio_ids = [int(model.audio_in_token_idx)] * len(chunks)
    input_ids = prefix + audio_ids + suffix

    text_embeds = model.model.embed_tokens(mx.array(input_ids)[None])
    dtype = text_embeds.dtype
    segments = [text_embeds[:, : len(prefix), :]]
    for feat in feature_list:
        segments.append(feat.astype(dtype))
    segments.append(text_embeds[:, len(prefix) + len(chunks) :, :])
    inputs_embeds = mx.concatenate(segments, axis=1)
    mx.eval(inputs_embeds)
    prompt_len = inputs_embeds.shape[1]

    # Prefill once over the injected embeddings: the reference generate_step
    # runs the same body through its chunked prefill plus a final last-token
    # call, producing these hidden states and logits.
    hidden = model.model(mx.array([], dtype=mx.int32), cache=model.make_cache(), input_embeddings=inputs_embeds)
    mx.eval(hidden)
    last_hidden = to_f32(hidden)[0, -1, :]
    logits = to_f32(model.lm_head(hidden[:, -1:, :]))[0, 0, :]
    top = np.argsort(logits)[::-1][:8]
    first_logits = {
        "argmax": int(top[0]),
        "argmax_value": float(logits[top[0]]),
        "top8": [[int(i), float(logits[i])] for i in top],
    }

    # Mirror Model.generate exactly: generate_step over the prepared
    # embeddings with a greedy sampler, stopping at the Qwen EOS ids.
    from mlx_audio.lm.generate import generate_step
    from mlx_audio.lm.sample_utils import make_sampler

    sampler = make_sampler(0.0, top_p=1.0, min_p=0.0, top_k=0)
    eos_token_ids = {151645, 151643}
    generated = []
    for token, _ in generate_step(
        prompt=mx.array([], dtype=mx.int32),
        input_embeddings=inputs_embeds.squeeze(0),
        model=model,
        max_tokens=MAX_TOKENS,
        sampler=sampler,
        prefill_step_size=2048,
    ):
        if int(token) in eos_token_ids:
            break
        generated.append(int(token))

    full_text = tok.decode(generated, skip_special_tokens=False)
    text = model._parse_output(full_text)

    # Per-step logits from an equivalent single-stream decode, so the Rust
    # test can compare the decision margins. The final step of the pinned
    # clip is a documented near tie: the reference bf16 logits for the
    # period (13) and <|im_end|> (151645) sit one bf16 ulp apart.
    cache = model.make_cache()
    hidden2 = model.model(
        mx.array([], dtype=mx.int32), cache=cache, input_embeddings=inputs_embeds
    )
    mx.eval(hidden2)
    prev = hidden2[:, -1:, :]
    final_step_logits = None
    for index, tid in enumerate(generated):
        step_logits = np.asarray(mx.astype(model.lm_head(prev), mx.float32))[0, 0, :]
        if index == len(generated) - 1:
            top = np.argsort(step_logits)[::-1][:8]
            final_step_logits = {
                "argmax": int(top[0]),
                "argmax_value": float(step_logits[top[0]]),
                "top8": [[int(i), float(step_logits[i])] for i in top],
                "token_values": {
                    "13": float(step_logits[13]),
                    "151645": float(step_logits[151645]),
                    "151643": float(step_logits[151643]),
                },
            }
        emb = model.model.embed_tokens(mx.array([tid]))
        prev = model.model(mx.array([0], dtype=mx.int32), cache=cache, input_embeddings=emb[None])
        mx.eval(prev)

    fixture = {
        "schema": "turbospark.higgs_audio_3.reference/1",
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
        "backbone": {
            "prompt": DEFAULT_PROMPT,
            "prompt_prefix_ids": prefix,
            "prompt_suffix_ids": suffix,
            "prompt_token_ids": input_ids,
            "prompt_tokens": int(prompt_len),
            "chunk_count": len(chunks),
            "chunk_ranges": [[int(s), int(e)] for s, e in ranges],
            "vad_cuts": [[int(s), int(e)] for s, e in cuts],
            "vad_mode": vad_mode,
            "audio_in_token_idx": int(model.audio_in_token_idx),
            "audio_rows": int(inputs_embeds.shape[1] - len(prefix) - len(suffix)),
            "eos_token_ids": sorted(int(v) for v in eos_token_ids),
            "input_features": selected_rows(mel0.transpose(0, 2, 1)),
            "encoder_output": selected_rows(encoded0),
            "audio_embeddings": selected_rows(feature_list[0]),
            "prefill_last_hidden": [float(v) for v in last_hidden],
            "first_logits": first_logits,
            "final_step_logits": final_step_logits,
            "generated_token_ids": generated,
            "raw_decoded_text": full_text,
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
                "vad_mode": vad_mode,
                "chunks": len(chunks),
                "audio_rows": fixture["backbone"]["audio_rows"],
                "prompt_tokens": int(prompt_len),
                "generated_tokens": len(generated),
                "first_token": first_logits["argmax"],
            }
        )
    )


def _chunk_ranges_with_cuts(total, chunk_samples, cuts, split_vads):
    """Reference vad_chunk_ranges with a nonempty cut list (the helper above
    only mirrors the empty-cuts fallback)."""
    if split_vads:
        wv_chunks = list(cuts)
    else:
        wv_chunks = []
        prev_e = 0
        for idx, (start, end) in enumerate(cuts):
            s = min(prev_e, start)
            e = total if idx == len(cuts) - 1 else end
            if e > s:
                wv_chunks.append((s, e))
            prev_e = e
    out = []
    for s, e in wv_chunks:
        pos = s
        while pos < e:
            nxt = min(e, pos + chunk_samples)
            out.append((pos, nxt))
            pos = nxt
    return out or [(0, total)]


if __name__ == "__main__":
    main()
