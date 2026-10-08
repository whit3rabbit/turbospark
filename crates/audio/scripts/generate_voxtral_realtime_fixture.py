#!/usr/bin/env python3
"""Generate a Voxtral Realtime 4B offline transcription witness from the pinned checkpoint.

Mirrors the offline buffered path of
mlx_audio.stt.models.voxtral_realtime.Model.generate at mlx-audio 0.5.7:
resample-free 16 kHz input, `_pad_audio_streaming` silence padding in
1280-sample token units, the 128-band log-mel frontend, the causal
sliding-window encoder with the 4x downsample + adapter projection, the
[BOS] + STREAMING_PAD prompt summed per position with the audio embeddings,
the Mistral-style decoder prefill, and the greedy decode until EOS followed
by the Tekken (sentencepiece-style BPE) detokenization.
"""

import argparse
import hashlib
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import soundfile as sf
from mlx_audio.stt.models.voxtral_realtime.config import _num_delay_tokens
from mlx_audio.stt.models.voxtral_realtime.voxtral_realtime import (
    _pad_audio_streaming,
)
from mlx_audio.stt.utils import load_model

SOURCE_COMMIT = "e1b19b9054bf163f5d812221a54fcc346f1890e9"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_f32le(array: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(array, dtype="<f4").tobytes()
    ).hexdigest()


def selected_spots(array: np.ndarray) -> dict:
    """Spots of a [T, C] matrix: edge, near-edge, center rows and columns."""
    if array.ndim == 3:
        array = array[0]
    if array.ndim == 1:
        array = array[:, None]
    rows = sorted({0, min(1, array.shape[0] - 1), array.shape[0] // 2, array.shape[0] - 1})
    columns = sorted({0, min(1, array.shape[1] - 1), array.shape[1] // 2, array.shape[1] - 1})
    return {
        "shape": list(array.shape),
        "rows": rows,
        "columns": columns,
        "values": [[float(array[row, col]) for col in columns] for row in rows],
    }


def vector_spots(array: np.ndarray, count: int = 16) -> dict:
    """Spots of a 1-D array: first and last four plus evenly spread samples."""
    n = array.shape[0]
    indices = sorted({0, 1, 2, 3, n // 4, n // 2, 3 * n // 4, n - 4, n - 3, n - 2, n - 1}
                     | set(np.linspace(0, n - 1, count, dtype=int).tolist()))
    return {
        "length": int(n),
        "indices": [int(i) for i in indices],
        "values": [float(array[i]) for i in indices],
    }


def log_softmax_top(logits: np.ndarray, index: int) -> float:
    shifted = logits.astype(np.float64) - logits.max()
    total = np.log(np.exp(shifted).sum())
    return float(shifted[index] - total)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()

    audio, sample_rate = sf.read(args.audio, dtype="float32", always_2d=False)
    if sample_rate != 16000:
        raise ValueError(f"expected 16 kHz audio, got {sample_rate} Hz")
    if audio.ndim == 2:
        audio = audio.mean(axis=1, dtype=np.float32)
    audio_np = np.asarray(audio, dtype=np.float32).flatten()

    model = load_model(str(args.model_dir))
    config = model.config
    if config.model_type != "voxtral_realtime":
        raise ValueError(f"unexpected model_type {config.model_type}")
    tokenizer = model._tokenizer

    delay_ms = config.transcription_delay_ms
    n_delay = _num_delay_tokens(delay_ms)
    n_left = config.n_left_pad_tokens
    n_right = (n_delay + 1) + 10

    # ---- Stage 1: offline streaming-mode padding (1280-sample units).
    padded = _pad_audio_streaming(audio_np, n_left, n_right)

    # ---- Stage 2: log-mel frontend (Model._prepare_mel).
    mel, mel_n_delay = model._prepare_mel(audio_np, delay_ms)
    mx.eval(mel)
    if mel_n_delay != n_delay:
        raise ValueError("delay token count disagrees between stages")
    if mel.shape[1] % 2 != 0:
        raise ValueError("mel frame count should already be even after the trim")
    mel_np = np.asarray(mel, dtype=np.float32)

    # ---- Stage 3: causal encoder, conv stem, adapter projection.
    conv_out = model.encoder.conv_stem(mel)
    mx.eval(conv_out)
    ds = model.encoder.config.downsample_factor
    n_audio_total = conv_out.shape[0] // ds
    encoder_path = "full" if conv_out.shape[0] <= model.encoder.config.sliding_window else "chunked"
    adapter_out = model.encoder(mel)
    mx.eval(adapter_out)
    adapter_np = np.asarray(mx.astype(adapter_out, mx.float32), dtype=np.float32)
    if encoder_path == "full":
        # The non-chunked and chunked paths must agree for in-window audio;
        # record the chunked result as a cross-check of that equivalence.
        chunked = mx.concatenate(list(model.encoder.encode_chunks(conv_out)), axis=0)
        chunked_adapter = model.encoder.downsample_and_project(chunked)
        worst = float(np.abs(np.asarray(chunked_adapter, dtype=np.float32) - adapter_np).max())
        print(f"chunked-vs-full adapter worst diff {worst:.3e}")

    # ---- Stage 4: prompt embeddings and prefill.
    prompt_len = 1 + n_left + n_delay
    prompt_ids = [config.bos_token_id] + [config.streaming_pad_token_id] * (n_left + n_delay)
    prompt_text_embeds = model.decoder.embed_tokens(mx.array(prompt_ids))
    prefix_embeds = adapter_out[:prompt_len] + prompt_text_embeds
    hidden, cache = model.decoder.forward(prefix_embeds, start_pos=0)
    logits = model.decoder.logits(hidden[-1])
    mx.eval(logits)
    hidden_np = np.asarray(mx.astype(hidden, mx.float32), dtype=np.float32)
    logits0 = np.asarray(mx.astype(logits, mx.float32), dtype=np.float32)
    argmax0 = int(np.argmax(logits0))
    top_ids = np.argsort(logits0)[::-1][:8]
    watch = sorted({0, 1, 2, 31, 32, 1000, 50_000, 100_000, 131_071,
                    argmax0} - {argmax0} | {argmax0})

    # ---- Stage 5: greedy decode (mirrors the offline generate loop,
    # including the trailing pending token the for/else reads).
    generated = []
    step_records = []
    next_token = argmax0

    def record_step(token: int, step_logits: np.ndarray) -> None:
        step_records.append({
            "token": int(token),
            "top_logprob": log_softmax_top(step_logits, int(token)),
        })

    record_step(next_token, logits0)
    broke = False
    for pos in range(prompt_len, n_audio_total):
        generated.append(int(next_token))
        if next_token == config.eos_token_id or len(generated) > 4096:
            broke = True
            break
        if pos < adapter_out.shape[0]:
            embed = adapter_out[pos] + model.decoder.embed_token(int(next_token))
        else:
            embed = model.decoder.embed_token(int(next_token))
        hidden, cache = model.decoder.forward(embed[None, :], start_pos=pos, cache=cache)
        logits = model.decoder.logits(hidden.squeeze(0))
        mx.eval(logits)
        step_logits = np.asarray(mx.astype(logits, mx.float32), dtype=np.float32)
        next_token = int(np.argmax(step_logits))
        record_step(next_token, step_logits)
    if not broke:
        generated.append(int(next_token))

    raw_generated = list(generated)
    if generated and generated[-1] == config.eos_token_id:
        generated = generated[:-1]
    text = tokenizer.decode(generated).strip()

    # ---- Cross-check against the unmodified upstream generate.
    upstream = model.generate(mx.array(audio_np), max_tokens=4096)
    if upstream.text != text:
        raise ValueError(f"manual decode {text!r} != upstream generate {upstream.text!r}")

    token_bytes = [tokenizer.token_bytes(t).hex() for t in raw_generated]

    document = {
        "provenance": {
            "source": "mlx-audio voxtral_realtime at commit " + SOURCE_COMMIT,
            "repository": "mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit",
            "revision": args.revision,
            "audio": args.audio.name,
            "audio_sha256": sha256_file(args.audio),
        },
        "padding": {
            "raw_samples": int(audio_np.shape[0]),
            "n_left_pad_tokens": int(n_left),
            "n_right_pad_tokens": int(n_right),
            "n_delay_tokens": int(n_delay),
            "transcription_delay_ms": float(delay_ms),
            "padded_samples": int(padded.shape[0]),
            "sha256_f32le": sha256_f32le(padded),
            "spots": vector_spots(np.asarray(padded, dtype=np.float32)),
        },
        "mel": selected_spots(mel_np.T),
        "encoder": {
            "path": encoder_path,
            "conv_frames": int(conv_out.shape[0]),
            "n_audio_total": int(n_audio_total),
            "adapter_len": int(adapter_out.shape[0]),
            "adapter": selected_spots(adapter_np),
        },
        "prompt": {
            "token_ids": [int(t) for t in prompt_ids],
            "bos_token_id": int(config.bos_token_id),
            "eos_token_id": int(config.eos_token_id),
            "streaming_pad_token_id": int(config.streaming_pad_token_id),
            "prompt_len": int(prompt_len),
        },
        "prefill_hidden_last_row": vector_spots(hidden_np[-1]),
        "first_logits": {
            "argmax": argmax0,
            "top_ids": [int(i) for i in top_ids],
            "top_values": [float(logits0[i]) for i in top_ids],
            "watch_indices": [int(i) for i in watch],
            "watch_values": [float(logits0[i]) for i in watch],
        },
        "steps": step_records,
        "generated_token_ids": [int(t) for t in raw_generated],
        "token_bytes_hex": token_bytes,
        "transcript_raw": tokenizer.decode(generated),
        "transcript": text,
    }
    args.output.write_text(json.dumps(document, indent=1))
    print(f"wrote {args.output}")
    print(
        f"audio {audio_np.shape[0]} samples, padded {padded.shape[0]}, mel {mel_np.shape}, "
        f"conv {conv_out.shape[0]} frames, adapter {adapter_out.shape[0]} tokens, "
        f"prompt {prompt_len}, n_audio_total {n_audio_total}"
    )
    print(f"generated {len(raw_generated)} tokens (incl. EOS: {raw_generated[-1] == config.eos_token_id})")
    print(f"transcript: {text!r}")


if __name__ == "__main__":
    main()
