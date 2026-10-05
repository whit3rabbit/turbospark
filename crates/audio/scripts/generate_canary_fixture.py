#!/usr/bin/env python3
"""Generate the Canary parity fixture from the pinned reference stack.

The script is a regeneration tool, never part of the tests. It runs the
mlx-audio 0.5.7 reference (source commit e1b19b9054bf163f5d812221a54fcc346f1890e9)
against the pinned checkpoint Mediform/canary-1b-v2-mlx-q8 at revision
0b6b32ee10f30c89e3ead7249bb636445e3019ee on the shared smoke clip.

Outputs crates/audio/testdata/canary_reference.json with the exact transcript,
greedy token ids, prompt token ids, the embedded SentencePiece tokenizer model,
log-mel and encoder output spot tensors, the decoder prefill last hidden row,
and the prefill logits top 8. The reference pipeline computes in bfloat16
(canary generate casts the mel to bfloat16 and every quantized layer
dequantizes into bfloat16); the fixture records float32 upcasts of those
bfloat16 tensors, so the Rust f32 port is compared with honest relative gates
exactly like the Mega-ASR decoder.
"""

import argparse
import base64
import hashlib
import json
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_audio.utils import apply_quantization, load_weights
from mlx_audio.stt.models.canary import Model
from mlx_audio.stt.models.canary.config import ModelConfig

MAX_TOKENS = 96

SPECIAL_TOKENS = [
    "<|startofcontext|>",
    "<|startoftranscript|>",
    "<|emo:undefined|>",
    "<|endoftext|>",
    "<|pnc|>",
    "<|nopnc|>",
    "<|noitn|>",
    "<|notimestamp|>",
    "<|nodiarize|>",
    "<|en|>",
    "<|de|>",
    "<|fr|>",
]


def to_f32(array):
    """Materialize an MLX array as float32 numpy (bf16 tensors upcast)."""
    mx.eval(array)
    return np.asarray(
        mx.astype(array, mx.float32) if array.dtype != mx.float32 else array
    )


def selected_rows(array):
    if isinstance(array, np.ndarray):
        values = array
    else:
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


def top8(logits: np.ndarray) -> dict:
    top = np.argsort(logits)[::-1][:8]
    return {
        "argmax": int(top[0]),
        "argmax_value": float(logits[top[0]]),
        "top8": [[int(i), float(logits[i])] for i in top],
    }


def load_canary(model_dir: Path):
    """Loads the checkpoint through the reference load path.

    This mirrors mlx_audio.utils.base_load_model for the canary family:
    construct from ModelConfig, sanitize the MLX-native names
    (encoder.* -> encoder.conformer.*, transf_decoder.* / head.classifier ->
    decoder.*), apply the config quantization block (8-bit, MLX default
    group 64), load weights, then attach the tokenizer through
    post_load_hook.
    """
    config = json.loads((model_dir / "config.json").read_text())
    config["model_path"] = str(model_dir)
    model = Model(ModelConfig.from_dict(json.loads(json.dumps(config))))
    weights = load_weights(model_dir)
    weights = model.sanitize(weights)
    apply_quantization(model, config, weights, getattr(model, "model_quant_predicate", None))
    model.load_weights(list(weights.items()))
    mx.eval(model.parameters())
    model.eval()
    model = Model.post_load_hook(model, model_dir)
    return model, config


def encode_fixture(model, samples: np.ndarray) -> dict:
    waveform = mx.array(samples, dtype=mx.float32)
    mel = model._preprocess_audio(waveform)
    mel_f32 = to_f32(mel)
    if mel_f32.ndim != 3:
        raise ValueError(f"expected [B, T, M] mel, got {mel_f32.shape}")
    frames = int(mel_f32.shape[1])
    enc_out, enc_len, enc_mask = model._encode_audio(mel.astype(mx.bfloat16))
    enc_out_f32 = to_f32(enc_out)
    return {
        "mel_frames": frames,
        "mel": selected_rows(mel_f32),
        "encoder_frames": int(np.asarray(enc_len)[0]),
        "encoder_output": selected_rows(enc_out_f32),
        "encoder_mask_all_valid": bool(np.asarray(enc_mask).sum() == enc_mask.size),
    }


def decode_fixture(model, samples: np.ndarray) -> dict:
    """Captures the decoder stages and greedy loop for the en->en pnc task."""
    tokenizer = model._tokenizer
    prompt = tokenizer.build_prompt_tokens(source_lang="en", target_lang="en", use_pnc=True)
    mel = model._preprocess_audio(mx.array(samples, dtype=mx.float32))
    enc_out, enc_len, enc_mask = model._encode_audio(mel.astype(mx.bfloat16))

    decoder = model.decoder
    tokens = mx.array([prompt], dtype=mx.int32)
    length = tokens.shape[1]
    hidden = decoder.embedding(tokens)
    position_ids = mx.arange(0, length)
    position_ids = mx.broadcast_to(position_ids[None, :], (1, length))
    hidden = hidden + decoder.position_embedding(position_ids)
    hidden = decoder.embedding_layer_norm(hidden)
    causal_mask = nn.MultiHeadAttention.create_additive_causal_mask(length)
    for block in decoder.blocks:
        # Pass the raw encoder mask; MultiHeadCrossAttention converts it to
        # the additive -1e9 form itself.
        hidden, _, _ = block(
            hidden,
            enc_out,
            encoder_mask=enc_mask,
            self_attn_mask=causal_mask,
            self_attn_cache=None,
            cross_attn_cache=None,
        )
    hidden = decoder.final_norm(hidden)
    mx.eval(hidden)
    prefill_hidden = to_f32(hidden)[0, -1, :]
    manual_logits = to_f32(decoder.output_proj(hidden))[0, -1, :]

    # Prefill through the reference call to obtain the KV cache the greedy
    # loop continues from; the manual pass above exists only to expose the
    # final hidden row, which __call__ does not return.
    logits_mx, cache = decoder(
        tokens,
        enc_out,
        encoder_mask=enc_mask,
        cache=None,
        start_pos=0,
    )
    mx.eval(logits_mx)
    reference_logits = to_f32(logits_mx)[0, -1, :]
    if int(np.argmax(manual_logits)) != int(np.argmax(reference_logits)):
        raise ValueError("manual prefill disagrees with the reference decoder call")
    logits = reference_logits
    first_logits = top8(logits)

    # Greedy loop, mirroring Model.generate exactly.
    eos_id = tokenizer.eos_id
    generated = []
    next_token = int(mx.array(logits).argmax())
    if next_token != eos_id:
        generated.append(next_token)
        for step in range(MAX_TOKENS - 1):
            token_ids = mx.array([[next_token]], dtype=mx.int32)
            step_logits, cache = decoder(
                token_ids,
                enc_out,
                encoder_mask=enc_mask,
                cache=cache,
                start_pos=len(prompt) + step,
            )
            mx.eval(step_logits)
            next_token = int(step_logits[:, -1, :].argmax())
            if next_token == eos_id:
                break
            generated.append(next_token)
    text = tokenizer.decode(generated)

    # Positional encoding witnesses from both reference tables. The encoder
    # table is recorded as the slice the forward consumes for this clip
    # (positions encoder_frames-1 down to -(encoder_frames-1)), which is
    # exactly the table the Rust port computes on the fly.
    pe_table = to_f32(model.encoder.conformer.pos_enc._pe)
    buffer_len = pe_table.shape[1]
    input_len = int(np.asarray(enc_len)[0])
    start_idx = buffer_len // 2 - (input_len - 1)
    end_idx = buffer_len // 2 + (input_len - 1) + 1
    enc_pe = pe_table[:, start_idx:end_idx, :]
    dec_pe = to_f32(decoder.position_embedding._pos_enc)
    return {
        "prompt_token_ids": prompt,
        "prompt_tokens": len(prompt),
        "prefill_last_hidden": [float(v) for v in prefill_hidden],
        "first_logits": first_logits,
        "generated_token_ids": generated,
        "generation_tokens": len(generated),
        "transcript": text.strip(),
        "encoder_pe_shape": list(enc_pe.shape),
        "encoder_pe_rows": [0, input_len // 2, input_len - 1],
        "encoder_pe_values": [
            [float(enc_pe[0, r, 0]), float(enc_pe[0, r, 1]), float(enc_pe[0, r, 2]), float(enc_pe[0, r, 3])]
            for r in [0, input_len // 2, input_len - 1]
        ],
        "decoder_pe_shape": list(dec_pe.shape),
        "decoder_pe_values": [
            [
                float(dec_pe[0, 0]),
                float(dec_pe[0, 1]),
                float(dec_pe[1, 0]),
                float(dec_pe[1, 1]),
                float(dec_pe[2, 0]),
                float(dec_pe[1023, 0]),
                float(dec_pe[1023, 1]),
            ]
        ],
    }


def tokenizer_fixture(model_dir: Path, decoded: dict) -> dict:
    config = json.loads((model_dir / "config.json").read_text())
    tokenizer_section = config.get("tokenizer")
    if not isinstance(tokenizer_section, dict) or "model_base64" not in tokenizer_section:
        raise ValueError("the pinned config.json must embed tokenizer.model_base64")

    import sentencepiece as spm

    proto = base64.b64decode(tokenizer_section["model_base64"])
    sp = spm.SentencePieceProcessor(model_proto=proto)
    pieces = [sp.id_to_piece(i) for i in range(sp.get_piece_size())]
    token2id = {piece: i for i, piece in enumerate(pieces)}
    selected_ids = sorted(
        set(
            list(range(0, 16))
            + decoded["prompt_token_ids"]
            + decoded["generated_token_ids"]
            + [token2id[t] for t in SPECIAL_TOKENS if t in token2id]
        )
    )
    # Second prompt through the reference builder semantics: de -> fr with
    # punctuation and capitalization disabled.
    prompt_de_fr_nopnc = [
        token2id["<|startofcontext|>"],
        token2id["<|startoftranscript|>"],
        token2id["<|emo:undefined|>"],
        token2id["<|de|>"],
        token2id["<|fr|>"],
        token2id["<|nopnc|>"],
        token2id["<|noitn|>"],
        token2id["<|notimestamp|>"],
        token2id["<|nodiarize|>"],
    ]
    return {
        "model_base64": tokenizer_section["model_base64"],
        "piece_count": len(pieces),
        "special_token_ids": {t: token2id[t] for t in SPECIAL_TOKENS if t in token2id},
        "selected_piece_ids": selected_ids,
        "selected_pieces": [pieces[i] for i in selected_ids],
        "generated_pieces": [pieces[i] for i in decoded["generated_token_ids"]],
        "prompt_de_fr_nopnc": prompt_de_fr_nopnc,
    }


def file_fingerprint(path: Path) -> dict:
    data = path.read_bytes()
    return {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--source-commit", default="e1b19b9054bf163f5d812221a54fcc346f1890e9")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).parent))
    from generate_mega_asr_fixture import read_wav_f32

    samples, sample_rate = read_wav_f32(args.audio)
    if sample_rate != 16000:
        raise ValueError(f"expected 16 kHz audio, got {sample_rate}")

    model, config = load_canary(args.model_dir)
    encoding = encode_fixture(model, samples)
    decoded = decode_fixture(model, samples)

    fixture = {
        "schema": "turbospark.canary.reference/1",
        "provenance": {
            "repository": "Mediform/canary-1b-v2-mlx-q8",
            "revision": args.revision,
            "source_commit": args.source_commit,
            "mlx_audio_version": "0.5.7",
            "checkpoint_files": {
                "config.json": file_fingerprint(args.model_dir / "config.json"),
                "model.safetensors": file_fingerprint(args.model_dir / "model.safetensors"),
            },
            "audio_path": "crates/audio/testdata/qwen3_forced_aligner_reference.wav",
            "audio_sha256": file_fingerprint(args.audio),
            "audio_samples": int(samples.shape[0]),
            "audio_sample_rate": 16000,
            "compute_dtype": (
                "bfloat16 (the reference casts the mel to bfloat16 and every "
                "quantized layer dequantizes into bfloat16)"
            ),
            "task": {"source_lang": "en", "target_lang": "en", "use_pnc": True},
        },
        "config": {
            "preprocessor": config.get("preprocessor"),
            "encoder": config.get("encoder"),
            "transf_decoder": config.get("transf_decoder"),
            "head": config.get("head"),
            "quantization": config.get("quantization"),
        },
        "encoding": encoding,
        "decoder": decoded,
        "tokenizer": tokenizer_fixture(args.model_dir, decoded),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, separators=(",", ":")) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "transcript": decoded["transcript"],
                "mel_frames": encoding["mel_frames"],
                "encoder_frames": encoding["encoder_frames"],
                "prompt_tokens": decoded["prompt_tokens"],
                "generated_tokens": decoded["generation_tokens"],
                "generated_ids": decoded["generated_token_ids"],
            }
        )
    )


if __name__ == "__main__":
    main()
