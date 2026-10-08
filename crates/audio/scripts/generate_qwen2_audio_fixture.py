#!/usr/bin/env python3
"""Generate a Qwen2-Audio offline transcription witness from the pinned checkpoint.

Mirrors the offline (single-audio) branch of
mlx_audio.stt.models.qwen2_audio.Model.generate at mlx-audio 0.5.7:
16 kHz mono audio padded to 30 seconds, the inline log-mel frontend, the
conv + attention-pooling audio encoder, the linear projector, the chat
prompt with 750 <|AUDIO|> placeholders spliced with the projected features,
then a causal prefill and greedy decode until <|im_end|>.
"""

import argparse
import hashlib
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import soundfile as sf
from mlx_audio.stt.utils import load_model

SOURCE_COMMIT = "e1b19b9054bf163f5d812221a54fcc346f1890e9"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def spots_of(value) -> dict:
    mx.eval(value)
    return selected_spots(np.asarray(mx.astype(value, mx.float32), dtype=np.float32))


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

    model = load_model(str(args.model_dir))
    tokenizer = model._processor.tokenizer
    if tokenizer.eos_token_id != 151645:
        raise ValueError(f"expected <|im_end|> (151645) as eos, got {tokenizer.eos_token_id}")
    if model.audio_token_id != 151646:
        raise ValueError(f"expected <|AUDIO|> (151646) as audio token, got {model.audio_token_id}")

    # ---- Frontend (mirrors Model._extract_features for an ndarray input).
    input_features, num_audio_tokens = model._extract_features(mx.array(audio))
    mx.eval(input_features)
    if num_audio_tokens != 750 or input_features.shape != (1, 128, 3001):
        raise ValueError(
            f"unexpected frontend geometry {input_features.shape} tokens {num_audio_tokens}"
        )

    # ---- Audio encoder (attention pooling) and projector.
    encoder_output = model.audio_tower(input_features)
    mx.eval(encoder_output)
    projected = model.multi_modal_projector(encoder_output)
    projected = mx.astype(projected, mx.float32)
    mx.eval(projected)

    # ---- Prompt (mirrors Model._build_prompt with the default instruction).
    prompt_ids = model._build_prompt([num_audio_tokens], None)
    mx.eval(prompt_ids)
    prompt_ids = [int(t) for t in prompt_ids.tolist()]
    prompt_text = tokenizer.decode(prompt_ids)
    from tokenizers import Tokenizer
    from tokenizers import AddedToken

    # Cross-check with the raw tokenizers BPE, the backend the Rust port
    # builds its tokenizer from. The pinned tokenizer.json carries only the
    # three Qwen chat tokens; the audio and timestamp control tokens exist
    # only in tokenizer_config.json added_tokens_decoder, so the port adds
    # the missing ones in id order.
    rust_style = Tokenizer.from_file(str(args.model_dir / "tokenizer.json"))
    config = json.load(open(args.model_dir / "tokenizer_config.json"))
    decoder = config["added_tokens_decoder"]
    for raw_id in sorted(decoder, key=int):
        entry = decoder[raw_id]
        token_id = int(raw_id)
        content = entry["content"]
        if rust_style.token_to_id(content) == token_id:
            continue
        added = AddedToken(
            content,
            single_word=entry.get("single_word", False),
            lstrip=entry.get("lstrip", False),
            rstrip=entry.get("rstrip", False),
            normalized=entry.get("normalized", False),
            special=entry.get("special", False),
        )
        if entry.get("special", False):
            rust_style.add_special_tokens([added])
        else:
            rust_style.add_tokens([added])
        if rust_style.token_to_id(content) != token_id:
            raise ValueError(f"added token {content} did not retain id {token_id}")
    fast_ids = rust_style.encode(prompt_text, add_special_tokens=False).ids
    if fast_ids != prompt_ids:
        raise ValueError("tokenizers BPE and HF tokenizer prompts disagree")
    audio_positions = [i for i, t in enumerate(prompt_ids) if t == model.audio_token_id]
    if len(audio_positions) != num_audio_tokens:
        raise ValueError("prompt audio placeholder count mismatch")

    # ---- Spliced embeddings (mirrors Model.get_input_embeddings).
    _prompt_ids_mx, inputs_embeds, _n = model.get_input_embeddings(mx.array(audio), None)
    mx.eval(inputs_embeds)
    if inputs_embeds.shape[1] != len(prompt_ids):
        raise ValueError("spliced embedding length mismatch")

    # ---- Prefill hidden states (cache-free) and first-step logits (the
    # cached prefill that generate_step performs, which also populates the
    # KV cache for the greedy loop).
    hidden = model.language_model.model(
        None, cache=None, input_embeddings=inputs_embeds
    )
    mx.eval(hidden)
    hidden_np = np.asarray(mx.astype(hidden[0], mx.float32), dtype=np.float32)

    cache = model.make_cache()
    prefill_logits = model(input_ids=None, cache=cache, input_embeddings=inputs_embeds)
    mx.eval(prefill_logits)
    logits0 = np.asarray(
        mx.astype(prefill_logits[0, -1], mx.float32), dtype=np.float32
    )
    argmax0 = int(np.argmax(logits0))
    top_ids = np.argsort(logits0)[::-1][:8]
    watch = sorted({0, 1, 1000, 50_000, 100_000, 151_643, 151_645, 151_646,
                    151_647, 151_648, 151_935, argmax0})

    # ---- Greedy decode with the upstream KV cache (mirrors generate_step
    # with the default greedy sampler and no logits processors).
    step_logits = mx.array(logits0)[None, None, :]
    generated = []
    step_records = []
    next_token = argmax0
    while next_token != tokenizer.eos_token_id and len(generated) < 4096:
        logprob = mx.log(mx.softmax(step_logits.astype(mx.float32), axis=-1))
        step_records.append({
            "token": int(next_token),
            "top_logprob": float(mx.max(logprob)),
        })
        generated.append(int(next_token))
        token_ids = mx.array([[next_token]])
        step_logits = model(input_ids=token_ids, cache=cache)
        mx.eval(step_logits)
        next_token = int(mx.argmax(step_logits[0, 0]))

    text = tokenizer.decode(generated, skip_special_tokens=True)

    # Cross-check against the unmodified upstream generate on the same input.
    upstream = model.generate(mx.array(audio), max_tokens=4096)
    if upstream.text.strip() != text.strip():
        raise ValueError(
            f"manual greedy {text.strip()!r} != upstream generate {upstream.text.strip()!r}"
        )

    document = {
        "provenance": {
            "source": "mlx-audio qwen2_audio at commit " + SOURCE_COMMIT,
            "repository": "mlx-community/Qwen2-Audio-7B-Instruct-4bit",
            "revision": args.revision,
            "audio": args.audio.name,
            "audio_sha256": sha256_file(args.audio),
        },
        "input_features": spots_of(input_features),
        "encoder_output": spots_of(encoder_output),
        "projected": spots_of(projected),
        "prompt": {
            "text": prompt_text,
            "token_ids": prompt_ids,
            "audio_token_id": int(model.audio_token_id),
            "audio_token_positions": audio_positions,
            "num_audio_tokens": int(num_audio_tokens),
        },
        "inputs_embeds_rows": selected_spots(
            np.concatenate([
                np.asarray(mx.astype(inputs_embeds[0, 0], mx.float32)),
                np.asarray(mx.astype(inputs_embeds[0, inputs_embeds.shape[1] // 2], mx.float32)),
                np.asarray(mx.astype(inputs_embeds[0, -1], mx.float32)),
            ])[None]
        ),
        "prefill_hidden_last_row": vector_spots(hidden_np[-1]),
        "first_logits": {
            "argmax": argmax0,
            "top_ids": [int(i) for i in top_ids],
            "top_values": [float(logits0[i]) for i in top_ids],
            "watch_indices": [int(i) for i in watch],
            "watch_values": [float(logits0[i]) for i in watch],
        },
        "steps": step_records,
        "generated_token_ids": [int(t) for t in generated],
        "transcript": text.strip(),
    }
    args.output.write_text(json.dumps(document, indent=1))
    print(f"wrote {args.output}")
    print(f"prompt tokens: {len(prompt_ids)}  audio tokens: {num_audio_tokens}")
    print(f"generated {len(generated)} tokens")
    print(f"transcript: {text.strip()!r}")


if __name__ == "__main__":
    main()
