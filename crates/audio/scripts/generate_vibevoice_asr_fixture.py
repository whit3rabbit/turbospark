#!/usr/bin/env python3
"""Generate a VibeVoice-ASR offline inference witness from the pinned checkpoint.

Mirrors the offline (single-window) branch of
mlx_audio.stt.models.vibevoice_asr.Model.generate at mlx-audio 0.5.7:
resample to 24 kHz, encode the acoustic and semantic tokenizer features,
project and combine them, build the default chat prompt with the speech
features spliced at the pad positions, then prefill and greedily decode
until one of the two Qwen EOS tokens.
"""

import argparse
import hashlib
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import soundfile as sf

from mlx_audio.stt.models.vibevoice_asr.config import ModelConfig
from mlx_audio.stt.models.vibevoice_asr.vibevoice_asr import Model
from mlx_audio.stt.utils import resample_audio

SOURCE_COMMIT = "e1b19b9054bf163f5d812221a54fcc346f1890e9"
CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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

    config = ModelConfig.from_dict(json.load(open(args.model_dir / "config.json")))
    model = Model(config)

    weights = {}
    for shard in sorted(args.model_dir.glob("model-*.safetensors")):
        weights.update(mx.load(str(shard), format="safetensors"))
    weights = model.sanitize(weights)
    if config.decoder_config.tie_word_embeddings:
        # The pinned checkpoint stores an lm_head that mlx_lm drops for tied
        # embeddings; logits come from embed_tokens as a linear layer.
        weights.pop("language_model.lm_head.weight", None)
    model.load_weights(list(weights.items()))
    model.eval()
    mx.eval(model.parameters())

    from transformers import Qwen2Tokenizer

    tokenizer = Qwen2Tokenizer.from_pretrained(str(args.model_dir))
    tokenizer.chat_template = CHAT_TEMPLATE
    model.tokenizer = tokenizer
    model._speech_start_id = tokenizer.convert_tokens_to_ids("<|object_ref_start|>")
    model._speech_end_id = tokenizer.convert_tokens_to_ids("<|object_ref_end|>")
    model._speech_pad_id = tokenizer.convert_tokens_to_ids("<|box_start|>")

    preprocessor = json.load(open(args.model_dir / "preprocessor_config.json"))
    model.sample_rate = preprocessor.get("target_sample_rate", model.sample_rate)
    model.speech_tok_compress_ratio = preprocessor.get(
        "speech_tok_compress_ratio", model.speech_tok_compress_ratio
    )
    model.normalize_audio = preprocessor.get("normalize_audio", model.normalize_audio)
    model.chunk_frames = preprocessor.get("chunk_frames", model.chunk_frames)
    model.lookahead_frames = preprocessor.get("lookahead_frames", model.lookahead_frames)

    text_chunk_end_id = tokenizer.convert_tokens_to_ids("<|text_chunk_end|>")
    unk_token_id = getattr(tokenizer, "unk_token_id", None)
    if text_chunk_end_id is not None and text_chunk_end_id != unk_token_id:
        model._text_chunk_end_id = text_chunk_end_id

    # ---- Frontend: resample to 24 kHz (normalize_audio is refused by the port).
    audio24 = np.asarray(
        resample_audio(audio, sample_rate, model.sample_rate), dtype=np.float32
    )
    audio_tensor = mx.array(audio24)[None]
    mx.eval(audio_tensor)

    # ---- Tokenizer encoders and connectors (mirrors Model.encode_speech).
    acoustic_tokens = model.acoustic_tokenizer.encode(audio_tensor)
    mx.eval(acoustic_tokens)
    acoustic_features = model.acoustic_connector(acoustic_tokens)
    mx.eval(acoustic_features)
    semantic_tokens = model.semantic_tokenizer.encode(audio_tensor)
    mx.eval(semantic_tokens)
    semantic_features = model.semantic_connector(semantic_tokens)
    mx.eval(semantic_features)
    speech_features = acoustic_features + semantic_features
    mx.eval(speech_features)

    # ---- Prompt (mirrors Model._build_prompt_tokens with no context).
    audio_duration = audio_tensor.shape[1] / 24000
    input_ids, acoustic_input_mask = model._build_prompt_tokens(
        speech_features, audio_duration, None
    )
    mx.eval(input_ids, acoustic_input_mask)
    prompt_ids = [int(t) for t in input_ids[0].tolist()]
    prompt_text = tokenizer.decode(prompt_ids)
    from tokenizers import Tokenizer

    # Cross-check with the raw tokenizers BPE, the backend the Rust port
    # builds its tokenizer from.
    rust_style = Tokenizer.from_file(str(args.model_dir / "tokenizer.json"))
    fast_ids = rust_style.encode(prompt_text, add_special_tokens=False).ids
    if fast_ids != prompt_ids:
        raise ValueError("tokenizers BPE and slow tokenizer prompts disagree")

    # ---- Prefill: embeddings with speech features spliced at pad positions.
    text_embeds = model.get_input_embeddings()(input_ids)
    speech = speech_features.astype(text_embeds.dtype)
    mask_expanded = mx.broadcast_to(acoustic_input_mask[:, :, None], text_embeds.shape)
    cumsum = mx.cumsum(acoustic_input_mask[0].astype(mx.int32))
    speech_idx = mx.clip(cumsum - 1, 0, speech_features.shape[1] - 1)
    expanded = speech_features[0][speech_idx]
    input_embeds = mx.where(mask_expanded, expanded[None], text_embeds)
    mx.eval(input_embeds)

    hidden = model.language_model.model(inputs=None, cache=None, input_embeddings=input_embeds)
    mx.eval(hidden)
    hidden_np = np.asarray(mx.astype(hidden[0], mx.float32), dtype=np.float32)

    embed_matrix = model.get_input_embeddings().weight
    mx.eval(embed_matrix)
    embed_np = np.asarray(mx.astype(embed_matrix, mx.float32), dtype=np.float32)
    logits0 = hidden_np[-1] @ embed_np.T
    top_ids = np.argsort(logits0)[::-1][:8]
    argmax0 = int(np.argmax(logits0))
    watch = sorted({0, 1, 1000, 50_000, 100_000, 151_643, 151_645, 151_646, 151_647,
                    151_648, 151_665, 151_935, argmax0})

    # ---- Greedy decode (mirrors stream_generate: greedy argmax, dual EOS).
    generated = []
    step_records = []
    for token, logprobs in model.stream_generate(
        input_ids=input_ids,
        speech_features=speech_features,
        acoustic_input_mask=acoustic_input_mask,
        max_tokens=8192,
    ):
        token = int(token)
        value = float(mx.max(logprobs))
        step_records.append({"token": token, "top_logprob": value})
        generated.append(token)

    text = tokenizer.decode(generated, skip_special_tokens=True)

    # stream_generate breaks on EOS before yielding it, so the generated
    # list ends at the last non-EOS token by construction.

    document = {
        "provenance": {
            "source": "mlx-audio vibevoice_asr at commit " + SOURCE_COMMIT,
            "repository": "microsoft/VibeVoice-ASR-Streaming-1.5B",
            "revision": args.revision,
            "audio": args.audio.name,
            "audio_sha256": sha256_file(args.audio),
        },
        "resampled_24k": {
            "sample_rate": 24000,
            "sha256_f32le": sha256_bytes(audio24.tobytes()),
            "spots": vector_spots(audio24),
        },
        "acoustic_tokens": spots_of(acoustic_tokens),
        "semantic_tokens": spots_of(semantic_tokens),
        "acoustic_features": spots_of(acoustic_features),
        "semantic_features": spots_of(semantic_features),
        "speech_features": spots_of(speech_features),
        "prompt": {
            "text": prompt_text,
            "token_ids": prompt_ids,
            "speech_pad_positions": int(acoustic_input_mask.sum()),
            "audio_duration_text": f"{audio_duration:.2f}",
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
        "generated_token_ids": [int(t) for t in generated],
        "transcript_raw": text,
        "transcript": text.strip(),
    }
    args.output.write_text(json.dumps(document, indent=1))
    print(f"wrote {args.output}")
    print(f"prompt tokens: {len(prompt_ids)}  speech frames: {speech_features.shape[1]}")
    print(f"generated {len(generated)} tokens")
    print(f"transcript: {text.strip()!r}")


if __name__ == "__main__":
    main()
