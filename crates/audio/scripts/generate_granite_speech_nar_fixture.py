#!/usr/bin/env python3
"""Generate a Granite Speech NAR inference witness from the pinned checkpoint."""

import argparse
import hashlib
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import soundfile as sf

from mlx_audio.stt.models.granite_speech_nar.config import ModelConfig
from mlx_audio.stt.models.granite_speech_nar.granite_speech_nar import (
    Model,
    _compute_features,
)
from mlx_audio.stt.models.granite_speech_nar.decoding import (
    add_insertion_slots,
    ctc_collapse_decode,
)

SOURCE_COMMIT = "e1b19b9054bf163f5d812221a54fcc346f1890e9"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_spots(array: np.ndarray) -> dict:
    if array.ndim == 3:
        array = array[0]
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
    waveform = mx.array(audio, dtype=mx.float32)

    config = ModelConfig.from_pretrained(args.model_dir)
    model = Model(config)

    weights = mx.load(str(args.model_dir / "model.safetensors"), format="safetensors")
    weights = model.sanitize(weights)
    model.load_weights(list(weights.items()))
    model.eval()
    mx.eval(model.parameters())

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir), trust_remote_code=True)
    model._tokenizer = tokenizer

    features = _compute_features(waveform)
    document_features = spots_of(features)

    # Stage-by-stage run mirroring Model._transcribe_tokens.
    feats = features[None].astype(mx.bfloat16)
    enc_out = model.encoder(feats)
    bpe_argmax = mx.argmax(enc_out.bpe_logits[0], axis=-1).astype(mx.int32)
    hypothesis = ctc_collapse_decode(bpe_argmax, blank_id=config.blank_token_id)
    mx.eval(enc_out.bpe_logits, hypothesis)

    fused = mx.concatenate(enc_out.hidden_states_for_projector, axis=-1)
    audio_embeds = model.projector(fused)
    audio_embeds = audio_embeds / config.text.embedding_multiplier
    mx.eval(audio_embeds)

    text_ids = add_insertion_slots(hypothesis, blank_id=config.blank_token_id,
                                   min_len=config.min_edit_sequence_length)
    text_embeds = model.editor.embed_tokens(text_ids).astype(audio_embeds.dtype)
    audio_len = audio_embeds.shape[1]
    flat_embeds = mx.concatenate([audio_embeds[0], text_embeds], axis=0)[None]
    position_ids = mx.arange(audio_len + text_ids.shape[0], dtype=mx.int32)
    logits = model.editor(inputs_embeds=flat_embeds, position_ids=position_ids,
                          logits_start=audio_len)
    mx.eval(logits)
    edited_argmax = mx.argmax(logits[0], axis=-1).astype(mx.int32)
    final_tokens = ctc_collapse_decode(edited_argmax, blank_id=config.blank_token_id)
    mx.eval(final_tokens)
    text = tokenizer.decode([int(t) for t in final_tokens.tolist()],
                            skip_special_tokens=True)

    char_logits = enc_out.char_logits
    hs_list = enc_out.hidden_states_for_projector
    document = {
        "provenance": {
            "source": "mlx-audio granite_speech_nar at commit " + SOURCE_COMMIT,
            "repository": "mlx-community/granite-speech-4.1-2b-nar-mlx",
            "revision": args.revision,
            "audio": args.audio.name,
            "audio_sha256": sha256_file(args.audio),
        },
        "input_features": document_features,
        "char_logits": spots_of(char_logits),
        "bpe_logits": spots_of(enc_out.bpe_logits),
        "hypothesis_token_ids": [int(t) for t in hypothesis.tolist()],
        "projector_hidden_state0": spots_of(hs_list[0]),
        "projector_hidden_state_last": spots_of(hs_list[-1]),
        "audio_embeddings": spots_of(audio_embeds),
        "editor_text_ids": [int(t) for t in text_ids.tolist()],
        "editor_logits": spots_of(logits[0]),
        "final_token_ids": [int(t) for t in final_tokens.tolist()],
        "transcript": text,
    }
    args.output.write_text(json.dumps(document, indent=1))
    print(f"wrote {args.output}")
    print(f"hypothesis: {[int(t) for t in hypothesis.tolist()]}")
    print(f"final: {text!r}")


if __name__ == "__main__":
    main()
