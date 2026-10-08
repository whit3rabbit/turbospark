#!/usr/bin/env python3
"""Generate a MOSS-Music inference witness from the pinned checkpoint."""

import argparse
import hashlib
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import soundfile as sf

from mlx_audio.stt.models.moss_music.config import ModelConfig
from mlx_audio.stt.models.moss_music.moss_music import Model
from mlx_audio.stt.models.moss_music.processor import MossMusicProcessor

SOURCE_COMMIT = "e1b19b9054bf163f5d812221a54fcc346f1890e9"
TRANSCRIPTION_PROMPT = "Please transcribe the lyrics of this clip."


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

    config = ModelConfig.from_dict(json.loads((args.model_dir / "config.json").read_text()))
    model = Model(config)
    weights = mx.load(str(args.model_dir / "model.safetensors"), format="safetensors")
    weights = model.sanitize(weights)
    from mlx_audio.utils import apply_quantization

    apply_quantization(model, json.loads((args.model_dir / "config.json").read_text()), weights, model.model_quant_predicate)
    model.load_weights(list(weights.items()))
    model.eval()
    mx.eval(model.parameters())
    model = Model.post_load_hook(model, args.model_dir)

    processor = model._ensure_processor()
    processed = processor(text=TRANSCRIPTION_PROMPT, audio=mx.array(audio))

    prompt_ids, input_embeds, deepstack, _ = model._build_prompt_embeddings(processed)
    mx.eval(input_embeds)
    if deepstack is not None:
        for ds in deepstack:
            mx.eval(ds)

    cache = model.make_cache()

    def model_call(ids, embeds, ds):
        return model(
            ids[None],
            cache=cache,
            input_embeddings=embeds[None] if embeds is not None else None,
            deepstack_embeddings=[x[None] for x in ds] if ds is not None else None,
        )

    total = int(prompt_ids.shape[0])
    processed_count = 0
    while total - processed_count > 1:
        remaining = (total - processed_count) - 1
        n = min(2048, remaining)
        ds_slice = [x[0, processed_count:processed_count + n] for x in deepstack] if deepstack is not None else None
        model_call(
            prompt_ids[processed_count:processed_count + n],
            input_embeds[0, processed_count:processed_count + n],
            ds_slice,
        )
        processed_count += n
    ds_tail = [x[0, processed_count:] for x in deepstack] if deepstack is not None else None
    logits = model_call(prompt_ids[processed_count:], input_embeds[0, processed_count:], ds_tail)[:, -1, :]
    mx.eval(logits)
    token = int(mx.argmax(logits).item())
    generated = []
    for _ in range(256):
        if token == int(model.config.eos_token_id):
            break
        generated.append(token)
        step_logits = model(mx.array([[token]], dtype=mx.int32), cache=cache)[:, -1, :]
        mx.eval(step_logits)
        token = int(mx.argmax(step_logits).item())

    text = processor.decode(generated, skip_special_tokens=True)
    text = model._strip_thinking(text)
    segments = model._parse_structured_segments(text)

    document = {
        "provenance": {
            "source": "mlx-audio moss_music at commit " + SOURCE_COMMIT,
            "repository": "mlx-community/MOSS-Music-8B-Thinking-4bit",
            "revision": args.revision,
            "audio": args.audio.name,
            "audio_sha256": sha256_file(args.audio),
            "transcription_prompt": TRANSCRIPTION_PROMPT,
            "greedy": True,
        },
        "input_features": spots_of(processed.audio_data[0]),
        "prompt_token_ids": [int(t) for t in prompt_ids.tolist()],
        "audio_input_mask_head": [bool(v) for v in processed.audio_input_mask[:64].tolist()],
        "audio_mask_true_count": int(np.array(processed.audio_input_mask).sum()),
        "prefill_last_hidden": spots_of(mx.astype(logits, mx.float32).reshape(1, -1)),
        "generated_token_ids": generated,
        "transcript_raw": processor.decode(generated, skip_special_tokens=True),
        "transcript": text,
        "segments": segments,
    }
    args.output.write_text(json.dumps(document, indent=1))
    print(f"wrote {args.output}")
    print(f"prompt tokens: {total}")
    print(f"generated: {generated}")
    print(f"transcript: {text!r}")
    print(f"segments: {segments}")


if __name__ == "__main__":
    main()
