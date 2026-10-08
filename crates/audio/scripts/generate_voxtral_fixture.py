#!/usr/bin/env python3
"""Generate a Voxtral Mini 3B inference witness from the pinned checkpoint."""

import argparse
import hashlib
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import soundfile as sf

from mlx_audio.stt.models.voxtral.config import ModelConfig
from mlx_audio.stt.models.voxtral.voxtral import Model

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

    config = ModelConfig.from_dict(json.loads((args.model_dir / "config.json").read_text()))
    model = Model(config)
    weights = mx.load(str(args.model_dir / "model-00001-of-00002.safetensors"), format="safetensors")
    weights.update(mx.load(str(args.model_dir / "model-00002-of-00002.safetensors"), format="safetensors"))
    weights = model.sanitize(weights)
    model.load_weights(list(weights.items()))
    model.eval()
    mx.eval(model.parameters())
    model = Model.post_load_hook(model, args.model_dir)

    repo_id = "mlx-community/Voxtral-Mini-3B-2507-bf16"
    inputs = model._processor.apply_transcription_request(
        language="en", audio=str(args.audio), model_id=repo_id
    )
    input_ids = mx.array(inputs["input_ids"])
    input_features = mx.array(inputs["input_features"]).transpose(0, 2, 1)
    mx.eval(input_ids, input_features)

    audio_embeds = model.get_audio_embeds(input_features)
    mx.eval(audio_embeds)
    tower_out = model.audio_tower(input_features)
    mx.eval(tower_out)

    from mlx_audio.lm.models.cache import make_prompt_cache

    cache = make_prompt_cache(model.language_model)
    input_embeddings = model._merge_input_embeddings(
        input_ids=input_ids, input_features=input_features
    )[0]
    logits = model.language_model(input_embeddings=input_embeddings[None], cache=cache)
    mx.eval(logits)
    token = int(mx.argmax(logits[:, -1, :]).item())
    eos_ids = set(model._processor.tokenizer.eos_token_ids)
    generated = []
    for _ in range(128):
        if token in eos_ids:
            break
        generated.append(token)
        step_logits = model.language_model(
            inputs=mx.array([[token]], dtype=mx.int32), cache=cache
        )
        mx.eval(step_logits)
        token = int(mx.argmax(step_logits[:, -1, :]).item())

    text = model._processor.decode(generated)

    document = {
        "provenance": {
            "source": "mlx-audio voxtral at commit " + SOURCE_COMMIT,
            "repository": "mlx-community/Voxtral-Mini-3B-2507-bf16",
            "revision": args.revision,
            "audio": args.audio.name,
            "audio_sha256": sha256_file(args.audio),
            "language": "en",
        },
        "input_ids": [int(t) for t in input_ids.flatten().tolist()],
        "input_features": spots_of(input_features),
        "tower_output": spots_of(tower_out),
        "audio_embeds": spots_of(audio_embeds),
        "prefill_last_logits": spots_of(logits[:, -1, :]),
        "generated_token_ids": generated,
        "transcript": text,
    }
    args.output.write_text(json.dumps(document, indent=1))
    print(f"wrote {args.output}")
    print(f"prompt tokens: {input_ids.shape}")
    print(f"features: {input_features.shape}")
    print(f"generated: {generated}")
    print(f"transcript: {text!r}")


if __name__ == "__main__":
    main()
