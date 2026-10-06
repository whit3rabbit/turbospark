#!/usr/bin/env python3
"""Generate a LASR CTC / MedASR inference witness from the pinned checkpoint.

This family has no working upstream end-to-end path in mlx-audio 0.5.7:

- The stock ``load()`` path runs ``LasrForCTC.sanitize``, which transposes
  every 3-axis conv weight to MLX layout. The public MLX conversion of
  MedASR already stores conv weights in MLX layout ``(out, kernel, in)``,
  so the stock sanitize double-transposes them and the model silently
  decodes garbage.
- ``LasrForCTC`` in mlx-audio defines no ``generate`` method, so the
  mlx-audio STT generate entry point raises AttributeError, and its
  ``decode`` returns an empty text string.

This generator therefore reproduces the verified path instead:

1. Features come from the transformers ``LASRProcessor``
   (``LASRFeatureExtractor``), the only working frontend for this family.
2. Weights are loaded raw from the safetensors file with NO conv
   transposition (the stored layout is already MLX layout), skipping
   ``rotary_emb.inv_freq`` buffers and squeezing only the stored
   ``(out, in, 1)`` CTC head to a linear ``(out, in)``.
3. The lasr.py model classes run forward to CTC logits in eval mode
   (MLX BatchNorm otherwise computes batch statistics instead of using
   the checkpoint's running statistics).
4. Greedy decoding follows the transformers ``LasrForCTC.generate`` and
   ``LasrTokenizer._decode`` contract: per-frame argmax, collapse
   consecutive repeats, drop the blank (pad) id, then decode with
   ``skip_special_tokens=True``.
"""

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path

MLX_AUDIO_SOURCE = Path("/Users/whit3rabbit/Documents/GitHub/mlx-audio")
SOURCE_COMMIT = "e1b19b9054bf163f5d812221a54fcc346f1890e9"

sys.path.insert(0, str(MLX_AUDIO_SOURCE))

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from transformers import AutoProcessor  # noqa: E402

from mlx_audio.stt.models.lasr_ctc.config import ModelConfig  # noqa: E402
from mlx_audio.stt.models.lasr_ctc.lasr import LasrForCTC  # noqa: E402


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
    return selected_spots(np.asarray(value, dtype=np.float32))


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

    processor = AutoProcessor.from_pretrained(str(args.model_dir))
    inputs = processor(audio, sampling_rate=16000, return_tensors="np")
    features = inputs["input_features"]
    if features.ndim != 3 or features.shape[-1] != 128:
        raise ValueError(f"unexpected LASR features shape {features.shape}")
    print(f"features: {features.shape}")

    config = ModelConfig.from_dict(json.loads((args.model_dir / "config.json").read_text()))
    model = LasrForCTC(config)

    # Raw weight load: NO conv transposition. The pinned conversion already
    # stores Conv1d weights in MLX layout (out, kernel, in). Only the CTC
    # head needs its stored (out, in, 1) kernel axis squeezed away.
    weights = dict(mx.load(str(args.model_dir / "model.safetensors"), format="safetensors"))
    weights = {name: value for name, value in weights.items() if "rotary_emb.inv_freq" not in name}
    head = weights["ctc_head.weight"]
    if head.ndim != 3 or head.shape[-1] != 1:
        raise ValueError(f"unexpected ctc_head.weight shape {head.shape}")
    weights["ctc_head.weight"] = head.reshape(head.shape[0], head.shape[1])
    model.load_weights(list(weights.items()), strict=False)
    model.eval()
    mx.eval(model.parameters())

    subsampled = model.encoder.subsampler(mx.array(features))
    cos, sin = model.encoder.rotary_emb(subsampled)
    hidden = subsampled
    layer_outputs = {}
    for index, layer in enumerate(model.encoder.layers):
        hidden = layer(hidden, position_embeddings=(cos, sin))
        if index in (0, len(model.encoder.layers) // 2, len(model.encoder.layers) - 1):
            layer_outputs[f"encoder_layer_{index}"] = hidden
    encoder_final = model.encoder.out_norm(hidden)
    logits = model.ctc_head(encoder_final)
    mx.eval(logits)
    mx.eval(subsampled)
    print(f"subsampled: {subsampled.shape}")
    print(f"logits: {logits.shape}")

    argmax_ids = [int(token) for token in np.asarray(mx.argmax(logits[0], axis=-1))]
    # The transformers LasrTokenizer collapses consecutive repeats BEFORE
    # dropping the blank (pad) id; keep that exact order.
    collapsed = [token for token, _ in itertools.groupby(argmax_ids)]
    collapsed = [token for token in collapsed if token != config.pad_token_id]
    transcript = processor.tokenizer.decode(collapsed, skip_special_tokens=True)

    tokenizer_document = json.loads((args.model_dir / "tokenizer.json").read_text())
    vocab_pairs = tokenizer_document["model"]["vocab"]
    if len(vocab_pairs) != config.vocab_size:
        raise ValueError(
            f"tokenizer vocab holds {len(vocab_pairs)} pieces, model vocab_size is "
            f"{config.vocab_size}"
        )
    special_ids = sorted(
        token_id for token_id in processor.tokenizer.all_special_ids if token_id < config.vocab_size
    )

    from transformers.models.lasr.feature_extraction_lasr import (  # noqa: E402
        linear_to_mel_weight_matrix,
    )

    mel_matrix = linear_to_mel_weight_matrix(
        num_mel_bins=128,
        num_spectrogram_bins=257,
        sample_rate=16000.0,
        lower_edge_hertz=125.0,
        upper_edge_hertz=7500.0,
        dtype=np.float64,
    )

    rope_cos = np.asarray(cos, dtype=np.float32)[0, :, 0, :]
    rope_sin = np.asarray(sin, dtype=np.float32)[0, :, 0, :]

    document = {
        "provenance": {
            "source": "mlx-audio lasr_ctc at commit " + SOURCE_COMMIT,
            "repository": "drankush-ai/medasr-mlx-fp32",
            "revision": args.revision,
            "audio": args.audio.name,
            "audio_sha256": sha256_file(args.audio),
            "model_file_sha256": sha256_file(args.model_dir / "model.safetensors"),
            "load_path": (
                "raw safetensors load with no conv transposition (the conversion already "
                "stores MLX (out, kernel, in) conv weights; the mlx-audio 0.5.7 stock "
                "sanitize double-transposes them) and the stored (out, in, 1) ctc_head "
                "squeezed to a linear"
            ),
            "frontend": (
                "transformers LASRProcessor: unfold framing (win 400, hop 160, no "
                "centering), symmetric Hann window in float64, 512-point rfft, power "
                "spectrum, lingvo-style kaldi-mel slope filterbank (257 bins, 125 to "
                "7500 Hz, DC bin excluded), clamp min 1e-5, natural log, cast to f32"
            ),
            "decode": (
                "per-frame argmax, collapse consecutive repeats, drop blank pad id, "
                "decode with skip_special_tokens=True like transformers "
                "LasrForCTC.generate plus LasrTokenizer._decode"
            ),
        },
        "config": {
            "model_type": "lasr_ctc",
            "vocab_size": config.vocab_size,
            "pad_token_id": config.pad_token_id,
            "hidden_size": config.encoder_config.hidden_size,
            "num_hidden_layers": config.encoder_config.num_hidden_layers,
            "num_attention_heads": config.encoder_config.num_attention_heads,
            "intermediate_size": config.encoder_config.intermediate_size,
            "hidden_act": config.encoder_config.hidden_act,
            "conv_kernel_size": config.encoder_config.conv_kernel_size,
            "conv_residual_weights": config.encoder_config.conv_residual_weights,
            "feed_forward_residual_weights": config.encoder_config.feed_forward_residual_weights,
            "num_mel_bins": config.encoder_config.num_mel_bins,
            "subsampling_conv_channels": config.encoder_config.subsampling_conv_channels,
            "subsampling_conv_kernel_size": config.encoder_config.subsampling_conv_kernel_size,
            "subsampling_conv_stride": config.encoder_config.subsampling_conv_stride,
            "layer_norm_eps": config.encoder_config.layer_norm_eps,
            "rope_theta": config.encoder_config.rope_theta,
        },
        "mel_matrix": {
            "shape": [257, 128],
            "spots": selected_spots(mel_matrix.astype(np.float32)),
        },
        "rope": {"cos": spots_of(rope_cos), "sin": spots_of(rope_sin)},
        "input_features": spots_of(features),
        "subsampler": spots_of(subsampled),
        "hidden_states": [
            {"name": name, "spots": spots_of(value)} for name, value in layer_outputs.items()
        ],
        "encoder_final": spots_of(encoder_final),
        "ctc_logits": spots_of(logits[0]),
        "argmax_token_ids": argmax_ids,
        "greedy_token_ids": collapsed,
        "special_token_ids": special_ids,
        "vocab": [piece for piece, _ in vocab_pairs],
        "transcript": transcript,
    }
    args.output.write_text(json.dumps(document, indent=1))
    print(f"wrote {args.output}")
    print(f"transcript: {transcript!r}")
    print(f"argmax ids: {argmax_ids}")
    print(f"greedy ids: {collapsed}")


if __name__ == "__main__":
    main()
