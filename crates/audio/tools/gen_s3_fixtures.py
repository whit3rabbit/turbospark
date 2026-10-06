#!/usr/bin/env python3
"""Regenerate the S3 tokenizer Rust parity fixtures.

Normative reference: mlx-audio at commit
e1b19b9054bf163f5d812221a54fcc346f1890e9
(mlX_audio/codec/models/s3, mlx 0.32.3).

Run from inside the mlx-audio checkout root so `mlx_audio` imports from
the pinned source, with the project venv interpreter:

    cd ../mlx-audio
    ../.venv-mlxaudio/bin/python \
        ../turbospark/crates/audio/tools/gen_s3_fixtures.py

Outputs into crates/audio/testdata/s3/:

- tiny_v1_config.json + tiny_v1_weights.safetensors
    Seeded tiny S3Tokenizer v1 (25 Hz stride). The codebook is written
    explicitly as `quantizer.embed` because the reference stores it on
    the underscore-private `VectorQuantization._codebook`, which the
    MLX parameter tree cannot carry (documented upstream defect).
- tiny_v2_config.json + tiny_v2_weights.safetensors
    Seeded tiny S3TokenizerV2 (FSQ quantizer, RoPE + FSMN attention).
- traces.json + npy files
    Codes, code lengths, quantization margins, dequantized frames for
    v1 (25 Hz and 50 Hz strides, padded and unpadded inputs), v2 short
    and >30 s sliding-window inputs, plus the log-mel front end golden.
- manifest.json
    File list with SHA-256 digests and generator metadata.

The script is deterministic: fixed seeds, no network access.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import fields
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from mlx_audio.codec.models.s3.model import S3Tokenizer
from mlx_audio.codec.models.s3.model_v2 import S3TokenizerV2
from mlx_audio.codec.models.s3.utils import log_mel_spectrogram

OUT = Path("../turbospark/crates/audio/testdata/s3")
REFERENCE_COMMIT = "e1b19b9054bf163f5d812221a54fcc346f1890e9"

V1_TINY = dict(
    n_mels=16,
    n_audio_ctx=64,
    n_audio_state=64,
    n_audio_head=4,
    n_audio_layer=2,
    n_codebook_size=128,
)
V2_TINY = dict(
    n_mels=16,
    n_audio_ctx=64,
    n_audio_state=128,
    n_audio_head=2,
    n_audio_layer=2,
    n_codebook_size=3**8,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_npy(name: str, arr: np.ndarray) -> None:
    np.save(OUT / name, np.ascontiguousarray(arr, dtype=np.float32))


def seed_model(model, seed: int) -> dict:
    flat = dict(tree_flatten(model.parameters()))
    rng = np.random.default_rng(seed)
    seeded = {
        name: mx.array((rng.standard_normal(v.shape) * 0.3).astype(np.float32))
        for name, v in flat.items()
    }
    model.load_weights(list(seeded.items()))
    mx.eval(model)
    return seeded


def code_margins(model_quantized_x: mx.array, codebook: mx.array) -> np.ndarray:
    """Best-vs-second distance margins for the quantized frames."""
    x = np.asarray(model_quantized_x, dtype=np.float32)
    embed = np.asarray(codebook, dtype=np.float32)
    x2 = (x**2).sum(axis=-1, keepdims=True)
    c2 = (embed**2).sum(axis=-1)[None, :]
    dist = x2 - 2.0 * x @ embed.T + c2
    order = np.argsort(dist, axis=-1, kind="stable")
    best = np.take_along_axis(dist, order[:, :1], axis=-1)
    second = np.take_along_axis(dist, order[:, 1:2], axis=-1)
    return (second - best).squeeze(-1)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    traces: dict = {}
    trace_files: list[str] = []

    # ---- v1 (25 Hz stride) -------------------------------------------
    cfg = V1_TINY
    mx.random.seed(101)
    v1 = S3Tokenizer("speech_tokenizer_v1_25hz", type("C", (), cfg)())
    seeded = seed_model(v1, 103)
    # The MLX tree cannot carry the v1 codebook (underscore attr); seed
    # and persist it under the loader-canonical name.
    rng = np.random.default_rng(107)
    codebook = mx.array(rng.standard_normal((cfg["n_codebook_size"], cfg["n_audio_state"])).astype(np.float32))
    v1.quantizer._codebook.embed = codebook
    weights = dict(seeded)
    weights["quantizer.embed"] = codebook
    mx.save_safetensors(str(OUT / "tiny_v1_weights.safetensors"), weights)
    (OUT / "tiny_v1_config.json").write_text(json.dumps(cfg, indent=2))
    (OUT / "tiny_v1_weights_manifest.json").write_text(
        json.dumps({"tiny_v1": [{"name": n, "shape": list(v.shape)} for n, v in sorted(weights.items())]}, indent=2)
    )

    mel_rng = np.random.default_rng(109)
    # The reference builds its masks from mel_len, so the mel must
    # arrive exact-width (batch-of-one has no padded tail).
    for tag, total in (("a", 40), ("b", 48)):
        mel = mel_rng.standard_normal((cfg["n_mels"], total)).astype(np.float32) * 0.5
        save_npy(f"v1_{tag}_mel.npy", mel)
        trace_files.append(f"v1_{tag}_mel.npy")
        batch = mx.array(mel)[None]
        hidden, code_len = v1.encoder(batch, mx.array([total]))
        normed = hidden / mx.sqrt(mx.sum(hidden**2, axis=-1, keepdims=True) + 1e-8)
        code, out_len = v1.quantize(batch, mx.array([total]))
        code = np.asarray(code, dtype=np.int32)[0]
        margins = code_margins(normed[0, : int(out_len[0])], codebook)
        traces[f"v1_{tag}_mel_len"] = total
        traces[f"v1_{tag}_code"] = code.tolist()
        traces[f"v1_{tag}_code_len"] = int(out_len[0])
        traces[f"v1_{tag}_min_margin"] = float(margins.min())
        deq = v1.quantizer.decode(mx.array(code)[None])  # (1, state, T)
        save_npy(f"v1_{tag}_dequant.npy", np.asarray(deq)[0])
        trace_files.append(f"v1_{tag}_dequant.npy")

    # 50 Hz stride through the same weights.
    v1_50 = S3Tokenizer("speech_tokenizer_v1", type("C", (), cfg)())
    v1_50.load_weights(list(seeded.items()))
    v1_50.quantizer._codebook.embed = codebook
    mx.eval(v1_50)
    mel = np.load(OUT / "v1_a_mel.npy")
    code, out_len = v1_50.quantize(mx.array(mel)[None], mx.array([40]))
    traces["v1_50hz_code"] = np.asarray(code, dtype=np.int32)[0].tolist()
    traces["v1_50hz_code_len"] = int(out_len[0])

    # ---- v2 (FSQ) -----------------------------------------------------
    cfg2 = V2_TINY
    mx.random.seed(111)

    # In upstream mlx-audio (commit e1b19b9), AudioEncoderV2.__call__
    # constructed mask as (B, 1, T) instead of 4D (B, 1, 1, T) (unlike v1).
    # When quantize batches multiple sliding windows, the 3D mask broadcasts
    # batch-as-head across SDPA, corrupting head 1 of segment 0 with segment 1's
    # pad mask. Patch mask to 4D (B, 1, 1, T) for clean per-window isolation.
    from mlx_audio.codec.models.s3.model_v2 import AudioEncoderV2
    from mlx_audio.codec.models.s3.utils import make_non_pad_mask, mask_to_bias
    import mlx.nn as nn

    def _fixed_encoder_call(self, x: mx.array, x_len: mx.array):
        mask = make_non_pad_mask(x_len)
        mask = mx.expand_dims(mask, axis=1)
        x = x.transpose(0, 2, 1)
        x = self.conv1(x * mask.transpose(0, 2, 1))
        x = nn.gelu(x)
        x_len = (x_len + 2 - 1 * (3 - 1) - 1) // self.stride + 1
        mask = make_non_pad_mask(x_len)
        x = self.conv2(x * mx.expand_dims(mask, axis=-1))
        x = nn.gelu(x)
        x_len = (x_len + 2 - 1 * (3 - 1) - 1) // 2 + 1
        mask = make_non_pad_mask(x_len)
        mask_pad = mx.expand_dims(mask, axis=-1)
        mask = mask_to_bias(mask, x.dtype)
        mask = mx.expand_dims(mx.expand_dims(mask, axis=1), axis=1)
        for block in self.blocks:
            x = block(x, mask, mask_pad, self._freqs_cis)
        return x, x_len

    AudioEncoderV2.__call__ = _fixed_encoder_call

    v2 = S3TokenizerV2("speech_tokenizer_v2", type("C", (), cfg2)())
    seed_model(v2, 113)

    mx.save_safetensors(str(OUT / "tiny_v2_weights.safetensors"), dict(tree_flatten(v2.parameters())))
    (OUT / "tiny_v2_config.json").write_text(json.dumps(cfg2, indent=2))

    mel_rng = np.random.default_rng(127)
    for tag, total in (("a", 36), ("b", 64)):
        mel = mel_rng.standard_normal((cfg2["n_mels"], total)).astype(np.float32) * 0.5
        save_npy(f"v2_{tag}_mel.npy", mel)
        trace_files.append(f"v2_{tag}_mel.npy")
        hidden, code_len = v2.encoder(mx.array(mel)[None], mx.array([total]))
        code, out_len = v2.quantize(mx.array(mel)[None], mx.array([total]))
        code = np.asarray(code, dtype=np.int32)[0]
        assert int(out_len[0]) == int(code_len[0])
        assert code.max() < 3**8 and code.min() >= 0
        traces[f"v2_{tag}_mel_len"] = total
        traces[f"v2_{tag}_code"] = code.tolist()
        traces[f"v2_{tag}_code_len"] = int(out_len[0])
        traces[f"v2_{tag}_unique_codes"] = int(len(set(code.tolist())))

    # Long audio: 3060 frames -> windows at 0 and 2600.
    total = 3060
    mel = mel_rng.standard_normal((cfg2["n_mels"], total)).astype(np.float32) * 0.5
    save_npy("v2_long_mel.npy", mel)
    trace_files.append("v2_long_mel.npy")
    code, out_len = v2.quantize(mx.array(mel)[None], mx.array([total]))
    code = np.asarray(code, dtype=np.int32)[0]
    traces["v2_long_mel_len"] = total
    traces["v2_long_code_len"] = int(out_len[0])
    np.save(OUT / "v2_long_code.npy", np.ascontiguousarray(code))

    # ---- log-mel front end --------------------------------------------
    rng = np.random.default_rng(131)
    wave = rng.standard_normal(16000).astype(np.float32) * 0.1
    save_npy("mel_wave.npy", wave)
    trace_files.append("mel_wave.npy")
    spec = log_mel_spectrogram(mx.array(wave), n_mels=128)
    save_npy("mel_golden.npy", np.asarray(spec))
    traces["mel_shape"] = [int(d) for d in spec.shape]
    trace_files.append("mel_golden.npy")

    (OUT / "traces.json").write_text(json.dumps(traces, indent=2))

    files = [
        "tiny_v1_config.json",
        "tiny_v1_weights.safetensors",
        "tiny_v1_weights_manifest.json",
        "tiny_v2_config.json",
        "tiny_v2_weights.safetensors",
        "traces.json",
        *trace_files,
        "v2_long_code.npy",
    ]
    manifest = {
        "generator": "tools/gen_s3_fixtures.py",
        "reference_commit": REFERENCE_COMMIT,
        "mlx_version": mx.__version__,
        "files": {name: digest(OUT / name) for name in files},
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(files) + 1} fixture files to {OUT}")


if __name__ == "__main__":
    main()
