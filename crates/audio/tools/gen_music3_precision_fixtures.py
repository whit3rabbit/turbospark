#!/usr/bin/env python3
"""Generate independent native BF16 Music 3 fixtures without network access.

From the TurboSpark root:
  PYTHONPATH=/tmp/turbospark-music3-mlx0323:/tmp/turbospark-music3-reference \
    ../mlx-audio/.venv/bin/python crates/audio/tools/gen_music3_precision_fixtures.py

The source and MLX version are checked before generation. Host arrays use f32
storage; their logical dtype is recorded separately. Historical FP32 fixtures
are untouched. The 201-frame model uses the reference's full 512-sample hop so
both waveform crop branches execute instead of falling back on a tiny wave.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from mlx_audio.music.models.minimax_music3 import ar
from mlx_audio.music.models.minimax_music3.config import (
    CHUNK_FRAMES, CHUNK_HOP, CROP_LEFT_LATENT, CROP_RIGHT_LATENT,
    DIT_CFG_SCALE, LATENT_HOP_LENGTH, OVERLAP_LATENT_LENGTH, ModelConfig,
)
from mlx_audio.music.models.minimax_music3.dit import FlowMatchingTransformer, _apply_partial_rotary
from mlx_audio.music.models.minimax_music3.euler import denoise_chunk
from mlx_audio.music.models.minimax_music3.minimax_music3 import (
    Model, _chunk_starts, _crop_waveform,
)
from mlx_audio.music.models.minimax_music3.vocoder import Snake1d

from probe_minimax_music3_reference import REFERENCE_SOURCE_SHA256, verify_reference

OUT = Path(__file__).resolve().parents[1] / "testdata/minimax_music3/precision"
PIN = "feb25a37b07923bae556e59111995071d66afa0d"
WEIGHT_SEED = 1234
OP_SEED = 20261005
SEED = 7
FRAMES = 3
STEPS = 2
TEXT = "Soft piano, warm melody, instrumental"
LYRICS = "[instrumental]"
PROFILES = [
    ("bf16", None, None, None),
    ("affine8", "affine", 8, 32),
    ("affine6", "affine", 6, 32),
    ("affine4", "affine", 4, 32),
    ("mxfp8", "mxfp8", 8, 32),
    ("mxfp4", "mxfp4", 4, 32),
    ("nvfp4", "nvfp4", 4, 16),
]


def dtype_name(array: mx.array) -> str:
    return str(array.dtype).removeprefix("mlx.core.")


def floats(array: mx.array) -> list[float]:
    mx.eval(array)
    return np.asarray(array.astype(mx.float32), dtype="<f4").ravel().tolist()


def integers(array: mx.array) -> list[int]:
    mx.eval(array)
    return np.asarray(array).ravel().tolist()


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="ascii")


def save_f32(path: Path, array: mx.array) -> dict:
    mx.eval(array)
    values = np.asarray(array.astype(mx.float32), dtype="<f4")
    if not np.isfinite(values).all():
        raise RuntimeError(f"non-finite fixture: {path}")
    values.tofile(path)
    return {"file": path.name, "shape": list(array.shape), "dtype": dtype_name(array)}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def provenance() -> dict:
    # These additional files control arithmetic outside the Music model tree.
    root = Path(ar.__file__).parents[3]
    paths = ["lm/models/base.py", "lm/models/activations.py", "lm/models/rope_utils.py"]
    additional = {name: digest(root / name) for name in paths if (root / name).exists()}
    return {
        "reference_pin": PIN, "mlx_version": mx.__version__,
        "reference_source_sha256": REFERENCE_SOURCE_SHA256,
        "additional_source_sha256": additional,
        "checkpoint_revision": "synthetic-seeded-1234",
        "precision": "checkpoint", "parameter_dtype": "bfloat16",
        "storage_dtype": "little-endian float32", "weight_seed": WEIGHT_SEED,
        "op_seed": OP_SEED, "device": str(mx.default_device()),
    }


def new_model(config: ModelConfig, mode=None, bits=None, group_size=None) -> Model:
    mx.random.seed(WEIGHT_SEED)
    model = Model(config)
    model.apply(lambda x: x.astype(mx.bfloat16) if mx.issubdtype(x.dtype, mx.floating) else x)
    mx.eval(model.parameters())
    if mode is not None:
        nn.quantize(model, mode=mode, bits=bits, group_size=group_size,
                    class_predicate=Model.model_quant_predicate)
        mx.eval(model.parameters())
    return model


def save_model(path: Path, model: Model, mode=None, bits=None, group_size=None) -> dict:
    path.mkdir(parents=True, exist_ok=True)
    params = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(str(path / "model.safetensors"), params)
    config = model.config.to_dict()
    config["torch_dtype"] = "bfloat16"
    if mode is not None:
        config["quantization"] = {"mode": mode, "bits": bits, "group_size": group_size}
    write_json(path / "config.json", config)
    return {name: {"dtype": dtype_name(x), "shape": list(x.shape)} for name, x in params.items()}


def capture_ar(path: Path, model: Model) -> tuple[mx.array, dict]:
    ids = model._text_ids(TEXT, LYRICS)
    codes, warmup = [], []
    original_frame = ar.ar_one_frame
    original_sample = ar.sample_top_k
    original_logits = ar.lm_logits
    cls = type(model.rvq_depth_decoder)
    original_depth = cls.__call__
    last_hidden = None
    collecting = True

    def logits(language_model, hidden):
        nonlocal last_hidden
        if collecting:
            last_hidden = hidden
        return original_logits(language_model, hidden)

    def depth(self, sequence):
        nonlocal last_hidden
        result = original_depth(self, sequence)
        if collecting:
            last_hidden = result[:, -1]
        return result

    def sample(values, key, top_k):
        result = original_sample(values, key, top_k)
        if collecting:
            index = len(warmup)
            hidden = save_f32(path / f"warmup_{index}_hidden.f32", last_hidden)
            logit = save_f32(path / f"warmup_{index}_logits.f32", values)
            warmup.append({"codebook": index, "key": integers(key),
                           "sampled": int(result[0].item()), "hidden": hidden, "logits": logit})
        return result

    def frame(*args, **kwargs):
        nonlocal collecting
        result = original_frame(*args, **kwargs)
        collecting = False
        if kwargs.get("emit_frame", True) and not result.ended:
            codes.append([int(result.semantic_code[0].item()), *integers(result.residual_codes[0])])
        return result

    ar.lm_logits, ar.sample_top_k, ar.ar_one_frame, cls.__call__ = logits, sample, frame, depth
    try:
        hiddens = ar.generate_frame_hiddens(model.language_model, model.rvq_depth_decoder,
                                          model.config, ids, max_frames=FRAMES, seed=SEED)
        mx.eval(hiddens)
    finally:
        ar.lm_logits, ar.sample_top_k, ar.ar_one_frame, cls.__call__ = (
            original_logits, original_sample, original_frame, original_depth)
    repeated = ar.generate_frame_hiddens(model.language_model, model.rvq_depth_decoder,
                                        model.config, ids, max_frames=FRAMES, seed=SEED)
    mx.eval(repeated)
    if not np.array_equal(np.asarray(hiddens.astype(mx.float32)),
                          np.asarray(repeated.astype(mx.float32))):
        raise RuntimeError("reference request reset changed AR hiddens")
    write_json(path / "codes.json", codes)
    write_json(path / "warmup.json", warmup)
    request = {
        **provenance(), "text": TEXT, "lyrics": LYRICS, "ids": integers(ids[0]),
        "frames": FRAMES, "emitted_frames": int(hiddens.shape[1]), "steps": STEPS,
        "seed": SEED, "request_reset_exact": True,
        "hiddens": save_f32(path / "hiddens.f32", hiddens),
    }
    return hiddens, request


def model_fixtures(out: Path) -> None:
    for name, mode, bits, group in PROFILES:
        path = out / name
        model = new_model(ModelConfig.tiny(), mode, bits, group)
        descriptors = save_model(path, model, mode, bits, group)
        hiddens, request = capture_ar(path, model)
        wave = model._run_flow(hiddens, STEPS, SEED)
        repeated = model._run_flow(hiddens, STEPS, SEED)
        mx.eval(wave, repeated)
        if not np.array_equal(np.asarray(wave.astype(mx.float32)),
                              np.asarray(repeated.astype(mx.float32))):
            raise RuntimeError("reference request reset changed flow waveform")
        request["wave"] = save_f32(path / "wave.f32", wave)
        request["encoding"] = name
        request["tensor_descriptors"] = descriptors
        write_json(path / "request.json", request)
        print(f"{name}: hiddens {hiddens.shape}, wave {wave.shape}", flush=True)
        del model
        mx.clear_cache()


def long_fixture(out: Path) -> None:
    path = out / "long201"
    config = replace(ModelConfig.tiny(), vocoder_upsampling_ratios=(8, 8, 4, 2))
    model = new_model(config)
    descriptors = save_model(path, model)
    hiddens = mx.random.normal((1, 201, config.hidden_size * config.num_codebooks),
                              key=mx.random.key(OP_SEED)).astype(mx.bfloat16)
    mx.eval(hiddens)
    starts = _chunk_starts(201)
    waves, chunks = [], []
    previous_latent, previous_condition = None, None
    mx.random.seed(SEED + 7)
    for index, start in enumerate(starts):
        condition = model.condition_encoder(hiddens[:, start:min(start + CHUNK_FRAMES, 201)])
        noise = mx.random.normal((1, config.dit_in_channels, condition.shape[1])).astype(condition.dtype)
        latents, condition = denoise_chunk(model.transformer, noise, condition, STEPS,
                                          DIT_CFG_SCALE, previous_latent, previous_condition)
        carry_start = max(0, latents.shape[-1] - 2 * OVERLAP_LATENT_LENGTH)
        carry_end = max(carry_start, latents.shape[-1] - OVERLAP_LATENT_LENGTH)
        previous_latent = latents[..., carry_start:carry_end]
        previous_condition = condition[:, carry_start:carry_end]
        wave = model.vocoder(latents)
        mx.eval(condition, noise, latents, wave)
        cropped = _crop_waveform(wave, index, len(starts))
        if cropped.shape[-1] >= wave.shape[-1]:
            raise RuntimeError("201-frame fixture did not exercise the real crop branch")
        chunks.append({"start": start, "frames": min(CHUNK_FRAMES, 201 - start),
                       "condition": save_f32(path / f"chunk_{index}_condition.f32", condition),
                       "noise": save_f32(path / f"chunk_{index}_noise.f32", noise),
                       "latents": save_f32(path / f"chunk_{index}_latents.f32", latents),
                       "wave_shape": list(wave.shape), "cropped_shape": list(cropped.shape),
                       "carry": [carry_start, carry_end]})
        waves.append(cropped)
    wave = mx.concatenate(waves, axis=-1)
    reference = model._run_flow(hiddens, STEPS, SEED)
    mx.eval(wave, reference)
    if not np.array_equal(np.asarray(wave.astype(mx.float32)), np.asarray(reference.astype(mx.float32))):
        raise RuntimeError("201-frame diagnostic differs from the native flow entry point")
    request = {**provenance(), "frames": 201, "steps": STEPS, "seed": SEED,
               "supplied_hiddens": True, "hiddens": save_f32(path / "hiddens.f32", hiddens),
               "wave": save_f32(path / "wave.f32", wave), "chunks": chunks,
               "chunk_frames": CHUNK_FRAMES, "chunk_hop": CHUNK_HOP,
               "latent_hop": LATENT_HOP_LENGTH, "overlap": OVERLAP_LATENT_LENGTH,
               "crop_left_latent": CROP_LEFT_LATENT, "crop_right_latent": CROP_RIGHT_LATENT,
               "tensor_descriptors": descriptors}
    write_json(path / "request.json", request)
    print(f"long201: native crop shapes {[c['cropped_shape'] for c in chunks]}", flush=True)
    del model
    mx.clear_cache()


def op_fixtures(out: Path) -> None:
    mx.random.seed(OP_SEED)
    cases = []

    def record(name: str, op: str, expected: mx.array, **fields):
        cases.append({"name": name, "op": op, "dtype": dtype_name(expected),
                      "shape": list(expected.shape), "expected": floats(expected), **fields})

    def rand(shape, scale=1.0):
        return (mx.random.normal(shape) * scale).astype(mx.bfloat16)

    for cols in (4, 64, 4096):
        x, weight = rand((2, cols), 2.0), rand((cols,), 0.8)
        y = mx.fast.rms_norm(x, weight, 1e-6)
        normalized = (x.astype(mx.float32) * mx.rsqrt(mx.mean(x.astype(mx.float32) ** 2,
                                                              axis=-1, keepdims=True) + 1e-6))
        single_round = (normalized * weight.astype(mx.float32)).astype(mx.bfloat16)
        record(f"rms_bf16_{cols}", "rms_norm", y, input=floats(x), weight=floats(weight),
               rows=2, cols=cols, eps=1e-6, single_round=floats(single_round))
    x, weight, bias = rand((2, 64), 2.0), rand((64,), 0.8), rand((64,), 0.2)
    y = mx.fast.layer_norm(x, weight, bias, 1e-5)
    normalized = (x.astype(mx.float32) - mx.mean(x.astype(mx.float32), axis=-1, keepdims=True))
    normalized *= mx.rsqrt(mx.mean(normalized ** 2, axis=-1, keepdims=True) + 1e-5)
    single_round = (normalized * weight.astype(mx.float32) + bias.astype(mx.float32)).astype(mx.bfloat16)
    record("layer_bf16_64", "layer_norm", y, input=floats(x), weight=floats(weight),
           bias=floats(bias), rows=2, cols=64, eps=1e-5, single_round=floats(single_round))

    x, up = rand((128,), 3.0), rand((128,), 1.0)
    record("sigmoid_bf16", "sigmoid", mx.sigmoid(x), input=floats(x))
    record("silu_bf16", "silu", nn.silu(x), input=floats(x),
           single_round=floats((x.astype(mx.float32) * mx.sigmoid(x.astype(mx.float32))).astype(mx.bfloat16)))
    record("swiglu_bf16", "swiglu", nn.silu(x) * up, input=floats(x), up=floats(up))
    record("weak_scalar_bf16", "scalar_multiply", x * 1.7, input=floats(x), scalar=1.7)
    strong = mx.array([1.7] * 128, dtype=mx.float32)
    record("strong_f32_promotes_bf16", "multiply", x * strong, input=floats(x),
           input_dtype="bfloat16", operand=floats(strong), operand_dtype="float32")
    record("residual_bf16", "add", x + up, input=floats(x), operand=floats(up))
    reduce_x = rand((8, 64))
    record("condition_sum_bf16", "sum", reduce_x.sum(axis=0), input=floats(reduce_x),
           input_shape=list(reduce_x.shape), axis=0)
    alpha = rand((1, 2, 1), 0.5) + 1.0
    snake_x = rand((1, 2, 16), 1.0)
    snake = Snake1d(2)
    snake.alpha = alpha
    record("snake_bf16", "snake", snake(snake_x), input=floats(snake_x), alpha=floats(alpha),
           input_shape=list(snake_x.shape))
    conditional, unconditional, noise, previous = [rand((128,), 1.0) for _ in range(4)]
    velocity = unconditional + 1.7 * (conditional - unconditional)
    record("euler_guidance_bf16", "euler_guidance", velocity,
           conditional=floats(conditional), unconditional=floats(unconditional), guidance=1.7)
    record("euler_update_bf16", "euler_update", noise + (0.4 - 0.2) * velocity,
           input=floats(noise), velocity=floats(velocity), delta=0.4 - 0.2)
    sigma = 0.37
    record("overlap_blend_bf16", "overlap_blend",
           (1.0 - (1.0 - 1e-6) * sigma) * noise + sigma * previous,
           noise=floats(noise), previous=floats(previous), sigma=sigma)

    for name, mode, bits, group in PROFILES:
        weight, bias = rand((32, 64), 0.2), rand((32,), 0.02)
        quant = None if mode is None else mx.quantize(weight, group_size=group, bits=bits, mode=mode)
        for rows in (1, 40):
            x = rand((rows, 64), 1.0)
            if quant is None:
                y = mx.addmm(bias, x, weight.T)
                product = x @ weight.T
                data = {"weight": floats(weight), "weight_dtype": dtype_name(weight), "encoding": "bf16"}
            else:
                args = {"transpose": True, "group_size": group, "bits": bits, "mode": mode}
                product = mx.quantized_matmul(x, quant[0], quant[1],
                                              quant[2] if mode == "affine" else None, **args)
                y = product + bias
                data = {"weight": integers(quant[0]), "weight_shape": list(quant[0].shape),
                        "encoding": name, "mode": mode, "bits": bits, "group_size": group,
                        "scales": floats(quant[1]) if mode == "affine" else integers(quant[1]),
                        "scale_dtype": dtype_name(quant[1]), "scale_shape": list(quant[1].shape)}
                if mode == "affine":
                    data["offsets"] = floats(quant[2])
                    data["offset_dtype"] = dtype_name(quant[2])
            record(f"linear_{name}_{rows}", "linear", y, input=floats(x), bias=floats(bias),
                   rows=rows, input_dim=64, output_dim=32, product=floats(product), **data)
        for dtype in (mx.float32, mx.float16):
            if mode is None:
                x = rand((2, 64)).astype(dtype)
                w, b = weight.astype(dtype), bias.astype(dtype)
                record(f"linear_dense_{dtype_name(x)}", "linear", mx.addmm(b, x, w.T),
                       input=floats(x), weight=floats(w), bias=floats(b), rows=2, input_dim=64,
                       output_dim=32, encoding=dtype_name(x), weight_dtype=dtype_name(w))

    for transpose in (False, True):
        ic, oc, kernel, length = 2, 3, 6 if transpose else 5, 9
        stride, pad, dilation = (3, 2, 1) if transpose else (2, 4, 2)
        x = rand((1, length, ic))
        weight, bias = rand((oc, kernel, ic), 0.3), rand((oc,), 0.03)
        if transpose:
            product = mx.conv_transpose1d(x, weight, stride=stride, padding=pad, dilation=dilation)
            stored_weight = weight.transpose(2, 0, 1)
            unrounded_product = mx.conv_transpose1d(x.astype(mx.float32), weight.astype(mx.float32),
                                                   stride=stride, padding=pad, dilation=dilation)
        else:
            product = mx.conv1d(x, weight, stride=stride, padding=pad, dilation=dilation)
            stored_weight = weight.transpose(0, 2, 1)
            unrounded_product = mx.conv1d(x.astype(mx.float32), weight.astype(mx.float32),
                                         stride=stride, padding=pad, dilation=dilation)
        y = product + bias
        record("transpose_conv_bf16" if transpose else "conv_bf16", "convolution",
               y[0].T, input=floats(x[0].T), weight=floats(stored_weight), bias=floats(bias),
               input_channels=ic, output_channels=oc, kernel=kernel, stride=stride,
               padding=pad, dilation=dilation, transpose=transpose, input_length=length,
               weight_shape=list(stored_weight.shape), product=floats(product[0].T),
               single_round=floats((unrounded_product + bias.astype(mx.float32)).astype(mx.bfloat16)[0].T))

    def attention_case(dim, queries, keys, causal, native_causal=False):
        batch, heads, kv_heads = 2, 4, 2
        q, k, v = rand((batch, heads, queries, dim)), rand((batch, kv_heads, keys, dim)), rand((batch, kv_heads, keys, dim))
        offset = keys - queries if causal else 0
        mask = mx.arange(keys)[None, :] <= (mx.arange(queries)[:, None] + offset) if causal else None
        if native_causal:
            mask = "causal"
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0 / math.sqrt(dim), mask=mask)
        record(f"attention_bf16_d{dim}_q{queries}_k{keys}", "attention",
               y.transpose(0, 2, 1, 3), q=floats(q.transpose(0, 2, 1, 3)),
               k=floats(k.transpose(0, 2, 1, 3)), v=floats(v.transpose(0, 2, 1, 3)),
               batch=batch, queries=queries, keys=keys, heads=heads, kv_heads=kv_heads,
               dim=dim, causal=causal, offset=offset, time_major=False)

    for dimensions in ((16, 1, 20, False), (16, 4, 7, True),
                       (64, 4, 7, True), (128, 1, 27, True), (128, 3, 7, True)):
        attention_case(*dimensions)

    x = rand((2, 4, 5, 16))
    rope = mx.fast.rope(x, dims=16, traditional=False, base=10000.0, scale=1.0, offset=3)
    record("qwen_fast_rope_bf16", "rope", rope.transpose(0, 2, 1, 3),
           input=floats(x.transpose(0, 2, 1, 3)), batch=2, seq=5, heads=4, dim=16,
           rotary_dim=16, offset=3, theta=10000.0)
    config = ModelConfig.tiny()
    rotary_model = new_model(config)
    cos, sin = rotary_model.transformer._rotary(5)
    x = rand((2, 5, 4, 16))
    record("dit_partial_rope_bf16", "partial_rope", _apply_partial_rotary(x, cos, sin),
           input=floats(x), cos=floats(cos), sin=floats(sin), batch=2, seq=5, heads=4,
           dim=16, rotary_dim=8, table_dtype="float32",
           single_round=floats(_apply_partial_rotary(x.astype(mx.float32), cos, sin).astype(mx.bfloat16)))
    timestep = mx.array([0.37, 0.8], dtype=mx.bfloat16)
    fourier = rotary_model.transformer.time_proj
    record("fourier_bf16", "fourier", fourier(timestep), timestep=floats(timestep),
           weight=floats(fourier.weight), embedding_dim=config.dit_fourier_dim)

    key = mx.random.key(14)
    lower = float(np.nextafter(np.float32(-1.0), np.float32(0.0)))
    uniforms = mx.random.uniform(lower, 1.0, (88200,), key=key)
    transformed = mx.erfinv(uniforms) * float(math.sqrt(2.0))
    generated = mx.random.normal((88200,), key=key)
    mx.eval(uniforms, transformed, generated)
    if not np.array_equal(np.asarray(transformed), np.asarray(generated)):
        raise RuntimeError("Gaussian uniforms do not reproduce native mx.random.normal")
    # Keep representative tails and values close to BF16 ties in a small JSON case.
    z = np.asarray(transformed)
    bits = z.view(np.uint32)
    tie_indices = np.argsort(np.abs((bits & 0xFFFF).astype(np.int32) - 0x8000))[:64]
    indices = sorted(set(range(64)) | set(tie_indices.tolist()) |
                     {int(np.argmax(z)), int(np.argmin(z)), 26189, 59810})
    selected = mx.array(indices, dtype=mx.int32)
    record("gaussian_bf16_ties", "normal_from_uniform", transformed[selected].astype(mx.bfloat16),
           input=floats(uniforms[selected]), input_dtype="float32", indices=indices,
           source_shape=[88200], seed=14, key=integers(key), native_float32=floats(transformed[selected]))

    # Append new coverage so existing deterministic operation inputs stay fixed.
    attention_case(128, 20, 20, True, native_causal=True)
    attention_case(64, 87, 87, False)
    tie_input = mx.array([-0.1650390625], dtype=mx.bfloat16).reshape(1, 1, 1)
    tie_snake = Snake1d(1)
    tie_output = tie_snake(tie_input)
    if floats(tie_output) != [-0.1376953125]:
        raise RuntimeError("native Snake power tie changed in the pinned MLX reference")
    sine = mx.sin(tie_input)
    record("snake_power_tie", "snake", tie_output, input=floats(tie_input), alpha=[1.0],
           input_shape=[1, 1, 1], native_sine=floats(sine), native_power=floats(sine ** 2),
           multiplied_square=floats(sine * sine))
    precise_tie_input = mx.array([0.033203125], dtype=mx.bfloat16).reshape(1, 1, 1)
    precise_tie_output = tie_snake(precise_tie_input)
    if floats(precise_tie_output) != [0.0341796875]:
        raise RuntimeError("native Snake precise power tie changed in the pinned reference")
    precise_sine = mx.sin(precise_tie_input)
    record("snake_precise_power_tie", "snake", precise_tie_output,
           input=floats(precise_tie_input), alpha=[1.0], input_shape=[1, 1, 1],
           native_sine=floats(precise_sine), native_power=floats(precise_sine ** 2))

    # Invoke the pinned method without allocating a transformer or its weights.
    for seq, rotary_dim in ((690, 8), (88, 64)):
        cos, sin = FlowMatchingTransformer._rotary(SimpleNamespace(rotary_dim=rotary_dim), seq)
        cos, sin = cos[:, :rotary_dim // 2], sin[:, :rotary_dim // 2]
        record(f"dit_rotary_tables_f32_seq{seq}_dim{rotary_dim}", "rotary_tables",
               mx.stack([cos, sin]), cos=floats(cos), sin=floats(sin), seq=seq,
               rotary_dim=rotary_dim, theta=10000.0, reference_method="FlowMatchingTransformer._rotary")

    wide_profiles = [(name, mode, bits, group, 512) for name, mode, bits, group in PROFILES]
    wide_profiles.append(("mxfp8", "mxfp8", 8, 32, 4096))
    for index, (name, mode, bits, group, width) in enumerate(wide_profiles):
        seed = OP_SEED + 100 + 3 * index
        weight = (mx.random.normal((8, width), key=mx.random.key(seed)) * 0.2).astype(mx.bfloat16)
        x = mx.random.normal((4, width), key=mx.random.key(seed + 1)).astype(mx.bfloat16)
        bias = (mx.random.normal((8,), key=mx.random.key(seed + 2)) * 0.02).astype(mx.bfloat16)
        if mode is None:
            product = x @ weight.T
            y = mx.addmm(bias, x, weight.T)
            data = {"weight": floats(weight), "weight_dtype": dtype_name(weight), "encoding": "bf16"}
        else:
            quant = mx.quantize(weight, group_size=group, bits=bits, mode=mode)
            product = mx.quantized_matmul(x, quant[0], quant[1],
                                         quant[2] if mode == "affine" else None,
                                         transpose=True, group_size=group, bits=bits, mode=mode)
            y = product + bias
            data = {"weight": integers(quant[0]), "weight_shape": list(quant[0].shape),
                    "encoding": name, "mode": mode, "bits": bits, "group_size": group,
                    "scales": floats(quant[1]) if mode == "affine" else integers(quant[1]),
                    "scale_dtype": dtype_name(quant[1]), "scale_shape": list(quant[1].shape)}
            if mode == "affine":
                data["offsets"] = floats(quant[2])
                data["offset_dtype"] = dtype_name(quant[2])
        record(f"linear_{name}_4_k{width}_n8", "linear", y, input=floats(x), bias=floats(bias),
               rows=4, input_dim=width, output_dim=8, product=floats(product), seed=seed,
               input_seed=seed + 1, bias_seed=seed + 2, input_dtype="bfloat16", bias_dtype="bfloat16", **data)
    write_json(out / "ops.json", {"provenance": provenance(), "cases": cases})
    print(f"ops: {len(cases)} independent native cases", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--ops-only", action="store_true")
    args = parser.parse_args()
    verify_reference()
    args.output.mkdir(parents=True, exist_ok=True)
    op_fixtures(args.output)
    if not args.ops_only:
        model_fixtures(args.output)
        long_fixture(args.output)
    files = [p for p in sorted(args.output.rglob("*")) if p.is_file() and p.name != "manifest.json"]
    write_json(args.output / "manifest.json", {
        **provenance(), "generator": Path(__file__).name,
        "generator_sha256": digest(Path(__file__)),
        "profiles": [p[0] for p in PROFILES], "frames": FRAMES, "steps": STEPS, "seed": SEED,
        "files": [{"path": p.relative_to(args.output).as_posix(), "bytes": p.stat().st_size,
                   "sha256": digest(p)} for p in files],
    })
    print(f"wrote {len(files)} files, {sum(p.stat().st_size for p in files)} bytes", flush=True)


if __name__ == "__main__":
    main()
