"""Dump independent stages from mlx-audio at feb25a37 for the Rust probe.

Run with PYTHONPATH pointing at the pinned reference checkout and its MLX
environment. --float32 is a controlled precision comparison, not the normal
checkpoint's BF16 generation contract. Arrays are little-endian f32, planar.
--trace records logical dtypes at native operation boundaries. --hiddens-file
skips AR; --noise-file supplies concatenated per-chunk Gaussian inputs.
"""

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys

import mlx.core as mx
import numpy as np
from mlx_audio.music import load
from mlx_audio.music.models.minimax_music3 import ar
from mlx_audio.music.models.minimax_music3 import dit, euler, fusion

REFERENCE_SOURCE_SHA256 = "a886c16bcb9322986a3a9adac0383ee95e666dbdbd2cf5af8a8c262010ab591a"
REFERENCE_PIN = "feb25a37b07923bae556e59111995071d66afa0d"
MXFP8_REVISION = "d00a12c3c7f80eb66379dd02dd0f30ed0ce2d96e"


def source_paths():
    names = [f"music/models/minimax_music3/{name}.py" for name in
             ["ar", "config", "depth", "dit", "euler", "fusion", "minimax_music3", "prompt", "sampling", "vocoder"]]
    return names + ["lm/models/qwen3.py", "lm/models/cache.py"]


def source_hashes():
    root = Path(ar.__file__).parents[3]
    names = source_paths() + ["lm/models/base.py", "lm/models/activations.py", "lm/models/rope_utils.py"]
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}


def verify_reference():
    if mx.__version__ != "0.32.3":
        raise RuntimeError(f"Music 3 parity requires MLX 0.32.3; found {mx.__version__}")
    root = Path(ar.__file__).parents[3]
    digest = hashlib.sha256()
    for name in source_paths():
        digest.update(name.encode() + b"\0" + (root / name).read_bytes())
    if digest.hexdigest() != REFERENCE_SOURCE_SHA256:
        raise RuntimeError("Music 3/Qwen3 source differs from the feb25a37 reference")


def dtype_name(value):
    return str(value.dtype).removeprefix("mlx.core.")


def read_f32(path):
    data = np.load(path, allow_pickle=False) if path.suffix == ".npy" else np.fromfile(path, dtype="<f4")
    if not np.isfinite(data).all():
        raise ValueError(f"non-finite supplied array: {path}")
    return np.asarray(data, dtype="<f4")


def infer_checkpoint_revision(model, explicit=None):
    if explicit is not None:
        return explicit
    metadata = model / "request.json"
    if metadata.exists():
        revision = json.loads(metadata.read_text()).get("checkpoint_revision")
        if revision is not None:
            return revision
    # macOS resolves /tmp to /private/tmp; compare canonical paths on both sides.
    if model.resolve() == Path("/tmp/turbospark-music3-mxfp8").resolve():
        return MXFP8_REVISION
    return None


class Trace:
    """Observe reference operations without replacing their implementations."""

    def __init__(self, model, output, enabled, max_calls, supplied_noise=None):
        self.output, self.enabled, self.max_calls = output, enabled, max_calls
        self.supplied_noise = supplied_noise
        self.noise_offset = 0
        self.noise_shapes = []
        self.condition_dtype = None
        self.entries, self.counts, self.hooks, self.original_classes = [], {}, {}, {}
        self.active_dit = None
        self.dit_rotary_call = 0
        self.rope_counts = {}
        self.condition_original = type(model.condition_encoder).__call__
        self.attach(model.condition_encoder, after=self.set_condition_dtype)
        if enabled:
            self.attach_model(model)

    def dump(self, stage, value):
        if not self.enabled:
            return
        call = self.counts.get(stage, 0)
        self.counts[stage] = call + 1
        if self.max_calls and call >= self.max_calls:
            return
        mx.eval(value)
        filename = f"trace_{stage}_{call}.f32"
        data = np.asarray(value.astype(mx.float32), dtype="<f4")
        if not np.isfinite(data).all():
            raise RuntimeError(f"non-finite reference trace at {stage}, call {call}")
        data.tofile(self.output / filename)
        self.entries.append({"stage": stage, "call": call, "dtype": dtype_name(value),
                             "shape": list(value.shape), "file": filename})

    def attach(self, module, before=None, after=None):
        entry = self.hooks.setdefault(id(module), {"before": [], "after": []})
        if before is not None:
            entry["before"].append(before if callable(before) else lambda x, name=before: self.dump(name, x))
        if after is not None:
            entry["after"].append(after if callable(after) else lambda x, name=after: self.dump(name, x))
        cls = type(module)
        if cls not in self.original_classes:
            self.original_classes[cls] = cls.__call__

    def attach_model(self, model):
        for index, layer in enumerate(model.language_model.model.layers):
            name = f"ar.{index}"
            self.attach(layer, before=f"{name}.input", after=f"{name}.output")
            self.attach(layer.input_layernorm, after=f"{name}.input_norm")
            attention = layer.self_attn
            for projection, stage in [("q_proj", "q"), ("k_proj", "k"), ("v_proj", "v")]:
                self.attach(getattr(attention, projection), after=f"{name}.{stage}")
            self.attach(attention.q_norm, after=f"{name}.q_norm")
            self.attach(attention.k_norm, after=f"{name}.k_norm")

            def rope(value, module_id=id(attention.rope), prefix=name):
                call = self.rope_counts.get(module_id, 0)
                self.rope_counts[module_id] = call + 1
                self.dump(f"{prefix}.{'q_rope' if call % 2 == 0 else 'k_rope'}", value.transpose(0, 2, 1, 3))

            self.attach(attention.rope, after=rope)
            self.attach(attention.o_proj, before=f"{name}.attention")
            self.attach(layer.post_attention_layernorm, before=f"{name}.residual", after=f"{name}.post_norm")
            for projection, stage in [("gate_proj", "gate"), ("up_proj", "up"), ("down_proj", "down")]:
                self.attach(getattr(layer.mlp, projection), after=f"{name}.{stage}")
            self.attach(layer.mlp.down_proj, before=f"{name}.swiglu")
        self.attach(model.language_model.model.norm, after="ar.final_norm")
        depth = model.rvq_depth_decoder
        for index, layer in enumerate(depth.layers):
            name = f"depth.{index}"
            self.attach(layer, before=f"{name}.input", after=f"{name}.output")
            if index == 0:
                self.attach(layer, before="depth.position")
            self.attach(layer.input_layernorm, after=f"{name}.input_norm")
            for projection, stage in [("to_q", "q"), ("to_k", "k"), ("to_v", "v")]:
                self.attach(getattr(layer.attn, projection), after=f"{name}.{stage}")
            self.attach(layer.attn.to_out, before=f"{name}.attention")
            self.attach(layer.post_attention_layernorm, before=f"{name}.residual", after=f"{name}.post_norm")
            for projection, stage in [("gate_proj", "gate"), ("up_proj", "up"), ("down_proj", "down")]:
                self.attach(getattr(layer, projection), after=f"{name}.{stage}")
            self.attach(layer.down_proj, before=f"{name}.swiglu")
        self.attach(depth.norm, after="depth.final_norm")
        self.attach(model.condition_encoder, before="flow.condition.input", after="flow.condition")
        transformer = model.transformer
        # DiT operates on one batch; retain the batch axis only for rotary dumps.
        single = lambda name: lambda x: self.dump(name, x[0])
        self.attach(transformer.preprocess_conv, after=single("dit.preprocess"))
        self.attach(transformer.proj_in, before=lambda x: self.dump("dit.pre_conv_residual", x[0].T),
                    after=single("dit.proj_in"))
        self.attach(transformer.time_proj, after="dit.time_fourier")
        self.attach(transformer.time_embed.linear_1, after="dit.time_linear1")
        self.attach(transformer.time_embed.linear_2, before="dit.time_silu", after="dit.time_token")
        for index, block in enumerate(transformer.transformer_blocks):
            name = f"dit.{index}"
            self.attach(block, before=single(f"{name}.input"), after=single(f"{name}.output"))
            self.attach(block.norm1, after=single(f"{name}.input_norm"))
            self.attach(block.norm2, before=single(f"{name}.residual"), after=single(f"{name}.post_norm"))
            for projection, stage in [("to_q", "q"), ("to_k", "k"), ("to_v", "v")]:
                self.attach(getattr(block.attn, projection), after=single(f"{name}.{stage}"))
            self.attach(block.attn.to_out[0], before=single(f"{name}.attention"), after=single(f"{name}.attn_out"))
            self.attach(block.ff_in, after=single(f"{name}.ff"))
            self.attach(block.ff_out, before=single(f"{name}.swiglu"), after=single(f"{name}.ff_out"))

            def dit_start(x, prefix=name):
                self.active_dit = prefix
                self.dit_rotary_call = 0

            self.attach(block.attn, before=dit_start)
        self.attach(transformer.proj_out, after=single("dit.proj_out"))
        self.attach(transformer.postprocess_conv, after=single("dit.postprocess"))
        self.attach(transformer, after=lambda x: self.dump("dit.post_conv_residual", x[0]))
        vocoder = model.vocoder

        def vocoder_input(latents):
            half = latents.shape[1] // 2
            for channel in range(2):
                self.dump(f"vocoder.{channel}.input", latents[0, channel * half:(channel + 1) * half])

        def stereo(name, nlc=False):
            def record(x):
                for half in range(2):
                    value = x[half].T if nlc else x[half]
                    self.dump(f"vocoder.{half}.{name}", value)
            return record

        self.attach(vocoder, before=vocoder_input,
                    after=lambda x: [self.dump(f"vocoder.{half}.output", x[0, half:half + 1]) for half in range(2)])
        self.attach(vocoder.dec_in_proj, after=stereo("proj", nlc=True))
        self.attach(vocoder.conv_in, after=stereo("conv_in", nlc=True))
        for index, block in enumerate(vocoder.blocks):
            name = f"block.{index}"
            self.attach(block.snake1, after=stereo(f"{name}.snake"))
            self.attach(block.conv_t1, after=stereo(f"{name}.transpose", nlc=True))
            for unit_index, unit in enumerate((block.res_unit1, block.res_unit2, block.res_unit3)):
                unit_name = f"{name}.unit.{unit_index}"
                self.attach(unit.snake1, after=stereo(f"{unit_name}.snake1"))
                self.attach(unit.conv1, after=stereo(f"{unit_name}.conv1", nlc=True))
                self.attach(unit.snake2, after=stereo(f"{unit_name}.snake2"))
                self.attach(unit.conv2, after=stereo(f"{unit_name}.conv2", nlc=True))
                self.attach(unit, after=stereo(f"{unit_name}.output"))
            self.attach(block, after=stereo(f"block.{index}"))

    def __enter__(self):
        for cls, original in self.original_classes.items():
            def call(module, *args, _original=original, **kwargs):
                hooks = self.hooks.get(id(module), {})
                for callback in hooks.get("before", []):
                    callback(args[0])
                result = _original(module, *args, **kwargs)
                for callback in hooks.get("after", []):
                    callback(result)
                return result
            cls.__call__ = call
        self.original_rotary = dit._apply_partial_rotary

        def rotary(*args, **kwargs):
            value = self.original_rotary(*args, **kwargs)
            if self.active_dit is not None:
                stage = "q_rope" if self.dit_rotary_call % 2 == 0 else "k_rope"
                self.dump(f"{self.active_dit}.{stage}", value)
                self.dit_rotary_call += 1
            return value

        dit._apply_partial_rotary = rotary
        self.original_rotary_tables = dit.FlowMatchingTransformer._rotary

        def rotary_tables(module, length):
            cos, sin = self.original_rotary_tables(module, length)
            half = module.rotary_dim // 2
            self.dump("dit.rotary.cos", cos[:, :half])
            self.dump("dit.rotary.sin", sin[:, :half])
            return cos, sin

        dit.FlowMatchingTransformer._rotary = rotary_tables
        self.original_normal = mx.random.normal

        def normal(*args, **kwargs):
            generated = self.original_normal(*args, **kwargs)
            shape = tuple(args[0] if args else kwargs["shape"])
            value = generated
            if self.supplied_noise is not None:
                count = int(np.prod(shape))
                end = self.noise_offset + count
                if end > self.supplied_noise.size:
                    raise ValueError(f"supplied noise has too few values for {shape}")
                value = mx.array(self.supplied_noise[self.noise_offset:end].reshape(shape), dtype=mx.float32)
                self.noise_offset = end
            self.noise_shapes.append(list(shape))
            self.dump("flow.noise", value.astype(self.condition_dtype or mx.float32)[0])
            return value

        mx.random.normal = normal
        self.original_trace = sys.gettrace()
        if self.enabled:
            # The pinned scheduler has no module boundary around its updates.
            # Line events observe its locals without reimplementing Euler math.
            lines, base = inspect.getsourcelines(euler.denoise_chunk)
            stages = {}
            for offset, line in enumerate(lines):
                if line.strip().startswith("conditional = transformer"):
                    stages[base + offset] = "input"
                elif line.strip().startswith("if guidance_scale =="):
                    stages[base + offset] = "conditional"
                elif line.strip().startswith("velocity = unconditional +"):
                    stages[base + offset] = "unconditional"
                elif line.strip() == "mx.eval(x)" and not stages.get(base + offset):
                    stages.setdefault(base + offset, "output")
                    break

            condition_lines, condition_base = inspect.getsourcelines(self.condition_original)
            condition_stages = {}
            for offset, line in enumerate(condition_lines):
                stripped = line.strip()
                if stripped.startswith("hidden = (hidden * weights"):
                    condition_stages[condition_base + offset] = "weights"
                elif stripped.startswith("hidden = self.layer_scale"):
                    condition_stages[condition_base + offset] = "reduced"
                elif stripped.startswith("hidden = self.proj"):
                    condition_stages[condition_base + offset] = "scaled"
                elif stripped.startswith("target = latent_length"):
                    condition_stages[condition_base + offset] = "projected"
                elif stripped.startswith("return nearest_interpolate_1d"):
                    condition_stages[condition_base + offset] = "resampled"
            if set(condition_stages.values()) != {"weights", "reduced", "scaled", "projected", "resampled"}:
                raise RuntimeError("pinned conditioning trace boundaries changed")

            def trace(frame, event, arg):
                if frame.f_code is self.condition_original.__code__:
                    if event == "line" and frame.f_lineno in condition_stages:
                        values = frame.f_locals
                        stage = condition_stages[frame.f_lineno]
                        if stage == "weights":
                            weights = mx.softmax(values["self"].layer_weight_logits.astype(mx.float32), axis=0)
                            self.dump("flow.condition.weights_f32", weights)
                            self.dump("flow.condition.weights", values["weights"])
                        elif stage == "resampled":
                            resampled = fusion.nearest_interpolate_1d(values["hidden"], values["target"])
                            self.dump("flow.condition.resampled", resampled[0])
                        else:
                            self.dump(f"flow.condition.{stage}", values["hidden"][0])
                    return trace
                if frame.f_code is not euler.denoise_chunk.__code__:
                    return None
                if event == "line" and frame.f_lineno in stages:
                    values = frame.f_locals
                    stage = stages[frame.f_lineno]
                    prefix = f"euler.{values['index']}"
                    self.dump(f"{prefix}.{stage}", values["x" if stage in ("input", "output") else stage][0])
                    if stage == "output":
                        self.dump(f"{prefix}.velocity", values["velocity"][0])
                return trace

            sys.settrace(trace)
        return self

    def __exit__(self, *args):
        for cls, original in self.original_classes.items():
            cls.__call__ = original
        dit._apply_partial_rotary = self.original_rotary
        dit.FlowMatchingTransformer._rotary = self.original_rotary_tables
        mx.random.normal = self.original_normal
        sys.settrace(self.original_trace)
        if self.enabled:
            (self.output / "trace.json").write_text(json.dumps(self.entries, indent=2) + "\n", encoding="ascii")

    def set_condition_dtype(self, condition):
        self.condition_dtype = condition.dtype


def capture_ar(model, ids, output, frames, seed):
    codes = []
    original = ar.ar_one_frame
    original_sampler = ar.sample_top_k
    original_lm_logits = ar.lm_logits
    depth_class = type(model.rvq_depth_decoder)
    original_depth_call = depth_class.__call__
    warmup = []
    last_hidden = None
    collecting = True

    def lm_logits(language_model, hidden):
        nonlocal last_hidden
        if collecting:
            last_hidden = hidden
        return original_lm_logits(language_model, hidden)

    def depth_call(self, sequence):
        nonlocal last_hidden
        hidden = original_depth_call(self, sequence)
        if collecting:
            last_hidden = hidden[:, -1]
        return hidden

    def sample(logits, key, top_k):
        result = original_sampler(logits, key, top_k)
        if collecting:
            codebook = len(warmup)
            mx.eval(last_hidden, logits, result[0], key)
            np.asarray(last_hidden.astype(mx.float32), dtype="<f4").tofile(output / f"warmup_{codebook}_hidden.f32")
            np.asarray(logits, dtype="<f4").tofile(output / f"warmup_{codebook}_logits.f32")
            warmup.append({"codebook": codebook, "key": np.asarray(key).tolist(), "sampled": int(result[0].item()),
                           "hidden_dtype": dtype_name(last_hidden), "hidden_shape": list(last_hidden.shape),
                           "logits_dtype": dtype_name(logits), "logits_shape": list(logits.shape)})
        return result

    def capture(*positional, **keywords):
        nonlocal collecting
        result = original(*positional, **keywords)
        collecting = False
        if keywords.get("emit_frame", True) and not result.ended:
            mx.eval(result.semantic_code, result.residual_codes)
            codes.append([int(result.semantic_code[0].item()), *np.asarray(result.residual_codes[0]).tolist()])
        return result

    ar.ar_one_frame = capture
    ar.sample_top_k = sample
    ar.lm_logits = lm_logits
    depth_class.__call__ = depth_call
    try:
        hiddens = ar.generate_frame_hiddens(
            model.language_model, model.rvq_depth_decoder, model.config,
            ids, max_frames=frames, seed=seed,
        )
    finally:
        ar.ar_one_frame = original
        ar.sample_top_k = original_sampler
        ar.lm_logits = original_lm_logits
        depth_class.__call__ = original_depth_call
    return hiddens, codes, warmup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--float32", action="store_true")
    parser.add_argument("--trace", action="store_true", help="write dtype-tagged native intermediate dumps")
    parser.add_argument("--trace-max-calls", type=int, default=0,
                        help="maximum dumps per stage, 0 records all calls")
    parser.add_argument("--hiddens-file", type=Path, help="supplied planar f32/npy [1,frames,codebooks*hidden] input; skip AR")
    parser.add_argument("--noise-file", type=Path, help="supplied f32/npy noise, concatenated in native chunk order")
    parser.add_argument("--checkpoint-revision", help="immutable checkpoint revision for probe provenance")
    args = parser.parse_args()
    if args.trace_max_calls < 0:
        parser.error("--trace-max-calls must be non-negative")
    verify_reference()
    model = load(str(args.model))
    if args.float32:
        # Packed integers and block scales retain their exact encoding.
        model.apply(lambda x: x.astype(mx.float32) if mx.issubdtype(x.dtype, mx.floating) else x)
        mx.eval(model.parameters())
    args.output.mkdir(parents=True, exist_ok=True)
    if args.hiddens_file:
        values = read_f32(args.hiddens_file)
        width = model.config.num_codebooks * model.config.hidden_size
        if values.size == 0 or values.size % width:
            raise ValueError(f"supplied hiddens require a positive multiple of width {width}")
        if values.ndim > 1 and values.shape != (1, values.size // width, width):
            raise ValueError("supplied hiddens must have shape [1,frames,codebooks*hidden]")
        dtype = mx.float32 if args.float32 else model.language_model.model.embed_tokens.weight.dtype
        hiddens = mx.array(values.reshape(1, -1, width), dtype=dtype)
        ids, codes, warmup = None, [], []
    else:
        ids = model._text_ids("Soft piano, warm melody, instrumental", "[instrumental]")
    noise = read_f32(args.noise_file).ravel() if args.noise_file else None
    tracer = Trace(model, args.output, args.trace, args.trace_max_calls, noise)
    with tracer:
        if ids is not None:
            hiddens, codes, warmup = capture_ar(model, ids, args.output, args.frames, args.seed)
        mx.eval(hiddens)
        wave = model._run_flow(hiddens, args.steps, args.seed)
        mx.eval(wave)
    if noise is not None and tracer.noise_offset != noise.size:
        raise ValueError(f"supplied noise contains {noise.size - tracer.noise_offset} unused values")
    np.asarray(hiddens.astype(mx.float32), dtype="<f4").tofile(args.output / "hiddens.f32")
    np.asarray(wave.astype(mx.float32), dtype="<f4").tofile(args.output / "wave.f32")
    (args.output / "codes.json").write_text(json.dumps(codes, indent=2) + "\n")
    (args.output / "warmup.json").write_text(json.dumps(warmup, indent=2) + "\n")
    revision = infer_checkpoint_revision(args.model, args.checkpoint_revision)
    (args.output / "request.json").write_text(json.dumps({
        "reference_pin": REFERENCE_PIN,
        "mlx_version": mx.__version__,
        "reference_source_sha256": REFERENCE_SOURCE_SHA256,
        "reference_source_hashes": source_hashes(), "checkpoint_revision": revision,
        "precision": "float32" if args.float32 else "checkpoint",
        "ids": np.asarray(ids[0]).tolist() if ids is not None else [],
        "frames": int(hiddens.shape[1]) if ids is None else args.frames,
        "emitted_frames": int(hiddens.shape[1]), "steps": args.steps, "seed": args.seed,
        "hiddens_dtype": dtype_name(hiddens), "hiddens_shape": list(hiddens.shape),
        "wave_dtype": dtype_name(wave), "wave_shape": list(wave.shape),
        "supplied_hiddens": str(args.hiddens_file) if args.hiddens_file else None,
        "supplied_noise": str(args.noise_file) if args.noise_file else None,
        "noise_shapes": tracer.noise_shapes, "trace_max_calls": args.trace_max_calls,
    }, indent=2) + "\n")
    print(json.dumps({"frames": int(hiddens.shape[1]), "sampled_frames": len(codes),
                      "shape": list(wave.shape), "finite": bool(np.isfinite(np.asarray(wave.astype(mx.float32))).all())}), flush=True)


if __name__ == "__main__":
    main()
