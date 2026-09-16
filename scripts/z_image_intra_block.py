#!/usr/bin/env python3
"""Capture a labeled Diffusers operation trace for one pinned Z-Image block."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from safetensors import safe_open

from z_image_compare import errors, quantize
from z_image_evidence import check_environment, sha256, validate_capture
from z_image_probe import REFERENCES


class Trace:
    def __init__(self):
        self.arrays = {}

    def save(self, name, value):
        value = value.detach().float().cpu().numpy()
        if not np.isfinite(value).all():
            raise ValueError(f"non-finite operation output: {name}")
        self.arrays[name] = value.copy()


class TracingProcessor:
    """Diffusers' processor with snapshots at the native operation seams."""

    _attention_backend = None
    _parallel_config = None

    def __init__(self, trace):
        self.trace = trace

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, freqs_cis=None):
        from diffusers.models.attention_dispatch import dispatch_attention_fn

        self.trace.save("attention.modulation", hidden_states)
        query = attn.to_q(hidden_states)
        self.trace.save("attention.q_projection", query)
        key = attn.to_k(hidden_states)
        self.trace.save("attention.k_projection", key)
        value = attn.to_v(hidden_states)
        self.trace.save("attention.v_projection", value)

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        def apply_rotary_emb(x_in, freqs):
            with torch.amp.autocast("cuda", enabled=False):
                x = torch.view_as_complex(x_in.float().reshape(*x_in.shape[:-1], -1, 2))
                x_out = torch.view_as_real(x * freqs.unsqueeze(2)).flatten(3)
                return x_out.type_as(x_in)

        if freqs_cis is not None:
            query = apply_rotary_emb(query, freqs_cis)
            key = apply_rotary_emb(key, freqs_cis)
        self.trace.save("attention.q_rope", query.flatten(2, 3))
        self.trace.save("attention.k_rope", key.flatten(2, 3))

        dtype = query.dtype
        query, key = query.to(dtype), key.to(dtype)
        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = attention_mask[:, None, None, :]
        hidden_states = dispatch_attention_fn(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0,
            is_causal=False, backend=self._attention_backend,
            parallel_config=self._parallel_config)
        hidden_states = hidden_states.flatten(2, 3).to(dtype)
        self.trace.save("attention.output", hidden_states)
        output = attn.to_out[0](hidden_states)
        self.trace.save("attention.output_projection", output)
        return output


def load_block(model, block, reference):
    mapping = json.loads(
        (model / "transformer/diffusion_pytorch_model.safetensors.index.json").read_text()
    )["weight_map"]
    prefix = f"layers.{block}."
    inventory = json.loads(Path("docs/verification/z-image-ig0-inputs.json").read_text())
    state = {}
    for shard in sorted({s for k, s in mapping.items() if k.startswith(prefix)}):
        path = "transformer/" + shard
        expected = inventory["weights"][path]["sha256"]
        if reference["verified_weights"].get(path) != expected:
            raise ValueError("checkpoint provenance mismatch")
        if sha256(model / path) != expected:
            raise ValueError("checkpoint payload drift")
        with safe_open(model / path, framework="pt", device="cpu") as reader:
            for key in reader.keys():
                if key.startswith(prefix):
                    state[key[len(prefix):]] = reader.get_tensor(key)
    return state


def quantize_block(block):
    rows = []
    with torch.inference_mode():
        for name, layer in block.named_modules():
            if not isinstance(layer, torch.nn.Linear):
                continue
            if not (name.startswith("attention.to_") or name.startswith("feed_forward.w")):
                continue
            weight = layer.weight.detach().cpu().numpy()
            dequantized = np.empty_like(weight)
            for start in range(0, len(weight), 64):
                dequantized[start:start + 64] = quantize(weight[start:start + 64], 64)[0]
            layer.weight.copy_(torch.from_numpy(dequantized))
            rows.append(name)
    if not rows:
        raise ValueError("no eligible linear weights were quantized")
    return {"layout": "affine-int4-group64-bf16-scale-bias",
            "emulation": "dequantized BF16 reference", "tensors": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--block", type=int, default=28)
    parser.add_argument("--model", type=Path, default=Path("target/ig0/model"))
    args = parser.parse_args()
    if args.block < 0 or args.block >= 30:
        raise ValueError("--block must be in 0..29")
    reference = validate_capture(args.run / "denoise.json")
    check_environment()
    from diffusers.models.transformers.transformer_z_image import ZImageTransformerBlock

    state = load_block(args.model, args.block, reference)
    with torch.device("meta"):
        block = ZImageTransformerBlock(args.block, 3840, 30, 30, 1e-5, True).eval()
    block.load_state_dict(state, assign=True)
    quantization = quantize_block(block)
    # Safetensors stores these payloads as FP32, while the pinned Diffusers
    # pipeline constructs the transformer with torch_dtype=torch.bfloat16.
    # Keep that conversion explicit before the MPS execution.
    block.to(dtype=torch.bfloat16, device="mps")

    root = args.run
    x = torch.from_numpy(np.load(root / f"block_{args.block:02}_input.npy").copy()).to(
        device="mps", dtype=torch.bfloat16)
    mask = torch.from_numpy(np.load(root / f"block_{args.block:02}_mask.npy").copy()).to("mps")
    freqs = torch.from_numpy(np.load(root / f"block_{args.block:02}_freqs.npy").copy()).to("mps")
    modulation = torch.from_numpy(
        np.load(root / f"block_{args.block:02}_modulation.npy").copy()
    ).to(device="mps", dtype=torch.bfloat16)

    trace = Trace()
    trace.save("input", x)
    block.attention.processor = TracingProcessor(trace)
    state_for_hooks = {}

    def modulation_hook(_, __, output):
        scale_msa, gate_msa, scale_mlp, gate_mlp = output.unsqueeze(1).chunk(4, dim=2)
        state_for_hooks["gate_msa"] = gate_msa.tanh()
        trace.save("modulation.scale_msa", 1.0 + scale_msa)
        trace.save("modulation.gate_msa", state_for_hooks["gate_msa"])
        trace.save("modulation.scale_mlp", 1.0 + scale_mlp)
        state_for_hooks["gate_mlp"] = gate_mlp.tanh()
        trace.save("modulation.gate_mlp", state_for_hooks["gate_mlp"])

    def attention_norm2_hook(_, __, output):
        trace.save("attention.output_norm", output)
        trace.save("attention.residual", x + state_for_hooks["gate_msa"] * output)

    def feed_forward_input_hook(_, inputs):
        trace.save("ffn.modulation", inputs[0])

    def feed_forward_hook(_, __, output):
        trace.save("ffn.silu_mul", torch.nn.functional.silu(state_for_hooks["w1"]) * state_for_hooks["w3"])

    def final_hook(_, __, output):
        trace.save("ffn.residual", output)

    block.adaLN_modulation.register_forward_hook(modulation_hook)
    block.attention_norm1.register_forward_hook(lambda _, __, output: trace.save("attention.rms_norm", output))
    block.attention_norm2.register_forward_hook(attention_norm2_hook)
    block.ffn_norm1.register_forward_hook(lambda _, __, output: trace.save("ffn.input_norm", output))
    block.ffn_norm2.register_forward_hook(lambda _, __, output: trace.save("ffn.output_norm", output))
    block.feed_forward.register_forward_pre_hook(feed_forward_input_hook)
    block.feed_forward.register_forward_hook(feed_forward_hook)
    block.feed_forward.w1.register_forward_hook(lambda _, __, output: state_for_hooks.update(w1=output) or trace.save("ffn.w1", output))
    block.feed_forward.w3.register_forward_hook(lambda _, __, output: state_for_hooks.update(w3=output) or trace.save("ffn.w3", output))
    block.feed_forward.w2.register_forward_hook(lambda _, __, output: trace.save("ffn.w2", output))
    block.register_forward_hook(final_hook)
    with torch.inference_mode():
        block(x, mask, freqs, modulation)

    expected = np.load(root / f"block_{args.block:02}_output.npy").copy()
    if not np.array_equal(trace.arrays["ffn.residual"].shape, expected.shape):
        raise ValueError("reference trace final shape mismatch")
    args.out.mkdir(parents=True, exist_ok=True)
    output_path = args.out / f"block_{args.block:02}_reference_intra_trace.npz"
    np.savez(output_path, **trace.arrays)
    report = {
        "block": args.block,
        "source_capture_sha256": sha256(root / "denoise.json"),
        "model_revision": reference["model_revision"],
        "diffusers_revision": REFERENCES["diffusers"][1],
        "quantization": quantization,
        "operations": list(trace.arrays),
        "reference_final_vs_capture": errors(trace.arrays["ffn.residual"], expected),
        "fixture_sha256": sha256(output_path),
        "scope": "one full 1024x1024 main-transformer block, BF16 reference with affine INT4 emulation",
    }
    (args.out / f"block_{args.block:02}_reference_intra_trace.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    main()
