"""Dump MLX 0.32.3 linear outputs on real Music 3 weights, for a bit-exact diff.

For a handful of real depth-decoder projections, run the operation MLX itself
runs (`x @ w.T` for dense bf16, `mx.quantized_matmul` for 4-bit affine, and the
flow condition encoder's kernel-3 `mx.conv1d`) at every
row count that matters (1 to 16, around the qmv/qmm threshold), and write the
weight, the input and MLX's output so that
`crates/gpu/examples/music3_mlx_linear_cmp.rs` can compare our Metal kernels
bit for bit with no sampling or model code in between.

    python gen_music3_mlx_linear_cases.py --model <pinned 4-bit dir> --output OUT

Needs MLX 0.32.3 (the pin the other Music 3 probes use). Arrays are little-endian;
x and y are f32 holding bf16 values.
"""

import argparse
import glob
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

ROWS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 13, 14, 15, 16]
# (name, tensor prefix, kind). Dense bf16 weights have only `.weight`.
CASES = [
    ("depth_projection", "rvq_depth_decoder.projection", "dense"),
    ("depth_head0", "rvq_depth_decoder.audio_heads.0", "dense"),
    ("depth_l0_to_q", "rvq_depth_decoder.layers.0.attn.to_q", "affine4"),
    ("depth_l0_gate", "rvq_depth_decoder.layers.0.gate_proj", "affine4"),
    ("depth_l0_down", "rvq_depth_decoder.layers.0.down_proj", "affine4"),
]


def bf16_bits(a):
    return np.asarray(a.view(mx.uint16)).astype("<u2")


def f32_of(a):
    return np.asarray(a.astype(mx.float32)).astype("<f4")


def main():
    if mx.__version__ != "0.32.3":
        raise SystemExit(f"Music 3 parity needs MLX 0.32.3, found {mx.__version__}")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    tensors = {}
    for path in sorted(glob.glob(str(args.model / "model-*.safetensors"))):
        tensors.update(mx.load(path))
    mx.random.seed(args.seed)
    manifest = {"mlx_version": mx.__version__, "rows": ROWS, "cases": []}
    for name, prefix, kind in CASES:
        weight = tensors[prefix + ".weight"]
        if kind == "dense":
            out_dim, in_dim = weight.shape
            (args.output / f"{name}.w.bin").write_bytes(bf16_bits(weight).tobytes())
            quant = None
        else:
            scales, biases = tensors[prefix + ".scales"], tensors[prefix + ".biases"]
            out_dim, packed = weight.shape
            in_dim = packed * 32 // 4
            (args.output / f"{name}.w.bin").write_bytes(np.asarray(weight).astype("<u4").tobytes())
            (args.output / f"{name}.scales.f32").write_bytes(f32_of(scales).tobytes())
            (args.output / f"{name}.biases.f32").write_bytes(f32_of(biases).tobytes())
            quant = {"bits": 4, "group_size": in_dim // scales.shape[1]}
        # One input block per case; row r uses its first r rows, so a smaller
        # row count is a prefix of a larger one.
        x = (mx.random.normal((max(ROWS), in_dim)) * 1.5).astype(mx.bfloat16)
        mx.eval(x)
        (args.output / f"{name}.x.f32").write_bytes(f32_of(x).tobytes())
        for rows in ROWS:
            xr = x[:rows]
            if kind == "dense":
                y = xr @ weight.T
            else:
                y = mx.quantized_matmul(
                    xr, weight, scales, biases, transpose=True,
                    group_size=quant["group_size"], bits=quant["bits"],
                )
            mx.eval(y)
            assert y.dtype == mx.bfloat16, y.dtype
            (args.output / f"{name}.y{rows}.f32").write_bytes(f32_of(y).tobytes())
        manifest["cases"].append({"name": name, "kind": kind, "in": int(in_dim), "out": int(out_dim), "quant": quant})
        print(name, kind, in_dim, out_dim, flush=True)
    # Convolutions MLX runs for the flow condition encoder and the vocoder. MLX
    # keeps convolution weights channels-last ([O, K, C], and [C_out, K, C_in]
    # for a transposed one); our device convolution takes PyTorch layouts
    # ([oc, ic, K] and [ic, oc, K]) and channel-major [C, len] activations.
    # (name, weight tensor, transposed, stride, padding, dilation, input lengths)
    convs = [
        ("condition_proj", "condition_encoder.proj", False, 1, 1, 1, [1, 2, 3, 5, 8, 16, 17, 40, 100]),
        ("voc_dec_in_proj", "vocoder.dec_in_proj", False, 1, 0, 1, [3, 10, 40]),
        ("voc_conv_in", "vocoder.conv_in", False, 1, 3, 1, [3, 10, 40]),
        ("voc_b0_conv_t1", "vocoder.blocks.0.conv_t1", True, 8, 4, 1, [3, 10, 40]),
        ("voc_b0_u1_conv1", "vocoder.blocks.0.res_unit1.conv1", False, 1, 3, 1, [10, 80, 160]),
        ("voc_b0_u2_conv1", "vocoder.blocks.0.res_unit2.conv1", False, 1, 9, 3, [10, 80, 160]),
        ("voc_b0_u3_conv1", "vocoder.blocks.0.res_unit3.conv1", False, 1, 27, 9, [10, 80, 160]),
        ("voc_b0_u1_conv2", "vocoder.blocks.0.res_unit1.conv2", False, 1, 0, 1, [10, 80, 160]),
    ]
    manifest["convs"] = []
    for name, prefix, transposed, stride, padding, dilation, lens in convs:
        weight = tensors[prefix + ".weight"]
        bias = tensors.get(prefix + ".bias")
        if transposed:
            out_ch, kernel, in_ch = weight.shape
            layout = weight.transpose(2, 0, 1)  # [ic, oc, K]
        else:
            out_ch, kernel, in_ch = weight.shape
            layout = weight.transpose(0, 2, 1)  # [oc, ic, K]
        (args.output / f"{name}.w.bin").write_bytes(bf16_bits(layout).tobytes())
        if bias is not None:
            (args.output / f"{name}.bias.f32").write_bytes(f32_of(bias).tobytes())
        for length in lens:
            x = (mx.random.normal((1, length, in_ch)) * 1.5).astype(mx.bfloat16)
            if transposed:
                raw = mx.conv_transpose1d(x, weight, stride=stride, padding=padding)
            else:
                raw = mx.conv1d(x, weight, stride=stride, padding=padding, dilation=dilation)
            y = raw + bias if bias is not None else raw
            mx.eval(x, raw, y)
            assert y.dtype == mx.bfloat16, y.dtype
            (args.output / f"{name}.x{length}.f32").write_bytes(f32_of(x[0].T).tobytes())
            (args.output / f"{name}.raw{length}.f32").write_bytes(f32_of(raw[0].T).tobytes())
            (args.output / f"{name}.y{length}.f32").write_bytes(f32_of(y[0].T).tobytes())
        manifest["convs"].append({
            "name": name, "in": int(in_ch), "out": int(out_ch), "kernel": int(kernel),
            "stride": stride, "pad": padding, "dilation": dilation, "transpose": transposed,
            "bias": bias is not None, "lens": lens,
        })
        print(name, "convT" if transposed else "conv", in_ch, out_ch, kernel, flush=True)
    (args.output / "cases.json").write_text(json.dumps(manifest, indent=1) + "\n")


if __name__ == "__main__":
    main()
