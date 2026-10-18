"""Dump MLX 0.32.3 linear outputs on real Music 3 weights, for a bit-exact diff.

For a handful of real depth-decoder projections, run the operation MLX itself
runs (`x @ w.T` for dense bf16, `mx.quantized_matmul` for 4-bit affine) at every
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
    (args.output / "cases.json").write_text(json.dumps(manifest, indent=1) + "\n")


if __name__ == "__main__":
    main()
