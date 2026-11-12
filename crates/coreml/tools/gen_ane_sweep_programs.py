"""Write the Core ML programs `tslab kernel --bench ane_matmul` times.

The bench (`crates/lab/src/kernel/ane_matmul.rs`) compares Steel alone, a
Core ML matmul on the Neural Engine, and the concurrent row split at a few
shares, per install shape and token count. It needs one program per
(K, ANE rows, tokens) and reads the `sweep.json` this script writes. Run it
with the shapes the bench prints when programs are missing:

    uv run --python 3.12 --with coremltools==9.0 --with numpy==2.2.6 \\
        python -I crates/coreml/tools/gen_ane_sweep_programs.py OUT_DIR \\
        --shapes 17408x5120,5120x17408 --tokens 32,128,512,1024,2048 \\
        [--shares 1.0,0.3,0.5,0.7] [--experts 4:32,8:64 --expert-dims 2048,512] \\
        [--const-int4 [--const-block 64]]

Each dense program computes `w[0:MA] @ a^T` in F16 with K sliced at 2048,
exactly as the prefill split's generator does (`gen_ane_prefill_programs.py`,
whose builder this script imports). The weight plane holds every row of the
shape, so one plane serves every share; `MA` is the share rounded down to a
multiple of 32 rows. Shape widths must be multiples of 32 halfs so the plane
stride Core ML pads matches the plane the bench allocates.

`--experts E:g` writes packed-expert programs: E experts of g token rows
each as ONE batched matmul `[E,1,g,K] x [E,1,R,K]^T`, for the gate/up shape
(R = intermediate, K = hidden) and the down shape (R = hidden,
K = intermediate) of `--expert-dims hidden,intermediate`. That is the "one
multi-expert Core ML call" form `docs/SPEEDUP_PLAYBOOK.md` names as the only
reopen path for ANE expert work; `tslab ane-plan` on the output says where
Core ML places it.
"""

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

FORMAT = "ane-sweep-v1"


def load_builder():
    source = Path(__file__).with_name("gen_ane_prefill_programs.py")
    spec = importlib.util.spec_from_file_location("ane_prefill_builder", source)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    return builder


def parse_list(text, conv=str):
    return [conv(v.strip()) for v in text.split(",") if v.strip()]


def experts_program(builder, e, g, k, rows):
    ct, mb, types = builder.ct, builder.mb, builder.types

    # The planes the bench binds are [1, 1, rows, cols] F16 surfaces, so the
    # experts are stacked along the rows and reshaped inside the program:
    # a is [1, 1, E*g, K], w is [1, 1, E*R, K], and the output comes back
    # as [1, 1, E*R, g] (expert e's rows at [e*R, (e+1)*R)).
    @mb.program(
        input_specs=[
            mb.TensorSpec((1, 1, e * g, k), types.fp16),
            mb.TensorSpec((1, 1, e * rows, k), types.fp16),
        ],
        opset_version=ct.target.macOS15,
    )
    def prog(a, w):
        a4 = mb.reshape(x=a, shape=[e, 1, g, k])
        w4 = mb.reshape(x=w, shape=[e, 1, rows, k])
        acc = None
        for first in range(0, k, builder.CHUNK):
            count = min(builder.CHUNK, k - first)
            sa = mb.slice_by_size(x=a4, begin=[0, 0, 0, first], size=[e, 1, g, count])
            sw = mb.slice_by_size(x=w4, begin=[0, 0, 0, first], size=[e, 1, rows, count])
            p = mb.matmul(x=sw, y=sa, transpose_x=False, transpose_y=True)
            acc = p if acc is None else mb.add(x=acc, y=p)
        return mb.identity(x=mb.reshape(x=acc, shape=[1, 1, e * rows, g]), name="o0")

    return ct.convert(
        prog,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=ct.precision.FLOAT16,
    )


def const_int4_program(builder, k, n, rows, block):
    """`w @ a^T` with the weights baked in as uint4 block-scaled constants.

    The F16 form takes the weights as an input plane the GPU fills with a
    dequant pass each micro-batch. This form carries them inside the program
    (`constexpr_blockwise_shift_scale`), so the bench skips that pass and the
    plane. The weights are random: the ANE's cost does not depend on their
    values, and this answers where Core ML places it and what a prediction
    costs, not a quality question. Coremltools writes the offset as a uint4
    zero point; this engine's affine layout has an fp16 bias instead
    (offset = -bias / scale), which the op also accepts, so a real-weights
    version would build the constants by hand. K is sliced at CHUNK as in
    the F16 form, one constant per slice.
    """
    import numpy as np
    from coremltools.optimize.coreml import (
        OpLinearQuantizerConfig,
        OptimizationConfig,
        linear_quantize_weights,
    )

    ct, mb, types = builder.ct, builder.mb, builder.types
    rng = np.random.default_rng(0)

    @mb.program(
        input_specs=[mb.TensorSpec((1, 1, n, k), types.fp16)],
        opset_version=ct.target.macOS15,
    )
    def prog(a):
        acc = None
        for first in range(0, k, builder.CHUNK):
            count = min(builder.CHUNK, k - first)
            sa = mb.slice_by_size(x=a, begin=[0, 0, 0, first], size=[1, 1, n, count])
            w = mb.const(
                val=(rng.standard_normal((1, 1, rows, count)) * 0.02).astype(np.float16)
            )
            p = mb.matmul(x=w, y=sa, transpose_x=False, transpose_y=True)
            acc = p if acc is None else mb.add(x=acc, y=p)
        return mb.identity(x=acc, name="o0")

    model = ct.convert(
        prog,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=ct.precision.FLOAT16,
    )
    config = OptimizationConfig(
        global_config=OpLinearQuantizerConfig(
            mode="linear",
            dtype="uint4",
            granularity="per_block",
            block_size=block,
            weight_threshold=0,
        )
    )
    return linear_quantize_weights(model, config)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out_dir")
    ap.add_argument("--shapes", default="", help="ROWSxCOLS pairs (output rows x K)")
    ap.add_argument("--tokens", default="32,128,512,1024,2048")
    ap.add_argument("--shares", default="1.0,0.3,0.5,0.7")
    ap.add_argument("--experts", default="", help="E:g pairs for packed-expert programs")
    ap.add_argument("--expert-dims", default="", help="hidden,intermediate for --experts")
    ap.add_argument(
        "--const-int4",
        action="store_true",
        help="also write, per shape, token count and share, a program with uint4 constant weights",
    )
    ap.add_argument("--const-block", type=int, default=64, help="block size of the uint4 scales")
    args = ap.parse_args()

    shapes = []
    for s in parse_list(args.shapes):
        rows, cols = (int(v) for v in s.lower().split("x"))
        if rows <= 0 or cols <= 0 or cols % 32 or rows % 32:
            raise SystemExit(f"shape {s}: rows and K must be positive multiples of 32")
        shapes.append((rows, cols))
    tokens = sorted(set(parse_list(args.tokens, int)))
    if not tokens or tokens[0] < 1 or tokens[-1] > 4096:
        raise SystemExit("token counts must be in 1..=4096")
    shares = sorted(set(parse_list(args.shares, float)), reverse=True)
    if any(not 0 < s <= 1 for s in shares):
        raise SystemExit("shares must be in (0, 1]")
    experts = [tuple(int(v) for v in e.split(":")) for e in parse_list(args.experts)]
    dims = parse_list(args.expert_dims, int)
    if experts and len(dims) != 2:
        raise SystemExit("--experts needs --expert-dims hidden,intermediate")
    if not shapes and not experts:
        raise SystemExit("nothing to write: give --shapes and/or --experts")

    builder = load_builder()
    ct = builder.ct
    os.makedirs(args.out_dir, exist_ok=True)
    programs = []

    def save(model, name):
        path = os.path.join(args.out_dir, name)
        if os.path.exists(path):
            print("exists", path)
        else:
            model.save(path)
            print("wrote", path)

    for rows, k in shapes:
        for n in tokens:
            for share in shares:
                ma = max(32, int(rows * share) // 32 * 32)
                name = f"ane_k{k}_rows{rows}_ma{ma}_n{n}.mlpackage"
                save(builder.build(k, n, [(ma, rows)]), name)
                programs.append({
                    "kind": "dense", "k": k, "rows": rows, "ane_rows": ma,
                    "plane_rows": rows, "tokens": n, "share": share, "package": name,
                })
    if args.const_int4:
        # One program per share too: the ANE bakes rows [0, MA) and Steel
        # takes the rest, like the F16 split. The whole-matrix name has no
        # `ma` so programs written before shares existed stay valid.
        for rows, k in shapes:
            for n in tokens:
                for share in shares:
                    ma = max(32, int(rows * share) // 32 * 32)
                    tag = "" if ma == rows else f"_ma{ma}"
                    name = f"ane_c4_k{k}_rows{rows}{tag}_n{n}.mlpackage"
                    save(const_int4_program(builder, k, n, ma, args.const_block), name)
                    programs.append({
                        "kind": "dense", "weights": "const_int4", "k": k, "rows": rows,
                        "ane_rows": ma, "plane_rows": rows, "tokens": n, "share": share,
                        "package": name,
                    })
    if experts:
        hidden, inter = dims
        for e, g in experts:
            if e <= 0 or g <= 0 or g % 32:
                raise SystemExit(f"--experts {e}:{g}: g must be a positive multiple of 32")
            for role, k, rows in [("gateup", hidden, inter), ("down", inter, hidden)]:
                name = f"experts_{role}_e{e}_g{g}_k{k}_rows{rows}.mlpackage"
                save(experts_program(builder, e, g, k, rows), name)
                programs.append({
                    "kind": "experts", "role": role, "experts": e, "group": g,
                    "k": k, "rows": rows, "package": name,
                })

    manifest = {
        "format": FORMAT,
        "generator": "crates/coreml/tools/gen_ane_sweep_programs.py",
        "coremltools": ct.__version__,
        "chunk": builder.CHUNK,
        "programs": programs,
    }
    with open(os.path.join(args.out_dir, "sweep.json"), "w") as f:
        json.dump(manifest, f, indent=1)
        f.write("\n")
    print(f"wrote {len(programs)} programs and sweep.json to {args.out_dir}")


if __name__ == "__main__":
    sys.exit(main())
