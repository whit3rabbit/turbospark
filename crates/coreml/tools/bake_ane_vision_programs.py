"""Bake one vision tower block's four projections into Core ML programs.

The sibling of `bake_ane_prefill_programs.py` for the vision tower: the
programs `tslab kernel --bench ane_matmul` times against the tower's own GPU
matmul (`gpu::encode_vision_matmul`, FP16) to answer whether the Neural
Engine runs a tower block faster once the conversions and the host handoff
are paid. One program per projection and patch-row bucket:

    qkv   rows 3*hidden  K hidden
    proj  rows hidden    K hidden
    fc1   rows inter     K hidden
    fc2   rows hidden    K inter

Each computes `w @ a^T` with `w` an F16 constant inside the program, K
sliced at the prefill builder's CHUNK. Constants, not an F16 input plane: the
tower's weights are FP16 already, so a Neural Engine tower would bake them
and pay no weight plane per block. The values are random (seeded): a dense
F16 matmul's ANE cost does not depend on them, and this answers time and
placement, not quality. The block's biases are not in the programs: with
the activation scaled per row for the handoff (`encode_ane_scale_rows`), a
bias would have to be scaled too, so it stays a GPU add, unpriced here.

Shapes come from the install, never typed: `--config` takes a `.gturbo`
install (its `manifest.json` vision block), a checkpoint `config.json`
(its `vision_config`), or a directory holding either.

    uv run --python 3.12 --with coremltools==9.0 --with numpy==2.2.6 \\
        python -I crates/coreml/tools/bake_ane_vision_programs.py OUT_DIR \\
        --config ~/.turbospark/models/text/qwen3vl-4b-vision \\
        [--tokens 1280,2048,4096]

The default buckets are the 512x640 page's 1280 patches (patch 16) and the
prefill generator's 2048 and 4096 ceiling; a larger page runs as several
predictions of the largest bucket. The directory's `sweep.json`
(`ane-sweep-v1`) gets entries with `"kind": "vision"`, `"weights":
"const_f16"` and `"role": "qkv" | "proj" | "fc1" | "fc2"`; rerunning adds to
it. Both dimensions of every matrix must be multiples of 32 halfs, the plane
stride the bench allocates: the qwen3_5 tower's 4304-wide MLP is refused.
"""

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

FORMAT = "ane-sweep-v1"
MAX_TOKENS = 4096


def load_builder():
    source = Path(__file__).with_name("gen_ane_prefill_programs.py")
    spec = importlib.util.spec_from_file_location("ane_prefill_builder", source)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    return builder


def tower_dims(path):
    """(hidden, intermediate, depth) of the vision tower `path` describes."""
    p = Path(path).expanduser()
    if p.is_dir():
        for name in ("manifest.json", "config.json"):
            if (p / name).is_file():
                p = p / name
                break
        else:
            raise SystemExit(f"{path}: no manifest.json or config.json")
    doc = json.loads(p.read_text())
    # A checkpoint's config.json nests `vision_config`; a .gturbo manifest
    # flattens it into `arch` as visionDepth, visionHiddenSize, ...
    v = doc.get("vision_config")
    if isinstance(v, dict) and v.get("depth"):
        return int(v["hidden_size"]), int(v["intermediate_size"]), int(v["depth"])
    arch = doc.get("arch") or {}
    if arch.get("visionDepth"):
        return (
            int(arch["visionHiddenSize"]),
            int(arch["visionIntermediateSize"]),
            int(arch["visionDepth"]),
        )
    raise SystemExit(f"{p}: no active vision tower (vision_config or arch.visionDepth)")


def const_f16_program(builder, k, n, rows, seed):
    ct, mb, types = builder.ct, builder.mb, builder.types
    rng = np.random.default_rng(seed)

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

    return ct.convert(
        prog,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=ct.precision.FLOAT16,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out_dir")
    ap.add_argument("--config", required=True, help="install dir, manifest.json or config.json")
    ap.add_argument("--tokens", default="1280,2048,4096", help="patch-row buckets")
    ap.add_argument("--roles", default="qkv,proj,fc1,fc2")
    args = ap.parse_args()

    hidden, inter, depth = tower_dims(args.config)
    shapes = {
        "qkv": (3 * hidden, hidden),
        "proj": (hidden, hidden),
        "fc1": (inter, hidden),
        "fc2": (hidden, inter),
    }
    roles = [r.strip() for r in args.roles.split(",") if r.strip()]
    for r in roles:
        if r not in shapes:
            raise SystemExit(f"--roles {r!r}: one of {', '.join(shapes)}")
        rows, k = shapes[r]
        if rows % 32 or k % 32:
            raise SystemExit(f"{r} is {rows}x{k}: both must be multiples of 32 halfs")
    tokens = sorted({int(t) for t in args.tokens.split(",") if t.strip()})
    if not tokens or tokens[0] < 1 or tokens[-1] > MAX_TOKENS:
        raise SystemExit(f"patch-row buckets must be in 1..={MAX_TOKENS}")

    builder = load_builder()
    os.makedirs(args.out_dir, exist_ok=True)
    manifest_path = os.path.join(args.out_dir, "sweep.json")
    programs = []
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            old = json.load(f)
        if old.get("format") != FORMAT:
            raise SystemExit(f"{manifest_path}: not an {FORMAT} manifest")
        programs = old.get("programs", [])

    for i, r in enumerate(roles):
        rows, k = shapes[r]
        for n in tokens:
            name = f"vision_{r}_k{k}_rows{rows}_n{n}.mlpackage"
            path = os.path.join(args.out_dir, name)
            if os.path.exists(path):
                print("exists", path)
            else:
                const_f16_program(builder, k, n, rows, 0x5EED + i).save(path)
                print("wrote", path)
            programs = [p for p in programs if p.get("package") != name]
            programs.append({
                "kind": "vision", "weights": "const_f16", "role": r, "k": k,
                "rows": rows, "ane_rows": rows, "tokens": n, "share": 1.0,
                "package": name, "tower": {"hidden": hidden, "intermediate": inter,
                                           "depth": depth},
            })

    manifest = {
        "format": FORMAT,
        "generator": "crates/coreml/tools/bake_ane_vision_programs.py",
        "coremltools": builder.ct.__version__,
        "chunk": builder.CHUNK,
        "programs": programs,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=1)
        f.write("\n")
    print(f"{len(roles) * len(tokens)} vision programs; sweep.json lists {len(programs)}")


if __name__ == "__main__":
    sys.exit(main())
