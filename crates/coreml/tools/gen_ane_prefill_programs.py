"""Write the Core ML programs for the opt-in ANE+Steel prefill split.

The split (`turbospark-runtime` feature `ane-prefill`,
`crates/runtime/src/families/qwen/ane_split.rs`) runs rows [0, MA) of each
dense qwen projection class on the Neural Engine while Steel computes the
rest. Each program computes, for one class and one token count N,

    o_i[1, 1, MA_i, N] = w_i[0:MA_i] @ a^T        (one or two matrices)

with every operand an F16 input, so one program serves every layer of its
shape. K is sliced at 2048 and the partial products are summed in F16. Each
weight input has WR_i rows, the largest MA_i of that matrix over the token
counts written, and the program reads only its first MA_i rows: one weight
plane per matrix then serves every token count.

The output directory gets one .mlpackage per class and token count plus a
`manifest.json` the runtime reads. Point `TURBOSPARK_ANE_PREFILL_DIR` at it.
The runtime refuses a manifest whose shapes do not match the install.

Run with a pinned, isolated interpreter. Building MIL programs is pure
Python, so no model download or device is involved:

    uv run --python 3.12 --with coremltools==9.0 --with numpy==2.2.6 \
        python -I crates/coreml/tools/gen_ane_prefill_programs.py <out_dir>

Options:

    --classes gateup,down,qg,qkvz,o   which classes split (default: all)
    --tokens 1024,2048                which token counts (default: both)
    --rows CLASS:N:MA[,MA]            override a preset share, e.g.
                                      --rows qkvz:2048:7168,4288

The preset shares are the ones measured on Qwen3.8-27B on an M4 Max
(docs/MOE_PRIOR_ART.md): each leaves the prediction at about 0.96 of the
Steel rows buffer it overlaps. Another chip has a different GPU-to-ANE
ratio, so its best shares differ. `TURBOSPARK_ANE_PREFILL_STATS=1` prints
per-class prediction and Steel times to re-balance from.
"""

import argparse
import json
import os
import sys

import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types

CHUNK = 2048
FORMAT = 1
MAX_TOKENS = 2048

# Qwen3.8-27B: hidden 5120, FFN 17408, attention q (with its gate) 2 x 24 x
# 256, GDN qkv 10240 and value 6144. Per class: K, each matrix's rows, and
# the measured ANE rows per token count.
PRESET = {
    "name": "qwen38-27b",
    "classes": {
        "gateup": {"k": 5120, "rows": [17408, 17408],
                   "ane_rows": {2048: [8192, 8192], 1024: [7168, 7168]}},
        "down": {"k": 17408, "rows": [5120],
                 "ane_rows": {2048: [2272], 1024: [1920]}},
        "qg": {"k": 5120, "rows": [12288],
               "ane_rows": {2048: [6592], 1024: [5696]}},
        "qkvz": {"k": 5120, "rows": [10240, 6144],
                 "ane_rows": {2048: [7168, 4288], 1024: [6208, 3712]}},
        "o": {"k": 6144, "rows": [5120],
              "ane_rows": {2048: [2272], 1024: [1856]}},
    },
}


def matmul(a, w, n, m, k):
    """w[0:m] @ a^T in F16, K in CHUNK slices. w may hold more than m rows."""
    acc = None
    for first in range(0, k, CHUNK):
        count = min(CHUNK, k - first)
        sa = mb.slice_by_size(x=a, begin=[0, 0, 0, first], size=[1, 1, n, count])
        sw = mb.slice_by_size(x=w, begin=[0, 0, 0, first], size=[1, 1, m, count])
        p = mb.matmul(x=sw, y=sa, transpose_x=False, transpose_y=True)
        acc = p if acc is None else mb.add(x=acc, y=p)
    return acc


def build(k, n, mats):
    """mats: [(MA, WR)], one or two matrices sharing the activation input."""
    specs = [mb.TensorSpec((1, 1, n, k), types.fp16)]
    specs += [mb.TensorSpec((1, 1, wr, k), types.fp16) for _, wr in mats]
    if len(mats) == 1:
        ((m0, _),) = mats

        @mb.program(input_specs=specs, opset_version=ct.target.macOS15)
        def prog(a, w0):
            return mb.identity(x=matmul(a, w0, n, m0, k), name="o0")

    elif len(mats) == 2:
        (m0, _), (m1, _) = mats

        @mb.program(input_specs=specs, opset_version=ct.target.macOS15)
        def prog(a, w0, w1):
            o0 = mb.identity(x=matmul(a, w0, n, m0, k), name="o0")
            o1 = mb.identity(x=matmul(a, w1, n, m1, k), name="o1")
            return o0, o1

    else:
        raise SystemExit("one or two matrices per program")
    return ct.convert(
        prog,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=ct.precision.FLOAT16,
    )


def parse_list(text, conv=str):
    return [conv(v.strip()) for v in text.split(",") if v.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out_dir")
    ap.add_argument("--classes", default="all")
    ap.add_argument("--tokens", default="1024,2048")
    ap.add_argument("--rows", action="append", default=[])
    args = ap.parse_args()

    preset = PRESET["classes"]
    classes = list(preset) if args.classes == "all" else parse_list(args.classes)
    for c in classes:
        if c not in preset:
            raise SystemExit(f"unknown class {c!r}; known: {', '.join(preset)}")
    tokens = sorted(set(parse_list(args.tokens, int)))
    if not tokens or tokens[0] < 1 or tokens[-1] > MAX_TOKENS:
        raise SystemExit(f"token counts must be in 1..={MAX_TOKENS}")

    shares = {c: dict(preset[c]["ane_rows"]) for c in classes}
    for spec in args.rows:
        c, n, rest = spec.split(":")
        if c not in shares:
            raise SystemExit(f"--rows {spec}: class {c!r} is not being written")
        shares[c][int(n)] = parse_list(rest, int)

    os.makedirs(args.out_dir, exist_ok=True)
    programs = []
    for c in classes:
        k, rows = preset[c]["k"], preset[c]["rows"]
        for n in tokens:
            if n not in shares[c]:
                raise SystemExit(f"no share for {c} at {n} tokens; pass --rows {c}:{n}:...")
            mas = shares[c][n]
            if len(mas) != len(rows):
                raise SystemExit(f"{c} has {len(rows)} matrices, got {len(mas)} rows at {n}")
            for ma, r in zip(mas, rows):
                if not (0 < ma < r and ma % 32 == 0):
                    raise SystemExit(f"{c} at {n}: {ma} ANE rows must be a multiple of 32 in (0, {r})")
        # One weight plane per matrix: the largest share over the counts.
        wrs = [max(shares[c][n][i] for n in tokens) for i in range(len(rows))]
        for n in tokens:
            mats = list(zip(shares[c][n], wrs))
            name = f"split_k{k}_n{n}" + "".join(f"_m{ma}w{wr}" for ma, wr in mats) + ".mlpackage"
            path = os.path.join(args.out_dir, name)
            if os.path.exists(path):
                print("exists", path)
            else:
                build(k, n, mats).save(path)
                print("wrote", path)
            programs.append({
                "class": c,
                "k": k,
                "tokens": n,
                "matrices": [
                    {"rows": r, "ane_rows": ma, "plane_rows": wr}
                    for r, (ma, wr) in zip(rows, mats)
                ],
                "package": name,
            })

    manifest = {
        "format": FORMAT,
        "preset": PRESET["name"],
        "generator": "crates/coreml/tools/gen_ane_prefill_programs.py",
        "coremltools": ct.__version__,
        "programs": programs,
    }
    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
        f.write("\n")
    print(f"wrote {len(programs)} programs and manifest.json to {args.out_dir}")


if __name__ == "__main__":
    sys.exit(main())
