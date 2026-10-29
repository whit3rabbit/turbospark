"""Bake an install's real INT4 weights into Core ML programs for the ANE.

`gen_ane_prefill_programs.py` writes programs that take their weights as F16
input planes, so one program serves every layer and the GPU dequantizes the
weights into a plane before each prediction. This tool writes the other
form: rows [0, MA) of one resident tensor become uint4 constants inside the
program (`constexpr_blockwise_shift_scale`), so there is no weight plane and
no dequant pass, and the ANE reads about a quarter of the bytes.

It reads `model_weights.bin` straight from a `.gturbo` install and uses the
resident INT4 affine layout: row-major codes with the low nibble first, then
BF16 scales, then BF16 biases, one per 64 columns, `w = q * scale + bias`.
The program computes `scale16 * (q - offset16)` with `offset = -bias / scale`
in fp16. A tensor is refused unless that reproduces the BF16 values: the
worst absolute error must stay under 1e-5 (the fp16 rounding of a typical
weight), or the worst error must stay under 0.05 of the group's own scale.
Scales below 6.1e-5 are subnormal in fp16, so a group with a tiny scale can
fail the second test and still pass the first.

Run with a pinned, isolated interpreter, as the sibling generators do:

    uv run --python 3.12 --with coremltools==9.0 --with numpy==2.2.6 \\
        python -I crates/coreml/tools/bake_ane_prefill_programs.py OUT_DIR \\
        --install ~/.turbospark/models/text/qwen38-27b.gturbo \\
        --tensors language_model.model.layers.0.mlp.gate_proj.weight \\
        --tokens 2048 [--share 0.5] [--with-f16]

One program per tensor, tokens and share. `--with-f16` also writes the
F16-plane program of the same shape, the form the runtime uses today, so a
bench can run both on the same real weights. The directory gets a
`sweep.json` (`ane-sweep-v1`, the manifest `tslab kernel --bench ane_matmul`
reads); rerunning with other tensors adds to it. Entries carry
`"weights": "const_int4"` and `"tensor"`.

Class mode writes what the runtime split reads (manifest format 2, see
`crates/runtime/src/families/qwen/ane_split.rs`): one program per projection
class, layer and token count, with the class's matrices in one program
(qkvz: in_proj_qkv and in_proj_z; gateup: gate and up) and the layer's own
INT4 weights inside it.

    ... bake_ane_prefill_programs.py OUT_DIR --install INSTALL \\
        --layers 0,3 [--classes all] --tokens 1024,2048 [--share 0.5 | --rows CLASS:N:MA[,MA]]

`--lut` writes the weights as 4-bit indices into a table of 16 fp16 values per
64 weights (`constexpr_lut_to_dense`) instead of a shift and a scale. The
values are the same grid; the Neural Engine holds about half the memory for it
(160 against 299 MiB per gateup program), which is what limits how many layers
can be baked on a 36 GB Mac. `--shared-f16 DIR` adds the F16-plane programs of
a format 1 directory to the manifest (weights `f16`), so layers that were not
baked still split. Name the baked layers and the rest fall back to them.

ANE rows default to the shares `gen_ane_prefill_programs.py` ships for the
F16 programs (the same split, so the two forms compare like for like).
`--share S` replaces them with S of each matrix's rows, rounded down to 32,
and `--rows` sets one class and token count by hand. The directory gets a
`manifest.json`; point `TURBOSPARK_ANE_PREFILL_DIR` at it. A layer or class
with no program runs on Steel.
"""

import argparse
import importlib.util
import json
import os
import re
import shutil
import struct
import sys
from pathlib import Path

import numpy as np
from coremltools.converters.mil.mil.types.type_int import np_uint4_dtype

FORMAT = "ane-sweep-v1"
GROUP = 64
# A scale below this is subnormal in fp16, with less than 10 bits of precision.
FP16_NORMAL_MIN = 6.103515625e-05
ABS_ERR_LIMIT = 1e-5
LEVEL_ERR_LIMIT = 0.05


def load_builder():
    source = Path(__file__).with_name("gen_ane_prefill_programs.py")
    spec = importlib.util.spec_from_file_location("ane_prefill_builder", source)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    return builder


def bf16(raw):
    """BF16 bit patterns (uint8 view) to float32."""
    return (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)


class Install:
    """The resident index of `model_weights.bin`, as `model-io`'s
    `resident_index::load` reads it: a 24-byte header, 72-byte entries, then
    names. Only INT4 affine entries (dtype 4) are used here."""

    def __init__(self, path):
        self.path = Path(path).expanduser() / "model_weights.bin"
        self.mm = np.memmap(self.path, dtype=np.uint8, mode="r")
        index_size, resident_size, count = struct.unpack("<QQQ", bytes(self.mm[:24]))
        if index_size + resident_size > len(self.mm):
            raise SystemExit(f"{self.path}: index region runs past the file")
        self.entries = {}
        for i in range(count):
            p = bytes(self.mm[24 + i * 72 : 24 + (i + 1) * 72])
            name_off, name_len, dtype = struct.unpack("<IHB", p[:7])
            offset, size = struct.unpack("<QQ", p[8:24])
            rows, cols, _, _ = struct.unpack("<IIII", p[24:40])
            s_off, s_size, b_off, b_size = struct.unpack("<QQQQ", p[40:72])
            name = bytes(self.mm[name_off : name_off + name_len]).decode()
            self.entries[name] = dict(
                dtype=dtype, offset=offset, size=size, rows=rows, cols=cols,
                s_off=s_off, s_size=s_size, b_off=b_off, b_size=b_size,
            )

    def int4(self, name):
        e = self.entries.get(name)
        if e is None:
            raise SystemExit(f"{name}: not in the resident index")
        if e["dtype"] != 4:
            raise SystemExit(f"{name}: dtype {e['dtype']} is not INT4 affine (4)")
        rows, cols = e["rows"], e["cols"]
        if cols % GROUP or e["size"] != rows * cols // 2:
            raise SystemExit(f"{name}: unexpected INT4 layout for {rows}x{cols}")
        return e

    def rows(self, name, ma):
        """Codes, scales and biases of rows [0, ma): uint8 (ma, cols) holding
        one code per element, float32 (ma, cols/64) scales and biases."""
        e = self.int4(name)
        rows, cols = e["rows"], e["cols"]
        if not 0 < ma <= rows:
            raise SystemExit(f"{name}: {ma} ANE rows of {rows}")
        packed = np.asarray(self.mm[e["offset"] : e["offset"] + ma * cols // 2])
        packed = packed.reshape(ma, cols // 2)
        q = np.empty((ma, cols), np.uint8)
        q[:, 0::2] = packed & 0x0F
        q[:, 1::2] = packed >> 4
        g = cols // GROUP
        s = bf16(np.asarray(self.mm[e["s_off"] : e["s_off"] + ma * g * 2])).reshape(ma, g)
        b = bf16(np.asarray(self.mm[e["b_off"] : e["b_off"] + ma * g * 2])).reshape(ma, g)
        return q, s, b


def fp16_mapping(name, s, b):
    """scale16 and offset16 for `w = q * s + b`, refusing a tensor the fp16
    form cannot reproduce. Returns them plus the worst errors seen."""
    s64, b64 = s.astype(np.float64), b.astype(np.float64)
    if (s64 == 0).any():
        raise SystemExit(f"{name}: a group has a zero scale, which has no offset form")
    s16 = s64.astype(np.float16)
    o16 = (-b64 / s64).astype(np.float16)
    s16d, o16d = s16.astype(np.float64), o16.astype(np.float64)
    worst_abs = worst_level = 0.0
    failing = 0
    for q in (0.0, 15.0):  # the error is linear in q, so the ends bound it
        err = np.abs(s16d * (q - o16d) - (q * s64 + b64))
        level = err / np.abs(s64)
        worst_abs = max(worst_abs, float(err.max()))
        worst_level = max(worst_level, float(level.max()))
        # A group fails only when it is over both limits.
        failing += int(((err > ABS_ERR_LIMIT) & (level > LEVEL_ERR_LIMIT)).sum())
    if failing:
        raise SystemExit(
            f"{name}: {failing} groups are not reproduced by the fp16 scale and "
            f"offset (worst abs {worst_abs:.2e}, worst {worst_level:.3f} level)"
        )
    return s16, o16, worst_abs, worst_level


def lut_table(name, s, b, s16, o16):
    """The 16 dequantized values of every group as fp16, `scale16 * (j -
    offset16)` computed in fp32, and the worst error against the exact
    `j * s + b` in units of the group's scale."""
    j = np.arange(16, dtype=np.float32)
    lut = (s16.astype(np.float32)[..., None] * (j - o16.astype(np.float32)[..., None])).astype(
        np.float16)
    exact = j * s.astype(np.float64)[..., None] + b.astype(np.float64)[..., None]
    err = np.abs(lut.astype(np.float64) - exact)
    return lut, float((err / np.abs(s.astype(np.float64))[..., None]).max())


def baked_program(builder, install, mats, n, lut=False):
    """One program computing `w[0:MA] @ a^T` for each `(tensor, MA)` in `mats`
    (outputs `o0`, `o1`, ...), all reading one activation input and all
    sharing K. Returns the model and the worst errors over the matrices.
    `lut` writes the weights as 4-bit indices into a per-group table of 16
    fp16 values (`constexpr_lut_to_dense`) instead of a shift and a scale:
    the Neural Engine holds about half the memory for it."""
    ct, mb, types = builder.ct, builder.mb, builder.types
    ks = {install.int4(name)["cols"] for name, _ in mats}
    if len(ks) != 1:
        raise SystemExit(f"matrices of one program must share K, got {sorted(ks)}")
    k = ks.pop()
    consts, worst_abs, worst_level = [], 0.0, 0.0
    for name, ma in mats:
        q, s, b = install.rows(name, ma)
        s16, o16, w_abs, w_level = fp16_mapping(name, s, b)
        if lut:
            table, t_level = lut_table(name, s, b, s16, o16)
            if t_level > LEVEL_ERR_LIMIT:
                raise SystemExit(f"{name}: the fp16 table is {t_level:.3f} level off")
            w_level = max(w_level, t_level)
            consts.append((ma, q, table, None))
        else:
            consts.append((ma, q, s16, o16))
        worst_abs, worst_level = max(worst_abs, w_abs), max(worst_level, w_level)

    def matmul_chain(a, ma, q, s16, o16):
        acc = None
        for first in range(0, k, builder.CHUNK):
            count = min(builder.CHUNK, k - first)
            g0, g1 = first // GROUP, (first + count) // GROUP
            # astype, not np.array(dtype=...): np.array drops the sub-byte tag
            # and quietly builds a uint8 constant, twice the bytes (and twice
            # the resident ANE footprint) of the uint4 this tool means to write.
            data = q[:, first : first + count].reshape(1, 1, ma, count).astype(np_uint4_dtype)
            if lut:
                # s16 holds the table here: (ma, groups, 16).
                w = mb.constexpr_lut_to_dense(
                    indices=data, lut=s16[:, g0:g1, :].reshape(1, 1, ma, g1 - g0, 16, 1))
            else:
                w = mb.constexpr_blockwise_shift_scale(
                    data=data,
                    scale=s16[:, g0:g1].reshape(1, 1, ma, g1 - g0),
                    offset=o16[:, g0:g1].reshape(1, 1, ma, g1 - g0),
                )
            sa = mb.slice_by_size(x=a, begin=[0, 0, 0, first], size=[1, 1, n, count])
            p = mb.matmul(x=w, y=sa, transpose_x=False, transpose_y=True)
            acc = p if acc is None else mb.add(x=acc, y=p)
        return acc

    specs = [mb.TensorSpec((1, 1, n, k), types.fp16)]
    if len(consts) == 1:
        @mb.program(input_specs=specs, opset_version=ct.target.macOS15)
        def prog(a):
            return mb.identity(x=matmul_chain(a, *consts[0]), name="o0")
    elif len(consts) == 2:
        @mb.program(input_specs=specs, opset_version=ct.target.macOS15)
        def prog(a):
            o0 = mb.identity(x=matmul_chain(a, *consts[0]), name="o0")
            o1 = mb.identity(x=matmul_chain(a, *consts[1]), name="o1")
            return o0, o1
    else:
        raise SystemExit("one or two matrices per program")
    # skip_model_load: the default compiles and loads every converted model,
    # which costs seconds per program, leaves temporary compiled copies
    # behind if the process is killed, and fills the Core ML and ANE caches
    # before the runtime has asked for the program. The bake only saves.
    model = ct.convert(
        prog,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=ct.precision.FLOAT16,
        skip_model_load=True,
    )
    return model, k, worst_abs, worst_level


def short(name):
    m = re.search(r"layers\.(\d+)\.(.+)\.weight$", name)
    return f"L{m.group(1)}_{m.group(2).replace('.', '_')}" if m else re.sub(r"\W", "_", name)


LAYER = "language_model.model.layers.{}."
# Class -> (tensor suffixes by layer kind). A class absent from a kind has no
# split point in that kind of layer.
CLASSES = {
    "qg": {"attn": ["self_attn.q_proj.weight"]},
    "qkvz": {"gdn": ["linear_attn.in_proj_qkv.weight", "linear_attn.in_proj_z.weight"]},
    "o": {"attn": ["self_attn.o_proj.weight"], "gdn": ["linear_attn.out_proj.weight"]},
    "gateup": {"attn": ["mlp.gate_proj.weight", "mlp.up_proj.weight"],
               "gdn": ["mlp.gate_proj.weight", "mlp.up_proj.weight"]},
    "down": {"attn": ["mlp.down_proj.weight"], "gdn": ["mlp.down_proj.weight"]},
}


def layer_kind(install, layer):
    prefix = LAYER.format(layer)
    if prefix + "linear_attn.in_proj_qkv.weight" in install.entries:
        return "gdn"
    if prefix + "self_attn.q_proj.weight" in install.entries:
        return "attn"
    raise SystemExit(f"layer {layer}: neither a GDN nor an attention layer in this install")


def class_rows(preset, cls, tokens, rows, share, overrides):
    """ANE rows per matrix of `cls` at `tokens`: an explicit `--rows`, else
    `--share`, else the F16 programs' shares from the sibling generator."""
    if (cls, tokens) in overrides:
        mas = overrides[(cls, tokens)]
    elif share is not None:
        mas = [max(32, int(r * share) // 32 * 32) for r in rows]
    else:
        mas = preset[cls]["ane_rows"].get(tokens)
        if mas is None:
            raise SystemExit(
                f"class {cls} has no preset ANE rows at {tokens} tokens; give --share or --rows"
            )
    if len(mas) != len(rows) or any(not 0 < m < r or m % 32 for m, r in zip(mas, rows)):
        raise SystemExit(f"class {cls} at {tokens}: ANE rows {mas} do not fit matrices of {rows} rows")
    return list(mas)


def add_shared_f16(src, out_dir, programs):
    """Adds the F16-plane programs of a format 1 manifest to `programs`, with
    their packages copied beside the baked ones. They serve every layer that
    has no baked program, so baking a subset of layers leaves the rest on the
    split the runtime already ships."""
    manifest = json.loads((src / "manifest.json").read_text())
    if manifest.get("format") != 1:
        raise SystemExit(f"{src}/manifest.json is not a format 1 manifest")
    for p in manifest["programs"]:
        dest = out_dir / p["package"]
        if not dest.exists():
            shutil.copytree(src / p["package"], dest)
        programs[p["package"]] = {**p, "weights": "f16"}


def bake_classes(args, install, builder, tokens):
    layers = [int(v) for v in args.layers.split(",") if v.strip()]
    wanted = list(CLASSES) if args.classes == "all" else [c.strip() for c in args.classes.split(",")]
    for c in wanted:
        if c not in CLASSES:
            raise SystemExit(f"unknown class {c!r}; known: {', '.join(CLASSES)}")
    overrides = {}
    for spec in args.rows:
        cls, n, rest = spec.split(":")
        overrides[(cls, int(n))] = [int(v) for v in rest.split(",")]
    preset = builder.PRESET["classes"]
    manifest_path = Path(args.out_dir) / "manifest.json"
    manifest = {"format": 2, "programs": []}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("format") != 2:
            raise SystemExit(f"{manifest_path} is not a format 2 manifest; use another directory")
    programs = {p["package"]: p for p in manifest["programs"]}
    for layer in layers:
        kind = layer_kind(install, layer)
        for cls in wanted:
            suffixes = CLASSES[cls].get(kind)
            if suffixes is None:
                continue
            names = [LAYER.format(layer) + s for s in suffixes]
            ents = [install.int4(nm) for nm in names]
            rows, k = [e["rows"] for e in ents], ents[0]["cols"]
            for n in tokens:
                mas = class_rows(preset, cls, n, rows, args.share, overrides)
                package = f"{'lut' if args.lut else 'bake'}_{cls}_l{layer}_n{n}.mlpackage"
                path = Path(args.out_dir) / package
                if path.exists():
                    print("exists", path)
                else:
                    model, _, w_abs, w_level = baked_program(
                        builder, install, list(zip(names, mas)), n, lut=args.lut)
                    model.save(str(path))
                    print(f"wrote {path}  ANE rows {mas}  (worst abs err {w_abs:.2e}, {w_level:.3f} level)")
                programs[package] = {
                    "class": cls, "layer": layer, "k": k, "tokens": n,
                    "weights": "const_int4", "package": package,
                    "matrices": [{"rows": r, "ane_rows": m} for r, m in zip(rows, mas)],
                }
    if args.shared_f16:
        add_shared_f16(Path(args.shared_f16), Path(args.out_dir), programs)
    manifest["format"] = 2
    manifest["generator"] = "crates/coreml/tools/bake_ane_prefill_programs.py"
    manifest["install"] = str(Path(args.install).expanduser())
    manifest["coremltools"] = builder.ct.__version__
    manifest["programs"] = list(programs.values())
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"{len(programs)} programs in {manifest_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out_dir")
    ap.add_argument("--install", required=True, help="a .gturbo install directory")
    ap.add_argument("--tensors", help="comma-separated resident tensor names (sweep.json mode)")
    ap.add_argument("--layers", help="comma-separated layer indices (class mode, manifest.json)")
    ap.add_argument("--classes", default="all", help="class mode: all or a comma list of qg,qkvz,o,gateup,down")
    ap.add_argument("--rows", action="append", default=[],
                    help="class mode: CLASS:TOKENS:MA[,MA], overrides the ANE rows")
    ap.add_argument("--tokens", default="2048")
    ap.add_argument("--share", type=float, default=None,
                    help="ANE share of the rows (default 0.5 per tensor; class mode defaults to the F16 programs' shares)")
    ap.add_argument("--lut", action="store_true",
                    help="class mode: 4-bit LUT weights, about half the Neural Engine memory of shift-scale")
    ap.add_argument("--shared-f16", metavar="DIR",
                    help="class mode: also list the F16 programs of this format 1 directory, for "
                         "the layers that have no baked program")
    ap.add_argument("--with-f16", action="store_true",
                    help="also write the F16-plane program of the same shape")
    args = ap.parse_args()
    if args.share is not None and not 0 < args.share < 1:
        raise SystemExit("--share must be in (0, 1)")
    if bool(args.tensors) == bool(args.layers):
        raise SystemExit("give --tensors (sweep.json mode) or --layers (class mode), not both or neither")
    tokens = sorted({int(v) for v in args.tokens.split(",") if v.strip()})
    if not tokens or tokens[0] < 1 or tokens[-1] > 4096:
        raise SystemExit("token counts must be in 1..=4096")

    install = Install(args.install)
    builder = load_builder()
    os.makedirs(args.out_dir, exist_ok=True)
    if args.layers:
        return bake_classes(args, install, builder, tokens)
    share = 0.5 if args.share is None else args.share
    manifest_path = Path(args.out_dir) / "sweep.json"
    manifest = {"format": FORMAT, "programs": []}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    programs = {p["package"]: p for p in manifest["programs"]}

    for name in [t.strip() for t in args.tensors.split(",") if t.strip()]:
        e = install.int4(name)
        rows, k = e["rows"], e["cols"]
        ma = max(32, int(rows * share) // 32 * 32)
        for n in tokens:
            package = f"ane_c4real_{short(name)}_ma{ma}_n{n}.mlpackage"
            path = Path(args.out_dir) / package
            if path.exists():
                print("exists", path)
            else:
                model, _, w_abs, w_level = baked_program(builder, install, [(name, ma)], n)
                model.save(str(path))
                print(f"wrote {path}  (worst abs err {w_abs:.2e}, {w_level:.3f} level)")
            programs[package] = {
                "kind": "dense", "weights": "const_int4", "tensor": name, "k": k,
                "rows": rows, "ane_rows": ma, "plane_rows": rows, "tokens": n,
                "share": ma / rows, "package": package,
            }
            if args.with_f16:
                f16 = f"ane_k{k}_rows{rows}_ma{ma}_n{n}.mlpackage"
                if not (Path(args.out_dir) / f16).exists():
                    builder.build(k, n, [(ma, rows)]).save(str(Path(args.out_dir) / f16))
                    print("wrote", Path(args.out_dir) / f16)
                programs[f16] = {
                    "kind": "dense", "k": k, "rows": rows, "ane_rows": ma,
                    "plane_rows": rows, "tokens": n, "share": ma / rows, "package": f16,
                }
    manifest["format"] = FORMAT
    manifest["generator"] = "crates/coreml/tools/bake_ane_prefill_programs.py"
    manifest["coremltools"] = builder.ct.__version__
    manifest["chunk"] = builder.CHUNK
    manifest["programs"] = list(programs.values())
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"{len(programs)} programs in {manifest_path}")


if __name__ == "__main__":
    sys.exit(main())
