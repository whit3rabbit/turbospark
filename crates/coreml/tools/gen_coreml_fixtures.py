"""Write the tiny Core ML fixture packages under crates/coreml/testdata/.

Each fixture exercises one marshalling path of turbospark-coreml with no
real model: f32 in/out, f16 in/out, int32 out, a multifunction package, and
a stateful accumulator. Expected outputs are simple closed forms the Rust
tests compute themselves, so the manifest only records provenance and
digests.

Run with a pinned, isolated interpreter. Building MIL programs is pure
Python, so no model download or device is involved:

    uv run --python 3.12 --with coremltools==9.0 --with numpy==2.2.6 \
        python -I crates/coreml/tools/gen_coreml_fixtures.py

Regeneration is separate from tests. Tests never run Python.
"""

import hashlib
import json
import os
import shutil
import sys
import tempfile

import coremltools as ct
import numpy as np
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types

OUT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "testdata"))


def convert(prog, target):
    return ct.convert(
        prog,
        convert_to="mlprogram",
        minimum_deployment_target=target,
        compute_precision=ct.precision.FLOAT32,
    )


def add_f32():
    # out = a + b, plus a second output so multi-output reads are covered.
    @mb.program(input_specs=[mb.TensorSpec((2, 3), types.fp32), mb.TensorSpec((2, 3), types.fp32)],
                opset_version=ct.target.macOS13)
    def prog(a, b):
        s = mb.add(x=a, y=b, name="sum")
        d = mb.sub(x=a, y=b, name="diff")
        return s, d

    return convert(prog, ct.target.macOS13)


def f16_scale():
    # y = 2 * x in float16 end to end.
    @mb.program(input_specs=[mb.TensorSpec((1, 4), types.fp16)], opset_version=ct.target.macOS13)
    def prog(x):
        return mb.mul(x=x, y=np.float16(2.0), name="y")

    return convert(prog, ct.target.macOS13)


def i32_argmax():
    # idx = argmax(x, axis=-1) as int32.
    @mb.program(input_specs=[mb.TensorSpec((1, 8), types.fp32)], opset_version=ct.target.macOS13)
    def prog(x):
        i = mb.reduce_argmax(x=x, axis=-1, keep_dims=False)
        return mb.cast(x=i, dtype="int32", name="idx")

    return convert(prog, ct.target.macOS13)


def scale_by(k):
    @mb.program(input_specs=[mb.TensorSpec((1, 4), types.fp32)], opset_version=ct.target.macOS15)
    def prog(x):
        return mb.mul(x=x, y=np.float32(k), name="y")

    return convert(prog, ct.target.macOS15)


def multifunction(path):
    # Two functions sharing one package: "double" (default) and "triple".
    tmp = tempfile.mkdtemp()
    try:
        d = os.path.join(tmp, "double.mlpackage")
        t = os.path.join(tmp, "triple.mlpackage")
        scale_by(2.0).save(d)
        scale_by(3.0).save(t)
        desc = ct.utils.MultiFunctionDescriptor()
        desc.add_function(d, src_function_name="main", target_function_name="double")
        desc.add_function(t, src_function_name="main", target_function_name="triple")
        desc.default_function_name = "double"
        ct.utils.save_multifunction(desc, path)
    finally:
        shutil.rmtree(tmp)


def accumulator():
    # acc += x; y = acc. State must be float16 in Core ML.
    @mb.program(
        input_specs=[
            mb.StateTensorSpec((1, 4), dtype=types.fp16),
            mb.TensorSpec((1, 4), types.fp16),
        ],
        opset_version=ct.target.macOS15,
    )
    def prog(acc, x):
        cur = mb.read_state(input=acc)
        nxt = mb.add(x=cur, y=x)
        mb.coreml_update_state(state=acc, value=nxt)
        return mb.identity(x=nxt, name="y")

    return ct.convert(
        prog,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=ct.precision.FLOAT16,
    )


def digest_tree(path):
    h = hashlib.sha256()
    for root, _, files in sorted(os.walk(path)):
        for f in sorted(files):
            p = os.path.join(root, f)
            h.update(os.path.relpath(p, path).encode())
            with open(p, "rb") as fh:
                h.update(fh.read())
    return h.hexdigest()


def main():
    os.makedirs(OUT, exist_ok=True)
    fixtures = {
        "add_f32.mlpackage": lambda p: add_f32().save(p),
        "f16_scale.mlpackage": lambda p: f16_scale().save(p),
        "i32_argmax.mlpackage": lambda p: i32_argmax().save(p),
        "multifunction.mlpackage": multifunction,
        "accumulator.mlpackage": lambda p: accumulator().save(p),
    }
    manifest = {
        "generator": "crates/coreml/tools/gen_coreml_fixtures.py",
        "python": sys.version.split()[0],
        "coremltools": ct.__version__,
        "numpy": np.__version__,
        "fixtures": {},
    }
    for name, build in fixtures.items():
        path = os.path.join(OUT, name)
        if os.path.exists(path):
            shutil.rmtree(path)
        build(path)
        manifest["fixtures"][name] = {"sha256_tree": digest_tree(path)}
        print("wrote", path)
    with open(os.path.join(OUT, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")


if __name__ == "__main__":
    main()
