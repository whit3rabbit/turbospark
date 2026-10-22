"""Generate isolated, shape-driven expert probes, separate from dense manifests.

uv run --python 3.12 --with coremltools==9.0 --with numpy==2.2.6 \
  python -I crates/coreml/tools/gen_ane_expert_programs.py OUTPUT \
  --hidden 2048 --intermediate 512

All weights and activations are F16 inputs. INT4 reads, dequantization,
gather/scatter, router weighting, and synchronization are external costs.
These programs alone cannot qualify expert offload or arithmetic parity.
"""
import argparse
import importlib.util
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--hidden", type=int, required=True)
    parser.add_argument("--intermediate", type=int, required=True)
    parser.add_argument("--tokens", default="8,16,17,24,32,64,128,256")
    args = parser.parse_args()
    try:
        tokens = sorted(set(map(int, args.tokens.split(","))))
    except ValueError:
        parser.error("tokens must be comma-separated integers")
    if not tokens or tokens[0] < 1 or tokens[-1] > 256:
        parser.error("tokens must be in 1..256")
    if any(n <= 0 or n % 64 for n in [args.hidden, args.intermediate]):
        parser.error("dimensions must be positive multiples of 64")
    if args.output.exists():
        parser.error("output must not exist")

    # Share the projection builder, never the dense preset or its manifest.
    source = Path(__file__).with_name("gen_ane_prefill_programs.py")
    spec = importlib.util.spec_from_file_location("projection_builder", source)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    ct, mb, types = builder.ct, builder.mb, builder.types

    def fused(hidden, intermediate, n):
        @mb.program(input_specs=[
            mb.TensorSpec((1, 1, n, hidden), types.fp16),
            mb.TensorSpec((1, 1, intermediate, hidden), types.fp16),
            mb.TensorSpec((1, 1, intermediate, hidden), types.fp16),
            mb.TensorSpec((1, 1, hidden, intermediate), types.fp16),
        ], opset_version=ct.target.macOS15)
        def program(a, wg, wu, wd):
            gate = builder.matmul(a, wg, n, intermediate, hidden)
            up = builder.matmul(a, wu, n, intermediate, hidden)
            active = mb.mul(x=mb.mul(x=gate, y=mb.sigmoid(x=gate)), y=up)
            return mb.identity(x=mb.matmul(x=wd, y=active), name="o0")
        return ct.convert(program, convert_to="mlprogram",
                          minimum_deployment_target=ct.target.macOS15,
                          compute_precision=ct.precision.FLOAT16)

    args.output.mkdir(parents=True)
    programs = []
    h, i = args.hidden, args.intermediate
    for n in tokens:
        for kind in ["gateup", "down", "fused"]:
            if kind == "gateup":
                model = builder.build(h, n, [(i, i), (i, i)])
            elif kind == "down":
                model = builder.build(i, n, [(h, h)])
            else:
                model = fused(h, i, n)
            name = f"{kind}_h{h}_i{i}_n{n}.mlpackage"
            model.save(str(args.output / name))
            programs.append(dict(name=name, kind=kind, tokens=n))
    manifest = dict(format="expert-probe-v1", hidden=h, intermediate=i,
                    input_dtype="f16", coremltools=ct.__version__, programs=programs,
                    includes_preparation=False, includes_router_weighting=False)
    (args.output / "experiment.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
