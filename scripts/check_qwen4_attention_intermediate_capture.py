#!/usr/bin/env python3
"""Compare a TurboSpark Qwen4 attention capture with llama.cpp eval-callback tensors.

Use the same GGUF checkpoint and exact token IDs for both captures. The CPU
reference files are the last prompt-token rows captured from named llama.cpp
graph nodes. This reports boundaries; it intentionally does not set a numeric
pass threshold because CPU/GPU accumulation and half-precision boundaries vary.
"""

import argparse
import json
import math
import struct
from pathlib import Path


REFERENCE_FILES = {
    "hc_normed": ("hc_norm-{layer}-0.f32", "plain"),
    "hc_gate": ("hc_gate-{layer}-0.f32", "sigmoid"),
    "hc_mixed": ("hc_mixed-{layer}-0.f32", "plain"),
    "branch_output": ("linear_attn_out-{layer}-0.f32", "plain"),
    "inject_gate": ("hc_inject-{layer}-0.f32", "inject"),
    "wide_after_join": ("hc_combine-{layer}-0.f32", "plain"),
}


def unpack(path: Path, code: str, count: int) -> tuple[float, ...]:
    raw = path.read_bytes()
    width = struct.calcsize(code)
    if len(raw) != count * width:
        raise ValueError(f"{path}: expected {count * width} bytes, got {len(raw)}")
    return struct.unpack("<" + code * count, raw)


def summarize(actual: tuple[float, ...], reference: tuple[float, ...]) -> str:
    if len(actual) != len(reference):
        raise ValueError(f"vector width mismatch: {len(actual)} != {len(reference)}")
    errors = [left - right for left, right in zip(actual, reference)]
    dot = math.fsum(left * right for left, right in zip(actual, reference))
    actual_norm = math.sqrt(math.fsum(value * value for value in actual))
    reference_norm = math.sqrt(math.fsum(value * value for value in reference))
    cosine = dot / (actual_norm * reference_norm) if actual_norm and reference_norm else 0.0
    rmse = math.sqrt(math.fsum(error * error for error in errors) / len(errors))
    return (
        f"width={len(actual)} max_abs={max(map(abs, errors), default=0.0):.7g} "
        f"rmse={rmse:.7g} turbo_l2={actual_norm:.7g} "
        f"llama_l2={reference_norm:.7g} cosine={cosine:.9g}"
    )


def transform(values: tuple[float, ...], kind: str, hc_count: int) -> tuple[float, ...]:
    if kind == "plain":
        return values
    if kind == "sigmoid":
        return tuple(1.0 / (1.0 + math.exp(-value)) for value in values)
    if kind == "inject":
        return tuple(2.0 / (1.0 + math.exp(-value / hc_count)) for value in values)
    raise ValueError(f"unsupported transform: {kind}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path, help="TurboSpark attention capture JSON")
    parser.add_argument("llama_dir", type=Path, help="llama.cpp eval-callback tensor directory")
    args = parser.parse_args()

    metadata = json.loads(args.capture.read_text())
    if metadata.get("format") != "qwen4_attention_intermediate_capture_v1":
        raise ValueError(f"unsupported capture format: {metadata.get('format')!r}")
    layer = int(metadata["layer"])
    wide_dim = int(metadata["wide_dim"])
    widths = {stage["name"]: int(stage["width"]) for stage in metadata["stages"]}
    data = unpack(args.capture.parent / metadata["data"], "e", len(widths) * wide_dim)
    hc_count = int(metadata["hc_count"])

    print(
        f"layer={layer} position={metadata['position']} wide_dim={wide_dim} "
        f"hidden_dim={metadata['hidden_dim']} hc_count={hc_count}"
    )
    for stage_index, (name, (filename_template, transform_kind)) in enumerate(REFERENCE_FILES.items()):
        if name not in widths:
            raise ValueError(f"TurboSpark capture is missing stage {name!r}")
        width = widths[name]
        turbo = data[stage_index * wide_dim : stage_index * wide_dim + width]
        filename = filename_template.format(layer=layer)
        reference_path = args.llama_dir / filename
        if not reference_path.is_file():
            raise FileNotFoundError(
                f"{reference_path} is missing; branch_output currently requires "
                "a GDN layer captured by llama.cpp eval-callback"
            )
        reference = transform(unpack(reference_path, "f", width), transform_kind, hc_count)
        print(f"{name}: {summarize(turbo, reference)}")


if __name__ == "__main__":
    main()
