#!/usr/bin/env python3
"""Check a TurboSpark Qwen4 GDN capture against SlotStream's L2 equation."""

import argparse
import json
import math
import struct
from pathlib import Path


def as_f16(value):
    return struct.unpack("<e", struct.pack("<e", value))[0]


def max_error(left, right):
    return max((abs(a - b) for a, b in zip(left, right)), default=0.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metadata", type=Path, help="capture JSON path")
    parser.add_argument("--atol", type=float, default=1e-3)
    args = parser.parse_args()

    metadata = json.loads(args.metadata.read_text())
    if metadata.get("format") != "qwen4_gdn_norm_capture_v1":
        raise SystemExit("unsupported capture format")
    q_heads = int(metadata["key_heads"])
    key_dim = int(metadata["key_dim"])
    v_heads = int(metadata["value_heads"])
    value_dim = int(metadata["value_dim"])
    width = int(metadata["qkv_width"])
    expected_width = 2 * q_heads * key_dim + v_heads * value_dim
    if width != expected_width:
        raise SystemExit(f"qkv width mismatch: capture={width}, shape={expected_width}")

    raw = (args.metadata.parent / metadata["data"]).read_bytes()
    half_count = 2 * width
    if len(raw) != half_count * 2:
        raise SystemExit(f"capture bytes mismatch: got {len(raw)}, expected {half_count * 2}")
    values = struct.unpack(f"<{half_count}e", raw)
    before = values[:width]
    after = values[width:]

    reference = list(before)
    legacy = list(before)
    reference_eps = float(metadata["reference_epsilon"])
    mean_eps = float(metadata["mean_space_epsilon"])
    if not math.isclose(mean_eps, reference_eps / key_dim, rel_tol=1e-6, abs_tol=1e-12):
        raise SystemExit("capture epsilon does not convert the reference value to mean space")
    for is_query, head_count in ((True, q_heads), (False, q_heads)):
        section_base = 0 if is_query else q_heads * key_dim
        for head in range(head_count):
            base = section_base + head * key_dim
            row = before[base : base + key_dim]
            sumsq = sum(value * value for value in row)
            ref_denom = math.sqrt(sumsq + reference_eps)
            q_scale = math.sqrt(key_dim) if is_query else 1.0
            old_scale = key_dim if is_query else math.sqrt(key_dim)
            old_denom = math.sqrt(sumsq / key_dim + 1e-6)
            for i, value in enumerate(row):
                reference[base + i] = as_f16(value / ref_denom / q_scale)
                legacy[base + i] = as_f16(value / old_denom / old_scale)

    qk_width = 2 * q_heads * key_dim
    ref_error = max_error(after[:qk_width], reference[:qk_width])
    legacy_error = max_error(after[:qk_width], legacy[:qk_width])
    changed_v = sum(
        left != right for left, right in zip(before[qk_width:], after[qk_width:])
    )
    print(
        f"layer={metadata['layer']} position={metadata['position']} "
        f"key_heads={q_heads} key_dim={key_dim}"
    )
    print(f"mean_space_epsilon={mean_eps:.9g} reference_epsilon={reference_eps:.9g}")
    print(f"max_abs_error_vs_slotstream_l2={ref_error:.9g}")
    print(f"max_abs_error_vs_legacy_mean_epsilon={legacy_error:.9g}")
    print(f"changed_v_elements={changed_v}/{width - qk_width}")
    if ref_error > args.atol:
        raise SystemExit(f"capture does not match the SlotStream equation (atol={args.atol})")
    if changed_v:
        raise SystemExit("normalization changed one or more v elements")


if __name__ == "__main__":
    main()
