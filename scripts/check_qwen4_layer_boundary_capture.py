#!/usr/bin/env python3
"""Summarize a Qwen4 residual capture and optionally compare layer outputs.

SlotStream's current_backend_reference.py writes layer_i.bin as float32 with
shape [sequence, hc_count * hidden]. Its last dimension matches this runtime
capture's after_moe_join stage. Compare only when checkpoint weights and input
token IDs are known to match; otherwise the reported difference is diagnostic,
not a parity result.

The llama.cpp callback mode compares all three stages against named float32
vectors emitted for the same final prompt token.
"""

import argparse
import json
import math
import struct
from pathlib import Path


STAGES = ("layer_input_after_ple", "after_attention_join", "after_moe_join")


def unpack_values(path: Path, dtype: str, count: int) -> tuple[float, ...]:
    data = path.read_bytes()
    width = 2 if dtype == "e" else 4
    if len(data) != count * width:
        raise ValueError(f"{path}: expected {count * width} bytes, got {len(data)}")
    return struct.unpack("<" + dtype * count, data)


def summarize(values: tuple[float, ...]) -> str:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return f"count={len(values)} finite=0"
    norm = math.sqrt(math.fsum(value * value for value in finite))
    return (
        f"count={len(values)} finite={len(finite)} "
        f"min={min(finite):.7g} max={max(finite):.7g} l2={norm:.7g}"
    )


def compare(actual: tuple[float, ...], expected: tuple[float, ...]) -> str:
    if len(actual) != len(expected):
        raise ValueError(f"vector width mismatch: {len(actual)} != {len(expected)}")
    errors = [a - b for a, b in zip(actual, expected)]
    dot = math.fsum(a * b for a, b in zip(actual, expected))
    actual_norm = math.sqrt(math.fsum(a * a for a in actual))
    expected_norm = math.sqrt(math.fsum(b * b for b in expected))
    cosine = dot / (actual_norm * expected_norm) if actual_norm and expected_norm else 0.0
    max_abs = max((abs(error) for error in errors), default=0.0)
    rmse = math.sqrt(math.fsum(error * error for error in errors) / len(errors))
    return (
        f"max_abs={max_abs:.7g} rmse={rmse:.7g} "
        f"actual_l2={actual_norm:.7g} reference_l2={expected_norm:.7g} "
        f"cosine={cosine:.9g}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path, help="Qwen4 layer-capture JSON header")
    parser.add_argument(
        "--reference-dir",
        type=Path,
        help="directory containing SlotStream layer_0.bin, layer_1.bin, ...",
    )
    parser.add_argument(
        "--llama-callback-dir",
        type=Path,
        help="directory containing final-token float32 tensors from the llama.cpp callback collector",
    )
    parser.add_argument(
        "--router-trace-json",
        type=Path,
        help="optional TurboSpark TURBOSPARK_ROUTER_HIST output with TURBOSPARK_ROUTER_TRACE enabled",
    )
    parser.add_argument(
        "--router-callback-dir",
        type=Path,
        help="directory containing llama.cpp ffn_moe_topk-L-0.f32 tensors; defaults to --llama-callback-dir",
    )
    parser.add_argument(
        "--router-trace-position",
        type=int,
        help="0-based input-token row to compare with the callback's final prompt token",
    )
    parser.add_argument(
        "--reference-position",
        type=int,
        default=-1,
        help="sequence row in each SlotStream output (default: last row)",
    )
    args = parser.parse_args()

    header = json.loads(args.capture.read_text())
    if header.get("format") != "qwen4_layer_boundary_capture_v1":
        raise ValueError(f"unsupported capture format: {header.get('format')!r}")
    layers = int(header["layers"])
    wide_dim = int(header["wide_dim"])
    data_path = args.capture.parent / header["data"]
    values = unpack_values(data_path, "e", len(STAGES) * layers * wide_dim)

    print(
        f"capture position={header['position']} layers={layers} "
        f"wide_dim={wide_dim} dtype={header['dtype']}"
    )
    for stage_index, stage in enumerate(STAGES):
        for layer in range(layers):
            start = (stage_index * layers + layer) * wide_dim
            vector = values[start : start + wide_dim]
            print(f"{stage} layer={layer} {summarize(vector)}")

    if args.reference_dir is not None and args.llama_callback_dir is not None:
        parser.error("use only one of --reference-dir and --llama-callback-dir")
    if args.router_trace_json is not None and args.llama_callback_dir is None:
        parser.error("--router-trace-json requires --llama-callback-dir")
    if args.router_callback_dir is not None and args.router_trace_json is None:
        parser.error("--router-callback-dir requires --router-trace-json")
    if (args.router_trace_json is None) != (args.router_trace_position is None):
        parser.error("--router-trace-json and --router-trace-position must be used together")

    if args.reference_dir is None and args.llama_callback_dir is None:
        return

    if args.llama_callback_dir is not None:
        callback_dir = args.llama_callback_dir
        missing = 0
        for layer in range(layers):
            input_name = (
                "hc_init-0.f32" if layer == 0 else (
                    "hc_input-1.f32" if layer == 1 else f"l_last-{layer - 1}-0.f32"
                )
            )
            names = (input_name, f"hc_combine-{layer}-0.f32", f"l_last-{layer}-0.f32")
            for stage_index, (stage, name) in enumerate(zip(STAGES, names)):
                reference_path = callback_dir / name
                if not reference_path.exists():
                    print(f"llama.cpp {stage} layer={layer} missing={name}")
                    missing += 1
                    continue
                reference = unpack_values(reference_path, "f", wide_dim)
                start = (stage_index * layers + layer) * wide_dim
                actual = values[start : start + wide_dim]
                print(f"llama.cpp {stage} layer={layer} {compare(actual, reference)}")
        print(
            "llama.cpp callback vectors are the final prompt token. Treat the "
            "deltas as parity evidence only when source GGUF and prompt token IDs match."
        )
        if missing:
            print(f"llama.cpp callback vectors missing={missing}; those stages were skipped")
        if args.router_trace_json is not None:
            trace_header = json.loads(args.router_trace_json.read_text())
            top_k = int(trace_header["top_k"])
            traces = trace_header["trace"]
            position = args.router_trace_position
            route_callback_dir = args.router_callback_dir or callback_dir
            exact = []
            order_only = []
            set_changes = []
            route_missing = []
            for layer in range(min(layers, len(traces))):
                reference_path = route_callback_dir / f"ffn_moe_topk-{layer}-0.f32"
                if not reference_path.exists():
                    route_missing.append(layer)
                    continue
                cpu_values = unpack_values(reference_path, "f", top_k)
                cpu_ids = [int(round(value)) for value in cpu_values]
                start = position * top_k
                turbo_ids = [int(value) for value in traces[layer][start : start + top_k]]
                if len(turbo_ids) != top_k:
                    route_missing.append(layer)
                    continue
                if cpu_ids == turbo_ids:
                    exact.append(layer)
                elif set(cpu_ids) == set(turbo_ids):
                    order_only.append((layer, cpu_ids, turbo_ids))
                else:
                    set_changes.append((layer, cpu_ids, turbo_ids))
            print(f"router position={position} top_k={top_k} exact_order_layers={exact}")
            print(f"router same_set_different_order={order_only}")
            print(f"router different_sets={set_changes}")
            if route_missing:
                print(f"router references missing or too short for layers={route_missing}")
            logits_capture = trace_header.get("router_logits")
            if logits_capture is not None:
                logits_layer = int(logits_capture["layer"])
                logits_pass = int(logits_capture["pass"])
                logits_values = logits_capture.get("values")
                if logits_values is None:
                    print(
                        f"router logits layer={logits_layer} pass={logits_pass} "
                        "capture_missing=true"
                    )
                elif logits_pass != position:
                    print(
                        f"router logits layer={logits_layer} pass={logits_pass} "
                        f"requested_position={position}; comparison skipped"
                    )
                else:
                    reference_path = route_callback_dir / f"ffn_moe_logits-{logits_layer}-0.f32"
                    if not reference_path.exists():
                        print(
                            f"router logits layer={logits_layer} pass={logits_pass} "
                            f"missing_reference={reference_path.name}"
                        )
                    else:
                        reference_logits = unpack_values(reference_path, "f", len(logits_values))
                        actual_logits = tuple(float(value) for value in logits_values)
                        cpu_ranked = sorted(
                            range(len(reference_logits)),
                            key=lambda expert: (-reference_logits[expert], expert),
                        )[: top_k + 1]
                        turbo_ranked = sorted(
                            range(len(actual_logits)),
                            key=lambda expert: (-actual_logits[expert], expert),
                        )[: top_k + 1]
                        print(
                            f"router logits layer={logits_layer} pass={logits_pass} "
                            f"{compare(actual_logits, reference_logits)}"
                        )
                        print(
                            f"router logit top_k_plus_one "
                            f"cpu={[(expert, round(reference_logits[expert], 7)) for expert in cpu_ranked]} "
                            f"turbo={[(expert, round(actual_logits[expert], 7)) for expert in turbo_ranked]}"
                        )
                        if len(cpu_ranked) > top_k and len(turbo_ranked) > top_k:
                            cpu_gap = reference_logits[cpu_ranked[top_k - 1]] - reference_logits[cpu_ranked[top_k]]
                            turbo_gap = actual_logits[turbo_ranked[top_k - 1]] - actual_logits[turbo_ranked[top_k]]
                            print(
                                f"router logit cutoff_gap cpu={cpu_gap:.7g} "
                                f"turbo={turbo_gap:.7g}"
                            )
        return

    if args.reference_position < 0:
        requested_position = None
    else:
        requested_position = args.reference_position
    for layer in range(layers):
        reference_path = args.reference_dir / f"layer_{layer}.bin"
        data = reference_path.read_bytes()
        row_bytes = wide_dim * 4
        if len(data) == 0 or len(data) % row_bytes:
            raise ValueError(
                f"{reference_path}: byte length {len(data)} is not a nonzero "
                f"multiple of row size {row_bytes}"
            )
        sequence = len(data) // row_bytes
        position = sequence - 1 if requested_position is None else requested_position
        if not 0 <= position < sequence:
            raise ValueError(
                f"{reference_path}: reference position {position} outside sequence length {sequence}"
            )
        expected = unpack_values(reference_path, "f", sequence * wide_dim)
        reference = expected[position * wide_dim : (position + 1) * wide_dim]
        start = (2 * layers + layer) * wide_dim
        actual = values[start : start + wide_dim]
        print(
            f"SlotStream after_moe_join layer={layer} "
            f"reference_position={position} {compare(actual, reference)}"
        )
    print(
        "Compare as parity evidence only if the checkpoint weights and input "
        "token IDs match; quantization differences can change these values."
    )


if __name__ == "__main__":
    main()
