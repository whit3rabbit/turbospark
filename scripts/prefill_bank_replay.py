#!/usr/bin/env python3
"""Screen a shared prefill expert bank against Flash Next's per-layer LFU.

Use a fresh-runner TURBOSPARK_ROUTER_TRACE capture with 128-token outer
chunks, default routed pipelining, and turn snapshots and speculation off.
Pass the actual templated prompt length so decode routes are excluded. The
same capture feeds all three arms. This projects requested expert bytes only.

`--layer-major` asks a different question of the same capture: bytes read per
prompt for batch-major against layer-major order, over `--chunks`, `--slots`
and `--prompt-tokens` lists. A chunk is read as the engine's routed
sub-batches (greedy, union of experts <= slots), one cache request each.

  batch_major_lfu   per-layer LFU that keeps experts across chunks (what the
                    engine has).
  batch_major_bank  no retention: the slots are emptied after every layer of
                    every chunk, the shape of a bank shared by all layers.
  layer_major       every chunk through layer L, then L+1. The slots are
                    emptied only between layers.
  compulsory        each (layer, expert) read once.

With private per-layer slots, order across layers changes nothing, so
layer_major equals batch_major_lfu; it differs from a design whose retention
the next layer would otherwise take away. The saving that matters for this
engine is `saving_vs_lfu_percent`; `saving_vs_bank_percent` is the upstream
(colibri) comparison. No runner change is justified under 20%.
"""

import argparse
import json
from pathlib import Path

from expert_policy_replay import from_capture
from moe_route_replay import routed_sub_batches
from pilot_ceiling import ExpertCache


BASELINE_ROWS = 16
WIDE_ROWS = 128


def lfu_misses(groups, rows, slots, num_experts, top_k):
    cache = ExpertCache(slots, num_experts)
    pipelined = slots >= 2 * top_k
    misses = 0
    for start in range(0, len(groups), rows):
        protected = ()
        for group in groups[start : start + rows]:
            misses += len(cache.plan(group, protected)[1])
            if pipelined:
                protected = group
    return misses


def bank_misses(groups):
    # The bank is overwritten at each layer boundary. An expert is read once
    # per layer per microbatch, including if the previous microbatch used it.
    return sum(
        len({expert for group in groups[start : start + WIDE_ROWS] for expert in group})
        for start in range(0, len(groups), WIDE_ROWS)
    )


def load_routed(trace, layout):
    """Per-layer pass groups and strides, checked against each other."""
    top_k = trace["top_k"]
    num_experts = trace["num_experts"]
    if not 0 < top_k <= num_experts:
        raise ValueError("expected 0 < top_k <= num_experts")
    for row in trace.get("trace", []):
        if len(row) % top_k or any(not 0 <= e < num_experts for e in row):
            raise ValueError("trace contains an incomplete pass or invalid expert ID")

    capture = from_capture(trace, "prefill capture")
    routed = capture["routed"]
    if any(len(groups) != capture["npass"] for groups in routed.values()):
        raise ValueError("routed layers have different pass counts")

    layers = layout["layers"]
    if (layout["expertsPerLayer"] != num_experts
            or layout["numLayers"] != len(layers)
            or list(routed) != list(range(len(layers)))
            or [entry["layer"] for entry in layers] != list(routed)):
        raise ValueError("expected one matching packed expert layout per routed layer")

    strides = {}
    for layer in routed:
        stride = layers[layer].get("expertStride", layout["expertStride"])
        if not 0 < stride <= layout["expertStride"]:
            raise ValueError(f"invalid expert stride at layer {layer}")
        strides[layer] = stride
    return top_k, num_experts, capture["npass"], routed, strides


def chunk_unions(groups, chunk, slots):
    """One chunk list per `chunk` tokens, each a list of sub-batch unions."""
    out = []
    for start in range(0, len(groups), chunk):
        part = groups[start : start + chunk]
        unions, at = [], 0
        for size in routed_sub_batches(part, slots):
            unions.append(sorted({e for group in part[at : at + size] for e in group}))
            at += size
        out.append(unions)
    return out


def order_misses(groups, chunk, slots, num_experts):
    """Expert reads of one layer under each loop order."""
    retained = ExpertCache(slots, num_experts)
    layer_major_cache = ExpertCache(slots, num_experts)
    chunks = chunk_unions(groups, chunk, slots)
    lfu = bank = 0
    for unions in chunks:
        emptied = ExpertCache(slots, num_experts)
        for union in unions:
            lfu += len(retained.plan(union)[1])
            bank += len(emptied.plan(union)[1])
    # Same requests in the same per-layer order, from a cache that is this
    # layer's alone for the whole prompt.
    layer_major = sum(
        len(layer_major_cache.plan(union)[1]) for unions in chunks for union in unions
    )
    return {
        "batch_major_lfu": lfu,
        "batch_major_bank": bank,
        "layer_major": layer_major,
        "compulsory": len({e for group in groups for e in group}),
    }


def order_replay(trace, layout, chunks, slot_counts, prompts):
    """Bytes per prompt for every (prompt, chunk, slots) cell."""
    top_k, num_experts, npass, routed, strides = load_routed(trace, layout)
    rows, skipped = [], []
    for prompt in prompts:
        if not 0 < prompt <= npass:
            skipped.append(prompt)
            continue
        for slots in slot_counts:
            if not top_k <= slots <= num_experts:
                raise ValueError("expected top_k <= slots <= num_experts")
            for chunk in chunks:
                if chunk < 1:
                    raise ValueError("chunk must be positive")
                total = {}
                for layer, groups in routed.items():
                    got = order_misses(groups[:prompt], chunk, slots, num_experts)
                    for name, misses in got.items():
                        total[name] = total.get(name, 0) + misses * strides[layer]
                saved = lambda base: 100 * (total[base] - total["layer_major"]) / total[base]
                rows.append({
                    "prompt_tokens": prompt, "slots_per_layer": slots, "chunk": chunk,
                    "bytes": total,
                    "saving_vs_lfu_percent": saved("batch_major_lfu"),
                    "saving_vs_bank_percent": saved("batch_major_bank"),
                })
    if not rows:
        raise ValueError(f"every prompt length is outside the {npass} captured passes")
    return {"captured_passes": npass, "routed_layers": len(routed),
            "skipped_prompts": skipped, "cells": rows}


def replay(trace, layout, slots, prompt_tokens):
    top_k, num_experts, npass, routed, strides = load_routed(trace, layout)
    if not top_k <= slots <= num_experts:
        raise ValueError("expected 0 < top_k <= slots <= num_experts")
    if not 0 < prompt_tokens <= npass:
        raise ValueError("prompt length is outside the captured passes")

    baseline = wide = bank = 0
    for layer, groups in routed.items():
        stride = strides[layer]
        prompt = groups[:prompt_tokens]
        baseline += stride * lfu_misses(prompt, BASELINE_ROWS, slots, num_experts, top_k)
        wide += stride * lfu_misses(prompt, WIDE_ROWS, slots, num_experts, top_k)
        bank += stride * bank_misses(prompt)

    return {
        "prompt_tokens": prompt_tokens,
        "routed_layers": len(routed),
        "slots_per_layer": slots,
        "requested_bytes": {"lfu_16": baseline, "lfu_128": wide, "bank_128": bank},
        "bank_reduction_percent": 100 * (baseline - bank) / baseline,
        "passes_screen": 5500 <= prompt_tokens <= 6500 and 10 * bank <= 9 * baseline,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--model", required=True, type=Path, help="Flash Next .gturbo install")
    parser.add_argument("--slots", help="resolved slots per layer; with --layer-major a list "
                        "(default 16,32,96)")
    parser.add_argument("--prompt-tokens", help="with --layer-major a list "
                        "(default 2048,8192,16384)")
    parser.add_argument("--layer-major", action="store_true",
                        help="compare batch-major and layer-major reads")
    parser.add_argument("--chunks", default="64,128,256,512")
    args = parser.parse_args()
    if not args.layer_major and not (args.slots and args.prompt_tokens):
        parser.error("--slots and --prompt-tokens are required")
    numbers = lambda text: [int(x) for x in text.split(",") if x.strip()]
    try:
        trace = json.loads(args.trace.read_text(encoding="utf-8"))
        layout_path = args.model / "packed_experts" / "layout.json"
        layout = json.loads(layout_path.read_text(encoding="utf-8"))
        if args.layer_major:
            result = order_replay(
                trace, layout, numbers(args.chunks), numbers(args.slots or "16,32,96"),
                numbers(args.prompt_tokens or "2048,8192,16384"))
        else:
            result = replay(trace, layout, int(args.slots), int(args.prompt_tokens))
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
