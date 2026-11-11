#!/usr/bin/env python3
"""Screen a shared prefill expert bank against Flash Next's per-layer LFU.

Use a fresh-runner TURBOSPARK_ROUTER_TRACE capture with 128-token outer
chunks, default routed pipelining, and turn snapshots and speculation off.
Pass the actual templated prompt length so decode routes are excluded. The
same capture feeds all three arms. This projects requested expert bytes only.
"""

import argparse
import json
from pathlib import Path

from expert_policy_replay import from_capture
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


def replay(trace, layout, slots, prompt_tokens):
    top_k = trace["top_k"]
    num_experts = trace["num_experts"]
    if not 0 < top_k <= slots <= num_experts:
        raise ValueError("expected 0 < top_k <= slots <= num_experts")
    for row in trace.get("trace", []):
        if len(row) % top_k or any(not 0 <= e < num_experts for e in row):
            raise ValueError("trace contains an incomplete pass or invalid expert ID")

    capture = from_capture(trace, "prefill capture")
    routed = capture["routed"]
    if any(len(groups) != capture["npass"] for groups in routed.values()):
        raise ValueError("routed layers have different pass counts")
    if not 0 < prompt_tokens <= capture["npass"]:
        raise ValueError("prompt length is outside the captured passes")

    layers = layout["layers"]
    if (layout["expertsPerLayer"] != num_experts
            or layout["numLayers"] != len(layers)
            or list(routed) != list(range(len(layers)))
            or [entry["layer"] for entry in layers] != list(routed)):
        raise ValueError("expected one matching packed expert layout per routed layer")

    baseline = wide = bank = 0
    for layer, groups in routed.items():
        stride = layers[layer].get("expertStride", layout["expertStride"])
        if not 0 < stride <= layout["expertStride"]:
            raise ValueError(f"invalid expert stride at layer {layer}")
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
    parser.add_argument("--slots", required=True, type=int, help="resolved slots per layer")
    parser.add_argument("--prompt-tokens", required=True, type=int)
    args = parser.parse_args()
    try:
        trace = json.loads(args.trace.read_text(encoding="utf-8"))
        layout_path = args.model / "packed_experts" / "layout.json"
        layout = json.loads(layout_path.read_text(encoding="utf-8"))
        result = replay(trace, layout, args.slots, args.prompt_tokens)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
