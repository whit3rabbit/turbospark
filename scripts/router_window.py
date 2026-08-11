#!/usr/bin/env python3
"""Expert-union cost of a batched verify, from MFERENCE_ROUTER_TRACE captures.

Usage:
    router_window.py capture.json [skip] [M1,M2,...]

`capture.json` is a run of turbospark-check under
`MFERENCE_ROUTER_HIST=capture.json MFERENCE_ROUTER_TRACE=1`, which records
each layer's top-k expert ids in pass order. `skip` drops that many leading
passes, which is how prefill is excluded: the trace covers every forward
pass, prefill included, so pass the PROMPT TOKEN COUNT here (default 0).
`M` values default to 2,4,8,16.

The question it answers, for ROADMAP's speculative-decoding item: a batched
verify of M consecutive tokens is ONE forward pass that must read the UNION
of those M tokens' routes, where M sequential decode steps read them one
group at a time. So the cost of a verify step, in expert bytes, is

    step_mult = distinct(window of M) / top_k

against a decode step's 1.0, and the block pays only if the expected number
of ACCEPTED tokens exceeds step_mult. `redundancy` is the same number over
M, i.e. the fraction of the sequential traffic the union saves; 1.00 means
adjacent tokens route disjointly and the union saves nothing.

Two caveats this cannot see, both of which make the real saving SMALLER
than `redundancy` suggests:
  - the expert slot cache already dedupes across sequential tokens, so the
    sequential arm's true cost is misses, not touches;
  - a rejected token's bytes are spent either way.
Read step_mult as the number to beat, not redundancy as a win.

Stdlib only, deterministic, no model access.
"""

import json
import sys


def windows(row, top_k, m, skip):
    """Distinct expert count over each window of `m` consecutive passes."""
    passes = [
        row[i * top_k : (i + 1) * top_k] for i in range(skip, len(row) // top_k)
    ]
    return [len(set().union(*passes[i : i + m])) for i in range(len(passes) - m + 1)]


def main():
    if not 2 <= len(sys.argv) <= 4:
        sys.exit(__doc__)
    with open(sys.argv[1]) as f:
        cap = json.load(f)
    skip = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    sizes = [int(s) for s in sys.argv[3].split(",")] if len(sys.argv) > 3 else [2, 4, 8, 16]

    if "trace" not in cap:
        sys.exit("capture has no trace: re-run with MFERENCE_ROUTER_TRACE=1")
    top_k = cap["top_k"]
    moe = [row for row in cap["trace"] if row]
    if not moe:
        sys.exit("capture has no MoE layers")
    total_passes = len(moe[0]) // top_k

    # The counts must be the trace's own histogram, which is the check that
    # the two halves of the capture describe the same run.
    for layer, row in enumerate(cap["trace"]):
        if not row:
            continue
        counts = cap["counts"][layer]
        for expert in row:
            counts[expert] -= 1
        if any(counts):
            sys.exit(f"layer {layer}: trace does not reproduce counts")

    print(
        f"{cap['num_experts']} experts/layer, top_k {top_k}, "
        f"{total_passes} passes ({skip} skipped), {len(moe)} MoE layers"
    )
    print("    M  distinct  step_mult  redundancy  breakeven_accept")
    for m in sizes:
        per_layer = [windows(row, top_k, m, skip) for row in moe]
        if not per_layer[0]:
            print(f"{m:>5}  (fewer than M passes after skip)")
            continue
        mean = sum(sum(w) / len(w) for w in per_layer) / len(per_layer)
        step_mult = mean / top_k
        print(
            f"{m:>5}  {mean:>8.1f}  {step_mult:>9.2f}  "
            f"{step_mult / m:>10.2f}  {step_mult:>16.2f}"
        )


if __name__ == "__main__":
    main()
