#!/usr/bin/env python3
"""Concentration and overlap analysis for TURBOSPARK_ROUTER_HIST captures.

Usage:
    router_hist.py A1.json[,A2.json,...] [B1.json[,B2.json,...]]

Each argument is a comma-separated group of histogram JSONs (one per
turbospark-check run) that are summed into one corpus. With one group it
reports per-layer routing concentration (how many experts cover 95% / 99%
of the routed mass, and how many were never routed). With two groups it
adds the Jaccard overlap of the two corpora's 95%-mass "hot" expert sets,
which is the go/no-go number for a domain-pruned or pre-warmed expert set:
high overlap means the domains route alike and there is nothing to prune.

Stdlib only, deterministic, no model access.
"""

import json
import sys


def load_group(arg):
    total = None
    for path in arg.split(","):
        with open(path) as f:
            hist = json.load(f)
        counts = hist["counts"]
        if total is None:
            total = [row[:] for row in counts]
        else:
            if len(counts) != len(total) or any(
                len(a) != len(b) for a, b in zip(counts, total)
            ):
                sys.exit(f"{path}: shape mismatch within group")
            for layer, row in enumerate(counts):
                for expert, n in enumerate(row):
                    total[layer][expert] += n
    return total


def hot_set(row, frac):
    """Smallest expert set covering `frac` of the layer's routed mass."""
    target = frac * sum(row)
    got, out = 0, set()
    for expert in sorted(range(len(row)), key=lambda e: -row[e]):
        if got >= target or row[expert] == 0:
            break
        out.add(expert)
        got += row[expert]
    return out


def main():
    if len(sys.argv) not in (2, 3):
        sys.exit(__doc__)
    a = load_group(sys.argv[1])
    b = load_group(sys.argv[2]) if len(sys.argv) == 3 else None
    num_experts = len(a[0]) if a else 0

    two = b is not None
    header = "layer  A:n95  A:n99  A:zero"
    if two:
        header += "  B:n95  B:n99  B:zero  jac95"
    print(f"{num_experts} experts/layer")
    print(header)

    sums = [0.0] * (7 if two else 3)
    moe_layers = 0
    for layer, row in enumerate(a):
        if sum(row) == 0:
            continue  # non-MoE layer
        moe_layers += 1
        h95 = hot_set(row, 0.95)
        cols = [len(h95), len(hot_set(row, 0.99)), row.count(0)]
        if two:
            brow = b[layer]
            bh95 = hot_set(brow, 0.95)
            jac = len(h95 & bh95) / len(h95 | bh95) if h95 | bh95 else 0.0
            cols += [len(bh95), len(hot_set(brow, 0.99)), brow.count(0), jac]
        for i, c in enumerate(cols):
            sums[i] += c
        fmt = "{:>5}  {:>5}  {:>5}  {:>6}"
        if two:
            fmt += "  {:>5}  {:>5}  {:>6}  {:.3f}"
        print(fmt.format(layer, *cols))

    if moe_layers:
        means = [s / moe_layers for s in sums]
        line = "mean   {:>5.1f}  {:>5.1f}  {:>6.1f}".format(*means[:3])
        if two:
            line += "  {:>5.1f}  {:>5.1f}  {:>6.1f}  {:.3f}".format(*means[3:])
        print(line)


if __name__ == "__main__":
    main()
