#!/usr/bin/env python3
"""Activation-sparsity analysis for MFERENCE_FFN_HIST captures.

Usage:
    ffn_sparsity.py capture1.json[,capture2.json,...]

Each argument is a comma-separated group of census JSONs (one per
turbospark-check run) summed into one corpus. Reports, per layer and as
means:

  n50/n90/n95/n99  neurons covering that fraction of cumulative |act| mass
                   (out of `inter`); the coverage curve that decides whether
                   a hot-neuron working set exists at all
  <t drops         fraction of activation MASS below a few magnitude
                   thresholds (the CATS-style operating curve: mass you
                   could zero, not merely elements)
  ov512..ov4096    mean overlap fraction of CONSECUTIVE tokens' top-K
                   neuron sets; the temporal locality a neuron cache needs

Then the working-set arithmetic: what fraction of FFN weight bytes a
hot-set of n95 neurons would keep resident, folded against the model's
non-FFN resident floor.

Sanity: per layer, sum(hist_count) == decode_passes * inter, or the
capture is refused. Stdlib only, deterministic, no model access.
"""

import json
import sys

# Muse Glimmer's non-FFN resident bytes (attention + embedding + head +
# norms) out of 15.67 GB total; used only for the closing arithmetic.
NON_FFN_GB = 4.8
FFN_GB = 10.9

THRESHOLDS = [1e-3, 1e-2, 1e-1]


def load_group(arg):
    total = None
    for path in arg.split(","):
        with open(path) as f:
            cap = json.load(f)
        for layer in range(cap["num_layers"]):
            got = sum(cap["hist_count"][layer])
            want = cap["decode_passes"] * cap["inter"]
            if got != want:
                sys.exit(f"{path}: layer {layer} hist_count sums to {got}, "
                         f"want decode_passes*inter = {want}: capture is broken")
        if total is None:
            total = cap
        else:
            for key in ("num_layers", "inter", "overlap_ks", "bucket_exp_offset"):
                if cap[key] != total[key]:
                    sys.exit(f"{path}: {key} mismatch within group")
            total["decode_passes"] += cap["decode_passes"]
            total["prefill_passes"] += cap["prefill_passes"]
            total["overlap_pairs"] += cap["overlap_pairs"]
            for field in ("mass", "hist_count", "hist_mass", "overlap_hits"):
                for layer, row in enumerate(cap[field]):
                    trow = total[field][layer]
                    for i, v in enumerate(row):
                        trow[i] += v
    return total


def n_frac(mass_row, frac):
    """Smallest neuron count covering `frac` of the layer's |act| mass."""
    target = frac * sum(mass_row)
    got, n = 0.0, 0
    for v in sorted(mass_row, reverse=True):
        if got >= target or v == 0.0:
            break
        got += v
        n += 1
    return n


def mass_below(cap, layer, threshold):
    """Fraction of the layer's activation mass in buckets under `threshold`.

    Bucket b covers [2^(b-off), 2^(b-off+1)); a bucket counts as "below"
    only if its whole range is, so this is a slight UNDER-estimate of the
    droppable mass -- the conservative direction for a go/no-go.
    """
    off = cap["bucket_exp_offset"]
    total = sum(cap["hist_mass"][layer])
    if total == 0.0:
        return 0.0
    below = sum(m for b, m in enumerate(cap["hist_mass"][layer])
                if b == 0 or (b < 63 and 2.0 ** (b - off + 1) <= threshold))
    return below / total


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    cap = load_group(sys.argv[1])
    inter, layers = cap["inter"], cap["num_layers"]
    ks = cap["overlap_ks"]
    pairs = cap["overlap_pairs"]
    print(f"{layers} layers x {inter} neurons, {cap['decode_passes']} decode "
          f"passes ({cap['prefill_passes']} prefill skipped), "
          f"{pairs} consecutive pairs")

    header = "layer   n50   n90   n95   n99"
    header += "".join(f"  <{t:g}" for t in THRESHOLDS)
    header += "".join(f"  ov{k}" for k in ks)
    print(header)

    ncols = 4 + len(THRESHOLDS) + len(ks)
    sums = [0.0] * ncols
    for layer in range(layers):
        mass = cap["mass"][layer]
        cols = [n_frac(mass, f) for f in (0.50, 0.90, 0.95, 0.99)]
        cols += [mass_below(cap, layer, t) for t in THRESHOLDS]
        cols += [cap["overlap_hits"][layer][ki] / (pairs * k) if pairs else 0.0
                 for ki, k in enumerate(ks)]
        for i, c in enumerate(cols):
            sums[i] += c
        print("{:>5}  {:>4}  {:>4}  {:>4}  {:>4}".format(layer, *cols[:4])
              + "".join(f"  {c:.3f}" for c in cols[4:]))

    means = [s / layers for s in sums]
    print("mean   {:>4.0f}  {:>4.0f}  {:>4.0f}  {:>4.0f}".format(*means[:4])
          + "".join(f"  {c:.3f}" for c in means[4:]))

    n95 = means[2]
    frac = n95 / inter
    print()
    print(f"n95 mean = {n95:.0f}/{inter} = {frac:.1%} of neurons carry 95% "
          f"of activation mass")
    print(f"working set at an n95 hot set: {NON_FFN_GB:.1f} GB non-FFN + "
          f"{frac * FFN_GB:.1f} GB of {FFN_GB:.1f} GB FFN "
          f"= {NON_FFN_GB + frac * FFN_GB:.1f} GB "
          f"(vs {NON_FFN_GB + FFN_GB:.1f} GB resident today)")
    print("gates: n95 <= ~15% of rows with high ov* -> a neuron cache is "
          "plausible; n95 >= ~50% -> dead end (docs/EXPERT_ROUTING.md style)")


if __name__ == "__main__":
    main()
