#!/usr/bin/env python3
"""What is the CEILING on a one-layer-ahead expert prefetcher?

Reads an `MFERENCE_ROUTER_HIST` capture taken with `MFERENCE_ROUTER_TRACE=1`
and answers, offline, the questions that decide whether a router-lookahead
prefetcher (colibri's PILOT) is worth building here -- WITHOUT building a
predictor first.

The reasoning, which is the whole point of running this before writing any
kernel: a prefetcher can only ever convert a MISS into a hit. So the miss
count is the ceiling on every predictor, perfect or not, and it is a property
of the trace and the cache policy alone. If the ceiling is small, a 71.6%-
recall predictor cannot be worth its dispatch, and no GEMV needs writing to
find that out.

Four things are reported, and only the last is new work:

1. Hit rate and misses per layer per pass, at each slot count. This is the
   ceiling. `docs/BENCHMARKS.md` has the io milliseconds to price it with.

2. Previous-token recall: how much of a layer's top-k repeats from the
   previous pass. colibri measures 41.3% for this predictor on GLM-5.2 and
   71.6% for PILOT, so this is the number that says whether the two models'
   routing has comparable structure at all.

3. Whether the previous token's selection would have let a prefetcher issue
   any read. `DEVIATIONS.md:793` states it cannot, because per-layer private
   caches keep the last token's experts resident. That is a claim about this
   engine recorded from upstream's measurement, and the trace can check it
   directly rather than inheriting it.

4. Miss composition: COLD (this expert has never been routed at this layer
   before) against EVICTED (it has, and the cache dropped it). This is the
   one number no existing page here carries, and it picks the lever:
   evictions are what a prefetcher or more slots can address, and a cold
   miss is compulsory -- no policy avoids the first touch.

Stdlib only, matching `scripts/router_hist.py`.
"""

import argparse
import json
import sys


class ExpertCache:
    """Port of `crates/streaming/src/expert_cache.rs`'s LFU policy.

    Faithful on the two details that change the answer. Eviction order is
    computed from the counts BEFORE this request's increment (the Rust sorts
    `evictable` and only then bumps `expert_use_count`), and empty slots sort
    ahead of occupied ones regardless of count.
    """

    def __init__(self, slot_count, num_experts):
        self.slot_count = slot_count
        self.slot_expert = [None] * slot_count
        self.slot_last_use = [0] * slot_count
        self.use_count = [0] * num_experts
        self.use_clock = 0

    def _eviction_key(self, slot):
        expert = self.slot_expert[slot]
        if expert is None:
            # (None, Some) => Less: empty slots are taken first.
            return (0, 0, self.slot_last_use[slot])
        return (1, self.use_count[expert], self.slot_last_use[slot])

    def plan(self, experts):
        """Returns (hits, misses) as lists of expert ids, and commits."""
        clock = self.use_clock + 1
        reserved = [False] * self.slot_count
        hits, misses = [], []
        assigned = []

        for expert in experts:
            slot = next(
                (
                    s
                    for s in range(self.slot_count)
                    if not reserved[s] and self.slot_expert[s] == expert
                ),
                None,
            )
            if slot is None:
                misses.append(expert)
            else:
                hits.append(expert)
                reserved[slot] = True
                assigned.append(slot)

        evictable = sorted(
            (s for s in range(self.slot_count) if not reserved[s]),
            key=self._eviction_key,
        )
        if len(misses) > len(evictable):
            raise RuntimeError(
                f"cache cannot place {len(misses)} misses in "
                f"{len(evictable)} evictable slots"
            )

        self.use_clock = clock
        for expert in experts:
            self.use_count[expert] += 1
        for slot in assigned:
            self.slot_last_use[slot] = clock
        for offset, expert in enumerate(misses):
            slot = evictable[offset]
            self.slot_expert[slot] = expert
            self.slot_last_use[slot] = clock

        return hits, misses


def passes(trace_row, top_k):
    """The flat row split back into one group of `top_k` ids per pass."""
    return [
        trace_row[i : i + top_k] for i in range(0, len(trace_row) - top_k + 1, top_k)
    ]


def analyse(capture, slot_counts, skip):
    top_k = capture["top_k"]
    num_experts = capture["num_experts"]
    trace = capture.get("trace")
    if not trace:
        sys.exit(
            "capture has no `trace` block: re-run the capture with "
            "MFERENCE_ROUTER_TRACE=1 set alongside MFERENCE_ROUTER_HIST"
        )

    routed = [(layer, passes(row, top_k)) for layer, row in enumerate(trace) if row]
    if not routed:
        sys.exit("capture routes on no layer at all")

    npass = min(len(p) for _, p in routed)
    if skip >= npass:
        sys.exit(f"--skip {skip} leaves no passes of {npass}")

    # Sanity invariant the histogram itself carries: per-layer counts must be
    # recomputable from the trace. A mismatch means the two halves of the
    # capture disagree and neither should be believed.
    for layer, layer_passes in routed:
        recomputed = [0] * num_experts
        for group in layer_passes:
            for expert in group:
                recomputed[expert] += 1
        if recomputed != capture["counts"][layer]:
            sys.exit(
                f"layer {layer}: trace does not reproduce the counts; "
                "the capture is internally inconsistent"
            )

    print(f"passes {npass} (analysing from {skip}), routed layers {len(routed)}, "
          f"top_k {top_k}, experts {num_experts}")
    print()

    # -- 2 and 3: predictor-free, cache-free facts about the routing itself.
    repeat_hits = repeat_total = 0
    for _, layer_passes in routed:
        for p in range(max(skip, 1), npass):
            previous = set(layer_passes[p - 1])
            repeat_hits += len(previous.intersection(layer_passes[p]))
            repeat_total += top_k
    print(f"previous-token recall  {100.0 * repeat_hits / repeat_total:.1f}%"
          f"   (colibri reports 41.3% for this predictor, 71.6% for PILOT)")
    print()

    # -- the PILOT prediction, when the capture carries one.
    pred = capture.get("pred")
    predicted = {}
    if pred:
        for layer, _ in routed:
            row = passes(pred[layer], top_k) if pred[layer] else []
            if row:
                predicted[layer] = row
        if predicted:
            hits = total = 0
            for layer, rows in predicted.items():
                actual = dict(routed)[layer]
                for p in range(max(skip, 0), min(npass, len(rows))):
                    hits += len(set(rows[p]).intersection(actual[p]))
                    total += top_k
            print(f"PILOT top-k recall     {100.0 * hits / total:.1f}%"
                  f"   (stale-state router, {len(predicted)} layers)")
            print()

    for slots in slot_counts:
        caches = {layer: ExpertCache(slots, num_experts) for layer, _ in routed}
        seen = {layer: set() for layer, _ in routed}

        miss_total = req_total = 0
        cold = evicted = 0
        prev_covered = 0
        pilot_covered = pilot_miss_total = 0
        pilot_fetched = 0

        for p in range(npass):
            for layer, layer_passes in routed:
                group = layer_passes[p]
                # Snapshot residency BEFORE the plan: a prefetcher issues its
                # reads while the previous layer computes, so what it would
                # have had to fetch is what was resident THEN. Reading it
                # after `plan` has already installed this pass's misses makes
                # every prefetch look free (measured: it reports 0 experts
                # read to save 2,455, which is impossible).
                resident_before = {
                    e for e in caches[layer].slot_expert if e is not None
                }
                _, misses = caches[layer].plan(group)
                if p >= skip:
                    req_total += len(group)
                    miss_total += len(misses)
                    previous = set(layer_passes[p - 1]) if p > 0 else set()
                    guess = None
                    if layer in predicted and p < len(predicted[layer]):
                        guess = set(predicted[layer][p])
                        pilot_miss_total += len(misses)
                        # What the prefetcher would actually have READ: its
                        # whole guess minus what was resident when it fired.
                        # The bytes it wastes are as much the story as the
                        # bytes it saves.
                        pilot_fetched += len(guess - resident_before)
                    for expert in misses:
                        if expert in seen[layer]:
                            evicted += 1
                        else:
                            cold += 1
                        if expert in previous:
                            prev_covered += 1
                        if guess is not None and expert in guess:
                            pilot_covered += 1
                seen[layer].update(group)

        hit_rate = 100.0 * (req_total - miss_total) / req_total
        per_layer_pass = miss_total / (len(routed) * (npass - skip))
        print(f"slots {slots:>3}   hit rate {hit_rate:5.1f}%   "
              f"misses/layer/pass {per_layer_pass:.2f}   ({miss_total} total)")
        if miss_total:
            print(f"            composition: {100.0 * cold / miss_total:5.1f}% cold "
                  f"(compulsory), {100.0 * evicted / miss_total:5.1f}% evicted "
                  f"(addressable)")
            print(f"            of those misses, {100.0 * prev_covered / miss_total:.1f}% "
                  f"were named by the PREVIOUS pass at that layer")
        if pilot_miss_total:
            print(f"            PILOT would cover "
                  f"{100.0 * pilot_covered / pilot_miss_total:.1f}% of misses, "
                  f"reading {pilot_fetched} experts to save {pilot_covered} "
                  f"({pilot_fetched / max(pilot_covered, 1):.2f} read per hit)")
            pilot_k_sweep(routed, predicted, num_experts, slots, npass, skip, top_k)
        print()


def pilot_k_sweep(routed, predicted, num_experts, slots, npass, skip, top_k):
    """colibri's `PILOT_K`: prefetch only the top k of the prediction.

    The head of the router's ranking is more reliable than its tail, so a
    narrower prefetch trades coverage for bandwidth. This is the knob that
    decides the whole question here, because the full-width prefetcher reads
    MORE bytes than it saves and the engine's expert read is a page-cache
    memcpy (`crates/streaming/CLAUDE.md` Gotcha 3) -- a bandwidth cost, not a
    latency one.

    `net` is the ratio of total expert reads under prefetch to the baseline's
    demand reads. Below 1.00 the prefetcher moves fewer bytes AND hides
    latency; above it, it is buying overlap with bandwidth.
    """
    print(f"            PILOT_K sweep (prefetch only the top k of the guess):")
    print(f"              k   coverage   read/hit   net bytes vs baseline")
    for k in range(1, top_k + 1):
        caches = {layer: ExpertCache(slots, num_experts) for layer, _ in routed}
        covered = fetched = misses_seen = 0
        for p in range(npass):
            for layer, layer_passes in routed:
                group = layer_passes[p]
                resident_before = {e for e in caches[layer].slot_expert if e is not None}
                _, misses = caches[layer].plan(group)
                if p < skip or layer not in predicted or p >= len(predicted[layer]):
                    continue
                guess = set(predicted[layer][p][:k])
                misses_seen += len(misses)
                fetched += len(guess - resident_before)
                covered += sum(1 for e in misses if e in guess)
        if not misses_seen:
            continue
        # Reads under prefetch: every speculative read, plus the misses the
        # guess did not name and which therefore still fault in on demand.
        total = fetched + (misses_seen - covered)
        print(f"              {k}   {100.0 * covered / misses_seen:6.1f}%   "
              f"{fetched / max(covered, 1):7.2f}    {total / misses_seen:.2f}x")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("capture", help="JSON written by MFERENCE_ROUTER_HIST")
    ap.add_argument("--slots", default="16,32",
                    help="comma-separated slot counts to simulate (default 16,32)")
    ap.add_argument("--skip", type=int, default=0,
                    help="passes to exclude from the statistics while still "
                         "warming the cache; use it to read decode alone")
    args = ap.parse_args()

    with open(args.capture, encoding="utf-8") as f:
        capture = json.load(f)

    analyse(capture, [int(s) for s in args.slots.split(",")], args.skip)


if __name__ == "__main__":
    main()
