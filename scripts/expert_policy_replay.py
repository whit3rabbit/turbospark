#!/usr/bin/env python3
"""How much is expert-cache POLICY worth, before anyone builds one?

Replays one or more `TURBOSPARK_ROUTER_HIST` captures (taken with
`TURBOSPARK_ROUTER_TRACE=1`) through several slot-cache policies at the same
slot count, and through the offline optimum, so a proposed policy change is
priced against its ceiling the way `scripts/pilot_ceiling.py` prices a
prefetcher. Stdlib only, deterministic, no model access.

Usage:
    expert_policy_replay.py a.json [b.json ...] [--slots 16,32]
                            [--skip 21 | --skip 21,30] [--half-life 64,256]

SEVERAL CAPTURES ARE REPLAYED BACK TO BACK THROUGH THE SAME CACHES. That is
the server's shape rather than the CLI's: `PreadExpertStreamer` is opened
once per runner and its `ExpertCache` is never reset, so a long-lived
process carries one request's routing statistics into the next. Every
published hit rate on this engine is one short process (one capture), which
cannot see that. Pass, say, a coding capture and then a general one to ask
what a domain change costs each policy.

Arms, all at the same slot count:

  lfu (engine)    `crates/streaming/src/expert_cache.rs`, through the SAME
                  port `pilot_ceiling.py` validated against the engine's own
                  recorded numbers. Its counts never age.
  lru             Least recently used, per layer.
  lfu-decay h=H   The engine's LFU with counts halving every H passes at that
                  layer (ds4's policy, which jitllm's ROADMAP item 12 also
                  lists unbuilt). Same tie-break and same count-before-
                  increment order as the engine.
  oracle          Belady's rule per layer: evict the resident expert whose
                  next request is farthest away, never one this pass needs.
                  No online policy at this slot count can miss less;
                  `test_expert_policy_replay.py` checks it against brute
                  force.
  pool lru,       ONE cache of slots x layers shared by every layer, the
  pool oracle     shape of jitllm's pager (`format/jlm/read.go` `evictOne`).
                  Same BYTES as the per-layer arms only when every layer's
                  expert stride is equal; a mixed-precision install is not.

Also printed: the compulsory floor (first touch of each (layer, expert) in
the whole replay), which no policy and no slot count avoids.

Read the oracle row first. If it sits a point or two above `lfu (engine)`,
policy work is closed at that slot count and slots are the lever
(`docs/DECODE_BUDGET.md`). The rows between say which mechanism gets how
much of the gap.
"""

import argparse
import bisect
import json
import math
import sys

from pilot_ceiling import ExpertCache, passes


class EngineLfu:
    """`pilot_ceiling.ExpertCache` behind the replay's interface."""

    name = "lfu (engine)"

    def __init__(self, slots, num_experts):
        self.cache = ExpertCache(slots, num_experts)

    def plan(self, group, now):
        _, misses = self.cache.plan(group)
        return len(misses)


class KeyedCache:
    """A slot cache over arbitrary keys whose eviction is a sort key.

    Slot identity never changes a miss count, so slots are a dict here;
    free capacity is taken before anything is evicted, which is the
    engine's empty-slots-first arm. `victim_key(key, now)` sorts ascending
    and the smallest is evicted first.
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.resident = set()

    def victim_key(self, key, now):
        raise NotImplementedError

    def touch(self, keys, now):
        """Bookkeeping for a committed request; called after eviction."""

    def plan(self, keys, now):
        if len(keys) > self.capacity:
            raise RuntimeError(f"{len(keys)} requested, {self.capacity} slots")
        wanted = set(keys)
        misses = [k for k in keys if k not in self.resident]
        need = len(self.resident) + len(misses) - self.capacity
        if need > 0:
            evictable = sorted(
                (k for k in self.resident if k not in wanted),
                key=lambda k: self.victim_key(k, now),
            )
            for k in evictable[:need]:
                self.resident.discard(k)
        self.resident.update(misses)
        self.touch(keys, now)
        return len(misses)


class Lru(KeyedCache):
    name = "lru"

    def __init__(self, capacity):
        super().__init__(capacity)
        self.last_use = {}

    def victim_key(self, key, now):
        return self.last_use[key]

    def touch(self, keys, now):
        for k in keys:
            self.last_use[k] = now


class DecayedLfu(KeyedCache):
    """LFU whose counts halve every `half_life` requests to this cache.

    Each request adds 2^(t/H) to its expert's stored value, which orders
    experts exactly as the decayed count sum 2^-((now-t)/H) does without
    touching every count per pass. The exponent is rebased before it can
    overflow a float.
    """

    REBASE_EXPONENT = 512

    def __init__(self, capacity, half_life):
        super().__init__(capacity)
        self.name = f"lfu-decay h={half_life}"
        self.half_life = half_life
        self.value = {}
        self.last_use = {}
        self.clock = 0
        self.base = 0

    def victim_key(self, key, now):
        return (self.value.get(key, 0.0), self.last_use.get(key, 0))

    def touch(self, keys, now):
        self.clock += 1
        exponent = (self.clock - self.base) / self.half_life
        if exponent > self.REBASE_EXPONENT:
            shift = math.ldexp(1.0, -self.REBASE_EXPONENT)
            self.value = {k: v * shift for k, v in self.value.items()}
            self.base += self.REBASE_EXPONENT * self.half_life
            exponent -= self.REBASE_EXPONENT
        weight = 2.0 ** exponent
        for k in keys:
            self.value[k] = self.value.get(k, 0.0) + weight
            self.last_use[k] = now


class Oracle(KeyedCache):
    """Belady: evict the key whose next request is farthest away."""

    name = "oracle"

    def __init__(self, capacity, uses):
        super().__init__(capacity)
        self.uses = uses

    def victim_key(self, key, now):
        times = self.uses[key]
        i = bisect.bisect_right(times, now)
        # Never requested again sorts first: it is the free eviction.
        return -(times[i] if i < len(times) else math.inf)


def load(path):
    with open(path, encoding="utf-8") as f:
        return from_capture(json.load(f), path)


def from_capture(capture, path):
    trace = capture.get("trace")
    if not trace:
        sys.exit(
            f"{path}: no `trace` block; re-run the capture with "
            "TURBOSPARK_ROUTER_TRACE=1 set alongside TURBOSPARK_ROUTER_HIST"
        )
    top_k, num_experts = capture["top_k"], capture["num_experts"]
    routed = {layer: passes(row, top_k) for layer, row in enumerate(trace) if row}
    if not routed:
        sys.exit(f"{path}: routes on no layer at all")
    # The histogram must be recomputable from the trace, as pilot_ceiling.py
    # requires; a capture whose halves disagree prices nothing.
    for layer, groups in routed.items():
        recomputed = [0] * num_experts
        for group in groups:
            if len(set(group)) != len(group):
                sys.exit(f"{path}: layer {layer} routes one expert twice in a pass")
            for expert in group:
                recomputed[expert] += 1
        if recomputed != capture["counts"][layer]:
            sys.exit(f"{path}: layer {layer}'s trace does not reproduce its counts")
    npass = min(len(g) for g in routed.values())
    return {"path": path, "top_k": top_k, "num_experts": num_experts,
            "routed": routed, "npass": npass}


def schedule(captures):
    """The replay's request order: capture, then pass, then layer.

    Returns `(capture_index, pass_index, layer, group)` rows, one per layer
    step, in the order the engine issues them.
    """
    steps = []
    for c, cap in enumerate(captures):
        layers = sorted(cap["routed"])
        for p in range(cap["npass"]):
            for layer in layers:
                steps.append((c, p, layer, cap["routed"][layer][p]))
    return steps


def next_use_index(steps):
    uses = {}
    for t, (_, _, layer, group) in enumerate(steps):
        for expert in group:
            uses.setdefault((layer, expert), []).append(t)
    return uses


def replay(steps, per_layer_factory, pool=None):
    """Miss counts per capture for one arm.

    `per_layer_factory(layer)` builds one layer's cache; `pool`, when given,
    is a single cache every layer shares, keyed by (layer, expert).
    """
    caches = {}
    for t, (c, p, layer, group) in enumerate(steps):
        if pool is not None:
            cache, keys = pool, [(layer, e) for e in group]
        else:
            if layer not in caches:
                caches[layer] = per_layer_factory(layer)
            cache, keys = caches[layer], list(group)
        missed = cache.plan(keys, t)
        yield c, p, len(group), missed


def tally(rows, skips):
    """Sum requests and misses per capture over the passes not skipped."""
    out = {}
    for c, p, requested, missed in rows:
        if p < skips[c]:
            continue
        req, mis = out.get(c, (0, 0))
        out[c] = (req + requested, mis + missed)
    return out


def arms(steps, slots, num_experts, layer_count, half_lives):
    uses = next_use_index(steps)
    per_layer_uses = {}
    for (layer, expert), times in uses.items():
        per_layer_uses.setdefault(layer, {})[expert] = times
    yield "lfu (engine)", replay(steps, lambda layer: EngineLfu(slots, num_experts))
    yield "lru", replay(steps, lambda layer: Lru(slots))
    for h in half_lives:
        yield f"lfu-decay h={h}", replay(steps, lambda layer, h=h: DecayedLfu(slots, h))
    yield "oracle", replay(steps, lambda layer: Oracle(slots, per_layer_uses[layer]))
    yield "pool lru", replay(steps, None, pool=Lru(slots * layer_count))
    yield "pool oracle", replay(steps, None, pool=Oracle(slots * layer_count, uses))


def compulsory(steps, skips):
    seen = set()
    out = {}
    for c, p, layer, group in steps:
        for expert in group:
            key = (layer, expert)
            if key not in seen:
                seen.add(key)
                if p >= skips[c]:
                    out[c] = out.get(c, 0) + 1
    return out


def parse_list(text, cast):
    return [cast(x) for x in text.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("captures", nargs="+", help="JSON written by TURBOSPARK_ROUTER_HIST")
    ap.add_argument("--slots", default="16,32",
                    help="comma-separated slot counts per layer (default 16,32)")
    ap.add_argument("--skip", default="0",
                    help="leading passes per capture to leave out of the statistics "
                         "while still warming the caches (one value for all, or one "
                         "per capture); the prompt token count reads decode alone")
    ap.add_argument("--half-life", default="64,256",
                    help="decayed-LFU half-lives in passes (default 64,256)")
    args = ap.parse_args()

    captures = [load(p) for p in args.captures]
    first = captures[0]
    for cap in captures[1:]:
        if (cap["top_k"], cap["num_experts"], sorted(cap["routed"])) != (
                first["top_k"], first["num_experts"], sorted(first["routed"])):
            sys.exit(f"{cap['path']}: top_k, expert count or routed layers differ "
                     f"from {first['path']}; these are not one install")
    skips = parse_list(args.skip, int)
    if len(skips) == 1:
        skips = skips * len(captures)
    if len(skips) != len(captures):
        sys.exit(f"--skip names {len(skips)} values for {len(captures)} captures")
    for cap, skip in zip(captures, skips):
        if skip >= cap["npass"]:
            sys.exit(f"{cap['path']}: --skip {skip} leaves no passes of {cap['npass']}")

    steps = schedule(captures)
    layer_count = len(first["routed"])
    top_k = first["top_k"]
    print(f"routed layers {layer_count}, top_k {top_k}, experts {first['num_experts']}")
    for c, (cap, skip) in enumerate(zip(captures, skips)):
        print(f"  #{c + 1} {cap['path']}: {cap['npass']} passes, analysing from {skip}")
    floor = compulsory(steps, skips)

    labels = [f"#{c + 1}" for c in range(len(captures))]
    for slots in parse_list(args.slots, int):
        if slots < top_k:
            sys.exit(f"--slots {slots} is below top_k {top_k}; the engine refuses it")
        print()
        print(f"slots {slots} per layer (pool: {slots * layer_count} shared)")
        header = "  ".join(f"{l:>7}" for l in labels)
        print(f"  {'arm':<16} {header}  {'all':>7}  misses/layer/pass")
        results = {}
        for name, rows in arms(steps, slots, first["num_experts"], layer_count,
                               parse_list(args.half_life, int)):
            per = tally(rows, skips)
            results[name] = per
            req = sum(r for r, _ in per.values())
            mis = sum(m for _, m in per.values())
            passes_counted = sum(cap["npass"] - s for cap, s in zip(captures, skips))
            cells = "  ".join(
                f"{100.0 * (per[c][0] - per[c][1]) / per[c][0]:6.1f}%"
                for c in range(len(captures)))
            print(f"  {name:<16} {cells}  {100.0 * (req - mis) / req:6.1f}%  "
                  f"{mis / (layer_count * passes_counted):.2f}")
        req_all = sum(r for r, _ in results["lru"].values())
        floor_all = sum(floor.values())
        print(f"  {'compulsory':<16} {'':>{9 * len(captures) - 2}}  "
              f"{100.0 * (req_all - floor_all) / req_all:6.1f}%  (no policy beats this)")

        # The oracle's defining property. If any online arm misses less, the
        # replay is wrong and none of these rows should be read.
        def total(name):
            return sum(m for _, m in results[name].values())
        for name in results:
            bound = "pool oracle" if name.startswith("pool") else "oracle"
            if total(name) < total(bound):
                sys.exit(f"inconsistent replay: {name} missed {total(name)} < "
                         f"{bound} {total(bound)}")
            if total(name) < floor_all:
                sys.exit(f"inconsistent replay: {name} beat the compulsory floor")
    print()
    print("Pool rows hold the same bytes as per-layer rows only if every layer's "
          "expert stride is equal.")


if __name__ == "__main__":
    main()
