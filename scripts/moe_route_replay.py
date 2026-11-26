#!/usr/bin/env python3
"""Analyze bounded TURBOSPARK_ROUTER_BATCHES captures; stdlib only.

No lifetime histogram or decoded window is an executable prefill batch.
--window concatenates contiguous prefill batches as a hypothetical window.
Cache and I/O data remain measured per original batch, never inferred.
--union-cap replays moe_prefill_pipeline::next_routed_sub_batch's greedy
shrink, so each row reports the routed sub-batch sizes that cap allows.

--phase verify reads the speculative verify batches instead: the output gains
`verify`, the distinct experts one verify reads, summed over layers. With
--expert-bytes and --committed-tokens (from `[spec-stats]`) it also prices
requested expert bytes per committed token. Requested, not read: the slot
cache serves some of them.

--streams 1,2,4,8 is a different analysis on the plain TURBOSPARK_ROUTER_TRACE
`trace` (no batch metadata needed): how the per-step expert union grows when N
independent sequences share one step. `ceiling` is N x top_k / union, the most
a batched decode of N streams can gain if expert reads were all the time.
Give N captures for N real streams; with fewer, the one capture is cut into N
equal segments, which shares a topic and so understates the union.
"""
import argparse
from collections import Counter
import json


def integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def routed_sub_batches(routes, cap):
    """Token counts of the greedy union-bounded sub-batches, in order.

    Mirrors next_routed_sub_batch: a sub-batch takes tokens while the
    union of their experts fits cap, and refuses a single token over cap.
    """
    sizes = []
    start = 0
    while start < len(routes):
        union = set(routes[start])
        if len(union) > cap:
            raise ValueError(f"one token routes {len(union)} experts against {cap} slots")
        n = 1
        while start + n < len(routes):
            new = set(routes[start + n]) - union
            if len(union) + len(new) > cap:
                break
            union |= new
            n += 1
        sizes.append(n)
        start += n
    return sizes


def stream_layers(capture, skip):
    """`{layer: [pass groups]}` from the legacy trace, from `skip` on."""
    k = integer(capture.get("top_k"), "top_k", 1)
    trace = capture.get("trace")
    if not trace:
        raise ValueError("no trace: re-run with TURBOSPARK_ROUTER_TRACE=1")
    out = {}
    for layer, row in enumerate(trace):
        if not row:
            continue
        if len(row) % k:
            raise ValueError("trace holds an incomplete pass")
        out[layer] = [row[i:i + k] for i in range(skip * k, len(row), k)]
    return out


def union_growth(captures, counts=(1, 2, 4, 8), skip=0):
    """Mean per-layer expert union of one step shared by N streams."""
    integer(skip, "skip")
    loaded = [stream_layers(c, skip) for c in captures]
    k = captures[0]["top_k"]
    if any(c["top_k"] != k or c["num_experts"] != captures[0]["num_experts"]
           or sorted(l) != sorted(loaded[0]) for c, l in zip(captures, loaded)):
        raise ValueError("captures are not from one install")
    rows = []
    for n in counts:
        integer(n, "streams", 1)
        if len(captures) >= n:
            streams, source = loaded[:n], "captures"
        elif len(captures) == 1:
            length = min(len(g) for g in loaded[0].values()) // n
            streams = [{layer: groups[i * length:(i + 1) * length]
                        for layer, groups in loaded[0].items()} for i in range(n)]
            source = "segments"
        else:
            raise ValueError(f"{len(captures)} captures cannot make {n} streams; "
                             "give N captures, or one to segment")
        steps = min(len(g) for s in streams for g in s.values())
        if steps < 1:
            raise ValueError(f"too few passes for {n} streams")
        sizes = [len({e for s in streams for e in s[layer][t]})
                 for layer in loaded[0] for t in range(steps)]
        union = sum(sizes) / len(sizes)
        rows.append({"streams": n, "source": source, "steps": steps,
                     "mean_union": union, "union_over_top_k": union / k,
                     "ceiling": n * k / union})
    return {"top_k": k, "routed_layers": len(loaded[0]), "skip": skip, "growth": rows}


def verify_summary(reports, expert_bytes=None, committed_tokens=None):
    """Distinct experts per verify (summed over layers) from batch rows."""
    per_verify = {}
    rows_per_verify = {}
    for r in reports:
        key = (r["request"], r["batch"])
        per_verify[key] = per_verify.get(key, 0) + r["expert_union"]
        rows_per_verify[key] = r["tokens"]
    if not per_verify:
        return {"verifies": 0}
    total = sum(per_verify.values())
    out = {"verifies": len(per_verify),
           "mean_rows": sum(rows_per_verify.values()) / len(per_verify),
           "mean_distinct_experts": total / len(per_verify)}
    if expert_bytes is not None:
        out["requested_bytes"] = total * expert_bytes
        if committed_tokens is not None:
            out["requested_bytes_per_committed_token"] = (
                total * expert_bytes / integer(committed_tokens, "committed_tokens", 1))
    return out


def analyze(capture, thresholds=(8, 17, 24, 32, 64), window=None,
            expert_bytes=None, allow_truncated=False, union_cap=None,
            phase="prefill"):
    experts = integer(capture.get("num_experts"), "num_experts", 1)
    layers = integer(capture.get("num_layers"), "num_layers", 1)
    k = integer(capture.get("top_k"), "top_k", 1)
    if k > experts:
        raise ValueError("top_k exceeds expert count")
    detail = capture.get("batches", {})
    if detail.get("version") != 1:
        raise ValueError("batch metadata version 1 required; legacy traces are ambiguous")
    if type(detail.get("truncated")) is not bool:
        raise ValueError("truncated must be boolean")
    if detail.get("truncated") and not allow_truncated:
        raise ValueError("truncated capture; use --allow-truncated for complete rows only")
    thresholds = tuple(integer(t, "threshold", 1) for t in thresholds)
    if expert_bytes is not None:
        integer(expert_bytes, "expert_bytes", 1)
    if window is not None:
        integer(window, "window", 1)
    if union_cap is not None:
        integer(union_cap, "union_cap", 1)
    rows = []
    seen = set()
    prior_offsets = {}
    for row in detail["rows"]:
        layer = integer(row["layer"], "layer")
        if layer >= layers:
            raise ValueError("layer outside model")
        start = integer(row["token_start"], "token_start")
        count = integer(row["token_count"], "token_count", 1)
        request = integer(row["request"], "request")
        batch = integer(row["batch"], "batch", 1)
        key = (request, batch, layer)
        if key in seen:
            raise ValueError("duplicate batch/layer")
        seen.add(key)
        routes = row["routes"]
        for route in routes:
            if len(route) != k or len(set(route)) != k:
                raise ValueError("route must contain top_k distinct experts")
            if any(type(e) is not int or not 0 <= e < experts for e in route):
                raise ValueError("invalid expert id")
        a = integer(row["trace_start"], "trace_start")
        b = integer(row["trace_end"], "trace_end")
        if b - a != len(routes) * k or a < prior_offsets.get(layer, 0):
            raise ValueError("invalid or overlapping trace offsets")
        prior_offsets[layer] = b
        if "trace" in capture and capture["trace"][layer][a:b] != [e for r in routes for e in r]:
            raise ValueError("batch routes disagree with legacy trace")
        if type(row["completed"]) is not bool:
            raise ValueError("completed must be boolean")
        if not row["completed"]:
            continue
        if len(routes) != count:
            raise ValueError("completed row has wrong token count")
        if row["phase"] != phase:
            continue
        for name in ("cache_hits", "cache_misses", "bytes_requested", "io_samples"):
            integer(row[name], name)
        if row["io_samples"] not in (0, count):
            raise ValueError("partial I/O observation in complete batch")
        if row["io_samples"] and row["cache_hits"] + row["cache_misses"] != count * k:
            raise ValueError("cache requests do not conserve selections")
        rows.append(dict(row, token_start=start, batch=batch))
    if window:
        grouped = {}
        for row in rows:
            grouped.setdefault((row["request"], row["layer"]), []).append(row)
        windows = []
        for group in grouped.values():
            group.sort(key=lambda r: r["token_start"])
            run = []
            end = None
            def flush():
                for first in range(0, len(run), window):
                    part = run[first:first + window]
                    windows.append(dict(group[0], routes=part, token_count=len(part),
                                        token_start=run_start + first))
            for row in group:
                if end is not None and row["token_start"] != end:
                    flush()
                    run = []
                if not run:
                    run_start = row["token_start"]
                run.extend(row["routes"])
                end = row["token_start"] + row["token_count"]
            if run:
                flush()
        rows = windows
    reports = []
    for row in rows:
        counts = Counter(e for route in row["routes"] for e in route)
        total = sum(counts.values())
        reports.append({
            "request": row["request"], "layer": row["layer"], "batch": row["batch"],
            "token_start": row["token_start"], "tokens": len(row["routes"]),
            "executable": window is None, "expert_union": len(counts),
            "selections": total, "counts": dict(sorted(counts.items())),
            "max_uses": max(counts.values(), default=0),
            "candidates": {str(t): {"experts": sum(n >= t for n in counts.values()),
                "selections": sum(n for n in counts.values() if n >= t),
                "coverage": sum(n for n in counts.values() if n >= t) / total,
                "modeled_weight_bytes": None if expert_bytes is None else
                    expert_bytes * sum(n >= t for n in counts.values())} for t in thresholds},
            "routed_sub_batches": None if union_cap is None else
                routed_sub_batches(row["routes"], union_cap),
            "measured_io": None if window or not row["io_samples"] else {
                name: row[name] for name in ("cache_hits", "cache_misses", "bytes_requested")},
        })
    return {"version": 1, "hypothetical_window": window, "union_cap": union_cap,
            "capture_truncated": detail["truncated"], "batches": reports}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("capture", nargs="+", help="several only with --streams")
    p.add_argument("--phase", default="prefill", choices=["prefill", "verify", "decode"])
    p.add_argument("--committed-tokens", type=int)
    p.add_argument("--streams", help="comma list of stream counts, e.g. 1,2,4,8")
    p.add_argument("--skip", type=int, default=0,
                   help="with --streams: leading passes to drop (the prompt)")
    p.add_argument("--window", type=int)
    p.add_argument("--thresholds", default="8,17,24,32,64")
    p.add_argument("--expert-bytes", type=int)
    p.add_argument("--allow-truncated", action="store_true")
    p.add_argument("--union-cap", type=int,
                   help="replay the routed sub-batch planner at this slot cap")
    args = p.parse_args()
    try:
        captures = []
        for path in args.capture:
            with open(path) as f:
                captures.append(json.load(f))
        if args.streams:
            result = union_growth(captures, tuple(map(int, args.streams.split(','))),
                                  args.skip)
        else:
            if len(captures) != 1:
                raise ValueError("one capture unless --streams is given")
            if args.window is not None and args.phase != "prefill":
                raise ValueError("--window concatenates prefill batches only")
            result = analyze(captures[0], tuple(map(int, args.thresholds.split(','))),
                             args.window, args.expert_bytes, args.allow_truncated,
                             args.union_cap, args.phase)
            if args.phase == "verify":
                result["verify"] = verify_summary(
                    result["batches"], args.expert_bytes, args.committed_tokens)
    except (OSError, ValueError, KeyError, TypeError, IndexError) as e:
        p.error(str(e))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
