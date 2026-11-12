#!/usr/bin/env python3
"""Summarize a Metal System Trace for tslab (evaluator item L5).

Usage: python3 -I trace_tables.py TRACE.trace PID

Exports two tables with `xcrun xctrace export` and prints one JSON object:

- encoders: the process's compute encoder intervals from
  `metal-gpu-intervals`: count, the window they span, GPU busy percent over
  that window (the union of intervals), idle gaps between them, mean encoder
  time. Low busy inside a decode window means launch-bound, high busy means
  kernel-time bound.
- shaders: `metal-shader-profiler-intervals` (needs the "Metal GPU
  Counters" instrument) grouped by kernel name: samples, sampled time, and
  share of the process's sampled time. Samples cover about a tenth of GPU
  time, so compare shares between two traces, never microseconds.

Standard library only. xctrace compresses repeated values: an element with
`id` defines a value, a later element with `ref` repeats it, and this
resolves both. Nothing from the trace is executed or evaluated.
"""

import json
import subprocess
import sys
import xml.etree.ElementTree as ET


def export(trace, schema):
    xpath = '/trace-toc/run[@number="1"]/data/table[@schema="%s"]' % schema
    out = subprocess.run(
        ["xcrun", "xctrace", "export", "--input", trace, "--xpath", xpath],
        capture_output=True,
        check=False,
    )
    if out.returncode != 0 or not out.stdout.strip():
        return None
    return ET.fromstring(out.stdout)


def rows(root):
    """Yield each row as {mnemonic: element}, with refs resolved."""
    if root is None:
        return
    node = root.find("node")
    if node is None:
        return
    schema = node.find("schema")
    cols = [c.findtext("mnemonic") for c in schema.findall("col")] if schema is not None else []
    defined = {}

    def resolve(el):
        ref = el.get("ref")
        if ref is not None:
            return defined.get(ref, el)
        ident = el.get("id")
        if ident is not None:
            defined[ident] = el
        # Nested definitions (a process's pid) must register too.
        for child in el.iter():
            if child is not el and child.get("id") is not None:
                defined[child.get("id")] = child
        return el

    for row in node.findall("row"):
        values = {}
        for mnemonic, el in zip(cols, list(row)):
            values[mnemonic] = resolve(el)
        yield values


def pid_of(el):
    if el is None or el.tag == "sentinel":
        return None
    pid = el.find("pid")
    if pid is not None and pid.text:
        return int(pid.text)
    return None


def num(el):
    try:
        return int(el.text)
    except (AttributeError, TypeError, ValueError):
        return None


def label(el):
    if el is None:
        return ""
    return el.get("fmt") or (el.text or "")


def percentile(sorted_values, q):
    if not sorted_values:
        return None
    i = min(len(sorted_values) - 1, int(round(q * (len(sorted_values) - 1))))
    return sorted_values[i]


def encoders(trace, pid):
    spans = []
    for r in rows(export(trace, "metal-gpu-intervals")):
        if pid_of(r.get("process")) != pid:
            continue
        if label(r.get("channel-name")) != "Compute":
            continue
        start, dur = num(r.get("start")), num(r.get("duration"))
        if start is None or dur is None:
            continue
        spans.append((start, start + dur))
    if not spans:
        return {"count": 0}
    spans.sort()
    busy, gaps = 0, []
    cur_s, cur_e = spans[0]
    for s, e in spans[1:]:
        if s > cur_e:
            busy += cur_e - cur_s
            gaps.append(s - cur_e)
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    busy += cur_e - cur_s
    window = spans[-1][1] - spans[0][0]
    gaps.sort()
    us = lambda ns: None if ns is None else ns / 1e3
    return {
        "count": len(spans),
        "window_ms": window / 1e6,
        "busy_percent": 100.0 * busy / window if window else None,
        "mean_encoder_us": sum(e - s for s, e in spans) / len(spans) / 1e3,
        "idle_gap_us": {
            "p50": us(percentile(gaps, 0.5)),
            "p90": us(percentile(gaps, 0.9)),
            "max": us(gaps[-1] if gaps else None),
            "count": len(gaps),
        },
    }


def shaders(trace, pid):
    by_name = {}
    total = 0
    for r in rows(export(trace, "metal-shader-profiler-intervals")):
        if pid_of(r.get("process")) != pid:
            continue
        dur = num(r.get("duration"))
        if dur is None:
            continue
        name = label(r.get("name"))
        # "dequant_int8_gemv_simd (0)": the suffix is a pipeline index.
        if name.endswith(")") and " (" in name:
            name = name[: name.rfind(" (")]
        entry = by_name.setdefault(name, {"name": name, "samples": 0, "sampled_us": 0.0})
        entry["samples"] += 1
        entry["sampled_us"] += dur / 1e3
        total += dur / 1e3
    out = sorted(by_name.values(), key=lambda e: -e["sampled_us"])
    for e in out:
        e["share"] = e["sampled_us"] / total if total else None
    return out


def main():
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    trace, pid = sys.argv[1], int(sys.argv[2])
    print(json.dumps({"pid": pid, "encoders": encoders(trace, pid), "shaders": shaders(trace, pid)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
