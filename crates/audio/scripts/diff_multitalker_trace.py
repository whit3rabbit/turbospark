#!/usr/bin/env python3
"""Compare a Rust Frozen E step trace with a NeMo trace and report the first divergence.

Both files are JSONL with one line per streaming step: the Rust side from
`examples/multitalker_meeting.rs` (`<id>.trace.jsonl`), the NeMo side from
`scripts/nemo_multitalker_trace.py` (`trace.jsonl`). Stages are compared in
pipeline order, so the first mismatching stage names where the port diverges:

  chunk geometry -> diarizer chunk preds -> active speakers -> targets
  -> pre-encode length -> tokens and frames -> decoded and cache lengths

Usage: diff_multitalker_trace.py RUST_TRACE NEMO_TRACE [--pred-tol 1e-4]
"""

import argparse
import json
import sys


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def max_abs(a, b):
    worst = 0.0
    for ra, rb in zip(a, b):
        for x, y in zip(ra, rb):
            worst = max(worst, abs(x - y))
    return worst


def main():
    p = argparse.ArgumentParser()
    p.add_argument("rust")
    p.add_argument("nemo")
    p.add_argument("--pred-tol", type=float, default=1e-4)
    a = p.parse_args()
    rust, nemo = load(a.rust), load(a.nemo)
    if len(rust) != len(nemo):
        print(f"step counts differ: rust {len(rust)} nemo {len(nemo)}")
    first = None
    counts = {"steps": 0, "pred_over_tol": 0, "active": 0, "targets": 0, "tokens": 0}
    worst_pred = 0.0
    for r, n in zip(rust, nemo):
        k = n["step"]
        counts["steps"] += 1
        checks = []
        if r["asr_chunk_frames"] != n["asr_chunk_frames"] or r["diar_input_frames"] != n["diar_input_frames"]:
            checks.append(("geometry", (r["asr_chunk_frames"], r["diar_input_frames"]), (n["asr_chunk_frames"], n["diar_input_frames"])))
        d = max_abs(r["chunk_preds"], n["chunk_preds"])
        worst_pred = max(worst_pred, d)
        if d > a.pred_tol:
            counts["pred_over_tol"] += 1
            checks.append(("chunk_preds", f"max |d| {d:.3g}", ""))
        if r["active"] != n["active"]:
            counts["active"] += 1
            checks.append(("active", r["active"], n["active"]))
        elif r["active"]:
            if r["spk_targets"] != n["spk_targets"] or r["bg_targets"] != n["bg_targets"]:
                counts["targets"] += 1
                checks.append(("targets", r["spk_targets"], n["spk_targets"]))
            if r["pre_encoded_len"] != n["pre_encoded_len"]:
                checks.append(("pre_encoded_len", r["pre_encoded_len"], n["pre_encoded_len"]))
            if r["tokens"] != n["tokens"] or r["token_frames"] != n["token_frames"]:
                counts["tokens"] += 1
                checks.append(("tokens", r["texts"], n["texts"]))
            if r["decoded_length"] != n["decoded_length"] or r["cache_last_channel_len"] != n["cache_last_channel_len"]:
                checks.append(("lengths", (r["decoded_length"], r["cache_last_channel_len"]), (n["decoded_length"], n["cache_last_channel_len"])))
        if checks and first is None:
            first = (k, checks)
    print(json.dumps(counts), f"worst chunk-pred |d| {worst_pred:.3g}")
    if first is None:
        print("no divergence: every compared stage matches on every step")
        return 0
    k, checks = first
    print(f"first divergence at step {k}:")
    for stage, got, want in checks:
        print(f"  {stage}: rust={got} nemo={want}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
