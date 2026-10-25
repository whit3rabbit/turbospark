#!/usr/bin/env python3
"""Score a SegLST against the frozen Frozen E references with the frozen scorer.

`tokens`, `edit` and `cpwer` are copied from the frozen qualification notebook
(TurboSpark_Nemotron_E_Frozen_Qualification_v3, cell 8A) without behavioral
changes. The Rust scorer in crates/bench/src/meeting/cpwer.rs must agree with
this script count for count. Dev-only: needs numpy and scipy.

Usage:
  frozen_e_score.py --manifest evaluation_manifest.json --seglst seglst.json [--clips ID,ID,...]
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

MATCHED_SIX = [
    "IS1000a_00510_00630",
    "IS1001a_00390_00510",
    "IS1002b_00880_01000",
    "IS1003a_00320_00440",
    "IS1004a_00140_00260",
    "IS1005a_00440_00560",
]


def tokens(s):
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", str(s).lower())


def edit(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def cpwer(ref_words, segs):
    ref = defaultdict(list)
    hyp = defaultdict(list)
    for w in sorted(ref_words, key=lambda x: x["start"]):
        ref[w["speaker"]].extend(tokens(w["text"]))
    for seg in sorted(segs, key=lambda x: float(x.get("start_time", 0))):
        hyp[str(seg["speaker"])].extend(tokens(seg["words"]))
    a = list(dict(sorted(ref.items())).values())
    b = list(dict(sorted(hyp.items())).values())
    k = max(len(a), len(b))
    a += [[] for _ in range(k - len(a))]
    b += [[] for _ in range(k - len(b))]
    if k > 8:
        raise ValueError("Too many speakers for this bounded comparison")
    cost = np.array([[edit(x, y) for y in b] for x in a], dtype=int)
    r, c = linear_sum_assignment(cost)
    errors = int(cost[r, c].sum())
    den = sum(map(len, a))
    return errors, den, len(ref), len(hyp)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--seglst", type=Path, required=True)
    p.add_argument("--clips", default="")
    a = p.parse_args()
    clips = json.loads(a.manifest.read_text())
    wanted = [c for c in a.clips.split(",") if c] or [c["id"] for c in clips]
    by_id = {c["id"]: c for c in clips}
    groups = defaultdict(list)
    for seg in json.loads(a.seglst.read_text()):
        groups[str(seg["session_id"])].append(seg)
    tot_e = tot_d = 0
    for cid in wanted:
        e, d, nr, nh = cpwer(by_id[cid]["reference_words"], groups.get(cid, []))
        tot_e += e
        tot_d += d
        print(f"{cid} errors={e} ref_tokens={d} ref_spk={nr} hyp_spk={nh}")
    print(f"TOTAL errors={tot_e} ref_tokens={tot_d} cpwer={tot_e / tot_d:.6f}")


if __name__ == "__main__":
    main()
