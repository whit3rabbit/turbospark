#!/usr/bin/env python3
"""Interleaved A/B of music3_bench flag variants, one fresh process per sample.

    cargo build --release -p turbospark-runtime --example music3_bench
    python3 ab_music3_flow.py --model DIR --duration 8 --steps 30 --runs 2 \
        --variant exact= --variant reuse3=--flow-uncond-interval,3 --out-dir OUT

Variants are `LABEL=FLAG,FLAG,...` (empty flags are the exact baseline).
Samples are interleaved per run so drift hits every arm alike. With
`--flow-only` the AR stage is skipped and the flow stage gets synthetic
hiddens (a timing aid; the audio is not music). Without it, the full request
runs and each variant writes a WAV that `spectral_distance.py` can compare.
Host state is recorded; a busy host is recorded, not hidden: timings from it
are not benchmark rows.
"""
import argparse, json, os, subprocess, sys
from pathlib import Path

CAPTION = "upbeat acoustic folk with warm guitar"
LYRICS = "[instrumental]"


def host():
    batt = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout
    return {"power": "AC" if "AC Power" in batt.splitlines()[0] else "other",
            "loadavg": os.getloadavg()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--binary", default="target/release/examples/music3_bench")
    ap.add_argument("--duration", type=float, default=8)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--flow-only", action="store_true")
    ap.add_argument("--variant", action="append", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    variants = []
    for spec in args.variant:
        label, _, flags = spec.partition("=")
        variants.append((label, [f for f in flags.split(",") if f]))
    results = []
    print(json.dumps({"host": host()}), flush=True)
    for run in range(args.runs):
        for label, flags in variants:
            cmd = [args.binary, args.model, "--caption", CAPTION, "--lyrics", LYRICS,
                   "--output", str(out / f"{label}.wav"), "--duration", str(args.duration),
                   "--steps", str(args.steps), "--seed", str(args.seed), "--warmup"] + flags
            if args.flow_only:
                cmd.append("--flow-only")
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode:
                sys.exit(f"{label} failed: {proc.stderr[-2000:]}")
            row = json.loads(proc.stdout.strip().splitlines()[-1])
            row.pop("dispatch", None)
            row.update(label=label, run=run, host=host())
            results.append(row)
            print(json.dumps(row), flush=True)
    (out / "results.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
