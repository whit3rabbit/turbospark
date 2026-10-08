#!/usr/bin/env python3
"""Benchmark MiniMax Music 3 generation on Metal, one fresh process per sample.

Build the runtime example first:

    cargo build --release -p turbospark-runtime --example music3_bench

Each sample runs `music3_bench` under `/usr/bin/time -l`, discards one
in-process warmup request (Metal pipeline compilation, first-touch costs),
then times a single request. Peak memory is the process `phys_footprint`,
which includes model open and the warmup, as a production session would.

Variants (`--binary LABEL=PATH`, repeatable) are interleaved per run so
thermal and background drift hits every arm alike. Every sample of one
scenario must produce byte-identical WAV output: a change in output hash
within or across variants is an error, not a data point. Pass
`--expect-wav-sha256 SCENARIO=HEX` to pin the hash to a recorded baseline.

The report records host, power source, load, commit, and checkpoint hashes.
A host that is not quiet is recorded, and refused unless `--allow-busy` is
given. Timings from a busy host are not benchmark rows.
"""

import argparse
import hashlib
import json
import os
import platform
import re
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PEAK = re.compile(r"^\s*(\d+)\s+peak memory footprint$", re.MULTILINE)

# Fixed prompt: the benchmark protocol pins these bytes, so a change is a
# new experiment.
CAPTION = "upbeat acoustic folk with warm guitar"
LYRICS = "[instrumental]"

DEFAULT_SCENARIOS = ("smoke=1:2", "short=8:1")


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def power_source():
    result = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True)
    first = result.stdout.splitlines()[0] if result.stdout else ""
    return "AC" if "AC Power" in first else ("battery" if "Battery" in first else "unknown")


def git_state():
    def run(*args):
        result = subprocess.run(["git", *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None

    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def parse_scenario(text):
    try:
        name, rest = text.split("=", 1)
        duration, steps = rest.split(":", 1)
        return {"name": name, "duration": float(duration), "steps": int(steps)}
    except ValueError:
        raise argparse.ArgumentTypeError(f"scenario must be NAME=SECONDS:STEPS, got {text!r}")


def parse_variant(text):
    label, sep, path = text.partition("=")
    if not sep:
        label, path = Path(text).stem, text
    return {"label": label, "binary": Path(path).resolve(strict=True)}


def run_sample(variant, model_dir, scenario, seed, wav_path):
    command = [
        "/usr/bin/time", "-l", str(variant["binary"]), str(model_dir),
        "--caption", CAPTION, "--lyrics", LYRICS,
        "--duration", str(scenario["duration"]), "--steps", str(scenario["steps"]),
        "--seed", str(seed), "--output", str(wav_path), "--warmup",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"{variant['label']} exited {result.returncode}:\n{result.stderr}")
    peak = PEAK.search(result.stderr)
    lines = [line for line in result.stdout.splitlines() if line.startswith("{")]
    if not peak or len(lines) != 1:
        raise RuntimeError(f"incomplete output from {variant['label']}:\n{result.stdout}{result.stderr}")
    sample = json.loads(lines[0])
    sample["peak_phys_footprint_bytes"] = int(peak.group(1))
    sample["wav_sha256"] = sha256_file(wav_path)
    return sample


def summarize(samples, field):
    values = [field(sample) for sample in samples]
    return {"median": statistics.median(values), "min": min(values), "max": max(values)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_dir", type=Path, help="converted MiniMax Music 3 checkpoint directory")
    parser.add_argument("--binary", action="append", type=parse_variant,
                        help="LABEL=PATH to a music3_bench build (repeatable; default target/release/examples/music3_bench)")
    parser.add_argument("--scenario", action="append", type=parse_scenario,
                        help=f"NAME=SECONDS:STEPS (repeatable; default {' '.join(DEFAULT_SCENARIOS)})")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--expect-wav-sha256", action="append", default=[], metavar="SCENARIO=HEX")
    parser.add_argument("--max-load", type=float, default=2.0, help="refuse above this 1-minute load average")
    parser.add_argument("--allow-busy", action="store_true", help="run on a busy or non-AC host (recorded in the report)")
    parser.add_argument("--skip-checkpoint-hash", action="store_true")
    parser.add_argument("--workdir", type=Path, default=Path("target/music3-bench"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if platform.system() != "Darwin":
        parser.error("phys_footprint sampling and Metal require macOS")
    if args.runs < 1:
        parser.error("--runs must be positive")
    variants = args.binary or [parse_variant("target/release/examples/music3_bench")]
    scenarios = args.scenario or [parse_scenario(s) for s in DEFAULT_SCENARIOS]
    model_dir = args.model_dir.resolve(strict=True)
    expected = dict(item.split("=", 1) for item in args.expect_wav_sha256)

    load = os.getloadavg()[0]
    power = power_source()
    host_quiet = power == "AC" and load <= args.max_load
    print(f"power={power} load_1m={load:.2f} quiet={host_quiet}", file=sys.stderr)
    if not host_quiet and not args.allow_busy:
        parser.error("host is not quiet (need AC power and low load); pass --allow-busy to record anyway")

    args.workdir.mkdir(parents=True, exist_ok=True)
    samples = []
    for scenario in scenarios:
        wav_path = args.workdir / f"{scenario['name']}.wav"
        for run in range(args.runs):
            order = variants if run % 2 == 0 else list(reversed(variants))
            for variant in order:
                sample = run_sample(variant, model_dir, scenario, args.seed, wav_path)
                sample.update(variant=variant["label"], scenario=scenario["name"], run=run + 1)
                samples.append(sample)
                print(
                    f"scenario={scenario['name']} variant={variant['label']} run={run + 1} "
                    f"generate_ms={sample['generate_ms']:.0f} rtf={sample['rtf']:.2f} "
                    f"peak_gib={sample['peak_phys_footprint_bytes'] / 2**30:.2f}",
                    file=sys.stderr,
                )

    # Output is deterministic for a fixed seed; any difference is a defect.
    for scenario in scenarios:
        hashes = {s["wav_sha256"] for s in samples if s["scenario"] == scenario["name"]}
        if len(hashes) != 1:
            raise RuntimeError(f"scenario {scenario['name']} produced {len(hashes)} distinct WAV outputs")
        want = expected.get(scenario["name"])
        if want and hashes != {want}:
            raise RuntimeError(f"scenario {scenario['name']} WAV {hashes.pop()} != expected {want}")

    report = {
        "protocol": "music3-metal-warm-interleaved-v1",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": {
            "platform": platform.platform(),
            "chip": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip(),
            "power": power,
            "load_1m": load,
            "quiet": host_quiet,
        },
        "git": git_state(),
        "prompt": {"caption": CAPTION, "lyrics": LYRICS, "seed": args.seed},
        "checkpoint": {
            "dir": str(model_dir),
            "sha256": None if args.skip_checkpoint_hash else {
                p.name: sha256_file(p) for p in sorted(model_dir.glob("*.safetensors"))
            },
        },
        "scenarios": scenarios,
        "samples": samples,
        "summary": {
            f"{scenario['name']}/{variant['label']}": {
                "generate_ms": summarize(arm, lambda s: s["generate_ms"]),
                "rtf": summarize(arm, lambda s: s["rtf"]),
                "open_ms": summarize(arm, lambda s: s["open_ms"]),
                "peak_phys_footprint_bytes": summarize(arm, lambda s: s["peak_phys_footprint_bytes"]),
                "stage_ms": {
                    key: summarize(arm, lambda s, key=key: s["stage_ms"][key])
                    for key in arm[0]["stage_ms"] if key != "chunks"
                },
                "wav_sha256": arm[0]["wav_sha256"],
            }
            for scenario in scenarios
            for variant in variants
            for arm in [[s for s in samples if s["scenario"] == scenario["name"] and s["variant"] == variant["label"]]]
        },
    }
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded)
    else:
        sys.stdout.write(encoded)


if __name__ == "__main__":
    main()
