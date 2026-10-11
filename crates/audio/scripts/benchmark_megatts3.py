#!/usr/bin/env python3
"""Benchmark MegaTTS3 voice cloning, one fresh process per sample.

Build the runtime example first:

    cargo build --release -p turbospark-runtime --example megatts3_bench

Each sample runs `megatts3_bench` under `/usr/bin/time -l`, discards one
in-process warmup request (Metal pipeline compilation, first-touch costs),
then times one cloning request: reference preparation, G2P, durations, the
diffusion steps and the WavVAE decode. Peak memory is the process
`phys_footprint`, which includes model open and the warmup, as a production
session would, and is what runtime admission reads.

A scenario is NAME=STEPS:DEVICE (device metal or cpu). All scenarios share
one reference WAV, one text and one seed. Every run visits every scenario,
alternating the order, so thermal and background drift hit each arm alike.
The output is deterministic for a fixed seed and device: two samples of one
scenario that write different WAV bytes are an error, not a data point.

Each sample also records what runtime admission would have checked before
the request (footprint at that moment plus the request's activation reserve)
and whether the measured peak stayed within it.

The report records host, power source, load, commit, and checkpoint hashes,
and each arm's spread ((max - min) / median of total_ms). A tight spread on
repeated identical runs is what licenses comparing arms; the load average
alone does not. A host that is not quiet is refused unless `--allow-busy` is
given, and timings from a busy host are not benchmark rows.
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
import tempfile
from datetime import datetime, timezone
from pathlib import Path

PEAK = re.compile(r"^\s*(\d+)\s+peak memory footprint$", re.MULTILINE)

# Fixed text: the protocol pins these bytes, so a change is a new experiment.
TEXT = "Hello world, this is a short test."

DEFAULT_SCENARIOS = ("metal-8=8:metal", "metal-16=16:metal", "metal-32=32:metal")


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
        steps, device = rest.split(":", 1)
        if device not in ("metal", "cpu"):
            raise ValueError
        return {"name": name, "steps": int(steps), "device": device}
    except ValueError:
        raise argparse.ArgumentTypeError(f"scenario must be NAME=STEPS:metal|cpu, got {text!r}")


def run_sample(binary, model_dir, reference, text, seed, scenario, workdir):
    wav_path = workdir / f"{scenario['name']}.wav"
    jobs = workdir / f"{scenario['name']}.json"
    jobs.write_text(json.dumps([{
        "reference": str(reference), "text": text, "steps": scenario["steps"],
        "seed": seed, "output": str(wav_path),
    }]))
    command = [
        "/usr/bin/time", "-l", str(binary), str(model_dir), "--jobs", str(jobs),
        "--device", scenario["device"], "--warmup",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"{scenario['name']} exited {result.returncode}:\n{result.stderr}")
    peak = PEAK.search(result.stderr)
    lines = [line for line in result.stdout.splitlines() if line.startswith("{")]
    opened = [line for line in result.stderr.splitlines() if line.startswith("{") and '"open_ms"' in line]
    if not peak or len(lines) != 1 or len(opened) != 1:
        raise RuntimeError(f"incomplete output from {scenario['name']}:\n{result.stdout}{result.stderr}")
    sample = json.loads(lines[0])
    opened = json.loads(opened[0])
    sample["open_ms"] = opened["open_ms"]
    sample["footprint_after_open_bytes"] = opened["footprint_after_open_bytes"]
    sample["peak_phys_footprint_bytes"] = int(peak.group(1))
    admission = sample.get("admission_bytes")
    sample["peak_within_admission"] = admission is not None and sample["peak_phys_footprint_bytes"] <= admission
    sample["wav_sha256"] = sha256_file(wav_path)
    return sample


def summarize(samples, field):
    values = [field(sample) for sample in samples]
    median = statistics.median(values)
    return {"median": median, "min": min(values), "max": max(values),
            "spread": (max(values) - min(values)) / median if median else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_dir", type=Path, help="converted or installed MegaTTS3 directory")
    parser.add_argument("--reference", type=Path, required=True, help="reference voice WAV")
    parser.add_argument("--text", default=TEXT, help="text to speak (default: the pinned sentence)")
    parser.add_argument("--binary", type=Path, default=Path("target/release/examples/megatts3_bench"))
    parser.add_argument("--scenario", action="append", type=parse_scenario,
                        help=f"NAME=STEPS:metal|cpu (repeatable; default {' '.join(DEFAULT_SCENARIOS)})")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-load", type=float, default=2.0, help="refuse above this 1-minute load average")
    parser.add_argument("--allow-busy", action="store_true", help="run on a busy or non-AC host (recorded in the report)")
    parser.add_argument("--skip-checkpoint-hash", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if platform.system() != "Darwin":
        parser.error("phys_footprint sampling and Metal require macOS")
    if args.runs < 1:
        parser.error("--runs must be positive")
    scenarios = args.scenario or [parse_scenario(s) for s in DEFAULT_SCENARIOS]
    model_dir = args.model_dir.resolve(strict=True)
    reference = args.reference.resolve(strict=True)
    binary = args.binary.resolve(strict=True)

    load = os.getloadavg()[0]
    power = power_source()
    host_quiet = power == "AC" and load <= args.max_load
    print(f"power={power} load_1m={load:.2f} quiet={host_quiet}", file=sys.stderr)
    if not host_quiet and not args.allow_busy:
        parser.error("host is not quiet (need AC power and low load); pass --allow-busy to record anyway")

    samples = []
    with tempfile.TemporaryDirectory(prefix="megatts3-bench-") as scratch:
        workdir = Path(scratch)
        for run in range(args.runs):
            order = scenarios if run % 2 == 0 else list(reversed(scenarios))
            for scenario in order:
                sample = run_sample(binary, model_dir, reference, args.text, args.seed, scenario, workdir)
                sample.update(scenario=scenario["name"], run=run + 1, load_1m_after=os.getloadavg()[0])
                samples.append(sample)
                print(
                    f"scenario={scenario['name']} run={run + 1} total_ms={sample['total_ms']:.0f} "
                    f"rtf={sample['rtf']:.2f} peak_gib={sample['peak_phys_footprint_bytes'] / 2**30:.2f}",
                    file=sys.stderr,
                )

    # Output is deterministic for a fixed seed and device; any difference is a defect.
    for scenario in scenarios:
        hashes = {s["wav_sha256"] for s in samples if s["scenario"] == scenario["name"]}
        if len(hashes) != 1:
            raise RuntimeError(f"scenario {scenario['name']} produced {len(hashes)} distinct WAV outputs")

    report = {
        "protocol": "megatts3-warm-interleaved-v1",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": {
            "platform": platform.platform(),
            "chip": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip(),
            "power": power,
            "load_1m": load,
            "quiet": host_quiet,
        },
        "git": git_state(),
        "request": {"reference": str(reference), "reference_sha256": sha256_file(reference),
                    "text": args.text, "seed": args.seed},
        "checkpoint": {
            "dir": str(model_dir),
            "sha256": None if args.skip_checkpoint_hash else {
                str(p.relative_to(model_dir)): sha256_file(p) for p in sorted(model_dir.glob("*/*.safetensors"))
            },
        },
        "scenarios": scenarios,
        "samples": samples,
        "summary": {
            scenario["name"]: {
                "total_ms": summarize(arm, lambda s: s["total_ms"]),
                "front_ms": summarize(arm, lambda s: s["front_ms"]),
                "rtf": summarize(arm, lambda s: s["rtf"]),
                "open_ms": summarize(arm, lambda s: s["open_ms"]),
                "peak_phys_footprint_bytes": summarize(arm, lambda s: s["peak_phys_footprint_bytes"]),
                "footprint_after_open_bytes": summarize(arm, lambda s: s["footprint_after_open_bytes"]),
                "admission_bytes": summarize(arm, lambda s: s["admission_bytes"]),
                "peak_within_admission": all(s["peak_within_admission"] for s in arm),
                "audio_seconds": arm[0]["audio_seconds"],
                "wav_sha256": arm[0]["wav_sha256"],
            }
            for scenario in scenarios
            for arm in [[s for s in samples if s["scenario"] == scenario["name"]]]
        },
    }
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded)
    else:
        sys.stdout.write(encoded)


if __name__ == "__main__":
    main()
