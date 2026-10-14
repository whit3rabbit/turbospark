#!/usr/bin/env python3
"""Warm, interleaved benchmark of the Core ML pipelines on macOS.

Build the probe first:

    cargo build --release -p turbospark-runtime --example coreml_backend_probe

Each sample is a fresh process on one arm: open, one discarded warmup, one
measured pass. Peak `phys_footprint` comes from `/usr/bin/time -l` and so
includes open and warmup, as a real session would. Arm order rotates every
run. Every sample records the 1-minute load average and the power source,
because a busy host or battery power changes the numbers; a run is only
quotable when both are clean (see docs/BENCHMARKING.md, Measurement hygiene).

    python3 crates/runtime/scripts/benchmark_coreml.py parakeet \\
        --coreml ~/models/coreml/parakeet-redux-coreml --cpu ~/models/parakeet-redux \\
        --wav clip.wav --runs 3 --output /tmp/parakeet.json
    python3 crates/runtime/scripts/benchmark_coreml.py embed \\
        --model ~/models/coreml/embeddinggemma-2-coreml --runs 3
    python3 crates/runtime/scripts/benchmark_coreml.py ocr \\
        --model ~/models/coreml/unlimited-ocr-coreml \\
        --pages crates/runtime/testdata/coreml_ocr/page1.png --runs 3

Output agreement across arms is reported (`digests_agree`), not enforced:
placements differ in the last bits, and the CPU Redux port is a different
implementation.
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

METRIC = re.compile(r"^coreml_probe (.*)$", re.MULTILINE)
PEAK = re.compile(r"^\s*(\d+)\s+peak memory footprint$", re.MULTILINE)
DIGEST = re.compile(r"^DIGEST: ([0-9a-f]+)$", re.MULTILINE)
BINARY = Path("target/release/examples/coreml_backend_probe")


def power_source():
    out = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout
    return "ac" if "AC Power" in out else "battery"


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def probe(args):
    load = os.getloadavg()[0]
    result = subprocess.run(["/usr/bin/time", "-l", str(BINARY), *map(str, args)],
                            capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"probe {args} exited {result.returncode}:\n{result.stdout}{result.stderr}")
    metric, peak, dig = METRIC.search(result.stderr), PEAK.search(result.stderr), DIGEST.search(result.stdout)
    if not (metric and peak and dig):
        raise RuntimeError(f"probe {args} output is incomplete:\n{result.stdout}{result.stderr}")
    sample = {}
    for pair in metric.group(1).split():
        key, value = pair.split("=", 1)
        try:
            sample[key] = float(value)
        except ValueError:
            sample[key] = value
    sample["peak_phys_footprint_bytes"] = int(peak.group(1))
    sample["digest"] = dig.group(1)
    sample["load_avg_1m_before"] = round(load, 2)
    sample["power"] = power_source()
    return sample


def summarize(samples, fields):
    out = {}
    for field in fields:
        values = [s[field] for s in samples if isinstance(s.get(field), float) or isinstance(s.get(field), int)]
        if values:
            out[field] = {"median": statistics.median(values), "min": min(values), "max": max(values)}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("task", choices=["parakeet", "embed", "ocr"])
    p.add_argument("--coreml", type=Path, help="parakeet: Core ML export directory")
    p.add_argument("--cpu", type=Path, help="parakeet: moondream/parakeet-redux directory (optional arm)")
    p.add_argument("--wav", type=Path)
    p.add_argument("--model", type=Path, help="embed / ocr: model directory")
    p.add_argument("--pages", type=Path, nargs="+")
    p.add_argument("--arms", nargs="+", help="subset of arms to run")
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    if platform.system() != "Darwin":
        p.error("Core ML and phys_footprint sampling require macOS")
    if not BINARY.is_file():
        p.error(f"{BINARY} is missing; build the example first")

    if a.task == "parakeet":
        if not (a.coreml and a.wav):
            p.error("parakeet needs --coreml and --wav")
        arms = {"coreml_gpu": a.coreml, "coreml_ane": a.coreml}
        if a.cpu:
            arms["cpu_redux"] = a.cpu
        command = lambda arm: ["parakeet", arms[arm], a.wav, arm]
        fields = ["measured_ms", "rtfx", "open_ms", "warmup_ms", "peak_phys_footprint_bytes"]
        inputs = {"wav_sha256": sha(a.wav)}
    elif a.task == "embed":
        if not a.model:
            p.error("embed needs --model")
        arms = {"ane": a.model, "gpu": a.model, "cpu": a.model}
        command = lambda arm: ["embed", a.model, arm]
        fields = ["measured_ms", "texts_per_s", "open_ms", "warmup_ms", "peak_phys_footprint_bytes"]
        inputs = {}
    else:
        if not (a.model and a.pages):
            p.error("ocr needs --model and --pages")
        arms = {"gpu": a.model}
        command = lambda arm: ["ocr", a.model, *a.pages]
        fields = ["measured_ms", "decode_tok_s", "encode_ms", "prefill_ms", "decode_ms",
                  "open_ms", "peak_phys_footprint_bytes"]
        inputs = {"pages_sha256": [sha(x) for x in a.pages]}
    names = [n for n in arms if not a.arms or n in a.arms]

    samples = []
    for run in range(a.runs):
        order = names[run % len(names):] + names[:run % len(names)]
        for arm in order:
            s = probe(command(arm))
            s["run"] = run + 1
            samples.append(s)
            print(f"run={run + 1} arm={arm} measured_ms={s['measured_ms']:.1f} "
                  f"peak_mib={s['peak_phys_footprint_bytes'] / 1048576:.0f} "
                  f"load={s['load_avg_1m_before']} power={s['power']}", file=sys.stderr)
    by_arm = {n: [s for s in samples if s["arm"] == n] for n in names}
    report = {
        "protocol": f"coreml-{a.task}-warm-interleaved-v1",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.platform(),
        "inputs": inputs,
        "digests_agree": len({s["digest"] for s in samples}) == 1,
        "digests_by_arm": {n: sorted({s["digest"] for s in v}) for n, v in by_arm.items()},
        "quiet_host": all(s["load_avg_1m_before"] < 4.0 and s["power"] == "ac" for s in samples),
        "samples": samples,
        "summary": {n: summarize(v, fields) for n, v in by_arm.items()},
    }
    text = json.dumps(report, indent=2) + "\n"
    if a.output:
        a.output.write_text(text)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
