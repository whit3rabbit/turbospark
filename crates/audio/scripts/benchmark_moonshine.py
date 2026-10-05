#!/usr/bin/env python3
"""Run warm, interleaved Moonshine CPU/Metal probes on macOS.

Build the uniquely named runtime example first. Each timed sample opens a
fresh process, warms the selected runner once, then transcribes the same WAV.
The peak includes open and warmup, as a production session would.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import sys
from datetime import datetime, timezone


METRIC = re.compile(
    r"^moonshine_(cpu|metal) open_ms=([0-9.]+) warmup_ms=([0-9.]+) "
    r"transcribe_ms=([0-9.]+) samples=(\d+)$",
    re.MULTILINE,
)
PEAK = re.compile(r"^\s*(\d+)\s+peak memory footprint$", re.MULTILINE)


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def probe(binary, model_dir, wav, backend):
    env = os.environ.copy()
    env["TURBOSPARK_MOONSHINE_DEVICE"] = backend
    env.pop("TURBOSPARK_MOONSHINE_PROFILE", None)
    command = ["/usr/bin/time", "-l", str(binary), str(model_dir), str(wav), "--warmup"]
    result = subprocess.run(command, env=env, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(
            f"{backend} probe exited {result.returncode}:\n{result.stdout}{result.stderr}"
        )
    metric = METRIC.search(result.stderr)
    peak = PEAK.search(result.stderr)
    transcript = result.stdout[6:].rstrip("\n") if result.stdout.startswith("TEXT: ") else None
    if not metric or metric.group(1) != backend or not peak or transcript is None:
        raise RuntimeError(f"{backend} probe output is incomplete:\n{result.stdout}{result.stderr}")
    return {
        "backend": backend,
        "open_ms": float(metric.group(2)),
        "warmup_ms": float(metric.group(3)),
        "transcribe_ms": float(metric.group(4)),
        "samples": int(metric.group(5)),
        "peak_phys_footprint_bytes": int(peak.group(1)),
        "transcript_sha256": hashlib.sha256(transcript.encode()).hexdigest(),
    }


def summarize(samples, field):
    values = [sample[field] for sample in samples]
    return {"median": statistics.median(values), "min": min(values), "max": max(values)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("wav", type=Path)
    parser.add_argument("--binary", type=Path, default=Path("target/release/examples/moonshine_backend_probe"))
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if platform.system() != "Darwin":
        parser.error("phys_footprint sampling requires macOS")
    if args.runs < 1:
        parser.error("--runs must be positive")
    binary = args.binary.resolve(strict=True)
    model_dir = args.model_dir.resolve(strict=True)
    wav = args.wav.resolve(strict=True)
    files = [model_dir / name for name in ("config.json", "model.safetensors", "tokenizer.json")]
    if not all(path.is_file() for path in files):
        parser.error("model directory needs config.json, model.safetensors, and tokenizer.json")

    samples = []
    for run in range(args.runs):
        order = ("cpu", "metal") if run % 2 == 0 else ("metal", "cpu")
        for backend in order:
            sample = probe(binary, model_dir, wav, backend)
            sample["run"] = run + 1
            samples.append(sample)
            print(
                f"run={run + 1} backend={backend} "
                f"transcribe_ms={sample['transcribe_ms']:.2f} "
                f"peak_mib={sample['peak_phys_footprint_bytes'] / 1048576:.1f}",
                file=sys.stderr,
            )
    if len({sample["transcript_sha256"] for sample in samples}) != 1:
        raise RuntimeError("CPU and Metal transcripts differ across interleaved runs")
    if len({sample["samples"] for sample in samples}) != 1:
        raise RuntimeError("CPU and Metal processed different sample counts")
    arms = {backend: [s for s in samples if s["backend"] == backend] for backend in ("cpu", "metal")}
    report = {
        "protocol": "moonshine-warm-interleaved-v1",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.platform(),
        "checkpoint_sha256": {path.name: digest(path) for path in files},
        "wav_sha256": digest(wav),
        "precompiled_metal": os.environ.get("TURBOSPARK_METAL_PRECOMPILED") == "1",
        "samples": samples,
        "summary": {
            backend: {
                "transcribe_ms": summarize(arm, "transcribe_ms"),
                "peak_phys_footprint_bytes": summarize(arm, "peak_phys_footprint_bytes"),
                "open_ms": summarize(arm, "open_ms"),
            }
            for backend, arm in arms.items()
        },
    }
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded)
    else:
        sys.stdout.write(encoded)


if __name__ == "__main__":
    main()
