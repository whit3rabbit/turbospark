#!/usr/bin/env python3
"""Compare first requests in fresh processes without hiding startup in warmup.

Archive population is recorded separately. This controls application caches;
it neither clears nor verifies the OS checkpoint page cache.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time


ARMS = {
    "source": (False, False),
    "ir": (True, False),
    "archive": (False, True),
    "ir-archive": (True, True),
}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def host_activity():
    result = subprocess.run(
        ["ps", "-Ao", "pid=,pcpu=,comm="], capture_output=True, text=True, check=True
    )
    rows = []
    for line in result.stdout.splitlines():
        values = line.strip().split(None, 2)
        if len(values) != 3:
            continue
        pid, cpu, command = values
        if int(pid) == os.getpid():
            continue
        rows.append({"pid": int(pid), "cpu_percent": float(cpu), "command": command})
    return sorted(rows, key=lambda row: row["cpu_percent"], reverse=True)[:10]


def snapshot(command, cwd=None):
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    return {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def validate_rows(rows, arm):
    if not rows:
        raise ValueError("probe returned no records")
    precompiled, archive = ARMS[arm]
    for row in rows:
        if row.get("protocol") != "moe-startup-v1" or row.get("discarded_warmups") != 0:
            raise ValueError("startup comparison requires unwarmed startup-probe records")
        if row.get("first_token_ms") is None or row.get("new_tokens", 0) == 0:
            raise ValueError("probe never reached its first usable token")
        if row.get("precompiled_requested") != precompiled:
            raise ValueError("precompiled request does not match the experiment arm")
        if row.get("pipeline_cache_requested") != archive:
            raise ValueError("pipeline-cache request does not match the experiment arm")
        counters = row["compiler_after_request"]
        if precompiled and counters["library_loads"] == 0:
            raise ValueError("precompiled arm silently fell back to source compilation")
        if precompiled and counters["library_compiles"] != 0:
            raise ValueError("precompiled arm partially fell back to source compilation")
        if archive and not counters["cache_enabled"]:
            raise ValueError("pipeline cache is unavailable on this host")
        if archive and row["request_index"] == 0 and counters["archive_hits"] == 0:
            raise ValueError("populated archive arm had no verified Metal archive hits")
        if archive and (counters["archive_misses"] or counters["archive_errors"]):
            raise ValueError("populated archive arm required compilation or cache recovery")


def compare_tokens(records):
    if not records:
        raise ValueError("no measured records")
    first = records[0]
    contract = ("manifest_sha256", "case", "context", "max_new", "expert_cache_slots", "residency", "shaping", "seed")
    expected = tuple(first[key] for key in contract)
    for row in records:
        if tuple(row[key] for key in contract) != expected:
            raise ValueError("records have different model or generation contracts")
        if (row["token_sha256"], row["new_tokens"], row["stop"]) != (
            first["token_sha256"], first["new_tokens"], first["stop"]
        ):
            raise ValueError("generated token digest/count/stop changed between arms")


def run_probe(args, arm, cache_dir, out, tag, population=False):
    precompiled, archive = ARMS[arm]
    # Inherited diagnostic seams can change arithmetic, residency, or timings.
    env = {key: value for key, value in os.environ.items() if not key.startswith("TURBOSPARK_")}
    env.update({
        "TURBOSPARK_METAL_PRECOMPILED": "1" if precompiled else "0",
        "TURBOSPARK_METAL_PIPELINE_CACHE": "1" if archive else "0",
        "TURBOSPARK_METAL_CACHE_DIR": str(cache_dir),
    })
    if args.io_diagnostics:
        env.update(TURBOSPARK_EXPERT_DISK_IO="1", TURBOSPARK_PHASES="1")
    command = [
        str(args.binary), "--model", str(args.model), "--case", args.case,
        "--max-new", str(args.max_new), "--context", str(args.context),
        "--slots", "16", "--residency", args.residency,
        "--shaping", args.shaping, "--repeats", str(args.repeats),
        "--reopens", "1", "--label", arm,
        "--page-cache-label", args.page_cache_label,
    ]
    if args.expert_trace:
        command += ["--expert-trace", str(args.expert_trace), "--prefetch-bytes", str(args.prefetch_bytes)]
    activity = host_activity()
    contended = any(row["cpu_percent"] >= 100 for row in activity)
    if contended and not args.allow_contended:
        raise ValueError("host has a process using >=100% CPU; retry quietly or use --allow-contended for smoke evidence")
    power_before = snapshot(["pmset", "-g", "batt"])
    thermal_before = snapshot(["pmset", "-g", "therm"])
    started = time.monotonic()
    try:
        result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=args.timeout_seconds)
    except subprocess.TimeoutExpired as error:
        # Retain partial evidence rather than leave a mapped or contended run unbounded.
        def decoded(value):
            return value.decode(errors="replace") if isinstance(value, bytes) else (value or "")
        (out / f"{tag}.stdout.jsonl").write_text(decoded(error.stdout))
        (out / f"{tag}.stderr.txt").write_text(decoded(error.stderr))
        (out / f"{tag}.launch.json").write_text(json.dumps({
            "command": command, "application_cache_dir": str(cache_dir),
            "archive_population": population, "host_activity_before": activity,
            "power_before": power_before, "thermal_before": thermal_before,
            "timed_out": True, "timeout_seconds": args.timeout_seconds,
        }, indent=2) + "\n")
        raise ValueError(f"{tag} exceeded {args.timeout_seconds}s; partial evidence retained in {out}") from error
    wall_seconds = time.monotonic() - started
    activity_after = host_activity()
    contended = contended or any(row["cpu_percent"] >= 100 for row in activity_after)
    (out / f"{tag}.stdout.jsonl").write_text(result.stdout)
    (out / f"{tag}.stderr.txt").write_text(result.stderr)
    (out / f"{tag}.launch.json").write_text(json.dumps({
        "command": command, "application_cache_dir": str(cache_dir),
        "archive_population": population, "host_activity_before": activity,
        "host_activity_after": activity_after,
        "power_before": power_before, "thermal_before": thermal_before,
        "thermal_after": snapshot(["pmset", "-g", "therm"]),
        "contended": contended, "wall_seconds": wall_seconds,
        "returncode": result.returncode,
    }, indent=2) + "\n")
    if result.returncode:
        raise ValueError(f"{tag} failed ({result.returncode}); inspect {out / (tag + '.stderr.txt')}")
    rows = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    if not population:
        validate_rows(rows, arm)
    return [{**row, "arm": arm, "contended": contended, "capture": tag} for row in rows]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arms", default="source,ir,archive,ir-archive")
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--case", default="short-explanation")
    parser.add_argument("--max-new", type=int, default=64)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--residency", choices=("streamed", "mapped"), default="streamed")
    parser.add_argument("--shaping", choices=("greedy", "sampled"), default="greedy")
    parser.add_argument("--page-cache-label", default="unknown")
    parser.add_argument("--expert-trace", type=Path)
    parser.add_argument("--prefetch-bytes", type=int, default=0)
    parser.add_argument("--io-diagnostics", action="store_true")
    parser.add_argument("--allow-contended", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=300)
    args = parser.parse_args()
    arms = args.arms.split(",")
    if args.pairs < 1 or len(set(arms)) != len(arms) or any(arm not in ARMS for arm in arms):
        parser.error("use positive --pairs and unique arms from source,ir,archive,ir-archive")
    if bool(args.expert_trace) != (args.prefetch_bytes > 0):
        parser.error("--expert-trace and positive --prefetch-bytes are required together")
    if not 0 < args.timeout_seconds <= 3600:
        parser.error("--timeout-seconds must be positive and at most 3600")
    args.binary = args.binary.expanduser().resolve(strict=True)
    args.model = args.model.expanduser().resolve(strict=True)
    if args.expert_trace:
        args.expert_trace = args.expert_trace.expanduser().resolve(strict=True)
    out = args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=False)
    cache_root = out / "application-cache"
    cache_root.mkdir()
    metadata = {
        "protocol": "moe-startup-v1", "binary_sha256": digest(args.binary),
        "manifest_sha256": digest(args.model / "manifest.json"),
        "platform": platform.platform(), "machine": platform.machine(),
        "os_page_cache_controlled": False, "archive_population_is_measured": False,
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "git_revision": snapshot(["git", "rev-parse", "HEAD"], Path(__file__).resolve().parents[1]),
        "git_status": snapshot(["git", "status", "--short"], Path(__file__).resolve().parents[1]),
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    measured = []
    population_rows = []
    for arm in arms:
        if ARMS[arm][1]:
            population_rows += run_probe(args, arm, cache_root / arm, out, f"populate-{arm}", population=True)
    for pair in range(args.pairs):
        order = arms if pair % 2 == 0 else list(reversed(arms))
        for arm in order:
            rows = run_probe(args, arm, cache_root / arm, out, f"pair-{pair}-{arm}")
            measured += [{**row, "pair_index": pair} for row in rows]
    compare_tokens(population_rows + measured)
    first_requests = [row for row in measured if row["request_index"] == 0]
    summary = {
        "tokens_identical": True,
        "quiet_at_launch_and_exit_pair_count_met": args.pairs >= 3 and not args.allow_contended and not args.io_diagnostics and not any(row["contended"] for row in measured),
        "default_promotion_supported": False,
        "os_page_cache_verified": False,
        "arms": {
            arm: {
                "first_requests": len([row for row in first_requests if row["arm"] == arm]),
                "median_load_to_first_token_ms": statistics.median(row["load_to_first_token_ms"] for row in first_requests if row["arm"] == arm),
                "load_to_first_token_ms": [row["load_to_first_token_ms"] for row in first_requests if row["arm"] == arm],
                "paired_delta_from_source_ms": [
                    row["load_to_first_token_ms"] - source["load_to_first_token_ms"]
                    for row in first_requests if row["arm"] == arm
                    for source in first_requests if source["arm"] == "source" and source["pair_index"] == row["pair_index"]
                ],
            } for arm in arms
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"output": str(out), **summary}, indent=2))


if __name__ == "__main__":
    main()
