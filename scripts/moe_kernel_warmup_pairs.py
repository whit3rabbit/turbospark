#!/usr/bin/env python3
"""Alternate compile-only preparation off/on, charging it to total startup."""

import argparse
import json
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import time

from moe_startup_pairs import ARMS, compare_tokens, digest, snapshot, validate_rows


def activity(owned_pid=None):
    result = snapshot(["ps", "-Ao", "pid=,ppid=,pcpu=,comm="])
    if result["returncode"]:
        raise RuntimeError("cannot inspect process activity")
    rows = []
    for line in result["stdout"].splitlines():
        parts = line.strip().split(None, 3)
        if len(parts) == 4:
            rows.append(dict(pid=int(parts[0]), ppid=int(parts[1]), cpu=float(parts[2]), command=parts[3]))
    owned = {os.getpid()}
    if owned_pid is not None:
        owned.add(owned_pid)
    while True:
        children = {row["pid"] for row in rows if row["ppid"] in owned}
        if children <= owned:
            break
        owned.update(children)
    return sorted((r for r in rows if r["pid"] not in owned), key=lambda r: r["cpu"], reverse=True)[:12]


def warmup_coverage(rows, enabled):
    for row in rows:
        if row.get("kernel_warmup_requested") != enabled:
            raise ValueError("warmup request does not match experiment mode")
        if enabled:
            if row["startup"]["kernel_warmup_unique_keys"] <= 0:
                raise ValueError("warmup prepared no keys")
            for key in ["pipeline_creations", "function_specializations", "library_compiles", "library_loads"]:
                if row["compiler_after_request"][key] != row["compiler_at_open"][key]:
                    raise ValueError(f"runtime compilation remained after warmup: {key}")
        elif row["startup"]["kernel_warmup_unique_keys"]:
            raise ValueError("warmup ran in the off arm")


def compare_pair(off, on):
    compare_tokens([*off, *on])
    if len(off) != len(on) or [r["request_index"] for r in off] != [r["request_index"] for r in on]:
        raise ValueError("incomplete or mismatched request sequence")
    if any(not isinstance(rows[0]["load_to_first_token_ms"], (int, float)) or
           not 0 < rows[0]["load_to_first_token_ms"] < float("inf") for rows in [off, on]):
        raise ValueError("first-request total startup is missing or invalid")
    delta = on[0]["load_to_first_token_ms"] - off[0]["load_to_first_token_ms"]
    return dict(total_startup_delta_ms=delta, total_startup_delta_percent=100 * delta / off[0]["load_to_first_token_ms"])


def run(args, out, cache, enabled, tag):
    env = {k: v for k, v in os.environ.items() if not k.startswith("TURBOSPARK_")}
    ir, archive = ARMS[args.cache]
    env.update(TURBOSPARK_METAL_PRECOMPILED=str(int(ir)), TURBOSPARK_METAL_PIPELINE_CACHE=str(int(archive)),
               TURBOSPARK_METAL_KERNEL_WARMUP=str(int(enabled)), TURBOSPARK_METAL_CACHE_DIR=str(cache))
    command = ["/usr/bin/time", "-l", str(args.binary), "--model", str(args.model), "--case", args.case,
               "--context", str(args.context), "--max-new", str(args.max_new), "--slots", "16",
               "--repeats", "2", "--shaping", args.shaping, "--page-cache-label", "unknown"]
    before = dict(power=snapshot(["pmset", "-g", "batt"]), thermal=snapshot(["pmset", "-g", "therm"]), activity=activity())
    started = time.monotonic()
    samples = []
    timed_out = False
    process = None
    failure = None
    try:
        with (out / f"{tag}.jsonl").open("w") as stdout, (out / f"{tag}.stderr").open("w") as stderr:
            process = subprocess.Popen(command, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
            while process.poll() is None:
                samples.append(dict(elapsed_s=time.monotonic() - started, activity=activity(process.pid)))
                if time.monotonic() - started > args.timeout_seconds:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    timed_out = True
                    break
                time.sleep(1)
    except BaseException as error:
        failure = str(error) or type(error).__name__
        raise
    finally:
        # A failed monitor or interrupt must not leave the measured process
        # consuming the host after this experiment has stopped recording it.
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
        if failure is not None:
            (out / f"{tag}.metadata.json").write_text(json.dumps(dict(
                command=command, before=before, periodic_activity=samples,
                failure=failure, timeout=timed_out,
                exit_code=process.returncode if process is not None else None,
                wall_seconds=time.monotonic() - started), indent=2) + "\n")
    after = dict(power=snapshot(["pmset", "-g", "batt"]), thermal=snapshot(["pmset", "-g", "therm"]), activity=activity())
    all_activity = [before["activity"], after["activity"], *[s["activity"] for s in samples]]
    contended = any(r["cpu"] >= 100 for sample in all_activity for r in sample)
    ac = all("'AC Power'" in snap["power"]["stdout"] for snap in [before, after])
    thermal_ok = all(snap["thermal"]["returncode"] == 0 and not re.search(r"(?:CPU|GPU)_Speed_Limit\s*=\s*(?!100\b)\d+", snap["thermal"]["stdout"]) for snap in [before, after])
    error_text = (out / f"{tag}.stderr").read_text()
    match = re.search(r"(\d+)\s+peak memory footprint", error_text)
    record = dict(command=command, environment={k: v for k, v in env.items() if k.startswith("TURBOSPARK_")},
                  before=before, after=after, periodic_activity=samples, contended=contended, ac_power=ac,
                  thermal_ok=thermal_ok, timeout=timed_out, exit_code=process.returncode,
                  wall_seconds=time.monotonic() - started,
                  whole_process_peak_bytes=int(match[1]) if match else None)
    (out / f"{tag}.metadata.json").write_text(json.dumps(record, indent=2) + "\n")
    if timed_out or process.returncode:
        raise RuntimeError(f"{tag}: probe failed or exceeded time limit; partial evidence retained")
    rows = [json.loads(line) for line in (out / f"{tag}.jsonl").read_text().splitlines()]
    if [row["request_index"] for row in rows] != [0, 1]:
        raise ValueError("probe did not produce both requested fresh/reset records")
    # Archive population is allowed to miss; measured archive processes must hit.
    if not tag.startswith("population"):
        validate_rows(rows, args.cache)
    warmup_coverage(rows, enabled)
    return rows, record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache", choices=ARMS, default="source")
    parser.add_argument("--case", default="short-explanation")
    parser.add_argument("--shaping", choices=["greedy", "sampled"], default="greedy")
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--max-new", type=int, default=64)
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=float, default=300)
    args = parser.parse_args()
    if args.pairs < 1 or args.context < 1 or args.max_new < 1 or args.timeout_seconds <= 0:
        parser.error("pair count, context, token budget and timeout must be positive")
    args.binary = args.binary.resolve()
    args.model = args.model.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    cache = (args.out / "cache").resolve()
    identity = dict(binary_sha256=digest(args.binary), manifest_sha256=digest(args.model / "manifest.json"),
                    git=snapshot(["git", "status", "--short"]), revision=snapshot(["git", "rev-parse", "HEAD"]),
                    platform=snapshot(["sw_vers"]), chip=snapshot(["sysctl", "-n", "machdep.cpu.brand_string"]),
                    cache=args.cache, case=args.case, shaping=args.shaping, context=args.context, max_new=args.max_new,
                    os_pages_and_apple_compiler_cache="unverified, previously exercised")
    (args.out / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    if ARMS[args.cache][1]:
        run(args, args.out, cache, True, "population-on")
    pairs = []
    records = []
    for index in range(args.pairs):
        arms = [False, True] if index % 2 == 0 else [True, False]
        rows = {}
        metadata = {}
        for enabled in arms:
            tag = f"pair-{index}-{'on' if enabled else 'off'}"
            rows[enabled], metadata[enabled] = run(args, args.out, cache, enabled, tag)
            records.extend(rows[enabled])
        eligible = all(m["ac_power"] and m["thermal_ok"] and not m["contended"] for m in metadata.values())
        pair = dict(index=index, **compare_pair(rows[False], rows[True]), eligible=eligible,
                    off_startup_ms=rows[False][0]["load_to_first_token_ms"], on_startup_ms=rows[True][0]["load_to_first_token_ms"],
                    warmup_ms=rows[True][0]["startup"]["kernel_warmup_ms"],
                    off_peak_bytes=metadata[False]["whole_process_peak_bytes"], on_peak_bytes=metadata[True]["whole_process_peak_bytes"])
        pairs.append(pair)
        print(json.dumps(pair), flush=True)
    compare_tokens(records)
    if digest(args.binary) != identity["binary_sha256"] or digest(args.model / "manifest.json") != identity["manifest_sha256"]:
        raise ValueError("binary or model manifest changed during experiment")
    qualified = [p for p in pairs if p["eligible"]]
    summary = dict(protocol="moe-kernel-warmup-v1", identity=identity, pairs=pairs,
                   exploratory_median_delta_ms=statistics.median(p["total_startup_delta_ms"] for p in pairs),
                   exploratory_median_delta_percent=statistics.median(p["total_startup_delta_percent"] for p in pairs),
                   eligible_pairs=len(qualified), required_eligible_pairs=3,
                   default_promotion_supported=False,
                   limitations=["OS pages and Apple compiler cache are unverified", "GPU isolation is not continuously verified", "Startup pairs do not establish quality or release readiness"])
    if qualified:
        summary["eligible_median_delta_ms"] = statistics.median(p["total_startup_delta_ms"] for p in qualified)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
