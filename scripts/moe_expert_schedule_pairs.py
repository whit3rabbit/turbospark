#!/usr/bin/env python3
"""Pair fresh-process expert scheduling arms, preserving rejected evidence.

This measures startup and first/reset requests without discarded warmups.
OS page residency and GPU isolation remain unverified; no defaults are promoted.
"""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import re
import signal
import statistics
import subprocess
import time

from moe_startup_pairs import compare_tokens, digest, validate_rows


EXPERIMENTS = {
    "overlap": {"off": ("streamed", False, "off"), "on": ("streamed", True, "off")},
    "mapped": {"off": ("mapped", False, "off"), "advice": ("mapped", False, "advice"),
               "touch": ("mapped", False, "touch")},
}
PROTOCOL = "moe-expert-schedule-v1"
MODEL_PROCESS = re.compile(r"(?:mlx_lm|llama-server|mference|ollama|memory_oracle|quality_gate|turbospark-(?:server|bench|check|startup-probe))", re.I)


def snapshot(command, cwd=None):
    try:
        result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=5)
        return dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
    except (OSError, subprocess.TimeoutExpired) as error:
        return dict(returncode=None, stdout="", stderr=str(error))


def write_json(path, value):
    # An interrupted experiment should retain the last complete summary.
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def cpu_seconds(value):
    days, separator, clock = value.partition("-")
    if not separator:
        clock, days = days, "0"
    parts = [float(part) for part in clock.split(":")]
    if len(parts) not in (2, 3):
        raise ValueError("unrecognized ps CPU time")
    seconds = parts[-1] + 60 * parts[-2]
    if len(parts) == 3:
        seconds += 3600 * parts[0]
    return 86400 * int(days) + seconds


def parse_activity(output, owned_pid=None):
    rows = []
    for line in output.splitlines():
        parts = line.strip().split(None, 5)
        if not parts:
            continue
        if len(parts) != 6:
            raise ValueError("incomplete ps activity row")
        rows.append(dict(pid=int(parts[0]), ppid=int(parts[1]), pgid=int(parts[2]),
                         cpu_percent=float(parts[3]), cpu_seconds=cpu_seconds(parts[4]), command=parts[5]))
    owned = {os.getpid()}
    if owned_pid is not None:
        owned.add(owned_pid)
    # Descendants include the measured process and the monitor's short-lived probes.
    while True:
        children = {row["pid"] for row in rows if row["ppid"] in owned}
        if children <= owned:
            break
        owned.update(children)
    return [row for row in rows if row["pid"] not in owned and row["pgid"] != owned_pid]


class HostMonitor:
    def __init__(self):
        self.previous = {}
        self.previous_at = None

    def sample(self, owned_pid=None):
        ps = snapshot(["ps", "-Ao", "pid=,ppid=,pgid=,pcpu=,time=,comm="])
        now = time.monotonic()
        sample = dict(monotonic_s=now, utc=datetime.now(timezone.utc).isoformat(),
                      power=snapshot(["pmset", "-g", "batt"]),
                      thermal=snapshot(["pmset", "-g", "therm"]))
        try:
            if ps["returncode"] != 0:
                raise ValueError("cannot inspect process activity: " + ps["stderr"])
            rows = parse_activity(ps["stdout"], owned_pid)
            if not rows:
                raise ValueError("ps returned no unrelated process records")
            interval = now - self.previous_at if self.previous_at is not None else None
            for row in rows:
                prior = self.previous.get(row["pid"])
                # ps %CPU supplies a conservative first/new-process observation;
                # established processes use interval CPU time, not lifetime averages.
                if prior is not None and interval and row["cpu_seconds"] >= prior:
                    row["cpu_percent"] = 100 * (row["cpu_seconds"] - prior) / interval
            self.previous = {row["pid"]: row["cpu_seconds"] for row in rows}
            self.previous_at = now
            sample.update(background_cpu_percent=sum(row["cpu_percent"] for row in rows),
                          max_background_cpu_percent=max(row["cpu_percent"] for row in rows),
                          background_cpu_by_pid={str(row["pid"]): row["cpu_percent"] for row in rows if row["cpu_percent"] > 0},
                          interval_s=interval, process_count=len(rows),
                          other_model_processes=[row for row in rows if MODEL_PROCESS.search(row["command"])],
                          activity=sorted(rows, key=lambda row: row["cpu_percent"], reverse=True)[:12])
        except (ValueError, TypeError) as error:
            sample.update(background_cpu_percent=None, activity=[], monitor_error=str(error))
        return sample


def host_eligibility(samples, threshold=5, sustained_seconds=2):
    reasons = set()
    elevated_since = {}
    if not samples:
        return dict(eligible=False, reasons=["missing host observations"])
    for sample in samples:
        if sample.get("monitor_error") or sample.get("background_cpu_percent") is None:
            reasons.add("activity monitoring failed")
        busy = {pid for pid, cpu in sample.get("background_cpu_by_pid", {}).items() if cpu > threshold}
        elevated_since = {pid: at for pid, at in elevated_since.items() if pid in busy}
        for pid in busy:
            # Interval CPU already describes the time preceding this observation.
            interval = sample.get("interval_s") or 0
            elevated_since.setdefault(pid, sample["monotonic_s"] - interval)
            if sample["monotonic_s"] - elevated_since[pid] >= sustained_seconds:
                reasons.add("sustained background CPU exceeds quiet threshold")
        if sample.get("other_model_processes"):
            reasons.add("another model process is present")
        power = sample.get("power", {})
        if power.get("returncode") != 0 or "'AC Power'" not in power.get("stdout", ""):
            reasons.add("AC power was not continuously observed")
        thermal = sample.get("thermal", {})
        if thermal.get("returncode") != 0:
            reasons.add("thermal monitoring failed")
        elif re.search(r"(?:CPU|GPU)_Speed_Limit\s*=\s*(?!100\b)\d+", thermal.get("stdout", "")):
            reasons.add("thermal speed limit was observed")
    return dict(eligible=not reasons, reasons=sorted(reasons))


def arm_environment(experiment, arm, cache, inherited=None):
    residency, overlap, prep = EXPERIMENTS[experiment][arm]
    source = os.environ if inherited is None else inherited
    env = {key: value for key, value in source.items() if not key.startswith("TURBOSPARK_")}
    env.update(TURBOSPARK_METAL_PRECOMPILED="0", TURBOSPARK_METAL_PIPELINE_CACHE="0",
               TURBOSPARK_METAL_KERNEL_WARMUP="0", TURBOSPARK_METAL_CACHE_DIR=str(cache),
               TURBOSPARK_QWEN_SHARED_READ_OVERLAP=str(int(overlap)), TURBOSPARK_MAPPED_DEMAND_PREP=prep)
    return residency, env


def validate_schedule(rows, args, arm):
    validate_rows(rows, "source")
    if [(row.get("open_index"), row.get("request_index")) for row in rows] != [(0, n) for n in range(args.repeats)]:
        raise ValueError("probe returned an incomplete or reordered request sequence")
    residency, overlap, prep = EXPERIMENTS[args.experiment][arm]
    for row in rows:
        expected = dict(residency=residency, shaping=args.shaping, case=args.case,
                        context=args.context, max_new=args.max_new, expert_cache_slots=16, prefix_reuse=False)
        if any(row.get(key) != value for key, value in expected.items()):
            raise ValueError("probe generation contract differs from the arm")
        if row.get("kernel_warmup_requested") is not False:
            raise ValueError("kernel warmup must be disabled for the scheduling comparison")
        if row.get("shared_read_overlap_requested") is not overlap or row.get("shared_read_overlap_effective") is not overlap:
            raise ValueError("shared-read overlap did not match or engage the arm")
        if row.get("mapped_demand_prep_requested") != prep or row.get("mapped_demand_prep_effective") != prep:
            raise ValueError("mapped demand preparation did not match or engage the arm")
        if row.get("mapped_demand_budget_bytes") != 16777216:
            raise ValueError("mapped demand preparation budget changed")
        counters = row.get("expert_schedule", {})
        if overlap and counters.get("shared_submissions_cumulative", 0) <= 0:
            raise ValueError("overlap arm submitted no expert reads")
        if prep != "off" and counters.get("mapped_prepare_calls_cumulative", 0) <= 0:
            raise ValueError("mapped arm prepared no experts")
        if prep == "advice" and (counters.get("mapped_advice_calls_cumulative", 0) <= 0 or
                                 counters.get("mapped_advice_failures_cumulative", 0)):
            raise ValueError("mapped advice was unavailable or failed")
        if prep == "touch" and counters.get("mapped_page_touches_cumulative", 0) <= 0:
            raise ValueError("mapped touch arm touched no pages")
        metrics = ["load_ms", "first_token_ms", "request_ms", "prefill_ms", "decode_ms"]
        if row["request_index"] == 0:
            metrics.append("load_to_first_token_ms")
        elif row.get("load_to_first_token_ms") is not None:
            raise ValueError("reset request must not claim a new load-to-first-token time")
        for metric in metrics:
            value = row.get(metric)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError("missing or invalid timing: " + metric)
        if row["decode_ms"] <= 0:
            raise ValueError("decode throughput is unavailable")
        opened = row.get("startup", {}).get("total_open_ms")
        if not isinstance(opened, (int, float)) or not math.isfinite(opened) or opened < 0:
            raise ValueError("runtime open timing is unavailable")


def compare_pair(arms):
    records = [row for rows in arms.values() for row in rows]
    compare_tokens(records)
    if len({tuple((row["open_index"], row["request_index"], row["prompt_tokens"]) for row in rows)
            for rows in arms.values()}) != 1:
        raise ValueError("request sequence or prompt token count changed between arms")
    baseline = arms["off"]
    def counters_for_request(rows, index):
        cumulative = rows[index].get("expert_schedule", {})
        previous = rows[index - 1].get("expert_schedule", {}) if index else {}
        return {key: value - previous.get(key, 0) for key, value in cumulative.items()}

    return {arm: [dict(request_index=row["request_index"],
                       prompt_tokens=row["prompt_tokens"], new_tokens=row["new_tokens"],
                       load_delta_ms=row["load_ms"] - reference["load_ms"],
                       runtime_open_delta_ms=row["startup"]["total_open_ms"] - reference["startup"]["total_open_ms"],
                       first_token_delta_ms=row["first_token_ms"] - reference["first_token_ms"],
                       load_to_first_token_delta_ms=(row["load_to_first_token_ms"] - reference["load_to_first_token_ms"])
                       if row["request_index"] == 0 else None,
                       request_delta_ms=row["request_ms"] - reference["request_ms"],
                       prefill_delta_ms=row["prefill_ms"] - reference["prefill_ms"],
                       decode_delta_ms=row["decode_ms"] - reference["decode_ms"],
                       expert_schedule_request=counters_for_request(rows, index),
                       expert_schedule_request_difference_from_off={
                           key: value - counters_for_request(baseline, index).get(key, 0)
                           for key, value in counters_for_request(rows, index).items()},
                       decode_delta_tokens_per_second=1000 * row["new_tokens"] / row["decode_ms"] -
                       1000 * reference["new_tokens"] / reference["decode_ms"])
                  for index, (row, reference) in enumerate(zip(rows, baseline))] for arm, rows in arms.items()}


def stop_process(process):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=2)
    except ProcessLookupError:
        process.wait(timeout=2)


class ProbeFailure(RuntimeError):
    pass


def run_probe(args, arm, out, tag):
    residency, env = arm_environment(args.experiment, arm, out / "application-cache")
    command = [str(args.binary), "--model", str(args.model), "--case", args.case,
               "--context", str(args.context), "--max-new", str(args.max_new), "--slots", "16",
               "--residency", residency, "--shaping", args.shaping, "--repeats", str(args.repeats),
               "--reopens", "1", "--label", arm, "--page-cache-label", args.page_cache_label]
    monitor = HostMonitor()
    preflight = []
    started = time.monotonic()
    while True:
        preflight.append(monitor.sample())
        if time.monotonic() - started >= args.quiet_seconds:
            break
        time.sleep(args.monitor_interval)
    preflight_verdict = host_eligibility(preflight, args.quiet_cpu_percent, args.sustained_seconds)
    allow_contended = getattr(args, "allow_contended", False)
    metadata = dict(command=command, environment={k: v for k, v in env.items() if k.startswith("TURBOSPARK_")},
                    preflight=preflight, preflight_verdict=preflight_verdict, periodic_host=[],
                    timed_out=False, exit_code=None, eligible=False, failure=None,
                    diagnostic_allow_contended=allow_contended)
    path = out / f"{tag}.launch.json"
    write_json(path, metadata)
    if not preflight_verdict["eligible"] and not allow_contended:
        metadata["failure"] = "quiet preflight rejected"
        write_json(path, metadata)
        raise ProbeFailure(f"{tag}: quiet preflight rejected; observations retained")
    process = None
    started = time.monotonic()
    try:
        with (out / f"{tag}.stdout.jsonl").open("w") as stdout, (out / f"{tag}.stderr.txt").open("w") as stderr:
            process = subprocess.Popen(command, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
            while process.poll() is None:
                metadata["periodic_host"].append(monitor.sample(process.pid))
                if time.monotonic() - started >= args.timeout_seconds:
                    metadata["timed_out"] = True
                    stop_process(process)
                    break
                time.sleep(min(args.monitor_interval, args.timeout_seconds))
            metadata["exit_code"] = process.returncode
        metadata["periodic_host"].append(monitor.sample(process.pid))
        if metadata["timed_out"] or process.returncode != 0:
            raise ProbeFailure(f"{tag}: probe failed or timed out; partial evidence retained")
        rows = [json.loads(line) for line in (out / f"{tag}.stdout.jsonl").read_text().splitlines() if line.strip()]
        validate_schedule(rows, args, arm)
        verdict = host_eligibility(preflight + metadata["periodic_host"], args.quiet_cpu_percent, args.sustained_seconds)
        reasons = verdict["reasons"]
        if allow_contended:
            reasons = [*reasons, "--allow-contended diagnostic excludes performance qualification"]
        metadata.update(eligible=verdict["eligible"] and not allow_contended, rejection_reasons=reasons)
        return rows, metadata
    except (OSError, ValueError, ProbeFailure) as error:
        metadata["failure"] = str(error)
        raise ProbeFailure(str(error)) from error
    finally:
        if process is not None:
            stop_process(process)
            metadata["exit_code"] = process.returncode
        metadata["wall_seconds"] = time.monotonic() - started
        write_json(path, metadata)


def summarize(attempts, required_pairs, allow_contended=False):
    qualified = [attempt for attempt in attempts if attempt["eligible"] and not allow_contended]
    return dict(protocol=PROTOCOL, attempts=attempts, eligible_pairs=len(qualified),
                diagnostic_allow_contended=allow_contended,
                requested_eligible_pairs=required_pairs, measurement_minimum_pairs=3,
                requested_pair_count_met=len(qualified) >= required_pairs,
                default_promotion_supported=False, os_page_cache_verified=False,
                eligible_first_request_median_deltas={
                    arm: {key: statistics.median(attempt["deltas"][arm][0][key] for attempt in qualified)
                          for key in ("load_delta_ms", "first_token_delta_ms", "load_to_first_token_delta_ms",
                                      "runtime_open_delta_ms", "request_delta_ms", "prefill_delta_ms",
                                      "decode_delta_ms", "decode_delta_tokens_per_second")}
                    for arm in qualified[0]["deltas"]} if qualified else {},
                limitations=["OS page residency and Apple's compiler cache are unverified",
                             "CPU, AC and thermal samples do not continuously verify GPU isolation",
                             "A token-parity timing comparison does not establish quality, memory or release gates"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("binary", "model", "out"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--experiment", choices=EXPERIMENTS, required=True)
    parser.add_argument("--case", default="short-explanation")
    parser.add_argument("--shaping", choices=("greedy", "sampled"), default="greedy")
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--max-new", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--max-attempts", type=int, default=9)
    parser.add_argument("--timeout-seconds", type=float, default=300)
    parser.add_argument("--monitor-interval", type=float, default=1)
    parser.add_argument("--quiet-seconds", type=float, default=3)
    parser.add_argument("--sustained-seconds", type=float, default=2)
    parser.add_argument("--quiet-cpu-percent", type=float, default=5,
                        help="sustained CPU use of one unrelated process, 100 percent is one core (maximum 5)")
    parser.add_argument("--page-cache-label", default="unknown")
    parser.add_argument("--allow-contended", action="store_true",
                        help="run bounded diagnostics despite preflight rejection; all captures remain ineligible")
    args = parser.parse_args(argv)
    if not all(1 <= value <= 100 for value in (args.pairs, args.max_attempts, args.repeats)):
        parser.error("pairs, max-attempts and repeats must each be in [1,100]")
    if args.pairs > args.max_attempts and not args.allow_contended:
        parser.error("pairs cannot exceed max-attempts without --allow-contended diagnostics")
    if not 0 < args.max_new < args.context or not 0 < args.timeout_seconds <= 3600:
        parser.error("token budget must fit context and timeout must be in (0,3600]")
    if not 0 < args.quiet_cpu_percent <= 5 or not 0.1 <= args.monitor_interval <= 5:
        parser.error("quiet CPU threshold must be in (0,5] and monitor interval in [0.1,5]")
    if not 0 < args.sustained_seconds <= args.quiet_seconds <= 60 or args.monitor_interval > args.sustained_seconds:
        parser.error("require monitor interval <= sustained seconds <= quiet seconds <= 60")
    args.binary = args.binary.expanduser().resolve(strict=True)
    args.model = args.model.expanduser().resolve(strict=True)
    args.out = args.out.expanduser().resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    identity = dict(protocol=PROTOCOL, binary_sha256=digest(args.binary),
                    harness_sha256=digest(Path(__file__).resolve()),
                    shared_helper_sha256=digest(root / "scripts" / "moe_startup_pairs.py"),
                    manifest_sha256=digest(args.model / "manifest.json"), platform=platform.platform(),
                    machine=platform.machine(), chip=snapshot(["sysctl", "-n", "machdep.cpu.brand_string"]),
                    os=snapshot(["sw_vers"]), git_revision=snapshot(["git", "rev-parse", "HEAD"], root),
                    git_status=snapshot(["git", "status", "--short"], root),
                    git_diff=snapshot(["git", "diff", "--stat"], root),
                    arguments={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()})
    write_json(args.out / "metadata.json", identity)
    attempts = []
    all_records = []
    try:
        for index in range(args.max_attempts):
            order = list(EXPERIMENTS[args.experiment])
            if index % 2:
                order.reverse()
            attempt = dict(index=index, order=order, eligible=False, failures={}, captures={}, deltas={})
            # Include the current partial attempt if parity or provenance fails.
            attempts.append(attempt)
            rows_by_arm = {}
            for arm in order:
                tag = f"attempt-{index}-{arm}"
                try:
                    rows, launch = run_probe(args, arm, args.out, tag)
                    if any(row["manifest_sha256"] != identity["manifest_sha256"] for row in rows):
                        raise ValueError("probe used a different model manifest")
                    rows_by_arm[arm] = rows
                    all_records.extend(rows)
                    attempt["captures"][arm] = dict(tag=tag, eligible=launch["eligible"] and not args.allow_contended,
                                                   rejection_reasons=launch.get("rejection_reasons", []), records=rows)
                except ProbeFailure as error:
                    # Keep running other modes, especially after a mapped baseline timeout.
                    attempt["failures"][arm] = str(error)
            if len(rows_by_arm) == len(order):
                attempt["deltas"] = compare_pair(rows_by_arm)
                compare_tokens(all_records)
                attempt["eligible"] = not args.allow_contended and all(capture["eligible"] for capture in attempt["captures"].values())
            if digest(args.binary) != identity["binary_sha256"] or digest(args.model / "manifest.json") != identity["manifest_sha256"]:
                raise ValueError("binary or manifest changed during experiment")
            write_json(args.out / "summary.json", summarize(attempts, args.pairs, args.allow_contended))
            print(json.dumps(dict(attempt=index, eligible=attempt["eligible"], failures=attempt["failures"])), flush=True)
            if sum(attempt["eligible"] for attempt in attempts) >= args.pairs:
                break
    except (ValueError, KeyboardInterrupt) as error:
        summary = summarize(attempts, args.pairs, args.allow_contended)
        summary["fatal_error"] = str(error) or "interrupted"
        write_json(args.out / "summary.json", summary)
        print(json.dumps(summary), flush=True)
        return 1
    summary = summarize(attempts, args.pairs, args.allow_contended)
    write_json(args.out / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    return 0 if summary["requested_pair_count_met"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
