#!/usr/bin/env python3
"""Run fresh-process, interleaved pairs of the ignored Metal attention bench.

Run on a quiet Metal host. The JSON output is a raw synthetic-kernel capture;
it does not establish an end-to-end performance result or acceptance pass.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_NAME = "attention_chunk_bench"
TEST_FILTER = "one_arm_attention_benchmark"
ARM_ENV = "TURBOSPARK_ATTENTION_BENCH_ARM"
ROWS_ENV = "TURBOSPARK_ATTENTION_BENCH_M"
REPETITIONS_ENV = "TURBOSPARK_ATTENTION_BENCH_REPETITIONS"
FIRST_QUERY_POSITION = 240
MAX_BATCH_ROWS = 16
DEFAULT_ROWS = 16
DEFAULT_REPETITIONS = 7
DEFAULT_PAIRS = 3
MIN_PAIRS = 3
HEAD_SHAPE = {"head_dim": 128, "num_q_heads": 32, "num_kv_heads": 8}
BUILD_INPUTS = (
    "Cargo.toml",
    "Cargo.lock",
    ".cargo",
    "rust-toolchain",
    "rust-toolchain.toml",
    "crates",
)


def validate_pair_count(pair_count):
    if type(pair_count) is not int or pair_count < MIN_PAIRS:
        raise ValueError(f"pair count must be an integer >= {MIN_PAIRS}")


def validate_inputs(rows, repetitions):
    if type(rows) is not int or not 1 <= rows <= MAX_BATCH_ROWS:
        raise ValueError(f"M must be an integer in 1..={MAX_BATCH_ROWS}")
    if type(repetitions) is not int or not 2 <= repetitions <= 100:
        raise ValueError("repetitions must be an integer in 2..=100")


def arm_order_for_pair(pair_index):
    if type(pair_index) is not int or pair_index < 0:
        raise ValueError("pair index must be a nonnegative integer")
    return ("batched", "serial") if pair_index % 2 == 0 else ("serial", "batched")


def _load_json_without_constants(value):
    def reject_constant(constant):
        raise ValueError(f"non-finite JSON number {constant}")

    return json.loads(value, parse_constant=reject_constant)


def _source_files(repo_root):
    for relative in BUILD_INPUTS:
        path = Path(repo_root) / relative
        if not path.exists():
            continue
        if path.is_symlink():
            raise RuntimeError(f"build input is a symlink: {relative}")
        if path.is_file():
            yield path, Path(relative)
            continue
        for current, directories, filenames in os.walk(path, followlinks=False):
            current_path = Path(current)
            directories.sort()
            filenames.sort()
            for directory in directories:
                child = current_path / directory
                if child.is_symlink():
                    raise RuntimeError(f"build input is a symlink: {child.relative_to(repo_root)}")
            for filename in filenames:
                child = current_path / filename
                if child.is_symlink():
                    raise RuntimeError(f"build input is a symlink: {child.relative_to(repo_root)}")
                if child.is_file():
                    yield child, child.relative_to(repo_root)


def source_snapshot(repo_root=REPO_ROOT, command_runner=subprocess.run):
    repo_root = Path(repo_root).resolve()
    revision_result = command_runner(
        ["git", "-C", str(repo_root), "rev-parse", "--verify", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if revision_result.returncode != 0:
        raise RuntimeError("could not resolve the benchmark source revision")
    revision = (revision_result.stdout or "").strip()
    if not revision:
        raise RuntimeError("benchmark source revision is empty")

    status_result = command_runner(
        [
            "git",
            "-C",
            str(repo_root),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            *BUILD_INPUTS,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if status_result.returncode != 0:
        raise RuntimeError("could not inspect benchmark build-input status")
    dirty_paths = [line[3:] for line in (status_result.stdout or "").splitlines() if line]

    digest = hashlib.sha256()
    for path, relative in sorted(_source_files(repo_root), key=lambda pair: pair[1].as_posix()):
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as source_file:
            for block in iter(lambda: source_file.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")

    return {
        "revision": revision,
        "build_inputs_sha256": digest.hexdigest(),
        "dirty_paths": dirty_paths,
    }


def parse_test_binary(cargo_stdout, repo_root):
    artifacts = []
    for line in cargo_stdout.splitlines():
        if not line.strip():
            continue
        try:
            item = _load_json_without_constants(line)
        except (json.JSONDecodeError, ValueError):
            continue
        target = item.get("target") if isinstance(item, dict) else None
        if (
            isinstance(item, dict)
            and item.get("reason") == "compiler-artifact"
            and isinstance(target, dict)
            and target.get("name") == TEST_NAME
            and "test" in target.get("kind", [])
        ):
            executable = item.get("executable")
            if not isinstance(executable, str) or not executable:
                raise RuntimeError("attention_chunk_bench test artifact has no executable")
            path = Path(executable)
            if not path.is_absolute():
                path = Path(repo_root) / path
            artifacts.append(path.resolve())

    if not artifacts:
        raise RuntimeError("no attention_chunk_bench test artifact in Cargo build output")
    if len(artifacts) != 1:
        raise RuntimeError("multiple attention_chunk_bench test artifacts in Cargo build output")
    if not artifacts[0].is_file():
        raise RuntimeError(f"attention_chunk_bench executable does not exist: {artifacts[0]}")
    return artifacts[0].absolute()


def build_test_binary(repo_root=REPO_ROOT, command_runner=subprocess.run):
    command = [
        "cargo",
        "test",
        "--locked",
        "-p",
        "turbospark-gpu",
        "--test",
        TEST_NAME,
        "--release",
        "--no-run",
        "--message-format=json-render-diagnostics",
    ]
    result = command_runner(
        command,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or "").strip()
        raise RuntimeError(f"release benchmark build failed ({result.returncode}): {detail}")
    return parse_test_binary(result.stdout or "", repo_root), command


def parse_record_line(stdout):
    candidates = []
    for line in stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("{") or '"arm"' in stripped:
            candidates.append(stripped)
    if len(candidates) != 1:
        raise RuntimeError(
            f"benchmark child must emit exactly one JSON record, found {len(candidates)}"
        )
    try:
        record = _load_json_without_constants(candidates[0])
    except (json.JSONDecodeError, ValueError) as error:
        raise RuntimeError(f"benchmark child emitted malformed JSON: {error}") from error
    if not isinstance(record, dict):
        raise RuntimeError("benchmark child JSON record must be an object")
    return candidates[0]


def _is_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source_file:
        for block in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _expect_number(record, field):
    value = record.get(field)
    if not _is_number(value):
        raise ValueError(f"{field} must be a finite number")
    return value


def validate_record(record, expected_arm, rows, repetitions):
    if not isinstance(record, dict):
        raise ValueError("benchmark record must be a JSON object")
    if record.get("arm") != expected_arm:
        raise ValueError(f"arm must be {expected_arm!r}")
    if type(record.get("M")) is not int or record["M"] != rows:
        raise ValueError(f"M must be {rows}")
    expected_positions = list(range(FIRST_QUERY_POSITION, FIRST_QUERY_POSITION + rows))
    if record.get("row_positions") != expected_positions:
        raise ValueError(f"row_positions must equal {expected_positions}")
    if record.get("head_shape") != HEAD_SHAPE:
        raise ValueError(f"head_shape must equal {HEAD_SHAPE}")
    if not isinstance(record.get("source_revision"), str) or not record["source_revision"]:
        raise ValueError("source_revision must be a nonempty string")
    if record.get("benchmark_source_dirty") is not False:
        raise ValueError("benchmark source must be clean")
    if record.get("release_mode") != "release":
        raise ValueError("release_mode must be 'release'")
    if not isinstance(record.get("metal_device"), str) or not record["metal_device"].strip():
        raise ValueError("metal_device must be a nonempty string")
    for field in ("repetitions", "sample_count"):
        if type(record.get(field)) is not int or record[field] != repetitions:
            raise ValueError(f"{field} must equal {repetitions}")

    samples = record.get("elapsed_seconds_per_repetition_samples")
    if not isinstance(samples, list) or len(samples) != repetitions:
        raise ValueError(
            f"elapsed_seconds_per_repetition_samples must contain {repetitions} values"
        )
    if any(not _is_number(sample) or sample <= 0 for sample in samples):
        raise ValueError("elapsed_seconds_per_repetition_samples must be finite and positive")

    sorted_samples = sorted(samples)
    middle = len(sorted_samples) // 2
    expected_median = (
        sorted_samples[middle]
        if len(sorted_samples) % 2
        else (sorted_samples[middle - 1] + sorted_samples[middle]) / 2
    )
    median = _expect_number(record, "elapsed_seconds_per_repetition_median")
    if not math.isclose(median, expected_median, rel_tol=0, abs_tol=1.1e-9):
        raise ValueError("reported per-repetition median does not match raw samples")
    elapsed_range = record.get("elapsed_seconds_per_repetition_range")
    if (
        not isinstance(elapsed_range, list)
        or len(elapsed_range) != 2
        or any(not _is_number(value) for value in elapsed_range)
        or not math.isclose(elapsed_range[0], sorted_samples[0], rel_tol=0, abs_tol=1.1e-9)
        or not math.isclose(elapsed_range[1], sorted_samples[-1], rel_tol=0, abs_tol=1.1e-9)
    ):
        raise ValueError("reported per-repetition range does not match raw samples")
    total = _expect_number(record, "elapsed_seconds_total")
    if not math.isclose(total, sum(samples), rel_tol=0, abs_tol=repetitions * 1.1e-9):
        raise ValueError("reported elapsed total does not match raw samples")


def _record_identity(record):
    return {
        "source_revision": record["source_revision"],
        "release_mode": record["release_mode"],
        "metal_device": record["metal_device"],
        "M": record["M"],
        "row_positions": record["row_positions"],
        "head_shape": record["head_shape"],
        "repetitions": record["repetitions"],
    }


def _check_compatible_record(record, identity):
    for field, expected in identity.items():
        if record[field] != expected:
            raise ValueError(f"{field} mismatch across matched benchmark runs")


def summarize_records(raw_records):
    summary = {}
    for arm in ("batched", "serial"):
        samples = [
            sample
            for item in raw_records
            if item["arm"] == arm
            for sample in item["record"]["elapsed_seconds_per_repetition_samples"]
        ]
        if not samples:
            raise ValueError(f"no timing samples captured for {arm}")
        summary[arm] = {
            "sample_count": len(samples),
            "median": statistics.median(samples),
            "range": {"min": min(samples), "max": max(samples)},
        }
    return summary


def run_pairs(
    pair_count=DEFAULT_PAIRS,
    rows=DEFAULT_ROWS,
    repetitions=DEFAULT_REPETITIONS,
    repo_root=REPO_ROOT,
    command_runner=subprocess.run,
):
    validate_pair_count(pair_count)
    validate_inputs(rows, repetitions)
    repo_root = Path(repo_root).resolve()

    source = source_snapshot(repo_root, command_runner)
    if source["dirty_paths"]:
        raise RuntimeError(
            "benchmark build inputs must be clean: " + ", ".join(source["dirty_paths"])
        )

    # Build once and execute the resulting test binary directly for every arm.
    binary, build_command = build_test_binary(repo_root, command_runner)
    if source_snapshot(repo_root, command_runner) != source:
        raise RuntimeError("benchmark build inputs changed during the release build")
    binary_hash = file_sha256(binary)

    raw_records = []
    identity = None
    for pair_index in range(pair_count):
        for position, arm in enumerate(arm_order_for_pair(pair_index), start=1):
            if source_snapshot(repo_root, command_runner) != source:
                raise RuntimeError("benchmark build inputs changed during measurement")
            if file_sha256(binary) != binary_hash:
                raise RuntimeError("benchmark test binary changed during measurement")
            command = [
                str(binary),
                TEST_FILTER,
                "--ignored",
                "--nocapture",
            ]
            environment = os.environ.copy()
            environment[ARM_ENV] = arm
            environment[ROWS_ENV] = str(rows)
            environment[REPETITIONS_ENV] = str(repetitions)
            result = command_runner(
                command,
                cwd=str(repo_root),
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                detail = (result.stderr or "").strip()
                raise RuntimeError(
                    f"benchmark child failed for pair {pair_index + 1} {arm} "
                    f"({result.returncode}): {detail}"
                )
            raw_json = parse_record_line(result.stdout or "")
            record = _load_json_without_constants(raw_json)
            validate_record(record, arm, rows, repetitions)
            if record["source_revision"] != source["revision"]:
                raise ValueError("benchmark source_revision does not match the built checkout")
            if identity is None:
                identity = _record_identity(record)
            else:
                _check_compatible_record(record, identity)
            raw_records.append(
                {
                    "pair": pair_index + 1,
                    "order_in_pair": position,
                    "arm": arm,
                    "command": command,
                    "raw_json": raw_json,
                    "record": record,
                }
            )

    if source_snapshot(repo_root, command_runner) != source:
        raise RuntimeError("benchmark build inputs changed during measurement")
    if file_sha256(binary) != binary_hash:
        raise RuntimeError("benchmark test binary changed during measurement")

    return {
        "build": {
            "command": build_command,
            "binary": str(binary),
            "mode": "release",
            "sha256": binary_hash,
        },
        "pair_count": pair_count,
        "source_identity": source,
        "inputs": {
            "M": rows,
            "repetitions_per_process": repetitions,
            "head_shape": HEAD_SHAPE,
            "row_positions": list(range(FIRST_QUERY_POSITION, FIRST_QUERY_POSITION + rows)),
        },
        "raw_records": raw_records,
        "summary": summarize_records(raw_records),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, default=DEFAULT_PAIRS)
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS, help="benchmark M (1..16)")
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--output", type=Path, help="also save the complete JSON report here")
    args = parser.parse_args(argv)

    try:
        report = run_pairs(args.pairs, args.rows, args.repetitions)
    except (RuntimeError, ValueError) as error:
        print(f"attention batch benchmark failed: {error}", file=sys.stderr)
        return 2

    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
