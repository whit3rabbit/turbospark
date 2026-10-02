#!/usr/bin/env python3
"""Compare pinned release binaries on matched dense-Llama prefill runs.

Each run is a fresh process. The bench binary performs and discards its own
warmup before reporting the measured long-synthesis prefill time. This runner
records the exact footer, artifact and binary hashes, commands, revisions,
host/device, and pair order. It does not run unless a model install and both
release binaries are supplied explicitly.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import socket
import statistics
import subprocess
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[1]
CASE = "long-synthesis"
PROMPT_PATH = "crates/bench/prompts/real-generation-v1/long-synthesis.txt"
CHUNK_SIZES = (32, 64, 128, 256, 512, 1024, 2048, 4096)
MIN_PAIRS = 3
DEFAULT_PAIRS = 3
DEFAULT_EXPERT_CACHE_SLOTS = 16
PINNED_BUILD_IDENTITIES = {
    "baseline": {
        "source_revision": "324b3f03ce94c85a5356c3ce9c6c5658fde81edc",
        "binary_sha256": "2ad05a09cb41a6d6410f90969047c1fe2f736db868fb4d9bbd26cbcabb2ab34a",
    },
    "candidate": {
        "source_revision": "80252a6f951c4f7e13acd3608855b67231cd9f36",
        "binary_sha256": "f8cf688db2c50fd5407cad7b35d9acefc944ac41d632549c4cfa06c408345f58",
    },
}
CLEARED_ENV_VARS = (
    "TURBOSPARK_PHASES",
    "TURBOSPARK_ROUTER_TRACE",
    "TURBOSPARK_PREFILL_CHUNK",
    "TURBOSPARK_ROUTED_BATCH",
    "TURBOSPARK_BATCHED_GEMV",
    "TURBOSPARK_DISPATCH_PROFILE",
    "TURBOSPARK_DEBUG_PROMPT_IDS",
    "TURBOSPARK_METAL_PRECISE_MATH",
    "TURBOSPARK_RESID_CAPTURE",
)
FOOTER_RE = re.compile(
    r"\[stop=([^\s\]]+) prefill=(\d+)tok/(\d+(?:\.\d+)?)s "
    r"new=(\d+)tok decode=(\d+(?:\.\d+)?)s tok/s=(\d+(?:\.\d+)?)\]"
)
FAMILY_RE = re.compile(
    r"^\s*family=(\S+) context=(\d+) max_new=(\d+) "
    r"expert_cache_slots=(\d+) kv_bits=(\S+)\s*$",
    re.MULTILINE,
)
PREFILL_RE = re.compile(
    r"^\s*prefill=chunked chunk_tokens=(\d+) routed_batch=(\S+) "
    r"batched_gemv=(\S+)\s*$",
    re.MULTILINE,
)
SHAPING_RE = re.compile(r"^\s*speculative=off shaping=protocol-sampled\s*$", re.MULTILINE)
CASE_ROW_RE = re.compile(r"^\s*long-synthesis\s+(\d+)\s+(\d+(?:\.\d+)?)\s+\d+\s+", re.MULTILINE)
IDENTITY_RE = re.compile(r"^[0-9a-fA-F]{40,64}$")
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
WORKSPACE_PACKAGE_RE = re.compile(
    r"(?ms)^\[workspace\.package\]\s*\n(.*?)(?=^\[|\Z)"
)
PACKAGE_VERSION_RE = re.compile(r'^\s*version\s*=\s*"([^"\n]+)"\s*$', re.MULTILINE)


def validate_inputs(pair_count, chunk_size):
    if type(pair_count) is not int or pair_count < MIN_PAIRS:
        raise ValueError(f"pair count must be an integer >= {MIN_PAIRS}")
    if type(chunk_size) is not int or chunk_size not in CHUNK_SIZES:
        raise ValueError(f"chunk size must be one of {CHUNK_SIZES}")


def arm_order_for_pair(pair_index):
    if type(pair_index) is not int or pair_index < 0:
        raise ValueError("pair index must be a nonnegative integer")
    return ("baseline", "candidate") if pair_index % 2 == 0 else ("candidate", "baseline")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source_file:
        for block in iter(lambda: source_file.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_files(root):
    root = Path(root)
    if root.is_symlink():
        raise ValueError("model install path must resolve to a real file or directory")
    if root.is_file():
        yield root, Path(root.name)
        return
    if not root.is_dir():
        raise ValueError(f"model install does not exist: {root}")
    for current, directories, filenames in os.walk(root, followlinks=False):
        current_path = Path(current)
        directories.sort()
        filenames.sort()
        for name in directories:
            path = current_path / name
            if path.is_symlink():
                raise ValueError(f"model install contains a symlink: {path}")
        for name in filenames:
            path = current_path / name
            if path.is_symlink():
                raise ValueError(f"model install contains a symlink: {path}")
            if path.is_file():
                yield path, path.relative_to(root)


def artifact_metadata(path):
    root = Path(path).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError("model install must be a .gturbo directory")
    manifest_path = root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"model install manifest is missing or malformed: {error}") from error
    arch = manifest.get("arch") if isinstance(manifest, dict) else None
    if not isinstance(arch, dict) or manifest.get("magic") != "GTURBO":
        raise ValueError("model install is not a valid GTURBO manifest")
    if arch.get("family") != "llama":
        raise ValueError("model install must declare the llama family")
    if type(arch.get("numExperts")) is not int or arch["numExperts"] != 0:
        raise ValueError("model install must be dense Llama with numExperts=0")
    if type(manifest.get("expertsPerLayer")) is not int or manifest["expertsPerLayer"] != 0:
        raise ValueError("model install must be dense Llama with expertsPerLayer=0")
    if not isinstance(manifest.get("modelID"), str) or not manifest["modelID"].strip():
        raise ValueError("model install manifest must declare modelID")
    entries = []
    total_bytes = 0
    for file_path, relative in sorted(_artifact_files(root), key=lambda item: item[1].as_posix()):
        stat_result = file_path.stat()
        entry = {
            "path": relative.as_posix(),
            "size_bytes": stat_result.st_size,
            "mtime_ns": stat_result.st_mtime_ns,
            "inode": stat_result.st_ino,
        }
        entries.append(entry)
        total_bytes += stat_result.st_size
    if not entries:
        raise ValueError(f"model install contains no regular files: {root}")
    return {
        "path": str(root),
        "file_count": len(entries),
        "total_bytes": total_bytes,
        "model_id": manifest["modelID"],
        "family": arch["family"],
        "num_experts": arch["numExperts"],
        "experts_per_layer": manifest["expertsPerLayer"],
        "source_snapshot_hash": manifest.get("sourceSnapshotHash"),
        "entries": entries,
    }


def artifact_snapshot(path):
    metadata = artifact_metadata(path)
    root = Path(metadata["path"])
    digest = hashlib.sha256()
    for entry in metadata["entries"]:
        relative = Path(entry["path"])
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        with (root / relative).open("rb") as source_file:
            for block in iter(lambda: source_file.read(8 * 1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
    return {**metadata, "sha256": digest.hexdigest()}


def _run_checked(command, command_runner, cwd):
    result = command_runner(
        command,
        cwd=str(cwd),
        capture_output=True,
        text=False,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr or b""
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"command failed ({result.returncode}): {command!r}: {stderr.strip()}")
    return result.stdout or b""


def resolve_revision(revision, label, repo_root=REPO_ROOT, command_runner=subprocess.run):
    if not isinstance(revision, str) or not IDENTITY_RE.fullmatch(revision):
        raise ValueError(f"{label} source identity must be a full Git commit hash")
    output = _run_checked(
        ["git", "rev-parse", "--verify", f"{revision}^{{commit}}"], command_runner, repo_root
    )
    resolved = output.decode("ascii").strip()
    if resolved.lower() != revision.lower():
        raise ValueError(f"{label} revision resolved to a different commit")
    return resolved


def prompt_sha256(revision, repo_root=REPO_ROOT, command_runner=subprocess.run):
    content = _run_checked(
        ["git", "show", f"{revision}:{PROMPT_PATH}"], command_runner, repo_root
    )
    if not content:
        raise RuntimeError(f"empty benchmark prompt in source revision {revision}")
    return hashlib.sha256(content).hexdigest()


def package_version(revision, repo_root=REPO_ROOT, command_runner=subprocess.run):
    manifest = _run_checked(
        ["git", "show", f"{revision}:Cargo.toml"], command_runner, repo_root
    ).decode("utf-8")
    section = WORKSPACE_PACKAGE_RE.search(manifest)
    match = PACKAGE_VERSION_RE.search(section.group(1)) if section else None
    if not match:
        raise ValueError(
            f"could not derive workspace package version from {revision}:Cargo.toml"
        )
    return match.group(1)


def host_identity(command_runner=subprocess.run, cwd=REPO_ROOT):
    result = command_runner(
        ["system_profiler", "SPDisplaysDataType", "-json"],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 or not (result.stdout or "").strip():
        detail = (result.stderr or "").strip()
        raise RuntimeError(f"could not read Metal device identity: {detail}")
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "macos_version": platform.mac_ver()[0],
        "system_profiler_displays_json": result.stdout,
    }


def _decode_output(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value or ""


def parse_run_output(
    stdout,
    stderr,
    expected_chunk,
    expected_model_path,
    expected_slots=DEFAULT_EXPERT_CACHE_SLOTS,
):
    stdout = _decode_output(stdout)
    stderr = _decode_output(stderr)
    combined = stdout + "\n" + stderr
    footers = list(FOOTER_RE.finditer(combined))
    if len(footers) != 1:
        raise ValueError(f"expected exactly one measured footer, found {len(footers)}")
    family_rows = list(FAMILY_RE.finditer(stdout))
    if len(family_rows) != 1:
        raise ValueError(f"expected exactly one resolved model header, found {len(family_rows)}")
    family, context, max_new, slots, kv_bits = family_rows[0].groups()
    if family != "llama":
        raise ValueError(f"expected dense-Llama family header, got {family!r}")
    if kv_bits != "off":
        raise ValueError("resolved model header must report unquantized KV")
    if int(slots) != expected_slots:
        raise ValueError(f"expert cache slots must equal {expected_slots}")
    prefill_rows = list(PREFILL_RE.finditer(stdout))
    if len(prefill_rows) != 1:
        raise ValueError(f"expected exactly one resolved chunk header, found {len(prefill_rows)}")
    chunk_tokens, routed_batch, batched_gemv = prefill_rows[0].groups()
    if int(chunk_tokens) != expected_chunk:
        raise ValueError(f"resolved chunk size must equal {expected_chunk}")
    if not SHAPING_RE.search(stdout):
        raise ValueError("run must use the sampled, non-speculative protocol")
    banner_prefix = f"turbospark-bench: real install {expected_model_path} on "
    banner_suffix = ", frozen protocol real-generation-v1"
    banners = [
        line
        for line in stdout.splitlines()
        if line.startswith("turbospark-bench: real install ") and line.endswith(banner_suffix)
    ]
    if len(banners) != 1:
        raise ValueError(f"expected exactly one real-install banner, found {len(banners)}")
    if not banners[0].startswith(banner_prefix):
        raise ValueError("real-install banner does not match the pinned model path")
    metal_chip_brand = banners[0][len(banner_prefix) : -len(banner_suffix)].strip()
    if not metal_chip_brand:
        raise ValueError("real-install banner has an empty device identity")
    case_rows = list(CASE_ROW_RE.finditer(stdout))
    if len(case_rows) != 1:
        raise ValueError(f"expected exactly one {CASE} measurement row, found {len(case_rows)}")

    footer = footers[0]
    (
        stop_reason,
        prompt_tokens,
        prefill_seconds_text,
        new_tokens,
        decode_seconds_text,
        tok_s_text,
    ) = footer.groups()
    prefill_seconds = float(prefill_seconds_text)
    if not math.isfinite(prefill_seconds) or prefill_seconds <= 0:
        raise ValueError("measured prefill time must be finite and positive")
    if int(prompt_tokens) <= 0 or int(new_tokens) <= 0:
        raise ValueError("measured prompt and generation token counts must be positive")
    row_prompt_tokens, row_prefill_seconds = case_rows[0].groups()
    if int(row_prompt_tokens) != int(prompt_tokens):
        raise ValueError("case row and measured footer disagree on prompt token count")
    if row_prefill_seconds != prefill_seconds_text:
        raise ValueError("case row and measured footer disagree on prefill time")
    return {
        "footer_raw": footer.group(0),
        "stop_reason": stop_reason,
        "prompt_tokens": int(prompt_tokens),
        "prefill_seconds_raw": prefill_seconds_text,
        "prefill_seconds": prefill_seconds,
        "new_tokens": int(new_tokens),
        "decode_seconds_raw": decode_seconds_text,
        "tokens_per_second_raw": tok_s_text,
        "family": family,
        "context_tokens": int(context),
        "max_new_tokens": int(max_new),
        "expert_cache_slots": int(slots),
        "kv_bits": kv_bits,
        "chunk_tokens": int(chunk_tokens),
        "routed_batch": routed_batch,
        "batched_gemv": batched_gemv,
        "metal_chip_brand": metal_chip_brand,
        "stdout": stdout,
        "stderr": stderr,
    }


def summarize(raw_runs):
    arms = {}
    for arm in ("baseline", "candidate"):
        times = [run["result"]["prefill_seconds"] for run in raw_runs if run["arm"] == arm]
        if not times:
            raise ValueError(f"no measurements captured for {arm}")
        arms[arm] = {
            "sample_count": len(times),
            "median_prefill_seconds": statistics.median(times),
            "range_prefill_seconds": {"min": min(times), "max": max(times)},
        }
    return arms


def _input_command(binary, model_path, chunk_size, expert_cache_slots):
    return [
        str(binary),
        "--model",
        str(model_path),
        "--case",
        CASE,
        "--prefill-chunk",
        str(chunk_size),
        "--kv-bits",
        "off",
        "--shaping",
        "protocol",
        "--speculative",
        "off",
        "--power-profile",
        "performance",
        "--expert-cache-slots",
        str(expert_cache_slots),
    ]


def run_pairs(
    baseline_binary,
    candidate_binary,
    baseline_binary_sha256,
    candidate_binary_sha256,
    baseline_revision,
    candidate_revision,
    model_install,
    chunk_size,
    pair_count=DEFAULT_PAIRS,
    baseline_diff_hash=None,
    candidate_diff_hash=None,
    expert_cache_slots=DEFAULT_EXPERT_CACHE_SLOTS,
    repo_root=REPO_ROOT,
    command_runner=subprocess.run,
    host_reader=host_identity,
):
    validate_inputs(pair_count, chunk_size)
    if type(expert_cache_slots) is not int or expert_cache_slots <= 0:
        raise ValueError("expert cache slots must be a positive integer")
    repo_root = Path(repo_root).resolve()
    binaries = {
        "baseline": Path(baseline_binary).expanduser().resolve(strict=True),
        "candidate": Path(candidate_binary).expanduser().resolve(strict=True),
    }
    if any(not path.is_file() or not os.access(path, os.X_OK) for path in binaries.values()):
        raise ValueError("both benchmark binaries must be executable regular files")
    binary_hashes = {arm: file_sha256(path) for arm, path in binaries.items()}
    expected_binary_hashes = {
        "baseline": baseline_binary_sha256,
        "candidate": candidate_binary_sha256,
    }
    for arm, expected_hash in expected_binary_hashes.items():
        if not isinstance(expected_hash, str) or not SHA256_RE.fullmatch(expected_hash):
            raise ValueError(f"{arm} recorded binary SHA-256 must contain 64 hex characters")
        if binary_hashes[arm].lower() != expected_hash.lower():
            raise ValueError(f"{arm} binary SHA-256 does not match the recorded artifact")
    revisions = {
        "baseline": resolve_revision(baseline_revision, "baseline", repo_root, command_runner),
        "candidate": resolve_revision(candidate_revision, "candidate", repo_root, command_runner),
    }
    for arm in ("baseline", "candidate"):
        pinned = PINNED_BUILD_IDENTITIES[arm]
        if revisions[arm].lower() != pinned["source_revision"].lower():
            raise ValueError(f"{arm} source revision does not match the pinned task 4.2 artifact")
        if expected_binary_hashes[arm].lower() != pinned["binary_sha256"].lower():
            raise ValueError(f"{arm} recorded SHA-256 does not match the pinned task 4.2 artifact")
    prompt_hashes = {
        arm: prompt_sha256(revision, repo_root, command_runner)
        for arm, revision in revisions.items()
    }
    if prompt_hashes["baseline"] != prompt_hashes["candidate"]:
        raise ValueError("baseline and candidate embed different long-synthesis prompts")
    package_versions = {
        arm: package_version(revision, repo_root, command_runner)
        for arm, revision in revisions.items()
    }

    model = artifact_snapshot(model_install)
    original_model_fingerprint = [
        {key: item[key] for key in ("path", "size_bytes", "mtime_ns", "inode")}
        for item in model["entries"]
    ]
    host = host_reader(command_runner, repo_root)
    raw_runs = []
    identity = None
    process_environment = os.environ.copy()
    cleared_environment = {}
    for name in CLEARED_ENV_VARS:
        if name in process_environment:
            cleared_environment[name] = process_environment.pop(name)
    for pair_index in range(pair_count):
        for order, arm in enumerate(arm_order_for_pair(pair_index), start=1):
            current_model = artifact_metadata(model["path"])
            if current_model["entries"] != original_model_fingerprint:
                raise RuntimeError("pinned model install changed during measurement")
            if {name: file_sha256(path) for name, path in binaries.items()} != binary_hashes:
                raise RuntimeError("release benchmark binary changed during measurement")

            command = _input_command(binaries[arm], model["path"], chunk_size, expert_cache_slots)
            started = time.monotonic()
            result = command_runner(
                command,
                cwd=str(repo_root),
                env=process_environment.copy(),
                capture_output=True,
                text=True,
                check=False,
            )
            process_wall_seconds = time.monotonic() - started
            if result.returncode != 0:
                detail = _decode_output(result.stderr).strip()
                raise RuntimeError(
                    f"{arm} process failed in pair {pair_index + 1} "
                    f"({result.returncode}): {detail}"
                )
            parsed = parse_run_output(
                result.stdout,
                result.stderr,
                chunk_size,
                model["path"],
                expert_cache_slots,
            )
            result_identity = {
                key: parsed[key]
                for key in (
                    "family",
                    "context_tokens",
                    "max_new_tokens",
                    "expert_cache_slots",
                "kv_bits",
                "chunk_tokens",
                "prompt_tokens",
                "routed_batch",
                    "batched_gemv",
                    "metal_chip_brand",
                )
            }
            if identity is None:
                identity = result_identity
            elif result_identity != identity:
                raise ValueError("resolved model, chunk, or device metadata differs across runs")
            raw_runs.append(
                {
                    "pair": pair_index + 1,
                    "order_in_pair": order,
                    "arm": arm,
                    "source_revision": revisions[arm],
                    "binary_path": str(binaries[arm]),
                    "binary_sha256": binary_hashes[arm],
                    "command": command,
                    "process_wall_seconds": process_wall_seconds,
                    "result": parsed,
                }
            )

    final_model = artifact_snapshot(model["path"])
    if (
        final_model["sha256"] != model["sha256"]
        or final_model["entries"] != original_model_fingerprint
    ):
        raise RuntimeError("pinned model install changed during measurement")
    if {name: file_sha256(path) for name, path in binaries.items()} != binary_hashes:
        raise RuntimeError("release benchmark binary changed during measurement")

    return {
        "protocol": {
            "case": CASE,
            "prompt_path": PROMPT_PATH,
            "prompt_sha256": prompt_hashes["baseline"],
            "chunk_size": chunk_size,
            "kv_bits": "off",
            "warmup_discarded_by_binary": True,
            "warmup_runs_per_process": 1,
            "pair_count": pair_count,
            "pair_order": "baseline/candidate on odd pairs, candidate/baseline on even pairs",
            "expert_cache_slots": expert_cache_slots,
        },
        "model_install": model,
        "host_device": host,
        "environment": {
            "cleared_turbospark_controls": cleared_environment,
            "note": "all other inherited environment values were identical for every process",
        },
        "builds": {
            arm: {
                "source_revision": revisions[arm],
                "package_version": package_versions[arm],
                "diff_hash": baseline_diff_hash if arm == "baseline" else candidate_diff_hash,
                "binary_path": str(binaries[arm]),
                "binary_sha256": binary_hashes[arm],
                "build_mode": "release",
            }
            for arm in ("baseline", "candidate")
        },
        "inputs": {
            "model_path": model["path"],
            "case": CASE,
            "chunk_size": chunk_size,
            "kv_bits": "off",
            "expert_cache_slots": expert_cache_slots,
        },
        "raw_runs": raw_runs,
        "summary": summarize(raw_runs),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-binary", type=Path, required=True)
    parser.add_argument("--candidate-binary", type=Path, required=True)
    parser.add_argument("--baseline-revision", required=True)
    parser.add_argument("--candidate-revision", required=True)
    parser.add_argument("--baseline-binary-sha256", required=True)
    parser.add_argument("--candidate-binary-sha256", required=True)
    parser.add_argument("--baseline-diff-hash")
    parser.add_argument("--candidate-diff-hash")
    parser.add_argument("--model-install", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, choices=CHUNK_SIZES, required=True)
    parser.add_argument("--pairs", type=int, default=DEFAULT_PAIRS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise ValueError(f"output already exists: {args.output}")
        report = run_pairs(
            baseline_binary=args.baseline_binary,
            candidate_binary=args.candidate_binary,
            baseline_binary_sha256=args.baseline_binary_sha256,
            candidate_binary_sha256=args.candidate_binary_sha256,
            baseline_revision=args.baseline_revision,
            candidate_revision=args.candidate_revision,
            baseline_diff_hash=args.baseline_diff_hash,
            candidate_diff_hash=args.candidate_diff_hash,
            model_install=args.model_install,
            chunk_size=args.chunk_size,
            pair_count=args.pairs,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"prefill benchmark failed: {error}", file=sys.stderr)
        return 2

    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered)
    sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
