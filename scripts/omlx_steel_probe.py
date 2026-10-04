#!/usr/bin/env python3
"""Prepare and run an isolated Metal experiment, outside the product path.

Pins oMLX and MLX sources, adapts Steel's INT4 loader to our BF16 companions
and FP16 activations, then compares it with our current batched shader.
Results are synthetic kernel evidence, not real-model quality or throughput.
See docs/MLX_KERNELS.md for the evidence boundary and promotion gates.
"""

import argparse
import hashlib
import json
import re
import statistics
import subprocess
import time
import urllib.request
from pathlib import Path

OMLX_COMMIT = "68c8c09f6f6a54fe36181f9aabba9525797d1b58"
MLX_COMMIT = "ce45c52505c8158ea48d2a54e8caae05efd86bfe"  # v0.31.1
REPO = Path(__file__).resolve().parents[1]
QUANTIZED = "omlx/custom_kernels/common/csrc/kernels/quantized_moe.h"
UTILS = "mlx/backend/metal/kernels/utils.h"
GEMM = "mlx/backend/metal/kernels/steel/gemm/gemm.h"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def prepare(output):
    identities = {}
    emitted = set()

    def fetch(repo, commit, path):
        url = f"https://raw.githubusercontent.com/{repo}/{commit}/{path}"
        with urllib.request.urlopen(url, timeout=30) as response:
            data = response.read()
        target = output / "upstream" / repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        identities[f"{repo}/{path}"] = {"commit": commit, "sha256": digest(data)}
        return data.decode()

    def flatten(path):
        if path in emitted:
            return ""
        emitted.add(path)
        source = fetch("ml-explore/mlx", MLX_COMMIT, path)
        source = re.sub(r'^#pragma once\s*$', "", source, flags=re.M)
        return re.sub(
            r'^#include "([^"]+)"', lambda m: flatten(m[1]), source, flags=re.M
        )

    header = flatten(UTILS) + flatten(GEMM)
    quantized = fetch("jundot/omlx", OMLX_COMMIT, QUANTIZED)
    # Keep upstream originals and the adapted composition separately so the
    # interface and arithmetic changes remain inspectable.
    start = quantized.index("template <\n    typename T,\n    short BROWS,")
    end = quantized.index("template <typename T, int group_size, int bits, int D>", start)
    loader = quantized[start:end]
    loader = loader.replace("const device T* scales", "const device bfloat16_t* scales")
    loader = loader.replace("const device T* biases", "const device bfloat16_t* biases")
    loader = loader.replace("T scale = *scales;", "float scale = float(*scales);")
    loader = loader.replace("T bias = *biases;", "float bias = float(*biases);")
    loader = loader.replace("dequantize<T, pack_factor, bits>", "ts_dequant4<T, pack_factor, bits>")
    qmm_start = quantized.index("template <\n    typename T,\n    const int group_size,", end)
    # This is qmm_t_impl, not a reconstruction of its tile machinery.
    qmm_end = quantized.index("template <", qmm_start + 10)
    qmm = quantized[qmm_start:qmm_end]
    if "METAL_FUNC void qmm_t_impl(" not in qmm:
        raise ValueError("pinned source no longer matches qmm_t_impl")
    qmm = qmm.replace("const device T* scales", "const device bfloat16_t* scales")
    qmm = qmm.replace("const device T* biases", "const device bfloat16_t* biases")
    helpers = """
// Derived from Apple MLX / oMLX, Apache-2.0. Originals are retained above.
using namespace metal;
#define MLX_MTL_CONST static constant constexpr const
MLX_MTL_CONST int SIMD_SIZE = 32;
template <int bits, int wsize = 8> inline constexpr short get_pack_factor() {
    static_assert(bits == 4, "probe supports INT4 only"); return wsize / bits;
}
template <int bits, int wsize = 8> inline constexpr short get_bytes_per_pack() {
    static_assert(bits == 4, "probe supports INT4 only"); return wsize / 8;
}
// BF16 scale/bias reads stay FP32 until the matrix operand is staged as FP16.
// Reinterpreting these planes as half would change the checkpoint values.
template <typename T, int N, int bits>
inline void ts_dequant4(const device uint8_t* w, float scale, float bias,
                       threadgroup T* dst) {
    static_assert(bits == 4, "probe supports INT4 only");
    for (int i = 0; i < N / 2; ++i) {
        dst[2*i] = T(float(w[i] & 15) * scale + bias);
        dst[2*i+1] = T(float(w[i] >> 4) * scale + bias);
    }
}
"""
    source = header + helpers + loader + qmm
    tiles = [(32, 32, 32), (32, 64, 64), (64, 32, 64)]
    for bm, bk, bn in tiles:
        source += f"""
kernel void ts_steel_{bm}_{bk}_{bn}(
    const device uint32_t* w [[buffer(0)]],
    const device bfloat16_t* scales [[buffer(1)]],
    const device bfloat16_t* biases [[buffer(2)]],
    const device half* x [[buffer(3)]], device half* y [[buffer(4)]],
    constant int& K [[buffer(5)]], constant int& N [[buffer(6)]],
    constant int& M [[buffer(7)]], uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {{
    threadgroup half Xs[{bm} * ({bk}+8)];
    threadgroup half Ws[{bn} * ({bk}+8)];
    qmm_t_impl<half, 64, 4, true, {bm}, {bk}, {bn}>(
        w, scales, biases, x, y, Xs, Ws, K, N, M, K, tid, lid, sg, lane);
}}
"""
    # Upstream copyright comments use a non-ASCII symbol; keep ASCII build
    # artifacts while preserving the exact originals and both licenses.
    source = source.replace("\u00a9", "(c)")
    (output / "steel.metal").write_text(source)
    files = [REPO / "crates/gpu/src/shaders/dequant_int4.metal",
             REPO / "crates/gpu/src/shaders/dequant_int4_batch.metal",
             REPO / "crates/gpu/src/dequant_int4_batch.rs",
             REPO / "scripts/omlx_steel_probe.swift", Path(__file__).resolve()]
    for path in files:
        identities[str(path.relative_to(REPO))] = {"sha256": digest(path.read_bytes())}
    (output / "control.metal").write_bytes(files[0].read_bytes() + files[1].read_bytes())
    for repo, commit in [("jundot/omlx", OMLX_COMMIT), ("ml-explore/mlx", MLX_COMMIT)]:
        fetch(repo, commit, "LICENSE")
    identities["adapted/steel.metal"] = {"sha256": digest(source.encode())}
    (output / "source-identities.json").write_text(json.dumps(identities, indent=2) + "\n")


def host_sample():
    def command(args):
        return subprocess.run(args, capture_output=True, text=True, timeout=5, check=True).stdout
    return {"unix_time": time.time(), "power": command(["pmset", "-g", "batt"]),
            "processes": command(["ps", "-A", "-o", "pid=,pcpu=,comm=", "-r"])}


def run_probe(binary, output, run):
    samples = []
    with (output / f"results-{run}.jsonl").open("w") as stdout, (output / f"stderr-{run}.log").open("w") as stderr:
        child = subprocess.Popen([str(binary), str(output)], stdout=stdout, stderr=stderr)
        try:
            deadline = time.monotonic() + 300
            while child.poll() is None:
                samples.append(host_sample())
                if time.monotonic() >= deadline:
                    raise TimeoutError("Metal probe exceeded five minutes")
                time.sleep(1)
            samples.append(host_sample())
            if child.returncode:
                raise subprocess.CalledProcessError(child.returncode, child.args)
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()
            (output / f"host-samples-{run}.json").write_text(json.dumps(samples, indent=2) + "\n")


def summarize(output, runs):
    cells = {}
    worst_cpu = 0.0
    worst_gpu = 0.0
    screening = []
    for run in range(runs):
        records = [json.loads(line) for line in (output / f"results-{run}.jsonl").read_text().splitlines()]
        samples = json.loads((output / f"host-samples-{run}.json").read_text())
        hot = sorted({line.strip() for sample in samples for line in sample["processes"].splitlines()
                      if len(line.split(None, 2)) == 3 and float(line.split(None, 2)[1]) >= 50
                      and line.split(None, 2)[2] != str(output / "probe")})
        all_ac = bool(samples) and all("AC Power" in s["power"] for s in samples)
        all_nominal = all(r["thermal"] == 0 for r in records if "thermal" in r)
        screening.append({"run": run, "all_ac": all_ac, "all_nominal": all_nominal,
                          "screen_eligible": all_ac and all_nominal and not hot,
                          "cpu_ge_50": hot, "host": records[0]})
        worst_cpu = max(worst_cpu, max(r["max_row_range_error_cpu"] for r in records if r["kind"] == "parity"))
        worst_gpu = max(worst_gpu, max(r["max_row_range_error_control"] for r in records if r["kind"] == "full_shape_parity"))
        timing = [r for r in records if r["kind"] == "timing"]
        for shape, m in sorted({(r["shape"], r["m"]) for r in timing}):
            group = [r for r in timing if r["shape"] == shape and r["m"] == m]
            medians = {arm: statistics.median(r["gpu_ms"] for r in group if r["arm"] == arm)
                       for arm in sorted({r["arm"] for r in group})}
            cell = cells.setdefault(f"{shape}/M={m}", [])
            cell.append({"run": run, "gpu_ms": medians,
                         "speedup": {arm: medians["control"] / value for arm, value in medians.items() if arm != "control"}})
    result = {"schema_version": 1, "scope": "synthetic kernel experiment; no model or default qualification",
              "runs": runs, "screening": screening, "max_row_range_error_cpu": worst_cpu,
              "max_row_range_error_control": worst_gpu, "parity_bound": 0.005, "cells": cells,
              "eligible_cells": {key: [r for r in values if screening[r["run"]]["screen_eligible"]]
                                 for key, values in cells.items()}}
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new artifact directory")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--runs", type=int, default=3, help="fresh processes (default: 3)")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    prepare(output)
    if args.prepare_only:
        return
    binary = output / "probe"
    subprocess.run(["xcrun", "swiftc", "-O", str(REPO / "scripts/omlx_steel_probe.swift"),
                    "-o", str(binary)], check=True)
    (output / "binary.sha256").write_text(digest(binary.read_bytes()) + "\n")
    for run in range(args.runs):
        run_probe(binary, output, run)
    summarize(output, args.runs)
    print(output / "summary.json")


if __name__ == "__main__":
    main()
