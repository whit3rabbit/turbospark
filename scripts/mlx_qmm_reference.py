#!/usr/bin/env python3
"""`c(M)` for MLX's own INT4 quantized matmul, at THIS port's matrix shapes.

The KERNEL-QUALITY counterpart to `mlx_prefill.py`, which compares whole
prompts. That script established the 4.94x prefill deficit; this one asks
which half of it belongs to the kernel by running the reference engine's
GEMM on the same seven matrices, on the same machine, against the same
yardstick `crates/gpu/tests/gemv_bandwidth_bench.rs` uses.

**WHY IT EXISTS.** `docs/BENCHMARKS.md` decomposes that deficit into ~2.3x of
micro-batch WIDTH and ~2.2x of kernel QUALITY, and prices the second by
comparing this port's `c(M)` at M=16 against mlx-lm's at M=512. Those are
different widths, so the comparison carries an assumption: that MLX's curve
is still falling between them. Nothing here has measured it. If MLX's `c(M)`
is also flat below M=64 then the width term is most of the gap and a better
kernel buys little; if MLX keeps falling where this port plateaus, the kernel
term is real and the shape difference is the lead. One table settles it.

**IT IS THE SAME `c` AS THE RUST BENCH, or the two are not comparable:**

    c(M) = (time for one call covering M rows) / M / (time for one M=1 call)

`1/M` would be free batching, `1.00` would mean batching bought nothing. The
Rust side computes it in `c_of_m_at_qwen38_shapes` and
`c_of_r_and_m_for_the_batched_kernel`; read those tables beside this one.

**WHAT IT DOES NOT MEASURE.** MLX's kernel is `qmm_t_impl`
(`mlx/backend/metal/kernels/quantized.h`): 128 threads, `WM = WN = 2`, both
operands staged into threadgroup memory, `simdgroup_matrix` MMA. This port's
batched kernel is scalar FP32 `fma`. So a gap here is a gap between two
DIFFERENT kernels and does not isolate any one design choice; it bounds what
is available, which is the question that decides whether to go looking.

**RUN CONDITIONS, and they are not optional** (AGENTS.md Gotchas 20, 22, 28,
43). AC power, a quiet machine, warmup discarded. `c` is a within-process
ratio over interleaved shapes, which is the shape that survives contention
best, but CPU contention still depresses both arms -- and the contaminating
load can be the session watching the run, so leave it alone. The header
reports the power source and load average so no table can silently be a
contaminated one. It does NOT report thermal pressure: `pmset -g therm` is
not that instrument (Gotcha 28), and the two that are need sudo or Rust.

Usage (the env is ephemeral, nothing is installed into this workspace):

    uv run --python 3.12 --with mlx -- python scripts/mlx_qmm_reference.py

    uv run --python 3.12 --with mlx -- python scripts/mlx_qmm_reference.py \
      --rounds 3 --max-m 128
"""

import argparse
import os
import platform
import subprocess
import sys
import time

import mlx.core as mx

# The seven matrices of `crates/gpu/tests/gemv_bandwidth_bench.rs`'s
# QWEN38_SHAPES, as (label, rows, cols) = (label, N, K). Copied rather than
# imported for the obvious reason, so a change there must be mirrored here;
# `--shapes` exists to check a subset without editing this list.
QWEN38_SHAPES = [
    ("gate/up    17408x5120", 17408, 5120),
    ("down       5120x17408", 5120, 17408),
    ("gdn_inproj 16480x5120", 16480, 5120),
    ("packed_q   12288x5120", 12288, 5120),
    ("o_proj      5120x6144", 5120, 6144),
    ("head      248320x5120", 248320, 5120),
    ("reference  32768x5120", 32768, 5120),
]

# This port's install is 4-bit affine at group 64 (`mlx-community/
# Qwen3.8-27B-4bit`), which is what makes the comparison controlled.
BITS = 4
GROUP_SIZE = 64

# The widths worth knowing. 16 is `MAX_PREFILL_BATCH` / `gpu::MAX_BATCH_ROWS`,
# 512 is mlx-lm's own default `prefill_step_size`, and everything between is
# where the published "width saturates at M=36" claim lives.
BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

# Each timed region is sized to last at least this long, so one DVFS state
# covers it. The Rust bench pins equal BYTES per call for the same reason;
# here the reps are calibrated per (shape, M) because the output slab, not
# the weight read, is what bounds how many fit in memory.
TARGET_SECONDS = 0.10
# Cap on the outputs held live inside one timed region. At M=512 against the
# 248320-row head a single output is 254 MB, so without this the head row
# would OOM rather than report.
MAX_OUTPUT_BYTES = 1 << 30


def machine_header() -> str:
    """Power source and load, the two contamination tells that need no sudo."""
    try:
        batt = subprocess.run(
            ["pmset", "-g", "batt"], capture_output=True, text=True, timeout=5
        ).stdout
        power = "AC" if "AC Power" in batt else "BATTERY"
    except Exception:
        power = "unknown"
    load = " ".join(f"{v:.2f}" for v in os.getloadavg())
    return (
        f"machine: {platform.machine()}  mlx {mx.__version__}  "
        f"power: {power}  load: {load}"
    )


def build_shape(rows: int, cols: int):
    """One quantized matrix, allocated ONCE per shape and reused.

    Allocating inside a timed region measures the allocation and the GPU's
    first-touch faults rather than the kernel; the Rust bench's `ShapeBuffers`
    carries the same warning after reading 35 then 68 GiB/s for one shape in
    consecutive rounds.
    """
    w = mx.random.normal((rows, cols)).astype(mx.float16)
    quantized = mx.quantize(w, group_size=GROUP_SIZE, bits=BITS)
    # MLX has returned 3 values historically and 4 since the `mode` argument
    # landed. Take the first three rather than unpacking, so a version bump
    # is not a crash in a script whose whole job is to be a stable reference.
    w_q, scales, biases = quantized[0], quantized[1], quantized[2]
    mx.eval(w_q, scales, biases)
    del w
    return w_q, scales, biases


def timed(w_q, scales, biases, x, reps: int) -> float:
    """`reps` back-to-back matmuls in one lazy graph, then one eval.

    The outputs are kept in a list rather than reduced: a `sum` or an `add`
    per rep would put an elementwise pass inside the region being timed, and
    at these shapes that is not negligible. `mx.synchronize` after the eval
    because `eval` returning is not the GPU being done.
    """
    outs = [
        mx.quantized_matmul(
            x, w_q, scales, biases, transpose=True, group_size=GROUP_SIZE, bits=BITS
        )
        for _ in range(reps)
    ]
    mx.eval(outs)
    mx.synchronize()
    start = time.perf_counter()
    outs = [
        mx.quantized_matmul(
            x, w_q, scales, biases, transpose=True, group_size=GROUP_SIZE, bits=BITS
        )
        for _ in range(reps)
    ]
    mx.eval(outs)
    mx.synchronize()
    return time.perf_counter() - start


def calibrate(w_q, scales, biases, x, rows: int, m: int) -> int:
    """Reps for a ~TARGET_SECONDS region, bounded by the output slab."""
    per_rep = timed(w_q, scales, biases, x, 2) / 2
    want = max(2, int(TARGET_SECONDS / max(per_rep, 1e-9)))
    cap = max(2, int(MAX_OUTPUT_BYTES / (m * rows * 2)))
    return min(want, cap)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--max-m", type=int, default=512)
    ap.add_argument("--shapes", type=str, default="")
    args = ap.parse_args()

    shapes = QWEN38_SHAPES
    if args.shapes:
        wanted = [s.strip() for s in args.shapes.split(",")]
        shapes = [s for s in shapes if any(w in s[0] for w in wanted)]
    widths = [m for m in BATCH_SIZES if m <= args.max_m]

    print()
    print(machine_header())
    print(
        f"c(M) = (time for M rows) / M / (time for one M=1 call);  "
        f"{BITS}-bit affine, group {GROUP_SIZE}, "
        f"best of {args.rounds} interleaved rounds"
    )
    print()

    # Build every shape first, so no allocation lands between timed regions.
    built = []
    for label, rows, cols in shapes:
        try:
            built.append((label, rows, cols, build_shape(rows, cols)))
        except Exception as exc:  # noqa: BLE001
            print(f"{label}  SKIPPED: {type(exc).__name__}: {exc}")
    if not built:
        return 1

    xs = {}
    for _, _, cols, _ in built:
        for m in widths:
            if (m, cols) not in xs:
                a = mx.random.normal((m, cols)).astype(mx.float16)
                mx.eval(a)
                xs[(m, cols)] = a

    reps = {}
    for label, rows, cols, (w_q, scales, biases) in built:
        for m in widths:
            reps[(label, m)] = calibrate(
                w_q, scales, biases, xs[(m, cols)], rows, m
            )

    # Interleave: one full pass over every (shape, M) cell per round, rather
    # than three consecutive rounds per cell. Thermal drift then lands on the
    # whole table instead of on whichever cell was measured last
    # (CLAUDE.local.md's A/B rule, applied inside one process).
    best = {}
    for _ in range(args.rounds):
        for label, rows, cols, (w_q, scales, biases) in built:
            for m in widths:
                n = reps[(label, m)]
                per_call = timed(w_q, scales, biases, xs[(m, cols)], n) / n
                key = (label, m)
                best[key] = min(best.get(key, float("inf")), per_call)

    header = "shape                  " + "".join(f"  M={m:<4}" for m in widths)
    print(header)
    for label, _, _, _ in built:
        base = best[(label, 1)]
        row = "".join(f"  {best[(label, m)] / m / base:>5.3f}" for m in widths)
        print(f"{label}{row}")

    print()
    print("ms per call at each width (the same cells, unnormalized):")
    print(header)
    for label, _, _, _ in built:
        row = "".join(f"  {best[(label, m)] * 1e3:>5.2f}" for m in widths)
        print(f"{label}{row}")

    print()
    print(
        "Read against `c_of_m_at_qwen38_shapes` and\n"
        "`c_of_r_and_m_for_the_batched_kernel` in\n"
        "crates/gpu/tests/gemv_bandwidth_bench.rs. If this table is FLAT below\n"
        "M=64 the width term dominates and a better kernel buys little; if it\n"
        "keeps falling where this port plateaus at ~0.44, the kernel term is\n"
        "real and MLX's tile shape is the lead worth following.\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
