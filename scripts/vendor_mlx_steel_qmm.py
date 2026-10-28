#!/usr/bin/env python3
"""Regenerate crates/gpu/src/shaders/dequant_int4_steel.metal from pinned MLX.

The output is MLX's Steel GEMM headers plus its own `QuantizedBlockLoader`
and `qmm_t_impl` (mlx/backend/metal/kernels/quantized.h), all MIT licensed,
adapted to this engine's resident INT4 layout:

  * scales and biases are BF16 planes, read as FP32 and only then rounded to
    the FP16 matrix operand (reinterpreting them as half would change the
    checkpoint's values);
  * the weight pointer is a byte pointer, because resident INT4 tensors sit
    at 2-byte aligned offsets (`dequant_int4.metal`'s alignment note) and a
    `uint32_t*` at such an address is undefined;
  * INT4, group 64 only.

The originals are fetched by commit and checked against the SHA-256 values
below, so a changed upstream file fails loudly instead of being vendored.
The output is ASCII; upstream's copyright symbol becomes "(c)".

Usage:  python3 -I scripts/vendor_mlx_steel_qmm.py
"""

import hashlib
import re
import urllib.request
from pathlib import Path

MLX_COMMIT = "ce45c52505c8158ea48d2a54e8caae05efd86bfe"  # v0.31.1
REPO = Path(__file__).resolve().parents[1]
OUTPUT = REPO / "crates/gpu/src/shaders/dequant_int4_steel.metal"
ROOTS = [
    "mlx/backend/metal/kernels/utils.h",
    "mlx/backend/metal/kernels/steel/gemm/gemm.h",
]
QUANTIZED = "mlx/backend/metal/kernels/quantized.h"
# Pinned identities of every upstream file the output is built from. A
# mismatch means the pin no longer describes the bytes; refuse rather than
# vendor something nobody reviewed.
EXPECTED_SHA256 = {
    QUANTIZED: "344780f5b159669219f993f9705b5f28504196d3fd2c7ab8756119c72e4e267f",
}
TILES = [(32, 32, 32), (64, 32, 64)]
# Tiles that also get an `_edge` entry: `aligned_N` false, so the last
# threadgroup column may be a partial tile (`load_safe` on the weights and a
# masked store). Full tiles take the same arithmetic as the aligned entry.
EDGE_TILES = [(32, 32, 32)]

fetched = {}


def fetch(path):
    if path not in fetched:
        url = f"https://raw.githubusercontent.com/ml-explore/mlx/{MLX_COMMIT}/{path}"
        with urllib.request.urlopen(url, timeout=30) as response:
            data = response.read()
        digest = hashlib.sha256(data).hexdigest()
        want = EXPECTED_SHA256.get(path)
        if want is not None and digest != want:
            raise SystemExit(f"{path}: sha256 {digest} does not match the pinned {want}")
        fetched[path] = (data.decode(), digest)
    return fetched[path][0]


def kernel_entry(bm, bk, bn, edge):
    """One `[[kernel]]` entry; `edge` instantiates `aligned_N = false`."""
    suffix = "_edge" if edge else ""
    aligned = "false" if edge else "true"
    return f"""
[[kernel, max_total_threads_per_threadgroup(128)]]
void dequant_int4_steel_{bm}_{bk}_{bn}{suffix}(
    const device uint8_t* w [[buffer(0)]],
    const device bfloat16_t* scales [[buffer(1)]],
    const device bfloat16_t* biases [[buffer(2)]],
    const device half* x [[buffer(3)]],
    device half* y [[buffer(4)]],
    constant int& K [[buffer(5)]],
    constant int& N [[buffer(6)]],
    constant int& M [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {{
    threadgroup half Xs[{bm} * ({bk} + 8)];
    threadgroup half Ws[{bn} * ({bk} + 8)];
    qmm_t_impl<half, 64, 4, {aligned}, {bm}, {bk}, {bn}>(
        w, scales, biases, x, y, Xs, Ws, K, N, M, tid, lid, simd_gid, simd_lid);
}}
"""


emitted = set()


def flatten(path):
    """Inline `#include "..."` recursively, each header once, in order."""
    if path in emitted:
        return ""
    emitted.add(path)
    source = fetch(path)
    source = re.sub(r"^#pragma once\s*$", "", source, flags=re.M)
    base = path.rsplit("/", 1)[0]

    def resolve(match):
        name = match[1]
        # MLX includes kernel headers by their path from the repository root.
        target = name if name.startswith("mlx/") else f"{base}/{name}"
        return flatten(target)

    return re.sub(r'^#include "([^"]+)"', resolve, source, flags=re.M)


def replace_once(text, old, new, expected=1):
    count = text.count(old)
    if count != expected:
        raise SystemExit(
            f"adaptation anchor found {count} times, expected {expected}: {old!r}"
        )
    return text.replace(old, new)


def main():
    headers = "".join(flatten(root) for root in ROOTS)
    quantized = fetch(QUANTIZED)

    start = quantized.index("template <\n    typename T,\n    short BROWS,")
    end = quantized.index("template <", start + 10)
    loader = quantized[start:end]
    if "struct QuantizedBlockLoader" not in loader:
        raise SystemExit("pinned quantized.h no longer starts QuantizedBlockLoader here")
    # The member and the constructor parameter of each plane, then the two
    # reads in load_unsafe and load_safe.
    for name in ("scales", "biases"):
        loader = replace_once(
            loader, f"const device T* {name};", f"const device bfloat16_t* {name};"
        )
        loader = replace_once(
            loader, f"const device T* {name}_,", f"const device bfloat16_t* {name}_,"
        )
    loader = replace_once(loader, "T scale = *scales;", "float scale = float(*scales);", 2)
    loader = replace_once(loader, "T bias = *biases;", "float bias = float(*biases);", 2)
    loader = replace_once(
        loader, "dequantize<T, pack_factor, bits>", "ts_dequant4<T, pack_factor, bits>", 2
    )

    qmm_start = quantized.index("METAL_FUNC void qmm_t_impl(")
    qmm_start = quantized.rindex("template <", 0, qmm_start)
    qmm_end = quantized.index("template <", qmm_start + 10)
    qmm = quantized[qmm_start:qmm_end]
    qmm = replace_once(qmm, "const device uint32_t* w,", "const device uint8_t* w,")
    qmm = replace_once(qmm, "const device T* scales,", "const device bfloat16_t* scales,")
    qmm = replace_once(qmm, "const device T* biases,", "const device bfloat16_t* biases,")
    qmm = replace_once(qmm, "auto wl = (const device uint8_t*)w;", "auto wl = w;")

    helpers = """
// ---- turbospark adaptation (not upstream) --------------------------------
using namespace metal;
#define MLX_MTL_CONST static constant constexpr const
MLX_MTL_CONST int SIMD_SIZE = 32;
template <int bits, int wsize = 8> inline constexpr short get_pack_factor() {
    static_assert(bits == 4, "INT4 only"); return wsize / bits;
}
template <int bits, int wsize = 8> inline constexpr short get_bytes_per_pack() {
    static_assert(bits == 4, "INT4 only"); return wsize / 8;
}
// Affine INT4 dequantization in FP32 from the BF16 companions, rounded once
// to the FP16 operand the matrix units consume.
template <typename T, int N, int bits>
inline void ts_dequant4(const device uint8_t* w, float scale, float bias,
                        threadgroup T* dst) {
    static_assert(bits == 4, "INT4 only");
    for (int i = 0; i < N / 2; ++i) {
        dst[2 * i] = T(float(w[i] & 15) * scale + bias);
        dst[2 * i + 1] = T(float(w[i] >> 4) * scale + bias);
    }
}
// ---- upstream quantized.h: QuantizedBlockLoader and qmm_t_impl -----------
"""

    kernels = """
// ---- turbospark kernel entries (not upstream) -----------------------------
// y[M, N] = x[M, K] * dequant(w[N, K])^T with FP16 x and y, packed INT4 w,
// BF16 scale and bias planes at group 64, FP32 accumulation. Grid is
// (ceil(N / BN), ceil(M / BM)) threadgroups of 128 threads. N must be a
// multiple of BN (aligned_N) except in the `_edge` entries, which accept any
// N; M is any size (rows past M are masked).
"""
    for bm, bk, bn in TILES:
        for edge in (False, True):
            if edge and (bm, bk, bn) not in EDGE_TILES:
                continue
            kernels += kernel_entry(bm, bk, bn, edge)

    license_text = fetch("LICENSE")
    provenance = "\n".join(
        f"//   {path}  sha256 {digest}" for path, (_, digest) in sorted(fetched.items())
    )
    banner = f"""// GENERATED by scripts/vendor_mlx_steel_qmm.py -- do not edit by hand.
//
// INT4 matrix-unit GEMM for prefill: Apple MLX's Steel GEMM machinery and its
// quantized `qmm_t_impl`, adapted to this engine's resident INT4 layout
// (BF16 scale/bias planes, byte weight pointer). Not bit-exact against
// `dequant_int4_gemm_simd`: the weight is rounded to an FP16 operand and the
// reduction order is the matrix units'. Prefill-only; see
// docs/BATCHED_PREFILL.md for the gate it is held to.
//
// Upstream: https://github.com/ml-explore/mlx at {MLX_COMMIT} (v0.31.1).
// Files, as fetched:
{provenance}
//
// MLX license (MIT), retained as its terms require:
//
""" + "\n".join(f"//   {line}".rstrip() for line in license_text.splitlines()) + "\n\n"

    source = banner + headers + helpers + loader + qmm + kernels
    source = source.replace("©", "(c)")
    non_ascii = sorted({ch for ch in source if ord(ch) > 127})
    if non_ascii:
        raise SystemExit(f"output still has non-ASCII characters: {non_ascii!r}")
    OUTPUT.write_text(source)
    print(f"wrote {OUTPUT.relative_to(REPO)} ({len(source.splitlines())} lines)")


if __name__ == "__main__":
    main()
