#include <metal_stdlib>
using namespace metal;

// ============================================================================
// dequant_q8_0 - GGUF Q8_0 dequant. PORT-LOCAL, not vendored: the Swift
// engine has no GGUF intake, so there is no upstream kernel to mirror. Its
// contract is `mrefrust_compute::dequant_q8_0_gemv` and nothing else.
//
// Layout per weight row of length N, exactly the bytes GGUF stores and the
// repack walk copies through untouched:
//   block   : 34 bytes = one f16 scale (little-endian) then 32 SIGNED weights.
//   row     : N/32 such blocks, contiguous.
//   value   : w[i] = float(int8(q[i])) * d[i/32].
//
// Three ways this differs from the affine INT8 sibling next door, each of
// which is silently wrong rather than a crash if carried over by habit:
//   1. Quants are SIGNED. Reading them as uchar mirrors half the weights.
//   2. The scale is INSIDE the block, not in a separate plane at group 64.
//   3. There is no bias term at all.
//
// The scale is loaded byte by byte rather than through a `device const half*`.
// A block stride of 34 is even, so a half load would in fact be aligned, but
// that is a property of the constant rather than of the format, and a future
// block type with an odd stride would fault rather than mis-round.
// ============================================================================

constant constexpr uint kQ8_0BlockElems = 32;
constant constexpr uint kQ8_0BlockBytes = 34;
constant constexpr uint kRowsPerTGQ8_0 = 8;

// y[m] = sum_n W[m, n] * x[n]. One SIMD group per output row: 32 lanes over a
// 32-element block, so each lane owns exactly one weight per block and the
// index arithmetic has no inner stride. Dispatch:
// threadgroupsPerGrid = (ceil(M / 8), 1, 1), threadsPerThreadgroup = (256,1,1).
[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void dequant_q8_0_gemv_simd(
    device const uint8_t* W      [[buffer(0)]],
    device const half*    x      [[buffer(1)]],
    device half*          y      [[buffer(2)]],
    constant uint&        M      [[buffer(3)]],
    constant uint&        N      [[buffer(4)]],
    uint                  tg_idx [[threadgroup_position_in_grid]],
    uint                  sg_idx [[simdgroup_index_in_threadgroup]],
    uint                  lane   [[thread_index_in_simdgroup]]
) {
    const uint row = tg_idx * kRowsPerTGQ8_0 + sg_idx;
    if (row >= M) return;

    const uint n_blocks = N / kQ8_0BlockElems;
    device const uint8_t* W_row = W + uint(row) * n_blocks * kQ8_0BlockBytes;

    float acc = 0.0f;
    for (uint b = 0; b < n_blocks; ++b) {
        device const uint8_t* blk = W_row + b * kQ8_0BlockBytes;
        ushort raw = ushort(blk[0]) | (ushort(blk[1]) << 8);
        float d = float(as_type<half>(raw));
        // int8_t, not uint8_t: see note 1 in the header.
        float q = float(int(as_type<int8_t>(blk[2 + lane])));
        float xv = float(x[b * kQ8_0BlockElems + lane]);
        acc = fma(d * q, xv, acc);
    }
    acc = simd_sum(acc);
    if (lane == 0) {
        y[row] = half(acc);
    }
}

// One row of a Q8_0 embedding table, dequantized into `out` and scaled.
// Sibling of `embed_lookup_int4` in `dequant_int4.metal`, and simpler for
// the usual Q8_0 reason: there are no scale or bias planes to index, so the
// row is one contiguous byte run and the scale comes from inside the block.
// One thread per element; `out_scale` is sqrt(hidden) for Gemma's scaled
// embeddings, 1.0 otherwise.
// KERN-04 Stage A bench arms for Q8_0, NOT used by the engine. A 34-byte block
// keeps block starts only 2-byte aligned, so the loads are `ushort`, not
// `uint`: each lane owns 4 consecutive quants of one block (two ushort loads,
// one ushort for the f16 scale, one half4 of x), 8 lanes per block, 4 blocks
// per simdgroup step. The scale is factored out of the block's 4-term sum
// (`d * sum(q * x)`). `_r2` runs two rows per simdgroup that share the x load
// (llama.cpp's NR0 = 2), 16 rows per threadgroup. Both read `W` 2-byte aligned
// and `x` 8-byte aligned; the host checks. A block count that is not a
// multiple of 4 is handled by the stride loop. The bits differ from the simd
// kernel (different reduction).
static inline float q8_0_wide_dot(device const uint8_t* blk, half4 xv, uint l) {
    const char2 a = as_type<char2>(*(device const ushort*)(blk + 2 + l));
    const char2 c = as_type<char2>(*(device const ushort*)(blk + 4 + l));
    float s = float(a.x) * float(xv[0]);
    s = fma(float(a.y), float(xv[1]), s);
    s = fma(float(c.x), float(xv[2]), s);
    s = fma(float(c.y), float(xv[3]), s);
    return s;
}

static inline float q8_0_scale(device const uint8_t* blk) {
    return float(as_type<half>(*(device const ushort*)blk));
}

[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void dequant_q8_0_gemv_wide(
    device const uint8_t* W      [[buffer(0)]],
    device const half*    x      [[buffer(1)]],
    device half*          y      [[buffer(2)]],
    constant uint&        M      [[buffer(3)]],
    constant uint&        N      [[buffer(4)]],
    uint                  tg_idx [[threadgroup_position_in_grid]],
    uint                  sg_idx [[simdgroup_index_in_threadgroup]],
    uint                  lane   [[thread_index_in_simdgroup]]
) {
    const uint row = tg_idx * kRowsPerTGQ8_0 + sg_idx;
    if (row >= M) return;
    const uint n_blocks = N / kQ8_0BlockElems;
    device const uint8_t* W_row = W + uint(row) * n_blocks * kQ8_0BlockBytes;
    const uint k = lane / 8u;
    const uint l = (lane % 8u) * 4u;
    float acc = 0.0f;
    for (uint b = k; b < n_blocks; b += 4u) {
        device const uint8_t* blk = W_row + b * kQ8_0BlockBytes;
        const half4 xv = *(device const half4*)(x + b * kQ8_0BlockElems + l);
        acc = fma(q8_0_scale(blk), q8_0_wide_dot(blk, xv, l), acc);
    }
    acc = simd_sum(acc);
    if (lane == 0) {
        y[row] = half(acc);
    }
}

[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void dequant_q8_0_gemv_wide_r2(
    device const uint8_t* W      [[buffer(0)]],
    device const half*    x      [[buffer(1)]],
    device half*          y      [[buffer(2)]],
    constant uint&        M      [[buffer(3)]],
    constant uint&        N      [[buffer(4)]],
    uint                  tg_idx [[threadgroup_position_in_grid]],
    uint                  sg_idx [[simdgroup_index_in_threadgroup]],
    uint                  lane   [[thread_index_in_simdgroup]]
) {
    const uint row0 = (tg_idx * kRowsPerTGQ8_0 + sg_idx) * 2u;
    if (row0 >= M) return;
    const bool has1 = row0 + 1u < M;
    const uint n_blocks = N / kQ8_0BlockElems;
    const uint row_bytes = n_blocks * kQ8_0BlockBytes;
    device const uint8_t* W0 = W + uint(row0) * row_bytes;
    device const uint8_t* W1 = has1 ? W0 + row_bytes : W0;
    const uint k = lane / 8u;
    const uint l = (lane % 8u) * 4u;
    float acc0 = 0.0f, acc1 = 0.0f;
    for (uint b = k; b < n_blocks; b += 4u) {
        const half4 xv = *(device const half4*)(x + b * kQ8_0BlockElems + l);
        device const uint8_t* b0 = W0 + b * kQ8_0BlockBytes;
        device const uint8_t* b1 = W1 + b * kQ8_0BlockBytes;
        acc0 = fma(q8_0_scale(b0), q8_0_wide_dot(b0, xv, l), acc0);
        acc1 = fma(q8_0_scale(b1), q8_0_wide_dot(b1, xv, l), acc1);
    }
    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);
    if (lane == 0) {
        y[row0] = half(acc0);
        if (has1) {
            y[row0 + 1u] = half(acc1);
        }
    }
}

kernel void embed_lookup_q8_0(
    device const uint8_t* table     [[buffer(0)]],   // [V, D/32 * 34] blocks
    device half*          out       [[buffer(1)]],   // [D] FP16
    constant uint&        token_id  [[buffer(2)]],
    constant uint&        D         [[buffer(3)]],
    constant float&       out_scale [[buffer(4)]],
    uint                  gid       [[thread_position_in_grid]]
) {
    if (gid >= D) return;
    const uint row_bytes = (D / kQ8_0BlockElems) * kQ8_0BlockBytes;
    device const uint8_t* blk =
        table + uint(token_id) * row_bytes + (gid / kQ8_0BlockElems) * kQ8_0BlockBytes;
    const ushort raw = ushort(blk[0]) | (ushort(blk[1]) << 8);
    const float d = float(as_type<half>(raw));
    const float q = float(int(as_type<int8_t>(blk[2 + (gid % kQ8_0BlockElems)])));
    out[gid] = half(q * d * out_scale);
}
