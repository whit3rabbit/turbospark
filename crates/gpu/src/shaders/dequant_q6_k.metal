#include <metal_stdlib>
using namespace metal;

// ============================================================================
// dequant_q6_k - GGUF Q6_K dequant. PORT-LOCAL, not vendored, for the same
// reason as its Q8_0 and Q4_K siblings: the Swift engine has no GGUF intake.
// Its contract is `mrefrust_compute::dequant_q6_k_gemv` and nothing else.
//
// Layout per weight row of length N, exactly the bytes GGUF stores and the
// repack walk copies through untouched:
//   superblock : 210 bytes over 256 elements =
//                  128 low-nibble bytes, 64 high-bit bytes,
//                  16 SIGNED int8 sub-block scales, f16 super-scale.
//   sub-block  : 16 elements, sixteen per superblock, one int8 scale each.
//   value      : w = d * sc[j] * (level - 32), level assembled from four low
//                bits in `ql` and two high bits in `qh`.
//
// Qwen 3.6's Q4_K_M carries exactly one Q6_K tensor (`output.weight`), which
// is the whole reason this exists. Four properties are silently wrong rather
// than a crash if carried over from either sibling:
//   1. Six bits per element are SPLIT ACROSS TWO RUNS. Reading `ql` alone
//      gives values in 0..16 where 0..64 was meant.
//   2. The quant is BIASED, not unsigned: 32 is subtracted from the stored
//      level. There is no per-sub-block min, unlike Q4_K.
//   3. The sixteen sub-block scales are SIGNED int8 plain bytes, and real
//      files carry negative ones. Reading them as uchar mirrors whole
//      16-element runs.
//   4. A superblock is two halves of 128 elements, and the four values one
//      `qh` byte serves sit 32 elements apart under scales `is`, `is + 2`,
//      `is + 4` and `is + 6` -- a stride of two, not one.
//
// The super-scale is loaded byte by byte, like the siblings: the alignment of
// a 210-byte stride is a property of the constant rather than of the format,
// and 210 is not even a multiple of 4.
// ============================================================================

constant constexpr uint kQ6_KBlockElems = 256;
constant constexpr uint kQ6_KSubElems = 16;
constant constexpr uint kQ6_KBlockBytes = 210;
constant constexpr uint kQ6_KQhAt = 128;
constant constexpr uint kQ6_KScalesAt = 192;
constant constexpr uint kQ6_KDAt = 208;
constant constexpr uint kQ6_KHalfElems = 128;
constant constexpr uint kRowsPerTGQ6_K = 8;

// One Q6_K output row dotted against `x`. 32 lanes over a 128-element half:
// each lane owns one `qh` byte and the four elements it serves, so a
// superblock is two passes and the whole row has no inner stride.
static inline float dequant_q6_k_row_simd(
    device const uint8_t* W_row,
    device const half* x,
    uint N,
    uint lane
) {
    const uint n_blocks = N / kQ6_KBlockElems;

    float acc = 0.0f;
    for (uint b = 0; b < n_blocks; ++b) {
        device const uint8_t* blk = W_row + b * kQ6_KBlockBytes;
        const ushort d_raw = ushort(blk[kQ6_KDAt]) | (ushort(blk[kQ6_KDAt + 1]) << 8);
        const float d = float(as_type<half>(d_raw));

        for (uint h = 0; h < kQ6_KBlockElems / kQ6_KHalfElems; ++h) {
            device const uint8_t* ql = blk + h * 64;
            device const uint8_t* qh = blk + kQ6_KQhAt + h * 32;
            device const uint8_t* sc = blk + kQ6_KScalesAt + h * 8;
            const uint is = lane / kQ6_KSubElems;

            const uint8_t lo = ql[lane];
            const uint8_t hi = ql[lane + 32];
            const uint8_t bits = qh[lane];
            // int(...) - 32 is note 2 in the header; as_type<int8_t> on the
            // scale is note 3.
            const float q0 = float(int((lo & 0xF) | ((bits & 3) << 4)) - 32);
            const float q1 = float(int((hi & 0xF) | (((bits >> 2) & 3) << 4)) - 32);
            const float q2 = float(int((lo >> 4) | (((bits >> 4) & 3) << 4)) - 32);
            const float q3 = float(int((hi >> 4) | (((bits >> 6) & 3) << 4)) - 32);

            const uint at = b * kQ6_KBlockElems + h * kQ6_KHalfElems + lane;
            acc = fma(d * float(as_type<int8_t>(sc[is])) * q0, float(x[at]), acc);
            acc = fma(d * float(as_type<int8_t>(sc[is + 2])) * q1, float(x[at + 32]), acc);
            acc = fma(d * float(as_type<int8_t>(sc[is + 4])) * q2, float(x[at + 64]), acc);
            acc = fma(d * float(as_type<int8_t>(sc[is + 6])) * q3, float(x[at + 96]), acc);
        }
    }
    return simd_sum(acc);
}

// y[m] = sum_n W[m, n] * x[n]. One SIMD group per output row. Dispatch:
// threadgroupsPerGrid = (ceil(M / 8), 1, 1), threadsPerThreadgroup = (256,1,1).
[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void dequant_q6_k_gemv_simd(
    device const uint8_t* W      [[buffer(0)]],
    device const half*    x      [[buffer(1)]],
    device half*          y      [[buffer(2)]],
    constant uint&        M      [[buffer(3)]],
    constant uint&        N      [[buffer(4)]],
    uint                  tg_idx [[threadgroup_position_in_grid]],
    uint                  sg_idx [[simdgroup_index_in_threadgroup]],
    uint                  lane   [[thread_index_in_simdgroup]]
) {
    const uint row = tg_idx * kRowsPerTGQ6_K + sg_idx;
    if (row >= M) return;

    const uint n_blocks = N / kQ6_KBlockElems;
    device const uint8_t* W_row = W + uint(row) * n_blocks * kQ6_KBlockBytes;

    const float acc = dequant_q6_k_row_simd(W_row, x, N, lane);
    if (lane == 0) {
        y[row] = half(acc);
    }
}
