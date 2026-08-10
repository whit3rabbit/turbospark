#include <metal_stdlib>
using namespace metal;

// ============================================================================
// dequant_q5_k - GGUF Q5_K dequant. PORT-LOCAL, not vendored, for the same
// reason as its Q8_0, Q4_K and Q6_K siblings: the Swift engine has no GGUF
// intake, so there is no upstream kernel to mirror. Its contract is
// `turbospark_compute::dequant_q5_k_gemv` and nothing else.
//
// ROADMAP Phase M2 added it for ONE tensor shape: Mixtral 8x7B's Q4_K_M
// carries `attn_output` in Q5_K while its experts are Q4_K. So there is
// deliberately no embedding-lookup and no routed-expert sibling here, the same
// judgement Q6_K's file records -- the real file asks for a resident GEMV and
// nothing else, and shipping more would be speculative.
//
// Layout per weight row of length N, exactly the bytes GGUF stores and the
// repack walk copies through untouched:
//   superblock : 176 bytes over 256 elements =
//                  f16 d, f16 dmin, 12 packed bytes, 32 qh bytes, 128 ql bytes.
//   sub-block  : 32 elements, eight per superblock, each with a 6-bit scale
//                and a 6-bit min, packed exactly as Q4_K packs them.
//   value      : w = d * sc[j] * q - dmin * m[j], with UNSIGNED 5-bit q.
//
// Q5_K is Q4_K plus one bit, and every part of that sentence is a trap:
//   1. The fifth bit lives in its OWN 32-byte run between the packed scales
//      and the nibbles, and it contributes 16 to the quant. Ignoring `qh`
//      halves the dynamic range and leaves values correctly signed,
//      correctly ordered and merely wrong.
//   2. `qh` is indexed by the element's position WITHIN a 32-element
//      sub-block, and the BIT it serves advances with the 64-element group:
//      bit `2g` for the low nibble half and `2g + 1` for the high one. One
//      `qh` byte is therefore read eight times at eight bit positions.
//   3. The per-sub-block MIN is still there and still SUBTRACTED. Q6_K's
//      symmetric bias-32 form is the odd one out among the K-quants, so a
//      habit carried from that sibling turns every weight non-negative.
//
// The 6-bit scale/min unpack is Q4_K's, byte for byte, so this file is
// compiled with `dequant_q4_k.metal` concatenated ahead of it and calls
// `q4_k_scale_min` rather than carrying a second copy of the split. A second
// hand-written copy is exactly the duplication that would let the two
// disagree.
//
// The two super-scales are loaded byte by byte for the same reason the
// siblings' are: alignment of a 176-byte stride is a property of the constant
// rather than of the format.
// ============================================================================

constant constexpr uint kQ5_KBlockElems = 256;
constant constexpr uint kQ5_KSubElems = 32;
constant constexpr uint kQ5_KBlockBytes = 176;
constant constexpr uint kQ5_KScalesAt = 4;
constant constexpr uint kQ5_KQhAt = 16;
constant constexpr uint kQ5_KQuantsAt = 48;
constant constexpr uint kRowsPerTGQ5_K = 8;

// Bytes one Q5_K row of `n` elements occupies.
static inline uint q5_k_row_bytes(uint n) {
    return n / kQ5_KBlockElems * kQ5_KBlockBytes;
}

// One Q5_K output row dotted against `x`, over 32 lanes: each lane owns one
// nibble byte AND the one `qh` byte at the same index, and therefore two
// elements 32 apart in two different sub-blocks. Four such groups tile a
// superblock.
static inline float dequant_q5_k_row_simd(
    device const uint8_t* W_row,
    device const half* x,
    uint N,
    uint lane
) {
    const uint n_blocks = N / kQ5_KBlockElems;

    float acc = 0.0f;
    for (uint b = 0; b < n_blocks; ++b) {
        device const uint8_t* blk = W_row + b * kQ5_KBlockBytes;
        const ushort d_raw = ushort(blk[0]) | (ushort(blk[1]) << 8);
        const ushort dmin_raw = ushort(blk[2]) | (ushort(blk[3]) << 8);
        const float d = float(as_type<half>(d_raw));
        const float dmin = float(as_type<half>(dmin_raw));
        device const uint8_t* packed = blk + kQ5_KScalesAt;
        device const uint8_t* qh = blk + kQ5_KQhAt;
        device const uint8_t* ql = blk + kQ5_KQuantsAt;

        const uint8_t bits = qh[lane];

        for (uint g = 0; g < kQ5_KBlockElems / (2 * kQ5_KSubElems); ++g) {
            float sc_lo, m_lo, sc_hi, m_hi;
            q4_k_scale_min(2 * g, packed, sc_lo, m_lo);
            q4_k_scale_min(2 * g + 1, packed, sc_hi, m_hi);

            const uint8_t byte = ql[g * kQ5_KSubElems + lane];
            // Note 1 and note 2 in the header: the +16 and the bit index.
            const float q_lo = float((byte & 0xF) + (((bits >> (2 * g)) & 1) << 4));
            const float q_hi = float((byte >> 4) + (((bits >> (2 * g + 1)) & 1) << 4));
            const float w_lo = d * sc_lo * q_lo - dmin * m_lo;
            const float w_hi = d * sc_hi * q_hi - dmin * m_hi;

            const uint at = b * kQ5_KBlockElems + g * 2 * kQ5_KSubElems + lane;
            acc = fma(w_lo, float(x[at]), acc);
            acc = fma(w_hi, float(x[at + kQ5_KSubElems]), acc);
        }
    }
    return simd_sum(acc);
}

// y[m] = sum_n W[m, n] * x[n]. One SIMD group per output row. Dispatch:
// threadgroupsPerGrid = (ceil(M / 8), 1, 1), threadsPerThreadgroup = (256,1,1).
[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void dequant_q5_k_gemv_simd(
    device const uint8_t* W      [[buffer(0)]],
    device const half*    x      [[buffer(1)]],
    device half*          y      [[buffer(2)]],
    constant uint&        M      [[buffer(3)]],
    constant uint&        N      [[buffer(4)]],
    uint                  tg_idx [[threadgroup_position_in_grid]],
    uint                  sg_idx [[simdgroup_index_in_threadgroup]],
    uint                  lane   [[thread_index_in_simdgroup]]
) {
    const uint row = tg_idx * kRowsPerTGQ5_K + sg_idx;
    if (row >= M) return;

    device const uint8_t* W_row = W + uint(row) * q5_k_row_bytes(N);
    const float acc = dequant_q5_k_row_simd(W_row, x, N, lane);
    if (lane == 0) {
        y[row] = half(acc);
    }
}
