#include <metal_stdlib>
using namespace metal;

// ============================================================================
// dequant_q4_k - GGUF Q4_K dequant. PORT-LOCAL, not vendored, for the same
// reason as its Q8_0 sibling: the Swift engine has no GGUF intake, so there
// is no upstream kernel to mirror. Its contract is
// `mrefrust_compute::dequant_q4_k_gemv` and nothing else.
//
// Layout per weight row of length N, exactly the bytes GGUF stores and the
// repack walk copies through untouched:
//   superblock : 144 bytes over 256 elements =
//                  f16 d, f16 dmin, 12 packed bytes, 128 nibble bytes.
//   sub-block  : 32 elements, eight per superblock, each with a 6-bit scale
//                and a 6-bit min quantized against d and dmin.
//   value      : w = d * sc[j] * q - dmin * m[j], with UNSIGNED 4-bit q.
//
// Q4_K is a two-level scheme, not a wider Q8_0, and three of its properties
// are silently wrong rather than a crash if carried over from the sibling:
//   1. The 6-bit sub-scales are SPLIT ACROSS BYTES for sub-blocks 4..8: the
//      low four bits live in packed[8..12], the high two in the top bits of
//      packed[0..8]. Reading only the low nibble gives a scale that is
//      merely too small.
//   2. The two nibbles of one byte are 32 elements APART, not adjacent, and
//      they belong to DIFFERENT sub-blocks (2g and 2g+1).
//   3. There IS a bias, and it is SUBTRACTED. Dropping it leaves every
//      reconstruction non-negative, which on real weights is finite,
//      correctly ordered within a sub-block, and wrong.
//
// The two super-scales are loaded byte by byte for the same reason the Q8_0
// scale is: alignment of a 144-byte stride is a property of the constant
// rather than of the format.
// ============================================================================

constant constexpr uint kQ4_KBlockElems = 256;
constant constexpr uint kQ4_KSubElems = 32;
constant constexpr uint kQ4_KBlockBytes = 144;
constant constexpr uint kQ4_KScalesAt = 4;
constant constexpr uint kQ4_KQuantsAt = 16;
constant constexpr uint kRowsPerTGQ4_K = 8;

// Sub-block `j`'s 6-bit scale and 6-bit min out of the 12 packed bytes.
// ggml calls this `get_scale_min_k4`; see note 1 in the header for why the
// second branch exists at all.
static inline void q4_k_scale_min(
    uint j,
    device const uint8_t* packed,
    thread float& scale,
    thread float& min_q
) {
    if (j < 4) {
        scale = float(packed[j] & 63);
        min_q = float(packed[j + 4] & 63);
    } else {
        scale = float((packed[j + 4] & 0xF) | ((packed[j - 4] >> 6) << 4));
        min_q = float((packed[j + 4] >> 4) | ((packed[j] >> 6) << 4));
    }
}

// Bytes one Q4_K row of `n` elements occupies. Shared with `moe_gguf.metal`,
// which is compiled with this file concatenated ahead of it.
static inline uint q4_k_row_bytes(uint n) {
    return n / kQ4_KBlockElems * kQ4_KBlockBytes;
}

// One Q4_K output row dotted against `x`, over 32 lanes: each lane owns
// exactly one nibble byte and therefore two elements 32 apart, in two
// different sub-blocks. Four such groups tile a superblock.
//
// Factored out rather than inlined into the kernel below because the routed
// expert pair in `moe_gguf.metal` needs the identical unpack, and a second
// hand-written copy of the 6-bit scale split is exactly the duplication that
// would let the two disagree.
static inline float dequant_q4_k_row_simd(
    device const uint8_t* W_row,
    device const half* x,
    uint N,
    uint lane
) {
    const uint n_blocks = N / kQ4_KBlockElems;

    float acc = 0.0f;
    for (uint b = 0; b < n_blocks; ++b) {
        device const uint8_t* blk = W_row + b * kQ4_KBlockBytes;
        ushort d_raw = ushort(blk[0]) | (ushort(blk[1]) << 8);
        ushort dmin_raw = ushort(blk[2]) | (ushort(blk[3]) << 8);
        const float d = float(as_type<half>(d_raw));
        const float dmin = float(as_type<half>(dmin_raw));
        device const uint8_t* packed = blk + kQ4_KScalesAt;
        device const uint8_t* qs = blk + kQ4_KQuantsAt;

        for (uint g = 0; g < kQ4_KBlockElems / (2 * kQ4_KSubElems); ++g) {
            float sc_lo, m_lo, sc_hi, m_hi;
            q4_k_scale_min(2 * g, packed, sc_lo, m_lo);
            q4_k_scale_min(2 * g + 1, packed, sc_hi, m_hi);

            const uint8_t byte = qs[g * kQ4_KSubElems + lane];
            const float w_lo = d * sc_lo * float(byte & 0xF) - dmin * m_lo;
            const float w_hi = d * sc_hi * float(byte >> 4) - dmin * m_hi;

            const uint at = b * kQ4_KBlockElems + g * 2 * kQ4_KSubElems + lane;
            acc = fma(w_lo, float(x[at]), acc);
            acc = fma(w_hi, float(x[at + kQ4_KSubElems]), acc);
        }
    }
    return simd_sum(acc);
}

// y[m] = sum_n W[m, n] * x[n]. One SIMD group per output row. Dispatch:
// threadgroupsPerGrid = (ceil(M / 8), 1, 1), threadsPerThreadgroup = (256,1,1).
[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void dequant_q4_k_gemv_simd(
    device const uint8_t* W      [[buffer(0)]],
    device const half*    x      [[buffer(1)]],
    device half*          y      [[buffer(2)]],
    constant uint&        M      [[buffer(3)]],
    constant uint&        N      [[buffer(4)]],
    uint                  tg_idx [[threadgroup_position_in_grid]],
    uint                  sg_idx [[simdgroup_index_in_threadgroup]],
    uint                  lane   [[thread_index_in_simdgroup]]
) {
    const uint row = tg_idx * kRowsPerTGQ4_K + sg_idx;
    if (row >= M) return;

    device const uint8_t* W_row = W + uint(row) * q4_k_row_bytes(N);
    const float acc = dequant_q4_k_row_simd(W_row, x, N, lane);
    if (lane == 0) {
        y[row] = half(acc);
    }
}

// One row of a Q4_K embedding table, dequantized into `out` and scaled.
// Sibling of `embed_lookup_q8_0`, and one thread per element rather than one
// SIMD group per row, matching the affine and Q8_0 lookups.
//
// The element-to-nibble mapping is the trap, and it is the same one the GEMV
// above navigates from the other direction: within a superblock, element `e`
// sits in group `g = e / 64`, takes the LOW nibble of byte `g * 32 + e % 32`
// when `(e % 64) < 32` and the HIGH nibble otherwise, and belongs to
// sub-block `2g` or `2g + 1` accordingly. Treating a byte's two nibbles as
// adjacent elements reads a plausible, wrong row.
kernel void embed_lookup_q4_k(
    device const uint8_t* table     [[buffer(0)]],   // [V, D/256 * 144] blocks
    device half*          out       [[buffer(1)]],   // [D] FP16
    constant uint&        token_id  [[buffer(2)]],
    constant uint&        D         [[buffer(3)]],
    constant float&       out_scale [[buffer(4)]],
    uint                  gid       [[thread_position_in_grid]]
) {
    if (gid >= D) return;
    device const uint8_t* blk = table + uint(token_id) * q4_k_row_bytes(D)
        + (gid / kQ4_KBlockElems) * kQ4_KBlockBytes;

    const ushort d_raw = ushort(blk[0]) | (ushort(blk[1]) << 8);
    const ushort dmin_raw = ushort(blk[2]) | (ushort(blk[3]) << 8);
    const float d = float(as_type<half>(d_raw));
    const float dmin = float(as_type<half>(dmin_raw));

    const uint e = gid % kQ4_KBlockElems;
    const uint g = e / (2 * kQ4_KSubElems);
    const uint upper = (e % (2 * kQ4_KSubElems)) / kQ4_KSubElems;
    const uint l = e % kQ4_KSubElems;

    float sc, m;
    q4_k_scale_min(2 * g + upper, blk + kQ4_KScalesAt, sc, m);
    const uint8_t byte = blk[kQ4_KQuantsAt + g * kQ4_KSubElems + l];
    const float q = float(upper == 0 ? (byte & 0xF) : (byte >> 4));
    out[gid] = half((d * sc * q - dmin * m) * out_scale);
}
