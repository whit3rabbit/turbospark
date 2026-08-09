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

// One row of a Q6_K embedding table, dequantized into `out` and scaled.
// Sibling of `embed_lookup_q4_k` and `embed_lookup_q8_0`, one thread per
// element rather than one SIMD group per row.
//
// It exists because ROADMAP Phase S's candidate checkpoint puts `token_embd`
// in Q6_K and ties the LM head to it. When Q6_K first landed this kernel was
// deliberately skipped -- the only real file using the type put it in
// `output.weight`, which needs a GEMV and nothing else -- and that call was
// right at the time; a second real file changed the answer.
//
// The element-to-byte mapping is the trap, and it is the GEMV's addressing
// read from the other direction. Within a 256-element superblock, element `e`
// sits in half `h = e / 128` and position `p = e % 128`. Inside that half:
//   p <  32  -> low nibble of ql[p],       high bits qh[p]      >> 0, scale is
//   p <  64  -> low nibble of ql[p],       qh[p - 32]   >> 2,    is + 2
//   p <  96  -> high nibble of ql[p - 64], qh[p - 64]   >> 4,    is + 4
//   p < 128  -> high nibble of ql[p - 64], qh[p - 96]   >> 6,    is + 6
// where `ql` is offset 64 bytes per half, `qh` 32, `sc` 8, and the scale index
// `is` is the LANE-equivalent `(p % 32) / 16`. Getting the scale stride wrong
// (one instead of two) mixes scales between quarters and is invisible unless
// the sub-blocks differ in magnitude.
kernel void embed_lookup_q6_k(
    device const uint8_t* table     [[buffer(0)]],   // [V, D/256 * 210] blocks
    device half*          out       [[buffer(1)]],   // [D] FP16
    constant uint&        token_id  [[buffer(2)]],
    constant uint&        D         [[buffer(3)]],
    constant float&       out_scale [[buffer(4)]],
    uint                  gid       [[thread_position_in_grid]]
) {
    if (gid >= D) return;
    const uint n_blocks = D / kQ6_KBlockElems;
    device const uint8_t* blk = table + uint(token_id) * n_blocks * kQ6_KBlockBytes
        + (gid / kQ6_KBlockElems) * kQ6_KBlockBytes;

    const ushort d_raw = ushort(blk[kQ6_KDAt]) | (ushort(blk[kQ6_KDAt + 1]) << 8);
    const float d = float(as_type<half>(d_raw));

    const uint e = gid % kQ6_KBlockElems;
    const uint h = e / kQ6_KHalfElems;
    const uint p = e % kQ6_KHalfElems;
    const uint quarter = p / 32;      // 0..4, selects the qh bit pair
    const uint lane = p % 32;         // the GEMV's lane, i.e. the qh byte

    device const uint8_t* ql = blk + h * 64;
    device const uint8_t* qh = blk + kQ6_KQhAt + h * 32;
    device const uint8_t* sc = blk + kQ6_KScalesAt + h * 8;

    // Quarters 0 and 1 take the LOW nibble of ql[lane] and ql[lane + 32];
    // quarters 2 and 3 take the HIGH nibble of the same two bytes.
    const uint8_t byte = ql[lane + (quarter % 2) * 32];
    const uint nib = (quarter < 2) ? uint(byte & 0xF) : uint(byte >> 4);
    const uint bits = (uint(qh[lane]) >> (2 * quarter)) & 3u;
    const float q = float(int(nib | (bits << 4)) - 32);

    const uint is = lane / kQ6_KSubElems + 2 * quarter;
    out[gid] = half(d * float(as_type<int8_t>(sc[is])) * q * out_scale);
}
