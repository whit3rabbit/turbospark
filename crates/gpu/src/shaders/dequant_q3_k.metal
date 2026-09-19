// ============================================================================
// dequant_q3_k - GGUF Q3_K dequant. PORT-LOCAL, not vendored, for the same
// reason as its Q4_K sibling: the Swift engine has no GGUF intake, so there
// is no upstream kernel to mirror. Its contract is
// `turbospark_compute::dequant_q3_k_gemv` and nothing else.
//
// Layout per weight row of length N, exactly the bytes GGUF stores and the
// repack walk copies through untouched:
//   superblock : 110 bytes over 256 elements =
//                  32 high-bit bytes, 64 two-bit quant bytes,
//                  12 packed scale bytes, then one closing f16 super-scale.
//   group      : 16 elements, sixteen per superblock, each with one 6-bit
//                scale quantized against the super-scale with a fixed bias
//                of 32.
//   value      : w = d * (sc[j] - 32) * level, where level is a 2-bit value
//                SHIFTED TO SIGNED by a high-bit run: level = q when that
//                run's bit is set and q - 4 when it is clear.
//
// Q3_K is the K-quant whose every sibling habit is wrong, and four of the
// wrong versions are finite, byte-aligned and length-correct rather than a
// crash:
//   1. The super-scale is the LAST field. Reading a Q4_K-style leading f16
//      turns 32 high-bit bytes into a garbage scale and every weight into a
//      plausible-magnitude wrong number.
//   2. The high-bit run is indexed by `e % 32` with the BIT chosen by
//      `e / 32`, so consecutive elements alternate bits within one byte.
//   3. The sixteen 6-bit scales are packed into twelve bytes through a
//      four-word shuffle (ggml's kmask1/kmask2 aux trick); the low nibble of
//      values 8..16 hides in the HIGH nibbles of the same eight bytes that
//      carry values 0..8's low nibbles.
//   4. A cleared high bit SUBTRACTS 4 from the 2-bit level. Reading levels
//      unsigned (the Q6_K habit) leaves correctly ordered, non-negative
//      weights.
//
// The CPU reference this must match is
// `turbospark_compute::quant_gguf::q3_k`, held to ggml itself by
// `q3_k_decodes_exactly_what_ggml_decodes` (generated oracle from
// `scripts/ggml_q3_k_oracle.c`).
// ============================================================================

#include <metal_stdlib>
using namespace metal;

constant constexpr uint kQ3_KBlockElems = 256;
constant constexpr uint kQ3_KGroupElems = 16;
constant constexpr uint kQ3_KBlockBytes = 110;
constant constexpr uint kQ3_KHmaskAt = 0;
constant constexpr uint kQ3_KQuantsAt = 32;
constant constexpr uint kQ3_KScalesAt = 96;
constant constexpr uint kQ3_KDAt = 108;
constant constexpr uint kRowsPerTGQ3_K = 8;

// Bytes one Q3_K row of `n` elements occupies. Shared with `moe_gguf.metal`
// by the same concatenation convention as the Q4_K one.
static inline uint q3_k_row_bytes(uint n) {
    return n / kQ3_KBlockElems * kQ3_KBlockBytes;
}

// The twelve packed scale bytes decoded into sixteen bias-subtracted scale
// factors already multiplied by the super-scale: `dl[g] = d * (s[g] - 32)`.
// The four-word shuffle is verbatim ggml; see note 3 in the header.
static inline void q3_k_decode_scales(
    device const uint8_t* blk,
    thread float* dl,
    thread float& d
) {
    const ushort d_raw = ushort(blk[kQ3_KDAt]) | (ushort(blk[kQ3_KDAt + 1]) << 8);
    d = float(as_type<half>(d_raw));

    uint aux[4] = {0u, 0u, 0u, 0u};
    for (uint i = 0; i < 3; ++i) {
        for (uint k = 0; k < 4; ++k) {
            aux[i] |= uint(blk[kQ3_KScalesAt + i * 4 + k]) << (8u * k);
        }
    }
    const uint kmask1 = 0x03030303u;
    const uint kmask2 = 0x0f0f0f0fu;
    const uint tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    for (uint g = 0; g < 16; ++g) {
        const uint byte = (aux[g / 4] >> (8u * (g % 4))) & 0xFFu;
        dl[g] = d * (float(byte) - 32.0f);
    }
}

// One Q3_K output row dotted against `x`, over 32 lanes: each lane owns one
// quant byte per 128-element half and therefore eight elements per
// superblock, four per half.
//
// Factored out rather than inlined into the kernel below for the same reason
// as the Q4_K one: the shape a routed pair would need is the identical
// unpack, and a second hand-written copy of the scale shuffle is exactly the
// duplication that would let the two disagree.
static inline float dequant_q3_k_row_simd(
    device const uint8_t* W_row,
    device const half* x,
    uint N,
    uint lane
) {
    const uint n_blocks = N / kQ3_KBlockElems;

    float acc = 0.0f;
    float dl[16];
    float d;
    for (uint b = 0; b < n_blocks; ++b) {
        device const uint8_t* blk = W_row + b * kQ3_KBlockBytes;
        q3_k_decode_scales(blk, dl, d);

        // One high-bit byte per lane covers every element this lane reads in
        // the superblock: element e uses byte e % 32, and this lane reads
        // elements at rem = lane within each group of 32.
        const uint8_t hm = blk[kQ3_KHmaskAt + lane];

        for (uint hi = 0; hi < 2; ++hi) {
            const uint8_t byte = blk[kQ3_KQuantsAt + hi * 32 + lane];
            const uint half_at = b * kQ3_KBlockElems + hi * 128;
            for (uint j = 0; j < 4; ++j) {
                const uint g = 8 * hi + 2 * j + (lane >= 16 ? 1u : 0u);
                const uint q = (byte >> (2u * j)) & 3u;
                const uint hbit = (uint(hm) >> (4u * hi + j)) & 1u;
                const float level = float(q) - (hbit != 0u ? 0.0f : 4.0f);
                acc = fma(dl[g] * level, float(x[half_at + 32u * j + lane]), acc);
            }
        }
    }
    return simd_sum(acc);
}

// y[m] = sum_n W[m, n] * x[n]. One SIMD group per output row. Dispatch:
// threadgroupsPerGrid = (ceil(M / 8), 1, 1), threadsPerThreadgroup = (256,1,1).
[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void dequant_q3_k_gemv_simd(
    device const uint8_t* W      [[buffer(0)]],
    device const half*    x      [[buffer(1)]],
    device half*          y      [[buffer(2)]],
    constant uint&        M      [[buffer(3)]],
    constant uint&        N      [[buffer(4)]],
    uint                  tg_idx [[threadgroup_position_in_grid]],
    uint                  sg_idx [[simdgroup_index_in_threadgroup]],
    uint                  lane   [[thread_index_in_simdgroup]]
) {
    const uint row = tg_idx * kRowsPerTGQ3_K + sg_idx;
    if (row >= M) return;

    device const uint8_t* W_row = W + uint(row) * q3_k_row_bytes(N);
    const float acc = dequant_q3_k_row_simd(W_row, x, N, lane);
    if (lane == 0) {
        y[row] = half(acc);
    }
}

// One row of a Q3_K embedding table, dequantized into `out` and scaled.
// Sibling of `embed_lookup_q4_k`, one thread per element. No real file in
// this port's orbit puts Q3_K in `token_embd` today (the pinned Q3_K_M keeps
// that tensor at Q4_K and its head at Q6_K), and this exists so the type's
// kernel set matches its siblings' rather than because a caller needs it --
// the Q6_K precedent, stated in `EXECUTABLE_GGUF_TYPES`' doc.
kernel void embed_lookup_q3_k(
    device const uint8_t* table     [[buffer(0)]],   // [V, D/256 * 110] blocks
    device half*          out       [[buffer(1)]],   // [D] FP16
    constant uint&        token_id  [[buffer(2)]],
    constant uint&        D         [[buffer(3)]],
    constant float&       out_scale [[buffer(4)]],
    uint                  gid       [[thread_position_in_grid]]
) {
    if (gid >= D) return;
    device const uint8_t* blk = table + uint(token_id) * q3_k_row_bytes(D)
        + (gid / kQ3_KBlockElems) * kQ3_KBlockBytes;

    float dl[16];
    float d;
    q3_k_decode_scales(blk, dl, d);

    const uint e = gid % kQ3_KBlockElems;
    const uint hi = e / 128;
    const uint j = (e % 128) / 32;
    const uint rem = e % 32;
    const uint8_t byte = blk[kQ3_KQuantsAt + hi * 32 + rem];
    const uint q = (uint(byte) >> (2u * j)) & 3u;
    const uint hbit = (uint(blk[kQ3_KHmaskAt + rem]) >> (4u * hi + j)) & 1u;
    const float level = float(q) - (hbit != 0u ? 0.0f : 4.0f);
    out[gid] = half(dl[8 * hi + 2 * j + rem / 16] * level * out_scale);
}
