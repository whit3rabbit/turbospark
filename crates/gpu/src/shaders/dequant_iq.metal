#include <metal_stdlib>
using namespace metal;

// ============================================================================
// dequant_iq - GGUF IQ-codebook dequant: IQ4_NL, IQ4_XS and IQ3_XXS
// (ROADMAP Phase S). PORT-LOCAL, not vendored, for the same reason as the
// Q8_0/Q4_K/Q6_K siblings: the Swift engine has no GGUF intake, so there is no
// upstream kernel to mirror. The contract is
// `turbospark_compute::quant_gguf_iq` and nothing else.
//
// WHAT MAKES THESE DIFFERENT FROM EVERY OTHER QUANT KERNEL HERE. Q8_0, Q4_K
// and Q6_K all reconstruct a weight ARITHMETICALLY from its stored bits. These
// three do not: they store an INDEX into a fixed table that ships with the
// format. There is no formula to fall back on, so a kernel either carries the
// table or it produces plausible garbage.
//
// THE TABLES BELOW ARE GENERATED, and by the same generator that produced the
// Rust ones -- `scripts/ggml_tables.c`, which recovers them by probing
// libggml's own `to_float`. Two copies of a 1 KB codebook is exactly the
// duplication that lets a CPU reference and its kernel drift apart, so they
// are not transcribed twice: they are printed once and pasted into both. If
// you change one, regenerate both.
//
// Lane assignment is ONE ELEMENT PER LANE for all three, which falls out of
// the layouts rather than being chosen: IQ4_NL's block is 32 elements, and
// IQ4_XS and IQ3_XXS both tile a 256-element superblock with 32-element
// sub-blocks. That is simpler than the Q4_K sibling, where a lane owns a BYTE
// and therefore two elements 32 apart.
// ============================================================================

constant constexpr uint kIq4NlBlockElems = 32;
constant constexpr uint kIq4NlBlockBytes = 18;
constant constexpr uint kIq4XsBlockElems = 256;
constant constexpr uint kIq4XsSubElems = 32;
constant constexpr uint kIq4XsBlockBytes = 136;
constant constexpr uint kIq3XxsBlockElems = 256;
constant constexpr uint kIq3XxsSubElems = 32;
constant constexpr uint kIq3XxsBlockBytes = 98;
// Where the eight sign-and-scale words start inside an IQ3_XXS superblock:
// after the f16 scale and the 64 grid-index bytes.
constant constexpr uint kIq3XxsAuxAt = 2 + kIq3XxsBlockElems / 4;
constant constexpr uint kRowsPerTGIq = 8;

// The 16 non-linear reconstruction levels IQ4_NL and IQ4_XS share. NOT evenly
// spaced: an affine read (`(q - 8) * d`) is finite, correctly ordered, and
// wrong, worst in the tails.
constant char kIq4NlValues[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
};

// The IQ3_XXS codebook: 256 entries of four 8-bit MAGNITUDES, drawn from the
// eight-value alphabet {4, 12, 20, 28, 36, 44, 52, 62}. All positive; the sign
// is carried separately, in the per-sub-block sign word.
constant uchar4 kIq3XxsGrid[256] = {
    { 4,  4,  4,  4}, {20,  4,  4,  4}, {36,  4,  4,  4}, {12, 12,  4,  4},
    {28, 12,  4,  4}, {62, 12,  4,  4}, { 4, 20,  4,  4}, {20, 20,  4,  4},
    {12, 28,  4,  4}, {20, 36,  4,  4}, {28, 62,  4,  4}, {44, 62,  4,  4},
    {12,  4, 12,  4}, {28,  4, 12,  4}, { 4, 12, 12,  4}, {20, 12, 12,  4},
    {12, 20, 12,  4}, {44, 20, 12,  4}, { 4, 28, 12,  4}, {20, 28, 12,  4},
    {12, 36, 12,  4}, {36, 44, 12,  4}, { 4, 62, 12,  4}, { 4,  4, 20,  4},
    {20,  4, 20,  4}, {36,  4, 20,  4}, {12, 12, 20,  4}, { 4, 20, 20,  4},
    {20, 20, 20,  4}, {12, 28, 20,  4}, {28, 28, 20,  4}, {62, 28, 20,  4},
    {12, 44, 20,  4}, {62, 44, 20,  4}, {44, 62, 20,  4}, {12,  4, 28,  4},
    {62,  4, 28,  4}, { 4, 12, 28,  4}, {20, 12, 28,  4}, {44, 20, 28,  4},
    { 4, 62, 28,  4}, {28, 12, 36,  4}, {62, 28, 36,  4}, {36, 36, 36,  4},
    {62, 44, 36,  4}, {28, 62, 36,  4}, {44, 62, 36,  4}, {12,  4, 44,  4},
    {62,  4, 44,  4}, {20, 28, 44,  4}, {20, 44, 44,  4}, {44, 28, 52,  4},
    {36, 52, 52,  4}, { 4, 12, 62,  4}, {36, 12, 62,  4}, {52, 12, 62,  4},
    {28, 36, 62,  4}, {12, 52, 62,  4}, {12,  4,  4, 12}, {28,  4,  4, 12},
    { 4, 12,  4, 12}, {20, 12,  4, 12}, {12, 20,  4, 12}, {28, 20,  4, 12},
    { 4, 28,  4, 12}, {20, 28,  4, 12}, {36, 28,  4, 12}, {62, 36,  4, 12},
    { 4, 44,  4, 12}, { 4,  4, 12, 12}, {20,  4, 12, 12}, {12, 12, 12, 12},
    { 4, 20, 12, 12}, {20, 20, 12, 12}, {12,  4, 20, 12}, {28,  4, 20, 12},
    { 4, 12, 20, 12}, {20, 12, 20, 12}, {12, 20, 20, 12}, { 4, 28, 20, 12},
    {20, 62, 20, 12}, { 4,  4, 28, 12}, {20,  4, 28, 12}, { 4, 20, 28, 12},
    {12, 28, 28, 12}, {52, 36, 28, 12}, {52, 52, 28, 12}, {12,  4, 36, 12},
    {44,  4, 36, 12}, { 4, 44, 36, 12}, { 4, 20, 44, 12}, {36, 20, 44, 12},
    {52, 36, 44, 12}, {12, 62, 44, 12}, {44,  4, 52, 12}, {20, 20, 62, 12},
    { 4, 36, 62, 12}, { 4,  4,  4, 20}, {20,  4,  4, 20}, {12, 12,  4, 20},
    {28, 12,  4, 20}, { 4, 20,  4, 20}, {20, 20,  4, 20}, {52, 20,  4, 20},
    {12, 28,  4, 20}, {20, 36,  4, 20}, {12,  4, 12, 20}, {28,  4, 12, 20},
    {44,  4, 12, 20}, { 4, 12, 12, 20}, {20, 12, 12, 20}, {12, 20, 12, 20},
    { 4, 28, 12, 20}, {28, 52, 12, 20}, {62, 52, 12, 20}, { 4, 62, 12, 20},
    { 4,  4, 20, 20}, {20,  4, 20, 20}, {12, 12, 20, 20}, {62, 12, 20, 20},
    { 4, 20, 20, 20}, {20, 20, 20, 20}, {62, 28, 20, 20}, { 4, 36, 20, 20},
    {44, 44, 20, 20}, {12,  4, 28, 20}, { 4, 12, 28, 20}, {36, 12, 28, 20},
    { 4, 62, 28, 20}, {36, 62, 28, 20}, {44, 28, 36, 20}, {28, 44, 36, 20},
    {28,  4, 44, 20}, {62, 20, 44, 20}, {12, 36, 44, 20}, {36, 62, 44, 20},
    {12,  4, 62, 20}, {28,  4, 62, 20}, {52, 12, 62, 20}, {44, 36, 62, 20},
    {12,  4,  4, 28}, { 4, 12,  4, 28}, {20, 12,  4, 28}, {12, 20,  4, 28},
    {28, 20,  4, 28}, { 4, 44,  4, 28}, {44, 52,  4, 28}, {20, 62,  4, 28},
    { 4,  4, 12, 28}, {20,  4, 12, 28}, { 4, 20, 12, 28}, {12, 28, 12, 28},
    {36, 36, 12, 28}, {52, 36, 12, 28}, {12,  4, 20, 28}, {28,  4, 20, 28},
    { 4, 12, 20, 28}, {44, 20, 20, 28}, {20, 44, 20, 28}, {20, 62, 20, 28},
    {12, 12, 28, 28}, {28, 28, 28, 28}, { 4, 28, 36, 28}, {62, 36, 36, 28},
    {20, 62, 36, 28}, { 4,  4, 44, 28}, {52,  4, 44, 28}, {20, 20, 44, 28},
    {44, 44, 44, 28}, {36, 12, 52, 28}, {52, 28, 52, 28}, {28, 52, 52, 28},
    {28, 28, 62, 28}, { 4, 52, 62, 28}, {36,  4,  4, 36}, {62, 12,  4, 36},
    {44, 28,  4, 36}, {62, 28,  4, 36}, {28, 44,  4, 36}, {62, 44,  4, 36},
    {36, 62, 12, 36}, { 4, 20, 20, 36}, {62, 28, 20, 36}, { 4, 36, 20, 36},
    { 4, 52, 20, 36}, {52, 52, 20, 36}, {62,  4, 28, 36}, {44, 36, 28, 36},
    {36,  4, 36, 36}, {12, 44, 36, 36}, {36, 52, 36, 36}, {44, 20, 44, 36},
    {28, 36, 44, 36}, { 4, 62, 44, 36}, {44,  4, 62, 36}, { 4, 12, 62, 36},
    {20, 12, 62, 36}, { 4, 28, 62, 36}, {20, 12,  4, 44}, {12, 36,  4, 44},
    { 4, 62,  4, 44}, { 4,  4, 12, 44}, {52,  4, 12, 44}, {52, 20, 12, 44},
    {44, 44, 12, 44}, {36, 12, 20, 44}, {20, 28, 20, 44}, {20, 62, 20, 44},
    {20,  4, 28, 44}, {28, 44, 28, 44}, { 4, 12, 36, 44}, {28, 20, 36, 44},
    {62, 20, 36, 44}, {20, 62, 36, 44}, {20,  4, 44, 44}, {12, 28, 44, 44},
    { 4, 44, 52, 44}, {36, 20, 62, 44}, {20, 36, 62, 44}, {36, 20,  4, 52},
    {36, 36,  4, 52}, {52, 36,  4, 52}, {36, 52,  4, 52}, {12, 20, 12, 52},
    {12, 52, 12, 52}, {62, 12, 20, 52}, {36, 52, 20, 52}, { 4, 28, 28, 52},
    {52, 28, 28, 52}, {36, 36, 36, 52}, {44,  4, 44, 52}, {20, 44, 44, 52},
    {28, 28, 52, 52}, {28,  4, 62, 52}, {12, 20, 62, 52}, {28,  4,  4, 62},
    {44,  4,  4, 62}, {62,  4,  4, 62}, { 4, 12,  4, 62}, {20, 28,  4, 62},
    {20, 44,  4, 62}, {52, 20, 12, 62}, { 4, 36, 12, 62}, {20, 12, 20, 62},
    {44, 36, 20, 62}, {20, 44, 20, 62}, { 4,  4, 28, 62}, {44, 12, 28, 62},
    {28, 28, 28, 62}, { 4, 52, 28, 62}, {12, 20, 36, 62}, {12, 36, 36, 62},
    { 4,  4, 44, 62}, {20,  4, 44, 62}, {36, 20, 44, 62}, { 4, 28, 52, 62},
};

// The eight sign bits a 7-bit IQ3_XXS sign index stands for.
//
// ggml ships a 128-entry table for this (`ksigns_iq2xs`) and it is an
// expression: the eighth sign is a PARITY bit over the seven stored ones, so
// every group of eight elements carries an even number of negatives. Computed
// rather than tabled because a computed byte cannot go stale, and checked
// against ggml for all 128 indices by `scripts/ggml_tables.c`.
//
// Dropping the parity bit is the interesting failure: seven of every eight
// elements keep the right sign, so the output still correlates well.
static inline uint iq3xxs_signs(uint index) {
    const uint i = index & 127u;
    return i | ((popcount(i) & 1u) << 7);
}

static inline uint iq4_nl_row_bytes(uint n) {
    return n / kIq4NlBlockElems * kIq4NlBlockBytes;
}

static inline uint iq4_xs_row_bytes(uint n) {
    return n / kIq4XsBlockElems * kIq4XsBlockBytes;
}

static inline uint iq3_xxs_row_bytes(uint n) {
    return n / kIq3XxsBlockElems * kIq3XxsBlockBytes;
}

static inline float iq_f16_at(device const uint8_t* p) {
    // Byte by byte for the same reason the Q8_0 and Q4_K scales are: the
    // alignment of an 18- or 98-byte stride is a property of the constant, not
    // of the format.
    const ushort raw = ushort(p[0]) | (ushort(p[1]) << 8);
    return float(as_type<half>(raw));
}

// One IQ4_NL output row dotted against `x`, over 32 lanes, one element each.
//
// The nibble split is the trap: a byte serves two elements SIXTEEN apart, not
// two adjacent ones. Lanes 0..16 take low nibbles, 16..32 take the high nibble
// of the SAME sixteen bytes.
static inline float dequant_iq4_nl_row_simd(
    device const uint8_t* W_row,
    device const half* x,
    uint N,
    uint lane
) {
    const uint n_blocks = N / kIq4NlBlockElems;
    const uint half_elems = kIq4NlBlockElems / 2;
    const uint byte_at = lane % half_elems;
    const bool upper = lane >= half_elems;

    float acc = 0.0f;
    for (uint b = 0; b < n_blocks; ++b) {
        device const uint8_t* blk = W_row + b * kIq4NlBlockBytes;
        const float d = iq_f16_at(blk);
        const uint8_t byte = blk[2 + byte_at];
        const uint q = upper ? uint(byte >> 4) : uint(byte & 0xF);
        const float w = d * float(kIq4NlValues[q]);
        acc = fma(w, float(x[b * kIq4NlBlockElems + lane]), acc);
    }
    return simd_sum(acc);
}

// One IQ4_XS output row dotted against `x`, over 32 lanes, one element per
// lane per sub-block.
//
// IQ4_XS is IQ4_NL's table under a two-level scale, and the scale is where it
// goes wrong quietly. Each of the eight 32-element sub-blocks carries a 6-bit
// scale SPLIT ACROSS TWO FIELDS: the low four bits are a nibble of
// `scales_l` (sub-block `ib` uses byte `ib / 2`, low nibble when `ib` is
// even), and the high two are bits `2 * ib` of the u16 `scales_h`. The result
// is BIASED: `dl = d * (ls - 32)`, so half the range is negative and a kernel
// that skips the bias mirrors whole sub-blocks rather than failing.
static inline float dequant_iq4_xs_row_simd(
    device const uint8_t* W_row,
    device const half* x,
    uint N,
    uint lane
) {
    const uint n_blocks = N / kIq4XsBlockElems;
    const uint half_elems = kIq4XsSubElems / 2;
    const uint byte_at = lane % half_elems;
    const bool upper = lane >= half_elems;

    float acc = 0.0f;
    for (uint b = 0; b < n_blocks; ++b) {
        device const uint8_t* blk = W_row + b * kIq4XsBlockBytes;
        const float d = iq_f16_at(blk);
        const uint scales_h = uint(blk[2]) | (uint(blk[3]) << 8);
        device const uint8_t* scales_l = blk + 4;
        device const uint8_t* qs = blk + 8;

        for (uint ib = 0; ib < kIq4XsBlockElems / kIq4XsSubElems; ++ib) {
            const uint lo = uint((scales_l[ib / 2] >> (4 * (ib % 2))) & 0xF);
            const uint hi = (scales_h >> (2 * ib)) & 3u;
            const float dl = d * (float(int(lo | (hi << 4))) - 32.0f);
            const uint8_t byte = qs[ib * half_elems + byte_at];
            const uint q = upper ? uint(byte >> 4) : uint(byte & 0xF);
            const uint at = b * kIq4XsBlockElems + ib * kIq4XsSubElems + lane;
            acc = fma(dl * float(kIq4NlValues[q]), float(x[at]), acc);
        }
    }
    return simd_sum(acc);
}

// One IQ3_XXS output row dotted against `x`, over 32 lanes, one element per
// lane per sub-block.
//
// A 256-element superblock holds an f16 `d`, 64 grid-index bytes, and eight
// u32 words. Each word serves one 32-element sub-block and packs five fields
// into 32 bits: four 7-bit sign indices in bits 0..28, and that sub-block's
// 4-bit scale in bits 28..32.
//
// Three things are silently wrong rather than a fault:
//   1. `db = d * (0.5 + nibble) * 0.5`, not `d * nibble`. The `0.5 +` means
//      scale nibble 0 is a real scale; dropping it zeroes one sub-block in
//      sixteen and dims the rest by a factor that varies per sub-block.
//   2. A grid index expands to FOUR elements and a sub-block consumes eight
//      indices. The pairing is interleaved: index `2l` fills elements
//      `8l .. 8l+4`, index `2l + 1` fills `8l+4 .. 8l+8`.
//   3. The grid holds magnitudes only. Reading an entry as signed turns a
//      quarter of the table into large negatives and still produces finite
//      output.
static inline float dequant_iq3_xxs_row_simd(
    device const uint8_t* W_row,
    device const half* x,
    uint N,
    uint lane
) {
    const uint n_blocks = N / kIq3XxsBlockElems;
    // Which of the eight elements this lane owns inside its group of eight,
    // and which of the four (index, sign-index) pairs that group is.
    const uint l = lane / 8;
    const uint j = lane % 8;
    const uint pair = j / 4;      // 0 -> grid index 2l, 1 -> grid index 2l + 1
    const uint slot = j % 4;      // which of the four values in that entry

    float acc = 0.0f;
    for (uint b = 0; b < n_blocks; ++b) {
        device const uint8_t* blk = W_row + b * kIq3XxsBlockBytes;
        const float d = iq_f16_at(blk);
        device const uint8_t* qs = blk + 2;
        device const uint8_t* aux = blk + kIq3XxsAuxAt;

        for (uint ib = 0; ib < kIq3XxsBlockElems / kIq3XxsSubElems; ++ib) {
            device const uint8_t* w = aux + 4 * ib;
            const uint word = uint(w[0]) | (uint(w[1]) << 8)
                | (uint(w[2]) << 16) | (uint(w[3]) << 24);
            const float db = d * (0.5f + float(word >> 28)) * 0.5f;
            const uint signs = iq3xxs_signs(word >> (7 * l));
            const uchar4 entry = kIq3XxsGrid[qs[ib * 8 + 2 * l + pair]];
            const float mag = float(entry[slot]);
            const float sign = (signs & (1u << (4 * pair + slot))) != 0 ? -1.0f : 1.0f;
            const uint at = b * kIq3XxsBlockElems + ib * kIq3XxsSubElems + lane;
            acc = fma(db * mag * sign, float(x[at]), acc);
        }
    }
    return simd_sum(acc);
}

// y[m] = sum_n W[m, n] * x[n], one SIMD group per output row. Dispatch:
// threadgroupsPerGrid = (ceil(M / 8), 1, 1), threadsPerThreadgroup = (256,1,1).
//
// Three near-identical kernels rather than one with a type constant, matching
// how every other quant GEMV here is shaped: the row helpers take different
// strides and MSL has no way to select one without a branch in the inner loop.
[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void dequant_iq4_nl_gemv_simd(
    device const uint8_t* W      [[buffer(0)]],
    device const half*    x      [[buffer(1)]],
    device half*          y      [[buffer(2)]],
    constant uint&        M      [[buffer(3)]],
    constant uint&        N      [[buffer(4)]],
    uint                  tg_idx [[threadgroup_position_in_grid]],
    uint                  sg_idx [[simdgroup_index_in_threadgroup]],
    uint                  lane   [[thread_index_in_simdgroup]]
) {
    const uint row = tg_idx * kRowsPerTGIq + sg_idx;
    if (row >= M) return;
    const float acc = dequant_iq4_nl_row_simd(W + row * iq4_nl_row_bytes(N), x, N, lane);
    if (lane == 0) y[row] = half(acc);
}

[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void dequant_iq4_xs_gemv_simd(
    device const uint8_t* W      [[buffer(0)]],
    device const half*    x      [[buffer(1)]],
    device half*          y      [[buffer(2)]],
    constant uint&        M      [[buffer(3)]],
    constant uint&        N      [[buffer(4)]],
    uint                  tg_idx [[threadgroup_position_in_grid]],
    uint                  sg_idx [[simdgroup_index_in_threadgroup]],
    uint                  lane   [[thread_index_in_simdgroup]]
) {
    const uint row = tg_idx * kRowsPerTGIq + sg_idx;
    if (row >= M) return;
    const float acc = dequant_iq4_xs_row_simd(W + row * iq4_xs_row_bytes(N), x, N, lane);
    if (lane == 0) y[row] = half(acc);
}

[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void dequant_iq3_xxs_gemv_simd(
    device const uint8_t* W      [[buffer(0)]],
    device const half*    x      [[buffer(1)]],
    device half*          y      [[buffer(2)]],
    constant uint&        M      [[buffer(3)]],
    constant uint&        N      [[buffer(4)]],
    uint                  tg_idx [[threadgroup_position_in_grid]],
    uint                  sg_idx [[simdgroup_index_in_threadgroup]],
    uint                  lane   [[thread_index_in_simdgroup]]
) {
    const uint row = tg_idx * kRowsPerTGIq + sg_idx;
    if (row >= M) return;
    const float acc = dequant_iq3_xxs_row_simd(W + row * iq3_xxs_row_bytes(N), x, N, lane);
    if (lane == 0) y[row] = half(acc);
}
