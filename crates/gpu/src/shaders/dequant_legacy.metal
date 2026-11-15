#include <metal_stdlib>
using namespace metal;

// ============================================================================
// dequant_legacy - GGUF Q4_0, Q4_1, Q5_0 and Q5_1 dequant. PORT-LOCAL, not
// vendored, like its Q8_0 and K-quant siblings. Contract: the CPU decoders
// `turbospark_compute::dequantize_q{4,5}_{0,1}`, which are themselves held to
// ggml's own `to_float` byte for byte (crates/repack gguf_quantize tests).
//
// All four are 32-element blocks, so one SIMD group per output row owns one
// element per lane per block, exactly as the Q8_0 kernel does, and the
// per-type work is the single decoder function below. Layouts, as GGUF
// stores them (every field little-endian):
//   Q4_0 : 18 B = f16 d,                  qs[16]            w = (q - 8) * d
//   Q4_1 : 20 B = f16 d, f16 m,           qs[16]            w = q * d + m
//   Q5_0 : 22 B = f16 d, u32 qh,          qs[16]            w = (q - 16) * d
//   Q5_1 : 24 B = f16 d, f16 m, u32 qh,   qs[16]            w = q * d + m
// Element e (0..32) lives in byte qs[e % 16]: the LOW nibble for e < 16 and
// the HIGH nibble for e >= 16. For the 5-bit types the fifth bit of element e
// is bit e of qh (not bit e % 16), and it contributes 16.
//
// Four traps, each leaving a finite plausible row rather than a crash:
//   1. Nibble halves: element j and j + 16 share a byte. Treating the nibbles
//      as consecutive elements (the Q4_K habit) interleaves two halves.
//   2. The Q4_0/Q5_0 bias (8 and 16) is subtracted, the Q4_1/Q5_1 minimum is
//      ADDED. Mixing the two turns every weight one-signed.
//   3. qh is indexed by the element, not by the byte: bit e, e in 0..32.
//   4. Every field is read byte by byte; strides of 18..24 bytes are even, but
//      that is a property of the constants and not of the format.
// ============================================================================

constant constexpr uint kLegacyBlockElems = 32;
constant constexpr uint kRowsPerTGLegacy = 8;

static inline float legacy_f16(device const uint8_t* p) {
    return float(as_type<half>(ushort(ushort(p[0]) | (ushort(p[1]) << 8))));
}

static inline uint legacy_u32(device const uint8_t* p) {
    return uint(p[0]) | (uint(p[1]) << 8) | (uint(p[2]) << 16) | (uint(p[3]) << 24);
}

// Nibble of element e from the 16-byte `qs` run.
static inline uint legacy_nibble(device const uint8_t* qs, uint e) {
    const uint byte = qs[e & 15];
    return (e < 16) ? (byte & 0xF) : (byte >> 4);
}

static inline float q4_0_elem(device const uint8_t* blk, uint e) {
    return (float(legacy_nibble(blk + 2, e)) - 8.0f) * legacy_f16(blk);
}

static inline float q4_1_elem(device const uint8_t* blk, uint e) {
    return float(legacy_nibble(blk + 4, e)) * legacy_f16(blk) + legacy_f16(blk + 2);
}

static inline float q5_0_elem(device const uint8_t* blk, uint e) {
    const uint q = legacy_nibble(blk + 6, e) | (((legacy_u32(blk + 2) >> e) & 1u) << 4);
    return (float(q) - 16.0f) * legacy_f16(blk);
}

static inline float q5_1_elem(device const uint8_t* blk, uint e) {
    const uint q = legacy_nibble(blk + 8, e) | (((legacy_u32(blk + 4) >> e) & 1u) << 4);
    return float(q) * legacy_f16(blk) + legacy_f16(blk + 2);
}

// y[m] = sum_n W[m, n] * x[n]. One SIMD group per output row, one element per
// lane per block. Dispatch: threadgroupsPerGrid = (ceil(M / 8), 1, 1),
// threadsPerThreadgroup = (256, 1, 1). Embedding lookups are one thread per
// element; `out_scale` is sqrt(hidden) for Gemma's scaled embeddings.
#define LEGACY_KERNELS(NAME, BLOCK_BYTES, ELEM)                                          \
[[kernel, max_total_threads_per_threadgroup(256)]]                                       \
kernel void dequant_##NAME##_gemv_simd(                                                  \
    device const uint8_t* W      [[buffer(0)]],                                          \
    device const half*    x      [[buffer(1)]],                                          \
    device half*          y      [[buffer(2)]],                                          \
    constant uint&        M      [[buffer(3)]],                                          \
    constant uint&        N      [[buffer(4)]],                                          \
    uint                  tg_idx [[threadgroup_position_in_grid]],                       \
    uint                  sg_idx [[simdgroup_index_in_threadgroup]],                     \
    uint                  lane   [[thread_index_in_simdgroup]]                           \
) {                                                                                      \
    const uint row = tg_idx * kRowsPerTGLegacy + sg_idx;                                 \
    if (row >= M) return;                                                                \
    const uint n_blocks = N / kLegacyBlockElems;                                         \
    device const uint8_t* W_row = W + ulong(row) * n_blocks * BLOCK_BYTES;               \
    float acc = 0.0f;                                                                    \
    for (uint b = 0; b < n_blocks; ++b) {                                                \
        const float w = ELEM(W_row + b * BLOCK_BYTES, lane);                             \
        acc = fma(w, float(x[b * kLegacyBlockElems + lane]), acc);                       \
    }                                                                                    \
    acc = simd_sum(acc);                                                                 \
    if (lane == 0) {                                                                     \
        y[row] = half(acc);                                                              \
    }                                                                                    \
}                                                                                        \
kernel void embed_lookup_##NAME(                                                         \
    device const uint8_t* table     [[buffer(0)]],                                       \
    device half*          out       [[buffer(1)]],                                       \
    constant uint&        token_id  [[buffer(2)]],                                       \
    constant uint&        D         [[buffer(3)]],                                       \
    constant float&       out_scale [[buffer(4)]],                                       \
    uint                  gid       [[thread_position_in_grid]]                          \
) {                                                                                      \
    if (gid >= D) return;                                                                \
    const ulong row_bytes = ulong(D / kLegacyBlockElems) * BLOCK_BYTES;                  \
    device const uint8_t* blk =                                                          \
        table + ulong(token_id) * row_bytes + ulong(gid / kLegacyBlockElems) * BLOCK_BYTES; \
    out[gid] = half(ELEM(blk, gid % kLegacyBlockElems) * out_scale);                     \
}

LEGACY_KERNELS(q4_0, 18, q4_0_elem)
LEGACY_KERNELS(q4_1, 20, q4_1_elem)
LEGACY_KERNELS(q5_0, 22, q5_0_elem)
LEGACY_KERNELS(q5_1, 24, q5_1_elem)
