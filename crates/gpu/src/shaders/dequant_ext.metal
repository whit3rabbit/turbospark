#include <metal_stdlib>
using namespace metal;

// ============================================================================
// dequant_ext - GGUF block types that ggml added after the K and IQ families:
// Q1_0, TQ1_0, TQ2_0, NVFP4, Q2_0 and MXFP4. PORT-LOCAL. Contract: the
// CPU decoders in `turbospark_compute` (`dequantize_q1_0`, ...), which are held
// to ggml's own `to_float` (type id 41 for Q1_0).
//
// Blocks here are wider than 32 elements, so one SIMD group per output row has
// each lane own elements lane, lane + 32, lane + 64, ... of every block, and
// the per-type work is the single ELEM function.
//   Q1_0 : 128 elems, 18 B = f16 d, bits[16]    w = bit ? +d : -d
//   TQ2_0: 256 elems, 66 B = qs[64], f16 d      w = (2-bit code - 1) * d
//   TQ1_0: 256 elems, 54 B = qs[48], qh[4], f16 d   w = (base-3 trit - 1) * d
//   NVFP4: 64 elems,  36 B = e4m3 scale[4], qs[32]  w = E2M1(code) * scale[e / 16]
//   Q2_0 : 64 elems,  18 B = f16 d, qs[16]      w = (2-bit code - 1) * d, LSB first
//   MXFP4: 32 elems,  17 B = e8m0 byte, qs[16]  w = FP4(code) * 2^(e-128), low nibble
//          of qs[j] is element j, high nibble element j + 16
// Element e of Q1_0 is bit (e % 8) of byte bits[e / 8], LSB first. TQ2_0, TQ1_0
// and NVFP4 put the scale at DIFFERENT ends (last, last, first); the element
// orders are in the CPU decoders' headers, each measured from ggml.
// ============================================================================

constant constexpr uint kRowsPerTGExt = 8;

static inline float ext_f16(device const uint8_t* p) {
    return float(as_type<half>(ushort(ushort(p[0]) | (ushort(p[1]) << 8))));
}

static inline float q1_0_elem(device const uint8_t* blk, uint e) {
    const float d = ext_f16(blk);
    return ((blk[2 + (e >> 3)] >> (e & 7)) & 1u) ? d : -d;
}

// TQ2_0: element e = chunk * 128 + l * 32 + m; code l of byte qs[chunk*32 + m].
static inline float tq2_0_elem(device const uint8_t* blk, uint e) {
    const uint chunk = e >> 7;
    const uint l = (e >> 5) & 3u;
    const uint m = e & 31u;
    const uint q = (uint(blk[chunk * 32u + m]) >> (2u * l)) & 3u;
    return (float(q) - 1.0f) * ext_f16(blk + 64);
}

// TQ1_0: one trit is ((byte * 3^plane) mod 256) * 3 >> 8. Segments: qs[0..32]
// give elements 0..160 (5 planes of 32), qs[32..48] give 160..240 (5 planes of
// 16), qh gives 240..256 (4 planes of 4).
static inline float tq1_0_elem(device const uint8_t* blk, uint e) {
    uint plane, byte_idx;
    if (e < 160u) {
        plane = e >> 5;
        byte_idx = e & 31u;
    } else if (e < 240u) {
        const uint r = e - 160u;
        plane = r >> 4;
        byte_idx = 32u + (r & 15u);
    } else {
        const uint r = e - 240u;
        plane = r >> 2;
        byte_idx = 48u + (r & 3u);
    }
    uint p3 = 1u;
    for (uint i = 0; i < plane; ++i) p3 *= 3u;
    const uint q = (uint(blk[byte_idx]) * p3) & 0xFFu;
    return (float((q * 3u) >> 8) - 1.0f) * ext_f16(blk + 52);
}

// NVFP4 scale byte: unsigned E4M3 with bit 7 ignored and 0x7F (NaN) -> 0,
// both measured from ggml.
static inline float nvfp4_scale(uint b) {
    if ((b & 0x7Fu) == 0x7Fu) return 0.0f;
    const uint ex = (b >> 3) & 15u;
    const float m = float(b & 7u);
    return (ex == 0u) ? (m * 0.125f * 0.015625f) : ((1.0f + m * 0.125f) * exp2(float(ex) - 7.0f));
}

static inline float nvfp4_elem(device const uint8_t* blk, uint e) {
    const uint s = e >> 4;
    const uint r = e & 15u;
    const uint q = blk[4u + s * 8u + (r & 7u)];
    const uint code = (r < 8u) ? (q & 15u) : (q >> 4);
    const float mag = (code & 7u) == 0u ? 0.0f
        : ((code & 7u) < 5u ? float(code & 7u) * 0.5f
        : ((code & 7u) == 5u ? 3.0f : ((code & 7u) == 6u ? 4.0f : 6.0f)));
    return ((code & 8u) ? -mag : mag) * nvfp4_scale(blk[s]);
}

static inline float q2_0_elem(device const uint8_t* blk, uint e) {
    const uint q = (uint(blk[2u + (e >> 2)]) >> (2u * (e & 3u))) & 3u;
    return (float(q) - 1.0f) * ext_f16(blk);
}

// ggml's MXFP4 codebook (index 8 is a second +0, not -0) and its E8M0 scale:
// e >= 2 is 2^(e - 128) built in the fp32 exponent field, e < 2 are the two
// subnormals 0x00200000 << e. Recovered from ggml, as in the CPU decoder.
constant constexpr float kExtMxfp4Values[16] = {
    0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 6.0f, 8.0f, 12.0f,
    0.0f, -1.0f, -2.0f, -3.0f, -4.0f, -6.0f, -8.0f, -12.0f};

static inline float mxfp4_elem(device const uint8_t* blk, uint e) {
    const uint x = blk[0];
    const uint bits = (x < 2u) ? (0x00200000u << x) : ((x - 1u) << 23);
    const uint q = blk[1u + (e & 15u)];
    const uint code = (e < 16u) ? (q & 15u) : (q >> 4);
    return kExtMxfp4Values[code] * as_type<float>(bits);
}

// Unsloth fork IQ1_XS / IQ1_XXS / IQ1_XXXS (ggml ids 64 to 66): IQ1_S's ternary
// codebook behind a narrower index. 256 elems; the layout and index
// reassembly are in `crates/compute/src/quant_gguf_iq1_narrow.rs`. The grid
// tables (`kIq1XsGridTable` ...) come from dequant_iq1_narrow_tables.metal,
// which is compiled BEFORE this file.
//   IQ1_XS  : 46 B = f16 d, qs[32], qh[8], sc[4]
//   IQ1_XXS : 42 B = f16 d, qs[32], qh[8]
//   IQ1_XXXS: 38 B = f16 d, qs[32], sc[4]
static inline float iq1n_value(ulong grid_word, uint j, float d, uint code, bool negative) {
    int g = int((grid_word >> (8u * j)) & 0xFFul);
    if (g > 127) g -= 256;
    return d * float(2u * code + 1u) * (float(g) + (negative ? -0.125f : 0.125f));
}

static inline float iq1_xs_elem(device const uint8_t* blk, uint e) {
    const uint g = e >> 3;
    const uint ib = g >> 2;
    const uint l = g & 3u;
    const uint idx = uint(blk[2u + g]) | (((uint(blk[34u + ib]) >> (2u * l)) & 3u) << 8);
    const uint nib = (uint(blk[42u + (ib >> 1)]) >> (4u * (ib & 1u))) & 0xFu;
    return iq1n_value(kIq1XsGridTable[idx], e & 7u, ext_f16(blk), nib & 7u, (nib & 8u) != 0u);
}

static inline float iq1_xxs_elem(device const uint8_t* blk, uint e) {
    const uint g = e >> 3;
    const uint ib = g >> 2;
    const uint l = g & 3u;
    const uint h = blk[34u + ib];
    const uint idx = uint(blk[2u + g]) | (((h >> l) & 1u) << 8);
    return iq1n_value(kIq1XxsGridTable[idx], e & 7u, ext_f16(blk), (h >> 4) & 7u, (h & 0x80u) != 0u);
}

static inline float iq1_xxxs_elem(device const uint8_t* blk, uint e) {
    const uint g = e >> 3;
    const uint ib = g >> 2;
    const uint idx = blk[2u + g];
    const uint nib = (uint(blk[34u + (ib >> 1)]) >> (4u * (ib & 1u))) & 0xFu;
    return iq1n_value(kIq1XxxsGridTable[idx], e & 7u, ext_f16(blk), nib & 7u, (nib & 8u) != 0u);
}

#define EXT_KERNELS(NAME, BLOCK_ELEMS, BLOCK_BYTES, ELEM)                                \
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
    const uint row = tg_idx * kRowsPerTGExt + sg_idx;                                    \
    if (row >= M) return;                                                                \
    const uint n_blocks = N / BLOCK_ELEMS;                                               \
    device const uint8_t* W_row = W + ulong(row) * n_blocks * BLOCK_BYTES;               \
    float acc = 0.0f;                                                                    \
    for (uint b = 0; b < n_blocks; ++b) {                                                \
        device const uint8_t* blk = W_row + b * BLOCK_BYTES;                             \
        for (uint e = lane; e < BLOCK_ELEMS; e += 32) {                                  \
            acc = fma(ELEM(blk, e), float(x[b * BLOCK_ELEMS + e]), acc);                 \
        }                                                                                \
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
    const ulong row_bytes = ulong(D / BLOCK_ELEMS) * BLOCK_BYTES;                        \
    device const uint8_t* blk =                                                          \
        table + ulong(token_id) * row_bytes + ulong(gid / BLOCK_ELEMS) * BLOCK_BYTES;    \
    out[gid] = half(ELEM(blk, gid % BLOCK_ELEMS) * out_scale);                           \
}

EXT_KERNELS(q1_0, 128, 18, q1_0_elem)
EXT_KERNELS(tq2_0, 256, 66, tq2_0_elem)
EXT_KERNELS(tq1_0, 256, 54, tq1_0_elem)
EXT_KERNELS(nvfp4, 64, 36, nvfp4_elem)
EXT_KERNELS(q2_0, 64, 18, q2_0_elem)
EXT_KERNELS(mxfp4, 32, 17, mxfp4_elem)
EXT_KERNELS(iq1_xs, 256, 46, iq1_xs_elem)
EXT_KERNELS(iq1_xxs, 256, 42, iq1_xxs_elem)
EXT_KERNELS(iq1_xxxs, 256, 38, iq1_xxxs_elem)
