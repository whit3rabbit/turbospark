//! Bench prototype: an 8-row INT4 GEMM on `simdgroup_matrix` with REGISTER
//! operands, read straight from the resident MLX-affine layout
//! (`docs/INT4_SMALL_M_MMA.md`, `docs/KAPITSA_KERNELS.md` item 7).
//!
//! `dequant_int4_mma.metal` lost at every width because it dequantized every
//! weight into threadgroup memory before `simdgroup_load`. This shape skips
//! both: each lane fills its two `thread_elements()` of the A operand from
//! its own packed words, and a 4-bit code becomes an exact half in one or
//! two integer ops. The activation side carries the permutation instead: a
//! prep pass writes the 8 token rows into an "X8 table" in fragment order,
//! so a lane's 16 codes for a 64-group are contiguous (one 8-byte read).
//!
//! The math per 64-group and output row n, token t:
//!   y[t][n] += s * sum_k q[n][k] x[t][k] + b * sum_k x[t][k]
//! with the code operand either `1024 + q` (half bits `0x6400 | q`, epilogue
//! subtracts `1024 s * sum`) or the subnormal `q * 2^-24` (bare bits,
//! epilogue scales by `s * 2^24`). Group sums of each token row come from
//! the prep pass.
//!
//! The protocol is the `dequant_int4_mma.metal` one: a shader kept OUT of
//! the production library, a correctness test that must pass before any
//! ratio means anything, and an `#[ignore]`d release bench whose ratio is
//! the deliverable. NOT bit-exact against the GEMV: the MMA chain sums in a
//! different order. The correctness test bounds the error instead.
//!
//! ```sh
//! cargo test -p turbospark-gpu --test int4_mma8_reg_bench --release -- --nocapture
//! cargo test -p turbospark-gpu --test int4_mma8_reg_bench --release -- --ignored --nocapture
//! ```

#![cfg(target_os = "macos")]

use metal::FunctionConstantValues;
use turbospark_gpu::{
    autorelease_pool, encode_dequant_int4_gemm_resident, encode_dequant_int4_gemv_resident,
    read_buffer_f16, Int4ResidentMatrix, MetalBuffer, MetalContext,
};

static SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant constexpr uint kG = 64;     // quantization group
constant constexpr uint kRows = 8;   // token rows of one tile

// Lane -> (fm, fn): the row and first column of the two elements a lane
// holds in an 8x8 simdgroup_matrix (MLX steel's documented layout).
inline uint2 frag_coord(uint lane) {
    const uint q = lane >> 2;
    return uint2((q & 4u) | ((lane >> 1) & 3u), ((q & 2u) << 1) | ((lane & 1u) << 1));
}

// Physical input (0..63 within a group) of k-slot j of fragment f. Chosen so
// that the lane holding slots (j, j+1), j = fn, reads its 16 codes as one
// contiguous run starting at 16 * (fn / 2): fragment f takes nibbles
// (f % 4) and (f % 4) + 4 of word f / 4 of that run.
inline uint phys(uint f, uint j) {
    return 16u * (j >> 1) + 8u * (f >> 2) + (f & 3u) + 4u * (j & 1u);
}

// X8 table: table[(g * 32 + lane) * 16 + 2 f + e] = x[fn + e][64 g + phys(f, fm)]
// and sums[g * 8 + t] = sum over group g of float(x[t][k]), in index order.
// Rows t >= R are zero. One threadgroup of 32 threads per group.
kernel void x8_table(
    device const half* x     [[buffer(0)]],   // [R][K], token-major
    device half*       table [[buffer(1)]],
    device float*      sums  [[buffer(2)]],
    constant uint&     K     [[buffer(3)]],
    constant uint&     R     [[buffer(4)]],
    uint g    [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]
) {
    const uint2 c = frag_coord(lane);
    device half* out = table + (g * 32u + lane) * 16u;
    for (uint f = 0; f < 8u; ++f) {
        const uint k = g * kG + phys(f, c.x);
        for (uint e = 0; e < 2u; ++e) {
            const uint t = c.y + e;
            out[2u * f + e] = t < R ? x[t * K + k] : half(0.0h);
        }
    }
    if (lane < kRows) {
        float s = 0.0f;
        if (lane < R) {
            for (uint k = 0; k < kG; ++k) s += float(x[lane * K + g * kG + k]);
        }
        sums[g * kRows + lane] = s;
    }
}

// c += a x b on register operands, fp32 accumulate.
inline void mma(thread float2& c, half2 a, half2 b) {
    simdgroup_half8x8 A, B;
    simdgroup_float8x8 C;
    A.thread_elements()[0] = a.x;
    A.thread_elements()[1] = a.y;
    B.thread_elements()[0] = b.x;
    B.thread_elements()[1] = b.y;
    C.thread_elements()[0] = c.x;
    C.thread_elements()[1] = c.y;
    simdgroup_multiply_accumulate(C, A, B, C);
    c = float2(C.thread_elements()[0], C.thread_elements()[1]);
}

struct Mma8Params {
    uint N;    // output rows
    uint K;    // inputs, a multiple of 64
    uint R;    // live token rows, 1..8
    uint ks;   // K slices per row block (simdgroups summed in fixed order)
    uint cs;   // row blocks per threadgroup
};

// One simdgroup owns NF fragments (8 NF output rows) over one K slice.
// Weights are read 2-byte aligned (the resident shape), as the GEMV does.
template <uint NF, bool SUB>
inline void mma8_impl(
    device const uchar*  W, device const bfloat* S, device const bfloat* Bi,
    device const half*   table, device const float* sums, device half* y,
    constant Mma8Params& p, threadgroup float2* red,
    uint tg, uint sg, uint lane
) {
    const uint2 c = frag_coord(lane);
    const uint fm = c.x, fn = c.y;
    const uint cb = sg / p.ks, kslice = sg % p.ks;
    const uint row0 = (tg * p.cs + cb) * NF * 8u;
    const uint G = p.K / kG;
    const uint g0 = kslice * G / p.ks, g1 = (kslice + 1u) * G / p.ks;
    const uint row_bytes = p.K / 2u;
    const uint chunk = fn >> 1;

    float2 acc[NF];
    for (uint f2 = 0; f2 < NF; ++f2) acc[f2] = float2(0.0f);

    for (uint g = g0; g < g1; ++g) {
        device const half4* tb = (device const half4*)(table + (g * 32u + lane) * 16u);
        const half4 x0 = tb[0], x1 = tb[1], x2 = tb[2], x3 = tb[3];
        const half2 b[8] = {x0.xy, x0.zw, x1.xy, x1.zw, x2.xy, x2.zw, x3.xy, x3.zw};
        const float2 sx = float2(sums[g * kRows + fn], sums[g * kRows + fn + 1u]);
        for (uint f2 = 0; f2 < NF; ++f2) {
            // The MMA is simdgroup-collective, so a row past N must not
            // skip it: its lanes multiply zero codes and drop the result.
            const uint n = row0 + 8u * f2 + fm;
            const bool live = n < p.N;
            uint w0 = 0u, w1 = 0u;
            float s = 0.0f, bi = 0.0f;
            if (live) {
                device const ushort* wp =
                    (device const ushort*)(W + n * row_bytes + g * (kG / 2u) + chunk * 8u);
                w0 = uint(wp[0]) | (uint(wp[1]) << 16);
                w1 = uint(wp[2]) | (uint(wp[3]) << 16);
                s = float(S[n * G + g]);
                bi = float(Bi[n * G + g]);
            }
            float2 dot = float2(0.0f);
            for (uint f = 0; f < 8u; ++f) {
                const uint word = f < 4u ? w0 : w1;
                uint bits = (word >> (4u * (f & 3u))) & 0x000F000Fu;
                if (!SUB) bits |= 0x64006400u;
                mma(dot, as_type<half2>(bits), b[f]);
            }
            if (SUB) {
                acc[f2] = fma(dot, float2(s * 16777216.0f), acc[f2]);
                acc[f2] = fma(sx, float2(bi), acc[f2]);
            } else {
                acc[f2] = fma(dot, float2(s), acc[f2]);
                acc[f2] = fma(sx, float2(fma(-1024.0f, s, bi)), acc[f2]);
            }
        }
    }

    if (p.ks > 1u) {
        for (uint f2 = 0; f2 < NF; ++f2) red[(sg * NF + f2) * 32u + lane] = acc[f2];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (kslice != 0u) return;
        for (uint f2 = 0; f2 < NF; ++f2) {
            float2 t = float2(0.0f);
            for (uint k = 0; k < p.ks; ++k) t += red[((sg + k) * NF + f2) * 32u + lane];
            acc[f2] = t;
        }
    }
    for (uint f2 = 0; f2 < NF; ++f2) {
        const uint n = row0 + 8u * f2 + fm;
        if (n >= p.N) continue;
        if (fn < p.R) y[fn * p.N + n] = half(acc[f2].x);
        if (fn + 1u < p.R) y[(fn + 1u) * p.N + n] = half(acc[f2].y);
    }
}

// Variant: the next group's operands are loaded before this group's MMAs
// run, and the 8 MMAs of a group alternate between two accumulators so
// they are not one dependent chain. Same epilogue; the two chains are
// summed per group, so a row's bits still depend on nothing but its own
// inputs.
struct Mma8Stage {
    half2 b[8];
    float2 sx;
};

template <uint NF, bool SUB>
inline void mma8_impl_pf(
    device const uchar*  W, device const bfloat* S, device const bfloat* Bi,
    device const half*   table, device const float* sums, device half* y,
    constant Mma8Params& p, threadgroup float2* red,
    uint tg, uint sg, uint lane
) {
    const uint2 c = frag_coord(lane);
    const uint fm = c.x, fn = c.y;
    const uint cb = sg / p.ks, kslice = sg % p.ks;
    const uint row0 = (tg * p.cs + cb) * NF * 8u;
    const uint G = p.K / kG;
    const uint g0 = kslice * G / p.ks, g1 = (kslice + 1u) * G / p.ks;
    const uint row_bytes = p.K / 2u;
    const uint chunk = fn >> 1;

    bool live[NF];
    device const ushort* wbase[NF];
    for (uint f2 = 0; f2 < NF; ++f2) {
        const uint n = row0 + 8u * f2 + fm;
        live[f2] = n < p.N;
        wbase[f2] = (device const ushort*)(W + (live[f2] ? n : 0u) * row_bytes + chunk * 8u);
    }

    float2 acc[NF];
    for (uint f2 = 0; f2 < NF; ++f2) acc[f2] = float2(0.0f);

    // Stage for group g: table, sums, and each fragment's words and coefficients.
    Mma8Stage st;
    uint w0[NF], w1[NF];
    float s[NF], bi[NF];
    auto load = [&](uint g, thread Mma8Stage& o, thread uint* a0, thread uint* a1,
                    thread float* ss, thread float* bb) {
        device const half4* tb = (device const half4*)(table + (g * 32u + lane) * 16u);
        const half4 x0 = tb[0], x1 = tb[1], x2 = tb[2], x3 = tb[3];
        o.b[0] = x0.xy; o.b[1] = x0.zw; o.b[2] = x1.xy; o.b[3] = x1.zw;
        o.b[4] = x2.xy; o.b[5] = x2.zw; o.b[6] = x3.xy; o.b[7] = x3.zw;
        o.sx = float2(sums[g * kRows + fn], sums[g * kRows + fn + 1u]);
        for (uint f2 = 0; f2 < NF; ++f2) {
            if (live[f2]) {
                device const ushort* wp = wbase[f2] + g * (kG / 4u);
                a0[f2] = uint(wp[0]) | (uint(wp[1]) << 16);
                a1[f2] = uint(wp[2]) | (uint(wp[3]) << 16);
                const uint n = row0 + 8u * f2 + fm;
                ss[f2] = float(S[n * G + g]);
                bb[f2] = float(Bi[n * G + g]);
            } else {
                a0[f2] = 0u; a1[f2] = 0u; ss[f2] = 0.0f; bb[f2] = 0.0f;
            }
        }
    };
    if (g0 < g1) load(g0, st, w0, w1, s, bi);
    for (uint g = g0; g < g1; ++g) {
        const Mma8Stage cur = st;
        uint c0[NF], c1[NF];
        float cs_[NF], cb_[NF];
        for (uint f2 = 0; f2 < NF; ++f2) {
            c0[f2] = w0[f2]; c1[f2] = w1[f2]; cs_[f2] = s[f2]; cb_[f2] = bi[f2];
        }
        if (g + 1u < g1) load(g + 1u, st, w0, w1, s, bi);
        for (uint f2 = 0; f2 < NF; ++f2) {
            float2 da = float2(0.0f), db = float2(0.0f);
            for (uint f = 0; f < 8u; f += 2u) {
                const uint word = f < 4u ? c0[f2] : c1[f2];
                uint ba = (word >> (4u * (f & 3u))) & 0x000F000Fu;
                uint bb2 = (word >> (4u * ((f + 1u) & 3u))) & 0x000F000Fu;
                if (!SUB) { ba |= 0x64006400u; bb2 |= 0x64006400u; }
                mma(da, as_type<half2>(ba), cur.b[f]);
                mma(db, as_type<half2>(bb2), cur.b[f + 1u]);
            }
            const float2 dot = da + db;
            if (SUB) {
                acc[f2] = fma(dot, float2(cs_[f2] * 16777216.0f), acc[f2]);
                acc[f2] = fma(cur.sx, float2(cb_[f2]), acc[f2]);
            } else {
                acc[f2] = fma(dot, float2(cs_[f2]), acc[f2]);
                acc[f2] = fma(cur.sx, float2(fma(-1024.0f, cs_[f2], cb_[f2])), acc[f2]);
            }
        }
    }

    if (p.ks > 1u) {
        for (uint f2 = 0; f2 < NF; ++f2) red[(sg * NF + f2) * 32u + lane] = acc[f2];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (kslice != 0u) return;
        for (uint f2 = 0; f2 < NF; ++f2) {
            float2 t = float2(0.0f);
            for (uint k = 0; k < p.ks; ++k) t += red[((sg + k) * NF + f2) * 32u + lane];
            acc[f2] = t;
        }
    }
    for (uint f2 = 0; f2 < NF; ++f2) {
        if (!live[f2]) continue;
        const uint n = row0 + 8u * f2 + fm;
        if (fn < p.R) y[fn * p.N + n] = half(acc[f2].x);
        if (fn + 1u < p.R) y[(fn + 1u) * p.N + n] = half(acc[f2].y);
    }
}

#define MMA8PF(NAME, NF, SUB)                                                       \
kernel void NAME(                                                                   \
    device const uchar*  W     [[buffer(0)]],                                       \
    device const bfloat* S     [[buffer(1)]],                                       \
    device const bfloat* Bi    [[buffer(2)]],                                       \
    device const half*   table [[buffer(3)]],                                       \
    device const float*  sums  [[buffer(4)]],                                       \
    device half*         y     [[buffer(5)]],                                       \
    constant Mma8Params& p     [[buffer(6)]],                                       \
    uint tg   [[threadgroup_position_in_grid]],                                     \
    uint sg   [[simdgroup_index_in_threadgroup]],                                   \
    uint lane [[thread_index_in_simdgroup]]                                         \
) {                                                                                 \
    threadgroup float2 red[16 * NF * 32];                                           \
    mma8_impl_pf<NF, SUB>(W, S, Bi, table, sums, y, p, red, tg, sg, lane);          \
}

MMA8PF(int4_mma8pf_sub_nf1, 1, true)
MMA8PF(int4_mma8pf_sub_nf2, 2, true)

// Variant: two groups per step, so a lane reads 16 contiguous bytes of its
// row per step instead of 8. The per-group epilogue is unchanged, so the
// summation order matches `mma8_impl` exactly.
template <uint NF>
inline void mma8_impl_g2(
    device const uchar*  W, device const bfloat* S, device const bfloat* Bi,
    device const half*   table, device const float* sums, device half* y,
    constant Mma8Params& p, threadgroup float2* red,
    uint tg, uint sg, uint lane
) {
    const uint2 c = frag_coord(lane);
    const uint fm = c.x, fn = c.y;
    const uint cb = sg / p.ks, kslice = sg % p.ks;
    const uint row0 = (tg * p.cs + cb) * NF * 8u;
    const uint G = p.K / kG;
    const uint g0 = kslice * G / p.ks, g1 = (kslice + 1u) * G / p.ks;
    const uint row_bytes = p.K / 2u;
    const uint chunk = fn >> 1;

    float2 acc[NF];
    for (uint f2 = 0; f2 < NF; ++f2) acc[f2] = float2(0.0f);

    uint g = g0;
    for (; g + 1u < g1; g += 2u) {
        half2 b[2][8];
        float2 sx[2];
        for (uint h = 0; h < 2u; ++h) {
            device const half4* tb = (device const half4*)(table + ((g + h) * 32u + lane) * 16u);
            const half4 x0 = tb[0], x1 = tb[1], x2 = tb[2], x3 = tb[3];
            b[h][0] = x0.xy; b[h][1] = x0.zw; b[h][2] = x1.xy; b[h][3] = x1.zw;
            b[h][4] = x2.xy; b[h][5] = x2.zw; b[h][6] = x3.xy; b[h][7] = x3.zw;
            sx[h] = float2(sums[(g + h) * kRows + fn], sums[(g + h) * kRows + fn + 1u]);
        }
        for (uint f2 = 0; f2 < NF; ++f2) {
            const uint n = row0 + 8u * f2 + fm;
            const bool live = n < p.N;
            uint w[2][2] = {{0u, 0u}, {0u, 0u}};
            float s[2] = {0.0f, 0.0f}, bi[2] = {0.0f, 0.0f};
            if (live) {
                for (uint h = 0; h < 2u; ++h) {
                    device const ushort* wp = (device const ushort*)(
                        W + n * row_bytes + (g + h) * (kG / 2u) + chunk * 8u);
                    w[h][0] = uint(wp[0]) | (uint(wp[1]) << 16);
                    w[h][1] = uint(wp[2]) | (uint(wp[3]) << 16);
                    s[h] = float(S[n * G + g + h]);
                    bi[h] = float(Bi[n * G + g + h]);
                }
            }
            for (uint h = 0; h < 2u; ++h) {
                float2 dot = float2(0.0f);
                for (uint f = 0; f < 8u; ++f) {
                    const uint word = f < 4u ? w[h][0] : w[h][1];
                    const uint bits = (word >> (4u * (f & 3u))) & 0x000F000Fu;
                    mma(dot, as_type<half2>(bits), b[h][f]);
                }
                acc[f2] = fma(dot, float2(s[h] * 16777216.0f), acc[f2]);
                acc[f2] = fma(sx[h], float2(bi[h]), acc[f2]);
            }
        }
    }
    for (; g < g1; ++g) {
        device const half4* tb = (device const half4*)(table + (g * 32u + lane) * 16u);
        const half4 x0 = tb[0], x1 = tb[1], x2 = tb[2], x3 = tb[3];
        const half2 b[8] = {x0.xy, x0.zw, x1.xy, x1.zw, x2.xy, x2.zw, x3.xy, x3.zw};
        const float2 sx = float2(sums[g * kRows + fn], sums[g * kRows + fn + 1u]);
        for (uint f2 = 0; f2 < NF; ++f2) {
            const uint n = row0 + 8u * f2 + fm;
            uint w0 = 0u, w1 = 0u;
            float s = 0.0f, bi = 0.0f;
            if (n < p.N) {
                device const ushort* wp =
                    (device const ushort*)(W + n * row_bytes + g * (kG / 2u) + chunk * 8u);
                w0 = uint(wp[0]) | (uint(wp[1]) << 16);
                w1 = uint(wp[2]) | (uint(wp[3]) << 16);
                s = float(S[n * G + g]);
                bi = float(Bi[n * G + g]);
            }
            float2 dot = float2(0.0f);
            for (uint f = 0; f < 8u; ++f) {
                const uint word = f < 4u ? w0 : w1;
                mma(dot, as_type<half2>((word >> (4u * (f & 3u))) & 0x000F000Fu), b[f]);
            }
            acc[f2] = fma(dot, float2(s * 16777216.0f), acc[f2]);
            acc[f2] = fma(sx, float2(bi), acc[f2]);
        }
    }

    if (p.ks > 1u) {
        for (uint f2 = 0; f2 < NF; ++f2) red[(sg * NF + f2) * 32u + lane] = acc[f2];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (kslice != 0u) return;
        for (uint f2 = 0; f2 < NF; ++f2) {
            float2 t = float2(0.0f);
            for (uint k = 0; k < p.ks; ++k) t += red[((sg + k) * NF + f2) * 32u + lane];
            acc[f2] = t;
        }
    }
    for (uint f2 = 0; f2 < NF; ++f2) {
        const uint n = row0 + 8u * f2 + fm;
        if (n >= p.N) continue;
        if (fn < p.R) y[fn * p.N + n] = half(acc[f2].x);
        if (fn + 1u < p.R) y[(fn + 1u) * p.N + n] = half(acc[f2].y);
    }
}

#define MMA8G2(NAME, NF)                                                            \
kernel void NAME(                                                                   \
    device const uchar*  W     [[buffer(0)]],                                       \
    device const bfloat* S     [[buffer(1)]],                                       \
    device const bfloat* Bi    [[buffer(2)]],                                       \
    device const half*   table [[buffer(3)]],                                       \
    device const float*  sums  [[buffer(4)]],                                       \
    device half*         y     [[buffer(5)]],                                       \
    constant Mma8Params& p     [[buffer(6)]],                                       \
    uint tg   [[threadgroup_position_in_grid]],                                     \
    uint sg   [[simdgroup_index_in_threadgroup]],                                   \
    uint lane [[thread_index_in_simdgroup]]                                         \
) {                                                                                 \
    threadgroup float2 red[16 * NF * 32];                                           \
    mma8_impl_g2<NF>(W, S, Bi, table, sums, y, p, red, tg, sg, lane);               \
}

MMA8G2(int4_mma8g2_sub_nf1, 1)
MMA8G2(int4_mma8g2_sub_nf2, 2)

// ABLATIONS of the base kernel (nf 1, subnormal codes), for timing only:
// the outputs are wrong by design. ABL 0 is the base kernel's work; 1 skips
// the MMAs (loads kept live); 2 reads weights and coefficients from the
// slice's first group every iteration (cache-resident, no weight DRAM
// traffic); 3 reads the table and sums from the first group (no table
// traffic); 4 is both 2 and 3 (compute alone).
template <uint ABL>
inline void mma8_abl_impl(
    device const uchar*  W, device const bfloat* S, device const bfloat* Bi,
    device const half*   table, device const float* sums, device half* y,
    constant Mma8Params& p, threadgroup float2* red,
    uint tg, uint sg, uint lane
) {
    const uint2 c = frag_coord(lane);
    const uint fm = c.x, fn = c.y;
    const uint kslice = sg % p.ks;
    const uint n = tg * 8u + fm;
    const uint G = p.K / kG;
    const uint g0 = kslice * G / p.ks, g1 = (kslice + 1u) * G / p.ks;
    const bool live = n < p.N;
    device const uchar* wrow = W + (live ? n : 0u) * (p.K / 2u) + (fn >> 1) * 8u;
    float2 acc = float2(0.0f);
    for (uint g = g0; g < g1; ++g) {
        const uint gt = (ABL == 3u || ABL == 4u) ? g0 : g;
        const uint gw = (ABL == 2u || ABL == 4u) ? g0 : g;
        device const half4* tb = (device const half4*)(table + (gt * 32u + lane) * 16u);
        const half4 x0 = tb[0], x1 = tb[1], x2 = tb[2], x3 = tb[3];
        const half2 b[8] = {x0.xy, x0.zw, x1.xy, x1.zw, x2.xy, x2.zw, x3.xy, x3.zw};
        const float2 sx = float2(sums[gt * kRows + fn], sums[gt * kRows + fn + 1u]);
        uint w0 = 0u, w1 = 0u;
        float s = 0.0f, bi = 0.0f;
        if (live) {
            if (ABL == 6u) {
                // One 8-byte load: needs 8-aligned weights (the bench pool
                // for this arm is allocated that way).
                const uint2 v = *(device const uint2*)(wrow + gw * (kG / 2u));
                w0 = v.x;
                w1 = v.y;
            } else {
                device const ushort* wp = (device const ushort*)(wrow + gw * (kG / 2u));
                w0 = uint(wp[0]) | (uint(wp[1]) << 16);
                w1 = uint(wp[2]) | (uint(wp[3]) << 16);
            }
            s = float(S[n * G + gw]);
            bi = float(Bi[n * G + gw]);
        }
        float2 dot = float2(0.0f);
        if (ABL == 5u) {
            // Persistent accumulator: one simdgroup_float8x8 for the whole
            // group, so each MMA only writes its A and B operands.
            simdgroup_float8x8 D = simdgroup_float8x8(0.0f);
            for (uint f = 0; f < 8u; ++f) {
                const uint word = f < 4u ? w0 : w1;
                simdgroup_half8x8 A, B;
                A.thread_elements()[0] = as_type<half2>((word >> (4u * (f & 3u))) & 0x000F000Fu).x;
                A.thread_elements()[1] = as_type<half2>((word >> (4u * (f & 3u))) & 0x000F000Fu).y;
                B.thread_elements()[0] = b[f].x;
                B.thread_elements()[1] = b[f].y;
                simdgroup_multiply_accumulate(D, A, B, D);
            }
            dot = float2(D.thread_elements()[0], D.thread_elements()[1]);
        } else {
            for (uint f = 0; f < 8u; ++f) {
                const uint word = f < 4u ? w0 : w1;
                const half2 a = as_type<half2>((word >> (4u * (f & 3u))) & 0x000F000Fu);
                if (ABL == 1u) {
                    dot += float2(a) * float2(b[f]);
                } else {
                    mma(dot, a, b[f]);
                }
            }
        }
        acc = fma(dot, float2(s * 16777216.0f), acc);
        acc = fma(sx, float2(bi), acc);
    }
    if (p.ks > 1u) {
        red[sg * 32u + lane] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (kslice != 0u) return;
        acc = float2(0.0f);
        for (uint k = 0; k < p.ks; ++k) acc += red[k * 32u + lane];
    }
    if (!live) return;
    if (fn < p.R) y[fn * p.N + n] = half(acc.x);
    if (fn + 1u < p.R) y[(fn + 1u) * p.N + n] = half(acc.y);
}

#define MMA8ABL(NAME, ABL)                                                          \
kernel void NAME(                                                                   \
    device const uchar*  W     [[buffer(0)]],                                       \
    device const bfloat* S     [[buffer(1)]],                                       \
    device const bfloat* Bi    [[buffer(2)]],                                       \
    device const half*   table [[buffer(3)]],                                       \
    device const float*  sums  [[buffer(4)]],                                       \
    device half*         y     [[buffer(5)]],                                       \
    constant Mma8Params& p     [[buffer(6)]],                                       \
    uint tg   [[threadgroup_position_in_grid]],                                     \
    uint sg   [[simdgroup_index_in_threadgroup]],                                   \
    uint lane [[thread_index_in_simdgroup]]                                         \
) {                                                                                 \
    threadgroup float2 red[16 * 32];                                                \
    mma8_abl_impl<ABL>(W, S, Bi, table, sums, y, p, red, tg, sg, lane);             \
}

MMA8ABL(int4_mma8_abl0, 0u)
MMA8ABL(int4_mma8_abl1, 1u)
MMA8ABL(int4_mma8_abl2, 2u)
MMA8ABL(int4_mma8_abl3, 3u)
MMA8ABL(int4_mma8_abl4, 4u)
MMA8ABL(int4_mma8_abl5, 5u)
MMA8ABL(int4_mma8_abl6, 6u)

#define MMA8(NAME, NF, SUB)                                                         \
kernel void NAME(                                                                   \
    device const uchar*  W     [[buffer(0)]],                                       \
    device const bfloat* S     [[buffer(1)]],                                       \
    device const bfloat* Bi    [[buffer(2)]],                                       \
    device const half*   table [[buffer(3)]],                                       \
    device const float*  sums  [[buffer(4)]],                                       \
    device half*         y     [[buffer(5)]],                                       \
    constant Mma8Params& p     [[buffer(6)]],                                       \
    uint tg   [[threadgroup_position_in_grid]],                                     \
    uint sg   [[simdgroup_index_in_threadgroup]],                                   \
    uint lane [[thread_index_in_simdgroup]]                                         \
) {                                                                                 \
    threadgroup float2 red[16 * NF * 32];                                           \
    mma8_impl<NF, SUB>(W, S, Bi, table, sums, y, p, red, tg, sg, lane);             \
}

MMA8(int4_mma8_sub_nf1, 1, true)
MMA8(int4_mma8_sub_nf2, 2, true)
MMA8(int4_mma8_sub_nf4, 4, true)
MMA8(int4_mma8_or_nf1, 1, false)
MMA8(int4_mma8_or_nf2, 2, false)
MMA8(int4_mma8_or_nf4, 4, false)
"#;

#[derive(Clone, Copy, Debug, PartialEq)]
struct Config {
    nf: usize,
    sub: bool,
    ks: usize,
    cs: usize,
    /// Prefetching, split-chain variant (`mma8_impl_pf`).
    pf: bool,
    /// Two groups per step (`mma8_impl_g2`), subnormal codes only.
    g2: bool,
}

impl Config {
    fn name(&self) -> &'static str {
        if self.g2 {
            return match (self.sub, self.nf) {
                (true, 1) => "int4_mma8g2_sub_nf1",
                (true, 2) => "int4_mma8g2_sub_nf2",
                _ => unreachable!("no such instantiation"),
            };
        }
        if self.pf {
            return match (self.sub, self.nf) {
                (true, 1) => "int4_mma8pf_sub_nf1",
                (true, 2) => "int4_mma8pf_sub_nf2",
                _ => unreachable!("no such instantiation"),
            };
        }
        match (self.sub, self.nf) {
            (true, 1) => "int4_mma8_sub_nf1",
            (true, 2) => "int4_mma8_sub_nf2",
            (true, 4) => "int4_mma8_sub_nf4",
            (false, 1) => "int4_mma8_or_nf1",
            (false, 2) => "int4_mma8_or_nf2",
            (false, 4) => "int4_mma8_or_nf4",
            _ => unreachable!("no such instantiation"),
        }
    }
}

fn xorshift(seed: u64) -> impl FnMut() -> u64 {
    let mut s = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    }
}

fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// One resident INT4 matrix laid out weights|scales|biases, weights at a
/// 2-byte-aligned (not 4-aligned) offset: the shape `.gturbo` hands out.
struct Matrix {
    buffer: MetalBuffer,
    weights_offset: u64,
    scales_offset: u64,
    biases_offset: u64,
    rows: usize,
    cols: usize,
    /// Host copy for the CPU reference (correctness fixtures only).
    host: Option<(Vec<u8>, Vec<u16>, Vec<u16>)>,
}

impl Matrix {
    fn from_parts(
        context: &MetalContext,
        rows: usize,
        cols: usize,
        packed: &[u8],
        scales: &[u16],
        biases: &[u16],
        keep_host: bool,
    ) -> Self {
        Self::from_parts_at(context, rows, cols, packed, scales, biases, keep_host, 2)
    }

    #[allow(clippy::too_many_arguments)]
    fn from_parts_at(
        context: &MetalContext,
        rows: usize,
        cols: usize,
        packed: &[u8],
        scales: &[u16],
        biases: &[u16],
        keep_host: bool,
        weights_offset: u64,
    ) -> Self {
        let scales_offset = (weights_offset + packed.len() as u64).next_multiple_of(2);
        let biases_offset = scales_offset + (scales.len() * 2) as u64;
        let mut bytes = vec![0u8; biases_offset as usize + biases.len() * 2];
        bytes[weights_offset as usize..][..packed.len()].copy_from_slice(packed);
        for (i, s) in scales.iter().enumerate() {
            bytes[scales_offset as usize + 2 * i..][..2].copy_from_slice(&s.to_le_bytes());
        }
        for (i, b) in biases.iter().enumerate() {
            bytes[biases_offset as usize + 2 * i..][..2].copy_from_slice(&b.to_le_bytes());
        }
        Self {
            buffer: context.new_buffer_with_data(&bytes),
            weights_offset,
            scales_offset,
            biases_offset,
            rows,
            cols,
            host: keep_host.then(|| (packed.to_vec(), scales.to_vec(), biases.to_vec())),
        }
    }

    /// Real quantized rows, for correctness.
    fn quantized(context: &MetalContext, rows: usize, cols: usize, seed: u64) -> Self {
        let mut next = xorshift(seed);
        let (mut packed, mut scales, mut biases) = (Vec::new(), Vec::new(), Vec::new());
        for _ in 0..rows {
            let row: Vec<f32> = (0..cols)
                .map(|_| ((next() % 2000) as f32 / 1000.0 - 1.0) * 0.5)
                .collect();
            let q = turbospark_compute::quantize_int4_affine(&row);
            packed.extend_from_slice(&q.packed);
            scales.extend_from_slice(&q.scales);
            biases.extend_from_slice(&q.biases);
        }
        Self::from_parts(context, rows, cols, &packed, &scales, &biases, true)
    }

    /// [`Self::random`] with the weights 16-byte aligned (8-byte load arms).
    fn random_aligned(context: &MetalContext, rows: usize, cols: usize, seed: u64) -> Self {
        let mut next = xorshift(seed);
        let packed: Vec<u8> = (0..rows * cols / 2).map(|_| next() as u8).collect();
        let groups = rows * cols / 64;
        let scales = vec![0x3C23u16; groups];
        let biases = vec![0xBDA4u16; groups];
        Self::from_parts_at(context, rows, cols, &packed, &scales, &biases, false, 16)
    }

    /// Random codes and plausible bf16 scale/bias, for timing only.
    fn random(context: &MetalContext, rows: usize, cols: usize, seed: u64) -> Self {
        let mut next = xorshift(seed);
        let packed: Vec<u8> = (0..rows * cols / 2).map(|_| next() as u8).collect();
        let groups = rows * cols / 64;
        let scales = vec![0x3C23u16; groups]; // bf16 ~0.01
        let biases = vec![0xBDA4u16; groups]; // bf16 ~-0.08
        Self::from_parts(context, rows, cols, &packed, &scales, &biases, false)
    }

    fn resident(&self) -> Int4ResidentMatrix<'_> {
        Int4ResidentMatrix {
            buffer: &self.buffer,
            weights_offset: self.weights_offset,
            scales_offset: self.scales_offset,
            biases_offset: self.biases_offset,
            rows: self.rows,
            cols: self.cols,
        }
    }

    fn cpu_reference(&self, x: &[half::f16], r: usize) -> Vec<f64> {
        let (packed, scales, biases) = self.host.as_ref().expect("host copy");
        let (n_rows, k) = (self.rows, self.cols);
        let g = k / 64;
        let mut y = vec![0.0f64; r * n_rows];
        for t in 0..r {
            for n in 0..n_rows {
                let mut acc = 0.0f64;
                for i in 0..k {
                    let byte = packed[n * k / 2 + i / 2];
                    let q = if i & 1 == 0 { byte & 0xF } else { byte >> 4 } as f64;
                    let s = bf16_to_f32(scales[n * g + i / 64]) as f64;
                    let b = bf16_to_f32(biases[n * g + i / 64]) as f64;
                    acc += (q * s + b) * x[t * k + i].to_f64();
                }
                y[t * n_rows + n] = acc;
            }
        }
        y
    }
}

/// Scratch for one GEMM call: token-major x, the X8 table and sums, y.
struct Rows {
    x: MetalBuffer,
    table: MetalBuffer,
    sums: MetalBuffer,
    y: MetalBuffer,
    host_x: Vec<half::f16>,
}

impl Rows {
    fn new(context: &MetalContext, n: usize, k: usize, seed: u64) -> Self {
        let mut next = xorshift(seed);
        let host_x: Vec<half::f16> = (0..8 * k)
            .map(|_| half::f16::from_f32((next() % 2000) as f32 / 1000.0 - 1.0))
            .collect();
        let bytes: Vec<u8> = host_x
            .iter()
            .flat_map(|v| v.to_bits().to_le_bytes())
            .collect();
        let groups = k / 64;
        Self {
            x: context.new_buffer_with_data(&bytes),
            table: context.new_output_buffer((groups * 32 * 16 * 2) as u64),
            sums: context.new_output_buffer((groups * 8 * 4) as u64),
            y: context.new_output_buffer((8 * n * 2) as u64),
            host_x,
        }
    }
}

fn encode_mma8(
    context: &mut MetalContext,
    pass: &turbospark_gpu::PassEncoder,
    w: &Matrix,
    rows: &Rows,
    r: usize,
    cfg: Config,
) {
    encode_x8_table(context, pass, rows, w.cols, r);
    encode_mma8_gemm(context, pass, w, rows, r, cfg);
}

/// The prep alone: once per distinct input in production, since every
/// projection that reads the same x reads the same table.
fn encode_x8_table(
    context: &mut MetalContext,
    pass: &turbospark_gpu::PassEncoder,
    rows: &Rows,
    k: usize,
    r: usize,
) {
    let prep = context
        .pipeline(SOURCE, "x8_table", &FunctionConstantValues::new(), b"")
        .unwrap();
    let (k32, r32) = ((k as u32).to_le_bytes(), (r as u32).to_le_bytes());
    pass.encode_threadgroups(
        &prep,
        &[(&rows.x, 0, 0), (&rows.table, 1, 0), (&rows.sums, 2, 0)],
        &[(&k32[..], 3), (&r32[..], 4)],
        (k / 64) as u64,
        32,
    );
}

/// The GEMM alone, reading a table an earlier prep wrote.
fn encode_mma8_gemm(
    context: &mut MetalContext,
    pass: &turbospark_gpu::PassEncoder,
    w: &Matrix,
    rows: &Rows,
    r: usize,
    cfg: Config,
) {
    let (n, k) = (w.rows, w.cols);
    let gemm = context
        .pipeline(SOURCE, cfg.name(), &FunctionConstantValues::new(), b"")
        .unwrap();
    let params: Vec<u8> = [n, k, r, cfg.ks, cfg.cs]
        .iter()
        .flat_map(|v| (*v as u32).to_le_bytes())
        .collect();
    let blocks = n.div_ceil(8 * cfg.nf);
    pass.encode_threadgroups(
        &gemm,
        &[
            (&w.buffer, 0, w.weights_offset),
            (&w.buffer, 1, w.scales_offset),
            (&w.buffer, 2, w.biases_offset),
            (&rows.table, 3, 0),
            (&rows.sums, 4, 0),
            (&rows.y, 5, 0),
        ],
        &[(&params[..], 6)],
        blocks.div_ceil(cfg.cs) as u64,
        (cfg.ks * cfg.cs * 32) as u64,
    );
}

fn max_abs_err(got: &[half::f16], want: &[f64]) -> f64 {
    got.iter()
        .zip(want)
        .map(|(g, w)| (g.to_f64() - w).abs())
        .fold(0.0, f64::max)
}

/// The MMA kernel must be as accurate as the shipped GEMM against an f64
/// reference, at every live-row count, remainder row block and config.
/// It is NOT bit-identical (different summation order), so this is the
/// gate that stands in for bit-identity.
#[test]
fn mma8_matches_the_reference_as_closely_as_the_shipped_gemm() {
    let mut context = MetalContext::new().expect("Metal device");
    let configs = [
        Config {
            nf: 1,
            sub: true,
            ks: 1,
            cs: 1,
            pf: false,
            g2: false,
        },
        Config {
            nf: 2,
            sub: true,
            ks: 2,
            cs: 2,
            pf: false,
            g2: false,
        },
        Config {
            nf: 4,
            sub: true,
            ks: 4,
            cs: 1,
            pf: false,
            g2: false,
        },
        Config {
            nf: 1,
            sub: false,
            ks: 1,
            cs: 4,
            pf: false,
            g2: false,
        },
        Config {
            nf: 2,
            sub: false,
            ks: 2,
            cs: 1,
            pf: false,
            g2: false,
        },
        Config {
            nf: 1,
            sub: true,
            ks: 4,
            cs: 1,
            pf: true,
            g2: false,
        },
        Config {
            nf: 2,
            sub: true,
            ks: 2,
            cs: 2,
            pf: true,
            g2: false,
        },
        Config {
            nf: 1,
            sub: true,
            ks: 4,
            cs: 1,
            pf: false,
            g2: true,
        },
        Config {
            nf: 2,
            sub: true,
            ks: 3,
            cs: 2,
            pf: false,
            g2: true,
        },
    ];
    for &(n, k) in &[(256usize, 2048usize), (44, 1024), (512, 4096)] {
        let w = Matrix::quantized(&context, n, k, 0xA8_000 + n as u64);
        let rows = Rows::new(&context, n, k, 0xA8_100 + k as u64);
        for r in [1usize, 3, 8] {
            let want = w.cpu_reference(&rows.host_x, r);
            let scale = want.iter().map(|v| v.abs()).fold(0.0, f64::max);

            let gemm_y = context.new_output_buffer((8 * n * 2) as u64);
            let pass = context.begin_pass();
            encode_dequant_int4_gemm_resident(
                &mut context,
                &pass,
                &w.resident(),
                (&rows.x, 0),
                (&gemm_y, 0),
                r,
            )
            .unwrap();
            pass.commit_and_wait();
            let gemm = read_buffer_f16(&gemm_y, 0, r * n);
            let gemm_err = max_abs_err(&gemm, &want);

            for cfg in configs {
                let pass = context.begin_pass();
                encode_mma8(&mut context, &pass, &w, &rows, r, cfg);
                pass.commit_and_wait();
                let got = read_buffer_f16(&rows.y, 0, r * n);
                let err = max_abs_err(&got, &want);
                let same = got.iter().zip(&gemm).filter(|(a, b)| a == b).count();
                println!(
                    "{n}x{k} R={r} {cfg:?}: max|err| {err:.3e} (gemm {gemm_err:.3e}, \
                     |y| {scale:.2}), bit-equal to gemm {same}/{}",
                    gemm.len()
                );
                // Two half ulps at the output's scale on top of the GEMM's
                // own error: anything past that is a layout or epilogue bug,
                // not summation order.
                let bound = 2.0 * gemm_err + scale * 2.0f64.powi(-10);
                assert!(
                    err <= bound,
                    "{n}x{k} R={r} {cfg:?}: max|err| {err:.3e} exceeds {bound:.3e}"
                );
            }
        }
    }
}

/// A token row's output must not depend on how many rows share the tile.
/// This is what lets decode (one row) and verify (several) share one
/// arithmetic and stay bit-exact against each other: run row 0 alone, then
/// inside a full tile of 8, and compare bits.
#[test]
fn a_rows_bits_do_not_depend_on_the_rows_beside_it() {
    let mut context = MetalContext::new().expect("Metal device");
    let configs = [
        Config {
            nf: 1,
            sub: true,
            ks: 1,
            cs: 1,
            pf: false,
            g2: false,
        },
        Config {
            nf: 1,
            sub: true,
            ks: 4,
            cs: 1,
            pf: false,
            g2: false,
        },
        Config {
            nf: 2,
            sub: false,
            ks: 2,
            cs: 2,
            pf: false,
            g2: false,
        },
        Config {
            nf: 1,
            sub: true,
            ks: 4,
            cs: 1,
            pf: true,
            g2: false,
        },
        Config {
            nf: 2,
            sub: true,
            ks: 2,
            cs: 2,
            pf: true,
            g2: false,
        },
        Config {
            nf: 1,
            sub: true,
            ks: 4,
            cs: 1,
            pf: false,
            g2: true,
        },
        Config {
            nf: 2,
            sub: true,
            ks: 3,
            cs: 2,
            pf: false,
            g2: true,
        },
    ];
    for &(n, k) in &[(256usize, 2048usize), (44, 1024)] {
        let w = Matrix::quantized(&context, n, k, 0xA9_000 + n as u64);
        let rows = Rows::new(&context, n, k, 0xA9_100 + k as u64);
        for cfg in configs {
            let mut by_r = Vec::new();
            for r in [1usize, 2, 5, 8] {
                let pass = context.begin_pass();
                encode_mma8(&mut context, &pass, &w, &rows, r, cfg);
                pass.commit_and_wait();
                by_r.push((r, read_buffer_f16(&rows.y, 0, n)));
            }
            for (r, row0) in &by_r[1..] {
                for (i, (a, b)) in row0.iter().zip(&by_r[0].1).enumerate() {
                    assert!(
                        a.to_bits() == b.to_bits(),
                        "{n}x{k} {cfg:?}: row 0 output {i} differs between R=1 and R={r}"
                    );
                }
            }
        }
    }
}

/// QWEN38 shapes without the head (~715 MB a copy), as in
/// `gemv_bandwidth_bench.rs`.
const SHAPES: [(&str, usize, usize); 5] = [
    ("gate/up    17408x5120", 17408, 5120),
    ("down       5120x17408", 5120, 17408),
    ("gdn_inproj 16480x5120", 16480, 5120),
    ("packed_q   12288x5120", 12288, 5120),
    ("o_proj      5120x6144", 5120, 6144),
];
/// Copies per shape, rotated so no call finds its weights in the SLC.
const POOL: usize = 4;
/// Weight bytes per timed arm. 512 MiB gave ~1.4 ms windows, short enough
/// for DVFS to matter; 2 GiB agreed with it to within the spread.
const TARGET_BYTES: usize = 2 * 1024 * 1024 * 1024;
const ROUNDS: usize = 5;

/// The ratio is the deliverable: an 8-row MMA call against the 1-row GEMV
/// (what a decode step pays) and against the shipped 8-row GEMM (what a
/// verify pays today). Configs are swept once per shape after a warmup; the
/// best is then timed against both baselines over interleaved rounds.
#[test]
#[ignore = "benchmark: needs a real Metal device, reports rather than asserts"]
fn mma8_ratio_against_the_gemv_and_the_shipped_gemm() {
    let mut context = MetalContext::new().expect("Metal device");
    let mut configs = Vec::new();
    for sub in [true, false] {
        for nf in [1, 2, 4] {
            for ks in [1, 2, 4, 8] {
                for cs in [1, 2, 4] {
                    if ks * cs <= 16 {
                        configs.push(Config {
                            nf,
                            sub,
                            ks,
                            cs,
                            pf: false,
                            g2: false,
                        });
                        if sub && nf <= 2 {
                            configs.push(Config {
                                nf,
                                sub,
                                ks,
                                cs,
                                pf: true,
                                g2: false,
                            });
                        }
                    }
                }
            }
        }
    }

    println!(
        "\nshape                   best config                          gemv1 ms  gemm8 ms  mma8 ms  \
         gemm8/gemv1  mma8/gemv1  mma8/gemm8  mma1/gemv1"
    );
    for (label, n, k) in SHAPES {
        let pool: Vec<Matrix> = (0..POOL)
            .map(|i| Matrix::random(&context, n, k, 0xB8_000 + i as u64))
            .collect();
        let rows = Rows::new(&context, n, k, 0xB8_100);
        let bytes = n * k / 2 + n * k / 64 * 4;
        let calls = (TARGET_BYTES / bytes).max(POOL);

        let gemv = |context: &mut MetalContext| -> f64 {
            autorelease_pool(|| {
                let pass = context.begin_pass();
                for i in 0..calls {
                    encode_dequant_int4_gemv_resident(
                        context,
                        &pass,
                        &pool[i % POOL].resident(),
                        (&rows.x, 0),
                        (&rows.y, 0),
                    )
                    .unwrap();
                }
                pass.commit_and_wait_with_gpu_time() * 1e3 / calls as f64
            })
        };
        let gemm = |context: &mut MetalContext| -> f64 {
            autorelease_pool(|| {
                let pass = context.begin_pass();
                for i in 0..calls {
                    encode_dequant_int4_gemm_resident(
                        context,
                        &pass,
                        &pool[i % POOL].resident(),
                        (&rows.x, 0),
                        (&rows.y, 0),
                        8,
                    )
                    .unwrap();
                }
                pass.commit_and_wait_with_gpu_time() * 1e3 / calls as f64
            })
        };
        let mma_r = |context: &mut MetalContext, cfg: Config, r: usize| -> f64 {
            autorelease_pool(|| {
                let pass = context.begin_pass();
                for i in 0..calls {
                    encode_mma8(context, &pass, &pool[i % POOL], &rows, r, cfg);
                }
                pass.commit_and_wait_with_gpu_time() * 1e3 / calls as f64
            })
        };
        let mma = |context: &mut MetalContext, cfg: Config| mma_r(context, cfg, 8);
        let gemm_only = |context: &mut MetalContext, cfg: Config| -> f64 {
            autorelease_pool(|| {
                let pass = context.begin_pass();
                encode_x8_table(context, &pass, &rows, k, 8);
                for i in 0..calls {
                    encode_mma8_gemm(context, &pass, &pool[i % POOL], &rows, 8, cfg);
                }
                pass.commit_and_wait_with_gpu_time() * 1e3 / calls as f64
            })
        };
        let prep_only = |context: &mut MetalContext| -> f64 {
            autorelease_pool(|| {
                let pass = context.begin_pass();
                for _ in 0..calls {
                    encode_x8_table(context, &pass, &rows, k, 8);
                }
                pass.commit_and_wait_with_gpu_time() * 1e3 / calls as f64
            })
        };

        // Warmup + fault-in (cold DVFS clocks would land on the first arm).
        gemv(&mut context);
        gemm(&mut context);
        mma(&mut context, configs[0]);

        let best = configs
            .iter()
            .map(|&c| (mma(&mut context, c), c))
            .fold(
                (f64::INFINITY, configs[0]),
                |a, b| if b.0 < a.0 { b } else { a },
            )
            .1;

        let (mut a, mut b, mut c, mut d) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let (mut e, mut f) = (Vec::new(), Vec::new());
        for round in 0..ROUNDS {
            // Rotate arm order each round to spread thermal drift.
            for arm in 0..6 {
                match (arm + round) % 6 {
                    4 => e.push(gemm_only(&mut context, best)),
                    5 => f.push(prep_only(&mut context)),
                    0 => a.push(gemv(&mut context)),
                    1 => b.push(gemm(&mut context)),
                    2 => c.push(mma(&mut context, best)),
                    // One live row through the same tile and config: the
                    // price of decode sharing verify's arithmetic.
                    _ => d.push(mma_r(&mut context, best, 1)),
                }
            }
        }
        let median = |v: &mut Vec<f64>| {
            v.sort_by(f64::total_cmp);
            v[v.len() / 2]
        };
        let (g1, g8, m8, m1) = (
            median(&mut a),
            median(&mut b),
            median(&mut c),
            median(&mut d),
        );
        let (go, po) = (median(&mut e), median(&mut f));
        println!(
            "{label}  {:<35}  {g1:8.3}  {g8:8.3}  {m8:7.3}  {:>10.2}x  {:>9.2}x  {:>9.2}x  {:>9.2}x",
            format!("{best:?}"),
            g8 / g1,
            m8 / g1,
            m8 / g8,
            m1 / g1
        );
        println!(
            "{:<23}  gemm alone {go:.4} ms ({:.2}x gemv1), prep alone {po:.4} ms",
            "",
            go / g1
        );
    }
    println!(
        "\nmma8 includes its X8-table prep dispatch per call (production would share one prep\n\
         across the projections that read the same input). Pass bar: mma8/gemv1 <= ~1.3x.\n\
         mma1/gemv1 is one live row through the same tile: what decode pays to share verify's\n\
         arithmetic.\n"
    );
}

/// How the K split should be chosen. `ks` changes summation order, so the
/// production rule must be a function of the shape alone (never of R).
/// Prints ms per `ks` for one config family at shapes the decode path has,
/// including the small ones (k/v at 1024 rows, in_proj_a/b at 48).
#[test]
#[ignore = "benchmark: needs a real Metal device, reports rather than asserts"]
fn mma8_k_split_sweep() {
    let mut context = MetalContext::new().expect("Metal device");
    let shapes: [(&str, usize, usize); 8] = [
        ("gate/up    17408x5120", 17408, 5120),
        ("down       5120x17408", 5120, 17408),
        ("packed_q   12288x5120", 12288, 5120),
        ("o_proj      5120x6144", 5120, 6144),
        ("k/v         1024x5120", 1024, 5120),
        ("in_proj_a     48x5120", 48, 5120),
        ("mtp_fc    5120x10240", 5120, 10240),
        ("reference  32768x5120", 32768, 5120),
    ];
    println!(
        "\nshape                  gemv1 ms   ks=1    ks=2    ks=4    ks=8    ks=16  (mma8/gemv1)"
    );
    for (label, n, k) in shapes {
        let pool: Vec<Matrix> = (0..POOL)
            .map(|i| Matrix::random(&context, n, k, 0xC8_000 + i as u64))
            .collect();
        let rows = Rows::new(&context, n, k, 0xC8_100);
        let bytes = n * k / 2 + n * k / 64 * 4;
        let calls = (TARGET_BYTES / bytes).clamp(POOL, 20_000);
        let gemv = |context: &mut MetalContext| -> f64 {
            autorelease_pool(|| {
                let pass = context.begin_pass();
                for i in 0..calls {
                    encode_dequant_int4_gemv_resident(
                        context,
                        &pass,
                        &pool[i % POOL].resident(),
                        (&rows.x, 0),
                        (&rows.y, 0),
                    )
                    .unwrap();
                }
                pass.commit_and_wait_with_gpu_time() * 1e3 / calls as f64
            })
        };
        let mma = |context: &mut MetalContext, ks: usize| -> f64 {
            let cfg = Config {
                nf: 1,
                sub: true,
                ks,
                cs: 1,
                pf: false,
                g2: false,
            };
            autorelease_pool(|| {
                let pass = context.begin_pass();
                for i in 0..calls {
                    encode_mma8(context, &pass, &pool[i % POOL], &rows, 8, cfg);
                }
                pass.commit_and_wait_with_gpu_time() * 1e3 / calls as f64
            })
        };
        gemv(&mut context);
        mma(&mut context, 4);
        let kss = [1usize, 2, 4, 8, 16];
        let mut best = vec![f64::INFINITY; kss.len()];
        let mut g1 = f64::INFINITY;
        for _ in 0..3 {
            g1 = g1.min(gemv(&mut context));
            for (j, &ks) in kss.iter().enumerate() {
                if ks <= k / 64 {
                    best[j] = best[j].min(mma(&mut context, ks));
                }
            }
        }
        let cells: Vec<String> = best
            .iter()
            .map(|t| {
                if t.is_finite() {
                    format!("{:6.2}x", t / g1)
                } else {
                    "     --".into()
                }
            })
            .collect();
        println!("{label}  {g1:8.4}  {}", cells.join(" "));
    }
}

/// Where the tile's time goes at one row: each ablation removes one cost
/// (`mma8_abl_impl`). Same K split as production (`ks` 4 at 1024+ row
/// blocks, 8 at 64+), cs 1, table prep done once outside the timed loop.
#[test]
#[ignore = "benchmark: needs a real Metal device, reports rather than asserts"]
fn mma8_ablation() {
    let mut context = MetalContext::new().expect("Metal device");
    let shapes: [(&str, usize, usize); 4] = [
        ("gate/up    17408x5120", 17408, 5120),
        ("down       5120x17408", 5120, 17408),
        ("packed_q   12288x5120", 12288, 5120),
        ("o_proj      5120x6144", 5120, 6144),
    ];
    let names = [
        ("base", "int4_mma8_abl0"),
        ("no-mma", "int4_mma8_abl1"),
        ("w-cached", "int4_mma8_abl2"),
        ("t-cached", "int4_mma8_abl3"),
        ("both", "int4_mma8_abl4"),
        ("persistent-acc", "int4_mma8_abl5"),
        ("uint2-load", "int4_mma8_abl6"),
    ];
    println!("\nshape                  gemv1 ms   base  no-mma  w-cached  t-cached  both  pers-acc  uint2  (x gemv1, R=1)");
    for (label, n, k) in shapes {
        let pool: Vec<Matrix> = (0..POOL)
            .map(|i| Matrix::random_aligned(&context, n, k, 0xE8_000 + i as u64))
            .collect();
        let rows = Rows::new(&context, n, k, 0xE8_100);
        let bytes = n * k / 2 + n * k / 64 * 4;
        let calls = (TARGET_BYTES / bytes).max(POOL);
        let ks = if n.div_ceil(8) >= 1024 { 4 } else { 8 };
        let gemv = |context: &mut MetalContext| -> f64 {
            autorelease_pool(|| {
                let pass = context.begin_pass();
                for i in 0..calls {
                    encode_dequant_int4_gemv_resident(
                        context,
                        &pass,
                        &pool[i % POOL].resident(),
                        (&rows.x, 0),
                        (&rows.y, 0),
                    )
                    .unwrap();
                }
                pass.commit_and_wait_with_gpu_time() * 1e3 / calls as f64
            })
        };
        let arm = |context: &mut MetalContext, name: &'static str| -> f64 {
            autorelease_pool(|| {
                let pass = context.begin_pass();
                encode_x8_table(context, &pass, &rows, k, 1);
                let pipe = context
                    .pipeline(SOURCE, name, &FunctionConstantValues::new(), b"")
                    .unwrap();
                let params: Vec<u8> = [n, k, 1, ks, 1]
                    .iter()
                    .flat_map(|v| (*v as u32).to_le_bytes())
                    .collect();
                for i in 0..calls {
                    let w = &pool[i % POOL];
                    pass.encode_threadgroups(
                        &pipe,
                        &[
                            (&w.buffer, 0, w.weights_offset),
                            (&w.buffer, 1, w.scales_offset),
                            (&w.buffer, 2, w.biases_offset),
                            (&rows.table, 3, 0),
                            (&rows.sums, 4, 0),
                            (&rows.y, 5, 0),
                        ],
                        &[(&params[..], 6)],
                        n.div_ceil(8) as u64,
                        (ks * 32) as u64,
                    );
                }
                pass.commit_and_wait_with_gpu_time() * 1e3 / calls as f64
            })
        };
        gemv(&mut context);
        for (_, name) in names {
            arm(&mut context, name);
        }
        if label.starts_with("gate") {
            for (tag, name) in names {
                let pipe = context
                    .pipeline(SOURCE, name, &FunctionConstantValues::new(), b"")
                    .unwrap();
                println!(
                    "  {tag:<15} max threads/threadgroup {}",
                    pipe.max_total_threads_per_threadgroup()
                );
            }
        }
        let mut g1 = Vec::new();
        let mut t: Vec<Vec<f64>> = vec![Vec::new(); names.len()];
        for round in 0..ROUNDS {
            g1.push(gemv(&mut context));
            for j in 0..names.len() {
                let j = (j + round) % names.len();
                t[j].push(arm(&mut context, names[j].1));
            }
        }
        let med = |v: &mut Vec<f64>| {
            v.sort_by(f64::total_cmp);
            v[v.len() / 2]
        };
        let g = med(&mut g1);
        let cells: Vec<String> = t
            .iter_mut()
            .map(|v| format!("{:6.2}x", med(v) / g))
            .collect();
        println!("{label}  {g:8.4}  {}", cells.join("  "));
    }
}
