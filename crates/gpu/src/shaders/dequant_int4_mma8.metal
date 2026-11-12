// INT4 GEMM for 1..8 token rows on simdgroup_matrix with REGISTER operands,
// read straight from the resident MLX-affine layout (low nibble first,
// groups of 64, BF16 scale and bias, 2-byte aligned).
// docs/INT4_SMALL_M_MMA.md holds the technique, the measurements and the
// exactness contract; crates/gpu/tests/int4_mma8_reg_bench.rs holds the
// prototype, its variants and the gates this file was promoted from.
//
// EXACTNESS CLASS: not bit-identical to dequant_int4_gemv (another
// summation order), but a token row's bits depend only on its own inputs,
// never on how many rows share the tile. That is what lets decode (one row)
// and verify (several) run THIS kernel and stay bit-exact against each
// other. Two things would break it and are therefore baked, not chosen at
// dispatch: the K split (FC_MMA8_KS, a function of the shape only) and the
// code mode (subnormal codes only).
//
// Function constants 40..42 (N, K, KS) are ALWAYS set and ALWAYS in the
// pipeline cache key (dequant_int4_mma8.rs).

#include <metal_stdlib>
using namespace metal;

constant uint FC_MMA8_N  [[function_constant(40)]];
constant uint FC_MMA8_K  [[function_constant(41)]];
constant uint FC_MMA8_KS [[function_constant(42)]];

constant constexpr uint kG = 64;     // quantization group
constant constexpr uint kRows = 8;   // token rows of one tile

// Lane -> (fm, fn): the row and first column of the two elements a lane
// holds in an 8x8 simdgroup_matrix (MLX steel's layout).
inline uint2 mma8_frag_coord(uint lane) {
    const uint q = lane >> 2;
    return uint2((q & 4u) | ((lane >> 1) & 3u), ((q & 2u) << 1) | ((lane & 1u) << 1));
}

// Physical input (0..63 within a group) of k-slot j of fragment f. The lane
// holding slots (j, j+1), j = fn, then reads its 16 codes as one contiguous
// run at 16 * (fn / 2): fragment f takes nibbles (f % 4) and (f % 4) + 4 of
// word f / 4 of that run.
inline uint mma8_phys(uint f, uint j) {
    return 16u * (j >> 1) + 8u * (f >> 2) + (f & 3u) + 4u * (j & 1u);
}

// X8 table: table[(g * 32 + lane) * 16 + 2 f + e] = x[fn + e][64 g + phys(f, fm)],
// sums[g * 8 + t] = sum over group g of float(x[t][k]) in index order.
// Rows t >= R are zero. One threadgroup of 32 threads per group.
kernel void dequant_int4_mma8_table(
    device const half* x     [[buffer(0)]],   // [R][K], token-major
    device half*       table [[buffer(1)]],
    device float*      sums  [[buffer(2)]],
    constant uint&     K     [[buffer(3)]],
    constant uint&     R     [[buffer(4)]],
    uint g    [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]
) {
    const uint2 c = mma8_frag_coord(lane);
    device half* out = table + (g * 32u + lane) * 16u;
    for (uint f = 0; f < 8u; ++f) {
        const uint k = g * kG + mma8_phys(f, c.x);
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
inline void mma8_mac(thread float2& c, half2 a, half2 b) {
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

// One threadgroup = FC_MMA8_KS simdgroups on the same 8 output rows, each
// over one K slice, summed in fixed slice order. Codes are the half
// subnormal q * 2^-24 (bare nibble bits), which the matrix units keep exact;
// the epilogue rescales by s * 2^24.
kernel void dequant_int4_mma8_gemm(
    device const uchar*  W     [[buffer(0)]],
    device const bfloat* S     [[buffer(1)]],
    device const bfloat* Bi    [[buffer(2)]],
    device const half*   table [[buffer(3)]],
    device const float*  sums  [[buffer(4)]],
    device half*         y     [[buffer(5)]],   // [R][N], token-major
    constant uint&       R     [[buffer(6)]],
    uint tg     [[threadgroup_position_in_grid]],
    uint kslice [[simdgroup_index_in_threadgroup]],
    uint lane   [[thread_index_in_simdgroup]]
) {
    threadgroup float2 red[16 * 32];
    const uint N = FC_MMA8_N, K = FC_MMA8_K, KS = FC_MMA8_KS;
    const uint2 c = mma8_frag_coord(lane);
    const uint fm = c.x, fn = c.y;
    const uint n = tg * 8u + fm;
    const uint G = K / kG;
    const uint g0 = kslice * G / KS, g1 = (kslice + 1u) * G / KS;
    // The MMA is simdgroup-collective: a row past N must not skip it, so its
    // lanes multiply zero codes and drop the result.
    const bool live = n < N;
    device const uchar* wrow = W + (live ? n : 0u) * (K / 2u) + (fn >> 1) * 8u;

    float2 acc = float2(0.0f);
    for (uint g = g0; g < g1; ++g) {
        device const half4* tb = (device const half4*)(table + (g * 32u + lane) * 16u);
        const half4 x0 = tb[0], x1 = tb[1], x2 = tb[2], x3 = tb[3];
        const half2 b[8] = {x0.xy, x0.zw, x1.xy, x1.zw, x2.xy, x2.zw, x3.xy, x3.zw};
        const float2 sx = float2(sums[g * kRows + fn], sums[g * kRows + fn + 1u]);
        uint w0 = 0u, w1 = 0u;
        float s = 0.0f, bi = 0.0f;
        if (live) {
            device const ushort* wp = (device const ushort*)(wrow + g * (kG / 2u));
            w0 = uint(wp[0]) | (uint(wp[1]) << 16);
            w1 = uint(wp[2]) | (uint(wp[3]) << 16);
            s = float(S[n * G + g]);
            bi = float(Bi[n * G + g]);
        }
        float2 dot = float2(0.0f);
        for (uint f = 0; f < 8u; ++f) {
            const uint word = f < 4u ? w0 : w1;
            mma8_mac(dot, as_type<half2>((word >> (4u * (f & 3u))) & 0x000F000Fu), b[f]);
        }
        acc = fma(dot, float2(s * 16777216.0f), acc);
        acc = fma(sx, float2(bi), acc);
    }

    if (KS > 1u) {
        red[kslice * 32u + lane] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (kslice != 0u) return;
        acc = float2(0.0f);
        for (uint k = 0; k < KS; ++k) acc += red[k * 32u + lane];
    }
    if (!live) return;
    if (fn < R) y[fn * N + n] = half(acc.x);
    if (fn + 1u < R) y[(fn + 1u) * N + n] = half(acc.y);
}
