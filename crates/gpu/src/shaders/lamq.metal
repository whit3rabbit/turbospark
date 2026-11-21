// ============================================================================
// LookAtMyQuant (LAMQ) kernels: the randomized Kronecker Hadamard transform,
// trellis and level GEMVs over packed rows, one-row decode (embedding), and
// the bucket-order scatter (head). Spec: docs/LAMQ.md. CPU contract:
// turbospark_compute::lamq, held by tests/lamq_parity.rs.
//
// PORT-LOCAL. Transcribed from the upstream reference decoders
// (runtime_mlx/inkling/kernels.py, MIT); accumulation order here is this
// port's own, so parity is to tolerance, not bit-exact.
//
// Easy to get fluently wrong (each is a mutation the parity test kills):
// - Column j's 12-bit state is read at bit (n - 1 - c) * k with c the LAST
//   column of its group; words past the row end read as zero.
// - H = kron(A, B), A Sylvester (applied as an FWHT over the outer index),
//   B the small dense factor. In: A X B^T, signs first. Out: A^T Y B (A is
//   symmetric), signs last. Both scaled by 1/sqrt(n).
// ============================================================================

#include <metal_stdlib>
using namespace metal;

// Widest vector the transform holds in threadgroup memory (the Inkling q and
// gate outputs). The host refuses anything larger.
constant constexpr uint kLamqMaxN = 3072;

struct LamqRhtParams {
    uint n;
    uint na;   // Sylvester factor, a power of two
    uint nb;   // dense factor
    uint dir;  // 0 = input rotation, 1 = output rotation
};

// in/out element types, as function constants so one source serves the
// half activations and the f32 intermediates. Indices 120-126 are used by
// no other shader in this crate; the expert pair compiles this file together
// with moe.metal (0-5, 40-43).
constant bool kRhtInHalf  [[function_constant(120)]];
constant bool kRhtOutHalf [[function_constant(121)]];

// One threadgroup per vector (vector v at v * n in both buffers).
kernel void lamq_rht(
    device const void*      x_raw  [[buffer(0)]],
    device const char*      signs  [[buffer(1)]],
    device const float*     B      [[buffer(2)]],
    device void*            y_raw  [[buffer(3)]],
    constant LamqRhtParams& p      [[buffer(4)]],
    uint                    v      [[threadgroup_position_in_grid]],
    uint                    tid    [[thread_position_in_threadgroup]],
    uint                    nt     [[threads_per_threadgroup]]
) {
    threadgroup float a[kLamqMaxN];
    threadgroup float t[kLamqMaxN];
    const uint n = p.n, na = p.na, nb = p.nb;
    const ulong base = ulong(v) * n;
    for (uint i = tid; i < n; i += nt) {
        float xi = kRhtInHalf ? float(((device const half*)x_raw)[base + i])
                              : ((device const float*)x_raw)[base + i];
        // Input rotation applies the signs before the transform.
        a[i] = p.dir == 0u ? xi * float(signs[i]) : xi;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Dense factor on the inner index: in X B^T, out Y B.
    for (uint idx = tid; idx < n; idx += nt) {
        const uint pp = idx / nb, j = idx % nb;
        float s = 0.0f;
        if (p.dir == 0u) {
            for (uint q = 0; q < nb; ++q) s += a[pp * nb + q] * B[j * nb + q];
        } else {
            for (uint q = 0; q < nb; ++q) s += a[pp * nb + q] * B[q * nb + j];
        }
        t[idx] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Sylvester factor on the outer index, as an in-place FWHT.
    for (uint h = 1; h < na; h <<= 1) {
        const uint pairs = (na / 2u) * nb;
        for (uint w = tid; w < pairs; w += nt) {
            const uint j = w % nb, k = w / nb;
            const uint lo = (k / h) * (2u * h) + (k % h);
            const uint i0 = lo * nb + j, i1 = (lo + h) * nb + j;
            const float u = t[i0], z = t[i1];
            t[i0] = u + z;
            t[i1] = u - z;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float inv = rsqrt(float(n));
    for (uint i = tid; i < n; i += nt) {
        float r = t[i] * inv;
        if (p.dir == 1u) r *= float(signs[i]);
        if (kRhtOutHalf) ((device half*)y_raw)[base + i] = half(r);
        else ((device float*)y_raw)[base + i] = r;
    }
}

// ----------------------------------------------------------------------------
// GEMV: y[r] = scale[r] * sum_j W~[r, j] * x[j], x already rotated (f32).
// One simdgroup per row, eight rows per threadgroup; the table is staged in
// threadgroup memory before the row bound is checked, so every thread meets
// the barrier.
// ----------------------------------------------------------------------------

struct LamqGemvParams {
    uint rows;
    uint cols;
    uint row_bytes;
    uint table_len; // table values (rows of the table times g)
};

constant uint kK      [[function_constant(122)]]; // bits per weight
constant uint kG      [[function_constant(123)]]; // weights per lookup
constant uint kShift  [[function_constant(124)]]; // hash shift, 0 = direct index
constant bool kOutHalf [[function_constant(125)]];
constant uint kLm     [[function_constant(126)]]; // 0 = trellis, 6 or 8 = levels

constant constexpr uint kLamqMaxTable = 4096;
constant constexpr uint kLamqRowsPerTG = 8;

static inline uint lamq_window(device const uint* w, uint words, uint bp) {
    const uint wi = bp >> 5, off = bp & 31u;
    const uint lo = w[wi] >> off;
    const uint hi = (off == 0u || wi + 1u >= words) ? 0u : (w[wi + 1u] << (32u - off));
    return (lo | hi) & 4095u;
}

static inline uint lamq_level_index(device const uchar* b, uint j) {
    if (kLm == 8u) return uint(b[j]);
    const uint q = j >> 2, t = j & 3u;
    const uint v = uint(b[3u * q]) | (uint(b[3u * q + 1u]) << 8) | (uint(b[3u * q + 2u]) << 16);
    return (v >> (6u * t)) & 63u;
}

kernel void lamq_gemv(
    device const uchar*      codes  [[buffer(0)]],
    device const half*       scales [[buffer(1)]],
    device const void*       table  [[buffer(2)]], // half LUT, or f32 levels when kLm != 0
    device const float*      x      [[buffer(3)]],
    device void*             y_raw  [[buffer(4)]],
    constant LamqGemvParams& p      [[buffer(5)]],
    uint                     tg     [[threadgroup_position_in_grid]],
    uint                     sg     [[simdgroup_index_in_threadgroup]],
    uint                     lane   [[thread_index_in_simdgroup]],
    uint                     tid    [[thread_position_in_threadgroup]],
    uint                     nt     [[threads_per_threadgroup]]
) {
    threadgroup float lut[kLamqMaxTable];
    for (uint i = tid; i < p.table_len; i += nt) {
        lut[i] = kLm != 0u ? ((device const float*)table)[i] : float(((device const half*)table)[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint row = tg * kLamqRowsPerTG + sg;
    if (row >= p.rows) return;
    device const uchar* rb = codes + ulong(row) * p.row_bytes;
    const uint n = p.cols;
    float acc = 0.0f;
    if (kLm != 0u) {
        for (uint j = lane; j < n; j += 32u) acc += lut[lamq_level_index(rb, j)] * x[j];
    } else {
        device const uint* w = (device const uint*)rb;
        const uint words = p.row_bytes / 4u;
        for (uint j = lane * kG; j < n; j += 32u * kG) {
            const uint s = lamq_window(w, words, (n - 1u - (j + kG - 1u)) * kK);
            const uint e = (kShift != 0u ? (s * 0x9E3779B1u) >> kShift : s) * kG;
            for (uint t = 0; t < kG; ++t) acc += lut[e + t] * x[j + t];
        }
    }
    acc = simd_sum(acc);
    if (lane != 0u) return;
    const float r = acc * float(scales[row]);
    if (kOutHalf) ((device half*)y_raw)[row] = half(r);
    else ((device float*)y_raw)[row] = r;
}

// One row decoded and scaled into f32 `out[cols]` (an embedding bucket row,
// before the output rotation). One thread per lookup group.
kernel void lamq_decode_row(
    device const uchar*      codes  [[buffer(0)]],
    device const half*       scales [[buffer(1)]],
    device const void*       table  [[buffer(2)]],
    device float*            out    [[buffer(3)]],
    constant LamqGemvParams& p      [[buffer(4)]],
    constant uint&           row    [[buffer(5)]],
    uint                     gid    [[thread_position_in_grid]]
) {
    const uint n = p.cols;
    device const uchar* rb = codes + ulong(row) * p.row_bytes;
    const float sc = float(scales[row]);
    if (kLm != 0u) {
        if (gid >= n) return;
        out[gid] = ((device const float*)table)[lamq_level_index(rb, gid)] * sc;
        return;
    }
    const uint j = gid * kG;
    if (j >= n) return;
    device const uint* w = (device const uint*)rb;
    const uint s = lamq_window(w, p.row_bytes / 4u, (n - 1u - (j + kG - 1u)) * kK);
    const uint e = (kShift != 0u ? (s * 0x9E3779B1u) >> kShift : s) * kG;
    for (uint t = 0; t < kG; ++t) out[j + t] = float(((device const half*)table)[e + t]) * sc;
}

// out[order[i]] = half(in[i]) for i < count: F32 bucket logits to the
// flow's FP16 logits in vocab order.
kernel void lamq_scatter(
    device const float* src   [[buffer(0)]],
    device const int*   order [[buffer(1)]],
    device half*        dst   [[buffer(2)]],
    constant uint&      count [[buffer(3)]],
    uint                gid   [[thread_position_in_grid]]
) {
    if (gid < count) dst[order[gid]] = half(src[gid]);
}
