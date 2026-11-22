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
// Largest dense factor (Paley-I of order 72, for 2304 and 1152). Its entries
// are exactly +1/-1, so it is staged as int8: 5 KB beside the two 12 KB
// vectors, inside the 32 KB threadgroup limit. The host refuses larger.
constant constexpr uint kLamqMaxNb = 72;

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
// One vector's rotation, the whole of it in this threadgroup: `a`, `t` and
// `B` are the caller's threadgroup arrays. Inputs and outputs are typed by
// flags so one body serves the half/f32 kernels and the batch below.
static inline void lamq_rht_vec(
    device const void* x_raw, bool x_half, device const char* signs,
    device const char* Bdev, device void* y_raw, bool y_half,
    uint n, uint na, uint nb, uint dir, uint tid, uint nt,
    threadgroup float* a, threadgroup float* t, threadgroup char* B
) {
    // Staged once per vector rather than read from device memory nb times per
    // element. float(+1/-1) is exact, so every product and sum is the same
    // f32 value as with the f32 factor this replaced.
    for (uint i = tid; i < nb * nb; i += nt) B[i] = Bdev[i];
    for (uint i = tid; i < n; i += nt) {
        float xi = x_half ? float(((device const half*)x_raw)[i]) : ((device const float*)x_raw)[i];
        // Input rotation applies the signs before the transform.
        a[i] = dir == 0u ? xi * float(signs[i]) : xi;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Dense factor on the inner index: in X B^T, out Y B.
    for (uint idx = tid; idx < n; idx += nt) {
        const uint pp = idx / nb, j = idx % nb;
        float s = 0.0f;
        if (dir == 0u) {
            for (uint q = 0; q < nb; ++q) s += a[pp * nb + q] * float(B[j * nb + q]);
        } else {
            for (uint q = 0; q < nb; ++q) s += a[pp * nb + q] * float(B[q * nb + j]);
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
        if (dir == 1u) r *= float(signs[i]);
        if (y_half) ((device half*)y_raw)[i] = half(r);
        else ((device float*)y_raw)[i] = r;
    }
}

kernel void lamq_rht(
    device const void*      x_raw  [[buffer(0)]],
    device const char*      signs  [[buffer(1)]],
    device const char*      Bdev   [[buffer(2)]],
    device void*            y_raw  [[buffer(3)]],
    constant LamqRhtParams& p      [[buffer(4)]],
    uint                    v      [[threadgroup_position_in_grid]],
    uint                    tid    [[thread_position_in_threadgroup]],
    uint                    nt     [[threads_per_threadgroup]]
) {
    threadgroup float a[kLamqMaxN];
    threadgroup float t[kLamqMaxN];
    threadgroup char B[kLamqMaxNb * kLamqMaxNb];
    const ulong base = ulong(v) * p.n;
    device const void* xv = kRhtInHalf ? (device const void*)((device const half*)x_raw + base)
                                       : (device const void*)((device const float*)x_raw + base);
    device void* yv = kRhtOutHalf ? (device void*)((device half*)y_raw + base)
                                   : (device void*)((device float*)y_raw + base);
    lamq_rht_vec(xv, kRhtInHalf, signs, Bdev, yv, kRhtOutHalf, p.n, p.na, p.nb, p.dir, tid, nt, a, t, B);
}

// Several input rotations of one width in one dispatch, one threadgroup per
// vector (so they run on as many cores): job v rotates half input x_sel
// (0..2) with the signs at byte s_off of `signs` into f32 y[v * n, ...).
// Each vector's arithmetic is the single-vector kernel's, so the outputs are
// bit-identical to rotating them one at a time.
struct LamqRhtJob {
    uint  x_sel;
    uint  pad;
    ulong s_off;
};

kernel void lamq_rht_in_batch(
    device const half*      x0     [[buffer(0)]],
    device const half*      x1     [[buffer(1)]],
    device const half*      x2     [[buffer(2)]],
    device const char*      signs  [[buffer(3)]],
    device const char*      Bdev   [[buffer(4)]],
    device float*           y      [[buffer(5)]],
    constant LamqRhtParams& p      [[buffer(6)]],
    constant LamqRhtJob*    jobs   [[buffer(7)]],
    uint                    v      [[threadgroup_position_in_grid]],
    uint                    tid    [[thread_position_in_threadgroup]],
    uint                    nt     [[threads_per_threadgroup]]
) {
    threadgroup float a[kLamqMaxN];
    threadgroup float t[kLamqMaxN];
    threadgroup char B[kLamqMaxNb * kLamqMaxNb];
    const LamqRhtJob j = jobs[v];
    device const half* x = j.x_sel == 0u ? x0 : (j.x_sel == 1u ? x1 : x2);
    lamq_rht_vec(x, true, signs + j.s_off, Bdev, y + ulong(v) * p.n, false,
                 p.n, p.na, p.nb, 0u, tid, nt, a, t, B);
}

// ----------------------------------------------------------------------------
// GEMV: y[r] = scale[r] * sum_j W~[r, j] * x[j], x already rotated (f32).
// One simdgroup per row, 32 rows per threadgroup; the table is staged in
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
constant constexpr uint kLamqRowsPerTG = 32; // = ROWS_PER_TG in lamq.rs

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
