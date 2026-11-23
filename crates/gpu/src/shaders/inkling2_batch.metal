// ============================================================================
// inkling2_batch - multi-row (speculative verify) forms of the Inkling 2
// decode kernels. PORT-LOCAL.
//
// Contract: row r of every kernel here computes EXACTLY the arithmetic of the
// single-row kernel it mirrors (`inkling2_ops.metal`, `rmsnorm.metal`,
// `minimax_router.metal`), in the same order, so a verify over M rows writes
// the bits M sequential decode calls write. `tests/inkling2_batch_parity.rs`
// holds that bit for bit. The row index is the threadgroup grid's y (or the
// last) dimension; every buffer carries an explicit per-row stride in
// elements, and the KV rows a conv or clip writes are RING-addressed:
// row r lives at `(base_row + r) % ring`.
//
// COMPILED CONCATENATED AFTER `moe.metal` for `router_select_next_one_simd`.
// ============================================================================

// --- Strided copy ------------------------------------------------------------
kernel void inkling2_copy_rows(
    device const half* src        [[buffer(0)]],
    device half*       dst        [[buffer(1)]],
    constant uint&     n          [[buffer(2)]],
    constant uint&     src_stride [[buffer(3)]],
    constant uint&     dst_stride [[buffer(4)]],
    uint2              gid        [[thread_position_in_grid]]
) {
    if (gid.x >= n) return;
    dst[ulong(gid.y) * dst_stride + gid.x] = src[ulong(gid.y) * src_stride + gid.x];
}

// --- RMS norm per row (rmsnorm_bf16w / rmsnorm_no_scale) ---------------------
//
// `ink_rms_block_inv` is `rms_block_inv` of rmsnorm.metal, copied so this file
// compiles without that file's function constants. One threadgroup per row,
// min(256, D) threads (the host matches the single-row dispatch width).
constant constexpr uint kInkRmsMaxSimdGroups = 8;

static inline float ink_rms_block_inv(
    device const half* x, uint D, float eps, uint lid, uint lsize,
    uint simd_lane_id, uint simd_group_id, uint simdgroups,
    threadgroup float* partial
) {
    float acc = 0.0f;
    for (uint i = lid; i < D; i += lsize) {
        float v = float(x[i]);
        acc = fma(v, v, acc);
    }
    acc = simd_sum(acc);
    if (simd_lane_id == 0) {
        partial[simd_group_id] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        float v = (simd_lane_id < simdgroups) ? partial[simd_lane_id] : 0.0f;
        v = simd_sum(v);
        if (simd_lane_id == 0) {
            float mean_sq = v / float(D);
            partial[0] = rsqrt(mean_sq + eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return partial[0];
}

// `has_w` 1: out = x * inv * w (rmsnorm_bf16w); 0: out = x * inv
// (rmsnorm_no_scale / rmsnorm_no_scale_perhead). Rows are `x + row * x_stride`
// and `out + row * out_stride`, row = (base_row + r) % ring when `ring` > 0
// (the per-head V norm on a ring slot), else r.
[[kernel, max_total_threads_per_threadgroup(256)]]
void inkling2_rms_rows(
    device const half*   x          [[buffer(0)]],
    device const bfloat* weight     [[buffer(1)]],
    device       half*   out        [[buffer(2)]],
    constant     uint&   D          [[buffer(3)]],
    constant     float&  eps        [[buffer(4)]],
    constant     uint&   x_stride   [[buffer(5)]],
    constant     uint&   out_stride [[buffer(6)]],
    constant     uint&   has_w      [[buffer(7)]],
    constant     uint&   base_row   [[buffer(8)]],
    constant     uint&   ring       [[buffer(9)]],
    uint2 tg               [[threadgroup_position_in_grid]],
    uint2 lid2             [[thread_position_in_threadgroup]],
    uint2 lsize2           [[threads_per_threadgroup]],
    uint  simd_lane_id     [[thread_index_in_simdgroup]],
    uint  simd_group_id    [[simdgroup_index_in_threadgroup]],
    uint  simdgroups       [[simdgroups_per_threadgroup]]
) {
    threadgroup float partial[kInkRmsMaxSimdGroups];
    const uint lid = lid2.x;
    const uint lsize = lsize2.x;
    // tg.x: the vector within the row (a head; 0 for whole-row norms).
    const uint r = tg.y;
    const uint row = ring > 0u ? (base_row + r) % ring : r;
    device const half* xr = x + ulong(row) * x_stride + ulong(tg.x) * D;
    device half* orow = out + ulong(row) * out_stride + ulong(tg.x) * D;
    const float inv = ink_rms_block_inv(xr, D, eps, lid, lsize,
                                        simd_lane_id, simd_group_id, simdgroups, partial);
    if (has_w != 0u) {
        for (uint i = lid; i < D; i += lsize) {
            float xv = float(xr[i]);
            float wv = float(weight[i]);
            orow[i] = half(xv * inv * wv);
        }
    } else {
        for (uint i = lid; i < D; i += lsize) {
            float xv = float(xr[i]);
            orow[i] = half(xv * inv);
        }
    }
}

// --- Depth mix per row ------------------------------------------------------
// `inkling2_depth_mix` with `states + row * states_stride` and
// `out + row * out_stride + q * h`; the queries are shared.
constant constexpr uint kInkBMaxDepthStates = 16;

kernel void inkling2_depth_mix_rows(
    device const half*    states        [[buffer(0)]],
    device const float*   queries       [[buffer(1)]],
    device half*          out           [[buffer(2)]],
    constant uint&        n_states      [[buffer(3)]],
    constant uint&        h             [[buffer(4)]],
    constant uint&        states_stride [[buffer(5)]],
    constant uint&        out_stride    [[buffer(6)]],
    uint2                 tg            [[threadgroup_position_in_grid]],
    uint2                 tid2          [[thread_position_in_threadgroup]],
    uint                  sg_idx        [[simdgroup_index_in_threadgroup]],
    uint                  lane          [[thread_index_in_simdgroup]]
) {
    const uint tid = tid2.x;
    threadgroup float part_ss[kInkBMaxDepthStates][8];
    threadgroup float part_dq[kInkBMaxDepthStates][8];
    threadgroup float w_s[kInkBMaxDepthStates];
    const uint q_idx = tg.x;
    device const half* sts = states + ulong(tg.y) * states_stride;
    device const float* q = queries + ulong(q_idx) * h;
    for (uint s = 0; s < n_states; ++s) {
        device const half* st = sts + ulong(s) * h;
        float ss = 0.0f;
        float dq = 0.0f;
        for (uint i = tid; i < h; i += 256u) {
            const float v = float(st[i]);
            ss = fma(v, v, ss);
            dq = fma(v, float(q[i]), dq);
        }
        ss = simd_sum(ss);
        dq = simd_sum(dq);
        if (lane == 0u) {
            part_ss[s][sg_idx] = ss;
            part_dq[s][sg_idx] = dq;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float sc[kInkBMaxDepthStates];
        float m = -INFINITY;
        const float inv_sqrt_h = rsqrt(float(h));
        for (uint s = 0; s < n_states; ++s) {
            float ss = 0.0f;
            float dq = 0.0f;
            for (uint g = 0; g < 8u; ++g) {
                ss += part_ss[s][g];
                dq += part_dq[s][g];
            }
            const float rinv = rsqrt(ss / float(h) + 1e-6f);
            part_ss[s][0] = rinv;
            sc[s] = rinv * dq * inv_sqrt_h;
            m = max(m, sc[s]);
        }
        float z = 0.0f;
        for (uint s = 0; s < n_states; ++s) {
            sc[s] = precise::exp(sc[s] - m);
            z += sc[s];
        }
        for (uint s = 0; s < n_states; ++s) w_s[s] = sc[s] / z * part_ss[s][0];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device half* o = out + ulong(tg.y) * out_stride + ulong(q_idx) * h;
    for (uint i = tid; i < h; i += 256u) {
        float acc = 0.0f;
        for (uint s = 0; s < n_states; ++s) {
            acc = fma(w_s[s], float(sts[ulong(s) * h + i]), acc);
        }
        o[i] = half(acc);
    }
}

// --- Loop controller combine per row ---------------------------------------
kernel void inkling2_control_combine_rows(
    device const half* cur      [[buffer(0)]],
    device const half* msg      [[buffer(1)]],
    device const half* gain     [[buffer(2)]],
    device const half* forget   [[buffer(3)]],
    device half*       out      [[buffer(4)]],
    constant uint&     n        [[buffer(5)]],
    constant uint4&    strides  [[buffer(6)]],   // cur, msg, gain/forget, out
    uint2              gid      [[thread_position_in_grid]]
) {
    const uint i = gid.x;
    if (i >= n) return;
    const ulong r = gid.y;
    const float c = float(cur[r * strides.x + i]);
    const float f = precise::tanh(float(forget[r * strides.z + i]));
    const float g = precise::tanh(float(gain[r * strides.z + i]));
    out[r * strides.w + i] = half(c - f * c + g * float(msg[r * strides.y + i]));
}

// --- Multi-row, multi-segment BF16 GEMV -------------------------------------
//
// `inkling2_bf16_gemv_multi` for up to 8 input rows: each simdgroup reads its
// weight row ONCE and dots it with every row's input, keeping one accumulator
// per row in the single-row kernel's lane order, so row m's output is the bits
// the single-row kernel writes for input m. Inputs are `x_i + m * x_stride[i]`,
// outputs `y_s + m * y_stride[s]`.
struct Inkling2GemvRowsTable {
    ulong w_off[8];
    uint rows[8];
    uint cols[8];
    uint x_idx[8];
    uint tg_start[9];
    uint n_seg;
    uint n_rows;
    uint x_stride[3];
    uint y_stride[8];
};

constant constexpr uint kInkBGemvRowsPerTG = 8;
constant constexpr uint kInkBMaxRows = 8;

kernel void inkling2_bf16_gemv_rows(
    device const uchar*              W   [[buffer(0)]],
    constant Inkling2GemvRowsTable&  t   [[buffer(1)]],
    device const half*               x0  [[buffer(2)]],
    device const half*               x1  [[buffer(3)]],
    device const half*               x2  [[buffer(4)]],
    device half*                     y0  [[buffer(5)]],
    device half*                     y1  [[buffer(6)]],
    device half*                     y2  [[buffer(7)]],
    device half*                     y3  [[buffer(8)]],
    device half*                     y4  [[buffer(9)]],
    device half*                     y5  [[buffer(10)]],
    device half*                     y6  [[buffer(11)]],
    device half*                     y7  [[buffer(12)]],
    uint                             tg   [[threadgroup_position_in_grid]],
    uint                             sg   [[simdgroup_index_in_threadgroup]],
    uint                             lane [[thread_index_in_simdgroup]]
) {
    uint s = 0;
    while (s + 1u < t.n_seg && tg >= t.tg_start[s + 1u]) ++s;
    const uint row = (tg - t.tg_start[s]) * kInkBGemvRowsPerTG + sg;
    if (row >= t.rows[s]) return;
    const uint xi = t.x_idx[s];
    device const half* x = xi == 0u ? x0 : (xi == 1u ? x1 : x2);
    const uint xs = t.x_stride[xi];
    const uint n4 = t.cols[s] / 4u;
    const uint nr = t.n_rows;
    device const bfloat4* w = (device const bfloat4*)(W + t.w_off[s]) + ulong(row) * n4;
    float acc[kInkBMaxRows] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (uint i = lane; i < n4; i += 32u) {
        const float4 wv = float4(w[i]);
        for (uint m = 0; m < kInkBMaxRows; ++m) {
            if (m < nr) {
                device const half4* xv = (device const half4*)(x + ulong(m) * xs);
                acc[m] += dot(wv, float4(xv[i]));
            }
        }
    }
    device half* y;
    switch (s) {
        case 0: y = y0; break;
        case 1: y = y1; break;
        case 2: y = y2; break;
        case 3: y = y3; break;
        case 4: y = y4; break;
        case 5: y = y5; break;
        case 6: y = y6; break;
        default: y = y7; break;
    }
    const uint ys = t.y_stride[s];
    for (uint m = 0; m < kInkBMaxRows; ++m) {
        if (m < nr) {
            const float v = simd_sum(acc[m]);
            if (lane == 0u) y[ulong(m) * ys + row] = half(v);
        }
    }
}

// The same kernel for up to 32 input rows (chunked prefill panels). A
// separate kernel rather than a wider `kInkBMaxRows`, so the 8-row form that
// speculative verify runs keeps its exact code. Row m's accumulator sees the
// same terms in the same lane order, so its bits are the single-row kernel's
// (`tests/inkling2_batch_parity.rs`). Tried and slower (2026-11-23): four
// weight rows per simdgroup sharing each input load, 117 against 98 ms per
// 32-row panel in the dispatch profile; the extra registers cost more than
// the input re-reads, which hit cache.
constant constexpr uint kInkBMaxRows32 = 32;

kernel void inkling2_bf16_gemv_rows32(
    device const uchar*              W   [[buffer(0)]],
    constant Inkling2GemvRowsTable&  t   [[buffer(1)]],
    device const half*               x0  [[buffer(2)]],
    device const half*               x1  [[buffer(3)]],
    device const half*               x2  [[buffer(4)]],
    device half*                     y0  [[buffer(5)]],
    device half*                     y1  [[buffer(6)]],
    device half*                     y2  [[buffer(7)]],
    device half*                     y3  [[buffer(8)]],
    device half*                     y4  [[buffer(9)]],
    device half*                     y5  [[buffer(10)]],
    device half*                     y6  [[buffer(11)]],
    device half*                     y7  [[buffer(12)]],
    uint                             tg   [[threadgroup_position_in_grid]],
    uint                             sg   [[simdgroup_index_in_threadgroup]],
    uint                             lane [[thread_index_in_simdgroup]]
) {
    uint s = 0;
    while (s + 1u < t.n_seg && tg >= t.tg_start[s + 1u]) ++s;
    const uint row = (tg - t.tg_start[s]) * kInkBGemvRowsPerTG + sg;
    if (row >= t.rows[s]) return;
    const uint xi = t.x_idx[s];
    device const half* x = xi == 0u ? x0 : (xi == 1u ? x1 : x2);
    const uint xs = t.x_stride[xi];
    const uint n4 = t.cols[s] / 4u;
    const uint nr = t.n_rows;
    device const bfloat4* w = (device const bfloat4*)(W + t.w_off[s]) + ulong(row) * n4;
    float acc[kInkBMaxRows32];
    for (uint m = 0; m < kInkBMaxRows32; ++m) acc[m] = 0.0f;
    for (uint i = lane; i < n4; i += 32u) {
        const float4 wv = float4(w[i]);
        for (uint m = 0; m < kInkBMaxRows32; ++m) {
            if (m < nr) {
                device const half4* xv = (device const half4*)(x + ulong(m) * xs);
                acc[m] += dot(wv, float4(xv[i]));
            }
        }
    }
    device half* y;
    switch (s) {
        case 0: y = y0; break;
        case 1: y = y1; break;
        case 2: y = y2; break;
        case 3: y = y3; break;
        case 4: y = y4; break;
        case 5: y = y5; break;
        case 6: y = y6; break;
        default: y = y7; break;
    }
    const uint ys = t.y_stride[s];
    for (uint m = 0; m < kInkBMaxRows32; ++m) {
        if (m < nr) {
            const float v = simd_sum(acc[m]);
            if (lane == 0u) y[ulong(m) * ys + row] = half(v);
        }
    }
}

// --- Matrix-unit BF16 GEMM for prefill panels (opt-in, KL-bounded) ---------
//
// The panel GEMV's job (`inkling2_bf16_gemv_rows32`) on simdgroup_matrix with
// register operands (the lane layout and register-operand technique of
// `dequant_int4_mma8.metal`, docs/INT4_SMALL_M_MMA.md). FP32 fragments hold
// the BF16 weights and FP16 inputs exactly; accumulation is FP32.
//
// EXACTNESS CLASS: not bit-identical to the GEMV (the matrix unit sums each
// 8-term k block in its own order), so prefill through it is KL-bounded
// against sequential decode, and speculative verify never runs it. But a
// token's output depends only on its own input row: C[o][t] reads weight row
// o and input row t only, K runs in one simdgroup in fixed 8-step order with
// no split, and token columns past n_rows are zero. So a row's bits do not
// depend on how many rows share the panel or how a prompt is chunked
// (`tests/inkling2_mma_gemm.rs`).
//
// One simdgroup = kInkMmaNF fragments of 8 output rows x up to 4 token tiles
// of 8; 8 simdgroups per threadgroup, so 64 * kInkMmaNF output rows per
// threadgroup (the host's `tg_start`, `INKLING2_MMA_ROWS_PER_TG`). Each input
// fragment read from threadgroup memory feeds all kInkMmaNF weight fragments.
// K runs in blocks of 64: the threadgroup stages every input row's block in
// threadgroup memory once (4-wide loads); each lane reads its 16 weights of
// the block per fragment with four 8-byte loads and its inputs with 16-byte
// threadgroup loads. That needs k permuted inside the block, the SAME way for
// every output and input row: k-step s, fragment k-slot kk is physical column
// 8 kk + s. The
// permutation and the block order are fixed, so row independence holds. K
// must be a multiple of 64 (host).
// Measured best at 1 fragment and 4 simdgroups (docs/INKLING2_PHASE0.md,
// chunked prefill step 3): more fragments or fewer simdgroups cost
// occupancy, which these shapes are short of.
#ifndef INK_MMA_NF
#define INK_MMA_NF 1
#endif
constant constexpr uint kInkMmaNF = INK_MMA_NF;
#ifndef INK_MMA_SG
#define INK_MMA_SG 4
#endif
// Simdgroups per threadgroup.
constant constexpr uint kInkMmaSG = INK_MMA_SG;
constant constexpr uint kInkMmaRowsPerTG = 8 * kInkMmaSG * kInkMmaNF;
constant constexpr uint kInkMmaKB = 64;
constant constexpr uint kInkMmaXPad = 4;  // threadgroup row stride 68 halfs

inline uint2 ink_mma_coord(uint lane) {
    const uint q = lane >> 2;
    return uint2((q & 4u) | ((lane >> 1) & 3u), ((q & 2u) << 1) | ((lane & 1u) << 1));
}

kernel void inkling2_bf16_gemm_rows_mma(
    device const uchar*              W   [[buffer(0)]],
    constant Inkling2GemvRowsTable&  t   [[buffer(1)]],
    device const half*               x0  [[buffer(2)]],
    device const half*               x1  [[buffer(3)]],
    device const half*               x2  [[buffer(4)]],
    device half*                     y0  [[buffer(5)]],
    device half*                     y1  [[buffer(6)]],
    device half*                     y2  [[buffer(7)]],
    device half*                     y3  [[buffer(8)]],
    device half*                     y4  [[buffer(9)]],
    device half*                     y5  [[buffer(10)]],
    device half*                     y6  [[buffer(11)]],
    device half*                     y7  [[buffer(12)]],
    uint                             tg   [[threadgroup_position_in_grid]],
    uint                             tid  [[thread_index_in_threadgroup]],
    uint                             sg   [[simdgroup_index_in_threadgroup]],
    uint                             lane [[thread_index_in_simdgroup]]
) {
    threadgroup half xt[32 * (kInkMmaKB + kInkMmaXPad)];
    uint s = 0;
    while (s + 1u < t.n_seg && tg >= t.tg_start[s + 1u]) ++s;
    const uint nrows = t.rows[s];
    const uint o0 = (tg - t.tg_start[s]) * kInkMmaRowsPerTG + sg * 8u * kInkMmaNF;
    // Every simdgroup stays for the staging barriers; one with no rows of
    // its own just skips the multiplies and the write.
    const bool sg_live = o0 < nrows;
    const uint2 c = ink_mma_coord(lane);
    const uint fm = c.x, fn = c.y;
    const uint xi = t.x_idx[s];
    device const half* x = xi == 0u ? x0 : (xi == 1u ? x1 : x2);
    const uint xs = t.x_stride[xi];
    const uint K = t.cols[s];
    const uint nr = t.n_rows;
    const uint ntile = (nr + 7u) / 8u;
    const uint XR = kInkMmaKB + kInkMmaXPad;
    device const bfloat* wbase = (device const bfloat*)(W + t.w_off[s]);
    bool live[kInkMmaNF];
    device const bfloat* wrow[kInkMmaNF];
    for (uint f = 0; f < kInkMmaNF; ++f) {
        const uint o = o0 + 8u * f + fm;
        live[f] = sg_live && o < nrows;
        wrow[f] = wbase + ulong(live[f] ? o : 0u) * K;
    }
    // A lane's weights of a block: physical columns 8 fn .. 8 fn + 15 (fn
    // even), k-slot fn then fn + 1 at step st being 8 fn + st, 8 fn + 8 + st.
    const uint p = fn >> 1;
    float2 acc[kInkMmaNF][4];
    for (uint f = 0; f < kInkMmaNF; ++f)
        for (uint tt = 0; tt < 4u; ++tt) acc[f][tt] = float2(0.0f);
    for (uint kb = 0; kb < K; kb += kInkMmaKB) {
        // Stage the block: 32 rows x 64 halfs in 4-wide loads, zero past
        // n_rows. 512 half4 slots over 256 threads.
        for (uint e = tid; e < 32u * (kInkMmaKB / 4u); e += 32u * kInkMmaSG) {
            const uint r = e / (kInkMmaKB / 4u), k4 = e % (kInkMmaKB / 4u);
            const half4 v = r < nr
                ? *(device const half4*)(x + ulong(r) * xs + kb + 4u * k4)
                : half4(0.0h);
            *(threadgroup half4*)(xt + r * XR + 4u * k4) = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg_live) {
            float wv[kInkMmaNF][16];
            for (uint f = 0; f < kInkMmaNF; ++f) {
                if (live[f]) {
                    device const bfloat4* w4 = (device const bfloat4*)(wrow[f] + kb + 16u * p);
                    for (uint q = 0; q < 4u; ++q) {
                        const float4 v = float4(w4[q]);
                        wv[f][4u * q] = v.x;
                        wv[f][4u * q + 1u] = v.y;
                        wv[f][4u * q + 2u] = v.z;
                        wv[f][4u * q + 3u] = v.w;
                    }
                } else {
                    for (uint q = 0; q < 16u; ++q) wv[f][q] = 0.0f;
                }
            }
            // B's k-slot fm at step st is physical column 8 fm + st: this
            // lane's inputs of the block are 8 contiguous halfs per row.
            for (uint tt = 0; tt < 4u; ++tt) {
                if (tt < ntile) {
                    const uint ta = tt * 8u + fn;
                    const half4 xa0 = *(threadgroup const half4*)(xt + ta * XR + 8u * fm);
                    const half4 xa1 = *(threadgroup const half4*)(xt + ta * XR + 8u * fm + 4u);
                    const half4 xb0 = *(threadgroup const half4*)(xt + (ta + 1u) * XR + 8u * fm);
                    const half4 xb1 = *(threadgroup const half4*)(xt + (ta + 1u) * XR + 8u * fm + 4u);
                    const float xa[8] = {float(xa0.x), float(xa0.y), float(xa0.z), float(xa0.w),
                                         float(xa1.x), float(xa1.y), float(xa1.z), float(xa1.w)};
                    const float xb[8] = {float(xb0.x), float(xb0.y), float(xb0.z), float(xb0.w),
                                         float(xb1.x), float(xb1.y), float(xb1.z), float(xb1.w)};
                    for (uint st = 0; st < 8u; ++st) {
                        simdgroup_float8x8 B;
                        B.thread_elements()[0] = xa[st];
                        B.thread_elements()[1] = xb[st];
                        for (uint f = 0; f < kInkMmaNF; ++f) {
                            simdgroup_float8x8 A, C;
                            A.thread_elements()[0] = wv[f][st];
                            A.thread_elements()[1] = wv[f][8u + st];
                            C.thread_elements()[0] = acc[f][tt].x;
                            C.thread_elements()[1] = acc[f][tt].y;
                            simdgroup_multiply_accumulate(C, A, B, C);
                            acc[f][tt] = float2(C.thread_elements()[0], C.thread_elements()[1]);
                        }
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (!sg_live) return;
    device half* y;
    switch (s) {
        case 0: y = y0; break;
        case 1: y = y1; break;
        case 2: y = y2; break;
        case 3: y = y3; break;
        case 4: y = y4; break;
        case 5: y = y5; break;
        case 6: y = y6; break;
        default: y = y7; break;
    }
    const uint ys = t.y_stride[s];
    for (uint f = 0; f < kInkMmaNF; ++f) {
        if (!live[f]) continue;
        const uint o = o0 + 8u * f + fm;
        for (uint tt = 0; tt < ntile; ++tt) {
            const uint ta = tt * 8u + fn, tb = ta + 1u;
            if (ta < nr) y[ulong(ta) * ys + o] = half(acc[f][tt].x);
            if (tb < nr) y[ulong(tb) * ys + o] = half(acc[f][tt].y);
        }
    }
}

// --- Short convolution over consecutive rows --------------------------------
//
// `inkling2_short_conv_decode` applied to rows 0..n_rows in order, the tail
// advancing between rows exactly as M sequential calls advance it. `snap`
// holds n_rows + 1 tails of `(taps - 1) * ch`: entry 0 is the tail at the
// block start, entry r + 1 the tail after row r, so a verify that keeps rows
// 0..k restores entry k (rollback). Output row r
// goes to `out + ((out_base + r) % out_ring) * out_stride` (`out_ring` 0: no
// ring). One thread per channel, as the single-row kernel.
constant constexpr uint kInkMaxConvTaps = 8;

kernel void inkling2_short_conv_rows(
    device const half*    x          [[buffer(0)]],
    device half*          tail       [[buffer(1)]],
    device const bfloat*  weight     [[buffer(2)]],
    device const half*    dyn        [[buffer(3)]],
    device half*          out        [[buffer(4)]],
    device half*          snap       [[buffer(5)]],
    constant uint&        ch         [[buffer(6)]],
    constant uint&        taps       [[buffer(7)]],
    constant uint&        heads      [[buffer(8)]],
    constant uint&        n_rows     [[buffer(9)]],
    constant uint&        x_stride   [[buffer(10)]],
    constant uint&        dyn_stride [[buffer(11)]],
    constant uint&        out_base   [[buffer(12)]],
    constant uint&        out_ring   [[buffer(13)]],
    constant uint&        out_stride [[buffer(14)]],
    uint                  c          [[thread_position_in_grid]]
) {
    if (c >= ch || taps > kInkMaxConvTaps) return;
    half tl[kInkMaxConvTaps];
    const uint tn = (taps - 1u) * ch;
    for (uint j = 0; j + 1u < taps; ++j) {
        tl[j] = tail[j * ch + c];
        snap[j * ch + c] = tl[j];
    }
    for (uint r = 0; r < n_rows; ++r) {
        const float xc = float(x[ulong(r) * x_stride + c]);
        float acc = 0.0f;
        for (uint j = 0; j + 1u < taps; ++j) {
            acc = fma(float(weight[c * taps + j]), float(tl[j]), acc);
        }
        acc = fma(float(weight[c * taps + taps - 1u]), xc, acc);
        float denom = float(taps + 1u);
        if (heads != 0u) {
            const uint head = c / (ch / heads);
            device const half* d = dyn + ulong(r) * dyn_stride + head * taps;
            acc = fma(float(d[0]), xc, acc);
            for (uint lag = 1; lag < taps; ++lag) {
                acc = fma(float(d[lag]), float(tl[taps - 1u - lag]), acc);
            }
            denom = float(2u * taps + 1u);
        }
        const uint orow = out_ring > 0u ? (out_base + r) % out_ring : r;
        out[ulong(orow) * out_stride + c] = half((acc + xc) * rsqrt(denom));
        for (uint j = 0; j + 2u < taps; ++j) tl[j] = tl[j + 1u];
        if (taps > 1u) tl[taps - 2u] = half(xc);
        device half* sn = snap + ulong(r + 1u) * tn;
        for (uint j = 0; j + 1u < taps; ++j) sn[j * ch + c] = tl[j];
    }
    for (uint j = 0; j + 1u < taps; ++j) tail[j * ch + c] = tl[j];
}

// --- L2 clip per row --------------------------------------------------------
// `inkling2_l2_clip` over `n_vec` vectors per row, row r at
// `x + ((base_row + r) % ring) * row_stride` (ring 0: row r), with row r's own
// post scale (`tau` of its position on full layers).
kernel void inkling2_l2_clip_rows(
    device half*          x            [[buffer(0)]],
    device const bfloat*  scale        [[buffer(1)]],
    constant uint&        n_vec        [[buffer(2)]],
    constant uint&        dim          [[buffer(3)]],
    constant uint&        scale_stride [[buffer(4)]],
    constant uint&        has_scale    [[buffer(5)]],
    constant float&       threshold    [[buffer(6)]],
    constant float*       post_scale   [[buffer(7)]],
    constant uint&        base_row     [[buffer(8)]],
    constant uint&        ring         [[buffer(9)]],
    constant uint&        row_stride   [[buffer(10)]],
    uint2                 tg           [[threadgroup_position_in_grid]],
    uint                  sg_idx       [[simdgroup_index_in_threadgroup]],
    uint                  lane         [[thread_index_in_simdgroup]]
) {
    const uint v = tg.x * 8u + sg_idx;
    if (v >= n_vec) return;
    const uint r = tg.y;
    const uint xrow = ring > 0u ? (base_row + r) % ring : r;
    device half* row = x + ulong(xrow) * row_stride + ulong(v) * dim;
    device const bfloat* s = scale + ulong(v) * scale_stride;
    float ss = 0.0f;
    for (uint i = lane; i < dim; i += 32u) {
        const float y = has_scale != 0u ? float(row[i]) * float(s[i]) : float(row[i]);
        ss = fma(y, y, ss);
    }
    const float norm = sqrt(simd_sum(ss));
    const float f = clamp(threshold / (norm + 1e-6f), 0.0f, 1.0f) * post_scale[r];
    for (uint i = lane; i < dim; i += 32u) {
        const float y = has_scale != 0u ? float(row[i]) * float(s[i]) : float(row[i]);
        row[i] = half(y * f);
    }
}

// --- Relative-bias table per row --------------------------------------------
kernel void inkling2_rel_table_rows(
    device const half*    r            [[buffer(0)]],
    device const bfloat*  proj         [[buffer(1)]],
    device float*         table        [[buffer(2)]],
    constant uint&        heads        [[buffer(3)]],
    constant uint&        d_rel        [[buffer(4)]],
    constant uint&        extent       [[buffer(5)]],
    constant uint&        r_stride     [[buffer(6)]],
    uint2                 gid2         [[thread_position_in_grid]]
) {
    const uint gid = gid2.x;
    if (gid >= heads * extent) return;
    device const half* rr = r + ulong(gid2.y) * r_stride;
    const uint h = gid / extent;
    const uint d = gid % extent;
    float acc = 0.0f;
    for (uint i = 0; i < d_rel; ++i) {
        acc = fma(float(rr[h * d_rel + i]), float(proj[i * extent + d]), acc);
    }
    table[ulong(gid2.y) * heads * extent + gid] = acc;
}

// --- Decode attention per row -----------------------------------------------
//
// `inkling2_attention_decode` for the query of row r at `pos0 + r`, over
// cached positions `first(r)..=pos0 + r` (`window` 0: from 0; else the last
// `window` positions). Causal inside the block falls out of the range: every
// row's K/V is written before this runs, and row r never reads past its own
// position. The ring must hold window + n_rows - 1 rows so no row of the block
// overwrites a position an earlier row still reads (the host checks).
kernel void inkling2_attention_rows(
    device const half*   q        [[buffer(0)]],
    device const half*   k        [[buffer(1)]],
    device const half*   v        [[buffer(2)]],
    device const float*  table    [[buffer(3)]],
    device const half*   sinks    [[buffer(4)]],
    device half*         out      [[buffer(5)]],
    constant uint&       window   [[buffer(6)]],
    constant uint&       pos0     [[buffer(7)]],
    constant uint&       kv_heads [[buffer(8)]],
    constant uint&       group    [[buffer(9)]],
    constant uint&       head_dim [[buffer(10)]],
    constant uint&       extent   [[buffer(11)]],
    constant uint&       ring     [[buffer(12)]],
    constant uint&       q_stride [[buffer(13)]],
    constant uint&       s_stride [[buffer(14)]],
    uint2                tg       [[threadgroup_position_in_grid]],
    uint                 sg_idx   [[simdgroup_index_in_threadgroup]],
    uint                 lane     [[thread_index_in_simdgroup]]
) {
    threadgroup float tg_m[8];
    threadgroup float tg_l[8];
    threadgroup float tg_acc[8][128];
    const uint h = tg.x;
    const uint rr = tg.y;
    const uint pos = pos0 + rr;
    const uint first = (window == 0u || pos + 1u <= window) ? 0u : pos + 1u - window;
    q += ulong(rr) * q_stride;
    out += ulong(rr) * q_stride;
    sinks += ulong(rr) * s_stride;
    // The table is [rows, heads, extent].
    table += ulong(rr) * (ulong(s_stride) * extent);
    const uint per = head_dim / 32u;
    const uint kvh = h / group;
    const uint kvd = kv_heads * head_dim;
    const float scale = rsqrt(float(head_dim));
    float qv[4];
    for (uint j = 0; j < per; ++j) qv[j] = float(q[h * head_dim + lane * per + j]);

    float m = -INFINITY;
    float l = 0.0f;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint kp = first + sg_idx; kp <= pos; kp += 8u) {
        const ulong row = ulong(kp % ring) * kvd + kvh * head_dim + lane * per;
        device const half* kr = k + row;
        float s = 0.0f;
        for (uint j = 0; j < per; ++j) s = fma(qv[j], float(kr[j]), s);
        s = simd_sum(s) * scale;
        const uint d = pos - kp;
        if (d < extent) s += table[h * extent + d];
        const float m_new = max(m, s);
        const float corr = precise::exp(m - m_new);
        const float w = precise::exp(s - m_new);
        l = l * corr + w;
        device const half* vr = v + row;
        for (uint j = 0; j < per; ++j) acc[j] = fma(w, float(vr[j]), acc[j] * corr);
        m = m_new;
    }
    if (lane == 0u) {
        tg_m[sg_idx] = m;
        tg_l[sg_idx] = l;
    }
    for (uint j = 0; j < per; ++j) tg_acc[sg_idx][lane * per + j] = acc[j];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg_idx != 0u) return;

    const float t = float(sinks[h]);
    float M = t;
    for (uint g = 0; g < 8u; ++g) M = max(M, tg_m[g]);
    float denom = precise::exp(t - M);
    float f[8];
    for (uint g = 0; g < 8u; ++g) {
        f[g] = (tg_l[g] > 0.0f) ? precise::exp(tg_m[g] - M) : 0.0f;
        denom += tg_l[g] * f[g];
    }
    for (uint j = 0; j < per; ++j) {
        const uint c = lane * per + j;
        float o = 0.0f;
        for (uint g = 0; g < 8u; ++g) o = fma(tg_acc[g][c], f[g], o);
        out[h * head_dim + c] = half(o / denom);
    }
}

// --- F32 router GEMV per row (minimax_router) -------------------------------
kernel void inkling2_router_rows(
    device const float* w        [[buffer(0)]],
    device const half*  x        [[buffer(1)]],
    device float*       out      [[buffer(2)]],
    constant uint&      hidden   [[buffer(3)]],
    constant uint&      x_stride [[buffer(4)]],
    constant uint&      experts  [[buffer(5)]],
    uint2               tg       [[threadgroup_position_in_grid]],
    uint                lane     [[thread_index_in_simdgroup]]
) {
    const uint row = tg.x;
    device const half* xr = x + ulong(tg.y) * x_stride;
    float sum = 0.0f;
    for (uint col = lane; col < hidden; col += 32) {
        sum += w[row * hidden + col] * float(xr[col]);
    }
    sum = simd_sum(sum);
    if (lane == 0) out[ulong(tg.y) * experts + row] = sum;
}

// --- Router selection per row -----------------------------------------------
// `inkling2_router_select_k8` with row r's logits at `logits + r * NE` and its
// selection at `out_* + r * 8`.
kernel void inkling2_router_select_rows(
    device const float* logits          [[buffer(0)]],
    device const float* correction_bias [[buffer(1)]],
    device uint*        out_indices     [[buffer(2)]],
    device half*        out_weights     [[buffer(3)]],
    constant uint&      num_experts     [[buffer(4)]],
    constant float&     route_scale     [[buffer(5)]],
    uint                r               [[threadgroup_position_in_grid]],
    uint                sg_idx          [[simdgroup_index_in_threadgroup]],
    uint                lane            [[thread_index_in_simdgroup]]
) {
    const uint NE = num_experts;
    if (sg_idx != 0u || NE > 32u * kRouterMaxPerLane) return;
    logits += ulong(r) * NE;
    out_indices += ulong(r) * 8u;
    out_weights += ulong(r) * 8u;
    const uint per = (NE + 31u) / 32u;
    const uint base = lane * per;
    float ch[kRouterMaxPerLane];
    for (uint j = 0; j < per; ++j) {
        const uint e = base + j;
        ch[j] = (e < NE) ? (1.0f / (1.0f + precise::exp(-logits[e])) + correction_bias[e])
                         : -INFINITY;
    }
    uint taken = 0u;
    uint top_idx[8];
    for (uint k = 0; k < 8; ++k) {
        top_idx[k] = router_select_next_one_simd(ch, taken, per, base);
    }
    if (lane != 0u) return;
    float scores[8];
    float sum = 0.0f;
    for (uint i = 0; i < 8; ++i) {
        scores[i] = 1.0f / (1.0f + precise::exp(-logits[top_idx[i]]));
        sum += scores[i];
    }
    const float denom = sum + 1e-20f;
    for (uint i = 0; i < 8; ++i) {
        out_indices[i] = top_idx[i];
        out_weights[i] = half(scores[i] / denom * route_scale);
    }
}
