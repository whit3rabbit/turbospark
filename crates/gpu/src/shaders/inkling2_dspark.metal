// ============================================================================
// inkling2_dspark - the Inkling 2 DSpark block drafter's own ops. PORT-LOCAL.
//
// Contract: the CPU reference in `crates/runtime/examples/inkling2_dspark_accept.rs`
// (which follows ollama `mlxrunner/model/inkling2/draft.go` and oMLX
// `patches/inkling2/dspark.py`). The drafter only PROPOSES: the target's
// verify decides every committed token, so a rounding difference here moves
// acceptance, never output. The residual stream is F32 with the reference's
// BF16 rounding of the stored residual before each add; projections take FP16
// inputs (the shared GEMV) and accumulate in F32.
// ============================================================================

#include <metal_stdlib>
using namespace metal;

static inline float dsp_round_bf16(float x) {
    return float(bfloat(x));
}

// Block reduce over a 256-thread threadgroup; every thread gets the sum.
static inline float dsp_tg_sum(float v, threadgroup float* part, uint tid, uint sg, uint lane) {
    v = simd_sum(v);
    if (lane == 0u) part[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float s = 0.0f;
    for (uint g = 0; g < 8u; ++g) s += part[g];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return s;
}

// --- Tap mix + RMS norm -----------------------------------------------------
// out[r] = rms(sum_k alpha[k] * taps[r][k], w), F32 math, FP16 out. `taps` is
// FP16 `[rows][n_taps][h]` (row stride `tap_stride` halfs). One threadgroup of
// 256 per row.
kernel void dspark_tap_mix_norm(
    device const half*   taps       [[buffer(0)]],
    device const bfloat* w          [[buffer(1)]],
    device half*         out        [[buffer(2)]],
    constant float4&     alpha      [[buffer(3)]],
    constant uint&       h          [[buffer(4)]],
    constant uint&       tap_stride [[buffer(5)]],
    constant float&      eps        [[buffer(6)]],
    uint                 r          [[threadgroup_position_in_grid]],
    uint                 tid        [[thread_position_in_threadgroup]],
    uint                 sg         [[simdgroup_index_in_threadgroup]],
    uint                 lane       [[thread_index_in_simdgroup]]
) {
    threadgroup float part[8];
    device const half* t = taps + ulong(r) * tap_stride;
    float ss = 0.0f;
    for (uint i = tid; i < h; i += 256u) {
        const float x = alpha.x * float(t[i]) + alpha.y * float(t[h + i]) +
                        alpha.z * float(t[2u * h + i]) + alpha.w * float(t[3u * h + i]);
        ss = fma(x, x, ss);
    }
    const float inv = rsqrt(dsp_tg_sum(ss, part, tid, sg, lane) / float(h) + eps);
    for (uint i = tid; i < h; i += 256u) {
        const float x = alpha.x * float(t[i]) + alpha.y * float(t[h + i]) +
                        alpha.z * float(t[2u * h + i]) + alpha.w * float(t[3u * h + i]);
        out[ulong(r) * h + i] = half(x * inv * float(w[i]));
    }
}

// --- RMS norm, F32 in, FP16 out ---------------------------------------------
kernel void dspark_rms_f32(
    device const float*  x    [[buffer(0)]],
    device const bfloat* w    [[buffer(1)]],
    device half*         out  [[buffer(2)]],
    constant uint&       h    [[buffer(3)]],
    constant float&      eps  [[buffer(4)]],
    uint                 r    [[threadgroup_position_in_grid]],
    uint                 tid  [[thread_position_in_threadgroup]],
    uint                 sg   [[simdgroup_index_in_threadgroup]],
    uint                 lane [[thread_index_in_simdgroup]]
) {
    threadgroup float part[8];
    device const float* xr = x + ulong(r) * h;
    float ss = 0.0f;
    for (uint i = tid; i < h; i += 256u) ss = fma(xr[i], xr[i], ss);
    const float inv = rsqrt(dsp_tg_sum(ss, part, tid, sg, lane) / float(h) + eps);
    for (uint i = tid; i < h; i += 256u) out[ulong(r) * h + i] = half(xr[i] * inv * float(w[i]));
}

// --- Block input: row 0 the anchor's normed embedding, rows 1.. mask_embed --
// `emb` is the anchor's embedding already RMS-normed (FP16); BF16-rounded as
// the reference does.
kernel void dspark_init_x(
    device const half*   emb  [[buffer(0)]],
    device const bfloat* mask [[buffer(1)]],
    device float*        x    [[buffer(2)]],
    constant uint&       h    [[buffer(3)]],
    uint2                gid  [[thread_position_in_grid]]
) {
    if (gid.x >= h) return;
    x[ulong(gid.y) * h + gid.x] = gid.y == 0u ? dsp_round_bf16(float(emb[gid.x])) : float(mask[gid.x]);
}

// --- Residual: x = bf16(x) + d ----------------------------------------------
kernel void dspark_residual(
    device float*      x [[buffer(0)]],
    device const half* d [[buffer(1)]],
    constant uint&     n [[buffer(2)]],
    uint               i [[thread_position_in_grid]]
) {
    if (i < n) x[i] = dsp_round_bf16(x[i]) + float(d[i]);
}

// --- SwiGLU: act = silu(g) * u ------------------------------------------------
kernel void dspark_silu_mul(
    device const half* g   [[buffer(0)]],
    device const half* u   [[buffer(1)]],
    device half*       act [[buffer(2)]],
    constant uint&     n   [[buffer(3)]],
    uint               i   [[thread_position_in_grid]]
) {
    if (i >= n) return;
    const float gv = float(g[i]);
    act[i] = half(gv / (1.0f + precise::exp(-gv)) * float(u[i]));
}

// --- NeoX RoPE in place on `[rows][n_heads][hd]` at `pos0 + row` -------------
kernel void dspark_rope_rows(
    device half*    x       [[buffer(0)]],
    constant uint&  n_heads [[buffer(1)]],
    constant uint&  hd      [[buffer(2)]],
    constant uint&  pos0    [[buffer(3)]],
    constant float& theta   [[buffer(4)]],
    uint3           gid     [[thread_position_in_grid]]
) {
    const uint half_d = hd / 2u;
    if (gid.x >= half_d || gid.y >= n_heads) return;
    const float pos = float(pos0 + gid.z);
    const float f = pos * precise::pow(theta, -2.0f * float(gid.x) / float(hd));
    const float s = precise::sin(f);
    const float c = precise::cos(f);
    device half* head = x + (ulong(gid.z) * n_heads + gid.y) * hd;
    const float a = float(head[gid.x]);
    const float b = float(head[gid.x + half_d]);
    head[gid.x] = half(a * c - b * s);
    head[gid.x + half_d] = half(b * c + a * s);
}

// --- Context K/V store: K roped at its position, both into the ring ----------
// `k`, `v` FP16 `[rows][kv_heads][hd]` (K already k_norm_ctx-normed); row r is
// position p0 + r, stored at cache row (p0 + r) % cap.
kernel void dspark_ctx_store(
    device const half* k       [[buffer(0)]],
    device const half* v       [[buffer(1)]],
    device half*       kc      [[buffer(2)]],
    device half*       vc      [[buffer(3)]],
    constant uint&     kv_heads[[buffer(4)]],
    constant uint&     hd      [[buffer(5)]],
    constant uint&     p0      [[buffer(6)]],
    constant uint&     cap     [[buffer(7)]],
    constant float&    theta   [[buffer(8)]],
    uint3              gid     [[thread_position_in_grid]]
) {
    const uint half_d = hd / 2u;
    if (gid.x >= half_d || gid.y >= kv_heads) return;
    const uint p = p0 + gid.z;
    const ulong src = (ulong(gid.z) * kv_heads + gid.y) * hd;
    const ulong dst = (ulong(p % cap) * kv_heads + gid.y) * hd;
    const float f = float(p) * precise::pow(theta, -2.0f * float(gid.x) / float(hd));
    const float s = precise::sin(f);
    const float c = precise::cos(f);
    const float a = float(k[src + gid.x]);
    const float b = float(k[src + gid.x + half_d]);
    kc[dst + gid.x] = half(a * c - b * s);
    kc[dst + gid.x + half_d] = half(b * c + a * s);
    vc[dst + gid.x] = v[src + gid.x];
    vc[dst + gid.x + half_d] = v[src + gid.x + half_d];
}

// --- Drafter attention --------------------------------------------------------
// Query row r of the block (head h) over context positions [first, t) from the
// ring (row p % cap) and all `n_block` block rows (bidirectional), scale
// 1/sqrt(hd), no sink, no bias. One threadgroup (8 simdgroups) per (head, row).
kernel void dspark_attention(
    device const half*  q        [[buffer(0)]],
    device const half*  kc       [[buffer(1)]],
    device const half*  vc       [[buffer(2)]],
    device const half*  kb       [[buffer(3)]],
    device const half*  vb       [[buffer(4)]],
    device half*        out      [[buffer(5)]],
    constant uint&      first    [[buffer(6)]],
    constant uint&      t        [[buffer(7)]],
    constant uint&      cap      [[buffer(8)]],
    constant uint&      n_block  [[buffer(9)]],
    constant uint&      heads    [[buffer(10)]],
    constant uint&      kv_heads [[buffer(11)]],
    constant uint&      hd       [[buffer(12)]],
    uint2               tg       [[threadgroup_position_in_grid]],
    uint                sg_idx   [[simdgroup_index_in_threadgroup]],
    uint                lane     [[thread_index_in_simdgroup]]
) {
    threadgroup float tg_m[8];
    threadgroup float tg_l[8];
    threadgroup float tg_acc[8][128];
    const uint h = tg.x;
    const uint r = tg.y;
    const uint per = hd / 32u;
    const uint kvh = h / (heads / kv_heads);
    const uint kvd = kv_heads * hd;
    const float scale = rsqrt(float(hd));
    device const half* qr = q + (ulong(r) * heads + h) * hd + lane * per;
    float qv[4];
    for (uint j = 0; j < per; ++j) qv[j] = float(qr[j]);
    const uint n_ctx = t - first;
    float m = -INFINITY;
    float l = 0.0f;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint j = sg_idx; j < n_ctx + n_block; j += 8u) {
        device const half* kr;
        device const half* vr;
        if (j < n_ctx) {
            const ulong row = ulong((first + j) % cap) * kvd + kvh * hd + lane * per;
            kr = kc + row;
            vr = vc + row;
        } else {
            const ulong row = ulong(j - n_ctx) * kvd + kvh * hd + lane * per;
            kr = kb + row;
            vr = vb + row;
        }
        float s = 0.0f;
        for (uint i = 0; i < per; ++i) s = fma(qv[i], float(kr[i]), s);
        s = simd_sum(s) * scale;
        const float m_new = max(m, s);
        const float corr = precise::exp(m - m_new);
        const float w = precise::exp(s - m_new);
        l = l * corr + w;
        for (uint i = 0; i < per; ++i) acc[i] = fma(w, float(vr[i]), acc[i] * corr);
        m = m_new;
    }
    if (lane == 0u) {
        tg_m[sg_idx] = m;
        tg_l[sg_idx] = l;
    }
    for (uint i = 0; i < per; ++i) tg_acc[sg_idx][lane * per + i] = acc[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg_idx != 0u) return;
    float M = -INFINITY;
    for (uint g = 0; g < 8u; ++g) M = max(M, tg_m[g]);
    float denom = 0.0f;
    float f[8];
    for (uint g = 0; g < 8u; ++g) {
        f[g] = (tg_l[g] > 0.0f) ? precise::exp(tg_m[g] - M) : 0.0f;
        denom += tg_l[g] * f[g];
    }
    device half* o = out + (ulong(r) * heads + h) * hd;
    for (uint i = 0; i < per; ++i) {
        const uint c = lane * per + i;
        float v = 0.0f;
        for (uint g = 0; g < 8u; ++g) v = fma(tg_acc[g][c], f[g], v);
        o[c] = half(v / denom);
    }
}

// --- Markov head: one step's RNN input --------------------------------------
// input = [state (rank) | w1[last] (rank) | hidden_i (h) | feature (h)], FP16.
kernel void dspark_markov_input(
    device const float*  state   [[buffer(0)]],
    device const bfloat* w1      [[buffer(1)]],
    device const uint*   last    [[buffer(2)]],
    device const half*   hidden  [[buffer(3)]],
    device const half*   feature [[buffer(4)]],
    device half*         input   [[buffer(5)]],
    constant uint&       rank    [[buffer(6)]],
    constant uint&       h       [[buffer(7)]],
    uint                 i       [[thread_position_in_grid]]
) {
    const uint n = 2u * rank + 2u * h;
    if (i >= n) return;
    float v;
    if (i < rank) {
        v = state[i];
    } else if (i < 2u * rank) {
        v = float(w1[ulong(last[0]) * rank + (i - rank)]);
    } else if (i < 2u * rank + h) {
        v = float(hidden[i - 2u * rank]);
    } else {
        v = float(feature[i - 2u * rank - h]);
    }
    input[i] = half(v);
}

// --- Markov head: gates ------------------------------------------------------
// g = rnn_proj(input) + bias, [keep | update | latent] of `rank` each:
// latent = tanh(g_l); keep = sigmoid(g_k); state = keep*state + (1-keep)*tanh(g_u).
kernel void dspark_markov_gate(
    device const half*   g      [[buffer(0)]],
    device const bfloat* bias   [[buffer(1)]],
    device float*        state  [[buffer(2)]],
    device half*         latent [[buffer(3)]],
    constant uint&       rank   [[buffer(4)]],
    uint                 j      [[thread_position_in_grid]]
) {
    if (j >= rank) return;
    const float gk = float(g[j]) + float(bias[j]);
    const float gu = float(g[rank + j]) + float(bias[rank + j]);
    const float gl = float(g[2u * rank + j]) + float(bias[2u * rank + j]);
    latent[j] = half(precise::tanh(gl));
    const float keep = 1.0f / (1.0f + precise::exp(-gk));
    state[j] = keep * state[j] + (1.0f - keep) * precise::tanh(gu);
}

// --- Markov head: corrected logits and a partial argmax ----------------------
// z[v] = base[v] + fold_b[v] + sum_j fold_t[j][v] * latent[j], v < n_vocab.
// The softcap is monotonic, so the argmax is taken on z. Each threadgroup
// writes its best (value, index); ties keep the lower index.
kernel void dspark_markov_logits(
    device const half*  base    [[buffer(0)]],
    device const half*  fold_b  [[buffer(1)]],
    device const half*  fold_t  [[buffer(2)]],
    device const half*  latent  [[buffer(3)]],
    device float*       part_v  [[buffer(4)]],
    device uint*        part_i  [[buffer(5)]],
    constant uint&      n_vocab [[buffer(6)]],
    constant uint&      stride  [[buffer(7)]],
    constant uint&      rank    [[buffer(8)]],
    uint                tgi     [[threadgroup_position_in_grid]],
    uint                tid     [[thread_position_in_threadgroup]],
    uint                sg      [[simdgroup_index_in_threadgroup]],
    uint                lane    [[thread_index_in_simdgroup]]
) {
    threadgroup float lat[256];
    threadgroup float sv[8];
    threadgroup uint si[8];
    for (uint j = tid; j < rank; j += 256u) lat[j] = float(latent[j]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint v = tgi * 256u + tid;
    float z = -INFINITY;
    if (v < n_vocab) {
        float acc = float(base[v]) + float(fold_b[v]);
        for (uint j = 0; j < rank; ++j) acc = fma(float(fold_t[ulong(j) * stride + v]), lat[j], acc);
        z = acc;
    }
    uint idx = v;
    for (uint off = 16u; off > 0u; off >>= 1u) {
        const float oz = simd_shuffle_down(z, off);
        const uint oi = simd_shuffle_down(idx, off);
        if (oz > z || (oz == z && oi < idx)) { z = oz; idx = oi; }
    }
    if (lane == 0u) { sv[sg] = z; si[sg] = idx; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float bz = sv[0];
        uint bi = si[0];
        for (uint g = 1; g < 8u; ++g) {
            if (sv[g] > bz || (sv[g] == bz && si[g] < bi)) { bz = sv[g]; bi = si[g]; }
        }
        part_v[tgi] = bz;
        part_i[tgi] = bi;
    }
}

// Reduces the partials; writes the token to `last[0]` and `proposals[step]`.
kernel void dspark_markov_argmax(
    device const float* part_v    [[buffer(0)]],
    device const uint*  part_i    [[buffer(1)]],
    device uint*        last      [[buffer(2)]],
    device uint*        proposals [[buffer(3)]],
    constant uint&      n_part    [[buffer(4)]],
    constant uint&      step      [[buffer(5)]],
    uint                tid       [[thread_position_in_threadgroup]],
    uint                sg        [[simdgroup_index_in_threadgroup]],
    uint                lane      [[thread_index_in_simdgroup]]
) {
    threadgroup float sv[8];
    threadgroup uint si[8];
    float z = -INFINITY;
    uint idx = 0xffffffffu;
    for (uint p = tid; p < n_part; p += 256u) {
        if (part_v[p] > z || (part_v[p] == z && part_i[p] < idx)) { z = part_v[p]; idx = part_i[p]; }
    }
    for (uint off = 16u; off > 0u; off >>= 1u) {
        const float oz = simd_shuffle_down(z, off);
        const uint oi = simd_shuffle_down(idx, off);
        if (oz > z || (oz == z && oi < idx)) { z = oz; idx = oi; }
    }
    if (lane == 0u) { sv[sg] = z; si[sg] = idx; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float bz = sv[0];
        uint bi = si[0];
        for (uint g = 1; g < 8u; ++g) {
            if (sv[g] > bz || (sv[g] == bz && si[g] < bi)) { bz = sv[g]; bi = si[g]; }
        }
        last[0] = bi;
        proposals[step] = bi;
    }
}

// --- BF16 matrix to FP16, transposed: out[j][i] = in[i][j] ---------------------
kernel void dspark_transpose_bf16(
    device const bfloat* src  [[buffer(0)]],
    device half*         dst  [[buffer(1)]],
    constant uint&       rows [[buffer(2)]],
    constant uint&       cols [[buffer(3)]],
    uint2                gid  [[thread_position_in_grid]]
) {
    if (gid.x >= cols || gid.y >= rows) return;
    dst[ulong(gid.x) * rows + gid.y] = half(float(src[ulong(gid.y) * cols + gid.x]));
}

// BF16 vector to FP16.
kernel void dspark_bf16_to_half(
    device const bfloat* src [[buffer(0)]],
    device half*         dst [[buffer(1)]],
    constant uint&       n   [[buffer(2)]],
    uint                 i   [[thread_position_in_grid]]
) {
    if (i < n) dst[i] = half(float(src[i]));
}
