// ============================================================================
// inkling2_ops - Inkling 2 (`inkling2_loop`) per-layer ops that no other
// family's kernel computes. PORT-LOCAL. Contract: the CPU reference in
// `turbospark_compute::inkling2` (`route_sigmoid`, `l2_clip`, `short_conv`,
// `block_attn_res_mix`), held by `tests/inkling2_ops_parity.rs`.
//
// COMPILED CONCATENATED AFTER `moe.metal` for `router_select_next_one_simd`, so
// the router's ranking and its lower-index tie rule are the SAME code every
// other top-k router in this port uses, not a copy that could drift.
//
// FP32 accumulate, FP16 activations, BF16 checkpoint weights throughout.
// ============================================================================

// --- Router: sigmoid scores, bias-corrected selection, top 8 ----------------
//
// Selection ranks `sigmoid(logit) + correction_bias`; the weights are the
// UNBIASED sigmoid scores of the selected experts, normalized to sum 1 and
// times `route_scale`. Output is in rank order (best first), which is the
// routed-slot order and therefore the reduction order. One SIMD group,
// num_experts in 8..=256 (the host checks).
kernel void inkling2_router_select_k8(
    device const float* logits          [[buffer(0)]],
    device const float* correction_bias [[buffer(1)]],
    device uint*        out_indices     [[buffer(2)]],
    device half*        out_weights     [[buffer(3)]],
    constant uint&      num_experts     [[buffer(4)]],
    constant float&     route_scale     [[buffer(5)]],
    uint                sg_idx          [[simdgroup_index_in_threadgroup]],
    uint                lane            [[thread_index_in_simdgroup]]
) {
    const uint NE = num_experts;
    if (sg_idx != 0u || NE > 32u * kRouterMaxPerLane) return;
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
    const float inv = route_scale / (sum + 1e-20f);
    for (uint i = 0; i < 8; ++i) {
        out_indices[i] = top_idx[i];
        out_weights[i] = half(scores[i] * inv);
    }
}

// --- L2 clip, in place ------------------------------------------------------
//
// For each of `n_vec` vectors of `dim`: y = x * scale (when a scale is bound),
// then y *= min(1, threshold / (|y| + 1e-6)), then y *= post_scale. The post
// scale is the full-attention `tau(pos)` on q and r (1.0 elsewhere), fused here
// because it applies to exactly the clipped vector. `scale` is indexed
// `scale[v * scale_stride + i]`: stride 0 shares one `[dim]` scale across the
// vectors, stride `dim` gives each its own. One SIMD group per vector.
kernel void inkling2_l2_clip(
    device half*          x            [[buffer(0)]],
    device const bfloat*  scale        [[buffer(1)]],
    constant uint&        n_vec        [[buffer(2)]],
    constant uint&        dim          [[buffer(3)]],
    constant uint&        scale_stride [[buffer(4)]],
    constant uint&        has_scale    [[buffer(5)]],
    constant float&       threshold    [[buffer(6)]],
    constant float&       post_scale   [[buffer(7)]],
    uint                  tg_idx       [[threadgroup_position_in_grid]],
    uint                  sg_idx       [[simdgroup_index_in_threadgroup]],
    uint                  lane         [[thread_index_in_simdgroup]]
) {
    const uint v = tg_idx * 8u + sg_idx;
    if (v >= n_vec) return;
    device half* row = x + ulong(v) * dim;
    device const bfloat* s = scale + ulong(v) * scale_stride;
    float ss = 0.0f;
    for (uint i = lane; i < dim; i += 32u) {
        const float y = has_scale != 0u ? float(row[i]) * float(s[i]) : float(row[i]);
        ss = fma(y, y, ss);
    }
    const float norm = sqrt(simd_sum(ss));
    const float f = clamp(threshold / (norm + 1e-6f), 0.0f, 1.0f) * post_scale;
    for (uint i = lane; i < dim; i += 32u) {
        const float y = has_scale != 0u ? float(row[i]) * float(s[i]) : float(row[i]);
        row[i] = half(y * f);
    }
}

// --- Causal short convolution with residual, one decode row ---------------
//
// `weight` is BF16 `[ch, taps]`, tap taps-1 on the CURRENT row. `tail` is FP16
// `[taps-1, ch]`, oldest first, advanced in place; it holds the conv INPUTS,
// which are FP16 already, so keeping it FP16 loses nothing. With `heads > 0`
// the full-attention K/V variant adds per-head taps `dyn[head * taps + lag]`
// (lag 0 = current row, FP16, from `weight_proj`) and the output scale becomes
// 1/sqrt(2 taps + 1) instead of 1/sqrt(taps + 1). Output = (conv + x) * scale.
// One thread per channel; each reads and writes only its own channel, so the
// in-place tail update needs no barrier.
kernel void inkling2_short_conv_decode(
    device const half*    x       [[buffer(0)]],
    device half*          tail    [[buffer(1)]],
    device const bfloat*  weight  [[buffer(2)]],
    device const half*    dyn     [[buffer(3)]],
    device half*          out     [[buffer(4)]],
    constant uint&        ch      [[buffer(5)]],
    constant uint&        taps    [[buffer(6)]],
    constant uint&        heads   [[buffer(7)]],
    uint                  c       [[thread_position_in_grid]]
) {
    if (c >= ch) return;
    const float xc = float(x[c]);
    // hist(j) for j < taps-1 is tail row j, hist(taps-1) is the current row.
    float acc = 0.0f;
    for (uint j = 0; j + 1u < taps; ++j) {
        acc = fma(float(weight[c * taps + j]), float(tail[j * ch + c]), acc);
    }
    acc = fma(float(weight[c * taps + taps - 1u]), xc, acc);
    float denom = float(taps + 1u);
    if (heads != 0u) {
        const uint head = c / (ch / heads);
        device const half* d = dyn + head * taps;
        acc = fma(float(d[0]), xc, acc);
        for (uint lag = 1; lag < taps; ++lag) {
            acc = fma(float(d[lag]), float(tail[(taps - 1u - lag) * ch + c]), acc);
        }
        denom = float(2u * taps + 1u);
    }
    out[c] = half((acc + xc) * rsqrt(denom));
    for (uint j = 0; j + 2u < taps; ++j) {
        tail[j * ch + c] = tail[(j + 1u) * ch + c];
    }
    if (taps > 1u) tail[(taps - 2u) * ch + c] = half(xc);
}

// --- Block attention residual ("depth attention") ---------------------------
//
// `states` is FP16 `[n_states, h]`: the earlier block outputs, then the
// current partial state LAST. For each query (one threadgroup each, BF16
// `[n_q, h]`): score_s = rinv_s * (state_s . q) / sqrt(h) with rinv_s the
// no-gamma RMS-norm factor (eps 1e-6), softmax over s, and out = sum_s w_s *
// rinv_s * state_s, i.e. the weighted sum of the NORMED states. One pass over
// the states computes both the sum of squares and the dot product.
constant constexpr uint kInkMaxDepthStates = 16;

kernel void inkling2_depth_mix(
    device const half*    states   [[buffer(0)]],
    device const bfloat*  queries  [[buffer(1)]],
    device half*          out      [[buffer(2)]],
    constant uint&        n_states [[buffer(3)]],
    constant uint&        h        [[buffer(4)]],
    uint                  q_idx    [[threadgroup_position_in_grid]],
    uint                  tid      [[thread_position_in_threadgroup]],
    uint                  sg_idx   [[simdgroup_index_in_threadgroup]],
    uint                  lane     [[thread_index_in_simdgroup]]
) {
    threadgroup float part_ss[kInkMaxDepthStates][8];
    threadgroup float part_dq[kInkMaxDepthStates][8];
    threadgroup float w_s[kInkMaxDepthStates];
    device const bfloat* q = queries + ulong(q_idx) * h;
    for (uint s = 0; s < n_states; ++s) {
        device const half* st = states + ulong(s) * h;
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
        float sc[kInkMaxDepthStates];
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
        // Fold the normalization factor into the weight once per state.
        for (uint s = 0; s < n_states; ++s) w_s[s] = sc[s] / z * part_ss[s][0];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device half* o = out + ulong(q_idx) * h;
    for (uint i = tid; i < h; i += 256u) {
        float acc = 0.0f;
        for (uint s = 0; s < n_states; ++s) {
            acc = fma(w_s[s], float(states[ulong(s) * h + i]), acc);
        }
        o[i] = half(acc);
    }
}

// --- Relative-bias table ----------------------------------------------------
//
// table[h * extent + d] = r_h . proj[:, d]. `r` FP16 `[heads, d_rel]` (clipped,
// times tau on full layers), `proj` BF16 `[d_rel, extent]` row-major. One
// thread per (h, d); d_rel is 16, so a plain loop beats any reduction.
kernel void inkling2_rel_table(
    device const half*    r       [[buffer(0)]],
    device const bfloat*  proj    [[buffer(1)]],
    device float*         table   [[buffer(2)]],
    constant uint&        heads   [[buffer(3)]],
    constant uint&        d_rel   [[buffer(4)]],
    constant uint&        extent  [[buffer(5)]],
    uint                  gid     [[thread_position_in_grid]]
) {
    if (gid >= heads * extent) return;
    const uint h = gid / extent;
    const uint d = gid % extent;
    float acc = 0.0f;
    for (uint i = 0; i < d_rel; ++i) {
        acc = fma(float(r[h * d_rel + i]), float(proj[i * extent + d]), acc);
    }
    table[gid] = acc;
}

// --- Decode attention with relative bias and a per-head sink ----------------
//
// One threadgroup per query head, 8 SIMD groups. Group g takes positions
// first + g, first + g + 8, ... with an online softmax; each lane owns
// head_dim / 32 channels (head_dim <= 128). The groups merge through
// threadgroup memory together with the sink: the sink logit t joins the
// denominator with a zero value, so out = sum e^(s-M) v / (e^(t-M) + sum
// e^(s-M)). logit = q.k / sqrt(head_dim) + table[h, pos - kpos] (no bias at
// distance >= extent). K and V are FP16 `[n_pos, kv_heads, head_dim]`.
constant constexpr uint kInkAttnMaxPerLane = 4;   // head_dim <= 128

kernel void inkling2_attention_decode(
    device const half*   q        [[buffer(0)]],
    device const half*   k        [[buffer(1)]],
    device const half*   v        [[buffer(2)]],
    device const float*  table    [[buffer(3)]],
    device const float*  sinks    [[buffer(4)]],
    device half*         out      [[buffer(5)]],
    constant uint&       first    [[buffer(6)]],
    constant uint&       pos      [[buffer(7)]],
    constant uint&       kv_heads [[buffer(8)]],
    constant uint&       group    [[buffer(9)]],
    constant uint&       head_dim [[buffer(10)]],
    constant uint&       extent   [[buffer(11)]],
    uint                 h        [[threadgroup_position_in_grid]],
    uint                 sg_idx   [[simdgroup_index_in_threadgroup]],
    uint                 lane     [[thread_index_in_simdgroup]]
) {
    threadgroup float tg_m[8];
    threadgroup float tg_l[8];
    threadgroup float tg_acc[8][128];
    const uint per = head_dim / 32u;
    const uint kvh = h / group;
    const uint kvd = kv_heads * head_dim;
    const float scale = rsqrt(float(head_dim));
    float qv[kInkAttnMaxPerLane];
    for (uint j = 0; j < per; ++j) qv[j] = float(q[h * head_dim + lane * per + j]);

    float m = -INFINITY;
    float l = 0.0f;
    float acc[kInkAttnMaxPerLane] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint kp = first + sg_idx; kp <= pos; kp += 8u) {
        device const half* kr = k + ulong(kp) * kvd + kvh * head_dim + lane * per;
        float s = 0.0f;
        for (uint j = 0; j < per; ++j) s = fma(qv[j], float(kr[j]), s);
        s = simd_sum(s) * scale;
        const uint d = pos - kp;
        if (d < extent) s += table[h * extent + d];
        const float m_new = max(m, s);
        const float corr = precise::exp(m - m_new);
        const float w = precise::exp(s - m_new);
        l = l * corr + w;
        device const half* vr = v + ulong(kp) * kvd + kvh * head_dim + lane * per;
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

    const float t = sinks[h];
    float M = t;
    for (uint g = 0; g < 8u; ++g) M = max(M, tg_m[g]);
    float denom = precise::exp(t - M);
    float f[8];
    for (uint g = 0; g < 8u; ++g) {
        // A group with no positions has m = -inf and l = 0: weight 0.
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
