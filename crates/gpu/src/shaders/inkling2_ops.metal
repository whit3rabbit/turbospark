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
    // `scores / sum * scale`, in that order: `route_sigmoid`'s rounding, so
    // the resident-expert path routes with the host path's exact weights.
    const float denom = sum + 1e-20f;
    for (uint i = 0; i < 8; ++i) {
        out_indices[i] = top_idx[i];
        out_weights[i] = half(scores[i] / denom * route_scale);
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
//
// The per-vector body is a helper so `inkling2_qkv_post` runs the SAME code:
// the fused kernel is held bit-identical to this one, not merely close.
static inline void ink2_l2_clip_vec(
    device half*         row,
    device const bfloat* s,
    uint                 dim,
    bool                 has_scale,
    float                threshold,
    float                post_scale,
    uint                 lane
) {
    // Fast math may not reassociate here: the fused and separate kernels
    // inline this body into different contexts, and reassociation orders
    // operands by definition rank, which differs between them.
    #pragma clang fp reassociate(off)
    float ss = 0.0f;
    for (uint i = lane; i < dim; i += 32u) {
        const float y = has_scale ? float(row[i]) * float(s[i]) : float(row[i]);
        ss = fma(y, y, ss);
    }
    const float norm = sqrt(simd_sum(ss));
    const float f = clamp(threshold / (norm + 1e-6f), 0.0f, 1.0f) * post_scale;
    for (uint i = lane; i < dim; i += 32u) {
        const float y = has_scale ? float(row[i]) * float(s[i]) : float(row[i]);
        row[i] = half(y * f);
    }
}

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
    ink2_l2_clip_vec(x + ulong(v) * dim, scale + ulong(v) * scale_stride, dim,
                     has_scale != 0u, threshold, post_scale, lane);
}

// One decode-row channel of the causal short conv (the body of
// `inkling2_short_conv_decode`, shared with `inkling2_qkv_post`). Advances
// the channel's tail in place and returns the FP16 output.
static inline half ink2_conv_channel(
    device const half*    x,
    device half*          tail,
    device const bfloat*  weight,
    device const half*    dyn,
    uint                  c,
    uint                  ch,
    uint                  taps,
    uint                  heads
) {
    // Fast math may not reassociate here: the fused and separate kernels
    // inline this body into different contexts, and reassociation orders
    // operands by definition rank, which differs between them.
    #pragma clang fp reassociate(off)
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
    const half result = half((acc + xc) * rsqrt(denom));
    for (uint j = 0; j + 2u < taps; ++j) {
        tail[j * ch + c] = tail[(j + 1u) * ch + c];
    }
    if (taps > 1u) tail[(taps - 2u) * ch + c] = half(xc);
    return result;
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
    out[c] = ink2_conv_channel(x, tail, weight, dyn, c, ch, taps, heads);
}

// --- Short conv fused with the residual step after it ------------------------
//
// The o-proj and MLP-output convs each feed the stream: `cur += conv` (mode 0,
// the `residual_add_fp16` arithmetic on the FP16-rounded conv output), or at
// the first layer of a residual block `save = cur; cur = conv` (mode 1, the
// two copies that push the finished block and start a new one). After oMLX
// a93fd7b's conv+residual kernels (Apache-2.0); this port's own code. Static
// taps only. One thread per channel, as `inkling2_short_conv_decode`.
kernel void inkling2_short_conv_residual(
    device const half*    x       [[buffer(0)]],
    device half*          tail    [[buffer(1)]],
    device const bfloat*  weight  [[buffer(2)]],
    device half*          cur     [[buffer(3)]],
    device half*          save    [[buffer(4)]],
    constant uint&        ch      [[buffer(5)]],
    constant uint&        taps    [[buffer(6)]],
    constant uint&        mode    [[buffer(7)]],
    uint                  c       [[thread_position_in_grid]]
) {
    if (c >= ch) return;
    // `dyn` is unread with heads 0; any half pointer satisfies the signature.
    const half o = ink2_conv_channel(x, tail, weight, x, c, ch, taps, 0u);
    if (mode == 0u) {
        cur[c] = half(float(cur[c]) + float(o));
    } else {
        save[c] = cur[c];
        cur[c] = o;
    }
}

// --- Fused attention post-projection (decode) --------------------------------
//
// After the attention-input GEMVs, one dispatch does what six did: the K and V
// short convs (into the slot's KV row), the K L2 clip, the V per-head unit RMS,
// and the q and r L2 clips with tau. After oMLX `qkv_post` (jundot/omlx
// 8dddad9a, Apache-2.0); the code here is this port's own and calls the same
// helpers as the separate kernels, so its output is bit-identical to the
// six-dispatch chain (`tests/inkling2_ops_parity.rs`).
//
// One SIMD group per vector: q heads, then r heads, then K heads, then V
// heads. Each lane of a K/V group convolves exactly the channels it later
// clips or normalizes (head * hd + lane + 32 j), so it reads back only its
// own writes and needs no barrier. The V norm reproduces
// `rmsnorm_no_scale_perhead`'s reduction tree (hd threads, one element each:
// a simd_sum per 32-element group, then a simd_sum of the group partials)
// inside one SIMD group. hd must be a multiple of 32 and at most 256 (host).
struct Ink2QkvPostParams {
    uint  heads;
    uint  kv_heads;
    uint  head_dim;
    uint  d_rel;
    uint  taps;
    uint  dyn_heads;   // kv_heads on full-attention layers, 0 on SWA
    float qk_clip;
    float rel_clip;
    float tau;
    float rms_eps;
};

kernel void inkling2_qkv_post(
    device half*          q        [[buffer(0)]],
    device const bfloat*  q_scale  [[buffer(1)]],
    device half*          r        [[buffer(2)]],
    device const bfloat*  r_scale  [[buffer(3)]],
    device const half*    k_pre    [[buffer(4)]],
    device const half*    v_pre    [[buffer(5)]],
    device half*          tail_k   [[buffer(6)]],
    device half*          tail_v   [[buffer(7)]],
    device const bfloat*  k_w      [[buffer(8)]],
    device const bfloat*  v_w      [[buffer(9)]],
    device const half*    dyn_k    [[buffer(10)]],
    device const half*    dyn_v    [[buffer(11)]],
    device half*          k_out    [[buffer(12)]],
    device half*          v_out    [[buffer(13)]],
    constant Ink2QkvPostParams& p  [[buffer(14)]],
    uint                  tg_idx   [[threadgroup_position_in_grid]],
    uint                  sg_idx   [[simdgroup_index_in_threadgroup]],
    uint                  lane     [[thread_index_in_simdgroup]]
) {
    // Fast math may not reassociate here: the fused and separate kernels
    // inline this body into different contexts, and reassociation orders
    // operands by definition rank, which differs between them.
    #pragma clang fp reassociate(off)
    const uint v = tg_idx * 8u + sg_idx;
    const uint H = p.heads, KVH = p.kv_heads, HD = p.head_dim;
    const uint kvd = KVH * HD;
    if (v < H) {
        ink2_l2_clip_vec(q + ulong(v) * HD, q_scale + ulong(v) * HD, HD, true,
                         p.qk_clip, p.tau, lane);
        return;
    }
    if (v < 2u * H) {
        const uint hh = v - H;
        ink2_l2_clip_vec(r + ulong(hh) * p.d_rel, r_scale + ulong(hh) * p.d_rel, p.d_rel,
                         true, p.rel_clip, p.tau, lane);
        return;
    }
    if (v < 2u * H + KVH) {
        const uint head = v - 2u * H;
        for (uint i = lane; i < HD; i += 32u) {
            const uint c = head * HD + i;
            k_out[c] = ink2_conv_channel(k_pre, tail_k, k_w, dyn_k, c, kvd, p.taps, p.dyn_heads);
        }
        ink2_l2_clip_vec(k_out + ulong(head) * HD, k_w, HD, false, p.qk_clip, 1.0f, lane);
        return;
    }
    if (v < 2u * H + 2u * KVH) {
        const uint head = v - 2u * H - KVH;
        device half* row = v_out + ulong(head) * HD;
        const uint groups = HD / 32u;
        float xs[8];
        float partial = 0.0f;
        for (uint g = 0; g < groups; ++g) {
            const uint c = head * HD + g * 32u + lane;
            const half o = ink2_conv_channel(v_pre, tail_v, v_w, dyn_v, c, kvd, p.taps, p.dyn_heads);
            row[g * 32u + lane] = o;
            xs[g] = float(o);
            // rms_block_inv: acc = fma(v, v, 0) per thread, simd_sum per group.
            const float sg = simd_sum(fma(xs[g], xs[g], 0.0f));
            if (lane == g) partial = sg;
        }
        const float total = simd_sum(lane < groups ? partial : 0.0f);
        const float inv = rsqrt(total / float(HD) + p.rms_eps);
        for (uint g = 0; g < groups; ++g) {
            row[g * 32u + lane] = half(xs[g] * inv);
        }
    }
}

// --- Block attention residual ("depth attention") ---------------------------
//
// `states` is FP16 `[n_states, h]`: the earlier block outputs, then the
// current partial state LAST. For each query (one threadgroup each, F32
// `[n_q, h]`, the checkpoint's own dtype): score_s = rinv_s * (state_s . q) / sqrt(h) with rinv_s the
// no-gamma RMS-norm factor (eps 1e-6), softmax over s, and out = sum_s w_s *
// rinv_s * state_s, i.e. the weighted sum of the NORMED states. One pass over
// the states computes both the sum of squares and the dot product.
constant constexpr uint kInkMaxDepthStates = 16;
// Elements per thread the fused mix+norm keeps in registers (h <= 256 * this).
constant constexpr uint kInkMixNormMaxPerThread = 16;

// State row `s` of the mix: `states` rows, except that the LAST state may
// live elsewhere (`cur`), so the fused kernel reads the stream in place
// instead of copying it into the state array first.
static inline device const half* ink2_state_row(
    device const half* states, device const half* cur, uint s, uint n_states, uint h
) {
    return s + 1u == n_states ? cur : states + ulong(s) * h;
}

// The mix weights `w_s[s]` for one query (the body of `inkling2_depth_mix`,
// shared with `inkling2_depth_mix_norm`). 256 threads; returns after the
// barrier that publishes `w_s`.
static inline void ink2_depth_mix_weights(
    device const half*  states,
    device const half*  cur,
    device const float* q,
    uint                n_states,
    uint                h,
    uint                tid,
    uint                sg_idx,
    uint                lane,
    threadgroup float (*part_ss)[8],
    threadgroup float (*part_dq)[8],
    threadgroup float*  w_s
) {
    for (uint s = 0; s < n_states; ++s) {
        device const half* st = ink2_state_row(states, cur, s, n_states, h);
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
}

static inline half ink2_depth_mix_out(
    device const half* states, device const half* cur, threadgroup const float* w_s,
    uint n_states, uint h, uint i
) {
    float acc = 0.0f;
    for (uint s = 0; s < n_states; ++s) {
        acc = fma(w_s[s], float(ink2_state_row(states, cur, s, n_states, h)[i]), acc);
    }
    return half(acc);
}

kernel void inkling2_depth_mix(
    device const half*    states   [[buffer(0)]],
    device const float*   queries  [[buffer(1)]],
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
    device const half* cur = states + ulong(n_states - 1u) * h;
    ink2_depth_mix_weights(states, cur, queries + ulong(q_idx) * h, n_states, h, tid, sg_idx,
                           lane, part_ss, part_dq, w_s);
    device half* o = out + ulong(q_idx) * h;
    for (uint i = tid; i < h; i += 256u) {
        o[i] = ink2_depth_mix_out(states, cur, w_s, n_states, h, i);
    }
}

// --- Depth mix fused with the RMS norm that follows it ----------------------
//
// After oMLX `attn_res_fused` (jundot/omlx 8dddad9a, Apache-2.0); this port's
// own code. The mix of `states[0..n_states-1]` plus `cur` (the LAST state,
// read in place: no copy into the state array), then for query j the BF16
// RMS norm with weight `w_j` (j < 3: the full layer's q/k/v input norms, or
// one norm). The norm repeats `rmsnorm_bf16w`'s 256-thread reduction tree on
// the FP16-rounded mix output (each thread owns the same elements in both),
// so `norm_out` is bit-identical to `inkling2_depth_mix` then
// `rmsnorm_bf16w`. `mix_out` is written only when `write_mix` is set.
kernel void inkling2_depth_mix_norm(
    device const half*    states    [[buffer(0)]],
    device const half*    cur       [[buffer(1)]],
    device const float*   queries   [[buffer(2)]],
    device half*          mix_out   [[buffer(3)]],
    device const bfloat*  w0        [[buffer(4)]],
    device const bfloat*  w1        [[buffer(5)]],
    device const bfloat*  w2        [[buffer(6)]],
    device half*          norm_out  [[buffer(7)]],
    constant uint&        n_states  [[buffer(8)]],
    constant uint&        h         [[buffer(9)]],
    constant float&       eps       [[buffer(10)]],
    constant uint&        write_mix [[buffer(11)]],
    uint                  q_idx     [[threadgroup_position_in_grid]],
    uint                  tid       [[thread_position_in_threadgroup]],
    uint                  sg_idx    [[simdgroup_index_in_threadgroup]],
    uint                  lane      [[thread_index_in_simdgroup]]
) {
    // Fast math may not reassociate here: the fused and separate kernels
    // inline this body into different contexts, and reassociation orders
    // operands by definition rank, which differs between them.
    #pragma clang fp reassociate(off)
    threadgroup float part_ss[kInkMaxDepthStates][8];
    threadgroup float part_dq[kInkMaxDepthStates][8];
    threadgroup float w_s[kInkMaxDepthStates];
    threadgroup float partial[8];
    ink2_depth_mix_weights(states, cur, queries + ulong(q_idx) * h, n_states, h, tid, sg_idx,
                           lane, part_ss, part_dq, w_s);
    // h <= 256 * kInkMixNormMaxPerThread (host).
    half o[kInkMixNormMaxPerThread];
    float acc = 0.0f;
    uint k = 0;
    for (uint i = tid; i < h; i += 256u, ++k) {
        o[k] = ink2_depth_mix_out(states, cur, w_s, n_states, h, i);
        const float v = float(o[k]);
        acc = fma(v, v, acc);
    }
    if (write_mix != 0u) {
        device half* mo = mix_out + ulong(q_idx) * h;
        k = 0;
        for (uint i = tid; i < h; i += 256u, ++k) mo[i] = o[k];
    }
    // rms_block_inv, 256 threads = 8 SIMD groups.
    acc = simd_sum(acc);
    if (lane == 0u) partial[sg_idx] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg_idx == 0u) {
        float v = (lane < 8u) ? partial[lane] : 0.0f;
        v = simd_sum(v);
        if (lane == 0u) partial[0] = rsqrt(v / float(h) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = partial[0];
    device const bfloat* w = q_idx == 0u ? w0 : (q_idx == 1u ? w1 : w2);
    device half* no = norm_out + ulong(q_idx) * h;
    k = 0;
    for (uint i = tid; i < h; i += 256u, ++k) {
        no[i] = half(float(o[k]) * inv * float(w[i]));
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
// distance >= extent). K and V are FP16 `[ring, kv_heads, head_dim]`, position
// kp at row kp % ring: a sliding-window slot keeps a window-sized ring, a full
// slot passes ring >= pos + 1 so the index is the position.
constant constexpr uint kInkAttnMaxPerLane = 4;   // head_dim <= 128

kernel void inkling2_attention_decode(
    device const half*   q        [[buffer(0)]],
    device const half*   k        [[buffer(1)]],
    device const half*   v        [[buffer(2)]],
    device const float*  table    [[buffer(3)]],
    device const half*   sinks    [[buffer(4)]],
    device half*         out      [[buffer(5)]],
    constant uint&       first    [[buffer(6)]],
    constant uint&       pos      [[buffer(7)]],
    constant uint&       kv_heads [[buffer(8)]],
    constant uint&       group    [[buffer(9)]],
    constant uint&       head_dim [[buffer(10)]],
    constant uint&       extent   [[buffer(11)]],
    constant uint&       ring     [[buffer(12)]],
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

// --- Loop controller combine -------------------------------------------------
//
// out = cur - tanh(f) * cur + tanh(g) * msg, elementwise, where g and f are the
// pre-tanh gain and forget projections (`gain_weight @ gate`, `forget_weight @
// gate`). The rank-64 gate and both projections are ordinary BF16 GEMVs; this
// is the only op of the controller no other kernel computes.
kernel void inkling2_control_combine(
    device const half* cur    [[buffer(0)]],
    device const half* msg    [[buffer(1)]],
    device const half* gain   [[buffer(2)]],
    device const half* forget [[buffer(3)]],
    device half*       out    [[buffer(4)]],
    constant uint&     n      [[buffer(5)]],
    uint               i      [[thread_position_in_grid]]
) {
    if (i >= n) return;
    const float c = float(cur[i]);
    const float f = precise::tanh(float(forget[i]));
    const float g = precise::tanh(float(gain[i]));
    out[i] = half(c - f * c + g * float(msg[i]));
}

// --- FP16 copy ---------------------------------------------------------------
kernel void inkling2_copy_half(
    device const half* src [[buffer(0)]],
    device half*       dst [[buffer(1)]],
    constant uint&     n   [[buffer(2)]],
    uint               i   [[thread_position_in_grid]]
) {
    if (i < n) dst[i] = src[i];
}

// ============================================================================
// inkling2_bf16_gemv_multi - up to 8 resident BF16 GEMVs in one dispatch.
//
// `y_s[r] = sum_c bf16(W_s[r, c]) * x_{x_idx[s]}[c]` for each segment s, every
// matrix addressed in place in one weights buffer (buffer 0, bound at offset
// 0; `w_off` are byte offsets into it). One simdgroup per output row, eight
// rows per threadgroup, 8-byte loads, no threadgroup barrier. The shared
// `bf16_gemv_rows` takes a 256-thread threadgroup per row with scalar loads
// and a barrier, and its ~650 launches a token (most of them a few hundred
// rows) were half of this family's GPU time in the dispatch profile.
//
// Threadgroups [tg_start[s], tg_start[s + 1]) belong to segment s. The host
// checks cols % 4 == 0, 8-byte-aligned rows and inputs, and n_seg <= 8.
// ============================================================================

struct Inkling2GemvTable {
    ulong w_off[8];
    uint rows[8];
    uint cols[8];
    uint x_idx[8];
    uint tg_start[9];
    uint n_seg;
};

constant constexpr uint kInkGemvRowsPerTG = 8;

kernel void inkling2_bf16_gemv_multi(
    device const uchar*              W   [[buffer(0)]],
    constant Inkling2GemvTable&      t   [[buffer(1)]],
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
    const uint row = (tg - t.tg_start[s]) * kInkGemvRowsPerTG + sg;
    if (row >= t.rows[s]) return;
    const uint xi = t.x_idx[s];
    device const half* x = xi == 0u ? x0 : (xi == 1u ? x1 : x2);
    const uint n4 = t.cols[s] / 4u;
    device const bfloat4* w = (device const bfloat4*)(W + t.w_off[s]) + ulong(row) * n4;
    device const half4* xv = (device const half4*)x;
    float acc = 0.0f;
    for (uint i = lane; i < n4; i += 32u) {
        acc += dot(float4(w[i]), float4(xv[i]));
    }
    acc = simd_sum(acc);
    if (lane != 0u) return;
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
    y[row] = half(acc);
}

// The same GEMV with TWO weight rows per simdgroup (16 per threadgroup): one
// input load feeds two independent weight streams, so twice the loads are in
// flight. Each row keeps its own accumulator and the exact term order of the
// one-row form, so its bits are the same (`tests/inkling2_ops_parity.rs`).
constant constexpr uint kInkGemv2RowsPerTG = 16;

kernel void inkling2_bf16_gemv_multi2(
    device const uchar*              W   [[buffer(0)]],
    constant Inkling2GemvTable&      t   [[buffer(1)]],
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
    const uint row0 = (tg - t.tg_start[s]) * kInkGemv2RowsPerTG + 2u * sg;
    const uint nrows = t.rows[s];
    if (row0 >= nrows) return;
    const bool two = row0 + 1u < nrows;
    const uint xi = t.x_idx[s];
    device const half* x = xi == 0u ? x0 : (xi == 1u ? x1 : x2);
    const uint n4 = t.cols[s] / 4u;
    device const bfloat4* wa = (device const bfloat4*)(W + t.w_off[s]) + ulong(row0) * n4;
    device const bfloat4* wb = two ? wa + n4 : wa;
    device const half4* xv = (device const half4*)x;
    float acc0 = 0.0f, acc1 = 0.0f;
    for (uint i = lane; i < n4; i += 32u) {
        const float4 xf = float4(xv[i]);
        acc0 += dot(float4(wa[i]), xf);
        acc1 += dot(float4(wb[i]), xf);
    }
    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);
    if (lane != 0u) return;
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
    y[row0] = half(acc0);
    if (two) y[row0 + 1u] = half(acc1);
}
