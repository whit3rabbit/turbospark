#include <metal_stdlib>
using namespace metal;

// ============================================================================
// mla.metal - DeepSeek V2 multi-head latent attention. PORT-LOCAL, not
// vendored: the Swift engine has no MLA, so there is no upstream kernel to
// mirror. Each kernel's contract is the function of the same name in
// `turbospark_compute::mla`, and nothing else.
//
// The absorbed form (docs/DEEPSEEK2_PHASE0.md): the cache holds ONE row per
// token per layer, `[c (kv_lora) ; k_pe (rope_dim)]` halves, V is the row's
// first `kv_lora` halves, and every head shares it (MLA + absorption is
// MQA). The per-head q nope half is folded through the up-projection
// BEFORE attention, and a second per-head projection maps the 512-wide
// attention back down to v_head_dim afterwards.
//
// The Q8_0 block read repeats the one in `dequant_q8_0.metal`: 34 bytes,
// f16 scale then 32 SIGNED weights, value = int8(q) * d. Signed, not
// unsigned -- see that file's note 1.
// ============================================================================

constant constexpr uint kMlaQ8BlockElems = 32;
constant constexpr uint kMlaQ8BlockBytes = 34;

// RMS-norm the first `rank` halves of a fused `[rank ; tail]` row in place;
// the tail (the rope-carried key) passes through for the rope kernel to
// rotate next. One threadgroup per row, 256 threads; the reduction mirrors
// the rms_norm kernel's. The weight reads as BF16 (u16, widened by a
// 16-bit shift) because every small resident tensor narrows to BF16 at
// transcode -- reading it as `half` mis-parses the exponent and produced
// exactly the fluent-garbage failure this note exists to prevent.
// Contract: `compute::mla_kv_norm` (weights dequantized to f32 there).
kernel void mla_kv_norm(
    device half*         row        [[buffer(0)]],
    device const uint16_t* weight  [[buffer(1)]],
    constant uint&       rank       [[buffer(2)]],
    constant float&      eps        [[buffer(3)]],
    constant uint&       row_stride [[buffer(4)]],
    uint                 tid        [[thread_position_in_threadgroup]],
    uint                 tg         [[threadgroup_position_in_grid]],
    uint                 tpg        [[threads_per_threadgroup]]
) {
    device half* r = row + uint(tg) * row_stride;
    // Reduction: shared partials across the threadgroup.
    threadgroup float partials[256];
    float acc = 0.0f;
    for (uint i = tid; i < rank; i += tpg) {
        float v = float(r[i]);
        acc = fma(v, v, acc);
    }
    partials[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tpg / 2; stride > 0; stride >>= 1) {
        if (tid < stride) partials[tid] += partials[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv = 1.0f / sqrt(partials[0] / float(rank) + eps);
    for (uint i = tid; i < rank; i += tpg) {
        float w = as_type<float>(uint(weight[i]) << 16);
        r[i] = half(float(r[i]) * inv * w);
    }
}

// Rope the trailing `rope_dim` of each `head_dim`-wide q head. The window
// starts at `window_offset` (= nope width) inside the head, which no rope
// kernel expresses: they all rotate from element 0. One thread per PAIR per
// head; dispatch (rotary_dim/2, num_heads, num_tokens).
// Contract: `compute::mla_rope_window`.
kernel void mla_rope_q_pe(
    device half*          data          [[buffer(0)]],
    device const float*   frequencies   [[buffer(1)]],
    constant uint&        position      [[buffer(2)]],
    constant uint&        head_dim      [[buffer(3)]],
    constant uint&        window_offset [[buffer(4)]],
    constant float&       mscale        [[buffer(5)]],
    constant uint&        rotary_dim    [[buffer(6)]],
    uint2                 gid           [[thread_position_in_grid]]
) {
    uint pair = gid.x;
    if (pair >= rotary_dim / 2) return;
    uint head = gid.y;
    device half* head_ptr = data + head * head_dim;
    float angle = float(position) * frequencies[pair];
    float c = cos(angle) * mscale;
    float s = sin(angle) * mscale;
    // CONSECUTIVE-element pairs, ggml's `ggml_rope_cache_init` layout
    // (cache[i0]/cache[i0+1], i0 even). NOT the half-split `(i, i + dim/2)`
    // pairing the rope.metal kernels use: ggml's own rope for this family
    // pairs neighbours, settled empirically against llama.cpp's per-layer
    // dump after the split form degraded every row past position 0 (pair 0
    // is extrapolated at 1.0 rad/position, so the two conventions differ by
    // a radian at position 1 already).
    uint lo = window_offset + 2 * pair;
    uint hi = lo + 1;
    float a = float(head_ptr[lo]);
    float b = float(head_ptr[hi]);
    head_ptr[lo] = half(a * c - b * s);
    head_ptr[hi] = half(a * s + b * c);
}

// Per-head q absorption, emitting the FUSED query row the attention kernel
// scores against the cache: out row h is `[q'_h (kv_lora) ; q_pe_h (rope)]`.
//
// THE PRODUCT IS TRANSPOSE-OF-ROW: q'_h[j] = sum_{i<nope} W_uk_h[i][j] *
// q_nope_h[i], where W_uk_h[i] is kv_b's row `h*(nope+v) + i` -- a 512-wide
// Q8_0 run -- and j indexes INTO that row. Absorption multiplies by the
// TRANSPOSE of the k up-projection, so a per-row GEMV over these rows (the
// shape every other kernel here has) is a DIFFERENT function: it reads 512
// rows per head where the head owns 128 and produces fluent garbage that a
// self-consistent fixture cannot catch.
//
// One threadgroup per head, 256 threads; each thread owns outputs j = tid
// and j = tid + 256 and reduces over the nope rows directly (no cross-
// thread reduce). The pe tail is gathered from the full q row. Contract:
// `compute::mla_absorb_q` plus a plain gather.
kernel void mla_absorb_q_q8_0(
    device const uint8_t* W        [[buffer(0)]],
    device const half*    q        [[buffer(1)]],   // [heads][nope + rope]
    device half*          out      [[buffer(2)]],   // [heads][kv_lora + rope]
    constant uint&        heads    [[buffer(3)]],
    constant uint&        nope     [[buffer(4)]],
    constant uint&        kv_lora  [[buffer(5)]],
    constant uint&        v_dim    [[buffer(6)]],
    constant uint&        rope_dim [[buffer(7)]],
    uint                  tg       [[threadgroup_position_in_grid]],
    uint                  tid      [[thread_position_in_threadgroup]],
    uint                  tpg      [[threads_per_threadgroup]]
) {
    if (tg >= heads) return;
    if (kv_lora != 2 * tpg) return; // each thread owns exactly 2 outputs
    uint head = tg;
    // A kv_b row is KV_LORA wide (the latent is the reduction-free axis of
    // the stored matrix), so the per-row byte run derives from kv_lora.
    uint row_bytes = kv_lora / kMlaQ8BlockElems * kMlaQ8BlockBytes;
    device const uint8_t* w_head = W + uint(head) * (nope + v_dim) * row_bytes;
    device const half* q_h = q + uint(head) * (nope + rope_dim);
    device half* out_h = out + uint(head) * (kv_lora + rope_dim);

    uint j0 = tid, j1 = tid + tpg;
    // The two outputs sit in DIFFERENT Q8_0 blocks (j1 - j0 = 256 > 32).
    // Every row has its OWN block scale, so the f16 scale is re-read per
    // row per output; only the block-local element index is loop-invariant.
    uint e0 = 2 + (j0 % kMlaQ8BlockElems);
    uint e1 = 2 + (j1 % kMlaQ8BlockElems);
    uint b0 = (j0 / kMlaQ8BlockElems) * kMlaQ8BlockBytes;
    uint b1 = (j1 / kMlaQ8BlockElems) * kMlaQ8BlockBytes;
    float acc0 = 0.0f, acc1 = 0.0f;
    for (uint i = 0; i < nope; ++i) {
        device const uint8_t* r0 = w_head + uint(i) * row_bytes + b0;
        device const uint8_t* r1 = w_head + uint(i) * row_bytes + b1;
        ushort raw0 = ushort(r0[0]) | (ushort)(ushort(r0[1]) << 8);
        ushort raw1 = ushort(r1[0]) | (ushort)(ushort(r1[1]) << 8);
        float d0 = float(as_type<half>(raw0));
        float d1 = float(as_type<half>(raw1));
        acc0 = fma(d0 * float(int(as_type<int8_t>(r0[e0]))), float(q_h[i]), acc0);
        acc1 = fma(d1 * float(int(as_type<int8_t>(r1[e1]))), float(q_h[i]), acc1);
    }

    out_h[j0] = half(acc0);
    out_h[j1] = half(acc1);
    // The pe tail, roped in place already.
    if (tid < rope_dim) {
        out_h[kv_lora + tid] = q_h[nope + tid];
    }
}


kernel void mla_attention_decode(
    device const half*   q         [[buffer(0)]],  // [heads][cache_row]
    device const half*   cache     [[buffer(1)]],  // [seq][cache_row]
    device half*         out       [[buffer(2)]],  // [heads][kv_lora]
    constant uint&       heads     [[buffer(3)]],
    constant uint&       cache_row [[buffer(4)]],
    constant uint&       kv_lora   [[buffer(5)]],
    constant uint&       seq_len   [[buffer(6)]],
    constant float&      scale     [[buffer(7)]],
    uint                 sg        [[simdgroup_index_in_threadgroup]],
    uint                 lane      [[thread_index_in_simdgroup]],
    uint                 tid       [[thread_position_in_threadgroup]],
    uint                 tpg       [[threads_per_threadgroup]],
    uint                 tg        [[threadgroup_position_in_grid]]
) {
    if (tg >= heads || kv_lora != 2 * tpg) return; // 512-wide accumulator contract
    threadgroup float shared[8]; // one slot per SIMD group
    device const half* q_h = q + tg * cache_row;

    float m = -INFINITY;
    float d = 0.0f;
    float acc0 = 0.0f, acc1 = 0.0f;
    uint a0 = tid, a1 = tid + tpg; // this thread's accumulator slices

    for (uint t = 0; t < seq_len; ++t) {
        device const half* k = cache + uint(t) * cache_row;
        float partial = 0.0f;
        for (uint i = tid; i < cache_row; i += tpg) {
            partial = fma(float(q_h[i]), float(k[i]), partial);
        }
        // threadgroup reduce: simd first, then across the 8 groups.
        partial = simd_sum(partial);
        if (lane == 0) shared[sg] = partial;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float total = scale * (shared[0] + shared[1] + shared[2] + shared[3]
                             + shared[4] + shared[5] + shared[6] + shared[7]);
        float m_new = max(m, total);
        float correction = exp(m - m_new);
        float p = exp(total - m_new);
        d = d * correction + p;
        acc0 = acc0 * correction + p * float(k[a0]);
        acc1 = acc1 * correction + p * float(k[a1]);
        m = m_new;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    out[tg * kv_lora + a0] = half(acc0 / d);
    out[tg * kv_lora + a1] = half(acc1 / d);
}

// Per-head v-combine over Q8_0 rows: out[h][j] = sum_i W_uv[h][j][i] *
// attn[h][i]. W rows for head h are kv_b's rows
// [h*(nope+v) + nope, h*(nope+v) + nope + v). Same shape as the absorb
// kernel with the reduce running over kv_lora. Contract:
// `compute::mla_v_combine`.
kernel void mla_v_combine_q8_0(
    device const uint8_t* W        [[buffer(0)]],
    device const half*    attn     [[buffer(1)]],
    device half*          out      [[buffer(2)]],
    constant uint&        heads    [[buffer(3)]],
    constant uint&        nope     [[buffer(4)]],
    constant uint&        kv_lora  [[buffer(5)]],
    constant uint&        v_dim    [[buffer(6)]],
    uint2                 tg       [[threadgroup_position_in_grid]],
    uint                  sg       [[simdgroup_index_in_threadgroup]],
    uint                  lane     [[thread_index_in_simdgroup]]
) {
    uint head = tg.y;
    if (head >= heads) return;
    uint row_in_head = tg.x * 8 + sg;
    if (row_in_head >= v_dim) return;
    uint w_row = head * (nope + v_dim) + nope + row_in_head;
    uint n_blocks = kv_lora / kMlaQ8BlockElems;
    device const uint8_t* wr = W + uint(w_row) * n_blocks * kMlaQ8BlockBytes;
    device const half* a_h = attn + head * kv_lora;

    float acc = 0.0f;
    for (uint b = 0; b < n_blocks; ++b) {
        device const uint8_t* blk = wr + b * kMlaQ8BlockBytes;
        ushort raw = ushort(blk[0]) | (ushort(blk[1]) << 8);
        float d = float(as_type<half>(raw));
        float qv = float(int(as_type<int8_t>(blk[2 + lane])));
        acc = fma(d * qv, float(a_h[b * kMlaQ8BlockElems + lane]), acc);
    }
    acc = simd_sum(acc);
    if (lane == 0) {
        out[head * v_dim + row_in_head] = half(acc);
    }
}

// Copy `count` halves from src+offset to dst+offset: the compressed row's
// trip from scratch into its cache slot after norm and rope ran in place.
// A 1.0-scale `scalar_mul` between two buffers, kept here so the MLA flow
// has one obvious thing to point at.
kernel void mla_cache_write(
    device const half* src    [[buffer(0)]],
    device half*       dst    [[buffer(1)]],
    constant uint&     count  [[buffer(2)]],
    uint               gid    [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    dst[gid] = src[gid];
}
