//! CPU references for the `deepseek2` MLA kernels in
//! `crates/gpu/src/shaders/mla.metal`. Port-local math with no upstream
//! sibling: these functions ARE the kernels' contract, the same role
//! `dequant_q8_0_gemv` plays for its shader.
//!
//! The absorbed form (see `docs/DEEPSEEK2_PHASE0.md`): per head, the q
//! nope half is folded through the up-projection so the score against a
//! cached token is ONE dot over the compressed row `[c ; k_pe]`, V is read
//! as the row's first `kv_lora` elements, and a second per-head projection
//! maps the 512-wide attention back down to `v_head_dim`.

/// RMS-norm the first `rank` elements of a fused `[rank ; tail]` row and
/// pass the tail through untouched. The reference for `mla_kv_norm`, whose
/// real tail is the rope-carried key the rope kernel then rotates in place.
#[must_use]
pub fn mla_kv_norm(kv_a: &[f32], rank: usize, weight: &[f32], eps: f32) -> Vec<f32> {
    assert!(
        rank > 0 && kv_a.len() >= rank,
        "row must carry at least one rank"
    );
    assert_eq!(weight.len(), rank, "weight must be rank-wide");
    let mut out = kv_a.to_vec();
    // `chunks_exact` leaves any trailing window as the tail it is: the
    // norm runs on whole `[rank]` latents only, and a row that stops short
    // of another rank has nothing more to normalize.
    for chunk in out.chunks_exact_mut(rank) {
        let sum: f32 = chunk.iter().map(|v| v * v).sum();
        let inv = 1.0 / (sum / rank as f32 + eps).sqrt();
        for (v, w) in chunk.iter_mut().zip(weight.iter()) {
            *v *= inv * w;
        }
    }
    out
}

/// Rotate the `rotary_dim`-wide window that starts at `window_offset`
/// inside each `head_dim`-wide head, CONSECUTIVE-element pairs:
/// `(window_offset + 2*i, window_offset + 2*i + 1)` takes angle
/// `position * freqs[i]`, both elements scaled by `mscale`. That pairing is
/// ggml's own (`ggml_rope_cache_init` steps `i0` by 2 and fills
/// `cache[i0]`/`cache[i0+1]`), and it was settled EMPIRICALLY, not from
/// prose: an earlier draft used the half-split `(i, i + dim/2)` pairing the
/// port's other rope kernels use; position 0 still matched (angles are all
/// zero there, so the pairing is invisible) and every row past it degraded
/// smoothly with position -- pair 0's extrapolated YaRN frequency is
/// 1.0 rad/position, so the two conventions differ by a full radian at
/// position 1 already. The llama.cpp per-layer dump matched this function's
/// consecutive form at corr 1.0000 (`docs/DEEPSEEK2_PHASE0.md`).
#[allow(clippy::too_many_arguments)]
pub fn mla_rope_window(
    data: &mut [f32],
    num_heads: usize,
    head_dim: usize,
    window_offset: usize,
    rotary_dim: usize,
    freqs: &[f32],
    position: f32,
    mscale: f32,
) {
    assert!(rotary_dim % 2 == 0);
    assert_eq!(freqs.len(), rotary_dim / 2);
    assert!(window_offset + rotary_dim <= head_dim);
    for h in 0..num_heads {
        let head = &mut data[h * head_dim..][..head_dim];
        for (pair, &freq) in freqs.iter().enumerate() {
            let angle = position * freq;
            let (s, c) = angle.sin_cos();
            let lo = window_offset + 2 * pair;
            let hi = lo + 1;
            let (a, b) = (head[lo], head[hi]);
            head[lo] = (a * c - b * s) * mscale;
            head[hi] = (a * s + b * c) * mscale;
        }
    }
}

/// The absorbed query: `q'_h = W_uk_h^T @ q_nope_h` for every head at once.
/// **THE TRANSPOSE IS THE WHOLE POINT, and dropping it produces a function
/// of the same shape with fluent-garbage output** (the one bug a real
/// install caught that a self-consistent fixture could not).
/// `w_uk[h]` is head h's nope half of the kv_b up-projection, DEQUANTIZED
/// to `[nope][kv_lora]` row-major -- row i is kv_b's row
/// `h * (nope + v) + i`, a `kv_lora`-wide slice of the latent -- and the
/// sum runs over i, the NOPE axis, with the output index j addressing INTO
/// each row.
#[must_use]
pub fn mla_absorb_q(
    q_nope: &[f32],
    w_uk: &[&[f32]],
    num_heads: usize,
    kv_lora: usize,
    nope: usize,
) -> Vec<f32> {
    assert_eq!(q_nope.len(), num_heads * nope);
    let mut out = vec![0.0f32; num_heads * kv_lora];
    for h in 0..num_heads {
        let w: &[f32] = w_uk[h];
        assert_eq!(w.len(), nope * kv_lora);
        for (i, qv) in q_nope[h * nope..][..nope].iter().enumerate() {
            let row = &w[i * kv_lora..][..kv_lora];
            for (j, o) in out[h * kv_lora..][..kv_lora].iter_mut().enumerate() {
                *o += row[j] * qv;
            }
        }
    }
    out
}

/// MQA attention over the compressed rows, the reference for
/// `mla_attention_decode`. `q` is `[heads][kv_lora + rope]` (the absorbed
/// nope half and the roped pe half concatenated per head); `k_rows` is the
/// cache, one `[c ; k_pe]` row per token with V read as its first
/// `kv_lora` elements. Two passes so the reference is obviously right:
/// scores first, then the softmax-weighted sum.
#[must_use]
pub fn mla_attention_decode(
    q: &[f32],
    k_rows: &[&[f32]],
    num_heads: usize,
    cache_row: usize,
    kv_lora: usize,
    scale: f32,
) -> Vec<f32> {
    assert!(kv_lora > 0 && cache_row > kv_lora);
    assert_eq!(q.len(), num_heads * cache_row);
    let seq = k_rows.len();
    let _ = seq;
    let mut out = vec![0.0f32; num_heads * kv_lora];
    for h in 0..num_heads {
        let q_h = &q[h * cache_row..][..cache_row];
        let scores: Vec<f32> = k_rows
            .iter()
            .map(|k| {
                assert_eq!(k.len(), cache_row);
                q_h.iter()
                    .zip(k.iter())
                    .map(|(qv, kv)| qv * kv)
                    .sum::<f32>()
                    * scale
            })
            .collect();
        let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let denom: f32 = scores.iter().map(|s| (s - max).exp()).sum();
        for (t, s) in scores.iter().enumerate() {
            let p = (s - max).exp() / denom;
            let c = &k_rows[t][..kv_lora];
            for (o, cv) in out[h * kv_lora..][..kv_lora].iter_mut().zip(c.iter()) {
                *o += p * cv;
            }
        }
    }
    out
}

/// The v-combine: `out_h = W_uv_h @ attn_h`, mapping the shared 512-wide
/// attention back to head h's `v_head_dim`-wide slice. `w_uv[h]` is the v
/// half of kv_b for head h, dequantized to `[v_head_dim][kv_lora]`.
#[must_use]
pub fn mla_v_combine(
    attn: &[f32],
    w_uv: &[&[f32]],
    num_heads: usize,
    kv_lora: usize,
    v_head_dim: usize,
) -> Vec<f32> {
    assert_eq!(attn.len(), num_heads * kv_lora);
    let mut out = vec![0.0f32; num_heads * v_head_dim];
    for h in 0..num_heads {
        let w: &[f32] = w_uv[h];
        assert_eq!(w.len(), v_head_dim * kv_lora);
        for j in 0..v_head_dim {
            let row = &w[j * kv_lora..][..kv_lora];
            let a = &attn[h * kv_lora..][..kv_lora];
            out[h * v_head_dim + j] = row.iter().zip(a.iter()).map(|(wv, av)| wv * av).sum();
        }
    }
    out
}
