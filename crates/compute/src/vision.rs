//! FP32 reference kernels for the `qwen3_5` vision tower (ROADMAP M-V2).
//!
//! Ground truth for `crates/gpu`'s vision Metal kernels, in the same
//! relationship every other module here has to its GPU sibling. Port-local:
//! the Swift engine has no vision tower, so there is no upstream kernel to
//! diff against and these functions are the only definition of what the
//! Metal ones compute.
//!
//! The tower's exact shape is read off `mlx_vlm/models/qwen3_vl/vision.py`
//! and recorded in `docs/VISION_PHASE0.md`: 27 blocks at hidden 1152, 16
//! heads of 72, intermediate 4304, out 5120, `spatial_merge_size` 2.
//!
//! # Two things here that look interchangeable and are not
//!
//! **LayerNorm, not RMSNorm.** Every norm in this tower is
//! `nn.LayerNorm(eps=1e-6)`: it subtracts the mean and adds a bias, where
//! every other family in this workspace uses RMSNorm and does neither.
//! Reaching for [`crate::rms_norm`] here produces finite, plausible output.
//!
//! **Two different GELUs in one forward pass.** The per-block MLP is
//! `nn.GELU(approx="tanh")` and the merger is a bare `nn.GELU()`, which is
//! the EXACT erf form. See [`gelu_erf`].

use crate::moe::gelu_tanh;

/// `y[i] = (x[i] - mean) / sqrt(var + eps) * weight[i] + bias[i]`.
///
/// The tower's `norm1`, `norm2` and the merger's `norm`, all at `eps = 1e-6`.
///
/// The variance is the BIASED one (divide by `d`, not `d - 1`), matching
/// every framework's LayerNorm. Using the unbiased estimator instead is a
/// factor of `d/(d-1)` on the scale, which at `d = 1152` is 0.09% -- far too
/// small to see in generated text and far too large to be right.
pub fn layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    assert_eq!(x.len(), weight.len(), "x and weight must match length");
    assert_eq!(x.len(), bias.len(), "x and bias must match length");
    let d = x.len();
    let mean: f32 = x.iter().sum::<f32>() / d as f32;
    let var: f32 = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;
    let inv_std = 1.0 / (var + eps).sqrt();
    x.iter()
        .zip(weight)
        .zip(bias)
        .map(|((xv, wv), bv)| (xv - mean) * inv_std * wv + bv)
        .collect()
}

/// The EXACT GELU, `0.5 * x * (1 + erf(x / sqrt(2)))`.
///
/// The merger's activation. Its sibling [`crate::gelu_tanh`] is the tanh
/// APPROXIMATION, which is what the per-block MLP uses, and the two are
/// different functions living in one forward pass -- so "which GELU" is a
/// property of the call site rather than of the tower.
///
/// The two agree to about 0.0003 absolute at their worst (near |x| = 2) and
/// to far less everywhere else, which is the problem: swapping them produces
/// output that is wrong by an amount no coherence check can see and that a
/// loose parity tolerance absorbs. `the_two_gelus_are_different_functions`
/// pins the gap so a fixture cannot be chosen where it vanishes.
///
/// `erf` is not in Rust's standard library, so it is computed here by
/// Abramowitz and Stegun 7.1.26, whose stated bound is 1.5e-7 absolute --
/// below f32's own resolution on a result in `[-1, 1]`.
pub fn gelu_erf(x: &[f32]) -> Vec<f32> {
    x.iter()
        .map(|&xv| {
            let z = xv as f64 * std::f64::consts::FRAC_1_SQRT_2;
            (0.5 * xv as f64 * (1.0 + erf(z))) as f32
        })
        .collect()
}

/// Abramowitz and Stegun 7.1.26, whose stated bound is 1.5e-7 absolute.
///
/// Odd, so the negative half is handled by symmetry rather than by a second
/// set of coefficients.
///
/// **In f64 rather than f32, and the coefficients are the published ones
/// unrounded.** Two reasons, and neither is speed. A&S's error bound is
/// 1.5e-7, which is barely under f32's own 6e-8 resolution near 1.0, so
/// evaluating the series in f32 puts the approximation error and the
/// arithmetic error at the same order and this reference stops being ground
/// truth. And truncating the constants to what f32 can hold (which is what
/// `clippy::excessive_precision` asks for) severs them from the published
/// table, so a reader can no longer check them against the source.
fn erf(x: f64) -> f64 {
    const A1: f64 = 0.254_829_592;
    const A2: f64 = -0.284_496_736;
    const A3: f64 = 1.421_413_741;
    const A4: f64 = -1.453_152_027;
    const A5: f64 = 1.061_405_429;
    const P: f64 = 0.327_591_1;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + P * x);
    let y = 1.0 - (((((A5 * t + A4) * t) + A3) * t + A2) * t + A1) * t * (-x * x).exp();
    sign * y
}

/// The tanh-approximation GELU, re-exported so a vision call site names its
/// activation from this module rather than from `moe`.
pub fn gelu_tanh_vision(x: &[f32]) -> Vec<f32> {
    gelu_tanh(x)
}

/// Apply the tower's 2-D rotary embedding to one token's head.
///
/// `head` is `head_dim` values; `freq_row` is `head_dim / 2` angles, the
/// height half followed by the width half, as
/// `turbospark_vision_io::vision_rope_freq_rows` emits them.
///
/// # Half-split pairs, and a freq ROW rather than a position
///
/// The rotation pairs element `i` with element `i + head_dim/2` -- the NeoX
/// half-split, not adjacent pairs -- and takes its angle from `freq_row[i]`.
/// The reference builds this by tiling the row to full width
/// (`mx.tile(cos, (1, 1, 2))`) and doing `x * cos + rotate_half(x) * sin`,
/// which is the same function written as a broadcast.
///
/// What differs from every other rope in this workspace is the SOURCE of the
/// angle. `rope_neox` and friends derive it from a scalar position and a
/// theta; here each token carries its own 36-wide row, because a patch has
/// a two-dimensional position and no single scalar encodes it. That is the
/// whole of the "vision" in this kernel -- the arithmetic below is ordinary.
pub fn rope_vision_2d(head: &[f32], freq_row: &[f32]) -> Vec<f32> {
    let d = head.len();
    assert!(d % 2 == 0, "head_dim must be even");
    let half = d / 2;
    assert_eq!(freq_row.len(), half, "freq row must be head_dim / 2 wide");

    let mut out = vec![0.0f32; d];
    for i in 0..half {
        let (cos, sin) = (freq_row[i].cos(), freq_row[i].sin());
        let (a, b) = (head[i], head[i + half]);
        out[i] = a * cos - b * sin;
        out[i + half] = b * cos + a * sin;
    }
    out
}

/// Bidirectional multi-head attention over one image's patches.
///
/// `q`, `k`, `v` are each `[seq, heads, head_dim]` row-major. Output is the
/// same shape, flattened.
///
/// # No mask, no cache, and that is the point
///
/// Every other attention in this workspace is CAUSAL and reads a KV cache:
/// it answers "what may position `t` see". A vision tower has neither. All
/// `seq` patches attend to all `seq` patches, the whole `seq x seq` matrix
/// is live, and nothing persists between images. Reusing
/// [`crate::causal_attention`] here would silently make the top-left patch
/// blind to everything below and right of it, which degrades an image rather
/// than corrupting it -- the failure mode no smoke test catches.
///
/// The reference splits on `cu_seqlens` so that a BATCH of images does not
/// attend across the boundary between them. One image is one span, which is
/// what this takes; a caller batching images calls it once per image.
pub fn bidirectional_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq: usize,
    heads: usize,
    head_dim: usize,
    scale: f32,
) -> Vec<f32> {
    let expect = seq * heads * head_dim;
    for (name, buf) in [("q", q), ("k", k), ("v", v)] {
        assert_eq!(buf.len(), expect, "{name} must be seq * heads * head_dim");
    }

    let mut out = vec![0.0f32; expect];
    let mut scores = vec![0.0f32; seq];
    for h in 0..heads {
        for i in 0..seq {
            let qi = (i * heads + h) * head_dim;
            // Softmax over every key, max-subtracted. The max is taken over
            // the real scores rather than seeded at zero: this tower's
            // activations reach into the hundreds by block 9 and the
            // thousands by block 26 (`docs/VISION_PHASE0.md` item 3), so an
            // unshifted exp overflows on a real page rather than on a
            // contrived one.
            let mut max = f32::NEG_INFINITY;
            for (j, slot) in scores.iter_mut().enumerate() {
                let kj = (j * heads + h) * head_dim;
                let dot: f32 = (0..head_dim).map(|d| q[qi + d] * k[kj + d]).sum();
                let s = dot * scale;
                *slot = s;
                if s > max {
                    max = s;
                }
            }
            let mut denom = 0.0f32;
            for slot in scores.iter_mut() {
                *slot = (*slot - max).exp();
                denom += *slot;
            }
            let inv = 1.0 / denom;
            let oi = (i * heads + h) * head_dim;
            for (j, &w) in scores.iter().enumerate() {
                let vj = (j * heads + h) * head_dim;
                let w = w * inv;
                for d in 0..head_dim {
                    out[oi + d] += w * v[vj + d];
                }
            }
        }
    }
    out
}

/// `out[m][n] = sum_k a[m][k] * b[n][k] + bias[n]`.
///
/// The reference for every GEMM in the tower: qkv, proj, fc1, fc2,
/// patch_embed and both merger projections. `b` is stored ROW-MAJOR BY
/// OUTPUT (`[n_out, k]`), which is how a `nn.Linear` weight ships and how
/// this port's resident index reads it, so the inner loop walks one output's
/// whole row contiguously and no transpose happens anywhere.
///
/// `bias` is optional because `merger.linear_fc1`/`fc2` and the block's
/// `proj` carry one while a caller building a fixture often does not.
pub fn matmul_bias(
    a: &[f32],
    b: &[f32],
    bias: Option<&[f32]>,
    m: usize,
    k: usize,
    n: usize,
) -> Vec<f32> {
    assert_eq!(a.len(), m * k, "a must be m * k");
    assert_eq!(b.len(), n * k, "b must be n * k (row-major by output)");
    if let Some(bias) = bias {
        assert_eq!(bias.len(), n, "bias must be n");
    }

    let mut out = vec![0.0f32; m * n];
    for row in 0..m {
        let a_row = &a[row * k..(row + 1) * k];
        for col in 0..n {
            let b_row = &b[col * k..(col + 1) * k];
            let dot: f32 = a_row.iter().zip(b_row).map(|(x, y)| x * y).sum();
            out[row * n + col] = dot + bias.map_or(0.0, |v| v[col]);
        }
    }
    out
}

/// The reciprocal-square-root attention scale, `head_dim^-0.5`.
pub fn attention_scale(head_dim: usize) -> f32 {
    (head_dim as f32).powf(-0.5)
}
