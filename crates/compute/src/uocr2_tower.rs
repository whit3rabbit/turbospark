//! CPU references and host arithmetic for Unlimited-OCR-2's SAM and CLIP
//! towers. `conv2d` and `relpos_attention` are the contracts of the kernels in
//! `crates/gpu/src/shaders/vision.metal`. The two table builders run live: a
//! position table is built once per view size, not per token.

/// CLIP's activation (upstream `quick_gelu`). NOT either GELU the qwen tower
/// uses: `x * sigmoid(1.702 x)` differs from them by up to about 2e-2.
pub fn quick_gelu(x: f32) -> f32 {
    x / (1.0 + (-1.702 * x).exp())
}

/// `[h, w, c_in]` to `[h', w', c_out]`, with `weight` `[out, kh, kw, in]`
/// (MLX layout) and zero padding.
#[allow(clippy::too_many_arguments)]
pub fn conv2d(
    x: &[f32],
    (h, w, c_in): (usize, usize, usize),
    weight: &[f32],
    (c_out, kh, kw): (usize, usize, usize),
    stride: usize,
    pad: usize,
    bias: Option<&[f32]>,
) -> (Vec<f32>, usize, usize) {
    let oh = (h + 2 * pad - kh) / stride + 1;
    let ow = (w + 2 * pad - kw) / stride + 1;
    let mut out = vec![0.0f32; oh * ow * c_out];
    for oy in 0..oh {
        for ox in 0..ow {
            for c in 0..c_out {
                let mut sum = 0.0f32;
                for ky in 0..kh {
                    for kx in 0..kw {
                        let (sy, sx) = (oy * stride + ky, ox * stride + kx);
                        if sy < pad || sx < pad || sy - pad >= h || sx - pad >= w {
                            continue;
                        }
                        let src = ((sy - pad) * w + (sx - pad)) * c_in;
                        let wgt = ((c * kh + ky) * kw + kx) * c_in;
                        for i in 0..c_in {
                            sum += x[src + i] * weight[wgt + i];
                        }
                    }
                }
                out[(oy * ow + ox) * c_out + c] = sum + bias.map_or(0.0, |b| b[c]);
            }
        }
    }
    (out, oh, ow)
}

/// SAM's decomposed relative-position table for one axis of `size`:
/// `[size, size, dim]`, indexed by `(query, key)`. A table of another length is
/// resampled linearly first, half-pixel centred, which is what a 640 view does
/// to a global block's 127 entries.
pub fn rel_pos_table(rel: &[f32], dim: usize, size: usize) -> Vec<f32> {
    let want = 2 * size - 1;
    let have = rel.len() / dim;
    let resampled: Vec<f32> = if have == want {
        rel.to_vec()
    } else {
        let scale = have as f32 / want as f32;
        let mut out = Vec::with_capacity(want * dim);
        for i in 0..want {
            let at = ((i as f32 + 0.5) * scale - 0.5).max(0.0);
            let lo = at.floor() as usize;
            let hi = (lo + 1).min(have - 1);
            let t = at - lo as f32;
            for d in 0..dim {
                out.push(rel[lo * dim + d] * (1.0 - t) + rel[hi * dim + d] * t);
            }
        }
        out
    };
    let mut table = Vec::with_capacity(size * size * dim);
    for q in 0..size {
        for k in 0..size {
            let at = q + size - 1 - k;
            table.extend_from_slice(&resampled[at * dim..(at + 1) * dim]);
        }
    }
    table
}

fn cubic(x: f32) -> f32 {
    const A: f32 = -0.5;
    let x = x.abs();
    if x < 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * A
    } else {
        0.0
    }
}

fn axis_weights(from: usize, to: usize) -> Vec<(usize, Vec<f32>)> {
    let scale = from as f32 / to as f32;
    let fs = scale.max(1.0);
    let support = 2.0 * fs;
    (0..to)
        .map(|i| {
            let center = (i as f32 + 0.5) * scale;
            let lo = ((center - support + 0.5).floor().max(0.0)) as usize;
            let hi = ((center + support + 0.5).floor() as usize).min(from);
            let mut w: Vec<f32> = (lo..hi)
                .map(|k| cubic((k as f32 + 0.5 - center) / fs))
                .collect();
            let total: f32 = w.iter().sum();
            w.iter_mut().for_each(|v| *v /= total);
            (lo, w)
        })
        .collect()
}

/// Antialiased bicubic of a `[side, side, dim]` position table, Pillow's kernel
/// (`a = -0.5`), which is `F.interpolate(..., antialias=True)` and upstream's
/// `bicubic_interpolate`. Width, then height, as Pillow runs them.
pub fn bicubic_resize_antialias(src: &[f32], side: usize, dim: usize, to: usize) -> Vec<f32> {
    if side == to {
        return src.to_vec();
    }
    let weights = axis_weights(side, to);
    let mut horizontal = vec![0.0f32; side * to * dim];
    for y in 0..side {
        for (x, (lo, w)) in weights.iter().enumerate() {
            for (j, wt) in w.iter().enumerate() {
                let at = (y * side + lo + j) * dim;
                for d in 0..dim {
                    horizontal[(y * to + x) * dim + d] += wt * src[at + d];
                }
            }
        }
    }
    let mut out = vec![0.0f32; to * to * dim];
    for (y, (lo, w)) in weights.iter().enumerate() {
        for x in 0..to {
            for (j, wt) in w.iter().enumerate() {
                let at = ((lo + j) * to + x) * dim;
                for d in 0..dim {
                    out[(y * to + x) * dim + d] += wt * horizontal[at + d];
                }
            }
        }
    }
    out
}

/// Windowed attention with SAM's decomposed bias, each window `side * side`
/// tokens, `q`/`k`/`v`/`out` `[seq, heads, head_dim]`, and `rel_h`/`rel_w` from
/// [`rel_pos_table`]. The bias is `q` dotted with the table, NOT scaled, added
/// to the scaled score.
#[allow(clippy::too_many_arguments)]
pub fn relpos_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    rel_h: &[f32],
    rel_w: &[f32],
    side: usize,
    heads: usize,
    head_dim: usize,
    scale: f32,
) -> Vec<f32> {
    let window = side * side;
    let seq = q.len() / (heads * head_dim);
    let mut out = vec![0.0f32; q.len()];
    let row = |t: usize, head: usize| (t * heads + head) * head_dim;
    for t in 0..seq {
        let (win, local) = (t / window, t % window);
        let (qh, qw) = (local / side, local % side);
        for head in 0..heads {
            let qr = &q[row(t, head)..row(t, head) + head_dim];
            let scores: Vec<f32> = (0..window)
                .map(|j| {
                    let (kh, kw) = (j / side, j % side);
                    let kr = row(win * window + j, head);
                    let rh = (qh * side + kh) * head_dim;
                    let rw = (qw * side + kw) * head_dim;
                    let (mut dot, mut bias) = (0.0f32, 0.0f32);
                    for d in 0..head_dim {
                        dot += qr[d] * k[kr + d];
                        bias += qr[d] * (rel_h[rh + d] + rel_w[rw + d]);
                    }
                    dot * scale + bias
                })
                .collect();
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let weights: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
            let total: f32 = weights.iter().sum();
            for (j, wgt) in weights.iter().enumerate() {
                let vr = row(win * window + j, head);
                for d in 0..head_dim {
                    out[row(t, head) + d] += (wgt / total) * v[vr + d];
                }
            }
        }
    }
    out
}
