//! FP32 reference for the Gemma 4 vision tower and its projector.
//!
//! Ground truth for the Metal tower, held to mlx-vlm by
//! `tests/gemma4_vision_reference.rs`. The facts and the traps are in
//! `docs/GEMMA4_VISION_PHASE0.md`. The shapes match the qwen3_5 tower in
//! [`crate::vision`] and almost nothing else does, so do not reach for that
//! module's norms or rope here:
//!
//! - every norm is RMSNorm with the weight applied directly ([`crate::rms_norm`]),
//!   and the block is sandwich-normed around both attention and the MLP;
//! - q and k are RMS-normed per head with a learned weight, v per head with
//!   NO weight, and attention runs at scale 1.0;
//! - the MLP is gated, `down(gelu_tanh(gate(x)) * up(x))`, no biases;
//! - the rope is axial: the first half of each head turns by the patch's x,
//!   the second half by its y, each half rotated within itself.

use crate::rms_norm::rms_norm;
use crate::vision::{bidirectional_attention, gelu_tanh_vision, matmul_bias};

/// Tower geometry. The real checkpoint's values are in the Phase 0 doc.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Gemma4VisionDims {
    pub hidden: usize,
    pub intermediate: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub patch_dim: usize,
    pub pos_table_rows: usize,
    pub pool: usize,
    pub text_hidden: usize,
    pub rope_theta: f32,
    pub eps: f32,
}

/// One block's weights, each `nn.Linear` row-major by output.
#[derive(Debug, Clone, Copy)]
pub struct Gemma4VisionBlock<'a> {
    pub ln_in: &'a [f32],
    pub q: &'a [f32],
    pub k: &'a [f32],
    pub v: &'a [f32],
    pub q_norm: &'a [f32],
    pub k_norm: &'a [f32],
    pub o: &'a [f32],
    pub ln_post_attn: &'a [f32],
    pub ln_pre_ffw: &'a [f32],
    pub gate: &'a [f32],
    pub up: &'a [f32],
    pub down: &'a [f32],
    pub ln_post_ffw: &'a [f32],
}

/// The whole tower plus the `embed_vision` projector.
#[derive(Debug, Clone)]
pub struct Gemma4VisionWeights<'a> {
    /// `[hidden, patch_dim]`.
    pub input_proj: &'a [f32],
    /// `[2, pos_table_rows, hidden]`: the x table, then the y table.
    pub pos_table: &'a [f32],
    pub blocks: Vec<Gemma4VisionBlock<'a>>,
    pub std_bias: &'a [f32],
    pub std_scale: &'a [f32],
    /// `[text_hidden, hidden]`.
    pub proj: &'a [f32],
}

/// Every stage's rows, for a parity test that has to say WHERE it diverged.
#[derive(Debug, Clone, PartialEq)]
pub struct Gemma4VisionStages {
    /// `[patches, hidden]`.
    pub patch_embed: Vec<f32>,
    /// `[patches, hidden]` after each block.
    pub blocks: Vec<Vec<f32>>,
    /// `[soft, hidden]`: pooled, scaled by `sqrt(hidden)`, standardized.
    pub tower: Vec<f32>,
    /// `[soft, text_hidden]`: what replaces the trunk's embedding rows.
    pub embedded: Vec<f32>,
}

/// RMSNorm each `dim`-wide row. `weight` `None` is the scale-free form.
fn rms_rows(x: &[f32], weight: Option<&[f32]>, dim: usize, eps: f32) -> Vec<f32> {
    let ones;
    let w = match weight {
        Some(w) => w,
        None => {
            ones = vec![1.0f32; dim];
            &ones
        }
    };
    x.chunks_exact(dim)
        .flat_map(|row| rms_norm(row, w, eps))
        .collect()
}

/// Rotate one head in place by an axial 2-D position.
///
/// The head splits into two parts of `head_dim / 2`: part 0 turns by `x`,
/// part 1 by `y`. Inside a part, element `j` pairs with `j + part/2`
/// (`rotate_half`), at inverse frequency `theta^(-2i / part)` for
/// `i = j mod part/2`. This is `apply_multidimensional_rope` in both
/// references, which split by the LAST axis of the position, x first.
pub fn axial_rope_2d(head: &mut [f32], x: usize, y: usize, theta: f32) {
    let part = head.len() / 2;
    assert_eq!(part * 2, head.len(), "head_dim must be even");
    assert_eq!(part % 2, 0, "head_dim must be a multiple of 4");
    let half = part / 2;
    for (d, pos) in [x, y].into_iter().enumerate() {
        let seg = &mut head[d * part..(d + 1) * part];
        for i in 0..half {
            let inv = theta.powf(-(2.0 * i as f32) / part as f32);
            let (sin, cos) = (pos as f32 * inv).sin_cos();
            let (a, b) = (seg[i], seg[i + half]);
            seg[i] = a * cos - b * sin;
            seg[i + half] = b * cos + a * sin;
        }
    }
}

/// Average each `pool x pool` block of a row-major `grid_w x grid_h` patch
/// grid into one row, soft tokens row-major over the blocks.
pub fn pool_by_position(
    rows: &[f32],
    positions: &[(usize, usize)],
    grid_w: usize,
    hidden: usize,
    pool: usize,
) -> Vec<f32> {
    let per_row = grid_w / pool;
    let soft = positions.len() / (pool * pool);
    let mut out = vec![0.0f32; soft * hidden];
    let inv = 1.0 / (pool * pool) as f32;
    for (i, &(x, y)) in positions.iter().enumerate() {
        let t = x / pool + per_row * (y / pool);
        for d in 0..hidden {
            out[t * hidden + d] += rows[i * hidden + d] * inv;
        }
    }
    out
}

/// One block over `[seq, hidden]` rows.
pub fn gemma4_vision_block(
    h: &[f32],
    positions: &[(usize, usize)],
    w: &Gemma4VisionBlock<'_>,
    dims: &Gemma4VisionDims,
) -> Vec<f32> {
    let (hid, hd, nh) = (dims.hidden, dims.head_dim, dims.heads);
    let seq = positions.len();
    let a = rms_rows(h, Some(w.ln_in), hid, dims.eps);
    let mut q = rms_rows(
        &matmul_bias(&a, w.q, None, seq, hid, nh * hd),
        Some(w.q_norm),
        hd,
        dims.eps,
    );
    let mut k = rms_rows(
        &matmul_bias(&a, w.k, None, seq, hid, nh * hd),
        Some(w.k_norm),
        hd,
        dims.eps,
    );
    let v = rms_rows(
        &matmul_bias(&a, w.v, None, seq, hid, nh * hd),
        None,
        hd,
        dims.eps,
    );
    for (t, &(x, y)) in positions.iter().enumerate() {
        for head in 0..nh {
            let at = (t * nh + head) * hd;
            axial_rope_2d(&mut q[at..at + hd], x, y, dims.rope_theta);
            axial_rope_2d(&mut k[at..at + hd], x, y, dims.rope_theta);
        }
    }
    let attn = bidirectional_attention(&q, &k, &v, seq, nh, hd, 1.0);
    let o = rms_rows(
        &matmul_bias(&attn, w.o, None, seq, nh * hd, hid),
        Some(w.ln_post_attn),
        hid,
        dims.eps,
    );
    let h1: Vec<f32> = h.iter().zip(&o).map(|(a, b)| a + b).collect();

    let m = rms_rows(&h1, Some(w.ln_pre_ffw), hid, dims.eps);
    let inter = dims.intermediate;
    let gate = gelu_tanh_vision(&matmul_bias(&m, w.gate, None, seq, hid, inter));
    let up = matmul_bias(&m, w.up, None, seq, hid, inter);
    let act: Vec<f32> = gate.iter().zip(&up).map(|(g, u)| g * u).collect();
    let down = rms_rows(
        &matmul_bias(&act, w.down, None, seq, inter, hid),
        Some(w.ln_post_ffw),
        hid,
        dims.eps,
    );
    h1.iter().zip(&down).map(|(a, b)| a + b).collect()
}

/// The tower and projector over the REAL patches of one image.
///
/// `pixels` are `[patches, patch_dim]` in `[0, 1]`, `positions` `(x, y)`
/// row-major over a `grid_w`-wide grid whose sides are multiples of `pool`.
/// Padding patches are not taken: the reference masks them as keys, zeroes
/// them before pooling, and drops their soft tokens, so running without them
/// is the same function.
pub fn gemma4_vision_forward(
    pixels: &[f32],
    positions: &[(usize, usize)],
    grid_w: usize,
    w: &Gemma4VisionWeights<'_>,
    dims: &Gemma4VisionDims,
) -> Gemma4VisionStages {
    let (hid, n) = (dims.hidden, positions.len());
    assert_eq!(
        pixels.len(),
        n * dims.patch_dim,
        "pixels must be [n, patch_dim]"
    );
    assert_eq!(
        n % (dims.pool * dims.pool),
        0,
        "patches must fill whole pool blocks"
    );

    // `2 * (x - 0.5)` happens in model code, not in the processor.
    let scaled: Vec<f32> = pixels.iter().map(|p| 2.0 * (p - 0.5)).collect();
    let mut h = matmul_bias(&scaled, w.input_proj, None, n, dims.patch_dim, hid);
    let table = dims.pos_table_rows * hid;
    for (t, &(x, y)) in positions.iter().enumerate() {
        assert!(
            x < dims.pos_table_rows && y < dims.pos_table_rows,
            "position past the table"
        );
        for d in 0..hid {
            h[t * hid + d] += w.pos_table[x * hid + d] + w.pos_table[table + y * hid + d];
        }
    }
    let patch_embed = h.clone();

    let mut blocks = Vec::with_capacity(w.blocks.len());
    for blk in &w.blocks {
        h = gemma4_vision_block(&h, positions, blk, dims);
        blocks.push(h.clone());
    }

    // FP32 throughout: `sqrt(hidden)` pushes these past FP16's range before
    // the standardize brings them back (Phase 0 trap 5).
    let root = (hid as f32).sqrt();
    let pooled = pool_by_position(&h, positions, grid_w, hid, dims.pool);
    let tower: Vec<f32> = pooled
        .chunks_exact(hid)
        .flat_map(|row| {
            row.iter()
                .enumerate()
                .map(|(d, v)| (v * root - w.std_bias[d]) * w.std_scale[d])
        })
        .collect();

    let soft = tower.len() / hid;
    let normed = rms_rows(&tower, None, hid, dims.eps);
    let embedded = matmul_bias(&normed, w.proj, None, soft, hid, dims.text_hidden);
    Gemma4VisionStages {
        patch_embed,
        blocks,
        tower,
        embedded,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn axial_rope_at_the_origin_is_the_identity() {
        let mut head: Vec<f32> = (0..8).map(|i| i as f32 - 3.5).collect();
        let before = head.clone();
        axial_rope_2d(&mut head, 0, 0, 100.0);
        assert_eq!(head, before);
    }

    #[test]
    fn axial_rope_turns_only_the_half_its_coordinate_owns() {
        let base: Vec<f32> = (0..8).map(|i| 1.0 + i as f32).collect();
        let mut by_x = base.clone();
        axial_rope_2d(&mut by_x, 3, 0, 100.0);
        assert_ne!(by_x[..4], base[..4]);
        assert_eq!(by_x[4..], base[4..]);
        let mut by_y = base.clone();
        axial_rope_2d(&mut by_y, 0, 3, 100.0);
        assert_eq!(by_y[..4], base[..4]);
        assert_ne!(by_y[4..], base[4..]);
    }

    #[test]
    fn pooling_averages_each_3x3_block_into_its_own_row() {
        // A 6x3 grid, two soft tokens; each patch's row is its x index.
        let positions: Vec<(usize, usize)> =
            (0..3).flat_map(|y| (0..6).map(move |x| (x, y))).collect();
        let rows: Vec<f32> = positions.iter().map(|&(x, _)| x as f32).collect();
        let out = pool_by_position(&rows, &positions, 6, 1, 3);
        assert_eq!(out, vec![1.0, 4.0]);
    }
}
