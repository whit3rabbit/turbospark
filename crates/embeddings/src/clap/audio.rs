//! HTSAT audio encoder (`ClapAudioEncoder`, unfused).
//!
//! Reference: transformers 4.57.1 `modeling_clap.py`:
//!
//! 1. `BatchNorm2d(num_mel_bins)` over the mel axis (eval statistics).
//! 2. `reshape_mel2img`: bicubic (`align_corners=True`) stretch of the time
//!    axis to `spec_width = spec_size * freq_ratio`, then fold the time
//!    axis into `freq_ratio` chunks stacked along frequency, giving a
//!    `spec_size` square image.
//! 3. Patch embed (`Conv2d` kernel = stride = patch) and LayerNorm.
//! 4. Swin stages: window attention with relative position bias, shifted
//!    windows (cyclic roll by `-window/2` plus the -100 region mask) on odd
//!    blocks, patch merging between stages. A stage whose resolution
//!    equals the window runs unshifted (`set_shift_and_window_size`).
//! 5. Final LayerNorm, then the mean over all tokens. The reference's
//!    tscam regrouping before `AdaptiveAvgPool1d(1)` only permutes tokens,
//!    so the pooled output is the plain token mean.
//!
//! Tokens are row-major `[h * W + w, channels]`, image rows being the
//! folded frequency axis and columns time, as in the reference's
//! `flatten(2).transpose(1, 2)`.

use turbospark_audio::nn::{LayerNorm, Linear};
use turbospark_audio::ops;

use super::config::{AudioConfig, HtsatGeometry, StageGeometry};
use crate::error::input_error;
use crate::weights::Weights;
use crate::Result;

/// Default `nn.LayerNorm` / `nn.BatchNorm2d` eps, used where the reference
/// does not pass `config.layer_norm_eps` (patch-embed norm, patch-merging
/// norm, final norm, batch norm).
const TORCH_EPS: f32 = 1e-5;
/// Additive value the reference writes for cross-region pairs in the
/// shifted-window mask (not -inf).
const SHIFT_MASK: f32 = -100.0;

pub(crate) fn layer_norm(w: &Weights, base: &str, width: usize, eps: f32) -> Result<LayerNorm> {
    Ok(LayerNorm::new(
        w.load(&format!("{base}.weight"), &[width])?,
        Some(w.load(&format!("{base}.bias"), &[width])?),
        eps,
    ))
}

struct SwinBlock {
    ln_before: LayerNorm,
    query: Linear,
    key: Linear,
    value: Linear,
    attn_out: Linear,
    ln_after: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    /// `[heads, t, t]` relative position bias, gathered once at load.
    rel_bias: Vec<f32>,
    shifted: bool,
}

struct PatchMerging {
    norm: LayerNorm,
    reduction: Linear,
}

struct Stage {
    geom: StageGeometry,
    blocks: Vec<SwinBlock>,
    downsample: Option<PatchMerging>,
    /// Token index for each windowed row, unshifted and shifted.
    gather_plain: Vec<usize>,
    gather_shift: Vec<usize>,
    /// `[windows, t, t]` additive mask for shifted blocks.
    shift_mask: Vec<f32>,
}

/// The HTSAT encoder up to the pooled `[hidden_size]` feature.
pub struct HtsatEncoder {
    geom: HtsatGeometry,
    mels: usize,
    bn_alpha: Vec<f32>,
    bn_beta: Vec<f32>,
    patch_w: Vec<f32>,
    patch_b: Vec<f32>,
    patch_norm: Option<LayerNorm>,
    stages: Vec<Stage>,
    norm: LayerNorm,
}

impl std::fmt::Debug for HtsatEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HtsatEncoder")
            .field("geometry", &self.geom)
            .finish_non_exhaustive()
    }
}

/// `relative_position_index` of `ClapAudioSelfAttention`, expanded into a
/// `[heads, t, t]` bias from the `[(2w-1)^2, heads]` table.
fn relative_bias(table: &[f32], window: usize, heads: usize) -> Vec<f32> {
    let t = window * window;
    let span = 2 * window - 1;
    let mut out = vec![0.0f32; heads * t * t];
    for i in 0..t {
        let (ih, iw) = (i / window, i % window);
        for j in 0..t {
            let (jh, jw) = (j / window, j % window);
            let idx = (ih + window - 1 - jh) * span + (iw + window - 1 - jw);
            for h in 0..heads {
                out[(h * t + i) * t + j] = table[idx * heads + h];
            }
        }
    }
    out
}

/// Windowed row -> token index for a `res x res` grid after a cyclic roll
/// by `-shift` (`torch.roll(x, (-s, -s))`: shifted[r] = x[(r + s) % res]).
/// Window order is `window_partition`'s: row blocks, then column blocks.
fn gather_index(res: usize, window: usize, shift: usize) -> Vec<usize> {
    let nw = res / window;
    let mut out = Vec::with_capacity(res * res);
    for wh in 0..nw {
        for ww in 0..nw {
            for i in 0..window {
                for j in 0..window {
                    let r = (wh * window + i + shift) % res;
                    let c = (ww * window + j + shift) % res;
                    out.push(r * res + c);
                }
            }
        }
    }
    out
}

/// `ClapAudioLayer.get_attn_mask`: regions are labeled in the shifted
/// frame by slices `[0, -window)`, `[-window, -shift)`, `[-shift, end)` per
/// axis; pairs from different regions get -100.
fn shift_mask(res: usize, window: usize, shift: usize) -> Vec<f32> {
    let region = |x: usize| {
        if x < res - window {
            0
        } else if x < res - shift {
            1
        } else {
            2
        }
    };
    let nw = res / window;
    let t = window * window;
    let mut out = vec![0.0f32; nw * nw * t * t];
    for wh in 0..nw {
        for ww in 0..nw {
            let w = wh * nw + ww;
            let ids: Vec<usize> = (0..t)
                .map(|i| region(wh * window + i / window) * 3 + region(ww * window + i % window))
                .collect();
            for i in 0..t {
                for j in 0..t {
                    if ids[i] != ids[j] {
                        out[(w * t + i) * t + j] = SHIFT_MASK;
                    }
                }
            }
        }
    }
    out
}

/// Attention inside each window. `q`, `k`, `v`, and `ctx` are `[windows *
/// t, heads * hd]` in windowed order. Windows run in parallel; each
/// writes only its own rows of `ctx`.
#[allow(clippy::too_many_arguments)]
fn window_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    ctx: &mut [f32],
    t: usize,
    heads: usize,
    hd: usize,
    bias: &[f32],
    mask: Option<&[f32]>,
) {
    let c = heads * hd;
    let windows = ctx.len() / (t * c);
    // The reference divides by sqrt(head_size) rather than multiplying.
    let denom = (hd as f64).sqrt() as f32;
    // Spawning costs tens of microseconds; small grids stay on the caller.
    let work = windows * t * t * c;
    let threads = if work < 1 << 18 {
        1
    } else {
        std::thread::available_parallelism()
            .map_or(1, |n| n.get())
            .min(windows)
            .max(1)
    };
    let per = windows.div_ceil(threads);
    std::thread::scope(|s| {
        for (chunk_i, chunk) in ctx.chunks_mut(per * t * c).enumerate() {
            s.spawn(move || {
                let mut scores = vec![0.0f32; t];
                for local in 0..chunk.len() / (t * c) {
                    let w = chunk_i * per + local;
                    let base = w * t;
                    for h in 0..heads {
                        let off = h * hd;
                        for i in 0..t {
                            let qi = &q[(base + i) * c + off..(base + i) * c + off + hd];
                            for (j, s) in scores.iter_mut().enumerate() {
                                let kj = &k[(base + j) * c + off..(base + j) * c + off + hd];
                                let mut acc = 0.0f32;
                                for (a, b) in qi.iter().zip(kj) {
                                    acc += a * b;
                                }
                                let mut val = acc / denom + bias[(h * t + i) * t + j];
                                if let Some(m) = mask {
                                    val += m[(w * t + i) * t + j];
                                }
                                *s = val;
                            }
                            ops::softmax_row(&mut scores);
                            let row = &mut chunk
                                [(local * t + i) * c + off..(local * t + i) * c + off + hd];
                            row.fill(0.0);
                            for (j, &p) in scores.iter().enumerate() {
                                let vj = &v[(base + j) * c + off..(base + j) * c + off + hd];
                                for (o, &vv) in row.iter_mut().zip(vj) {
                                    *o += p * vv;
                                }
                            }
                        }
                    }
                }
            });
        }
    });
}

impl SwinBlock {
    fn forward(&self, x: &mut [f32], stage: &Stage) {
        let g = &stage.geom;
        let c = g.dim;
        let n = g.resolution * g.resolution;
        let t = g.window * g.window;
        let mut y = x.to_vec();
        self.ln_before.apply(&mut y, n);
        let gather = if self.shifted {
            &stage.gather_shift
        } else {
            &stage.gather_plain
        };
        let mut xw = vec![0.0f32; n * c];
        for (row, &src) in gather.iter().enumerate() {
            xw[row * c..(row + 1) * c].copy_from_slice(&y[src * c..(src + 1) * c]);
        }
        let q = self.query.forward(&xw, n);
        let k = self.key.forward(&xw, n);
        let v = self.value.forward(&xw, n);
        let mut ctx = vec![0.0f32; n * c];
        let mask = self.shifted.then_some(stage.shift_mask.as_slice());
        window_attention(
            &q,
            &k,
            &v,
            &mut ctx,
            t,
            g.heads,
            c / g.heads,
            &self.rel_bias,
            mask,
        );
        let o = self.attn_out.forward(&ctx, n);
        // window_reverse plus the reverse roll put windowed row r back at
        // token gather[r].
        for (row, &dst) in gather.iter().enumerate() {
            for (a, b) in x[dst * c..(dst + 1) * c]
                .iter_mut()
                .zip(&o[row * c..(row + 1) * c])
            {
                *a += b;
            }
        }
        let mut y = x.to_vec();
        self.ln_after.apply(&mut y, n);
        let mut hidden = self.fc1.forward(&y, n);
        ops::gelu_erf(&mut hidden);
        let out = self.fc2.forward(&hidden, n);
        for (a, b) in x.iter_mut().zip(&out) {
            *a += b;
        }
    }
}

impl PatchMerging {
    /// `[res^2, c] -> [(res/2)^2, 2c]`; concat order x0 (even row, even
    /// col), x1 (odd, even), x2 (even, odd), x3 (odd, odd).
    fn forward(&self, x: &[f32], res: usize, c: usize) -> Vec<f32> {
        let half = res / 2;
        let mut cat = vec![0.0f32; half * half * 4 * c];
        for r in 0..half {
            for col in 0..half {
                let dst = (r * half + col) * 4 * c;
                for (q, (dr, dc)) in [(0, 0), (1, 0), (0, 1), (1, 1)].into_iter().enumerate() {
                    let src = ((2 * r + dr) * res + 2 * col + dc) * c;
                    cat[dst + q * c..dst + (q + 1) * c].copy_from_slice(&x[src..src + c]);
                }
            }
        }
        self.norm.apply(&mut cat, half * half);
        self.reduction.forward(&cat, half * half)
    }
}

/// `nn.functional.interpolate(mode="bicubic", align_corners=True)` along
/// the time axis of `[t_in, m]`, as PyTorch's CPU kernel computes it (f32
/// source index and weights, A = -0.75, edge-clamped taps, taps summed in
/// order). The frequency axis is unchanged: with align_corners and equal
/// sizes its weights are exactly [0, 1, 0, 0].
fn bicubic_time(x: &[f32], t_in: usize, m: usize, t_out: usize) -> Vec<f32> {
    const A: f32 = -0.75;
    let conv1 = |x: f32| ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0;
    let conv2 = |x: f32| ((A * x - 5.0 * A) * x + 8.0 * A) * x - 4.0 * A;
    let scale = if t_out > 1 {
        (t_in - 1) as f32 / (t_out - 1) as f32
    } else {
        0.0
    };
    let mut out = vec![0.0f32; t_out * m];
    for o in 0..t_out {
        let real = scale * o as f32;
        let idx = (real.floor() as usize).min(t_in - 1);
        let lambda = (real - idx as f32).clamp(0.0, 1.0);
        let x2 = 1.0 - lambda;
        let w = [
            conv2(lambda + 1.0),
            conv1(lambda),
            conv1(x2),
            conv2(x2 + 1.0),
        ];
        let taps = [
            idx.saturating_sub(1),
            idx,
            (idx + 1).min(t_in - 1),
            (idx + 2).min(t_in - 1),
        ];
        for f in 0..m {
            let mut acc = x[taps[0] * m + f] * w[0];
            for k in 1..4 {
                acc += x[taps[k] * m + f] * w[k];
            }
            out[o * m + f] = acc;
        }
    }
    out
}

impl HtsatEncoder {
    /// Loads `{prefix}batch_norm`, `patch_embed`, `layers.*`, and `norm`
    /// (prefix `audio_model.audio_encoder.` in a `ClapModel` checkpoint).
    pub fn load(w: &Weights, cfg: &AudioConfig, prefix: &str) -> Result<Self> {
        let geom = cfg.geometry()?;
        let mels = cfg.num_mel_bins;
        let ln_eps = cfg.layer_norm_eps as f32;
        let bn = |n: &str| w.load(&format!("{prefix}batch_norm.{n}"), &[mels]);
        let (bw, bb, mean, var) = (
            bn("weight")?,
            bn("bias")?,
            bn("running_mean")?,
            bn("running_var")?,
        );
        // PyTorch's eval batch norm: alpha = w / sqrt(var + eps), beta =
        // b - mean * alpha, y = x * alpha + beta.
        let mut bn_alpha = Vec::with_capacity(mels);
        let mut bn_beta = Vec::with_capacity(mels);
        for i in 0..mels {
            let alpha = 1.0 / (var[i] + TORCH_EPS).sqrt() * bw[i];
            bn_alpha.push(alpha);
            bn_beta.push(bb[i] - mean[i] * alpha);
        }
        let c0 = cfg.patch_embeds_hidden_size;
        let p = geom.patch;
        let patch_w = w.load(&format!("{prefix}patch_embed.proj.weight"), &[c0, 1, p, p])?;
        let patch_b = w.load(&format!("{prefix}patch_embed.proj.bias"), &[c0])?;
        let patch_norm = if cfg.enable_patch_layer_norm {
            Some(layer_norm(
                w,
                &format!("{prefix}patch_embed.norm"),
                c0,
                TORCH_EPS,
            )?)
        } else {
            None
        };
        let mut stages = Vec::with_capacity(geom.stages.len());
        for (si, sg) in geom.stages.iter().enumerate() {
            let d = sg.dim;
            let t = sg.window * sg.window;
            let span = 2 * sg.window - 1;
            let mut blocks = Vec::with_capacity(sg.depth);
            for bi in 0..sg.depth {
                let b = format!("{prefix}layers.{si}.blocks.{bi}");
                let table = w.load(
                    &format!("{b}.attention.self.relative_position_bias_table"),
                    &[span * span, sg.heads],
                )?;
                blocks.push(SwinBlock {
                    ln_before: layer_norm(w, &format!("{b}.layernorm_before"), d, ln_eps)?,
                    query: w.linear(
                        &format!("{b}.attention.self.query"),
                        d,
                        d,
                        cfg.qkv_bias,
                        None,
                    )?,
                    key: w.linear(&format!("{b}.attention.self.key"), d, d, cfg.qkv_bias, None)?,
                    value: w.linear(
                        &format!("{b}.attention.self.value"),
                        d,
                        d,
                        cfg.qkv_bias,
                        None,
                    )?,
                    attn_out: w.linear(&format!("{b}.attention.output.dense"), d, d, true, None)?,
                    ln_after: layer_norm(w, &format!("{b}.layernorm_after"), d, ln_eps)?,
                    fc1: w.linear(
                        &format!("{b}.intermediate.dense"),
                        d,
                        geom.mlp_hidden[si],
                        true,
                        None,
                    )?,
                    fc2: w.linear(
                        &format!("{b}.output.dense"),
                        geom.mlp_hidden[si],
                        d,
                        true,
                        None,
                    )?,
                    rel_bias: relative_bias(&table, sg.window, sg.heads),
                    shifted: bi % 2 == 1 && sg.shift > 0,
                });
            }
            let downsample = if si + 1 < geom.stages.len() {
                let base = format!("{prefix}layers.{si}.downsample");
                Some(PatchMerging {
                    norm: layer_norm(w, &format!("{base}.norm"), 4 * d, TORCH_EPS)?,
                    reduction: w.linear(&format!("{base}.reduction"), 4 * d, 2 * d, false, None)?,
                })
            } else {
                None
            };
            stages.push(Stage {
                gather_plain: gather_index(sg.resolution, sg.window, 0),
                gather_shift: gather_index(sg.resolution, sg.window, sg.shift),
                shift_mask: if sg.shift > 0 {
                    shift_mask(sg.resolution, sg.window, sg.shift)
                } else {
                    Vec::new()
                },
                geom: sg.clone(),
                blocks,
                downsample,
            });
            debug_assert_eq!(
                t * (sg.resolution / sg.window).pow(2),
                sg.resolution * sg.resolution
            );
        }
        let norm = layer_norm(w, &format!("{prefix}norm"), geom.num_features, TORCH_EPS)?;
        Ok(Self {
            geom,
            mels,
            bn_alpha,
            bn_beta,
            patch_w,
            patch_b,
            patch_norm,
            stages,
            norm,
        })
    }

    pub fn geometry(&self) -> &HtsatGeometry {
        &self.geom
    }

    /// Pooled `[num_features]` output for `[frames, mels]` log-mel input.
    /// `hook` sees each named intermediate (`batch_norm`, `mel2img`,
    /// `patch_embed`, `stage{i}_blocks`, `stage{i}_out`, `norm`, `pooled`).
    pub fn forward_with(
        &self,
        features: &[f32],
        frames: usize,
        mels: usize,
        hook: &mut dyn FnMut(&str, &[f32]),
    ) -> Result<Vec<f32>> {
        let g = &self.geom;
        if mels != self.mels || features.len() != frames * mels {
            return Err(input_error(format!(
                "features are {} values for {frames}x{mels}; the model expects {} mel bins",
                features.len(),
                self.mels
            )));
        }
        if frames == 0 || frames > g.spec_width {
            // The reference raises "the wav size should be less than or
            // equal to the swin input size".
            return Err(input_error(format!(
                "{frames} frames; the model takes 1..={} frames",
                g.spec_width
            )));
        }
        let mut x = features.to_vec();
        for row in x.chunks_exact_mut(mels) {
            for (i, v) in row.iter_mut().enumerate() {
                *v = *v * self.bn_alpha[i] + self.bn_beta[i];
            }
        }
        hook("batch_norm", &x);
        if frames < g.spec_width {
            x = bicubic_time(&x, frames, mels, g.spec_width);
        }
        // Fold [R * S, M] time-major into an [R * M, S] image:
        // img[c * M + m][t] = x[c * S + t][m].
        let s = g.image;
        let mut img = vec![0.0f32; s * s];
        for chunk in 0..g.freq_ratio {
            for t in 0..s {
                for m in 0..mels {
                    img[(chunk * mels + m) * s + t] = x[(chunk * s + t) * mels + m];
                }
            }
        }
        hook("mel2img", &img);

        let p = g.patch;
        let grid = g.grid;
        let c0 = self.patch_b.len();
        let mut h = vec![0.0f32; grid * grid * c0];
        for gh in 0..grid {
            for gw in 0..grid {
                let tok = &mut h[(gh * grid + gw) * c0..(gh * grid + gw + 1) * c0];
                for (o, slot) in tok.iter_mut().enumerate() {
                    let mut acc = self.patch_b[o];
                    for kh in 0..p {
                        let src = (gh * p + kh) * s + gw * p;
                        let wrow = &self.patch_w[(o * p + kh) * p..(o * p + kh + 1) * p];
                        for (a, b) in img[src..src + p].iter().zip(wrow) {
                            acc += a * b;
                        }
                    }
                    *slot = acc;
                }
            }
        }
        if let Some(norm) = &self.patch_norm {
            norm.apply(&mut h, grid * grid);
        }
        hook("patch_embed", &h);

        for (i, stage) in self.stages.iter().enumerate() {
            for block in &stage.blocks {
                block.forward(&mut h, stage);
            }
            hook(&format!("stage{i}_blocks"), &h);
            if let Some(ds) = &stage.downsample {
                h = ds.forward(&h, stage.geom.resolution, stage.geom.dim);
            }
            hook(&format!("stage{i}_out"), &h);
        }
        let c = g.num_features;
        let tokens = h.len() / c;
        self.norm.apply(&mut h, tokens);
        hook("norm", &h);
        let mut pooled = vec![0.0f32; c];
        for row in h.chunks_exact(c) {
            for (a, b) in pooled.iter_mut().zip(row) {
                *a += b;
            }
        }
        for v in pooled.iter_mut() {
            *v /= tokens as f32;
        }
        hook("pooled", &pooled);
        Ok(pooled)
    }
}

#[cfg(test)]
mod unit {
    use super::*;

    #[test]
    fn relative_index_matches_reference_formula() {
        // Window 2: table rows (dh + 1) * 3 + (dw + 1) for dh = ih - jh.
        let table: Vec<f32> = (0..9).map(|v| v as f32).collect();
        let b = relative_bias(&table, 2, 1);
        // Token 0 = (0,0) vs token 3 = (1,1): dh = dw = -1 -> row 0.
        assert_eq!(b[3], 0.0);
        // Token 3 vs token 0: dh = dw = +1 -> row 8.
        assert_eq!(b[3 * 4], 8.0);
        assert_eq!(b[0], 4.0);
    }

    #[test]
    fn shift_mask_regions() {
        // res 8, window 4, shift 2: window 0 (top-left) is one region.
        let m = shift_mask(8, 4, 2);
        assert!(m[..16 * 16].iter().all(|&v| v == 0.0));
        // The bottom-right window mixes four regions.
        let last = &m[3 * 256..4 * 256];
        assert_eq!(
            last.iter().filter(|&&v| v == SHIFT_MASK).count(),
            16 * 16 - 4 * 4 * 4
        );
    }

    #[test]
    fn bicubic_identity_on_equal_sizes() {
        let x: Vec<f32> = (0..12).map(|v| v as f32 * 0.37 - 1.0).collect();
        assert_eq!(bicubic_time(&x, 6, 2, 6), x);
    }
}
