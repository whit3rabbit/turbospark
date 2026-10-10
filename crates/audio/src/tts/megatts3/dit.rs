//! MegaTTS3's latent diffusion transformer (`tts/modules/llm_dit/dit.py`).
//!
//! `Diffusion.inference` predicts a flow-matching velocity for a batch of
//! three CFG rows that share the noisy latent `x` and the timestep and differ
//! only in conditioning:
//!
//! | row | phones/tones | prompt latent |
//! |---|---|---|
//! | 0 spk+txt | real | present |
//! | 1 txt | real | zeroed |
//! | 2 uncond | mask tokens (`n_vocab - 1`, `tone_vocab - 1`) | zeroed |
//!
//! combined as `uncond + p_w * (txt - uncond) + t_w * (spk_txt - txt)`.
//! Everything that does not depend on `x` or the timestep (the phone
//! encoder, the 4x duration expansion, the conditioning projections) is
//! computed once in [`Dit::prepare`]; each step then runs only the
//! transformer for the three rows.
//!
//! Weights come from the converted `diffusion_transformer/model.safetensors`.
//! The checkpoint also carries `bn_eos_bos`, `seg_embed` and
//! `time_embedding.*`, which the reference never uses (it loads with
//! `strict=False`); they are ignored here for the same reason.

use std::path::Path;

use turbospark_model_io::safetensors::SafetensorsFile;

use super::rel_transformer::{sinusoid_positions, RelTransformer, RelTransformerConfig};
use crate::codec::conv::{load_bias, load_mlx_conv_weight, BiasMode, Conv1d};
use crate::codec::wnconv::load_f32_shaped;
use crate::nn::Linear;
use crate::{ops, Result, SpeechError};

const MAX_LAYERS: usize = 64;
const MAX_DIM: usize = 8192;
/// Rotary base of the reference (`precompute_freqs_cis`, theta 10000).
const ROPE_THETA: f32 = 10_000.0;
/// `nn.LayerNorm(elementwise_affine=False, eps=1e-6)` in every adaLN block.
const LN_EPS: f32 = 1e-6;

#[derive(Debug, Clone, PartialEq)]
pub struct DitConfig {
    /// Transformer width (`encoder_dim`).
    pub dim: usize,
    pub layers: usize,
    pub heads: usize,
    /// Hidden width of the SiLU feed-forward.
    pub ffn_hidden: usize,
    /// Latent channels in and out.
    pub channels: usize,
    pub local_cond_dim: usize,
    pub ctx_mask_dim: usize,
    /// Sinusoid width of the F5-style time embedding.
    pub time_freq_dim: usize,
    pub phone_vocab: usize,
    pub tone_vocab: usize,
    pub ph_layers: usize,
    pub ph_heads: usize,
    pub ph_filter: usize,
    pub ph_kernel: usize,
}

impl DitConfig {
    /// The shipped MegaTTS3 checkpoint.
    pub fn megatts3() -> Self {
        Self {
            dim: 1024,
            layers: 24,
            heads: 16,
            ffn_hidden: 2048,
            channels: 32,
            local_cond_dim: 512,
            ctx_mask_dim: 16,
            time_freq_dim: 256,
            phone_vocab: 302,
            tone_vocab: 32,
            ph_layers: 6,
            ph_heads: 4,
            ph_filter: 2048,
            ph_kernel: 3,
        }
    }

    pub fn from_json(value: &serde_json::Value) -> Result<Self> {
        let u = |f: &str| -> Result<usize> {
            value
                .get(f)
                .and_then(|v| v.as_u64())
                .map(|n| n as usize)
                .ok_or_else(|| SpeechError::BadConfig {
                    field: f.to_string(),
                    why: "missing or not an unsigned integer".to_string(),
                })
        };
        let cfg = Self {
            dim: u("dim")?,
            layers: u("layers")?,
            heads: u("heads")?,
            ffn_hidden: u("ffn_hidden")?,
            channels: u("channels")?,
            local_cond_dim: u("local_cond_dim")?,
            ctx_mask_dim: u("ctx_mask_dim")?,
            time_freq_dim: u("time_freq_dim")?,
            phone_vocab: u("phone_vocab")?,
            tone_vocab: u("tone_vocab")?,
            ph_layers: u("ph_layers")?,
            ph_heads: u("ph_heads")?,
            ph_filter: u("ph_filter")?,
            ph_kernel: u("ph_kernel")?,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn validate(&self) -> Result<()> {
        let bad = |f: &str, why: &str| SpeechError::BadConfig {
            field: format!("dit.{f}"),
            why: why.to_string(),
        };
        for (name, v) in [
            ("dim", self.dim),
            ("ffn_hidden", self.ffn_hidden),
            ("channels", self.channels),
            ("local_cond_dim", self.local_cond_dim),
            ("ctx_mask_dim", self.ctx_mask_dim),
            ("time_freq_dim", self.time_freq_dim),
        ] {
            if v == 0 || v > MAX_DIM {
                return Err(bad(name, "must be in 1..=8192"));
            }
        }
        if self.layers == 0 || self.layers > MAX_LAYERS {
            return Err(bad("layers", "must be in 1..=64"));
        }
        if self.heads == 0 || self.dim % self.heads != 0 || (self.dim / self.heads) % 2 != 0 {
            return Err(bad("heads", "must divide dim into an even head width"));
        }
        if self.time_freq_dim < 4 || self.time_freq_dim % 2 != 0 {
            return Err(bad("time_freq_dim", "must be even and at least 4"));
        }
        if self.phone_vocab < 2 || self.tone_vocab < 2 {
            return Err(bad("vocab", "need room for a mask token"));
        }
        self.ph_config().validate()
    }

    pub(crate) fn ph_config(&self) -> RelTransformerConfig {
        RelTransformerConfig {
            n_vocab: self.phone_vocab,
            hidden: self.dim,
            filter: self.ph_filter,
            heads: self.ph_heads,
            layers: self.ph_layers,
            kernel: self.ph_kernel,
        }
    }
}

/// Frames per latent frame: the two stride-2 convs of `ling_pre_net`.
pub const LING_STRIDE: usize = 4;

struct Block {
    wq: Linear,
    wk: Linear,
    wv: Linear,
    wo: Linear,
    /// `[6 * dim, dim]` -> shift/scale/gate for attention and MLP.
    ada: Linear,
    w1: Linear,
    w2: Linear,
}

pub struct Dit {
    cfg: DitConfig,
    ctx_mask_proj: Linear,
    local_cond_project: Linear,
    x_prenet: Linear,
    prenet: Linear,
    postnet: Linear,
    time_in: Linear,
    time_out: Linear,
    blocks: Vec<Block>,
    final_ada: Linear,
    out_proj: Linear,
    ph_encoder: RelTransformer,
    tone_embed: Vec<f32>,
    ling_pre_net: [Conv1d; 2],
}

/// Everything about one request that does not change between sampler steps.
pub struct DitCondition {
    /// Total latent frames (prompt + generated).
    pub frames: usize,
    /// Leading frames held fixed by the prompt.
    pub prompt_frames: usize,
    /// Per row `prenet(local_cond) + x_ling`, each `[frames, dim]`.
    base: [Vec<f32>; 3],
    /// `1 - ctx_mask` per frame: 0 inside the prompt, 1 after.
    keep: Vec<f32>,
    /// Rotary tables `[frames, head_dim / 2]` for positions `0..frames`.
    rope_cos: Vec<f32>,
    rope_sin: Vec<f32>,
}

impl Dit {
    pub fn open(dir: &Path) -> Result<Self> {
        let file = SafetensorsFile::open(&dir.join("model.safetensors"))?;
        Self::load(DitConfig::megatts3(), &file)
    }

    pub fn load(cfg: DitConfig, file: &SafetensorsFile) -> Result<Self> {
        cfg.validate()?;
        let d = cfg.dim;
        let mut blocks = Vec::with_capacity(cfg.layers);
        for l in 0..cfg.layers {
            let p = format!("encoder.layers.{l}");
            blocks.push(Block {
                wq: Linear::load(file, &format!("{p}.attention.wq"), d, d, true)?,
                wk: Linear::load(file, &format!("{p}.attention.wk"), d, d, true)?,
                wv: Linear::load(file, &format!("{p}.attention.wv"), d, d, true)?,
                wo: Linear::load(file, &format!("{p}.attention.wo"), d, d, true)?,
                ada: Linear::load(file, &format!("{p}.attention_norm.linear"), d, 6 * d, true)?,
                w1: Linear::load(
                    file,
                    &format!("{p}.feed_forward.w1"),
                    d,
                    cfg.ffn_hidden,
                    true,
                )?,
                w2: Linear::load(
                    file,
                    &format!("{p}.feed_forward.w2"),
                    cfg.ffn_hidden,
                    d,
                    true,
                )?,
            });
        }
        let conv = |i: usize| -> Result<Conv1d> {
            let name = format!("ling_pre_net.{i}");
            Ok(Conv1d {
                in_ch: d,
                out_ch: d,
                kernel: 4,
                stride: 2,
                padding: 1,
                dilation: 1,
                groups: 1,
                weight: load_mlx_conv_weight(file, &format!("{name}.weight"), d, 4, d)?,
                bias: load_bias(file, &format!("{name}.bias"), d, BiasMode::Required)?,
            })
        };
        Ok(Self {
            ctx_mask_proj: Linear::load(file, "ctx_mask_proj", 1, cfg.ctx_mask_dim, true)?,
            local_cond_project: Linear::load(
                file,
                "local_cond_project",
                cfg.channels + cfg.ctx_mask_dim,
                cfg.local_cond_dim,
                true,
            )?,
            x_prenet: Linear::load(file, "x_prenet", cfg.channels, d, true)?,
            prenet: Linear::load(file, "prenet", cfg.local_cond_dim, d, true)?,
            postnet: Linear::load(file, "postnet", d, cfg.channels, true)?,
            time_in: Linear::load(file, "f5_time_embed.time_mlp.0", cfg.time_freq_dim, d, true)?,
            time_out: Linear::load(file, "f5_time_embed.time_mlp.2", d, d, true)?,
            blocks,
            final_ada: Linear::load(file, "encoder.norm.linear", d, 2 * d, true)?,
            out_proj: Linear::load(file, "encoder.out_proj", d, d, true)?,
            ph_encoder: RelTransformer::load(file, "ph_encoder", cfg.ph_config())?,
            tone_embed: load_f32_shaped(file, "tone_embed.weight", &[cfg.tone_vocab, d])?,
            ling_pre_net: [conv(0)?, conv(1)?],
            cfg,
        })
    }

    pub fn config(&self) -> &DitConfig {
        &self.cfg
    }

    /// `forward_ling_encoder`: phones/tones `[N]` -> `[N, dim]`.
    pub(crate) fn phone_encoding(&self, phone: &[i32], tone: &[i32]) -> Result<Vec<f32>> {
        let d = self.cfg.dim;
        let pos = sinusoid_positions(phone.len(), d);
        let tones = ops::embedding(&self.tone_embed, d, tone);
        let other: Vec<f32> = tones.iter().zip(&pos).map(|(a, b)| a + b).collect();
        self.ph_encoder.forward(phone, &other)
    }

    /// `forward_ling_encoder` followed by `expand_states` and `ling_pre_net`:
    /// phones/tones `[N]` and `mel2ph [4 * frames]` -> `[frames, dim]`.
    pub(crate) fn ling(
        &self,
        phone: &[i32],
        tone: &[i32],
        mel2ph: &[i32],
        frames: usize,
    ) -> Result<Vec<f32>> {
        let d = self.cfg.dim;
        let n = phone.len();
        let enc = self.phone_encoding(phone, tone)?;
        // expand_states: index 0 is the zero row, index i is phone i - 1.
        let tm = mel2ph.len();
        let mut expanded = vec![0.0f32; d * tm];
        for (t, &m) in mel2ph.iter().enumerate() {
            if m < 0 || m as usize > n {
                return Err(SpeechError::Input {
                    why: format!("mel2ph[{t}] = {m} outside 0..={n}"),
                });
            }
            if m > 0 {
                let row = &enc[(m as usize - 1) * d..m as usize * d];
                for (c, &v) in row.iter().enumerate() {
                    expanded[c * tm + t] = v;
                }
            }
        }
        // Two stride-2 convs, channel-major, no activation in between.
        let h1 = self.ling_pre_net[0].forward(&expanded);
        let h2 = self.ling_pre_net[1].forward(&h1);
        debug_assert_eq!(h2.len(), d * frames);
        Ok(transpose(&h2, d, frames))
    }

    /// Precomputes the step-independent parts of the three CFG rows.
    ///
    /// `mel2ph` is the 100 fps phone index per frame (1-based, 0 = none) of the
    /// prompt followed by the target; `prompt_latent` is `[prompt_frames,
    /// channels]` and must not be longer than the output.
    pub fn prepare(
        &self,
        phone: &[i32],
        tone: &[i32],
        mel2ph: &[i32],
        prompt_latent: &[f32],
    ) -> Result<DitCondition> {
        let cfg = &self.cfg;
        let ch = cfg.channels;
        if phone.is_empty() || phone.len() != tone.len() {
            return Err(SpeechError::Input {
                why: format!(
                    "phone ({}) and tone ({}) must be equal and non-empty",
                    phone.len(),
                    tone.len()
                ),
            });
        }
        if mel2ph.is_empty() || mel2ph.len() % LING_STRIDE != 0 {
            return Err(SpeechError::Input {
                why: format!(
                    "mel2ph length {} is not a positive multiple of {LING_STRIDE}",
                    mel2ph.len()
                ),
            });
        }
        if tone.iter().any(|&t| t < 0 || t as usize >= cfg.tone_vocab) {
            return Err(SpeechError::Input {
                why: format!("tone id outside 0..{}", cfg.tone_vocab),
            });
        }
        let frames = mel2ph.len() / LING_STRIDE;
        if prompt_latent.len() % ch != 0 || prompt_latent.len() / ch > frames {
            return Err(SpeechError::Input {
                why: format!(
                    "prompt latent of {} values does not fit {frames} frames x {ch} channels",
                    prompt_latent.len()
                ),
            });
        }
        let prompt_frames = prompt_latent.len() / ch;

        let ling_real = self.ling(phone, tone, mel2ph, frames)?;
        let mask_phone = vec![cfg.phone_vocab as i32 - 1; phone.len()];
        let mask_tone = vec![cfg.tone_vocab as i32 - 1; phone.len()];
        let ling_mask = self.ling(&mask_phone, &mask_tone, mel2ph, frames)?;

        // local_cond = local_cond_project([lat_ctx, ctx_mask_proj(ctx_mask)]).
        // Past the prompt the mask is 0, so only the projection bias survives.
        let local = |with_prompt: bool| -> Vec<f32> {
            let mut cat = vec![0.0f32; frames * (ch + cfg.ctx_mask_dim)];
            for t in 0..frames {
                let in_prompt = t < prompt_frames;
                let row = &mut cat[t * (ch + cfg.ctx_mask_dim)..(t + 1) * (ch + cfg.ctx_mask_dim)];
                if with_prompt && in_prompt {
                    row[..ch].copy_from_slice(&prompt_latent[t * ch..(t + 1) * ch]);
                }
                let m = if in_prompt { 1.0 } else { 0.0 };
                for (j, v) in row[ch..].iter_mut().enumerate() {
                    *v = m * self.ctx_mask_proj.weight[j]
                        + self.ctx_mask_proj.bias.as_ref().map_or(0.0, |b| b[j]);
                }
            }
            let lc = self.local_cond_project.forward(&cat, frames);
            self.prenet.forward(&lc, frames)
        };
        let with = local(true);
        let without = local(false);
        let add =
            |a: &[f32], b: &[f32]| -> Vec<f32> { a.iter().zip(b).map(|(x, y)| x + y).collect() };
        let keep = (0..frames)
            .map(|t| if t < prompt_frames { 0.0 } else { 1.0 })
            .collect();
        let (rope_cos, rope_sin) = ops::rope_tables(frames, cfg.dim / cfg.heads, ROPE_THETA);
        Ok(DitCondition {
            frames,
            prompt_frames,
            base: [
                add(&with, &ling_real),
                add(&without, &ling_real),
                add(&without, &ling_mask),
            ],
            keep,
            rope_cos,
            rope_sin,
        })
    }

    fn time_embedding(&self, t: f32) -> Vec<f32> {
        let half = self.cfg.time_freq_dim / 2;
        let step = (10_000.0f32).ln() / (half as f32 - 1.0);
        let mut sin_cos = vec![0.0f32; self.cfg.time_freq_dim];
        for i in 0..half {
            let a = 1000.0 * t * (-(i as f32) * step).exp();
            sin_cos[i] = a.sin();
            sin_cos[half + i] = a.cos();
        }
        let mut h = self.time_in.forward(&sin_cos, 1);
        ops::silu(&mut h);
        self.time_out.forward(&h, 1)
    }

    /// One transformer pass over `batch` stacked rows `[batch * frames, dim]`.
    ///
    /// Stacking changes no value: every linear, norm and activation works per
    /// position (each output is the same `k`-ascending sum however many rows
    /// share the call), and attention runs on `batch * heads` independent
    /// planes, so a row's heads only see that row's keys. The adaLN
    /// projections depend only on the timestep and are shared by all rows.
    fn run_rows(
        &self,
        cond: &DitCondition,
        mut x: Vec<f32>,
        batch: usize,
        temb_silu: &[f32],
    ) -> Vec<f32> {
        let frames = cond.frames;
        let rows = batch * frames;
        let d = self.cfg.dim;
        let heads = self.cfg.heads;
        let hd = d / heads;
        debug_assert_eq!(x.len(), rows * d);
        let ones = vec![1.0f32; d];
        let (cos, sin) = (&cond.rope_cos, &cond.rope_sin);
        // Row-major `[batch * frames, dim]` <-> head-major `[batch * heads, frames, hd]`.
        let split = |y: &[f32]| -> Vec<f32> {
            let mut out = Vec::with_capacity(y.len());
            for row in y.chunks_exact(frames * d) {
                out.extend(ops::split_heads(row, frames, heads, hd));
            }
            out
        };
        let merge = |y: &[f32]| -> Vec<f32> {
            let mut out = Vec::with_capacity(y.len());
            for row in y.chunks_exact(frames * d) {
                out.extend(ops::merge_heads(row, frames, heads, hd));
            }
            out
        };
        for block in &self.blocks {
            let m = block.ada.forward(temb_silu, 1);
            let (shift_a, scale_a, gate_a) = (&m[..d], &m[d..2 * d], &m[2 * d..3 * d]);
            let (shift_m, scale_m, gate_m) = (&m[3 * d..4 * d], &m[4 * d..5 * d], &m[5 * d..6 * d]);

            let mut n = x.clone();
            ops::layernorm(&mut n, rows, d, &ones, None, LN_EPS);
            modulate(&mut n, scale_a, shift_a, d);
            let mut q = split(&block.wq.forward(&n, rows));
            let mut k = split(&block.wk.forward(&n, rows));
            let v = split(&block.wv.forward(&n, rows));
            ops::rope_interleaved(&mut q, batch * heads, frames, hd, cos, sin);
            ops::rope_interleaved(&mut k, batch * heads, frames, hd, cos, sin);
            let a = ops::mha(
                &q,
                &k,
                &v,
                None,
                batch * heads,
                frames,
                frames,
                hd,
                1.0 / (hd as f32).sqrt(),
            );
            let o = block.wo.forward(&merge(&a), rows);
            for (row, orow) in x.chunks_exact_mut(d).zip(o.chunks_exact(d)) {
                for j in 0..d {
                    row[j] += gate_a[j] * orow[j];
                }
            }

            let mut n = x.clone();
            ops::layernorm(&mut n, rows, d, &ones, None, LN_EPS);
            modulate(&mut n, scale_m, shift_m, d);
            let mut h = block.w1.forward(&n, rows);
            ops::silu(&mut h);
            let f = block.w2.forward(&h, rows);
            for (row, frow) in x.chunks_exact_mut(d).zip(f.chunks_exact(d)) {
                for j in 0..d {
                    row[j] += gate_m[j] * frow[j];
                }
            }
        }
        // AdaLNZero_Out: scale first, then shift.
        let m = self.final_ada.forward(temb_silu, 1);
        ops::layernorm(&mut x, rows, d, &ones, None, LN_EPS);
        modulate(&mut x, &m[..d], &m[d..], d);
        self.postnet.forward(&self.out_proj.forward(&x, rows), rows)
    }

    /// Stacks `x_prenet(x) + base[r]` for the requested CFG rows.
    fn stacked_input(&self, cond: &DitCondition, xp: &[f32], rows: &[usize]) -> Vec<f32> {
        let mut h = Vec::with_capacity(rows.len() * xp.len());
        for &r in rows {
            h.extend(xp.iter().zip(&cond.base[r]).map(|(a, b)| a + b));
        }
        h
    }

    /// CFG-combined velocity for noisy latent `x [frames, channels]` at
    /// timestep `sigma`.
    pub fn velocity(
        &self,
        cond: &DitCondition,
        x: &[f32],
        sigma: f32,
        p_w: f32,
        t_w: f32,
    ) -> Result<Vec<f32>> {
        let ch = self.cfg.channels;
        let frames = cond.frames;
        if x.len() != frames * ch {
            return Err(SpeechError::Input {
                why: format!("x has {} values, expected {}", x.len(), frames * ch),
            });
        }
        let mut x_in = x.to_vec();
        for (t, row) in x_in.chunks_exact_mut(ch).enumerate() {
            for v in row.iter_mut() {
                *v *= cond.keep[t];
            }
        }
        let xp = self.x_prenet.forward(&x_in, frames);
        let mut temb = self.time_embedding(sigma);
        ops::silu(&mut temb);
        let h = self.stacked_input(cond, &xp, &[0, 1, 2]);
        let out = self.run_rows(cond, h, 3, &temb);
        let n = frames * ch;
        let (spk_txt, txt, uncond) = (&out[..n], &out[n..2 * n], &out[2 * n..]);
        Ok((0..n)
            .map(|i| uncond[i] + p_w * (txt[i] - uncond[i]) + t_w * (spk_txt[i] - txt[i]))
            .collect())
    }

    /// The uncombined output of each CFG row run alone, for the batching
    /// invariance test.
    #[cfg(test)]
    pub(super) fn rows_unbatched(&self, cond: &DitCondition, x: &[f32], sigma: f32) -> Vec<f32> {
        let xp = self.x_prenet.forward(x, cond.frames);
        let mut temb = self.time_embedding(sigma);
        ops::silu(&mut temb);
        (0..3)
            .flat_map(|r| self.run_rows(cond, self.stacked_input(cond, &xp, &[r]), 1, &temb))
            .collect()
    }

    #[cfg(test)]
    pub(super) fn rows_batched(&self, cond: &DitCondition, x: &[f32], sigma: f32) -> Vec<f32> {
        let xp = self.x_prenet.forward(x, cond.frames);
        let mut temb = self.time_embedding(sigma);
        ops::silu(&mut temb);
        self.run_rows(cond, self.stacked_input(cond, &xp, &[0, 1, 2]), 3, &temb)
    }
}

/// `x * (1 + scale) + shift`, scale and shift broadcast over rows.
fn modulate(x: &mut [f32], scale: &[f32], shift: &[f32], d: usize) {
    for row in x.chunks_exact_mut(d) {
        for j in 0..d {
            row[j] = row[j] * (1.0 + scale[j]) + shift[j];
        }
    }
}

fn transpose(x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = x[r * cols + c];
        }
    }
    out
}
