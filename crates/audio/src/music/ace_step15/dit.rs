//! ACE-Step 1.5 condition encoder, DiT decoder, and turbo sampler.
//!
//! Reference: `AceStepConditionEncoder`, `AceStepDiTModel`,
//! `AceStepDiTLayer`, `TimestepEmbedding` and
//! `AceStepConditionGenerationModel.generate_audio` in
//! `modeling_acestep_v15_turbo.py` at the pinned checkpoint revision.
//!
//! Scope: one batch item with no padding, which is what upstream builds
//! for a single request (`padding="longest"` over one sequence). The
//! encoder packs `[lyric, timbre, text]`; with every mask set, the
//! reference's mask-sorted pack is that plain concatenation. One
//! reference clip gives one timbre token. The FSQ tokenizer and
//! detokenizer (cover hints) are not ported yet.
//!
//! Layout: time-major `[frames, channels]` latents, as the reference
//! `[B, T, C]`.

use super::dcw::Dcw;
use super::transformer::{
    active_window, linear, rms, rope_tables, AceConfig, Attention, EncoderStack, Loader, Mlp, Proj,
};
use crate::nn::{load_tensor, Linear, RmsNorm};
use crate::{ops, Result, SpeechError};

/// Sinusoidal width of `TimestepEmbedding` (`in_channels=256`).
const TIME_FREQ_DIM: usize = 256;

/// `AceStepConditionEncoder`: text projection, lyric encoder, timbre
/// encoder, packed into one cross-attention sequence.
#[derive(Clone)]
pub struct ConditionEncoder {
    text_projector: Proj,
    lyric: EncoderStack,
    timbre: EncoderStack,
    text_dim: usize,
    timbre_dim: usize,
    hidden: usize,
}

/// Conditioning inputs for one request.
#[derive(Debug, Clone, Copy)]
pub struct Conditioning<'a> {
    /// Text encoder last hidden state `[text_len, text_hidden_dim]`.
    pub text: &'a [f32],
    pub text_len: usize,
    /// Text encoder token embeddings of the lyrics `[lyric_len, text_hidden_dim]`.
    pub lyric: &'a [f32],
    pub lyric_len: usize,
    /// Reference acoustic latents `[refer_len, timbre_hidden_dim]`.
    pub refer: &'a [f32],
    pub refer_len: usize,
}

impl ConditionEncoder {
    pub fn load(l: &Loader<'_>, c: &AceConfig) -> Result<Self> {
        Ok(Self {
            text_projector: linear(
                l,
                "encoder.text_projector",
                c.text_hidden_dim,
                c.hidden_size,
                false,
            )?,
            lyric: EncoderStack::load(
                l,
                "encoder.lyric_encoder",
                c,
                c.text_hidden_dim,
                c.num_lyric_encoder_hidden_layers,
            )?,
            timbre: EncoderStack::load(
                l,
                "encoder.timbre_encoder",
                c,
                c.timbre_hidden_dim,
                c.num_timbre_encoder_hidden_layers,
            )?,
            text_dim: c.text_hidden_dim,
            timbre_dim: c.timbre_hidden_dim,
            hidden: c.hidden_size,
        })
    }

    /// Returns the packed encoder states `[lyric_len + 1 + text_len, hidden]`.
    pub fn forward(&self, cond: &Conditioning<'_>) -> Result<(Vec<f32>, usize)> {
        let check = |name: &str, data: &[f32], len: usize, width: usize| {
            if len == 0 || data.len() != len * width {
                return Err(SpeechError::Input {
                    why: format!(
                        "{name}: expected {len} x {width} values, got {}",
                        data.len()
                    ),
                });
            }
            Ok(())
        };
        check("text", cond.text, cond.text_len, self.text_dim)?;
        check("lyric", cond.lyric, cond.lyric_len, self.text_dim)?;
        check("refer", cond.refer, cond.refer_len, self.timbre_dim)?;
        let text = self.text_projector.forward(cond.text, cond.text_len)?;
        let lyric = self.lyric.forward(cond.lyric, cond.lyric_len)?;
        // The timbre encoder's output at position 0 is the timbre token
        // (the special token prepend is commented out upstream).
        let timbre = self.timbre.forward(cond.refer, cond.refer_len)?;
        let d = self.hidden;
        let mut packed = Vec::with_capacity((cond.lyric_len + 1 + cond.text_len) * d);
        packed.extend_from_slice(&lyric);
        packed.extend_from_slice(&timbre[..d]);
        packed.extend_from_slice(&text);
        Ok((packed, cond.lyric_len + 1 + cond.text_len))
    }
}

/// `TimestepEmbedding`: sinusoid of `1000 t`, two linears, and a 6-way
/// modulation projection.
#[derive(Clone)]
struct TimestepEmbedding {
    linear_1: Proj,
    linear_2: Proj,
    time_proj: Proj,
}

impl TimestepEmbedding {
    fn load(l: &Loader<'_>, prefix: &str, d: usize) -> Result<Self> {
        Ok(Self {
            linear_1: linear(l, &format!("{prefix}.linear_1"), TIME_FREQ_DIM, d, true)?,
            linear_2: linear(l, &format!("{prefix}.linear_2"), d, d, true)?,
            time_proj: linear(l, &format!("{prefix}.time_proj"), d, 6 * d, true)?,
        })
    }

    /// Returns `(temb [d], proj [6 * d])`.
    fn forward(&self, t: f32) -> Result<(Vec<f32>, Vec<f32>)> {
        let half = TIME_FREQ_DIM / 2;
        let t = t * 1000.0;
        let neg_log = -(10000f64.ln() as f32);
        let mut freq = vec![0.0f32; TIME_FREQ_DIM];
        for i in 0..half {
            let f = (neg_log * i as f32 / half as f32).exp();
            let a = t * f;
            freq[i] = a.cos();
            freq[half + i] = a.sin();
        }
        let mut h = self.linear_1.forward(&freq, 1)?;
        ops::silu(&mut h);
        let temb = self.linear_2.forward(&h, 1)?;
        let mut act = temb.clone();
        ops::silu(&mut act);
        let proj = self.time_proj.forward(&act, 1)?;
        Ok((temb, proj))
    }
}

/// `AceStepDiTLayer`: AdaLN self-attention, plain cross-attention, AdaLN
/// MLP, with gated residuals.
#[derive(Clone)]
struct DitLayer {
    self_norm: RmsNorm,
    self_attn: Attention,
    cross_norm: RmsNorm,
    cross_attn: Attention,
    mlp_norm: RmsNorm,
    mlp: Mlp,
    /// `[6, d]`: shift, scale, gate for attention, then for the MLP.
    scale_shift: Vec<f32>,
    span: Option<usize>,
}

impl DitLayer {
    fn load(l: &Loader<'_>, prefix: &str, c: &AceConfig, idx: usize) -> Result<Self> {
        let file = l.file;
        let (d, eps) = (c.hidden_size, c.rms_norm_eps);
        Ok(Self {
            self_norm: rms(l, &format!("{prefix}.self_attn_norm"), d, eps)?,
            self_attn: Attention::load(l, &format!("{prefix}.self_attn"), c)?,
            cross_norm: rms(l, &format!("{prefix}.cross_attn_norm"), d, eps)?,
            cross_attn: Attention::load(l, &format!("{prefix}.cross_attn"), c)?,
            mlp_norm: rms(l, &format!("{prefix}.mlp_norm"), d, eps)?,
            mlp: Mlp::load(l, &format!("{prefix}.mlp"), c)?,
            scale_shift: load_tensor(file, &format!("{prefix}.scale_shift_table"), &[1, 6, d])?,
            span: c.span(idx),
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        x: &mut [f32],
        seq: usize,
        proj: &[f32],
        rope: (&[f32], &[f32]),
        enc: &[f32],
        enc_seq: usize,
    ) -> Result<()> {
        let d = self.self_norm.width();
        // (table + temb_proj).chunk(6): rows are shift, scale, gate,
        // c_shift, c_scale, c_gate.
        let m: Vec<f32> = self
            .scale_shift
            .iter()
            .zip(proj)
            .map(|(a, b)| a + b)
            .collect();
        let row = |i: usize| &m[i * d..(i + 1) * d];
        let modulate = |h: &mut [f32], shift: &[f32], scale: &[f32]| {
            for r in h.chunks_exact_mut(d) {
                for ((v, sh), sc) in r.iter_mut().zip(shift).zip(scale) {
                    *v = *v * (1.0 + sc) + sh;
                }
            }
        };
        let gated_add = |x: &mut [f32], y: &[f32], gate: &[f32]| {
            for (xr, yr) in x.chunks_exact_mut(d).zip(y.chunks_exact(d)) {
                for ((xv, yv), g) in xr.iter_mut().zip(yr).zip(gate) {
                    *xv += yv * g;
                }
            }
        };

        let mut h = x.to_vec();
        self.self_norm.apply(&mut h, seq);
        modulate(&mut h, row(0), row(1));
        let a = self
            .self_attn
            .forward_self(&h, seq, rope, active_window(self.span, seq))?;
        gated_add(x, &a, row(2));

        let mut h = x.to_vec();
        self.cross_norm.apply(&mut h, seq);
        let a = self.cross_attn.forward_cross(&h, seq, enc, enc_seq)?;
        for (xv, av) in x.iter_mut().zip(&a) {
            *xv += av;
        }

        let mut h = x.to_vec();
        self.mlp_norm.apply(&mut h, seq);
        modulate(&mut h, row(3), row(4));
        let f = self.mlp.forward(&h, seq)?;
        gated_add(x, &f, row(5));
        Ok(())
    }
}

/// `AceStepDiTModel`: patchify, timestep embeddings, DiT layers, AdaLN
/// output head, de-patchify.
#[derive(Clone)]
pub struct Dit {
    config: AceConfig,
    /// `proj_in` conv (kernel = stride = patch) as a linear over the
    /// flattened patch `[patch * in_channels]`, time-major within it.
    proj_in: Linear,
    time_embed: TimestepEmbedding,
    time_embed_r: TimestepEmbedding,
    condition_embedder: Proj,
    layers: Vec<DitLayer>,
    norm_out: RmsNorm,
    /// `[2, d]`: shift, scale.
    scale_shift_out: Vec<f32>,
    /// `proj_out` transposed conv as a linear `d -> [patch * acoustic]`.
    proj_out: Linear,
}

impl Dit {
    pub fn load(l: &Loader<'_>, c: &AceConfig) -> Result<Self> {
        let file = l.file;
        let (d, p) = (c.hidden_size, c.patch_size);
        let (cin, a) = (c.in_channels, c.audio_acoustic_hidden_dim);
        // Conv1d weight [d, cin, p] -> linear [d, p * cin] with the
        // kernel tap outer, matching a time-major flattened patch.
        let w = load_tensor(file, "decoder.proj_in.1.weight", &[d, cin, p])?;
        let mut wl = vec![0.0f32; d * p * cin];
        for o in 0..d {
            for ci in 0..cin {
                for k in 0..p {
                    wl[o * p * cin + k * cin + ci] = w[(o * cin + ci) * p + k];
                }
            }
        }
        let b = load_tensor(file, "decoder.proj_in.1.bias", &[d])?;
        let proj_in = Linear::new(wl, Some(b), p * cin, d);
        // ConvTranspose1d weight [d, a, p]: out[p_idx * patch + k, o] =
        // sum_i h[p_idx, i] * w[i, o, k] + b[o].
        let w = load_tensor(file, "decoder.proj_out.1.weight", &[d, a, p])?;
        let mut wo = vec![0.0f32; p * a * d];
        for i in 0..d {
            for o in 0..a {
                for k in 0..p {
                    wo[(k * a + o) * d + i] = w[(i * a + o) * p + k];
                }
            }
        }
        let bo = load_tensor(file, "decoder.proj_out.1.bias", &[a])?;
        let bias: Vec<f32> = (0..p).flat_map(|_| bo.iter().copied()).collect();
        let proj_out = Linear::new(wo, Some(bias), d, p * a);
        Ok(Self {
            proj_in,
            time_embed: TimestepEmbedding::load(l, "decoder.time_embed", d)?,
            time_embed_r: TimestepEmbedding::load(l, "decoder.time_embed_r", d)?,
            condition_embedder: linear(l, "decoder.condition_embedder", d, d, true)?,
            layers: (0..c.num_hidden_layers)
                .map(|i| DitLayer::load(l, &format!("decoder.layers.{i}"), c, i))
                .collect::<Result<_>>()?,
            norm_out: rms(l, "decoder.norm_out", d, c.rms_norm_eps)?,
            scale_shift_out: load_tensor(file, "decoder.scale_shift_table", &[1, 2, d])?,
            proj_out,
            config: c.clone(),
        })
    }

    pub fn config(&self) -> &AceConfig {
        &self.config
    }

    /// Projects packed encoder states once; the result is reused by
    /// every step (the reference recomputes it per call, same values).
    pub fn embed_condition(&self, enc: &[f32], enc_seq: usize) -> Result<Vec<f32>> {
        self.condition_embedder.forward(enc, enc_seq)
    }

    /// Velocity for noisy latents `x [frames, acoustic]` at time `t`
    /// (`timestep_r = t`, as the sampler calls it) with context
    /// `[frames, 2 * acoustic]` and embedded encoder states.
    pub fn forward(
        &self,
        x: &[f32],
        context: &[f32],
        frames: usize,
        t: f32,
        enc: &[f32],
        enc_seq: usize,
    ) -> Result<Vec<f32>> {
        let c = &self.config;
        let (a, p, d) = (c.audio_acoustic_hidden_dim, c.patch_size, c.hidden_size);
        if frames == 0 || x.len() != frames * a || context.len() != frames * 2 * a {
            return Err(SpeechError::Input {
                why: format!(
                    "latents {} and context {} do not match {frames} frames of width {a}",
                    x.len(),
                    context.len()
                ),
            });
        }
        if enc.len() != enc_seq * d || enc_seq == 0 {
            return Err(SpeechError::Input {
                why: format!("encoder states {} != {enc_seq} x {d}", enc.len()),
            });
        }
        let (temb_t, proj_t) = self.time_embed.forward(t)?;
        // time_embed_r sees `timestep - timestep_r`; the sampler passes
        // timestep_r = t, and t - t is exactly +0.0 for any finite t.
        let (temb_r, proj_r) = self.time_embed_r.forward(0.0)?;
        let temb: Vec<f32> = temb_t.iter().zip(&temb_r).map(|(x, y)| x + y).collect();
        let proj: Vec<f32> = proj_t.iter().zip(&proj_r).map(|(x, y)| x + y).collect();

        // cat([context, x], -1), zero-padded to a whole patch.
        let patches = frames.div_ceil(p);
        let cin = 3 * a;
        let mut h = vec![0.0f32; patches * p * cin];
        for f in 0..frames {
            let row = &mut h[f * cin..(f + 1) * cin];
            row[..2 * a].copy_from_slice(&context[f * 2 * a..(f + 1) * 2 * a]);
            row[2 * a..].copy_from_slice(&x[f * a..(f + 1) * a]);
        }
        let mut h = self.proj_in.forward(&h, patches);

        let (cos, sin) = rope_tables(patches, c.head_dim, c.rope_theta);
        for layer in &self.layers {
            layer.forward(&mut h, patches, &proj, (&cos, &sin), enc, enc_seq)?;
        }

        self.norm_out.apply(&mut h, patches);
        let (shift, scale) = (&self.scale_shift_out[..d], &self.scale_shift_out[d..]);
        for r in h.chunks_exact_mut(d) {
            for (i, v) in r.iter_mut().enumerate() {
                *v = *v * (1.0 + (scale[i] + temb[i])) + (shift[i] + temb[i]);
            }
        }
        let mut out = self.proj_out.forward(&h, patches);
        out.truncate(frames * a);
        Ok(out)
    }
}

/// Turbo schedules (`SHIFT_TIMESTEPS`, fix_nfe 8), final zero excluded.
const SHIFT_1: [f64; 8] = [1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125];
const SHIFT_2: [f64; 8] = [
    1.0,
    0.9333333333333333,
    0.8571428571428571,
    0.7692307692307693,
    0.6666666666666666,
    0.5454545454545454,
    0.4,
    0.2222222222222222,
];
const SHIFT_3: [f64; 8] = [
    1.0,
    0.9545454545454546,
    0.9,
    0.8333333333333334,
    0.75,
    0.6428571428571429,
    0.5,
    0.3,
];

/// The turbo timestep schedule for `shift`, rounded to the nearest of
/// the three trained shifts as the reference does.
pub fn turbo_schedule(shift: f32) -> [f32; 8] {
    let s = [1.0f32, 2.0, 3.0]
        .into_iter()
        .min_by(|a, b| (a - shift).abs().total_cmp(&(b - shift).abs()))
        .unwrap_or(3.0);
    let table = if s == 1.0 {
        SHIFT_1
    } else if s == 2.0 {
        SHIFT_2
    } else {
        SHIFT_3
    };
    // torch.tensor(list, dtype=float32)
    table.map(|v| v as f32)
}

/// One sampling condition: context latents `[frames, 2 * acoustic]`
/// and embedded encoder states `[enc_seq, hidden]`.
#[derive(Debug, Clone, Copy)]
pub struct StepCondition<'a> {
    pub context: &'a [f32],
    pub enc: &'a [f32],
    pub enc_seq: usize,
}

/// `cover_steps = int(num_steps * audio_cover_strength)`: the step at
/// which a cover run leaves the source hints. `None` when the strength
/// is 1.0 or more, since the reference then never builds the fallback.
pub fn cover_switch_step(steps: usize, strength: f32) -> Option<usize> {
    if strength >= 1.0 {
        return None;
    }
    // Python: int(len(schedule) * float) truncates toward zero.
    Some((steps as f64 * strength.max(0.0) as f64) as usize)
}

/// Euler ODE sampling as `generate_audio` runs it (`infer_method="ode"`):
/// `x -= v * (t_i - t_{i+1})`, and the last step returns `x0 = x - v * t`.
/// Steps before `fallback.0` use `primary`, later steps use `fallback.1`
/// (cover: hints context, then the silence context and non-cover text).
/// With `dcw`, every step (the final one too) is followed by the wavelet
/// correction against `denoised = x_before - v * t`, as upstream's turbo
/// sampler does by default. `noise` is the initial latent `[frames,
/// acoustic]`. `on_step` sees each step index before it runs and may stop
/// the loop by returning false.
#[allow(clippy::too_many_arguments)]
pub fn sample_turbo(
    dit: &Dit,
    noise: &[f32],
    frames: usize,
    schedule: &[f32],
    primary: StepCondition<'_>,
    fallback: Option<(usize, StepCondition<'_>)>,
    dcw: Option<Dcw>,
    mut on_step: impl FnMut(usize) -> bool,
) -> Result<Option<Vec<f32>>> {
    let a = dit.config().audio_acoustic_hidden_dim;
    let dcw = dcw.filter(Dcw::is_active);
    let mut x = noise.to_vec();
    for (i, &t) in schedule.iter().enumerate() {
        if !on_step(i) {
            return Ok(None);
        }
        let cond = match fallback {
            Some((at, f)) if i >= at => f,
            _ => primary,
        };
        let v = dit.forward(&x, cond.context, frames, t, cond.enc, cond.enc_seq)?;
        // denoised = x_before_step - v * t, from the latent before the step.
        let denoised: Option<Vec<f32>> =
            dcw.map(|_| x.iter().zip(&v).map(|(xv, vv)| xv - vv * t).collect());
        if i + 1 == schedule.len() {
            for (xv, vv) in x.iter_mut().zip(&v) {
                *xv -= vv * t;
            }
        } else {
            // dt is a Python float difference of the two f32 schedule
            // values, then cast to f32 by the multiply.
            let dt = (t as f64 - schedule[i + 1] as f64) as f32;
            for (xv, vv) in x.iter_mut().zip(&v) {
                *xv -= vv * dt;
            }
        }
        if let (Some(d), Some(den)) = (dcw, denoised) {
            d.apply(&mut x, &den, frames, a, t)?;
        }
    }
    Ok(Some(x))
}
