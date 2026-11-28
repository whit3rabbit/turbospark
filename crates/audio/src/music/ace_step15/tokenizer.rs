//! ACE-Step 1.5 audio tokenizer and detokenizer: the 5 Hz FSQ round trip
//! that turns source latents into cover hints.
//!
//! Reference: `AceStepConditionGenerationModel.tokenize` / `detokenize`,
//! `AceStepAudioTokenizer`, `AttentionPooler` and `AudioTokenDetokenizer`
//! in `modeling_acestep_v15_turbo.py` at the pinned checkpoint revision,
//! and `ResidualFSQ` / `FSQ` from vector-quantize-pytorch 1.27.20 with
//! the defaults the tokenizer constructs them with (one quantizer,
//! `bound_hard_clamp=True`, `preserve_symmetry=True`).
//!
//! Layout: time-major `[frames, channels]` latents; tokens are
//! `[frames / pool, hidden]`.

use super::transformer::{linear, rms, rope_tables, AceConfig, EncoderLayer, Loader, Proj};
use crate::nn::{load_tensor, RmsNorm};
use crate::{Result, SpeechError};

/// Single-level finite scalar quantizer with input and output
/// projections (`ResidualFSQ` with `num_quantizers=1`).
#[derive(Clone)]
pub struct Fsq {
    project_in: Proj,
    project_out: Proj,
    levels: Vec<usize>,
}

impl Fsq {
    fn load(l: &Loader<'_>, prefix: &str, c: &AceConfig) -> Result<Self> {
        let file = l.file;
        let n = c.fsq_levels.len();
        // The 6-wide projections are tiny; keep them on the host so the
        // discontinuous floor sees host f32 rounding, as the CPU gate does.
        let l = &Loader::cpu(file);
        Ok(Self {
            project_in: linear(l, &format!("{prefix}.project_in"), c.hidden_size, n, true)?,
            project_out: linear(l, &format!("{prefix}.project_out"), n, c.hidden_size, true)?,
            levels: c.fsq_levels.clone(),
        })
    }

    pub fn codebook_size(&self) -> usize {
        self.levels.iter().product()
    }

    /// Quantizes `x [rows, hidden]`; returns the projected codes
    /// `[rows, hidden]` and one codebook index per row.
    pub fn forward(&self, x: &[f32], rows: usize) -> Result<(Vec<f32>, Vec<i32>)> {
        let n = self.levels.len();
        let mut z = self.project_in.forward(x, rows)?;
        let mut indices = Vec::with_capacity(rows);
        for row in z.chunks_exact_mut(n) {
            let mut index = 0.0f32;
            let mut basis = 1.0f32;
            for (v, &l) in row.iter_mut().zip(&self.levels) {
                let lm1 = (l - 1) as f32;
                // ResidualFSQ soft clamp: tanh(x / c) * c, c = 1 + 1/(L-1).
                let c = 1.0 + 1.0 / lm1;
                let soft = (*v / c).tanh() * c;
                // Layer 0 divides by scale L^-0 = 1.0 (a no-op), then the
                // symmetry-preserving bound with a hard clamp.
                let bracket = (lm1 * (soft.clamp(-1.0, 1.0) + 1.0) / 2.0 + 0.5).floor();
                let code = (2.0 / lm1) * bracket - 1.0;
                // codes_to_indices: level index from the code, then the
                // mixed-radix sum, rounded once at the end.
                index += (code + 1.0) / (2.0 / lm1) * basis;
                basis *= l as f32;
                *v = code;
            }
            indices.push(index.round() as i32);
        }
        Ok((self.project_out.forward(&z, rows)?, indices))
    }
}

/// `AceStepAudioTokenizer`: acoustic projection, attention pooling of
/// each `pool`-frame window into one token, FSQ.
#[derive(Clone)]
pub struct AudioTokenizer {
    acoustic_proj: Proj,
    pool_embed: Proj,
    pool_token: Vec<f32>,
    pool_layers: Vec<EncoderLayer>,
    pool_norm: RmsNorm,
    fsq: Fsq,
    config: AceConfig,
}

/// `AudioTokenDetokenizer`: expands each token into `pool` frames with
/// learned per-position offsets.
#[derive(Clone)]
pub struct AudioDetokenizer {
    embed: Proj,
    special: Vec<f32>,
    layers: Vec<EncoderLayer>,
    norm: RmsNorm,
    proj_out: Proj,
    config: AceConfig,
}

fn layers(l: &Loader<'_>, prefix: &str, c: &AceConfig) -> Result<Vec<EncoderLayer>> {
    (0..c.num_attention_pooler_hidden_layers)
        .map(|i| EncoderLayer::load(l, &format!("{prefix}.layers.{i}"), c, i))
        .collect()
}

/// Runs a short bidirectional stack over each `seq`-row group of `x` in
/// place (`(b t) p c` in the reference: every group is its own sequence
/// with positions `0..seq`).
fn run_groups(
    layers: &[EncoderLayer],
    x: &mut [f32],
    groups: usize,
    seq: usize,
    c: &AceConfig,
) -> Result<()> {
    let (cos, sin) = rope_tables(seq, c.head_dim, c.rope_theta);
    let d = c.hidden_size;
    for g in x.chunks_exact_mut(seq * d).take(groups) {
        for layer in layers {
            layer.forward(g, seq, (&cos, &sin))?;
        }
    }
    Ok(())
}

impl AudioTokenizer {
    pub fn load(l: &Loader<'_>, c: &AceConfig) -> Result<Self> {
        let file = l.file;
        let d = c.hidden_size;
        let p = "tokenizer.attention_pooler";
        Ok(Self {
            acoustic_proj: linear(
                l,
                "tokenizer.audio_acoustic_proj",
                c.audio_acoustic_hidden_dim,
                d,
                true,
            )?,
            pool_embed: linear(l, &format!("{p}.embed_tokens"), d, d, true)?,
            pool_token: load_tensor(file, &format!("{p}.special_token"), &[1, 1, d])?,
            pool_layers: layers(l, p, c)?,
            pool_norm: rms(l, &format!("{p}.norm"), d, c.rms_norm_eps)?,
            fsq: Fsq::load(l, "tokenizer.quantizer", c)?,
            config: c.clone(),
        })
    }

    /// Tokenizes `src [frames, acoustic]`. When `frames` is not a whole
    /// number of windows the tail is padded with the leading frames of
    /// `silence [silence_frames, acoustic]`, as the reference does.
    /// Returns `(tokens [frames.div_ceil(pool), hidden], indices)`.
    pub fn tokenize(
        &self,
        src: &[f32],
        frames: usize,
        silence: &[f32],
    ) -> Result<(Vec<f32>, Vec<i32>)> {
        let c = &self.config;
        let (a, pool, d) = (
            c.audio_acoustic_hidden_dim,
            c.pool_window_size,
            c.hidden_size,
        );
        if frames == 0 || src.len() != frames * a {
            return Err(SpeechError::Input {
                why: format!("source has {} values, expected {frames} x {a}", src.len()),
            });
        }
        let pad = (pool - frames % pool) % pool;
        if silence.len() % a != 0 || silence.len() / a < pad {
            return Err(SpeechError::Input {
                why: format!("silence latent too short to pad {pad} frames"),
            });
        }
        let mut x = src.to_vec();
        x.extend_from_slice(&silence[..pad * a]);
        let windows = (frames + pad) / pool;

        let h = self.acoustic_proj.forward(&x, frames + pad)?;
        let h = self.pool_embed.forward(&h, frames + pad)?;
        // Prepend the special token to every window: [windows, pool + 1, d].
        let seq = pool + 1;
        let mut g = vec![0.0f32; windows * seq * d];
        for w in 0..windows {
            let dst = &mut g[w * seq * d..(w + 1) * seq * d];
            dst[..d].copy_from_slice(&self.pool_token);
            dst[d..].copy_from_slice(&h[w * pool * d..(w + 1) * pool * d]);
        }
        run_groups(&self.pool_layers, &mut g, windows, seq, c)?;
        // Norm is per row, so normalizing only the pooled rows is exact.
        let mut pooled = Vec::with_capacity(windows * d);
        for w in 0..windows {
            pooled.extend_from_slice(&g[w * seq * d..w * seq * d + d]);
        }
        self.pool_norm.apply(&mut pooled, windows);
        self.fsq.forward(&pooled, windows)
    }

    pub fn fsq(&self) -> &Fsq {
        &self.fsq
    }
}

impl AudioDetokenizer {
    pub fn load(l: &Loader<'_>, c: &AceConfig) -> Result<Self> {
        let file = l.file;
        let (d, pool) = (c.hidden_size, c.pool_window_size);
        let p = "detokenizer";
        Ok(Self {
            embed: linear(l, &format!("{p}.embed_tokens"), d, d, true)?,
            special: load_tensor(file, &format!("{p}.special_tokens"), &[1, pool, d])?,
            layers: layers(l, p, c)?,
            norm: rms(l, &format!("{p}.norm"), d, c.rms_norm_eps)?,
            proj_out: linear(
                l,
                &format!("{p}.proj_out"),
                d,
                c.audio_acoustic_hidden_dim,
                true,
            )?,
            config: c.clone(),
        })
    }

    /// `tokens [windows, hidden]` -> `[windows * pool, acoustic]`.
    pub fn detokenize(&self, tokens: &[f32], windows: usize) -> Result<Vec<f32>> {
        let c = &self.config;
        let (pool, d) = (c.pool_window_size, c.hidden_size);
        if windows == 0 || tokens.len() != windows * d {
            return Err(SpeechError::Input {
                why: format!(
                    "tokens have {} values, expected {windows} x {d}",
                    tokens.len()
                ),
            });
        }
        let e = self.embed.forward(tokens, windows)?;
        let mut g = vec![0.0f32; windows * pool * d];
        for w in 0..windows {
            let tok = &e[w * d..(w + 1) * d];
            for p in 0..pool {
                let row = &mut g[(w * pool + p) * d..(w * pool + p + 1) * d];
                let sp = &self.special[p * d..(p + 1) * d];
                for ((r, t), s) in row.iter_mut().zip(tok).zip(sp) {
                    *r = t + s;
                }
            }
        }
        run_groups(&self.layers, &mut g, windows, pool, c)?;
        self.norm.apply(&mut g, windows * pool);
        self.proj_out.forward(&g, windows * pool)
    }
}

/// Cover hints for `src [frames, acoustic]`: tokenize, detokenize, crop
/// back to `frames` (`prepare_condition` with `is_covers=1`).
pub fn cover_hints(
    tokenizer: &AudioTokenizer,
    detokenizer: &AudioDetokenizer,
    src: &[f32],
    frames: usize,
    silence: &[f32],
) -> Result<Vec<f32>> {
    let (tokens, _) = tokenizer.tokenize(src, frames, silence)?;
    let windows = tokens.len() / tokenizer.config.hidden_size;
    let mut hints = detokenizer.detokenize(&tokens, windows)?;
    hints.truncate(frames * tokenizer.config.audio_acoustic_hidden_dim);
    Ok(hints)
}
