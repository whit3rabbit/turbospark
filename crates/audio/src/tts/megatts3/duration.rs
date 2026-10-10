//! MegaTTS3's autoregressive duration LM (`ar_dur/ar_dur_predictor.py`) and the
//! chunk-level duration post-processing of `frontend_function.py`.
//!
//! The model is a pre-LN rotary decoder (8 layers, width 512, 8 heads, GELU
//! FFN, no attention bias) that regresses one duration per phone (`ar_mse`: a
//! Softplus on a 1-wide head, rounded). Its text side is a [`RelTransformer`]
//! plus a tone embedding and a sinusoid position embedding, projected by
//! `enc_proj`. Inference is always incremental:
//!
//! 1. [`DurationLm::prompt`] teacher-forces the prompt's own durations
//!    (`make_dur_prompt`), keeping only the key/value state. The outputs are
//!    discarded, exactly as upstream does.
//! 2. [`DurationLm::predict`] clones that state per text chunk, feeds the
//!    prompt's last duration as the first decoder input, and decodes one
//!    duration per phone (`dur_pred`'s call into `infer`).
//! 3. [`finalize_durations`] applies the jitter, silence floors, and 8-frame
//!    padding of `dur_pred`, and [`length_regulate`] expands to a `mel2ph`.
//!
//! Keys are cached after rotation at their cache index, which is what the
//! reference recomputes on every step; queries rotate at the caller's
//! explicit position (`spk_pos_ids_flat`, equal to the cache length).
//!
//! The checkpoint's `char_encoder.*` and `char_empty_embed` are never used at
//! inference (no character tokens are passed) and are ignored.

use std::path::Path;

use turbospark_model_io::safetensors::SafetensorsFile;

use super::rel_transformer::{sinusoid_positions, RelTransformer, RelTransformerConfig};
use crate::codec::conv::{load_bias, BiasMode};
use crate::codec::wnconv::load_f32_shaped;
use crate::nn::Linear;
use crate::{ops, Result, SpeechError};

const ROPE_BASE: f32 = 10_000.0;
/// `nn.LayerNorm` default.
const LN_EPS: f32 = 1e-5;
/// Upstream truncates the text features at `max_tokens`.
const MAX_TOKENS: usize = 6000;
const MAX_LAYERS: usize = 64;
const MAX_DIM: usize = 8192;

/// Silence and sentence-end phone ids (full stop, exclamation, question, `sil`
/// in the shipped phone dictionary), floored at 64 frames.
const LONG_PAUSE_PHONES: [i32; 4] = [148, 153, 166, 145];
/// Comma / semicolon phone ids floored at 32 frames.
const SHORT_PAUSE_PHONES: [i32; 2] = [163, 165];
/// Frames are padded to a multiple of this (`frames_multiple`).
pub const FRAMES_MULTIPLE: i32 = 8;

#[derive(Debug, Clone, PartialEq)]
pub struct DurationConfig {
    pub hidden: usize,
    pub layers: usize,
    /// Fixed at 8 upstream (`RotTransformerDecoderLayer`'s default).
    pub heads: usize,
    pub ffn_hidden: usize,
    /// Largest duration code; the embedding has `code_size + 2` rows
    /// (0, 1..=code_size, and the start token `code_size + 1`).
    pub code_size: usize,
    pub phone_vocab: usize,
    pub tone_vocab: usize,
    pub enc_filter: usize,
    pub enc_heads: usize,
    pub enc_layers: usize,
    pub enc_kernel: usize,
}

impl DurationConfig {
    /// The shipped MegaTTS3 checkpoint.
    pub fn megatts3() -> Self {
        Self {
            hidden: 512,
            layers: 8,
            heads: 8,
            ffn_hidden: 2048,
            code_size: 128,
            phone_vocab: 174,
            tone_vocab: 23,
            enc_filter: 1024,
            enc_heads: 2,
            enc_layers: 4,
            enc_kernel: 3,
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
            hidden: u("hidden")?,
            layers: u("layers")?,
            heads: u("heads")?,
            ffn_hidden: u("ffn_hidden")?,
            code_size: u("code_size")?,
            phone_vocab: u("phone_vocab")?,
            tone_vocab: u("tone_vocab")?,
            enc_filter: u("enc_filter")?,
            enc_heads: u("enc_heads")?,
            enc_layers: u("enc_layers")?,
            enc_kernel: u("enc_kernel")?,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn validate(&self) -> Result<()> {
        let bad = |f: &str, why: &str| SpeechError::BadConfig {
            field: format!("duration.{f}"),
            why: why.to_string(),
        };
        for (name, v) in [("hidden", self.hidden), ("ffn_hidden", self.ffn_hidden)] {
            if v == 0 || v > MAX_DIM {
                return Err(bad(name, "must be in 1..=8192"));
            }
        }
        if self.layers == 0 || self.layers > MAX_LAYERS {
            return Err(bad("layers", "must be in 1..=64"));
        }
        if self.heads == 0 || self.hidden % self.heads != 0 || (self.hidden / self.heads) % 2 != 0 {
            return Err(bad("heads", "must divide hidden into an even head width"));
        }
        if self.code_size == 0 || self.code_size > 4096 {
            return Err(bad("code_size", "must be in 1..=4096"));
        }
        if self.phone_vocab < 2 || self.tone_vocab < 2 {
            return Err(bad("vocab", "need at least 2 entries"));
        }
        self.enc_config().validate()
    }

    fn enc_config(&self) -> RelTransformerConfig {
        RelTransformerConfig {
            n_vocab: self.phone_vocab,
            hidden: self.hidden,
            filter: self.enc_filter,
            heads: self.enc_heads,
            layers: self.enc_layers,
            kernel: self.enc_kernel,
        }
    }
}

struct Norm {
    weight: Vec<f32>,
    bias: Vec<f32>,
}

impl Norm {
    fn load(file: &SafetensorsFile, prefix: &str, c: usize) -> Result<Self> {
        Ok(Self {
            weight: load_f32_shaped(file, &format!("{prefix}.weight"), &[c])?,
            bias: load_f32_shaped(file, &format!("{prefix}.bias"), &[c])?,
        })
    }

    fn apply(&self, x: &mut [f32], rows: usize) {
        ops::layernorm(
            x,
            rows,
            self.weight.len(),
            &self.weight,
            Some(&self.bias),
            LN_EPS,
        );
    }
}

struct DecLayer {
    ln1: Norm,
    /// Fused `[3 * hidden, hidden]` q/k/v projection, no bias.
    in_proj: Linear,
    out_proj: Linear,
    ln2: Norm,
    ffn_1: Linear,
    ffn_2: Linear,
}

/// Cached keys (already rotated) and values, each `[len, hidden]`.
#[derive(Clone, Default)]
struct LayerKv {
    k: Vec<f32>,
    v: Vec<f32>,
}

/// Decoder key/value state after some number of positions.
#[derive(Clone)]
pub struct DurationState {
    layers: Vec<LayerKv>,
    len: usize,
}

impl DurationState {
    /// Positions cached so far (the prompt's phone count after priming).
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// The primed decoder for one voice prompt.
#[derive(Clone)]
pub struct DurationPrompt {
    state: DurationState,
    /// The prompt's last duration code, the first decoder input of every chunk.
    last_token: i32,
}

impl DurationPrompt {
    pub fn phones(&self) -> usize {
        self.state.len
    }
}

pub struct DurationLm {
    cfg: DurationConfig,
    encoder: RelTransformer,
    tone_embed: Vec<f32>,
    enc_proj: Linear,
    code_embed: Vec<f32>,
    layers: Vec<DecLayer>,
    final_norm: Norm,
    head: Linear,
}

impl DurationLm {
    pub fn open(dir: &Path) -> Result<Self> {
        let file = SafetensorsFile::open(&dir.join("model.safetensors"))?;
        Self::load(DurationConfig::megatts3(), &file)
    }

    pub fn load(cfg: DurationConfig, file: &SafetensorsFile) -> Result<Self> {
        cfg.validate()?;
        let h = cfg.hidden;
        // A kernel-1 conv `[out, 1, in]` is a Linear with the same flat layout.
        let conv1x1 = |name: &str, in_ch: usize, out_ch: usize| -> Result<Linear> {
            let w = load_f32_shaped(file, &format!("{name}.weight"), &[out_ch, 1, in_ch])?;
            let b = load_bias(file, &format!("{name}.bias"), out_ch, BiasMode::Required)?;
            Ok(Linear::new(w, b, in_ch, out_ch))
        };
        let mut layers = Vec::with_capacity(cfg.layers);
        for l in 0..cfg.layers {
            let p = format!("layers.{l}.op");
            layers.push(DecLayer {
                ln1: Norm::load(file, &format!("{p}.layer_norm1"), h)?,
                in_proj: Linear::new(
                    load_f32_shaped(file, &format!("{p}.self_attn.in_proj_weight"), &[3 * h, h])?,
                    None,
                    h,
                    3 * h,
                ),
                out_proj: Linear::load(file, &format!("{p}.self_attn.out_proj"), h, h, false)?,
                ln2: Norm::load(file, &format!("{p}.layer_norm2"), h)?,
                ffn_1: conv1x1(&format!("{p}.ffn.ffn_1.1"), h, cfg.ffn_hidden)?,
                ffn_2: Linear::load(file, &format!("{p}.ffn.ffn_2"), cfg.ffn_hidden, h, true)?,
            });
        }
        Ok(Self {
            encoder: RelTransformer::load(file, "encoder", cfg.enc_config())?,
            tone_embed: load_f32_shaped(
                file,
                "ling_label_embed_layers.tone.weight",
                &[cfg.tone_vocab, h],
            )?,
            enc_proj: Linear::load(file, "enc_proj", h, h, true)?,
            code_embed: load_f32_shaped(file, "code_emb.weight", &[cfg.code_size + 2, h])?,
            layers,
            final_norm: Norm::load(file, "layer_norm", h)?,
            head: Linear::load(file, "project_out_dim.0", h, 1, true)?,
            cfg,
        })
    }

    pub fn config(&self) -> &DurationConfig {
        &self.cfg
    }

    /// `forward_ling_encoder`: phones and tones -> `[N, hidden]` text features.
    pub(crate) fn text_features(&self, phone: &[i32], tone: &[i32]) -> Result<Vec<f32>> {
        let h = self.cfg.hidden;
        if phone.is_empty() || phone.len() != tone.len() {
            return Err(SpeechError::Input {
                why: format!(
                    "phone ({}) and tone ({}) must be equal and non-empty",
                    phone.len(),
                    tone.len()
                ),
            });
        }
        if phone.len() > MAX_TOKENS {
            return Err(SpeechError::Input {
                why: format!(
                    "{} phones exceeds the {MAX_TOKENS} the duration LM accepts",
                    phone.len()
                ),
            });
        }
        if tone
            .iter()
            .any(|&t| t < 0 || t as usize >= self.cfg.tone_vocab)
        {
            return Err(SpeechError::Input {
                why: format!("tone id outside 0..{}", self.cfg.tone_vocab),
            });
        }
        let pos = sinusoid_positions(phone.len(), h);
        let tones = ops::embedding(&self.tone_embed, h, tone);
        let other: Vec<f32> = tones.iter().zip(&pos).map(|(a, b)| a + b).collect();
        let enc = self.encoder.forward(phone, &other)?;
        Ok(self.enc_proj.forward(&enc, phone.len()))
    }

    fn embed_code(&self, code: i32) -> Result<&[f32]> {
        let h = self.cfg.hidden;
        if code < 0 || code as usize >= self.cfg.code_size + 2 {
            // The reference raises IndexError in `code_emb` here.
            return Err(SpeechError::Input {
                why: format!(
                    "duration code {code} outside the embedding (0..={})",
                    self.cfg.code_size + 1
                ),
            });
        }
        Ok(&self.code_embed[code as usize * h..(code as usize + 1) * h])
    }

    /// One decoder position: embeds `code`, adds the text feature, runs every
    /// layer against (and appending to) `state`, and returns the final hidden.
    fn step(
        &self,
        code: i32,
        text: &[f32],
        pos: usize,
        state: &mut DurationState,
    ) -> Result<Vec<f32>> {
        let h = self.cfg.hidden;
        let heads = self.cfg.heads;
        let hd = h / heads;
        let mut x: Vec<f32> = self
            .embed_code(code)?
            .iter()
            .zip(text)
            .map(|(a, b)| a + b)
            .collect();
        let scale = (hd as f32).powf(-0.5);
        let cache_idx = state.len;
        let (qc, qs) = ops::rope_tables_range(pos, 1, hd, ROPE_BASE);
        let (kc, ks) = ops::rope_tables_range(cache_idx, 1, hd, ROPE_BASE);
        for (layer, kv) in self.layers.iter().zip(state.layers.iter_mut()) {
            let mut n = x.clone();
            layer.ln1.apply(&mut n, 1);
            let qkv = layer.in_proj.forward(&n, 1);
            let mut q: Vec<f32> = qkv[..h].iter().map(|v| v * scale).collect();
            let mut k = qkv[h..2 * h].to_vec();
            ops::rope_neox(&mut q, heads, 1, hd, &qc, &qs);
            ops::rope_neox(&mut k, heads, 1, hd, &kc, &ks);
            kv.k.extend_from_slice(&k);
            kv.v.extend_from_slice(&qkv[2 * h..]);
            let len = cache_idx + 1;
            let mut attn = vec![0.0f32; h];
            let mut scores = vec![0.0f32; len];
            for head in 0..heads {
                let c = head * hd;
                let qh = &q[c..c + hd];
                for (t, s) in scores.iter_mut().enumerate() {
                    *s = qh
                        .iter()
                        .zip(&kv.k[t * h + c..t * h + c + hd])
                        .map(|(a, b)| a * b)
                        .sum();
                }
                ops::softmax_row(&mut scores);
                for (t, &p) in scores.iter().enumerate() {
                    for (o, &v) in attn[c..c + hd]
                        .iter_mut()
                        .zip(&kv.v[t * h + c..t * h + c + hd])
                    {
                        *o += p * v;
                    }
                }
            }
            let o = layer.out_proj.forward(&attn, 1);
            for (a, b) in x.iter_mut().zip(&o) {
                *a += b;
            }
            let mut n = x.clone();
            layer.ln2.apply(&mut n, 1);
            let mut f = layer.ffn_1.forward(&n, 1);
            ops::gelu_erf(&mut f);
            let f = layer.ffn_2.forward(&f, 1);
            for (a, b) in x.iter_mut().zip(&f) {
                *a += b;
            }
        }
        state.len += 1;
        self.final_norm.apply(&mut x, 1);
        Ok(x)
    }

    /// Duration regression of one position: `softplus(head(hidden))`.
    fn duration_of(&self, hidden: &[f32]) -> f32 {
        let d = self.head.forward(hidden, 1)[0];
        // F.softplus(beta = 1, threshold = 20)
        if d > 20.0 {
            d
        } else {
            d.exp().ln_1p()
        }
    }

    fn empty_state(&self) -> DurationState {
        DurationState {
            layers: vec![LayerKv::default(); self.cfg.layers],
            len: 0,
        }
    }

    /// `make_dur_prompt`: primes the decoder with the prompt's phones, tones and
    /// duration codes (`prompt_duration_tokens`), teacher-forced.
    pub fn prompt(
        &self,
        phone: &[i32],
        tone: &[i32],
        dur_tokens: &[i32],
    ) -> Result<DurationPrompt> {
        let n = phone.len();
        if dur_tokens.len() != n {
            return Err(SpeechError::Input {
                why: format!("{} duration codes for {n} prompt phones", dur_tokens.len()),
            });
        }
        if dur_tokens
            .iter()
            .any(|&d| d < 1 || d as usize > self.cfg.code_size)
        {
            return Err(SpeechError::Input {
                why: format!(
                    "prompt duration codes must be in 1..={}",
                    self.cfg.code_size
                ),
            });
        }
        let text = self.text_features(phone, tone)?;
        let h = self.cfg.hidden;
        let mut state = self.empty_state();
        for i in 0..n {
            // Step i consumes the start token, then the previous prompt duration.
            let code = if i == 0 {
                self.cfg.code_size as i32 + 1
            } else {
                dur_tokens[i - 1]
            };
            self.step(code, &text[i * h..(i + 1) * h], i, &mut state)?;
        }
        Ok(DurationPrompt {
            state,
            last_token: dur_tokens[n - 1],
        })
    }

    /// One chunk's raw duration codes (`dur_pred`'s call into `infer`). Also
    /// returns the unrounded regression outputs for tracing.
    pub fn predict_with_raw(
        &self,
        prompt: &DurationPrompt,
        phone: &[i32],
        tone: &[i32],
    ) -> Result<(Vec<i32>, Vec<f32>)> {
        let text = self.text_features(phone, tone)?;
        let h = self.cfg.hidden;
        let mut state = prompt.state.clone();
        let base = state.len;
        let mut input = prompt.last_token;
        let mut codes = Vec::with_capacity(phone.len());
        let mut raw = Vec::with_capacity(phone.len());
        for step in 0..phone.len() {
            let hidden = self.step(
                input,
                &text[step * h..(step + 1) * h],
                base + step,
                &mut state,
            )?;
            let d = self.duration_of(&hidden);
            if !d.is_finite() {
                return Err(SpeechError::Input {
                    why: "duration LM produced a non-finite value".to_string(),
                });
            }
            raw.push(d);
            // torch.round ties to even; step 0 may be 0, later steps are >= 1.
            let floor = if step == 0 { 0 } else { 1 };
            let code = (d.round_ties_even() as i64).clamp(floor, i32::MAX as i64) as i32;
            codes.push(code);
            input = code;
        }
        Ok((codes, raw))
    }

    pub fn predict(
        &self,
        prompt: &DurationPrompt,
        phone: &[i32],
        tone: &[i32],
    ) -> Result<Vec<i32>> {
        Ok(self.predict_with_raw(prompt, phone, tone)?.0)
    }
}

/// `make_dur_prompt`'s token construction: frames per phone in the prompt
/// alignment, clamped to `code_size - 1`, plus one (codes `1..=code_size`).
pub fn prompt_duration_tokens(
    mel2ph_ref: &[i32],
    n_phones: usize,
    code_size: usize,
) -> Result<Vec<i32>> {
    let mut counts = vec![0i64; n_phones];
    for (t, &m) in mel2ph_ref.iter().enumerate() {
        if m < 0 || m as usize > n_phones {
            return Err(SpeechError::Input {
                why: format!("mel2ph_ref[{t}] = {m} outside 0..={n_phones}"),
            });
        }
        if m > 0 {
            counts[m as usize - 1] += 1;
        }
    }
    Ok(counts
        .into_iter()
        .map(|c| c.min(code_size as i64 - 1) as i32 + 1)
        .collect())
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChunkOptions {
    /// Jitter strength (`dur_disturb`, 0.1 upstream).
    pub disturb: f32,
    /// Speed factor (`dur_alpha`, 1.0 upstream).
    pub alpha: f32,
    pub is_first: bool,
    pub is_final: bool,
}

impl Default for ChunkOptions {
    fn default() -> Self {
        Self {
            disturb: 0.1,
            alpha: 1.0,
            is_first: true,
            is_final: true,
        }
    }
}

/// The post-processing of `dur_pred` after `infer`: shift to frame counts,
/// edge handling, random jitter, silence floors, and padding to a multiple of
/// [`FRAMES_MULTIPLE`]. `uniform` fills a slice with U[0, 1) draws; it is
/// called twice, for the coin flips then the jitter magnitudes, like the two
/// `torch.rand_like` calls upstream.
pub fn finalize_durations(
    raw: &[i32],
    phone: &[i32],
    code_size: usize,
    opts: ChunkOptions,
    uniform: &mut dyn FnMut(&mut [f32]),
) -> Result<Vec<i32>> {
    let n = raw.len();
    if n == 0 || phone.len() != n {
        return Err(SpeechError::Input {
            why: format!("{} durations for {} phones", n, phone.len()),
        });
    }
    if !(opts.alpha.is_finite()
        && opts.alpha > 0.0
        && opts.disturb.is_finite()
        && opts.disturb >= 0.0)
    {
        return Err(SpeechError::Input {
            why: "alpha must be positive and disturb non-negative".to_string(),
        });
    }
    let max_code = code_size as i32 - 1;
    let mut d: Vec<i32> = raw.iter().map(|&c| (c - 1).clamp(0, max_code)).collect();
    let last = n - 1;
    if opts.is_final {
        d[last] = d[last].clamp(64, 128);
    } else {
        d[last] += 32; // crossfade margin
    }
    let mut choice = vec![0.0f32; n];
    let mut jitter = vec![0.0f32; n];
    uniform(&mut choice);
    uniform(&mut jitter);
    let mut f: Vec<f32> = (0..n)
        .map(|i| {
            let r = 1.0 + jitter[i] * opts.disturb;
            let v = d[i] as f32;
            let c = if choice[i] > 0.5 { 1.0 } else { 0.0 };
            v * r * c + v / r * (1.0 - c)
        })
        .map(|v| (v * opts.alpha).round_ties_even().clamp(0.0, 127.0))
        .collect();
    for (v, &p) in f.iter_mut().zip(phone) {
        if LONG_PAUSE_PHONES.contains(&p) {
            *v = v.max(64.0);
        } else if SHORT_PAUSE_PHONES.contains(&p) {
            *v = v.max(32.0);
        }
    }
    if opts.is_first {
        f[0] = 8.0;
    }
    let total: f32 = f.iter().sum();
    let npad = FRAMES_MULTIPLE as f32 - total % FRAMES_MULTIPLE as f32;
    if npad < FRAMES_MULTIPLE as f32 {
        f[last] += npad;
    }
    Ok(f.into_iter().map(|v| v as i32).collect())
}

/// `LengthRegulator`: phone `i` (1-based) fills frames `[prev_i, cum_i)` where
/// `cum` is the running sum of `dur`. For non-negative durations that is `i`
/// repeated `dur[i - 1]` times. The aligner's duration fix-up can drive a
/// duration negative; the reference then builds an empty interval for that
/// phone and lets the next interval start early, summing the indices of any
/// frames two phones claim. This follows that arithmetic exactly. The output
/// has `sum(dur)` frames (none when the sum is not positive).
pub fn length_regulate(dur: &[i32]) -> Vec<i32> {
    let total: i64 = dur.iter().map(|&d| i64::from(d)).sum();
    if total <= 0 {
        return Vec::new();
    }
    let mut out = vec![0i32; total as usize];
    let mut prev = 0i64;
    for (i, &d) in dur.iter().enumerate() {
        let cum = prev + i64::from(d);
        for p in prev.max(0)..cum.min(total) {
            out[p as usize] += i as i32 + 1;
        }
        prev = cum;
    }
    out
}
