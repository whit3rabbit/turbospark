//! MegaTTS3's prompt aligner (`aligner/whisper_small.py` and `align()` in
//! `frontend_function.py`).
//!
//! A small Whisper-architecture model (6 encoder + 6 decoder layers, width
//! 512, vocab 6800) that reads the prompt's 16 kHz log-mel and greedily emits
//! interleaved phone and timestamp tokens: `<BOS 798> phone t phone t ...
//! <EOS 799>`. Timestamps are `800 + frame` at 100 frames per second.
//! [`align_tokens`] turns that stream into the phone/tone/`mel2ph` triple the
//! DiT and the duration LM consume.
//!
//! The reference re-runs the whole decoder over the growing sequence at every
//! step with no cache; this port keeps a key/value cache, which is exactly
//! equivalent for causal attention with absolute position embeddings. The
//! encoder positional table and every other tensor come from the checkpoint
//! (the encoder table is stored even though upstream registers sinusoids).

use std::path::Path;

use turbospark_model_io::safetensors::SafetensorsFile;

use super::duration::FRAMES_MULTIPLE;
use crate::codec::conv::{load_bias, load_mlx_conv_weight, BiasMode, Conv1d};
use crate::codec::wnconv::load_f32_shaped;
use crate::nn::Linear;
use crate::whisper::whisper_log_mel;
use crate::{ops, Result, SpeechError};

/// Start and end tokens of the alignment stream.
pub const BOS_TOKEN: i32 = 798;
pub const EOS_TOKEN: i32 = 799;
/// Greedy decode budget of the reference loop (`range(768)`).
pub const MAX_NEW_TOKENS: usize = 768;
/// Timestamp tokens are `TIME_BASE + frame`.
const TIME_BASE: i32 = 800;
const LN_EPS: f32 = 1e-5;
const MAX_LAYERS: usize = 64;
const MAX_STATE: usize = 4096;

#[derive(Debug, Clone, PartialEq)]
pub struct AlignerConfig {
    pub n_mels: usize,
    pub state: usize,
    /// Fixed at 8 upstream; not derivable from tensor shapes.
    pub heads: usize,
    pub enc_layers: usize,
    pub dec_layers: usize,
    pub enc_ctx: usize,
    pub dec_ctx: usize,
    pub vocab: usize,
}

impl AlignerConfig {
    /// The shipped MegaTTS3 checkpoint.
    pub fn megatts3() -> Self {
        Self {
            n_mels: 80,
            state: 512,
            heads: 8,
            enc_layers: 6,
            dec_layers: 6,
            enc_ctx: 3000,
            dec_ctx: 2048,
            vocab: 6800,
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
            n_mels: u("n_mels")?,
            state: u("state")?,
            heads: u("heads")?,
            enc_layers: u("enc_layers")?,
            dec_layers: u("dec_layers")?,
            enc_ctx: u("enc_ctx")?,
            dec_ctx: u("dec_ctx")?,
            vocab: u("vocab")?,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn validate(&self) -> Result<()> {
        let bad = |f: &str, why: &str| SpeechError::BadConfig {
            field: format!("aligner.{f}"),
            why: why.to_string(),
        };
        if self.n_mels != 80 {
            return Err(bad("n_mels", "the log-mel frontend produces 80 bands"));
        }
        if self.state == 0 || self.state > MAX_STATE {
            return Err(bad("state", "must be in 1..=4096"));
        }
        if self.heads == 0 || self.state % self.heads != 0 {
            return Err(bad("heads", "must divide state"));
        }
        for (name, n) in [
            ("enc_layers", self.enc_layers),
            ("dec_layers", self.dec_layers),
        ] {
            if n == 0 || n > MAX_LAYERS {
                return Err(bad(name, "must be in 1..=64"));
            }
        }
        if self.enc_ctx == 0
            || self.enc_ctx > 1 << 16
            || self.dec_ctx == 0
            || self.dec_ctx > 1 << 16
        {
            return Err(bad("ctx", "must be in 1..=65536"));
        }
        if self.vocab as i32 <= EOS_TOKEN || self.vocab > 1 << 20 {
            return Err(bad(
                "vocab",
                "must exceed the end token and stay below 2^20",
            ));
        }
        Ok(())
    }
}

struct Ln {
    weight: Vec<f32>,
    bias: Vec<f32>,
}

impl Ln {
    fn load(file: &SafetensorsFile, prefix: &str, c: usize) -> Result<Self> {
        Ok(Self {
            weight: load_f32_shaped(file, &format!("{prefix}.weight"), &[c])?,
            bias: load_f32_shaped(file, &format!("{prefix}.bias"), &[c])?,
        })
    }

    fn apply(&self, x: &[f32], rows: usize) -> Vec<f32> {
        let mut out = x.to_vec();
        ops::layernorm(
            &mut out,
            rows,
            self.weight.len(),
            &self.weight,
            Some(&self.bias),
            LN_EPS,
        );
        out
    }
}

/// `MultiHeadAttention`: query/value/out have a bias, key does not.
struct Attn {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
}

impl Attn {
    fn load(file: &SafetensorsFile, prefix: &str, c: usize) -> Result<Self> {
        Ok(Self {
            q: Linear::load(file, &format!("{prefix}.query"), c, c, true)?,
            k: Linear::load(file, &format!("{prefix}.key"), c, c, false)?,
            v: Linear::load(file, &format!("{prefix}.value"), c, c, true)?,
            o: Linear::load(file, &format!("{prefix}.out"), c, c, true)?,
        })
    }
}

struct Mlp {
    up: Linear,
    down: Linear,
}

impl Mlp {
    fn load(file: &SafetensorsFile, prefix: &str, c: usize) -> Result<Self> {
        Ok(Self {
            up: Linear::load(file, &format!("{prefix}.mlp.0"), c, 4 * c, true)?,
            down: Linear::load(file, &format!("{prefix}.mlp.2"), 4 * c, c, true)?,
        })
    }

    fn forward(&self, x: &[f32], rows: usize) -> Vec<f32> {
        let mut h = self.up.forward(x, rows);
        ops::gelu_erf(&mut h);
        self.down.forward(&h, rows)
    }
}

struct EncBlock {
    attn: Attn,
    attn_ln: Ln,
    mlp: Mlp,
    mlp_ln: Ln,
}

struct DecBlock {
    attn: Attn,
    attn_ln: Ln,
    cross: Attn,
    cross_ln: Ln,
    mlp: Mlp,
    mlp_ln: Ln,
}

pub struct Aligner {
    cfg: AlignerConfig,
    conv1: Conv1d,
    conv2: Conv1d,
    enc_pos: Vec<f32>,
    enc_blocks: Vec<EncBlock>,
    ln_post: Ln,
    tok_emb: Vec<f32>,
    dec_pos: Vec<f32>,
    dec_blocks: Vec<DecBlock>,
    dec_ln: Ln,
    out_proj: Linear,
}

/// Encoder output `[frames, state]` with the per-layer cross-attention
/// keys/values the decoder reuses at every step.
pub struct AudioFeatures {
    frames: usize,
    /// The encoder output `[frames, state]`.
    out: Vec<f32>,
    cross: Vec<(Vec<f32>, Vec<f32>)>,
}

impl AudioFeatures {
    pub fn frames(&self) -> usize {
        self.frames
    }

    /// The encoder output `[frames, state]`.
    pub fn data(&self) -> &[f32] {
        &self.out
    }
}

/// Decoder self-attention cache: per layer `[len, state]` keys and values.
struct DecState {
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    len: usize,
}

fn conv_k(
    file: &SafetensorsFile,
    name: &str,
    in_ch: usize,
    out_ch: usize,
    stride: usize,
) -> Result<Conv1d> {
    Ok(Conv1d {
        in_ch,
        out_ch,
        kernel: 3,
        stride,
        padding: 1,
        dilation: 1,
        groups: 1,
        weight: load_mlx_conv_weight(file, &format!("{name}.weight"), out_ch, 3, in_ch)?,
        bias: load_bias(file, &format!("{name}.bias"), out_ch, BiasMode::Required)?,
    })
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

impl Aligner {
    pub fn open(dir: &Path) -> Result<Self> {
        let file = SafetensorsFile::open(&dir.join("model.safetensors"))?;
        Self::load(AlignerConfig::megatts3(), &file)
    }

    pub fn load(cfg: AlignerConfig, file: &SafetensorsFile) -> Result<Self> {
        cfg.validate()?;
        let c = cfg.state;
        let mut enc_blocks = Vec::new();
        for l in 0..cfg.enc_layers {
            let p = format!("encoder.blocks.{l}");
            enc_blocks.push(EncBlock {
                attn: Attn::load(file, &format!("{p}.attn"), c)?,
                attn_ln: Ln::load(file, &format!("{p}.attn_ln"), c)?,
                mlp: Mlp::load(file, &p, c)?,
                mlp_ln: Ln::load(file, &format!("{p}.mlp_ln"), c)?,
            });
        }
        let mut dec_blocks = Vec::new();
        for l in 0..cfg.dec_layers {
            let p = format!("decoder.blocks.{l}");
            dec_blocks.push(DecBlock {
                attn: Attn::load(file, &format!("{p}.attn"), c)?,
                attn_ln: Ln::load(file, &format!("{p}.attn_ln"), c)?,
                cross: Attn::load(file, &format!("{p}.cross_attn"), c)?,
                cross_ln: Ln::load(file, &format!("{p}.cross_attn_ln"), c)?,
                mlp: Mlp::load(file, &p, c)?,
                mlp_ln: Ln::load(file, &format!("{p}.mlp_ln"), c)?,
            });
        }
        Ok(Self {
            conv1: conv_k(file, "encoder.conv1", cfg.n_mels, c, 1)?,
            conv2: conv_k(file, "encoder.conv2", c, c, 2)?,
            enc_pos: load_f32_shaped(file, "encoder.positional_embedding", &[cfg.enc_ctx, c])?,
            enc_blocks,
            ln_post: Ln::load(file, "encoder.ln_post", c)?,
            tok_emb: load_f32_shaped(file, "decoder.token_embedding.weight", &[cfg.vocab, c])?,
            dec_pos: load_f32_shaped(file, "decoder.positional_embedding", &[cfg.dec_ctx, c])?,
            dec_blocks,
            dec_ln: Ln::load(file, "decoder.ln", c)?,
            out_proj: Linear::load(file, "decoder.out_proj", c, cfg.vocab, true)?,
            cfg,
        })
    }

    pub fn config(&self) -> &AlignerConfig {
        &self.cfg
    }

    /// `embed_audio`: log-mel `[n_mels, frames]` (channel-major) -> features.
    pub fn encode(&self, mel: &[f32], frames: usize) -> Result<AudioFeatures> {
        let c = self.cfg.state;
        if frames == 0 || mel.len() != self.cfg.n_mels * frames {
            return Err(SpeechError::Input {
                why: format!(
                    "mel has {} values, expected {} x {frames}",
                    mel.len(),
                    self.cfg.n_mels
                ),
            });
        }
        let mut h = self.conv1.forward(mel);
        ops::gelu_erf(&mut h);
        let mut h = self.conv2.forward(&h);
        ops::gelu_erf(&mut h);
        let t = h.len() / c;
        if t > self.cfg.enc_ctx {
            return Err(SpeechError::Input {
                why: format!(
                    "{t} encoder frames exceed the {} positions",
                    self.cfg.enc_ctx
                ),
            });
        }
        let mut x = transpose(&h, c, t);
        for (v, p) in x.iter_mut().zip(&self.enc_pos[..t * c]) {
            *v += p;
        }
        let heads = self.cfg.heads;
        let hd = c / heads;
        for b in &self.enc_blocks {
            let n = b.attn_ln.apply(&x, t);
            let q = ops::split_heads(&b.attn.q.forward(&n, t), t, heads, hd);
            let k = ops::split_heads(&b.attn.k.forward(&n, t), t, heads, hd);
            let v = ops::split_heads(&b.attn.v.forward(&n, t), t, heads, hd);
            let a = ops::mha(&q, &k, &v, None, heads, t, t, hd, 1.0 / (hd as f32).sqrt());
            let o = b.attn.o.forward(&ops::merge_heads(&a, t, heads, hd), t);
            for (v, a) in x.iter_mut().zip(&o) {
                *v += a;
            }
            let n = b.mlp_ln.apply(&x, t);
            let m = b.mlp.forward(&n, t);
            for (v, a) in x.iter_mut().zip(&m) {
                *v += a;
            }
        }
        let feats = self.ln_post.apply(&x, t);
        let cross = self
            .dec_blocks
            .iter()
            .map(|b| (b.cross.k.forward(&feats, t), b.cross.v.forward(&feats, t)))
            .collect();
        Ok(AudioFeatures {
            frames: t,
            out: feats,
            cross,
        })
    }

    fn new_state(&self) -> DecState {
        DecState {
            k: vec![Vec::new(); self.cfg.dec_layers],
            v: vec![Vec::new(); self.cfg.dec_layers],
            len: 0,
        }
    }

    /// One decoder position; returns the logits `[vocab]`.
    fn step(&self, token: i32, feats: &AudioFeatures, st: &mut DecState) -> Result<Vec<f32>> {
        let c = self.cfg.state;
        let heads = self.cfg.heads;
        let hd = c / heads;
        if token < 0 || token as usize >= self.cfg.vocab {
            return Err(SpeechError::Input {
                why: format!("token {token} outside the vocabulary of {}", self.cfg.vocab),
            });
        }
        if st.len >= self.cfg.dec_ctx {
            return Err(SpeechError::Input {
                why: format!(
                    "decoder context of {} positions exhausted",
                    self.cfg.dec_ctx
                ),
            });
        }
        let mut x: Vec<f32> = self.tok_emb[token as usize * c..(token as usize + 1) * c]
            .iter()
            .zip(&self.dec_pos[st.len * c..(st.len + 1) * c])
            .map(|(a, b)| a + b)
            .collect();
        let scale = 1.0 / (hd as f32).sqrt();
        for (l, b) in self.dec_blocks.iter().enumerate() {
            let n = b.attn_ln.apply(&x, 1);
            st.k[l].extend_from_slice(&b.attn.k.forward(&n, 1));
            st.v[l].extend_from_slice(&b.attn.v.forward(&n, 1));
            let q = b.attn.q.forward(&n, 1);
            let o = attend(&q, &st.k[l], &st.v[l], st.len + 1, heads, hd, scale);
            let o = b.attn.o.forward(&o, 1);
            for (v, a) in x.iter_mut().zip(&o) {
                *v += a;
            }
            let n = b.cross_ln.apply(&x, 1);
            let q = b.cross.q.forward(&n, 1);
            let (ck, cv) = &feats.cross[l];
            let o = attend(&q, ck, cv, feats.frames, heads, hd, scale);
            let o = b.cross.o.forward(&o, 1);
            for (v, a) in x.iter_mut().zip(&o) {
                *v += a;
            }
            let n = b.mlp_ln.apply(&x, 1);
            let m = b.mlp.forward(&n, 1);
            for (v, a) in x.iter_mut().zip(&m) {
                *v += a;
            }
        }
        st.len += 1;
        let n = self.dec_ln.apply(&x, 1);
        Ok(self.out_proj.forward(&n, 1))
    }

    /// Teacher-forced logits for every position of `tokens` (`[n, vocab]`).
    /// Equals the reference's full-sequence `logits(token, features)`.
    pub fn logits_for(&self, feats: &AudioFeatures, tokens: &[i32]) -> Result<Vec<f32>> {
        let mut st = self.new_state();
        let mut out = Vec::with_capacity(tokens.len() * self.cfg.vocab);
        for &t in tokens {
            out.extend(self.step(t, feats, &mut st)?);
        }
        Ok(out)
    }

    /// Greedy decode from [`BOS_TOKEN`]: stops after emitting [`EOS_TOKEN`] or
    /// `max_new` tokens. The returned stream includes the BOS (and the EOS
    /// when reached), like the reference's `alignment_tokens`.
    pub fn generate(&self, feats: &AudioFeatures, max_new: usize) -> Result<Vec<i32>> {
        let mut st = self.new_state();
        let mut tokens = vec![BOS_TOKEN];
        let mut last = BOS_TOKEN;
        for _ in 0..max_new {
            let logits = self.step(last, feats, &mut st)?;
            // torch.argmax over the softmax: the first maximum wins.
            let mut best = 0usize;
            for (i, &v) in logits.iter().enumerate() {
                if v > logits[best] {
                    best = i;
                }
            }
            last = best as i32;
            tokens.push(last);
            if last == EOS_TOKEN {
                break;
            }
        }
        Ok(tokens)
    }
}

/// Single-query multi-head attention over `[len, c]` keys and values.
fn attend(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    len: usize,
    heads: usize,
    hd: usize,
    scale: f32,
) -> Vec<f32> {
    let c = heads * hd;
    let mut out = vec![0.0f32; c];
    let mut scores = vec![0.0f32; len];
    for h in 0..heads {
        let o = h * hd;
        let qh = &q[o..o + hd];
        for (t, s) in scores.iter_mut().enumerate() {
            *s = qh
                .iter()
                .zip(&k[t * c + o..t * c + o + hd])
                .map(|(a, b)| a * b)
                .sum::<f32>()
                * scale;
        }
        ops::softmax_row(&mut scores);
        for (t, &p) in scores.iter().enumerate() {
            for (acc, &x) in out[o..o + hd].iter_mut().zip(&v[t * c + o..t * c + o + hd]) {
                *acc += p * x;
            }
        }
    }
    out
}

// ------------------------------------------------------------- frontend

/// `win_size` of the DiT config: the prompt is padded to this STFT window.
const WIN_SIZE: usize = 1200;
/// Trailing silence `preprocess` appends to the 24 kHz prompt.
const TAIL_PAD: usize = 12_000;
/// Prompt sample rate and the aligner's input rate.
pub const PROMPT_RATE: u32 = 24_000;
pub const ALIGNER_RATE: u32 = 16_000;

/// `preprocess`'s zero padding of the 24 kHz prompt: up to one sample short of
/// a multiple of the 1200-sample window, then 12000 more zeros.
pub fn pad_prompt_audio(wav_24k: &[f32]) -> Vec<f32> {
    let mut out = wav_24k.to_vec();
    if out.len() % WIN_SIZE < WIN_SIZE - 1 {
        out.resize(out.len() + (WIN_SIZE - 1 - out.len() % WIN_SIZE), 0.0);
    }
    out.resize(out.len() + TAIL_PAD, 0.0);
    out
}

/// 24 kHz prompt -> 16 kHz aligner input with the crate's sinc-Hann
/// polyphase resampler.
///
/// The reference uses librosa's `soxr_hq`, which this crate does not ship (see
/// `resample.rs`). A scan against soxr on real speech found its response close
/// to an ideal low-pass at 0.99 of Nyquist with a long filter: the torchaudio
/// default (width 6) leaves a 1.8e-2 waveform gap, width 64 brings it to 4e-4,
/// and lowering the rolloff to mimic soxr's nominal 91% passband made it worse.
/// The alignment is not bit-identical to the soxr path (some timestamps move by
/// one 10 ms frame); `tests_aligner.rs` pins the measured behavior.
pub fn resample_for_aligner(wav_24k: &[f32]) -> Result<Vec<f32>> {
    let options = crate::resample::SincHannOptions {
        lowpass_filter_width: 64,
        ..crate::resample::SincHannOptions::default()
    };
    crate::resample::resample_mono_sinc_hann(wav_24k, PROMPT_RATE, ALIGNER_RATE, &options)
        .map_err(|e| SpeechError::Audio(e.to_string()))
}

/// The prompt's log-mel as the aligner sees it: Whisper's 80-band log-mel of
/// the 16 kHz audio (no 30 s padding), channel-major `[80, frames]`, cut to a
/// multiple of [`FRAMES_MULTIPLE`]. Returns `(mel, frames)`.
pub fn prompt_mel(wav_16k: &[f32]) -> Result<(Vec<f32>, usize)> {
    if wav_16k.is_empty() || wav_16k.iter().any(|v| !v.is_finite()) {
        return Err(SpeechError::Input {
            why: "prompt audio is empty or contains NaN/inf".to_string(),
        });
    }
    let rows = whisper_log_mel(wav_16k, 80).map_err(|e| SpeechError::Audio(e.to_string()))?;
    let frames = rows.len() / FRAMES_MULTIPLE as usize * FRAMES_MULTIPLE as usize;
    if frames == 0 {
        return Err(SpeechError::Input {
            why: "prompt audio is too short to align".to_string(),
        });
    }
    let mut mel = vec![0.0f32; 80 * frames];
    for (t, row) in rows.iter().take(frames).enumerate() {
        for (m, &v) in row.iter().enumerate() {
            mel[m * frames + t] = v;
        }
    }
    Ok((mel, frames))
}

// --------------------------------------------------------------- parsing

/// The aligner's output for one prompt: phone ids, tone ids, and the 100 fps
/// phone index per frame (1-based), already a multiple of 8 frames long.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AlignedPrompt {
    pub phone: Vec<i32>,
    pub tone: Vec<i32>,
    pub mel2ph: Vec<i32>,
}

/// `split_ph_timestamp`: `[phone, t, phone, t, ...]` -> phones, tones, and
/// durations (differences of consecutive timestamps).
///
/// Chinese phones fold phone and tone into one token in `200..=788`
/// (`(id - 201) / 6 + 3`, tone from the remainder); every other phone id is an
/// English/punctuation phone with tone 3.
pub fn split_ph_timestamp(tokens: &[i32]) -> Result<(Vec<i32>, Vec<i32>, Vec<i32>)> {
    let mut ph = Vec::new();
    let mut tone = Vec::new();
    let mut dur = Vec::new();
    let mut cur = 0i32;
    for (i, &raw) in tokens.iter().enumerate() {
        let item = if raw >= TIME_BASE {
            raw - TIME_BASE
        } else {
            raw
        };
        if i % 2 == 0 {
            if (200..=788).contains(&item) {
                ph.push((item - 200 - 1).div_euclid(6) + 3);
                let t = (item - 200 - 1).rem_euclid(6) + 1;
                tone.push(if t == 1 { 4 } else { t + 9 });
            } else {
                ph.push(item);
                tone.push(3);
            }
        } else {
            dur.push(item - cur);
            cur = item;
        }
    }
    if ph.len() != dur.len() {
        // The reference asserts equal lengths; an odd stream is a truncated decode.
        return Err(SpeechError::Input {
            why: format!(
                "alignment stream has {} phones but {} timestamps",
                ph.len(),
                dur.len()
            ),
        });
    }
    Ok((ph, tone, dur))
}

/// The duration fix-ups of `align()`: pad the last phone up to the mel frame
/// count, or take one frame at a time off each phone in turn (which can drive
/// short phones negative, as upstream does) until the total fits.
fn fit_durations(dur: &mut [i32], frames: i32) {
    let sum: i32 = dur.iter().sum();
    if sum < frames {
        if let Some(last) = dur.last_mut() {
            *last += frames - sum;
        }
    } else if sum > frames {
        let mut diff = sum - frames;
        'outer: loop {
            for d in dur.iter_mut() {
                *d -= 1;
                diff -= 1;
                if diff == 0 {
                    break 'outer;
                }
            }
        }
    }
}

/// Parses a decoded stream (as returned by [`Aligner::generate`], with its
/// BOS and final token) into an [`AlignedPrompt`] for a prompt of
/// `mel_frames` frames (already a multiple of 8).
pub fn align_tokens(stream: &[i32], mel_frames: usize) -> Result<AlignedPrompt> {
    if stream.len() < 4 {
        return Err(SpeechError::Input {
            why: "alignment stream too short".to_string(),
        });
    }
    // The reference always drops the first and the last token.
    let (phone, tone, mut dur) = split_ph_timestamp(&stream[1..stream.len() - 1])?;
    if phone.is_empty() {
        return Err(SpeechError::Input {
            why: "aligner produced no phones".to_string(),
        });
    }
    fit_durations(&mut dur, mel_frames as i32);
    let mut mel2ph = super::duration::length_regulate(&dur);
    mel2ph.truncate(mel2ph.len() / FRAMES_MULTIPLE as usize * FRAMES_MULTIPLE as usize);
    Ok(AlignedPrompt {
        phone,
        tone,
        mel2ph,
    })
}
