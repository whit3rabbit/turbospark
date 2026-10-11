//! `MegaTts3`: the end-to-end voice-cloning pipeline
//! (`MegaTTS3DiTInfer.preprocess` and `.forward`).
//!
//! 1. [`MegaTts3::prepare_prompt`] (`preprocess`): pad the 24 kHz prompt,
//!    measure its loudness, align it (phones, tones, `mel2ph`), encode it to a
//!    latent with the WavVAE encoder, and prime the duration LM.
//! 2. [`MegaTts3::synthesize`] (`forward`): normalize and chunk the text, then
//!    per chunk run G2P, durations, the DiT sampler and the WavVAE decoder, trim
//!    the prompt region, match the prompt's loudness, and crossfade the chunks.
//!
//! Randomness is injected ([`Randomness`]) in the reference's draw order, so a
//! recorded stream reproduces a PyTorch run. English text only; see
//! [`super::text`] for the normalizer's scope.

use std::path::Path;

use super::aligner::{
    align_tokens, pad_prompt_audio, prompt_mel, resample_for_aligner, Aligner,
    MAX_NEW_TOKENS as ALIGNER_MAX_NEW, PROMPT_RATE,
};
use super::dit::Dit;
use super::duration::{
    finalize_durations, length_regulate, prompt_duration_tokens, ChunkOptions, DurationLm,
    DurationPrompt, FRAMES_MULTIPLE,
};
use super::g2p::G2p;
use super::postprocess::{integrated_loudness, match_loudness, Crossfader};
use super::rng::Randomness;
use super::sampler::{sample_controlled, SamplerParams};
use super::text::{chunk_english, ensure_english, normalize_english, DEFAULT_MAX_CHARS};
use crate::codec::wavvae::WavVae;
use crate::resample::{resample_mono_sinc_hann, SincHannOptions};
use crate::{Result, SpeechError};

/// Output and prompt sample rate.
pub const SAMPLE_RATE: u32 = PROMPT_RATE;
/// `model_type` of the root `config.json` the converter writes; runtimes
/// dispatch on it.
pub const MODEL_TYPE: &str = "megatts3";
/// The Space cuts prompts to 28 s (`cut_wav(max_len=28)`).
pub const MAX_PROMPT_SECONDS: usize = 28;
/// Latent frames cover this many output samples (`vae_stride * hop_size`).
const SAMPLES_PER_LATENT: usize = 960;
/// Mel frames per latent frame.
const FRAMES_PER_LATENT: usize = 4;
/// Longest output the DiT is asked for in one pass (latent frames, about 164 s).
const MAX_LATENT_FRAMES: usize = 4096;

#[derive(Debug, Clone, PartialEq)]
pub struct SynthParams {
    /// Euler steps of the sampler (`time_step`).
    pub steps: usize,
    /// Intelligibility weight.
    pub p_w: f32,
    /// Similarity weight.
    pub t_w: f32,
    /// Duration jitter strength (`dur_disturb`).
    pub dur_disturb: f32,
    /// Speed factor (`dur_alpha`).
    pub dur_alpha: f32,
    /// UTF-8 byte budget per text chunk.
    pub max_chunk_bytes: usize,
}

impl Default for SynthParams {
    /// The Space's guidance and duration settings (the upstream CLI's are
    /// `p_w` 1.6, `t_w` 2.5) at Balanced steps. The README's step study found
    /// no intelligibility or speaker-similarity loss at 16 steps against the
    /// upstream 32 (`Quality::Best`).
    fn default() -> Self {
        Self {
            steps: Quality::Balanced.steps(),
            p_w: 1.4,
            t_w: 3.0,
            dur_disturb: 0.1,
            dur_alpha: 1.0,
            max_chunk_bytes: DEFAULT_MAX_CHARS,
        }
    }
}

/// Named step counts for a speed and quality choice. Each step is one DiT
/// pass over the three guidance rows, so time scales with steps; the README's
/// step study measured what fewer steps cost.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Quality {
    Fast,
    Balanced,
    Best,
}

impl Quality {
    pub const ALL: [Self; 3] = [Self::Fast, Self::Balanced, Self::Best];

    pub const fn steps(self) -> usize {
        match self {
            Self::Fast => 8,
            Self::Balanced => 16,
            Self::Best => 32,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Fast => "fast",
            Self::Balanced => "balanced",
            Self::Best => "best",
        }
    }

    pub fn parse(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|q| q.name() == name)
    }
}

/// Everything derived from one voice prompt.
#[derive(Clone)]
pub struct VoicePrompt {
    pub phone: Vec<i32>,
    pub tone: Vec<i32>,
    pub mel2ph: Vec<i32>,
    /// Prompt latent `[frames, 32]`.
    pub latent: Vec<f32>,
    pub(crate) duration: DurationPrompt,
    /// Integrated loudness of the padded prompt, the target of every chunk.
    pub loudness: f64,
}

impl VoicePrompt {
    pub fn latent_frames(&self) -> usize {
        self.latent.len() / 32
    }
}

pub struct MegaTts3 {
    aligner: Aligner,
    wavvae: WavVae,
    dit: Dit,
    duration: DurationLm,
    g2p: G2p,
}

impl MegaTts3 {
    /// Opens a converted checkpoint directory (`tools/convert_megatts3.py`):
    /// `aligner_lm/`, `diffusion_transformer/`, `duration_lm/`, `wavvae/`, `g2p/`.
    /// A root `config.json`, when present, must name [`MODEL_TYPE`] and
    /// declare `voice_clone` matching whether the WavVAE has its encoder.
    pub fn open(dir: &Path) -> Result<Self> {
        Self::open_with_backend(dir, None)
    }

    /// [`open`](Self::open) with the DiT's projections and attention and the
    /// WavVAE decoder's convolutions on `backend` (a runtime's device). The
    /// aligner, duration LM, G2P and WavVAE encoder stay on the host.
    pub fn open_with_backend(
        dir: &Path,
        backend: Option<std::rc::Rc<dyn crate::backend::ComputeBackend>>,
    ) -> Result<Self> {
        let wavvae = WavVae::open(&dir.join("wavvae"))?.with_backend(backend.clone());
        check_root_config(dir, wavvae.has_encoder())?;
        Ok(Self {
            aligner: Aligner::open(&dir.join("aligner_lm"))?,
            wavvae,
            dit: Dit::open(&dir.join("diffusion_transformer"))?.with_backend(backend),
            duration: DurationLm::open(&dir.join("duration_lm"))?,
            g2p: G2p::open(&dir.join("g2p"))?,
        })
    }

    /// Whether the DiT runs on a device backend.
    pub fn using_device(&self) -> bool {
        self.dit.on_device()
    }

    /// Assembles a pipeline from already-loaded parts.
    pub fn from_parts(
        aligner: Aligner,
        wavvae: WavVae,
        dit: Dit,
        duration: DurationLm,
        g2p: G2p,
    ) -> Self {
        Self {
            aligner,
            wavvae,
            dit,
            duration,
            g2p,
        }
    }

    /// Whether voice cloning from a raw waveform is possible (the official
    /// decoder-only WavVAE has no encoder).
    pub fn can_clone(&self) -> bool {
        self.wavvae.has_encoder()
    }

    /// `preprocess`: mono 24 kHz prompt audio, at most [`MAX_PROMPT_SECONDS`]
    /// are used.
    pub fn prepare_prompt(&self, wav_24k: &[f32], rng: &mut dyn Randomness) -> Result<VoicePrompt> {
        self.prepare_prompt_with(wav_24k, None, rng)
    }

    /// [`prepare_prompt`](Self::prepare_prompt) with the 16 kHz aligner input
    /// supplied instead of resampled, to isolate everything downstream of the
    /// resampler (the reference uses soxr, which this crate does not ship).
    pub(crate) fn prepare_prompt_with(
        &self,
        wav_24k: &[f32],
        wav_16k: Option<&[f32]>,
        rng: &mut dyn Randomness,
    ) -> Result<VoicePrompt> {
        if !self.can_clone() {
            return Err(SpeechError::Unsupported {
                why: "this WavVAE checkpoint has no encoder (official decoder-only release); \
                      voice cloning from a waveform needs the community encoder"
                    .to_string(),
            });
        }
        if wav_24k.is_empty() || wav_24k.iter().any(|v| !v.is_finite()) {
            return Err(SpeechError::Input {
                why: "prompt audio is empty or contains NaN/inf".to_string(),
            });
        }
        let used = &wav_24k[..wav_24k.len().min(MAX_PROMPT_SECONDS * SAMPLE_RATE as usize)];
        let padded = pad_prompt_audio(used);
        let loudness = integrated_loudness(&padded, SAMPLE_RATE)?;
        if !loudness.is_finite() {
            return Err(SpeechError::Input {
                why: "prompt audio is silent".to_string(),
            });
        }

        let resampled;
        let wav16 = match wav_16k {
            Some(w) => w,
            None => {
                resampled = resample_for_aligner(&padded)?;
                &resampled
            }
        };
        let (mel, frames) = prompt_mel(wav16)?;
        let feats = self.aligner.encode(&mel, frames)?;
        let stream = self.aligner.generate(&feats, ALIGNER_MAX_NEW)?;
        let aligned = align_tokens(&stream, frames)?;

        let posterior = self.wavvae.encode(&padded)?;
        let mut eps = vec![0.0f32; posterior.mean.len()];
        rng.normal(&mut eps);
        let latent = posterior.sample(&eps)?;
        let keep = (aligned.mel2ph.len() / FRAMES_PER_LATENT).min(posterior.frames);
        let latent = latent[..keep * posterior.z_dim].to_vec();

        let tokens = prompt_duration_tokens(
            &aligned.mel2ph,
            aligned.phone.len(),
            self.duration.config().code_size,
        )?;
        let duration = self
            .duration
            .prompt(&aligned.phone, &aligned.tone, &tokens)?;
        Ok(VoicePrompt {
            phone: aligned.phone,
            tone: aligned.tone,
            mel2ph: aligned.mel2ph,
            latent,
            duration,
            loudness,
        })
    }

    /// `forward`: synthesizes `text` in the prompt's voice at 24 kHz.
    pub fn synthesize(
        &mut self,
        prompt: &VoicePrompt,
        text: &str,
        params: &SynthParams,
        rng: &mut dyn Randomness,
    ) -> Result<Vec<f32>> {
        let mut out = Vec::new();
        self.synthesize_controlled(prompt, text, params, rng, &mut |_| Ok(()), &mut |pcm| {
            out.extend(pcm);
            true
        })?;
        Ok(out)
    }

    /// [`synthesize`](Self::synthesize) with progress, cancellation and
    /// streaming. `progress` runs before each chunk and after every diffusion
    /// step; an error from it stops synthesis and is returned. `emit` receives
    /// the finished audio in order (all but the crossfade tail of each chunk
    /// as soon as the chunk is decoded); returning `false` stops early with
    /// `Ok`. The concatenated output equals `synthesize` bit for bit.
    pub fn synthesize_controlled(
        &mut self,
        prompt: &VoicePrompt,
        text: &str,
        params: &SynthParams,
        rng: &mut dyn Randomness,
        progress: &mut dyn FnMut(SynthProgress) -> Result<()>,
        emit: &mut dyn FnMut(Vec<f32>) -> bool,
    ) -> Result<()> {
        let chunks = plan_chunks(text, params.max_chunk_bytes)?;
        let code_size = self.duration.config().code_size;
        let prompt_frames = prompt.latent_frames();
        let mut fader = Crossfader::new(SAMPLE_RATE);
        let last = chunks.len() - 1;
        for (i, chunk) in chunks.iter().enumerate() {
            let at = |step| SynthProgress {
                chunk: i,
                chunks: chunks.len(),
                step,
                steps: params.steps,
            };
            progress(at(0))?;
            let (ph, tone) = self.g2p.phonemize(chunk)?;
            let raw = self.duration.predict(&prompt.duration, &ph, &tone)?;
            let opts = ChunkOptions {
                disturb: params.dur_disturb,
                alpha: params.dur_alpha,
                is_first: i == 0,
                is_final: i == last,
            };
            let dur = finalize_durations(&raw, &ph, code_size, opts, &mut |o| rng.uniform(o))?;
            let mel2ph_pred = length_regulate(&dur);

            let (phone, tones, mel2ph) = dit_inputs(prompt, &ph, &tone, &mel2ph_pred);
            if mel2ph.len() / FRAMES_PER_LATENT > MAX_LATENT_FRAMES {
                return Err(SpeechError::Input {
                    why: format!(
                        "chunk needs {} latent frames, above the {MAX_LATENT_FRAMES} limit",
                        mel2ph.len() / FRAMES_PER_LATENT
                    ),
                });
            }
            let cond = self.dit.prepare(&phone, &tones, &mel2ph, &prompt.latent)?;
            let sampler = SamplerParams {
                steps: params.steps,
                p_w: params.p_w,
                t_w: params.t_w,
            };
            let mut x = sample_controlled(
                &self.dit,
                &cond,
                sampler,
                &mut |o| rng.normal(o),
                &mut |done| progress(at(done)),
            )?;
            // The prompt region is replaced by the prompt's own latent.
            x[..prompt.latent.len()].copy_from_slice(&prompt.latent);
            let audio = self.wavvae.decode(&x)?;
            let trimmed = &audio[prompt_frames * SAMPLES_PER_LATENT..];
            let loud = integrated_loudness(trimmed, SAMPLE_RATE)?;
            let ready = fader.push(&match_loudness(trimmed, loud, prompt.loudness))?;
            if !ready.is_empty() && !emit(ready) {
                return Ok(());
            }
        }
        let tail = fader.finish();
        if !tail.is_empty() {
            emit(tail);
        }
        Ok(())
    }
}

/// Where a synthesis is: reported before each text chunk (`step` 0) and after
/// every diffusion step of that chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SynthProgress {
    /// 0-based chunk index.
    pub chunk: usize,
    pub chunks: usize,
    /// Diffusion steps finished in this chunk.
    pub step: usize,
    pub steps: usize,
}

/// The text checks synthesis starts with (English only, not empty after
/// normalization), returning the chunks it will synthesize. Cheap; callers
/// use it to refuse a request before queueing it.
pub fn plan_chunks(text: &str, max_chunk_bytes: usize) -> Result<Vec<String>> {
    ensure_english(text)?;
    let chunks = chunk_english(&normalize_english(text), max_chunk_bytes);
    if chunks.is_empty() || chunks.iter().all(|c| c.trim().is_empty()) {
        return Err(SpeechError::Input {
            why: "text is empty after normalization".to_string(),
        });
    }
    Ok(chunks)
}

/// Conservative peak activation bytes (above the loaded weights) for one
/// synthesis: a prompt of `prompt_samples` at 24 kHz, a longest text chunk of
/// `longest_chunk_bytes`, and duration factor `dur_alpha`.
///
/// Measured as peak process footprint (what runtime admission reads), one
/// reading each on an M4 Max: 6.6 GB at about 120 latent frames (prompt plus
/// target), 11.1 GB at about 730 and 13.8 GB at about 880, over about 5.8 GB
/// of resident weights; the Metal path peaked no higher. That is at most
/// about 9.5 MiB per frame, mostly the WavVAE (encoder on the reference,
/// decoder on prompt plus target). The target is bounded at 3 latent frames
/// per text byte, above the 1.5 to 2.2 observed.
pub fn activation_reserve_bytes(
    prompt_samples: usize,
    longest_chunk_bytes: usize,
    dur_alpha: f32,
) -> u64 {
    const PER_FRAME: u64 = 10 << 20; // above the measured 9.5 MiB
    const SLACK: u64 = 256 << 20;
    let used = prompt_samples.min(MAX_PROMPT_SECONDS * SAMPLE_RATE as usize);
    let prompt_frames = used.div_ceil(SAMPLES_PER_LATENT);
    let alpha = if dur_alpha.is_finite() {
        dur_alpha.max(1.0)
    } else {
        1.0
    };
    let target = ((longest_chunk_bytes as f64) * 3.0 * f64::from(alpha)).ceil() as usize;
    let frames = (prompt_frames + target).min(MAX_LATENT_FRAMES + prompt_frames);
    frames as u64 * PER_FRAME + SLACK
}

/// Mono reference audio at `rate` Hz resampled to the 24 kHz prompt rate.
/// The reference loads prompts with librosa (soxr); the long sinc-Hann filter
/// is the closest portable match (see the aligner resampler notes).
pub fn resample_reference(mono: &[f32], rate: u32) -> Result<Vec<f32>> {
    if rate == SAMPLE_RATE {
        return Ok(mono.to_vec());
    }
    let options = SincHannOptions {
        lowpass_filter_width: 64,
        ..SincHannOptions::default()
    };
    resample_mono_sinc_hann(mono, rate, SAMPLE_RATE, &options)
        .map_err(|e| SpeechError::Audio(e.to_string()))
}

/// The root `config.json` the converter writes. Older converted directories
/// have none; when present it must agree with what was loaded.
pub(super) fn check_root_config(dir: &Path, can_clone: bool) -> Result<()> {
    let path = dir.join("config.json");
    let text = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => {
            return Err(SpeechError::Audio(format!(
                "reading {}: {e}",
                path.display()
            )))
        }
    };
    let bad = |field: &str, why: String| SpeechError::BadConfig {
        field: format!("config.json {field}"),
        why,
    };
    let value: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| bad("(file)", e.to_string()))?;
    let model_type = value.get("model_type").and_then(|v| v.as_str());
    if model_type != Some(MODEL_TYPE) {
        return Err(bad(
            "model_type",
            format!("expected {MODEL_TYPE:?}, found {model_type:?}"),
        ));
    }
    let declared = value.get("voice_clone").and_then(|v| v.as_bool());
    if declared != Some(can_clone) {
        return Err(bad(
            "voice_clone",
            format!("declares {declared:?} but the WavVAE encoder is present: {can_clone}"),
        ));
    }
    Ok(())
}

/// `prepare_inputs_for_dit`: concatenates the prompt and chunk phones, tones and
/// alignments, resets tones outside {0, 4, 11..=15} to 3 (the English tone), and
/// cuts `mel2ph` to a multiple of 8 frames.
pub(super) fn dit_inputs(
    prompt: &VoicePrompt,
    ph_pred: &[i32],
    tone_pred: &[i32],
    mel2ph_pred: &[i32],
) -> (Vec<i32>, Vec<i32>, Vec<i32>) {
    let offset = prompt.phone.len() as i32;
    let mut mel2ph = prompt.mel2ph.clone();
    mel2ph.extend(mel2ph_pred.iter().map(|&m| m + offset));
    mel2ph.truncate(mel2ph.len() / FRAMES_MULTIPLE as usize * FRAMES_MULTIPLE as usize);
    let mut phone = prompt.phone.clone();
    phone.extend_from_slice(ph_pred);
    let mut tone = prompt.tone.clone();
    tone.extend_from_slice(tone_pred);
    for t in tone.iter_mut() {
        if !(*t == 4 || (11..=15).contains(t) || *t == 0) {
            *t = 3;
        }
    }
    (phone, tone, mel2ph)
}
