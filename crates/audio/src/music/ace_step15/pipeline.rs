//! ACE-Step 1.5 text2music, end to end: prompt, text encoder, condition
//! encoder, turbo sampling, VAE decode.
//!
//! Reference: upstream's handler at the pinned commit for the inputs it
//! builds for one text2music request (`prepare_batch_data`,
//! `_prepare_text_conditioning_inputs`, `_build_chunk_masks_and_src_latents`,
//! `infer_refer_latent`, `_prepare_target_latents_and_wavs`) and the
//! decode post-processing (`generate_music_decode.py`):
//!
//! - latent frames `max(128, int(duration * 48000) // 1920)`;
//! - source latents and the timbre reference come from the silence latent
//!   (an all-zero reference is `silence[:750]`), the chunk mask is all ones;
//! - turbo schedule at `shift` (upstream recommends 3.0 for turbo), ODE;
//! - non-finite or all-zero latents are refused;
//! - the silence latent comes from the checkpoint's `silence_latent.pt`
//!   through [`super::silence::read_pt`], which never unpickles;
//! - the decoded waveform is divided by its peak when the peak exceeds 1.
//!
//! Not reproduced: torch's RNG (the initial noise comes from [`gaussian`]
//! or the caller, so a seed does not give upstream's audio). Upstream's
//! DCW sampler correction is available ([`Text2Music::dcw`]); upstream
//! enables [`Dcw::TURBO`] by default for turbo.

use std::path::Path;
use std::rc::Rc;
use std::time::{Duration, Instant};

use turbospark_model_io::safetensors::SafetensorsFile;

use super::dcw::Dcw;
use super::dit::{
    sample_turbo, turbo_schedule, ConditionEncoder, Conditioning, Dit, StepCondition,
};
use super::prompt::{caption_prompt, lyric_prompt, Metas, PromptTokenizer, Task};
use super::text::TextEncoder;
use super::transformer::{AceConfig, Loader};
use super::vae::{OobleckConfig, OobleckVae};
use crate::backend::ComputeBackend;
use crate::nn::load_tensor;
use crate::{Result, SpeechError};

/// Output sample rate of the Oobleck VAE.
pub const SAMPLE_RATE: u32 = 48_000;
/// Upstream's duration range for one request, in seconds.
pub const MAX_DURATION_SECONDS: f64 = 600.0;
/// Frames of silence latent used as the default timbre reference (30 s).
const REFERENCE_FRAMES: usize = 750;
/// Minimum latent length upstream generates (`max(128, ...)`).
const MIN_FRAMES: usize = 128;
/// Tiled decode window, upstream's CUDA default; bit-identical to untiled
/// on the real checkpoint.
const DECODE_CHUNK: usize = 512;
const DECODE_OVERLAP: usize = 64;

/// One text2music request.
#[derive(Debug, Clone)]
pub struct Text2Music<'a> {
    pub caption: &'a str,
    /// Lyrics with section tags, or `[Instrumental]`.
    pub lyrics: &'a str,
    /// Vocal language code written into the lyric prompt (upstream
    /// default `en`).
    pub language: &'a str,
    pub bpm: Option<u32>,
    pub key_scale: &'a str,
    pub time_signature: &'a str,
    pub duration_seconds: f64,
    /// Turbo shift; rounded to the trained 1, 2 or 3.
    pub shift: f32,
    pub seed: u64,
    /// Initial latent noise `[frames, 64]`, replacing the seeded draw
    /// (parity tests replay a reference draw).
    pub noise: Option<&'a [f32]>,
    /// Wavelet sampler correction; upstream's turbo default is
    /// [`Dcw::TURBO`], `None` reproduces the checkpoint's own sampler.
    pub dcw: Option<Dcw>,
}

/// Where a run is, for progress and cancellation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    Text,
    Condition,
    Step { index: usize, total: usize },
    Decode,
}

/// The rendered song and what produced it.
#[derive(Debug, Clone)]
pub struct Generated {
    /// Channel-major `[2, samples]` at [`SAMPLE_RATE`].
    pub audio: Vec<f32>,
    pub samples: usize,
    /// Time-major `[frames, 64]` latents before decode.
    pub latents: Vec<f32>,
    pub frames: usize,
    pub caption_truncated: bool,
    pub lyrics_truncated: bool,
    pub timings: Timings,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Timings {
    pub text: Duration,
    pub condition: Duration,
    pub sampling: Duration,
    pub decode: Duration,
}

/// Seeded standard-normal draws (SplitMix64 uniforms, Box-Muller).
/// Deterministic per seed; not torch's generator.
pub fn gaussian(seed: u64, count: usize) -> Vec<f32> {
    let mut state = seed;
    let mut next = || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        // 53-bit uniform in (0, 1].
        ((z >> 11) as f64 + 1.0) / (1u64 << 53) as f64
    };
    let mut out = Vec::with_capacity(count + 1);
    while out.len() < count {
        let (u1, u2) = (next(), next());
        let r = (-2.0 * u1.ln()).sqrt();
        let a = std::f64::consts::TAU * u2;
        out.push((r * a.cos()) as f32);
        out.push((r * a.sin()) as f32);
    }
    out.truncate(count);
    out
}

/// The loaded model: checkpoint root with `acestep-v15-turbo/`, `vae/`
/// and `Qwen3-Embedding-0.6B/`.
pub struct AceStep15 {
    tokenizer: PromptTokenizer,
    text: TextEncoder,
    cond: ConditionEncoder,
    dit: Dit,
    vae: OobleckVae,
    /// Time-major `[frames, 64]`.
    silence: Vec<f32>,
    silence_frames: usize,
    acoustic: usize,
}

impl AceStep15 {
    /// Loads every component. With `backend`, DiT projections and VAE
    /// convolutions are resident on the device; the text encoder stays on
    /// the host.
    pub fn load(root: &Path, backend: Option<Rc<dyn ComputeBackend>>) -> Result<Self> {
        let dit_dir = root.join("acestep-v15-turbo");
        let read_json = |path: &Path| -> Result<serde_json::Value> {
            let text = std::fs::read_to_string(path).map_err(|e| SpeechError::BadConfig {
                field: path.display().to_string(),
                why: e.to_string(),
            })?;
            serde_json::from_str(&text).map_err(|e| SpeechError::BadConfig {
                field: path.display().to_string(),
                why: e.to_string(),
            })
        };
        let config = AceConfig::from_json(&read_json(&dit_dir.join("config.json"))?)?;
        if !config.is_turbo {
            return Err(SpeechError::Unsupported {
                why: "only the turbo checkpoint is ported (8-step schedule, no CFG)".into(),
            });
        }
        let acoustic = config.audio_acoustic_hidden_dim;
        // The checkpoint's pickle is read structurally (no unpickling); a
        // trusted conversion to safetensors is accepted when that is all
        // there is.
        let pt = dit_dir.join("silence_latent.pt");
        let converted = dit_dir.join("silence_latent.safetensors");
        let (silence, frames) = if pt.is_file() {
            if acoustic != super::silence::CHANNELS {
                return Err(SpeechError::BadConfig {
                    field: "audio_acoustic_hidden_dim".into(),
                    why: format!("{acoustic} does not match the silence latent's 64 channels"),
                });
            }
            (super::silence::read_pt(&pt)?, super::silence::FRAMES)
        } else if converted.is_file() {
            let sfile = SafetensorsFile::open(&converted)?;
            let frames = sfile
                .descriptor("silence_latent")
                .and_then(|d| (d.shape.len() == 2 && d.shape[1] == acoustic).then_some(d.shape[0]))
                .ok_or_else(|| SpeechError::Tensor {
                    name: "silence_latent".into(),
                    why: format!("expected [frames, {acoustic}]"),
                })?;
            (
                load_tensor(&sfile, "silence_latent", &[frames, acoustic])?,
                frames,
            )
        } else {
            return Err(SpeechError::BadConfig {
                field: "silence_latent.pt".into(),
                why: "missing from acestep-v15-turbo/".into(),
            });
        };
        if frames < REFERENCE_FRAMES {
            return Err(SpeechError::Tensor {
                name: "silence_latent".into(),
                why: format!("{frames} frames, need at least {REFERENCE_FRAMES}"),
            });
        }

        let file = SafetensorsFile::open(&dit_dir.join("model.safetensors"))?;
        let loader = match &backend {
            Some(b) => Loader::with_backend(&file, b.clone()),
            None => Loader::cpu(&file),
        };
        let cond = ConditionEncoder::load(&loader, &config)?;
        let dit = Dit::load(&loader, &config)?;

        let vae_dir = root.join("vae");
        let vae_config = OobleckConfig::from_json(&read_json(&vae_dir.join("config.json"))?)?;
        if vae_config.sampling_rate != SAMPLE_RATE
            || vae_config.latent_channels() != acoustic
            || vae_config.audio_channels != 2
        {
            return Err(SpeechError::BadConfig {
                field: "vae/config.json".into(),
                why: "expected the 48 kHz stereo VAE with the DiT's latent width".into(),
            });
        }
        let vfile = SafetensorsFile::open(&vae_dir.join("diffusion_pytorch_model.safetensors"))?;
        let mut vae = OobleckVae::load(vae_config, &vfile)?;
        if let Some(b) = &backend {
            vae.attach_backend(b.as_ref())?;
        }

        let text_dir = root.join("Qwen3-Embedding-0.6B");
        let text = TextEncoder::load(&text_dir)?;
        if text.hidden_size() != config.text_hidden_dim {
            return Err(SpeechError::BadConfig {
                field: "text_hidden_dim".into(),
                why: "text encoder width does not match the DiT".into(),
            });
        }
        Ok(Self {
            tokenizer: PromptTokenizer::load(&text_dir)?,
            text,
            cond,
            dit,
            vae,
            silence,
            silence_frames: frames,
            acoustic,
        })
    }

    /// `_get_silence_latent_slice`: the first `len` frames, tiled when the
    /// stored latent is shorter.
    fn silence_slice(&self, len: usize) -> Vec<f32> {
        let a = self.acoustic;
        let mut out = Vec::with_capacity(len * a);
        while out.len() < len * a {
            let take = (len * a - out.len()).min(self.silence_frames * a);
            out.extend_from_slice(&self.silence[..take]);
        }
        out
    }

    /// Latent frames for a duration, as upstream sizes its silent target.
    pub fn frames_for(duration_seconds: f64) -> usize {
        let samples = (duration_seconds * SAMPLE_RATE as f64) as usize;
        (samples / 1920).max(MIN_FRAMES)
    }

    /// Runs one request. `progress` sees each stage before it runs and
    /// cancels the run by returning false (`Ok(None)`).
    pub fn text2music(
        &mut self,
        req: &Text2Music<'_>,
        mut progress: impl FnMut(Stage) -> bool,
    ) -> Result<Option<Generated>> {
        let d = req.duration_seconds;
        if !(d.is_finite() && d > 0.0 && d <= MAX_DURATION_SECONDS) {
            return Err(SpeechError::Input {
                why: format!("duration {d} s outside (0, {MAX_DURATION_SECONDS}]"),
            });
        }
        let mut timings = Timings::default();
        let a = self.acoustic;
        let frames = Self::frames_for(d);

        if !progress(Stage::Text) {
            return Ok(None);
        }
        let started = Instant::now();
        let metas = Metas {
            bpm: req.bpm,
            key_scale: req.key_scale.to_string(),
            time_signature: req.time_signature.to_string(),
            duration_seconds: Some(d),
        };
        let caption = self.tokenizer.caption(&caption_prompt(
            Task::Text2Music.instruction(),
            req.caption,
            &metas,
        ))?;
        let lyrics = self
            .tokenizer
            .lyrics(&lyric_prompt(req.lyrics, req.language))?;
        let text = self.text.encode_caption(&caption.ids)?;
        let lyric = self.text.embed_lyrics(&lyrics.ids)?;
        timings.text = started.elapsed();

        if !progress(Stage::Condition) {
            return Ok(None);
        }
        let started = Instant::now();
        let refer = &self.silence[..REFERENCE_FRAMES * a];
        let (enc, enc_seq) = self.cond.forward(&Conditioning {
            text: &text,
            text_len: caption.ids.len(),
            lyric: &lyric,
            lyric_len: lyrics.ids.len(),
            refer,
            refer_len: REFERENCE_FRAMES,
        })?;
        let enc = self.dit.embed_condition(&enc, enc_seq)?;
        let src = self.silence_slice(frames);
        let mut context = Vec::with_capacity(frames * 2 * a);
        for f in 0..frames {
            context.extend_from_slice(&src[f * a..(f + 1) * a]);
            context.extend(std::iter::repeat_n(1.0f32, a));
        }
        let noise = match req.noise {
            Some(n) if n.len() == frames * a => n.to_vec(),
            Some(n) => {
                return Err(SpeechError::Input {
                    why: format!("noise has {} values, expected {frames} x {a}", n.len()),
                })
            }
            None => gaussian(req.seed, frames * a),
        };
        timings.condition = started.elapsed();

        let started = Instant::now();
        let schedule = turbo_schedule(req.shift);
        let total = schedule.len();
        let mut cancelled = false;
        let latents = sample_turbo(
            &self.dit,
            &noise,
            frames,
            &schedule,
            StepCondition {
                context: &context,
                enc: &enc,
                enc_seq,
            },
            None,
            req.dcw,
            |index| {
                let go = progress(Stage::Step { index, total });
                cancelled |= !go;
                go
            },
        )?;
        let Some(latents) = latents.filter(|_| !cancelled) else {
            return Ok(None);
        };
        timings.sampling = started.elapsed();
        if latents.iter().any(|v| !v.is_finite()) {
            return Err(SpeechError::Unsupported {
                why: "generation produced non-finite latents".into(),
            });
        }
        if latents.iter().all(|&v| v == 0.0) {
            return Err(SpeechError::Unsupported {
                why: "generation produced all-zero latents".into(),
            });
        }

        if !progress(Stage::Decode) {
            return Ok(None);
        }
        let started = Instant::now();
        let mut channel_major = vec![0.0f32; frames * a];
        for f in 0..frames {
            for c in 0..a {
                channel_major[c * frames + f] = latents[f * a + c];
            }
        }
        let (mut audio, samples) =
            self.vae
                .decode_tiled(&channel_major, frames, DECODE_CHUNK, DECODE_OVERLAP)?;
        let peak = audio.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        if peak > 1.0 {
            for v in &mut audio {
                *v /= peak;
            }
        }
        timings.decode = started.elapsed();
        Ok(Some(Generated {
            audio,
            samples,
            latents,
            frames,
            caption_truncated: caption.truncated,
            lyrics_truncated: lyrics.truncated,
            timings,
        }))
    }
}
