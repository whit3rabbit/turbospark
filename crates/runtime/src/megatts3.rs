//! MegaTTS3 voice cloning on the native audio worker.
//!
//! The pipeline itself is `audio::tts::megatts3`. With [`Device::Metal`] the
//! DiT's per-block projections and attention and the WavVAE decoder's
//! convolutions run on `gpu::Music3Device` (F32 weights, tiled float GEMM and
//! convolution, batched non-causal attention); everything else stays on the
//! CPU. Preparing a reference voice (aligner, WavVAE encoder,
//! duration prompt) takes seconds, so the runner keeps the last few prepared
//! voices. A prepared voice also consumes one random draw (the encoder's
//! posterior sample), so the cache stores the generator state after that draw:
//! a hit continues exactly where a miss would, and a seed gives the same audio
//! whether or not the voice was cached.

use std::path::Path;
use std::rc::Rc;

use audio::backend::{
    AttentionShape, ComputeBackend, ConvShape, DType, DeviceWeight, WeightData, WeightEncoding,
};
use audio::tts::megatts3::{
    plan_chunks, resample_reference, MegaTts3, SeededRng, SynthParams, SynthProgress, VoicePrompt,
};
use audio::{Result as AudioResult, SpeechError};
use gpu::{Music3Device, Music3Encoding, Music3Weight};

/// Where the DiT's heavy operations run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    Cpu,
    Metal,
}

fn metal_error(e: gpu::GpuError) -> SpeechError {
    SpeechError::Unsupported {
        why: format!("MegaTTS3 Metal: {e}"),
    }
}

/// The device: F32 weights only, the tiled float GEMM and convolution, and
/// batched non-causal attention. Anything else is refused rather than
/// approximated.
struct MetalBackend {
    device: Music3Device,
}
struct MetalWeight {
    inner: Music3Weight,
}
impl ComputeBackend for MetalBackend {
    fn load_weight(&self, data: WeightData<'_>) -> AudioResult<Rc<dyn DeviceWeight>> {
        if data.encoding != WeightEncoding::F32 || data.dtype != DType::F32 {
            return Err(SpeechError::Unsupported {
                why: "MegaTTS3 Metal loads its F32 checkpoint only".into(),
            });
        }
        let inner = self
            .device
            .load_weight(data.shape, data.bytes, Music3Encoding::F32, &[], &[], &[])
            .map_err(metal_error)?;
        Ok(Rc::new(MetalWeight { inner }))
    }
    fn attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        s: AttentionShape,
        d: DType,
    ) -> AudioResult<Vec<f32>> {
        if d != DType::F32 || s.causal || s.kv_time_major || s.offset != 0 || s.queries != s.keys {
            return Err(SpeechError::Unsupported {
                why: "MegaTTS3 attention is non-causal F32 self-attention".into(),
            });
        }
        self.device
            .attention_typed(
                q,
                k,
                v,
                s.batch,
                s.queries,
                s.keys,
                s.heads,
                s.kv_heads,
                s.dim,
                false,
                false,
                0,
                gpu::Music3DType::F32,
            )
            .map_err(metal_error)
    }
}
impl DeviceWeight for MetalWeight {
    fn linear(
        &self,
        x: &[f32],
        bias: Option<&[f32]>,
        rows: usize,
        input: usize,
        output: usize,
        d: DType,
    ) -> AudioResult<Vec<f32>> {
        if d != DType::F32 {
            return Err(SpeechError::Unsupported {
                why: "MegaTTS3 linear must remain F32".into(),
            });
        }
        // Full float tiles without operand narrowing (the Kokoro F32 path).
        self.inner
            .linear_f32(x, bias, rows, input, output)
            .map_err(metal_error)
    }
    fn embedding(&self, _ids: &[i32], _width: usize) -> AudioResult<Vec<f32>> {
        Err(SpeechError::Unsupported {
            why: "MegaTTS3 Metal has no embeddings".into(),
        })
    }
    fn convolution(
        &self,
        x: &[f32],
        bias: Option<&[f32]>,
        s: ConvShape,
        d: DType,
    ) -> AudioResult<Vec<f32>> {
        self.convolution_grouped(x, bias, s, 1, d)
    }
    fn convolution_grouped(
        &self,
        x: &[f32],
        bias: Option<&[f32]>,
        s: ConvShape,
        groups: usize,
        d: DType,
    ) -> AudioResult<Vec<f32>> {
        if d != DType::F32 {
            return Err(SpeechError::Unsupported {
                why: "MegaTTS3 convolution must remain F32".into(),
            });
        }
        self.inner
            .convolution_f32_grouped(
                x,
                bias,
                s.input_channels,
                s.output_channels,
                s.kernel,
                s.stride,
                s.padding,
                s.dilation,
                s.transpose,
                groups,
            )
            .map_err(metal_error)
    }
}

/// Prepared voices kept per runner. Each holds the reference at 24 kHz (at most
/// 30 s, 2.9 MB), the prompt latent and the duration LM's key/value state.
const PROMPT_CACHE: usize = 4;

/// One cloning request, already validated by the caller.
pub struct CloneRequest<'a> {
    pub text: &'a str,
    /// Mono reference audio at `reference_rate` Hz.
    pub reference: &'a [f32],
    pub reference_rate: u32,
    pub seed: u64,
    pub params: SynthParams,
}

/// Progress of [`MegaTts3Runner::synthesize_controlled`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CloneProgress {
    /// Preparing the reference voice (skipped when it is cached).
    Reference,
    /// `completed` of `total` diffusion steps across all text chunks.
    Diffusion { completed: usize, total: usize },
}

struct CachedVoice {
    seed: u64,
    reference_24k: Vec<f32>,
    prompt: VoicePrompt,
    rng_after_prompt: SeededRng,
}

pub struct MegaTts3Runner {
    tts: MegaTts3,
    /// Most recently used last.
    voices: Vec<CachedVoice>,
    device: Device,
}

impl MegaTts3Runner {
    /// Opens a converted directory on `device`. Refuses one without the
    /// WavVAE encoder: this runner only clones.
    pub fn open(path: &Path, device: Device) -> audio::Result<Self> {
        let backend: Option<Rc<dyn ComputeBackend>> = match device {
            Device::Cpu => None,
            Device::Metal => Some(Rc::new(MetalBackend {
                device: Music3Device::new().map_err(metal_error)?,
            })),
        };
        let tts = MegaTts3::open_with_backend(path, backend)?;
        if !tts.can_clone() {
            return Err(audio::SpeechError::Unsupported {
                why: "this MegaTTS3 install has no WavVAE encoder, so it cannot clone a voice"
                    .into(),
            });
        }
        Ok(Self {
            tts,
            voices: Vec::new(),
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    /// Clones the reference voice and streams `text` in it at 24 kHz.
    /// `progress` errors cancel; `emit` returning `false` stops early.
    pub fn synthesize_controlled(
        &mut self,
        request: &CloneRequest<'_>,
        progress: &mut dyn FnMut(CloneProgress) -> audio::Result<()>,
        emit: &mut dyn FnMut(Vec<f32>) -> bool,
    ) -> audio::Result<()> {
        let chunks = plan_chunks(request.text, request.params.max_chunk_bytes)?.len();
        let reference_24k = resample_reference(request.reference, request.reference_rate)?;
        let (prompt, mut rng) = match self.cached(request.seed, &reference_24k) {
            Some(voice) => voice,
            None => {
                progress(CloneProgress::Reference)?;
                let mut rng = SeededRng::new(request.seed);
                let prompt = self.tts.prepare_prompt(&reference_24k, &mut rng)?;
                self.remember(CachedVoice {
                    seed: request.seed,
                    reference_24k,
                    prompt: prompt.clone(),
                    rng_after_prompt: rng.clone(),
                });
                (prompt, rng)
            }
        };
        let steps = request.params.steps;
        let total = chunks * steps;
        let tts = &mut self.tts;
        // Every device call autoreleases command objects; without an inner
        // pool they accumulate for the life of the worker thread.
        gpu::autorelease_pool(|| {
            tts.synthesize_controlled(
                &prompt,
                request.text,
                &request.params,
                &mut rng,
                &mut |p: SynthProgress| {
                    progress(CloneProgress::Diffusion {
                        completed: p.chunk * steps + p.step,
                        total,
                    })
                },
                emit,
            )
        })
    }

    /// A cached voice for exactly this reference and seed, moved to the back.
    fn cached(&mut self, seed: u64, reference_24k: &[f32]) -> Option<(VoicePrompt, SeededRng)> {
        let index = self
            .voices
            .iter()
            .position(|v| v.seed == seed && same_bits(&v.reference_24k, reference_24k))?;
        let voice = self.voices.remove(index);
        let hit = (voice.prompt.clone(), voice.rng_after_prompt.clone());
        self.voices.push(voice);
        Some(hit)
    }

    fn remember(&mut self, voice: CachedVoice) {
        if self.voices.len() == PROMPT_CACHE {
            self.voices.remove(0);
        }
        self.voices.push(voice);
    }

    #[cfg(test)]
    pub(crate) fn cached_voices(&self) -> usize {
        self.voices.len()
    }
}

/// Exact equality, so `-0.0` and `0.0` (which `==` merges) stay distinct keys.
fn same_bits(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

#[cfg(test)]
mod tests {
    use super::*;
    use audio::tts::megatts3::{Dit, DitConfig};

    /// Synthetic DiT inputs: `phones` phones over `frames` latent frames
    /// (4 mel frames each), the first `prompt` frames held by a latent.
    /// Returns phones, tones, `mel2ph`, the prompt latent and a noisy latent.
    #[allow(clippy::type_complexity)]
    fn dit_inputs(
        cfg: &DitConfig,
        phones: usize,
        frames: usize,
        prompt: usize,
    ) -> (Vec<i32>, Vec<i32>, Vec<i32>, Vec<f32>, Vec<f32>) {
        let phone: Vec<i32> = (0..phones)
            .map(|i| 1 + (i * 7 % (cfg.phone_vocab - 2)) as i32)
            .collect();
        let tone = vec![3; phones];
        let mel2ph: Vec<i32> = (0..4 * frames)
            .map(|t| 1 + (t * phones / (4 * frames)) as i32)
            .collect();
        let wave = |n: usize, k: usize| -> Vec<f32> {
            (0..n)
                .map(|i| (((i * k) % 113) as f32 - 56.0) / 40.0)
                .collect()
        };
        let latent = wave(prompt * cfg.channels, 29);
        let x = wave(frames * cfg.channels, 37);
        (phone, tone, mel2ph, latent, x)
    }

    fn compare(a: &[f32], b: &[f32]) -> (f32, f32) {
        let diff = a
            .iter()
            .zip(b)
            .fold(0.0f32, |m, (x, y)| m.max((x - y).abs()));
        let scale = a.iter().fold(0.0f32, |m, x| m.max(x.abs()));
        (diff, scale)
    }

    /// Metal against the CPU reference on the committed tiny checkpoint. The
    /// device sums in a different order, so the bound is on the observed
    /// rounding gap, not equality.
    #[test]
    fn metal_dit_tracks_the_cpu_on_the_tiny_fixture() {
        let Ok(device) = Music3Device::new() else {
            eprintln!("no Metal device; skipped");
            return;
        };
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../audio/testdata/megatts3_dit");
        let load = || {
            let text = std::fs::read_to_string(dir.join("tiny_config.json")).unwrap();
            let cfg = DitConfig::from_json(&serde_json::from_str(&text).unwrap()).unwrap();
            let file =
                model_io::safetensors::SafetensorsFile::open(&dir.join("tiny_weights.safetensors"))
                    .unwrap();
            Dit::load(cfg, &file).unwrap()
        };
        let cpu = load();
        let metal = load().with_backend(Some(Rc::new(MetalBackend { device })));
        assert!(metal.on_device());
        let (phone, tone, mel2ph, latent, x) = dit_inputs(cpu.config(), 9, 12, 3);
        let cc = cpu.prepare(&phone, &tone, &mel2ph, &latent).unwrap();
        let mc = metal.prepare(&phone, &tone, &mel2ph, &latent).unwrap();
        for sigma in [0.05f32, 0.5, 0.95] {
            let want = cpu.velocity(&cc, &x, sigma, 1.4, 3.0).unwrap();
            let got = gpu::autorelease_pool(|| metal.velocity(&mc, &x, sigma, 1.4, 3.0)).unwrap();
            let (diff, scale) = compare(&want, &got);
            eprintln!("tiny DiT sigma {sigma}: max diff {diff:e} (peak {scale})");
            // Observed about 6e-7 relative on an M4 Max; the bound leaves 16x room.
            assert!(diff <= 1e-5 * scale.max(1.0), "sigma {sigma}: {diff}");
        }
    }

    /// The WavVAE decoder on Metal against the CPU on the committed tiny
    /// checkpoint (device convolutions; odd-ratio transposed convs stay on
    /// the host).
    #[test]
    fn metal_wavvae_decoder_tracks_the_cpu_on_the_tiny_fixture() {
        use audio::codec::wavvae::{WavVae, WavVaeConfig};
        let Ok(device) = Music3Device::new() else {
            eprintln!("no Metal device; skipped");
            return;
        };
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../audio/testdata/wavvae");
        let load = || {
            let text = std::fs::read_to_string(dir.join("tiny_config.json")).unwrap();
            let cfg = WavVaeConfig::from_json(&serde_json::from_str(&text).unwrap()).unwrap();
            let file =
                model_io::safetensors::SafetensorsFile::open(&dir.join("tiny_weights.safetensors"))
                    .unwrap();
            WavVae::load(cfg, &file).unwrap()
        };
        let cpu = load();
        let metal = load().with_backend(Some(Rc::new(MetalBackend { device })));
        let z = cpu.config().z_dim;
        let latent: Vec<f32> = (0..9 * z)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) / 40.0)
            .collect();
        let want = cpu.decode(&latent).unwrap();
        let got = gpu::autorelease_pool(|| metal.decode(&latent)).unwrap();
        let (diff, scale) = compare(&want, &got);
        eprintln!("tiny WavVAE decode: max diff {diff:e} (peak {scale})");
        assert_eq!(got.len(), want.len());
        assert!(diff <= 1e-5, "{diff}");
    }

    /// Real checkpoint: one velocity at a realistic length on both paths,
    /// with timings (one reading each, not a benchmark).
    #[test]
    #[ignore = "requires TURBOSPARK_MEGATTS3_DIR and a Metal device"]
    fn real_metal_dit_tracks_the_cpu() {
        let dir = std::env::var_os("TURBOSPARK_MEGATTS3_DIR").unwrap();
        let frames: usize = std::env::var("TURBOSPARK_MEGATTS3_FRAMES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(250);
        let path = Path::new(&dir).join("diffusion_transformer");
        let cpu = Dit::open(&path).unwrap();
        let metal = Dit::open(&path)
            .unwrap()
            .with_backend(Some(Rc::new(MetalBackend {
                device: Music3Device::new().unwrap(),
            })));
        let (phone, tone, mel2ph, latent, x) = dit_inputs(cpu.config(), 40, frames, frames / 4);
        let cc = cpu.prepare(&phone, &tone, &mel2ph, &latent).unwrap();
        let mc = metal.prepare(&phone, &tone, &mel2ph, &latent).unwrap();
        let started = std::time::Instant::now();
        let want = cpu.velocity(&cc, &x, 0.5, 1.4, 3.0).unwrap();
        let cpu_time = started.elapsed();
        // The first device call uploads the weights; time the second.
        let mut got = Vec::new();
        let mut metal_times = Vec::new();
        for _ in 0..3 {
            let started = std::time::Instant::now();
            got = gpu::autorelease_pool(|| metal.velocity(&mc, &x, 0.5, 1.4, 3.0)).unwrap();
            metal_times.push(started.elapsed());
        }
        let (diff, scale) = compare(&want, &got);
        eprintln!(
            "real DiT, {frames} frames: max diff {diff:e} (peak {scale}); CPU {cpu_time:.2?}, Metal {metal_times:.2?}"
        );
        assert!(diff <= 1e-3 * scale.max(1.0), "{diff}");
    }

    /// Whole pipeline on Metal: writes the WAV for a listening or ASR check and
    /// reports how far it is from the CPU path for the same seed.
    #[test]
    #[ignore = "requires TURBOSPARK_MEGATTS3_DIR, TURBOSPARK_MEGATTS3_REFERENCE and a Metal device"]
    fn real_metal_pipeline_matches_the_cpu_pipeline() {
        let dir = std::env::var_os("TURBOSPARK_MEGATTS3_DIR").unwrap();
        let wav = std::env::var_os("TURBOSPARK_MEGATTS3_REFERENCE").unwrap();
        let wave = audio::read_wav_f32(Path::new(&wav)).unwrap();
        let steps: usize = std::env::var("TURBOSPARK_MEGATTS3_STEPS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(8);
        let request = CloneRequest {
            text: "Hello world, this is a short test.",
            reference: &wave.samples,
            reference_rate: wave.sample_rate,
            seed: 7,
            params: SynthParams {
                steps,
                ..SynthParams::default()
            },
        };
        // Two runs per runner: the first uploads device weights and prepares
        // the voice; the second shows the warm cost of a request.
        let run = |device: Device| {
            let mut runner = MegaTts3Runner::open(Path::new(&dir), device).unwrap();
            let mut times = Vec::new();
            let mut out = Vec::new();
            for _ in 0..2 {
                let started = std::time::Instant::now();
                out.clear();
                runner
                    .synthesize_controlled(&request, &mut |_| Ok(()), &mut |pcm| {
                        out.extend(pcm);
                        true
                    })
                    .unwrap();
                times.push(started.elapsed());
            }
            (out, times)
        };
        let (metal, metal_time) = run(Device::Metal);
        let (cpu, cpu_time) = run(Device::Cpu);
        let n = metal.len().min(cpu.len());
        let (diff, _) = compare(&cpu[..n], &metal[..n]);
        let dot: f64 = cpu[..n]
            .iter()
            .zip(&metal[..n])
            .map(|(a, b)| f64::from(*a) * f64::from(*b))
            .sum();
        let norm = |x: &[f32]| x.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>().sqrt();
        let cosine = dot / (norm(&cpu[..n]) * norm(&metal[..n]));
        eprintln!(
            "pipeline, {steps} steps: lengths CPU {} Metal {}, max diff {diff:e}, cosine {cosine:.6}; CPU {cpu_time:.1?}, Metal {metal_time:.1?} (first, warm)",
            cpu.len(),
            metal.len()
        );
        if let Some(out) = std::env::var_os("TURBOSPARK_MEGATTS3_METAL_WAV") {
            let wave = audio::Waveform::mono(24_000, metal.clone()).unwrap();
            std::fs::write(out, audio::write_wav_f32(&wave)).unwrap();
        }
        assert!(metal.iter().all(|v| v.is_finite()));
        assert_eq!(
            metal.len(),
            cpu.len(),
            "durations come from the CPU duration LM"
        );
    }

    #[test]
    fn same_bits_distinguishes_signed_zero_and_length() {
        assert!(same_bits(&[0.0, 1.0], &[0.0, 1.0]));
        assert!(!same_bits(&[0.0], &[-0.0]));
        assert!(!same_bits(&[0.0], &[0.0, 0.0]));
    }

    /// Real weights: a cache hit must produce exactly the audio of a miss, and
    /// cancellation stops mid-run.
    #[test]
    #[ignore = "requires TURBOSPARK_MEGATTS3_DIR and TURBOSPARK_MEGATTS3_REFERENCE"]
    fn real_cache_hit_equals_miss_and_cancel_stops() {
        let dir = std::env::var_os("TURBOSPARK_MEGATTS3_DIR").unwrap();
        let wav = std::env::var_os("TURBOSPARK_MEGATTS3_REFERENCE").unwrap();
        let wave = audio::read_wav_f32(Path::new(&wav)).unwrap();
        let channels = usize::from(wave.channels).max(1);
        let mono: Vec<f32> = wave
            .samples
            .chunks_exact(channels)
            .map(|f| f.iter().sum::<f32>() / channels as f32)
            .collect();
        let mut runner = MegaTts3Runner::open(Path::new(&dir), Device::Cpu).unwrap();
        let request = CloneRequest {
            text: "Hello world.",
            reference: &mono,
            reference_rate: wave.sample_rate,
            seed: 5,
            params: SynthParams {
                steps: 2,
                ..SynthParams::default()
            },
        };
        let run = |runner: &mut MegaTts3Runner| {
            let (mut out, mut events) = (Vec::new(), Vec::new());
            runner
                .synthesize_controlled(
                    &request,
                    &mut |p| {
                        events.push(p);
                        Ok(())
                    },
                    &mut |pcm| {
                        out.extend(pcm);
                        true
                    },
                )
                .unwrap();
            (out, events)
        };
        let (miss, miss_events) = run(&mut runner);
        assert_eq!(runner.cached_voices(), 1);
        let (hit, hit_events) = run(&mut runner);
        assert_eq!(runner.cached_voices(), 1);
        assert!(
            !miss.is_empty() && same_bits(&miss, &hit),
            "cache hit changed the audio"
        );
        assert_eq!(miss_events[0], CloneProgress::Reference);
        assert!(!hit_events.contains(&CloneProgress::Reference));
        assert_eq!(
            hit_events.last(),
            Some(&CloneProgress::Diffusion {
                completed: 2,
                total: 2
            })
        );

        // The seed is part of the key: the prompt consumed a draw from it.
        let reseeded = CloneRequest {
            seed: 6,
            params: request.params.clone(),
            ..request
        };
        let mut prepared = false;
        runner
            .synthesize_controlled(
                &reseeded,
                &mut |p| {
                    prepared |= p == CloneProgress::Reference;
                    Ok(())
                },
                &mut |_| true,
            )
            .unwrap();
        assert!(prepared, "a different seed reused a cached voice");
        assert_eq!(runner.cached_voices(), 2);

        let mut emitted = 0usize;
        let stopped = runner.synthesize_controlled(
            &request,
            &mut |p| match p {
                CloneProgress::Diffusion { completed: 1, .. } => Err(audio::SpeechError::Input {
                    why: "cancelled".into(),
                }),
                _ => Ok(()),
            },
            &mut |pcm| {
                emitted += pcm.len();
                true
            },
        );
        assert!(stopped.is_err());
        assert_eq!(emitted, 0, "no audio after a cancel before decode");
    }
}
