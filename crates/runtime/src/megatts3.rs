//! MegaTTS3 voice cloning on the native audio worker.
//!
//! The pipeline itself is `audio::tts::megatts3` and runs on the CPU until its
//! Metal path qualifies. Preparing a reference voice (aligner, WavVAE encoder,
//! duration prompt) takes seconds, so the runner keeps the last few prepared
//! voices. A prepared voice also consumes one random draw (the encoder's
//! posterior sample), so the cache stores the generator state after that draw:
//! a hit continues exactly where a miss would, and a seed gives the same audio
//! whether or not the voice was cached.

use std::path::Path;

use audio::tts::megatts3::{
    plan_chunks, resample_reference, MegaTts3, SeededRng, SynthParams, SynthProgress, VoicePrompt,
};

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
}

impl MegaTts3Runner {
    /// Opens a converted directory. Refuses one without the WavVAE encoder:
    /// this runner only clones.
    pub fn open(path: &Path) -> audio::Result<Self> {
        let tts = MegaTts3::open(path)?;
        if !tts.can_clone() {
            return Err(audio::SpeechError::Unsupported {
                why: "this MegaTTS3 install has no WavVAE encoder, so it cannot clone a voice"
                    .into(),
            });
        }
        Ok(Self {
            tts,
            voices: Vec::new(),
        })
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
        self.tts.synthesize_controlled(
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
        let mut runner = MegaTts3Runner::open(Path::new(&dir)).unwrap();
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
