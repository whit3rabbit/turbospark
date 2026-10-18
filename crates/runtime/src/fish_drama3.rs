//! Fish Audio Drama 3 on Metal: the headless trunk as the portable
//! generation loop's backbone.
//!
//! The loop, prompt, sampler and tower are `audio::tts::fish_drama3`; this
//! adapts [`HiddenTrunk`] (streamed routed experts on the qwen36 flow) to its
//! `Backbone` trait. Facts and gates: `docs/FISH_DRAMA3_PHASE0.md`.

use audio::tts::fish_drama3::generate::{Backbone, BackboneRow};
use audio::{Result, SpeechError};

use crate::hidden_trunk::{HiddenTrunk, PrefixSnapshot, TrunkRow};
use crate::real_forward_types::RealForwardError;

fn trunk_err(e: RealForwardError) -> SpeechError {
    SpeechError::Unsupported {
        why: format!("drama3 trunk: {e}"),
    }
}

fn token(id: i32) -> Result<u32> {
    u32::try_from(id).map_err(|_| SpeechError::Input {
        why: format!("negative token id {id}"),
    })
}

/// [`HiddenTrunk`] as a Drama 3 backbone.
pub struct TrunkBackbone {
    trunk: HiddenTrunk,
}

impl TrunkBackbone {
    pub fn new(trunk: HiddenTrunk) -> Self {
        TrunkBackbone { trunk }
    }

    pub fn trunk(&self) -> &HiddenTrunk {
        &self.trunk
    }
}

impl Backbone for TrunkBackbone {
    type Snapshot = PrefixSnapshot;

    fn hidden_size(&self) -> usize {
        self.trunk.hidden_size()
    }

    fn embedding_row(&mut self, id: i32, out: &mut [f32]) -> Result<()> {
        self.trunk.embedding_row(token(id)?, out).map_err(trunk_err)
    }

    fn prefill(&mut self, rows: &[BackboneRow], hidden_out: &mut [f32]) -> Result<()> {
        let rows = rows
            .iter()
            .map(|row| {
                Ok(match row {
                    BackboneRow::Token(id) => TrunkRow::Token(token(*id)?),
                    BackboneRow::Embedding(values) => TrunkRow::Embedding(values),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        self.trunk.prefill(&rows, hidden_out).map_err(trunk_err)
    }

    fn step(&mut self, row: &BackboneRow, hidden_out: &mut [f32]) -> Result<()> {
        let row = match row {
            BackboneRow::Token(id) => TrunkRow::Token(token(*id)?),
            BackboneRow::Embedding(values) => TrunkRow::Embedding(values),
        };
        self.trunk.step(row, hidden_out).map_err(trunk_err)
    }

    fn snapshot(&mut self) -> Result<PrefixSnapshot> {
        Ok(self.trunk.snapshot())
    }

    fn restore(&mut self, snapshot: &PrefixSnapshot) -> Result<()> {
        self.trunk.restore(snapshot).map_err(trunk_err)
    }

    fn reset(&mut self) {
        self.trunk.reset();
    }
}

/// The trunk architecture a Drama 3 install must carry, derived from its
/// `config.json` rather than read off the trunk's own manifest, so the open's
/// exact manifest check compares two independent sources. Drama 3's text
/// backbone is the Qwen 3.6 35B-A3B baseline field for field apart from its
/// vocabulary (`crates/repack/tests/fish_drama3.rs`); a checkpoint that differs
/// anywhere else is refused at open rather than run.
pub fn expected_trunk_arch(
    install: &std::path::Path,
) -> std::result::Result<model_io::ArchConfig, String> {
    let text = std::fs::read_to_string(install.join("config.json")).map_err(|e| e.to_string())?;
    let root: serde_json::Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
    let vocab = root
        .get("text_config")
        .and_then(|t| t.get("vocab_size"))
        .and_then(serde_json::Value::as_i64)
        .ok_or("config.json: text_config.vocab_size missing")?;
    let mut arch = model_io::qwen_gdn_moe_35b_a3b();
    arch.vocab_size = vocab;
    Ok(arch)
}

/// Routed-expert slots per layer when a host opens Drama 3 without saying:
/// the qwen36 default (`docs/BENCHMARKS.md`'s streamed rows use 16).
pub const DEFAULT_EXPERT_SLOTS: usize = 16;

/// Trunk rows: the system block, two history turns of up to
/// `DEFAULT_MAX_FRAMES` frames each plus their text, the target text, and the
/// chunk's own frames. 8192 rows of full-attention KV is 160 MiB.
pub const MAX_CONTEXT: usize = 8192;

/// Host bytes a Drama 3 open commits, for admission: the trunk's resident
/// weights, the routed-expert slot cache at [`DEFAULT_EXPERT_SLOTS`], the
/// tower and codec (the codec widens to f32 as stored), the codec's Metal
/// weight copies, trunk and tower KV at [`MAX_CONTEXT`], and two
/// recurrent-state snapshots.
pub fn open_bytes(install: &std::path::Path) -> std::result::Result<u64, String> {
    use audio::tts::fish_drama3::generate::{CODEC_FILE, SIDE_FILE, TRUNK_DIR};
    let size = |p: std::path::PathBuf| -> std::result::Result<u64, String> {
        std::fs::metadata(&p)
            .map(|m| m.len())
            .map_err(|e| format!("{}: {e}", p.display()))
    };
    let resident = size(install.join(TRUNK_DIR).join("model_weights.bin"))?;
    let side = size(install.join(SIDE_FILE))?;
    let codec = size(install.join(CODEC_FILE))?;
    let arch = model_io::qwen_gdn_moe_35b_a3b();
    // One expert's gate, up and down at 4 bits with BF16 scales and biases.
    let expert = 1_769_472u64;
    let slots = DEFAULT_EXPERT_SLOTS as u64 * arch.num_layers as u64 * expert;
    let kv = MAX_CONTEXT as u64 * 10 * 2 * 2 * 256 * 2;
    // The Metal tower allocates its KV whole: 13 blocks of K and V rows, 2
    // heads of 256 f16 each.
    let tower_kv = MAX_CONTEXT as u64 * 13 * 2 * 2 * 256 * 2;
    let snapshots = 2 * (64 << 20);
    // The codec worker uploads the conv and linear weights it uses (the
    // decoder's on the first piece, the encoder's on the first reference).
    // The transformers and codebooks stay on the host, so the file bounds
    // the copies from above.
    let codec_device = codec;
    Ok(resident + slots + side + codec + codec_device + kv + tower_kv + snapshots)
}

/// The portable Drama 3 model with the Metal tower, its KV sized for
/// `max_context` rows (the trunk's, since the tower sees every trunk row).
pub fn open_drama(
    install: &std::path::Path,
    max_context: usize,
) -> audio::Result<audio::tts::fish_drama3::generate::Drama3> {
    audio::tts::fish_drama3::generate::Drama3::open_with(install, |side, config| {
        let tower = crate::fish_drama3_tower::MetalTower::load(
            side,
            config.tower.clone(),
            config.semantic_band(),
            max_context,
        )?;
        Ok(Box::new(tower))
    })
}

/// Frames in the first streamed piece, then in every later one. The reference
/// streams every `streaming_interval` (0.32 s, 7 frames) and lets the first
/// piece be shorter for a faster first audio.
const FIRST_PIECE_FRAMES: usize = 4;
const PIECE_FRAMES: usize = 7;

/// Caller overrides of the sampler. `None` keeps the product default, Fish
/// Audio's serving recipe (`SamplingParams::FISH_SERVING`, 1024 frames).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Drama3Sampling {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<usize>,
    /// An upper bound on one chunk's frames. The text-scaled cap still applies
    /// beneath it, so this can shorten a chunk and never lengthen it.
    pub max_frames: Option<usize>,
}

impl Drama3Sampling {
    /// The generation options these overrides give, or why they are refused.
    pub fn options(
        &self,
        seed: u64,
    ) -> std::result::Result<audio::tts::fish_drama3::generate::GenerateOptions, String> {
        use audio::tts::fish_drama3::generate::GenerateOptions;
        use audio::tts::fish_drama3::sampling::{SamplingParams, DEFAULT_MAX_FRAMES};
        let default = GenerateOptions::default().sampling;
        let sampling = SamplingParams {
            temperature: self.temperature.unwrap_or(default.temperature),
            top_k: self.top_k.unwrap_or(default.top_k),
            top_p: self.top_p.unwrap_or(default.top_p),
        };
        sampling.validate().map_err(|e| e.to_string())?;
        let max_frames = self.max_frames.unwrap_or(DEFAULT_MAX_FRAMES);
        if !(1..=DEFAULT_MAX_FRAMES).contains(&max_frames) {
            return Err(format!(
                "max_frames {max_frames} is outside 1 through {DEFAULT_MAX_FRAMES}"
            ));
        }
        Ok(GenerateOptions {
            sampling,
            max_frames,
            seed,
            ..GenerateOptions::default()
        })
    }
}

/// One synthesis request.
pub struct Drama3Request<'a> {
    pub text: &'a str,
    /// Mono reference audio, its sample rate and its transcript, for cloning.
    pub reference: Option<(&'a [f32], u32, &'a str)>,
    pub seed: u64,
    pub sampling: Drama3Sampling,
}

/// Drama 3 on the headless Metal trunk: the portable model, the trunk and the
/// Fish S1 DAC, with the last reference voice's codes cached.
pub struct FishDrama3Runner {
    drama: audio::tts::fish_drama3::generate::Drama3,
    backbone: TrunkBackbone,
    codec: crate::fish_drama3_codec::CodecWorker,
    /// Content hash of the last reference and its encoded utterance.
    voice: Option<(u64, audio::tts::fish_drama3::prompt::Utterance)>,
    pause_guard: Option<audio::tts::fish_drama3::generate::PauseGuard>,
}

impl FishDrama3Runner {
    /// Opens a Drama 3 install (`config.json`, `tokenizer.json`,
    /// `tts_side.safetensors`, `codec.safetensors`, `trunk/`).
    pub fn open(
        install: &std::path::Path,
        expert_cache_slots: model_io::ExpertCacheSlots,
        residency: model_io::ExpertResidency,
        max_context: usize,
    ) -> std::result::Result<Self, String> {
        use audio::tts::fish_drama3::generate::{CODEC_FILE, TRUNK_DIR};
        let drama = open_drama(install, max_context).map_err(|e| e.to_string())?;
        let trunk_dir = install.join(TRUNK_DIR);
        let arch = expected_trunk_arch(install)?;
        let trunk = HiddenTrunk::open(&trunk_dir, arch, max_context, expert_cache_slots, residency)
            .map_err(|e| e.to_string())?;
        let codec = crate::fish_drama3_codec::CodecWorker::open(&install.join(CODEC_FILE), true)?;
        Ok(FishDrama3Runner {
            drama,
            backbone: TrunkBackbone::new(trunk),
            codec,
            voice: None,
            pause_guard: audio::tts::fish_drama3::generate::GenerateOptions::default().pause_guard,
        })
    }

    /// Whether generation resamples a pause the model may never leave
    /// (`PauseGuard`, on by default). `None` runs the reference's loop, for an
    /// A/B against it.
    pub fn set_pause_guard(
        &mut self,
        guard: Option<audio::tts::fish_drama3::generate::PauseGuard>,
    ) {
        self.pause_guard = guard;
    }

    /// Metal buffers the trunk's context has allocated so far (see
    /// [`HiddenTrunk::gpu_buffer_allocations`]). The tower allocates only at
    /// load. The codec allocates per call by design, so it has no flat count
    /// to assert and the footprint oracle covers it.
    pub fn gpu_buffer_allocations(&self) -> u64 {
        self.backbone.trunk().gpu_buffer_allocations()
    }

    /// The codec's output rate (44.1 kHz).
    pub fn sample_rate(&self) -> u32 {
        self.codec.sample_rate()
    }

    /// The encoded reference utterance, from the cache when the same audio and
    /// transcript were used last.
    fn reference(
        &mut self,
        samples: &[f32],
        rate: u32,
        transcript: &str,
    ) -> std::result::Result<audio::tts::fish_drama3::prompt::Utterance, String> {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        rate.hash(&mut hasher);
        // Length-prefixed so bytes cannot move between the two parts.
        samples.len().hash(&mut hasher);
        for v in samples {
            v.to_bits().hash(&mut hasher);
        }
        transcript.hash(&mut hasher);
        let key = hasher.finish();
        if let Some((cached, utterance)) = &self.voice {
            if *cached == key {
                return Ok(utterance.clone());
            }
        }
        let wave = audio::Waveform {
            sample_rate: rate,
            channels: 1,
            samples: samples.to_vec(),
        };
        let target = self.codec.sample_rate();
        let mono = if rate == target {
            wave.samples
        } else {
            audio::to_mono_resampled(
                &wave,
                target,
                &audio::MonoResampleStrategy::SincHann(Default::default()),
            )
            .map_err(|e| e.to_string())?
        };
        let (codes, frames) = self.codec.encode(mono)?;
        let codes: Vec<Vec<i32>> = codes.into_iter().map(|b| b[..frames].to_vec()).collect();
        let utterance = audio::tts::fish_drama3::prompt::Utterance {
            text: transcript.to_string(),
            codes,
        };
        self.voice = Some((key, utterance.clone()));
        Ok(utterance)
    }

    /// Speaks `request`, handing PCM to `emit` as it is decoded. `emit`
    /// returns false to stop, and `stop` is polled once per frame. Returns
    /// what the take produced, including whether it likely stopped speaking
    /// early.
    ///
    /// Generation runs on this thread; the codec runs on a scoped worker fed
    /// frame by frame, so decoding overlaps the next frames' generation.
    pub fn synthesize(
        &mut self,
        request: &Drama3Request<'_>,
        stop: &(dyn Fn() -> bool + Sync),
        mut emit: impl FnMut(Vec<f32>) -> bool + Send,
    ) -> std::result::Result<Drama3Take, String> {
        use audio::tts::fish_drama3::generate::{ChunkStop, Flow, FrameEvent};
        // Refused before the reference is encoded, which costs seconds.
        let mut options = request.sampling.options(request.seed)?;
        options.pause_guard = self.pause_guard;
        let references = match request.reference {
            Some((samples, rate, transcript)) => vec![self.reference(samples, rate, transcript)?],
            None => Vec::new(),
        };
        // The codec decodes on its own thread while generation runs here, and
        // its samples are handed to `emit` between frames and after the last.
        let (frames_tx, pcm_rx) = self.codec.stream(FIRST_PIECE_FRAMES, PIECE_FRAMES)?;
        let mut delivery = Delivery::default();
        let mut chunks = Vec::new();
        let result = self.drama.generate(
            &mut self.backbone,
            &references,
            request.text,
            options,
            |event| {
                while let Ok(piece) = pcm_rx.try_recv() {
                    delivery.take(piece, &mut emit);
                }
                if delivery.stopped || stop() {
                    return Flow::Stop;
                }
                match event {
                    FrameEvent::Frame { codes, .. } => {
                        if frames_tx.send(Some(codes)).is_err() {
                            return Flow::Stop;
                        }
                    }
                    // A frame the pause guard holds back reaches the codec
                    // only if the pause ends, so nothing goes out yet. The
                    // stop flag above is still polled once per frame.
                    FrameEvent::Held { .. } => {}
                    // A chunk cut short still plays, so it never ends silently.
                    FrameEvent::ChunkEnd {
                        chunk,
                        frames,
                        limit,
                        stop,
                        retries,
                    } => {
                        if stop != ChunkStop::Eos {
                            eprintln!(
                                "fish_drama3: chunk {chunk} stopped without EOS at {frames} frames ({stop:?})"
                            );
                        }
                        chunks.push(ChunkReport {
                            chunk,
                            frames,
                            limit,
                            stop,
                            retries,
                        });
                    }
                }
                Flow::Continue
            },
        );
        let _ = frames_tx.send(None);
        drop(frames_tx);
        // The codec closes the channel once the tail is decoded. A stopped
        // request drops it instead, which ends the codec's job.
        while !delivery.stopped {
            match pcm_rx.recv() {
                Ok(piece) => delivery.take(piece, &mut emit),
                Err(_) => break,
            }
        }
        result.map_err(|e| e.to_string())?;
        match delivery.error {
            Some(e) => Err(e),
            None => Ok(Drama3Take {
                emitted: delivery.emitted,
                chunks,
                longest_silence_frames: delivery.silence.longest(),
            }),
        }
    }
}

/// How one text chunk of a take ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChunkReport {
    pub chunk: usize,
    pub frames: usize,
    /// The most frames the chunk could take (its text cap or `max_frames`).
    pub limit: usize,
    pub stop: audio::tts::fish_drama3::generate::ChunkStop,
    /// Times the pause guard rolled a long pause back and drew again.
    pub retries: usize,
}

/// Samples per codec frame (46.4 ms at 44.1 kHz).
const FRAME_SAMPLES: usize = 2048;
/// A codec frame is silent under this RMS (dBFS, full scale 1.0), the rule
/// `docs/FISH_DRAMA3_PHASE0.md` and `fish_drama3_wer` use.
const SILENT_DBFS: f32 = -45.0;
/// A silent run this long means the take stopped speaking. Completed takes
/// paused at most 78 frames between turns (`docs/FISH_DRAMA3_PHASE0.md`), and
/// a take that went silent after its first sentence ran 359 frames before EOS.
pub const STUCK_SILENCE_FRAMES: usize = 160;

/// What one [`FishDrama3Runner::synthesize`] call produced.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Drama3Take {
    /// Samples handed to `emit`.
    pub emitted: usize,
    /// Every chunk that finished, in order. A stopped request reports only
    /// the chunks it finished.
    pub chunks: Vec<ChunkReport>,
    /// The longest run of silent codec frames in the emitted audio.
    pub longest_silence_frames: usize,
}

impl Drama3Take {
    /// Whether the take likely stopped speaking before its text ended: a
    /// chunk ran to its cap without EOS, or the audio held a silence no
    /// completed take has (the pause trap, `docs/FISH_DRAMA3_PHASE0.md`).
    /// The words were never generated, so a retake with another seed is the
    /// remedy.
    pub fn stopped_early(&self) -> bool {
        use audio::tts::fish_drama3::generate::ChunkStop;
        self.chunks.iter().any(|c| c.stop != ChunkStop::Eos)
            || self.longest_silence_frames >= STUCK_SILENCE_FRAMES
    }
}

/// The longest run of silent codec frames in PCM fed in order, frames
/// counted from the first sample. A final partial frame is measured too.
#[derive(Clone, Copy, Default)]
struct SilenceMeter {
    power: f32,
    filled: usize,
    run: usize,
    longest: usize,
}

impl SilenceMeter {
    fn feed(&mut self, pcm: &[f32]) {
        for v in pcm {
            self.power += v * v;
            self.filled += 1;
            if self.filled == FRAME_SAMPLES {
                self.close_frame();
            }
        }
    }

    fn close_frame(&mut self) {
        let floor = 10f32.powf(SILENT_DBFS / 20.0);
        let rms = (self.power / self.filled as f32).sqrt();
        self.run = if rms < floor { self.run + 1 } else { 0 };
        self.longest = self.longest.max(self.run);
        self.power = 0.0;
        self.filled = 0;
    }

    fn longest(&self) -> usize {
        let mut tail = *self;
        if tail.filled > 0 {
            tail.close_frame();
        }
        tail.longest
    }
}

/// Samples handed to the caller so far, and whether to stop.
#[derive(Default)]
struct Delivery {
    emitted: usize,
    stopped: bool,
    error: Option<String>,
    silence: SilenceMeter,
}

impl Delivery {
    fn take(
        &mut self,
        piece: std::result::Result<Vec<f32>, String>,
        emit: &mut impl FnMut(Vec<f32>) -> bool,
    ) {
        if self.stopped {
            return;
        }
        match piece {
            Ok(pcm) if pcm.is_empty() => {}
            Ok(pcm) => {
                self.emitted += pcm.len();
                self.silence.feed(&pcm);
                self.stopped = !emit(pcm);
            }
            Err(e) => {
                self.error = Some(e);
                self.stopped = true;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use audio::tts::fish_drama3::generate::{ChunkStop, GenerateOptions, PauseGuard, TextCap};
    use audio::tts::fish_drama3::sampling::{SamplingParams, DEFAULT_MAX_FRAMES};

    /// No overrides is the product default, Fish Audio's serving recipe, the
    /// same options `GenerateOptions::default` gives the portable loop.
    #[test]
    fn no_overrides_keep_the_product_options() {
        let got = Drama3Sampling::default().options(7).unwrap();
        let want = GenerateOptions::default();
        assert_eq!(got.sampling, SamplingParams::FISH_SERVING);
        assert_eq!(got.sampling, want.sampling);
        assert_eq!(got.max_frames, DEFAULT_MAX_FRAMES);
        assert_eq!(got.max_frames, want.max_frames);
        assert_eq!(got.seed, 7);
        assert_eq!(got.text_cap, Some(TextCap::MEASURED));
        assert_eq!(got.overlap_tower_tail, want.overlap_tower_tail);
        assert_eq!(got.pause_guard, Some(PauseGuard::MEASURED));
    }

    #[test]
    fn each_override_lands_in_its_own_field() {
        let got = Drama3Sampling {
            temperature: Some(0.9),
            top_p: Some(0.8),
            top_k: Some(12),
            max_frames: Some(300),
        }
        .options(0)
        .unwrap();
        assert_eq!(got.sampling.temperature, 0.9);
        assert_eq!(got.sampling.top_p, 0.8);
        assert_eq!(got.sampling.top_k, 12);
        assert_eq!(got.max_frames, 300);
        // One override leaves the others on the recipe.
        let only_k = Drama3Sampling {
            top_k: Some(5),
            ..Default::default()
        }
        .options(0)
        .unwrap();
        assert_eq!(only_k.sampling.top_k, 5);
        assert_eq!(
            only_k.sampling.temperature,
            SamplingParams::FISH_SERVING.temperature
        );
        assert_eq!(only_k.sampling.top_p, SamplingParams::FISH_SERVING.top_p);
        assert_eq!(only_k.max_frames, DEFAULT_MAX_FRAMES);
    }

    #[test]
    fn settings_the_sampler_cannot_use_are_refused() {
        let refused = |s: Drama3Sampling| s.options(0).is_err();
        let none = Drama3Sampling::default();
        assert!(refused(Drama3Sampling {
            temperature: Some(f32::NAN),
            ..none
        }));
        assert!(refused(Drama3Sampling {
            temperature: Some(-0.1),
            ..none
        }));
        assert!(refused(Drama3Sampling {
            top_p: Some(0.0),
            ..none
        }));
        assert!(refused(Drama3Sampling {
            top_p: Some(1.5),
            ..none
        }));
        assert!(refused(Drama3Sampling {
            top_k: Some(0),
            ..none
        }));
        assert!(refused(Drama3Sampling {
            max_frames: Some(0),
            ..none
        }));
        assert!(refused(Drama3Sampling {
            max_frames: Some(DEFAULT_MAX_FRAMES + 1),
            ..none
        }));
        // The edges of the allowed ranges stay allowed.
        let edges = Drama3Sampling {
            temperature: Some(0.0),
            top_p: Some(1.0),
            top_k: Some(1),
            max_frames: Some(1),
        };
        assert!(edges.options(0).is_ok());
        let top = Drama3Sampling {
            max_frames: Some(DEFAULT_MAX_FRAMES),
            ..none
        };
        assert!(top.options(0).is_ok());
    }

    fn chunk(stop: ChunkStop) -> ChunkReport {
        ChunkReport {
            chunk: 0,
            frames: 100,
            limit: 424,
            stop,
            retries: 0,
        }
    }

    #[test]
    fn a_take_stopped_early_when_a_chunk_missed_eos_or_went_silent() {
        let done = Drama3Take {
            emitted: 1,
            chunks: vec![chunk(ChunkStop::Eos), chunk(ChunkStop::Eos)],
            longest_silence_frames: STUCK_SILENCE_FRAMES - 1,
        };
        assert!(!done.stopped_early());
        for stop in [ChunkStop::TextCap, ChunkStop::MaxFrames] {
            let mut capped = done.clone();
            capped.chunks[1] = chunk(stop);
            assert!(capped.stopped_early(), "{stop:?}");
        }
        // EOS after a long silence: the words were still never spoken.
        let silent = Drama3Take {
            longest_silence_frames: STUCK_SILENCE_FRAMES,
            ..done.clone()
        };
        assert!(silent.stopped_early());
        assert!(!Drama3Take::default().stopped_early());
    }

    fn frames(count: usize, amplitude: f32) -> Vec<f32> {
        vec![amplitude; count * FRAME_SAMPLES]
    }

    fn longest(pieces: &[Vec<f32>]) -> usize {
        let mut meter = SilenceMeter::default();
        for piece in pieces {
            meter.feed(piece);
        }
        meter.longest()
    }

    #[test]
    fn the_meter_finds_the_longest_silent_run_across_pieces() {
        // -45 dBFS is an RMS of about 0.0056.
        let mut pcm = frames(3, 0.1);
        pcm.extend(frames(5, 0.001));
        pcm.extend(frames(2, 0.1));
        pcm.extend(frames(7, 0.001));
        pcm.extend(frames(1, 0.1));
        assert_eq!(longest(std::slice::from_ref(&pcm)), 7);
        // Codec pieces do not line up with frames.
        let pieces: Vec<Vec<f32>> = pcm.chunks(1_000).map(<[f32]>::to_vec).collect();
        assert_eq!(longest(&pieces), 7);
        assert_eq!(longest(&[]), 0);
    }

    /// Every piece handed to the caller is measured, so the take's longest
    /// silence is the emitted audio's.
    #[test]
    fn delivered_pieces_feed_the_silence_meter() {
        let mut delivery = Delivery::default();
        let mut heard = 0usize;
        let mut emit = |pcm: Vec<f32>| {
            heard += pcm.len();
            true
        };
        delivery.take(Ok(frames(2, 0.1)), &mut emit);
        delivery.take(Ok(frames(3, 0.001)), &mut emit);
        delivery.take(Ok(Vec::new()), &mut emit);
        assert_eq!(delivery.silence.longest(), 3);
        assert_eq!(delivery.emitted, 5 * FRAME_SAMPLES);
        assert_eq!(heard, delivery.emitted);
    }

    #[test]
    fn the_meter_floor_is_minus_45_dbfs_and_counts_a_partial_last_frame() {
        assert_eq!(longest(&[frames(2, 0.0050)]), 2);
        assert_eq!(longest(&[frames(2, 0.0063)]), 0);
        let mut tail = frames(2, 0.1);
        tail.extend(vec![0.0; 100]);
        assert_eq!(longest(&[tail]), 1);
    }
}
