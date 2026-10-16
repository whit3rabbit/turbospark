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

/// One synthesis request.
pub struct Drama3Request<'a> {
    pub text: &'a str,
    /// Mono reference audio, its sample rate and its transcript, for cloning.
    pub reference: Option<(&'a [f32], u32, &'a str)>,
    pub seed: u64,
}

/// Drama 3 on the headless Metal trunk: the portable model, the trunk and the
/// Fish S1 DAC, with the last reference voice's codes cached.
pub struct FishDrama3Runner {
    drama: audio::tts::fish_drama3::generate::Drama3,
    backbone: TrunkBackbone,
    codec: crate::fish_drama3_codec::CodecWorker,
    /// Content hash of the last reference and its encoded utterance.
    voice: Option<(u64, audio::tts::fish_drama3::prompt::Utterance)>,
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
        })
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
    /// returns false to stop, and `stop` is polled once per frame. Returns the
    /// samples emitted.
    ///
    /// Generation runs on this thread; the codec runs on a scoped worker fed
    /// frame by frame, so decoding overlaps the next frames' generation.
    pub fn synthesize(
        &mut self,
        request: &Drama3Request<'_>,
        stop: &(dyn Fn() -> bool + Sync),
        mut emit: impl FnMut(Vec<f32>) -> bool + Send,
    ) -> std::result::Result<usize, String> {
        use audio::tts::fish_drama3::generate::{ChunkStop, Flow, FrameEvent, GenerateOptions};
        let references = match request.reference {
            Some((samples, rate, transcript)) => vec![self.reference(samples, rate, transcript)?],
            None => Vec::new(),
        };
        let options = GenerateOptions {
            seed: request.seed,
            ..GenerateOptions::default()
        };
        // The codec decodes on its own thread while generation runs here, and
        // its samples are handed to `emit` between frames and after the last.
        let (frames_tx, pcm_rx) = self.codec.stream(FIRST_PIECE_FRAMES, PIECE_FRAMES)?;
        let mut delivery = Delivery::default();
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
                    // A chunk cut short still plays, so it never ends silently.
                    FrameEvent::ChunkEnd {
                        chunk,
                        frames,
                        stop,
                    } if stop != ChunkStop::Eos => {
                        eprintln!(
                            "fish_drama3: chunk {chunk} stopped without EOS at {frames} frames ({stop:?})"
                        );
                    }
                    FrameEvent::ChunkEnd { .. } => {}
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
            None => Ok(delivery.emitted),
        }
    }
}

/// Samples handed to the caller so far, and whether to stop.
#[derive(Default)]
struct Delivery {
    emitted: usize,
    stopped: bool,
    error: Option<String>,
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
                self.stopped = !emit(pcm);
            }
            Err(e) => {
                self.error = Some(e);
                self.stopped = true;
            }
        }
    }
}
