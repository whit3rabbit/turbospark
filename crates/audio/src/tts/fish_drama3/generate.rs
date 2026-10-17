//! The Drama 3 generation loop, over any [`Backbone`].
//!
//! Port of `Model.generate`, `_frames`, `_prefill` and `_step` in mlx-audio
//! `fish_drama3/drama3.py` at [`super::REFERENCE_COMMIT`]. Each decode step
//! samples the frame's main token (EOS or a semantic token) from the
//! backbone's hidden state, lets the tower sample the other codebooks, and
//! feeds the frame back into the backbone. Frames go out through a callback
//! as they are produced; turning them into audio is the codec's job, on its
//! own stage.
//!
//! Long text is generated chunk by chunk (`chunking::split_text`); each chunk
//! sees the first and the previous chunk as history. The system block (the
//! references) is prefilled once per call and restored for every chunk.
//!
//! A chunk ends at EOS, at a cap scaled to its text ([`TextCap`], not in the
//! reference), or at `max_frames`. The reference stops only at the last, and
//! under its recipe about one clip in twenty runs away to it.

use super::chunking::{split_text, spoken_text, validate_text};
use super::config::Drama3Config;
use super::prompt::{build_frames, Prompt, Utterance};
use super::sampling::{self, SamplingParams, DEFAULT_MAX_FRAMES, RAS_WINDOW};
use super::tower::{CpuTower, Tower, TowerModel};
use crate::music::minimax_music3::rng::KeySequence;
use crate::{Result, SpeechError};
use turbospark_model_io::safetensors::SafetensorsFile;

/// One backbone input row.
#[derive(Clone, Debug)]
pub enum BackboneRow {
    /// A token id, looked up in the backbone's own table.
    Token(i32),
    /// A host-computed input row (an audio frame's mixed embedding).
    Embedding(Vec<f32>),
}

/// The text backbone as the loop drives it: rows in, post-final-norm hidden
/// states out. `crates/runtime`'s `HiddenTrunk` is the Metal implementation.
pub trait Backbone {
    /// A restore point for [`Backbone::restore`].
    type Snapshot;

    /// Width of a hidden state and of an input row.
    fn hidden_size(&self) -> usize;
    /// The backbone's own embedding row for `token`.
    fn embedding_row(&mut self, token: i32, out: &mut [f32]) -> Result<()>;
    /// Feeds `rows` in order; `hidden_out` receives one hidden state per row.
    fn prefill(&mut self, rows: &[BackboneRow], hidden_out: &mut [f32]) -> Result<()>;
    /// Feeds one row.
    fn step(&mut self, row: &BackboneRow, hidden_out: &mut [f32]) -> Result<()>;
    /// A restore point at the current position.
    fn snapshot(&mut self) -> Result<Self::Snapshot>;
    /// Returns to `snapshot`, dropping every row after it.
    fn restore(&mut self, snapshot: &Self::Snapshot) -> Result<()>;
    /// Empties the backbone.
    fn reset(&mut self);
}

/// A per-chunk frame cap scaled to the chunk's text, so a sampling runaway
/// stops near the longest the text plausibly takes instead of at
/// `max_frames`. Tokens are the backbone tokenizer's, counted over the
/// chunk's words (`chunking::spoken_text`): unlike bytes or words, they count
/// digits one by one and stay close across scripts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TextCap {
    /// Frames any chunk may take, so a one-word chunk still fits.
    pub base_frames: usize,
    /// Frames per token of the chunk's words.
    pub frames_per_token: usize,
    /// Frames per `<|speaker:N|>` tag, a turn's opening pause.
    pub frames_per_speaker_tag: usize,
    /// Frames per cue or paired tag (`[sighs]`, `<whisper>...</whisper>`).
    pub frames_per_cue: usize,
}

impl TextCap {
    /// From 125 normal clips under the reference recipe (this port and
    /// mlx-audio, 4-bit): the slowest ran 13.6 frames per token, the median
    /// 6.8, and every clip stays under half its cap. A sigh added at most
    /// about 50 frames. Speaker tags have no normal measurement (both dialogue
    /// clips ran away), so their 1 s is a turn gap with margin.
    /// `docs/FISH_DRAMA3_PHASE0.md` (sampling recipe) has the numbers.
    pub const MEASURED: TextCap = TextCap {
        base_frames: 64,
        frames_per_token: 24,
        frames_per_speaker_tag: 22,
        frames_per_cue: 108,
    };

    /// The cap for a chunk of `tokens` word tokens, `speaker_tags` speaker
    /// tags and `cues` cues.
    pub fn frames(&self, tokens: usize, speaker_tags: usize, cues: usize) -> usize {
        self.base_frames
            .saturating_add(self.frames_per_token.saturating_mul(tokens))
            .saturating_add(self.frames_per_speaker_tag.saturating_mul(speaker_tags))
            .saturating_add(self.frames_per_cue.saturating_mul(cues))
    }
}

/// What one call generates with.
#[derive(Clone, Copy, Debug)]
pub struct GenerateOptions {
    pub sampling: SamplingParams,
    /// Frames one chunk may generate before it is cut off, whatever its text.
    pub max_frames: usize,
    /// A tighter cap per chunk, scaled to its text. `None` leaves only
    /// `max_frames`, as the reference does.
    pub text_cap: Option<TextCap>,
    /// Seeds every draw; a fixed seed fixes the codes.
    pub seed: u64,
    /// Run the next backbone step beside the tower's last residual layers.
    /// The next backbone row needs only `main` and `c1`, which the tower's
    /// first pass yields, so the step and the tail of the frame (heads 1 and
    /// up) overlap on two threads. The draws are made in the same order, so
    /// the codes are the serial loop's (`overlap_and_serial_frames_agree`).
    pub overlap_tower_tail: bool,
}

impl Default for GenerateOptions {
    fn default() -> Self {
        GenerateOptions {
            sampling: SamplingParams::FISH_SERVING,
            max_frames: DEFAULT_MAX_FRAMES,
            text_cap: Some(TextCap::MEASURED),
            seed: 0,
            overlap_tower_tail: true,
        }
    }
}

/// Why a chunk ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkStop {
    /// The model ended it.
    Eos,
    /// It reached the cap scaled to its text before EOS, most likely a
    /// sampling runaway.
    TextCap,
    /// It reached `max_frames` before EOS.
    MaxFrames,
}

/// What the loop reports as it goes.
#[derive(Debug, Clone, PartialEq)]
pub enum FrameEvent {
    /// A generated frame's codes, `num_codebooks` of them.
    Frame { chunk: usize, codes: Vec<i32> },
    /// A text chunk finished; `frames` is its count and `limit` the most it
    /// could take (its text cap or `max_frames`).
    ChunkEnd {
        chunk: usize,
        frames: usize,
        limit: usize,
        stop: ChunkStop,
    },
}

/// Whether the caller wants the loop to keep going.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Flow {
    Continue,
    Stop,
}

/// Totals for one call.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GenerateStats {
    pub chunks: usize,
    pub frames: usize,
    pub prompt_rows: usize,
    pub stopped: bool,
}

/// Files of a Drama 3 install, beside `config.json`.
pub const TOKENIZER_FILE: &str = "tokenizer.json";
/// The tower and semantic head (`crates/repack`'s Drama intake).
pub const SIDE_FILE: &str = "tts_side.safetensors";
/// The Fish S1 DAC weights.
pub const CODEC_FILE: &str = "codec.safetensors";
/// The headless text trunk install.
pub const TRUNK_DIR: &str = "trunk";

/// The TTS side of Drama 3: config, tokenizer and tower.
pub struct Drama3 {
    pub config: Drama3Config,
    pub tokenizer: tokenizers::Tokenizer,
    pub tower: Box<dyn TowerModel + Send>,
}

/// The state after the system block, restored for every chunk. The tower
/// only needs its row count: rows past it are rewritten by the next chunk.
struct VoicePrefix<S> {
    backbone: S,
    rows: usize,
}

impl Drama3 {
    /// Loads `config.json`, the tokenizer and the side tensors from a Drama 3
    /// install, checking the tokenizer against the config, with the f32
    /// reference tower.
    pub fn open(install: &std::path::Path) -> Result<Self> {
        Self::open_with(install, |side, config| {
            let tower = Tower::load(side, config.tower.clone(), config.semantic_band())?;
            Ok(Box::new(CpuTower::new(tower)))
        })
    }

    /// [`Drama3::open`] with the tower `build` makes from the side file, so a
    /// device tower never loads the reference one first.
    pub fn open_with(
        install: &std::path::Path,
        build: impl FnOnce(&SafetensorsFile, &Drama3Config) -> Result<Box<dyn TowerModel + Send>>,
    ) -> Result<Self> {
        let text = std::fs::read_to_string(install.join("config.json")).map_err(|e| {
            SpeechError::BadConfig {
                field: "config.json".into(),
                why: e.to_string(),
            }
        })?;
        let root: serde_json::Value =
            serde_json::from_str(&text).map_err(|e| SpeechError::BadConfig {
                field: "config.json".into(),
                why: e.to_string(),
            })?;
        let config = Drama3Config::from_json(&root)?;
        let tokenizer =
            tokenizers::Tokenizer::from_file(install.join(TOKENIZER_FILE)).map_err(|e| {
                SpeechError::BadConfig {
                    field: TOKENIZER_FILE.into(),
                    why: e.to_string(),
                }
            })?;
        config.check_tokenizer(&tokenizer)?;
        let side = SafetensorsFile::open(&install.join(SIDE_FILE))?;
        let tower = build(&side, &config)?;
        Ok(Drama3 {
            config,
            tokenizer,
            tower,
        })
    }

    fn row_for(&self, backbone: &mut impl Backbone, frame: &[i32]) -> Result<BackboneRow> {
        let embeddings = self.tower.embeddings();
        if !embeddings.is_semantic(frame[0]) {
            return Ok(BackboneRow::Token(frame[0]));
        }
        let mut text = vec![0.0f32; backbone.hidden_size()];
        backbone.embedding_row(frame[0], &mut text)?;
        Ok(BackboneRow::Embedding(
            embeddings.embed_backbone(frame, &text),
        ))
    }

    /// Feeds prompt rows `[start, stop)` into both models and returns row
    /// `stop - 1`'s hidden state. The tower is teacher-forced with the
    /// following row, so it covers `[start, min(stop, T - 1))`.
    fn prefill<B: Backbone>(
        &mut self,
        backbone: &mut B,
        rows: &[Vec<i32>],
        start: usize,
        stop: usize,
    ) -> Result<Vec<f32>> {
        let dim = backbone.hidden_size();
        let inputs = rows[start..stop]
            .iter()
            .map(|row| self.row_for(backbone, row))
            .collect::<Result<Vec<_>>>()?;
        let mut hidden = vec![0.0f32; inputs.len() * dim];
        backbone.prefill(&inputs, &mut hidden)?;
        let labelled = stop.min(rows.len() - 1);
        if labelled > start {
            let frames: Vec<&[i32]> = rows[start..labelled].iter().map(Vec::as_slice).collect();
            let labels: Vec<&[i32]> = rows[start + 1..=labelled]
                .iter()
                .map(Vec::as_slice)
                .collect();
            self.tower
                .prefill(&hidden[..(labelled - start) * dim], &frames, &labels)?;
        }
        Ok(hidden[(stop - start - 1) * dim..].to_vec())
    }

    /// Frames `chunk` may generate, and why it ends if it uses them all.
    fn frame_limit(&self, chunk: &str, options: &GenerateOptions) -> Result<(usize, ChunkStop)> {
        let Some(cap) = options.text_cap else {
            return Ok((options.max_frames, ChunkStop::MaxFrames));
        };
        let spoken = spoken_text(chunk);
        let tokens = self
            .tokenizer
            .encode(spoken.words.as_str(), false)
            .map_err(|e| SpeechError::Input {
                why: format!("tokenizing {:?}: {e}", spoken.words),
            })?
            .get_ids()
            .len();
        let frames = cap.frames(tokens, spoken.speaker_tags, spoken.cues);
        Ok(if frames < options.max_frames {
            (frames, ChunkStop::TextCap)
        } else {
            (options.max_frames, ChunkStop::MaxFrames)
        })
    }

    /// Speech for `text` in the voice of `references` (empty: the model picks
    /// one), reported frame by frame through `on_event`.
    pub fn generate<B: Backbone>(
        &mut self,
        backbone: &mut B,
        references: &[Utterance],
        text: &str,
        options: GenerateOptions,
        mut on_event: impl FnMut(FrameEvent) -> Flow,
    ) -> Result<GenerateStats> {
        validate_text(text, "text")?;
        options.sampling.validate()?;
        if options.max_frames == 0 {
            return Err(SpeechError::Input {
                why: "max_frames must be at least 1".into(),
            });
        }
        if options.text_cap.is_some_and(|cap| cap.base_frames == 0) {
            return Err(SpeechError::Input {
                why: "text_cap.base_frames must be at least 1".into(),
            });
        }
        if backbone.hidden_size() != self.config.hidden_size {
            return Err(SpeechError::Input {
                why: format!(
                    "the backbone's hidden size {} is not the config's {}",
                    backbone.hidden_size(),
                    self.config.hidden_size
                ),
            });
        }
        let candidates = self.config.candidate_ids();
        let band = self.config.semantic_band();
        let eos = self.config.eos_token_id;
        let dim = backbone.hidden_size();
        let mut keys = KeySequence::new(options.seed);
        let mut stats = GenerateStats::default();
        let mut prefix: Option<VoicePrefix<B::Snapshot>> = None;
        let mut pinned: Option<Utterance> = None;
        let mut latest: Option<Utterance> = None;

        backbone.reset();
        for (chunk_index, chunk) in split_text(text).iter().enumerate() {
            let history: Vec<Utterance> = pinned.iter().chain(&latest).cloned().collect();
            let Prompt { rows, system_rows } =
                build_frames(&self.tokenizer, &self.config, references, &history, chunk)?;
            stats.prompt_rows += rows.len();

            // The system block once per call, then a restore per chunk.
            match &prefix {
                Some(p) if p.rows == system_rows => {
                    backbone.restore(&p.backbone)?;
                    self.tower.truncate(p.rows)?;
                }
                _ => {
                    backbone.reset();
                    self.tower.truncate(0)?;
                    self.prefill(backbone, &rows, 0, system_rows)?;
                    prefix = Some(VoicePrefix {
                        backbone: backbone.snapshot()?,
                        rows: system_rows,
                    });
                }
            }
            let mut hidden = self.prefill(backbone, &rows, system_rows, rows.len())?;
            let (limit, cut) = self.frame_limit(chunk, &options)?;

            let mut history_tokens: Vec<i32> = rows[rows.len().saturating_sub(RAS_WINDOW)..]
                .iter()
                .map(|r| r[0])
                .collect();
            let mut frame = rows[rows.len() - 1].clone();
            let mut codes_by_frame: Vec<Vec<i32>> = Vec::new();
            let mut eos_seen = false;
            for _ in 0..limit {
                let logits = self.tower.semantic_logits(&hidden)?;
                let main = sampling::main_token(
                    &logits,
                    &candidates,
                    options.sampling,
                    &history_tokens,
                    band.clone(),
                    keys.next(),
                    keys.next(),
                );
                if main == eos {
                    eos_seen = true;
                    break;
                }
                let mut sample = |_: usize, head: &[f32]| {
                    sampling::sample(head, options.sampling, keys.next()) as i32
                };
                let codes = if options.overlap_tower_tail {
                    let [c0, c1] =
                        self.tower
                            .begin_residual_codes(&hidden, main, &frame, &mut sample)?;
                    // The backbone input reads `main` and `c1` only
                    // (`TowerEmbeddings::embed_backbone`).
                    let row = self.row_for(backbone, &[main, c0, c1])?;
                    let tower = &mut self.tower;
                    hidden = vec![0.0f32; dim];
                    let next = &mut hidden;
                    let (tail, stepped) = std::thread::scope(|scope| {
                        let tail = scope.spawn(move || tower.finish_residual_codes(&mut sample));
                        let stepped = backbone.step(&row, next);
                        (tail.join(), stepped)
                    });
                    let codes = tail.map_err(|_| SpeechError::Unsupported {
                        why: "the tower thread panicked".into(),
                    })??;
                    stepped?;
                    codes
                } else {
                    let codes = self
                        .tower
                        .residual_codes(&hidden, main, &frame, &mut sample)?;
                    let mut new_frame = vec![main];
                    new_frame.extend(&codes);
                    let row = self.row_for(backbone, &new_frame)?;
                    hidden = vec![0.0f32; dim];
                    backbone.step(&row, &mut hidden)?;
                    codes
                };
                let mut new_frame = vec![main];
                new_frame.extend(&codes);
                // `history[1:] + [main]`: a window shorter than RAS_WINDOW
                // keeps its length, as the reference's does.
                if !history_tokens.is_empty() {
                    history_tokens.remove(0);
                }
                history_tokens.push(main);
                frame = new_frame;
                codes_by_frame.push(codes.clone());
                stats.frames += 1;
                if on_event(FrameEvent::Frame {
                    chunk: chunk_index,
                    codes,
                }) == Flow::Stop
                {
                    stats.stopped = true;
                    return Ok(stats);
                }
            }
            stats.chunks += 1;
            let frames = codes_by_frame.len();
            if on_event(FrameEvent::ChunkEnd {
                chunk: chunk_index,
                frames,
                limit,
                stop: if eos_seen { ChunkStop::Eos } else { cut },
            }) == Flow::Stop
            {
                stats.stopped = true;
                return Ok(stats);
            }
            if frames > 0 {
                let books = self.config.tower.num_codebooks;
                let turn = Utterance {
                    text: chunk.clone(),
                    codes: (0..books)
                        .map(|b| codes_by_frame.iter().map(|f| f[b]).collect())
                        .collect(),
                };
                if pinned.is_none() {
                    pinned = Some(turn);
                } else {
                    latest = Some(turn);
                }
            }
        }
        Ok(stats)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tts::fish_drama3::tower::TowerConfig;
    use turbospark_model_io::safetensors::SafetensorsFile;

    /// A deterministic stand-in for the trunk: a leaky tanh recurrence over
    /// its input rows, so every hidden state depends on every earlier row.
    struct FakeBackbone {
        dim: usize,
        state: Vec<f32>,
        snapshots: usize,
        restores: usize,
        rows_fed: usize,
    }

    impl FakeBackbone {
        fn new(dim: usize) -> Self {
            FakeBackbone {
                dim,
                state: vec![0.0; dim],
                snapshots: 0,
                restores: 0,
                rows_fed: 0,
            }
        }

        fn token_row(&self, token: i32) -> Vec<f32> {
            (0..self.dim)
                .map(|i| (((token as usize * 131 + i * 17) % 97) as f32 / 97.0 - 0.5) * 2.0)
                .collect()
        }
    }

    impl Backbone for FakeBackbone {
        type Snapshot = Vec<f32>;

        fn hidden_size(&self) -> usize {
            self.dim
        }

        fn embedding_row(&mut self, token: i32, out: &mut [f32]) -> Result<()> {
            out.copy_from_slice(&self.token_row(token));
            Ok(())
        }

        fn prefill(&mut self, rows: &[BackboneRow], hidden_out: &mut [f32]) -> Result<()> {
            for (row, out) in rows.iter().zip(hidden_out.chunks_exact_mut(self.dim)) {
                self.step(row, out)?;
            }
            Ok(())
        }

        fn step(&mut self, row: &BackboneRow, hidden_out: &mut [f32]) -> Result<()> {
            let x = match row {
                BackboneRow::Token(t) => self.token_row(*t),
                BackboneRow::Embedding(v) => v.clone(),
            };
            for ((s, xi), out) in self.state.iter_mut().zip(&x).zip(hidden_out.iter_mut()) {
                *s = (0.6 * *s + xi).tanh();
                *out = *s * 3.0;
            }
            self.rows_fed += 1;
            Ok(())
        }

        fn snapshot(&mut self) -> Result<Vec<f32>> {
            self.snapshots += 1;
            Ok(self.state.clone())
        }

        fn restore(&mut self, snapshot: &Vec<f32>) -> Result<()> {
            self.restores += 1;
            self.state = snapshot.clone();
            Ok(())
        }

        fn reset(&mut self) {
            self.state = vec![0.0; self.dim];
        }
    }

    fn drama() -> Drama3 {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/fish_drama3");
        let golden: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("tower.json")).unwrap())
                .unwrap();
        let tower_cfg = TowerConfig::from_json(&golden["tower"]).unwrap();
        let config = Drama3Config {
            semantic_start_token_id: 32,
            semantic_end_token_id: 47,
            eos_token_id: 2,
            hidden_size: tower_cfg.dim,
            tower: tower_cfg.clone(),
        };
        let file = SafetensorsFile::open(&dir.join("tower_side.safetensors")).unwrap();
        let tower = Tower::load(&file, tower_cfg, config.semantic_band()).unwrap();
        let tokenizer =
            tokenizers::Tokenizer::from_file(dir.join("prompt_tokenizer.json")).unwrap();
        config.check_tokenizer(&tokenizer).unwrap();
        Drama3 {
            config,
            tokenizer,
            tower: Box::new(CpuTower::new(tower)),
        }
    }

    fn run(
        d: &mut Drama3,
        text: &str,
        seed: u64,
        max_frames: usize,
    ) -> (Vec<FrameEvent>, FakeBackbone) {
        let mut backbone = FakeBackbone::new(d.config.hidden_size);
        let mut events = Vec::new();
        let options = GenerateOptions {
            max_frames,
            seed,
            ..GenerateOptions::default()
        };
        d.generate(&mut backbone, &[], text, options, |e| {
            events.push(e);
            Flow::Continue
        })
        .unwrap();
        (events, backbone)
    }

    fn run_with(d: &mut Drama3, text: &str, options: GenerateOptions) -> Vec<FrameEvent> {
        let mut backbone = FakeBackbone::new(d.config.hidden_size);
        let mut events = Vec::new();
        d.generate(&mut backbone, &[], text, options, |e| {
            events.push(e);
            Flow::Continue
        })
        .unwrap();
        events
    }

    /// The last chunk's frame count and stop.
    fn ending(events: &[FrameEvent]) -> (usize, ChunkStop) {
        match events.last() {
            Some(FrameEvent::ChunkEnd { frames, stop, .. }) => (*frames, *stop),
            other => panic!("the call must end with a chunk end, got {other:?}"),
        }
    }

    fn tokens(d: &Drama3, words: &str) -> usize {
        d.tokenizer.encode(words, false).unwrap().get_ids().len()
    }

    /// The product samples with Fish Audio's serving recipe. Under the
    /// mlx-audio recipe a sentence gap trapped 4 of 12 seeds of one script in
    /// silence, against 0 of 12 (`docs/FISH_DRAMA3_PHASE0.md`).
    #[test]
    fn the_default_samples_with_fish_audios_serving_recipe() {
        let sampling = GenerateOptions::default().sampling;
        assert_eq!(sampling, SamplingParams::FISH_SERVING);
        assert_eq!(
            (sampling.temperature, sampling.top_k, sampling.top_p),
            (1.0, 30, 0.9)
        );
        assert_ne!(sampling, SamplingParams::REFERENCE);
    }

    #[test]
    fn the_text_cap_adds_base_tokens_tags_and_cues() {
        let cap = TextCap::MEASURED;
        assert_eq!(cap.frames(0, 0, 0), cap.base_frames);
        assert_eq!(
            cap.frames(10, 1, 2),
            cap.base_frames
                + 10 * cap.frames_per_token
                + cap.frames_per_speaker_tag
                + 2 * cap.frames_per_cue
        );
        // Eval sentence s07, 21 tokens: 26 s of audio, against 47.6 s at
        // max_frames.
        assert_eq!(cap.frames(21, 0, 0), 568);
        let huge = TextCap {
            frames_per_token: usize::MAX,
            ..cap
        };
        assert_eq!(huge.frames(2, 0, 0), usize::MAX);
    }

    #[test]
    fn a_chunk_limit_counts_its_words_and_cues_and_never_passes_max_frames() {
        let d = drama();
        let cap = TextCap::MEASURED;
        let chunk = "[sighs] Hallo du.";
        let expected = cap.frames(tokens(&d, " Hallo du."), 0, 1);
        let options = GenerateOptions::default();
        assert_eq!(
            d.frame_limit(chunk, &options).unwrap(),
            (expected, ChunkStop::TextCap)
        );
        let tight = GenerateOptions {
            max_frames: expected,
            ..options
        };
        assert_eq!(
            d.frame_limit(chunk, &tight).unwrap(),
            (expected, ChunkStop::MaxFrames)
        );
        let off = GenerateOptions {
            text_cap: None,
            ..options
        };
        assert_eq!(
            d.frame_limit(chunk, &off).unwrap(),
            (DEFAULT_MAX_FRAMES, ChunkStop::MaxFrames)
        );
    }

    #[test]
    fn the_text_cap_cuts_a_runaway_chunk_short_and_says_so() {
        let mut d = drama();
        let text = "Hallo du.";
        // One frame per token on top of one: tight enough for this tower.
        let cap = TextCap {
            base_frames: 1,
            frames_per_token: 1,
            frames_per_speaker_tag: 0,
            frames_per_cue: 0,
        };
        let limit = cap.frames(tokens(&d, text), 0, 0);
        let uncapped = |seed| GenerateOptions {
            max_frames: 64,
            text_cap: None,
            seed,
            ..GenerateOptions::default()
        };
        let (seed, free) = (0..64)
            .map(|seed| (seed, run_with(&mut d, text, uncapped(seed))))
            .find(|(_, events)| ending(events).0 > limit)
            .expect("a seed whose chunk runs past the cap");
        let capped = run_with(
            &mut d,
            text,
            GenerateOptions {
                text_cap: Some(cap),
                ..uncapped(seed)
            },
        );
        assert_eq!(ending(&capped), (limit, ChunkStop::TextCap));
        assert!(
            matches!(capped.last(), Some(FrameEvent::ChunkEnd { limit: l, .. }) if *l == limit),
            "the chunk end reports the cap it ran into"
        );
        // The cap cuts the same frames short; it does not change them.
        assert_eq!(capped[..limit], free[..limit]);
        // A cap at max_frames leaves max_frames in charge.
        let at_max = run_with(
            &mut d,
            text,
            GenerateOptions {
                max_frames: limit,
                text_cap: Some(cap),
                ..uncapped(seed)
            },
        );
        assert_eq!(ending(&at_max), (limit, ChunkStop::MaxFrames));
    }

    /// The overlapped loop draws in the serial loop's order and feeds the
    /// backbone the same rows, so every frame's codes agree, across chunks.
    #[test]
    fn overlap_and_serial_frames_agree() {
        let text = "Hallo du. Wie geht es dir heute? Mir geht es gut, danke.";
        let mut by_mode = Vec::new();
        for overlap in [false, true] {
            let mut d = drama();
            let mut backbone = FakeBackbone::new(d.config.hidden_size);
            let mut events = Vec::new();
            // The recipe and seed this test was sized for: the fake model
            // ends early at seed 5 under the default recipe. Draw order does
            // not depend on the recipe.
            let options = GenerateOptions {
                sampling: SamplingParams::REFERENCE,
                max_frames: 12,
                seed: 5,
                overlap_tower_tail: overlap,
                ..GenerateOptions::default()
            };
            d.generate(&mut backbone, &[], text, options, |e| {
                events.push(e);
                Flow::Continue
            })
            .unwrap();
            // The fake's state folds in every row it was fed, in order.
            let fed: Vec<u32> = backbone.state.iter().map(|v| v.to_bits()).collect();
            by_mode.push((events, (fed, backbone.rows_fed)));
        }
        let frames = |events: &[FrameEvent]| {
            events
                .iter()
                .filter(|e| matches!(e, FrameEvent::Frame { .. }))
                .count()
        };
        assert!(frames(&by_mode[0].0) > 8, "too few frames to compare");
        assert_eq!(by_mode[0].0, by_mode[1].0, "frame codes");
        assert_eq!(by_mode[0].1, by_mode[1].1, "backbone rows");
    }

    #[test]
    fn a_seed_fixes_the_codes_and_codes_stay_in_range() {
        let mut d = drama();
        let (a, _) = run(&mut d, "Hallo du.", 4, 8);
        let (b, _) = run(&mut d, "Hallo du.", 4, 8);
        assert_eq!(a, b);
        let books = d.config.tower.num_codebooks;
        for event in &a {
            if let FrameEvent::Frame { codes, .. } = event {
                assert_eq!(codes.len(), books);
                assert!(codes.iter().all(|&c| (0..16).contains(&c)), "{codes:?}");
            }
        }
        assert!(matches!(
            a.last(),
            Some(FrameEvent::ChunkEnd { chunk: 0, .. })
        ));
    }

    #[test]
    fn long_text_restores_the_system_block_for_every_chunk() {
        let mut d = drama();
        let text = "Ein kurzer Satz mit etwas Inhalt. ".repeat(14);
        let (events, backbone) = run(&mut d, &text, 1, 3);
        let ends: Vec<usize> = events
            .iter()
            .filter_map(|e| match e {
                FrameEvent::ChunkEnd { chunk, .. } => Some(*chunk),
                _ => None,
            })
            .collect();
        assert!(ends.len() >= 2, "expected several chunks, got {ends:?}");
        assert_eq!(ends, (0..ends.len()).collect::<Vec<_>>());
        assert_eq!(backbone.snapshots, 1, "the system block is prefilled once");
        assert_eq!(
            backbone.restores,
            ends.len() - 1,
            "and restored for every later chunk"
        );
    }

    #[test]
    fn stop_ends_the_call_after_the_current_frame() {
        let mut d = drama();
        let mut backbone = FakeBackbone::new(d.config.hidden_size);
        let mut frames = 0;
        let stats = d
            .generate(
                &mut backbone,
                &[],
                "Hallo du.",
                GenerateOptions {
                    max_frames: 8,
                    ..Default::default()
                },
                |e| {
                    if matches!(e, FrameEvent::Frame { .. }) {
                        frames += 1;
                        return Flow::Stop;
                    }
                    Flow::Continue
                },
            )
            .unwrap();
        assert!(stats.stopped || frames == 0);
        assert!(frames <= 1);
    }

    #[test]
    fn refuses_bad_options_and_mismatched_backbones() {
        let mut d = drama();
        let mut backbone = FakeBackbone::new(d.config.hidden_size);
        let zero = GenerateOptions {
            max_frames: 0,
            ..Default::default()
        };
        assert!(d
            .generate(&mut backbone, &[], "Hallo.", zero, |_| Flow::Continue)
            .is_err());
        let no_base = GenerateOptions {
            text_cap: Some(TextCap {
                base_frames: 0,
                ..TextCap::MEASURED
            }),
            ..Default::default()
        };
        assert!(d
            .generate(&mut backbone, &[], "Hallo.", no_base, |_| Flow::Continue)
            .is_err());
        let mut narrow = FakeBackbone::new(d.config.hidden_size - 1);
        assert!(d
            .generate(&mut narrow, &[], "Hallo.", Default::default(), |_| {
                Flow::Continue
            })
            .is_err());
        assert!(d
            .generate(&mut backbone, &[], "<|im_end|>", Default::default(), |_| {
                Flow::Continue
            })
            .is_err());
    }
}
