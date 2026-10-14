//! EmbeddingGemma 2 word-CTC transcription on Core ML (experiment).
//!
//! Backs the portable `audio::stt::embeddinggemma2` pipeline with the three
//! compiled programs of Haku's eg2demo v5 distribution:
//!
//! - `eg2_audio_l11.mlmodelc`: `input_features [1,2000,128]` and
//!   `feature_mask [1,2000]` (f32) to `l11 [1,500,1024]` (f16).
//! - `reader_P6.mlmodelc`: `l11 [1,500,1024]` and `mask [1,500]` (f32) to
//!   `vec [1,500,768]` and `blank [1,500,1]` (f16).
//! - `scorer_blk_d128_k16_b4_t500.mlmodelc`: `vec [1,500,768]` (f16) to the
//!   sorted top-16 block ids `bidx [1,500,16]` (i32).
//!
//! Every program's description is checked against these shapes at open, so
//! a different distribution is refused rather than mis-decoded. All three
//! are iOS 18 ML programs, so macOS 15 is required. The log-mel frontend
//! uses the shipped `hann_320.f32` and `mel_filters_257x128.f32` tables.
//!
//! What is and is not verified lives in
//! `crates/audio/src/stt/embeddinggemma2/README.md`.

use std::path::Path;
use std::time::{Duration, Instant};

use audio::stt::embeddinggemma2::frontend::NUM_MELS;
use audio::stt::embeddinggemma2::scorer::{BLOCK_SIZE, SCORER_COLUMNS, TOP_BLOCKS};
use audio::stt::embeddinggemma2::windows::{ENCODER_FRAMES, MEL_FRAMES};
use audio::stt::embeddinggemma2::{
    AcousticBackend, BlockShortlist, BlockShortlister, Calibration, LogMelFrontend, ReaderOutput,
    Transcriber, WordList, WordTable, EMBEDDING_DIM, ENCODER_DIM,
};
use audio::SpeechError;
use turbospark_coreml::{ComputeUnits, CoreMlModel, DType, FeatureKind, LoadOptions, Tensor};

pub const ENCODER_PROGRAM: &str = "eg2_audio_l11.mlmodelc";
pub const READER_PROGRAM: &str = "reader_P6.mlmodelc";
pub const SCORER_PROGRAM: &str = "scorer_blk_d128_k16_b4_t500.mlmodelc";

const ENCODER_UNITS_ENV: &str = "TURBOSPARK_EG2_CTC_ENCODER_UNITS";
const READER_UNITS_ENV: &str = "TURBOSPARK_EG2_CTC_READER_UNITS";
const SCORER_UNITS_ENV: &str = "TURBOSPARK_EG2_CTC_SCORER_UNITS";

/// The transcriber over the Core ML encoder, reader, and scorer.
pub type Eg2CtcCoreMl = Transcriber<CoreMlAcoustic, CoreMlScorer>;

/// Compute units per program.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Eg2CtcOptions {
    pub encoder_units: ComputeUnits,
    pub reader_units: ComputeUnits,
    pub scorer_units: ComputeUnits,
}

impl Default for Eg2CtcOptions {
    /// `All` everywhere: the `MLModelConfiguration` default, which is what
    /// a stock app that never sets `computeUnits` runs with.
    fn default() -> Self {
        Self {
            encoder_units: ComputeUnits::All,
            reader_units: ComputeUnits::All,
            scorer_units: ComputeUnits::All,
        }
    }
}

impl Eg2CtcOptions {
    /// Defaults, overridable per program through
    /// `TURBOSPARK_EG2_CTC_{ENCODER,READER,SCORER}_UNITS` (`cpu`,
    /// `cpu_gpu`, `cpu_ane`, `all`). An unknown value is an error.
    pub fn from_env() -> Result<Self, String> {
        let read = |name: &str, default: ComputeUnits| -> Result<ComputeUnits, String> {
            match std::env::var(name) {
                Ok(value) => ComputeUnits::parse(value.trim()).ok_or_else(|| {
                    format!("{name}={value} is not one of cpu, cpu_gpu, cpu_ane, all")
                }),
                Err(_) => Ok(default),
            }
        };
        let defaults = Self::default();
        Ok(Self {
            encoder_units: read(ENCODER_UNITS_ENV, defaults.encoder_units)?,
            reader_units: read(READER_UNITS_ENV, defaults.reader_units)?,
            scorer_units: read(SCORER_UNITS_ENV, defaults.scorer_units)?,
        })
    }
}

/// Accumulated prediction time for one program.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StageTime {
    pub calls: usize,
    pub total: Duration,
}

impl StageTime {
    fn add(&mut self, started: Instant) {
        self.calls += 1;
        self.total += started.elapsed();
    }
}

/// The encoder and reader programs.
pub struct CoreMlAcoustic {
    encoder: CoreMlModel,
    reader: CoreMlModel,
    pub encode_time: StageTime,
    pub read_time: StageTime,
}

impl std::fmt::Debug for CoreMlAcoustic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CoreMlAcoustic")
            .field("encoder", &self.encoder.path())
            .field("reader", &self.reader.path())
            .finish_non_exhaustive()
    }
}

impl CoreMlAcoustic {
    pub fn open(dir: &Path, opts: &Eg2CtcOptions) -> Result<Self, String> {
        require_macos_15()?;
        let encoder = load(dir, ENCODER_PROGRAM, opts.encoder_units)?;
        expect(
            &encoder,
            Side::Input,
            "input_features",
            DType::F32,
            &[1, MEL_FRAMES, NUM_MELS],
        )?;
        expect(
            &encoder,
            Side::Input,
            "feature_mask",
            DType::F32,
            &[1, MEL_FRAMES],
        )?;
        expect(
            &encoder,
            Side::Output,
            "l11",
            DType::F16,
            &[1, ENCODER_FRAMES, ENCODER_DIM],
        )?;
        let reader = load(dir, READER_PROGRAM, opts.reader_units)?;
        expect(
            &reader,
            Side::Input,
            "l11",
            DType::F32,
            &[1, ENCODER_FRAMES, ENCODER_DIM],
        )?;
        expect(
            &reader,
            Side::Input,
            "mask",
            DType::F32,
            &[1, ENCODER_FRAMES],
        )?;
        expect(
            &reader,
            Side::Output,
            "vec",
            DType::F16,
            &[1, ENCODER_FRAMES, EMBEDDING_DIM],
        )?;
        expect(
            &reader,
            Side::Output,
            "blank",
            DType::F16,
            &[1, ENCODER_FRAMES, 1],
        )?;
        Ok(Self {
            encoder,
            reader,
            encode_time: StageTime::default(),
            read_time: StageTime::default(),
        })
    }
}

impl AcousticBackend for CoreMlAcoustic {
    fn encode(&mut self, features: &[f32], mask: &[f32]) -> audio::Result<Vec<f32>> {
        let features = Tensor::f32(vec![1, MEL_FRAMES, NUM_MELS], features.to_vec())
            .map_err(|e| failure(ENCODER_PROGRAM, e))?;
        let mask = Tensor::f32(vec![1, MEL_FRAMES], mask.to_vec())
            .map_err(|e| failure(ENCODER_PROGRAM, e))?;
        let started = Instant::now();
        let out = self
            .encoder
            .predict(&[("input_features", &features), ("feature_mask", &mask)])
            .map_err(|e| failure(ENCODER_PROGRAM, e))?;
        self.encode_time.add(started);
        output(
            &out,
            ENCODER_PROGRAM,
            "l11",
            &[1, ENCODER_FRAMES, ENCODER_DIM],
        )
    }

    fn read(&mut self, hidden: &[f32], mask: &[f32]) -> audio::Result<ReaderOutput> {
        let hidden = Tensor::f32(vec![1, ENCODER_FRAMES, ENCODER_DIM], hidden.to_vec())
            .map_err(|e| failure(READER_PROGRAM, e))?;
        let mask = Tensor::f32(vec![1, ENCODER_FRAMES], mask.to_vec())
            .map_err(|e| failure(READER_PROGRAM, e))?;
        let started = Instant::now();
        let out = self
            .reader
            .predict(&[("l11", &hidden), ("mask", &mask)])
            .map_err(|e| failure(READER_PROGRAM, e))?;
        self.read_time.add(started);
        Ok(ReaderOutput {
            vectors: output(
                &out,
                READER_PROGRAM,
                "vec",
                &[1, ENCODER_FRAMES, EMBEDDING_DIM],
            )?,
            blank: output(&out, READER_PROGRAM, "blank", &[1, ENCODER_FRAMES, 1])?,
        })
    }
}

/// The compiled block scorer, the reference's shortlist stage.
pub struct CoreMlScorer {
    model: CoreMlModel,
    pub time: StageTime,
}

impl std::fmt::Debug for CoreMlScorer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CoreMlScorer")
            .field("model", &self.model.path())
            .finish_non_exhaustive()
    }
}

impl CoreMlScorer {
    pub fn open(dir: &Path, units: ComputeUnits) -> Result<Self, String> {
        require_macos_15()?;
        let model = load(dir, SCORER_PROGRAM, units)?;
        expect(
            &model,
            Side::Input,
            "vec",
            DType::F16,
            &[1, ENCODER_FRAMES, EMBEDDING_DIM],
        )?;
        expect(
            &model,
            Side::Output,
            "bidx",
            DType::I32,
            &[1, ENCODER_FRAMES, TOP_BLOCKS],
        )?;
        Ok(Self {
            model,
            time: StageTime::default(),
        })
    }
}

impl BlockShortlister for CoreMlScorer {
    fn shortlist(&mut self, vectors: &[f32], frames: usize) -> audio::Result<BlockShortlist> {
        if frames != ENCODER_FRAMES {
            return Err(SpeechError::Input {
                why: format!("the compiled scorer takes {ENCODER_FRAMES} frames, not {frames}"),
            });
        }
        // The reader's `vec` is f16 already, so this narrowing is exact for
        // pipeline input. It refuses non-finite values instead of passing
        // them to the program.
        let input = Tensor::f16_from_f32("vec", vec![1, ENCODER_FRAMES, EMBEDDING_DIM], vectors)
            .map_err(|e| failure(SCORER_PROGRAM, e))?;
        let started = Instant::now();
        let out = self
            .model
            .predict(&[("vec", &input)])
            .map_err(|e| failure(SCORER_PROGRAM, e))?;
        self.time.add(started);
        let bidx = out
            .get("bidx")
            .filter(|t| t.shape() == [1, ENCODER_FRAMES, TOP_BLOCKS])
            .and_then(Tensor::as_i32)
            .ok_or_else(|| tensor_error(SCORER_PROGRAM, "bidx is missing or misshaped"))?;
        let block_count = SCORER_COLUMNS / BLOCK_SIZE;
        let blocks = bidx
            .iter()
            .map(|&id| {
                u32::try_from(id)
                    .ok()
                    .filter(|&block| (block as usize) < block_count)
                    .ok_or_else(|| {
                        tensor_error(
                            SCORER_PROGRAM,
                            format!("block id {id} is outside 0..{block_count}"),
                        )
                    })
            })
            .collect::<audio::Result<Vec<u32>>>()?;
        BlockShortlist::new(ENCODER_FRAMES, TOP_BLOCKS, blocks)
    }
}

/// Opens the full transcriber: shipped frontend tables, word list and
/// table, calibration, and the three Core ML programs.
pub fn open(dir: &Path, opts: &Eg2CtcOptions) -> Result<Eg2CtcCoreMl, String> {
    let backend = CoreMlAcoustic::open(dir, opts)?;
    let scorer = CoreMlScorer::open(dir, opts.scorer_units)?;
    open_with(dir, backend, scorer)
}

/// Assembles a transcriber over any backend and shortlister (for example
/// the Core ML backend with the portable CPU scorer).
pub fn open_with<B: AcousticBackend, S: BlockShortlister>(
    dir: &Path,
    backend: B,
    shortlister: S,
) -> Result<Transcriber<B, S>, String> {
    let err = |e: SpeechError| e.to_string();
    Transcriber::new(
        LogMelFrontend::load(dir).map_err(err)?,
        backend,
        shortlister,
        WordList::load(dir).map_err(err)?,
        WordTable::load(dir).map_err(err)?,
        Calibration::load(dir).map_err(err)?,
    )
    .map_err(err)
}

fn require_macos_15() -> Result<(), String> {
    let (major, _, _) = turbospark_coreml::os_version();
    if major < 15 {
        return Err(format!(
            "the eg2demo programs are iOS 18 ML programs and need macOS 15 or newer (this is {major})"
        ));
    }
    Ok(())
}

fn load(dir: &Path, program: &str, units: ComputeUnits) -> Result<CoreMlModel, String> {
    CoreMlModel::load(&dir.join(program), &LoadOptions::new(units))
        .map_err(|e| format!("{program}: {e}"))
}

#[derive(Clone, Copy)]
enum Side {
    Input,
    Output,
}

/// Refuses a program whose feature differs from the pinned description.
fn expect(
    model: &CoreMlModel,
    side: Side,
    name: &str,
    dtype: DType,
    shape: &[usize],
) -> Result<(), String> {
    let (specs, what) = match side {
        Side::Input => (model.inputs(), "input"),
        Side::Output => (model.outputs(), "output"),
    };
    let program = model.path().display();
    let spec = specs
        .iter()
        .find(|spec| spec.name == name)
        .ok_or_else(|| format!("{program} has no {what} `{name}`"))?;
    match &spec.kind {
        FeatureKind::MultiArray {
            dtype: Some(d),
            shape: Some(s),
        } if *d == dtype && s.as_slice() == shape => Ok(()),
        other => Err(format!(
            "{program}: {what} `{name}` is {other:?}, expected {dtype} {shape:?}"
        )),
    }
}

fn output(
    out: &std::collections::HashMap<String, Tensor>,
    program: &str,
    name: &str,
    shape: &[usize],
) -> audio::Result<Vec<f32>> {
    out.get(name)
        .filter(|t| t.shape() == shape)
        .and_then(Tensor::to_f32_vec)
        .ok_or_else(|| tensor_error(program, format!("{name} is missing or misshaped")))
}

fn failure(program: &str, error: turbospark_coreml::CoreMlError) -> SpeechError {
    tensor_error(program, error.to_string())
}

fn tensor_error(program: &str, why: impl Into<String>) -> SpeechError {
    SpeechError::Tensor {
        name: program.into(),
        why: why.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn asset_dir() -> Option<std::path::PathBuf> {
        std::env::var_os("TURBOSPARK_EG2_ASSET_DIR").map(Into::into)
    }

    #[test]
    fn options_read_per_program_units_and_refuse_unknown_ones() {
        // The only test that touches these variables, so no other test
        // races with it.
        std::env::set_var(SCORER_UNITS_ENV, "cpu");
        let o = Eg2CtcOptions::from_env();
        std::env::set_var(SCORER_UNITS_ENV, "tpu");
        let bad = Eg2CtcOptions::from_env();
        std::env::remove_var(SCORER_UNITS_ENV);
        let o = o.unwrap();
        assert_eq!(o.scorer_units, ComputeUnits::CpuOnly);
        assert_eq!(o.encoder_units, ComputeUnits::All);
        assert_eq!(o.reader_units, ComputeUnits::All);
        assert!(bad.unwrap_err().contains(SCORER_UNITS_ENV));
    }

    /// The checked-in forced-aligner clip ("The quick brown fox jumps over
    /// the lazy dog."), 16 kHz mono f32.
    fn reference_clip() -> Vec<f32> {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../audio/testdata/qwen3_forced_aligner_reference.wav");
        let wave = audio::wav::read_wav_f32_bytes(&std::fs::read(path).unwrap()).unwrap();
        assert_eq!((wave.sample_rate, wave.channels), (16_000, 1));
        wave.samples
    }

    #[test]
    #[ignore = "requires the pinned eg2demo v5 assets in TURBOSPARK_EG2_ASSET_DIR and macOS 15"]
    fn transcribes_the_reference_clip_with_either_scorer() {
        let dir = asset_dir().expect("set TURBOSPARK_EG2_ASSET_DIR");
        let samples = reference_clip();
        let expected = "the quick brown fox jumps over the lazy dog";

        let mut model = open(&dir, &Eg2CtcOptions::default()).unwrap();
        let transcript = model.transcribe(&samples).unwrap();
        assert_eq!(transcript.text, expected);
        assert_eq!(transcript.windows, 1);
        // 44715 samples plus the 8000-sample tail: 329 valid mel frames,
        // 83 encoder frames.
        assert_eq!(transcript.frame_path.len(), 83);
        assert_eq!(
            (
                model.backend.encode_time.calls,
                model.shortlister.time.calls
            ),
            (1, 1)
        );
        // Word spikes are ordered and inside the clip.
        let starts: Vec<f64> = transcript.words.iter().map(|w| w.start_seconds).collect();
        assert!(
            starts.windows(2).all(|pair| pair[0] < pair[1]),
            "{starts:?}"
        );
        assert!(transcript.words.last().unwrap().end_seconds <= 83.0 * 0.04 + 1e-9);

        // The portable CPU scorer shortlists the same words on this clip.
        let backend = CoreMlAcoustic::open(&dir, &Eg2CtcOptions::default()).unwrap();
        let cpu = audio::stt::embeddinggemma2::CpuBlockScorer::load(&dir).unwrap();
        let mut portable = open_with(&dir, backend, cpu).unwrap();
        let cpu_transcript = portable.transcribe(&samples).unwrap();
        assert_eq!(cpu_transcript.frame_path, transcript.frame_path);
    }

    #[test]
    fn missing_programs_are_refused_with_the_program_name() {
        let dir = std::env::temp_dir().join(format!("ts-eg2-ctc-missing-{}", std::process::id()));
        // Before macOS 15 the OS gate refuses first, naming the requirement.
        let (encoder, scorer) = if turbospark_coreml::os_version().0 < 15 {
            ("macOS 15", "macOS 15")
        } else {
            (ENCODER_PROGRAM, SCORER_PROGRAM)
        };
        let error = CoreMlAcoustic::open(&dir, &Eg2CtcOptions::default()).unwrap_err();
        assert!(error.contains(encoder), "{error}");
        let error = CoreMlScorer::open(&dir, ComputeUnits::CpuOnly).unwrap_err();
        assert!(error.contains(scorer), "{error}");
    }
}
