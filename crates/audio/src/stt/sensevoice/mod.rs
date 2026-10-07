//! SenseVoice Small CTC, ported from mlx-audio 0.5.7.
//!
//! Source files: `mlx_audio/stt/models/sensevoice/{config.py,sensevoice.py}`
//! at commit `e1b19b9054bf163f5d812221a54fcc346f1890e9`.
//! The pinned profile is the F32 MLX conversion, which stores the FSMN
//! kernels in the raw `[channels, 1, kernel]` layout consumed here.

pub(super) mod frontend;
pub(super) mod sanm;

use std::fs;
use std::path::Path;

use serde_json::Value;
use turbospark_model_io::safetensors::SafetensorsFile;

use crate::nn::{argmax, bad_config, load_tensor, LayerNorm, Linear};
use crate::stt::wav2vec::ctc::CtcCollapse;
use crate::{Result, SpeechError};
use sanm::{add_sinusoidal_positions, FsmnLayout, SanmConfig, SanmEncoderLayer};

/// Immutable Hugging Face checkpoint profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SenseVoiceProfile {
    pub name: &'static str,
    pub repository: &'static str,
    pub revision: &'static str,
}

pub const SENSEVOICE_SMALL: SenseVoiceProfile = SenseVoiceProfile {
    name: "SenseVoice Small",
    repository: "mlx-community/SenseVoiceSmall",
    revision: "8ddd966bd96243cff196422f81f0c5d955814792",
};

const VOCAB_SIZE: usize = 25_055;
const INPUT_SIZE: usize = 560;
const OUTPUT_SIZE: usize = 512;
const ATTENTION_HEADS: usize = 4;
const LINEAR_UNITS: usize = 2048;
const ENCODER_BLOCKS: usize = 50;
const TP_BLOCKS: usize = 20;
const FSMN_KERNEL: usize = 11;
const MEL_BINS: usize = 80;
const LFR_M: usize = 7;
const LFR_N: usize = 6;
/// Every SenseVoice LayerNorm uses this epsilon; the shared `nn::LayerNorm`
/// stores it instead of hardcoding it in `apply`.
const LAYER_NORM_EPS: f32 = 1e-5;

/// Output from a single offline SenseVoice inference pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SenseVoiceOutput {
    pub text: String,
    pub language: String,
    pub emotion: String,
    pub event: String,
    pub token_ids: Vec<usize>,
}

#[derive(Debug, Clone, Copy)]
struct Config {
    frontend: frontend::FrontendConfig,
    vocabulary: usize,
    input_size: usize,
    output_size: usize,
    attention_heads: usize,
    linear_units: usize,
    encoder_blocks: usize,
    tp_blocks: usize,
    fsmn_kernel: usize,
    sanm_shift: usize,
}

fn validate_sanm(_: usize, _: usize, kernel: usize, left_padding: usize) -> Result<()> {
    if left_padding > kernel - 1 {
        return Err(bad_config("sanm_shift", "padding exceeds the FSMN kernel"));
    }
    Ok(())
}

impl Config {
    fn sanm(&self) -> SanmConfig {
        SanmConfig {
            output_size: self.output_size,
            linear_units: self.linear_units,
            attention_heads: self.attention_heads,
            fsmn_kernel: self.fsmn_kernel,
            sanm_shift: self.sanm_shift,
            layer_norm_eps: LAYER_NORM_EPS,
            fsmn_layout: FsmnLayout::ChannelsOneKernel,
            validate: validate_sanm,
        }
    }

    fn load(path: &Path) -> Result<Self> {
        let text = fs::read_to_string(path).map_err(|error| SpeechError::Input {
            why: format!("cannot read {}: {error}", path.display()),
        })?;
        let root: Value = serde_json::from_str(&text)
            .map_err(|error| bad_config("config.json", error.to_string()))?;
        if root.get("model_type").and_then(Value::as_str) != Some("sensevoice") {
            return Err(bad_config("model_type", "expected sensevoice"));
        }

        let encoder = root
            .get("encoder_conf")
            .and_then(Value::as_object)
            .ok_or_else(|| bad_config("encoder_conf", "must be an object"))?;
        let frontend_json = root
            .get("frontend_conf")
            .and_then(Value::as_object)
            .ok_or_else(|| bad_config("frontend_conf", "must be an object"))?;
        let get_positive = |object: &serde_json::Map<String, Value>, field: &str| {
            object
                .get(field)
                .and_then(Value::as_u64)
                .and_then(|value| usize::try_from(value).ok())
                .filter(|&value| value > 0)
                .ok_or_else(|| bad_config(field, "must be a positive integer"))
        };
        let vocabulary = root
            .get("vocab_size")
            .and_then(Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .filter(|&value| value > 0)
            .ok_or_else(|| bad_config("vocab_size", "must be a positive integer"))?;
        let input_size = root
            .get("input_size")
            .and_then(Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .filter(|&value| value > 0)
            .ok_or_else(|| bad_config("input_size", "must be a positive integer"))?;
        let output_size = get_positive(encoder, "output_size")?;
        let attention_heads = get_positive(encoder, "attention_heads")?;
        let linear_units = get_positive(encoder, "linear_units")?;
        let encoder_blocks = get_positive(encoder, "num_blocks")?;
        let tp_blocks = get_positive(encoder, "tp_blocks")?;
        let fsmn_kernel = get_positive(encoder, "kernel_size")?;
        let sanm_shift = encoder
            .get("sanm_shift")
            .and_then(Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .ok_or_else(|| bad_config("sanm_shift", "must be a non-negative integer"))?;
        if encoder.get("normalize_before").and_then(Value::as_bool) != Some(true) {
            return Err(bad_config(
                "normalize_before",
                "only the verified pre-normalized SenseVoice layout is supported",
            ));
        }
        let window = frontend_json
            .get("window")
            .and_then(Value::as_str)
            .ok_or_else(|| bad_config("window", "must be a string"))?;
        if window != "hamming" {
            return Err(bad_config(
                "window",
                "only the verified hamming window is supported",
            ));
        }
        let frontend = frontend::FrontendConfig {
            sample_rate: get_positive(frontend_json, "fs")?,
            num_mels: get_positive(frontend_json, "n_mels")?,
            frame_length_ms: get_positive(frontend_json, "frame_length")?,
            frame_shift_ms: get_positive(frontend_json, "frame_shift")?,
            lfr_m: get_positive(frontend_json, "lfr_m")?,
            lfr_n: get_positive(frontend_json, "lfr_n")?,
        };
        if vocabulary != VOCAB_SIZE
            || input_size != INPUT_SIZE
            || output_size != OUTPUT_SIZE
            || attention_heads != ATTENTION_HEADS
            || linear_units != LINEAR_UNITS
            || encoder_blocks != ENCODER_BLOCKS
            || tp_blocks != TP_BLOCKS
            || fsmn_kernel != FSMN_KERNEL
            || sanm_shift != 0
            || frontend.sample_rate != 16_000
            || frontend.num_mels != MEL_BINS
            || frontend.lfr_m != LFR_M
            || frontend.lfr_n != LFR_N
            || input_size != frontend.num_mels * frontend.lfr_m
            || output_size % attention_heads != 0
        {
            return Err(bad_config(
                "config.json",
                "checkpoint dimensions differ from the pinned SenseVoice Small profile",
            ));
        }
        Ok(Self {
            frontend,
            vocabulary,
            input_size,
            output_size,
            attention_heads,
            linear_units,
            encoder_blocks,
            tp_blocks,
            fsmn_kernel,
            sanm_shift,
        })
    }
}

/// Loaded SenseVoice Small model. The checkpoint is loaded from a local HF
/// snapshot, so inference never downloads weights implicitly.
pub struct SenseVoiceSmall {
    config: Config,
    embedding: Vec<f32>,
    first_block: SanmEncoderLayer,
    encoder: Vec<SanmEncoderLayer>,
    after_norm: LayerNorm,
    tp_encoder: Vec<SanmEncoderLayer>,
    tp_norm: LayerNorm,
    ctc: Linear,
    cmvn_means: Vec<f32>,
    cmvn_istd: Vec<f32>,
    pieces: Vec<SentencePiece>,
}

impl SenseVoiceSmall {
    /// Load SenseVoice Small from a completed local model snapshot.
    pub fn load(model_dir: &Path) -> Result<Self> {
        let config = Config::load(&model_dir.join("config.json"))?;
        let weights = SafetensorsFile::open(&model_dir.join("model.safetensors"))?;
        let embedding = load_tensor(&weights, "embed.weight", &[16, config.input_size])?;
        let sanm = config.sanm();
        let first_block =
            SanmEncoderLayer::load(&weights, "encoder.encoders0.0", config.input_size, &sanm)?;
        let mut encoder = Vec::with_capacity(config.encoder_blocks - 1);
        for index in 0..config.encoder_blocks - 1 {
            encoder.push(SanmEncoderLayer::load(
                &weights,
                &format!("encoder.encoders.{index}"),
                config.output_size,
                &sanm,
            )?);
        }
        let after_norm = LayerNorm::load(
            &weights,
            "encoder.after_norm",
            config.output_size,
            LAYER_NORM_EPS,
        )?;
        let mut tp_encoder = Vec::with_capacity(config.tp_blocks);
        for index in 0..config.tp_blocks {
            tp_encoder.push(SanmEncoderLayer::load(
                &weights,
                &format!("encoder.tp_encoders.{index}"),
                config.output_size,
                &sanm,
            )?);
        }
        let tp_norm = LayerNorm::load(
            &weights,
            "encoder.tp_norm",
            config.output_size,
            LAYER_NORM_EPS,
        )?;
        let ctc = Linear::load(
            &weights,
            "ctc.ctc_lo",
            config.output_size,
            config.vocabulary,
            true,
        )?;
        let (cmvn_means, cmvn_istd) = parse_mvn(&model_dir.join("am.mvn"), config.input_size)?;
        let pieces = parse_sentencepiece(&model_dir.join("chn_jpn_yue_eng_ko_spectok.bpe.model"))?;
        if pieces.len() != config.vocabulary {
            return Err(bad_config(
                "SentencePiece vocabulary",
                format!(
                    "expected {} pieces, got {}",
                    config.vocabulary,
                    pieces.len()
                ),
            ));
        }
        Ok(Self {
            config,
            embedding,
            first_block,
            encoder,
            after_norm,
            tp_encoder,
            tp_norm,
            ctc,
            cmvn_means,
            cmvn_istd,
            pieces,
        })
    }

    pub fn profile(&self) -> SenseVoiceProfile {
        SENSEVOICE_SMALL
    }

    /// Transcribe a mono 16 kHz waveform with greedy CTC decoding.
    pub fn transcribe(&self, samples: &[f32]) -> Result<String> {
        Ok(self.transcribe_with_options(samples, "auto", false)?.text)
    }

    /// Transcribe with optional language selection and inverse text
    /// normalization, returning SenseVoice's language, emotion, and event
    /// tags alongside the text.
    pub fn transcribe_with_options(
        &self,
        samples: &[f32],
        language: &str,
        use_itn: bool,
    ) -> Result<SenseVoiceOutput> {
        self.transcribe_impl(samples, language, use_itn, None)
    }

    fn transcribe_impl(
        &self,
        samples: &[f32],
        language: &str,
        use_itn: bool,
        mut trace: Option<&mut Vec<StageSnapshot>>,
    ) -> Result<SenseVoiceOutput> {
        let language_id = match language {
            "auto" => 0,
            "zh" => 3,
            "en" => 4,
            "yue" => 7,
            "ja" => 11,
            "ko" => 12,
            "nospeech" => 13,
            _ => {
                return Err(SpeechError::Input {
                    why: format!("unsupported SenseVoice language {language:?}"),
                })
            }
        };
        let (features, feature_frames) = frontend::extract_features(
            samples,
            self.config.frontend,
            &self.cmvn_means,
            &self.cmvn_istd,
        )?;
        let total_frames = feature_frames + 4;
        let input_width = self.config.input_size;
        let mut hidden = vec![0.0f32; total_frames * input_width];
        let query_ids = [language_id, 1usize, 2usize, if use_itn { 14 } else { 15 }];
        for (frame, &query_id) in query_ids.iter().enumerate() {
            let source = &self.embedding[query_id * input_width..(query_id + 1) * input_width];
            hidden[frame * input_width..(frame + 1) * input_width].copy_from_slice(source);
        }
        hidden[4 * input_width..].copy_from_slice(&features);
        add_sinusoidal_positions(
            &mut hidden,
            total_frames,
            input_width,
            self.config.output_size,
        );
        capture_stage(
            &mut trace,
            "positioned_encoder_input",
            &hidden,
            total_frames,
            input_width,
        );

        hidden = self.first_block.forward(&hidden, total_frames);
        capture_stage(
            &mut trace,
            "first_block",
            &hidden,
            total_frames,
            self.config.output_size,
        );
        for layer in &self.encoder {
            hidden = layer.forward(&hidden, total_frames);
        }
        self.after_norm.apply(&mut hidden, total_frames);
        capture_stage(
            &mut trace,
            "encoder_stack",
            &hidden,
            total_frames,
            self.config.output_size,
        );
        for layer in &self.tp_encoder {
            hidden = layer.forward(&hidden, total_frames);
        }
        self.tp_norm.apply(&mut hidden, total_frames);
        capture_stage(
            &mut trace,
            "encoder_output",
            &hidden,
            total_frames,
            self.config.output_size,
        );

        let logits = self.ctc.forward(&hidden, total_frames);
        let language = rich_tag(argmax(&logits[..self.config.vocabulary]), "language");
        let emotion = rich_tag(
            argmax(&logits[self.config.vocabulary..2 * self.config.vocabulary]),
            "emotion",
        );
        let event = rich_tag(
            argmax(&logits[2 * self.config.vocabulary..3 * self.config.vocabulary]),
            "event",
        );
        let token_ids = greedy_ctc(
            &logits[4 * self.config.vocabulary..],
            feature_frames,
            self.config.vocabulary,
        );
        let text = decode_sentencepiece(&self.pieces, &token_ids)?;
        Ok(SenseVoiceOutput {
            text,
            language,
            emotion,
            event,
            token_ids,
        })
    }
}

struct StageSnapshot {
    name: &'static str,
    shape: [usize; 2],
    rows: Vec<usize>,
    columns: Vec<usize>,
    values: Vec<Vec<f32>>,
}

fn capture_stage(
    trace: &mut Option<&mut Vec<StageSnapshot>>,
    name: &'static str,
    data: &[f32],
    rows: usize,
    columns: usize,
) {
    let Some(trace) = trace.as_deref_mut() else {
        return;
    };
    let mut selected_rows = vec![0, 1.min(rows - 1), rows / 2, rows - 1];
    selected_rows.sort_unstable();
    selected_rows.dedup();
    let mut selected_columns = vec![0, 1.min(columns - 1), columns / 2, columns - 1];
    selected_columns.sort_unstable();
    selected_columns.dedup();
    let values = selected_rows
        .iter()
        .map(|&row| {
            selected_columns
                .iter()
                .map(|&column| data[row * columns + column])
                .collect()
        })
        .collect();
    trace.push(StageSnapshot {
        name,
        shape: [rows, columns],
        rows: selected_rows,
        columns: selected_columns,
        values,
    });
}

struct SentencePiece {
    text: String,
    kind: u64,
}

fn parse_mvn(path: &Path, width: usize) -> Result<(Vec<f32>, Vec<f32>)> {
    let text = fs::read_to_string(path).map_err(|error| SpeechError::Input {
        why: format!("cannot read {}: {error}", path.display()),
    })?;
    let means = parse_mvn_vector(&text, "<AddShift>")?;
    let istd = parse_mvn_vector(&text, "<Rescale>")?;
    if means.len() != width || istd.len() != width {
        return Err(bad_config(
            "am.mvn",
            format!("expected two vectors of length {width}"),
        ));
    }
    Ok((means, istd))
}

fn parse_mvn_vector(text: &str, marker: &str) -> Result<Vec<f32>> {
    let section = text
        .find(marker)
        .and_then(|start| text[start..].find('[').map(|open| start + open + 1))
        .ok_or_else(|| bad_config("am.mvn", format!("missing {marker} values")))?;
    let end = text[section..]
        .find(']')
        .map(|offset| section + offset)
        .ok_or_else(|| bad_config("am.mvn", format!("unterminated {marker} vector")))?;
    text[section..end]
        .split_whitespace()
        .map(|item| {
            item.parse::<f32>()
                .map_err(|error| bad_config("am.mvn", error.to_string()))
        })
        .collect()
}

fn parse_sentencepiece(path: &Path) -> Result<Vec<SentencePiece>> {
    let bytes = fs::read(path).map_err(|error| SpeechError::Input {
        why: format!("cannot read {}: {error}", path.display()),
    })?;
    let mut pieces = Vec::new();
    let mut offset = 0usize;
    while offset < bytes.len() {
        let tag = read_varint(&bytes, &mut offset)
            .ok_or_else(|| bad_config("SentencePiece model", "invalid protobuf tag"))?;
        let field = tag >> 3;
        let wire = (tag & 7) as u8;
        if field == 1 && wire == 2 {
            let message = read_bytes(&bytes, &mut offset)
                .ok_or_else(|| bad_config("SentencePiece model", "invalid piece record"))?;
            pieces.push(parse_piece(message)?);
        } else {
            skip_field(&bytes, &mut offset, wire)
                .ok_or_else(|| bad_config("SentencePiece model", "invalid protobuf field"))?;
        }
    }
    Ok(pieces)
}

fn parse_piece(message: &[u8]) -> Result<SentencePiece> {
    let mut offset = 0usize;
    let mut text = None;
    let mut kind = 1u64;
    while offset < message.len() {
        let tag = read_varint(message, &mut offset)
            .ok_or_else(|| bad_config("SentencePiece model", "invalid piece field tag"))?;
        let field = tag >> 3;
        let wire = (tag & 7) as u8;
        match (field, wire) {
            (1, 2) => {
                let value = read_bytes(message, &mut offset)
                    .ok_or_else(|| bad_config("SentencePiece model", "invalid piece text"))?;
                text = Some(
                    String::from_utf8(value.to_vec())
                        .map_err(|error| bad_config("SentencePiece model", error.to_string()))?,
                );
            }
            (3, 0) => {
                kind = read_varint(message, &mut offset)
                    .ok_or_else(|| bad_config("SentencePiece model", "invalid piece type"))?;
            }
            _ => skip_field(message, &mut offset, wire)
                .ok_or_else(|| bad_config("SentencePiece model", "invalid piece field"))?,
        }
    }
    Ok(SentencePiece {
        text: text.ok_or_else(|| bad_config("SentencePiece model", "piece has no text"))?,
        kind,
    })
}

fn read_varint(bytes: &[u8], offset: &mut usize) -> Option<u64> {
    let mut value = 0u64;
    for shift in (0..70).step_by(7) {
        let byte = *bytes.get(*offset)?;
        *offset += 1;
        if shift == 63 && byte > 1 {
            return None;
        }
        value |= u64::from(byte & 0x7f).checked_shl(shift)?;
        if byte & 0x80 == 0 {
            return Some(value);
        }
    }
    None
}

fn read_bytes<'a>(bytes: &'a [u8], offset: &mut usize) -> Option<&'a [u8]> {
    let length = usize::try_from(read_varint(bytes, offset)?).ok()?;
    let end = (*offset).checked_add(length)?;
    let result = bytes.get(*offset..end)?;
    *offset = end;
    Some(result)
}

fn skip_field(bytes: &[u8], offset: &mut usize, wire: u8) -> Option<()> {
    match wire {
        0 => {
            read_varint(bytes, offset)?;
        }
        1 => *offset = (*offset).checked_add(8)?,
        2 => {
            read_bytes(bytes, offset)?;
        }
        5 => *offset = (*offset).checked_add(4)?,
        _ => return None,
    }
    (*offset <= bytes.len()).then_some(())
}

fn decode_sentencepiece(pieces: &[SentencePiece], ids: &[usize]) -> Result<String> {
    let mut bytes = Vec::new();
    for &id in ids {
        let piece = pieces.get(id).ok_or_else(|| SpeechError::Tensor {
            name: "SentencePiece id".into(),
            why: format!("token id {id} is outside the vocabulary"),
        })?;
        match piece.kind {
            3 | 5 => {}
            2 => bytes.extend_from_slice("\u{fffd}".as_bytes()),
            6 => {
                if let Some(byte) = parse_byte_piece(&piece.text) {
                    bytes.push(byte);
                } else {
                    bytes.extend_from_slice(piece.text.as_bytes());
                }
            }
            _ => bytes.extend_from_slice(piece.text.as_bytes()),
        }
    }
    let text = String::from_utf8_lossy(&bytes).replace('\u{2581}', " ");
    Ok(text.trim().to_owned())
}

fn parse_byte_piece(piece: &str) -> Option<u8> {
    let value = piece.strip_prefix("<0x")?.strip_suffix('>')?;
    if value.len() != 2 {
        return None;
    }
    u8::from_str_radix(value, 16).ok()
}

fn greedy_ctc(logits: &[f32], frames: usize, vocab: usize) -> Vec<usize> {
    let mut collapse = CtcCollapse::new(0);
    for frame in 0..frames {
        collapse.push(argmax(&logits[frame * vocab..(frame + 1) * vocab]));
    }
    collapse.finish()
}

fn rich_tag(token: usize, kind: &str) -> String {
    match kind {
        "language" => match token {
            24_884 => "zh".into(),
            24_885 => "en".into(),
            24_888 => "yue".into(),
            24_892 => "ja".into(),
            24_896 => "ko".into(),
            24_992 => "nospeech".into(),
            _ => "unknown".into(),
        },
        "emotion" => match token {
            25_001 => "happy".into(),
            25_002 => "sad".into(),
            25_003 => "angry".into(),
            25_004 => "neutral".into(),
            25_005 => "fearful".into(),
            25_006 => "disgusted".into(),
            25_007 => "surprised".into(),
            25_008 => "other".into(),
            25_009 => "unk".into(),
            _ => format!("token_{token}"),
        },
        "event" => match token {
            24_993 => "Speech".into(),
            24_995 => "BGM".into(),
            24_997 => "Laughter".into(),
            24_999 => "Applause".into(),
            _ => format!("token_{token}"),
        },
        _ => unreachable!("known SenseVoice tag kind"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str = include_str!("../../../testdata/sensevoice_reference.json");

    fn fixture() -> Value {
        serde_json::from_str(FIXTURE).expect("valid SenseVoice reference fixture")
    }

    #[test]
    fn kaldi_fbank_lfr_and_cmvn_match_pinned_mlx_reference() {
        let fixture = fixture();
        let audio_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("testdata/qwen3_forced_aligner_reference.wav");
        let waveform = crate::wav::read_wav_f32(&audio_path).expect("reference WAV loads");
        assert_eq!(waveform.sample_rate, 16_000);
        assert_eq!(waveform.channels, 1);

        let means: Vec<f32> = serde_json::from_value(fixture["cmvn_means"].clone()).unwrap();
        let istd: Vec<f32> = serde_json::from_value(fixture["cmvn_istd"].clone()).unwrap();
        let expected: Vec<Vec<f32>> =
            serde_json::from_value(fixture["frontend_features"].clone()).unwrap();
        let front_config = frontend::FrontendConfig {
            sample_rate: 16_000,
            num_mels: MEL_BINS,
            frame_length_ms: 25,
            frame_shift_ms: 10,
            lfr_m: LFR_M,
            lfr_n: LFR_N,
        };
        let raw_actual = frontend::compute_fbank(&waveform.samples, front_config).unwrap();
        let raw_expected: Vec<Vec<f32>> = serde_json::from_value(fixture["fbank"].clone()).unwrap();
        let raw_expected = raw_expected.into_iter().flatten().collect::<Vec<_>>();
        let (raw_max_index, raw_max) = raw_actual
            .iter()
            .zip(&raw_expected)
            .enumerate()
            .map(|(index, (actual, expected))| (index, (actual - expected).abs()))
            .max_by(|left, right| left.1.total_cmp(&right.1))
            .unwrap();
        assert!(
            raw_max < 0.01,
            "maximum raw FBANK difference {raw_max} at row {}, mel {}",
            raw_max_index / MEL_BINS,
            raw_max_index % MEL_BINS
        );
        let (actual, frames) =
            frontend::extract_features(&waveform.samples, front_config, &means, &istd)
                .expect("features compute");
        assert_eq!(frames, expected.len());
        assert_eq!(actual.len(), expected.iter().map(Vec::len).sum::<usize>());
        let mut max_abs = 0.0f32;
        let expected = expected.into_iter().flatten().collect::<Vec<_>>();
        let mut max_index = 0;
        for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
            let difference = (actual - expected).abs();
            if difference > max_abs {
                max_abs = difference;
                max_index = index;
            }
        }
        assert!(
            max_abs < 0.01,
            "maximum frontend difference {max_abs} at row {}, col {}; values {} vs {}; raw fbank max difference {raw_max} at row {}, mel {}, values {} vs {}; row actual {:?}, row expected {:?}",
            max_index / INPUT_SIZE,
            max_index % INPUT_SIZE,
            actual[max_index],
            expected[max_index],
            raw_max_index / MEL_BINS,
            raw_max_index % MEL_BINS,
            raw_actual[raw_max_index],
            raw_expected[raw_max_index],
            &actual[(max_index / INPUT_SIZE) * INPUT_SIZE + 150
                ..(max_index / INPUT_SIZE) * INPUT_SIZE + 170],
            &expected[(max_index / INPUT_SIZE) * INPUT_SIZE + 150
                ..(max_index / INPUT_SIZE) * INPUT_SIZE + 170]
        );
    }

    #[test]
    fn ctc_collapse_keeps_repetitions_separated_by_blank() {
        let logits = [
            0.0, 1.0, // token 1
            0.0, 1.0, // repeated token 1 collapses
            1.0, 0.0, // blank
            0.0, 1.0, // token 1 after blank is kept
        ];
        assert_eq!(greedy_ctc(&logits, 4, 2), vec![1, 1]);
    }

    #[test]
    fn sentencepiece_decode_skips_control_and_unpacks_byte_pieces() {
        let pieces = vec![
            SentencePiece {
                text: "<unk>".into(),
                kind: 2,
            },
            SentencePiece {
                text: "<s>".into(),
                kind: 3,
            },
            SentencePiece {
                text: "\u{2581}hello".into(),
                kind: 1,
            },
            SentencePiece {
                text: "\u{2581}world".into(),
                kind: 1,
            },
            SentencePiece {
                text: "<0x21>".into(),
                kind: 6,
            },
        ];
        assert_eq!(
            decode_sentencepiece(&pieces, &[1, 2, 3, 4]).unwrap(),
            "hello world!"
        );
    }

    #[test]
    #[ignore = "requires the pinned SenseVoiceSmall snapshot in TURBOSPARK_SENSEVOICE_MODEL_DIR"]
    fn pinned_checkpoint_matches_mlx_transcript_and_tags() {
        let model_dir = std::env::var_os("TURBOSPARK_SENSEVOICE_MODEL_DIR")
            .expect("set TURBOSPARK_SENSEVOICE_MODEL_DIR to the pinned local snapshot");
        let model = SenseVoiceSmall::load(Path::new(&model_dir)).expect("checkpoint loads");
        let audio_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("testdata/qwen3_forced_aligner_reference.wav");
        let waveform = crate::wav::read_wav_f32(&audio_path).expect("reference WAV loads");
        let mut stages = Vec::new();
        let output = model
            .transcribe_impl(&waveform.samples, "auto", false, Some(&mut stages))
            .expect("checkpoint transcribes");
        let fixture = fixture();
        assert_eq!(output.text, fixture["transcript"].as_str().unwrap());
        assert_eq!(output.language, fixture["language"].as_str().unwrap());
        assert_eq!(output.emotion, fixture["emotion"].as_str().unwrap());
        assert_eq!(output.event, fixture["event"].as_str().unwrap());
        let expected_tokens: Vec<usize> =
            serde_json::from_value(fixture["token_ids"].clone()).unwrap();
        assert_eq!(output.token_ids, expected_tokens);
        assert_eq!(stages.len(), 4);
        for stage in stages {
            let expected = &fixture[stage.name];
            let expected_shape: Vec<usize> =
                serde_json::from_value(expected["shape"].clone()).unwrap();
            let expected_rows: Vec<usize> =
                serde_json::from_value(expected["rows"].clone()).unwrap();
            let expected_columns: Vec<usize> =
                serde_json::from_value(expected["columns"].clone()).unwrap();
            let expected_values: Vec<Vec<f32>> =
                serde_json::from_value(expected["values"].clone()).unwrap();
            assert_eq!(stage.shape.as_slice(), expected_shape);
            assert_eq!(stage.rows, expected_rows);
            assert_eq!(stage.columns, expected_columns);
            let max_abs = stage
                .values
                .iter()
                .flatten()
                .zip(expected_values.iter().flatten())
                .map(|(actual, expected)| (actual - expected).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_abs < 0.01,
                "stage {} maximum absolute difference was {max_abs}",
                stage.name
            );
        }
    }
}
