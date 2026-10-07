//! Mega-ASR: routed robust transcription over the shared Qwen3-ASR backbone.
//!
//! Reference: `mlx_audio/stt/models/mega_asr/` (mega_asr.py, router.py,
//! config.py, convert_lora.py, lora.py) at mlx-audio 0.5.7, commit
//! `e1b19b9054bf163f5d812221a54fcc346f1890e9`. The audio tower, text decoder,
//! log-mel frontend, tokenizer loader, and prompt construction are reused from
//! `crate::stt::qwen3_asr` rather than duplicated; this module adds the
//! audio-quality router, the LoRA adapter table, and the family profile
//! rules. See README.md for the pinned profiles and verification evidence.

use std::path::Path;

use serde_json::Value;
use turbospark_model_io::safetensors::SafetensorsFile;
use turbospark_tokenizer::Tokenizer;

use crate::quant::QuantScheme;
use crate::stt::qwen3_asr::decoder::{chat_stop_ids, greedy_generate, is_chat_stop, Decoder};
use crate::stt::qwen3_asr::encoder::AudioEncoder;
use crate::stt::qwen3_asr::frontend::compute_features;
use crate::stt::qwen3_asr::{load_tokenizer, prompt_token_ids, transcript_from_tokens};
use crate::{Result, SpeechError};

pub mod config;
pub mod lora;
pub mod router;

pub use config::{MegaConfig, ProfileKind, RouterSettings};
pub use lora::{LoraAdapter, LoraModule};
pub use router::{AudioQualityRouter, RouteDecision, RouterStages};

/// Immutable Hugging Face checkpoint reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MegaAsrProfile {
    pub name: &'static str,
    pub repository: &'static str,
    pub revision: &'static str,
}

/// The pinned always-on-robust profile: the Qwen3-ASR-1.7B backbone with the
/// robustness LoRA folded in before 8-bit quantization (see
/// `merge_metadata.json` in the distribution). It carries no router and no
/// LoRA file, so every clip runs the robust path.
pub const MEGA_ASR_8BIT: MegaAsrProfile = MegaAsrProfile {
    name: "Mega-ASR 8-bit always-on robust",
    repository: "mlx-community/Mega-ASR-8bit",
    revision: "b9c3c7020f94944205df7f7b5d5d1ce96678d74f",
};

/// One sparse stage tensor witness: the row-major `[shape]` values at the
/// recorded row and column indices, matching the fixture generator's
/// `selected_rows`.
#[derive(Debug, Clone, PartialEq)]
pub struct StageWitness {
    pub name: String,
    pub shape: Vec<usize>,
    pub rows: Vec<usize>,
    pub columns: Vec<usize>,
    pub values: Vec<Vec<f32>>,
}

/// The first decoded-step logits summary: the argmax plus the top-16 token
/// ids and values (the fixture records the reference top-8, which must
/// survive inside this port's top-16 under bf16-vs-f32 drift).
#[derive(Debug, Clone, PartialEq)]
pub struct FirstLogits {
    pub argmax: u32,
    pub argmax_value: f32,
    pub top16: Vec<(u32, f32)>,
}

/// Stages captured by a checkpoint-gated transcription.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TranscribeStages {
    pub input_features: Option<StageWitness>,
    pub audio_embeddings: Option<StageWitness>,
    pub prompt_token_ids: Vec<i32>,
    pub prefill_last_hidden: Option<Vec<f32>>,
    pub first_logits: Option<FirstLogits>,
    pub generated_token_ids: Vec<u32>,
}

/// Loaded Mega-ASR model: Qwen3-ASR backbone plus the optional router and
/// LoRA adapter of the dynamic profile.
pub struct MegaAsr {
    config: MegaConfig,
    encoder: AudioEncoder,
    decoder: Decoder,
    tokenizer: Tokenizer,
    router: Option<AudioQualityRouter>,
    lora: Option<LoraAdapter>,
}

impl MegaAsr {
    /// Loads one local checkpoint directory. The profile is detected from
    /// `config.json` plus `merge_metadata.json`; see
    /// [`MegaConfig::detect_profile`] for the rules. Tests never download.
    pub fn load(model_dir: &Path) -> Result<Self> {
        let config_json: Value = read_json(&model_dir.join("config.json"))?;
        let profile = MegaConfig::detect_profile(model_dir, &config_json)?;
        let config = MegaConfig::from_json(&config_json, profile)?;

        let weights = SafetensorsFile::open(&model_dir.join("model.safetensors"))?;
        let scheme = QuantScheme {
            bits: config.backbone.quant_bits,
            group_size: config.backbone.quant_group_size,
        };
        let encoder = AudioEncoder::load(&weights, &config.backbone.audio)?;
        let decoder = Decoder::load(&weights, &config.backbone.text, scheme)?;
        let tokenizer = load_tokenizer(model_dir)?;
        for (token, expected) in [
            ("<|audio_pad|>", config.backbone.audio_token_id),
            ("<|audio_start|>", config.backbone.audio_start_token_id),
            ("<|audio_end|>", config.backbone.audio_end_token_id),
        ] {
            if tokenizer.token_to_id(token).map(|id| id as i32) != Some(expected) {
                return Err(SpeechError::BadConfig {
                    field: "tokenizer.json".into(),
                    why: format!("{token} id does not match config.json"),
                });
            }
        }

        // The dynamic profile refuses to run without trained router weights.
        // The reference would silently route with a freshly initialized
        // router, whose decision is meaningless noise; the pre-merged robust
        // profile has no router by construction.
        let router = match profile {
            ProfileKind::Dynamic => {
                let settings = config.router.as_ref().expect("dynamic profile router");
                let path = model_dir.join(&config.router_weights);
                if !path.is_file() {
                    return Err(SpeechError::BadConfig {
                        field: "router_weights".into(),
                        why: format!(
                            "the dynamic profile needs trained router weights at {} (the \
                             reference would silently route with an untrained router)",
                            path.display()
                        ),
                    });
                }
                Some(AudioQualityRouter::from_safetensors(
                    &SafetensorsFile::open(&path)?,
                    Some(settings),
                )?)
            }
            ProfileKind::PreMergedRobust => None,
        };
        let lora = match profile {
            ProfileKind::Dynamic => {
                let path = model_dir.join(&config.lora_weights);
                if path.is_file() {
                    Some(LoraAdapter::load_factors(&path)?)
                } else {
                    None
                }
            }
            ProfileKind::PreMergedRobust => None,
        };

        Ok(Self {
            config,
            encoder,
            decoder,
            tokenizer,
            router,
            lora,
        })
    }

    pub fn profile(&self) -> MegaAsrProfile {
        MEGA_ASR_8BIT
    }

    pub fn config(&self) -> &MegaConfig {
        &self.config
    }

    /// Routes one clip through the audio-quality router. Refuses when the
    /// distribution carries no router (the pre-merged robust profile always
    /// runs the robust path and cannot route).
    pub fn route(&self, samples: &[f32]) -> Result<RouteDecision> {
        let router = self
            .router
            .as_ref()
            .ok_or_else(|| SpeechError::Unsupported {
                why: "this Mega-ASR distribution carries no router (LoRA pre-merged); it always \
                  runs the robust path"
                    .into(),
            })?;
        Ok(router.route(samples)?.1)
    }

    /// Transcribes mono PCM already sampled at 16 kHz with greedy decoding.
    pub fn transcribe(&self, samples: &[f32]) -> Result<String> {
        self.transcribe_with_options(samples, None, 512)
    }

    /// Transcribes one mono 16 kHz clip, optionally prompting a supported
    /// output language. `max_tokens` bounds greedy decoding. The prompt and
    /// chat template match the reference exactly (see
    /// `qwen3_asr::prompt_token_ids`).
    pub fn transcribe_with_options(
        &self,
        samples: &[f32],
        language: Option<&str>,
        max_tokens: usize,
    ) -> Result<String> {
        self.transcribe_impl(samples, language, max_tokens, None)
            .map(|(text, _)| text)
    }

    /// Transcription with optional stage capture for checkpoint-gated parity
    /// tests. The greedy loop mirrors `qwen3_asr::Qwen3Asr`, whose public
    /// constructor cannot be reused because it insists on the plain
    /// qwen3_asr model type and has no router or LoRA surface.
    pub(crate) fn transcribe_impl(
        &self,
        samples: &[f32],
        language: Option<&str>,
        max_tokens: usize,
        mut stages: Option<&mut TranscribeStages>,
    ) -> Result<(String, Vec<u32>)> {
        if max_tokens == 0 {
            return Err(SpeechError::Input {
                why: "Mega-ASR max_tokens must be positive".into(),
            });
        }
        if samples.is_empty() || samples.iter().any(|s| !s.is_finite()) {
            return Err(SpeechError::Input {
                why: "Mega-ASR audio must be non-empty and finite".into(),
            });
        }
        let language = language
            .map(|requested| {
                self.config
                    .backbone
                    .supported_languages
                    .iter()
                    .find(|known| known.eq_ignore_ascii_case(requested))
                    .map(String::as_str)
                    .ok_or_else(|| SpeechError::Input {
                        why: format!("unsupported Mega-ASR language: {requested}"),
                    })
            })
            .transpose()?;

        // The router decides which decode path runs. With no LoRA table the
        // reference's apply_deltas call is a no-op, so both decisions decode
        // identically and the port skips the pretense. A degraded-route decode
        // with real deltas needs a dense base to merge them into; the shared
        // decoder keeps its weights private, so that combination is refused
        // until the dense-base runtime path is wired (README remaining gates).
        if let Some(router) = &self.router {
            let decision = router.route(samples)?.1;
            if decision.use_lora && self.lora.is_some() {
                return Err(SpeechError::Unsupported {
                    why: "runtime LoRA application over the reused Qwen3-ASR backbone is not \
                          wired; the pinned always-on-robust profile carries the deltas \
                          pre-merged and never reaches this path"
                        .into(),
                });
            }
        }

        let features = compute_features(samples)?;
        if let Some(stages) = stages.as_deref_mut() {
            stages.input_features = Some(witness(
                "input_features",
                &[
                    features.values.len() / features.frames.max(1),
                    features.frames,
                ],
                &features.values,
                &[],
                &[],
            ));
        }
        let audio_embeddings = self.encoder.forward(&features)?;
        if audio_embeddings.len() % self.decoder.hidden_size != 0 {
            return Err(SpeechError::Tensor {
                name: "audio_tower.output".into(),
                why: "encoder output is not a whole number of decoder embeddings".into(),
            });
        }
        let audio_rows = audio_embeddings.len() / self.decoder.hidden_size;
        if audio_rows == 0 {
            return Err(SpeechError::Input {
                why: "Mega-ASR audio encoder produced no features".into(),
            });
        }
        if let Some(stages) = stages.as_deref_mut() {
            stages.audio_embeddings = Some(witness(
                "audio_embeddings",
                &[audio_rows, self.decoder.hidden_size],
                &audio_embeddings,
                &[],
                &[],
            ));
        }

        let (token_ids, audio_positions) =
            prompt_token_ids(&self.config.backbone, &self.tokenizer, audio_rows, language)?;
        if let Some(stages) = stages.as_deref_mut() {
            stages.prompt_token_ids = token_ids.clone();
        }
        let mut embeddings = self.decoder.embed(&token_ids)?;
        for (audio_row, &position) in audio_positions.iter().enumerate() {
            let target = position * self.decoder.hidden_size;
            let source = audio_row * self.decoder.hidden_size;
            embeddings[target..target + self.decoder.hidden_size]
                .copy_from_slice(&audio_embeddings[source..source + self.decoder.hidden_size]);
        }

        let stop_ids = chat_stop_ids(&self.tokenizer);
        let rows = embeddings.len() / self.decoder.hidden_size;
        let (last_hidden, cache) = self.decoder.prefill(&embeddings, rows);
        if let Some(stages) = stages.as_deref_mut() {
            stages.prefill_last_hidden = Some(last_hidden.clone());
        }
        let logits = self.decoder.logits(&last_hidden);
        // Recorded only when the loop is entered, as before.
        let first_logits = (max_tokens > 0).then(|| top_logits(&logits));
        let generated = greedy_generate(
            &self.decoder,
            &logits,
            cache,
            max_tokens,
            |next| is_chat_stop(&stop_ids, next),
            "Mega-ASR",
            |_, _| {},
        )?;
        if let Some(stages) = stages {
            stages.first_logits = first_logits;
            stages.generated_token_ids = generated.clone();
        }

        let text = transcript_from_tokens(&self.tokenizer, &generated, language)?;
        Ok((text, generated))
    }
}

/// Reads and parses a JSON file next to the checkpoint.
fn read_json(path: &Path) -> Result<Value> {
    let bytes = std::fs::read(path).map_err(|error| SpeechError::BadConfig {
        field: path.display().to_string(),
        why: format!("cannot read: {error}"),
    })?;
    serde_json::from_slice(&bytes).map_err(|error| SpeechError::BadConfig {
        field: path.display().to_string(),
        why: error.to_string(),
    })
}

fn top_logits(logits: &[f32]) -> FirstLogits {
    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_by(|a, b| logits[*b].total_cmp(&logits[*a]));
    FirstLogits {
        argmax: order[0] as u32,
        argmax_value: logits[order[0]],
        top16: order
            .iter()
            .take(16)
            .map(|&index| (index as u32, logits[index]))
            .collect(),
    }
}

/// Builds the fixture's sparse row/column witness: rows and columns default to
/// the reference's selected set (first two, middle, last) when not given.
fn witness(
    name: &str,
    shape: &[usize],
    values: &[f32],
    rows: &[usize],
    columns: &[usize],
) -> StageWitness {
    let (rows_total, columns_total) = match shape.len() {
        1 => (1usize, shape[0]),
        2 => (shape[0], shape[1]),
        _ => (values.len(), 1),
    };
    let mut row_set: Vec<usize> = if rows.is_empty() {
        vec![
            0,
            1.min(rows_total.saturating_sub(1)),
            rows_total / 2,
            rows_total - 1,
        ]
    } else {
        rows.to_vec()
    };
    row_set.sort_unstable();
    row_set.dedup();
    let mut column_set: Vec<usize> = if columns.is_empty() {
        vec![
            0,
            1.min(columns_total.saturating_sub(1)),
            columns_total / 2,
            columns_total - 1,
        ]
    } else {
        columns.to_vec()
    };
    column_set.sort_unstable();
    column_set.dedup();
    let mut out = Vec::with_capacity(row_set.len());
    for &row in &row_set {
        let base = if shape.len() == 1 {
            0
        } else {
            row * columns_total
        };
        out.push(
            column_set
                .iter()
                .map(|&column| values[base + column])
                .collect(),
        );
    }
    StageWitness {
        name: name.to_owned(),
        shape: shape.to_vec(),
        rows: row_set,
        columns: column_set,
        values: out,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lora::test_support::write_safetensors;

    const TOLERANCE_STAGE: f32 = 2.0e-4;
    const TOLERANCE_LOGITS: f32 = 2.0e-4;
    const TOLERANCE_PROB: f32 = 5.0e-5;

    fn fixture() -> serde_json::Value {
        serde_json::from_str(include_str!("../../../testdata/mega_asr_reference.json")).unwrap()
    }

    fn smoke_samples() -> Vec<f32> {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("testdata/qwen3_forced_aligner_reference.wav");
        let audio = crate::wav::read_wav_f32(&path).expect("reference WAV loads");
        assert_eq!(audio.sample_rate, 16_000);
        assert_eq!(audio.channels, 1);
        audio.samples
    }

    /// The fixture's documented deterministic degraded waveform; every step
    /// is a single round-to-nearest f32 operation, so Python and Rust produce
    /// bit-identical samples.
    fn degraded(samples: &[f32]) -> Vec<f32> {
        samples
            .iter()
            .enumerate()
            .map(|(index, &value)| value + 0.2f32 * ((index % 97) as f32 / 97.0f32 - 0.5f32))
            .collect()
    }

    fn decode_base64(raw: &str) -> Vec<f32> {
        fn value(byte: u8) -> Option<u32> {
            match byte {
                b'A'..=b'Z' => Some((byte - b'A') as u32),
                b'a'..=b'z' => Some((byte - b'a') as u32 + 26),
                b'0'..=b'9' => Some((byte - b'0') as u32 + 52),
                b'+' => Some(62),
                b'/' => Some(63),
                _ => None,
            }
        }
        let mut bytes = Vec::with_capacity(raw.len() / 4 * 3);
        let mut buffer = 0u32;
        let mut bits = 0u32;
        for byte in raw.bytes() {
            if let Some(entry) = value(byte) {
                buffer = (buffer << 6) | entry;
                bits += 6;
                if bits >= 8 {
                    bits -= 8;
                    bytes.push(((buffer >> bits) & 0xFF) as u8);
                }
            }
        }
        bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect()
    }

    fn compare_witness(actual: &StageWitness, expected: &serde_json::Value, tolerance: f32) -> f32 {
        let shape: Vec<usize> = serde_json::from_value(expected["shape"].clone()).unwrap();
        let rows: Vec<usize> = serde_json::from_value(expected["rows"].clone()).unwrap();
        let columns: Vec<usize> = serde_json::from_value(expected["columns"].clone()).unwrap();
        let values: Vec<Vec<f32>> = serde_json::from_value(expected["values"].clone()).unwrap();
        assert_eq!(actual.shape, shape, "stage {} shape", actual.name);
        assert_eq!(actual.rows, rows, "stage {} rows", actual.name);
        assert_eq!(actual.columns, columns, "stage {} columns", actual.name);
        let mut max_abs = 0.0f32;
        for (actual_row, expected_row) in actual.values.iter().zip(&values) {
            for (a, e) in actual_row.iter().zip(expected_row) {
                max_abs = max_abs.max((a - e).abs());
            }
        }
        assert!(
            max_abs <= tolerance,
            "stage {} maximum absolute difference {max_abs} exceeds {tolerance}",
            actual.name
        );
        max_abs
    }

    #[test]
    fn router_forward_matches_the_pinned_reference() {
        let fixture = fixture();
        let router_section = &fixture["router"];
        let names: Vec<String> =
            serde_json::from_value(router_section["weight_names"].clone()).unwrap();
        let packed = decode_base64(router_section["weights_base64"].as_str().unwrap());
        let mut offset = 0usize;
        let mut tensors = std::collections::BTreeMap::new();
        for name in &names {
            let shape: Vec<usize> =
                serde_json::from_value(router_section["weight_shapes"][name].clone()).unwrap();
            let count: usize = shape.iter().product();
            tensors.insert(
                name.clone(),
                (shape.clone(), packed[offset..offset + count].to_vec()),
            );
            offset += count;
        }
        assert_eq!(offset, packed.len(), "every packed f32 must be consumed");
        let router = AudioQualityRouter::from_tensors(&tensors, None).unwrap();

        let clean = smoke_samples();
        let samples_for = |name: &str| match name {
            "smoke_clean" => clean.clone(),
            "smoke_degraded" => degraded(&clean),
            other => panic!("unexpected router input {other}"),
        };
        for input in router_section["inputs"].as_array().unwrap() {
            let name = input["name"].as_str().unwrap();
            let (stages, decision) = router.route(&samples_for(name)).unwrap();
            let mut stage_max = 0.0f32;
            let mut logit_max = 0.0f32;
            let logmel_max = compare_witness(
                &witness(
                    "logmel",
                    &[
                        stages.logmel_frames,
                        stages.logmel.len() / stages.logmel_frames,
                    ],
                    &stages.logmel,
                    &[],
                    &[],
                ),
                &input["logmel"],
                TOLERANCE_STAGE,
            );
            stage_max = stage_max.max(logmel_max);
            let frontend_max = compare_witness(
                &witness(
                    "frontend_hidden",
                    &[
                        stages.frontend_hidden.len() / router.d_model(),
                        router.d_model(),
                    ],
                    &stages.frontend_hidden,
                    &[],
                    &[],
                ),
                &input["frontend_hidden"],
                TOLERANCE_STAGE,
            );
            stage_max = stage_max.max(frontend_max);
            let transformer_max = compare_witness(
                &witness(
                    "transformer_hidden",
                    &[
                        stages.transformer_hidden.len() / router.d_model(),
                        router.d_model(),
                    ],
                    &stages.transformer_hidden,
                    &[],
                    &[],
                ),
                &input["transformer_hidden"],
                TOLERANCE_STAGE,
            );
            stage_max = stage_max.max(transformer_max);
            let pooled: Vec<f32> = serde_json::from_value(input["pooled"].clone()).unwrap();
            let pooled_max = stages
                .pooled
                .iter()
                .zip(&pooled)
                .map(|(a, e)| (a - e).abs())
                .fold(0.0f32, f32::max);
            stage_max = stage_max.max(pooled_max);
            assert!(
                pooled_max <= TOLERANCE_STAGE,
                "pooled difference {pooled_max}"
            );
            let logits: [f32; 2] = serde_json::from_value::<Vec<f32>>(input["logits"].clone())
                .unwrap()[..2]
                .try_into()
                .unwrap();
            for (a, e) in stages.logits.iter().zip(logits) {
                logit_max = logit_max.max((a - e).abs());
                assert!((a - e).abs() <= TOLERANCE_LOGITS, "logits {a} vs {e}");
            }
            let expected_prob: f32 =
                serde_json::from_value(input["degraded_prob"].clone()).unwrap();
            assert!(
                (decision.degraded_prob - expected_prob).abs() <= TOLERANCE_PROB,
                "degraded_prob {} vs {expected_prob}",
                decision.degraded_prob
            );
            let expected_lora: bool = serde_json::from_value(input["use_lora"].clone()).unwrap();
            assert_eq!(decision.use_lora, expected_lora, "routing for {name}");
            eprintln!(
                "mega_asr router {name}: max stage abs diff {stage_max:.3e}, logits diff \
                 {logit_max:.3e}, prob diff {prob_max:.3e}",
                stage_max = stage_max,
                logit_max = logit_max,
                prob_max = (decision.degraded_prob - expected_prob).abs(),
            );
        }
    }

    #[test]
    fn lora_deltas_match_the_pinned_reference() {
        let fixture = fixture();
        let lora_section = &fixture["lora"];
        let modules: Vec<(String, usize, usize, usize)> =
            serde_json::from_value(lora_section["modules"].clone()).unwrap();
        assert_eq!(
            modules.len(),
            lora_section["module_count"].as_u64().unwrap() as usize
        );
        assert_eq!(lora_section["scaling"].as_f64().unwrap(), 1.0);
        for (name, output, input, rank) in &modules {
            // The factors file keeps the decoder modules under both
            // "model.layers.N.*" and the alias "layers.N.*"; the reference
            // resolves both to the same linear and sums the deltas.
            assert!(
                name.starts_with("model.layers.")
                    || name.starts_with("layers.")
                    || name.starts_with("audio_tower."),
                "unexpected module name {name}"
            );
            assert!(*rank > 0 && *output > 0 && *input > 0);
            assert!(output >= rank && input >= rank);
        }

        for selected in lora_section["selected"].as_array().unwrap() {
            let name = selected["name"].as_str().unwrap();
            let a = decode_base64(selected["a_base64"].as_str().unwrap());
            let b = decode_base64(selected["b_base64"].as_str().unwrap());
            let rank = selected["rank"].as_u64().unwrap() as usize;
            let (_, output, input, _) = modules
                .iter()
                .find(|(module, _, _, _)| module == name)
                .expect("selected module exists in the inventory");
            let module = LoraModule {
                rank,
                output: *output,
                input: *input,
                scaling: 1.0,
                a,
                b,
            };
            let delta = module.materialize_delta();
            let rows: Vec<usize> = serde_json::from_value(selected["delta_rows"].clone()).unwrap();
            let columns: Vec<usize> =
                serde_json::from_value(selected["delta_columns"].clone()).unwrap();
            let values: Vec<Vec<f32>> =
                serde_json::from_value(selected["delta_values"].clone()).unwrap();
            let mut max_abs = 0.0f32;
            for (&row, expected_row) in rows.iter().zip(&values) {
                for (&column, expected) in columns.iter().zip(expected_row) {
                    max_abs = max_abs.max((delta[row * input + column] - expected).abs());
                }
            }
            assert!(
                max_abs <= TOLERANCE_STAGE,
                "delta spot difference {max_abs} for {name}"
            );
            let max_expected: f64 =
                serde_json::from_value(selected["delta_max_abs"].clone()).unwrap();
            let sum_expected: f64 =
                serde_json::from_value(selected["delta_sum_abs"].clone()).unwrap();
            let max_actual = delta.iter().fold(0.0f64, |m, v| m.max(v.abs() as f64));
            let sum_actual: f64 = delta.iter().map(|v| v.abs() as f64).sum();
            assert!(
                (max_actual - max_expected).abs() <= 1e-3,
                "delta max {max_actual} vs {max_expected} for {name}"
            );
            eprintln!(
                "mega_asr lora {name}: spot max {max_abs:.3e}, delta max abs \
                 {max_actual:.6}, delta sum abs {sum_actual:.6}"
            );
            let relative_sum = (sum_actual - sum_expected).abs() / sum_expected.abs().max(1.0);
            assert!(
                relative_sum <= 2.0e-3,
                "delta sum relative difference {relative_sum} for {name}"
            );
        }
    }

    #[test]
    fn lora_factor_file_loader_round_trips_the_pinned_factors() {
        let fixture = fixture();
        let lora_section = &fixture["lora"];
        let modules: Vec<(String, usize, usize, usize)> =
            serde_json::from_value(lora_section["modules"].clone()).unwrap();
        let dir = std::env::temp_dir().join(format!("mega-lora-roundtrip-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("lora.safetensors");
        let mut tensors = std::collections::BTreeMap::new();
        for selected in lora_section["selected"].as_array().unwrap() {
            let name = selected["name"].as_str().unwrap();
            let rank = selected["rank"].as_u64().unwrap() as usize;
            let a = decode_base64(selected["a_base64"].as_str().unwrap());
            let b = decode_base64(selected["b_base64"].as_str().unwrap());
            let (_, output, input, _) = modules
                .iter()
                .find(|(module, _, _, _)| module == name)
                .unwrap();
            tensors.insert(format!("{name}.lora_A"), (vec![rank, *input], a));
            tensors.insert(format!("{name}.lora_B"), (vec![*output, rank], b));
        }
        write_safetensors(&path, &tensors);
        let adapter = LoraAdapter::load_factors(&path).unwrap();
        assert_eq!(adapter.module_count(), 3);
        for selected in lora_section["selected"].as_array().unwrap() {
            let name = selected["name"].as_str().unwrap();
            let rank = selected["rank"].as_u64().unwrap() as usize;
            let (_, output, input, _) = modules
                .iter()
                .find(|(module, _, _, _)| module == name)
                .unwrap();
            let module = adapter.module(name).unwrap();
            assert_eq!(module.rank, rank);
            assert_eq!(module.output, *output);
            assert_eq!(module.input, *input);
            assert_eq!(module.scaling, 1.0);
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn pre_merged_config_from_fixture_matches_the_pinned_profile() {
        let fixture = fixture();
        let config =
            MegaConfig::from_json(&fixture["config_json"], ProfileKind::PreMergedRobust).unwrap();
        assert_eq!(config.profile, ProfileKind::PreMergedRobust);
        assert_eq!(config.backbone.text.num_hidden_layers, 28);
        assert_eq!(config.backbone.audio.encoder_layers, 24);
        assert_eq!(config.backbone.quant_bits, 8);
    }

    #[test]
    #[ignore = "requires the pinned Mega-ASR checkpoint in TURBOSPARK_MEGA_ASR_MODEL_DIR"]
    fn pinned_checkpoint_matches_reference_stages_and_transcript() {
        let model_dir = std::env::var_os("TURBOSPARK_MEGA_ASR_MODEL_DIR")
            .expect("set TURBOSPARK_MEGA_ASR_MODEL_DIR to the pinned local snapshot");
        let model = MegaAsr::load(Path::new(&model_dir)).expect("checkpoint loads");
        let samples = smoke_samples();
        let mut stages = TranscribeStages::default();
        let (text, _) = model
            .transcribe_impl(&samples, None, 96, Some(&mut stages))
            .expect("checkpoint transcribes");
        let fixture = fixture();
        assert_eq!(
            text,
            fixture["backbone"]["transcript"].as_str().unwrap(),
            "the pinned transcript must match the mlx-audio reference exactly"
        );

        let expected_tokens: Vec<u32> =
            serde_json::from_value(fixture["backbone"]["generated_token_ids"].clone()).unwrap();
        assert_eq!(stages.generated_token_ids, expected_tokens);
        let expected_prompt: Vec<i32> =
            serde_json::from_value(fixture["backbone"]["prompt_token_ids"].clone()).unwrap();
        assert_eq!(stages.prompt_token_ids, expected_prompt);

        let features_max = compare_witness(
            stages.input_features.as_ref().unwrap(),
            &fixture["backbone"]["input_features"],
            2.0e-4,
        );
        let embeddings_max = compare_witness(
            stages.audio_embeddings.as_ref().unwrap(),
            &fixture["backbone"]["audio_embeddings"],
            2.0e-4,
        );
        let expected_hidden: Vec<f32> =
            serde_json::from_value(fixture["backbone"]["prefill_last_hidden"].clone()).unwrap();
        let hidden = stages.prefill_last_hidden.as_ref().unwrap();
        assert_eq!(hidden.len(), expected_hidden.len());
        let hidden_max = hidden
            .iter()
            .zip(&expected_hidden)
            .map(|(a, e)| (a - e).abs())
            .fold(0.0f32, f32::max);
        let expected_max = expected_hidden.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        // The reference decoder computes in bf16 end to end (quantized
        // weights carry bf16 scales, activations round to bf16 between
        // ops); this port computes the dequantized f32 pipeline. Drift of a
        // few percent of the row scale through 28 layers is the expected
        // bf16-vs-f32 gap, while the greedy decode stays token-identical.
        assert!(
            hidden_max <= 6.0e-2 * expected_max,
            "prefill hidden difference {hidden_max} exceeds 6% of the row scale \
             {expected_max}"
        );
        let hidden_relative = hidden_max / expected_max;
        let first = stages.first_logits.as_ref().unwrap();
        let expected_argmax = fixture["backbone"]["first_logits"]["argmax"]
            .as_u64()
            .unwrap() as u32;
        assert_eq!(first.argmax, expected_argmax, "first greedy token id");
        let expected_top8: Vec<(u32, f64)> =
            serde_json::from_value(fixture["backbone"]["first_logits"]["top8"].clone()).unwrap();
        let expected_top8_max = expected_top8
            .iter()
            .map(|(_, v)| v.abs())
            .fold(0.0f64, f64::max);
        // The token ids of the reference top-8 must all survive the drift
        // (checked against this port's top-16, since bf16 rounding can
        // reorder near ties), and every top-8 logit value must stay within
        // 5% of the reference.
        let mut top_logit_max = 0.0f64;
        for (expected_id, expected_value) in &expected_top8 {
            let actual_value = first
                .top16
                .iter()
                .find(|(id, _)| id == expected_id)
                .unwrap_or_else(|| {
                    panic!("reference top-8 token {expected_id} left this port's top-16")
                })
                .1 as f64;
            top_logit_max = top_logit_max.max((actual_value - expected_value).abs());
            assert!(
                (actual_value - expected_value).abs() <= 5.0e-2 * expected_value.abs(),
                "top-8 logit for {expected_id}: {actual_value} vs {expected_value}"
            );
        }
        let top8_relative = top_logit_max / expected_top8_max;
        eprintln!(
            "mega_asr checkpoint parity: input_features max {features_max:.3e}, \
             audio_embeddings max {embeddings_max:.3e}, prefill hidden max {hidden_max:.3e} \
             (relative to row max {hidden_relative:.3e}), top-8 logit max {top_logit_max:.3e} \
             (relative {top8_relative:.3e})"
        );
    }
}
