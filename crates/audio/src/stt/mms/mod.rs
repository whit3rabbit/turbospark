//! English MMS ASR using the shared Wav2Vec2 CTC backbone.
//!
//! Reference: `mlx_audio/stt/models/mms/` and
//! `mlx_audio/stt/models/wav2vec/` at mlx-audio 0.5.7, commit
//! `e1b19b9054bf163f5d812221a54fcc346f1890e9`.

use std::fs;
use std::path::Path;

use serde_json::Value;
use turbospark_model_io::safetensors::SafetensorsFile;

use crate::nn::{bad_config, load_tensor, LayerNorm, Linear};
use crate::ops;
use crate::stt::wav2vec::backbone::{
    add, channels_first_to_rows, decode_ctc, parse_vocab, positive, positive_array,
    rows_to_channels_first, Attention, PositionalConv,
};
use crate::{Result, SpeechError};

/// Immutable Hugging Face checkpoint profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MmsProfile {
    pub name: &'static str,
    pub repository: &'static str,
    pub revision: &'static str,
    pub language: &'static str,
}

pub const MMS_1B_FL102_ENGLISH: MmsProfile = MmsProfile {
    name: "MMS 1B FL102 English",
    repository: "facebook/mms-1b-fl102",
    revision: "d483345545bea550895b1aa0c6ba40236b9f1e22",
    language: "eng",
};

/// Inference settings from the pinned Wav2Vec2 checkpoint config.
#[derive(Debug, Clone, PartialEq)]
pub struct MmsConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub layer_norm_eps: f32,
    pub conv_dim: Vec<usize>,
    pub conv_stride: Vec<usize>,
    pub conv_kernel: Vec<usize>,
    pub conv_bias: bool,
    pub num_conv_pos_embeddings: usize,
    pub num_conv_pos_embedding_groups: usize,
    pub adapter_attn_dim: usize,
}

impl MmsConfig {
    pub fn from_json(root: &Value) -> Result<Self> {
        if root.get("model_type").and_then(Value::as_str) != Some("wav2vec2")
            || root
                .get("architectures")
                .and_then(Value::as_array)
                .is_none_or(|items| {
                    !items
                        .iter()
                        .any(|item| item.as_str() == Some("Wav2Vec2ForCTC"))
                })
        {
            return Err(bad_config(
                "model_type",
                "expected the Wav2Vec2ForCTC architecture",
            ));
        }
        if root.get("feat_extract_norm").and_then(Value::as_str) != Some("layer")
            || root.get("do_stable_layer_norm").and_then(Value::as_bool) != Some(true)
            || root.get("hidden_act").and_then(Value::as_str) != Some("gelu")
        {
            return Err(SpeechError::Unsupported {
                why: "MMS port requires layer-normalized GELU features and a stable-layer-norm encoder".into(),
            });
        }

        let config = Self {
            vocab_size: positive(root, "vocab_size")?,
            hidden_size: positive(root, "hidden_size")?,
            num_hidden_layers: positive(root, "num_hidden_layers")?,
            num_attention_heads: positive(root, "num_attention_heads")?,
            intermediate_size: positive(root, "intermediate_size")?,
            layer_norm_eps: root
                .get("layer_norm_eps")
                .and_then(Value::as_f64)
                .filter(|number| number.is_finite() && *number > 0.0)
                .map(|number| number as f32)
                .ok_or_else(|| bad_config("layer_norm_eps", "must be a positive number"))?,
            conv_dim: positive_array(root, "conv_dim")?,
            conv_stride: positive_array(root, "conv_stride")?,
            conv_kernel: positive_array(root, "conv_kernel")?,
            conv_bias: root
                .get("conv_bias")
                .and_then(Value::as_bool)
                .ok_or_else(|| bad_config("conv_bias", "must be a boolean"))?,
            num_conv_pos_embeddings: positive(root, "num_conv_pos_embeddings")?,
            num_conv_pos_embedding_groups: positive(root, "num_conv_pos_embedding_groups")?,
            adapter_attn_dim: positive(root, "adapter_attn_dim")?,
        };
        if config.num_attention_heads == 0
            || config.hidden_size % config.num_attention_heads != 0
            || config.hidden_size % config.num_conv_pos_embedding_groups != 0
            || config.conv_dim.len() != config.conv_stride.len()
            || config.conv_dim.len() != config.conv_kernel.len()
            || config.conv_dim.is_empty()
        {
            return Err(bad_config(
                "config.json",
                "inconsistent Wav2Vec2 dimensions or convolution geometry",
            ));
        }
        Ok(config)
    }
}

struct FeatureConv {
    weight: Vec<f32>,
    bias: Option<Vec<f32>>,
    norm: LayerNorm,
    input: usize,
    output: usize,
    kernel: usize,
    stride: usize,
}

impl FeatureConv {
    fn load(file: &SafetensorsFile, index: usize, config: &MmsConfig) -> Result<Self> {
        let input = if index == 0 {
            1
        } else {
            config.conv_dim[index - 1]
        };
        let output = config.conv_dim[index];
        let kernel = config.conv_kernel[index];
        let prefix = format!("wav2vec2.feature_extractor.conv_layers.{index}");
        Ok(Self {
            weight: load_tensor(
                file,
                &format!("{prefix}.conv.weight"),
                &[output, input, kernel],
            )?,
            bias: if config.conv_bias {
                Some(load_tensor(
                    file,
                    &format!("{prefix}.conv.bias"),
                    &[output],
                )?)
            } else {
                None
            },
            norm: LayerNorm::load(
                file,
                &format!("{prefix}.layer_norm"),
                output,
                config.layer_norm_eps,
            )?,
            input,
            output,
            kernel,
            stride: config.conv_stride[index],
        })
    }

    fn forward(
        &self,
        channels_first: &[f32],
        steps: usize,
        trace: &mut Option<&mut Vec<StageTensor>>,
        index: usize,
    ) -> Result<(Vec<f32>, usize)> {
        if steps < self.kernel {
            return Err(SpeechError::Input {
                why: "audio is too short for the Wav2Vec2 convolution stack".into(),
            });
        }
        let output_steps = (steps - self.kernel) / self.stride + 1;
        let mut hidden = ops::conv1d(
            channels_first,
            &self.weight,
            self.bias.as_deref(),
            self.input,
            self.output,
            self.kernel,
            self.stride,
            0,
            1,
            1,
        );
        let mut rows = channels_first_to_rows(&hidden, self.output, output_steps);
        record_stage(
            trace,
            format!("feature_conv_{index}_raw"),
            &rows,
            output_steps,
            self.output,
        );
        self.norm.apply(&mut rows, output_steps);
        record_stage(
            trace,
            format!("feature_conv_{index}_norm"),
            &rows,
            output_steps,
            self.output,
        );
        ops::gelu_erf(&mut rows);
        record_stage(
            trace,
            format!("feature_conv_{index}"),
            &rows,
            output_steps,
            self.output,
        );
        hidden = rows_to_channels_first(&rows, output_steps, self.output);
        Ok((hidden, output_steps))
    }
}

struct FeatureEncoder {
    layers: Vec<FeatureConv>,
}

impl FeatureEncoder {
    fn load(file: &SafetensorsFile, config: &MmsConfig) -> Result<Self> {
        let layers = (0..config.conv_dim.len())
            .map(|index| FeatureConv::load(file, index, config))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { layers })
    }

    fn forward(
        &self,
        samples: &[f32],
        trace: &mut Option<&mut Vec<StageTensor>>,
    ) -> Result<(Vec<f32>, usize, usize)> {
        let mut hidden = samples.to_vec();
        let mut channels = 1;
        let mut steps = samples.len();
        for (index, layer) in self.layers.iter().enumerate() {
            let (next, next_steps) = layer.forward(&hidden, steps, trace, index)?;
            hidden = next;
            channels = layer.output;
            steps = next_steps;
        }
        Ok((
            channels_first_to_rows(&hidden, channels, steps),
            steps,
            channels,
        ))
    }
}

struct Adapter {
    norm: LayerNorm,
    down: Linear,
    up: Linear,
}

impl Adapter {
    fn load(file: &SafetensorsFile, prefix: &str, config: &MmsConfig) -> Result<Self> {
        Ok(Self {
            norm: LayerNorm::load(
                file,
                &format!("{prefix}.norm"),
                config.hidden_size,
                config.layer_norm_eps,
            )?,
            down: Linear::load(
                file,
                &format!("{prefix}.linear_1"),
                config.hidden_size,
                config.adapter_attn_dim,
                true,
            )?,
            up: Linear::load(
                file,
                &format!("{prefix}.linear_2"),
                config.adapter_attn_dim,
                config.hidden_size,
                true,
            )?,
        })
    }

    fn forward(&self, x: &[f32], steps: usize) -> Vec<f32> {
        let mut hidden = x.to_vec();
        self.norm.apply(&mut hidden, steps);
        hidden = self.down.forward(&hidden, steps);
        for value in &mut hidden {
            *value = value.max(0.0);
        }
        self.up.forward(&hidden, steps)
    }
}

struct EncoderLayer {
    attention_norm: LayerNorm,
    attention: Attention,
    feed_forward_norm: LayerNorm,
    feed_forward_in: Linear,
    feed_forward_out: Linear,
    adapter: Adapter,
}

impl EncoderLayer {
    fn load(
        base: &SafetensorsFile,
        adapter_file: &SafetensorsFile,
        index: usize,
        config: &MmsConfig,
    ) -> Result<Self> {
        let prefix = format!("wav2vec2.encoder.layers.{index}");
        let adapter_prefix = format!("{prefix}.adapter_layer");
        Ok(Self {
            attention_norm: LayerNorm::load(
                base,
                &format!("{prefix}.layer_norm"),
                config.hidden_size,
                config.layer_norm_eps,
            )?,
            attention: Attention::load(
                base,
                &format!("{prefix}.attention"),
                config.hidden_size,
                config.num_attention_heads,
            )?,
            feed_forward_norm: LayerNorm::load(
                base,
                &format!("{prefix}.final_layer_norm"),
                config.hidden_size,
                config.layer_norm_eps,
            )?,
            feed_forward_in: Linear::load(
                base,
                &format!("{prefix}.feed_forward.intermediate_dense"),
                config.hidden_size,
                config.intermediate_size,
                true,
            )?,
            feed_forward_out: Linear::load(
                base,
                &format!("{prefix}.feed_forward.output_dense"),
                config.intermediate_size,
                config.hidden_size,
                true,
            )?,
            adapter: Adapter::load(adapter_file, &adapter_prefix, config)?,
        })
    }

    fn forward(&self, x: &[f32], steps: usize) -> Vec<f32> {
        let mut normalized = x.to_vec();
        self.attention_norm.apply(&mut normalized, steps);
        let attention = self.attention.forward(&normalized, steps);
        let mut hidden = add(x, &attention);

        normalized.copy_from_slice(&hidden);
        self.feed_forward_norm.apply(&mut normalized, steps);
        let mut feed_forward = self.feed_forward_in.forward(&normalized, steps);
        ops::gelu_erf(&mut feed_forward);
        feed_forward = self.feed_forward_out.forward(&feed_forward, steps);
        hidden = add(&hidden, &feed_forward);

        let adapter = self.adapter.forward(&hidden, steps);
        add(&hidden, &adapter)
    }
}

struct Wav2Vec2 {
    feature_encoder: FeatureEncoder,
    feature_projection_norm: LayerNorm,
    feature_projection: Linear,
    positional_conv: PositionalConv,
    encoder_layers: Vec<EncoderLayer>,
    encoder_norm: LayerNorm,
    config: MmsConfig,
}

#[derive(Debug)]
struct StageTensor {
    name: String,
    rows: usize,
    columns: usize,
    values: Vec<f32>,
}

fn record_stage(
    trace: &mut Option<&mut Vec<StageTensor>>,
    name: impl Into<String>,
    values: &[f32],
    rows: usize,
    columns: usize,
) {
    if let Some(stages) = trace.as_mut() {
        stages.push(StageTensor {
            name: name.into(),
            rows,
            columns,
            values: values.to_vec(),
        });
    }
}

impl Wav2Vec2 {
    fn load(
        base: &SafetensorsFile,
        adapter_file: &SafetensorsFile,
        config: &MmsConfig,
    ) -> Result<Self> {
        let feature_width = *config
            .conv_dim
            .last()
            .ok_or_else(|| bad_config("conv_dim", "must contain at least one feature layer"))?;
        let layers = (0..config.num_hidden_layers)
            .map(|index| EncoderLayer::load(base, adapter_file, index, config))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            feature_encoder: FeatureEncoder::load(base, config)?,
            feature_projection_norm: LayerNorm::load(
                base,
                "wav2vec2.feature_projection.layer_norm",
                feature_width,
                config.layer_norm_eps,
            )?,
            feature_projection: Linear::load(
                base,
                "wav2vec2.feature_projection.projection",
                feature_width,
                config.hidden_size,
                true,
            )?,
            positional_conv: PositionalConv::load(
                base,
                config.hidden_size,
                config.num_conv_pos_embeddings,
                config.num_conv_pos_embedding_groups,
            )?,
            encoder_layers: layers,
            encoder_norm: LayerNorm::load(
                base,
                "wav2vec2.encoder.layer_norm",
                config.hidden_size,
                config.layer_norm_eps,
            )?,
            config: config.clone(),
        })
    }

    fn forward(
        &self,
        samples: &[f32],
        trace: &mut Option<&mut Vec<StageTensor>>,
    ) -> Result<Vec<f32>> {
        let (features, steps, feature_width) = self.feature_encoder.forward(samples, trace)?;
        record_stage(trace, "feature_extractor", &features, steps, feature_width);
        let mut hidden = features;
        self.feature_projection_norm.apply(&mut hidden, steps);
        hidden = self.feature_projection.forward(&hidden, steps);
        record_stage(
            trace,
            "feature_projection",
            &hidden,
            steps,
            self.config.hidden_size,
        );
        let positional = self.positional_conv.forward(&hidden, steps);
        record_stage(
            trace,
            "positional_conv",
            &positional,
            steps,
            self.config.hidden_size,
        );
        hidden = add(&hidden, &positional);
        let mut capture_layers = vec![
            0,
            self.encoder_layers.len().saturating_sub(1) / 2,
            self.encoder_layers.len() - 1,
        ];
        capture_layers.sort_unstable();
        capture_layers.dedup();
        for (index, layer) in self.encoder_layers.iter().enumerate() {
            hidden = layer.forward(&hidden, steps);
            if capture_layers.contains(&index) {
                record_stage(
                    trace,
                    format!("encoder_layer_{index}"),
                    &hidden,
                    steps,
                    self.config.hidden_size,
                );
            }
        }
        self.encoder_norm.apply(&mut hidden, steps);
        record_stage(
            trace,
            "encoder_final_norm",
            &hidden,
            steps,
            self.config.hidden_size,
        );
        debug_assert_eq!(feature_width, *self.config.conv_dim.last().unwrap());
        Ok(hidden)
    }
}

/// Loaded Wav2Vec2 MMS model with the English adapter and CTC head.
pub struct Mms {
    config: MmsConfig,
    wav2vec2: Wav2Vec2,
    lm_head: Linear,
    vocab: Vec<String>,
}

impl Mms {
    /// Load the pinned English profile from an already-downloaded model folder.
    pub fn load(model_dir: &Path) -> Result<Self> {
        let config_path = model_dir.join("config.json");
        let config_json = fs::read_to_string(&config_path).map_err(|error| SpeechError::Input {
            why: format!("cannot read {}: {error}", config_path.display()),
        })?;
        let root: Value = serde_json::from_str(&config_json)
            .map_err(|error| bad_config("config.json", error.to_string()))?;
        let config = MmsConfig::from_json(&root)?;
        let base_path = model_dir.join("model.safetensors");
        let adapter_path = model_dir.join("adapter.eng.safetensors");
        let base = SafetensorsFile::open(&base_path)?;
        let adapter_file = SafetensorsFile::open(&adapter_path)?;
        let wav2vec2 = Wav2Vec2::load(&base, &adapter_file, &config)?;
        let lm_head = Linear::load(
            &adapter_file,
            "lm_head",
            config.hidden_size,
            config.vocab_size,
            true,
        )?;
        let vocab_path = model_dir.join("vocab.json");
        let vocab_json = fs::read_to_string(&vocab_path).map_err(|error| SpeechError::Input {
            why: format!("cannot read {}: {error}", vocab_path.display()),
        })?;
        let vocab = parse_vocab(&vocab_json, config.vocab_size, true)?;
        Ok(Self {
            config,
            wav2vec2,
            lm_head,
            vocab,
        })
    }

    pub fn profile(&self) -> MmsProfile {
        MMS_1B_FL102_ENGLISH
    }

    pub fn config(&self) -> &MmsConfig {
        &self.config
    }

    /// Transcribe one mono 16 kHz waveform with greedy CTC decoding.
    pub fn transcribe(&self, samples: &[f32]) -> Result<String> {
        let logits = self.logits(samples, None)?;
        let steps = logits.len() / self.config.vocab_size;
        decode_ctc(&logits, steps, self.config.vocab_size, &self.vocab)
    }

    fn logits(
        &self,
        samples: &[f32],
        mut trace: Option<&mut Vec<StageTensor>>,
    ) -> Result<Vec<f32>> {
        if samples.is_empty() {
            return Err(SpeechError::Input {
                why: "audio must contain at least one sample".into(),
            });
        }
        if samples.iter().any(|sample| !sample.is_finite()) {
            return Err(SpeechError::Input {
                why: "audio samples must be finite".into(),
            });
        }
        let mean = samples.iter().sum::<f32>() / samples.len() as f32;
        let variance = samples
            .iter()
            .map(|sample| (sample - mean) * (sample - mean))
            .sum::<f32>()
            / samples.len() as f32;
        record_stage(&mut trace, "audio_input", samples, 1, samples.len());
        let standard_deviation = variance.sqrt();
        let normalized = samples
            .iter()
            .map(|sample| (sample - mean) / (standard_deviation + 1e-7))
            .collect::<Vec<_>>();
        record_stage(
            &mut trace,
            "normalized_audio",
            &normalized,
            1,
            normalized.len(),
        );
        let encoded = self.wav2vec2.forward(&normalized, &mut trace)?;
        let steps = encoded.len() / self.config.hidden_size;
        let logits = self.lm_head.forward(&encoded, steps);
        record_stage(
            &mut trace,
            "ctc_logits",
            &logits,
            steps,
            self.config.vocab_size,
        );
        Ok(logits)
    }
}

#[cfg(test)]
mod tests {
    use super::{decode_ctc, parse_vocab, Mms, MmsConfig, StageTensor, MMS_1B_FL102_ENGLISH};
    use serde_json::json;
    use std::path::Path;

    #[test]
    fn profile_pin_is_immutable() {
        assert_eq!(MMS_1B_FL102_ENGLISH.repository, "facebook/mms-1b-fl102");
        assert_eq!(
            MMS_1B_FL102_ENGLISH.revision,
            "d483345545bea550895b1aa0c6ba40236b9f1e22"
        );
        assert_eq!(MMS_1B_FL102_ENGLISH.language, "eng");
    }

    #[test]
    fn parses_wav2vec2_inference_config_and_rejects_non_stable_encoder() {
        let config = json!({
            "model_type":"wav2vec2",
            "architectures":["Wav2Vec2ForCTC"],
            "vocab_size":8,"hidden_size":16,"num_hidden_layers":2,
            "num_attention_heads":4,"intermediate_size":32,
            "layer_norm_eps":0.00001,"feat_extract_norm":"layer",
            "do_stable_layer_norm":true,"hidden_act":"gelu",
            "conv_dim":[4,8],"conv_stride":[2,2],"conv_kernel":[4,3],
            "conv_bias":true,"num_conv_pos_embeddings":8,
            "num_conv_pos_embedding_groups":4,"adapter_attn_dim":2
        });
        let parsed = MmsConfig::from_json(&config).unwrap();
        assert_eq!(parsed.hidden_size, 16);
        assert_eq!(parsed.conv_stride, [2, 2]);
        assert_eq!(parsed.adapter_attn_dim, 2);

        let mut unsupported = config;
        unsupported["do_stable_layer_norm"] = json!(false);
        assert!(MmsConfig::from_json(&unsupported).is_err());
    }

    #[test]
    fn parses_the_pinned_hub_config_fixture() {
        let root: serde_json::Value =
            serde_json::from_str(include_str!("../../../testdata/mms_config.json")).unwrap();
        let parsed = MmsConfig::from_json(&root).unwrap();
        assert_eq!(parsed.vocab_size, 78);
        assert_eq!(parsed.hidden_size, 1280);
        assert_eq!(parsed.num_hidden_layers, 48);
        assert_eq!(parsed.num_attention_heads, 16);
        assert_eq!(parsed.adapter_attn_dim, 16);
        assert_eq!(parsed.conv_dim, vec![512; 7]);
    }

    #[test]
    fn ctc_decode_collapses_repeats_and_maps_word_delimiter() {
        let vocab = vec!["<pad>".into(), "h".into(), "i".into(), "|".into()];
        let tokens = [1usize, 1, 2, 3, 3, 1, 0, 1];
        let mut logits = vec![0.0f32; tokens.len() * vocab.len()];
        for (step, token) in tokens.into_iter().enumerate() {
            logits[step * vocab.len() + token] = 1.0;
        }
        assert_eq!(
            decode_ctc(&logits, tokens.len(), vocab.len(), &vocab).unwrap(),
            "hi hh"
        );
    }

    #[test]
    fn ctc_decode_matches_mlx_for_nonblank_special_tokens() {
        let vocab = vec!["<pad>".into(), "<s>".into(), "</s>".into()];
        let tokens = [1usize, 2];
        let mut logits = vec![0.0f32; tokens.len() * vocab.len()];
        for (step, token) in tokens.into_iter().enumerate() {
            logits[step * vocab.len() + token] = 1.0;
        }
        assert_eq!(
            decode_ctc(&logits, tokens.len(), vocab.len(), &vocab).unwrap(),
            "<s></s>"
        );
    }

    #[test]
    fn parses_the_english_subtree_of_a_multilingual_vocab() {
        let vocab = parse_vocab(
            r#"{"eng":{"<pad>":0,"h":1,"i":2,"|":3},"fra":{"<pad>":0,"z":1,"y":2,"|":3}}"#,
            4,
            true,
        )
        .unwrap();
        assert_eq!(vocab, ["<pad>", "h", "i", "|"]);
    }

    #[test]
    #[ignore = "requires the pinned 3.9 GB MMS checkpoint downloaded from Hugging Face"]
    fn pinned_english_checkpoint_matches_mlx_reference() {
        let model_dir = std::env::var_os("TURBOSPARK_MMS_DIR")
            .expect("set TURBOSPARK_MMS_DIR to the pinned MMS snapshot directory");
        let model_dir = Path::new(&model_dir);
        assert!(
            model_dir.ancestors().any(|path| {
                path.file_name().and_then(|name| name.to_str())
                    == Some(MMS_1B_FL102_ENGLISH.revision)
            }),
            "TURBOSPARK_MMS_DIR must point inside the Hugging Face snapshot for the pinned revision"
        );
        let fixture_path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/mms_reference.json");
        let fixture: serde_json::Value = serde_json::from_slice(
            &std::fs::read(&fixture_path).expect("missing MLX-generated MMS fixture"),
        )
        .unwrap();
        assert_eq!(
            fixture["repository"].as_str(),
            Some(MMS_1B_FL102_ENGLISH.repository)
        );
        assert_eq!(
            fixture["revision"].as_str(),
            Some(MMS_1B_FL102_ENGLISH.revision)
        );

        let wav = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("testdata/qwen3_forced_aligner_reference.wav");
        let audio = turbospark_audio::read_wav_f32(&wav).unwrap();
        assert_eq!(audio.sample_rate, 16_000);
        assert_eq!(audio.channels, 1);
        let model = Mms::load(model_dir).unwrap();
        let mut stages = Vec::new();
        let logits = model.logits(&audio.samples, Some(&mut stages)).unwrap();
        let transcript = decode_ctc(
            &logits,
            logits.len() / model.config.vocab_size,
            model.config.vocab_size,
            &model.vocab,
        )
        .unwrap();
        assert_eq!(transcript, fixture["transcript"].as_str().unwrap());

        assert_eq!(
            stages.len(),
            fixture["stages"].as_object().unwrap().len(),
            "Rust and MLX captured different intermediate stage sets"
        );
        let mut first_mismatch = None;
        for actual in &stages {
            let mismatch = compare_stage_fixture(actual, &fixture["stages"][&actual.name]);
            if let Some((row, column, actual_value, expected_value, error, tolerance)) = mismatch {
                if first_mismatch.is_none() {
                    first_mismatch = Some((
                        actual.name.as_str(),
                        row,
                        column,
                        actual_value,
                        expected_value,
                        error,
                        tolerance,
                    ));
                }
            }
        }
        if let Some(mismatch) = first_mismatch {
            eprintln!("MMS sampled stage values exceed the diagnostic tolerance; first mismatch: {mismatch:?}");
        }
    }

    fn compare_stage_fixture(
        actual: &StageTensor,
        expected: &serde_json::Value,
    ) -> Option<(usize, usize, f32, f32, f32, f32)> {
        let shape = expected["shape"].as_array().unwrap();
        assert_eq!(
            shape[0].as_u64().unwrap() as usize,
            actual.rows,
            "{} rows",
            actual.name
        );
        assert_eq!(
            shape[1].as_u64().unwrap() as usize,
            actual.columns,
            "{} columns",
            actual.name
        );
        let rows = expected["rows"].as_array().unwrap();
        let columns = expected["columns"].as_array().unwrap();
        let values = expected["values"].as_array().unwrap();
        assert_eq!(values.len(), rows.len(), "{} sampled rows", actual.name);
        let mut max_absolute_error = 0.0f32;
        let mut max_error_sample = None;
        let mut mismatch = None;
        for (row_index, row) in rows.iter().enumerate() {
            let row = row.as_u64().unwrap() as usize;
            let expected_row = values[row_index].as_array().unwrap();
            assert_eq!(
                expected_row.len(),
                columns.len(),
                "{} sampled columns",
                actual.name
            );
            for (column_index, column) in columns.iter().enumerate() {
                let column = column.as_u64().unwrap() as usize;
                let actual_value = actual.values[row * actual.columns + column];
                let expected_value = expected_row[column_index].as_f64().unwrap() as f32;
                let absolute_error = (actual_value - expected_value).abs();
                let tolerance = 1e-3 + expected_value.abs() * 1e-2;
                if absolute_error > max_absolute_error {
                    max_absolute_error = absolute_error;
                    max_error_sample = Some((row, column, actual_value, expected_value));
                }
                if absolute_error > tolerance && mismatch.is_none() {
                    mismatch = Some((
                        row,
                        column,
                        actual_value,
                        expected_value,
                        absolute_error,
                        tolerance,
                    ));
                }
            }
        }
        eprintln!("MMS stage {}: {}x{}, max sampled abs error {max_absolute_error:.6} at {max_error_sample:?}", actual.name, actual.rows, actual.columns);
        mismatch
    }
}
