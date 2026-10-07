//! Wav2Vec2 base CTC speech recognition.
//!
//! Reference: `mlx_audio/stt/models/wav2vec/` (wav2vec.py,
//! feature_extractor.py) at mlx-audio 0.5.7, commit
//! `e1b19b9054bf163f5d812221a54fcc346f1890e9`. The upstream module exposes a
//! bare Wav2Vec2Model backbone; its sanitize drops the CTC head, and the
//! mlx-audio STT loader does not auto-route `model_type "wav2vec2"`
//! checkpoints. This family port keeps the checkpoint's own `lm_head` so the
//! standalone distribution is transcribable, matching transformers
//! `Wav2Vec2ForCTC`.

use std::fs;
use std::path::Path;

use serde_json::Value;
use turbospark_model_io::safetensors::SafetensorsFile;

use crate::nn::{bad_config, load_tensor, LayerNorm, Linear};
use crate::ops;
use crate::{Result, SpeechError};

pub(super) mod backbone;
pub(crate) mod ctc;

use backbone::{
    add, channels_first_to_rows, decode_ctc, parse_vocab, positive, positive_array, Attention,
    PositionalConv,
};

/// Immutable Hugging Face checkpoint profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Wav2VecProfile {
    pub name: &'static str,
    pub repository: &'static str,
    pub revision: &'static str,
}

pub const WAV2VEC2_BASE_960H: Wav2VecProfile = Wav2VecProfile {
    name: "Wav2Vec2 Base 960h",
    repository: "facebook/wav2vec2-base-960h",
    revision: "22aad52d435eb6dbaf354bdad9b0da84ce7d6156",
};

/// Inference settings from the pinned Wav2Vec2 checkpoint config.
#[derive(Debug, Clone, PartialEq)]
pub struct Wav2VecConfig {
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
}

impl Wav2VecConfig {
    /// Parses a Wav2Vec2ForCTC config for the non-stable group-norm family.
    ///
    /// The stable-layer-norm encoder variant lives in the `mms` family port;
    /// this module owns the classic post-norm layout.
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
        if root.get("feat_extract_norm").and_then(Value::as_str) != Some("group")
            || root.get("do_stable_layer_norm").and_then(Value::as_bool) != Some(false)
            || root.get("hidden_act").and_then(Value::as_str) != Some("gelu")
        {
            return Err(SpeechError::Unsupported {
                why: "the wav2vec family requires group-normalized GELU features and the non-stable post-norm encoder; stable-layer-norm checkpoints belong to the mms family".into(),
            });
        }
        if root
            .get("adapter_attn_dim")
            .and_then(Value::as_u64)
            .is_some()
        {
            return Err(SpeechError::Unsupported {
                why: "adapter checkpoints belong to the mms family".into(),
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

/// First feature convolution: conv, then per-channel group normalization over
/// the temporal dimension (MLX `nn.GroupNorm(num_groups=channels)`,
/// pytorch-compatible epsilon placement), then GELU.
struct GroupNormConv {
    weight: Vec<f32>,
    bias: Option<Vec<f32>>,
    norm_weight: Vec<f32>,
    norm_bias: Vec<f32>,
    epsilon: f32,
    input: usize,
    output: usize,
    kernel: usize,
    stride: usize,
}

impl GroupNormConv {
    fn load(file: &SafetensorsFile, index: usize, config: &Wav2VecConfig) -> Result<Self> {
        let prefix = format!("wav2vec2.feature_extractor.conv_layers.{index}");
        Ok(Self {
            weight: load_tensor(
                file,
                &format!("{prefix}.conv.weight"),
                &[config.conv_dim[index], 1, config.conv_kernel[index]],
            )?,
            bias: if config.conv_bias {
                Some(load_tensor(
                    file,
                    &format!("{prefix}.conv.bias"),
                    &[config.conv_dim[index]],
                )?)
            } else {
                None
            },
            norm_weight: load_tensor(
                file,
                &format!("{prefix}.layer_norm.weight"),
                &[config.conv_dim[index]],
            )?,
            norm_bias: load_tensor(
                file,
                &format!("{prefix}.layer_norm.bias"),
                &[config.conv_dim[index]],
            )?,
            epsilon: config.layer_norm_eps,
            input: 1,
            output: config.conv_dim[index],
            kernel: config.conv_kernel[index],
            stride: config.conv_stride[index],
        })
    }

    /// `channels_first` is `[channel * steps + step]`; the norm statistics are
    /// per channel over the whole sequence.
    fn forward(&self, channels_first: &[f32], steps: usize) -> Result<Vec<f32>> {
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
        for channel in 0..self.output {
            let row = &mut hidden[channel * output_steps..(channel + 1) * output_steps];
            let mean = row.iter().sum::<f32>() / output_steps as f32;
            let variance = row
                .iter()
                .map(|value| (value - mean) * (value - mean))
                .sum::<f32>()
                / output_steps as f32;
            let scale = self.norm_weight[channel];
            let shift = self.norm_bias[channel];
            for value in row.iter_mut() {
                *value = (*value - mean) / (variance + self.epsilon).sqrt() * scale + shift;
            }
        }
        ops::gelu_erf(&mut hidden);
        Ok(hidden)
    }
}

/// Feature convolutions 1..n: conv and GELU with no normalization.
struct PlainConv {
    weight: Vec<f32>,
    bias: Option<Vec<f32>>,
    input: usize,
    output: usize,
    kernel: usize,
    stride: usize,
}

impl PlainConv {
    fn load(file: &SafetensorsFile, index: usize, config: &Wav2VecConfig) -> Result<Self> {
        let prefix = format!("wav2vec2.feature_extractor.conv_layers.{index}");
        Ok(Self {
            weight: load_tensor(
                file,
                &format!("{prefix}.conv.weight"),
                &[
                    config.conv_dim[index],
                    config.conv_dim[index - 1],
                    config.conv_kernel[index],
                ],
            )?,
            bias: if config.conv_bias {
                Some(load_tensor(
                    file,
                    &format!("{prefix}.conv.bias"),
                    &[config.conv_dim[index]],
                )?)
            } else {
                None
            },
            input: config.conv_dim[index - 1],
            output: config.conv_dim[index],
            kernel: config.conv_kernel[index],
            stride: config.conv_stride[index],
        })
    }

    fn forward(&self, channels_first: &[f32], steps: usize) -> Result<Vec<f32>> {
        if steps < self.kernel {
            return Err(SpeechError::Input {
                why: "audio is too short for the Wav2Vec2 convolution stack".into(),
            });
        }
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
        ops::gelu_erf(&mut hidden);
        Ok(hidden)
    }
}

struct FeatureEncoder {
    group_norm_conv: GroupNormConv,
    plain_convs: Vec<PlainConv>,
}

impl FeatureEncoder {
    fn load(file: &SafetensorsFile, config: &Wav2VecConfig) -> Result<Self> {
        let group_norm_conv = GroupNormConv::load(file, 0, config)?;
        let plain_convs = (1..config.conv_dim.len())
            .map(|index| PlainConv::load(file, index, config))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            group_norm_conv,
            plain_convs,
        })
    }

    fn forward(&self, samples: &[f32]) -> Result<(Vec<f32>, usize, usize)> {
        let conv = &self.group_norm_conv;
        let mut hidden = conv.forward(samples, samples.len())?;
        let mut steps = (samples.len() - conv.kernel) / conv.stride + 1;
        for layer in &self.plain_convs {
            let next = layer.forward(&hidden, steps)?;
            steps = (steps - layer.kernel) / layer.stride + 1;
            hidden = next;
        }
        let channels = self
            .plain_convs
            .last()
            .map(|layer| layer.output)
            .unwrap_or(conv.output);
        Ok((
            channels_first_to_rows(&hidden, channels, steps),
            steps,
            channels,
        ))
    }
}

/// Classic post-norm encoder layer: attention on the un-normalized input,
/// residual, layer norm, feed forward on the normalized input, residual,
/// final layer norm.
struct EncoderLayer {
    attention: Attention,
    attention_norm: LayerNorm,
    feed_forward_in: Linear,
    feed_forward_out: Linear,
    final_norm: LayerNorm,
}

impl EncoderLayer {
    fn load(file: &SafetensorsFile, index: usize, config: &Wav2VecConfig) -> Result<Self> {
        let prefix = format!("wav2vec2.encoder.layers.{index}");
        Ok(Self {
            attention: Attention::load(
                file,
                &format!("{prefix}.attention"),
                config.hidden_size,
                config.num_attention_heads,
            )?,
            attention_norm: LayerNorm::load(
                file,
                &format!("{prefix}.layer_norm"),
                config.hidden_size,
                config.layer_norm_eps,
            )?,
            feed_forward_in: Linear::load(
                file,
                &format!("{prefix}.feed_forward.intermediate_dense"),
                config.hidden_size,
                config.intermediate_size,
                true,
            )?,
            feed_forward_out: Linear::load(
                file,
                &format!("{prefix}.feed_forward.output_dense"),
                config.intermediate_size,
                config.hidden_size,
                true,
            )?,
            final_norm: LayerNorm::load(
                file,
                &format!("{prefix}.final_layer_norm"),
                config.hidden_size,
                config.layer_norm_eps,
            )?,
        })
    }

    fn forward(&self, x: &[f32], steps: usize) -> Vec<f32> {
        let attention = self.attention.forward(x, steps);
        let mut hidden = add(x, &attention);
        self.attention_norm.apply(&mut hidden, steps);
        let mut feed_forward = self.feed_forward_in.forward(&hidden, steps);
        ops::gelu_erf(&mut feed_forward);
        feed_forward = self.feed_forward_out.forward(&feed_forward, steps);
        hidden = add(&hidden, &feed_forward);
        self.final_norm.apply(&mut hidden, steps);
        hidden
    }
}

struct Wav2Vec2 {
    feature_encoder: FeatureEncoder,
    feature_projection_norm: LayerNorm,
    feature_projection: Linear,
    positional_conv: PositionalConv,
    encoder_norm: LayerNorm,
    encoder_layers: Vec<EncoderLayer>,
    config: Wav2VecConfig,
}

impl Wav2Vec2 {
    fn load(file: &SafetensorsFile, config: &Wav2VecConfig) -> Result<Self> {
        let feature_width = *config
            .conv_dim
            .last()
            .ok_or_else(|| bad_config("conv_dim", "must contain at least one feature layer"))?;
        let encoder_layers = (0..config.num_hidden_layers)
            .map(|index| EncoderLayer::load(file, index, config))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            feature_encoder: FeatureEncoder::load(file, config)?,
            feature_projection_norm: LayerNorm::load(
                file,
                "wav2vec2.feature_projection.layer_norm",
                feature_width,
                config.layer_norm_eps,
            )?,
            feature_projection: Linear::load(
                file,
                "wav2vec2.feature_projection.projection",
                feature_width,
                config.hidden_size,
                true,
            )?,
            positional_conv: PositionalConv::load(
                file,
                config.hidden_size,
                config.num_conv_pos_embeddings,
                config.num_conv_pos_embedding_groups,
            )?,
            encoder_norm: LayerNorm::load(
                file,
                "wav2vec2.encoder.layer_norm",
                config.hidden_size,
                config.layer_norm_eps,
            )?,
            encoder_layers,
            config: config.clone(),
        })
    }

    fn forward(&self, samples: &[f32]) -> Result<Vec<f32>> {
        let (features, steps, feature_width) = self.feature_encoder.forward(samples)?;
        let mut hidden = features;
        self.feature_projection_norm.apply(&mut hidden, steps);
        hidden = self.feature_projection.forward(&hidden, steps);
        let positional = self.positional_conv.forward(&hidden, steps);
        hidden = add(&hidden, &positional);
        // The non-stable encoder normalizes after the positional convolution,
        // before the layer stack, and applies no final norm.
        self.encoder_norm.apply(&mut hidden, steps);
        for layer in &self.encoder_layers {
            hidden = layer.forward(&hidden, steps);
        }
        debug_assert_eq!(feature_width, *self.config.conv_dim.last().unwrap());
        Ok(hidden)
    }
}

/// Loaded Wav2Vec2 base model with its own CTC head.
pub struct Wav2Vec {
    config: Wav2VecConfig,
    wav2vec2: Wav2Vec2,
    lm_head: Linear,
    vocab: Vec<String>,
}

impl Wav2Vec {
    /// Load the pinned profile from an already-downloaded model folder.
    pub fn load(model_dir: &Path) -> Result<Self> {
        let config_path = model_dir.join("config.json");
        let config_json = fs::read_to_string(&config_path).map_err(|error| SpeechError::Input {
            why: format!("cannot read {}: {error}", config_path.display()),
        })?;
        let root: Value = serde_json::from_str(&config_json)
            .map_err(|error| bad_config("config.json", error.to_string()))?;
        let config = Wav2VecConfig::from_json(&root)?;
        let base = SafetensorsFile::open(&model_dir.join("model.safetensors"))?;
        let wav2vec2 = Wav2Vec2::load(&base, &config)?;
        let lm_head = Linear::load(
            &base,
            "lm_head",
            config.hidden_size,
            config.vocab_size,
            true,
        )?;
        let vocab_json = fs::read_to_string(model_dir.join("vocab.json")).map_err(|error| {
            SpeechError::Input {
                why: format!("cannot read vocab.json: {error}"),
            }
        })?;
        let vocab = parse_vocab(&vocab_json, config.vocab_size, false)?;
        Ok(Self {
            config,
            wav2vec2,
            lm_head,
            vocab,
        })
    }

    pub fn profile(&self) -> Wav2VecProfile {
        WAV2VEC2_BASE_960H
    }

    pub fn config(&self) -> &Wav2VecConfig {
        &self.config
    }

    /// Transcribe one mono 16 kHz waveform with greedy CTC decoding.
    pub fn transcribe(&self, samples: &[f32]) -> Result<String> {
        let logits = self.logits(samples)?;
        let steps = logits.len() / self.config.vocab_size;
        decode_ctc(&logits, steps, self.config.vocab_size, &self.vocab)
    }

    fn logits(&self, samples: &[f32]) -> Result<Vec<f32>> {
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
        // HF feature extractor convention: epsilon inside the square root.
        let mean = samples.iter().sum::<f32>() / samples.len() as f32;
        let variance = samples
            .iter()
            .map(|sample| (sample - mean) * (sample - mean))
            .sum::<f32>()
            / samples.len() as f32;
        let normalized = samples
            .iter()
            .map(|sample| (sample - mean) / (variance + 1e-7).sqrt())
            .collect::<Vec<_>>();
        let encoded = self.wav2vec2.forward(&normalized)?;
        let steps = encoded.len() / self.config.hidden_size;
        Ok(self.lm_head.forward(&encoded, steps))
    }
}

#[cfg(test)]
mod tests {
    use super::{add, decode_ctc, parse_vocab, Wav2Vec, Wav2VecConfig, WAV2VEC2_BASE_960H};
    use serde_json::{json, Value};
    use std::path::Path;

    const FIXTURE: &str = include_str!("../../../testdata/wav2vec_reference.json");

    #[derive(serde::Deserialize)]
    struct Spots {
        shape: Vec<usize>,
        rows: Vec<usize>,
        columns: Vec<usize>,
        values: Vec<Vec<f32>>,
    }

    fn load_fixture() -> Value {
        serde_json::from_str(FIXTURE).expect("valid wav2vec reference fixture")
    }

    fn load_pinned_model() -> Option<Wav2Vec> {
        let model_dir = std::env::var_os("TURBOSPARK_WAV2VEC_MODEL_DIR")?;
        Some(Wav2Vec::load(Path::new(&model_dir)).expect("pinned wav2vec checkpoint loads"))
    }

    fn normalized_waveform() -> Vec<f32> {
        let audio_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("testdata/qwen3_forced_aligner_reference.wav");
        let waveform = crate::wav::read_wav_f32(&audio_path).expect("reference WAV loads");
        assert_eq!(waveform.sample_rate, 16_000);
        let samples = &waveform.samples;
        let mean = samples.iter().sum::<f32>() / samples.len() as f32;
        let variance = samples
            .iter()
            .map(|sample| (sample - mean) * (sample - mean))
            .sum::<f32>()
            / samples.len() as f32;
        samples
            .iter()
            .map(|sample| (sample - mean) / (variance + 1e-7).sqrt())
            .collect()
    }

    fn compare_spots(
        actual: &[f32],
        rows: usize,
        columns: usize,
        spots: &Spots,
        label: &str,
    ) -> f32 {
        assert_eq!([rows, columns], spots.shape.as_slice(), "{label} shape");
        let mut worst = 0.0f32;
        for (row_index, &row) in spots.rows.iter().enumerate() {
            for (column_index, &column) in spots.columns.iter().enumerate() {
                let expected = spots.values[row_index][column_index];
                let diff = (actual[row * columns + column] - expected).abs();
                worst = worst.max(diff);
                assert!(diff < 2.0e-4, "{label} [{row},{column}] differs by {diff}");
            }
        }
        worst
    }

    #[test]
    fn profile_pin_is_immutable() {
        assert_eq!(WAV2VEC2_BASE_960H.repository, "facebook/wav2vec2-base-960h");
        assert_eq!(
            WAV2VEC2_BASE_960H.revision,
            "22aad52d435eb6dbaf354bdad9b0da84ce7d6156"
        );
    }

    #[test]
    fn parses_the_base_config_and_rejects_stable_encoder_and_adapters() {
        let config = json!({
            "model_type":"wav2vec2",
            "architectures":["Wav2Vec2ForCTC"],
            "vocab_size":32,"hidden_size":768,"num_hidden_layers":12,
            "num_attention_heads":12,"intermediate_size":3072,
            "layer_norm_eps":0.00001,"feat_extract_norm":"group",
            "do_stable_layer_norm":false,"hidden_act":"gelu",
            "conv_dim":[512,512,512,512,512,512,512],"conv_stride":[5,2,2,2,2,2,2],
            "conv_kernel":[10,3,3,3,3,2,2],"conv_bias":false,
            "num_conv_pos_embeddings":128,"num_conv_pos_embedding_groups":16
        });
        let parsed = Wav2VecConfig::from_json(&config).unwrap();
        assert_eq!(parsed.hidden_size, 768);
        assert_eq!(parsed.conv_kernel[0], 10);

        let mut stable = config.clone();
        stable["do_stable_layer_norm"] = json!(true);
        assert!(Wav2VecConfig::from_json(&stable).is_err());

        let mut adapter = config;
        adapter["adapter_attn_dim"] = json!(64);
        assert!(Wav2VecConfig::from_json(&adapter).is_err());
    }

    #[test]
    fn ctc_decode_collapses_repeats_and_maps_the_word_separator() {
        let vocab = vec![
            "<pad>".to_owned(),
            "|".to_owned(),
            "E".to_owned(),
            "T".to_owned(),
        ];
        let logits = vec![
            1.0, 0.0, 0.0, 0.0, // blank
            0.0, 0.0, 1.0, 0.0, // E
            0.0, 0.0, 1.0, 0.0, // E repeated
            0.0, 0.0, 0.0, 1.0, // T
            0.0, 1.0, 0.0, 0.0, // separator
            0.0, 0.0, 0.0, 1.0, // T
        ];
        assert_eq!(decode_ctc(&logits, 6, 4, &vocab).unwrap(), "ET T");
    }

    #[test]
    fn parses_a_flat_token_to_id_vocab() {
        let vocab = parse_vocab(r#"{"<pad>": 0, "|": 1, "E": 2}"#, 3, false).unwrap();
        assert_eq!(vocab, ["<pad>", "|", "E"]);
        assert!(parse_vocab(r#"{"<pad>": 0}"#, 2, false).is_err());
    }

    #[test]
    fn fixture_provenance_pins_the_reference_run() {
        let fixture = load_fixture();
        assert_eq!(
            fixture["provenance"]["revision"],
            "22aad52d435eb6dbaf354bdad9b0da84ce7d6156"
        );
        assert_eq!(
            fixture["provenance"]["source"],
            "mlx-audio wav2vec at commit e1b19b9054bf163f5d812221a54fcc346f1890e9"
        );
        assert_eq!(
            fixture["provenance"]["normalization"],
            "(x - mean) / sqrt(var + 1e-7)"
        );
    }

    #[test]
    #[ignore = "requires the pinned Wav2Vec2 checkpoint in TURBOSPARK_WAV2VEC_MODEL_DIR"]
    fn pinned_checkpoint_matches_the_fixture_stages_and_transcript() {
        let Some(model) = load_pinned_model() else {
            eprintln!("skipping: TURBOSPARK_WAV2VEC_MODEL_DIR is unset");
            return;
        };
        let fixture = load_fixture();
        let normalized = normalized_waveform();

        let fixture_normalized = &fixture["normalized_audio"];
        let first: Vec<f32> = fixture_normalized["first"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_f64().unwrap() as f32)
            .collect();
        for (index, expected) in first.iter().enumerate() {
            assert!((normalized[index] - expected).abs() < 1.0e-6);
        }

        let conv0 = model
            .wav2vec2
            .feature_encoder
            .group_norm_conv
            .forward(&normalized, normalized.len())
            .unwrap();
        let conv0_steps =
            (normalized.len() - model.config.conv_kernel[0]) / model.config.conv_stride[0] + 1;
        let conv0_spots: Spots =
            serde_json::from_value(fixture["conv0_group_norm"].clone()).unwrap();
        compare_spots(
            &conv0,
            model.config.conv_dim[0],
            conv0_steps,
            &conv0_spots,
            "conv0_group_norm",
        );

        let (features, steps, channels) =
            model.wav2vec2.feature_encoder.forward(&normalized).unwrap();
        let feature_spots: Spots =
            serde_json::from_value(fixture["feature_extractor"].clone()).unwrap();
        compare_spots(
            &features,
            steps,
            channels,
            &feature_spots,
            "feature_extractor",
        );

        let mut projected = features;
        model
            .wav2vec2
            .feature_projection_norm
            .apply(&mut projected, steps);
        projected = model.wav2vec2.feature_projection.forward(&projected, steps);
        let projection_spots: Spots =
            serde_json::from_value(fixture["feature_projection"].clone()).unwrap();
        compare_spots(
            &projected,
            steps,
            model.config.hidden_size,
            &projection_spots,
            "feature_projection",
        );

        let positional = model.wav2vec2.positional_conv.forward(&projected, steps);
        let mut hidden = add(&projected, &positional);
        model.wav2vec2.encoder_norm.apply(&mut hidden, steps);

        // The reference's output_hidden_states tuple records each layer's
        // INPUT state (and the final output after the loop), not each
        // layer's output.
        for (index, layer) in model.wav2vec2.encoder_layers.iter().enumerate() {
            for entry in fixture["hidden_states"].as_array().unwrap() {
                if entry["name"] == format!("encoder_layer_{index}") {
                    let spots: Spots = serde_json::from_value(entry["spots"].clone()).unwrap();
                    compare_spots(
                        &hidden,
                        steps,
                        model.config.hidden_size,
                        &spots,
                        &format!("encoder_layer_{index}_input"),
                    );
                }
            }
            hidden = layer.forward(&hidden, steps);
        }
        for entry in fixture["hidden_states"].as_array().unwrap() {
            if entry["name"] == "encoder_final" {
                let spots: Spots = serde_json::from_value(entry["spots"].clone()).unwrap();
                compare_spots(
                    &hidden,
                    steps,
                    model.config.hidden_size,
                    &spots,
                    "encoder_final",
                );
            }
        }

        let logits = model.lm_head.forward(&hidden, steps);
        let logit_spots: Spots = serde_json::from_value(fixture["ctc_logits"].clone()).unwrap();
        // The 768-length CTC dot product amplifies the hidden-state spot
        // tolerance; measured worst spot is ~2.2e-3, so the logits gate is
        // 5.0e-3 while the transcript and token ids stay exact.
        let mut worst = 0.0f32;
        for (row_index, &row) in logit_spots.rows.iter().enumerate() {
            for (column_index, &column) in logit_spots.columns.iter().enumerate() {
                let expected = logit_spots.values[row_index][column_index];
                let diff = (logits[row * model.config.vocab_size + column] - expected).abs();
                worst = worst.max(diff);
                assert!(
                    diff < 5.0e-3,
                    "ctc_logits [{row},{column}] differs by {diff}"
                );
            }
        }
        eprintln!("wav2vec fixture parity: ctc_logits worst {worst:.3e} (gate 5.0e-3)");

        let transcript = decode_ctc(&logits, steps, model.config.vocab_size, &model.vocab).unwrap();
        let expected = fixture["transcript"].as_str().unwrap();
        assert_eq!(
            transcript, expected,
            "checkpoint transcript must match the pinned MLX reference"
        );
        let expected_tokens: Vec<usize> = fixture["greedy_token_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|token| token.as_u64().unwrap() as usize)
            .collect();
        let mut tokens = Vec::new();
        let mut previous = None;
        for row in logits.chunks_exact(model.config.vocab_size) {
            let token = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0;
            if Some(token) != previous && token != 0 {
                tokens.push(token);
            }
            previous = Some(token);
        }
        assert_eq!(tokens, expected_tokens, "greedy token ids must match");
    }
}
