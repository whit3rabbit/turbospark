//! Shared dense LM backbones for the mlx-audio model families.
//!
//! Port of the `mlx_audio/lm` subset the TTS families import (vendored from
//! mlx-lm at the pinned mlx-audio commit): the KV caches (including the
//! rotating sliding-window ring), the rope constructions, and the llama,
//! qwen2, and qwen3 dense backbones plus the stage-1 additions gpt2,
//! cohere2, bailing_moe, and switch_layers, and the stage-2 falcon_h1
//! hybrid with its Mamba2 selective scan (`ssm`). Everything here is
//! portable single-batch f32: the reference generation loop these families
//! drive is batch 1, and Metal offload is a later runtime optimization.
//!
//! Layouts follow the crate conventions: activations are row-major
//! `[positions, width]`, linear weights stay in the HF `[out, in]` layout
//! (the bailing expert stacks are `[expert, out, in]`), and cache buffers
//! are `[heads, total_positions, head_dim]`. Token IDs crossing into this
//! module are signed 32-bit.

pub mod bailing_moe;
pub mod cache;
pub mod cohere2;
pub mod engine;
pub(crate) mod execution;
pub(crate) use execution::Execution;
pub mod falcon_h1;
pub mod gpt2;
pub mod llama;
pub mod qwen2;
pub mod qwen3;
pub mod rope;
pub mod ssm;
pub mod switch_layers;

#[cfg(test)]
mod tests;

pub use bailing_moe::BailingMoeModel;
pub use cache::{make_prompt_cache, KvCache, LayerCaches, PromptCache, RotatingKvCache};
pub use cohere2::Cohere2Model;
pub use engine::{Backbone, DenseConfig, Input};
pub use falcon_h1::{compute_mup_vector, FalconConfig, FalconH1Model, FalconLayerCache};
pub use gpt2::Gpt2Model;
pub use llama::LlamaModel;
pub use qwen2::Qwen2Model;
pub use qwen3::Qwen3Model;
pub use rope::Rope;
pub use switch_layers::SwitchGlu;

use std::path::Path;

use serde_json::Value;
use turbospark_model_io::safetensors::SafetensorsFile;

use crate::nn::{self, Linear};
use crate::quant::{self, QuantScheme};
use crate::{Result, SpeechError};

/// Checkpoint access for the LM backbones: the opened shards, the
/// quantization scheme from `config.json`, and the name prefix the family's
/// checkpoint uses (`model.` for a full-model save, empty for backbone-only
/// saves).
pub struct Weights {
    files: Vec<SafetensorsFile>,
    scheme: Option<CheckpointScheme>,
    prefix: String,
    config: Value,
    /// Opt-in tensor-name remap applied after the prefix on the prefixed
    /// loads (`linear`, `embedding`, `tensor`). For checkpoints whose raw
    /// layout names the backbone tensors differently from the engine reads
    /// (the voxtral_tts consolidated Mistral layout: `layers.N.attention.wq`
    /// against the engine's `layers.N.self_attn.q_proj`). Absolute-name
    /// loads bypass the map on purpose: they exist for family tensors that
    /// live outside the backbone.
    name_map: Option<NameMap>,
}

type NameMap = std::sync::Arc<dyn Fn(&str) -> String + Send + Sync>;

/// The quantization scheme a checkpoint was saved with. [`Weights::open`]
/// reads the MLX groupwise affine block from `config.json`;
/// [`Weights::open_single_with_scheme`] carries a caller-parsed scheme such
/// as the OmniVoice rowwise int8 format recorded in `mlx_manifest.json`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckpointScheme {
    /// MLX groupwise affine: U32-packed values with per-group scales and
    /// biases, `w = q * scale + bias`.
    Affine(QuantScheme),
    /// OmniVoice rowwise symmetric int8: I8 `[rows, groups, group_size]`
    /// weights with F16 `[rows, groups]` scales, `w = q * scale`, no biases.
    OmnivoiceRowwise,
}

/// Bias policy for a linear load.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bias {
    /// The tensor must exist.
    Required,
    /// Used when present.
    Optional,
    /// Never loaded even when present.
    None,
}

impl Weights {
    /// Opens `model.safetensors` (or the shards of its index) plus
    /// `config.json` from `dir`.
    pub fn open(dir: &Path) -> Result<Self> {
        let files = nn::open_shards(dir)?;
        let config = quant::read_json(&dir.join("config.json"))?;
        let scheme = config.get("quantization").and_then(|q| {
            Some(CheckpointScheme::Affine(QuantScheme {
                bits: q.get("bits")?.as_u64()? as u32,
                group_size: q.get("group_size")?.as_u64()? as usize,
            }))
        });
        // Full-model saves nest the backbone under `model.`; backbone-only
        // saves do not. The embed table exists in both, so it disambiguates.
        let prefix = if files
            .iter()
            .any(|f| f.contains_tensor("model.embed_tokens.weight"))
        {
            "model.".to_owned()
        } else {
            String::new()
        };
        Ok(Self {
            files,
            scheme,
            prefix,
            config,
            name_map: None,
        })
    }

    /// Builds checkpoint access from one already-opened checkpoint
    /// path (families that share the checkpoint file across their own
    /// submodules, like the indextts gpt2 block). No quantization
    /// scheme is assumed; `config()` stays empty.
    pub fn open_single(path: &Path) -> Result<Self> {
        let file = SafetensorsFile::open(path)?;
        Ok(Self {
            files: vec![file],
            scheme: None,
            prefix: String::new(),
            config: Value::Null,
            name_map: None,
        })
    }

    /// [`Self::open_single`] with a caller-parsed quantization scheme
    /// (the OmniVoice rowwise int8 format lives in `mlx_manifest.json`,
    /// not `config.json`). Quantized tensors without a matching scheme
    /// are refused, never guessed.
    pub fn open_single_with_scheme(path: &Path, scheme: Option<CheckpointScheme>) -> Result<Self> {
        let file = SafetensorsFile::open(path)?;
        Ok(Self {
            files: vec![file],
            scheme,
            prefix: String::new(),
            config: Value::Null,
            name_map: None,
        })
    }

    /// Sets the name prefix every tensor load prepends.
    pub fn with_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.prefix = prefix.into();
        self
    }

    /// Sets the opt-in tensor-name remap applied after the prefix on the
    /// prefixed loads (see the `name_map` field docs). Additive: families
    /// whose checkpoints already use the engine's names never call this.
    pub fn with_name_map(mut self, map: NameMap) -> Self {
        self.name_map = Some(map);
        self
    }

    fn map_name(&self, name: &str) -> String {
        match &self.name_map {
            Some(map) => map(name),
            None => name.to_owned(),
        }
    }

    pub fn config(&self) -> &Value {
        &self.config
    }

    /// The opened checkpoint shards, for families that load deeply
    /// nested submodule trees (codec decoders, diffusion heads) through
    /// their own tensor loaders instead of per-tensor absolute lookups.
    pub fn shards(&self) -> &[SafetensorsFile] {
        &self.files
    }

    /// True when `{prefix}{name}.weight` (or its scales) exists in a shard.
    pub fn contains(&self, name: &str) -> bool {
        self.find(name).is_some()
    }

    /// True when the absolute `{name}` (or `{name}.weight`) exists in a
    /// shard, ignoring the prefix. Used for family tensors that live
    /// outside the backbone prefix (`stop_predictor.*`).
    pub fn contains_absolute(&self, name: &str) -> bool {
        self.files
            .iter()
            .any(|f| f.contains_tensor(name) || f.contains_tensor(&format!("{name}.weight")))
    }

    /// Loads a plain tensor by its absolute base name, ignoring the
    /// prefix. Used for family tensors that live outside the backbone
    /// prefix; symmetric with [`Weights::linear_absolute`].
    pub fn tensor_absolute(&self, base: &str, shape: &[usize]) -> Result<Vec<f32>> {
        let index = self
            .files
            .iter()
            .position(|f| f.contains_tensor(base))
            .ok_or_else(|| SpeechError::Tensor {
                name: base.to_owned(),
                why: "tensor is missing".into(),
            })?;
        nn::load_tensor(&self.files[index], base, shape)
    }

    fn find(&self, name: &str) -> Option<usize> {
        let base = format!("{}{}", self.prefix, self.map_name(name));
        self.files.iter().position(|f| {
            f.contains_tensor(&format!("{base}.weight"))
                || f.contains_tensor(&format!("{base}.scales"))
        })
    }

    /// Loads one linear by its absolute base name, ignoring the prefix.
    /// The LM head sits outside `model.` in full-model saves.
    pub fn linear_absolute(
        &self,
        base: &str,
        input: usize,
        output: usize,
        bias: Bias,
    ) -> Result<Linear> {
        let (weight, bias_value) = self.load_weight_with_prefix("", base, input, output)?;
        let bias_value = match bias {
            Bias::None => None,
            Bias::Required => Some(bias_value.ok_or_else(|| SpeechError::Tensor {
                name: format!("{base}.bias"),
                why: "tensor is missing".into(),
            })?),
            Bias::Optional => bias_value,
        };
        Ok(Linear::new(weight, bias_value, input, output))
    }

    /// Loads one linear as `[out, in]` f32 (dequantizing groupwise-affine
    /// tensors when the checkpoint is quantized).
    pub fn linear(&self, name: &str, input: usize, output: usize, bias: Bias) -> Result<Linear> {
        let (weight, bias_value) =
            self.load_weight_with_prefix(&self.prefix, name, input, output)?;
        let bias_value = match bias {
            Bias::None => None,
            Bias::Required => Some(bias_value.ok_or_else(|| SpeechError::Tensor {
                name: format!("{}{}.bias", self.prefix, name),
                why: "tensor is missing".into(),
            })?),
            Bias::Optional => bias_value,
        };
        Ok(Linear::new(weight, bias_value, input, output))
    }

    /// Loads an embedding table `[rows, cols]` as flat f32.
    pub fn embedding(&self, name: &str, rows: usize, cols: usize) -> Result<Vec<f32>> {
        let (weight, _) = self.load_weight_with_prefix(&self.prefix, name, rows, cols)?;
        Ok(weight)
    }

    /// Shared weight load for linears and embedding tables: dequantizes
    /// groupwise-affine tensors when the checkpoint is quantized, plain
    /// f32/F16/BF16 otherwise, and validates the element count.
    fn load_weight_with_prefix(
        &self,
        prefix: &str,
        name: &str,
        rows: usize,
        cols: usize,
    ) -> Result<(Vec<f32>, Option<Vec<f32>>)> {
        let base = format!("{prefix}{}", self.map_name(name));
        let index = self
            .files
            .iter()
            .position(|f| {
                f.contains_tensor(&format!("{base}.weight"))
                    || f.contains_tensor(&format!("{base}.scales"))
            })
            .ok_or_else(|| SpeechError::Tensor {
                name: format!("{base}.weight"),
                why: "tensor is missing".into(),
            })?;
        let affine_quantized = self.files[index].contains_tensor(&format!("{base}.scales"));
        let rowwise_quantized = self.files[index].contains_tensor(&format!("{base}.weight.scales"));
        // load_quantized ignores the scheme for plain tensors; a quantized
        // tensor without a matching scheme is a refusal, not a guess.
        let (weight, bias) = match self.scheme {
            Some(CheckpointScheme::Affine(scheme)) => {
                if rowwise_quantized {
                    return Err(SpeechError::Tensor {
                        name: format!("{base}.weight.scales"),
                        why: "rowwise-quantized tensor but the checkpoint scheme is affine".into(),
                    });
                }
                quant::load_quantized(&self.files[index], &base, scheme)?
            }
            Some(CheckpointScheme::OmnivoiceRowwise) => {
                if affine_quantized {
                    return Err(SpeechError::Tensor {
                        name: format!("{base}.scales"),
                        why: "affine-quantized tensor but the checkpoint scheme is rowwise".into(),
                    });
                }
                if rowwise_quantized {
                    (
                        quant::load_rowwise_i8(&self.files[index], &format!("{base}.weight"))?,
                        None,
                    )
                } else {
                    // A plain layer inside the rowwise checkpoint (norms,
                    // biases): load it directly.
                    (
                        self.files[index].load_as_f32(&format!("{base}.weight"))?,
                        self.files[index]
                            .contains_tensor(&format!("{base}.bias"))
                            .then(|| self.files[index].load_as_f32(&format!("{base}.bias")))
                            .transpose()?,
                    )
                }
            }
            None if affine_quantized => {
                return Err(SpeechError::Tensor {
                    name: format!("{base}.scales"),
                    why: "quantized tensor but config.json has no quantization block".into(),
                });
            }
            None if rowwise_quantized => {
                return Err(SpeechError::Tensor {
                    name: format!("{base}.weight.scales"),
                    why: "rowwise-quantized tensor but no checkpoint scheme was parsed".into(),
                });
            }
            None => (
                self.files[index].load_as_f32(&format!("{base}.weight"))?,
                self.files[index]
                    .contains_tensor(&format!("{base}.bias"))
                    .then(|| self.files[index].load_as_f32(&format!("{base}.bias")))
                    .transpose()?,
            ),
        };
        if weight.len()
            != rows.checked_mul(cols).ok_or_else(|| SpeechError::Tensor {
                name: format!("{base}.weight"),
                why: "shape overflows".into(),
            })?
        {
            return Err(SpeechError::Tensor {
                name: format!("{base}.weight"),
                why: format!("expected {rows}x{cols}, got {} values", weight.len()),
            });
        }
        Ok((weight, bias))
    }

    /// Loads a plain tensor when present, `None` otherwise.
    pub fn tensor_optional(&self, name: &str, shape: &[usize]) -> Result<Option<Vec<f32>>> {
        let index = match self.find(name) {
            Some(index) => index,
            None => return Ok(None),
        };
        if !self.files[index].contains_tensor(&format!("{}{}.weight", self.prefix, name)) {
            return Ok(None);
        }
        Ok(Some(self.tensor(name, shape)?))
    }

    /// Loads a plain tensor with an exact shape.
    pub fn tensor(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        let base = format!("{}{}", self.prefix, self.map_name(name));
        let index = self.find(name).ok_or_else(|| SpeechError::Tensor {
            name: format!("{base}.weight"),
            why: "tensor is missing".into(),
        })?;
        nn::load_tensor(&self.files[index], &format!("{base}.weight"), shape)
    }

    /// Loads a bare (non-linear) tensor by exact name, prefix included.
    /// Used for parameters that are not `{name}.weight` pairs, like the
    /// bailing MoE router bias.
    pub fn raw_tensor(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        let base = format!("{}{}", self.prefix, name);
        let index = self
            .files
            .iter()
            .position(|f| f.contains_tensor(&base))
            .ok_or_else(|| SpeechError::Tensor {
                name: base.clone(),
                why: "tensor is missing".into(),
            })?;
        nn::load_tensor(&self.files[index], &base, shape)
    }
}
