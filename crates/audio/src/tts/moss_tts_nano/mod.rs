//! MOSS-TTS-Nano TTS family (OpenMOSS MOSS-TTS-Nano wrapper).
//!
//! Port of `mlx_audio/tts/models/moss_tts_nano/` at mlx-audio
//! [`e1b19b9054bf163f5d812221a54fcc346f1890e9`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/tts/models/moss_tts_nano):
//! a two-transformer architecture over per-channel VQ audio codes. A
//! global GPT-2-style transformer (`transformer`, rope positions, fused
//! biased QKV, gelu_new MLPs) consumes one embedding row per time step -
//! the text token embedding plus the sum of the per-channel audio
//! embeddings - and a one-layer local transformer predicts, per frame,
//! the continue/stop text token (argmax over the `audio_assistant_slot`
//! and `audio_end` embedding rows used as heads) and then one code per
//! VQ channel through the per-channel embedding heads. Frames decode
//! through the MOSS audio tokenizer (48 kHz deployed). The GPT-2 here is
//! the family-local `gpt2.py` variant (rope positions, learned absolute
//! optional, tanh GELU), which differs from the shared [`crate::lm`]
//! gpt2 backbone (learned positions only, no rope), so it is ported in
//! this module reusing the shared attention math, rope, and KV cache.
//! The `moss_tts` delay-pattern family shares none of this module; the
//! families stay separate per the inventory. See [README.md](README.md)
//! for the verification evidence and open items.
//!
//! Scope of this port, matching the reference single-prompt path: batch
//! size 1 (the reference refuses batches too), non-streaming decode, and
//! prompt construction from pre-tokenized ids. Text tokenization
//! (sentencepiece) and the token-budget sentence chunking of `text.py`
//! are caller concerns; [`lightweight_normalize_text`] and the prompt
//! id assembly helpers are ported.

use serde_json::Value;

use self::sampling as family_sampling;
use crate::codec::moss_audio_tokenizer::MossAudioTokenizer;
use crate::lm::cache::KvCache;
use crate::lm::engine::{split_fused_linear, Attention};
use crate::music::minimax_music3::rng;
use crate::nn::{json as cfg, LayerNorm, Linear};
use crate::ops;
use crate::{quant, Result, SpeechError};

mod sampling;

/// Reference audio tokenizer repo (`config.DEFAULT_AUDIO_TOKENIZER_REPO`).
pub const DEFAULT_AUDIO_TOKENIZER_REPO: &str = "mlx-community/MOSS-Audio-Tokenizer-Nano";

/// `config.GPT2Config` defaults the deployed checkpoints keep.
pub const DEFAULT_VOCAB_SIZE: usize = 16384;
pub const DEFAULT_N_EMBD: usize = 768;
pub const DEFAULT_N_INNER: usize = 3072;
pub const DEFAULT_ROPE_BASE: f32 = 10000.0;
/// `config.ModelConfig` special-token defaults.
pub const DEFAULT_AUDIO_PAD_TOKEN_ID: i32 = 1024;
pub const DEFAULT_PAD_TOKEN_ID: i32 = 3;
pub const DEFAULT_IM_START_TOKEN_ID: i32 = 4;
pub const DEFAULT_IM_END_TOKEN_ID: i32 = 5;
pub const DEFAULT_AUDIO_START_TOKEN_ID: i32 = 6;
pub const DEFAULT_AUDIO_END_TOKEN_ID: i32 = 7;
pub const DEFAULT_AUDIO_USER_SLOT_TOKEN_ID: i32 = 8;
pub const DEFAULT_AUDIO_ASSISTANT_SLOT_TOKEN_ID: i32 = 9;

/// `text.py` prompt template strings; the caller encodes them with the
/// checkpoint's sentencepiece tokenizer and hands the ids to the
/// assembly helpers below.
pub const USER_ROLE_PREFIX: &str = "user\n";
pub const USER_TEMPLATE_REFERENCE_PREFIX: &str = "<user_inst>\n- Reference(s):\n";
pub const USER_TEMPLATE_AFTER_REFERENCE: &str = "\n- Instruction:\nNone\n- Tokens:\nNone\n- Quality:\nNone\n- Sound Event:\nNone\n- Ambient Sound:\nNone\n- Language:\nNone\n- Text:\n";
pub const USER_TEMPLATE_SUFFIX: &str = "\n</user_inst>";
pub const ASSISTANT_TURN_PREFIX: &str = "\n";
pub const ASSISTANT_ROLE_PREFIX: &str = "assistant\n";

/// The family-local `config.GPT2Config` (the fields the runtime reads;
/// dropout probabilities and initializer ranges are training-only).
#[derive(Debug, Clone)]
pub struct Gpt2Config {
    pub vocab_size: usize,
    pub n_positions: usize,
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_inner: usize,
    pub activation_function: String,
    pub layer_norm_epsilon: f32,
    pub scale_attn_weights: bool,
    pub scale_attn_by_inverse_layer_idx: bool,
    pub position_embedding_type: String,
    pub rope_base: f32,
}

impl Gpt2Config {
    /// `GPT2Config.from_dict`: the HF alias fields (`hidden_size`,
    /// `num_hidden_layers`, `num_attention_heads`, `intermediate_size`)
    /// feed the gpt2 names when those are absent, and every field keeps
    /// the dataclass default.
    pub fn from_json(value: &Value) -> Result<Self> {
        let u = |key: &str| -> Result<usize> { cfg::usize_field(value, key) };
        let u_or = |key: &str, default: usize| {
            value
                .get(key)
                .and_then(Value::as_u64)
                .map(|n| n as usize)
                .unwrap_or(default)
        };
        let n_embd = if value.get("n_embd").is_some() {
            u("n_embd")?
        } else {
            u_or("hidden_size", DEFAULT_N_EMBD)
        };
        let n_layer = if value.get("n_layer").is_some() {
            u("n_layer")?
        } else {
            u_or("num_hidden_layers", 12)
        };
        let n_head = if value.get("n_head").is_some() {
            u("n_head")?
        } else {
            u_or("num_attention_heads", 12)
        };
        let n_inner = if value.get("n_inner").is_some() {
            u("n_inner")?
        } else {
            u_or("intermediate_size", DEFAULT_N_INNER)
        };
        // Refuse here, before any `n_embd % n_head` or head_dim division:
        // a zero head count would otherwise panic on the first stack load.
        if n_head == 0 || n_embd % n_head != 0 {
            return Err(SpeechError::BadConfig {
                field: "n_head".into(),
                why: format!("n_head {n_head} must be non-zero and divide n_embd {n_embd}"),
            });
        }
        let position_embedding_type = value
            .get("position_embedding_type")
            .and_then(Value::as_str)
            .unwrap_or("rope")
            .to_owned();
        if position_embedding_type != "rope" {
            return Err(SpeechError::Unsupported {
                why: format!(
                    "position_embedding_type {position_embedding_type}: only the rope path is ported (no shipped checkpoint uses learned positions)"
                ),
            });
        }
        Ok(Self {
            vocab_size: u_or("vocab_size", DEFAULT_VOCAB_SIZE),
            n_positions: u_or("n_positions", 32768),
            n_embd,
            n_layer,
            n_head,
            n_inner,
            activation_function: value
                .get("activation_function")
                .and_then(Value::as_str)
                .unwrap_or("gelu_new")
                .to_owned(),
            layer_norm_epsilon: value
                .get("layer_norm_epsilon")
                .and_then(Value::as_f64)
                .unwrap_or(1e-5) as f32,
            scale_attn_weights: value
                .get("scale_attn_weights")
                .and_then(Value::as_bool)
                .unwrap_or(true),
            scale_attn_by_inverse_layer_idx: value
                .get("scale_attn_by_inverse_layer_idx")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            position_embedding_type,
            rope_base: value
                .get("rope_base")
                .and_then(Value::as_f64)
                .unwrap_or(DEFAULT_ROPE_BASE as f64) as f32,
        })
    }
}

/// The TTS `ModelConfig` (`config.ModelConfig.from_dict`).
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub gpt2_config: Gpt2Config,
    pub n_vq: usize,
    pub audio_vocab_size: usize,
    pub audio_codebook_sizes: Vec<usize>,
    pub audio_pad_token_id: i32,
    pub pad_token_id: i32,
    pub im_start_token_id: i32,
    pub im_end_token_id: i32,
    pub audio_start_token_id: i32,
    pub audio_end_token_id: i32,
    pub audio_user_slot_token_id: i32,
    pub audio_assistant_slot_token_id: i32,
    pub audio_tokenizer_sample_rate: u32,
    pub local_transformer_layers: usize,
}

impl ModelConfig {
    pub fn from_json(value: &Value) -> Result<Self> {
        let gpt2_config = Gpt2Config::from_json(value.get("gpt2_config").unwrap_or(&Value::Null))?;
        let n_vq = value.get("n_vq").and_then(Value::as_u64).unwrap_or(16) as usize;
        // Zero channels would make the per-channel tables and the frame
        // width degenerate (and `n_vq + 1` context math assumes at least one).
        if n_vq == 0 {
            return Err(SpeechError::BadConfig {
                field: "n_vq".into(),
                why: "must be at least 1".into(),
            });
        }
        let audio_vocab_size = value
            .get("audio_vocab_size")
            .and_then(Value::as_u64)
            .unwrap_or(1024) as usize;
        let audio_codebook_sizes = match value.get("audio_codebook_sizes").and_then(Value::as_array)
        {
            Some(list) => {
                let sizes: Vec<usize> = list
                    .iter()
                    .map(|v| v.as_u64().map(|n| n as usize))
                    .collect::<Option<Vec<usize>>>()
                    .ok_or_else(|| SpeechError::BadConfig {
                        field: "audio_codebook_sizes".into(),
                        why: "must be a list of integers".into(),
                    })?;
                if sizes.len() != n_vq {
                    return Err(SpeechError::BadConfig {
                        field: "audio_codebook_sizes".into(),
                        why: format!(
                            "must have one entry per VQ channel (expected {n_vq}, got {})",
                            sizes.len()
                        ),
                    });
                }
                sizes
            }
            None => vec![audio_vocab_size; n_vq],
        };
        let i = |key: &str, default: i32| {
            value
                .get(key)
                .and_then(Value::as_i64)
                .map(|n| n as i32)
                .unwrap_or(default)
        };
        Ok(Self {
            gpt2_config,
            n_vq,
            audio_vocab_size,
            audio_codebook_sizes,
            audio_pad_token_id: i("audio_pad_token_id", DEFAULT_AUDIO_PAD_TOKEN_ID),
            pad_token_id: i("pad_token_id", DEFAULT_PAD_TOKEN_ID),
            im_start_token_id: i("im_start_token_id", DEFAULT_IM_START_TOKEN_ID),
            im_end_token_id: i("im_end_token_id", DEFAULT_IM_END_TOKEN_ID),
            audio_start_token_id: i("audio_start_token_id", DEFAULT_AUDIO_START_TOKEN_ID),
            audio_end_token_id: i("audio_end_token_id", DEFAULT_AUDIO_END_TOKEN_ID),
            audio_user_slot_token_id: i(
                "audio_user_slot_token_id",
                DEFAULT_AUDIO_USER_SLOT_TOKEN_ID,
            ),
            audio_assistant_slot_token_id: i(
                "audio_assistant_slot_token_id",
                DEFAULT_AUDIO_ASSISTANT_SLOT_TOKEN_ID,
            ),
            audio_tokenizer_sample_rate: value
                .get("audio_tokenizer_sample_rate")
                .and_then(Value::as_u64)
                .unwrap_or(48000) as u32,
            local_transformer_layers: value
                .get("local_transformer_layers")
                .and_then(Value::as_u64)
                .unwrap_or(1) as usize,
        })
    }

    /// The local transformer configuration
    /// (`ModelConfig.local_gpt2_config`): no token embedding positions,
    /// one context of `n_vq + 1` positions, the global depth otherwise.
    pub fn local_gpt2_config(&self) -> Gpt2Config {
        let mut config = self.gpt2_config.clone();
        config.n_positions = self.n_vq + 1;
        config.n_layer = self.local_transformer_layers;
        config
    }
}

/// Tanh GELU (`gpt2.gelu_new`), exact GELU, or SiLU, by config name.
fn apply_activation(name: &str, values: &mut [f32]) -> Result<()> {
    match name {
        "gelu_new" => ops::gelu_tanh(values),
        "gelu" => ops::gelu_erf(values),
        "silu" => {
            for v in values.iter_mut() {
                *v *= ops::sigmoid(*v);
            }
        }
        other => {
            return Err(SpeechError::Unsupported {
                why: format!("activation_function {other}"),
            })
        }
    }
    Ok(())
}

/// The family-local `gpt2.MLP`: `fc_out(activation(fc_in(x)))`, both
/// linears biased.
struct Mlp {
    fc_in: Linear,
    fc_out: Linear,
    activation: String,
}

impl Mlp {
    fn forward(&self, x: &[f32], rows: usize) -> Result<Vec<f32>> {
        let mut hidden = self.fc_in.forward(x, rows);
        apply_activation(&self.activation, &mut hidden)?;
        Ok(self.fc_out.forward(&hidden, rows))
    }
}

struct Block {
    ln_1: LayerNorm,
    attn: Attention,
    ln_2: LayerNorm,
    mlp: Mlp,
}

/// The family-local `gpt2.GPT2Model`: optional token embedding (the
/// global transformer has one, the local one does not), rope positions,
/// LayerNorm pre-norm blocks, and the final LayerNorm.
pub(crate) struct Gpt2 {
    pub(crate) config: Gpt2Config,
    pub(crate) wte: Option<Vec<f32>>,
    blocks: Vec<Block>,
    ln_f: LayerNorm,
}

impl Gpt2 {
    /// Loads one stack under `name` (`transformer` or
    /// `local_transformer`). `use_token_embedding` mirrors the reference
    /// constructor flag (the local transformer's `wte` is dropped by the
    /// checkpoint sanitize).
    fn load(
        weights: &crate::lm::Weights,
        name: &str,
        config: &Gpt2Config,
        use_token_embedding: bool,
    ) -> Result<Self> {
        if config.n_embd % config.n_head != 0 {
            return Err(SpeechError::BadConfig {
                field: "n_embd".into(),
                why: format!(
                    "{} not divisible by n_head {}",
                    config.n_embd, config.n_head
                ),
            });
        }
        if config.position_embedding_type != "rope" {
            return Err(SpeechError::Unsupported {
                why: format!(
                    "position_embedding_type {}: only rope is ported",
                    config.position_embedding_type
                ),
            });
        }
        // `RotaryEmbedding(head_dim, rope_base)`: adjacent-pair rotation
        // with `angle = position * base ** (-2 d / dim)`, which is the
        // shared rope's plain traditional construction.
        let rope = crate::lm::rope::Rope::initialize(
            config.n_embd / config.n_head,
            config.rope_base,
            true,
            None,
            None,
        )?;
        let head_dim = config.n_embd / config.n_head;
        let base_scale = if config.scale_attn_weights {
            (head_dim as f64).powf(-0.5) as f32
        } else {
            1.0
        };
        // No pre-sizing: `n_layer` is an untrusted config value and the loop
        // below fails on the first missing tensor, so a huge count cannot
        // reach an allocation.
        let mut blocks = Vec::new();
        for index in 0..config.n_layer {
            let layer = format!("{name}.h.{index}.");
            let fused = weights.linear(
                &format!("{layer}attn.c_attn"),
                config.n_embd,
                3 * config.n_embd,
                crate::lm::Bias::Required,
            )?;
            // mx.split(qkv, 3, axis=-1): equal consecutive output ranges.
            let (q_proj, k_proj, v_proj) = split_fused_linear(&fused, config.n_embd, config.n_embd);
            let layer_scale = if config.scale_attn_by_inverse_layer_idx {
                base_scale / (index + 1) as f32
            } else {
                base_scale
            };
            let attn = Attention {
                q_proj,
                k_proj,
                v_proj,
                o_proj: weights.linear(
                    &format!("{layer}attn.c_proj"),
                    config.n_embd,
                    config.n_embd,
                    crate::lm::Bias::Required,
                )?,
                q_norm: None,
                k_norm: None,
                n_heads: config.n_head,
                n_kv_heads: config.n_head,
                head_dim,
                scale: layer_scale,
                rope: Some(rope.clone()),
                window: None,
                bidirectional: false,
                exec: None,
            };
            let mlp = Mlp {
                fc_in: weights.linear(
                    &format!("{layer}mlp.fc_in"),
                    config.n_embd,
                    config.n_inner,
                    crate::lm::Bias::Required,
                )?,
                fc_out: weights.linear(
                    &format!("{layer}mlp.fc_out"),
                    config.n_inner,
                    config.n_embd,
                    crate::lm::Bias::Required,
                )?,
                activation: config.activation_function.clone(),
            };
            blocks.push(Block {
                ln_1: load_layer_norm(
                    weights,
                    &format!("{layer}ln_1"),
                    config.n_embd,
                    config.layer_norm_epsilon,
                )?,
                attn,
                ln_2: load_layer_norm(
                    weights,
                    &format!("{layer}ln_2"),
                    config.n_embd,
                    config.layer_norm_epsilon,
                )?,
                mlp,
            });
        }
        Ok(Self {
            config: config.clone(),
            wte: if use_token_embedding {
                Some(weights.embedding(&format!("{name}.wte"), config.vocab_size, config.n_embd)?)
            } else {
                None
            },
            blocks,
            ln_f: load_layer_norm(
                weights,
                &format!("{name}.ln_f"),
                config.n_embd,
                config.layer_norm_epsilon,
            )?,
        })
    }

    /// Runs the stack over row-major embeddings `[rows, n_embd]`;
    /// `caches` advances when present (the global transformer), and the
    /// causal mask is the additive mask of the reference with an
    /// all-true padding mask (batch 1, unpadded). Returns the post
    /// `ln_f` hidden states.
    pub(crate) fn forward(
        &mut self,
        embeds: &[f32],
        mut caches: Option<&mut Vec<KvCache>>,
    ) -> Result<Vec<f32>> {
        let width = self.config.n_embd;
        let rows = embeds.len() / width;
        if rows == 0 {
            return Err(SpeechError::Input {
                why: "empty transformer input".into(),
            });
        }
        if let Some(caches) = caches.as_deref() {
            if caches.len() != self.blocks.len() {
                return Err(SpeechError::Input {
                    why: format!(
                        "{} layer caches for {} layers",
                        caches.len(),
                        self.blocks.len()
                    ),
                });
            }
        }
        let offset = caches
            .as_deref()
            .and_then(|c| c.first())
            .map(|c| c.offset)
            .unwrap_or(0);
        let mut h = embeds.to_vec();
        for (index, block) in self.blocks.iter_mut().enumerate() {
            let mut attended = h.clone();
            block.ln_1.apply(&mut attended, rows);
            let attn = block.attn.forward(
                &attended,
                rows,
                offset,
                caches
                    .as_deref_mut()
                    .map(|c| &mut c[index] as &mut dyn crate::lm::cache::LayerKv),
            )?;
            for (row, out) in attn.chunks_exact(width).enumerate() {
                let base = row * width;
                for (col, value) in out.iter().enumerate() {
                    h[base + col] += value;
                }
            }
            let mut fed = h.clone();
            block.ln_2.apply(&mut fed, rows);
            let mlp = block.mlp.forward(&fed, rows)?;
            for (row, out) in mlp.chunks_exact(width).enumerate() {
                let base = row * width;
                for (col, value) in out.iter().enumerate() {
                    h[base + col] += value;
                }
            }
        }
        self.ln_f.apply(&mut h, rows);
        Ok(h)
    }
}

fn load_layer_norm(
    weights: &crate::lm::Weights,
    name: &str,
    width: usize,
    epsilon: f32,
) -> Result<LayerNorm> {
    let weight = weights.tensor(name, &[width])?;
    let bias = weights.raw_tensor(&format!("{name}.bias"), &[width])?;
    Ok(LayerNorm::new(weight, Some(bias), epsilon))
}

/// Why the frame loop stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// The continue/stop text token was not the assistant slot.
    EndOfText,
    /// `max_new_frames` frames were generated.
    MaxFrames,
}

/// Sampling and generation options with the reference
/// `generate_audio_token_ids` defaults.
#[derive(Debug, Clone)]
pub struct GenerateParams {
    /// `None` resolves to `n_vq` (`_resolve_nq`).
    pub nq: Option<usize>,
    pub max_new_frames: usize,
    pub do_sample: bool,
    pub text_temperature: f32,
    pub text_top_p: f32,
    pub text_top_k: usize,
    pub audio_temperature: f32,
    pub audio_top_p: f32,
    pub audio_top_k: usize,
    pub audio_repetition_penalty: f32,
    pub use_kv_cache: bool,
}

impl Default for GenerateParams {
    fn default() -> Self {
        Self {
            nq: None,
            max_new_frames: 375,
            do_sample: true,
            text_temperature: 1.0,
            text_top_p: 1.0,
            text_top_k: 50,
            audio_temperature: 0.8,
            audio_top_p: 0.95,
            audio_top_k: 25,
            audio_repetition_penalty: 1.2,
            use_kv_cache: true,
        }
    }
}

/// One generated prompt: the frames (`n_vq` wide, audio-pad padded), the
/// codebook-major codes the decode consumes, and the decoded waveform.
#[derive(Debug, Clone)]
pub struct Generated {
    pub frames: Vec<Vec<i32>>,
    /// `[nq][T]`, the first `effective_nq` books of the frame transpose
    /// (the decode drops audio-pad filler channels).
    pub codes: Vec<Vec<i32>>,
    pub samples: Vec<f32>,
    pub stopped_by: StopReason,
}

/// The prompt id assembly of `text.py` with caller-owned tokenization.
/// `im_start` and the encoded template pieces come from the checkpoint's
/// sentencepiece tokenizer; the returned ids compose exactly like the
/// reference builders.
pub fn build_user_prompt_prefix(
    im_start_token_id: i32,
    role_prefix_ids: &[i32],
    reference_prefix_ids: &[i32],
) -> Vec<i32> {
    let mut ids = Vec::with_capacity(1 + role_prefix_ids.len() + reference_prefix_ids.len());
    ids.push(im_start_token_id);
    ids.extend_from_slice(role_prefix_ids);
    ids.extend_from_slice(reference_prefix_ids);
    ids
}

pub fn build_user_prompt_after_reference(after_reference_ids: &[i32]) -> Vec<i32> {
    after_reference_ids.to_vec()
}

pub fn build_assistant_prompt_prefix(
    suffix_ids: &[i32],
    im_end_token_id: i32,
    turn_prefix_ids: &[i32],
    im_start_token_id: i32,
    role_prefix_ids: &[i32],
) -> Vec<i32> {
    let mut ids = Vec::with_capacity(
        suffix_ids.len() + 1 + turn_prefix_ids.len() + 1 + role_prefix_ids.len(),
    );
    ids.extend_from_slice(suffix_ids);
    ids.push(im_end_token_id);
    ids.extend_from_slice(turn_prefix_ids);
    ids.push(im_start_token_id);
    ids.extend_from_slice(role_prefix_ids);
    ids
}

/// `build_prompt_token_ids` (the continuation prompt): the user prefix,
/// the encoded `"None"` reference, the after-reference template, the
/// text, and the assistant prefix.
pub fn build_prompt_token_ids(
    prefix_ids: &[i32],
    none_reference_ids: &[i32],
    after_reference_ids: &[i32],
    text_ids: &[i32],
    assistant_prefix_ids: &[i32],
) -> Vec<i32> {
    let mut ids = Vec::with_capacity(
        prefix_ids.len()
            + none_reference_ids.len()
            + after_reference_ids.len()
            + text_ids.len()
            + assistant_prefix_ids.len(),
    );
    ids.extend_from_slice(prefix_ids);
    ids.extend_from_slice(none_reference_ids);
    ids.extend_from_slice(after_reference_ids);
    ids.extend_from_slice(text_ids);
    ids.extend_from_slice(assistant_prefix_ids);
    ids
}

/// `lightweight_normalize_text`: strip, map CR/LF to spaces, and
/// collapse whitespace runs.
pub fn lightweight_normalize_text(text: &str) -> String {
    text.split_whitespace().collect::<Vec<&str>>().join(" ")
}

/// The loaded family model.
pub struct Model {
    pub config: ModelConfig,
    transformer: Gpt2,
    local_transformer: Gpt2,
    audio_embeddings: Vec<Vec<f32>>,
}

impl Model {
    /// Opens a checkpoint directory (`config.json` + safetensors).
    pub fn open(dir: &std::path::Path) -> Result<Self> {
        let config_value = quant::read_json(&dir.join("config.json"))?;
        let weights = crate::lm::Weights::open(dir)?;
        Self::load(&weights, &config_value)
    }

    /// Loads from opened weights and the raw config dict. Tensor names
    /// are the checkpoint's own (`transformer.*`, `local_transformer.*`,
    /// `audio_embeddings.{channel}`); the reference sanitize drops the
    /// redundant `text_lm_head`/`audio_lm_heads` (tied to the embedding
    /// tables used as heads here), the local `wte`, and the `wpe`
    /// tables of rope configurations.
    pub fn load(weights: &crate::lm::Weights, config: &Value) -> Result<Self> {
        let config = ModelConfig::from_json(config)?;
        let global_config = config.gpt2_config.clone();
        let local_config = config.local_gpt2_config();
        let transformer = Gpt2::load(weights, "transformer", &global_config, true)?;
        let local_transformer = Gpt2::load(weights, "local_transformer", &local_config, false)?;
        let mut audio_embeddings = Vec::with_capacity(config.n_vq);
        for (channel, size) in config.audio_codebook_sizes.iter().enumerate() {
            audio_embeddings.push(weights.embedding(
                &format!("audio_embeddings.{channel}"),
                *size,
                config.gpt2_config.n_embd,
            )?);
        }
        Ok(Self {
            config,
            transformer,
            local_transformer,
            audio_embeddings,
        })
    }

    /// Reference `_build_inputs_embeds`: the text embedding of column 0
    /// plus the masked sum of the per-channel audio embeddings (audio-pad
    /// columns contribute zero; out-of-range codes are refused where the
    /// reference embedding lookup would raise).
    fn build_inputs_embeds(&self, rows: &[Vec<i32>]) -> Result<Vec<f32>> {
        let wte = self
            .transformer
            .wte
            .as_ref()
            .ok_or_else(|| SpeechError::Input {
                why: "the global transformer has no token embedding".into(),
            })?;
        let width = self.config.n_vq + 1;
        let embd = self.config.gpt2_config.n_embd;
        let mut out = vec![0.0f32; rows.len() * embd];
        for (index, row) in rows.iter().enumerate() {
            if row.len() != width {
                return Err(SpeechError::Input {
                    why: format!(
                        "expected input_ids shape [batch, seq, {}], got rows of {}",
                        width,
                        row.len()
                    ),
                });
            }
            let base = index * embd;
            let text = lookup_row(wte, self.config.gpt2_config.vocab_size, embd, row[0])?;
            out[base..base + embd].copy_from_slice(text);
            for channel in 0..self.config.n_vq {
                let id = row[channel + 1];
                if id == self.config.audio_pad_token_id {
                    continue;
                }
                let size = self.config.audio_codebook_sizes[channel];
                if id < 0 || id as usize >= size {
                    return Err(SpeechError::Input {
                        why: format!("channel {channel} code {id} outside codebook size {size}"),
                    });
                }
                let embed =
                    &self.audio_embeddings[channel][id as usize * embd..(id as usize + 1) * embd];
                for (col, value) in embed.iter().enumerate() {
                    out[base + col] += value;
                }
            }
        }
        Ok(out)
    }

    /// Reference `_text_lm_head` / `_audio_lm_head`: the embedding tables
    /// are the heads (`hidden @ table.T`).
    fn head_logits(table: &[f32], vocab: usize, embd: usize, hidden: &[f32]) -> Vec<f32> {
        ops::linear(hidden, table, None, 1, embd, vocab)
    }

    /// Reference `generate_audio_token_ids` for batch 1: the global
    /// transformer (cache stepped once per frame) proposes the
    /// continue/stop token through the local transformer and the tied
    /// text head; while it is the assistant slot, the local transformer
    /// predicts one code per channel through the per-channel embedding
    /// heads with the sign-aware repetition penalty over each channel's
    /// full history. Returns the generated frames and why the loop
    /// stopped; decoding runs through [`Model::generate`] or the caller's
    /// [`MossAudioTokenizer`].
    pub fn generate_audio_token_ids(
        &mut self,
        prompt_rows: &[Vec<i32>],
        params: &GenerateParams,
        seed: u64,
    ) -> Result<(Vec<Vec<i32>>, StopReason)> {
        let width = self.config.n_vq + 1;
        if prompt_rows.is_empty() {
            return Err(SpeechError::Input {
                why: "empty prompt".into(),
            });
        }
        if prompt_rows.iter().any(|row| row.len() != width) {
            return Err(SpeechError::Input {
                why: format!("prompt rows must be [{width}] wide"),
            });
        }
        let effective_nq = self.resolve_nq(params.nq)?;
        let embd = self.config.gpt2_config.n_embd;
        let vocab = self.config.gpt2_config.vocab_size;
        let assistant_slot = self.config.audio_assistant_slot_token_id;
        let audio_end = self.config.audio_end_token_id;
        let mut keys = rng::KeySequence::new(seed);

        // `use_kv_cache`: the cache steps one row per frame; without it
        // every frame replays the whole prompt plus generated rows
        // cacheless.
        let mut caches = if params.use_kv_cache {
            Some(crate::lm::make_prompt_cache(
                self.config.gpt2_config.n_layer,
            ))
        } else {
            None
        };
        let mut history_rows: Vec<Vec<i32>> = prompt_rows.to_vec();
        let mut generated: Vec<Vec<i32>> = Vec::new();
        let mut stopped_by = StopReason::MaxFrames;

        for _ in 0..params.max_new_frames {
            let embeds = self.build_inputs_embeds(&history_rows)?;
            let global_hidden = match caches.as_mut() {
                Some(caches) => self.transformer.forward(&embeds, Some(caches))?,
                None => self.transformer.forward(&embeds, None)?,
            };
            let global_last = &global_hidden[(global_hidden.len() / embd - 1) * embd..];

            // Continue/stop choice over the two candidate embeddings
            // (`sample_assistant_text_token`, top-k capped at 2).
            let local_hidden = self.local_transformer.forward(global_last, None)?;
            let local_last = &local_hidden[local_hidden.len() - embd..];
            let text_logits = Self::head_logits(
                self.transformer
                    .wte
                    .as_ref()
                    .ok_or_else(|| SpeechError::Input {
                        why: "the global transformer has no token embedding".into(),
                    })?,
                vocab,
                embd,
                local_last,
            );
            let sampled = family_sampling::sample_assistant_text_token(
                &text_logits,
                assistant_slot,
                audio_end,
                params.do_sample,
                params.text_temperature,
                Some(params.text_top_k),
                Some(params.text_top_p),
                keys.next(),
            )?;
            let next_text_token = [assistant_slot, audio_end][sampled];
            if next_text_token != assistant_slot {
                stopped_by = StopReason::EndOfText;
                break;
            }

            // One code per channel: the local transformer replays
            // [global hidden, previous channel embedding, ...] cacheless.
            let wte = self
                .transformer
                .wte
                .as_ref()
                .ok_or_else(|| SpeechError::Input {
                    why: "the global transformer has no token embedding".into(),
                })?;
            let mut current_local_input = lookup_row(wte, vocab, embd, next_text_token)?.to_vec();
            let mut local_rows: Vec<f32> = global_last.to_vec();
            let mut frame = Vec::with_capacity(effective_nq);
            for channel in 0..effective_nq {
                local_rows.extend_from_slice(&current_local_input);
                let local_hidden = self.local_transformer.forward(&local_rows, None)?;
                let local_last = &local_hidden[local_hidden.len() - embd..];
                let size = self.config.audio_codebook_sizes[channel];
                let channel_logits =
                    Self::head_logits(&self.audio_embeddings[channel], size, embd, local_last);
                let previous: Vec<i32> = generated.iter().map(|f| f[channel]).collect();
                let token = family_sampling::sample_next_token(
                    &channel_logits,
                    params.do_sample,
                    params.audio_temperature,
                    Some(params.audio_top_k),
                    Some(params.audio_top_p),
                    if generated.is_empty() {
                        None
                    } else {
                        Some(&previous)
                    },
                    params.audio_repetition_penalty,
                    keys.next(),
                )?;
                frame.push(token as i32);
                current_local_input =
                    self.audio_embeddings[channel][token * embd..(token + 1) * embd].to_vec();
            }

            // Pad the frame to n_vq with the audio-pad id.
            frame.resize(self.config.n_vq, self.config.audio_pad_token_id);
            generated.push(frame.clone());

            let mut next_row = Vec::with_capacity(width);
            next_row.push(assistant_slot);
            next_row.extend_from_slice(&frame);
            if params.use_kv_cache {
                history_rows = vec![next_row];
            } else {
                history_rows.push(next_row);
            }
        }
        Ok((generated, stopped_by))
    }

    /// Reference `_resolve_nq`.
    pub fn resolve_nq(&self, nq: Option<usize>) -> Result<usize> {
        match nq {
            None => Ok(self.config.n_vq),
            Some(nq) => {
                if nq < 1 || nq > self.config.n_vq {
                    return Err(SpeechError::Input {
                        why: format!("nq must be in [1, {}], got {nq}", self.config.n_vq),
                    });
                }
                Ok(nq)
            }
        }
    }

    /// Reference `build_inference_input_ids` for batch 1: the composed
    /// prompt rows (`n_vq + 1` wide) and the all-true attention mask the
    /// port drops (single unpadded batch). `text_ids` is the encoded
    /// chunk; `prompt_text_ids` is the encoded `ref_text` (continuation
    /// mode); `prompt_codes` is `[frames][<= n_vq]` reference audio.
    /// `template` carries the caller-encoded template strings of
    /// [`USER_ROLE_PREFIX`] and friends (the checkpoint's sentencepiece
    /// tokenizer owns that encoding).
    pub fn build_inference_input_ids(
        &self,
        mode: PromptMode,
        template: &TemplateIds,
        text_ids: &[i32],
        prompt_text_ids: Option<&[i32]>,
        prompt_codes: Option<&[Vec<i32>]>,
    ) -> Result<Vec<Vec<i32>>> {
        let mut rows: Vec<Vec<i32>> = Vec::new();
        match mode {
            PromptMode::VoiceClone => {
                let codes = prompt_codes.ok_or_else(|| SpeechError::Input {
                    why: "voice_clone mode requires prompt_codes".into(),
                })?;
                if prompt_text_ids.is_some() {
                    return Err(SpeechError::Input {
                        why: "voice_clone mode does not accept prompt_text".into(),
                    });
                }
                let mut prefix_ids = build_user_prompt_prefix(
                    self.config.im_start_token_id,
                    template.user_role_prefix,
                    template.user_template_reference_prefix,
                );
                prefix_ids.push(self.config.audio_start_token_id);
                rows.extend(text_rows(
                    &prefix_ids,
                    self.config.n_vq,
                    self.config.audio_pad_token_id,
                ));
                rows.extend(self.audio_prefix_rows(codes, self.config.audio_user_slot_token_id)?);
                let mut suffix_ids = vec![self.config.audio_end_token_id];
                suffix_ids.extend(build_user_prompt_after_reference(
                    template.user_template_after_reference,
                ));
                suffix_ids.extend_from_slice(text_ids);
                suffix_ids.extend(build_assistant_prompt_prefix(
                    template.user_template_suffix,
                    self.config.im_end_token_id,
                    template.assistant_turn_prefix,
                    self.config.im_start_token_id,
                    template.assistant_role_prefix,
                ));
                suffix_ids.push(self.config.audio_start_token_id);
                rows.extend(text_rows(
                    &suffix_ids,
                    self.config.n_vq,
                    self.config.audio_pad_token_id,
                ));
            }
            PromptMode::Continuation => {
                if prompt_text_ids.is_none() != prompt_codes.is_none() {
                    return Err(SpeechError::Input {
                        why: "continuation mode accepts target text only, or both prompt_text and prompt_codes".into(),
                    });
                }
                let mut combined = Vec::new();
                if let Some(prompt_text_ids) = prompt_text_ids {
                    combined.extend_from_slice(prompt_text_ids);
                }
                combined.extend_from_slice(text_ids);
                let prompt_ids = build_prompt_token_ids(
                    &build_user_prompt_prefix(
                        self.config.im_start_token_id,
                        template.user_role_prefix,
                        template.user_template_reference_prefix,
                    ),
                    template.none_reference,
                    template.user_template_after_reference,
                    &combined,
                    &build_assistant_prompt_prefix(
                        template.user_template_suffix,
                        self.config.im_end_token_id,
                        template.assistant_turn_prefix,
                        self.config.im_start_token_id,
                        template.assistant_role_prefix,
                    ),
                );
                rows.extend(text_rows(
                    &prompt_ids,
                    self.config.n_vq,
                    self.config.audio_pad_token_id,
                ));
                rows.extend(text_rows(
                    &[self.config.audio_start_token_id],
                    self.config.n_vq,
                    self.config.audio_pad_token_id,
                ));
                if let Some(codes) = prompt_codes {
                    rows.extend(
                        self.audio_prefix_rows(codes, self.config.audio_assistant_slot_token_id)?,
                    );
                }
            }
        }
        Ok(rows)
    }

    /// Reference `_build_audio_prefix_rows`: all-audio-pad rows with the
    /// slot token in column 0 and up to `n_vq` reference codes filling
    /// the audio channels.
    fn audio_prefix_rows(&self, codes: &[Vec<i32>], slot_token_id: i32) -> Result<Vec<Vec<i32>>> {
        let width = self.config.n_vq + 1;
        let mut rows = Vec::with_capacity(codes.len());
        for frame in codes {
            if frame.len() > self.config.n_vq {
                return Err(SpeechError::Input {
                    why: format!(
                        "prompt_audio_codes must have shape [frames, n_vq], got rows of {}",
                        frame.len()
                    ),
                });
            }
            let mut row = vec![self.config.audio_pad_token_id; width];
            row[0] = slot_token_id;
            for (channel, &code) in frame.iter().enumerate() {
                row[channel + 1] = code;
            }
            rows.push(row);
        }
        Ok(rows)
    }

    /// Runs one prompt end to end: [`Model::generate_audio_token_ids`]
    /// then the MOSS audio tokenizer decode of the first `effective_nq`
    /// codebooks (the reference `decode_audio_token_ids`; the audio-pad
    /// filler channels beyond `effective_nq` are not decoded).
    /// `do_sample=False` is fully deterministic; `seed` only feeds the
    /// categorical draws.
    pub fn generate(
        &mut self,
        tokenizer: &MossAudioTokenizer,
        prompt_rows: &[Vec<i32>],
        params: &GenerateParams,
        seed: u64,
    ) -> Result<Generated> {
        let (frames, stopped_by) = self.generate_audio_token_ids(prompt_rows, params, seed)?;
        let effective_nq = self.resolve_nq(params.nq)?;
        let codes = frames_to_codes(&frames, effective_nq);
        let samples = if frames.is_empty() {
            Vec::new()
        } else {
            tokenizer.decode(&codes)?
        };
        Ok(Generated {
            frames,
            codes,
            samples,
            stopped_by,
        })
    }
}

/// The caller-encoded `text.py` template pieces
/// (`build_user_prompt_prefix` and friends take these ids).
#[derive(Debug, Clone, Copy)]
pub struct TemplateIds<'a> {
    /// `encode(USER_ROLE_PREFIX)`.
    pub user_role_prefix: &'a [i32],
    /// `encode(USER_TEMPLATE_REFERENCE_PREFIX)`.
    pub user_template_reference_prefix: &'a [i32],
    /// `encode(USER_TEMPLATE_AFTER_REFERENCE)`.
    pub user_template_after_reference: &'a [i32],
    /// `encode(USER_TEMPLATE_SUFFIX)`.
    pub user_template_suffix: &'a [i32],
    /// `encode(ASSISTANT_TURN_PREFIX)`.
    pub assistant_turn_prefix: &'a [i32],
    /// `encode(ASSISTANT_ROLE_PREFIX)`.
    pub assistant_role_prefix: &'a [i32],
    /// `encode("None")`, the continuation reference stand-in.
    pub none_reference: &'a [i32],
}

/// Reference `_build_text_rows`: all-audio-pad rows with the text token
/// in column 0.
fn text_rows(token_ids: &[i32], n_vq: usize, audio_pad_token_id: i32) -> Vec<Vec<i32>> {
    token_ids
        .iter()
        .map(|&id| {
            let mut row = vec![audio_pad_token_id; n_vq + 1];
            row[0] = id;
            row
        })
        .collect()
}

/// Embedding row lookup refusing negative and out-of-vocab ids.
fn lookup_row(table: &[f32], vocab: usize, width: usize, token: i32) -> Result<&[f32]> {
    let row = usize::try_from(token).map_err(|_| SpeechError::Input {
        why: format!("negative token id {token}"),
    })?;
    if row >= vocab {
        return Err(SpeechError::Input {
            why: format!("token id {token} outside vocab {vocab}"),
        });
    }
    Ok(&table[row * width..(row + 1) * width])
}

/// Generation prompt mode (`mode` in the reference `build_inference_input_ids`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromptMode {
    VoiceClone,
    Continuation,
}

/// Transposes the generated frames `[T][K]` into the codebook-major
/// `[K][T]` layout the MOSS audio tokenizer decode consumes. An empty
/// frame list stays empty (the reference returns a `[1, 0, n_vq]` token
/// tensor and nothing to decode).
pub fn frames_to_codes(frames: &[Vec<i32>], codebooks: usize) -> Vec<Vec<i32>> {
    if frames.is_empty() {
        return Vec::new();
    }
    let mut codes = vec![Vec::with_capacity(frames.len()); codebooks];
    for frame in frames {
        for (book, code) in frame.iter().take(codebooks).enumerate() {
            codes[book].push(*code);
        }
    }
    codes
}

#[cfg(test)]
mod tests;
