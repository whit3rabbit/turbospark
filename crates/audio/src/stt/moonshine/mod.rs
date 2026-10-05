//! Moonshine: Useful Sensors' lightweight English ASR.
//!
//! Reference: `mlx_audio/stt/models/moonshine/moonshine.py` (v0.5.7).
//! Architecture: a three-conv front end (tanh, GroupNorm, gelu, gelu),
//! a bidirectional encoder with partial interleaved RoPE (90% of each
//! head dim, GPT-J pair style) and bias-free LayerNorms, then a causal
//! decoder with cross-attention and a gated SiLU MLP, greedy decoded
//! from `decoder_start_token_id` until `eos_token_id`.
//!
//! Checkpoints (UsefulSensors/moonshine-tiny and siblings) store conv
//! weights in the PyTorch `[out, in, kernel]` layout this crate's conv
//! kernel consumes directly, plus a `model.` tensor-name prefix the
//! loader strips. The tokenizer is a sentencepiece-style BPE shipped as
//! `tokenizer.json`; only decoding is implemented here (STT never
//! encodes text).

use std::collections::HashMap;
use std::path::Path;

use turbospark_model_io::safetensors::SafetensorsFile;

use crate::ops;
use crate::{Result, SpeechError};

/// Model dimensions, resolved from `config.json` with the reference
/// defaults (moonshine-tiny shape).
#[derive(Debug, Clone)]
pub struct MoonshineConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub encoder_layers: usize,
    pub decoder_layers: usize,
    pub encoder_heads: usize,
    pub decoder_heads: usize,
    pub encoder_kv_heads: usize,
    pub decoder_kv_heads: usize,
    pub partial_rotary_factor: f32,
    pub rope_theta: f32,
    pub decoder_start_token_id: u32,
    pub eos_token_id: u32,
    pub tie_word_embeddings: bool,
}

impl MoonshineConfig {
    pub fn from_json(v: &serde_json::Value) -> Result<Self> {
        let get = |key: &str| -> Result<serde_json::Value> {
            v.get(key).cloned().ok_or_else(|| SpeechError::BadConfig {
                field: key.to_string(),
                why: "missing from config.json".to_string(),
            })
        };
        let as_usize = |x: serde_json::Value| x.as_u64().map(|n| n as usize);
        // Refuse mistyped integers before resolving defaults; truncating
        // token IDs or defaulting a malformed head count changes the model.
        for field in [
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "encoder_num_hidden_layers",
            "decoder_num_hidden_layers",
            "encoder_num_attention_heads",
            "decoder_num_attention_heads",
            "encoder_num_key_value_heads",
            "decoder_num_key_value_heads",
        ] {
            if v.get(field)
                .and_then(|x| x.as_u64())
                .and_then(|n| usize::try_from(n).ok())
                .is_none()
            {
                return Err(SpeechError::BadConfig {
                    field: field.to_string(),
                    why: "must be an integer fitting usize".to_string(),
                });
            }
        }
        for field in ["decoder_start_token_id", "eos_token_id"] {
            if let Some(value) = v.get(field) {
                if value.as_u64().and_then(|n| u32::try_from(n).ok()).is_none() {
                    return Err(SpeechError::BadConfig {
                        field: field.to_string(),
                        why: "must be an unsigned 32-bit integer".to_string(),
                    });
                }
            }
        }
        let encoder_heads = as_usize(get("encoder_num_attention_heads")?).ok_or_else(|| {
            SpeechError::BadConfig {
                field: "encoder_num_attention_heads".to_string(),
                why: "not an integer".to_string(),
            }
        })?;
        let decoder_heads = as_usize(get("decoder_num_attention_heads")?).ok_or_else(|| {
            SpeechError::BadConfig {
                field: "decoder_num_attention_heads".to_string(),
                why: "not an integer".to_string(),
            }
        })?;
        let config = MoonshineConfig {
            vocab_size: as_usize(get("vocab_size")?).unwrap_or(32768),
            hidden_size: as_usize(get("hidden_size")?).unwrap_or(288),
            intermediate_size: as_usize(get("intermediate_size")?).unwrap_or(1152),
            encoder_layers: as_usize(get("encoder_num_hidden_layers")?).unwrap_or(6),
            decoder_layers: as_usize(get("decoder_num_hidden_layers")?).unwrap_or(6),
            encoder_heads,
            decoder_heads,
            encoder_kv_heads: as_usize(get("encoder_num_key_value_heads")?)
                .unwrap_or(encoder_heads),
            decoder_kv_heads: as_usize(get("decoder_num_key_value_heads")?)
                .unwrap_or(decoder_heads),
            partial_rotary_factor: get("partial_rotary_factor")
                .ok()
                .and_then(|x| x.as_f64())
                .map(|x| x as f32)
                .unwrap_or(0.9),
            rope_theta: get("rope_theta")
                .ok()
                .and_then(|x| x.as_f64())
                .map(|x| x as f32)
                .unwrap_or(10_000.0),
            decoder_start_token_id: get("decoder_start_token_id")
                .ok()
                .and_then(|x| x.as_u64())
                .unwrap_or(1) as u32,
            eos_token_id: get("eos_token_id")
                .ok()
                .and_then(|x| x.as_u64())
                .unwrap_or(2) as u32,
            tie_word_embeddings: get("tie_word_embeddings")
                .ok()
                .and_then(|x| x.as_bool())
                .unwrap_or(true),
        };
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<()> {
        let bad = |field: &str, why: &str| SpeechError::BadConfig {
            field: field.to_string(),
            why: why.to_string(),
        };
        for (field, value) in [
            ("hidden_size", self.hidden_size),
            ("intermediate_size", self.intermediate_size),
            ("vocab_size", self.vocab_size),
            ("encoder_num_hidden_layers", self.encoder_layers),
            ("decoder_num_hidden_layers", self.decoder_layers),
        ] {
            if value == 0 {
                return Err(bad(field, "must be positive"));
            }
        }
        for (field, heads, kv_heads) in [
            (
                "encoder_num_attention_heads",
                self.encoder_heads,
                self.encoder_kv_heads,
            ),
            (
                "decoder_num_attention_heads",
                self.decoder_heads,
                self.decoder_kv_heads,
            ),
        ] {
            if heads == 0 || kv_heads == 0 || self.hidden_size % heads != 0 || heads % kv_heads != 0
            {
                return Err(bad(
                    field,
                    "heads must divide hidden_size and key/value heads must divide heads",
                ));
            }
        }
        if !(self.partial_rotary_factor.is_finite()
            && self.partial_rotary_factor > 0.0
            && self.partial_rotary_factor <= 1.0)
        {
            return Err(bad("partial_rotary_factor", "must be finite and in (0, 1]"));
        }
        if !self.rope_theta.is_finite() || self.rope_theta <= 0.0 {
            return Err(bad("rope_theta", "must be finite and positive"));
        }
        if self.vocab_size > i32::MAX as usize
            || self.decoder_start_token_id as usize >= self.vocab_size
            || self.eos_token_id as usize >= self.vocab_size
        {
            return Err(bad(
                "vocab_size",
                "token IDs must fit the signed embedding vocabulary",
            ));
        }
        for factors in [
            [self.hidden_size, self.hidden_size, 14],
            [self.hidden_size, self.intermediate_size, 2],
            [self.hidden_size, self.vocab_size, 1],
        ] {
            if factors
                .into_iter()
                .try_fold(1usize, |n, v| n.checked_mul(v))
                .is_none_or(|n| n > isize::MAX as usize / 4)
            {
                return Err(bad(
                    "hidden_size",
                    "tensor dimensions overflow the allocation limit",
                ));
            }
        }
        Ok(())
    }
}

/// One attention projection set. `bias` is absent on every moonshine
/// projection in shipped checkpoints.
struct Attn {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    o: Vec<f32>,
}

struct EncoderLayer {
    attn: Attn,
    ln1: Vec<f32>,
    ln2: Vec<f32>,
    fc1: Vec<f32>,
    fc1_b: Vec<f32>,
    fc2: Vec<f32>,
    fc2_b: Vec<f32>,
}

struct DecoderLayer {
    self_attn: Attn,
    cross_attn: Attn,
    ln1: Vec<f32>,
    ln2: Vec<f32>,
    ln3: Vec<f32>,
    fc1: Vec<f32>,
    fc1_b: Vec<f32>,
    fc2: Vec<f32>,
    fc2_b: Vec<f32>,
}

/// The loaded model.
pub struct Moonshine {
    config: MoonshineConfig,
    conv1: Vec<f32>,
    groupnorm_w: Vec<f32>,
    groupnorm_b: Vec<f32>,
    conv2: Vec<f32>,
    conv2_b: Vec<f32>,
    conv3: Vec<f32>,
    conv3_b: Vec<f32>,
    enc_layers: Vec<EncoderLayer>,
    enc_norm: Vec<f32>,
    embed: Vec<f32>,
    dec_layers: Vec<DecoderLayer>,
    dec_norm: Vec<f32>,
    proj_out: Option<Vec<f32>>,
    tokenizer: Tokenizer,
}

struct AttnShape {
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    rotary: usize,
}

impl Moonshine {
    /// Opens a model directory: `config.json`, `model.safetensors`,
    /// `tokenizer.json`.
    pub fn open(dir: &Path) -> Result<Self> {
        let config: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).map_err(
                |e| SpeechError::BadConfig {
                    field: "config.json".to_string(),
                    why: e.to_string(),
                },
            )?)
            .map_err(|e| SpeechError::BadConfig {
                field: "config.json".to_string(),
                why: e.to_string(),
            })?;
        let config = MoonshineConfig::from_json(&config)?;
        let file = SafetensorsFile::open(&dir.join("model.safetensors"))?;
        let tokenizer = Tokenizer::load(&dir.join("tokenizer.json"))?;

        // Tensor names: shipped checkpoints carry the `model.` prefix;
        // accept both that and the stripped form.
        let name = |key: &str| -> String {
            if file.contains_tensor(&format!("model.{key}")) {
                format!("model.{key}")
            } else {
                key.to_string()
            }
        };
        let f32v = |key: &str| -> Result<Vec<f32>> {
            let n = name(key);
            file.load_as_f32(&n).map_err(|e| SpeechError::Tensor {
                name: n,
                why: e.to_string(),
            })
        };
        let opt_f32v = |key: &str| -> Result<Option<Vec<f32>>> {
            let n = name(key);
            if file.contains_tensor(&n) {
                Ok(Some(file.load_as_f32(&n)?))
            } else {
                Ok(None)
            }
        };
        let conv1 = f32v("encoder.conv1.weight")?;
        let groupnorm_w = f32v("encoder.groupnorm.weight")?;
        let groupnorm_b = f32v("encoder.groupnorm.bias")?;
        let conv2 = f32v("encoder.conv2.weight")?;
        let conv2_b = f32v("encoder.conv2.bias")?;
        let conv3 = f32v("encoder.conv3.weight")?;
        let conv3_b = f32v("encoder.conv3.bias")?;
        let enc_norm = f32v("encoder.layer_norm.weight")?;
        let embed = f32v("decoder.embed_tokens.weight")?;
        let dec_norm = f32v("decoder.norm.weight")?;
        let proj_out = opt_f32v("proj_out.weight")?;

        let load_attn = |base: &str| -> Result<Attn> {
            Ok(Attn {
                q: f32v(&format!("{base}.q_proj.weight"))?,
                k: f32v(&format!("{base}.k_proj.weight"))?,
                v: f32v(&format!("{base}.v_proj.weight"))?,
                o: f32v(&format!("{base}.o_proj.weight"))?,
            })
        };
        let mut enc_layers = Vec::with_capacity(config.encoder_layers);
        for i in 0..config.encoder_layers {
            enc_layers.push(EncoderLayer {
                attn: load_attn(&format!("encoder.layers.{i}.self_attn"))?,
                ln1: f32v(&format!("encoder.layers.{i}.input_layernorm.weight"))?,
                ln2: f32v(&format!(
                    "encoder.layers.{i}.post_attention_layernorm.weight"
                ))?,
                fc1: f32v(&format!("encoder.layers.{i}.mlp.fc1.weight"))?,
                fc1_b: f32v(&format!("encoder.layers.{i}.mlp.fc1.bias"))?,
                fc2: f32v(&format!("encoder.layers.{i}.mlp.fc2.weight"))?,
                fc2_b: f32v(&format!("encoder.layers.{i}.mlp.fc2.bias"))?,
            });
        }
        let mut dec_layers = Vec::with_capacity(config.decoder_layers);
        for i in 0..config.decoder_layers {
            dec_layers.push(DecoderLayer {
                self_attn: load_attn(&format!("decoder.layers.{i}.self_attn"))?,
                cross_attn: load_attn(&format!("decoder.layers.{i}.encoder_attn"))?,
                ln1: f32v(&format!("decoder.layers.{i}.input_layernorm.weight"))?,
                ln2: f32v(&format!(
                    "decoder.layers.{i}.post_attention_layernorm.weight"
                ))?,
                ln3: f32v(&format!("decoder.layers.{i}.final_layernorm.weight"))?,
                fc1: f32v(&format!("decoder.layers.{i}.mlp.fc1.weight"))?,
                fc1_b: f32v(&format!("decoder.layers.{i}.mlp.fc1.bias"))?,
                fc2: f32v(&format!("decoder.layers.{i}.mlp.fc2.weight"))?,
                fc2_b: f32v(&format!("decoder.layers.{i}.mlp.fc2.bias"))?,
            });
        }
        Ok(Moonshine {
            config,
            conv1,
            groupnorm_w,
            groupnorm_b,
            conv2,
            conv2_b,
            conv3,
            conv3_b,
            enc_layers,
            enc_norm,
            embed,
            dec_layers,
            dec_norm,
            proj_out,
            tokenizer,
        })
    }

    /// Conv front end: `[samples]` -> `[seq, hidden]` encoder input.
    fn frontend(&self, samples: &[f32]) -> Result<Vec<f32>> {
        let d = self.config.hidden_size;
        let mut x = ops::conv1d(samples, &self.conv1, None, 1, d, 127, 64, 0, 1, 1);
        let seq = x.len() / d;
        for v in x.iter_mut() {
            *v = v.tanh();
        }
        // x is channel-major [d, seq] here (the conv kernel's output
        // layout); GroupNorm normalizes globally either way but the
        // affine is per channel.
        groupnorm_channels(&mut x, seq, d, &self.groupnorm_w, &self.groupnorm_b, 1e-5);
        let mut x = ops::conv1d(
            &x,
            &self.conv2,
            Some(&self.conv2_b),
            d,
            2 * d,
            7,
            3,
            0,
            1,
            1,
        );
        ops::gelu_erf(&mut x);
        let mut x = ops::conv1d(
            &x,
            &self.conv3,
            Some(&self.conv3_b),
            2 * d,
            d,
            3,
            2,
            0,
            1,
            1,
        );
        ops::gelu_erf(&mut x);
        // The transformer consumes [seq, hidden]; the conv stack ran in
        // the kernel's channel-major layout, so transpose once here.
        let seq3 = x.len() / d;
        let mut out = vec![0.0f32; x.len()];
        for c in 0..d {
            for s in 0..seq3 {
                out[s * d + c] = x[c * seq3 + s];
            }
        }
        Ok(out)
    }

    /// Encoder: frontend + transformer stack. Returns `[seq, hidden]`.
    pub fn encode(&self, samples: &[f32]) -> Result<Vec<f32>> {
        // The unpadded 127/64, 7/3, 3/2 conv stack needs 895 samples
        // to produce its first encoder position.
        if samples.len() < 895 || samples.iter().any(|v| !v.is_finite()) {
            return Err(SpeechError::Input {
                why: "Moonshine requires at least 895 finite 16 kHz samples".to_string(),
            });
        }
        let d = self.config.hidden_size;
        let mut x = self.frontend(samples)?;
        let seq = x.len() / d;
        let heads = self.config.encoder_heads;
        let kv_heads = self.config.encoder_kv_heads;
        let head_dim = d / heads;
        let rotary = (head_dim as f32 * self.config.partial_rotary_factor) as usize;
        let rotary = rotary - (rotary % 2);
        let (cos, sin) = ops::rope_tables(seq, rotary, self.config.rope_theta);
        for layer in &self.enc_layers {
            let mut h = x.clone();
            ops::layernorm(&mut h, seq, d, &layer.ln1, None, 1e-5);
            let attn = attention_block(
                &h,
                KvSource::Project,
                seq,
                d,
                &AttnShape {
                    heads,
                    kv_heads,
                    head_dim,
                    rotary,
                },
                &layer.attn,
                Some((&cos, &sin, 0)),
                None,
            )?;
            for (r, a) in x.iter_mut().zip(&attn) {
                *r += a;
            }
            let mut h = x.clone();
            ops::layernorm(&mut h, seq, d, &layer.ln2, None, 1e-5);
            let mut mlp = ops::linear(
                &h,
                &layer.fc1,
                Some(&layer.fc1_b),
                seq,
                d,
                self.config.intermediate_size,
            );
            ops::gelu_erf(&mut mlp);
            let mlp = ops::linear(
                &mlp,
                &layer.fc2,
                Some(&layer.fc2_b),
                seq,
                self.config.intermediate_size,
                d,
            );
            for (r, a) in x.iter_mut().zip(&mlp) {
                *r += a;
            }
        }
        ops::layernorm(&mut x, seq, d, &self.enc_norm, None, 1e-5);
        Ok(x)
    }

    /// One greedy decode step over the cached decoder. `cache` holds the
    /// self-attention keys/values (grown each step) and the precomputed
    /// cross-attention keys/values (computed once).
    fn decode_step(
        &self,
        token: u32,
        encoder_out: &[f32],
        enc_seq: usize,
        cache: &mut DecodeCache,
    ) -> Result<Vec<f32>> {
        let d = self.config.hidden_size;
        let heads = self.config.decoder_heads;
        let kv_heads = self.config.decoder_kv_heads;
        let head_dim = d / heads;
        let rotary = (head_dim as f32 * self.config.partial_rotary_factor) as usize;
        let rotary = rotary - (rotary % 2);
        let pos = cache.self_kv[0]
            .as_ref()
            .map_or(0, |(k, _)| k.len() / (kv_heads * head_dim));
        let (cos, sin) = ops::rope_tables(pos + 1, rotary, self.config.rope_theta);
        let mut x = ops::embedding(&self.embed, d, &[token as i32]);
        let _debug = std::env::var("MOONSHINE_DEBUG").is_ok();
        let first_step = std::env::var("MOONSHINE_DUMP_STEPOK").is_ok();
        if first_step {
            std::env::remove_var("MOONSHINE_DUMP_STEPOK");
            eprintln!("rust decode_step token={} embed[:4]={:?}", token, &x[..4]);
            let mut bytes = Vec::new();
            for v in &x {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            let _ = std::fs::write("/tmp/rust_emb1.raw", bytes);
        }
        for (i, layer) in self.dec_layers.iter().enumerate() {
            // Self attention over the single new token, keys/values
            // appended to the cache. With one query position the
            // reference's causal mask is never built (T == 1).
            let mut h = x.clone();
            ops::layernorm(&mut h, 1, d, &layer.ln1, None, 1e-5);
            let attn = attention_block(
                &h,
                KvSource::Project,
                1,
                d,
                &AttnShape {
                    heads,
                    kv_heads,
                    head_dim,
                    rotary,
                },
                &layer.self_attn,
                Some((&cos, &sin, pos)),
                Some(&mut cache.self_kv[i]),
            )?;
            for (r, a) in x.iter_mut().zip(&attn) {
                *r += a;
            }
            // Cross attention: cache the projected encoder k/v forever.
            let mut h = x.clone();
            ops::layernorm(&mut h, 1, d, &layer.ln2, None, 1e-5);
            let (ck, cv) = cache.cross_kv[i].get_or_insert_with(|| {
                let k = ops::linear(
                    encoder_out,
                    &layer.cross_attn.k,
                    None,
                    enc_seq,
                    d,
                    kv_heads * head_dim,
                );
                let v = ops::linear(
                    encoder_out,
                    &layer.cross_attn.v,
                    None,
                    enc_seq,
                    d,
                    kv_heads * head_dim,
                );
                (k, v)
            });
            let attn = attention_block(
                &h,
                KvSource::Provide(ck, cv),
                1,
                d,
                &AttnShape {
                    heads,
                    kv_heads,
                    head_dim,
                    rotary,
                },
                &layer.cross_attn,
                None,
                None,
            )?;
            for (r, a) in x.iter_mut().zip(&attn) {
                *r += a;
            }
            // final_layernorm BEFORE the MLP, the reference's own order.
            let mut h = x.clone();
            ops::layernorm(&mut h, 1, d, &layer.ln3, None, 1e-5);
            let mut h2 = ops::linear(
                &h,
                &layer.fc1,
                Some(&layer.fc1_b),
                1,
                d,
                2 * self.config.intermediate_size,
            );
            // Split into (x, gate); gate gets the SiLU.
            let mid = self.config.intermediate_size;
            let mut gate = h2[mid..2 * mid].to_vec();
            h2.truncate(mid);
            for v in gate.iter_mut() {
                *v /= 1.0 + (-*v).exp();
            }
            for (a, g) in h2.iter_mut().zip(&gate) {
                *a *= g;
            }
            let mlp = ops::linear(&h2, &layer.fc2, Some(&layer.fc2_b), 1, mid, d);
            for (r, a) in x.iter_mut().zip(&mlp) {
                *r += a;
            }
        }
        ops::layernorm(&mut x, 1, d, &self.dec_norm, None, 1e-5);
        Ok(x)
    }

    /// Transcribes 16 kHz mono PCM greedily (the reference's
    /// `temperature = 0` path).
    pub fn transcribe(&self, samples: &[f32]) -> Result<String> {
        let encoder_out = self.encode(samples)?;
        let enc_seq = encoder_out.len() / self.config.hidden_size;
        let mut cache = DecodeCache {
            self_kv: vec![None; self.dec_layers.len()],
            cross_kv: vec![None; self.dec_layers.len()],
        };
        let mut token = self.config.decoder_start_token_id;
        let mut generated: Vec<u32> = Vec::new();
        for _ in 0..448 {
            let hidden = self.decode_step(token, &encoder_out, enc_seq, &mut cache)?;
            // Logits over the vocab: tied embedding head unless the
            // checkpoint carries a separate proj_out.
            let logits = match &self.proj_out {
                Some(p) => ops::linear(
                    &hidden,
                    p,
                    None,
                    1,
                    self.config.hidden_size,
                    self.config.vocab_size,
                ),
                None => {
                    // h @ embed.T
                    let mut logits = vec![0.0f32; self.config.vocab_size];
                    let cols = self.config.hidden_size;
                    for (v, logit) in logits.iter_mut().enumerate() {
                        let row = &self.embed[v * cols..(v + 1) * cols];
                        *logit = row.iter().zip(hidden.iter()).map(|(e, h)| e * h).sum();
                    }
                    logits
                }
            };
            let (best, _) = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .ok_or(SpeechError::Input {
                    why: "empty logits".to_string(),
                })?;
            if best as u32 == self.config.eos_token_id {
                break;
            }
            token = best as u32;
            generated.push(token);
        }
        Ok(self.tokenizer.decode(&generated))
    }
}

/// Decoder KV cache: self-attention keys/values `[kv_heads * head_dim,
/// cached_len]` per layer is flattened to one buffer shared across
/// layers as `[k; v]` pairs; cross-attention projections are computed
/// once from the encoder output.
struct DecodeCache {
    /// One self-attention KV buffer PER LAYER (the reference builds a
    /// cache dict per layer; sharing one buffer corrupts every layer
    /// after the first).
    self_kv: Vec<Option<(Vec<f32>, Vec<f32>)>>,
    cross_kv: Vec<Option<(Vec<f32>, Vec<f32>)>>,
}

/// Where an attention block gets its keys and values from.
enum KvSource<'a> {
    /// Project from the block input (self attention).
    Project,
    /// Precomputed keys/values (cross attention over the encoder).
    Provide(&'a [f32], &'a [f32]),
}

/// Attention over one or more query positions.
///
/// `x` is `[q_len, d]`. With [`KvSource::Project`], keys/values project
/// from `x` and, when `self_cache` is `Some`, append to (or initialize)
/// the cached buffer; the decoder passes the same cache every step.
/// Rope applies interleaved-style to the leading `rotary` dims of q and
/// k at `offset`. Returns `[q_len, d]` after the output projection.
#[allow(clippy::too_many_arguments)]
fn attention_block(
    x: &[f32],
    kv_source: KvSource,
    q_len: usize,
    d: usize,
    shape: &AttnShape,
    attn: &Attn,
    rope: Option<(&[f32], &[f32], usize)>,
    self_cache: Option<&mut Option<(Vec<f32>, Vec<f32>)>>,
) -> Result<Vec<f32>> {
    let AttnShape {
        heads,
        kv_heads,
        head_dim,
        rotary,
    } = *shape;
    let (src, kv_len) = match kv_source {
        KvSource::Project => (x, q_len),
        KvSource::Provide(k, v) => {
            debug_assert_eq!(k.len(), v.len());
            // The source only feeds the projections below; cross
            // attention callers pass the cached k/v directly, so the
            // projections here would be wasted work. Cross callers must
            // use Provide with cached values and skip projection, which
            // the decoder does by passing cached pairs.
            (x, k.len() / (kv_heads * head_dim))
        }
    };
    let mut q = ops::linear(x, &attn.q, None, q_len, d, heads * head_dim);
    let projected;
    let (k_all, v_all): (&[f32], &[f32]);
    match kv_source {
        KvSource::Project => {
            let mut k = ops::linear(src, &attn.k, None, kv_len, d, kv_heads * head_dim);
            let v = ops::linear(src, &attn.v, None, kv_len, d, kv_heads * head_dim);
            if let Some((cos, sin, offset)) = rope {
                rope_heads(&mut q, heads, q_len, head_dim, rotary, cos, sin, offset);
                rope_heads(&mut k, kv_heads, kv_len, head_dim, rotary, cos, sin, offset);
            }
            match self_cache {
                Some(slot) => {
                    let (cached_k, cached_v) = slot.get_or_insert_with(|| (Vec::new(), Vec::new()));
                    cached_k.extend_from_slice(&k);
                    cached_v.extend_from_slice(&v);
                    k_all = cached_k;
                    v_all = cached_v;
                }
                None => {
                    projected = (k, v);
                    k_all = &projected.0;
                    v_all = &projected.1;
                }
            }
        }
        KvSource::Provide(k, v) => {
            k_all = k;
            v_all = v;
        }
    }
    let total_len = k_all.len() / (kv_heads * head_dim);
    let scale = (head_dim as f32).powf(-0.5);
    let mut out = vec![0.0f32; q_len * d];
    let mut q_head = vec![0.0f32; q_len * head_dim];
    let q_stride = heads * head_dim;
    let kv_stride = kv_heads * head_dim;
    for h in 0..heads {
        let kv_h = h / (heads / kv_heads);
        for t in 0..q_len {
            q_head[t * head_dim..(t + 1) * head_dim].copy_from_slice(
                &q[t * q_stride + h * head_dim..t * q_stride + (h + 1) * head_dim],
            );
        }
        let o = ops::sdpa_strided(
            &q_head,
            k_all,
            kv_h * head_dim,
            kv_stride,
            v_all,
            kv_h * head_dim,
            kv_stride,
            None,
            q_len,
            total_len,
            head_dim,
            head_dim,
            scale,
        );
        for t in 0..q_len {
            out[t * d + h * head_dim..t * d + (h + 1) * head_dim]
                .copy_from_slice(&o[t * head_dim..(t + 1) * head_dim]);
        }
    }
    Ok(ops::linear(&out, &attn.o, None, q_len, d, d))
}

/// Applies interleaved rope to `x [seq, heads * head_dim]` (the layout
/// the linear projections emit) for positions `offset..offset+seq`,
/// rotary on the leading `rotary` dims of each head.
#[allow(
    clippy::too_many_arguments,
    reason = "attention kernels carry the tensor shape explicitly"
)]
fn rope_heads(
    x: &mut [f32],
    heads: usize,
    seq: usize,
    head_dim: usize,
    rotary: usize,
    cos: &[f32],
    sin: &[f32],
    offset: usize,
) {
    let half = rotary / 2;
    let stride = heads * head_dim;
    for h in 0..heads {
        for t in 0..seq {
            let base = t * stride + h * head_dim;
            let pos = offset + t;
            for d in 0..half {
                let c = cos[pos * half + d];
                let s = sin[pos * half + d];
                let a = x[base + 2 * d];
                let b = x[base + 2 * d + 1];
                x[base + 2 * d] = a * c - b * s;
                x[base + 2 * d + 1] = b * c + a * s;
            }
        }
    }
}

/// GroupNorm with one group over channel-major `x [channels, seq]`:
/// statistics over all channels and positions, then the per-channel
/// affine transform.
fn groupnorm_channels(x: &mut [f32], seq: usize, channels: usize, w: &[f32], b: &[f32], eps: f32) {
    let n = seq * channels;
    let mean = x.iter().sum::<f32>() / n as f32;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
    let inv = 1.0 / (var + eps).sqrt();
    for v in x.iter_mut() {
        *v = (*v - mean) * inv;
    }
    for c in 0..channels {
        for v in &mut x[c * seq..(c + 1) * seq] {
            *v = *v * w[c] + b[c];
        }
    }
}

/// Minimal tokenizer.json decoder: vocab lookup, ByteFallback, space
/// join, leading-space strip. Encoding text is out of scope for STT.
pub struct Tokenizer {
    id_to_token: HashMap<u32, String>,
    special_ids: std::collections::HashSet<u32>,
}

impl Tokenizer {
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).map_err(|e| SpeechError::BadConfig {
            field: "tokenizer.json".to_string(),
            why: e.to_string(),
        })?;
        let v: serde_json::Value =
            serde_json::from_str(&text).map_err(|e| SpeechError::BadConfig {
                field: "tokenizer.json".to_string(),
                why: e.to_string(),
            })?;
        let mut id_to_token = HashMap::new();
        if let Some(vocab) = v
            .get("model")
            .and_then(|m| m.get("vocab"))
            .and_then(|x| x.as_object())
        {
            for (token, id) in vocab {
                id_to_token.insert(id.as_u64().unwrap_or(0) as u32, token.clone());
            }
        }
        let mut special_ids = std::collections::HashSet::new();
        if let Some(added) = v.get("added_tokens").and_then(|x| x.as_array()) {
            for t in added {
                if let (Some(id), Some(content)) = (
                    t.get("id").and_then(|v| v.as_u64()),
                    t.get("content").and_then(|v| v.as_str()),
                ) {
                    id_to_token.insert(id as u32, content.to_string());
                }
                if t.get("special").and_then(|s| s.as_bool()).unwrap_or(false) {
                    if let Some(id) = t.get("id").and_then(|i| i.as_u64()) {
                        special_ids.insert(id as u32);
                    }
                }
            }
        }
        if id_to_token.is_empty() {
            return Err(SpeechError::BadConfig {
                field: "tokenizer.json".to_string(),
                why: "no model.vocab entries".to_string(),
            });
        }
        Ok(Tokenizer {
            id_to_token,
            special_ids,
        })
    }

    /// The reference decode chain: skip special tokens, map `▁` to a
    /// space, resolve `<0xNN>` byte-fallback tokens to raw bytes, fuse,
    /// and strip one leading space.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut pieces: Vec<u8> = Vec::new();
        let mut pending_bytes: Vec<u8> = Vec::new();
        let flush = |pending: &mut Vec<u8>, out: &mut Vec<u8>| {
            out.append(pending);
        };
        for &id in ids {
            if self.special_ids.contains(&id) {
                continue;
            }
            let Some(token) = self.id_to_token.get(&id) else {
                continue;
            };
            if let Some(hex) = token.strip_prefix("<0x").and_then(|s| s.strip_suffix('>')) {
                if let Ok(byte) = u8::from_str_radix(hex, 16) {
                    pending_bytes.push(byte);
                    continue;
                }
            }
            flush(&mut pending_bytes, &mut pieces);
            let text = token.replace('▁', " ");
            pieces.extend_from_slice(text.as_bytes());
        }
        flush(&mut pending_bytes, &mut pieces);
        let text = String::from_utf8_lossy(&pieces).to_string();
        text.strip_prefix(' ').unwrap_or(&text).to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_tokenizer() -> Tokenizer {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "moonshine-tok-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tokenizer.json");
        std::fs::write(
            &path,
            serde_json::json!({
                "model": {"vocab": {"▁hello": 0, "<0xF0>": 1, "<0x9F>": 2, "<0x91>": 3, "world": 4, "▁the": 5}},
                "added_tokens": [
                    {"id": 6, "special": true, "content": "<|startofcontext|>"},
                    {"id": 7, "special": false, "content": "[PAUSE]"}
                ]
            })
            .to_string(),
        )
        .unwrap();
        let tokenizer = Tokenizer::load(&path).unwrap();
        std::fs::remove_dir_all(dir).unwrap();
        tokenizer
    }

    #[test]
    fn tokenizer_decode_joins_and_strips() {
        let tok = test_tokenizer();
        // Each " word" piece carries its own sentencepiece space mark.
        assert_eq!(tok.decode(&[0, 5]), "hello the");
        assert_eq!(tok.decode(&[4]), "world");
    }

    #[test]
    fn tokenizer_byte_fallback_recomposes_utf8() {
        let tok = test_tokenizer();
        // <0xF0><0x9F><0x91> is the first three bytes of an emoji; the
        // lossy decode keeps the replacement character, proving bytes
        // flow through as bytes rather than text.
        let out = tok.decode(&[1, 2, 3]);
        assert_eq!(out, "\u{FFFD}");
    }

    #[test]
    fn tokenizer_skips_special_tokens_only() {
        let tok = test_tokenizer();
        // Non-special added tokens are part of the decoded vocabulary.
        assert_eq!(tok.decode(&[6, 0, 7]), "hello[PAUSE]");
    }

    #[test]
    fn groupnorm_normalizes_then_affines() {
        // channel-major [ch 2, seq 2]: ch0 = [1, 3], ch1 = [2, 4]
        let mut x = vec![1.0, 3.0, 2.0, 4.0];
        groupnorm_channels(&mut x, 2, 2, &[1.0, 1.0], &[0.0, 0.0], 0.0);
        let mean = 2.5;
        let inv = 1.0 / (1.25f32).sqrt();
        assert!((x[0] - (1.0 - mean) * inv).abs() < 1e-5);
        assert!((x[3] - (4.0 - mean) * inv).abs() < 1e-5);
    }

    #[test]
    fn attention_cache_appends_once_and_matches_prefix_attention() {
        let identity = vec![1.0, 0.0, 0.0, 1.0];
        let attn = Attn {
            q: identity.clone(),
            k: identity.clone(),
            v: identity.clone(),
            o: identity,
        };
        let shape = AttnShape {
            heads: 1,
            kv_heads: 1,
            head_dim: 2,
            rotary: 0,
        };
        let mut cache = None;
        let first = attention_block(
            &[1.0, 0.0],
            KvSource::Project,
            1,
            2,
            &shape,
            &attn,
            None,
            Some(&mut cache),
        )
        .unwrap();
        assert_eq!(first, vec![1.0, 0.0]);
        let second = attention_block(
            &[0.0, 1.0],
            KvSource::Project,
            1,
            2,
            &shape,
            &attn,
            None,
            Some(&mut cache),
        )
        .unwrap();
        let (keys, values) = cache.as_ref().unwrap();
        assert_eq!(keys, &[1.0, 0.0, 0.0, 1.0]);
        assert_eq!(values, keys);
        let expected = ops::sdpa(
            &[0.0, 1.0],
            keys,
            values,
            None,
            1,
            2,
            2,
            2,
            1.0 / 2.0f32.sqrt(),
        );
        assert_eq!(second, expected);
        let cross = attention_block(
            &[0.0, 1.0],
            KvSource::Provide(keys, values),
            1,
            2,
            &shape,
            &attn,
            None,
            None,
        )
        .unwrap();
        assert_eq!(cross, expected);
    }
}

#[cfg(test)]
mod config_regression {
    use super::*;

    #[test]
    fn rejects_configs_that_cannot_form_attention_or_decoder_state() {
        let valid = serde_json::json!({ "vocab_size": 32768, "hidden_size": 288, "intermediate_size": 1152,
            "encoder_num_hidden_layers": 6, "decoder_num_hidden_layers": 6,
            "encoder_num_attention_heads": 8, "decoder_num_attention_heads": 8,
            "encoder_num_key_value_heads": 8, "decoder_num_key_value_heads": 8 });
        for (key, value) in [
            ("encoder_num_attention_heads", serde_json::json!(0)),
            ("decoder_num_hidden_layers", serde_json::json!(0)),
            ("encoder_num_key_value_heads", serde_json::json!(3)),
            ("partial_rotary_factor", serde_json::json!(2.0)),
            ("decoder_start_token_id", serde_json::json!(32768)),
            ("decoder_start_token_id", serde_json::json!(4294967297u64)),
            ("hidden_size", serde_json::json!("bad")),
        ] {
            let mut invalid = valid.clone();
            invalid[key] = value;
            assert!(MoonshineConfig::from_json(&invalid).is_err(), "{key}");
        }
    }

    #[test]
    fn reads_the_checkpoints_partial_rotary_factor() {
        let value = serde_json::json!({ "vocab_size": 32768, "hidden_size": 288, "intermediate_size": 1152,
            "encoder_num_hidden_layers": 6, "decoder_num_hidden_layers": 6,
            "encoder_num_attention_heads": 8, "decoder_num_attention_heads": 8,
            "encoder_num_key_value_heads": 8, "decoder_num_key_value_heads": 8,
            "partial_rotary_factor": 0.5 });
        assert_eq!(
            MoonshineConfig::from_json(&value)
                .unwrap()
                .partial_rotary_factor,
            0.5
        );
    }
}
