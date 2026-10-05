//! Whisper weight container and safetensors loader.
//!
//! Maps the Hugging Face whisper checkpoint (`model.safetensors`) into the
//! owned f32 tensors the reference kernels consume. Names are tried with
//! and without the `model.` prefix because exports differ on that; dtypes
//! F32/F16/BF16 all load through `SafetensorsFile::load_as_f32`. Missing
//! tensors are `ModelError`s that name the tensor, so a partial or foreign
//! distribution refuses at open instead of running wrong.

use model_io::safetensors::SafetensorsFile;
use model_io::whisper_config::WhisperConfig;
use model_io::ModelError;

/// Owned weights for one whisper encoder layer, all f32. The checkpoint
/// has no k projection bias (openai convention); q, v, and out carry one.
pub struct WhisperEncoderLayerOwned {
    pub q: Vec<f32>,
    pub q_bias: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub v_bias: Vec<f32>,
    pub out: Vec<f32>,
    pub out_bias: Vec<f32>,
    pub ln1_weight: Vec<f32>,
    pub ln1_bias: Vec<f32>,
    pub fc1: Vec<f32>,
    pub fc1_bias: Vec<f32>,
    pub fc2: Vec<f32>,
    pub fc2_bias: Vec<f32>,
    pub ln2_weight: Vec<f32>,
    pub ln2_bias: Vec<f32>,
}

/// Owned weights for one whisper decoder layer, all f32.
pub struct WhisperDecoderLayerOwned {
    pub self_q: Vec<f32>,
    pub self_q_bias: Vec<f32>,
    pub self_k: Vec<f32>,
    pub self_v: Vec<f32>,
    pub self_v_bias: Vec<f32>,
    pub self_out: Vec<f32>,
    pub self_out_bias: Vec<f32>,
    pub ln_self_weight: Vec<f32>,
    pub ln_self_bias: Vec<f32>,
    pub cross_q: Vec<f32>,
    pub cross_q_bias: Vec<f32>,
    pub cross_k: Vec<f32>,
    pub cross_v: Vec<f32>,
    pub cross_v_bias: Vec<f32>,
    pub cross_out: Vec<f32>,
    pub cross_out_bias: Vec<f32>,
    pub ln_cross_weight: Vec<f32>,
    pub ln_cross_bias: Vec<f32>,
    pub fc1: Vec<f32>,
    pub fc1_bias: Vec<f32>,
    pub fc2: Vec<f32>,
    pub fc2_bias: Vec<f32>,
    pub ln_fc_weight: Vec<f32>,
    pub ln_fc_bias: Vec<f32>,
}

/// Complete whisper weights.
pub struct WhisperWeights {
    pub conv1: Vec<f32>,
    pub conv1_bias: Vec<f32>,
    pub conv2: Vec<f32>,
    pub conv2_bias: Vec<f32>,
    /// `[max_source_positions, d_model]` sinusoidal table.
    pub enc_positions: Vec<f32>,
    pub enc_layers: Vec<WhisperEncoderLayerOwned>,
    pub enc_ln_weight: Vec<f32>,
    pub enc_ln_bias: Vec<f32>,
    /// `[vocab, d_model]` token embedding, tied with the output projection.
    pub embed_tokens: Vec<f32>,
    /// `[max_target_positions, d_model]` sinusoidal table.
    pub dec_positions: Vec<f32>,
    pub dec_layers: Vec<WhisperDecoderLayerOwned>,
    pub dec_ln_weight: Vec<f32>,
    pub dec_ln_bias: Vec<f32>,
}

/// Loads a tensor trying both the `model.`-prefixed and bare names.
fn load(file: &SafetensorsFile, name: &str) -> Result<Vec<f32>, ModelError> {
    if file.contains_tensor(name) {
        return file.load_as_f32(name);
    }
    let prefixed = format!("model.{name}");
    if file.contains_tensor(&prefixed) {
        return file.load_as_f32(&prefixed);
    }
    Err(ModelError::TensorNotFound {
        name: name.to_string(),
    })
}

impl WhisperWeights {
    /// Loads every tensor the kernels consume, validating shapes against
    /// the config as it goes. Dispatches on the config: HF layouts
    /// (`d_model`, `*_proj.weight` names) through the HF path, MLX
    /// openai-layout conversions (`n_audio_state`, `blocks.attn.query`
    /// names, groupwise 8-bit quantization) through the MLX path.
    pub fn load_from_safetensors(
        file: &SafetensorsFile,
        config: &WhisperConfig,
    ) -> Result<Self, ModelError> {
        config.validate_weights(file)?;
        if config.quantization.is_some() || file.contains_tensor("decoder.token_embedding.weight") {
            return Self::load_mlx(file, config);
        }
        Self::load_hf(file, config)
    }

    /// HF-layout loader: `model.`-prefixed or bare `*_proj` names, F16 /
    /// BF16 / F32 tensors.
    fn load_hf(file: &SafetensorsFile, config: &WhisperConfig) -> Result<Self, ModelError> {
        let d = config.d_model;
        let shape_check = |name: &str, got: &[f32], want: usize| -> Result<(), ModelError> {
            if got.len() != want {
                return Err(ModelError::TensorSizeMismatch {
                    name: name.to_string(),
                    expected: want as u64,
                    actual: got.len() as u64,
                });
            }
            Ok(())
        };

        // HF stores conv weights [out, in, k], the layout the kernels read.
        let conv1 = load(file, "encoder.conv1.weight")?;
        shape_check("encoder.conv1.weight", &conv1, d * config.n_mels * 3)?;
        let conv1_bias = load(file, "encoder.conv1.bias")?;
        shape_check("encoder.conv1.bias", &conv1_bias, d)?;
        let conv2 = load(file, "encoder.conv2.weight")?;
        shape_check("encoder.conv2.weight", &conv2, d * d * 3)?;
        let conv2_bias = load(file, "encoder.conv2.bias")?;
        shape_check("encoder.conv2.bias", &conv2_bias, d)?;

        let enc_positions = load(file, "encoder.embed_positions.weight")?;
        shape_check(
            "encoder.embed_positions.weight",
            &enc_positions,
            config.max_source_positions * d,
        )?;
        let enc_ln_weight = load(file, "encoder.layer_norm.weight")?;
        let enc_ln_bias = load(file, "encoder.layer_norm.bias")?;
        let embed_tokens = load(file, "decoder.embed_tokens.weight")?;
        shape_check(
            "decoder.embed_tokens.weight",
            &embed_tokens,
            config.vocab_size * d,
        )?;
        let dec_positions = load(file, "decoder.embed_positions.weight")?;
        shape_check(
            "decoder.embed_positions.weight",
            &dec_positions,
            config.max_target_positions * d,
        )?;
        let dec_ln_weight = load(file, "decoder.layer_norm.weight")?;
        let dec_ln_bias = load(file, "decoder.layer_norm.bias")?;

        let mut enc_layers = Vec::with_capacity(config.encoder_layers());
        for i in 0..config.encoder_layers() {
            let p = format!("encoder.layers.{i}.");
            let q = load(file, &format!("{p}self_attn.q_proj.weight"))?;
            shape_check(&format!("{p}self_attn.q_proj.weight"), &q, d * d)?;
            enc_layers.push(WhisperEncoderLayerOwned {
                q,
                q_bias: load(file, &format!("{p}self_attn.q_proj.bias"))?,
                k: load(file, &format!("{p}self_attn.k_proj.weight"))?,
                v: load(file, &format!("{p}self_attn.v_proj.weight"))?,
                v_bias: load(file, &format!("{p}self_attn.v_proj.bias"))?,
                out: load(file, &format!("{p}self_attn.out_proj.weight"))?,
                out_bias: load(file, &format!("{p}self_attn.out_proj.bias"))?,
                ln1_weight: load(file, &format!("{p}self_attn_layer_norm.weight"))?,
                ln1_bias: load(file, &format!("{p}self_attn_layer_norm.bias"))?,
                fc1: load(file, &format!("{p}fc1.weight"))?,
                fc1_bias: load(file, &format!("{p}fc1.bias"))?,
                fc2: load(file, &format!("{p}fc2.weight"))?,
                fc2_bias: load(file, &format!("{p}fc2.bias"))?,
                ln2_weight: load(file, &format!("{p}final_layer_norm.weight"))?,
                ln2_bias: load(file, &format!("{p}final_layer_norm.bias"))?,
            });
        }

        let mut dec_layers = Vec::with_capacity(config.decoder_layers());
        for i in 0..config.decoder_layers() {
            let p = format!("decoder.layers.{i}.");
            dec_layers.push(WhisperDecoderLayerOwned {
                self_q: load(file, &format!("{p}self_attn.q_proj.weight"))?,
                self_q_bias: load(file, &format!("{p}self_attn.q_proj.bias"))?,
                self_k: load(file, &format!("{p}self_attn.k_proj.weight"))?,
                self_v: load(file, &format!("{p}self_attn.v_proj.weight"))?,
                self_v_bias: load(file, &format!("{p}self_attn.v_proj.bias"))?,
                self_out: load(file, &format!("{p}self_attn.out_proj.weight"))?,
                self_out_bias: load(file, &format!("{p}self_attn.out_proj.bias"))?,
                ln_self_weight: load(file, &format!("{p}self_attn_layer_norm.weight"))?,
                ln_self_bias: load(file, &format!("{p}self_attn_layer_norm.bias"))?,
                cross_q: load(file, &format!("{p}encoder_attn.q_proj.weight"))?,
                cross_q_bias: load(file, &format!("{p}encoder_attn.q_proj.bias"))?,
                cross_k: load(file, &format!("{p}encoder_attn.k_proj.weight"))?,
                cross_v: load(file, &format!("{p}encoder_attn.v_proj.weight"))?,
                cross_v_bias: load(file, &format!("{p}encoder_attn.v_proj.bias"))?,
                cross_out: load(file, &format!("{p}encoder_attn.out_proj.weight"))?,
                cross_out_bias: load(file, &format!("{p}encoder_attn.out_proj.bias"))?,
                ln_cross_weight: load(file, &format!("{p}encoder_attn_layer_norm.weight"))?,
                ln_cross_bias: load(file, &format!("{p}encoder_attn_layer_norm.bias"))?,
                fc1: load(file, &format!("{p}fc1.weight"))?,
                fc1_bias: load(file, &format!("{p}fc1.bias"))?,
                fc2: load(file, &format!("{p}fc2.weight"))?,
                fc2_bias: load(file, &format!("{p}fc2.bias"))?,
                ln_fc_weight: load(file, &format!("{p}final_layer_norm.weight"))?,
                ln_fc_bias: load(file, &format!("{p}final_layer_norm.bias"))?,
            });
        }

        Ok(Self {
            conv1,
            conv1_bias,
            conv2,
            conv2_bias,
            enc_positions,
            enc_layers,
            enc_ln_weight,
            enc_ln_bias,
            embed_tokens,
            dec_positions,
            dec_layers,
            dec_ln_weight,
            dec_ln_bias,
        })
    }

    /// MLX openai-layout loader (`mlx-community/whisper-*` conversions).
    ///
    /// Names follow openai's original module tree (`encoder.blocks.{i}.
    /// attn.query`), the top-level config quantization block describes the
    /// groupwise affine scheme, and linear weights pack as U32 words with
    /// per-group F16 scales and biases. The dequant formula --
    /// `w = q * scale + bias` over unsigned bytes unpacked little-endian
    /// from each word -- was verified element-wise against the fp32
    /// checkpoint (mean abs error 1.5e-4); do not "fix" the sign without
    /// repeating that check.
    fn load_mlx(file: &SafetensorsFile, config: &WhisperConfig) -> Result<Self, ModelError> {
        let quant = config
            .quantization
            .unwrap_or(model_io::whisper_config::WhisperQuantization {
                bits: 8,
                group_size: 64,
            });
        // The two schemes the mlx-community whisper conversions ship, each
        // verified element-wise against the fp32 checkpoint before being
        // implemented: 8-bit groups of 64 (1.5e-4) and 4-bit groups of 64
        // (2.4e-3, which is 4-bit quantization noise, not error).
        if !((quant.bits == 8 || quant.bits == 4) && quant.group_size == 64) {
            return Err(ModelError::BadConfig {
                field: "quantization".to_string(),
                why: format!(
                    "the kernels implement 4-bit and 8-bit groups of 64; this conversion is {}-bit groups of {}",
                    quant.bits, quant.group_size
                ),
            });
        }
        let d = config.d_model;
        let shape_check = |name: &str, got: &[f32], want: usize| -> Result<(), ModelError> {
            if got.len() != want {
                return Err(ModelError::TensorSizeMismatch {
                    name: name.to_string(),
                    expected: want as u64,
                    actual: got.len() as u64,
                });
            }
            Ok(())
        };

        // One MLX tensor view: an f32 direct load or an 8-bit dequant.
        // `out_dim`/`in_dim` are the logical linear shapes [out, in].
        let linear = |base: &str,
                      out_dim: usize,
                      in_dim: usize|
         -> Result<(Vec<f32>, Vec<f32>), ModelError> {
            let weight_name = format!("{base}.weight");
            let bias = file
                .descriptor(&format!("{base}.bias"))
                .map(|_| file.load_as_f32(&format!("{base}.bias")))
                .transpose()?
                .unwrap_or_else(|| vec![0.0; out_dim]);
            if file.contains_tensor(&format!("{base}.scales")) {
                // Quantized: U32 words carry `32 / bits` values each,
                // little-endian; scales/biases are F16 [out, in / group].
                let packed_raw = file.raw_bytes(&weight_name)?;
                if packed_raw.len() % 4 != 0 {
                    return Err(ModelError::IndexCorrupt {
                        detail: format!("{weight_name}: unaligned U32 length"),
                    });
                }
                let words: Vec<u32> = packed_raw
                    .chunks_exact(4)
                    .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
                    .collect();
                let scales = file.load_as_f32(&format!("{base}.scales"))?;
                let biases = file.load_as_f32(&format!("{base}.biases"))?;
                let groups = in_dim / quant.group_size;
                if scales.len() != out_dim * groups || biases.len() != out_dim * groups {
                    return Err(ModelError::TensorSizeMismatch {
                        name: format!("{base}.scales"),
                        expected: (out_dim * groups) as u64,
                        actual: scales.len() as u64,
                    });
                }
                // Unpack each word into `per_word` values in value order:
                // bytes little-endian, and for 4-bit the LOW nibble of
                // each byte precedes the high nibble (both orders verified
                // against the fp32 checkpoint; the 8-bit byte walk is the
                // 4-bit sequence's every-other-element sibling).
                let per_word = 32 / quant.bits;
                if words.len() * per_word != out_dim * in_dim {
                    return Err(ModelError::TensorSizeMismatch {
                        name: weight_name,
                        expected: (out_dim * in_dim) as u64,
                        actual: (words.len() * per_word) as u64,
                    });
                }
                let values: Vec<f32> = words
                    .iter()
                    .flat_map(|&w| {
                        (0..per_word).map(move |k| {
                            if quant.bits == 8 {
                                f32::from((w >> (k * 8)) as u8)
                            } else {
                                let shift = k * 4;
                                f32::from(((w >> shift) & 0xF) as u8)
                            }
                        })
                    })
                    .collect();
                let mut out = vec![0.0f32; out_dim * in_dim];
                for r in 0..out_dim {
                    let row = &values[r * in_dim..(r + 1) * in_dim];
                    for g in 0..groups {
                        let s = scales[r * groups + g];
                        let b = biases[r * groups + g];
                        for k in 0..quant.group_size {
                            out[r * in_dim + g * quant.group_size + k] =
                                row[g * quant.group_size + k] * s + b;
                        }
                    }
                }
                Ok((out, bias))
            } else {
                // Unquantized linear (F16/F32).
                Ok((file.load_as_f32(&weight_name)?, bias))
            }
        };

        // A plain f32 vector (norms, positional tables).
        let vec = |name: &str| -> Result<Vec<f32>, ModelError> { file.load_as_f32(name) };

        // Conv weights ship F16 [out, 3, in]; the kernels read [out, in, 3].
        let conv_transpose = |flat: &[f32], out_dim: usize, in_dim: usize| -> Vec<f32> {
            let mut out = vec![0.0f32; flat.len()];
            for o in 0..out_dim {
                for k in 0..3 {
                    for i in 0..in_dim {
                        out[o * in_dim * 3 + i * 3 + k] = flat[o * 3 * in_dim + k * in_dim + i];
                    }
                }
            }
            out
        };

        let (conv1_t, conv1_bias) = linear("encoder.conv1", d, config.n_mels * 3)?;
        let conv1 = conv_transpose(&conv1_t, d, config.n_mels);
        shape_check("encoder.conv1", &conv1, d * config.n_mels * 3)?;
        let (conv2_t, conv2_bias) = linear("encoder.conv2", d, d * 3)?;
        let conv2 = conv_transpose(&conv2_t, d, d);
        shape_check("encoder.conv2", &conv2, d * d * 3)?;

        // MLX conversions omit the encoder's sinusoidal table (mlx-examples
        // computes it at runtime), so generate it with the reference
        // formula -- verified element-wise against the stored checkpoint
        // table (max abs diff 2.4e-4, f32 precision) before shipping here.
        let enc_positions = match file.load_as_f32("encoder.positional_embedding") {
            Ok(stored) => stored,
            Err(ModelError::TensorNotFound { .. }) => {
                compute::whisper::sinusoids(config.max_source_positions, d)
            }
            Err(error) => return Err(error),
        };
        shape_check(
            "encoder.positional_embedding",
            &enc_positions,
            config.max_source_positions * d,
        )?;
        let enc_ln_weight = vec("encoder.ln_post.weight")?;
        let enc_ln_bias = vec("encoder.ln_post.bias")?;
        // The token embedding is quantized like any other linear.
        let (embed_tokens, _) = linear("decoder.token_embedding", config.vocab_size, d)?;
        shape_check(
            "decoder.token_embedding",
            &embed_tokens,
            config.vocab_size * d,
        )?;
        let dec_positions = vec("decoder.positional_embedding")?;
        shape_check(
            "decoder.positional_embedding",
            &dec_positions,
            config.max_target_positions * d,
        )?;
        let dec_ln_weight = vec("decoder.ln.weight")?;
        let dec_ln_bias = vec("decoder.ln.bias")?;

        let mut enc_layers = Vec::with_capacity(config.encoder_layers());
        for i in 0..config.encoder_layers() {
            let p = format!("encoder.blocks.{i}");
            let (q, q_bias) = linear(&format!("{p}.attn.query"), d, d)?;
            let (v, v_bias) = linear(&format!("{p}.attn.value"), d, d)?;
            let (out, out_bias) = linear(&format!("{p}.attn.out"), d, d)?;
            // The MLP order is decided by shape: mlp1 maps d -> 4d.
            let (m1, m1b) = linear(&format!("{p}.mlp1"), 4 * d, d)?;
            let (m2, m2b) = linear(&format!("{p}.mlp2"), d, 4 * d)?;
            let (fc1, fc1_bias, fc2, fc2_bias) = if m1.len() == 4 * d * d {
                (m1, m1b, m2, m2b)
            } else {
                (m2, m2b, m1, m1b)
            };
            enc_layers.push(WhisperEncoderLayerOwned {
                q,
                q_bias,
                k: linear(&format!("{p}.attn.key"), d, d)?.0,
                v,
                v_bias,
                out,
                out_bias,
                ln1_weight: vec(&format!("{p}.attn_ln.weight"))?,
                ln1_bias: vec(&format!("{p}.attn_ln.bias"))?,
                fc1,
                fc1_bias,
                fc2,
                fc2_bias,
                ln2_weight: vec(&format!("{p}.mlp_ln.weight"))?,
                ln2_bias: vec(&format!("{p}.mlp_ln.bias"))?,
            });
        }

        let mut dec_layers = Vec::with_capacity(config.decoder_layers());
        for i in 0..config.decoder_layers() {
            let p = format!("decoder.blocks.{i}");
            let (sq, sqb) = linear(&format!("{p}.attn.query"), d, d)?;
            let (sv, svb) = linear(&format!("{p}.attn.value"), d, d)?;
            let (so, sob) = linear(&format!("{p}.attn.out"), d, d)?;
            let (cq, cqb) = linear(&format!("{p}.cross_attn.query"), d, d)?;
            let (cv, cvb) = linear(&format!("{p}.cross_attn.value"), d, d)?;
            let (co, cob) = linear(&format!("{p}.cross_attn.out"), d, d)?;
            let (m1, m1b) = linear(&format!("{p}.mlp1"), 4 * d, d)?;
            let (m2, m2b) = linear(&format!("{p}.mlp2"), d, 4 * d)?;
            let (fc1, fc1_bias, fc2, fc2_bias) = if m1.len() == 4 * d * d {
                (m1, m1b, m2, m2b)
            } else {
                (m2, m2b, m1, m1b)
            };
            dec_layers.push(WhisperDecoderLayerOwned {
                self_q: sq,
                self_q_bias: sqb,
                self_k: linear(&format!("{p}.attn.key"), d, d)?.0,
                self_v: sv,
                self_v_bias: svb,
                self_out: so,
                self_out_bias: sob,
                ln_self_weight: vec(&format!("{p}.attn_ln.weight"))?,
                ln_self_bias: vec(&format!("{p}.attn_ln.bias"))?,
                cross_q: cq,
                cross_q_bias: cqb,
                cross_k: linear(&format!("{p}.cross_attn.key"), d, d)?.0,
                cross_v: cv,
                cross_v_bias: cvb,
                cross_out: co,
                cross_out_bias: cob,
                ln_cross_weight: vec(&format!("{p}.cross_attn_ln.weight"))?,
                ln_cross_bias: vec(&format!("{p}.cross_attn_ln.bias"))?,
                fc1,
                fc1_bias,
                fc2,
                fc2_bias,
                ln_fc_weight: vec(&format!("{p}.mlp_ln.weight"))?,
                ln_fc_bias: vec(&format!("{p}.mlp_ln.bias"))?,
            });
        }

        Ok(Self {
            conv1,
            conv1_bias,
            conv2,
            conv2_bias,
            enc_positions,
            enc_layers,
            enc_ln_weight,
            enc_ln_bias,
            embed_tokens,
            dec_positions,
            dec_layers,
            dec_ln_weight,
            dec_ln_bias,
        })
    }
}
