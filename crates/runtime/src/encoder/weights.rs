//! Encoder weights container and loader from safetensors.
//!
//! Handles both unquantized BF16 weights (e.g. BGE-small) and MLX 8-bit affine
//! quantized weights (e.g. Snowflake Arctic Embed).

use compute::quant::{dequantize_int8_affine, Int8AffineRow};
use compute::EncoderLayerWeights;
use model_io::encoder_config::EncoderConfig;
use model_io::safetensors::SafetensorsFile;
use model_io::ModelError;

/// Owned weights for one encoder layer in FP32.
pub struct EncoderLayerWeightsOwned {
    pub q_weight: Vec<f32>,
    pub q_bias: Vec<f32>,
    pub k_weight: Vec<f32>,
    pub k_bias: Vec<f32>,
    pub v_weight: Vec<f32>,
    pub v_bias: Vec<f32>,
    pub out_weight: Vec<f32>,
    pub out_bias: Vec<f32>,
    pub attn_ln_weight: Vec<f32>,
    pub attn_ln_bias: Vec<f32>,
    pub intermediate_weight: Vec<f32>,
    pub intermediate_bias: Vec<f32>,
    pub mlp_out_weight: Vec<f32>,
    pub mlp_out_bias: Vec<f32>,
    pub mlp_ln_weight: Vec<f32>,
    pub mlp_ln_bias: Vec<f32>,
}

impl EncoderLayerWeightsOwned {
    /// Borrow as compute references.
    pub fn as_borrowed(&self) -> EncoderLayerWeights<'_> {
        EncoderLayerWeights {
            q_weight: &self.q_weight,
            q_bias: &self.q_bias,
            k_weight: &self.k_weight,
            k_bias: &self.k_bias,
            v_weight: &self.v_weight,
            v_bias: &self.v_bias,
            out_weight: &self.out_weight,
            out_bias: &self.out_bias,
            attn_ln_weight: &self.attn_ln_weight,
            attn_ln_bias: &self.attn_ln_bias,
            intermediate_weight: &self.intermediate_weight,
            intermediate_bias: &self.intermediate_bias,
            mlp_out_weight: &self.mlp_out_weight,
            mlp_out_bias: &self.mlp_out_bias,
            mlp_ln_weight: &self.mlp_ln_weight,
            mlp_ln_bias: &self.mlp_ln_bias,
        }
    }
}

/// Complete weights for an encoder model.
pub struct EncoderWeights {
    pub word_embeddings: Vec<f32>,
    pub position_embeddings: Vec<f32>,
    pub token_type_embeddings: Option<Vec<f32>>,
    pub emb_ln_weight: Vec<f32>,
    pub emb_ln_bias: Vec<f32>,
    pub layers: Vec<EncoderLayerWeightsOwned>,
}

impl EncoderWeights {
    /// Load weights from a safetensors file according to config.
    pub fn load_from_safetensors(
        file: &SafetensorsFile,
        config: &EncoderConfig,
    ) -> Result<Self, ModelError> {
        let is_int8 = config.is_int8_quantized();

        // Helper to load either unquantized or 8-bit quantized linear weights
        let load_matrix = |name: &str| -> Result<Vec<f32>, ModelError> {
            let weight_name = format!("{name}.weight");
            if is_int8 && file.contains_tensor(&format!("{name}.scales")) {
                let packed = file.raw_bytes(&weight_name)?;
                let scales_u16 = file.load_as_u16(&format!("{name}.scales"))?;
                let biases_u16 = file.load_as_u16(&format!("{name}.biases"))?;
                let row = Int8AffineRow {
                    packed: packed.to_vec(),
                    scales: scales_u16,
                    biases: biases_u16,
                };
                Ok(dequantize_int8_affine(&row, packed.len()))
            } else {
                file.load_as_f32(&weight_name)
            }
        };

        // Helper to load bias (defaults to zeros if not found)
        let load_bias = |name: &str, dim: usize| -> Vec<f32> {
            let bias_name = format!("{name}.bias");
            file.load_as_f32(&bias_name)
                .unwrap_or_else(|_| vec![0.0f32; dim])
        };

        // 1. Load Embeddings
        let word_embeddings = load_matrix("embeddings.word_embeddings")
            .or_else(|_| load_matrix("word_embeddings"))?;
        let position_embeddings = load_matrix("embeddings.position_embeddings")
            .or_else(|_| load_matrix("position_embeddings"))?;
        let token_type_embeddings = if config.type_vocab_size > 0 {
            load_matrix("embeddings.token_type_embeddings")
                .or_else(|_| load_matrix("token_type_embeddings"))
                .ok()
        } else {
            None
        };

        let emb_ln_weight = file
            .load_as_f32("embeddings.LayerNorm.weight")
            .or_else(|_| file.load_as_f32("LayerNorm.weight"))?;
        let emb_ln_bias = file
            .load_as_f32("embeddings.LayerNorm.bias")
            .or_else(|_| file.load_as_f32("LayerNorm.bias"))?;

        // 2. Load Layers
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let prefix = format!("encoder.layer.{i}");

            let q_weight = load_matrix(&format!("{prefix}.attention.self.query"))?;
            let q_bias = load_bias(
                &format!("{prefix}.attention.self.query"),
                config.hidden_size,
            );

            let k_weight = load_matrix(&format!("{prefix}.attention.self.key"))?;
            let k_bias = load_bias(&format!("{prefix}.attention.self.key"), config.hidden_size);

            let v_weight = load_matrix(&format!("{prefix}.attention.self.value"))?;
            let v_bias = load_bias(
                &format!("{prefix}.attention.self.value"),
                config.hidden_size,
            );

            let out_weight = load_matrix(&format!("{prefix}.attention.output.dense"))?;
            let out_bias = load_bias(
                &format!("{prefix}.attention.output.dense"),
                config.hidden_size,
            );

            let attn_ln_weight =
                file.load_as_f32(&format!("{prefix}.attention.output.LayerNorm.weight"))?;
            let attn_ln_bias =
                file.load_as_f32(&format!("{prefix}.attention.output.LayerNorm.bias"))?;

            let intermediate_weight = load_matrix(&format!("{prefix}.intermediate.dense"))?;
            let intermediate_bias = load_bias(
                &format!("{prefix}.intermediate.dense"),
                config.intermediate_size,
            );

            let mlp_out_weight = load_matrix(&format!("{prefix}.output.dense"))?;
            let mlp_out_bias = load_bias(&format!("{prefix}.output.dense"), config.hidden_size);

            let mlp_ln_weight = file.load_as_f32(&format!("{prefix}.output.LayerNorm.weight"))?;
            let mlp_ln_bias = file.load_as_f32(&format!("{prefix}.output.LayerNorm.bias"))?;

            layers.push(EncoderLayerWeightsOwned {
                q_weight,
                q_bias,
                k_weight,
                k_bias,
                v_weight,
                v_bias,
                out_weight,
                out_bias,
                attn_ln_weight,
                attn_ln_bias,
                intermediate_weight,
                intermediate_bias,
                mlp_out_weight,
                mlp_out_bias,
                mlp_ln_weight,
                mlp_ln_bias,
            });
        }

        Ok(Self {
            word_embeddings,
            position_embeddings,
            token_type_embeddings,
            emb_ln_weight,
            emb_ln_bias,
            layers,
        })
    }
}
