//! FP32 reference kernels for BERT and XLM-RoBERTa encoder models.
//!
//! Ground truth for encoder execution, following the Post-LN transformer
//! specification used by BGE and Snowflake Arctic Embed:
//! - Learned 1D position embeddings (with optional padding offset for RoBERTa)
//! - Word, position, and token-type embedding summation followed by LayerNorm
//! - Multi-head bidirectional self-attention with linear bias
//! - Post-attention LayerNorm: `LN(x + SelfAttention(x))`
//! - Intermediate dense projection + GELU + output dense projection
//! - Post-FFN LayerNorm: `LN(x + MLP(x))`
//! - CLS token pooling (row 0) followed by Euclidean L2 normalization

use crate::moe::gelu_tanh;
use crate::vision::{bidirectional_attention, gelu_erf, layer_norm, matmul_bias};

/// Configuration parameters for an encoder forward pass.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EncoderReferenceConfig {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub layer_norm_eps: f32,
    pub position_offset: usize,
    pub use_tanh_gelu: bool,
}

impl EncoderReferenceConfig {
    /// Configuration for BAAI/bge-small-en-v1.5.
    pub fn bge_small() -> Self {
        Self {
            hidden_size: 384,
            num_attention_heads: 12,
            intermediate_size: 1536,
            layer_norm_eps: 1e-12,
            position_offset: 0,
            use_tanh_gelu: false,
        }
    }

    /// Configuration for snowflake-arctic-embed-l-v2.0.
    pub fn arctic_embed_l() -> Self {
        Self {
            hidden_size: 1024,
            num_attention_heads: 16,
            intermediate_size: 4096,
            layer_norm_eps: 1e-5,
            position_offset: 2, // pad_token_id (1) + 1
            use_tanh_gelu: false,
        }
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

/// Compute input embeddings for a token sequence:
/// `LayerNorm(word_emb[token_id] + pos_emb[pos + offset] + type_emb[type_id])`.
#[allow(clippy::too_many_arguments)]
pub fn encoder_embeddings_lookup(
    input_ids: &[u32],
    token_type_ids: Option<&[u32]>,
    word_embeddings: &[f32],
    position_embeddings: &[f32],
    token_type_embeddings: Option<&[f32]>,
    gamma: &[f32],
    beta: &[f32],
    config: &EncoderReferenceConfig,
) -> Vec<f32> {
    let seq = input_ids.len();
    let hidden = config.hidden_size;
    let mut out = vec![0.0f32; seq * hidden];

    for (pos, &token_id) in input_ids.iter().enumerate() {
        let word_start = token_id as usize * hidden;
        let pos_start = (pos + config.position_offset) * hidden;
        let type_id = token_type_ids.map_or(0, |t| t[pos] as usize);
        let type_start = type_id * hidden;

        assert!(
            word_start + hidden <= word_embeddings.len(),
            "token_id {token_id} (offset {word_start}) exceeds word_embeddings len {}",
            word_embeddings.len()
        );
        assert!(
            pos_start + hidden <= position_embeddings.len(),
            "position {pos} (offset {pos_start}) exceeds position_embeddings len {}",
            position_embeddings.len()
        );

        let row_start = pos * hidden;
        for i in 0..hidden {
            let mut val = word_embeddings[word_start + i] + position_embeddings[pos_start + i];
            if let Some(type_table) = token_type_embeddings {
                val += type_table[type_start + i];
            }
            out[row_start + i] = val;
        }

        // Apply LayerNorm to this token row
        let normed = layer_norm(
            &out[row_start..row_start + hidden],
            gamma,
            beta,
            config.layer_norm_eps,
        );
        out[row_start..row_start + hidden].copy_from_slice(&normed);
    }

    out
}

/// Weights for one encoder layer.
pub struct EncoderLayerWeights<'a> {
    pub q_weight: &'a [f32],
    pub q_bias: &'a [f32],
    pub k_weight: &'a [f32],
    pub k_bias: &'a [f32],
    pub v_weight: &'a [f32],
    pub v_bias: &'a [f32],
    pub out_weight: &'a [f32],
    pub out_bias: &'a [f32],
    pub attn_ln_weight: &'a [f32],
    pub attn_ln_bias: &'a [f32],
    pub intermediate_weight: &'a [f32],
    pub intermediate_bias: &'a [f32],
    pub mlp_out_weight: &'a [f32],
    pub mlp_out_bias: &'a [f32],
    pub mlp_ln_weight: &'a [f32],
    pub mlp_ln_bias: &'a [f32],
}

/// Run one Post-LN encoder block:
/// 1. Self-attention with Q/K/V linear projections + additive biases
/// 2. Residual connection + LayerNorm: `h = LN(x + Attention(x))`
/// 3. MLP: `GELU(h * W_inter + b_inter) * W_out + b_out`
/// 4. Residual connection + LayerNorm: `y = LN(h + MLP(h))`
pub fn encoder_block_forward(
    x: &[f32],
    seq: usize,
    weights: &EncoderLayerWeights,
    config: &EncoderReferenceConfig,
) -> Vec<f32> {
    let hidden = config.hidden_size;
    let heads = config.num_attention_heads;
    let head_dim = config.head_dim();
    let inter_dim = config.intermediate_size;
    let scale = (head_dim as f32).powf(-0.5);

    // Q, K, V projections
    let q = matmul_bias(
        x,
        weights.q_weight,
        Some(weights.q_bias),
        seq,
        hidden,
        hidden,
    );
    let k = matmul_bias(
        x,
        weights.k_weight,
        Some(weights.k_bias),
        seq,
        hidden,
        hidden,
    );
    let v = matmul_bias(
        x,
        weights.v_weight,
        Some(weights.v_bias),
        seq,
        hidden,
        hidden,
    );

    // Bidirectional attention
    let attn_scores = bidirectional_attention(&q, &k, &v, seq, heads, head_dim, scale);

    // Attention output projection
    let attn_dense = matmul_bias(
        &attn_scores,
        weights.out_weight,
        Some(weights.out_bias),
        seq,
        hidden,
        hidden,
    );

    // Residual add + LayerNorm (Post-LN)
    let mut post_attn = vec![0.0f32; seq * hidden];
    for pos in 0..seq {
        let row_start = pos * hidden;
        let mut row_sum = vec![0.0f32; hidden];
        for i in 0..hidden {
            row_sum[i] = x[row_start + i] + attn_dense[row_start + i];
        }
        let normed = layer_norm(
            &row_sum,
            weights.attn_ln_weight,
            weights.attn_ln_bias,
            config.layer_norm_eps,
        );
        post_attn[row_start..row_start + hidden].copy_from_slice(&normed);
    }

    // Intermediate dense projection
    let intermediate = matmul_bias(
        &post_attn,
        weights.intermediate_weight,
        Some(weights.intermediate_bias),
        seq,
        hidden,
        inter_dim,
    );

    // GELU activation
    let activated = if config.use_tanh_gelu {
        gelu_tanh(&intermediate)
    } else {
        gelu_erf(&intermediate)
    };

    // Output dense projection
    let mlp_dense = matmul_bias(
        &activated,
        weights.mlp_out_weight,
        Some(weights.mlp_out_bias),
        seq,
        inter_dim,
        hidden,
    );

    // Residual add + LayerNorm (Post-LN)
    let mut out = vec![0.0f32; seq * hidden];
    for pos in 0..seq {
        let row_start = pos * hidden;
        let mut row_sum = vec![0.0f32; hidden];
        for i in 0..hidden {
            row_sum[i] = post_attn[row_start + i] + mlp_dense[row_start + i];
        }
        let normed = layer_norm(
            &row_sum,
            weights.mlp_ln_weight,
            weights.mlp_ln_bias,
            config.layer_norm_eps,
        );
        out[row_start..row_start + hidden].copy_from_slice(&normed);
    }

    out
}

/// CLS token pooling: extracts token 0 (row 0) from the final hidden state
/// and normalizes it to unit Euclidean L2 norm: `v / ||v||_2`.
pub fn cls_pool_and_normalize(hidden_states: &[f32], hidden_size: usize) -> Vec<f32> {
    assert!(
        hidden_states.len() >= hidden_size,
        "hidden_states must have at least one token row"
    );
    let cls_token = &hidden_states[0..hidden_size];

    let sum_sq: f32 = cls_token.iter().map(|&v| v * v).sum();
    let norm = sum_sq.sqrt().max(1e-12);
    let inv_norm = 1.0 / norm;

    cls_token.iter().map(|&v| v * inv_norm).collect()
}

/// Cosine similarity between two vectors of equal length. Both sides are
/// L2-normalized first, so arbitrary nonzero inputs give a true cosine in
/// [-1, 1]; a zero vector has no direction and yields 0.0. Vectors from
/// [`cls_pool_and_normalize`] are already unit, so for encoder embeddings
/// this stays the dot product it always was.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "embeddings must match dimension");
    let norm_sq = |v: &[f32]| v.iter().map(|&x| x * x).sum::<f32>();
    let (na, nb) = (norm_sq(a), norm_sq(b));
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    a.iter().zip(b).map(|(&x, &y)| x * y).sum::<f32>() / na.sqrt() / nb.sqrt()
}
