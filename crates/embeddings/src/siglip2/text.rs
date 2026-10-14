//! Text tower (`Siglip2TextTransformer`).
//!
//! Token + learned absolute position embedding, the shared encoder with NO
//! attention mask, `final_layer_norm`, then the hidden state at the LAST
//! position through the biased `head` linear.
//!
//! Why no mask and why the last position: `Siglip2Processor` pads every
//! text to 64 tokens with `<pad>` (id 0) and its tokenizer declares
//! `model_input_names = ["input_ids"]`, so no `attention_mask` reaches the
//! model and pads attend and are attended to. The pooled state is
//! `last_hidden_state[:, -1]`, a pad slot for any text shorter than 63
//! tokens. This is the "sticky EOS" convention inherited from SigLIP1,
//! whose tokenizer padded with EOS so the last slot was always an EOS;
//! SigLIP2's Gemma tokenizer pads with `<pad> = 0` instead, and the model
//! was trained that way, so the last slot is used regardless.

use turbospark_audio::nn::{LayerNorm, Linear};

use super::config::TextConfig;
use super::encoder::{layer_norm, load_layers, Dims, EncoderLayer};
use crate::error::{input_error, tensor_error};
use crate::weights::{EmbeddingTable, Weights};
use crate::Result;

const PREFIX: &str = "text_model";
const TOKEN_TABLE: &str = "text_model.embeddings.token_embedding.weight";

/// Token table. The real vocabulary is 256000 x 768 f32 (786 MB), and a
/// forward reads 64 rows, so an F32 table is read row by row from the
/// mapped file instead of copied to the heap.
enum TokenTable {
    Mapped {
        weights: Weights,
        rows: usize,
        cols: usize,
    },
    Dense(EmbeddingTable),
}

impl std::fmt::Debug for TokenTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TokenTable::Mapped { rows, cols, .. } => write!(f, "Mapped({rows}x{cols})"),
            TokenTable::Dense(t) => write!(f, "Dense({}x{})", t.rows(), t.cols()),
        }
    }
}

impl TokenTable {
    fn gather(&self, ids: &[u32]) -> Result<Vec<f32>> {
        match self {
            TokenTable::Dense(t) => t.gather(ids),
            TokenTable::Mapped {
                weights,
                rows,
                cols,
            } => {
                let bytes = weights.file(TOKEN_TABLE)?.raw_bytes(TOKEN_TABLE)?;
                let mut out = Vec::with_capacity(ids.len() * cols);
                for &id in ids {
                    let r = id as usize;
                    if r >= *rows {
                        return Err(input_error(format!(
                            "token id {id} out of range for {rows} embedding rows"
                        )));
                    }
                    let row = &bytes[r * cols * 4..(r + 1) * cols * 4];
                    out.extend(
                        row.chunks_exact(4)
                            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])),
                    );
                }
                Ok(out)
            }
        }
    }
}

/// The text tower.
#[derive(Debug)]
pub struct TextTower {
    cfg: TextConfig,
    tokens: TokenTable,
    /// `[max_position_embeddings, hidden]`.
    position: Vec<f32>,
    layers: Vec<EncoderLayer>,
    final_layer_norm: LayerNorm,
    head: Linear,
}

/// Intermediate tensors of one forward.
#[derive(Debug, Clone)]
pub struct TextTrace {
    /// `[len, hidden]` token + position embedding.
    pub embeddings: Vec<f32>,
    /// `[len, hidden]` after the encoder and `final_layer_norm`.
    pub last_hidden: Vec<f32>,
    /// `[projection_size]` head output (unnormalized text embedding).
    pub pooled: Vec<f32>,
}

impl TextTower {
    /// Loads the tower, keeping `w` alive when the token table is read
    /// from the mapping.
    pub fn load(w: Weights, cfg: &TextConfig) -> Result<Self> {
        let h = cfg.hidden_size;
        let d = Dims {
            hidden: h,
            intermediate: cfg.intermediate_size,
            heads: cfg.num_attention_heads,
            eps: cfg.layer_norm_eps,
        };
        let shape = w.shape(TOKEN_TABLE)?;
        if shape != [cfg.vocab_size, h] {
            return Err(tensor_error(
                TOKEN_TABLE,
                format!("expected shape [{}, {h}], got {shape:?}", cfg.vocab_size),
            ));
        }
        let dtype = w
            .file(TOKEN_TABLE)?
            .descriptor(TOKEN_TABLE)
            .map(|d| d.dtype.to_uppercase())
            .unwrap_or_default();
        let position = w.load(
            &format!("{PREFIX}.embeddings.position_embedding.weight"),
            &[cfg.max_position_embeddings, h],
        )?;
        let layers = load_layers(&w, PREFIX, cfg.num_hidden_layers, d)?;
        let final_layer_norm = layer_norm(&w, &format!("{PREFIX}.final_layer_norm"), h, d.eps)?;
        let head = w.linear(
            &format!("{PREFIX}.head"),
            h,
            cfg.projection_size,
            true,
            None,
        )?;
        let tokens = if dtype == "F32" {
            let len = w.file(TOKEN_TABLE)?.raw_bytes(TOKEN_TABLE)?.len();
            if len != cfg.vocab_size * h * 4 {
                return Err(tensor_error(
                    TOKEN_TABLE,
                    format!("{len} bytes for an F32 table"),
                ));
            }
            TokenTable::Mapped {
                weights: w,
                rows: cfg.vocab_size,
                cols: h,
            }
        } else {
            let base = format!("{PREFIX}.embeddings.token_embedding");
            TokenTable::Dense(w.embedding(&base, cfg.vocab_size, h, None)?)
        };
        Ok(Self {
            tokens,
            position,
            layers,
            final_layer_norm,
            head,
            cfg: cfg.clone(),
        })
    }

    pub fn config(&self) -> &TextConfig {
        &self.cfg
    }

    /// Runs the tower over already padded ids and keeps every stage.
    pub fn forward_trace(&self, ids: &[u32]) -> Result<TextTrace> {
        let len = ids.len();
        if len == 0 || len > self.cfg.max_position_embeddings {
            return Err(input_error(format!(
                "{len} token ids; need 1..={}",
                self.cfg.max_position_embeddings
            )));
        }
        let h = self.cfg.hidden_size;
        let mut x = self.tokens.gather(ids)?;
        for (xi, pi) in x.iter_mut().zip(&self.position[..len * h]) {
            *xi += pi;
        }
        let embeddings = x.clone();
        for layer in &self.layers {
            layer.forward(&mut x, len);
        }
        self.final_layer_norm.apply(&mut x, len);
        let pooled = self.head.forward(&x[(len - 1) * h..], 1);
        Ok(TextTrace {
            embeddings,
            last_hidden: x,
            pooled,
        })
    }

    /// The unnormalized pooled text embedding.
    pub fn forward(&self, ids: &[u32]) -> Result<Vec<f32>> {
        Ok(self.forward_trace(ids)?.pooled)
    }
}
