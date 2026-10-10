//! The melotts BERT (`bert.py`): the bert-base transformer whose
//! third-to-last hidden state feeds the ja_bert features.
//!
//! Port of the reference `BertModel` at mlx-audio
//! [`e1b19b9054bf163f5d812221a54fcc346f1890e9`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/tts/models/melotts).
//! The shipped checkpoints store the converted bert-base weights under
//! `bert_weights.npz` (the reference remaps them into the MLX names this
//! loader reads: `embeddings.*`, `encoder.layers.{i}.*`, `pooler.*`).
//! The attention mask is additive like the reference: masked key
//! positions get `-inf` scores before the softmax (the batch-1
//! full-length path has no mask). Dropout is eval-identity and skipped.
//! The embedding LayerNorm uses the config `layer_norm_eps` (1e-12 for
//! bert-base); the per-layer `ln1`/`ln2` are plain `nn.LayerNorm`
//! (eps 1e-5).

use turbospark_model_io::safetensors::SafetensorsFile;

use crate::{ops, Result, SpeechError};

use super::BertDims;

/// The transformer layer LayerNorm eps (`nn.LayerNorm` default).
const LAYER_NORM_EPS: f32 = 1e-5;

/// The BERT encoder (`bert.py BertModel`).
pub struct BertModel {
    word_emb: Vec<f32>,
    token_type_emb: Vec<f32>,
    position_emb: Vec<f32>,
    norm_w: Vec<f32>,
    norm_b: Vec<f32>,
    embedding_eps: f32,
    layers: Vec<Layer>,
    pooler_w: Vec<f32>,
    pooler_b: Vec<f32>,
    max_position_embeddings: usize,
    num_heads: usize,
    hidden: usize,
}

struct Layer {
    query_w: Vec<f32>,
    query_b: Vec<f32>,
    key_w: Vec<f32>,
    key_b: Vec<f32>,
    value_w: Vec<f32>,
    value_b: Vec<f32>,
    out_w: Vec<f32>,
    out_b: Vec<f32>,
    ln1_w: Vec<f32>,
    ln1_b: Vec<f32>,
    linear1_w: Vec<f32>,
    linear1_b: Vec<f32>,
    mlp: usize,
    linear2_w: Vec<f32>,
    linear2_b: Vec<f32>,
    ln2_w: Vec<f32>,
    ln2_b: Vec<f32>,
}

impl Layer {
    /// Row-major `[tokens, hidden]` -> row-major `[tokens, hidden]`
    /// (post-norm transformer layer). `mask` marks padded key
    /// positions; their scores are `-inf` before the softmax.
    fn forward(
        &self,
        x: &[f32],
        tokens: usize,
        hidden: usize,
        num_heads: usize,
        mask: Option<&[u32]>,
    ) -> Vec<f32> {
        let head_dim = hidden / num_heads;
        let mut attn = vec![0.0f32; tokens * hidden];
        for head in 0..num_heads {
            let mut q = vec![0.0f32; tokens * head_dim];
            let mut k = vec![0.0f32; tokens * head_dim];
            let mut v = vec![0.0f32; tokens * head_dim];
            for (i, row) in x.chunks(hidden).enumerate() {
                for d in 0..head_dim {
                    let src = head * head_dim + d;
                    q[i * head_dim + d] = row
                        .iter()
                        .zip(self.query_w[src * hidden..(src + 1) * hidden].iter())
                        .map(|(a, w)| a * w)
                        .sum::<f32>()
                        + self.query_b[src];
                    k[i * head_dim + d] = row
                        .iter()
                        .zip(self.key_w[src * hidden..(src + 1) * hidden].iter())
                        .map(|(a, w)| a * w)
                        .sum::<f32>()
                        + self.key_b[src];
                    v[i * head_dim + d] = row
                        .iter()
                        .zip(self.value_w[src * hidden..(src + 1) * hidden].iter())
                        .map(|(a, w)| a * w)
                        .sum::<f32>()
                        + self.value_b[src];
                }
            }
            let scale = 1.0 / (head_dim as f32).sqrt();
            let mut scores = vec![0.0f32; tokens * tokens];
            for (i, row) in scores.chunks_mut(tokens).enumerate() {
                for (j, slot) in row.iter_mut().enumerate() {
                    *slot = q[i * head_dim..(i + 1) * head_dim]
                        .iter()
                        .zip(&k[j * head_dim..(j + 1) * head_dim])
                        .map(|(a, b)| a * b)
                        .sum::<f32>()
                        * scale;
                }
            }
            // The additive attention mask: masked key columns get -inf
            // (`mx.where(attention_mask == 0, -inf, 0.0)` in the
            // reference).
            if let Some(mask) = mask {
                for (row, masked) in scores.chunks_mut(tokens).zip(mask) {
                    if *masked == 0 {
                        for slot in row.iter_mut() {
                            *slot = f32::NEG_INFINITY;
                        }
                    }
                }
            }
            softmax_rows(&mut scores, tokens);
            let mut head_out = vec![0.0f32; tokens * head_dim];
            for i in 0..tokens {
                for d in 0..head_dim {
                    head_out[i * head_dim + d] = scores[i * tokens..(i + 1) * tokens]
                        .iter()
                        .enumerate()
                        .map(|(j, p)| p * v[j * head_dim + d])
                        .sum();
                }
            }
            for i in 0..tokens {
                for d in 0..head_dim {
                    attn[i * hidden + head * head_dim + d] = head_out[i * head_dim + d];
                }
            }
        }
        // out_proj + residual + ln1.
        let mut sum = vec![0.0f32; tokens * hidden];
        for (i, row) in attn.chunks(hidden).enumerate() {
            for d in 0..hidden {
                let projected: f32 = row
                    .iter()
                    .zip(self.out_w[d * hidden..(d + 1) * hidden].iter())
                    .map(|(a, w)| a * w)
                    .sum::<f32>()
                    + self.out_b[d];
                sum[i * hidden + d] = projected + x[i * hidden + d];
            }
        }
        ops::layernorm(
            &mut sum,
            tokens,
            hidden,
            &self.ln1_w,
            Some(&self.ln1_b),
            LAYER_NORM_EPS,
        );
        // FFN + residual + ln2.
        let mut ffn = vec![0.0f32; tokens * self.mlp];
        for (i, row) in sum.chunks(hidden).enumerate() {
            for d in 0..self.mlp {
                ffn[i * self.mlp + d] = row
                    .iter()
                    .zip(self.linear1_w[d * hidden..(d + 1) * hidden].iter())
                    .map(|(a, w)| a * w)
                    .sum::<f32>()
                    + self.linear1_b[d];
            }
        }
        ops::gelu_erf(&mut ffn);
        let mut out = vec![0.0f32; tokens * hidden];
        for (i, row) in ffn.chunks(self.mlp).enumerate() {
            for d in 0..hidden {
                out[i * hidden + d] = row
                    .iter()
                    .zip(self.linear2_w[d * self.mlp..(d + 1) * self.mlp].iter())
                    .map(|(a, w)| a * w)
                    .sum::<f32>()
                    + self.linear2_b[d]
                    + sum[i * hidden + d];
            }
        }
        ops::layernorm(
            &mut out,
            tokens,
            hidden,
            &self.ln2_w,
            Some(&self.ln2_b),
            LAYER_NORM_EPS,
        );
        out
    }
}

impl BertModel {
    /// Loads from the converted BERT weights with the MLX names
    /// (`embeddings.word_embeddings.weight`,
    /// `encoder.layers.0.attention.query_proj.weight`, ...).
    pub fn load(file: &SafetensorsFile, dims: &BertDims) -> Result<Self> {
        let read = |name: &str| -> Result<Vec<f32>> {
            file.load_as_f32(name).map_err(|e| SpeechError::Tensor {
                name: name.to_string(),
                why: e.to_string(),
            })
        };
        let mut layers = Vec::new();
        for i in 0..dims.num_hidden_layers {
            let base = format!("encoder.layers.{i}");
            let mlp = dims.intermediate_size;
            layers.push(Layer {
                query_w: read(&format!("{base}.attention.query_proj.weight"))?,
                query_b: read(&format!("{base}.attention.query_proj.bias"))?,
                key_w: read(&format!("{base}.attention.key_proj.weight"))?,
                key_b: read(&format!("{base}.attention.key_proj.bias"))?,
                value_w: read(&format!("{base}.attention.value_proj.weight"))?,
                value_b: read(&format!("{base}.attention.value_proj.bias"))?,
                out_w: read(&format!("{base}.attention.out_proj.weight"))?,
                out_b: read(&format!("{base}.attention.out_proj.bias"))?,
                ln1_w: read(&format!("{base}.ln1.weight"))?,
                ln1_b: read(&format!("{base}.ln1.bias"))?,
                linear1_w: read(&format!("{base}.linear1.weight"))?,
                linear1_b: read(&format!("{base}.linear1.bias"))?,
                mlp,
                linear2_w: read(&format!("{base}.linear2.weight"))?,
                linear2_b: read(&format!("{base}.linear2.bias"))?,
                ln2_w: read(&format!("{base}.ln2.weight"))?,
                ln2_b: read(&format!("{base}.ln2.bias"))?,
            });
        }
        Ok(BertModel {
            word_emb: read("embeddings.word_embeddings.weight")?,
            token_type_emb: read("embeddings.token_type_embeddings.weight")?,
            position_emb: read("embeddings.position_embeddings.weight")?,
            norm_w: read("embeddings.norm.weight")?,
            norm_b: read("embeddings.norm.bias")?,
            embedding_eps: dims.layer_norm_eps,
            layers,
            pooler_w: read("pooler.weight")?,
            pooler_b: read("pooler.bias")?,
            max_position_embeddings: dims.max_position_embeddings,
            num_heads: dims.num_attention_heads,
            hidden: dims.hidden_size,
        })
    }

    /// The hidden states after every layer, preceded by the embedding
    /// output (`output_hidden_states` in the reference). `mask` marks
    /// padded key positions (None on the full-length path).
    fn hidden_stack(
        &self,
        input_ids: &[u32],
        token_type_ids: &[u32],
        mask: Option<&[u32]>,
    ) -> Result<Vec<Vec<f32>>> {
        let n = input_ids.len();
        if n == 0 || n > self.max_position_embeddings {
            return Err(SpeechError::Input {
                why: "bert input must fit the position table".to_string(),
            });
        }
        let h = self.hidden;
        let mut x = vec![0.0f32; n * h];
        for (i, &id) in input_ids.iter().enumerate() {
            if id as usize * h >= self.word_emb.len() {
                return Err(SpeechError::Input {
                    why: "bert input ids do not fit the word embedding".to_string(),
                });
            }
            let token_type = *token_type_ids.first().unwrap_or(&0) as usize;
            if token_type * h >= self.token_type_emb.len() {
                return Err(SpeechError::Input {
                    why: "bert token type ids do not fit the type embedding".to_string(),
                });
            }
            for d in 0..h {
                x[i * h + d] = self.word_emb[id as usize * h + d]
                    + self.token_type_emb[token_type * h + d]
                    + self.position_emb[i * h + d];
            }
        }
        ops::layernorm(
            &mut x,
            n,
            h,
            &self.norm_w,
            Some(&self.norm_b),
            self.embedding_eps,
        );
        let mut stack = vec![x.clone()];
        for layer in &self.layers {
            x = layer.forward(&x, n, h, self.num_heads, mask);
            stack.push(x.clone());
        }
        Ok(stack)
    }

    /// The reference `extract_features`: the third-to-last hidden state
    /// (`all_hidden_states[-3]`), row-major `[tokens, hidden]`.
    pub fn extract_features(
        &self,
        input_ids: &[u32],
        token_type_ids: &[u32],
        mask: Option<&[u32]>,
    ) -> Result<Vec<f32>> {
        let stack = self.hidden_stack(input_ids, token_type_ids, mask)?;
        let index = stack
            .len()
            .checked_sub(3)
            .ok_or_else(|| SpeechError::Input {
                why: "bert needs at least three hidden states".to_string(),
            })?;
        Ok(stack[index].clone())
    }

    /// The reference `pooled`: `tanh(pooler(last_hidden[:, 0]))`.
    pub fn pooled(
        &self,
        input_ids: &[u32],
        token_type_ids: &[u32],
        mask: Option<&[u32]>,
    ) -> Result<Vec<f32>> {
        let stack = self.hidden_stack(input_ids, token_type_ids, mask)?;
        let last = stack.last().expect("at least the embedding output");
        let cls = &last[..self.hidden];
        let mut out = ops::linear(
            cls,
            &self.pooler_w,
            Some(&self.pooler_b),
            1,
            self.hidden,
            self.hidden,
        );
        for v in out.iter_mut() {
            *v = v.tanh();
        }
        Ok(out)
    }

    /// The token count the features will carry.
    pub fn tokens(&self, input_ids: &[u32]) -> usize {
        input_ids.len()
    }

    /// The reference `get_bert_features`: the token hidden states
    /// repeated per `word2ph` entry and transposed to the channel-major
    /// `[hidden, phones]` layout the ja_bert projection consumes.
    pub fn phone_features(
        &self,
        input_ids: &[u32],
        token_type_ids: &[u32],
        mask: Option<&[u32]>,
        word2ph: &[usize],
    ) -> Result<Vec<f32>> {
        let states = self.extract_features(input_ids, token_type_ids, mask)?;
        let hidden = self.hidden;
        let tokens = states.len() / hidden;
        if tokens != word2ph.len() {
            return Err(SpeechError::Input {
                why: format!(
                    "bert tokens ({tokens}) != word2ph length ({})",
                    word2ph.len()
                ),
            });
        }
        let total: usize = word2ph.iter().sum();
        let mut out = vec![0.0f32; hidden * total];
        let mut phone = 0usize;
        for (token, &count) in word2ph.iter().enumerate() {
            for _ in 0..count {
                for d in 0..hidden {
                    out[d * total + phone] = states[token * hidden + d];
                }
                phone += 1;
            }
        }
        Ok(out)
    }
}

fn softmax_rows(values: &mut [f32], columns: usize) {
    for row in values.chunks_mut(columns) {
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for v in row.iter_mut() {
            *v = (*v - max).exp();
            sum += *v;
        }
        for v in row.iter_mut() {
            *v /= sum;
        }
    }
}
