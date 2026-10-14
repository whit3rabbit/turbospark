//! RoBERTa text encoder (`ClapTextModel` with its pooler).
//!
//! Reference: transformers 4.57.1 `modeling_clap.py` (`ClapTextEmbeddings`,
//! `ClapTextLayer`, `ClapTextPooler`). Post-LayerNorm BERT layers with erf
//! GELU; the pooled output is `tanh(dense(h[0]))`, the `<s>` token, which
//! is what `ClapModel.get_text_features` projects.
//!
//! Position ids follow `create_position_ids_from_input_ids`: non-pad tokens
//! count from `padding_idx + 1`, and a pad id keeps position `padding_idx`
//! without advancing the counter. Embeddings are summed as
//! `(word + token_type) + position`, then LayerNorm.

use turbospark_audio::nn::{LayerNorm, Linear};
use turbospark_audio::ops;

use super::audio::layer_norm;
use super::config::TextConfig;
use crate::error::input_error;
use crate::weights::{EmbeddingTable, Weights};
use crate::Result;

struct TextLayer {
    query: Linear,
    key: Linear,
    value: Linear,
    attn_out: Linear,
    attn_ln: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    out_ln: LayerNorm,
}

/// The RoBERTa tower up to the pooled `[hidden_size]` output.
pub struct RobertaEncoder {
    cfg: TextConfig,
    word: EmbeddingTable,
    position: Vec<f32>,
    /// Row 0 of `token_type_embeddings`; CLAP never passes token types.
    token_type: Vec<f32>,
    emb_ln: LayerNorm,
    layers: Vec<TextLayer>,
    pooler: Linear,
}

impl std::fmt::Debug for RobertaEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RobertaEncoder")
            .field("layers", &self.layers.len())
            .field("hidden", &self.cfg.hidden_size)
            .finish_non_exhaustive()
    }
}

/// `create_position_ids_from_input_ids(ids, padding_idx)`.
pub fn position_ids(ids: &[u32], pad: u32) -> Vec<usize> {
    let mut count = 0usize;
    ids.iter()
        .map(|&id| {
            if id == pad {
                pad as usize
            } else {
                count += 1;
                count + pad as usize
            }
        })
        .collect()
}

impl RobertaEncoder {
    /// Loads `{prefix}embeddings`, `encoder.layer.*`, and `pooler`
    /// (prefix `text_model.` in a `ClapModel` checkpoint).
    pub fn load(w: &Weights, cfg: &TextConfig, prefix: &str) -> Result<Self> {
        cfg.validate()?;
        let h = cfg.hidden_size;
        let eps = cfg.layer_norm_eps as f32;
        let word = w.embedding(
            &format!("{prefix}embeddings.word_embeddings"),
            cfg.vocab_size,
            h,
            None,
        )?;
        let position = w.load(
            &format!("{prefix}embeddings.position_embeddings.weight"),
            &[cfg.max_position_embeddings, h],
        )?;
        let token_type = w.load(
            &format!("{prefix}embeddings.token_type_embeddings.weight"),
            &[cfg.type_vocab_size, h],
        )?[..h]
            .to_vec();
        let emb_ln = layer_norm(w, &format!("{prefix}embeddings.LayerNorm"), h, eps)?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let b = format!("{prefix}encoder.layer.{i}");
            let inter = cfg.intermediate_size;
            layers.push(TextLayer {
                query: w.linear(&format!("{b}.attention.self.query"), h, h, true, None)?,
                key: w.linear(&format!("{b}.attention.self.key"), h, h, true, None)?,
                value: w.linear(&format!("{b}.attention.self.value"), h, h, true, None)?,
                attn_out: w.linear(&format!("{b}.attention.output.dense"), h, h, true, None)?,
                attn_ln: layer_norm(w, &format!("{b}.attention.output.LayerNorm"), h, eps)?,
                fc1: w.linear(&format!("{b}.intermediate.dense"), h, inter, true, None)?,
                fc2: w.linear(&format!("{b}.output.dense"), inter, h, true, None)?,
                out_ln: layer_norm(w, &format!("{b}.output.LayerNorm"), h, eps)?,
            });
        }
        let pooler = w.linear(&format!("{prefix}pooler.dense"), h, h, true, None)?;
        Ok(Self {
            cfg: cfg.clone(),
            word,
            position,
            token_type,
            emb_ln,
            layers,
            pooler,
        })
    }

    pub fn config(&self) -> &TextConfig {
        &self.cfg
    }

    /// Pooled `[hidden_size]` output for one unpadded id sequence (with
    /// its `<s>` / `</s>`). `hook` sees `embeddings`, `layer{i}`, and
    /// `pooled`.
    ///
    /// The reference batches padded sequences under an attention mask;
    /// masked keys get `finfo.min` and contribute exactly zero, so one
    /// unpadded sequence gives the same result up to rounding.
    pub fn forward_with(
        &self,
        ids: &[u32],
        hook: &mut dyn FnMut(&str, &[f32]),
    ) -> Result<Vec<f32>> {
        let cfg = &self.cfg;
        let h = cfg.hidden_size;
        let n = ids.len();
        if n == 0 {
            return Err(input_error("empty token sequence"));
        }
        if n > cfg.max_tokens() {
            // The reference indexes past the position table and fails.
            return Err(input_error(format!(
                "{n} tokens exceed the {} the position table covers",
                cfg.max_tokens()
            )));
        }
        let mut x = self.word.gather(ids)?;
        for (row, pos) in x
            .chunks_exact_mut(h)
            .zip(position_ids(ids, cfg.pad_token_id))
        {
            let p = &self.position[pos * h..(pos + 1) * h];
            for ((v, t), q) in row.iter_mut().zip(&self.token_type).zip(p) {
                *v = (*v + t) + q;
            }
        }
        self.emb_ln.apply(&mut x, n);
        hook("embeddings", &x);

        let heads = cfg.num_attention_heads;
        let hd = h / heads;
        // HF: attention_head_size ** -0.5, multiplied into the scores.
        let scale = (hd as f64).powf(-0.5) as f32;
        for (i, layer) in self.layers.iter().enumerate() {
            let q = ops::split_heads(&layer.query.forward(&x, n), n, heads, hd);
            let k = ops::split_heads(&layer.key.forward(&x, n), n, heads, hd);
            let v = ops::split_heads(&layer.value.forward(&x, n), n, heads, hd);
            let ctx = ops::merge_heads(
                &ops::mha(&q, &k, &v, None, heads, n, n, hd, scale),
                n,
                heads,
                hd,
            );
            let mut a = layer.attn_out.forward(&ctx, n);
            for (o, r) in a.iter_mut().zip(&x) {
                *o += r;
            }
            layer.attn_ln.apply(&mut a, n);
            let mut inter = layer.fc1.forward(&a, n);
            ops::gelu_erf(&mut inter);
            let mut out = layer.fc2.forward(&inter, n);
            for (o, r) in out.iter_mut().zip(&a) {
                *o += r;
            }
            layer.out_ln.apply(&mut out, n);
            x = out;
            hook(&format!("layer{i}"), &x);
        }
        let mut pooled = self.pooler.forward(&x[..h], 1);
        for v in pooled.iter_mut() {
            *v = v.tanh();
        }
        hook("pooled", &pooled);
        Ok(pooled)
    }
}

#[cfg(test)]
mod unit {
    use super::*;

    #[test]
    fn positions_start_after_padding_idx() {
        assert_eq!(position_ids(&[0, 5, 9, 2], 1), vec![2, 3, 4, 5]);
        // A pad id keeps position padding_idx and does not advance.
        assert_eq!(position_ids(&[0, 1, 9, 2], 1), vec![2, 1, 3, 4]);
    }
}
