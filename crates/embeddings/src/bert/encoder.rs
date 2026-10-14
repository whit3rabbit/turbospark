//! BERT / XLM-RoBERTa encoder (`BertModel`, `XLMRobertaModel` without the
//! pooler).
//!
//! Reference: transformers 4.57.1 `models/bert/modeling_bert.py` and
//! `models/xlm_roberta/modeling_xlm_roberta.py` (sdpa path):
//!
//! ```text
//! x   = LayerNorm((word[id] + token_type[0]) + position[pos])
//! a   = LayerNorm(attn_out(mha(x)) + x)          (BertSelfOutput)
//! out = LayerNorm(fc2(gelu(fc1(a))) + a)          (BertIntermediate/Output)
//! ```
//!
//! Post-LN, biased projections, attention scale `1/sqrt(head_dim)` (the
//! sdpa default), erf GELU. Token type is row 0: the BERT tokenizer emits
//! all-zero `token_type_ids` for one segment and XLM-R falls back to the
//! zero buffer. Positions are `0..n` for BERT and the
//! `create_position_ids_from_input_ids` rule for XLM-R (see
//! [`BertConfig::position_ids`]).
//!
//! A padding mask, when given, is the additive key mask transformers builds
//! for sdpa (`finfo(f32).min` on masked keys, shared by every query row and
//! head); padded query rows are still computed, as in the reference.
//!
//! Token tables: an F32 or BF16 table is gathered row by row from the
//! mapped safetensors file, so arctic-embed-l-v2.0's 250002 x 1024 table
//! (1 GB as f32) never lands on the heap. MLX-quantized tables stay packed
//! ([`EmbeddingTable::Quantized`]). Only an F16 table is widened at load.

use turbospark_audio::nn::{LayerNorm, Linear};
use turbospark_audio::ops;

use super::config::BertConfig;
use crate::error::{bad_config, input_error, tensor_error};
use crate::weights::{EmbeddingTable, Weights};
use crate::Result;

/// Stored element type of a table read from the mapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RowDtype {
    F32,
    Bf16,
}

/// An embedding table (word, position, or token type).
#[derive(Debug)]
enum Table {
    /// Rows decoded from the mapped file on every gather.
    Mapped {
        name: String,
        rows: usize,
        cols: usize,
        dtype: RowDtype,
    },
    /// Dense on the heap (F16) or MLX-packed.
    Loaded(EmbeddingTable),
}

impl Table {
    fn load(w: &Weights, base: &str, rows: usize, cols: usize, c: &BertConfig) -> Result<Self> {
        let name = format!("{base}.weight");
        if w.is_quantized(base) {
            return Ok(Table::Loaded(w.embedding(
                base,
                rows,
                cols,
                c.quantization,
            )?));
        }
        let shape = w.shape(&name)?;
        if shape != [rows, cols] {
            return Err(tensor_error(
                &name,
                format!("expected shape [{rows}, {cols}], got {shape:?}"),
            ));
        }
        let dtype = w
            .file(&name)?
            .descriptor(&name)
            .map(|d| d.dtype.to_uppercase())
            .unwrap_or_default();
        let (dtype, width) = match dtype.as_str() {
            "F32" => (RowDtype::F32, 4),
            "BF16" => (RowDtype::Bf16, 2),
            _ => return Ok(Table::Loaded(w.embedding(base, rows, cols, None)?)),
        };
        let len = w.file(&name)?.raw_bytes(&name)?.len();
        if len != rows * cols * width {
            return Err(tensor_error(
                &name,
                format!("{len} bytes for {dtype:?} [{rows}, {cols}]"),
            ));
        }
        Ok(Table::Mapped {
            name,
            rows,
            cols,
            dtype,
        })
    }

    fn gather(&self, w: &Weights, ids: &[u32]) -> Result<Vec<f32>> {
        let (name, rows, cols, dtype) = match self {
            Table::Loaded(t) => return t.gather(ids),
            Table::Mapped {
                name,
                rows,
                cols,
                dtype,
            } => (name, *rows, *cols, *dtype),
        };
        let bytes = w.file(name)?.raw_bytes(name)?;
        let mut out = Vec::with_capacity(ids.len() * cols);
        for &id in ids {
            let r = id as usize;
            if r >= rows {
                return Err(input_error(format!(
                    "{name}: index {id} out of range for {rows} rows"
                )));
            }
            match dtype {
                RowDtype::F32 => out.extend(
                    bytes[r * cols * 4..(r + 1) * cols * 4]
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])),
                ),
                // BF16 is the top half of an f32: widening is exact.
                RowDtype::Bf16 => out.extend(
                    bytes[r * cols * 2..(r + 1) * cols * 2]
                        .chunks_exact(2)
                        .map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16)),
                ),
            }
        }
        Ok(out)
    }

    fn is_mapped(&self) -> bool {
        matches!(self, Table::Mapped { .. })
    }
}

fn layer_norm(w: &Weights, base: &str, width: usize, eps: f32) -> Result<LayerNorm> {
    Ok(LayerNorm::new(
        w.load(&format!("{base}.weight"), &[width])?,
        Some(w.load(&format!("{base}.bias"), &[width])?),
        eps,
    ))
}

#[derive(Debug, Clone)]
struct Layer {
    query: Linear,
    key: Linear,
    value: Linear,
    attn_out: Linear,
    attn_ln: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    out_ln: LayerNorm,
}

/// The encoder up to `last_hidden_state`.
pub struct Encoder {
    config: BertConfig,
    /// Kept open so mapped tables can be read at inference.
    weights: Weights,
    word: Table,
    position: Table,
    /// Row 0 of `token_type_embeddings`.
    token_type: Vec<f32>,
    emb_ln: LayerNorm,
    layers: Vec<Layer>,
}

impl std::fmt::Debug for Encoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Encoder")
            .field("arch", &self.config.arch)
            .field("layers", &self.layers.len())
            .field("hidden", &self.config.hidden_size)
            .field("word", &self.word)
            .finish_non_exhaustive()
    }
}

impl Encoder {
    /// Loads `embeddings.*` and `encoder.layer.*` (no prefix, as in every
    /// pinned checkpoint and MLX conversion). `pooler.*` is ignored:
    /// sentence-transformers pools `last_hidden_state`.
    pub fn load(weights: Weights, config: &BertConfig) -> Result<Self> {
        let c = config;
        let (h, inter) = (c.hidden_size, c.intermediate_size);
        let eps = c.layer_norm_eps;
        let q = c.quantization;
        let w = &weights;
        if !w.contains("embeddings.word_embeddings.weight") {
            return Err(tensor_error(
                "embeddings.word_embeddings.weight",
                "missing (a prefixed BertModel/RobertaModel layout is not supported)",
            ));
        }
        let extra = format!(
            "encoder.layer.{}.attention.self.query.weight",
            c.num_hidden_layers
        );
        if w.contains(&extra) {
            return Err(bad_config(
                "num_hidden_layers",
                format!("{} but the checkpoint has {extra}", c.num_hidden_layers),
            ));
        }
        let word = Table::load(w, "embeddings.word_embeddings", c.vocab_size, h, c)?;
        let position = Table::load(
            w,
            "embeddings.position_embeddings",
            c.max_position_embeddings,
            h,
            c,
        )?;
        let token_type = Table::load(
            w,
            "embeddings.token_type_embeddings",
            c.type_vocab_size,
            h,
            c,
        )?
        .gather(w, &[0])?;
        let emb_ln = layer_norm(w, "embeddings.LayerNorm", h, eps)?;
        let mut layers = Vec::with_capacity(c.num_hidden_layers);
        for i in 0..c.num_hidden_layers {
            let b = format!("encoder.layer.{i}");
            let lin = |n: &str, input: usize, output: usize| {
                w.linear(&format!("{b}.{n}"), input, output, true, q)
            };
            layers.push(Layer {
                query: lin("attention.self.query", h, h)?,
                key: lin("attention.self.key", h, h)?,
                value: lin("attention.self.value", h, h)?,
                attn_out: lin("attention.output.dense", h, h)?,
                attn_ln: layer_norm(w, &format!("{b}.attention.output.LayerNorm"), h, eps)?,
                fc1: lin("intermediate.dense", h, inter)?,
                fc2: lin("output.dense", inter, h)?,
                out_ln: layer_norm(w, &format!("{b}.output.LayerNorm"), h, eps)?,
            });
        }
        Ok(Self {
            config: config.clone(),
            word,
            position,
            token_type,
            emb_ln,
            layers,
            weights,
        })
    }

    pub fn config(&self) -> &BertConfig {
        &self.config
    }

    /// True when the word table is read from the mapping (F32/BF16).
    pub fn word_table_is_mapped(&self) -> bool {
        self.word.is_mapped()
    }

    /// `last_hidden_state [ids.len(), hidden]`, every token attended.
    pub fn forward(&self, ids: &[u32]) -> Result<Vec<f32>> {
        self.forward_traced(ids, None, |_, _| {})
    }

    /// [`forward`](Self::forward) under an attention mask (`1` real, `0`
    /// padding), handing `trace` the states HF returns as
    /// `hidden_states`: index 0 is the embedding output (after its
    /// LayerNorm), index `i` the output of layer `i - 1`. The last one is
    /// `last_hidden_state` and is returned.
    pub fn forward_traced(
        &self,
        ids: &[u32],
        mask: Option<&[u32]>,
        mut trace: impl FnMut(usize, &[f32]),
    ) -> Result<Vec<f32>> {
        let c = &self.config;
        let n = ids.len();
        if n == 0 {
            return Err(input_error("empty token sequence"));
        }
        if let Some(m) = mask {
            if m.len() != n {
                return Err(input_error(format!("mask length {} != {n} ids", m.len())));
            }
            if m.iter().all(|&v| v == 0) {
                return Err(input_error("attention mask has no real token"));
            }
        }
        let positions = c.position_ids(ids);
        if let Some(&p) = positions
            .iter()
            .find(|&&p| p as usize >= c.max_position_embeddings)
        {
            return Err(input_error(format!(
                "position {p} needs more than the {} position rows (at most {} tokens)",
                c.max_position_embeddings,
                c.max_tokens()
            )));
        }
        let h = c.hidden_size;
        let w = &self.weights;
        let mut x = self.word.gather(w, ids)?;
        let pos = self.position.gather(w, &positions)?;
        for (row, p) in x.chunks_exact_mut(h).zip(pos.chunks_exact(h)) {
            for ((v, t), q) in row.iter_mut().zip(&self.token_type).zip(p) {
                *v = (*v + t) + q;
            }
        }
        self.emb_ln.apply(&mut x, n);
        trace(0, &x);

        // Additive key mask, built only when something is masked (with an
        // all-ones mask transformers passes no mask to sdpa either).
        let additive = mask.filter(|m| m.contains(&0)).map(|m| {
            let row: Vec<f32> = m
                .iter()
                .map(|&v| if v != 0 { 0.0 } else { f32::MIN })
                .collect();
            row.repeat(n)
        });
        let heads = c.num_attention_heads;
        let hd = c.head_dim();
        let scale = (1.0 / (hd as f64).sqrt()) as f32;
        for (i, layer) in self.layers.iter().enumerate() {
            let q = ops::split_heads(&layer.query.forward(&x, n), n, heads, hd);
            let k = ops::split_heads(&layer.key.forward(&x, n), n, heads, hd);
            let v = ops::split_heads(&layer.value.forward(&x, n), n, heads, hd);
            let ctx = ops::merge_heads(
                &ops::mha(&q, &k, &v, additive.as_deref(), heads, n, n, hd, scale),
                n,
                heads,
                hd,
            );
            let mut a = layer.attn_out.forward(&ctx, n);
            for (o, r) in a.iter_mut().zip(&x) {
                *o += r;
            }
            layer.attn_ln.apply(&mut a, n);
            let mut mid = layer.fc1.forward(&a, n);
            ops::gelu_erf(&mut mid);
            let mut out = layer.fc2.forward(&mid, n);
            for (o, r) in out.iter_mut().zip(&a) {
                *o += r;
            }
            layer.out_ln.apply(&mut out, n);
            x = out;
            trace(i + 1, &x);
        }
        Ok(x)
    }
}
