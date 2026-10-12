//! The Dia encoder/decoder stacks (`layers.py` `Encoder`, `Decoder`):
//! an embedding plus pre-norm attention/MLP encoder over the
//! byte-level text ids, and a per-channel-summed-embedding decoder with
//! grouped-query self attention and precomputed cross attention.

use super::config::DiaConfig;
use super::layers::{load_dense_linear, load_tensor, Attention, KvStore, Mlp, Rotary};
use crate::nn::{Linear, RmsNorm};
use crate::{ops, Result};
use turbospark_model_io::safetensors::SafetensorsFile;

/// Embedding lookup with the observed MLX gather semantics: an
/// out-of-range id produces a zero row (the reference can embed ids
/// pushed past the vocabulary by the post-EOS bookkeeping; the pinned
/// MLX maps them to zeros, so the port does too). A negative id wraps
/// like a Python index.
pub(crate) fn embed_lookup(weight: &[f32], cols: usize, id: i32) -> &[f32] {
    let rows = weight.len() / cols;
    if rows == 0 {
        return &[];
    }
    if id < 0 {
        let back = ((-id) as usize).min(rows);
        let row = rows - back;
        return &weight[row * cols..(row + 1) * cols];
    }
    let r = id as usize;
    if r >= rows {
        return &[];
    }
    &weight[r * cols..(r + 1) * cols]
}

/// One pre-norm self-attention plus post-norm MLP encoder layer.
struct EncoderLayer {
    pre_sa_norm: RmsNorm,
    self_attention: Attention,
    post_sa_norm: RmsNorm,
    mlp: Mlp,
    embed_dim: usize,
}

impl EncoderLayer {
    fn load(file: &SafetensorsFile, prefix: &str, config: &DiaConfig) -> Result<Self> {
        let enc = &config.model.encoder;
        let eps = config.model.normalization_layer_epsilon;
        let dim = enc.stack.n_embd;
        let rotary = Rotary::new(
            config.model.rope_min_timescale,
            config.model.rope_max_timescale,
        );
        let norm = |name: &str| -> Result<RmsNorm> {
            Ok(RmsNorm::new(
                load_tensor(file, &format!("{prefix}.{name}.weight"), &[dim])?,
                eps,
            ))
        };
        Ok(Self {
            pre_sa_norm: norm("pre_sa_norm")?,
            self_attention: Attention::load(
                file,
                &format!("{prefix}.self_attention"),
                dim,
                dim,
                enc.n_head,
                enc.n_head,
                enc.head_dim,
                false,
                dim,
                rotary.clone(),
            )?,
            post_sa_norm: norm("post_sa_norm")?,
            mlp: Mlp::load(
                file,
                &format!("{prefix}.mlp"),
                dim,
                enc.stack.n_hidden,
                enc.stack.mlp_activations,
                enc.stack.use_pre_norm,
                eps,
            )?,
            embed_dim: dim,
        })
    }

    fn forward(&self, x: &mut [f32], positions: &[i32], masks: &[Vec<f32>]) -> Result<()> {
        let tokens = x.len() / (2 * self.embed_dim);
        let mut normed = x.to_vec();
        self.pre_sa_norm.apply(&mut normed, 2 * tokens);
        let attended = self.self_attention.forward(
            &normed,
            Some(&normed),
            tokens,
            positions,
            Some(masks),
            None,
            None,
        )?;
        for (dst, src) in x.iter_mut().zip(attended) {
            *dst += src;
        }
        let mut normed = x.to_vec();
        self.post_sa_norm.apply(&mut normed, 2 * tokens);
        self.mlp.forward(&mut normed, 2 * tokens);
        for (dst, src) in x.iter_mut().zip(normed) {
            *dst += src;
        }
        Ok(())
    }
}

/// The text encoder (`Encoder`).
pub(crate) struct Encoder {
    embedding: Vec<f32>,
    layers: Vec<EncoderLayer>,
    norm: RmsNorm,
    embed_dim: usize,
}

impl Encoder {
    pub(crate) fn load(file: &SafetensorsFile, config: &DiaConfig) -> Result<Self> {
        let enc = &config.model.encoder;
        let dim = enc.stack.n_embd;
        let mut layers = Vec::with_capacity(enc.stack.n_layer);
        for i in 0..enc.stack.n_layer {
            layers.push(EncoderLayer::load(
                file,
                &format!("encoder.layers.{i}"),
                config,
            )?);
        }
        Ok(Self {
            embedding: load_tensor(
                file,
                "encoder.embedding.weight",
                &[config.model.src_vocab_size * dim],
            )?,
            layers,
            norm: RmsNorm::new(
                load_tensor(file, "encoder.norm.weight", &[dim])?,
                config.model.normalization_layer_epsilon,
            ),
            embed_dim: dim,
        })
    }

    /// `src_ids [2][S]` with the additive `[S, S]` mask per row and the
    /// shared positions. Returns `[2 * S, embed]`.
    pub(crate) fn forward(
        &self,
        src_ids: &[Vec<i32>],
        positions: &[i32],
        masks: &[Vec<f32>],
    ) -> Result<Vec<f32>> {
        let seq = positions.len();
        let mut x = vec![0.0f32; 2 * seq * self.embed_dim];
        for (b, row_ids) in src_ids.iter().enumerate() {
            for (t, &id) in row_ids.iter().take(seq).enumerate() {
                let row = embed_lookup(&self.embedding, self.embed_dim, id);
                let dst = (b * seq + t) * self.embed_dim;
                x[dst..dst + self.embed_dim].copy_from_slice(row);
            }
        }
        for layer in &self.layers {
            layer.forward(&mut x, positions, masks)?;
        }
        self.norm.apply(&mut x, 2 * seq);
        Ok(x)
    }
}

/// One decoder layer: pre-norm self attention (GQA, cache), pre-norm
/// cross attention (precomputed cache), pre-norm MLP.
struct DecoderLayer {
    pre_sa_norm: RmsNorm,
    self_attention: Attention,
    pre_ca_norm: RmsNorm,
    cross_attention: Attention,
    pre_mlp_norm: RmsNorm,
    mlp: Mlp,
}

impl DecoderLayer {
    fn load(file: &SafetensorsFile, prefix: &str, config: &DiaConfig) -> Result<Self> {
        let dec = &config.model.decoder;
        let enc_dim = config.model.encoder.stack.n_embd;
        let eps = config.model.normalization_layer_epsilon;
        let dim = dec.stack.n_embd;
        let rotary = Rotary::new(
            config.model.rope_min_timescale,
            config.model.rope_max_timescale,
        );
        let norm = |name: &str| -> Result<RmsNorm> {
            Ok(RmsNorm::new(
                load_tensor(file, &format!("{prefix}.{name}.weight"), &[dim])?,
                eps,
            ))
        };
        Ok(Self {
            pre_sa_norm: norm("pre_sa_norm")?,
            pre_ca_norm: norm("pre_ca_norm")?,
            pre_mlp_norm: norm("pre_mlp_norm")?,
            self_attention: Attention::load(
                file,
                &format!("{prefix}.self_attention"),
                dim,
                dim,
                dec.gqa_query_heads,
                dec.kv_heads,
                dec.gqa_head_dim,
                false,
                dim,
                rotary.clone(),
            )?,
            cross_attention: Attention::load(
                file,
                &format!("{prefix}.cross_attention"),
                dim,
                enc_dim,
                dec.cross_query_heads,
                dec.cross_query_heads,
                dec.cross_head_dim,
                true,
                dim,
                rotary,
            )?,
            mlp: Mlp::load(
                file,
                &format!("{prefix}.mlp"),
                dim,
                dec.stack.n_hidden,
                dec.stack.mlp_activations,
                dec.stack.use_pre_norm,
                eps,
            )?,
        })
    }

    /// Prefill over the whole prompt block (reference audio prompts),
    /// filling the self cache. `self_masks[b]` is the additive
    /// `[T, T]` bias, `cross_masks[b]` the additive `[T, S]` bias.
    #[allow(clippy::too_many_arguments)]
    fn prefill(
        &self,
        x: &mut [f32],
        tokens: usize,
        positions: &[i32],
        self_masks: &[Vec<f32>],
        cross_masks: &[Vec<f32>],
        self_cache: &mut KvStore,
        cross_cache: &KvStore,
    ) -> Result<()> {
        let dim = x.len() / (2 * tokens);
        let normed = {
            let mut n = x.to_vec();
            self.pre_sa_norm.apply(&mut n, 2 * tokens);
            n
        };
        let attended = self.self_attention.forward(
            &normed,
            Some(&normed),
            tokens,
            positions,
            Some(self_masks),
            Some(self_cache),
            None,
        )?;
        for (dst, src) in x.iter_mut().zip(attended) {
            *dst += src;
        }
        let normed = {
            let mut n = x.to_vec();
            self.pre_ca_norm.apply(&mut n, 2 * tokens);
            n
        };
        let attended = self.cross_attention.forward(
            &normed,
            None,
            tokens,
            positions,
            Some(cross_masks),
            None,
            Some(cross_cache),
        )?;
        for (dst, src) in x.iter_mut().zip(attended) {
            *dst += src;
        }
        let mut normed = x.to_vec();
        self.pre_mlp_norm.apply(&mut normed, 2 * tokens);
        self.mlp.forward(&mut normed, 2 * tokens);
        for (dst, src) in x.iter_mut().zip(normed) {
            *dst += src;
        }
        let _ = dim;
        Ok(())
    }
}

/// The audio-token decoder (`Decoder`): one embedding table per
/// channel, summed into the stream.
pub(crate) struct Decoder {
    embeddings: Vec<Vec<f32>>,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    logits_dense: Linear,
    num_channels: usize,
    embed_dim: usize,
}

impl Decoder {
    pub(crate) fn load(file: &SafetensorsFile, config: &DiaConfig) -> Result<Self> {
        let dec = &config.model.decoder;
        let model = &config.model;
        let dim = dec.stack.n_embd;
        let channels = config.data.channels;
        let mut embeddings = Vec::with_capacity(channels);
        for c in 0..channels {
            embeddings.push(load_tensor(
                file,
                &format!("decoder.embeddings.{c}.weight"),
                &[model.tgt_vocab_size * dim],
            )?);
        }
        let mut layers = Vec::with_capacity(dec.stack.n_layer);
        for i in 0..dec.stack.n_layer {
            layers.push(DecoderLayer::load(
                file,
                &format!("decoder.layers.{i}"),
                config,
            )?);
        }
        Ok(Self {
            embeddings,
            layers,
            norm: RmsNorm::new(
                load_tensor(file, "decoder.norm.weight", &[dim])?,
                model.normalization_layer_epsilon,
            ),
            logits_dense: load_dense_linear(
                file,
                "decoder.logits_dense",
                dim,
                channels * model.tgt_vocab_size,
            )?,
            num_channels: channels,
            embed_dim: dim,
        })
    }

    /// Fresh self-attention caches sized for the decode budget.
    pub(crate) fn new_self_caches(&self, max_len: usize) -> Vec<KvStore> {
        self.layers
            .iter()
            .map(|layer| {
                let sa = &layer.self_attention;
                KvStore::new(sa.num_query_heads(), max_len, sa.head_dim())
            })
            .collect()
    }

    /// `precompute_cross_attention_kv`: per-layer cross caches holding
    /// the roped k/v of the encoder output for the whole sequence.
    pub(crate) fn precompute_cross_kv(
        &self,
        encoder_out: &[f32],
        seq: usize,
        src_positions: &[i32],
    ) -> Vec<KvStore> {
        self.layers
            .iter()
            .map(|layer| {
                let cross = &layer.cross_attention;
                let (k, v) = cross.project_kv(encoder_out, seq, src_positions);
                let mut store = KvStore::new(cross.num_kv_heads(), seq, cross.head_dim());
                store.prefill(&k, &v, seq);
                store
            })
            .collect()
    }

    fn embed_step(&self, ids: &[i32]) -> Vec<f32> {
        let mut x = vec![0.0f32; 2 * self.embed_dim];
        for (c, &id) in ids.iter().enumerate() {
            let row = embed_lookup(&self.embeddings[c], self.embed_dim, id);
            for b in 0..2 {
                let dst = b * self.embed_dim;
                for (d, &v) in row.iter().enumerate() {
                    x[dst + d] += v;
                }
            }
        }
        x
    }

    fn embed_block(&self, ids: &[Vec<Vec<i32>>], tokens: usize) -> Vec<f32> {
        let mut x = vec![0.0f32; 2 * tokens * self.embed_dim];
        for c in 0..self.num_channels {
            for (b, row_block) in ids.iter().enumerate() {
                for (t, row_ids) in row_block.iter().take(tokens).enumerate() {
                    let row = embed_lookup(&self.embeddings[c], self.embed_dim, row_ids[c]);
                    let dst = (b * tokens + t) * self.embed_dim;
                    for (d, &v) in row.iter().enumerate() {
                        x[dst + d] += v;
                    }
                }
            }
        }
        x
    }

    /// One decode step for both rows: `tgt_ids` is the per-channel
    /// input id row `[C]` (shared across the two batch rows, exactly
    /// like the reference). Returns `[2, C * vocab]` f32 logits.
    pub(crate) fn decode_step(
        &self,
        tgt_ids: &[i32],
        position: i32,
        cross_mask: &[Vec<f32>],
        self_caches: &mut [KvStore],
        cross_caches: &[KvStore],
    ) -> Vec<f32> {
        let mut x = self.embed_step(tgt_ids);
        for (layer, (self_cache, cross_cache)) in self
            .layers
            .iter()
            .zip(self_caches.iter_mut().zip(cross_caches))
        {
            let mut normed = x.clone();
            layer.pre_sa_norm.apply(&mut normed, 2);
            let attended =
                layer
                    .self_attention
                    .step(&normed, position, None, Some(self_cache), None);
            for (dst, src) in x.iter_mut().zip(attended) {
                *dst += src;
            }
            let normed = {
                let mut n = x.clone();
                layer.pre_ca_norm.apply(&mut n, 2);
                n
            };
            let attended = layer.cross_attention.step(
                &normed,
                position,
                Some(cross_mask),
                None,
                Some(cross_cache),
            );
            for (dst, src) in x.iter_mut().zip(attended) {
                *dst += src;
            }
            let mut normed = x.clone();
            layer.pre_mlp_norm.apply(&mut normed, 2);
            layer.mlp.forward(&mut normed, 2);
            for (dst, src) in x.iter_mut().zip(normed) {
                *dst += src;
            }
        }
        self.norm.apply(&mut x, 2);
        self.logits_dense.forward(&x, 2)
    }

    /// The prefill pass (reference audio prompts): runs the layers over
    /// the whole prompt block to fill the self caches. The reference
    /// discards the logits output, so they are not computed.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prefill(
        &self,
        tgt_ids: &[Vec<Vec<i32>>],
        tokens: usize,
        positions: &[i32],
        self_masks: &[Vec<f32>],
        cross_masks: &[Vec<f32>],
        self_caches: &mut [KvStore],
        cross_caches: &[KvStore],
    ) -> Result<()> {
        let mut x = self.embed_block(tgt_ids, tokens);
        for (layer, (self_cache, cross_cache)) in self
            .layers
            .iter()
            .zip(self_caches.iter_mut().zip(cross_caches))
        {
            layer.prefill(
                &mut x,
                tokens,
                positions,
                self_masks,
                cross_masks,
                self_cache,
                cross_cache,
            )?;
        }
        // The reference computes and drops the final norm + logits; the
        // norm is evaluated here so the parity surface matches a
        // cacheless probe if a fixture ever wants it.
        self.norm.apply(&mut x, 2 * tokens);
        ops::checksum(&x);
        Ok(())
    }
}
