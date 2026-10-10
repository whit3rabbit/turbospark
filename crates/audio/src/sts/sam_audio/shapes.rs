//! Load-time shape contract for SAM-Audio.
//!
//! The forward passes index flat `Vec<f32>` buffers with strides taken from
//! the config (`dim`, `head_dim`, `n_heads * d_kv`, ...) while the weights
//! carry their own shapes from the checkpoint. A checkpoint whose config is
//! self-consistent but whose tensors disagree with it would pass every
//! per-tensor check at load and then panic on an out-of-range index during
//! inference. This module states each dependency once and rejects the
//! mismatch at load, so inference can rely on it.

use super::{
    Attention, DiT, EmbedAnchors, Linear, PatcherResnet, ProjectionLayer, Result, Rms,
    SamAudioConfig, T5Encoder,
};
use crate::error::SpeechError;

fn bad(name: &str, why: String) -> SpeechError {
    SpeechError::Tensor {
        name: name.to_string(),
        why,
    }
}

/// `weight` is `[out, in]`; `bias`, when present, has `out` entries.
fn linear(name: &str, l: &Linear, out: usize, inp: usize) -> Result<()> {
    if l.out_dim != out || l.in_dim != inp {
        return Err(bad(
            name,
            format!(
                "expected (out, in) = ({out}, {inp}), got ({}, {})",
                l.out_dim, l.in_dim
            ),
        ));
    }
    if out.checked_mul(inp) != Some(l.weight.len()) {
        return Err(bad(
            name,
            format!(
                "weight holds {} values, expected {out} x {inp}",
                l.weight.len()
            ),
        ));
    }
    if let Some(bias) = &l.bias {
        if bias.len() != out {
            return Err(bad(
                name,
                format!("bias holds {} values, expected {out}", bias.len()),
            ));
        }
    }
    Ok(())
}

fn len(name: &str, got: usize, want: usize) -> Result<()> {
    if got != want {
        return Err(bad(name, format!("expected {want} values, got {got}")));
    }
    Ok(())
}

fn rms(name: &str, r: &Rms, want: usize) -> Result<()> {
    len(name, r.weight.len(), want)
}

/// SwiGLU triple: `w1`/`w3` map `inp -> hidden`, `w2` maps `hidden -> out`.
/// Returns `hidden` so callers can relate it to other tensors.
fn projection(name: &str, p: &ProjectionLayer, inp: usize, out: Option<usize>) -> Result<usize> {
    let hidden = p.w1.out_dim;
    linear(&format!("{name}.w1"), &p.w1, hidden, inp)?;
    if let Some(w3) = &p.w3 {
        linear(&format!("{name}.w3"), w3, hidden, inp)?;
    }
    linear(
        &format!("{name}.w2"),
        &p.w2,
        out.unwrap_or(p.w2.out_dim),
        hidden,
    )?;
    Ok(p.w2.out_dim)
}

fn attention(name: &str, a: &Attention, dim: usize, head_dim: usize, kv_in: usize) -> Result<()> {
    // The head-dim-major reshape strides every projection by `dim`, so all
    // four outputs (and the query/output inputs) must be exactly `dim` wide.
    linear(&format!("{name}.wq"), &a.wq, dim, dim)?;
    linear(&format!("{name}.wk"), &a.wk, dim, kv_in)?;
    linear(&format!("{name}.wv"), &a.wv, dim, kv_in)?;
    linear(&format!("{name}.wo"), &a.wo, dim, dim)?;
    for (suffix, norm) in [("q_norm", &a.q_norm), ("k_norm", &a.k_norm)] {
        if let Some(norm) = norm {
            rms(&format!("{name}.{suffix}"), norm, head_dim)?;
        }
    }
    Ok(())
}

fn patcher(p: &PatcherResnet, dim: usize) -> Result<()> {
    let base = "transformer.x_embedder.block";
    len(base, p.channels, dim)?;
    for (i, (norm, conv)) in p.blocks.iter().enumerate() {
        let name = format!("{base}.block{}", i + 1);
        len(&format!("{name}.groupnorm.weight"), norm.weight.len(), dim)?;
        len(&format!("{name}.groupnorm.bias"), norm.bias.len(), dim)?;
        // Conv1d k=3: rows are [out][k][in], so `in_dim` is 3 * channels.
        linear(&format!("{name}.project"), conv, dim, 3 * dim)?;
    }
    if let Some(to_out) = &p.to_out {
        linear(&format!("{base}.to_out"), to_out, dim, dim)?;
    }
    Ok(())
}

fn t5(config: &SamAudioConfig, enc: &T5Encoder) -> Result<()> {
    let d = enc.d_model;
    let inner = enc.n_heads * enc.d_kv;
    len(
        "text_encoder.model.shared.weight",
        enc.shared.len(),
        enc.vocab * d,
    )?;
    len(
        "text_encoder.model.encoder.final_layer_norm.weight",
        enc.final_norm.len(),
        d,
    )?;
    for (i, block) in enc.blocks.iter().enumerate() {
        let p = format!("text_encoder.model.encoder.block.{i}");
        len(&format!("{p}.layer.0.layer_norm"), block.attn_norm.len(), d)?;
        len(&format!("{p}.layer.1.layer_norm"), block.ffn_norm.len(), d)?;
        let a = &block.attn;
        let ap = format!("{p}.layer.0.SelfAttention");
        linear(&format!("{ap}.q"), &a.q, inner, d)?;
        linear(&format!("{ap}.k"), &a.k, inner, d)?;
        linear(&format!("{ap}.v"), &a.v, inner, d)?;
        linear(&format!("{ap}.o"), &a.o, d, inner)?;
        if let Some(table) = &a.relative_bias {
            // Indexed `weight[bucket * n_heads + head]`.
            linear(
                &format!("{ap}.relative_attention_bias"),
                table,
                config.text_encoder.relative_attention_num_buckets,
                enc.n_heads,
            )?;
        }
        let f = &block.ffn;
        let fp = format!("{p}.layer.1.DenseReluDense");
        let ff = f.wi_0.out_dim;
        linear(&format!("{fp}.wi_0"), &f.wi_0, ff, d)?;
        if let Some(wi_1) = &f.wi_1 {
            linear(&format!("{fp}.wi_1"), wi_1, ff, d)?;
        }
        linear(&format!("{fp}.wo"), &f.wo, d, ff)?;
    }
    Ok(())
}

fn dit(dit: &DiT, config: &SamAudioConfig, codebook_dim: usize) -> Result<()> {
    let dim = dit.dim;
    let head_dim = dit.head_dim;
    let tc = &config.transformer;

    patcher(&dit.x_embedder, dim)?;

    // Context embedder: memory (already `dim` wide) -> context width that the
    // cross-attention K/V projections consume.
    let ctx = projection(
        "transformer.y_embedder.projection",
        &dit.y_embedder,
        dim,
        None,
    )?;
    // Timestep embedder feeds `t_block`, which must produce the six
    // modulation vectors, and the final scale/shift, both `dim` wide.
    projection(
        "transformer.t_embedder.projection",
        &dit.t_embedder,
        tc.frequency_embedding_dim,
        Some(dim),
    )?;
    linear("transformer.t_block", &dit.t_block, 6 * dim, dim)?;
    len(
        "transformer.final_layer_scale_shift_table",
        dit.final_layer_scale_shift_table.len(),
        2 * dim,
    )?;
    rms("transformer.norm", &dit.norm, dim)?;
    // The ODE adds the velocity to a `[frames, 2 * codebook_dim]` state.
    linear("transformer.output", &dit.output, 2 * codebook_dim, dim)?;

    for (i, layer) in dit.layers.iter().enumerate() {
        let lp = format!("transformer.layers.{i}");
        len(
            &format!("{lp}.scale_shift_table"),
            layer.scale_shift_table.len(),
            6 * dim,
        )?;
        rms(&format!("{lp}.attention_norm"), &layer.attention_norm, dim)?;
        rms(&format!("{lp}.ffn_norm"), &layer.ffn_norm, dim)?;
        attention(
            &format!("{lp}.attention"),
            &layer.attention,
            dim,
            head_dim,
            dim,
        )?;
        attention(
            &format!("{lp}.cross_attention"),
            &layer.cross_attention,
            dim,
            head_dim,
            ctx,
        )?;
        projection(
            &format!("{lp}.feed_forward"),
            &layer.feed_forward,
            dim,
            Some(dim),
        )?;
    }
    Ok(())
}

/// Cross-checks every loaded tensor against the config and against the
/// tensors it feeds. `codebook_dim` comes from the codec.
pub(super) fn check(
    config: &SamAudioConfig,
    codebook_dim: usize,
    text_encoder: &T5Encoder,
    transformer: &DiT,
    proj: &Linear,
    embed_anchors: &EmbedAnchors,
    memory_proj: &Linear,
) -> Result<()> {
    let dim = transformer.dim;
    t5(config, text_encoder)?;
    dit(transformer, config, codebook_dim)?;

    // `velocity` writes the noisy state at the front of each row and the
    // features at the back; a row narrower than both would slice out of range.
    let need = codebook_dim
        .checked_mul(2)
        .ok_or_else(|| bad("proj", "codebook_dim overflows".to_string()))?;
    if config.in_channels < need {
        return Err(SpeechError::BadConfig {
            field: "in_channels".to_string(),
            why: format!(
                "{} is smaller than 2 * codebook_dim ({need})",
                config.in_channels
            ),
        });
    }
    linear("proj", proj, dim, config.in_channels)?;
    linear("memory_proj", memory_proj, dim, config.text_encoder.d_model)?;

    let ed = embed_anchors.embedding_dim;
    if ed == 0 || embed_anchors.embed.is_empty() || embed_anchors.embed.len() % ed != 0 {
        return Err(bad(
            "embed_anchors.embed.weight",
            format!(
                "{} values is not a nonzero multiple of anchor_embedding_dim {ed}",
                embed_anchors.embed.len()
            ),
        ));
    }
    linear("embed_anchors.proj", &embed_anchors.proj, dim, ed)?;
    Ok(())
}
