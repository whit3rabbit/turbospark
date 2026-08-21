use foundation::TokenId;

use super::{ring_spans, RESIDUAL_RESCALE};
use crate::families::qwen::dflash::{DFLASH_MASK_TOKEN, DFLASH_RESIDUAL_EPS};
use crate::real_forward::{RealForwardError, RealForwardRunner};
use crate::real_forward_dispatch::{encode_embed_any, encode_gemm_any};
use crate::real_forward_utils::norm_view;

impl RealForwardRunner {
    /// The block forward proper: embeddings, `layers` drafter layers, the
    /// final norm, and the two head GEMMs. Assumes the context KV is
    /// current up to `base`.
    pub(crate) fn dflash_forward_block(
        &mut self,
        anchor: TokenId,
        base: usize,
    ) -> Result<(), RealForwardError> {
        let arch = self.arch.clone();
        let vocab = arch.vocab_size as usize;
        let use_silu = arch.hidden_activation.contains("silu");
        let gpu_err = RealForwardError::Gpu;
        let embed_name = "language_model.model.embed_tokens.weight";
        let head_name = if arch.tie_word_embeddings {
            embed_name.to_string()
        } else {
            "language_model.lm_head.weight".to_string()
        };

        let pass = self.context.begin_pass_labeled("dflash draft");
        let (context, weights, index, dflash) = (
            &mut self.context,
            &self.weights,
            &self.index,
            self.real_dflash.as_ref().expect("checked"),
        );
        let s = &dflash.shape;
        let hidden = s.hidden;
        let rows = dflash.block + 1;
        let row_bytes = hidden as u64 * 2;
        let kv_spans = ring_spans(dflash.kv.capacity(0), base, rows);

        // Row 0 is the bonus token; rows 1..block are masks. Both enter at
        // 1/DFLASH_RESIDUAL_SCALE, which is where the residual stream's
        // scaled representation starts; the final norm takes it back out.
        encode_embed_any(
            context,
            &pass,
            weights,
            index,
            embed_name,
            (&dflash.x, 0),
            anchor as u32,
            hidden as u32,
            RESIDUAL_RESCALE,
        )?;
        for r in 1..rows {
            encode_embed_any(
                context,
                &pass,
                weights,
                index,
                embed_name,
                (&dflash.x, r as u64 * row_bytes),
                DFLASH_MASK_TOKEN as u32,
                hidden as u32,
                RESIDUAL_RESCALE,
            )?;
        }

        for layer in 0..s.layers {
            super::layer::encode_dflash_layer(
                context, &pass, weights, index, dflash, layer, base, rows, &kv_spans, use_silu,
            )?;
        }

        // Final norm, then the TRUNK's lm_head (the drafter shares it) and
        // the selector's hidden projection.
        let final_norm = norm_view(weights, index, "dflash.norm.weight", hidden)?;
        for r in 0..rows {
            let row = r as u64 * row_bytes;
            gpu::encode_rms_norm_bf16w(
                context,
                &pass,
                (&dflash.x, row),
                final_norm,
                (&dflash.normed, row),
                hidden as u32,
                DFLASH_RESIDUAL_EPS,
            )
            .map_err(gpu_err)?;
        }
        encode_gemm_any(
            context,
            &pass,
            weights,
            index,
            &head_name,
            vocab,
            hidden,
            (&dflash.normed, 0),
            (&dflash.logits, 0),
            rows,
        )?;
        encode_gemm_any(
            context,
            &pass,
            weights,
            index,
            "dflash.candidate_selector.hidden_projection.weight",
            s.rank,
            hidden,
            (&dflash.normed, 0),
            (&dflash.hproj, 0),
            rows,
        )?;
        pass.commit_and_wait();
        Ok(())
    }
}
