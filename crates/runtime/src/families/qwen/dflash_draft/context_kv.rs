use super::{ring_spans, DFLASH_THETA};
use crate::families::qwen::RMS_EPS;
use crate::real_forward::{RealForwardError, RealForwardRunner};
use crate::real_forward_dispatch::encode_gemm_any;
use crate::real_forward_utils::norm_view;

impl RealForwardRunner {
    /// The context-KV write: capture rows `[0, rows)` become target-derived
    /// KV at positions `[capture_base, capture_base + rows)`, and the
    /// cursor advances to `capture_base + rows`.
    ///
    /// The rows are written from CAPTURE_BASE and not from the cursor
    /// because the write must cover the whole span the last verify
    /// accepted, part of which sits BEHIND the post-rewind cursor: the
    /// accepted proposals' slots still hold DRAFT-written KV (mask-token
    /// inputs), and the reference overwrites every accepted row with
    /// target-derived KV each round. Rows behind the cursor that were
    /// already correct are rewritten with identical values, which is
    /// idempotent by determinism.
    ///
    /// The recipe is the reference's `precompute_and_store_context_kv`,
    /// split into the per-layer projections this engine already dispatches
    /// rather than one fused GEMM: fc, `hidden_norm`, then per layer
    /// `k_proj`/`v_proj` straight into the slots (batched: the slot stride
    /// IS the kernel's row stride, asserted at build), per-head `k_norm` in
    /// place, RoPE in place at each row's own position.
    pub(crate) fn dflash_context_write(&mut self, rows: usize) -> Result<(), RealForwardError> {
        let (hidden, aux_count, layers, num_kv, head_dim) = {
            let d = self.real_dflash.as_ref().expect("caller checked");
            (
                d.shape.hidden,
                d.shape.aux_count,
                d.shape.layers,
                d.shape.num_kv_heads,
                d.shape.head_dim,
            )
        };
        let aux_width = hidden * aux_count;
        let kv_dim = num_kv * head_dim;
        let base_pos = self
            .real_dflash
            .as_ref()
            .expect("caller checked")
            .capture_base;
        // BEFORE the pass, not after it: this is a pure function of
        // `capture_base`, `rows` and the cursor, and refusing after
        // `commit_and_wait` would have already written the rows the
        // refusal says were not written.
        {
            let d = self.real_dflash.as_ref().expect("caller checked");
            if base_pos + rows < d.kv.position() {
                return Err(RealForwardError::Unsupported(format!(
                    "the DFlash2 context write would move the cursor BACKWARD: {} to {}",
                    d.kv.position(),
                    base_pos + rows
                )));
            }
        }

        let pass = self.context.begin_pass_labeled("dflash ctx kv");
        let (context, weights, index, dflash) = (
            &mut self.context,
            &self.weights,
            &self.index,
            self.real_dflash.as_ref().expect("caller checked"),
        );

        encode_gemm_any(
            context,
            &pass,
            weights,
            index,
            "dflash.fc.weight",
            hidden,
            aux_width,
            (&dflash.capture, 0),
            (&dflash.ctx_combined, 0),
            rows,
        )?;
        let hidden_norm = norm_view(weights, index, "dflash.hidden_norm.weight", hidden)?;
        for r in 0..rows {
            let row = (r * hidden) as u64 * 2;
            gpu::encode_rms_norm_bf16w(
                context,
                &pass,
                (&dflash.ctx_combined, row),
                hidden_norm,
                (&dflash.ctx_normed, row),
                hidden as u32,
                RMS_EPS,
            )
            .map_err(RealForwardError::Gpu)?;
        }
        let spans = ring_spans(dflash.kv.capacity(0), base_pos, rows);
        for layer in 0..layers {
            let name = |s: &str| format!("dflash.layers.{layer}.{s}");
            let k_norm = norm_view(weights, index, &name("self_attn.k_norm.weight"), head_dim)?;
            for is_k in [true, false] {
                let suffix = if is_k {
                    "self_attn.k_proj.weight"
                } else {
                    "self_attn.v_proj.weight"
                };
                for &(row0, count) in spans.iter().filter(|s| s.1 > 0) {
                    let (buf, off) = if is_k {
                        dflash.kv.k_slot(layer, base_pos + row0)
                    } else {
                        dflash.kv.v_slot(layer, base_pos + row0)
                    };
                    encode_gemm_any(
                        context,
                        &pass,
                        weights,
                        index,
                        &name(suffix),
                        kv_dim,
                        hidden,
                        (&dflash.ctx_normed, (row0 * hidden * 2) as u64),
                        (buf, off as u64),
                        count,
                    )?;
                }
            }
            for r in 0..rows {
                // Per POSITION rather than `k_off + r * stride`: the same
                // wrap the spans above split is what makes the arithmetic
                // form wrong on a straddling round.
                let (k_buf, k_row) = dflash.kv.k_slot(layer, base_pos + r);
                let k_row = k_row as u64;
                gpu::encode_rms_norm_bf16w_perhead(
                    context,
                    &pass,
                    (k_buf, k_row),
                    k_norm,
                    (k_buf, k_row),
                    num_kv as u32,
                    head_dim as u32,
                    RMS_EPS,
                )
                .map_err(RealForwardError::Gpu)?;
                gpu::encode_rope_proportional_neox(
                    context,
                    &pass,
                    (k_buf, k_row),
                    (base_pos + r) as u32,
                    num_kv as u32,
                    head_dim as u32,
                    (head_dim / 2) as u32,
                    DFLASH_THETA,
                )
                .map_err(RealForwardError::Gpu)?;
            }
        }
        // The wait is unconditional: priming exists FOR the KV rows, and
        // they are not written until the GPU has run this pass.
        pass.commit_and_wait();
        let d = self.real_dflash.as_mut().expect("caller checked");
        let target = d.capture_base + rows;
        d.kv.advance_by(target - d.kv.position());
        // Consumed: a later draft must not rewrite these rows, which the
        // next trunk pass replaces wholesale.
        d.capture_rows = 0;
        Ok(())
    }
}
