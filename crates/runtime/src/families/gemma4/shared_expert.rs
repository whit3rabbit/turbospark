//! Shared (dense) MLP expert branch dispatches for Gemma 4.

use crate::families::gemma4::moe;
use crate::real_forward::RealForwardRunner;
use crate::real_forward_dispatch::{encode_gemm_any, encode_gemv_any};
use crate::real_forward_types::RealForwardError;
use crate::real_forward_utils::{layer_tensor, norm_view};

const RMS_EPS: f32 = 1e-6;

impl RealForwardRunner {
    /// The shared (dense) expert branch: INT8 gate/up on `dense_x`, gated
    /// activation, down projection, then `post_feedforward_layernorm_1`,
    /// encoded into its own command buffer and committed without waiting.
    ///
    /// `slot` names which token of a prefill micro-batch this is; it reads
    /// that token's `dense_x` row. `h1_out` is the row the down
    /// projection and `post_feedforward_layernorm_1` write into: the
    /// single-row `h1` on every existing path, and `batch_h1[t]` on the
    /// batched routed path, whose tail consumes all M tokens' shared
    /// outputs only after they have all been produced.
    pub(crate) fn encode_shared_expert_branch(
        &mut self,
        layer: usize,
        hidden: usize,
        inter: usize,
        use_silu: bool,
        slot: &moe::RoutedSlot,
        h1_out: (&gpu::MetalBuffer, u64),
    ) -> Result<(), RealForwardError> {
        let gpu_err = RealForwardError::Gpu;
        let x_off = (slot.token * hidden * 2) as u64;
        let shared_pass = self.context.begin_pass_labeled("shared-expert cb");
        let real = self.real.as_ref().expect("real state present");
        for (name, rows, cols, x_buf, y_buf) in [
            (
                layer_tensor(layer, "mlp.gate_proj.weight"),
                inter,
                hidden,
                &real.dense_x,
                &self.scratch.ffn_gate,
            ),
            (
                layer_tensor(layer, "mlp.up_proj.weight"),
                inter,
                hidden,
                &real.dense_x,
                &self.scratch.ffn_up,
            ),
        ] {
            encode_gemv_any(
                &mut self.context,
                &shared_pass,
                &self.weights,
                &self.index,
                &name,
                rows,
                cols,
                (x_buf, x_off),
                (y_buf, 0),
            )?;
        }
        let act = if use_silu {
            gpu::encode_silu_mul
        } else {
            gpu::encode_gelu_mul
        };
        act(
            &mut self.context,
            &shared_pass,
            (&self.scratch.ffn_gate, 0),
            (&self.scratch.ffn_up, 0),
            (&self.scratch.ffn_act, 0),
            inter as u32,
        )
        .map_err(gpu_err)?;
        encode_gemv_any(
            &mut self.context,
            &shared_pass,
            &self.weights,
            &self.index,
            &layer_tensor(layer, "mlp.down_proj.weight"),
            hidden,
            inter,
            (&self.scratch.ffn_act, 0),
            h1_out,
        )?;
        let post_ffn1 = norm_view(
            &self.weights,
            &self.index,
            &layer_tensor(layer, "post_feedforward_layernorm_1.weight"),
            hidden,
        )?;
        gpu::encode_rms_norm_bf16w(
            &mut self.context,
            &shared_pass,
            h1_out,
            post_ffn1,
            h1_out,
            hidden as u32,
            RMS_EPS,
        )
        .map_err(gpu_err)?;
        shared_pass.commit();
        Ok(())
    }

    /// The same branch for ALL `m` tokens of a prefill micro-batch, in one
    /// command buffer instead of `m` (`TURBOSPARK_BATCHED_GEMV`,
    /// `docs/BATCHED_PREFILL.md`).
    ///
    /// Two things are batched and one deliberately is not. The three
    /// projections become M-row GEMMs through `encode_gemm_any`, which is
    /// where the weight amortization lives; and the whole micro-batch shares
    /// ONE `begin_pass_labeled`, where the per-token form opens and commits
    /// `m` of them, which is host work the dispatch ranking never sees.
    /// The gated activation and `post_feedforward_layernorm_1` stay looped
    /// per row -- both are elementwise or per-row reductions with no weights
    /// to amortize, they are in the 21.2% of prefill that cannot batch at
    /// all, and looping them is what `families/qwen/batched_layers.rs`
    /// already does for the same pair of dispatches.
    ///
    /// `dense_x` is read from row 0 for all M rows because it already holds
    /// `MAX_PREFILL_BATCH` token-major rows, which is exactly the layout
    /// `encode_gemm_any` documents for its `x`.
    pub(crate) fn encode_shared_expert_branch_batched(
        &mut self,
        layer: usize,
        hidden: usize,
        inter: usize,
        use_silu: bool,
        m: usize,
    ) -> Result<(), RealForwardError> {
        let gpu_err = RealForwardError::Gpu;
        // Cloned up front so the `&mut self.context` borrows below do not
        // contend with the `&self.real` these live behind. Cheap: a
        // `MetalBuffer` clone is a handle, not the memory.
        let real = self.real.as_ref().expect("real state present");
        let dense_x = real.dense_x.clone();
        let batched = real.batched();
        let (gate, up, act, h1) = (
            batched.batch_ffn_gate.clone(),
            batched.batch_ffn_up.clone(),
            batched.batch_ffn_act.clone(),
            batched.batch_h1.clone(),
        );
        let shared_pass = self
            .context
            .begin_pass_labeled("shared-expert cb (batched)");
        for (suffix, out) in [("mlp.gate_proj.weight", &gate), ("mlp.up_proj.weight", &up)] {
            encode_gemm_any(
                &mut self.context,
                &shared_pass,
                &self.weights,
                &self.index,
                &layer_tensor(layer, suffix),
                inter,
                hidden,
                (&dense_x, 0),
                (out, 0),
                m,
            )?;
        }
        let activation = if use_silu {
            gpu::encode_silu_mul
        } else {
            gpu::encode_gelu_mul
        };
        for t in 0..m {
            let row = (t * inter * 2) as u64;
            activation(
                &mut self.context,
                &shared_pass,
                (&gate, row),
                (&up, row),
                (&act, row),
                inter as u32,
            )
            .map_err(gpu_err)?;
        }
        encode_gemm_any(
            &mut self.context,
            &shared_pass,
            &self.weights,
            &self.index,
            &layer_tensor(layer, "mlp.down_proj.weight"),
            hidden,
            inter,
            (&act, 0),
            (&h1, 0),
            m,
        )?;
        let post_ffn1 = norm_view(
            &self.weights,
            &self.index,
            &layer_tensor(layer, "post_feedforward_layernorm_1.weight"),
            hidden,
        )?;
        for t in 0..m {
            let row = (t * hidden * 2) as u64;
            gpu::encode_rms_norm_bf16w(
                &mut self.context,
                &shared_pass,
                (&h1, row),
                post_ffn1,
                (&h1, row),
                hidden as u32,
                RMS_EPS,
            )
            .map_err(gpu_err)?;
        }
        shared_pass.commit();
        Ok(())
    }
}
