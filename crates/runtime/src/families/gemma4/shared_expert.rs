//! Shared (dense) MLP expert branch dispatches for Gemma 4.

use crate::families::gemma4::moe;
use crate::real_forward::RealForwardRunner;
use crate::real_forward_dispatch::{encode_gemm_any, encode_gemv_any};
use crate::real_forward_types::RealForwardError;
use crate::real_forward_utils::{layer_tensor, norm_view};

const RMS_EPS: f32 = 1e-6;

impl RealForwardRunner {
    /// The DENSE trunk's whole FFN arm, encoded into the caller's pass:
    /// gate/up on `dense_x` (the attention tail's `pre_feedforward_layernorm`
    /// output), gated activation, down projection, the plain
    /// `post_feedforward_layernorm`, the residual add, and `layer_scalar`'s
    /// residual scale -- the same end-of-layer scalar the MoE tail applies,
    /// which the real dense carrier carries per layer even with the MoE
    /// block off.
    ///
    /// This is the MoE sandwich minus the second branch: no
    /// `post_feedforward_layernorm_1` (the shared branch's own norm, absent
    /// in a dense checkpoint), no routed half, no `h2`, no router. One row,
    /// slot 0 -- the sequential decode path is always token 0.
    pub(crate) fn encode_gemma4_dense_ffn(
        &mut self,
        pass: &gpu::PassEncoder,
        layer: usize,
        hidden: usize,
        inter: usize,
        use_silu: bool,
    ) -> Result<(), RealForwardError> {
        let gpu_err = RealForwardError::Gpu;
        let real = self.real.as_ref().expect("real state present");
        for (name, rows, cols, y_buf) in [
            (
                layer_tensor(layer, "mlp.gate_proj.weight"),
                inter,
                hidden,
                &self.scratch.ffn_gate,
            ),
            (
                layer_tensor(layer, "mlp.up_proj.weight"),
                inter,
                hidden,
                &self.scratch.ffn_up,
            ),
        ] {
            encode_gemv_any(
                &mut self.context,
                pass,
                &self.weights,
                &self.index,
                &name,
                rows,
                cols,
                (&real.dense_x, 0),
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
            pass,
            (&self.scratch.ffn_gate, 0),
            (&self.scratch.ffn_up, 0),
            (&self.scratch.ffn_act, 0),
            inter as u32,
        )
        .map_err(gpu_err)?;
        encode_gemv_any(
            &mut self.context,
            pass,
            &self.weights,
            &self.index,
            &layer_tensor(layer, "mlp.down_proj.weight"),
            hidden,
            inter,
            (&self.scratch.ffn_act, 0),
            (&real.h1, 0),
        )?;
        let post_ffn = norm_view(
            &self.weights,
            &self.index,
            &layer_tensor(layer, "post_feedforward_layernorm.weight"),
            hidden,
        )?;
        gpu::encode_rms_norm_bf16w(
            &mut self.context,
            pass,
            (&real.h1, 0),
            post_ffn,
            (&self.scratch.ffn_normed, 0),
            hidden as u32,
            RMS_EPS,
        )
        .map_err(gpu_err)?;
        gpu::encode_residual_add(
            &mut self.context,
            pass,
            (&self.scratch.x, 0),
            (&self.scratch.ffn_normed, 0),
            hidden as u32,
        )
        .map_err(gpu_err)?;
        gpu::encode_scalar_mul(
            &mut self.context,
            pass,
            (&self.scratch.x, 0),
            real.layer_scalar[layer],
            hidden as u32,
        )
        .map_err(gpu_err)?;
        Ok(())
    }

    /// [`encode_gemma4_dense_ffn`] for ALL `m` tokens of a prefill
    /// micro-batch: the three projections as M-row GEMMs through
    /// `encode_gemm_any` (the INT8-affine batched kernel), the activation
    /// and the sandwich tail looped per row -- the same batch/loop line
    /// `encode_shared_expert_branch_batched` draws, ending in the plain
    /// `post_feedforward_layernorm`, the residual add and `layer_scalar`
    /// instead of the MoE split's per-branch norms and second branch.
    pub(crate) fn encode_gemma4_dense_ffn_batched(
        &mut self,
        pass: &gpu::PassEncoder,
        layer: usize,
        hidden: usize,
        inter: usize,
        use_silu: bool,
        m: usize,
    ) -> Result<(), RealForwardError> {
        let gpu_err = RealForwardError::Gpu;
        let real = self.real.as_ref().expect("real state present");
        // Cloned up front so the `&mut self.context` borrows below do not
        // contend with the `&self.real` / `&self.scratch` these live behind
        // (`encode_shared_expert_branch_batched`'s own comment: a
        // `MetalBuffer` clone is a handle, not the memory).
        let dense_x = real.dense_x.clone();
        // M rows, not the single-row `h1`: the batched down projection
        // writes `m * hidden`, and the same token-major row the tail reads
        // must have held all of them (`batch_h1`'s own doc).
        let batched = real.batched();
        let h1 = batched.batch_h1.clone();
        let layer_scalar = real.layer_scalar[layer];
        // M-row scratch for the M-row GEMMs: the single-row `scratch.ffn_*`
        // decode buffers are the per-token shared branch's, and an M-row
        // write into one is an overflow that surfaces as garbage, not a
        // crash (`encode_shared_expert_branch_batched` reached for the
        // batched seven for the same reason).
        let (gate, up, act, normed) = (
            batched.batch_ffn_gate.clone(),
            batched.batch_ffn_up.clone(),
            batched.batch_ffn_act.clone(),
            self.scratch.ffn_normed.clone(),
        );
        for (suffix, out) in [("mlp.gate_proj.weight", &gate), ("mlp.up_proj.weight", &up)] {
            encode_gemm_any(
                &mut self.context,
                pass,
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
                pass,
                (&gate, row),
                (&up, row),
                (&act, row),
                inter as u32,
            )
            .map_err(gpu_err)?;
        }
        encode_gemm_any(
            &mut self.context,
            pass,
            &self.weights,
            &self.index,
            &layer_tensor(layer, "mlp.down_proj.weight"),
            hidden,
            inter,
            (&act, 0),
            (&h1, 0),
            m,
        )?;
        let post_ffn = norm_view(
            &self.weights,
            &self.index,
            &layer_tensor(layer, "post_feedforward_layernorm.weight"),
            hidden,
        )?;
        for t in 0..m {
            let row = (t * hidden * 2) as u64;
            gpu::encode_rms_norm_bf16w(
                &mut self.context,
                pass,
                (&h1, row),
                post_ffn,
                (&normed, 0),
                hidden as u32,
                RMS_EPS,
            )
            .map_err(gpu_err)?;
            // Reusing `ffn_normed` per row is the single-row scratch rule:
            // each norm's output is consumed by the add before the next
            // dispatch writes it (`crates/gpu` Gotcha 8).
            gpu::encode_residual_add(
                &mut self.context,
                pass,
                (&self.scratch.x, row),
                (&normed, 0),
                hidden as u32,
            )
            .map_err(gpu_err)?;
            gpu::encode_scalar_mul(
                &mut self.context,
                pass,
                (&self.scratch.x, row),
                layer_scalar,
                hidden as u32,
            )
            .map_err(gpu_err)?;
        }
        Ok(())
    }

    /// The shared (dense) expert branch: INT8 gate/up on `dense_x`, gated
    /// activation, down projection, then `post_feedforward_layernorm_1`,
    /// encoded into its own command buffer and committed without waiting.
    /// The returned handle drains on drop so an error cannot release the
    /// external no-copy weight backing while this work is still running.
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
    ) -> Result<gpu::CommittedPass, RealForwardError> {
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
        Ok(shared_pass.commit().waiting_on_drop())
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
    ) -> Result<gpu::CommittedPass, RealForwardError> {
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
        Ok(shared_pass.commit().waiting_on_drop())
    }
}
