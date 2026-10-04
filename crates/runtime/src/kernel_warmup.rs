//! Bounded Qwen affine-MoE experiment; selection follows the loaded model.

use gpu::{KernelWarmup as K, KernelWarmupPlan, KernelWarmupStats};
use model_io::ModelFamily;

use crate::real_forward_layout::RoutedBlobLayout;
use crate::real_forward_utils::entry;
use crate::{RealForwardError, RealForwardRunner};

impl RealForwardRunner {
    /// Compile selected text kernels without a forward pass, expert I/O, or
    /// state reset. Other paths retain lazy compilation and are refused here.
    pub fn prepare_kernel_warmup(
        &mut self,
        max_context: u32,
    ) -> Result<KernelWarmupStats, RealForwardError> {
        let started = std::time::Instant::now();
        let unsupported = |detail: &str| {
            RealForwardError::Unsupported(format!(
                "compile-only warmup experiment requires {detail}"
            ))
        };
        if self.arch.family != ModelFamily::QwenGdnMoe {
            return Err(unsupported("the Qwen GDN MoE family"));
        }
        let qwen = self
            .real_qwen
            .as_ref()
            .ok_or_else(|| unsupported("Qwen state"))?;
        if qwen.hadamard.is_some()
            || self.real_mtp.is_some()
            || self.real_dflash.is_some()
            || self.steering.is_some()
            || self.batched_gemv_prefill
            || self.routed_batch_prefill
            || self.arch.top_k_experts != 8
            || !self.arch.shared_expert_gated
            || !self.arch.attn_output_gate
            || self.arch.final_logit_softcap != 0.0
        {
            return Err(unsupported("unfolded top-8 gated MoE with speculation, steering, batched paths and logit softcap off"));
        }
        if max_context == 0
            || self.routed_layouts.is_empty()
            || self.routed_layouts.iter().any(|l| {
                l.phase1 != RoutedBlobLayout::Affine || l.phase2 != RoutedBlobLayout::Affine
            })
        {
            return Err(unsupported("a positive context and affine routed experts"));
        }
        let mut plan = KernelWarmupPlan::default();
        for kernel in [
            K::RmsNorm,
            K::ResidualAdd,
            K::SigmoidScalarMul,
            K::MoeRouter,
        ] {
            plan.register(kernel);
        }
        let silu = self.arch.hidden_activation.contains("silu");
        plan.register(if silu { K::SiluMul } else { K::GeluMul });
        for phase2 in [false, true] {
            plan.register(K::MoeAffine {
                phase2,
                use_silu: silu,
            });
        }
        let embedding = "language_model.model.embed_tokens.weight";
        match entry(&self.index, embedding)?.dtype {
            4 => plan.register(K::Int4Embedding),
            // BF16 embedding lookup is a host slice followed by a buffer write.
            1 => (),
            _ => return Err(unsupported("INT4 or BF16 embeddings")),
        }
        let hidden = self.arch.hidden_size as u32;
        let shape = qwen.shape;
        let head_dim = self.arch.full_head_dim as u32;
        for layer in 0..self.arch.num_layers as usize {
            let name = |suffix: &str| crate::families::qwen::layer_tensor(layer, suffix);
            if entry(&self.index, &name("mlp.gate.weight"))?.dtype != 5 {
                return Err(unsupported("the INT8 MoE router"));
            }
            if self.arch.layer_is_linear(layer) {
                for kernel in [
                    K::GdnConvDecode,
                    K::GdnQkNorm,
                    K::GdnDeltaDecode,
                    K::GdnGatedNorm,
                ] {
                    plan.register(kernel);
                }
                if entry(&self.index, &name("linear_attn.in_proj_qkv.weight"))?.dtype == 4 {
                    for suffix in ["in_proj_z.weight", "in_proj_a.weight", "in_proj_b.weight"] {
                        if entry(&self.index, &name(&format!("linear_attn.{suffix}")))?.dtype != 4 {
                            return Err(unsupported("matching INT4 fused GDN projections"));
                        }
                    }
                    plan.register(K::GdnInProjection {
                        qkv: shape.qkv_dim(),
                        z: shape.value_dim(),
                        ab: shape.num_v_heads,
                        cols: hidden,
                    });
                } else {
                    for (suffix, rows) in [
                        ("in_proj_qkv.weight", shape.qkv_dim()),
                        ("in_proj_z.weight", shape.value_dim()),
                        ("in_proj_a.weight", shape.num_v_heads),
                        ("in_proj_b.weight", shape.num_v_heads),
                    ] {
                        self.register_warmup_matrix(
                            &mut plan,
                            &name(&format!("linear_attn.{suffix}")),
                            rows,
                            hidden,
                        )?;
                    }
                }
                self.register_warmup_matrix(
                    &mut plan,
                    &name("linear_attn.out_proj.weight"),
                    hidden,
                    shape.value_dim(),
                )?;
            } else {
                if self.kv.quant_tables().is_some() || self.kv.ring_capacity(layer) != 0 {
                    return Err(unsupported("linear FP16 full-attention KV"));
                }
                for kernel in [
                    K::RmsNormPerHead,
                    K::RopeNeoxSubdim,
                    K::SplitQGate,
                    K::SigmoidGateMul,
                ] {
                    plan.register(kernel);
                }
                plan.register_attention(self.arch.attention_scale as f32, 0, max_context, false);
                for (suffix, rows, cols) in [
                    (
                        "q_proj.weight",
                        2 * self.arch.num_heads as u32 * head_dim,
                        hidden,
                    ),
                    (
                        "k_proj.weight",
                        self.arch.num_full_kv_heads as u32 * head_dim,
                        hidden,
                    ),
                    (
                        "v_proj.weight",
                        self.arch.num_full_kv_heads as u32 * head_dim,
                        hidden,
                    ),
                    (
                        "o_proj.weight",
                        hidden,
                        self.arch.num_heads as u32 * head_dim,
                    ),
                ] {
                    self.register_warmup_matrix(
                        &mut plan,
                        &name(&format!("self_attn.{suffix}")),
                        rows,
                        cols,
                    )?;
                }
            }
            for (suffix, rows, cols) in [
                (
                    "mlp.shared_expert.gate_proj.weight",
                    self.arch.intermediate_size as u32,
                    hidden,
                ),
                (
                    "mlp.shared_expert.up_proj.weight",
                    self.arch.intermediate_size as u32,
                    hidden,
                ),
                (
                    "mlp.shared_expert.down_proj.weight",
                    hidden,
                    self.arch.intermediate_size as u32,
                ),
                ("mlp.shared_expert_gate.weight", 1, hidden),
            ] {
                self.register_warmup_matrix(&mut plan, &name(suffix), rows, cols)?;
            }
        }
        let head = if self.arch.tie_word_embeddings {
            embedding
        } else {
            "language_model.lm_head.weight"
        };
        self.register_warmup_matrix(&mut plan, head, self.arch.vocab_size as u32, hidden)?;
        let mut stats = plan
            .prepare(&mut self.context)
            .map_err(RealForwardError::Gpu)?;
        stats.elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
        Ok(stats)
    }

    fn register_warmup_matrix(
        &self,
        plan: &mut KernelWarmupPlan,
        name: &str,
        rows: u32,
        cols: u32,
    ) -> Result<(), RealForwardError> {
        let kernel = match entry(&self.index, name)?.dtype {
            4 => K::Int4Gemv { rows, cols },
            5 => K::Int8Gemv,
            1 => K::Bf16Gemv,
            dtype => {
                return Err(RealForwardError::Unsupported(format!(
                    "compile-only warmup has no resident key for {name}, dtype {dtype}"
                )))
            }
        };
        plan.register(kernel);
        Ok(())
    }

    pub fn metal_pipeline_inventory(&self) -> Vec<(String, Vec<u8>)> {
        self.context.pipeline_inventory()
    }
}
