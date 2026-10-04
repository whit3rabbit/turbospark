//! Compile-only preparation using the production specialization builders.
//! Shared shader sources use statics because the in-process cache keys their
//! addresses. Separate const expansions can miss pipelines prepared here.

use std::collections::HashSet;
use std::time::Instant;

use metal::FunctionConstantValues;

use crate::{autorelease_pool, GpuError, MetalContext};

/// Only compile-time inputs belong here. Buffer addresses and expert IDs do not.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KernelWarmup {
    Int4Gemv {
        rows: u32,
        cols: u32,
    },
    Int4Embedding,
    Int8Gemv,
    MoeRouter,
    Bf16Gemv,
    GdnInProjection {
        qkv: u32,
        z: u32,
        ab: u32,
        cols: u32,
    },
    GdnConvDecode,
    GdnQkNorm,
    GdnDeltaDecode,
    GdnGatedNorm,
    RmsNorm,
    RmsNormPerHead,
    RopeNeoxSubdim,
    SiluMul,
    GeluMul,
    ResidualAdd,
    SigmoidGateMul,
    SigmoidScalarMul,
    SplitQGate,
    MoeAffine {
        phase2: bool,
        use_silu: bool,
    },
    AttentionDecode {
        combine: bool,
        scale_bits: u32,
        ring_capacity: u32,
        chunks: u32,
        has_sinks: bool,
    },
}

/// Host compilation work. Preparation never creates a command buffer.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct KernelWarmupStats {
    pub registrations: usize,
    pub unique_keys: usize,
    pub pipeline_creations: u64,
    pub elapsed_ms: f64,
}

/// Metadata registration is cheap; preparation is an explicit separate boundary.
#[derive(Default)]
pub struct KernelWarmupPlan {
    kernels: Vec<KernelWarmup>,
    seen: HashSet<KernelWarmup>,
    registrations: usize,
}

impl KernelWarmupPlan {
    pub fn register(&mut self, kernel: KernelWarmup) {
        self.registrations += 1;
        if self.seen.insert(kernel) {
            self.kernels.push(kernel);
        }
    }

    /// Traverse reachable dispatch buckets with the runtime's own mapping.
    pub fn register_attention(
        &mut self,
        scale: f32,
        ring_capacity: u32,
        max_range: u32,
        has_sinks: bool,
    ) {
        let mut range = 1u32;
        while range <= max_range {
            for combine in [false, true] {
                self.register(KernelWarmup::AttentionDecode {
                    combine,
                    scale_bits: scale.to_bits(),
                    ring_capacity,
                    chunks: crate::attention_decode::chunks_for(range),
                    has_sinks,
                });
            }
            // chunks_for uses power-of-two buckets; all transitions are reached
            // here without enumerating every context position.
            let Some(next) = range.checked_mul(2) else {
                break;
            };
            range = next;
        }
    }

    pub fn prepare(&self, context: &mut MetalContext) -> Result<KernelWarmupStats, GpuError> {
        let started = Instant::now();
        // Reject malformed plans before preparing any of their pipelines.
        for kernel in &self.kernels {
            kernel.validate()?;
        }
        let before = context.compilation_stats().pipeline_creations;
        for kernel in &self.kernels {
            autorelease_pool(|| kernel.prepare(context))?;
        }
        Ok(KernelWarmupStats {
            registrations: self.registrations,
            unique_keys: self.kernels.len(),
            pipeline_creations: context.compilation_stats().pipeline_creations - before,
            elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
        })
    }
}

impl KernelWarmup {
    fn validate(self) -> Result<(), GpuError> {
        let valid = match self {
            Self::Int4Gemv { rows, cols } => rows > 0 && cols > 0 && cols % 64 == 0,
            Self::GdnInProjection { qkv, z, ab, cols } => {
                qkv > 0 && z > 0 && ab > 0 && cols > 0 && cols % 64 == 0
            }
            Self::AttentionDecode {
                scale_bits, chunks, ..
            } => {
                f32::from_bits(scale_bits).is_finite()
                    && f32::from_bits(scale_bits) > 0.0
                    && chunks.is_power_of_two()
                    && chunks <= crate::attention_decode::MAX_CHUNKS
            }
            _ => true,
        };
        if valid {
            Ok(())
        } else {
            Err(GpuError::InvalidInput(format!("warmup key {self:?}")))
        }
    }

    fn prepare(self, context: &mut MetalContext) -> Result<(), GpuError> {
        use KernelWarmup::*;
        match self {
            Int4Gemv { rows, cols } => {
                let (constants, key) = crate::dequant_int4_gemv::specialized_constants(rows, cols);
                context.pipeline(
                    crate::dequant_int4_gemv::SOURCE,
                    "dequant_int4_gemv_simd",
                    &constants,
                    &key,
                )?;
            }
            Int4Embedding => {
                context.pipeline(
                    crate::dequant_int4_gemv::SOURCE,
                    "embed_lookup_int4",
                    &crate::dequant_int4_gemv::unused_function_constants(),
                    b"",
                )?;
            }
            Int8Gemv => {
                context.pipeline(
                    crate::dequant_int8_gemv::SOURCE,
                    "dequant_int8_gemv_simd",
                    &crate::dequant_int8_gemv::unused_function_constants(),
                    b"",
                )?;
            }
            MoeRouter => {
                context.pipeline(
                    crate::moe_decode::moe_decode_source(),
                    "router_gemv_gemma4_r4",
                    &crate::moe_decode::moe_function_constants(false),
                    &crate::moe_decode::constants_key(false),
                )?;
            }
            Bf16Gemv => {
                context.pipeline(
                    crate::gemv_bf16::SOURCE,
                    "bf16_gemv_rows",
                    &FunctionConstantValues::new(),
                    b"",
                )?;
            }
            GdnInProjection { qkv, z, ab, cols } => {
                crate::gdn::in_proj_pipeline(context, qkv, z, ab, cols)?;
            }
            GdnConvDecode | GdnQkNorm | GdnDeltaDecode | GdnGatedNorm => {
                let name = match self {
                    GdnConvDecode => "gdn_conv_mix_decode",
                    GdnQkNorm => "gdn_qk_norm",
                    GdnDeltaDecode => "gdn_delta_step_decode",
                    _ => "gdn_gated_norm",
                };
                crate::gdn::pipeline(context, name)?;
            }
            RmsNorm | RmsNormPerHead => {
                let name = if self == RmsNorm {
                    "rmsnorm_bf16w"
                } else {
                    "rmsnorm_bf16w_perhead"
                };
                context.pipeline(
                    crate::rms_norm::SOURCE,
                    name,
                    &crate::rms_norm::unused_function_constants(),
                    b"",
                )?;
            }
            RopeNeoxSubdim => {
                context.pipeline(
                    crate::rope::SOURCE,
                    "rope_neox_subdim",
                    &crate::rope::unused_function_constants(),
                    b"",
                )?;
            }
            MoeAffine { phase2, use_silu } => {
                let name = if phase2 {
                    "moe_phase2_down_reduce_k8"
                } else {
                    "moe_phase1_gate_up_act_u16load"
                };
                context.pipeline(
                    crate::moe_decode::moe_decode_source(),
                    name,
                    &crate::moe_decode::moe_function_constants(use_silu),
                    &crate::moe_decode::constants_key(use_silu),
                )?;
            }
            AttentionDecode {
                combine,
                scale_bits,
                ring_capacity,
                chunks,
                has_sinks,
            } => {
                let scale = f32::from_bits(scale_bits);
                let name = if combine {
                    "attention_decode_combine"
                } else {
                    "attention_decode_partial"
                };
                context.pipeline(
                    crate::attention_decode::SOURCE,
                    name,
                    &crate::attention_decode::attention_function_constants(
                        scale,
                        ring_capacity,
                        chunks,
                        has_sinks,
                    ),
                    &crate::attention_decode::attention_constants_key(
                        scale,
                        ring_capacity,
                        chunks,
                        has_sinks,
                    ),
                )?;
            }
            SiluMul | GeluMul | ResidualAdd | SigmoidGateMul | SigmoidScalarMul | SplitQGate => {
                let name = match self {
                    SiluMul => "silu_mul_fp16",
                    GeluMul => "gelu_mul_fp16",
                    ResidualAdd => "residual_add_fp16",
                    SigmoidGateMul => "sigmoid_gate_mul_fp16",
                    SigmoidScalarMul => "sigmoid_scalar_mul_fp16",
                    _ => "split_q_gate_fp16",
                };
                context.pipeline(
                    crate::utility::SOURCE,
                    name,
                    &FunctionConstantValues::new(),
                    b"",
                )?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_plan_is_rejected_before_preparing_any_pipeline() {
        let mut context = MetalContext::new().unwrap();
        let mut plan = KernelWarmupPlan::default();
        plan.register(KernelWarmup::RmsNorm);
        plan.register(KernelWarmup::Int4Gemv { rows: 64, cols: 0 });
        assert!(matches!(
            plan.prepare(&mut context),
            Err(GpuError::InvalidInput(_))
        ));
        assert_eq!(context.compilation_stats().pipeline_creations, 0);
        assert_eq!(context.compilation_stats().library_compiles, 0);
        assert_eq!(context.buffer_allocation_count(), 0);
    }
}
