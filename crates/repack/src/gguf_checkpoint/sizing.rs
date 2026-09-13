//! Header-only storage sizing for the witnessed MiniMax GGUF contract.
use super::{plan, GgufRepackError};
use crate::{arch_from_gguf, map_gguf_name, GgufHeader, GgufMapping, GTURBO_PAGE_BYTES};
use model_io::ModelFamily;

#[derive(Debug)]
pub struct MiniMaxSizing {
    pub resident_bytes: u64,
    pub expert_file_bytes: u64,
    pub max_expert_stride: u64,
    pub eight_slot_bytes: u64,
    pub kv_8192_bytes: u64,
}
impl MiniMaxSizing {
    /// Includes a conservative 64 MiB allowance for tokenizer and JSON sidecars.
    pub fn install_bytes(&self) -> u64 {
        self.resident_bytes
            .saturating_add(self.expert_file_bytes)
            .saturating_add(64 << 20)
    }
}

pub fn minimax_gguf_sizing(header: &GgufHeader) -> Result<MiniMaxSizing, GgufRepackError> {
    let arch = arch_from_gguf(header)?;
    if arch.family != ModelFamily::MiniMaxM2 {
        return Err(GgufRepackError::UnsupportedFamily {
            family: arch.family.as_str(),
        });
    }
    let plan = plan::classify(header, arch.family, arch.num_layers as usize)?;
    let overflow = || GgufRepackError::ShapeMismatch {
        tensor: "MiniMax sizing".into(),
        detail: "byte count overflow".into(),
    };
    let add = |a: u64, b: u64| a.checked_add(b).ok_or_else(overflow);
    let mul = |a: u64, b: u64| a.checked_mul(b).ok_or_else(overflow);
    let align = |a: u64, n: u64| add(a, n - 1).map(|v| v / n * n);
    let mut index = 24u64;
    let mut payload = 0u64;
    for name in &plan.resident {
        let GgufMapping::Resident(canonical) = map_gguf_name(name, arch.family)? else {
            unreachable!()
        };
        index = add(index, add(72, canonical.len() as u64)?)?;
        let info = &header.tensors[*name];
        let source_bytes = info.byte_size(name)?;
        let fp32 = canonical.ends_with(".mlp.gate.weight")
            || canonical.ends_with(".mlp.e_score_correction_bias");
        if fp32 && info.ggml_type != 0 {
            return Err(GgufRepackError::ShapeMismatch {
                tensor: name.to_string(),
                detail: "MiniMax router and correction bias must be F32".into(),
            });
        }
        let bytes = if info.ggml_type == 0 && !fp32 {
            source_bytes / 2
        } else {
            source_bytes
        };
        payload = add(align(payload, 4)?, bytes)?;
    }
    let resident_bytes = add(align(index, GTURBO_PAGE_BYTES)?, payload)?;
    let mut max_expert_stride = 0;
    let mut expert_file_bytes = 0;
    let mut eight_slot_bytes = 0;
    for sources in plan.routed.values() {
        let mut used = 0;
        for source in sources {
            used = add(used, plan::per_expert_bytes(header, source, &arch)?)?;
        }
        let stride = align(used, GTURBO_PAGE_BYTES)?;
        max_expert_stride = max_expert_stride.max(stride);
        expert_file_bytes = add(expert_file_bytes, mul(stride, arch.num_experts as u64)?)?;
        eight_slot_bytes = add(eight_slot_bytes, mul(stride, 8)?)?;
    }
    let kv_8192_bytes = [
        arch.num_layers as u64,
        arch.num_full_kv_heads as u64,
        arch.full_head_dim as u64,
        8192,
        2,
        2,
    ]
    .into_iter()
    .try_fold(1, mul)?;
    Ok(MiniMaxSizing {
        resident_bytes,
        expert_file_bytes,
        max_expert_stride,
        eight_slot_bytes,
        kv_8192_bytes,
    })
}
