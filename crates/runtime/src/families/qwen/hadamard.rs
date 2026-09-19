//! The runtime half of the Hadamard-folded weight contract (the Bonsai-2
//! line, `prism-ml/Ternary-Bonsai-2-27B-mlx-2bit`): the plan built once at
//! open from the manifest's `hadamard` section, the sign vectors uploaded to
//! the GPU, and the transformed-input scratch the dense flow's folded
//! matmuls read.
//!
//! THE CONTRACT, restated from the crate doc (`docs/BONSAI2.md`): a folded
//! checkpoint stores `W' = W * diag(signs) * H_b / sqrt(block)`, so the
//! engine transforms ACTIVATIONS, never weights -- `forward` (signs, then
//! butterflies) on every folded entry's input, `inverse` (butterflies, then
//! signs) on the embedding's dequantized rows. The residual stream is
//! therefore in the ORIGINAL basis everywhere, exactly as prism's bundled
//! runtime keeps it, which is what makes the vision-tower blit arm safe: a
//! tower row lands in the stream untransformed, and the transforms happen at
//! weight-input boundaries only.
//!
//! Per-NAME, not per-family: every folded entry's input picks the
//! transformed buffer; every name absent from `folded` (the checkpoint's
//! unquantized `in_proj_a`/`in_proj_b`, which prism ships F32 in the
//! ORIGINAL basis) reads the raw norm. The sign vectors are shared per input
//! width -- verified byte-equal across all 402 packed modules of the real
//! checkpoint at walk time -- so one buffer per width serves every folded
//! entry of that width.

use std::collections::HashSet;

use model_io::{ManifestHadamard, ResidentIndex};

use crate::real_forward::RealForwardError;
use crate::real_forward_utils::entry;

/// One checkpoint's folded-weight plan: the sign buffers, the folded and
/// inverse name sets, and the single-row scratch the decode flow's transform
/// dispatches write. The chunked prefill runs per token through this same
/// single-row scratch (`prefill_layers.rs`'s per-token arm norms into
/// `scratch.normed` at offset 0), so no second M-row set exists; the
/// batched-GEMV arm refuses a folded install by name instead.
pub(crate) struct HadamardPlan {
    /// Butterfly width in elements; also `signs`' segment size.
    pub(crate) block: u32,
    /// One buffer per distinct folded input width: F32 +/-1, `width` long.
    signs: Vec<(u32, gpu::MetalBuffer)>,
    /// Resident entry names whose INPUT activations carry the forward
    /// transform.
    pub(crate) folded: HashSet<String>,
    /// Resident entry names whose OUTPUT rows carry the inverse transform
    /// (the embedding).
    pub(crate) inverse: HashSet<String>,
    /// `[hidden]`: the embedding lookup's landing row, transformed into
    /// `scratch.x` by the inverse.
    pub(crate) embed_row: gpu::MetalBuffer,
    /// `[hidden]`: the input-layernorm output, forward-transformed; read by
    /// every folded `q/k/v` and GDN `in_proj_qkv`/`in_proj_z`.
    pub(crate) normed_h: gpu::MetalBuffer,
    /// `[hidden]`: the post-attention norm output, forward-transformed; read
    /// by every folded `mlp.gate_proj`/`up_proj`.
    pub(crate) moe_x_h: gpu::MetalBuffer,
    /// `[num_heads * full_head_dim]`: the gated attention output,
    /// forward-transformed for folded `o_proj`.
    pub(crate) attn_out_h: gpu::MetalBuffer,
    /// `[num_v_heads * value_head_dim]`: the gated GDN output,
    /// forward-transformed for folded `out_proj`.
    pub(crate) gdn_out_h: gpu::MetalBuffer,
    /// `[ffn_intermediate]`: the gated FFN activation, forward-transformed
    /// for folded `mlp.down_proj`.
    pub(crate) ffn_act_h: gpu::MetalBuffer,
}

impl HadamardPlan {
    /// Builds the plan from the manifest section and the raw `hadamard.bin`
    /// bytes. Refuses at OPEN anything the flow would otherwise trip over at
    /// token 1: a folded or inverse name the resident index does not carry,
    /// a sign width the butterfly cannot run, or a `hadamard.bin` shorter
    /// than its manifest declares.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn build(
        context: &mut gpu::MetalContext,
        section: &ManifestHadamard,
        signs_bytes: &[u8],
        index: &ResidentIndex,
        hidden: usize,
        q_dim: usize,
        inter: usize,
        value_dim: usize,
    ) -> Result<Self, RealForwardError> {
        let unsupported = |detail: String| Err(RealForwardError::Unsupported(detail));
        if section.folded.is_empty() {
            return unsupported(
                "manifest.hadamard carries no folded entries; an inverse-only contract \
                 has no weight this flow reads"
                    .to_string(),
            );
        }
        let block = section.block as u32;
        let mut signs = Vec::new();
        for s in &section.signs {
            let start = s.offset as usize;
            let end = start + s.bytes as usize;
            let raw = signs_bytes.get(start..end).ok_or_else(|| {
                RealForwardError::Unsupported(format!(
                    "hadamard.bin is {} bytes but its manifest declares signs at [{}..{}]",
                    signs_bytes.len(),
                    start,
                    end
                ))
            })?;
            let buffer = context.new_output_buffer(raw.len() as u64);
            gpu::write_buffer_bytes(&buffer, 0, raw);
            signs.push((s.width as u32, buffer));
        }
        for name in section.folded.iter().chain(section.inverse.iter()) {
            entry(index, name)?;
        }
        // The MoE half of this flow is NOT wired: no folded MoE checkpoint
        // exists, and a router/shared/routed transform would touch call sites
        // this plan does not own. Refusing here names the gap instead of
        // letting a future folded MoE checkpoint decode half-transformed.
        if !section
            .folded
            .iter()
            .any(|n| n.contains(".mlp.gate.weight"))
            && index.entries.keys().any(|k| k.contains(".mlp.gate.weight"))
        {
            return unsupported(
                "this install looks like the MoE half of the qwen flow (routed `.mlp.gate` \
                 present) while carrying a hadamard section; folded MoE is not wired"
                    .to_string(),
            );
        }
        let halfs = |n: usize| context.new_output_buffer((n.max(1) * 2) as u64);
        Ok(Self {
            block,
            signs,
            folded: section.folded.iter().cloned().collect(),
            inverse: section.inverse.iter().cloned().collect(),
            embed_row: halfs(hidden),
            normed_h: halfs(hidden),
            moe_x_h: halfs(hidden),
            attn_out_h: halfs(q_dim),
            gdn_out_h: halfs(value_dim),
            ffn_act_h: halfs(inter),
        })
    }

    pub(crate) fn is_folded(&self, name: &str) -> bool {
        self.folded.contains(name)
    }

    pub(crate) fn is_inverse(&self, name: &str) -> bool {
        self.inverse.contains(name)
    }

    fn signs_for(&self, width: u32) -> Result<&gpu::MetalBuffer, RealForwardError> {
        self.signs
            .iter()
            .find(|(w, _)| *w == width)
            .map(|(_, b)| b)
            .ok_or_else(|| {
                RealForwardError::Unsupported(format!(
                    "no manifest.hadamard sign vector for activation width {width}; the \
                     install's section does not cover every folded input width"
                ))
            })
    }

    /// `dst[..] = fwht(src[..])` over `rows` contiguous rows of `width`
    /// halfs. `forward` is signs-then-butterfly; the inverse (butterfly then
    /// signs) is the same dispatch with it cleared.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn transform(
        &self,
        context: &mut gpu::MetalContext,
        pass: &gpu::PassEncoder,
        src: (&gpu::MetalBuffer, u64),
        dst: (&gpu::MetalBuffer, u64),
        rows: u32,
        width: u32,
        forward: bool,
    ) -> Result<(), RealForwardError> {
        let signs = self.signs_for(width)?;
        gpu::encode_hadamard_fwht(
            context,
            pass,
            src,
            dst,
            (signs, 0),
            rows,
            width,
            self.block,
            forward,
        )
        .map_err(RealForwardError::Gpu)
    }
}
