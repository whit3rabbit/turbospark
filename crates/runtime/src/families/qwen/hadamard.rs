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
    /// a folded or inverse name no call site routes through a transform (see
    /// [`routing_error`]), a sign width the butterfly cannot run, or a
    /// `hadamard.bin` shorter than its manifest declares.
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
        if let Some(detail) = routing_error(&section.folded, &section.inverse, &|name| {
            index
                .entries
                .get(name)
                // 4 is the fused-int4 packed dtype `attn.rs`'s fused in-proj
                // arm dispatches on; other dtypes take the per-suffix arm.
                .map(|e| e.dtype == 4)
                .unwrap_or(false)
        }) {
            return unsupported(detail);
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

/// The tensor-name suffixes the dense flow's transform call sites actually
/// route. A folded name outside this set would dispatch its matmul on the RAW
/// activation against rotated weights -- silent wrong math, the exact failure
/// prism's GGUF loader guards against by refusing tensors outside its
/// verified projection kinds -- so the plan refuses it at open with the name
/// in the message. Layer names match by suffix, which covers the same
/// encoders running under the MTP head's prefix for free.
const ROUTED_FOLDED_SUFFIXES: &[&str] = &[
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "linear_attn.in_proj_qkv.weight",
    "linear_attn.in_proj_z.weight",
    "linear_attn.in_proj_a.weight",
    "linear_attn.in_proj_b.weight",
    "linear_attn.out_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
    // The untied head, and the tied head (the embedding resolving as the
    // head by its own name).
    "language_model.lm_head.weight",
    "language_model.model.embed_tokens.weight",
];

/// The one inverse site: the embedding lookup, whose two drivers
/// (`produce.rs` and the chunked `prefill.rs`) both key on this exact name.
const ROUTED_INVERSE_NAME: &str = "language_model.model.embed_tokens.weight";

/// Shared-input groups whose call site keys the transformed buffer on ONE
/// member's name (`q_proj` standing for q/k/v, `gate_proj` for gate/up). A
/// manifest folding part of a group would leave the unlisted siblings reading
/// the raw activation, so each site folds all of a group or none of it.
const SHARED_INPUT_GROUPS: &[&[&str]] = &[
    &[
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
    ],
    &["mlp.gate_proj.weight", "mlp.up_proj.weight"],
];

/// The GDN in-proj quartet. The per-suffix arm routes each member by its own
/// name, but the fused-int4 arm (taken whenever the site's `in_proj_qkv`
/// entry is dtype 4) feeds ONE input buffer to all four resident matrices, so
/// a partial quartet has no correct encoding on that dtype. The real
/// checkpoint's quartet is partial (qkv and z folded, unquantized a and b
/// not) and routes per-suffix because its dtype is not 4.
const IN_PROJ_QUARTET: &[&str] = &[
    "linear_attn.in_proj_qkv.weight",
    "linear_attn.in_proj_z.weight",
    "linear_attn.in_proj_a.weight",
    "linear_attn.in_proj_b.weight",
];

/// Checks a manifest section's folded and inverse names against the routing
/// the flow actually implements (see the constants above). Returns the
/// refusal detail, or `None` when every name routes. `in_proj_qkv_is_int4`
/// reports whether a site's `linear_attn.in_proj_qkv.weight` dispatches the
/// fused-int4 arm, the one case where a partial quartet cannot be expressed.
fn routing_error(
    folded: &[String],
    inverse: &[String],
    in_proj_qkv_is_int4: &dyn Fn(&str) -> bool,
) -> Option<String> {
    for name in folded {
        if !ROUTED_FOLDED_SUFFIXES.iter().any(|s| name.ends_with(s)) {
            return Some(format!(
                "manifest.hadamard folds {name}, which no call site in this flow routes \
                 through a transform; it would decode on the raw activation against \
                 rotated weights"
            ));
        }
    }
    for name in inverse {
        if name != ROUTED_INVERSE_NAME {
            return Some(format!(
                "manifest.hadamard inverses {name}; the only inverse site in this flow is \
                 the embedding lookup at {ROUTED_INVERSE_NAME}"
            ));
        }
    }
    // A site is a name minus its member suffix: the per-layer (or head)
    // prefix one shared call site serves. Each group must be all-or-nothing
    // at every site it appears at, keyed as a bitmask over member indices.
    fn site_masks(folded: &[String], group: &[&str]) -> Vec<(String, usize)> {
        let mut sites: Vec<(String, usize)> = Vec::new();
        for name in folded {
            for (member, suffix) in group.iter().enumerate() {
                if let Some(site) = name.strip_suffix(suffix) {
                    match sites.iter_mut().find(|(s, _)| s == site) {
                        Some((_, mask)) => *mask |= 1 << member,
                        None => sites.push((site.to_string(), 1 << member)),
                    }
                }
            }
        }
        sites
    }
    let partial_group_error = |sites: &[(String, usize)], group: &[&str]| -> Option<String> {
        let full = (1usize << group.len()) - 1;
        sites.iter().find_map(|(site, mask)| {
            (*mask != full).then(|| {
                format!(
                    "manifest.hadamard folds part of the {site}* [{}] group; it shares one \
                     call site and must fold together or not at all",
                    group.join(", ")
                )
            })
        })
    };
    // The attention and FFN groups are unconditional: their call sites key on
    // one member's name whatever the dtypes are.
    for group in SHARED_INPUT_GROUPS {
        if let Some(detail) = partial_group_error(&site_masks(folded, group), group) {
            return Some(detail);
        }
    }
    // The in-proj quartet only on the fused-int4 dtype, where the per-suffix
    // arm is bypassed and one buffer feeds all four matrices.
    for (site, mask) in site_masks(folded, IN_PROJ_QUARTET) {
        if mask != (1usize << IN_PROJ_QUARTET.len()) - 1
            && in_proj_qkv_is_int4(&format!("{site}{}", IN_PROJ_QUARTET[0]))
        {
            return Some(format!(
                "manifest.hadamard folds part of the {site}linear_attn.in_proj_* quartet \
                 on a fused-int4 in_proj, which feeds one input buffer to all four \
                 matrices; the quartet must fold together or not at all"
            ));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::routing_error;

    fn names(names: &[&str]) -> Vec<String> {
        names.iter().map(|s| s.to_string()).collect()
    }

    /// One full-attention layer, one GDN layer, the FFN both share and the
    /// untied head: the shape classes the real checkpoint's 401 folded names
    /// fall into (q/k/v keyed as a group on q_proj, gate/up on gate_proj, the
    /// partial in-proj quartet routing per-suffix on a non-fused dtype).
    #[test]
    fn the_real_checkpoints_name_shape_routes() {
        let folded = names(&[
            "language_model.model.layers.0.self_attn.q_proj.weight",
            "language_model.model.layers.0.self_attn.k_proj.weight",
            "language_model.model.layers.0.self_attn.v_proj.weight",
            "language_model.model.layers.0.self_attn.o_proj.weight",
            "language_model.model.layers.1.linear_attn.in_proj_qkv.weight",
            "language_model.model.layers.1.linear_attn.in_proj_z.weight",
            "language_model.model.layers.1.mlp.gate_proj.weight",
            "language_model.model.layers.1.mlp.up_proj.weight",
            "language_model.model.layers.1.mlp.down_proj.weight",
            "language_model.lm_head.weight",
        ]);
        let inverse = names(&["language_model.model.embed_tokens.weight"]);
        assert_eq!(routing_error(&folded, &inverse, &|_| false), None);
    }

    /// The tied head resolves through the embedding's own name in BOTH lists
    /// (folded as the head's input, inverse as the embedding's output).
    #[test]
    fn the_tied_head_folds_the_embedding_under_both_lists() {
        let embed = "language_model.model.embed_tokens.weight";
        let folded = names(&[
            "language_model.model.layers.0.mlp.gate_proj.weight",
            "language_model.model.layers.0.mlp.up_proj.weight",
            embed,
        ]);
        let inverse = names(&[embed]);
        assert_eq!(routing_error(&folded, &inverse, &|_| false), None);
    }

    #[test]
    fn an_unrouted_folded_name_is_refused_by_name() {
        let folded = names(&["language_model.model.layers.0.self_attn.conv1d.weight"]);
        let detail =
            routing_error(&folded, &[], &|_| false).expect("conv1d is not a routed matmul");
        assert!(
            detail.contains("conv1d"),
            "the refusal names the tensor: {detail}"
        );
    }

    #[test]
    fn an_unrouted_inverse_name_is_refused() {
        let inverse = names(&["language_model.lm_head.weight"]);
        assert!(routing_error(&[], &inverse, &|_| false).is_some());
    }

    #[test]
    fn a_partially_folded_attention_group_is_refused() {
        // q folded, k and v not: the site keys the transformed buffer on
        // q_proj alone, so the k/v matmuls would read the raw activation.
        let folded = names(&[
            "language_model.model.layers.0.self_attn.q_proj.weight",
            "language_model.model.layers.0.self_attn.o_proj.weight",
        ]);
        assert!(routing_error(&folded, &[], &|_| false).is_some());
    }

    #[test]
    fn a_partially_folded_gate_up_group_is_refused() {
        let folded = names(&["language_model.model.layers.0.mlp.gate_proj.weight"]);
        assert!(routing_error(&folded, &[], &|_| false).is_some());
    }

    #[test]
    fn a_partial_in_proj_quartet_refuses_only_on_the_fused_int4_dtype() {
        let folded = names(&[
            "language_model.model.layers.0.linear_attn.in_proj_qkv.weight",
            "language_model.model.layers.0.linear_attn.in_proj_a.weight",
        ]);
        // Non-fused dtype: the per-suffix arm routes each member by its own
        // name, so a partial quartet is contract-legal here.
        assert_eq!(routing_error(&folded, &[], &|_| false), None);
        // Fused-int4 dtype: one buffer feeds all four matrices; no correct
        // encoding exists.
        assert!(routing_error(&folded, &[], &|_| true).is_some());
    }

    #[test]
    fn a_complete_in_proj_quartet_routes_on_the_fused_int4_dtype() {
        let folded = names(&[
            "language_model.model.layers.0.linear_attn.in_proj_qkv.weight",
            "language_model.model.layers.0.linear_attn.in_proj_z.weight",
            "language_model.model.layers.0.linear_attn.in_proj_a.weight",
            "language_model.model.layers.0.linear_attn.in_proj_b.weight",
        ]);
        assert_eq!(routing_error(&folded, &[], &|_| true), None);
    }
}
