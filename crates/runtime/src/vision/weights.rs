//! Where the tower's weights are: the nine resident tensors as offsets into
//! the shared resident buffer, and the twelve per-block roles as offsets into
//! whichever streamer slot the block was read into (ROADMAP M-V4).
//!
//! # This is the tower's only weight reader, and that is a CONSTRAINT
//!
//! `readable_resident_dtype` honours FP16 (tag 2) under the `vision.` prefix
//! and refuses it everywhere else, because `norm_view` and `read_bf16_host`
//! are dtype-BLIND -- they resolve an unquantized tensor by byte width and
//! decode it as BF16, so an FP16 tensor either of them reaches is misread
//! rather than rejected (AGENTS.md Gotcha 45). What makes the scoped
//! exception safe is that nothing under `vision.` is reachable from a
//! text-path helper. [`fp16_view`] below is why: it checks the tag as well as
//! the width, so this module cannot accidentally become the general reader.

use model_io::ResidentIndex;

use crate::real_forward_layout::DTYPE_RAW_FP16;
use crate::real_forward_types::RealForwardError;
use crate::vision::shape::VisionShape;

/// The twelve per-block roles, in forward-pass order.
///
/// The same order and the same spellings as
/// `turbospark_repack::VISION_BLOCK_ROLES`, which is what the walk writes
/// into each blob's `sub_tensors` map. Restated here rather than imported
/// because `crates/repack` is a DEV-dependency of this crate: taking a
/// production dependency on the repack walk to read six-character strings
/// would invert the dependency direction for nothing.
/// `the_role_table_matches_the_writers` asserts the two agree, so the
/// duplication cannot drift silently.
pub(crate) const BLOCK_ROLES: [&str; 12] = [
    "ln1_w", "ln1_b", "qkv_w", "qkv_b", "proj_w", "proj_b", "ln2_w", "ln2_b", "fc1_w", "fc1_b",
    "fc2_w", "fc2_b",
];

/// Index into [`BLOCK_ROLES`] and into [`BlockRoles::offsets`].
#[derive(Debug, Clone, Copy)]
pub(crate) enum Role {
    Ln1W = 0,
    Ln1B = 1,
    QkvW = 2,
    QkvB = 3,
    ProjW = 4,
    ProjB = 5,
    Ln2W = 6,
    Ln2B = 7,
    Fc1W = 8,
    Fc1B = 9,
    Fc2W = 10,
    Fc2B = 11,
}

/// One block's twelve sub-tensor offsets, relative to the blob's start (which
/// is offset 0 of the slot buffer it is read into).
#[derive(Debug, Clone)]
pub(crate) struct BlockRoles {
    pub(crate) offsets: [u64; 12],
}

impl BlockRoles {
    pub(crate) fn at(&self, role: Role) -> u64 {
        self.offsets[role as usize]
    }
}

/// The nine non-block tensors, as byte offsets into the resident GPU buffer.
///
/// Offsets rather than `(buffer, offset)` pairs because every one of them
/// binds the SAME buffer: the whole resident region is one zero-copy
/// `MTLBuffer` (`gpu::ResidentGpuWeights`), exactly as every text projection
/// binds it.
#[derive(Debug, Clone)]
pub(crate) struct VisionResident {
    pub(crate) patch_w: u64,
    pub(crate) patch_b: u64,
    pub(crate) merger_norm_w: u64,
    pub(crate) merger_norm_b: u64,
    pub(crate) merger_fc1_w: u64,
    pub(crate) merger_fc1_b: u64,
    pub(crate) merger_fc2_w: u64,
    pub(crate) merger_fc2_b: u64,
    /// `pos_embed.weight`, kept as a HOST offset (into the mmap) rather than a
    /// GPU one: the interpolation is a gather with per-patch weights and this
    /// port has no gather kernel, so the table is read to host once at open
    /// and blended there. See `stages::pos_embed_rows`.
    pub(crate) pos_embed_host: usize,
    /// One entry per deepstack merger, in `VisionShape::deepstack` order:
    /// merger `k`'s output is injected after TRUNK layer `k`. EMPTY for a
    /// tower whose config declares no deepstack indexes.
    pub(crate) deepstack_mergers: Vec<DeepstackMergerWeights>,
}

/// One deepstack merger's six resident offsets.
///
/// The same six roles as the main merger, with one structural difference the
/// offsets alone cannot show: the reference builds the deepstack mergers with
/// `use_postshuffle_norm=True` (`mlx_vlm/models/qwen3_vl/vision.py`), so the
/// norm is sized `hidden * merge^2` and runs on the CONCATENATED
/// `[merged, merger_input]` rows, where the main merger norms per patch row
/// and reshapes after. The checkpoint's own shapes agree (the deepstack norm
/// weight is [4096] where the main one is [1024] on the 4B tower), and
/// `resolve` sizes every entry from [`VisionShape`] so a tensor at the wrong
/// width is refused rather than read as a different number.
#[derive(Debug, Clone)]
pub(crate) struct DeepstackMergerWeights {
    pub(crate) norm_w: u64,
    pub(crate) norm_b: u64,
    pub(crate) fc1_w: u64,
    pub(crate) fc1_b: u64,
    pub(crate) fc2_w: u64,
    pub(crate) fc2_b: u64,
}

/// Resolve a `vision.`-prefixed resident tensor to its GPU byte offset,
/// checking BOTH its dtype tag and its byte count.
///
/// The tag check is the load-bearing half. A BF16 tensor of the same element
/// count has the same `size_bytes`, so a width check alone would accept one
/// and the kernels -- which bind `half` -- would read it as a different
/// number, finite and wrong. This is the mirror of the hazard the name-scoped
/// gate exists to prevent, pointing the other way.
fn fp16_view(
    index: &ResidentIndex,
    name: &str,
    expect_elems: usize,
) -> Result<u64, RealForwardError> {
    let e = index
        .entries
        .get(name)
        .ok_or_else(|| RealForwardError::MissingTensor(name.to_string()))?;
    if e.dtype != DTYPE_RAW_FP16 {
        return Err(RealForwardError::Unsupported(format!(
            "vision tensor {name} carries dtype {}, but the tower's kernels bind FP16 (tag {}); \
             a same-width BF16 tensor would be read as a different number rather than refused",
            e.dtype, DTYPE_RAW_FP16
        )));
    }
    if e.size_bytes as usize != expect_elems * 2 {
        return Err(RealForwardError::Unsupported(format!(
            "vision tensor {name}: {} bytes does not match expected FP16 [{expect_elems}]",
            e.size_bytes
        )));
    }
    Ok(e.file_offset - index.header.index_size)
}

/// The prefix the walk writes the tower's resident tensors under.
///
/// `turbospark_repack`'s `VISION_INSTALL_PREFIX`, restated for
/// [`BLOCK_ROLES`]' reason. `the_resident_prefix_matches_the_writers` pins the
/// agreement.
pub(crate) use crate::real_forward_layout::VISION_PREFIX;

/// The deepstack mergers' name level under [`VISION_PREFIX`].
///
/// `turbospark_repack`'s `VISION_DEEPSTACK_MERGER_PREFIX`, restated for the
/// same reason; `the_deepstack_prefix_matches_the_writers` pins it.
pub(crate) const DEEPSTACK_MERGER_PREFIX: &str = "deepstack_merger_list.";

impl VisionResident {
    /// Resolve all nine off the resident index.
    ///
    /// Every shape is stated from [`VisionShape`] rather than read off the
    /// entry, so a tensor of the right dtype and the wrong shape is refused
    /// here instead of producing a correctly-sized GEMM over the wrong
    /// matrix.
    pub(crate) fn resolve(
        index: &ResidentIndex,
        shape: &VisionShape,
    ) -> Result<Self, RealForwardError> {
        let h = shape.hidden;
        let m = shape.merger_input();
        let name = |tail: &str| format!("{VISION_PREFIX}{tail}");

        // `patch_embed.proj.weight` is RANK 5 in the checkpoint -- `[out, T,
        // P_h, P_w, C]` -- and the walk records it flattened to its GEMM
        // shape `(hidden, patch_dim)`, verbatim, with NO permutation. M-V1
        // emits patch rows in `(T, P_h, P_w, C)` order for exactly this
        // reason (`crates/vision-io` Gotcha 1), so both operands read
        // row-major and neither side transposes. Transposing here mixes the
        // wrong channels into the wrong positions and the tower reads a
        // different image, fluently.
        let patch_w = fp16_view(index, &name("patch_embed.proj.weight"), h * shape.patch_dim)?;
        let patch_b = fp16_view(index, &name("patch_embed.proj.bias"), h)?;
        let pos_entry = index
            .entries
            .get(&name("pos_embed.weight"))
            .ok_or_else(|| RealForwardError::MissingTensor(name("pos_embed.weight")))?;
        // Checked through the same helper so the dtype gate applies, then
        // kept as a HOST offset.
        let _ = fp16_view(index, &name("pos_embed.weight"), shape.pos_rows * h)?;
        let pos_embed_host = (pos_entry.file_offset - index.header.index_size) as usize;

        Ok(Self {
            patch_w,
            patch_b,
            // The merger's norm is over `hidden` PER PATCH ROW, not over the
            // merged 4608-wide row: the reference builds its `PatchMerger`
            // with `use_postshuffle_norm=False`, so `norm` is sized
            // `config.hidden_size` and the reshape happens AFTER it. A norm
            // over the wide row is a different function that still runs.
            merger_norm_w: fp16_view(index, &name("merger.norm.weight"), h)?,
            merger_norm_b: fp16_view(index, &name("merger.norm.bias"), h)?,
            merger_fc1_w: fp16_view(index, &name("merger.linear_fc1.weight"), m * m)?,
            merger_fc1_b: fp16_view(index, &name("merger.linear_fc1.bias"), m)?,
            merger_fc2_w: fp16_view(
                index,
                &name("merger.linear_fc2.weight"),
                shape.out_hidden * m,
            )?,
            merger_fc2_b: fp16_view(index, &name("merger.linear_fc2.bias"), shape.out_hidden)?,
            pos_embed_host,
            deepstack_mergers: Self::resolve_deepstack(index, shape, m)?,
        })
    }

    /// Resolve one [`DeepstackMergerWeights`] per declared index.
    ///
    /// DECLARED MEANS REQUIRED: a config that names deepstack indexes against
    /// an install missing the merger tensors is refused here rather than
    /// injecting nothing, because a tower that silently skips its deepstack
    /// produces a competent-but-wrong trunk -- the exact failure class the
    /// all-slots refusals exist for. The widths are the POST-SHUFFLE ones
    /// (`use_postshuffle_norm=True`): the norm runs on `merger_input` rows,
    /// not on `hidden` rows like the main merger's.
    fn resolve_deepstack(
        index: &ResidentIndex,
        shape: &VisionShape,
        m: usize,
    ) -> Result<Vec<DeepstackMergerWeights>, RealForwardError> {
        let name = |k: usize, tail: &str| {
            format!("{VISION_PREFIX}{DEEPSTACK_MERGER_PREFIX}{k}.{tail}")
        };
        let mut mergers = Vec::with_capacity(shape.deepstack.len());
        for k in 0..shape.deepstack.len() {
            mergers.push(DeepstackMergerWeights {
                norm_w: fp16_view(index, &name(k, "norm.weight"), m)?,
                norm_b: fp16_view(index, &name(k, "norm.bias"), m)?,
                fc1_w: fp16_view(index, &name(k, "linear_fc1.weight"), m * m)?,
                fc1_b: fp16_view(index, &name(k, "linear_fc1.bias"), m)?,
                fc2_w: fp16_view(index, &name(k, "linear_fc2.weight"), shape.out_hidden * m)?,
                fc2_b: fp16_view(index, &name(k, "linear_fc2.bias"), shape.out_hidden)?,
            });
        }
        Ok(mergers)
    }
}

/// Resolve one block's twelve roles off its `packed_vision/layout.json`
/// entry.
///
/// # Two refusals, both by name
///
/// **Every role must be FP16.** The walk writes `"fp16"` for all twelve; a
/// hand-edited layout is what this catches, and the failure it prevents is
/// the same misread `fp16_view` guards on the resident side.
///
/// **Every offset must be 4-byte aligned.** Sub-tensors are packed
/// back-to-back with no per-role padding, so alignment is a property of the
/// preceding tensors' sizes rather than of the format. The real tower's
/// element counts are all even and every offset lands, which is exactly the
/// kind of accident worth asserting: Metal's `setBuffer:offset:` requires it,
/// and a violation is a validation-layer abort a long way from the layout
/// that produced it.
/// Expected byte size for a block role given the tower's shape.
pub(crate) fn role_size(role: &str, shape: &VisionShape) -> Option<u64> {
    let h = shape.hidden as u64;
    let inter = shape.intermediate as u64;
    let elems = match role {
        "ln1_w" | "ln1_b" | "ln2_w" | "ln2_b" | "proj_b" | "fc2_b" => h,
        "qkv_w" => 3 * h * h,
        "qkv_b" => 3 * h,
        "proj_w" => h * h,
        "fc1_w" => inter * h,
        "fc1_b" => inter,
        "fc2_w" => h * inter,
        _ => return None,
    };
    Some(elems * 2)
}

pub(crate) fn resolve_block_roles(
    entry: &model_io::ExpertEntry,
    block: usize,
    stride: u64,
    shape: &VisionShape,
) -> Result<BlockRoles, RealForwardError> {
    let mut offsets = [0u64; 12];
    for (i, role) in BLOCK_ROLES.iter().enumerate() {
        let sub = entry.sub_tensors.get(*role).ok_or_else(|| {
            RealForwardError::MissingTensor(format!(
                "vision block {block} is missing the sub-tensor {role}"
            ))
        })?;
        if sub.dtype != "fp16" {
            return Err(RealForwardError::Unsupported(format!(
                "vision block {block} role {role} carries dtype {:?}, but the tower's kernels \
                 bind FP16; a same-width type would be read as a different number",
                sub.dtype
            )));
        }
        if let Some(expected) = role_size(role, shape) {
            if sub.size != expected {
                return Err(RealForwardError::Unsupported(format!(
                    "vision block {block} role {role} size {} does not match expected size {expected}",
                    sub.size
                )));
            }
        }
        if sub.offset % 4 != 0 {
            return Err(RealForwardError::Unsupported(format!(
                "vision block {block} role {role} sits at blob offset {}, which is not 4-byte \
                 aligned; Metal cannot bind a buffer there",
                sub.offset
            )));
        }
        if sub.offset + sub.size > stride {
            return Err(RealForwardError::Unsupported(format!(
                "vision block {block} role {role} spans {}..{} of a {stride}-byte blob",
                sub.offset,
                sub.offset + sub.size
            )));
        }
        offsets[i] = sub.offset;
    }
    Ok(BlockRoles { offsets })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn test_shape() -> VisionShape {
        VisionShape {
            depth: 1,
            hidden: 16,
            intermediate: 32,
            heads: 2,
            head_dim: 8,
            merge: 2,
            out_hidden: 16,
            patch_dim: 16,
            pos_rows: 16,
            pos_side: 4,
            patch_size: 16,
            deepstack: Vec::new(),
        }
    }

    fn make_entry_with_sub(role: &str, size: u64) -> model_io::ExpertEntry {
        let shape = test_shape();
        let mut sub_tensors = BTreeMap::new();
        let mut offset = 0u64;
        for &r in &BLOCK_ROLES {
            let r_size = if r == role {
                size
            } else {
                role_size(r, &shape).unwrap()
            };
            sub_tensors.insert(
                r.to_string(),
                model_io::SubTensorEntry {
                    offset,
                    size: r_size,
                    dtype: "fp16".to_string(),
                },
            );
            offset += r_size;
        }
        model_io::ExpertEntry {
            expert: 0,
            offset: 0,
            size: offset,
            sub_tensors,
        }
    }

    #[test]
    fn block_roles_validates_exact_sizes() {
        let shape = test_shape();
        let entry = make_entry_with_sub("ln1_w", role_size("ln1_w", &shape).unwrap());
        assert!(resolve_block_roles(&entry, 0, 100_000, &shape).is_ok());
    }

    #[test]
    fn block_role_wrong_size_is_refused() {
        let shape = test_shape();
        let wrong_size = role_size("ln1_w", &shape).unwrap() + 2;
        let entry = make_entry_with_sub("ln1_w", wrong_size);
        let err = resolve_block_roles(&entry, 0, 100_000, &shape).expect_err("wrong size");
        match err {
            RealForwardError::Unsupported(msg) => {
                assert!(
                    msg.contains("role ln1_w size 34 does not match expected size 32"),
                    "msg: {msg}"
                );
            }
            other => panic!("expected Unsupported error, got {other:?}"),
        }
    }
}
