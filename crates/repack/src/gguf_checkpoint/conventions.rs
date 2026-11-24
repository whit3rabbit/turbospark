//! GGUF source tensor layout conventions (V-head de-interleaving and RoPE rotary unpermutations).

use model_io::{ArchConfig, ModelFamily};

use super::types::GgufRepackError;

struct VHeadAxis {
    columns: bool,
    base: usize,
    span: usize,
}

#[derive(Clone, Copy)]
struct VHeadLayout {
    /// The source already stores grouped V heads (`qwen3next`), so the restore
    /// is the identity. Only `ssm_a` -> `A_log` still needs the walk.
    already_grouped: bool,
    num_k_heads: usize,
    num_v_heads: usize,
    values_per_k_head: usize,
}

fn v_head_axis(
    name: &str,
    canonical: &str,
    arch: &ArchConfig,
) -> Result<Option<VHeadAxis>, GgufRepackError> {
    // Qwen's gated-DeltaNet families. The convention belongs to llama.cpp's
    // converter. Restore the grouped V heads used by the runtime from the
    // converter's tiled order; the V/K ratio varies by family.
    //
    // **THE DENSE HALF WAS EXCLUDED AND IT FAILED EXACTLY AS Gotcha 33 SAYS
    // THIS CLASS FAILS**: the first `qwen35` install loaded, decoded, never
    // errored, and produced word salad. Nothing upstream can see it -- every
    // name mapped, every shape checked out, the manifest validated. Only
    // running the model does, which is why the convention check is a Phase 1
    // gate rather than a tidy-up.
    if !matches!(
        arch.family,
        ModelFamily::QwenGdnMoe | ModelFamily::QwenGdnDense | ModelFamily::Qwen4Exp
    ) {
        return Ok(None);
    }
    let la = &arch.linear_attention;
    let rows = |base: usize, span: usize| {
        Ok(Some(VHeadAxis {
            columns: false,
            base,
            span,
        }))
    };
    let dimension_error = || GgufRepackError::ShapeMismatch {
        tensor: name.to_string(),
        detail: "V-head dimensions are not positive and representable".to_string(),
    };
    let num_k_heads = usize::try_from(la.num_k_heads).map_err(|_| dimension_error())?;
    let key_head_dim = usize::try_from(la.key_head_dim).map_err(|_| dimension_error())?;
    let width = usize::try_from(la.value_head_dim).map_err(|_| dimension_error())?;
    let v_at = num_k_heads
        .checked_mul(key_head_dim)
        .and_then(|v| v.checked_mul(2))
        .ok_or_else(dimension_error)?;
    let Some((_, suffix)) = canonical.rsplit_once("linear_attn.") else {
        return Ok(None);
    };
    // A `qwen3next` GGUF keeps HF's grouped V-head order (llama.cpp reorders
    // to tiled only for Qwen 3.5), so nothing but the `A_log` value transform
    // applies to it. Returning `None` here also keeps the block-split refusal
    // away from tensors that are never shuffled.
    if la.source_v_heads_grouped && suffix != "A_log" {
        return Ok(None);
    }
    match suffix {
        "A_log" | "dt_bias" | "in_proj_a.weight" | "in_proj_b.weight" => rows(0, 1),
        "conv1d.weight" | "in_proj_qkv.weight" => rows(v_at, width),
        "in_proj_z.weight" => rows(0, width),
        "out_proj.weight" => Ok(Some(VHeadAxis {
            columns: true,
            base: 0,
            span: width,
        })),
        _ => Ok(None),
    }
}

fn v_head_layout(name: &str, arch: &ArchConfig) -> Result<VHeadLayout, GgufRepackError> {
    let num_k_heads = usize::try_from(arch.linear_attention.num_k_heads).map_err(|_| {
        GgufRepackError::ShapeMismatch {
            tensor: name.to_string(),
            detail: format!(
                "the V-head reorder needs a representable num_k_heads, got {}",
                arch.linear_attention.num_k_heads
            ),
        }
    })?;
    let num_v_heads = usize::try_from(arch.linear_attention.num_v_heads).map_err(|_| {
        GgufRepackError::ShapeMismatch {
            tensor: name.to_string(),
            detail: format!(
                "the V-head reorder needs a representable num_v_heads, got {}",
                arch.linear_attention.num_v_heads
            ),
        }
    })?;
    if num_k_heads == 0 || num_v_heads == 0 || num_v_heads % num_k_heads != 0 {
        return Err(GgufRepackError::ShapeMismatch {
            tensor: name.to_string(),
            detail: format!(
                "V-head reorder needs positive num_k_heads and num_v_heads divisible by it, \
                 got {num_k_heads} and {num_v_heads}"
            ),
        });
    }
    Ok(VHeadLayout {
        already_grouped: arch.linear_attention.source_v_heads_grouped,
        num_k_heads,
        num_v_heads,
        values_per_k_head: num_v_heads / num_k_heads,
    })
}

/// Restore grouped `[key_head, value_head_within_key, span]` order from the
/// converter's tiled `[value_head_within_key, key_head, span]` order.
fn restore_grouped_v_heads<T: Copy>(data: &mut [T], base: usize, span: usize, layout: VHeadLayout) {
    if layout.already_grouped {
        return;
    }
    let end = base + layout.num_v_heads * span;
    let source = data[base..end].to_vec();
    for key_head in 0..layout.num_k_heads {
        for value_head in 0..layout.values_per_k_head {
            let grouped_head = key_head * layout.values_per_k_head + value_head;
            let tiled_head = value_head * layout.num_k_heads + key_head;
            data[base + grouped_head * span..base + (grouped_head + 1) * span]
                .copy_from_slice(&source[tiled_head * span..(tiled_head + 1) * span]);
        }
    }
}

/// Undo llama.cpp's ROTARY PAIR PERMUTATION on a `llama`-architecture
/// `attn_q` / `attn_k` (ROADMAP Phase M2).
///
/// **The same class as the Qwen V-head convention above -- a source that is
/// right about every NAME and wrong about what a tensor MEANS (AGENTS.md
/// Gotcha 33) -- and it was found the same way: the real install opened,
/// decoded, and produced degenerate text.**
///
/// ggml rotates ADJACENT pairs `(2i, 2i+1)` for this architecture, where this
/// port's `rope_proportional_neox` rotates half-split pairs
/// `(i, head_dim/2 + i)`. Same rotation, different element pairing, and
/// llama.cpp's HF converter absorbs the difference into the WEIGHTS: it
/// reshapes each head's rows to `(2, D/2)` and swaps to `(D/2, 2)`. So GGUF
/// row `j` holds what the original layout kept at `b * (D/2) + a` for
/// `a = j / 2`, `b = j % 2`, and undoing it means reading install row `i`
/// from GGUF row `(i % (D/2)) * 2 + i / (D/2)`.
///
/// Rows only, and rows are contiguous byte runs in every block layout here,
/// so this moves bytes without decoding any -- the same property that let the
/// Qwen fix permute quantized tensors directly.
///
/// V is NOT touched: it never goes through RoPE.
fn rotary_row_source(i: usize, head_dim: usize) -> usize {
    let half = head_dim / 2;
    (i % half) * 2 + i / half
}

/// True for the two tensors that need it, on the families that do: `llama`,
/// and Muse Glimmer, whose converter applies the same permutation
/// (`conversion/muse_glimmer.py::_unpermute_for_rope`, the inverse of
/// transformers' `_permute_for_rope`, so ggml gets interleaved pairs).
fn needs_rotary_unpermute(canonical: &str, arch: &ArchConfig) -> bool {
    matches!(arch.family, ModelFamily::Llama | ModelFamily::MuseGlimmer)
        && (canonical.ends_with("self_attn.q_proj.weight")
            || canonical.ends_with("self_attn.k_proj.weight"))
}

/// Muse Glimmer's four per-layer norms. The converter adds 1 to every tensor
/// named `*layernorm.weight` (`MuseGlimmerModel.norm_shift`), and this
/// family's flow applies `1 + w` itself, so the install needs the stored
/// value back. The final norm (`model.norm.weight`) is not shifted by the
/// converter and is not matched here.
fn is_muse_centered_norm(canonical: &str, arch: &ArchConfig) -> bool {
    arch.family == ModelFamily::MuseGlimmer
        && canonical.starts_with("language_model.model.layers.")
        && [
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ]
        .iter()
        .any(|n| canonical.ends_with(&format!(".{n}.weight")))
}

/// Every Gemma 3 norm tensor. llama.cpp's converter adds 1 to each tensor
/// whose name ends `norm.weight` (the four per-layer norms, the per-head q/k
/// norms AND the final norm), because Gemma 3's RMSNorm computes
/// `x * (1 + w)`. This family's flow applies `1 + w` itself
/// (`encode_rms_norm_bf16w_centered`), so the install needs the stored value
/// back. Unlike Muse Glimmer the FINAL norm is shifted too.
fn is_gemma3_centered_norm(canonical: &str, arch: &ArchConfig) -> bool {
    // Gemma 2 bakes `+1` into every norm the same way (`GemmaModel`).
    matches!(arch.family, ModelFamily::Gemma3 | ModelFamily::Gemma2)
        && canonical.ends_with("norm.weight")
}

/// The dense `qwen35` multi-token-prediction head's seven norms. llama.cpp's
/// converter adds 1 to every `*norm.weight` but `linear_attn.norm`
/// (`Qwen3NextModel.modify_tensors`), and its rename to `blk.<n>.nextn.*`
/// runs first, so the head's norms are shifted along with the trunk's. The
/// runtime reads the TRUNK's plain (`x * w`, the shift is wanted) and the
/// HEAD's centered (`x * (1 + w)`, `QkNormConvention::Centered` and
/// `encode_rms_norm_bf16w_centered` in `families/qwen/mtp.rs`), because
/// mlx-vlm leaves the head unshifted. So only the head gets the value back.
/// Reading it shifted is not subtle: `docs/MTP.md` measured the true token
/// at median rank 248,308 of 248,320.
///
/// AN EXPLICIT LIST, not a suffix test on the canonical name. llama.cpp's
/// rule matches its OWN names (`enorm`, `hnorm`), and two of their canonical
/// names (`pre_fc_norm_embedding`, `pre_fc_norm_hidden`) do not end in
/// `norm.weight`, so a canonical suffix test silently skips them.
fn is_qwen_gdn_mtp_head_norm(canonical: &str, arch: &ArchConfig) -> bool {
    const HEAD_NORMS: [&str; 7] = [
        "mtp.pre_fc_norm_embedding.weight",
        "mtp.pre_fc_norm_hidden.weight",
        "mtp.norm.weight",
        "mtp.layers.0.input_layernorm.weight",
        "mtp.layers.0.post_attention_layernorm.weight",
        "mtp.layers.0.self_attn.q_norm.weight",
        "mtp.layers.0.self_attn.k_norm.weight",
    ];
    arch.family == ModelFamily::QwenGdnDense && HEAD_NORMS.contains(&canonical)
}

fn unpermute_rotary_rows(
    name: &str,
    rows: usize,
    head_dim: usize,
    bytes: &mut [u8],
) -> Result<(), GgufRepackError> {
    let shape_err = |detail: String| GgufRepackError::ShapeMismatch {
        tensor: name.to_string(),
        detail,
    };
    if head_dim == 0 || head_dim % 2 != 0 || rows == 0 || rows % head_dim != 0 {
        return Err(shape_err(format!(
            "{rows} rows do not split into whole heads of {head_dim}"
        )));
    }
    if bytes.len() % rows != 0 {
        return Err(shape_err(format!(
            "{} bytes is not a whole number of {rows} rows",
            bytes.len()
        )));
    }
    let row_bytes = bytes.len() / rows;
    let original = bytes.to_vec();
    for head_start in (0..rows).step_by(head_dim) {
        for i in 0..head_dim {
            let src = head_start + rotary_row_source(i, head_dim);
            let dst = head_start + i;
            bytes[dst * row_bytes..(dst + 1) * row_bytes]
                .copy_from_slice(&original[src * row_bytes..(src + 1) * row_bytes]);
        }
    }
    Ok(())
}

pub(crate) fn apply_source_convention(
    name: &str,
    canonical: &str,
    arch: &ArchConfig,
    cols: usize,
    values: &mut [f32],
) -> Result<(), GgufRepackError> {
    if is_muse_centered_norm(canonical, arch)
        || is_gemma3_centered_norm(canonical, arch)
        || is_qwen_gdn_mtp_head_norm(canonical, arch)
    {
        // Exact for the published file: the converter's `w + 1` is computed
        // in f32 on a BF16 weight, and the difference reproduces that weight
        // (measured against `mlx-community/Muse-Glimmer-30B-4bit`).
        for v in values.iter_mut() {
            *v -= 1.0;
        }
        return Ok(());
    }
    let Some(axis) = v_head_axis(name, canonical, arch)? else {
        return Ok(());
    };
    let layout = v_head_layout(name, arch)?;
    let shape_err = |detail: String| GgufRepackError::ShapeMismatch {
        tensor: name.to_string(),
        detail,
    };
    if axis.columns {
        if cols == 0 || values.len() % cols != 0 {
            return Err(shape_err(format!(
                "{} values do not form whole rows of {cols} columns",
                values.len()
            )));
        }
        let end = axis
            .span
            .checked_mul(layout.num_v_heads)
            .and_then(|body| axis.base.checked_add(body));
        if end != Some(cols) {
            return Err(shape_err(format!(
                "{cols} columns do not fill {} + {} x {} V-head values",
                axis.base, layout.num_v_heads, axis.span
            )));
        }
        let rows = values.len() / cols;
        for row in 0..rows {
            restore_grouped_v_heads(values, row * cols + axis.base, axis.span, layout);
        }
        return Ok(());
    }
    let base = axis.base.checked_mul(cols);
    let span = axis.span.checked_mul(cols);
    let end = span
        .and_then(|span| layout.num_v_heads.checked_mul(span))
        .and_then(|body| base.and_then(|base| base.checked_add(body)));
    let (Some(base), Some(span), Some(end)) = (base, span, end) else {
        return Err(shape_err("V-head shape arithmetic overflowed".to_string()));
    };
    if span == 0 || end != values.len() {
        return Err(shape_err(format!(
            "{} values do not fill {base} + {} x {span}",
            values.len(),
            layout.num_v_heads
        )));
    }

    if canonical.ends_with("linear_attn.A_log") {
        for v in values.iter_mut() {
            if !v.is_finite() || *v >= 0.0 {
                return Err(shape_err(format!(
                    "ssm_a holds -exp(A_log) and must be finite and negative, found {v}"
                )));
            }
            *v = (-*v).ln();
        }
    }
    restore_grouped_v_heads(values, base, span, layout);
    Ok(())
}

/// The refusal text when a `span`-column V head is not a whole number of
/// `ggml_type` blocks, or `None` when the byte de-interleave is safe.
fn block_split_detail(span: usize, ggml_type: u32) -> Option<String> {
    let block_elements = crate::gguf_header::ggml_type_block(ggml_type)
        .map(|(elements, _)| elements as usize)
        .unwrap_or(1);
    (block_elements > 1 && span % block_elements != 0).then(|| {
        format!(
            "a {span}-column V head is not a whole number of {block_elements}-element \
             blocks, so the de-interleave would split one: dequantizing is the only \
             way to permute this tensor, and this walk copies bytes"
        )
    })
}

/// Header-only form of the whole-block refusal in
/// `apply_source_convention_bytes`: `Some(reason)` when the install walk
/// would refuse this tensor because a V head splits a quantization block.
///
/// The catalog probe calls this so it cannot report RUNNABLE for a file the
/// walk refuses (an Ornith `qwen3_5` with a Q4_K `ssm_out` did exactly that:
/// every type had a kernel, and the probe only checked types). Qwen4Exp's
/// dequantized types never reach the byte path, so they return `None`.
pub fn gguf_v_head_block_split(
    name: &str,
    canonical: &str,
    arch: &ArchConfig,
    ggml_type: u32,
) -> Option<String> {
    if arch.family == ModelFamily::Qwen4Exp
        && super::transcode::qwen4exp_tensor_is_transcoded(canonical, ggml_type)
    {
        return None;
    }
    // A malformed axis is refused by the walk's own shape checks; this
    // predicate answers only the block question.
    let axis = v_head_axis(name, canonical, arch).ok()??;
    if !axis.columns {
        return None;
    }
    block_split_detail(axis.span, ggml_type)
}

pub(crate) fn apply_source_convention_bytes(
    name: &str,
    canonical: &str,
    arch: &ArchConfig,
    (rows, cols): (usize, usize),
    ggml_type: u32,
    bytes: &mut [u8],
) -> Result<(), GgufRepackError> {
    if needs_rotary_unpermute(canonical, arch) {
        return unpermute_rotary_rows(name, rows, arch.full_head_dim as usize, bytes);
    }
    let Some(axis) = v_head_axis(name, canonical, arch)? else {
        return Ok(());
    };
    let layout = v_head_layout(name, arch)?;
    let shape_err = |detail: String| GgufRepackError::ShapeMismatch {
        tensor: name.to_string(),
        detail,
    };
    let along = if axis.columns { cols } else { rows };
    let end = layout
        .num_v_heads
        .checked_mul(axis.span)
        .and_then(|body| axis.base.checked_add(body));
    if axis.span == 0 || end != Some(along) {
        return Err(shape_err(format!(
            "a {rows}x{cols} tensor's V axis does not fill {} + {} x {}",
            axis.base, layout.num_v_heads, axis.span
        )));
    }
    if rows == 0 || bytes.len() % rows != 0 {
        return Err(shape_err(format!(
            "{} bytes is not a whole number of {rows} rows",
            bytes.len()
        )));
    }
    let row_bytes = bytes.len() / rows;

    if !axis.columns {
        restore_grouped_v_heads(bytes, axis.base * row_bytes, axis.span * row_bytes, layout);
        return Ok(());
    }
    // **WHOLE BLOCKS, NOT MERELY WHOLE BYTES.** A byte-divisibility check is
    // necessary and NOT sufficient, and the gap is not hypothetical: on a
    // Q4_K `out_proj` at 4096 columns a 128-column head is 72 bytes, which
    // divides evenly and is HALF a 144-byte superblock. Permuting at that
    // granularity cuts through the packed 6-bit scale header, so the tensor
    // dequantizes to non-finite values -- an install that writes, validates,
    // opens, and dies at the sampler with "score vector must not contain a
    // non-finite entry", four layers from the cause.
    //
    // The condition is on ELEMENTS: a head must be a whole number of blocks.
    // Q8_0's block is 32 elements and 128 is a multiple of it, which is why
    // every Qwen install before Ornith passed -- `Q4_K_M` puts Q8_0 on the
    // attention and gated-DeltaNet tensors (AGENTS.md Gotcha 29) and only
    // Ornith's converter put `ssm_out` at Q4_K. Same trap as Gotcha 37: a
    // check that is correct for exactly as long as one file exercises it.
    //
    // REFUSED rather than shuffled by bytes when a block spans heads from
    // different key-head groups. The inverse tile-to-grouped permutation
    // would split the source block, so dequantizing is required.
    if let Some(detail) = block_split_detail(axis.span, ggml_type) {
        return Err(shape_err(detail));
    }
    if row_bytes * axis.span % cols != 0 {
        return Err(shape_err(format!(
            "a {row_bytes}-byte row of {cols} columns has no whole-byte \
             {}-column V head: the block is wider than one head",
            axis.span
        )));
    }
    let head_bytes = row_bytes * axis.span / cols;
    let base_bytes = row_bytes * axis.base / cols;
    for r in 0..rows {
        restore_grouped_v_heads(
            &mut bytes[r * row_bytes..(r + 1) * row_bytes],
            base_bytes,
            head_bytes,
            layout,
        );
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;

    // ggml type ids: Q8_0 = 8, Q4_K = 12, Q6_K = 14, IQ4_XS = 23.
    const OUT_PROJ: &str = "model.layers.0.linear_attn.out_proj.weight";

    #[test]
    fn v_head_block_split_refuses_superblock_types_on_a_gdn_out_projection() {
        for family in [ModelFamily::QwenGdnDense, ModelFamily::QwenGdnMoe] {
            let mut arch = model_io::known_architecture(family);
            arch.linear_attention.value_head_dim = 128;
            for ggml_type in [12, 14, 23] {
                let reason =
                    gguf_v_head_block_split("blk.0.ssm_out.weight", OUT_PROJ, &arch, ggml_type)
                        .unwrap_or_else(|| panic!("{family:?} type {ggml_type} splits a block"));
                assert!(reason.contains("128-column V head"), "{reason}");
            }
            assert_eq!(
                gguf_v_head_block_split("blk.0.ssm_out.weight", OUT_PROJ, &arch, 8),
                None,
                "a 32-element Q8_0 block fits a 128-column head"
            );
        }
    }

    #[test]
    fn v_head_block_split_matches_the_walk_refusal() {
        // The probe's answer and the byte walk's must be the same answer.
        let mut arch = model_io::known_architecture(ModelFamily::QwenGdnDense);
        arch.linear_attention.num_k_heads = 1;
        arch.linear_attention.num_v_heads = 2;
        arch.linear_attention.value_head_dim = 128;
        // One Q4_K row of 256 columns: one 144-byte superblock.
        let mut bytes = vec![0u8; 144];
        let walk = apply_source_convention_bytes(
            "blk.0.ssm_out.weight",
            OUT_PROJ,
            &arch,
            (1, 256),
            12,
            &mut bytes,
        );
        let probe = gguf_v_head_block_split("blk.0.ssm_out.weight", OUT_PROJ, &arch, 12);
        match walk {
            Err(GgufRepackError::ShapeMismatch { detail, .. }) => {
                assert_eq!(Some(detail), probe)
            }
            other => panic!("the walk must refuse, got {:?}", other.is_ok()),
        }
    }

    #[test]
    fn v_head_block_split_skips_qwen4exp_dequantized_types_and_row_axis_tensors() {
        let mut arch = model_io::known_architecture(ModelFamily::Qwen4Exp);
        arch.linear_attention.value_head_dim = 128;
        assert_eq!(
            gguf_v_head_block_split("blk.0.ssm_out.weight", OUT_PROJ, &arch, 12),
            None,
            "Qwen4Exp dequantizes a Q4_K out_proj before permuting"
        );
        let dense = model_io::known_architecture(ModelFamily::QwenGdnDense);
        assert_eq!(
            gguf_v_head_block_split(
                "blk.0.attn_gate.weight",
                "model.layers.0.linear_attn.in_proj_z.weight",
                &dense,
                12
            ),
            None,
            "row-axis permutations move whole rows and never split a block"
        );
    }

    #[test]
    fn qwen4_out_projection_restores_three_v_heads_per_key_per_output_row() {
        let mut arch = model_io::known_architecture(ModelFamily::Qwen4Exp);
        arch.linear_attention.num_k_heads = 2;
        arch.linear_attention.key_head_dim = 1;
        arch.linear_attention.num_v_heads = 6;
        arch.linear_attention.value_head_dim = 2;
        let mut values = (0..24).map(|n| n as f32).collect::<Vec<_>>();

        apply_source_convention(
            "blk.0.ssm_out.weight",
            "language_model.model.layers.0.linear_attn.out_proj.weight",
            &arch,
            12,
            &mut values,
        )
        .expect("restore grouped V-head columns");

        assert_eq!(
            values,
            vec![
                0., 1., 4., 5., 8., 9., 2., 3., 6., 7., 10., 11., 12., 13., 16., 17., 20., 21.,
                14., 15., 18., 19., 22., 23.
            ]
        );
    }

    #[test]
    fn qwen_gdn_out_projection_preserves_the_two_to_one_v_head_order() {
        let mut arch = model_io::known_architecture(ModelFamily::QwenGdnMoe);
        arch.linear_attention.num_k_heads = 2;
        arch.linear_attention.key_head_dim = 1;
        arch.linear_attention.num_v_heads = 4;
        arch.linear_attention.value_head_dim = 2;
        let mut values = (0..16).map(|n| n as f32).collect::<Vec<_>>();

        apply_source_convention(
            "blk.0.ssm_out.weight",
            "language_model.model.layers.0.linear_attn.out_proj.weight",
            &arch,
            8,
            &mut values,
        )
        .expect("restore grouped V-head columns");

        assert_eq!(
            values,
            vec![0., 1., 4., 5., 2., 3., 6., 7., 8., 9., 12., 13., 10., 11., 14., 15.]
        );
    }

    #[test]
    fn qwen4exp_qkv_projection_reorders_only_v_rows_by_key_head_group() {
        let mut arch = model_io::known_architecture(ModelFamily::Qwen4Exp);
        arch.linear_attention.num_k_heads = 2;
        arch.linear_attention.key_head_dim = 1;
        arch.linear_attention.num_v_heads = 6;
        arch.linear_attention.value_head_dim = 2;
        let mut values = (0..16).map(|n| n as f32).collect::<Vec<_>>();

        apply_source_convention(
            "blk.0.ssm_qkv.weight",
            "language_model.model.layers.0.linear_attn.in_proj_qkv.weight",
            &arch,
            1,
            &mut values,
        )
        .expect("restore grouped V rows");

        assert_eq!(
            values,
            vec![0., 1., 2., 3., 4., 5., 8., 9., 12., 13., 6., 7., 10., 11., 14., 15.]
        );
    }

    #[test]
    fn qwen4exp_restores_grouped_a_log_and_inverts_the_ssm_a_encoding() {
        let mut arch = model_io::known_architecture(ModelFamily::Qwen4Exp);
        arch.linear_attention.num_k_heads = 2;
        arch.linear_attention.key_head_dim = 1;
        arch.linear_attention.num_v_heads = 6;
        arch.linear_attention.value_head_dim = 2;
        let mut values = [-1., -2., -3., -4., -5., -6.];

        apply_source_convention(
            "blk.0.ssm_a",
            "language_model.model.layers.0.linear_attn.A_log",
            &arch,
            1,
            &mut values,
        )
        .expect("recover grouped A_log");

        let expected = [
            1.0f32.ln(),
            3.0f32.ln(),
            5.0f32.ln(),
            2.0f32.ln(),
            4.0f32.ln(),
            6.0f32.ln(),
        ];
        for (actual, expected) in values.into_iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
        }
    }
    #[test]
    fn muse_per_layer_norms_lose_the_converters_plus_one_and_the_final_norm_keeps_its_value() {
        let arch = model_io::known_architecture(ModelFamily::MuseGlimmer);
        for tail in [
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ] {
            let mut values = [1.0f32, 1.5, 0.25, 4.5];
            apply_source_convention(
                "blk.3.some_norm.weight",
                &format!("language_model.model.layers.3.{tail}.weight"),
                &arch,
                1,
                &mut values,
            )
            .unwrap();
            assert_eq!(values, [0.0, 0.5, -0.75, 3.5], "{tail}");
        }
        // The final norm is unshifted in the file and must stay as it is.
        let mut values = [0.0169f32, 2.0];
        apply_source_convention(
            "output_norm.weight",
            "language_model.model.norm.weight",
            &arch,
            1,
            &mut values,
        )
        .unwrap();
        assert_eq!(values, [0.0169, 2.0]);
        // Another family's norm of the same name is not touched.
        let other = model_io::known_architecture(ModelFamily::Gemma4);
        let mut values = [1.0f32];
        apply_source_convention(
            "blk.0.attn_norm.weight",
            "language_model.model.layers.0.input_layernorm.weight",
            &other,
            1,
            &mut values,
        )
        .unwrap();
        assert_eq!(values, [1.0]);
    }

    #[test]
    fn muse_q_and_k_rows_take_the_rotary_unpermute_and_v_does_not() {
        let arch = model_io::known_architecture(ModelFamily::MuseGlimmer);
        let head_dim = arch.full_head_dim as usize;
        let rows = 2 * head_dim;
        // One byte per row, so a row's value is its source index.
        let source: Vec<u8> = (0..rows).map(|r| (r % 251) as u8).collect();
        for (canonical, permuted) in [
            (
                "language_model.model.layers.0.self_attn.q_proj.weight",
                true,
            ),
            (
                "language_model.model.layers.0.self_attn.k_proj.weight",
                true,
            ),
            (
                "language_model.model.layers.0.self_attn.v_proj.weight",
                false,
            ),
        ] {
            let mut bytes = source.clone();
            apply_source_convention_bytes("n", canonical, &arch, (rows, 1), 1, &mut bytes).unwrap();
            if !permuted {
                assert_eq!(bytes, source, "{canonical}");
                continue;
            }
            // Install row i reads GGUF row (i % half) * 2 + i / half, per head.
            let half = head_dim / 2;
            for head in 0..2 {
                for i in 0..head_dim {
                    let want = source[head * head_dim + (i % half) * 2 + i / half];
                    assert_eq!(bytes[head * head_dim + i], want, "{canonical} row {i}");
                }
            }
        }
    }

    /// The dense `qwen35` head's seven norms lose llama.cpp's `+1`; the
    /// trunk's identically-named norms keep it, because the runtime reads the
    /// trunk plain and the head centered.
    #[test]
    fn qwen35_dense_unshifts_only_the_mtp_head_norms() {
        let arch = model_io::known_architecture(ModelFamily::QwenGdnDense);
        for canonical in [
            "mtp.pre_fc_norm_embedding.weight",
            "mtp.pre_fc_norm_hidden.weight",
            "mtp.norm.weight",
            "mtp.layers.0.input_layernorm.weight",
            "mtp.layers.0.post_attention_layernorm.weight",
            "mtp.layers.0.self_attn.q_norm.weight",
            "mtp.layers.0.self_attn.k_norm.weight",
        ] {
            let mut values = [1.0f32, 1.5, 0.25];
            apply_source_convention("n", canonical, &arch, 1, &mut values).unwrap();
            assert_eq!(values, [0.0, 0.5, -0.75], "{canonical}");
        }
        for canonical in [
            "language_model.model.layers.3.input_layernorm.weight",
            "language_model.model.layers.3.self_attn.q_norm.weight",
            "language_model.model.norm.weight",
            "mtp.fc.weight",
        ] {
            let mut values = [1.0f32, 1.5, 0.25];
            apply_source_convention("n", canonical, &arch, 1, &mut values).unwrap();
            assert_eq!(values, [1.0, 1.5, 0.25], "{canonical}");
        }
    }

    #[test]
    fn gemma3_shifts_every_norm_including_qk_and_final_and_never_permutes_rows() {
        let arch = model_io::known_architecture(ModelFamily::Gemma3);
        for canonical in [
            "language_model.model.layers.3.input_layernorm.weight",
            "language_model.model.layers.3.post_attention_layernorm.weight",
            "language_model.model.layers.3.pre_feedforward_layernorm.weight",
            "language_model.model.layers.3.post_feedforward_layernorm.weight",
            "language_model.model.layers.3.self_attn.q_norm.weight",
            "language_model.model.layers.3.self_attn.k_norm.weight",
            // Unlike Muse Glimmer the final norm is shifted by the converter.
            "language_model.model.norm.weight",
        ] {
            let mut values = [1.0f32, 1.5, 0.25, 4.5];
            apply_source_convention("n", canonical, &arch, 1, &mut values).unwrap();
            assert_eq!(values, [0.0, 0.5, -0.75, 3.5], "{canonical}");
        }
        // A projection matrix is not a norm.
        let mut values = [1.0f32, 2.0];
        apply_source_convention(
            "n",
            "language_model.model.layers.0.mlp.gate_proj.weight",
            &arch,
            1,
            &mut values,
        )
        .unwrap();
        assert_eq!(values, [1.0, 2.0]);
        // llama.cpp applies no rotary permutation to Gemma 3 (read off the
        // real file against the HF q_proj/k_proj), so q/k rows stay put.
        let rows = 2 * arch.full_head_dim as usize;
        let source: Vec<u8> = (0..rows).map(|r| (r % 251) as u8).collect();
        for canonical in [
            "language_model.model.layers.0.self_attn.q_proj.weight",
            "language_model.model.layers.0.self_attn.k_proj.weight",
        ] {
            let mut bytes = source.clone();
            apply_source_convention_bytes("n", canonical, &arch, (rows, 1), 1, &mut bytes).unwrap();
            assert_eq!(bytes, source, "{canonical}");
        }
    }
}
