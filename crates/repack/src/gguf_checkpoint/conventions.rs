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
        num_k_heads,
        num_v_heads,
        values_per_k_head: num_v_heads / num_k_heads,
    })
}

/// Restore grouped `[key_head, value_head_within_key, span]` order from the
/// converter's tiled `[value_head_within_key, key_head, span]` order.
fn restore_grouped_v_heads<T: Copy>(data: &mut [T], base: usize, span: usize, layout: VHeadLayout) {
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

/// True for the two tensors that need it, on the one family that does.
fn needs_rotary_unpermute(canonical: &str, arch: &ArchConfig) -> bool {
    arch.family == ModelFamily::Llama
        && (canonical.ends_with("self_attn.q_proj.weight")
            || canonical.ends_with("self_attn.k_proj.weight"))
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
    let block_elements = crate::gguf_header::ggml_type_block(ggml_type)
        .map(|(elements, _)| elements as usize)
        .unwrap_or(1);
    if block_elements > 1 && axis.span % block_elements != 0 {
        return Err(shape_err(format!(
            "a {}-column V head is not a whole number of {block_elements}-element \
             blocks, so the de-interleave would split one: dequantizing is the only \
             way to permute this tensor, and this walk copies bytes",
            axis.span
        )));
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
}
