//! GGUF source tensor layout conventions (V-head de-interleaving and RoPE rotary unpermutations).

use model_io::{ArchConfig, ModelFamily};

use super::types::GgufRepackError;

struct VHeadAxis {
    columns: bool,
    base: usize,
    span: usize,
}

fn v_head_axis(
    name: &str,
    canonical: &str,
    arch: &ArchConfig,
) -> Result<Option<VHeadAxis>, GgufRepackError> {
    // BOTH Qwen halves. The convention belongs to llama.cpp's CONVERTER and
    // to the gated-DeltaNet block, and the two halves share both -- the dense
    // one is the same `linear_attn.*` inventory with a different FFN below it.
    //
    // **THE DENSE HALF WAS EXCLUDED AND IT FAILED EXACTLY AS Gotcha 33 SAYS
    // THIS CLASS FAILS**: the first `qwen35` install loaded, decoded, never
    // errored, and produced word salad. Nothing upstream can see it -- every
    // name mapped, every shape checked out, the manifest validated. Only
    // running the model does, which is why the convention check is a Phase 1
    // gate rather than a tidy-up.
    if !matches!(
        arch.family,
        ModelFamily::QwenGdnMoe | ModelFamily::QwenGdnDense
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

fn permute_v_heads<T: Copy>(data: &mut [T], base: usize, span: usize, heads: usize) {
    let source = data.to_owned();
    for h in 0..heads {
        let to = if h < heads / 2 {
            2 * h
        } else {
            2 * (h - heads / 2) + 1
        };
        data[base + to * span..base + (to + 1) * span]
            .copy_from_slice(&source[base + h * span..base + (h + 1) * span]);
    }
}

fn even_v_heads(name: &str, arch: &ArchConfig) -> Result<usize, GgufRepackError> {
    let heads = usize::try_from(arch.linear_attention.num_v_heads).map_err(|_| {
        GgufRepackError::ShapeMismatch {
            tensor: name.to_string(),
            detail: format!(
                "the V-head de-interleave needs a representable num_v_heads, got {}",
                arch.linear_attention.num_v_heads
            ),
        }
    })?;
    if heads < 2 || heads % 2 != 0 {
        return Err(GgufRepackError::ShapeMismatch {
            tensor: name.to_string(),
            detail: format!("the V-head de-interleave needs an even num_v_heads, got {heads}"),
        });
    }
    Ok(heads)
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
    let heads = even_v_heads(name, arch)?;
    let shape_err = |detail: String| GgufRepackError::ShapeMismatch {
        tensor: name.to_string(),
        detail,
    };
    if axis.columns {
        return Err(shape_err(
            "a column-axis V-head tensor is not expected to arrive as F32".to_string(),
        ));
    }
    let base = axis.base.checked_mul(cols);
    let span = axis.span.checked_mul(cols);
    let end = span
        .and_then(|span| heads.checked_mul(span))
        .and_then(|body| base.and_then(|base| base.checked_add(body)));
    let (Some(base), Some(span), Some(end)) = (base, span, end) else {
        return Err(shape_err("V-head shape arithmetic overflowed".to_string()));
    };
    if span == 0 || end != values.len() {
        return Err(shape_err(format!(
            "{} values do not fill {base} + {heads} x {span}",
            values.len()
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
    permute_v_heads(values, base, span, heads);
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
    let heads = even_v_heads(name, arch)?;
    let shape_err = |detail: String| GgufRepackError::ShapeMismatch {
        tensor: name.to_string(),
        detail,
    };
    let along = if axis.columns { cols } else { rows };
    let end = heads
        .checked_mul(axis.span)
        .and_then(|body| axis.base.checked_add(body));
    if axis.span == 0 || end != Some(along) {
        return Err(shape_err(format!(
            "a {rows}x{cols} tensor's V axis does not fill {} + {heads} x {}",
            axis.base, axis.span
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
        permute_v_heads(bytes, axis.base * row_bytes, axis.span * row_bytes, heads);
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
    // REFUSED rather than shuffled in halves, which is what Gotcha 7 already
    // says this case must do. The de-interleave maps output heads `2k` and
    // `2k+1` from source heads `k` and `k + H/2`, so one output superblock
    // draws on two DIFFERENT source superblocks -- there is no byte-level
    // rearrangement that fixes it, only dequantizing and requantizing.
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
        permute_v_heads(
            &mut bytes[r * row_bytes..(r + 1) * row_bytes],
            base_bytes,
            head_bytes,
            heads,
        );
    }
    Ok(())
}
