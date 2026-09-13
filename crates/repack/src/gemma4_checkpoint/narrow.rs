//! Unquantized BF16 narrowing and packed quantized resident tensor preparation.

use super::config::{is_supported_affine_shape, Gemma4Error, Gemma4Quant, AFFINE_GROUP_SIZE};
use super::shards::{le_u16, Gemma4Shards};
use crate::resident_writer::{ResidentEntrySpec, ResidentTensorSpec, DTYPE_BF16, DTYPE_FP16};

/// One unquantized tensor narrowed to BF16, which is the only unquantized
/// width this port can dispatch.
///
/// This replaced a `raw_dtype_tag` that mapped `BF16`/`F16`/`F32` onto the
/// three raw tags and was DELETED rather than left beside it: nothing in
/// `crates/runtime` reads tags 2 or 3, so its only remaining use would have
/// been to record a claim no reader honours.
pub struct NarrowedRaw {
    pub bytes: Vec<u8>,
    /// Always [`DTYPE_BF16`]; a field so a caller cannot forget to change the
    /// tag when it changes the bytes.
    pub dtype: u8,
    /// How many values did not survive the narrowing exactly. Zero for a BF16
    /// source, and NOT always zero otherwise -- see the header.
    pub lossy: usize,
}

/// Narrows an unquantized tensor to BF16, counting the values that lose bits.
///
/// **This is the safetensors sibling of the GGUF walk's `transcode_f32`, and
/// it is owed for the same reason with a DIFFERENT measurement behind it.**
/// The GGUF case narrows F32 that llama.cpp had upcast from BF16, so it is
/// exactly lossless and was measured to be. This one narrows F16, and F16 is
/// not a widened BF16: it carries 10 mantissa bits against BF16's 7, so a
/// value only survives if it happens to sit on the coarser grid.
///
/// Measured on the real `prism-ml/Bonsai-27B-mlx-1bit`, whose every
/// unquantized tensor is F16 (2026-08-14, ranged reads off the published
/// file, deterministic):
///
/// | tensor | values | lossy | worst relative |
/// |---|---|---|---|
/// | `input_layernorm.weight` | 5120 | 4364 | 0.003891 |
/// | `post_attention_layernorm.weight` | 5120 | 3738 | 0.003717 |
/// | `self_attn.q_norm.weight` | 256 | 216 | 0.003690 |
/// | `self_attn.k_norm.weight` | 256 | 222 | 0.003344 |
/// | `model.norm.weight` | 5120 | 2583 | 0.003891 |
/// | `linear_attn.{A_log,dt_bias,norm.weight,conv1d.weight}` | 41184 | **0** | 0.000000 |
///
/// So the loss is confined to the five RMS-norm families and is bounded by
/// BF16's own quantum, 2^-8; the gated-DeltaNet tensors are exactly
/// representable because this QAT checkpoint stores them on a coarse grid
/// (its layer 0 `A_log` has ONE distinct value across 48 elements and its
/// `conv1d` 538 across 40,960).
///
/// **The alternative was an FP16-weight variant of `rms_norm_bf16w` and its
/// `_perhead` sibling**, which is two kernels plus a dtype threaded through
/// every `norm_view` call site in four family flows. It is the fix if
/// ROADMAP's step 5 cross-engine KL lands above its backend floor, and the
/// first place to look if it does; it is not worth two kernels on the
/// strength of a 0.4% perturbation of a norm scale in a model whose weight
/// matrices are ONE BIT.
pub fn narrow_raw_to_bf16(
    tensor: &str,
    dtype: &str,
    bytes: Vec<u8>,
) -> Result<NarrowedRaw, Gemma4Error> {
    match dtype {
        "BF16" => Ok(NarrowedRaw {
            bytes,
            dtype: DTYPE_BF16,
            lossy: 0,
        }),
        "F16" | "F32" => {
            let values = decode_raw_to_f32(tensor, dtype, &bytes)?;
            let mut out = Vec::with_capacity(values.len() * 2);
            let mut lossy = 0usize;
            for value in values {
                let narrowed = compute::f32_to_bf16(value);
                if compute::bf16_to_f32(narrowed) != value {
                    lossy += 1;
                }
                out.extend_from_slice(&narrowed.to_le_bytes());
            }
            Ok(NarrowedRaw {
                bytes: out,
                dtype: DTYPE_BF16,
                lossy,
            })
        }
        other => Err(Gemma4Error::UnsupportedDtype {
            tensor: tensor.to_string(),
            dtype: other.to_string(),
        }),
    }
}

/// Decodes raw BF16/F16/F32 bytes into f32 values, in file order.
///
/// The shared parse step behind [`narrow_raw_to_bf16`]'s F16/F32 arm (which
/// re-narrows the result back to BF16) and [`quantize_gating_matrix_int8`]
/// (which quantizes it to INT8-affine instead). Unlike `narrow_raw_to_bf16`, this
/// also decodes a `BF16` source rather than passing it through: the router
/// quantizer needs f32 regardless of the source width.
fn decode_raw_to_f32(tensor: &str, dtype: &str, bytes: &[u8]) -> Result<Vec<f32>, Gemma4Error> {
    let width = match dtype {
        "BF16" | "F16" => 2,
        "F32" => 4,
        other => {
            return Err(Gemma4Error::UnsupportedDtype {
                tensor: tensor.to_string(),
                dtype: other.to_string(),
            })
        }
    };
    if bytes.len() % width != 0 {
        return Err(Gemma4Error::ShapeMismatch {
            tensor: tensor.to_string(),
            detail: format!(
                "{} bytes is not a whole number of {dtype} values",
                bytes.len()
            ),
        });
    }
    Ok(bytes
        .chunks_exact(width)
        .map(|chunk| match dtype {
            "BF16" => compute::bf16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]])),
            "F16" => compute::f16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]])),
            _ => f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]),
        })
        .collect())
}

/// Force-quantizes one of `qwen4_exp`'s small GATING matrices (the MoE
/// router `mlp.gate.weight`, or the shared expert's sigmoid gate
/// `mlp.shared_expert_gate.weight`) to INT8-affine at repack time, for a
/// checkpoint that ships it raw rather than pre-packed as `U32` -- mirroring
/// the GGUF walk's `transcode_f32` router target
/// (`gguf_checkpoint/transcode.rs`, `crates/repack` CLAUDE.md Gotcha 6),
/// whose `int8_transcode_targets` names the SAME two tensors for
/// `QwenGdnMoe`. Both are `[rows, hidden]` matrices with a small row count
/// (`num_experts` for the router, 1 for the shared-expert gate) that MLX's
/// default quantizer skips; `crates/runtime`'s `encode_gemv_any` has no
/// unquantized-BF16 GEMV kernel, so either one reaching the resident index
/// as raw BF16 fails at the first dispatch with "no dispatched GEMV kernel"
/// (`mlp.shared_expert_gate.weight`) or the family's own dtype-5 check
/// (`mlp.gate.weight`, `families/qwen4/moe.rs`). Every other MoE family's
/// versions of these tensors already reach INT8 via `pass_through_packed`,
/// because their upstream conversion pre-packs them; this is the one
/// safetensors checkpoint here whose upstream conversion left them unpacked.
pub fn quantize_gating_matrix_int8(
    shards: &Gemma4Shards<'_>,
    name: &str,
    dtype: &str,
    expected_shape: (usize, usize),
) -> Result<ResidentEntrySpec, Gemma4Error> {
    let w = shards.info(name)?;
    if w.shape.len() != 2 {
        return Err(Gemma4Error::ShapeMismatch {
            tensor: name.to_string(),
            detail: format!("expected rank-2 gating weight, got {:?}", w.shape),
        });
    }
    let rows = usize::try_from(w.shape[0]).map_err(|_| Gemma4Error::ShapeMismatch {
        tensor: name.to_string(),
        detail: format!("router row count {} does not fit usize", w.shape[0]),
    })?;
    let cols = usize::try_from(w.shape[1]).map_err(|_| Gemma4Error::ShapeMismatch {
        tensor: name.to_string(),
        detail: format!("router column count {} does not fit usize", w.shape[1]),
    })?;
    if (rows, cols) != expected_shape {
        return Err(Gemma4Error::ShapeMismatch {
            tensor: name.to_string(),
            detail: format!(
                "expected gating weight {}x{}, got {rows}x{cols}",
                expected_shape.0, expected_shape.1
            ),
        });
    }
    let group = AFFINE_GROUP_SIZE as usize;
    if cols % group != 0 {
        return Err(Gemma4Error::ShapeMismatch {
            tensor: name.to_string(),
            detail: format!("row length {cols} is not a multiple of the {group}-element group"),
        });
    }
    let value_count = rows
        .checked_mul(cols)
        .ok_or_else(|| Gemma4Error::ShapeMismatch {
            tensor: name.to_string(),
            detail: format!("gating weight element count overflows usize: {rows}x{cols}"),
        })?;
    let bytes = shards.read(name)?;
    let element_bytes = match dtype {
        "BF16" | "F16" => 2,
        "F32" => 4,
        other => {
            return Err(Gemma4Error::UnsupportedDtype {
                tensor: name.to_string(),
                dtype: other.to_string(),
            })
        }
    };
    let expected_bytes =
        value_count
            .checked_mul(element_bytes)
            .ok_or_else(|| Gemma4Error::ShapeMismatch {
                tensor: name.to_string(),
                detail: format!("gating weight byte count overflows usize: {value_count} values"),
            })?;
    if bytes.len() != expected_bytes {
        return Err(Gemma4Error::ShapeMismatch {
            tensor: name.to_string(),
            detail: format!(
                "{} bytes do not fill {rows}x{cols} {dtype} values",
                bytes.len()
            ),
        });
    }
    let values = decode_raw_to_f32(name, dtype, &bytes)?;
    let quantized: Vec<_> = (0..rows)
        .map(|r| compute::quantize_int8_affine(&values[r * cols..(r + 1) * cols]))
        .collect();
    Ok(ResidentEntrySpec::Int8(
        crate::repack::resident_spec_from_int8_rows(name, &quantized, cols),
    ))
}

/// One unquantized tensor converted to FP16, for the vision tower alone
/// (ROADMAP M-V3).
pub struct ConvertedFp16 {
    pub bytes: Vec<u8>,
    /// Always [`DTYPE_FP16`], a field for [`NarrowedRaw::dtype`]'s reason.
    pub dtype: u8,
    /// Values that landed in FP16's SUBNORMAL range, where the mantissa is
    /// truncated, plus those that flushed to zero. Everything in the normal
    /// range is exact -- see the header.
    pub lossy: usize,
}

/// Converts an unquantized tensor to FP16, refusing values FP16 cannot hold.
///
/// **THE TOWER IS THE ONE COMPONENT THIS PORT KEEPS AT FP16, AND THAT MAKES
/// THIS THE MIRROR OF `narrow_raw_to_bf16` RATHER THAN A SECOND COPY OF IT.**
/// Every text tensor is narrowed to BF16 because BF16 is the only unquantized
/// width the text kernels dispatch (AGENTS.md Gotcha 45). The vision kernels
/// landed in M-V2 bind `half` and are alone in `crates/gpu` in doing so, so
/// for these tensors the rule points the other way: sending them through the
/// BF16 narrowing would throw away three mantissa bits to store a precision
/// nothing then reads.
///
/// **BOTH PUBLISHED CHECKPOINTS ARE COVERED AND THEY DISAGREE ON THE SOURCE
/// DTYPE.** `prism-ml/Bonsai-27B-mlx-1bit` ships the tower F16, so that arm is
/// a verbatim copy. `mlx-community/Qwen3.8-27B-4bit` -- the artifact the real
/// M-V3 gate streams -- ships the SAME 333 tensors at the SAME shapes in
/// **BF16**, so that arm is a real conversion. A walk that only handled F16
/// would pass every fixture built from the 1-bit file and refuse the very
/// checkpoint it exists to ingest.
///
/// **BF16 to FP16 IS EXACT IN THE NORMAL RANGE, AND THE DIRECTION IS WHAT
/// MAKES IT SO.** BF16 carries 7 mantissa bits against FP16's 10, so the
/// mantissa WIDENS and cannot round. What can go wrong is the EXPONENT, in
/// both directions, and the two ends are treated differently on purpose:
///
/// - **Above FP16's 65,504 the value is REFUSED by name, never clamped.**
///   BF16 reaches 3.4e38, so a large weight becomes `inf`, and an `inf` weight
///   is the failure mode AGENTS.md Gotcha 59 is about -- it does not crash,
///   it propagates into activations as NaN, and NaN then reads as a PERFECT
///   score on every rank and top-k instrument downstream. A refusal at repack
///   costs one message; the alternative costs a session.
/// - **Below FP16's smallest normal (6.1e-5) the value degrades gracefully**
///   into FP16 subnormals and eventually to zero, and that is counted rather
///   than refused. A weight that small contributes nothing a norm can see, and
///   refusing one would reject a checkpoint over a value it does not use.
///
/// The count is reported through the streamed writer's `progress` callback for
/// `narrow_raw_to_bf16`'s reason: a lossy step in silence is how a quality
/// question becomes a mystery three phases later.
pub fn convert_raw_to_fp16(
    tensor: &str,
    dtype: &str,
    bytes: Vec<u8>,
) -> Result<ConvertedFp16, Gemma4Error> {
    let mismatch = |detail: String| Gemma4Error::ShapeMismatch {
        tensor: tensor.to_string(),
        detail,
    };
    match dtype {
        "F16" => {
            if bytes.len() % 2 != 0 {
                return Err(mismatch(format!(
                    "{} bytes is not a whole number of F16 values",
                    bytes.len()
                )));
            }
            Ok(ConvertedFp16 {
                bytes,
                dtype: DTYPE_FP16,
                lossy: 0,
            })
        }
        "BF16" | "F32" => {
            let width = if dtype == "BF16" { 2 } else { 4 };
            if bytes.len() % width != 0 {
                return Err(mismatch(format!(
                    "{} bytes is not a whole number of {dtype} values",
                    bytes.len()
                )));
            }
            let mut out = Vec::with_capacity(bytes.len() / width * 2);
            let mut lossy = 0usize;
            for chunk in bytes.chunks_exact(width) {
                let value = if width == 2 {
                    compute::bf16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]]))
                } else {
                    f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
                };
                let converted = compute::f32_to_f16(value);
                let back = compute::f16_to_f32(converted);
                // A source that was already non-finite is passed through: the
                // checkpoint says so, and inventing a refusal for it would
                // blame this conversion for the publisher's bytes. Only a
                // FINITE value that became non-finite is this step's doing.
                if value.is_finite() && !back.is_finite() {
                    return Err(Gemma4Error::ShapeMismatch {
                        tensor: tensor.to_string(),
                        detail: format!(
                            "value {value:e} exceeds FP16's 65504 maximum; the vision tower is \
                             held at FP16 end to end and an overflow here would reach a kernel \
                             as inf"
                        ),
                    });
                }
                if back != value {
                    lossy += 1;
                }
                out.extend_from_slice(&converted.to_le_bytes());
            }
            Ok(ConvertedFp16 {
                bytes: out,
                dtype: DTYPE_FP16,
                lossy,
            })
        }
        other => Err(Gemma4Error::UnsupportedDtype {
            tensor: tensor.to_string(),
            dtype: format!("{other} in a vision tower tensor (expected F16 or BF16)"),
        }),
    }
}

/// Packs quantized tensor weights and companion scale/bias arrays into resident entry spec.
pub fn pass_through_packed(
    shards: &Gemma4Shards<'_>,
    name: &str,
    quant: &Gemma4Quant,
) -> Result<ResidentEntrySpec, Gemma4Error> {
    pass_through_packed_impl(shards, name, quant, false).map(|(spec, _)| spec)
}

/// Reads a Qwen2/Qwen2.5 packed tensor whose 4/8-bit companions are F16 in
/// the MLX source checkpoint. The resident affine kernels consume BF16
/// companions, so the planes are narrowed while they are ingested rather
/// than copied as F16 bits under a BF16 manifest tag.
pub fn pass_through_packed_qwen2(
    shards: &Gemma4Shards<'_>,
    name: &str,
    quant: &Gemma4Quant,
) -> Result<ResidentEntrySpec, Gemma4Error> {
    pass_through_packed_impl(shards, name, quant, true).map(|(spec, _)| spec)
}

/// Qwen2's streamed writer needs the narrowing count for its conversion
/// report. The public wrapper above keeps the existing pass-through API
/// focused on the entry itself.
pub(crate) fn pass_through_packed_qwen2_with_loss(
    shards: &Gemma4Shards<'_>,
    name: &str,
    quant: &Gemma4Quant,
) -> Result<(ResidentEntrySpec, Vec<(String, usize)>), Gemma4Error> {
    pass_through_packed_impl(shards, name, quant, true)
}

fn pass_through_packed_impl(
    shards: &Gemma4Shards<'_>,
    name: &str,
    quant: &Gemma4Quant,
    qwen2_f16_companions: bool,
) -> Result<(ResidentEntrySpec, Vec<(String, usize)>), Gemma4Error> {
    let w = shards.info(name)?;
    if w.shape.len() != 2 {
        return Err(Gemma4Error::ShapeMismatch {
            tensor: name.to_string(),
            detail: format!("expected rank-2 packed weight, got {:?}", w.shape),
        });
    }
    let base = name.strip_suffix(".weight").unwrap_or(name);
    let bits = quant.bits_for(base);
    let group_size = quant.group_size;
    if !is_supported_affine_shape(bits, group_size) {
        return Err(Gemma4Error::UnsupportedDtype {
            tensor: name.to_string(),
            dtype: format!("{bits}-bit quantization at group {group_size}"),
        });
    }
    // Elements per packed u32 word. The formula covers one and two bits as
    // well as four and eight; what does NOT generalize is everything below it.
    let factor = 32 / bits as u64;
    // **The source companion dtype is a function of the bit width, and this is
    // the one axis in the whole walk that fails silently if it is wrong.** MLX
    // writes companions in the checkpoint's own dtype: BF16 for the INT4/INT8
    // installs this port already reads, FP16 for the 1-bit and 2-bit ones, and
    // FP16 for Qwen2's 4/8-bit source planes. The two are the same width and
    // share no exponent field, so accepting either without conversion would
    // produce an install of exactly the right SIZE whose scales are wrong by
    // orders of magnitude -- 0.0271 read as 1.7e-16. Hence a required dtype
    // per width rather than a set of allowed ones, except for Qwen2's explicit
    // F16-to-BF16 conversion below.
    //
    // Both sub-4-bit checkpoints happen to be F16 and both non-Qwen2 4/8-bit
    // ones BF16, so this reads as a threshold and is not one: it is a table of
    // what each published file carries, and a future 2-bit checkpoint in BF16
    // would be a third row rather than a moved boundary.
    let companion_dtype = match bits {
        1 | 2 => "F16",
        _ => "BF16",
    };
    let qwen2_wide_f16 = qwen2_f16_companions && matches!(bits, 4 | 8);
    let scales_name = format!("{base}.scales");
    let biases_name = format!("{base}.biases");
    for companion in [&scales_name, &biases_name] {
        if !shards.contains(companion) {
            return Err(Gemma4Error::MissingCompanion(name.to_string()));
        }
        let c = shards.info(companion)?;
        let accepted = if qwen2_wide_f16 {
            matches!(c.dtype.as_str(), "F16" | "BF16")
        } else {
            c.dtype == companion_dtype
        };
        if !accepted {
            return Err(Gemma4Error::UnsupportedDtype {
                tensor: companion.to_string(),
                dtype: if qwen2_wide_f16 {
                    format!(
                        "{} companions on a {bits}-bit Qwen2 tensor (expected F16 or BF16)",
                        c.dtype
                    )
                } else {
                    format!(
                        "{} companions on a {bits}-bit tensor (expected {companion_dtype})",
                        c.dtype
                    )
                },
            });
        }
    }
    let rows = w.shape[0];
    let cols = w.shape[1] * factor;
    let packed = shards.read(name)?;
    // Scales and biases are checked against `rows*cols/group` below; the
    // packed run itself was not, so a shard whose `data_offsets` disagree
    // with its declared `shape` (a corrupt but non-hostile checkpoint) would
    // otherwise write an index entry whose shape and byte size disagree.
    let expected_packed_bytes = (rows * w.shape[1] * 4) as usize;
    if packed.len() != expected_packed_bytes {
        return Err(Gemma4Error::ShapeMismatch {
            tensor: name.to_string(),
            detail: format!(
                "packed run is {} bytes, expected {expected_packed_bytes} for a {rows}x{} \
                 u32 packing of shape {rows}x{cols}",
                packed.len(),
                w.shape[1]
            ),
        });
    }
    let (scales, scales_lossy) = read_companion_plane(shards, &scales_name, qwen2_wide_f16)?;
    let (biases, biases_lossy) = read_companion_plane(shards, &biases_name, qwen2_wide_f16)?;
    let group = group_size as u64;
    let expected_groups = (rows * cols / group) as usize;
    if cols % group != 0 || scales.len() != expected_groups || biases.len() != expected_groups {
        return Err(Gemma4Error::ShapeMismatch {
            tensor: name.to_string(),
            detail: format!(
                "shape {rows}x{cols} with {} scales / {} biases does not match \
                 the {group}-element groups this checkpoint declares",
                scales.len(),
                biases.len()
            ),
        });
    }
    // Writer invariant: `ResidentTensorSpec` stores shape as `u32`. A silent
    // truncation here would write a plausible, wrong shape with no error at
    // any later read, so this is asserted rather than cast blindly.
    assert!(
        rows <= u64::from(u32::MAX) && cols <= u64::from(u32::MAX),
        "{name}: shape {rows}x{cols} exceeds the 32-bit resident index shape fields"
    );
    let spec = ResidentTensorSpec {
        name: name.to_string(),
        packed,
        scales,
        biases,
        rows: rows as u32,
        cols: cols as u32,
    };
    let spec = match bits {
        1 => ResidentEntrySpec::Int1(spec),
        2 => ResidentEntrySpec::Int2(spec),
        4 => ResidentEntrySpec::Int4(spec),
        _ => ResidentEntrySpec::Int8(spec),
    };
    let losses = if scales_lossy == 0 && biases_lossy == 0 {
        Vec::new()
    } else {
        vec![(scales_name, scales_lossy), (biases_name, biases_lossy)]
            .into_iter()
            .filter(|(_, count)| *count > 0)
            .collect()
    };
    Ok((spec, losses))
}

fn read_companion_plane(
    shards: &Gemma4Shards<'_>,
    name: &str,
    narrow_f16: bool,
) -> Result<(Vec<u16>, usize), Gemma4Error> {
    let dtype = shards.info(name)?.dtype.as_str();
    let bytes = shards.read(name)?;
    if narrow_f16 && dtype == "F16" {
        let values = decode_raw_to_f32(name, "F16", &bytes)?;
        let mut lossy = 0usize;
        let bits = values
            .into_iter()
            .map(|value| {
                let narrowed = compute::f32_to_bf16(value);
                if compute::bf16_to_f32(narrowed) != value {
                    lossy += 1;
                }
                narrowed
            })
            .collect();
        Ok((bits, lossy))
    } else {
        Ok((le_u16(&bytes), 0))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::quantize_gating_matrix_int8;
    use crate::gemma4_checkpoint::{Gemma4Error, Gemma4Shards};
    use crate::ranged_download::MemoryRangeSource;
    use crate::safetensors_header::{SafetensorsHeader, TensorInfo};

    const NAME: &str = "language_model.model.layers.0.mlp.gate.weight";

    fn shards_with_shape(shape: Vec<u64>) -> (SafetensorsHeader, MemoryRangeSource<'static>) {
        let header = SafetensorsHeader {
            tensors: BTreeMap::from([(
                NAME.to_string(),
                TensorInfo {
                    dtype: "BF16".to_string(),
                    shape,
                    data_offsets: (0, 0),
                },
            )]),
            metadata: None,
            header_len: 0,
        };
        static EMPTY_FILE: [u8; 8] = [0; 8];
        (header, MemoryRangeSource::new(&EMPTY_FILE))
    }

    #[test]
    fn malicious_gating_dimensions_are_refused_without_panicking() {
        let (header, source) = shards_with_shape(vec![1 << 63, 64]);
        let shards = Gemma4Shards::single(&header, &source);
        let hostile_rows = usize::try_from(1_u64 << 63).expect("test requires a 64-bit host");
        let err = quantize_gating_matrix_int8(&shards, NAME, "BF16", (hostile_rows, 64))
            .expect_err("overflowing shape must be refused");
        assert!(matches!(err, Gemma4Error::ShapeMismatch { .. }));
    }

    #[test]
    fn gating_dimensions_must_match_the_architecture() {
        let (header, source) = shards_with_shape(vec![3, 64]);
        let shards = Gemma4Shards::single(&header, &source);
        let err = quantize_gating_matrix_int8(&shards, NAME, "BF16", (2, 64))
            .expect_err("checkpoint shape must match architecture");
        let Gemma4Error::ShapeMismatch { detail, .. } = err else {
            panic!("expected shape mismatch");
        };
        assert!(detail.contains("expected gating weight 2x64"), "{detail}");
    }
}
