//! Unquantized BF16 narrowing and packed quantized resident tensor preparation.

use super::config::{is_supported_affine_shape, Gemma4Error, Gemma4Quant};
use super::shards::{le_u16, Gemma4Shards};
use crate::resident_writer::{ResidentEntrySpec, ResidentTensorSpec, DTYPE_BF16};

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
    let mismatch = |detail: String| Gemma4Error::ShapeMismatch {
        tensor: tensor.to_string(),
        detail,
    };
    match dtype {
        "BF16" => Ok(NarrowedRaw {
            bytes,
            dtype: DTYPE_BF16,
            lossy: 0,
        }),
        "F16" | "F32" => {
            let width = if dtype == "F16" { 2 } else { 4 };
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
                    compute::f16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]]))
                } else {
                    f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
                };
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

/// Packs quantized tensor weights and companion scale/bias arrays into resident entry spec.
pub fn pass_through_packed(
    shards: &Gemma4Shards<'_>,
    name: &str,
    quant: &Gemma4Quant,
) -> Result<ResidentEntrySpec, Gemma4Error> {
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
    // **The companion dtype is a function of the bit width, and this is the
    // one axis in the whole walk that fails silently if it is wrong.** MLX
    // writes companions in the checkpoint's own dtype: BF16 for the INT4/INT8
    // installs this port already reads, FP16 for the 1-bit and 2-bit ones. The
    // two are the same width and share no exponent field, so accepting either
    // here would produce an install of exactly the right SIZE whose scales are
    // wrong by orders of magnitude -- 0.0271 read as 1.7e-16. Hence a
    // required dtype per width rather than a set of allowed ones.
    //
    // Both sub-4-bit checkpoints happen to be F16 and both 4/8-bit ones BF16,
    // so this reads as a threshold and is not one: it is a table of what each
    // published file carries, and a future 2-bit checkpoint in BF16 would be a
    // third row rather than a moved boundary.
    let companion_dtype = match bits {
        1 | 2 => "F16",
        _ => "BF16",
    };
    let scales_name = format!("{base}.scales");
    let biases_name = format!("{base}.biases");
    for companion in [&scales_name, &biases_name] {
        if !shards.contains(companion) {
            return Err(Gemma4Error::MissingCompanion(name.to_string()));
        }
        let c = shards.info(companion)?;
        if c.dtype != companion_dtype {
            return Err(Gemma4Error::UnsupportedDtype {
                tensor: companion.to_string(),
                dtype: format!(
                    "{} companions on a {bits}-bit tensor (expected {companion_dtype})",
                    c.dtype
                ),
            });
        }
    }
    let rows = w.shape[0];
    let cols = w.shape[1] * factor;
    let packed = shards.read(name)?;
    let scales = le_u16(&shards.read(&scales_name)?);
    let biases = le_u16(&shards.read(&biases_name)?);
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
    let spec = ResidentTensorSpec {
        name: name.to_string(),
        packed,
        scales,
        biases,
        rows: rows as u32,
        cols: cols as u32,
    };
    Ok(match bits {
        1 => ResidentEntrySpec::Int1(spec),
        2 => ResidentEntrySpec::Int2(spec),
        4 => ResidentEntrySpec::Int4(spec),
        _ => ResidentEntrySpec::Int8(spec),
    })
}
