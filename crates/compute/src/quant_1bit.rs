//! 1-bit affine groupwise quantization reference (MLX `affine` layout at
//! `bits = 1`).
//!
//! A separate module from `quant.rs` for the reason `quant_gguf.rs` is
//! separate from both: the layout is not a narrower `Int4AffineRow`. Three
//! things differ at once, and each one is a silent wrong answer rather than
//! a failure if it is carried over:
//!
//! - **The group size is not 64.** `quant.rs` pins [`crate::quant::GROUP_SIZE`]
//!   at 64 and every affine GEMV in `crates/gpu` specializes on it. The real
//!   1-bit checkpoint groups by 128, so the group size is a PARAMETER here
//!   rather than a constant. Getting it wrong reads every scale off by a
//!   factor of the group ratio and still produces finite, ordered output.
//! - **The companions are FP16, not BF16.** `quant.rs` carries scale and bias
//!   as raw `u16` BF16 bit patterns, which is what the affine GEMV kernels
//!   read as `device const bfloat*`. MLX writes this checkpoint's `scales`
//!   and `biases` as safetensors `F16`. The two are both 16 bits wide and
//!   share no exponent field, so reading one as the other is undetectable
//!   by any length check and wrong by orders of magnitude.
//! - **The element per byte count is 8, not 2.** Bit `k` of byte `j` is
//!   element `8 * j + k`.
//!
//! **THE CONVENTION BELOW WAS MEASURED, NOT READ OFF A FORMAT DOC.**
//! `crates/compute/tests/quant_1bit.rs` carries the oracle: 32 packed bytes
//! and their two FP16 companion pairs, lifted by ranged read out of the real
//! `prism-ml/Bonsai-27B-mlx-1bit` `model.safetensors`, against the 256
//! floats MLX's own `mx.dequantize` produces from them. The LSB-first order
//! is exact on all 245,760 elements of the probed tensor and the MSB-first
//! alternative matches none of them.
//!
//! # Layout
//!
//! One row of `n` weights, `n` a multiple of the group size:
//!
//! - `packed`: `n / 8` bytes. Element `i` is bit `i % 8` of byte `i / 8`,
//!   least significant bit first. MLX stores the same run as `n / 32`
//!   little-endian `u32` words; the byte view is identical and carries no
//!   endianness assumption, which is why it is the one used here.
//! - `scales` / `biases`: one FP16 bit pattern each per group.
//!
//! An element decodes as `w = q * scale + bias` with `q` in `{0, 1}`, so a
//! group's two representable values are `bias` and `bias + scale`.
//!
//! # The symmetry is a property of the checkpoint, not of the format
//!
//! Every group of the probed tensor has `bias == -scale / 2` exactly, so its
//! two values are `-scale/2` and `+scale/2`: symmetric binary weights
//! carried in an affine container. That is what a QAT binary checkpoint
//! produces and it is what makes a `+/-1` accumulate-then-scale GEMV
//! possible.
//!
//! It is deliberately NOT assumed anywhere in this module. The container
//! permits any `(scale, bias)` pair, nothing in the file declares the
//! symmetry, and a kernel that bakes it in decodes a non-symmetric group to
//! plausible wrong weights. [`is_symmetric`] MEASURES it instead, which is
//! what a repack walk should call before selecting a symmetric fast path.

use foundation::LogitValue as F16;

/// The group size the real 1-bit checkpoint declares
/// (`config.json`: `quantization.group_size`).
///
/// Not a constant of the format. Pass the checkpoint's own value; this
/// exists so a caller writing a fixture does not invent one.
pub const BONSAI_GROUP_SIZE: usize = 128;

/// Widen an FP16 bit pattern to `f32`.
///
/// Routed through `foundation::LogitValue` rather than a local bit shift:
/// binary16 has a 5-bit exponent and an implicit-leading-bit mantissa, so
/// unlike BF16 it is not a truncation of an FP32 word (AGENTS.md Gotcha 3).
#[inline]
pub fn f16_to_f32(bits: u16) -> f32 {
    F16::from_bits(bits).to_f32()
}

/// Narrow an `f32` to an FP16 bit pattern.
#[inline]
pub fn f32_to_f16(x: f32) -> u16 {
    F16::from_f32(x).to_bits()
}

/// One row of 1-bit-affine packed weights: `n / 8` bytes, plus one FP16
/// scale and one FP16 bias per group.
///
/// `group_size` travels with the row because it is a property of the
/// checkpoint (see the module header), and a row separated from it cannot
/// be decoded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Int1AffineRow {
    pub packed: Vec<u8>,
    pub scales: Vec<u16>,
    pub biases: Vec<u16>,
    pub group_size: usize,
}

impl Int1AffineRow {
    /// Number of weights this row decodes to.
    pub fn len(&self) -> usize {
        self.packed.len() * 8
    }

    /// True when the row carries no weights.
    pub fn is_empty(&self) -> bool {
        self.packed.is_empty()
    }

    /// The unsigned 1-bit quant at element `i`, least significant bit first.
    #[inline]
    pub fn quant(&self, i: usize) -> u8 {
        (self.packed[i / 8] >> (i % 8)) & 1
    }
}

/// Checks the `bias == -scale / 2` symmetry group by group, on the decoded
/// FP16 values rather than on the bit patterns.
///
/// Returns the number of groups that are NOT symmetric, so `0` means the row
/// is symmetric binary and a caller may take a `+/-1` fast path on it. The
/// comparison is exact: with `bias` and `scale` both FP16 and the ratio a
/// power of two, halving is exact whenever the result is normal, so a
/// tolerance here would hide a genuinely asymmetric group instead of
/// absorbing rounding.
pub fn asymmetric_group_count(row: &Int1AffineRow) -> usize {
    row.scales
        .iter()
        .zip(row.biases.iter())
        .filter(|(&s, &b)| f16_to_f32(b) != -f16_to_f32(s) / 2.0)
        .count()
}

/// True when every group of the row is symmetric binary (`+/- scale / 2`).
pub fn is_symmetric(row: &Int1AffineRow) -> bool {
    asymmetric_group_count(row) == 0
}

fn check_shape(n: usize, group_size: usize) {
    assert!(group_size > 0, "group size must be positive");
    assert!(
        group_size % 8 == 0,
        "group size {group_size} is not a whole number of bytes"
    );
    assert!(
        n % group_size == 0,
        "row length {n} is not a multiple of the group size {group_size}"
    );
}

fn check_row_shape(row: &Int1AffineRow, n: usize) {
    assert_eq!(n, row.len(), "n must equal packed.len() * 8");
    check_shape(n, row.group_size);
    let n_groups = n / row.group_size;
    assert_eq!(
        row.scales.len(),
        n_groups,
        "scales.len() must equal n_groups"
    );
    assert_eq!(
        row.biases.len(),
        n_groups,
        "biases.len() must equal n_groups"
    );
}

/// Symmetric 1-bit affine quantize: `q in {0, 1}`, `w ~= q * scale + bias`
/// with `bias = -scale / 2`, so the two representable values are
/// `+/- scale / 2`.
///
/// **This is not the min/max affine rule the INT4 and INT8 quantizers use,
/// and that is deliberate.** At four bits, `scale = (max - min) / 15` with
/// `bias = min` spends its two endpoints on the group's extremes and its
/// interior levels on everything else. At ONE bit there is no interior: the
/// same rule reduces to "round each weight to whichever of the group's two
/// extremes is nearer", which quantizes a symmetric distribution to a pair
/// of outliers and drives the reconstruction error far above the sign rule.
/// It also would not reproduce the convention the real QAT checkpoint uses,
/// so a fixture built with it would exercise a layout no file has.
///
/// The scale is `2 * mean(|w|)`, which is the least-squares magnitude for a
/// sign-quantized group, and is rounded through FP16 before the quantize
/// pass so decode reproduces the same `q` the row stores. A group of all
/// zeros gets a zero scale and decodes back to zeros.
pub fn quantize_int1_affine_symmetric(row: &[f32], group_size: usize) -> Int1AffineRow {
    check_shape(row.len(), group_size);
    let n_groups = row.len() / group_size;
    let mut packed = vec![0u8; row.len() / 8];
    let mut scales = vec![0u16; n_groups];
    let mut biases = vec![0u16; n_groups];

    for g in 0..n_groups {
        let group = &row[g * group_size..(g + 1) * group_size];
        let mean_abs = group.iter().map(|w| w.abs()).sum::<f32>() / group_size as f32;
        let s_bits = f32_to_f16(2.0 * mean_abs);
        let scale = f16_to_f32(s_bits);
        // Stored rather than recomputed on decode: the file carries an
        // independent bias plane and this module refuses to assume the two
        // are related (see the module header).
        let b_bits = f32_to_f16(-scale / 2.0);
        scales[g] = s_bits;
        biases[g] = b_bits;

        for (k, &w) in group.iter().enumerate() {
            // Sign rule. The midpoint of the two representable values is
            // `bias + scale/2`, which for a symmetric group is 0, so
            // "nearer representable value" is exactly "non-negative".
            if w >= 0.0 {
                let i = g * group_size + k;
                packed[i / 8] |= 1 << (i % 8);
            }
        }
    }

    Int1AffineRow {
        packed,
        scales,
        biases,
        group_size,
    }
}

/// Dequantize an [`Int1AffineRow`] to `n` FP32 elements.
pub fn dequantize_int1_affine(r: &Int1AffineRow, n: usize) -> Vec<f32> {
    check_row_shape(r, n);
    let mut out = vec![0f32; n];
    for g in 0..n / r.group_size {
        let scale = f16_to_f32(r.scales[g]);
        let bias = f16_to_f32(r.biases[g]);
        for k in 0..r.group_size {
            let i = g * r.group_size + k;
            out[i] = r.quant(i) as f32 * scale + bias;
        }
    }
    out
}

/// FP32 reference for the 1-bit-affine embedding lookup: one row of a
/// `[V, D]` table, dequantized and scaled by `out_scale` (Gemma's
/// `sqrt(hidden)` post-embedding scale, or `1.0` for a raw dequant).
///
/// The sibling of [`crate::quant::embed_lookup_int4`], and it exists because
/// the real checkpoint QUANTIZES ITS EMBEDDING TABLE AT ONE BIT. That was
/// read off the safetensors header rather than assumed: 498 tensors carry
/// `.scales`, and `language_model.model.embed_tokens` and
/// `language_model.lm_head` are two of them, both `U32 [248320, 160]` (160
/// packed words is 5120 elements, the hidden size). The head goes through
/// the resident GEMV like any other matrix; the table needs this, so 1-bit
/// is a type with a GEMV and a lookup and no routed-expert pair -- the
/// per-type footing AGENTS.md Gotcha 29 describes, decided by what the one
/// real file puts where.
///
/// `group_size` is a parameter for the module header's reason. Note the
/// bound checks are on the SCALE plane as well as the packed one: at one bit
/// a row is eight times shorter than at eight, so a `token_id` that walks
/// off the end of one plane can still be inside the other.
pub fn embed_lookup_int1(
    table_packed: &[u8],
    table_scales: &[u16],
    table_biases: &[u16],
    token_id: usize,
    d: usize,
    group_size: usize,
    out_scale: f32,
) -> Vec<f32> {
    check_shape(d, group_size);
    let groups_per_row = d / group_size;
    let row_bytes = d / 8;
    let pack_base = token_id * row_bytes;
    let scale_base = token_id * groups_per_row;
    assert!(
        pack_base + row_bytes <= table_packed.len(),
        "token out of range"
    );
    assert!(
        scale_base + groups_per_row <= table_scales.len(),
        "scales out of range"
    );
    assert!(
        scale_base + groups_per_row <= table_biases.len(),
        "biases out of range"
    );

    let mut out = vec![0f32; d];
    for (i, out_val) in out.iter_mut().enumerate() {
        let byte = table_packed[pack_base + (i / 8)];
        let q = (byte >> (i % 8)) & 1;
        let g = i / group_size;
        let scale = f16_to_f32(table_scales[scale_base + g]);
        let bias = f16_to_f32(table_biases[scale_base + g]);
        *out_val = (q as f32 * scale + bias) * out_scale;
    }
    out
}

/// Dequantize-and-multiply reference GEMV: `out[row] = sum_i w[row][i] * x[i]`.
///
/// Accumulates in FP32 in element order, which is the order a per-row GPU
/// kernel's lane partials have to be reduced in to match. Mirrors
/// [`crate::quant::dequant_int4_gemv`].
pub fn dequant_int1_gemv(weight_rows: &[Int1AffineRow], x: &[f32], n: usize) -> Vec<f32> {
    assert!(!weight_rows.is_empty(), "weight_rows must not be empty");
    assert_eq!(x.len(), n, "x.len() must equal n");
    let mut out = vec![0f32; weight_rows.len()];
    for (r, row) in weight_rows.iter().enumerate() {
        let w = dequantize_int1_affine(row, n);
        out[r] = w.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
    }
    out
}

/// The `+/-1` form of [`dequant_int1_gemv`], for a row that
/// [`is_symmetric`] accepts.
///
/// Exists to state what a symmetric-fast-path GEMV kernel computes, and to
/// give it something exact to be checked against: a group contributes
/// `(scale / 2) * sum(+/-x)`, so the weights never materialize. It PANICS on
/// an asymmetric row rather than silently returning the affine answer,
/// because the interesting failure is a caller taking this path on a row
/// that does not qualify.
///
/// It is not bit-identical to [`dequant_int1_gemv`] and is not meant to be:
/// factoring the scale out of the group changes the summation order, so the
/// two agree to FP32 rounding rather than exactly. That is the same tradeoff
/// AGENTS.md Gotcha 27 is about, and it is the reason this is a separate
/// function instead of an optimization inside the one above.
pub fn dequant_int1_gemv_symmetric(weight_rows: &[Int1AffineRow], x: &[f32], n: usize) -> Vec<f32> {
    assert!(!weight_rows.is_empty(), "weight_rows must not be empty");
    assert_eq!(x.len(), n, "x.len() must equal n");
    let mut out = vec![0f32; weight_rows.len()];
    for (r, row) in weight_rows.iter().enumerate() {
        check_row_shape(row, n);
        assert!(
            is_symmetric(row),
            "row {r} is not symmetric binary; use dequant_int1_gemv"
        );
        let mut acc = 0f32;
        for g in 0..n / row.group_size {
            let half = f16_to_f32(row.scales[g]) / 2.0;
            let mut signed = 0f32;
            for k in 0..row.group_size {
                let i = g * row.group_size + k;
                signed += if row.quant(i) == 1 { x[i] } else { -x[i] };
            }
            acc += half * signed;
        }
        out[r] = acc;
    }
    out
}
