//! Affine int4/int8 groupwise quantization reference (MLX `affine` layout).
//!
//! Ported from `Infrastructure/ModelIO/Quantization.swift`. BF16 scale/bias
//! per group of [`GROUP_SIZE`] elements are carried as raw `u16` bit patterns
//! (top 16 bits of the FP32 word) so the same buffer shape matches what the
//! GPU kernels read as `device const bfloat*`.

/// Elements per quantization group.
pub const GROUP_SIZE: usize = 64;

/// Widen a BF16 bit pattern to `f32`. Pure bit-shift: BF16 is exactly the top
/// 16 bits of an FP32 word, so no rounding is needed on decode.
#[inline]
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Narrow an `f32` to a BF16 bit pattern with round-half-to-even.
///
/// NaN is handled separately: the round-half-to-even add below can carry a
/// NaN's mantissa bits into the exponent field, landing on `+inf` or `-0.0`
/// instead of a narrower NaN (a probed `0x7F800001` narrows to `+inf`, and
/// `0x7FFFFFFF` to `-0.0`; only the canonical `f32::NAN` bit pattern happens
/// to survive). A weight that is NaN in a source checkpoint would otherwise
/// be written to the `.gturbo` narrowing path as a finite value and never be
/// seen again. Finite and infinite inputs are unaffected by this branch.
#[inline]
pub fn f32_to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    if x.is_nan() {
        // Top 16 bits (sign + exponent + high mantissa) with the quiet bit
        // forced, so the narrowed value stays a NaN even if the truncated
        // high mantissa bits were all zero.
        return ((bits >> 16) | 0x0040) as u16;
    }
    let lsb = (bits >> 16) & 1;
    let rounding_bias = 0x7FFFu32.wrapping_add(lsb);
    (bits.wrapping_add(rounding_bias) >> 16) as u16
}

/// One row of INT4-affine packed weights: `N / 2` bytes (two nibbles per
/// byte, low nibble = even index, high nibble = odd index), plus one
/// BF16 scale and one BF16 bias per group of [`GROUP_SIZE`] elements.
#[derive(Debug, Clone)]
pub struct Int4AffineRow {
    pub packed: Vec<u8>,
    pub scales: Vec<u16>,
    pub biases: Vec<u16>,
}

/// One row of INT8-affine packed weights: `N` unsigned bytes, plus one BF16
/// scale and one BF16 bias per group of [`GROUP_SIZE`] elements.
#[derive(Debug, Clone)]
pub struct Int8AffineRow {
    pub packed: Vec<u8>,
    pub scales: Vec<u16>,
    pub biases: Vec<u16>,
}

/// Affine 4-bit quantize: `q in [0, 15]`, `w ~= q * scale + bias`, scale/bias
/// derived from the per-group min/max and rounded through BF16 before the
/// quantize pass, so decode reproduces the same `q` the row stores.
pub fn quantize_int4_affine(row: &[f32]) -> Int4AffineRow {
    assert!(
        row.len() % GROUP_SIZE == 0,
        "row length {} is not a multiple of {GROUP_SIZE}",
        row.len()
    );
    let n_groups = row.len() / GROUP_SIZE;
    let mut packed = vec![0u8; row.len() / 2];
    let mut scales = vec![0u16; n_groups];
    let mut biases = vec![0u16; n_groups];

    for g in 0..n_groups {
        let group = &row[g * GROUP_SIZE..(g + 1) * GROUP_SIZE];
        let wmin = group.iter().copied().fold(f32::INFINITY, f32::min);
        let wmax = group.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let (scale_f, bias_f) = if wmax == wmin {
            (1.0, wmin)
        } else {
            ((wmax - wmin) / 15.0, wmin)
        };
        let s_bits = f32_to_bf16(scale_f);
        let b_bits = f32_to_bf16(bias_f);
        scales[g] = s_bits;
        biases[g] = b_bits;
        let scale = bf16_to_f32(s_bits);
        let bias = bf16_to_f32(b_bits);
        let inv_scale = if scale == 0.0 { 0.0 } else { 1.0 / scale };

        for (k, &w) in group.iter().enumerate() {
            let q = (((w - bias) * inv_scale).round() as i32).clamp(0, 15) as u8;
            let byte_idx = g * (GROUP_SIZE / 2) + (k / 2);
            if k & 1 == 0 {
                packed[byte_idx] = (packed[byte_idx] & 0xF0) | (q & 0x0F);
            } else {
                packed[byte_idx] = (packed[byte_idx] & 0x0F) | (q << 4);
            }
        }
    }
    Int4AffineRow {
        packed,
        scales,
        biases,
    }
}

/// Dequantize an [`Int4AffineRow`] to `n` FP32 elements.
pub fn dequantize_int4_affine(r: &Int4AffineRow, n: usize) -> Vec<f32> {
    assert_eq!(n, r.packed.len() * 2, "n must equal packed.len() * 2");
    assert!(
        n % GROUP_SIZE == 0,
        "row length {n} is not a multiple of {GROUP_SIZE}"
    );
    let n_groups = n / GROUP_SIZE;
    assert_eq!(r.scales.len(), n_groups, "scales.len() must equal n_groups");
    assert_eq!(r.biases.len(), n_groups, "biases.len() must equal n_groups");
    let mut out = vec![0f32; n];
    for g in 0..n_groups {
        let scale = bf16_to_f32(r.scales[g]);
        let bias = bf16_to_f32(r.biases[g]);
        for k in 0..GROUP_SIZE {
            let byte_idx = g * (GROUP_SIZE / 2) + (k / 2);
            let b = r.packed[byte_idx];
            let nibble = if k & 1 == 0 { b & 0x0F } else { b >> 4 };
            out[g * GROUP_SIZE + k] = nibble as f32 * scale + bias;
        }
    }
    out
}

/// Affine 8-bit quantize: `q in [0, 255]`, `w ~= q * scale + bias`.
pub fn quantize_int8_affine(row: &[f32]) -> Int8AffineRow {
    assert!(
        row.len() % GROUP_SIZE == 0,
        "row length {} is not a multiple of {GROUP_SIZE}",
        row.len()
    );
    let n_groups = row.len() / GROUP_SIZE;
    let mut packed = vec![0u8; row.len()];
    let mut scales = vec![0u16; n_groups];
    let mut biases = vec![0u16; n_groups];

    for g in 0..n_groups {
        let group = &row[g * GROUP_SIZE..(g + 1) * GROUP_SIZE];
        let wmin = group.iter().copied().fold(f32::INFINITY, f32::min);
        let wmax = group.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let (scale_f, bias_f) = if wmax == wmin {
            (1.0, wmin)
        } else {
            ((wmax - wmin) / 255.0, wmin)
        };
        let s_bits = f32_to_bf16(scale_f);
        let b_bits = f32_to_bf16(bias_f);
        scales[g] = s_bits;
        biases[g] = b_bits;
        let scale = bf16_to_f32(s_bits);
        let bias = bf16_to_f32(b_bits);
        let inv_scale = if scale == 0.0 { 0.0 } else { 1.0 / scale };

        for (k, &w) in group.iter().enumerate() {
            let q = (((w - bias) * inv_scale).round() as i32).clamp(0, 255) as u8;
            packed[g * GROUP_SIZE + k] = q;
        }
    }
    Int8AffineRow {
        packed,
        scales,
        biases,
    }
}

/// Dequantize an [`Int8AffineRow`] to `n` FP32 elements.
pub fn dequantize_int8_affine(r: &Int8AffineRow, n: usize) -> Vec<f32> {
    assert_eq!(n, r.packed.len(), "n must equal packed.len()");
    assert!(
        n % GROUP_SIZE == 0,
        "row length {n} is not a multiple of {GROUP_SIZE}"
    );
    let n_groups = n / GROUP_SIZE;
    assert_eq!(r.scales.len(), n_groups, "scales.len() must equal n_groups");
    assert_eq!(r.biases.len(), n_groups, "biases.len() must equal n_groups");
    let mut out = vec![0f32; n];
    for g in 0..n_groups {
        let scale = bf16_to_f32(r.scales[g]);
        let bias = bf16_to_f32(r.biases[g]);
        for k in 0..GROUP_SIZE {
            out[g * GROUP_SIZE + k] = r.packed[g * GROUP_SIZE + k] as f32 * scale + bias;
        }
    }
    out
}

/// FP32 reference for the INT4-affine groupwise GEMV `y = W * x`.
pub fn dequant_int4_gemv(weight_rows: &[Int4AffineRow], x: &[f32], n: usize) -> Vec<f32> {
    assert!(!weight_rows.is_empty());
    assert_eq!(x.len(), n);
    assert!(n % GROUP_SIZE == 0, "N must be a multiple of {GROUP_SIZE}");
    weight_rows
        .iter()
        .map(|row| {
            let w_row = dequantize_int4_affine(row, n);
            w_row.iter().zip(x.iter()).map(|(w, xv)| w * xv).sum()
        })
        .collect()
}

/// FP32 reference for the INT8-affine groupwise GEMV `y = W * x`.
pub fn dequant_int8_gemv(weight_rows: &[Int8AffineRow], x: &[f32], n: usize) -> Vec<f32> {
    assert!(!weight_rows.is_empty());
    assert_eq!(x.len(), n);
    assert!(n % GROUP_SIZE == 0, "N must be a multiple of {GROUP_SIZE}");
    weight_rows
        .iter()
        .map(|row| {
            let w_row = dequantize_int8_affine(row, n);
            w_row.iter().zip(x.iter()).map(|(w, xv)| w * xv).sum()
        })
        .collect()
}

/// FP32 reference for the INT8-affine embedding lookup: `out = dequantize(table[token_id])`.
///
/// `table_packed` is `V * D` unsigned bytes; `table_scales` / `table_biases`
/// are `V * (D / GROUP_SIZE)` BF16 bit patterns.
pub fn embed_lookup_int8(
    table_packed: &[u8],
    table_scales: &[u16],
    table_biases: &[u16],
    token_id: usize,
    d: usize,
) -> Vec<f32> {
    assert!(d % GROUP_SIZE == 0, "D must be a multiple of {GROUP_SIZE}");
    let groups_per_row = d / GROUP_SIZE;
    let pack_base = token_id * d;
    let scale_base = token_id * groups_per_row;
    assert!(pack_base + d <= table_packed.len(), "token out of range");
    assert!(
        scale_base + groups_per_row <= table_scales.len(),
        "scales out of range"
    );
    assert!(
        scale_base + groups_per_row <= table_biases.len(),
        "biases out of range"
    );

    let mut out = vec![0f32; d];
    for g in 0..groups_per_row {
        let scale = bf16_to_f32(table_scales[scale_base + g]);
        let bias = bf16_to_f32(table_biases[scale_base + g]);
        let group_base = pack_base + g * GROUP_SIZE;
        for k in 0..GROUP_SIZE {
            out[g * GROUP_SIZE + k] = table_packed[group_base + k] as f32 * scale + bias;
        }
    }
    out
}

/// FP32 reference for the INT4-affine embedding lookup. `table_packed` carries
/// `V * D / 2` bytes (low nibble = even index, high nibble = odd index).
/// `out_scale` applies a post-lookup scale (e.g. `sqrt(hidden_size)` for
/// Gemma 4's post-embedding scale; `1.0` for the raw dequant).
pub fn embed_lookup_int4(
    table_packed: &[u8],
    table_scales: &[u16],
    table_biases: &[u16],
    token_id: usize,
    d: usize,
    out_scale: f32,
) -> Vec<f32> {
    assert!(d % GROUP_SIZE == 0, "D must be a multiple of {GROUP_SIZE}");
    let groups_per_row = d / GROUP_SIZE;
    let row_bytes = d / 2;
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
        let byte = table_packed[pack_base + (i >> 1)];
        let q = if i & 1 == 0 { byte & 0x0F } else { byte >> 4 };
        let g = i / GROUP_SIZE;
        let scale = bf16_to_f32(table_scales[scale_base + g]);
        let bias = bf16_to_f32(table_biases[scale_base + g]);
        *out_val = (q as f32 * scale + bias) * out_scale;
    }
    out
}
