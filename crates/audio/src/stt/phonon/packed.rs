//! Packed five-value decoder linears for Phonon-1.
//!
//! Reference: `mlx_audio/stt/models/phonon/packed.py` at mlx-audio 0.5.7,
//! commit `e1b19b9054bf163f5d812221a54fcc346f1890e9`. The checkpoint stores
//! each decoder linear as exact base-5 symbols packed ten per 24 bits (rows
//! of 3-byte little-endian groups), plus slim group metadata: one bf16 base
//! scale per output row and one bf16 residual scale per linear. At load the
//! codes unpack to the pair of native MLX 2-bit affine planes (the exact
//! integer semantics of the reference Metal kernel), and both planes
//! dequantize groupwise into one fused f32 weight `base + residual`.
//! The reference evaluates the two planes as separate quantized matmuls and
//! sums the outputs; this port folds the sum into the weights instead, which
//! is order-different f32 rounding of the same real number and is gated by
//! the staged tolerances in the family README.

use crate::{Result, SpeechError};

/// Group width of both affine planes. The reference refuses any other value.
pub const GROUP_SIZE: usize = 128;

/// Base-5 powers for the ten symbol positions inside one 24-bit group,
/// mirroring the divisor table of the reference Metal kernel.
const FIFTHS: [u32; 10] = [1, 5, 25, 125, 625, 3125, 15625, 78125, 390625, 1953125];

/// Packed row width in bytes: ten base-5 symbols per 3-byte group.
pub fn packed_bytes_per_row(in_features: usize) -> Result<usize> {
    if in_features == 0 {
        return Err(SpeechError::Tensor {
            name: "quint5_q".into(),
            why: "input width must be positive".into(),
        });
    }
    let symbol_groups = in_features
        .checked_add(9)
        .ok_or_else(|| SpeechError::Tensor {
            name: "quint5_q".into(),
            why: "input width overflows the packed layout".into(),
        })?
        / 10;
    symbol_groups
        .checked_mul(3)
        .ok_or_else(|| SpeechError::Tensor {
            name: "quint5_q".into(),
            why: "packed row width overflows".into(),
        })
}

/// The five learned values as (base code, residual code) pairs of the two
/// 2-bit planes. The base plane carries `-alpha, 0, +alpha` (code minus one,
/// times the row scale) and the residual plane the same over the per-linear
/// residual scale; the base-5 digit selects their sum.
fn symbol_planes(symbol: u32) -> (u32, u32) {
    match symbol {
        0 => (0, 0),
        1 => (0, 2),
        2 => (1, 1),
        3 => (2, 0),
        _ => (2, 2),
    }
}

/// Reads the 2-bit code at one column of a packed plane row (one u32 word
/// per 16 columns, value `v` at bit `2 * v`, the packing
/// `mx.quantized_matmul` reads for 2-bit weights).
pub(crate) fn plane_code(words: &[u32], column: usize) -> u32 {
    (words[column / 16] >> (2 * (column % 16))) & 0b11
}

/// Unpacks the quint5 byte plane `[out_features, packed row]` into the pair
/// of code planes `[out_features, in_features / 16]` of u32 words. This is a
/// literal port of the reference Metal kernel: each group of 16 logical
/// positions reads its base-5 digits from 3-byte little-endian payloads and
/// maps digit to plane codes through the fixed five-value table.
pub fn unpack_codes(
    quint5: &[u8],
    out_features: usize,
    in_features: usize,
) -> Result<(Vec<u32>, Vec<u32>)> {
    if in_features % 16 != 0 {
        return Err(SpeechError::Tensor {
            name: "quint5_q".into(),
            why: format!("{in_features} is not a multiple of 16 positions per word"),
        });
    }
    let bytes_per_row = packed_bytes_per_row(in_features)?;
    if quint5.len() != out_features * bytes_per_row {
        return Err(SpeechError::Tensor {
            name: "quint5_q".into(),
            why: format!(
                "expected {} bytes for [{out_features}, {bytes_per_row}], got {}",
                out_features * bytes_per_row,
                quint5.len()
            ),
        });
    }
    let words_per_row = in_features / 16;
    let mut base = vec![0u32; out_features * words_per_row];
    let mut residual = vec![0u32; out_features * words_per_row];
    for row in 0..out_features {
        let row_start = row * bytes_per_row;
        for word in 0..words_per_row {
            let mut base_word = 0u32;
            let mut residual_word = 0u32;
            for j in 0..16usize {
                let logical = word * 16 + j;
                let group = logical / 10;
                let offset = logical - group * 10;
                let start = row_start + group * 3;
                let payload = quint5[start] as u32
                    | (quint5[start + 1] as u32) << 8
                    | (quint5[start + 2] as u32) << 16;
                let symbol = (payload / FIFTHS[offset]) % 5;
                let (base_code, residual_code) = symbol_planes(symbol);
                base_word |= base_code << (2 * j);
                residual_word |= residual_code << (2 * j);
            }
            let target = row * words_per_row + word;
            base[target] = base_word;
            residual[target] = residual_word;
        }
    }
    Ok((base, residual))
}

/// The four groupwise affine metadata arrays of the two planes, each
/// `[out_features, groups]` row-major.
#[derive(Debug, Clone, PartialEq)]
pub struct PlaneMetadata {
    pub base_scales: Vec<f32>,
    pub base_biases: Vec<f32>,
    pub residual_scales: Vec<f32>,
    pub residual_biases: Vec<f32>,
}

/// Builds the slim runtime metadata of the reference `materialize` step from
/// the distributed form: one base scale per output row, one residual scale
/// per linear, and biases equal to the negated scales.
pub fn slim_metadata(
    base_alpha: &[f32],
    residual_scale: &[f32],
    out_features: usize,
    groups: usize,
) -> Result<PlaneMetadata> {
    if base_alpha.len() != out_features {
        return Err(SpeechError::Tensor {
            name: "base_alpha".into(),
            why: format!(
                "expected {out_features} row scales, got {}",
                base_alpha.len()
            ),
        });
    }
    if residual_scale.len() != 1 {
        return Err(SpeechError::Tensor {
            name: "residual_scale".into(),
            why: format!("expected one residual scale, got {}", residual_scale.len()),
        });
    }
    let residual = residual_scale[0];
    let mut metadata = PlaneMetadata {
        base_scales: vec![0.0f32; out_features * groups],
        base_biases: vec![0.0f32; out_features * groups],
        residual_scales: vec![0.0f32; out_features * groups],
        residual_biases: vec![0.0f32; out_features * groups],
    };
    for (row, &alpha) in base_alpha.iter().enumerate() {
        for group in 0..groups {
            let slot = row * groups + group;
            metadata.base_scales[slot] = alpha;
            metadata.base_biases[slot] = -alpha;
            metadata.residual_scales[slot] = residual;
            metadata.residual_biases[slot] = -residual;
        }
    }
    Ok(metadata)
}

/// Materializes the fused f32 weight `[out_features, in_features]` (row
/// major): `w = (base_code * base_scale + base_bias) + (residual_code *
/// residual_scale + residual_bias)` with affine groups of [`GROUP_SIZE`]
/// along the input dimension. Both planes must share the group geometry, as
/// the reference's two `mx.quantized_matmul` calls do.
pub fn materialize_weight(
    base_q: &[u32],
    residual_q: &[u32],
    metadata: &PlaneMetadata,
    out_features: usize,
    in_features: usize,
) -> Result<Vec<f32>> {
    if in_features % GROUP_SIZE != 0 {
        return Err(SpeechError::Tensor {
            name: "packed linear".into(),
            why: format!("in_features {in_features} is not divisible by group size {GROUP_SIZE}"),
        });
    }
    let groups = in_features / GROUP_SIZE;
    let words_per_row = in_features / 16;
    let metadata_len = out_features * groups;
    let PlaneMetadata {
        base_scales,
        base_biases,
        residual_scales,
        residual_biases,
    } = metadata;
    for (name, values) in [
        ("base_q", base_q.len()),
        ("residual_q", residual_q.len()),
        ("base_scales", base_scales.len()),
        ("base_biases", base_biases.len()),
        ("residual_scales", residual_scales.len()),
        ("residual_biases", residual_biases.len()),
    ] {
        let expected = match name {
            "base_q" | "residual_q" => out_features * words_per_row,
            _ => metadata_len,
        };
        if values != expected {
            return Err(SpeechError::Tensor {
                name: name.into(),
                why: format!(
                    "expected {expected} values for [{out_features}, {in_features}], got {values}"
                ),
            });
        }
    }
    let output_len = out_features
        .checked_mul(in_features)
        .filter(|&count| count <= isize::MAX as usize / 4)
        .ok_or_else(|| SpeechError::Tensor {
            name: "packed linear".into(),
            why: "materialized shape overflows".into(),
        })?;
    let mut out = vec![0.0f32; output_len];
    for row in 0..out_features {
        let base_row = &base_q[row * words_per_row..(row + 1) * words_per_row];
        let residual_row = &residual_q[row * words_per_row..(row + 1) * words_per_row];
        for column in 0..in_features {
            let group = row * groups + column / GROUP_SIZE;
            let base_value =
                plane_code(base_row, column) as f32 * base_scales[group] + base_biases[group];
            let residual_value = plane_code(residual_row, column) as f32 * residual_scales[group]
                + residual_biases[group];
            out[row * in_features + column] = base_value + residual_value;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::{
        materialize_weight, packed_bytes_per_row, plane_code, slim_metadata, unpack_codes,
    };

    /// Packs one row of base-5 symbols (low digit first) into the 3-byte
    /// little-endian groups of the transport layout; the test-side inverse
    /// of `unpack_codes`.
    fn pack_row(symbols: &[u32], bytes_per_row: usize) -> Vec<u8> {
        let mut row = vec![0u8; bytes_per_row];
        for (group, digits) in symbols.chunks(10).enumerate() {
            let mut payload = 0u32;
            for (position, &symbol) in digits.iter().enumerate() {
                assert!(symbol < 5, "symbols are base-5 digits");
                payload += symbol * super::FIFTHS[position];
            }
            for byte in 0..3 {
                row[group * 3 + byte] = ((payload >> (8 * byte)) & 0xFF) as u8;
            }
        }
        row
    }

    #[test]
    fn packed_row_width_matches_the_reference_formula() {
        assert_eq!(packed_bytes_per_row(1024).unwrap(), 309);
        assert_eq!(packed_bytes_per_row(3072).unwrap(), 924);
        assert_eq!(packed_bytes_per_row(16).unwrap(), 6);
        assert!(packed_bytes_per_row(0).is_err());
        assert!(packed_bytes_per_row(usize::MAX).is_err());
    }

    #[test]
    fn every_symbol_survives_every_position_of_the_unpack() {
        // 160 positions = 10 packed groups of 10 symbols = 48 bytes, and
        // 160 / 16 = 10 output words per row.
        let in_features = 160;
        let bytes_per_row = packed_bytes_per_row(in_features).unwrap();
        for symbol in 0..5u32 {
            // Digit to plane codes: 0 -> (0, 0), 1 -> (0, 2), 2 -> (1, 1),
            // 3 -> (2, 0), 4 -> (2, 2).
            let (base_word, residual_word) = match symbol {
                0 => (0u32, 0u32),
                1 => (0, 0b10101010101010101010101010101010),
                2 => (
                    0b01010101010101010101010101010101,
                    0b01010101010101010101010101010101,
                ),
                3 => (0b10101010101010101010101010101010, 0),
                _ => (
                    0b10101010101010101010101010101010,
                    0b10101010101010101010101010101010,
                ),
            };
            let row = pack_row(&vec![symbol; in_features], bytes_per_row);
            let (base, residual) = unpack_codes(&row, 1, in_features).unwrap();
            assert_eq!(base.len(), in_features / 16);
            assert_eq!(base[0], base_word, "base plane for symbol {symbol}");
            assert_eq!(
                residual[0], residual_word,
                "residual plane for symbol {symbol}"
            );
        }
    }

    #[test]
    fn mixed_symbols_read_back_in_transport_order() {
        // One symbol per position, position i carries digit i % 5.
        let in_features = 160;
        let bytes_per_row = packed_bytes_per_row(in_features).unwrap();
        let symbols: Vec<u32> = (0..in_features).map(|i| (i % 5) as u32).collect();
        let row = pack_row(&symbols, bytes_per_row);
        let (base, residual) = unpack_codes(&row, 1, in_features).unwrap();
        for (position, &symbol) in symbols.iter().enumerate() {
            let (base_code, residual_code) = match symbol {
                0 => (0u32, 0u32),
                1 => (0, 2),
                2 => (1, 1),
                3 => (2, 0),
                _ => (2, 2),
            };
            assert_eq!(plane_code(&base, position), base_code, "base at {position}");
            assert_eq!(
                plane_code(&residual, position),
                residual_code,
                "residual at {position}"
            );
        }
    }

    #[test]
    fn slim_metadata_materializes_the_exact_five_values() {
        let alpha = 0.25f32;
        let residual = 0.0625f32;
        let metadata = slim_metadata(&[alpha], &[residual], 1, 1).unwrap();
        assert_eq!(metadata.base_biases[0], -alpha);
        assert_eq!(metadata.residual_biases[0], -residual);
        // One row of 128 positions carrying each of the five digits.
        let in_features = 128;
        let bytes_per_row = packed_bytes_per_row(in_features).unwrap();
        let row = pack_row(&[0, 1, 2, 3, 4], bytes_per_row);
        let (base, residual_q) = unpack_codes(&row, 1, in_features).unwrap();
        let weight = materialize_weight(&base, &residual_q, &metadata, 1, in_features).unwrap();
        let expected = [
            -(alpha + residual),
            -alpha + residual,
            0.0,
            alpha - residual,
            alpha + residual,
        ];
        for (position, &value) in weight.iter().take(5).enumerate() {
            assert_eq!(value, expected[position], "position {position}");
        }
        for value in &weight[5..] {
            assert_eq!(*value, expected[0], "padding repeats the first digit");
        }
    }

    #[test]
    fn materialize_refuses_mismatched_geometry() {
        let metadata = slim_metadata(&[1.0, 1.0], &[1.0], 2, 1).unwrap();
        // in_features 96 is not divisible by the 128 group size.
        assert!(materialize_weight(&[0; 6], &[0; 6], &metadata, 2, 96).is_err());
        // Wrong word count for the declared width.
        assert!(materialize_weight(&[0; 7], &[0; 8], &metadata, 2, 128).is_err());
        // Wrong metadata length.
        let mut short = metadata.clone();
        short.base_scales.truncate(1);
        assert!(materialize_weight(&[0; 8], &[0; 8], &short, 2, 128).is_err());
    }
}
