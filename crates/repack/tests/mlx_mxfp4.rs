//! MLX mxfp4 re-encoding: GGUF re-blocking for routed experts and BF16
//! dequantization for resident matrices.
//!
//! Two independent references, because a test written from the same mental
//! model as the codec can agree with it while both are wrong:
//!
//! - an OCP decoder written here from the format (E2M1 grid, sign bit 3,
//!   `2^(e - 127)`), compared with `==` over every code at every legal
//!   exponent, against both outputs. The GGUF side goes through
//!   `compute::dequantize_mxfp4`, whose own oracle is ggml
//!   (`crates/compute/tests/quant_gguf_mxfp4.rs`), so this also pins the
//!   claim that the two conventions differ only in nibble order;
//! - MLX itself: `fixtures/mlx_mxfp4_linear_oracle.json` holds matrices
//!   `mx.quantize(mode="mxfp4")` produced and the `mx.quantized_matmul`
//!   result, copied from Music 3's precision fixture. The product is BF16, so
//!   that comparison is held to a tolerance, and the test proves the
//!   tolerance still separates the right nibble order from the wrong one.

use turbospark_repack::{
    gguf_mxfp4_row_bytes, mlx_mxfp4_dequant_to_bf16, mlx_mxfp4_reblock_to_gguf,
};

/// The OCP reading of one MLX code, written from the format, not the codec.
fn ocp(code: u8, exponent: u8) -> f32 {
    let magnitude = [0.0f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0][usize::from(code & 7)];
    let signed = if code & 8 == 0 { magnitude } else { -magnitude };
    signed * 2.0f32.powi(i32::from(exponent) - 127)
}

/// Packs one row of codes into MLX's u32 layout: element `i` at bits
/// `4 * (i % 8)` of word `i / 8`, little-endian.
fn pack_mlx(codes: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    for word in codes.chunks(8) {
        let w = word
            .iter()
            .enumerate()
            .fold(0u32, |acc, (i, &c)| acc | (u32::from(c) << (4 * i)));
        out.extend_from_slice(&w.to_le_bytes());
    }
    out
}

fn bf16_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16))
        .collect()
}

#[test]
fn every_code_at_every_legal_exponent_decodes_identically_both_ways() {
    for exponent in 0u8..=254 {
        for shift in 0u8..16 {
            // Every code appears, and elements `j` and `j + 16` never share
            // one, so swapping GGUF's two nibble halves is visible.
            let codes: Vec<u8> = (0u8..32).map(|i| (i + shift + (i / 16) * 5) % 16).collect();
            let expected: Vec<f32> = codes.iter().map(|&c| ocp(c, exponent)).collect();
            let weight = pack_mlx(&codes);

            let gguf = mlx_mxfp4_reblock_to_gguf("t", &weight, &[exponent], 1, 32).unwrap();
            assert_eq!(gguf.len(), gguf_mxfp4_row_bytes(32));
            assert_eq!(
                gguf[0], exponent,
                "the exponent byte carries over unchanged"
            );
            let via_gguf = compute::dequantize_mxfp4(&gguf, 32);
            assert_eq!(
                via_gguf, expected,
                "gguf reblock, exponent {exponent} shift {shift}"
            );

            let bf16 = mlx_mxfp4_dequant_to_bf16("t", &weight, &[exponent], 1, 32).unwrap();
            let via_bf16 = bf16_to_f32(&bf16);
            assert_eq!(
                via_bf16, expected,
                "bf16 dequant, exponent {exponent} shift {shift}"
            );
        }
    }
}

#[test]
fn rows_and_groups_stay_in_order() {
    // 3 rows x 64 columns: two groups per row with distinct exponents, so a
    // row or group stride slip lands a value under the wrong scale.
    let (rows, cols) = (3usize, 64usize);
    // Not periodic in the group width, so two groups never hold the same
    // bytes and reading the wrong one shows.
    let codes: Vec<u8> = (0..rows * cols)
        .map(|i| ((i * 7 + i / 13) % 16) as u8)
        .collect();
    let scales: Vec<u8> = (0..rows * cols / 32).map(|g| 120 + g as u8).collect();
    let weight = pack_mlx(&codes);
    let expected: Vec<f32> = codes
        .iter()
        .enumerate()
        .map(|(i, &c)| ocp(c, scales[i / 32]))
        .collect();

    let gguf = mlx_mxfp4_reblock_to_gguf("t", &weight, &scales, rows, cols).unwrap();
    let row_bytes = gguf_mxfp4_row_bytes(cols);
    for r in 0..rows {
        let row = compute::dequantize_mxfp4(&gguf[r * row_bytes..(r + 1) * row_bytes], cols);
        assert_eq!(row, expected[r * cols..(r + 1) * cols], "row {r}");
    }
    let bf16 = mlx_mxfp4_dequant_to_bf16("t", &weight, &scales, rows, cols).unwrap();
    assert_eq!(bf16_to_f32(&bf16), expected);
}

#[test]
fn exponent_0xff_and_bad_shapes_are_refused() {
    let weight = pack_mlx(&[1u8; 32]);
    let err = mlx_mxfp4_reblock_to_gguf("t", &weight, &[0xFF], 1, 32).unwrap_err();
    assert!(err.to_string().contains("0xFF"), "{err}");
    assert!(mlx_mxfp4_dequant_to_bf16("t", &weight, &[0xFF], 1, 32).is_err());
    // A width that is not a whole number of groups.
    assert!(mlx_mxfp4_reblock_to_gguf("t", &weight[..8], &[127], 1, 16).is_err());
    // A scale plane of the wrong length.
    assert!(mlx_mxfp4_reblock_to_gguf("t", &weight, &[127, 127], 1, 32).is_err());
}

#[derive(serde::Deserialize)]
struct Oracle {
    cases: Vec<OracleCase>,
}

#[derive(serde::Deserialize)]
struct OracleCase {
    name: String,
    weight: Vec<u32>,
    scales: Vec<u8>,
    input: Vec<f32>,
    product: Vec<f32>,
    rows: usize,
    input_dim: usize,
    output_dim: usize,
}

/// `y = W x` per batch row through compute's GGUF MXFP4 GEMV reference.
fn gemv(gguf: &[u8], case: &OracleCase) -> Vec<f32> {
    let row_bytes = gguf_mxfp4_row_bytes(case.input_dim);
    let rows: Vec<&[u8]> = gguf.chunks_exact(row_bytes).collect();
    case.input
        .chunks_exact(case.input_dim)
        .flat_map(|x| compute::dequant_mxfp4_gemv(&rows, x, case.input_dim))
        .collect()
}

fn worst_excess(ours: &[f32], mlx: &[f32]) -> f32 {
    // MLX rounds its product to BF16 (relative quantum 2^-8), so allow a
    // little over that plus a small absolute floor for near-zero outputs.
    ours.iter()
        .zip(mlx)
        .map(|(a, b)| (a - b).abs() - (0.01 * b.abs() + 0.02))
        .fold(f32::NEG_INFINITY, f32::max)
}

#[test]
fn reblocked_bytes_reproduce_mlx_quantized_matmul() {
    let oracle: Oracle =
        serde_json::from_str(include_str!("fixtures/mlx_mxfp4_linear_oracle.json")).unwrap();
    assert_eq!(oracle.cases.len(), 2);
    for case in &oracle.cases {
        let weight: Vec<u8> = case.weight.iter().flat_map(|w| w.to_le_bytes()).collect();
        let gguf = mlx_mxfp4_reblock_to_gguf(
            &case.name,
            &weight,
            &case.scales,
            case.output_dim,
            case.input_dim,
        )
        .unwrap();
        let ours = gemv(&gguf, case);
        assert_eq!(ours.len(), case.rows * case.output_dim);
        let excess = worst_excess(&ours, &case.product);
        assert!(
            excess <= 0.0,
            "{}: off MLX by {excess} past tolerance",
            case.name
        );

        // The tolerance has teeth: reading MLX's adjacent nibble pairs as if
        // they were already GGUF's split halves (the trap this codec exists
        // to avoid) must fall outside it.
        let mut naive = Vec::new();
        for (g, &e) in case.scales.iter().enumerate() {
            naive.push(e);
            naive.extend_from_slice(&weight[g * 16..g * 16 + 16]);
        }
        let wrong = gemv(&naive, case);
        assert!(
            worst_excess(&wrong, &case.product) > 0.0,
            "{}: the adjacent-order misreading passes, so the tolerance proves nothing",
            case.name
        );
    }
}
