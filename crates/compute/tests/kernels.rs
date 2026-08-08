//! Kernel-level tests for the CPU reference implementations in
//! `turbospark_compute`. Numerics parity with the Swift/Metal implementation is
//! out of scope; these tests check the documented math contracts.

use turbospark_compute::{
    apply_streamed_routed, bf16_to_f32, causal_attention, dequant_int4_gemv, dequant_int8_gemv,
    dequantize_int4_affine, dequantize_int8_affine, embed_lookup_int4, embed_lookup_int8,
    f32_to_bf16, gelu_tanh, logit_softcap_softmax, max_abs_diff, quantize_int4_affine,
    quantize_int8_affine, rel_error, rms_norm, rope_neox, rope_paired, run_ffn, wht, Int4AffineRow,
};

#[test]
fn rms_norm_matches_hand_computed_value() {
    let x = [3.0f32, 4.0];
    let w = [1.0f32, 1.0];
    let out = rms_norm(&x, &w, 0.0);
    let inv_rms = 1.0 / ((25.0f32 / 2.0).sqrt());
    assert!((out[0] - 3.0 * inv_rms).abs() < 1e-6);
    assert!((out[1] - 4.0 * inv_rms).abs() < 1e-6);
}

#[test]
fn wht_is_involutory_up_to_scale() {
    let x = [1.0f32, 0.0, 0.0, 0.0];
    let once = wht(&x);
    let twice = wht(&once);
    // WHT is its own inverse (normalized): applying it twice recovers input.
    for (a, b) in twice.iter().zip(x.iter()) {
        assert!((a - b).abs() < 1e-6);
    }
}

#[test]
fn wht_single_element_is_identity() {
    let x = [5.0f32];
    assert_eq!(wht(&x), vec![5.0f32]);
}

#[test]
#[should_panic(expected = "power of two")]
fn wht_rejects_non_power_of_two() {
    let x = [1.0f32, 2.0, 3.0];
    let _ = wht(&x);
}

#[test]
fn rope_paired_zero_position_is_identity() {
    let input: Vec<f32> = (0..8).map(|i| i as f32).collect();
    let out = rope_paired(&input, 1, 1, 8, 8, 0, 10000.0);
    for (a, b) in out.iter().zip(input.iter()) {
        assert!((a - b).abs() < 1e-5);
    }
}

#[test]
fn rope_paired_preserves_pair_norm() {
    let input = [1.0f32, 0.0];
    let out = rope_paired(&input, 1, 1, 2, 2, 5, 10000.0);
    let norm_in = (input[0] * input[0] + input[1] * input[1]).sqrt();
    let norm_out = (out[0] * out[0] + out[1] * out[1]).sqrt();
    assert!((norm_in - norm_out).abs() < 1e-5);
}

#[test]
fn rope_neox_zero_position_is_identity() {
    let input: Vec<f32> = (0..8).map(|i| i as f32).collect();
    let out = rope_neox(&input, 1, 1, 8, 4, 0, 10000.0);
    for (a, b) in out.iter().zip(input.iter()) {
        assert!((a - b).abs() < 1e-5);
    }
}

#[test]
fn causal_attention_single_position_returns_that_value() {
    let head_dim = 4;
    let q = vec![1.0f32; head_dim];
    let k = vec![1.0f32; head_dim];
    let v: Vec<f32> = (0..head_dim).map(|i| i as f32).collect();
    let out = causal_attention(&q, &k, &v, head_dim, 1, 1, 1, None, None);
    assert_eq!(out, v);
}

#[test]
fn causal_attention_window_restricts_to_recent_positions() {
    let head_dim = 2;
    let num_kv = 3;
    // Uniform queries/keys so softmax is uniform over the visible window;
    // only the newest position (v = [1, 1]) should be visible with window=1.
    let q = vec![0.0f32; head_dim];
    let k = vec![0.0f32; head_dim * num_kv];
    let mut v = vec![0.0f32; head_dim * num_kv];
    v[2 * head_dim] = 1.0;
    v[2 * head_dim + 1] = 1.0;
    let out = causal_attention(&q, &k, &v, head_dim, 1, 1, num_kv, Some(1), None);
    assert!((out[0] - 1.0).abs() < 1e-6);
    assert!((out[1] - 1.0).abs() < 1e-6);
}

#[test]
fn bf16_round_trip_is_close() {
    let x = 3.14158f32;
    let bits = f32_to_bf16(x);
    let back = bf16_to_f32(bits);
    assert!((x - back).abs() < 0.02);
}

#[test]
fn int4_affine_round_trip_within_quant_tolerance() {
    let row: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.1).collect();
    let packed = quantize_int4_affine(&row);
    let back = dequantize_int4_affine(&packed, 64);
    // 4-bit code over a 6.3-wide range: step ~= 0.42, so max error is half
    // a step plus BF16 rounding slack on scale/bias.
    let err = max_abs_diff(&back, &row);
    assert!(err < 0.25, "err = {err}");
}

#[test]
fn int8_affine_round_trip_within_quant_tolerance() {
    let row: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.1).collect();
    let packed = quantize_int8_affine(&row);
    let back = dequantize_int8_affine(&packed, 64);
    let err = max_abs_diff(&back, &row);
    assert!(err < 0.02, "err = {err}");
}

#[test]
fn int4_affine_constant_group_round_trips_exactly() {
    let row = vec![2.5f32; 64];
    let packed = quantize_int4_affine(&row);
    let back = dequantize_int4_affine(&packed, 64);
    for v in back {
        assert!((v - 2.5).abs() < 1e-4);
    }
}

#[test]
fn dequant_int4_gemv_matches_identity_row() {
    // A single all-zero-except-one-hot row dequantizes to a constant vector
    // (min == max per group), so the dot product against x is scale*sum(x).
    let row = vec![1.0f32; 64];
    let weight_row = quantize_int4_affine(&row);
    let x = vec![1.0f32; 64];
    let y = dequant_int4_gemv(&[weight_row], &x, 64);
    assert!((y[0] - 64.0).abs() < 0.1, "y = {:?}", y);
}

#[test]
fn dequant_int8_gemv_matches_identity_row() {
    let row = vec![1.0f32; 64];
    let weight_row = quantize_int8_affine(&row);
    let x = vec![1.0f32; 64];
    let y = dequant_int8_gemv(&[weight_row], &x, 64);
    assert!((y[0] - 64.0).abs() < 0.01, "y = {:?}", y);
}

#[test]
fn embed_lookup_int8_reads_the_requested_row() {
    let d = 64;
    let vocab = 2;
    let mut packed = vec![0u8; vocab * d];
    for i in 0..d {
        packed[d + i] = 10; // token 1's row is constant 10
    }
    let scales = vec![f32_to_bf16(1.0); vocab];
    let biases = vec![f32_to_bf16(0.0); vocab];
    let out = embed_lookup_int8(&packed, &scales, &biases, 1, d);
    assert!(out.iter().all(|&v| (v - 10.0).abs() < 1e-3));
}

#[test]
fn embed_lookup_int4_applies_out_scale() {
    let d = 64;
    let packed = vec![0x11u8; d / 2]; // nibble 1 in both halves
    let scales = vec![f32_to_bf16(1.0)];
    let biases = vec![f32_to_bf16(0.0)];
    let out = embed_lookup_int4(&packed, &scales, &biases, 0, d, 2.0);
    assert!(out.iter().all(|&v| (v - 2.0).abs() < 1e-3));
}

#[test]
fn gelu_tanh_zero_is_zero() {
    let out = gelu_tanh(&[0.0]);
    assert!(out[0].abs() < 1e-6);
}

#[test]
fn run_ffn_shape_and_finiteness() {
    // F must be a multiple of GROUP_SIZE (64) for the quantizer, so this
    // shape-only smoke test uses D = F = 64.
    let d = 64;
    let f = 64;
    let gate_rows: Vec<Int4AffineRow> = (0..f)
        .map(|_| quantize_int4_affine(&vec![0.1f32; d]))
        .collect();
    let up_rows: Vec<Int4AffineRow> = (0..f)
        .map(|_| quantize_int4_affine(&vec![0.2f32; d]))
        .collect();
    let down_rows: Vec<Int4AffineRow> = (0..d)
        .map(|_| quantize_int4_affine(&vec![0.05f32; f]))
        .collect();
    let x = vec![1.0f32; d];
    let out = run_ffn(&gate_rows, &up_rows, &down_rows, &x, d, f);
    assert_eq!(out.len(), d);
    assert!(out.iter().all(|v| v.is_finite()));
}

#[test]
fn apply_streamed_routed_zero_weight_is_identity() {
    let d = 64;
    let f = 64;
    let gate = [quantize_int4_affine(&vec![0.1f32; d])];
    let up = [quantize_int4_affine(&vec![0.1f32; d])];
    let down: Vec<Int4AffineRow> = (0..d)
        .map(|_| quantize_int4_affine(&vec![0.1f32; f]))
        .collect();
    let routed_gate = vec![vec![gate[0].clone(); f]];
    let routed_up = vec![vec![up[0].clone(); f]];
    let routed_down = vec![down];
    let x = vec![1.0f32; d];
    let residual = vec![7.0f32; d];
    let out = apply_streamed_routed(
        &x,
        &residual,
        &routed_gate,
        &routed_up,
        &routed_down,
        &[],
        &[],
        d,
        f,
    );
    assert_eq!(out, residual);
}

#[test]
fn logit_softcap_softmax_sums_to_one() {
    let x = [1.0f32, 2.0, 3.0, -4.0];
    let out = logit_softcap_softmax(&x, 30.0);
    let sum: f32 = out.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
}

#[test]
fn logit_softcap_softmax_caps_extreme_logits() {
    // With softcap 30, extremely large logits saturate tanh to +-1, so the
    // capped values converge to +-30 regardless of magnitude beyond that.
    let x = [1_000_000.0f32, -1_000_000.0];
    let out = logit_softcap_softmax(&x, 30.0);
    assert!(out[0] > 0.999_999);
    assert!(out[1] < 0.000_001);
}

#[test]
fn rel_error_zero_for_identical_vectors() {
    let a = [1.0f32, 2.0, 3.0];
    assert_eq!(rel_error(&a, &a), 0.0);
}

#[test]
fn rel_error_uses_absolute_floor_when_reference_is_all_zero() {
    let a = [0.0f32, 1e-7];
    let r = [0.0f32, 0.0];
    let err = rel_error(&a, &r);
    assert!((err - 1e-7 / 1e-6).abs() < 1e-6);
}
