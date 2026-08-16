#![cfg(target_os = "macos")]
//! Parity tests for `rmsnorm_bf16w` (scaled RMSNorm, the learned-weight
//! form real checkpoints need) and `embed_lookup_int4` (GPU embedding
//! row dequant) against their `turbospark_compute` references.

use half::f16;
use turbospark_gpu::MetalContext;

fn to_le(v: &[f16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 2);
    for x in v {
        out.extend_from_slice(&x.to_bits().to_le_bytes());
    }
    out
}

fn read_halfs(buffer: &metal::Buffer, len: usize) -> Vec<f32> {
    let ptr = buffer.contents() as *const u16;
    let bits = unsafe { std::slice::from_raw_parts(ptr, len) };
    bits.iter().map(|&b| f16::from_bits(b).to_f32()).collect()
}

/// BF16 bit pattern of an f32 (truncation, matching the repack format).
fn f32_to_bf16_bits(v: f32) -> u16 {
    (v.to_bits() >> 16) as u16
}

#[test]
fn rmsnorm_bf16w_matches_cpu_reference() {
    let mut context = MetalContext::new().expect("Metal device");
    let d = 128usize;
    let eps = 1e-6f32;
    let x16: Vec<f16> = (0..d)
        .map(|i| f16::from_f32(((i as f32) * 0.23).sin()))
        .collect();
    let w32: Vec<f32> = (0..d).map(|i| 0.5 + ((i as f32) * 0.17).cos()).collect();
    let w_bits: Vec<u16> = w32.iter().map(|&v| f32_to_bf16_bits(v)).collect();

    // Reference uses the BF16-rounded weights the kernel actually reads.
    let w_rounded: Vec<f32> = w_bits
        .iter()
        .map(|&b| f32::from_bits((b as u32) << 16))
        .collect();
    let x32: Vec<f32> = x16.iter().map(|v| v.to_f32()).collect();
    let expected = turbospark_compute::rms_norm(&x32, &w_rounded, eps);

    let x_buf = context.new_buffer_with_data(&to_le(&x16));
    let w_bytes: Vec<u8> = w_bits.iter().flat_map(|b| b.to_le_bytes()).collect();
    let w_buf = context.new_buffer_with_data(&w_bytes);
    let out_buf = context.new_output_buffer((d * 2) as u64);

    let pass = context.begin_pass();
    turbospark_gpu::encode_rms_norm_bf16w(
        &mut context,
        &pass,
        (&x_buf, 0),
        (&w_buf, 0),
        (&out_buf, 0),
        d as u32,
        eps,
    )
    .expect("encode");
    pass.commit_and_wait();

    let got = read_halfs(&out_buf, d);
    for i in 0..d {
        let diff = (got[i] - expected[i]).abs();
        assert!(
            diff <= 2e-3_f32.max(expected[i].abs() * 1e-2),
            "i={i}: got {} want {}",
            got[i],
            expected[i]
        );
    }
}

/// Runs one norm kernel over `x` with BF16 weight bits `w_bits`, returning
/// FP32. `centered` picks `rmsnorm_bf16w_centered` over `rmsnorm_bf16w`.
fn run_norm(
    context: &mut MetalContext,
    x16: &[f16],
    w_bits: &[u16],
    eps: f32,
    centered: bool,
) -> Vec<f32> {
    let d = x16.len();
    let x_buf = context.new_buffer_with_data(&to_le(x16));
    let w_bytes: Vec<u8> = w_bits.iter().flat_map(|b| b.to_le_bytes()).collect();
    let w_buf = context.new_buffer_with_data(&w_bytes);
    let out_buf = context.new_output_buffer((d * 2) as u64);

    let pass = context.begin_pass();
    let encode = if centered {
        turbospark_gpu::encode_rms_norm_bf16w_centered
    } else {
        turbospark_gpu::encode_rms_norm_bf16w
    };
    encode(
        context,
        &pass,
        (&x_buf, 0),
        (&w_buf, 0),
        (&out_buf, 0),
        d as u32,
        eps,
    )
    .expect("encode");
    pass.commit_and_wait();
    read_halfs(&out_buf, d)
}

/// A weight vector CENTRED AT ZERO, which is what a `CenteredRMSNorm`
/// checkpoint actually stores, spanning [-0.6, +0.6].
///
/// **The range is the point.** A fixture whose weights sit near zero makes
/// `x * w` and `x * (1 + w)` numerically close in the first case and simply
/// small in the second, so a parity test written on one could pass against
/// either kernel. `the_two_norm_conventions_are_different_functions` asserts
/// that this fixture discriminates, rather than leaving it to be assumed.
fn centered_weight_bits(d: usize) -> Vec<u16> {
    (0..d)
        .map(|i| f32_to_bf16_bits(0.6 * ((i as f32) * 0.31).sin()))
        .collect()
}

#[test]
fn rmsnorm_bf16w_centered_matches_cpu_reference() {
    let mut context = MetalContext::new().expect("Metal device");
    let d = 128usize;
    let eps = 1e-8f32; // this family's POST-norm epsilon
    let x16: Vec<f16> = (0..d)
        .map(|i| f16::from_f32(((i as f32) * 0.23).sin()))
        .collect();
    let w_bits = centered_weight_bits(d);

    // The reference reads the BF16-rounded weights the kernel actually sees.
    let w_rounded: Vec<f32> = w_bits
        .iter()
        .map(|&b| f32::from_bits((b as u32) << 16))
        .collect();
    let x32: Vec<f32> = x16.iter().map(|v| v.to_f32()).collect();
    let expected = turbospark_compute::rms_norm_centered(&x32, &w_rounded, eps);

    let got = run_norm(&mut context, &x16, &w_bits, eps, true);
    for i in 0..d {
        let diff = (got[i] - expected[i]).abs();
        assert!(
            diff <= 2e-3_f32.max(expected[i].abs() * 1e-2),
            "i={i}: got {} want {}",
            got[i],
            expected[i]
        );
    }
}

/// THE FIXTURE MUST DISCRIMINATE, and this states it rather than assuming
/// it.
///
/// `muse_glimmer` dispatches BOTH kernels in one forward pass -- centered on
/// its four per-layer norms, plain on `model.norm` -- so the failure to
/// defend against is not "the kernel is wrong" but "the two kernels are the
/// same function and nobody noticed". On a tidy fixture with weights near
/// zero they very nearly are.
#[test]
fn the_two_norm_conventions_are_different_functions() {
    let mut context = MetalContext::new().expect("Metal device");
    let d = 128usize;
    let eps = 1e-5f32;
    let x16: Vec<f16> = (0..d)
        .map(|i| f16::from_f32(((i as f32) * 0.23).sin()))
        .collect();
    let w_bits = centered_weight_bits(d);

    let plain = run_norm(&mut context, &x16, &w_bits, eps, false);
    let centered = run_norm(&mut context, &x16, &w_bits, eps, true);

    let max_gap = plain
        .iter()
        .zip(&centered)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_gap > 0.5,
        "the fixture cannot tell the two conventions apart (max gap {max_gap}); \
         a parity test on it would pass against either kernel"
    );
}

/// The `+1` is exactly a `+1`: at a stored weight of ZERO the centered form
/// must reproduce the plain form at a stored weight of ONE.
///
/// This pins the constant itself, where the parity test above would also
/// pass for `(c + w)` at any `c` the CPU reference happened to share.
#[test]
fn a_zero_centered_weight_is_unit_scale() {
    let mut context = MetalContext::new().expect("Metal device");
    let d = 64usize;
    let eps = 1e-6f32;
    let x16: Vec<f16> = (0..d)
        .map(|i| f16::from_f32(((i as f32) * 0.41).cos()))
        .collect();

    let zeros = vec![f32_to_bf16_bits(0.0); d];
    let ones = vec![f32_to_bf16_bits(1.0); d];

    let centered_at_zero = run_norm(&mut context, &x16, &zeros, eps, true);
    let plain_at_one = run_norm(&mut context, &x16, &ones, eps, false);

    for i in 0..d {
        assert_eq!(
            centered_at_zero[i], plain_at_one[i],
            "i={i}: centered(w=0) must equal plain(w=1) bit for bit"
        );
    }
}

#[test]
fn embed_lookup_int4_matches_cpu_reference() {
    let mut context = MetalContext::new().expect("Metal device");
    let vocab = 16usize;
    let d = 64usize;
    let token = 7usize;
    let out_scale = (d as f32).sqrt();

    // Quantize a deterministic table row set.
    let mut packed = Vec::new();
    let mut scales = Vec::new();
    let mut biases = Vec::new();
    for r in 0..vocab {
        let row: Vec<f32> = (0..d)
            .map(|i| (((r * 131 + i) as f32) * 0.13).sin())
            .collect();
        let q = turbospark_compute::quantize_int4_affine(&row);
        packed.extend_from_slice(&q.packed);
        scales.extend_from_slice(&q.scales);
        biases.extend_from_slice(&q.biases);
    }

    let expected = turbospark_compute::quant::embed_lookup_int4(
        &packed, &scales, &biases, token, d, out_scale,
    );

    let u16_le = |v: &[u16]| -> Vec<u8> { v.iter().flat_map(|b| b.to_le_bytes()).collect() };
    let table_buf = context.new_buffer_with_data(&packed);
    let scales_buf = context.new_buffer_with_data(&u16_le(&scales));
    let biases_buf = context.new_buffer_with_data(&u16_le(&biases));
    let out_buf = context.new_output_buffer((d * 2) as u64);

    let pass = context.begin_pass();
    turbospark_gpu::encode_embed_lookup_int4(
        &mut context,
        &pass,
        (&table_buf, 0),
        (&scales_buf, 0),
        (&biases_buf, 0),
        (&out_buf, 0),
        token as u32,
        d as u32,
        out_scale,
    )
    .expect("encode");
    pass.commit_and_wait();

    let got = read_halfs(&out_buf, d);
    for i in 0..d {
        let diff = (got[i] - expected[i]).abs();
        assert!(
            diff <= 5e-3_f32.max(expected[i].abs() * 1e-2),
            "i={i}: got {} want {}",
            got[i],
            expected[i]
        );
    }
}
