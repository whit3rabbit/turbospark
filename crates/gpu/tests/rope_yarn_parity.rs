#![cfg(target_os = "macos")]
//! YaRN rope parity (ROADMAP M5, the `gpt-oss` family): `rope_neox_freqs`
//! against `turbospark_compute::yarn_frequencies` plus a straight-line
//! rotation, on real hardware.
//!
//! TWO SEPARATE CLAIMS ARE UNDER TEST HERE and they fail in different ways.
//! The kernel claim is ordinary parity: given a frequency table it must
//! rotate the halves of each head the way NeoX does and scale by `mscale`.
//! The REFERENCE claim is that the table is YaRN's, and no comparison
//! against this port can establish that -- so the cases below pin it against
//! ggml's own arithmetic, spelled out independently from `rope_yarn`,
//! `rope_yarn_ramp` and `ggml_rope_yarn_corr_dims`, at gpt-oss's own
//! parameters.

use half::f16;
use turbospark_gpu::MetalContext;

/// gpt-oss-20b's, read off the published header: base 150000, factor 32,
/// original context 4096, beta_fast 32, beta_slow 1, head_dim 64.
const BASE: f32 = 150_000.0;
const FACTOR: f32 = 32.0;
const ORIG_CTX: i64 = 4096;
const BETA_FAST: f32 = 32.0;
const BETA_SLOW: f32 = 1.0;
const HEAD_DIM: usize = 64;

fn to_le(v: &[f16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 2);
    for x in v {
        out.extend_from_slice(&x.to_bits().to_le_bytes());
    }
    out
}

/// ggml's `rope_yarn` written out again from its own source, deliberately
/// NOT calling the crate under test. A reference checked against itself is
/// not a check.
fn ggml_yarn_angle(pair: usize) -> (f32, f32) {
    let n_dims = HEAD_DIM as f32;
    let corr = |n_rot: f32| {
        n_dims * (ORIG_CTX as f32 / (n_rot * 2.0 * std::f32::consts::PI)).ln() / (2.0 * BASE.ln())
    };
    let low = corr(BETA_FAST).floor().max(0.0);
    let high = corr(BETA_SLOW).ceil().min(n_dims - 1.0);

    let i0 = 2 * pair;
    let freq_scale = 1.0 / FACTOR;
    let theta_extrap = BASE.powf(-(i0 as f32) / n_dims);
    let theta_interp = freq_scale * theta_extrap;

    let y = (i0 as f32 / 2.0 - low) / (high - low).max(0.001);
    let ramp_mix = 1.0 - y.clamp(0.0, 1.0);
    let theta = theta_interp * (1.0 - ramp_mix) + theta_extrap * ramp_mix;

    let mscale = 1.0 + 0.1 * (1.0f32 / freq_scale).ln();
    (theta, mscale)
}

/// The frequency table is YaRN's, checked against ggml's arithmetic rather
/// than against itself.
#[test]
fn the_frequency_table_reproduces_ggmls_yarn() {
    let spec = turbospark_compute::yarn_frequencies(
        HEAD_DIM, BASE, FACTOR, ORIG_CTX, BETA_FAST, BETA_SLOW,
    );
    assert_eq!(spec.frequencies.len(), HEAD_DIM / 2);
    for pair in 0..HEAD_DIM / 2 {
        let (want, want_mscale) = ggml_yarn_angle(pair);
        let got = spec.frequencies[pair];
        assert!(
            (got - want).abs() <= want.abs() * 1e-6 + 1e-12,
            "pair {pair}: got {got} want {want}"
        );
        assert!((spec.mscale - want_mscale).abs() <= 1e-6);
    }
}

/// The correction dims for gpt-oss's parameters, asserted as the concrete
/// numbers rather than only as agreement.
///
/// Below `low` the ramp is fully EXTRAPOLATING (frequency untouched) and
/// above `high` fully INTERPOLATING (frequency divided by `factor`); an
/// off-by-one in `floor`/`ceil` or a swapped beta moves the boundary and
/// changes long-context behaviour only, which is the failure no smoke test
/// reaches.
#[test]
fn the_ramp_runs_between_the_expected_dimensions() {
    let spec = turbospark_compute::yarn_frequencies(
        HEAD_DIM, BASE, FACTOR, ORIG_CTX, BETA_FAST, BETA_SLOW,
    );
    let plain = turbospark_compute::yarn_frequencies(HEAD_DIM, BASE, 0.0, 0, 0.0, 0.0);

    // Pair 0 is well below the fast correction dim: pure extrapolation, so
    // YaRN leaves it exactly where an unscaled rope would put it.
    assert!(
        (spec.frequencies[0] - plain.frequencies[0]).abs() <= 1e-9,
        "the lowest pair must be untouched by YaRN"
    );
    // The last pair is well above the slow correction dim: pure
    // interpolation, i.e. divided by `factor`.
    let last = HEAD_DIM / 2 - 1;
    let want = plain.frequencies[last] / FACTOR;
    assert!(
        (spec.frequencies[last] - want).abs() <= want.abs() * 1e-6,
        "the highest pair must be fully interpolated: got {} want {want}",
        spec.frequencies[last]
    );
    // And something in between must be neither, or the ramp is not a ramp.
    //
    // THE UNITS ARE THE TRAP, and this assertion is where it surfaced: ggml
    // computes the correction dims in ELEMENT units and then compares them
    // against `i0 / 2`, which is the PAIR index. For gpt-oss's parameters
    // they come out at 8 and 18, so the ramp runs over pairs 8..18 of 32 --
    // pair 6 is fully extrapolating, which is what the first version of this
    // test asserted was mid-ramp and got a flat contradiction for.
    for below in [0usize, 6] {
        assert!(
            (spec.frequencies[below] - plain.frequencies[below]).abs() <= 1e-9,
            "pair {below} is below the fast correction dim and must be untouched"
        );
    }
    let mid = 13;
    let extrap = plain.frequencies[mid];
    let interp = extrap / FACTOR;
    let got = spec.frequencies[mid];
    assert!(
        got < extrap && got > interp,
        "pair {mid} should be mid-ramp: got {got}, extrap {extrap}, interp {interp}"
    );
}

/// `mscale` is NOT 1.0, which is the whole point of reading llama.cpp's
/// double-application cancellation carefully.
#[test]
fn the_magnitude_scale_is_not_one() {
    let spec = turbospark_compute::yarn_frequencies(
        HEAD_DIM, BASE, FACTOR, ORIG_CTX, BETA_FAST, BETA_SLOW,
    );
    let want = 1.0 + 0.1 * FACTOR.ln();
    assert!((spec.mscale - want).abs() <= 1e-6, "got {}", spec.mscale);
    assert!(spec.mscale > 1.34 && spec.mscale < 1.35, "{}", spec.mscale);
}

/// The kernel, against the table and a straight-line NeoX rotation.
#[test]
fn the_kernel_rotates_by_the_frequency_table() {
    let mut context = MetalContext::new().expect("Metal device");
    let num_heads = 4usize;
    let position = 137u32;
    let spec = turbospark_compute::yarn_frequencies(
        HEAD_DIM, BASE, FACTOR, ORIG_CTX, BETA_FAST, BETA_SLOW,
    );

    let src: Vec<f16> = (0..num_heads * HEAD_DIM)
        .map(|i| f16::from_f32(((i as f32) * 0.13).sin()))
        .collect();

    let mut expected: Vec<f32> = src.iter().map(|x| x.to_f32()).collect();
    for h in 0..num_heads {
        for pair in 0..HEAD_DIM / 2 {
            let angle = position as f32 * spec.frequencies[pair];
            let (c, s) = (angle.cos() * spec.mscale, angle.sin() * spec.mscale);
            let lo = h * HEAD_DIM + pair;
            let hi = h * HEAD_DIM + HEAD_DIM / 2 + pair;
            let (x0, x1) = (expected[lo], expected[hi]);
            expected[lo] = x0 * c - x1 * s;
            expected[hi] = x0 * s + x1 * c;
        }
    }

    let data = context.new_buffer_with_data(&to_le(&src));
    let freq_bytes: Vec<u8> = spec
        .frequencies
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    let freqs = context.new_buffer_with_data(&freq_bytes);

    let pass = context.begin_pass();
    turbospark_gpu::encode_rope_neox_freqs(
        &mut context,
        &pass,
        (&data, 0),
        position,
        num_heads as u32,
        HEAD_DIM as u32,
        (HEAD_DIM / 2) as u32,
        (&freqs, 0),
        spec.mscale,
    )
    .expect("encode");
    pass.commit_and_wait();

    let got: Vec<f32> = {
        let ptr = data.contents() as *const u16;
        let bits = unsafe { std::slice::from_raw_parts(ptr, num_heads * HEAD_DIM) };
        bits.iter().map(|&b| f16::from_bits(b).to_f32()).collect()
    };
    for i in 0..got.len() {
        let diff = (got[i] - expected[i]).abs();
        assert!(
            diff <= 3e-3_f32.max(expected[i].abs() * 2e-2),
            "i={i}: got {} want {}",
            got[i],
            expected[i]
        );
    }
}

/// With scaling OFF, the table is the ordinary unscaled one and `mscale` is
/// 1.0, so this kernel agrees with `rope_proportional_neox`.
///
/// That is what makes the new kernel a superset rather than a fork: a family
/// with no YaRN could dispatch either and get the same bytes.
#[test]
fn without_scaling_it_agrees_with_the_scalar_theta_kernel() {
    let mut context = MetalContext::new().expect("Metal device");
    let num_heads = 2usize;
    let position = 41u32;
    let spec = turbospark_compute::yarn_frequencies(HEAD_DIM, BASE, 0.0, 0, 0.0, 0.0);
    assert_eq!(spec.mscale, 1.0);

    let src: Vec<f16> = (0..num_heads * HEAD_DIM)
        .map(|i| f16::from_f32(((i as f32) * 0.23).cos()))
        .collect();

    let a = context.new_buffer_with_data(&to_le(&src));
    let b = context.new_buffer_with_data(&to_le(&src));
    let freq_bytes: Vec<u8> = spec
        .frequencies
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    let freqs = context.new_buffer_with_data(&freq_bytes);

    let pass = context.begin_pass();
    turbospark_gpu::encode_rope_neox_freqs(
        &mut context,
        &pass,
        (&a, 0),
        position,
        num_heads as u32,
        HEAD_DIM as u32,
        (HEAD_DIM / 2) as u32,
        (&freqs, 0),
        spec.mscale,
    )
    .expect("encode freqs");
    turbospark_gpu::encode_rope_proportional_neox(
        &mut context,
        &pass,
        (&b, 0),
        position,
        num_heads as u32,
        HEAD_DIM as u32,
        (HEAD_DIM / 2) as u32,
        BASE,
    )
    .expect("encode scalar");
    pass.commit_and_wait();

    let read = |buf: &metal::Buffer| -> Vec<u16> {
        let ptr = buf.contents() as *const u16;
        unsafe { std::slice::from_raw_parts(ptr, num_heads * HEAD_DIM) }.to_vec()
    };
    let (ga, gb) = (read(&a), read(&b));
    for i in 0..ga.len() {
        let (x, y) = (
            f16::from_bits(ga[i]).to_f32(),
            f16::from_bits(gb[i]).to_f32(),
        );
        assert!((x - y).abs() <= 2e-3, "i={i}: freqs {x} vs scalar {y}");
    }
}
