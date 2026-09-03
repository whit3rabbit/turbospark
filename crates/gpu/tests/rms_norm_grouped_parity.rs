#![cfg(target_os = "macos")]
//! Parity tests for `rmsnorm_bf16w_grouped_centered` (`qwen4_exp`'s `hc_norm`
//! and PLE norms, `docs/QWEN4_PHASE0.md` item 9's norm taxonomy row 1)
//! against `turbospark_compute::rms_norm_grouped_centered`.
//!
//! **THE DISCRIMINATION CASE HERE IS THE ONE THE PHASE 0 DOC ITSELF FLAGS AS
//! OWED.** This kernel and `rmsnorm_bf16w_centered` (the PLAIN full-width
//! form, `qwen4_exp`'s own `pre_fc_norm_hidden`) both bind a `[D]` BF16
//! weight at the SAME `D = 10240` and both apply a centered `(1 + w)` scale,
//! so no buffer shape distinguishes a call site that reached for the wrong
//! one. The two differ only in whether the reduction is ONE statistic over
//! the whole vector or FOUR independent ones, which is invisible on any
//! fixture whose groups happen to share a magnitude -- exactly the fixture a
//! tidy test would reach for.

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

fn f32_to_bf16_bits(v: f32) -> u16 {
    (v.to_bits() >> 16) as u16
}

/// A weight vector centered at zero (`hc_norm`'s weights are initialized to
/// zero per the reference, `docs/QWEN4_PHASE0.md` item 9), spanning
/// [-0.6, +0.6] for `the_two_norm_conventions_are_different_functions`'s
/// reason in `scaled_norm_and_embed.rs`: near zero, `x*w` and `x*(1+w)`
/// converge, but that trap is about the WRONG convention. This file's trap is
/// the wrong REDUCTION SCOPE, which a narrow weight range does not mask.
fn grouped_weight_bits(d: usize) -> Vec<u16> {
    (0..d)
        .map(|i| f32_to_bf16_bits(0.6 * ((i as f32) * 0.29).sin()))
        .collect()
}

/// Runs the grouped kernel over `x` (`groups * group_dim` halfs) with the
/// full-width weight `w_bits`, returning FP32.
fn run_grouped(
    context: &mut MetalContext,
    x16: &[f16],
    w_bits: &[u16],
    groups: u32,
    eps: f32,
) -> Vec<f32> {
    let total = x16.len();
    let group_dim = total as u32 / groups;
    let x_buf = context.new_buffer_with_data(&to_le(x16));
    let w_bytes: Vec<u8> = w_bits.iter().flat_map(|b| b.to_le_bytes()).collect();
    let w_buf = context.new_buffer_with_data(&w_bytes);
    let out_buf = context.new_output_buffer((total * 2) as u64);

    let pass = context.begin_pass();
    turbospark_gpu::encode_rms_norm_bf16w_grouped_centered(
        context,
        &pass,
        (&x_buf, 0),
        (&w_buf, 0),
        (&out_buf, 0),
        groups,
        group_dim,
        eps,
    )
    .expect("encode");
    pass.commit_and_wait();
    read_halfs(&out_buf, total)
}

/// Same widths, `rmsnorm_bf16w_centered` (the PLAIN full-width sibling).
fn run_plain(context: &mut MetalContext, x16: &[f16], w_bits: &[u16], eps: f32) -> Vec<f32> {
    let d = x16.len();
    let x_buf = context.new_buffer_with_data(&to_le(x16));
    let w_bytes: Vec<u8> = w_bits.iter().flat_map(|b| b.to_le_bytes()).collect();
    let w_buf = context.new_buffer_with_data(&w_bytes);
    let out_buf = context.new_output_buffer((d * 2) as u64);

    let pass = context.begin_pass();
    turbospark_gpu::encode_rms_norm_bf16w_centered(
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

/// The kernel against the CPU reference, per group, at the model's real
/// proportions scaled down (`hc_count = 4`).
///
/// **Every group gets DIFFERENT activations at a DIFFERENT scale**, phase-
/// and amplitude-shifted by group index. That is what makes a mis-derived
/// group stride (reading `x + i` instead of `x + group * GD + i`) visible: a
/// fixture repeating one row's magnitude across groups could not tell a
/// stride bug from a correct one, because the wrong group would still
/// produce a plausible-looking normalized row.
#[test]
fn rmsnorm_bf16w_grouped_centered_matches_cpu_reference() {
    let mut context = MetalContext::new().expect("Metal device");
    let groups = 4usize;
    let group_dim = 64usize;
    let d = groups * group_dim;
    let eps = 1e-6f32;
    let x16: Vec<f16> = (0..d)
        .map(|i| {
            let g = (i / group_dim) as f32;
            let lane = (i % group_dim) as f32;
            // Amplitude grows with group index so each group's own RMS
            // genuinely differs -- a kernel that pooled all four groups into
            // one statistic (the plain-form bug) would read a MIDDLE value
            // that no single group's correct output can match.
            f16::from_f32((1.0 + g) * (lane * 0.23 + g * 1.7).sin())
        })
        .collect();
    let w_bits = grouped_weight_bits(d);
    let w_rounded: Vec<f32> = w_bits
        .iter()
        .map(|&b| f32::from_bits((b as u32) << 16))
        .collect();
    let x32: Vec<f32> = x16.iter().map(|v| v.to_f32()).collect();

    let expected = turbospark_compute::rms_norm_grouped_centered(&x32, &w_rounded, groups, eps);
    let got = run_grouped(&mut context, &x16, &w_bits, groups as u32, eps);

    assert_eq!(got.len(), expected.len());
    for i in 0..d {
        let diff = (got[i] - expected[i]).abs();
        assert!(
            diff <= 2e-3_f32.max(expected[i].abs() * 1e-2),
            "i={i} (group {}): got {} want {}",
            i / group_dim,
            got[i],
            expected[i]
        );
    }
}

/// **THE FIXTURE MUST DISCRIMINATE GROUPED FROM PLAIN**, and this states it
/// rather than assuming it. `hc_norm` (grouped) and `pre_fc_norm_hidden`
/// (plain) are two DIFFERENT tensors at the identical 10240-element width,
/// both centered, both a single full-width BF16 weight -- so the failure
/// to defend against is the two kernels quietly answering the same question
/// on a fixture whose groups happen to share a magnitude.
#[test]
fn the_grouped_and_plain_centered_forms_are_different_functions() {
    let mut context = MetalContext::new().expect("Metal device");
    let groups = 4usize;
    let group_dim = 64usize;
    let d = groups * group_dim;
    let eps = 1e-6f32;
    // Group amplitudes span 1x to 4x, so the pooled (plain) statistic sits
    // well away from any one group's own.
    let x16: Vec<f16> = (0..d)
        .map(|i| {
            let g = (i / group_dim) as f32;
            let lane = (i % group_dim) as f32;
            f16::from_f32((1.0 + g) * (lane * 0.23).sin())
        })
        .collect();
    let w_bits = grouped_weight_bits(d);

    let plain = run_plain(&mut context, &x16, &w_bits, eps);
    let grouped = run_grouped(&mut context, &x16, &w_bits, groups as u32, eps);

    let max_gap = plain
        .iter()
        .zip(&grouped)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_gap > 0.1,
        "the fixture cannot tell the grouped and plain reductions apart \
         (max gap {max_gap}); a parity test on it would pass against either kernel"
    );
}
