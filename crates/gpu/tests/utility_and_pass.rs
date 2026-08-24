#![cfg(target_os = "macos")]
//! Parity tests for the `utility.metal` elementwise kernels (against the
//! `turbospark_compute` reference math) and for the `PassEncoder` batching
//! path (an encoded chain must be bit-identical to the one-shot dispatch
//! wrappers it replaces).

use half::f16;
use turbospark_gpu::MetalContext;

fn to_le(v: &[f16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 2);
    for x in v {
        out.extend_from_slice(&x.to_bits().to_le_bytes());
    }
    out
}

fn read_halfs(buffer: &metal::Buffer, len: usize) -> Vec<f16> {
    let ptr = buffer.contents() as *const u16;
    let bits = unsafe { std::slice::from_raw_parts(ptr, len) };
    bits.iter().map(|&b| f16::from_bits(b)).collect()
}

fn test_vec(n: usize, step: f32) -> Vec<f16> {
    (0..n)
        .map(|i| f16::from_f32((i as f32 * step).sin() * 2.0))
        .collect()
}

#[test]
fn gelu_mul_matches_compute_reference() {
    let mut context = MetalContext::new().expect("Metal device");
    let n = 256usize;
    let gate = test_vec(n, 0.13);
    let up = test_vec(n, 0.29);

    let gate_buf = context.new_buffer_with_data(&to_le(&gate));
    let up_buf = context.new_buffer_with_data(&to_le(&up));
    let out_buf = context.new_output_buffer((n * 2) as u64);

    let pass = context.begin_pass();
    turbospark_gpu::encode_gelu_mul(
        &mut context,
        &pass,
        (&gate_buf, 0),
        (&up_buf, 0),
        (&out_buf, 0),
        n as u32,
    )
    .expect("encode");
    pass.commit_and_wait();
    let got = read_halfs(&out_buf, n);

    let gate32: Vec<f32> = gate.iter().map(|x| x.to_f32()).collect();
    let expected32 = turbospark_compute::moe::gelu_tanh(&gate32);
    for i in 0..n {
        let want = expected32[i] * up[i].to_f32();
        let diff = (got[i].to_f32() - want).abs();
        // The kernel clamps tanh's argument at +/-20 (saturated there at
        // FP32 anyway) and rounds through FP16; small tolerance.
        assert!(
            diff <= 2e-3_f32.max(want.abs() * 2e-2),
            "i={i}: got {} want {want}",
            got[i].to_f32()
        );
    }
}

#[test]
fn silu_mul_matches_reference_formula() {
    let mut context = MetalContext::new().expect("Metal device");
    let n = 256usize;
    let gate = test_vec(n, 0.17);
    let up = test_vec(n, 0.31);

    let gate_buf = context.new_buffer_with_data(&to_le(&gate));
    let up_buf = context.new_buffer_with_data(&to_le(&up));
    let out_buf = context.new_output_buffer((n * 2) as u64);

    let pass = context.begin_pass();
    turbospark_gpu::encode_silu_mul(
        &mut context,
        &pass,
        (&gate_buf, 0),
        (&up_buf, 0),
        (&out_buf, 0),
        n as u32,
    )
    .expect("encode");
    pass.commit_and_wait();
    let got = read_halfs(&out_buf, n);

    for i in 0..n {
        let g = gate[i].to_f32();
        let want = (g / (1.0 + (-g).exp())) * up[i].to_f32();
        let diff = (got[i].to_f32() - want).abs();
        assert!(
            diff <= 2e-3_f32.max(want.abs() * 2e-2),
            "i={i}: got {} want {want}",
            got[i].to_f32()
        );
    }
}

#[test]
fn residual_add_matches_host_fp16_add() {
    let mut context = MetalContext::new().expect("Metal device");
    let n = 300usize; // deliberately not a multiple of the threadgroup size
    let hidden = test_vec(n, 0.07);
    let delta = test_vec(n, 0.11);

    let hidden_buf = context.new_buffer_with_data(&to_le(&hidden));
    let delta_buf = context.new_buffer_with_data(&to_le(&delta));

    let pass = context.begin_pass();
    turbospark_gpu::encode_residual_add(
        &mut context,
        &pass,
        (&hidden_buf, 0),
        (&delta_buf, 0),
        n as u32,
    )
    .expect("encode");
    pass.commit_and_wait();
    let got = read_halfs(&hidden_buf, n);

    for i in 0..n {
        let want = f16::from_f32(hidden[i].to_f32() + delta[i].to_f32());
        assert_eq!(got[i].to_bits(), want.to_bits(), "i={i}");
    }
}

/// A chain encoded through one `PassEncoder` (serial dispatch order) must
/// be bit-identical to running the same kernels through the one-shot
/// wrappers with host round-trips between them.
#[test]
fn encoded_chain_matches_one_shot_dispatches() {
    let mut context = MetalContext::new().expect("Metal device");
    let n = 128usize;
    let x = test_vec(n, 0.23);
    let eps = 1e-6_f32;

    // One-shot: rmsnorm, then rmsnorm of the result again (any chain works
    // for the equivalence check; two steps prove ordering).
    let step1 = turbospark_gpu::rms_norm_no_scale(&mut context, &x, eps).expect("one-shot 1");
    let step2 = turbospark_gpu::rms_norm_no_scale(&mut context, &step1, eps).expect("one-shot 2");

    // Batched: both dispatches in one command buffer, chained on GPU.
    let x_buf = context.new_buffer_with_data(&to_le(&x));
    let mid_buf = context.new_output_buffer((n * 2) as u64);
    let out_buf = context.new_output_buffer((n * 2) as u64);
    let pass = context.begin_pass();
    turbospark_gpu::encode_rms_norm_no_scale(
        &mut context,
        &pass,
        (&x_buf, 0),
        (&mid_buf, 0),
        n as u32,
        eps,
    )
    .expect("encode 1");
    turbospark_gpu::encode_rms_norm_no_scale(
        &mut context,
        &pass,
        (&mid_buf, 0),
        (&out_buf, 0),
        n as u32,
        eps,
    )
    .expect("encode 2");
    pass.commit_and_wait();
    let batched = read_halfs(&out_buf, n);

    for i in 0..n {
        assert_eq!(batched[i].to_bits(), step2[i].to_bits(), "i={i}");
    }
}

#[test]
fn logit_softcap_matches_the_fused_kernel_cap() {
    let mut context = MetalContext::new().expect("Metal device");
    let n = 512usize;
    let softcap = 30.0f32;
    // Span the saturating range: the cap only matters where |z| >> softcap.
    let logits: Vec<f16> = (0..n)
        .map(|i| f16::from_f32((i as f32 - n as f32 / 2.0) * 1.5))
        .collect();

    let buf = context.new_buffer_with_data(&to_le(&logits));
    let pass = context.begin_pass();
    turbospark_gpu::encode_logit_softcap(&mut context, &pass, (&buf, 0), softcap, n as u32)
        .expect("encode");
    pass.commit_and_wait();
    let got = read_halfs(&buf, n);

    for i in 0..n {
        let want = softcap * (logits[i].to_f32() / softcap).tanh();
        let diff = (got[i].to_f32() - want).abs();
        assert!(diff < 0.02, "i={i} got {} want {want}", got[i].to_f32());
        assert!(got[i].to_f32().abs() <= softcap, "i={i} escapes the cap");
    }

    // The cap alone must NOT normalize: a softmaxed vector would sum to 1.
    let sum: f32 = got.iter().map(|v| v.to_f32()).sum();
    assert!(
        (sum - 1.0).abs() > 0.5,
        "the softcap kernel must not softmax; got sum {sum}"
    );
}

/// Qwen 3.6's three gating kernels, against `turbospark_compute::gating`.
#[test]
fn sigmoid_gate_mul_matches_compute_reference() {
    let mut context = MetalContext::new().expect("Metal device");
    let n = 256usize;
    let out = test_vec(n, 0.17);
    let gate = test_vec(n, 0.41);

    let out_buf = context.new_buffer_with_data(&to_le(&out));
    let gate_buf = context.new_buffer_with_data(&to_le(&gate));
    let pass = context.begin_pass();
    turbospark_gpu::encode_sigmoid_gate_mul(
        &mut context,
        &pass,
        (&out_buf, 0),
        (&gate_buf, 0),
        n as u32,
    )
    .expect("dispatch");
    pass.commit_and_wait();

    let got: Vec<f32> = read_halfs(&out_buf, n).iter().map(|h| h.to_f32()).collect();
    let want = turbospark_compute::sigmoid_gate_mul(
        &out.iter().map(|h| h.to_f32()).collect::<Vec<_>>(),
        &gate.iter().map(|h| h.to_f32()).collect::<Vec<_>>(),
    );
    let err = turbospark_compute::max_abs_diff(&got, &want);
    assert!(err < 1e-2, "err = {err}");
}

#[test]
fn sigmoid_scalar_mul_applies_one_gate_to_every_element() {
    let mut context = MetalContext::new().expect("Metal device");
    let n = 128usize;
    let y = test_vec(n, 0.23);
    let gate = vec![f16::from_f32(-0.75)];

    let y_buf = context.new_buffer_with_data(&to_le(&y));
    let gate_buf = context.new_buffer_with_data(&to_le(&gate));
    let pass = context.begin_pass();
    turbospark_gpu::encode_sigmoid_scalar_mul(
        &mut context,
        &pass,
        (&y_buf, 0),
        (&gate_buf, 0),
        n as u32,
    )
    .expect("dispatch");
    pass.commit_and_wait();

    let got: Vec<f32> = read_halfs(&y_buf, n).iter().map(|h| h.to_f32()).collect();
    let want = turbospark_compute::sigmoid_scalar_mul(
        &y.iter().map(|h| h.to_f32()).collect::<Vec<_>>(),
        gate[0].to_f32(),
    );
    let err = turbospark_compute::max_abs_diff(&got, &want);
    assert!(err < 1e-2, "err = {err}");
}

#[test]
fn split_q_gate_deinterleaves_per_head_pairs() {
    let mut context = MetalContext::new().expect("Metal device");
    let (heads, dim) = (4usize, 32usize);
    let packed = test_vec(heads * 2 * dim, 0.11);

    let packed_buf = context.new_buffer_with_data(&to_le(&packed));
    let q_buf = context.new_output_buffer((heads * dim * 2) as u64);
    let gate_buf = context.new_output_buffer((heads * dim * 2) as u64);
    let pass = context.begin_pass();
    turbospark_gpu::encode_split_q_gate(
        &mut context,
        &pass,
        (&packed_buf, 0),
        (&q_buf, 0),
        (&gate_buf, 0),
        heads as u32,
        dim as u32,
    )
    .expect("dispatch");
    pass.commit_and_wait();

    // A pure permutation: assert the exact bits, not a tolerance.
    let (want_q, want_gate) = turbospark_compute::split_q_gate(
        &packed.iter().map(|h| h.to_f32()).collect::<Vec<_>>(),
        heads,
        dim,
    );
    let got_q: Vec<f32> = read_halfs(&q_buf, heads * dim)
        .iter()
        .map(|h| h.to_f32())
        .collect();
    let got_gate: Vec<f32> = read_halfs(&gate_buf, heads * dim)
        .iter()
        .map(|h| h.to_f32())
        .collect();
    assert_eq!(got_q, want_q);
    assert_eq!(got_gate, want_gate);
}

// ===========================================================================
// steer_direction_fp16 — the directional-steering edit (ROADMAP item 9, in
// its runtime form). Contract: `turbospark_compute::steering`.
//
// Deliberately NOT a multiple of the 256-thread threadgroup: the kernel's
// strided read loop has a remainder only at a length like this, and a fixture
// of exactly 256 or 512 would exercise the tidy case alone.
// ===========================================================================

const STEER_D: usize = 320;

/// A direction that is neither unit-norm nor orthogonal to [`steer_row`].
///
/// Non-unit is load-bearing: `inv_norm` cancels out of every mode when
/// `||d|| == 1`, so a fixture built from a normalized direction passes
/// against a kernel that ignores the parameter entirely.
fn steer_direction() -> Vec<f16> {
    (0..STEER_D)
        .map(|i| f16::from_f32(((i as f32) * 0.37).sin() * 1.5))
        .collect()
}

fn steer_row() -> Vec<f16> {
    (0..STEER_D)
        .map(|i| f16::from_f32(((i as f32) * 0.11).cos() * 3.0))
        .collect()
}

fn as_f32(v: &[f16]) -> Vec<f32> {
    v.iter().map(|h| h.to_f32()).collect()
}

/// A row carrying a LARGE component along the direction, which [`steer_row`]
/// does not.
///
/// It exists because the equivalent CPU fixture was MEASURED unable to do the
/// job: it reads a rescale factor of 1.006, so `Renorm` and `Ablate` agree on
/// it to within 0.6% and comparing them there proves almost nothing. Not an
/// inflated fixture either -- on the real `qwen38-27b` the extracted direction
/// is 19-28% of the residual stream at the late layers
/// (`docs/OBLITERATION.md`), which is the regime the mode exists for.
fn steer_row_carrying() -> Vec<f16> {
    let d = steer_direction();
    steer_row()
        .iter()
        .zip(d.iter())
        .map(|(x, dv)| f16::from_f32(x.to_f32() + 4.0 * dv.to_f32()))
        .collect()
}

/// Runs one dispatch over a single row and returns `(edited row, coefficient)`.
fn steer_once(
    mode: foundation::SteeringMode,
    alpha: f32,
    target: f32,
    gate: f32,
) -> (Vec<f16>, f32) {
    steer_once_on(&steer_row(), mode, alpha, target, gate)
}

/// [`steer_once`] over a caller-supplied row, which the `Renorm` cases need:
/// that mode's whole effect scales with how much of the direction the row
/// carries, so the default fixture cannot exercise it.
fn steer_once_on(
    x: &[f16],
    mode: foundation::SteeringMode,
    alpha: f32,
    target: f32,
    gate: f32,
) -> (Vec<f16>, f32) {
    let mut context = MetalContext::new().expect("Metal device");
    let d = steer_direction();

    let x_buf = context.new_buffer_with_data(&to_le(x));
    let d_buf = context.new_buffer_with_data(&to_le(&d));
    let coeff_buf = context.new_output_buffer(4);

    let params = turbospark_gpu::SteerParams {
        d_len: STEER_D as u32,
        rows: 1,
        row_stride: STEER_D as u32,
        mode,
        alpha,
        inv_norm: turbospark_compute::inv_norm(&as_f32(&d)),
        target,
        gate_threshold: gate,
    };

    let pass = context.begin_pass();
    turbospark_gpu::encode_steer_direction(
        &mut context,
        &pass,
        (&x_buf, 0),
        (&d_buf, 0),
        (&coeff_buf, 0),
        &params,
    )
    .expect("encode");
    pass.commit_and_wait();

    (
        read_halfs(&x_buf, STEER_D),
        turbospark_gpu::read_f32_buffer(&coeff_buf, 1)[0],
    )
}

/// The CPU reference over the same fixture, rounded to FP16 at the one place
/// the kernel stores.
fn steer_reference(mode: foundation::SteeringMode, alpha: f32, target: f32, gate: f32) -> Vec<f16> {
    steer_reference_on(&steer_row(), mode, alpha, target, gate)
}

/// [`steer_reference`] over a caller-supplied row, paired with
/// [`steer_once_on`].
fn steer_reference_on(
    row: &[f16],
    mode: foundation::SteeringMode,
    alpha: f32,
    target: f32,
    gate: f32,
) -> Vec<f16> {
    let d = as_f32(&steer_direction());
    let mut x = as_f32(row);
    turbospark_compute::steer_in_place(
        &mut x,
        &d,
        mode,
        alpha,
        turbospark_compute::inv_norm(&d),
        target,
        gate,
    );
    x.iter().map(|v| f16::from_f32(*v)).collect()
}

/// The GPU reduces the dot product in a different ORDER from the reference
/// (a strided read into a two-stage `simd_sum`, against a sequential sum), and
/// FP addition is not associative, so this compares within a tolerance rather
/// than bitwise. The tolerance is one FP16 quantum at the fixture's magnitude:
/// values run to ~3, where FP16 resolves 2^-9 ≈ 0.002.
fn assert_steer_close(got: &[f16], want: &[f16], label: &str) {
    for i in 0..STEER_D {
        let diff = (got[i].to_f32() - want[i].to_f32()).abs();
        assert!(
            diff <= 0.004,
            "{label}: i={i} got {} want {} (diff {diff})",
            got[i].to_f32(),
            want[i].to_f32()
        );
    }
}

/// The wire values the shader's `kSteerMode*` constants switch on.
///
/// This pins the RUST side against an accidental reorder of the enum. The
/// shader side is pinned by the three parity cases below: if the codes
/// disagreed across the boundary, asking for `Ablate` would run `Add`, and
/// `steer_ablate_matches_compute_reference` would fail against a result that
/// is perfectly finite and perfectly wrong.
#[test]
fn steering_mode_codes_are_pinned() {
    assert_eq!(foundation::SteeringMode::Ablate.as_u32(), 0);
    assert_eq!(foundation::SteeringMode::Add.as_u32(), 1);
    assert_eq!(foundation::SteeringMode::Clamp.as_u32(), 2);
    assert_eq!(foundation::SteeringMode::Renorm.as_u32(), 3);
}

/// **Assert the fixture discriminates before believing any parity case.**
///
/// At a small `alpha`, at a direction near-orthogonal to the row, or at a
/// `target` near the row's own coefficient, the three modes converge and a
/// tidy fixture would pass against a kernel that implements only one of them
/// — the same trap `the_two_norm_conventions_are_different_functions` exists
/// for one file over, and AGENTS.md Gotchas 48 and 50 state generally.
///
/// `Renorm` is deliberately NOT in this test and has its own
/// (`renorm_is_not_ablate_on_the_gpu`). It cannot use this fixture: its whole
/// effect scales with the direction's share of the row, and on `steer_row`
/// that share puts its rescale factor at 1.006 -- so it would be compared
/// against `ablate` on a row where the two genuinely almost agree, and the
/// 0.05 bar below would fail for a correct kernel.
#[test]
fn the_original_three_modes_are_different_functions() {
    let (ablate, _) = steer_once(foundation::SteeringMode::Ablate, 1.0, 4.0, 0.0);
    let (add, _) = steer_once(foundation::SteeringMode::Add, 1.0, 4.0, 0.0);
    let (clamp, _) = steer_once(foundation::SteeringMode::Clamp, 1.0, 4.0, 0.0);
    let base = steer_row();

    for (a, b, label) in [
        (&ablate, &add, "ablate vs add"),
        (&ablate, &clamp, "ablate vs clamp"),
        (&add, &clamp, "add vs clamp"),
    ] {
        let worst = (0..STEER_D)
            .map(|i| (a[i].to_f32() - b[i].to_f32()).abs())
            .fold(0.0f32, f32::max);
        assert!(worst > 0.05, "{label} agree to {worst}; fixture is blind");
    }

    // And each must actually move the row, or "different from each other"
    // could still mean two of them are no-ops.
    for (edited, label) in [(&ablate, "ablate"), (&add, "add"), (&clamp, "clamp")] {
        let worst = (0..STEER_D)
            .map(|i| (edited[i].to_f32() - base[i].to_f32()).abs())
            .fold(0.0f32, f32::max);
        assert!(worst > 0.05, "{label} left the row unmoved");
    }
}

/// Run at `alpha = 0.6` and a NON-ZERO `target`, and both are deliberate.
///
/// At `alpha = 1.0` with `target = 0.0` this case cannot see the difference
/// between ablate and clamp, because at those parameters they are the same
/// function (see `full_ablation_and_a_zero_clamp_are_the_same_edit`). A
/// mutation swapping the two mode codes in the shader passed this case at
/// those parameters. A fractional alpha separates them, and a target that
/// ablate must IGNORE pins that it reads no such parameter.
#[test]
fn steer_ablate_matches_compute_reference() {
    let (got, _) = steer_once(foundation::SteeringMode::Ablate, 0.6, 4.0, 0.0);
    let want = steer_reference(foundation::SteeringMode::Ablate, 0.6, 4.0, 0.0);
    assert_steer_close(&got, &want, "ablate");
}

/// Ablation at `alpha = 1` and a clamp to `target = 0` are the SAME edit:
/// both drive the coefficient to zero, and both reduce to a scale of
/// `-c * inv_norm^2`. Stated as a test rather than left as a trap, because
/// it is the reason the case above cannot be run at those parameters, and
/// because it is a genuine cross-check: two independently written branches
/// of the kernel must agree where the algebra says they must.
#[test]
fn full_ablation_and_a_zero_clamp_are_the_same_edit() {
    let (ablate, _) = steer_once(foundation::SteeringMode::Ablate, 1.0, 0.0, 0.0);
    let (clamp, _) = steer_once(foundation::SteeringMode::Clamp, 1.0, 0.0, 0.0);
    for i in 0..STEER_D {
        assert_eq!(
            ablate[i].to_bits(),
            clamp[i].to_bits(),
            "the two branches disagree at {i}"
        );
    }
}

#[test]
fn steer_add_matches_compute_reference() {
    let (got, _) = steer_once(foundation::SteeringMode::Add, 0.75, 0.0, 0.0);
    let want = steer_reference(foundation::SteeringMode::Add, 0.75, 0.0, 0.0);
    assert_steer_close(&got, &want, "add");
}

#[test]
fn steer_clamp_matches_compute_reference() {
    let (got, _) = steer_once(foundation::SteeringMode::Clamp, 1.0, 4.0, 0.0);
    let want = steer_reference(foundation::SteeringMode::Clamp, 1.0, 4.0, 0.0);
    assert_steer_close(&got, &want, "clamp");
}

/// Run at `alpha = 0.6` for the reason `steer_ablate_matches_compute_
/// reference` runs there and one more besides: `renorm`'s rescale
/// coefficient is `alpha * (2 - alpha)`, which is EQUAL to `alpha` at exactly
/// 1.0. A case at full strength alone is blind to the difference between the
/// correct expression and a plain `alpha`, and full strength is the natural
/// value to reach for, since making it usable is the mode's whole purpose.
///
/// On `steer_row_carrying` rather than the default row, because the whole
/// effect scales with the direction's share and the default row's is 0.6%.
#[test]
fn steer_renorm_matches_compute_reference() {
    let row = steer_row_carrying();
    let (got, _) = steer_once_on(&row, foundation::SteeringMode::Renorm, 0.6, 4.0, 0.0);
    let want = steer_reference_on(&row, foundation::SteeringMode::Renorm, 0.6, 4.0, 0.0);
    assert_steer_close(&got, &want, "renorm");
}

/// `renorm`'s defining property, asserted on the GPU rather than inferred
/// from the reference agreeing with it: the edited row must carry the norm
/// the original did.
///
/// Paired with `steer_renorm_still_ablates` below, and neither alone pins the
/// edit -- a no-op preserves the norm, and plain `ablate` removes the
/// component. The tolerance is loose because the row is stored FP16 and the
/// norm is a sum over 320 such values.
#[test]
fn steer_renorm_preserves_the_row_norm() {
    fn l2(v: &[f16]) -> f32 {
        v.iter()
            .map(|a| a.to_f32() * a.to_f32())
            .sum::<f32>()
            .sqrt()
    }
    let row = steer_row_carrying();
    let before = l2(&row);
    for alpha in [0.6f32, 1.0] {
        let (got, _) = steer_once_on(&row, foundation::SteeringMode::Renorm, alpha, 0.0, 0.0);
        let after = l2(&got);
        assert!(
            (after - before).abs() / before < 0.01,
            "alpha {alpha}: norm moved {before} -> {after}"
        );
    }
}

/// The other half: it must still project the direction OUT. Preserving the
/// norm while leaving the component alone is the identity.
#[test]
fn steer_renorm_still_ablates() {
    let row = steer_row_carrying();
    let d = as_f32(&steer_direction());
    let inv = turbospark_compute::inv_norm(&d);
    let before = turbospark_compute::unit_coefficient(&as_f32(&row), &d, inv);
    assert!(before.abs() > 1.0, "fixture carries no direction to remove");

    let (got, _) = steer_once_on(&row, foundation::SteeringMode::Renorm, 1.0, 0.0, 0.0);
    let after = turbospark_compute::unit_coefficient(&as_f32(&got), &d, inv);
    assert!(
        after.abs() < before.abs() * 0.02,
        "component survived renorm: {before} -> {after}"
    );
}

/// `renorm` and `ablate` must be different edits on the GPU too, and **the
/// fixture is checked capable of showing that first**: as the coefficient
/// goes to zero there is no norm to restore, the rescale factor goes to 1,
/// and the two converge. On the DEFAULT row they agree to 0.6%, which is why
/// this uses `steer_row_carrying` -- the same Gotcha 48 discipline
/// `the_three_modes_are_different_functions` applies above.
#[test]
fn renorm_is_not_ablate_on_the_gpu() {
    let row = steer_row_carrying();
    let xx: f32 = as_f32(&row).iter().map(|v| v * v).sum();
    let d = as_f32(&steer_direction());
    let inv = turbospark_compute::inv_norm(&d);
    let c_hat = turbospark_compute::unit_coefficient(&as_f32(&row), &d, inv);
    let gamma = turbospark_compute::renorm_gamma(xx, c_hat, 1.0);
    assert!(
        gamma > 1.01,
        "fixture cannot discriminate: gamma is {gamma}, so the two edits coincide on it"
    );

    let (ablated, _) = steer_once_on(&row, foundation::SteeringMode::Ablate, 1.0, 0.0, 0.0);
    let (renormed, _) = steer_once_on(&row, foundation::SteeringMode::Renorm, 1.0, 0.0, 0.0);
    let worst = (0..STEER_D)
        .map(|i| (ablated[i].to_f32() - renormed[i].to_f32()).abs())
        .fold(0.0f32, f32::max);
    assert!(
        worst > 0.05,
        "renorm reproduced ablate's row (worst {worst})"
    );
}

/// The reported coefficient is the pre-edit one along the UNIT direction, and
/// it must agree with the reference's. This is the number the whole
/// measurement story rests on: it says how much of the direction the stream
/// carried, which is the steered-vs-unsteered signal without a second model.
#[test]
fn the_reported_coefficient_matches_the_reference() {
    let d = as_f32(&steer_direction());
    let x = as_f32(&steer_row());
    let want = turbospark_compute::unit_coefficient(&x, &d, turbospark_compute::inv_norm(&d));
    assert!(want.abs() > 0.5, "fixture carries no direction to report");

    for mode in [
        foundation::SteeringMode::Ablate,
        foundation::SteeringMode::Add,
        foundation::SteeringMode::Clamp,
    ] {
        let (_, got) = steer_once(mode, 1.0, 4.0, 0.0);
        assert!(
            (got - want).abs() <= want.abs() * 1e-2,
            "{mode:?}: coefficient {got} against reference {want}"
        );
    }
}

/// **The null control.** With the kernel dispatched and `alpha` zero, the row
/// must come back BIT-IDENTICAL, in the two modes that scale by alpha. One
/// comparison catches a wrong reduction, a wrong buffer offset and a wrong
/// row stride, and it is the arm the real-model probe will lean on to prove
/// that steering-off costs nothing.
#[test]
fn alpha_zero_leaves_the_row_bit_identical() {
    let base = steer_row();
    for mode in [
        foundation::SteeringMode::Ablate,
        foundation::SteeringMode::Add,
    ] {
        let (got, _) = steer_once(mode, 0.0, 0.0, 0.0);
        for i in 0..STEER_D {
            assert_eq!(
                got[i].to_bits(),
                base[i].to_bits(),
                "{mode:?} at alpha=0 moved element {i}"
            );
        }
    }
}

/// The gate suppresses the edit and still reports, so a gated run stays
/// measurable rather than going dark.
#[test]
fn the_gate_blocks_the_edit_but_reports_the_coefficient() {
    let base = steer_row();
    let (got, coeff) = steer_once(foundation::SteeringMode::Ablate, 1.0, 0.0, 1.0e6);
    for i in 0..STEER_D {
        assert_eq!(
            got[i].to_bits(),
            base[i].to_bits(),
            "gated edit fired at {i}"
        );
    }
    assert!(coeff.abs() > 0.0, "gated edit reported no coefficient");
}

/// `rows` and `row_stride` must edit each row independently and touch nothing
/// between them. The gap is filled with a sentinel and asserted untouched,
/// which is what separates a correct stride from one that reads in bytes:
/// at `rows == 1` the two are indistinguishable.
#[test]
fn multi_row_edits_each_row_and_nothing_between_them() {
    let mut context = MetalContext::new().expect("Metal device");
    let rows = 3usize;
    let stride = STEER_D + 7; // a gap, and not a round number
    let d = steer_direction();
    let row = steer_row();

    let sentinel = f16::from_f32(-9.0);
    let mut flat = Vec::with_capacity(rows * stride);
    for _ in 0..rows {
        flat.extend_from_slice(&row);
        flat.extend(std::iter::repeat_n(sentinel, stride - STEER_D));
    }

    let x_buf = context.new_buffer_with_data(&to_le(&flat));
    let d_buf = context.new_buffer_with_data(&to_le(&d));
    let coeff_buf = context.new_output_buffer((rows * 4) as u64);

    let params = turbospark_gpu::SteerParams {
        d_len: STEER_D as u32,
        rows: rows as u32,
        row_stride: stride as u32,
        mode: foundation::SteeringMode::Ablate,
        // A fractional alpha for the reason `steer_ablate_matches_compute_
        // reference` uses one: at 1.0 this could not tell ablate from clamp.
        alpha: 0.6,
        inv_norm: turbospark_compute::inv_norm(&as_f32(&d)),
        target: 0.0,
        gate_threshold: 0.0,
    };

    let pass = context.begin_pass();
    turbospark_gpu::encode_steer_direction(
        &mut context,
        &pass,
        (&x_buf, 0),
        (&d_buf, 0),
        (&coeff_buf, 0),
        &params,
    )
    .expect("encode");
    pass.commit_and_wait();

    let got = read_halfs(&x_buf, rows * stride);
    let want = steer_reference(foundation::SteeringMode::Ablate, 0.6, 0.0, 0.0);
    for r in 0..rows {
        let base = r * stride;
        assert_steer_close(&got[base..base + STEER_D], &want, &format!("row {r}"));
        for g in STEER_D..stride {
            assert_eq!(
                got[base + g].to_bits(),
                sentinel.to_bits(),
                "row {r} spilled into the gap at {g}"
            );
        }
    }

    // Identical rows must report identical coefficients.
    let coeffs = turbospark_gpu::read_f32_buffer(&coeff_buf, rows);
    for r in 1..rows {
        assert_eq!(
            coeffs[r], coeffs[0],
            "row {r} reported a different coefficient"
        );
    }
}
