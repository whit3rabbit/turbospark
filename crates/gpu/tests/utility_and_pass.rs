#![cfg(target_os = "macos")]
//! Parity tests for the `utility.metal` elementwise kernels (against the
//! `mrefrust_compute` reference math) and for the `PassEncoder` batching
//! path (an encoded chain must be bit-identical to the one-shot dispatch
//! wrappers it replaces).

use half::f16;
use mrefrust_gpu::MetalContext;

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
    mrefrust_gpu::encode_gelu_mul(
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
    let expected32 = mrefrust_compute::moe::gelu_tanh(&gate32);
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
    mrefrust_gpu::encode_silu_mul(
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
    mrefrust_gpu::encode_residual_add(
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
    let step1 = mrefrust_gpu::rms_norm_no_scale(&mut context, &x, eps).expect("one-shot 1");
    let step2 = mrefrust_gpu::rms_norm_no_scale(&mut context, &step1, eps).expect("one-shot 2");

    // Batched: both dispatches in one command buffer, chained on GPU.
    let x_buf = context.new_buffer_with_data(&to_le(&x));
    let mid_buf = context.new_output_buffer((n * 2) as u64);
    let out_buf = context.new_output_buffer((n * 2) as u64);
    let pass = context.begin_pass();
    mrefrust_gpu::encode_rms_norm_no_scale(
        &mut context,
        &pass,
        (&x_buf, 0),
        (&mid_buf, 0),
        n as u32,
        eps,
    )
    .expect("encode 1");
    mrefrust_gpu::encode_rms_norm_no_scale(
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
    mrefrust_gpu::encode_logit_softcap(&mut context, &pass, (&buf, 0), softcap, n as u32)
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

/// Qwen 3.6's three gating kernels, against `mrefrust_compute::gating`.
#[test]
fn sigmoid_gate_mul_matches_compute_reference() {
    let mut context = MetalContext::new().expect("Metal device");
    let n = 256usize;
    let out = test_vec(n, 0.17);
    let gate = test_vec(n, 0.41);

    let out_buf = context.new_buffer_with_data(&to_le(&out));
    let gate_buf = context.new_buffer_with_data(&to_le(&gate));
    let pass = context.begin_pass();
    mrefrust_gpu::encode_sigmoid_gate_mul(
        &mut context,
        &pass,
        (&out_buf, 0),
        (&gate_buf, 0),
        n as u32,
    )
    .expect("dispatch");
    pass.commit_and_wait();

    let got: Vec<f32> = read_halfs(&out_buf, n).iter().map(|h| h.to_f32()).collect();
    let want = mrefrust_compute::sigmoid_gate_mul(
        &out.iter().map(|h| h.to_f32()).collect::<Vec<_>>(),
        &gate.iter().map(|h| h.to_f32()).collect::<Vec<_>>(),
    );
    let err = mrefrust_compute::max_abs_diff(&got, &want);
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
    mrefrust_gpu::encode_sigmoid_scalar_mul(
        &mut context,
        &pass,
        (&y_buf, 0),
        (&gate_buf, 0),
        n as u32,
    )
    .expect("dispatch");
    pass.commit_and_wait();

    let got: Vec<f32> = read_halfs(&y_buf, n).iter().map(|h| h.to_f32()).collect();
    let want = mrefrust_compute::sigmoid_scalar_mul(
        &y.iter().map(|h| h.to_f32()).collect::<Vec<_>>(),
        gate[0].to_f32(),
    );
    let err = mrefrust_compute::max_abs_diff(&got, &want);
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
    mrefrust_gpu::encode_split_q_gate(
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
    let (want_q, want_gate) = mrefrust_compute::split_q_gate(
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
