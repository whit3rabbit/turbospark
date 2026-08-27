//! The `qwen3_5` vision tower's Metal kernels against
//! `turbospark_compute::vision`, on real hardware (ROADMAP M-V2).
//!
//! These kernels are PORT-LOCAL -- the Swift engine has no vision tower -- so
//! the CPU reference is the only definition of what they compute and this file
//! is the only thing holding them to it.
//!
//! Every parity case is paired with a DISCRIMINATION case asserting that its
//! fixture can tell the kernel from the plausible wrong kernel. The tower has
//! three places where two functions nearly coincide (the two GELUs, LayerNorm
//! against RMSNorm, bidirectional against causal attention), and in all three
//! a tidy fixture passes against either.
#![cfg(target_os = "macos")]

use half::f16;
use turbospark_compute::vision::{
    attention_scale, bidirectional_attention, gelu_erf, gelu_tanh_vision, layer_norm, matmul_bias,
    rope_vision_2d,
};
use turbospark_gpu::{
    encode_vision_attention, encode_vision_gelu, encode_vision_layer_norm, encode_vision_matmul,
    encode_vision_residual_add, encode_vision_rope_2d, GeluKind, MetalContext,
    MAX_ATTENTION_HEAD_DIM,
};

// ------------------------------------------------------------------ helpers

fn to_le(values: &[f16]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|v| v.to_bits().to_le_bytes())
        .collect()
}

fn read_f16(buffer: &metal::Buffer, len: usize) -> Vec<f32> {
    let ptr = buffer.contents() as *const u16;
    // SAFETY-equivalent note for a test: the buffer was created with exactly
    // `len * 2` bytes by `new_output_buffer` and the pass has been waited on.
    let bits = unsafe { std::slice::from_raw_parts(ptr, len) };
    bits.iter().map(|&b| f16::from_bits(b).to_f32()).collect()
}

fn half_of(values: &[f32]) -> Vec<f16> {
    values.iter().map(|&v| f16::from_f32(v)).collect()
}

/// A deterministic, non-flat, non-degenerate spread.
fn wave(n: usize, phase: f32, amp: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32) * 0.37 + phase).sin() * amp)
        .collect()
}

/// The largest absolute difference, and the index it occurred at.
fn worst(a: &[f32], b: &[f32]) -> (f32, usize) {
    let mut best = (0.0f32, 0usize);
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        let d = (x - y).abs();
        if d > best.0 {
            best = (d, i);
        }
    }
    best
}

/// FP16 storage carries about 3 decimal digits, so a value of order 1 rounds
/// at ~5e-4 on the way in and again on the way out. Every bound here is
/// stated against the magnitude actually present rather than as a bare
/// absolute, because these kernels run on activations reaching the hundreds.
fn fp16_bound(values: &[f32]) -> f32 {
    let scale = values.iter().fold(1.0f32, |m, v| m.max(v.abs()));
    2e-3 * scale
}

// ---------------------------------------------------------------- LayerNorm

#[test]
fn layer_norm_matches_the_cpu_reference() {
    let mut context = MetalContext::new().expect("Metal device");
    let (rows, d) = (3usize, 256usize);
    let eps = 1e-6f32;

    let x: Vec<f32> = wave(rows * d, 0.0, 6.0);
    let weight: Vec<f32> = wave(d, 1.7, 1.5);
    let bias: Vec<f32> = wave(d, 2.9, 0.5);

    let x_buf = context.new_buffer_with_data(&to_le(&half_of(&x)));
    let w_buf = context.new_buffer_with_data(&to_le(&half_of(&weight)));
    let b_buf = context.new_buffer_with_data(&to_le(&half_of(&bias)));
    let out_buf = context.new_output_buffer((rows * d * 2) as u64);

    let pass = context.begin_pass();
    encode_vision_layer_norm(
        &mut context,
        &pass,
        (&x_buf, 0),
        (&w_buf, 0),
        (&b_buf, 0),
        (&out_buf, 0),
        rows as u32,
        d as u32,
        eps,
    )
    .expect("encode");
    pass.commit_and_wait();
    let gpu = read_f16(&out_buf, rows * d);

    // The reference runs on the FP16-ROUNDED inputs, not the f32 originals:
    // the kernel never sees the originals, so comparing against them would
    // fold the storage rounding into the parity bound and hide a real error
    // of the same size.
    let xr: Vec<f32> = half_of(&x).iter().map(|v| v.to_f32()).collect();
    let wr: Vec<f32> = half_of(&weight).iter().map(|v| v.to_f32()).collect();
    let br: Vec<f32> = half_of(&bias).iter().map(|v| v.to_f32()).collect();
    let mut cpu = Vec::with_capacity(rows * d);
    for r in 0..rows {
        cpu.extend(layer_norm(&xr[r * d..(r + 1) * d], &wr, &br, eps));
    }

    let (err, at) = worst(&gpu, &cpu);
    assert!(
        err <= fp16_bound(&cpu),
        "worst {err} at {at}: gpu {} cpu {}",
        gpu[at],
        cpu[at]
    );
}

#[test]
fn layer_norm_puts_the_eps_inside_the_square_root() {
    // COVERS WHAT THE PARITY CASE ABOVE CANNOT. At the shipped `eps` of 1e-6
    // against a variance of order one, `sqrt(var + eps)` and `sqrt(var) + eps`
    // differ by less than FP16 can represent, so the parity case passes
    // against either -- measured, not assumed: that mutation survived it.
    // `eps` is a runtime argument, so a large one is a legitimate input and
    // is the only way to make the formula's shape observable.
    let mut context = MetalContext::new().expect("Metal device");
    let d = 64usize;
    // mean 0, variance 1: the divisor is sqrt(2) one way and 2 the other.
    let x: Vec<f32> = (0..d)
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    let weight = vec![1.0f32; d];
    let bias = vec![0.0f32; d];

    let x_buf = context.new_buffer_with_data(&to_le(&half_of(&x)));
    let w_buf = context.new_buffer_with_data(&to_le(&half_of(&weight)));
    let b_buf = context.new_buffer_with_data(&to_le(&half_of(&bias)));
    let out_buf = context.new_output_buffer((d * 2) as u64);
    let pass = context.begin_pass();
    encode_vision_layer_norm(
        &mut context,
        &pass,
        (&x_buf, 0),
        (&w_buf, 0),
        (&b_buf, 0),
        (&out_buf, 0),
        1,
        d as u32,
        1.0,
    )
    .expect("encode");
    pass.commit_and_wait();
    let gpu = read_f16(&out_buf, d);

    // Inside:  1 / sqrt(1 + 1) = 0.70710678
    // Outside: 1 / (sqrt(1) + 1) = 0.5
    assert!(
        (gpu[0] - std::f32::consts::FRAC_1_SQRT_2).abs() < 2e-3,
        "got {}: eps looks like it is outside the square root",
        gpu[0]
    );
}

#[test]
fn the_layer_norm_fixture_can_tell_it_from_rms_norm() {
    // GUARDS THE CASE ABOVE. On a zero-mean row LayerNorm and RMSNorm agree,
    // and a symmetric fixture is the natural one to reach for. This asserts
    // the fixture's rows are mean-shifted enough that the two are far apart,
    // so a kernel that skipped the mean subtraction could not pass.
    let d = 256usize;
    let x: Vec<f32> = wave(d, 0.0, 6.0).iter().map(|v| v + 4.0).collect();
    let weight = vec![1.0f32; d];
    let bias = vec![0.0f32; d];
    let ln = layer_norm(&x, &weight, &bias, 1e-6);
    let rms = turbospark_compute::rms_norm(&x, &weight, 1e-6);
    let (gap, _) = worst(&ln, &rms);
    assert!(
        gap > 0.5,
        "the two norms differ by only {gap} on this fixture"
    );
}

// --------------------------------------------------------------------- GELU

#[test]
fn both_gelus_match_their_cpu_references() {
    let mut context = MetalContext::new().expect("Metal device");
    let n = 512usize;
    // Spanning the range where the two forms differ most (|x| near 2) as well
    // as both saturating tails.
    let x: Vec<f32> = (0..n).map(|i| (i as f32 - 256.0) * 0.03).collect();

    for (kind, reference) in [
        (GeluKind::Tanh, gelu_tanh_vision as fn(&[f32]) -> Vec<f32>),
        (GeluKind::Erf, gelu_erf as fn(&[f32]) -> Vec<f32>),
    ] {
        let buf = context.new_buffer_with_data(&to_le(&half_of(&x)));
        let pass = context.begin_pass();
        encode_vision_gelu(&mut context, &pass, (&buf, 0), n as u32, kind).expect("encode");
        pass.commit_and_wait();
        let gpu = read_f16(&buf, n);

        let xr: Vec<f32> = half_of(&x).iter().map(|v| v.to_f32()).collect();
        let cpu = reference(&xr);
        let (err, at) = worst(&gpu, &cpu);
        assert!(
            err <= fp16_bound(&cpu),
            "{kind:?}: worst {err} at x={}: gpu {} cpu {}",
            xr[at],
            gpu[at],
            cpu[at]
        );
    }
}

#[test]
fn the_two_gelu_kernels_are_different_pipelines() {
    // THE TRAP THIS GUARDS is `crates/gpu` Gotcha 1: if the two shared one
    // kernel with a mode uniform whose byte missed the pipeline cache's key,
    // the second dispatch in a process would silently reuse the first's
    // pipeline. They are separate kernel NAMES, so that cannot happen -- but
    // asserting the outputs differ is what proves the selection reaches the
    // GPU at all, rather than both arms running the same shader.
    let mut context = MetalContext::new().expect("Metal device");
    let n = 128usize;
    // |x| near 2 is where the two forms are furthest apart.
    let x: Vec<f32> = (0..n).map(|i| 1.5 + (i as f32) * 0.01).collect();

    let mut outputs = Vec::new();
    for kind in [GeluKind::Tanh, GeluKind::Erf] {
        let buf = context.new_buffer_with_data(&to_le(&half_of(&x)));
        let pass = context.begin_pass();
        encode_vision_gelu(&mut context, &pass, (&buf, 0), n as u32, kind).expect("encode");
        pass.commit_and_wait();
        outputs.push(read_f16(&buf, n));
    }
    let (gap, _) = worst(&outputs[0], &outputs[1]);
    assert!(
        gap > 0.0,
        "the two GELU kernels produced identical output; the selection is not reaching the GPU"
    );
}

// --------------------------------------------------------------------- RoPE

#[test]
fn rope_2d_matches_the_cpu_reference() {
    let mut context = MetalContext::new().expect("Metal device");
    // The tower's real head geometry.
    let (seq, heads, head_dim) = (5usize, 3usize, 72usize);
    let half = head_dim / 2;

    let qkv: Vec<f32> = wave(seq * heads * head_dim, 0.4, 3.0);
    let freqs: Vec<f32> = (0..seq * half).map(|i| (i as f32) * 0.017 - 1.0).collect();

    let qkv_buf = context.new_buffer_with_data(&to_le(&half_of(&qkv)));
    let freq_buf = context.new_buffer_with_data(&to_le(&half_of(&freqs)));

    let pass = context.begin_pass();
    encode_vision_rope_2d(
        &mut context,
        &pass,
        (&qkv_buf, 0),
        (&freq_buf, 0),
        seq as u32,
        heads as u32,
        head_dim as u32,
    )
    .expect("encode");
    pass.commit_and_wait();
    let gpu = read_f16(&qkv_buf, seq * heads * head_dim);

    let qr: Vec<f32> = half_of(&qkv).iter().map(|v| v.to_f32()).collect();
    let fr: Vec<f32> = half_of(&freqs).iter().map(|v| v.to_f32()).collect();
    let mut cpu = vec![0.0f32; qr.len()];
    for token in 0..seq {
        let row = &fr[token * half..(token + 1) * half];
        for head in 0..heads {
            let base = (token * heads + head) * head_dim;
            let rotated = rope_vision_2d(&qr[base..base + head_dim], row);
            cpu[base..base + head_dim].copy_from_slice(&rotated);
        }
    }

    let (err, at) = worst(&gpu, &cpu);
    assert!(
        err <= fp16_bound(&cpu),
        "worst {err} at {at}: gpu {} cpu {}",
        gpu[at],
        cpu[at]
    );
}

#[test]
fn rope_2d_shares_one_freq_row_across_heads() {
    // The freq table is `[seq, head_dim/2]` and NOT `[seq, heads,
    // head_dim/2]`: every head of a token rotates by the same angles. A
    // kernel indexing the table per head would read past the end or, worse,
    // read a neighbouring token's row. Identical head inputs must therefore
    // come back identical.
    let mut context = MetalContext::new().expect("Metal device");
    let (seq, heads, head_dim) = (4usize, 3usize, 8usize);
    let half = head_dim / 2;

    let one_head: Vec<f32> = wave(head_dim, 0.9, 2.0);
    let mut qkv = Vec::new();
    for _ in 0..seq * heads {
        qkv.extend_from_slice(&one_head);
    }
    let freqs: Vec<f32> = (0..seq * half).map(|i| 0.3 + i as f32 * 0.21).collect();

    let qkv_buf = context.new_buffer_with_data(&to_le(&half_of(&qkv)));
    let freq_buf = context.new_buffer_with_data(&to_le(&half_of(&freqs)));
    let pass = context.begin_pass();
    encode_vision_rope_2d(
        &mut context,
        &pass,
        (&qkv_buf, 0),
        (&freq_buf, 0),
        seq as u32,
        heads as u32,
        head_dim as u32,
    )
    .expect("encode");
    pass.commit_and_wait();
    let gpu = read_f16(&qkv_buf, seq * heads * head_dim);

    for token in 0..seq {
        let first = &gpu[(token * heads) * head_dim..(token * heads + 1) * head_dim];
        for head in 1..heads {
            let other =
                &gpu[(token * heads + head) * head_dim..(token * heads + head + 1) * head_dim];
            assert_eq!(
                first, other,
                "token {token} head {head} rotated differently"
            );
        }
    }
    // And the rotation actually happened, so the check above is not comparing
    // untouched inputs.
    let untouched: Vec<f32> = half_of(&one_head).iter().map(|v| v.to_f32()).collect();
    let (moved, _) = worst(&gpu[..head_dim], &untouched);
    assert!(moved > 0.01, "the kernel left the first head unrotated");
}

#[test]
fn an_odd_head_dim_is_refused() {
    let mut context = MetalContext::new().expect("Metal device");
    let buf = context.new_buffer_with_data(&to_le(&half_of(&[0.0; 8])));
    let pass = context.begin_pass();
    let err = encode_vision_rope_2d(&mut context, &pass, (&buf, 0), (&buf, 0), 1, 1, 7);
    assert!(
        err.is_err(),
        "an odd head_dim must be refused, not truncated"
    );
    pass.commit_and_wait();
}

// ---------------------------------------------------------------- attention

fn run_attention(
    context: &mut MetalContext,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq: usize,
    heads: usize,
    head_dim: usize,
) -> (Vec<f32>, Vec<f32>) {
    let n = seq * heads * head_dim;
    let scale = attention_scale(head_dim);
    let q_buf = context.new_buffer_with_data(&to_le(&half_of(q)));
    let k_buf = context.new_buffer_with_data(&to_le(&half_of(k)));
    let v_buf = context.new_buffer_with_data(&to_le(&half_of(v)));
    let out_buf = context.new_output_buffer((n * 2) as u64);

    let pass = context.begin_pass();
    encode_vision_attention(
        context,
        &pass,
        (&q_buf, 0),
        (&k_buf, 0),
        (&v_buf, 0),
        (&out_buf, 0),
        seq as u32,
        heads as u32,
        head_dim as u32,
        scale,
    )
    .expect("encode");
    pass.commit_and_wait();
    let gpu = read_f16(&out_buf, n);

    let qr: Vec<f32> = half_of(q).iter().map(|v| v.to_f32()).collect();
    let kr: Vec<f32> = half_of(k).iter().map(|v| v.to_f32()).collect();
    let vr: Vec<f32> = half_of(v).iter().map(|v| v.to_f32()).collect();
    let cpu = bidirectional_attention(&qr, &kr, &vr, seq, heads, head_dim, scale);
    (gpu, cpu)
}

#[test]
fn attention_matches_the_cpu_reference_at_the_towers_head_geometry() {
    let mut context = MetalContext::new().expect("Metal device");
    let (seq, heads, head_dim) = (37usize, 16usize, 72usize);
    let n = seq * heads * head_dim;
    let q = wave(n, 0.0, 2.0);
    let k = wave(n, 1.1, 2.0);
    let v = wave(n, 2.3, 3.0);

    let (gpu, cpu) = run_attention(&mut context, &q, &k, &v, seq, heads, head_dim);
    let (err, at) = worst(&gpu, &cpu);
    assert!(
        err <= fp16_bound(&cpu),
        "worst {err} at {at}: gpu {} cpu {}",
        gpu[at],
        cpu[at]
    );
}

#[test]
fn attention_is_correct_when_the_key_count_is_under_the_simd_group_count() {
    // The threadgroup splits keys across 8 SIMD groups, so at `seq < 8` some
    // groups process NO keys and carry a running max of -infinity. Merging
    // those naively computes `exp(-inf - -inf)`, which is NaN, and NaN reads
    // as a perfect score on any rank instrument (AGENTS.md Gotcha 59). Every
    // short sequence is checked, not just one.
    let mut context = MetalContext::new().expect("Metal device");
    for seq in 1usize..=9 {
        let (heads, head_dim) = (2usize, 72usize);
        let n = seq * heads * head_dim;
        let q = wave(n, 0.5, 2.0);
        let k = wave(n, 1.5, 2.0);
        let v = wave(n, 2.5, 2.0);
        let (gpu, cpu) = run_attention(&mut context, &q, &k, &v, seq, heads, head_dim);
        assert!(
            gpu.iter().all(|v| v.is_finite()),
            "seq {seq}: non-finite output"
        );
        let (err, at) = worst(&gpu, &cpu);
        assert!(
            err <= fp16_bound(&cpu),
            "seq {seq}: worst {err} at {at}: gpu {} cpu {}",
            gpu[at],
            cpu[at]
        );
    }
}

#[test]
fn attention_is_bidirectional_and_the_fixture_can_tell() {
    // The property that separates this kernel from `attention_decode`, on a
    // fixture built so a causal mask would be visible: position 0's query
    // aligns with the LAST key alone, so under a mask it would return its own
    // value instead.
    let mut context = MetalContext::new().expect("Metal device");
    let (seq, heads, head_dim) = (6usize, 1usize, 8usize);
    let n = seq * heads * head_dim;
    let mut q = vec![0.0f32; n];
    let mut k = vec![0.0f32; n];
    let mut v = vec![0.0f32; n];
    for j in 0..seq {
        k[j * head_dim] = if j == seq - 1 { 12.0 } else { -12.0 };
        v[j * head_dim] = j as f32;
    }
    q[0] = 1.0;

    let (gpu, cpu) = run_attention(&mut context, &q, &k, &v, seq, heads, head_dim);
    assert!(
        (gpu[0] - (seq - 1) as f32).abs() < 0.05,
        "position 0 did not reach the last value: got {}",
        gpu[0]
    );
    // A causal kernel would return position 0's own value here, so this
    // fixture genuinely discriminates.
    assert!(
        (cpu[0] - v[0]).abs() > 1.0,
        "the fixture cannot tell a causal kernel from a bidirectional one"
    );
}

#[test]
fn attention_survives_the_towers_real_activation_scale() {
    // `docs/VISION_PHASE0.md` item 3 measures absmax 9,024 at block 26 on the
    // largest legal page. The running max is what keeps `exp` from
    // overflowing there, and it is load-bearing on real data.
    let mut context = MetalContext::new().expect("Metal device");
    let (seq, heads, head_dim) = (12usize, 2usize, 72usize);
    let n = seq * heads * head_dim;
    let q: Vec<f32> = (0..n).map(|i| 200.0 + (i % 17) as f32).collect();
    let k: Vec<f32> = (0..n).map(|i| 200.0 - (i % 13) as f32).collect();
    let v = wave(n, 0.7, 4.0);

    let (gpu, cpu) = run_attention(&mut context, &q, &k, &v, seq, heads, head_dim);
    assert!(
        gpu.iter().all(|v| v.is_finite()),
        "overflowed to non-finite"
    );
    let (err, at) = worst(&gpu, &cpu);
    assert!(
        err <= fp16_bound(&cpu),
        "worst {err} at {at}: gpu {} cpu {}",
        gpu[at],
        cpu[at]
    );
}

#[test]
fn an_oversized_head_dim_is_refused_rather_than_truncated() {
    let mut context = MetalContext::new().expect("Metal device");
    let buf = context.new_buffer_with_data(&to_le(&half_of(&[0.0; 8])));
    let pass = context.begin_pass();
    let err = encode_vision_attention(
        &mut context,
        &pass,
        (&buf, 0),
        (&buf, 0),
        (&buf, 0),
        (&buf, 0),
        1,
        1,
        MAX_ATTENTION_HEAD_DIM + 32,
        1.0,
    );
    assert!(err.is_err(), "an oversized head_dim must be refused");
    pass.commit_and_wait();
}

// ------------------------------------------------------------------- matmul

#[test]
fn matmul_matches_the_cpu_reference_with_and_without_bias() {
    let mut context = MetalContext::new().expect("Metal device");
    let (m, k, n) = (5usize, 320usize, 48usize);
    let a = wave(m * k, 0.0, 1.5);
    let b = wave(n * k, 1.3, 0.8);
    let bias = wave(n, 2.1, 2.0);

    for with_bias in [false, true] {
        let a_buf = context.new_buffer_with_data(&to_le(&half_of(&a)));
        let b_buf = context.new_buffer_with_data(&to_le(&half_of(&b)));
        let bias_buf = context.new_buffer_with_data(&to_le(&half_of(&bias)));
        let out_buf = context.new_output_buffer((m * n * 2) as u64);

        let pass = context.begin_pass();
        encode_vision_matmul(
            &mut context,
            &pass,
            (&a_buf, 0),
            (&b_buf, 0),
            with_bias.then_some((&bias_buf, 0)),
            (&out_buf, 0),
            m as u32,
            k as u32,
            n as u32,
        )
        .expect("encode");
        pass.commit_and_wait();
        let gpu = read_f16(&out_buf, m * n);

        let ar: Vec<f32> = half_of(&a).iter().map(|v| v.to_f32()).collect();
        let br: Vec<f32> = half_of(&b).iter().map(|v| v.to_f32()).collect();
        let biasr: Vec<f32> = half_of(&bias).iter().map(|v| v.to_f32()).collect();
        let cpu = matmul_bias(&ar, &br, with_bias.then_some(&biasr[..]), m, k, n);

        let (err, at) = worst(&gpu, &cpu);
        assert!(
            err <= fp16_bound(&cpu),
            "with_bias={with_bias}: worst {err} at {at}: gpu {} cpu {}",
            gpu[at],
            cpu[at]
        );
    }
}

#[test]
fn matmul_reads_the_weight_row_major_by_output() {
    // Reading `b` as `[k, n]` instead of `[n, k]` transposes every projection
    // while keeping every buffer-size check happy. An asymmetric selector
    // weight makes the orientation directly visible.
    let mut context = MetalContext::new().expect("Metal device");
    let (m, k, n) = (1usize, 4usize, 2usize);
    let a = [1.0f32, 2.0, 3.0, 4.0];
    // Output 0 selects a[0]; output 1 selects 10 * a[1].
    let b = [1.0f32, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0];

    let a_buf = context.new_buffer_with_data(&to_le(&half_of(&a)));
    let b_buf = context.new_buffer_with_data(&to_le(&half_of(&b)));
    let out_buf = context.new_output_buffer((m * n * 2) as u64);
    let pass = context.begin_pass();
    encode_vision_matmul(
        &mut context,
        &pass,
        (&a_buf, 0),
        (&b_buf, 0),
        None,
        (&out_buf, 0),
        m as u32,
        k as u32,
        n as u32,
    )
    .expect("encode");
    pass.commit_and_wait();
    let gpu = read_f16(&out_buf, m * n);
    assert!((gpu[0] - 1.0).abs() < 1e-3, "{gpu:?}");
    assert!((gpu[1] - 20.0).abs() < 1e-2, "{gpu:?}");
}

#[test]
fn matmul_runs_at_every_real_projection_shape() {
    // The tower's own shapes at one row each, so a dimension swapped between
    // `k` and `n` fails here rather than at a buffer-size check in M-V4.
    let mut context = MetalContext::new().expect("Metal device");
    for (k, n, label) in [
        (1536usize, 1152usize, "patch_embed"),
        (1152, 3456, "qkv"),
        (1152, 1152, "proj"),
        (1152, 4304, "mlp.fc1"),
        (4304, 1152, "mlp.fc2"),
        (4608, 4608, "merger.fc1"),
        (4608, 5120, "merger.fc2"),
    ] {
        let a = wave(k, 0.2, 0.5);
        let b = wave(n * k, 0.9, 0.05);
        let a_buf = context.new_buffer_with_data(&to_le(&half_of(&a)));
        let b_buf = context.new_buffer_with_data(&to_le(&half_of(&b)));
        let out_buf = context.new_output_buffer((n * 2) as u64);
        let pass = context.begin_pass();
        encode_vision_matmul(
            &mut context,
            &pass,
            (&a_buf, 0),
            (&b_buf, 0),
            None,
            (&out_buf, 0),
            1,
            k as u32,
            n as u32,
        )
        .expect("encode");
        pass.commit_and_wait();
        let gpu = read_f16(&out_buf, n);

        let ar: Vec<f32> = half_of(&a).iter().map(|v| v.to_f32()).collect();
        let br: Vec<f32> = half_of(&b).iter().map(|v| v.to_f32()).collect();
        let cpu = matmul_bias(&ar, &br, None, 1, k, n);
        let (err, at) = worst(&gpu, &cpu);
        assert!(
            err <= fp16_bound(&cpu),
            "{label}: worst {err} at {at}: gpu {} cpu {}",
            gpu[at],
            cpu[at]
        );
    }
}

// ------------------------------------------------------------------ residual

#[test]
fn residual_add_is_elementwise() {
    let mut context = MetalContext::new().expect("Metal device");
    let n = 300usize;
    let y = wave(n, 0.0, 3.0);
    let x = wave(n, 1.9, 2.0);

    let y_buf = context.new_buffer_with_data(&to_le(&half_of(&y)));
    let x_buf = context.new_buffer_with_data(&to_le(&half_of(&x)));
    let pass = context.begin_pass();
    encode_vision_residual_add(&mut context, &pass, (&y_buf, 0), (&x_buf, 0), n as u32)
        .expect("encode");
    pass.commit_and_wait();
    let gpu = read_f16(&y_buf, n);

    let yr: Vec<f32> = half_of(&y).iter().map(|v| v.to_f32()).collect();
    let xr: Vec<f32> = half_of(&x).iter().map(|v| v.to_f32()).collect();
    let cpu: Vec<f32> = yr.iter().zip(&xr).map(|(a, b)| a + b).collect();
    let (err, at) = worst(&gpu, &cpu);
    assert!(err <= fp16_bound(&cpu), "worst {err} at {at}");
}
