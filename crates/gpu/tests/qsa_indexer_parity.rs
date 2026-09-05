#![cfg(target_os = "macos")]
//! Parity tests for `qsa_pool_blocks_mean_fp16` and `qsa_score_blocks_fp16`
//! (`qwen4_exp`'s QSA block indexer: pooling and scoring,
//! `docs/QWEN4_PHASE0.md` section 5) against
//! `turbospark_compute::qsa_indexer::{pool_blocks_mean, score_blocks}`.
//!
//! Groundwork, not yet dispatched from any decode flow -- see
//! `crates/compute/src/qsa_indexer.rs`'s module doc for the resolved
//! norm/RoPE convention these kernels compose with, and
//! `families/qwen4/mod.rs` for the current "no indexer code in this port"
//! scope statement these kernels are the first step past.

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

fn run_pool(
    context: &mut MetalContext,
    keys: &[f16],
    compress_ratio: u32,
    head_dim: u32,
) -> Vec<f32> {
    let visible = keys.len() as u32 / head_dim;
    let num_blocks = visible / compress_ratio;
    let keys_buf = context.new_buffer_with_data(&to_le(keys));
    let pooled_buf = context.new_output_buffer((num_blocks * head_dim) as u64 * 2);

    let pass = context.begin_pass();
    turbospark_gpu::encode_qsa_pool_blocks_mean(
        context,
        &pass,
        (&keys_buf, 0),
        (&pooled_buf, 0),
        compress_ratio,
        head_dim,
        num_blocks,
    )
    .expect("encode");
    pass.commit_and_wait();
    read_halfs(&pooled_buf, (num_blocks * head_dim) as usize)
}

/// The pooling kernel against the CPU reference, with every row DISTINCT so
/// a wrong block stride or a dropped row reads a wrong number rather than a
/// coincidentally-plausible one, and a RAGGED TAIL present that the kernel
/// must never read (only `num_blocks` complete blocks are dispatched).
#[test]
fn pool_blocks_mean_matches_cpu_reference() {
    let mut context = MetalContext::new().expect("Metal device");
    let head_dim = 16u32;
    let compress_ratio = 4u32;
    let num_blocks = 3u32;
    // 3 complete blocks (12 rows) plus a 2-row ragged tail the kernel must
    // ignore -- if it read past its own dispatch it would corrupt block 2.
    let visible = num_blocks * compress_ratio + 2;

    let keys16: Vec<f16> = (0..visible * head_dim)
        .map(|i| {
            let row = (i / head_dim) as f32;
            let d = (i % head_dim) as f32;
            f16::from_f32((row * 0.31 + d * 0.07).sin() * (1.0 + row * 0.1))
        })
        .collect();
    let keys32: Vec<f32> = keys16.iter().map(|v| v.to_f32()).collect();
    // CPU reference only ever sees the complete-block prefix, matching what
    // the kernel is told to pool (num_blocks derived from `visible`, tail
    // dropped by the caller before dispatch -- exactly what the real
    // wiring will do once it exists).
    let complete_len = (num_blocks * compress_ratio * head_dim) as usize;
    let expected = turbospark_compute::pool_blocks_mean(
        &keys32[..complete_len],
        head_dim as usize,
        compress_ratio as usize,
    );

    let got = run_pool(&mut context, &keys16, compress_ratio, head_dim);

    assert_eq!(got.len(), expected.len());
    for i in 0..got.len() {
        let diff = (got[i] - expected[i]).abs();
        assert!(
            diff <= 2e-3_f32.max(expected[i].abs() * 1e-2),
            "i={i}: got {} want {}",
            got[i],
            expected[i]
        );
    }
}

/// **THE MEAN, NOT THE SUM.** `keys` fixed at `1.0` everywhere makes a sum
/// read `compress_ratio` (4.0) where a correct mean reads `1.0` -- a
/// fixture whose expected values could plausibly be either would not catch
/// a dropped division.
#[test]
fn pool_blocks_mean_divides_by_compress_ratio_not_just_summing() {
    let mut context = MetalContext::new().expect("Metal device");
    let head_dim = 8u32;
    let compress_ratio = 4u32;
    let num_blocks = 2u32;
    let visible = num_blocks * compress_ratio;

    let keys16 = vec![f16::from_f32(1.0); (visible * head_dim) as usize];
    let got = run_pool(&mut context, &keys16, compress_ratio, head_dim);
    for (i, &v) in got.iter().enumerate() {
        assert!(
            (v - 1.0).abs() < 1e-2,
            "i={i}: got {v}, expected the MEAN 1.0 (a kernel that summed \
             instead would read {compress_ratio}.0)"
        );
    }
}

fn run_score(
    context: &mut MetalContext,
    q: &[f16],
    pooled: &[f16],
    num_heads: u32,
    head_dim: u32,
) -> Vec<f32> {
    let num_blocks = pooled.len() as u32 / head_dim;
    let q_buf = context.new_buffer_with_data(&to_le(q));
    let pooled_buf = context.new_buffer_with_data(&to_le(pooled));
    let scores_buf = context.new_output_buffer(num_blocks as u64 * 4);

    let pass = context.begin_pass();
    turbospark_gpu::encode_qsa_score_blocks(
        context,
        &pass,
        (&q_buf, 0),
        (&pooled_buf, 0),
        (&scores_buf, 0),
        num_heads,
        head_dim,
        num_blocks,
    )
    .expect("encode");
    pass.commit_and_wait();
    turbospark_gpu::read_f32_buffer(&scores_buf, num_blocks as usize)
}

/// The scoring kernel against the CPU reference, several blocks and several
/// heads, with values chosen so the relu-after-sum-over-heads
/// parenthesization matters: some individual head/block dot products are
/// negative while the total across heads is not (and vice versa), so a
/// kernel that applied relu PER HEAD before summing would read a different,
/// generally larger, number.
#[test]
fn score_blocks_matches_cpu_reference() {
    let mut context = MetalContext::new().expect("Metal device");
    let head_dim = 32u32;
    let num_heads = 4u32;
    let num_blocks = 3u32;

    let q16: Vec<f16> = (0..num_heads * head_dim)
        .map(|i| {
            let h = (i / head_dim) as f32;
            let d = (i % head_dim) as f32;
            f16::from_f32((d * 0.19 - h * 0.7).sin() * (1.0 + h * 0.3))
        })
        .collect();
    let pooled16: Vec<f16> = (0..num_blocks * head_dim)
        .map(|i| {
            let b = (i / head_dim) as f32;
            let d = (i % head_dim) as f32;
            f16::from_f32((d * 0.11 + b * 1.3).cos() * (1.0 + b * 0.2))
        })
        .collect();

    let q32: Vec<f32> = q16.iter().map(|v| v.to_f32()).collect();
    let pooled32: Vec<f32> = pooled16.iter().map(|v| v.to_f32()).collect();
    let expected =
        turbospark_compute::score_blocks(&q32, &pooled32, num_heads as usize, head_dim as usize);

    let got = run_score(&mut context, &q16, &pooled16, num_heads, head_dim);

    assert_eq!(got.len(), expected.len());
    for i in 0..got.len() {
        let diff = (got[i] - expected[i]).abs();
        assert!(
            diff <= 5e-2_f32.max(expected[i].abs() * 2e-2),
            "block {i}: got {} want {}",
            got[i],
            expected[i]
        );
    }
    // At least one block must have a genuinely nonzero score, or the relu
    // discrimination below is vacuous (everything already clamps to 0).
    assert!(
        expected.iter().any(|&s| s > 0.1),
        "fixture must produce at least one clearly-positive score, got {expected:?}"
    );
}

fn to_bf16_le(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 2);
    for &x in v {
        out.extend_from_slice(&turbospark_compute::f32_to_bf16(x).to_le_bytes());
    }
    out
}

/// `encode_qsa_advance_blocks` (pool -> norm -> RoPE, composed) against the
/// SAME three CPU-reference functions called by hand, in the SAME order --
/// `turbospark_compute::qsa_indexer`'s own `the_full_indexer_chain_composes`
/// test does this composition on the CPU side; this is its GPU-kernel
/// counterpart, at the identical parameters, so the fixture values below
/// double as a cross-check that the two independently-composed chains agree.
#[test]
fn advance_blocks_matches_the_composed_cpu_reference_chain() {
    let mut context = MetalContext::new().expect("Metal device");
    let head_dim = 8u32;
    let rotary_dim = 4u32;
    let theta = 10000.0f32;
    let eps = 1e-6f32;
    let compress_ratio = 2u32;
    let key_start_position = 0u32;

    // 2 complete blocks, no ragged tail: block 0's raw rows are both
    // all-1s, block 1's rows are distinct and nonuniform.
    #[rustfmt::skip]
    let raw_keys32: Vec<f32> = vec![
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        3.0, -2.0, 0.5, 4.0, -1.0, 2.5, 0.0, 1.5,
        1.0, 1.0, 2.0, -3.0, 0.5, 0.0, 2.0, -1.0,
    ];
    let raw_keys16: Vec<f16> = raw_keys32.iter().map(|&v| f16::from_f32(v)).collect();
    let k_norm_w32 = vec![0.1f32, -0.2, 0.05, 0.3, -0.1, 0.0, 0.2, -0.05];

    let raw_buf = context.new_buffer_with_data(&to_le(&raw_keys16));
    let pooled_buf = context.new_output_buffer(2 * head_dim as u64 * 2);
    let weight_buf = context.new_buffer_with_data(&to_bf16_le(&k_norm_w32));

    let pass = context.begin_pass();
    turbospark_gpu::encode_qsa_advance_blocks(
        &mut context,
        &pass,
        (&raw_buf, 0),
        (&pooled_buf, 0),
        (&weight_buf, 0),
        compress_ratio,
        head_dim,
        rotary_dim,
        theta,
        eps,
        0,
        2,
        key_start_position,
    )
    .expect("encode");
    pass.commit_and_wait();
    let got = read_halfs(&pooled_buf, (2 * head_dim) as usize);

    // The identical chain, composed by hand from the CPU reference.
    let pooled32 = turbospark_compute::pool_blocks_mean(
        &raw_keys32,
        head_dim as usize,
        compress_ratio as usize,
    );
    let mut expected = vec![0.0f32; pooled32.len()];
    for b in 0..2usize {
        let row = &pooled32[b * head_dim as usize..(b + 1) * head_dim as usize];
        let normed = turbospark_compute::rms_norm_centered(row, &k_norm_w32, eps);
        let block_position = key_start_position as usize + b * compress_ratio as usize;
        let roped = turbospark_compute::rope_neox_subdim(
            &normed,
            1,
            1,
            head_dim as usize,
            rotary_dim as usize,
            block_position,
            theta,
        );
        expected[b * head_dim as usize..(b + 1) * head_dim as usize].copy_from_slice(&roped);
    }

    assert_eq!(got.len(), expected.len());
    for i in 0..got.len() {
        let diff = (got[i] - expected[i]).abs();
        assert!(
            diff <= 5e-2_f32.max(expected[i].abs() * 2e-2),
            "i={i}: got {} want {}",
            got[i],
            expected[i]
        );
    }
}

/// **`key_start_position` MUST reach RoPE, not just `block_index *
/// compress_ratio`.** A fixture whose raw key history starts at absolute
/// position 0 (every other test in this file) cannot tell a dropped
/// `key_start_position` term apart from a correct one -- the two formulas
/// coincide there by construction, the identical self-relative-fixture trap
/// `crates/compute/src/qsa_indexer.rs`'s own `the_full_indexer_chain_composes`
/// test hit once already this session (there over `rotary_dim`, here over
/// this argument). This fixture starts at position 100 specifically so a
/// dropped offset reads a wrong angle rather than a coincidentally correct
/// one; mutation-checked in a prior session pass, where dropping the
/// offset survived every OTHER test in this file silently.
#[test]
fn advance_blocks_ropes_at_the_absolute_key_start_position() {
    let mut context = MetalContext::new().expect("Metal device");
    let head_dim = 8u32;
    let rotary_dim = 4u32;
    let theta = 10000.0f32;
    let eps = 1e-6f32;
    let compress_ratio = 2u32;
    let key_start_position = 100u32;

    let raw_keys32 = vec![1.0f32; (compress_ratio * head_dim) as usize];
    let raw_keys16: Vec<f16> = raw_keys32.iter().map(|&v| f16::from_f32(v)).collect();
    let k_norm_w32 = vec![0.1f32, -0.2, 0.05, 0.3, -0.1, 0.0, 0.2, -0.05];

    let raw_buf = context.new_buffer_with_data(&to_le(&raw_keys16));
    let pooled_buf = context.new_output_buffer(head_dim as u64 * 2);
    let weight_buf = context.new_buffer_with_data(&to_bf16_le(&k_norm_w32));

    let pass = context.begin_pass();
    turbospark_gpu::encode_qsa_advance_blocks(
        &mut context,
        &pass,
        (&raw_buf, 0),
        (&pooled_buf, 0),
        (&weight_buf, 0),
        compress_ratio,
        head_dim,
        rotary_dim,
        theta,
        eps,
        0,
        1,
        key_start_position,
    )
    .expect("encode");
    pass.commit_and_wait();
    let got = read_halfs(&pooled_buf, head_dim as usize);

    let pooled32 = turbospark_compute::pool_blocks_mean(
        &raw_keys32,
        head_dim as usize,
        compress_ratio as usize,
    );
    let normed = turbospark_compute::rms_norm_centered(&pooled32, &k_norm_w32, eps);
    let expected = turbospark_compute::rope_neox_subdim(
        &normed,
        1,
        1,
        head_dim as usize,
        rotary_dim as usize,
        key_start_position as usize,
        theta,
    );

    for i in 0..got.len() {
        let diff = (got[i] - expected[i]).abs();
        assert!(
            diff <= 5e-2_f32.max(expected[i].abs() * 2e-2),
            "i={i}: got {} want {} (RoPE'd at position {key_start_position})",
            got[i],
            expected[i]
        );
    }
}

/// **THE INCREMENTAL DESIGN'S WHOLE POINT**: calling `encode_qsa_advance_blocks`
/// for block 1 alone, after block 0 was already advanced, must leave block
/// 0's row untouched -- a caller that re-pooled/re-normed/re-roped the
/// whole history every time a new block completed would still pass the
/// single-call test above; only a two-call, incremental sequence can catch
/// a version that clobbers or drifts an already-computed block.
#[test]
fn advance_blocks_leaves_earlier_blocks_untouched_on_a_later_incremental_call() {
    let mut context = MetalContext::new().expect("Metal device");
    let head_dim = 8u32;
    let rotary_dim = 4u32;
    let theta = 10000.0f32;
    let eps = 1e-6f32;
    let compress_ratio = 2u32;

    #[rustfmt::skip]
    let raw_keys32: Vec<f32> = vec![
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        3.0, -2.0, 0.5, 4.0, -1.0, 2.5, 0.0, 1.5,
        1.0, 1.0, 2.0, -3.0, 0.5, 0.0, 2.0, -1.0,
    ];
    let raw_keys16: Vec<f16> = raw_keys32.iter().map(|&v| f16::from_f32(v)).collect();
    let k_norm_w32 = vec![0.1f32, -0.2, 0.05, 0.3, -0.1, 0.0, 0.2, -0.05];

    let raw_buf = context.new_buffer_with_data(&to_le(&raw_keys16));
    let pooled_buf = context.new_output_buffer(2 * head_dim as u64 * 2);
    let weight_buf = context.new_buffer_with_data(&to_bf16_le(&k_norm_w32));

    // First call: advance block 0 alone.
    let pass1 = context.begin_pass();
    turbospark_gpu::encode_qsa_advance_blocks(
        &mut context,
        &pass1,
        (&raw_buf, 0),
        (&pooled_buf, 0),
        (&weight_buf, 0),
        compress_ratio,
        head_dim,
        rotary_dim,
        theta,
        eps,
        0,
        1,
        0,
    )
    .expect("encode 1");
    pass1.commit_and_wait();
    let after_first = read_halfs(&pooled_buf, (2 * head_dim) as usize);

    // Second call: advance block 1 alone (first_new_block = 1).
    let pass2 = context.begin_pass();
    turbospark_gpu::encode_qsa_advance_blocks(
        &mut context,
        &pass2,
        (&raw_buf, 0),
        (&pooled_buf, 0),
        (&weight_buf, 0),
        compress_ratio,
        head_dim,
        rotary_dim,
        theta,
        eps,
        1,
        1,
        0,
    )
    .expect("encode 2");
    pass2.commit_and_wait();
    let after_second = read_halfs(&pooled_buf, (2 * head_dim) as usize);

    let hd = head_dim as usize;
    assert_eq!(
        &after_first[..hd],
        &after_second[..hd],
        "block 0's row must be identical before and after the incremental \
         call that only advances block 1"
    );
    // Block 1 must actually have moved from whatever garbage the fresh
    // buffer started with (all zeros).
    assert!(
        after_second[hd..].iter().any(|&v| v.abs() > 1e-3),
        "block 1 must be nonzero after being advanced"
    );
}

/// **RELU IS OUTSIDE THE HEAD SUM.** `q` has TWO heads that individually
/// dot to a strongly negative and a strongly positive value against the
/// same pooled block, chosen so their SUM is negative (relu-after-sum ->
/// 0) while relu-BEFORE-sum would keep the positive head's contribution
/// and report a large positive score instead.
#[test]
fn score_blocks_applies_relu_after_summing_across_heads() {
    let mut context = MetalContext::new().expect("Metal device");
    let head_dim = 4u32;
    let num_heads = 2u32;

    // pooled block = [1, 1, 1, 1].
    let pooled16 = vec![f16::from_f32(1.0); head_dim as usize];
    // head 0: dot = 4 + 4 + 4 + 4 = ... use distinct magnitudes instead so
    // relu-before-sum vs relu-after-sum give clearly different signs.
    // head 0 dot = 3+3+3+3 = 12 (positive), head 1 dot = -5-5-5-5 = -20
    // (negative). Sum = -8 -> relu(-8) = 0. relu(12)+relu(-20) = 12+0 = 12.
    let q16: Vec<f16> = vec![
        f16::from_f32(3.0),
        f16::from_f32(3.0),
        f16::from_f32(3.0),
        f16::from_f32(3.0),
        f16::from_f32(-5.0),
        f16::from_f32(-5.0),
        f16::from_f32(-5.0),
        f16::from_f32(-5.0),
    ];

    let got = run_score(&mut context, &q16, &pooled16, num_heads, head_dim);
    assert_eq!(got.len(), 1);
    assert!(
        got[0].abs() < 1e-3,
        "relu(sum over heads) must be 0 here (sum = -8), got {} \
         (a per-head relu would read 12 / sqrt(4) = 6.0)",
        got[0]
    );
}
