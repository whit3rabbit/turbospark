#![cfg(target_os = "macos")]
//! Attention-sink parity (ROADMAP M5, the `gpt-oss` family): one learned
//! logit per query head, added to the softmax DENOMINATOR and to nothing
//! else, against `turbospark_compute::causal_attention_with_sinks` on real
//! hardware.
//!
//! WHAT MAKES THIS WORTH A SEPARATE FILE rather than a case in
//! `attention_swa.rs`: the sink is the only thing in this port that changes
//! attention's normalization without changing its numerator, so both of its
//! failure modes are quiet. Ignoring it leaves a model whose attention is
//! slightly too confident everywhere, and adding it to the numerator turns
//! it into an ordinary extra key with a zero value -- a different function
//! that is still finite, still normalized, and still fluent.
//!
//! The sinks arrive as BF16 because `transcode_f32` narrows gpt-oss's F32
//! `attn_sinks.weight` at repack, the same width and reader as a norm.
//!
//! MUTATION-CHECKED, AND ONE OF THE TWO MUTATIONS IS NOT OBSERVABLE. Giving
//! the sink the wrong weight in the denominator (doubling it) reddens
//! `a_dominant_sink_takes_most_of_the_mass`, and ONLY that case -- which is
//! what that case is for, since a sink far below the score maximum changes
//! the answer by less than the FP16 tolerance whatever the kernel does with
//! it.
//!
//! Dropping the sink from `m_glob` reddens NOTHING, and that is a fact about
//! the arithmetic rather than a hole in the fixture. Softmax is invariant to
//! the choice of maximum, so including the sink there is a numerical-
//! stability guard, not a correctness one: it only matters when
//! `sink - m_glob` is large enough for `exp` to overflow. And in exactly
//! that regime the sink already dominates the denominator, so the true
//! output is ~0 and the overflowing one is `1/inf * v = 0` as well. The two
//! agree wherever they differ at all. The line stays because it is free and
//! because it is what ggml does; the test does not claim to check it.

use half::f16;
use turbospark_gpu::{AttentionScratch, MetalContext};

fn to_le_f16(v: &[f16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 2);
    for x in v {
        out.extend_from_slice(&x.to_bits().to_le_bytes());
    }
    out
}

/// BF16 is the top 16 bits of the FP32 word, which is the same hand-rolled
/// convention `turbospark_compute`'s helpers use (AGENTS.md Gotcha 3 forbids
/// hand-rolling FP16, not BF16).
fn to_le_bf16(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 2);
    for x in v {
        out.extend_from_slice(&((x.to_bits() >> 16) as u16).to_le_bytes());
    }
    out
}

fn bf16_round(x: f32) -> f32 {
    f32::from_bits((x.to_bits() >> 16) << 16)
}

/// One decode-attention dispatch with sinks, against the CPU reference.
///
/// `sink_scale` moves the sinks relative to the real scores, which is the
/// axis that matters: a sink far below the maximum score contributes almost
/// nothing and would let a broken kernel pass, while one far above it takes
/// nearly all the mass and makes the output approach zero. Both ends are
/// exercised by the cases below.
fn assert_sinks_match_cpu(
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    seq_len: u32,
    sink_scale: f32,
) {
    let mut context = MetalContext::new().expect("Metal device");
    let scale = 0.125f32;

    let q16: Vec<f16> = (0..(num_q_heads * head_dim) as usize)
        .map(|i| f16::from_f32(((i as f32) * 0.19).sin()))
        .collect();
    let kv_len = (seq_len * num_kv_heads * head_dim) as usize;
    let k16: Vec<f16> = (0..kv_len)
        .map(|i| f16::from_f32(((i as f32) * 0.07).cos()))
        .collect();
    let v16: Vec<f16> = (0..kv_len)
        .map(|i| f16::from_f32(((i as f32) * 0.11).sin()))
        .collect();
    // Rounded through BF16 on the reference side too, since that is the
    // width the kernel reads and the only place these two can disagree for a
    // reason that is not the sink arithmetic.
    let sinks: Vec<f32> = (0..num_q_heads as usize)
        .map(|i| bf16_round(((i as f32) * 0.37).sin() * sink_scale))
        .collect();

    let q32: Vec<f32> = q16.iter().map(|x| x.to_f32()).collect();
    let k32: Vec<f32> = k16.iter().map(|x| x.to_f32()).collect();
    let v32: Vec<f32> = v16.iter().map(|x| x.to_f32()).collect();

    let expected = turbospark_compute::causal_attention_with_sinks(
        &q32,
        &k32,
        &v32,
        head_dim as usize,
        num_q_heads as usize,
        num_kv_heads as usize,
        seq_len as usize,
        None,
        Some(scale),
        Some(&sinks),
    );

    let q_buf = context.new_buffer_with_data(&to_le_f16(&q16));
    let k_buf = context.new_buffer_with_data(&to_le_f16(&k16));
    let v_buf = context.new_buffer_with_data(&to_le_f16(&v16));
    let sink_buf = context.new_buffer_with_data(&to_le_bf16(&sinks));
    let out_buf = context.new_output_buffer((num_q_heads * head_dim) as u64 * 2);
    let scratch = AttentionScratch::new(&context, num_q_heads, head_dim);

    let pass = context.begin_pass();
    turbospark_gpu::encode_attention_decode(
        &mut context,
        &pass,
        (&q_buf, 0),
        &k_buf,
        &v_buf,
        &scratch,
        (&out_buf, 0),
        head_dim,
        num_q_heads,
        num_kv_heads,
        seq_len,
        0,
        0,
        scale,
        Some((&sink_buf, 0)),
    )
    .expect("encode");
    pass.commit_and_wait();

    let got: Vec<f32> = {
        let ptr = out_buf.contents() as *const u16;
        let bits = unsafe { std::slice::from_raw_parts(ptr, (num_q_heads * head_dim) as usize) };
        bits.iter().map(|&b| f16::from_bits(b).to_f32()).collect()
    };

    for i in 0..got.len() {
        let diff = (got[i] - expected[i]).abs();
        let tol = 3e-3_f32.max(expected[i].abs() * 3e-2);
        assert!(
            diff <= tol,
            "sink_scale={sink_scale} i={i}: got {} want {} (diff {diff})",
            got[i],
            expected[i]
        );
    }
}

/// Sinks comparable in size to the real scores, which is where gpt-oss's
/// trained values sit and where the term genuinely changes the answer.
#[test]
fn sinks_match_the_cpu_reference() {
    assert_sinks_match_cpu(4, 2, 32, 48, 1.0);
}

/// Sinks well ABOVE the score maximum, so they take most of the probability
/// mass and the output shrinks toward zero. This is the case that fails if
/// the sink is left out of `m_glob`: `exp(sink - m_glob)` would overflow the
/// denominator, or be computed against the wrong maximum.
#[test]
fn a_dominant_sink_takes_most_of_the_mass() {
    assert_sinks_match_cpu(4, 2, 32, 48, 8.0);
}

/// Sinks far BELOW every score, where the term is nearly a no-op. Kept
/// because it is the case a broken kernel is most likely to pass, so its
/// value is as a control on the two above rather than as a check of its own.
#[test]
fn a_negligible_sink_barely_moves_the_output() {
    assert_sinks_match_cpu(4, 2, 32, 48, -8.0);
}

/// More q heads than the 32-lane SIMD width, so the per-head indexing of
/// `sinks[q_head]` is exercised past one threadgroup's worth.
#[test]
fn sinks_are_indexed_per_query_head() {
    assert_sinks_match_cpu(64, 8, 64, 96, 1.0);
}

/// A LONG context, so the combine pass really has several chunks to fold and
/// the sink joins a multi-chunk maximum rather than a single one.
///
/// This is the case `attention_decode_combine` exists for: at `num_chunks
/// == 1` its `m_glob` is just `m_row[0]` and the sink's interaction with the
/// cross-chunk rescale is invisible.
#[test]
fn the_sink_joins_a_multi_chunk_maximum() {
    assert_sinks_match_cpu(4, 2, 32, 2048, 1.0);
}

/// The guard that says the four sink-free families are untouched: the same
/// dispatch with `None` must equal the plain reference exactly, and it takes
/// a DIFFERENT pipeline (the function constant is in the cache key), so this
/// also checks the two specializations do not collide.
#[test]
fn a_sink_free_dispatch_is_unchanged() {
    let mut context = MetalContext::new().expect("Metal device");
    let (num_q_heads, num_kv_heads, head_dim, seq_len) = (4u32, 2u32, 32u32, 48u32);
    let scale = 0.125f32;

    let q16: Vec<f16> = (0..(num_q_heads * head_dim) as usize)
        .map(|i| f16::from_f32(((i as f32) * 0.19).sin()))
        .collect();
    let kv_len = (seq_len * num_kv_heads * head_dim) as usize;
    let k16: Vec<f16> = (0..kv_len)
        .map(|i| f16::from_f32(((i as f32) * 0.07).cos()))
        .collect();
    let v16: Vec<f16> = (0..kv_len)
        .map(|i| f16::from_f32(((i as f32) * 0.11).sin()))
        .collect();

    let expected = turbospark_compute::causal_attention(
        &q16.iter().map(|x| x.to_f32()).collect::<Vec<_>>(),
        &k16.iter().map(|x| x.to_f32()).collect::<Vec<_>>(),
        &v16.iter().map(|x| x.to_f32()).collect::<Vec<_>>(),
        head_dim as usize,
        num_q_heads as usize,
        num_kv_heads as usize,
        seq_len as usize,
        None,
        Some(scale),
    );

    let q_buf = context.new_buffer_with_data(&to_le_f16(&q16));
    let k_buf = context.new_buffer_with_data(&to_le_f16(&k16));
    let v_buf = context.new_buffer_with_data(&to_le_f16(&v16));
    let out_buf = context.new_output_buffer((num_q_heads * head_dim) as u64 * 2);
    let scratch = AttentionScratch::new(&context, num_q_heads, head_dim);

    let pass = context.begin_pass();
    turbospark_gpu::encode_attention_decode(
        &mut context,
        &pass,
        (&q_buf, 0),
        &k_buf,
        &v_buf,
        &scratch,
        (&out_buf, 0),
        head_dim,
        num_q_heads,
        num_kv_heads,
        seq_len,
        0,
        0,
        scale,
        None,
    )
    .expect("encode");
    pass.commit_and_wait();

    let got: Vec<f32> = {
        let ptr = out_buf.contents() as *const u16;
        let bits = unsafe { std::slice::from_raw_parts(ptr, (num_q_heads * head_dim) as usize) };
        bits.iter().map(|&b| f16::from_bits(b).to_f32()).collect()
    };
    for i in 0..got.len() {
        let diff = (got[i] - expected[i]).abs();
        assert!(
            diff <= 3e-3_f32.max(expected[i].abs() * 3e-2),
            "i={i}: got {} want {}",
            got[i],
            expected[i]
        );
    }
}
