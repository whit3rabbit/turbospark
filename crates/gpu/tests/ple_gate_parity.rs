#![cfg(target_os = "macos")]
//! Parity tests for `ple_gate_fp16` (`qwen4_exp`'s PLE gate,
//! `docs/QWEN4_PHASE0.md` section 4) against `turbospark_compute::ple_gate`.
//!
//! **THE SIGNED SQRT IS THE TRAP THIS FILE EXISTS FOR.** The reference reads
//! `sign(gate) * sqrt(max(|gate|, 1e-6))`, not a plain `sqrt`, and the doc
//! that specifies it calls this out by name as "the sort of line that reads
//! as a typo and is not." A kernel that dropped the sign (or used the wrong
//! zero convention for it) produces finite, plausible numbers, never a
//! crash -- so this file's fixtures deliberately span both signs of the
//! dot product.

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

fn run_ple_gate(
    context: &mut MetalContext,
    key: &[f16],
    query: &[f16],
    value: &[f16],
    groups: u32,
) -> Vec<f32> {
    let total = key.len();
    assert_eq!(total, query.len());
    let group_dim = total as u32 / groups;
    assert_eq!(value.len(), group_dim as usize);
    let key_buf = context.new_buffer_with_data(&to_le(key));
    let query_buf = context.new_buffer_with_data(&to_le(query));
    let value_buf = context.new_buffer_with_data(&to_le(value));
    let out_buf = context.new_output_buffer((total * 2) as u64);

    let pass = context.begin_pass();
    turbospark_gpu::encode_ple_gate(
        context,
        &pass,
        (&key_buf, 0),
        (&query_buf, 0),
        (&value_buf, 0),
        (&out_buf, 0),
        groups,
        group_dim,
    )
    .expect("encode");
    pass.commit_and_wait();
    read_halfs(&out_buf, total)
}

/// The kernel against the CPU reference, at `qwen4_exp`'s real `C = 4`
/// proportions with `H` scaled down. Groups are seeded so their dot
/// products land on BOTH sides of zero (checked below), which is what
/// exercises the signed sqrt in both directions in one run.
#[test]
fn ple_gate_matches_cpu_reference() {
    let mut context = MetalContext::new().expect("Metal device");
    let groups = 4usize;
    let group_dim = 256usize;
    let total = groups * group_dim;

    let key16: Vec<f16> = (0..total)
        .map(|i| {
            let c = (i / group_dim) as f32;
            let lane = (i % group_dim) as f32;
            f16::from_f32(0.3 * (lane * 0.19 + c * 1.1).sin())
        })
        .collect();
    // Alternate sign per group by flipping query's phase, so group dot
    // products land on both sides of zero rather than all agreeing.
    let query16: Vec<f16> = (0..total)
        .map(|i| {
            let c = (i / group_dim) as f32;
            let lane = (i % group_dim) as f32;
            let sign = if (c as usize) % 2 == 0 { 1.0 } else { -1.0 };
            f16::from_f32(sign * 0.3 * (lane * 0.19 + c * 1.1 + 0.4).sin())
        })
        .collect();
    let value16: Vec<f16> = (0..group_dim)
        .map(|i| f16::from_f32((i as f32 * 0.11).cos() * 0.7))
        .collect();

    let key32: Vec<f32> = key16.iter().map(|v| v.to_f32()).collect();
    let query32: Vec<f32> = query16.iter().map(|v| v.to_f32()).collect();
    let value32: Vec<f32> = value16.iter().map(|v| v.to_f32()).collect();

    // Confirm the fixture actually spans both signs before trusting the
    // parity case to have exercised the signed sqrt in both directions.
    let mut saw_positive = false;
    let mut saw_negative = false;
    for c in 0..groups {
        let base = c * group_dim;
        let dot: f32 = (0..group_dim)
            .map(|i| key32[base + i] * query32[base + i])
            .sum();
        if dot > 0.0 {
            saw_positive = true;
        }
        if dot < 0.0 {
            saw_negative = true;
        }
    }
    assert!(
        saw_positive && saw_negative,
        "fixture must span both signs of the dot product to exercise the \
         signed sqrt in both directions"
    );

    let expected = turbospark_compute::ple_gate(&key32, &query32, &value32, groups);
    let got = run_ple_gate(&mut context, &key16, &query16, &value16, groups as u32);

    assert_eq!(got.len(), expected.len());
    for i in 0..total {
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

/// **THE SIGN MUST FLIP THE GATE, NOT JUST ITS MAGNITUDE.** Two fixtures
/// identical except that one group's `query` is negated: the dot product
/// (and therefore the pre-sqrt `gate`) flips sign, so a correct
/// implementation of `sign(gate) * sqrt(|gate|)` must move the SIGMOID
/// output measurably (sigmoid is monotonic and not symmetric about 0.5 in a
/// way a sign-dropping bug could fake here, since the two gates have
/// different MAGNITUDES too -- this is not a degenerate point). A kernel
/// that computed `sqrt(max(gate, 1e-6))` (dropping the sign entirely, and
/// clamping negative gates to the SAME floor regardless of magnitude) would
/// produce IDENTICAL output for both cases whenever `gate <= 0`, which this
/// asserts against.
#[test]
fn flipping_the_dot_products_sign_moves_the_gate() {
    let mut context = MetalContext::new().expect("Metal device");
    let groups = 1usize;
    let group_dim = 64usize;

    let key16: Vec<f16> = (0..group_dim)
        .map(|i| f16::from_f32(0.4 * (i as f32 * 0.23).sin()))
        .collect();
    let query_pos: Vec<f16> = (0..group_dim)
        .map(|i| f16::from_f32(0.4 * (i as f32 * 0.23).sin()))
        .collect();
    let query_neg: Vec<f16> = query_pos
        .iter()
        .map(|v| f16::from_f32(-v.to_f32()))
        .collect();
    let value16: Vec<f16> = (0..group_dim)
        .map(|i| f16::from_f32((i as f32 * 0.13).cos()))
        .collect();

    let pos_out = run_ple_gate(&mut context, &key16, &query_pos, &value16, groups as u32);
    let neg_out = run_ple_gate(&mut context, &key16, &query_neg, &value16, groups as u32);

    let max_gap = pos_out
        .iter()
        .zip(&neg_out)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_gap > 1e-2,
        "flipping query's sign (and so the dot product's sign) must move \
         the gate measurably (max gap {max_gap}); a kernel that dropped the \
         sign would produce the same output for both"
    );
}
