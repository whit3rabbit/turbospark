#![cfg(target_os = "macos")]
//! Parity tests for `hc_mix_fp16` and `hc_inject_add_fp16` (`qwen4_exp`'s
//! hyper-connection mix and scatter, `docs/QWEN4_PHASE0.md` section 3)
//! against `turbospark_compute::hc_mix` / `hc_inject_add`.
//!
//! **NOT `HyperConnectionConfig`'s Sinkhorn-normalised mHC.** That is
//! DeepSeek-V4-Flash's mechanism and has no kernel anywhere in this port;
//! this file tests `qwen4_exp`'s low-rank silu/sigmoid mix, a different
//! algorithm that happens to share a name prefix in the config struct.

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

fn run_hc_mix(context: &mut MetalContext, w: &[f16], normed: &[f16], groups: u32) -> Vec<f32> {
    let total = w.len();
    assert_eq!(total, normed.len());
    let group_dim = total as u32 / groups;
    let w_buf = context.new_buffer_with_data(&to_le(w));
    let normed_buf = context.new_buffer_with_data(&to_le(normed));
    let out_buf = context.new_output_buffer((group_dim as usize * 2) as u64);

    let pass = context.begin_pass();
    turbospark_gpu::encode_hc_mix(
        context,
        &pass,
        (&w_buf, 0),
        (&normed_buf, 0),
        (&out_buf, 0),
        groups,
        group_dim,
    )
    .expect("encode");
    pass.commit_and_wait();
    read_halfs(&out_buf, group_dim as usize)
}

/// The kernel against the CPU reference, at `qwen4_exp`'s real `C = 4`
/// proportions with `H` scaled down for a fast test.
///
/// Every stream gets DIFFERENT values, both in `w` (the gate, which sits in
/// `[0, 1]` post-sigmoid) and in `normed` (which can be any sign) -- so a
/// kernel that read the wrong stream's slice, or dropped one stream from the
/// mean entirely, disagrees with a per-stream-varying reference in a way a
/// uniform fixture could not show.
#[test]
fn hc_mix_matches_cpu_reference() {
    let mut context = MetalContext::new().expect("Metal device");
    let groups = 4usize;
    let group_dim = 256usize;
    let total = groups * group_dim;

    let w16: Vec<f16> = (0..total)
        .map(|i| {
            let c = (i / group_dim) as f32;
            let lane = (i % group_dim) as f32;
            // Sigmoid range: [0, 1], distinct per stream.
            f16::from_f32(0.5 + 0.4 * ((lane * 0.19 + c * 0.9).sin()))
        })
        .collect();
    let normed16: Vec<f16> = (0..total)
        .map(|i| {
            let c = (i / group_dim) as f32;
            let lane = (i % group_dim) as f32;
            f16::from_f32((1.0 + c) * (lane * 0.23 - c * 1.3).sin())
        })
        .collect();

    let w32: Vec<f32> = w16.iter().map(|v| v.to_f32()).collect();
    let normed32: Vec<f32> = normed16.iter().map(|v| v.to_f32()).collect();
    let expected = turbospark_compute::hc_mix(&w32, &normed32, groups, group_dim);

    let got = run_hc_mix(&mut context, &w16, &normed16, groups as u32);

    assert_eq!(got.len(), expected.len());
    for i in 0..group_dim {
        let diff = (got[i] - expected[i]).abs();
        assert!(
            diff <= 2e-3_f32.max(expected[i].abs() * 1e-2),
            "i={i}: got {} want {}",
            got[i],
            expected[i]
        );
    }
}

/// **THE MEAN, NOT THE SUM.** A kernel that summed the four streams instead
/// of averaging them would still pass a fixture whose expected values allow
/// for either (e.g. all streams near zero); this asserts the actual scale by
/// checking a fixture where `w = 1` everywhere and `normed`'s streams take
/// four DISTINCT constant values, so the mean has one unambiguous answer:
/// the arithmetic mean of the four constants, not their sum.
#[test]
fn hc_mix_divides_by_group_count_not_just_summing() {
    let mut context = MetalContext::new().expect("Metal device");
    let groups = 4usize;
    let group_dim = 32usize;
    let total = groups * group_dim;

    let w16 = vec![f16::from_f32(1.0); total];
    let stream_values = [2.0f32, 4.0, 6.0, 8.0]; // mean = 5.0, sum = 20.0
    let normed16: Vec<f16> = (0..total)
        .map(|i| f16::from_f32(stream_values[i / group_dim]))
        .collect();

    let got = run_hc_mix(&mut context, &w16, &normed16, groups as u32);
    for (i, &v) in got.iter().enumerate() {
        assert!(
            (v - 5.0).abs() < 1e-2,
            "i={i}: got {v}, expected the MEAN 5.0 (a kernel that summed \
             instead would read 20.0)"
        );
    }
}

fn run_hc_inject_add(
    context: &mut MetalContext,
    raw: &[f16],
    out: &[f16],
    inject_w: &[f16],
    groups: u32,
) -> Vec<f32> {
    let total = raw.len();
    let group_dim = total as u32 / groups;
    assert_eq!(out.len(), group_dim as usize);
    assert_eq!(inject_w.len(), groups as usize);
    // In place: the buffer starts as `raw` and the kernel accumulates onto it.
    let hidden_buf = context.new_buffer_with_data(&to_le(raw));
    let out_buf = context.new_buffer_with_data(&to_le(out));
    let inject_buf = context.new_buffer_with_data(&to_le(inject_w));

    let pass = context.begin_pass();
    turbospark_gpu::encode_hc_inject_add(
        context,
        &pass,
        (&hidden_buf, 0),
        (&out_buf, 0),
        (&inject_buf, 0),
        groups,
        group_dim,
    )
    .expect("encode");
    pass.commit_and_wait();
    read_halfs(&hidden_buf, total)
}

/// The kernel against the CPU reference. Every stream's `raw` slice and
/// `inject_w` scalar are distinct, so a broadcast that read the wrong
/// `inject_w[c]` or scattered to the wrong stream disagrees rather than
/// happening to land on a shared value.
#[test]
fn hc_inject_add_matches_cpu_reference() {
    let mut context = MetalContext::new().expect("Metal device");
    let groups = 4usize;
    let group_dim = 256usize;
    let total = groups * group_dim;

    let raw16: Vec<f16> = (0..total)
        .map(|i| {
            let c = (i / group_dim) as f32;
            let lane = (i % group_dim) as f32;
            f16::from_f32((1.0 + c) * (lane * 0.17 - c * 0.6).sin())
        })
        .collect();
    let out16: Vec<f16> = (0..group_dim)
        .map(|i| f16::from_f32((i as f32 * 0.29).cos() * 0.8))
        .collect();
    // `2 * sigmoid(...)` has range [0, 2]; four distinct values in that band.
    let inject_w16: Vec<f16> = [0.2f32, 0.9, 1.4, 1.8]
        .iter()
        .map(|&v| f16::from_f32(v))
        .collect();

    let raw32: Vec<f32> = raw16.iter().map(|v| v.to_f32()).collect();
    let out32: Vec<f32> = out16.iter().map(|v| v.to_f32()).collect();
    let inject32: Vec<f32> = inject_w16.iter().map(|v| v.to_f32()).collect();
    let expected = turbospark_compute::hc_inject_add(&raw32, &out32, &inject32, groups, group_dim);

    let got = run_hc_inject_add(&mut context, &raw16, &out16, &inject_w16, groups as u32);

    assert_eq!(got.len(), expected.len());
    for i in 0..total {
        let diff = (got[i] - expected[i]).abs();
        assert!(
            diff <= 2e-2_f32.max(expected[i].abs() * 1e-2),
            "i={i} (stream {}): got {} want {}",
            i / group_dim,
            got[i],
            expected[i]
        );
    }
}

/// **BROADCAST, NOT A SECOND STREAM AXIS ON `out`.** `out` is `[group_dim]`
/// only -- one value per lane, shared by construction across every stream --
/// so a kernel that (wrongly) indexed `out` by `c * group_dim + h` instead of
/// `h` alone would read out of bounds or a garbage neighbor rather than a
/// plausible number. This fixes `inject_w` at a single nonzero value and
/// zero `raw`, so the expected output is exactly `inject_w[c] * out[h]` with
/// nothing else contributing -- a fixture that could not tell a correct
/// broadcast from an over-indexed one would pass either way.
#[test]
fn hc_inject_add_broadcasts_out_across_every_stream() {
    let mut context = MetalContext::new().expect("Metal device");
    let groups = 4usize;
    let group_dim = 16usize;
    let total = groups * group_dim;

    let raw16 = vec![f16::from_f32(0.0); total];
    let out16: Vec<f16> = (0..group_dim)
        .map(|i| f16::from_f32(1.0 + i as f32 * 0.1))
        .collect();
    let inject_w16 = vec![f16::from_f32(2.0); groups];

    let got = run_hc_inject_add(&mut context, &raw16, &out16, &inject_w16, groups as u32);
    for c in 0..groups {
        for h in 0..group_dim {
            let expected = 2.0 * (1.0 + h as f32 * 0.1);
            let idx = c * group_dim + h;
            assert!(
                (got[idx] - expected).abs() < 1e-2,
                "c={c} h={h}: got {} want {expected}",
                got[idx]
            );
        }
    }
}
