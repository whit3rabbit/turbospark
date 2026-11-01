//! The Gemma 4 tower's three new Metal kernels against
//! `turbospark_compute::vision_gemma4`, on real hardware.
//!
//! As in `vision_parity.rs`, each parity case is paired with a check that the
//! fixture can tell the kernel from its plausible wrong twin: RMSNorm from
//! the `1 + w` form, the axial rope from the whole-head half-split rope, and
//! an in-place norm from one that reads its own writes.
#![cfg(target_os = "macos")]

use half::f16;
use turbospark_compute::rms_norm::rms_norm;
use turbospark_compute::vision::rope_vision_2d;
use turbospark_compute::vision_gemma4::axial_rope_2d;
use turbospark_gpu::{
    encode_vision_mul, encode_vision_rms_norm, encode_vision_rope_axial, MetalContext,
};

fn read_f16(buffer: &metal::Buffer, len: usize) -> Vec<f32> {
    let ptr = buffer.contents() as *const u16;
    // The buffer holds at least `len` halves and its pass has been waited on.
    let bits = unsafe { std::slice::from_raw_parts(ptr, len) };
    bits.iter().map(|&b| f16::from_bits(b).to_f32()).collect()
}

fn half_of(values: &[f32]) -> Vec<f16> {
    values.iter().map(|&v| f16::from_f32(v)).collect()
}

/// What FP16 storage did to the input, so the reference sees the same data.
fn rounded(values: &[f32]) -> Vec<f32> {
    half_of(values).iter().map(|v| v.to_f32()).collect()
}

fn wave(n: usize, phase: f32, amp: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32) * 0.37 + phase).sin() * amp)
        .collect()
}

fn worst(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).fold(0.0f32, |m, (x, y)| {
        assert!(x.is_finite() && y.is_finite(), "non-finite value");
        m.max((x - y).abs())
    })
}

fn bound(values: &[f32]) -> f32 {
    2e-3 * values.iter().fold(1.0f32, |m, v| m.max(v.abs()))
}

fn run_rms(rows: usize, d: usize, x: &[f32], w: Option<&[f32]>, in_place: bool) -> Vec<f32> {
    let mut context = MetalContext::new().expect("Metal device");
    let xb = context.new_buffer_with_data(&half_of(x));
    let wb = w.map(|w| context.new_buffer_with_data(&half_of(w)));
    let ob = context.new_output_buffer((rows * d * 2) as u64);
    let pass = context.begin_pass();
    let out = if in_place { (&xb, 0) } else { (&ob, 0) };
    encode_vision_rms_norm(
        &mut context,
        &pass,
        (&xb, 0),
        wb.as_ref().map(|b| (b, 0)),
        out,
        rows as u32,
        d as u32,
        1e-6,
    )
    .expect("encode");
    pass.commit_and_wait();
    read_f16(if in_place { &xb } else { &ob }, rows * d)
}

fn cpu_rms(rows: usize, d: usize, x: &[f32], w: Option<&[f32]>) -> Vec<f32> {
    let ones = vec![1.0f32; d];
    let w = w.map(rounded).unwrap_or(ones);
    rounded(x)
        .chunks_exact(d)
        .take(rows)
        .flat_map(|row| rms_norm(row, &w, 1e-6))
        .collect()
}

#[test]
fn rms_norm_matches_the_reference_weighted_and_unweighted() {
    let (rows, d) = (5usize, 1152usize);
    let x = wave(rows * d, 0.3, 40.0);
    let w = wave(d, 1.1, 0.8);
    let want = cpu_rms(rows, d, &x, Some(&w));
    let got = run_rms(rows, d, &x, Some(&w), false);
    assert!(worst(&got, &want) <= bound(&want), "weighted");

    let want = cpu_rms(rows, d, &x, None);
    let got = run_rms(rows, d, &x, None, false);
    assert!(worst(&got, &want) <= bound(&want), "unweighted");

    // Discrimination: the centered `1 + w` form is measurably different here.
    let shifted: Vec<f32> = w.iter().map(|v| 1.0 + v).collect();
    let wrong = cpu_rms(rows, d, &x, Some(&shifted));
    let got = run_rms(rows, d, &x, Some(&w), false);
    assert!(worst(&got, &wrong) > 10.0 * bound(&wrong));
}

/// The per-head form the tower uses for q, k and v: many short rows.
#[test]
fn rms_norm_per_head_and_in_place_matches() {
    let (rows, d) = (7 * 16usize, 72usize);
    let x = wave(rows * d, 2.0, 9.0);
    let w = wave(d, 0.2, 1.5);
    let want = cpu_rms(rows, d, &x, Some(&w));
    let got = run_rms(rows, d, &x, Some(&w), true);
    assert!(worst(&got, &want) <= bound(&want), "in place");
}

fn freq_rows(positions: &[(usize, usize)], head_dim: usize, theta: f32) -> Vec<f32> {
    let part = head_dim / 2;
    let quarter = head_dim / 4;
    positions
        .iter()
        .flat_map(|&(x, y)| {
            [x, y].into_iter().flat_map(move |p| {
                (0..quarter).map(move |i| p as f32 * theta.powf(-(2.0 * i as f32) / part as f32))
            })
        })
        .collect()
}

#[test]
fn axial_rope_matches_the_reference() {
    let (heads, hd) = (16usize, 72usize);
    let positions: Vec<(usize, usize)> = (0..4)
        .flat_map(|y| (0..9).map(move |x| (x * 7, y * 11)))
        .collect();
    let seq = positions.len();
    let x = wave(seq * heads * hd, 0.9, 3.0);
    let freqs = freq_rows(&positions, hd, 100.0);

    let mut context = MetalContext::new().expect("Metal device");
    let qb = context.new_buffer_with_data(&half_of(&x));
    let fb = context.new_buffer_with_data(&freqs);
    let pass = context.begin_pass();
    encode_vision_rope_axial(
        &mut context,
        &pass,
        (&qb, 0),
        (&fb, 0),
        seq as u32,
        heads as u32,
        hd as u32,
    )
    .expect("encode");
    pass.commit_and_wait();
    let got = read_f16(&qb, seq * heads * hd);

    let mut want = rounded(&x);
    for (t, &(px, py)) in positions.iter().enumerate() {
        for h in 0..heads {
            let at = (t * heads + h) * hd;
            axial_rope_2d(&mut want[at..at + hd], px, py, 100.0);
        }
    }
    assert!(worst(&got, &want) <= bound(&want), "axial rope");

    // Discrimination: the whole-head half-split rope at the same angles.
    let mut wrong = rounded(&x);
    for (t, chunk) in wrong.chunks_exact_mut(hd).enumerate() {
        let row = &freqs[(t / heads) * (hd / 2)..(t / heads + 1) * (hd / 2)];
        chunk.copy_from_slice(&rope_vision_2d(chunk, row));
    }
    assert!(worst(&got, &wrong) > 10.0 * bound(&wrong));
}

#[test]
fn mul_matches_the_reference() {
    let n = 1000usize;
    let a = wave(n, 0.1, 4.0);
    let b = wave(n, 2.3, 3.0);
    let mut context = MetalContext::new().expect("Metal device");
    let ab = context.new_buffer_with_data(&half_of(&a));
    let bb = context.new_buffer_with_data(&half_of(&b));
    let pass = context.begin_pass();
    encode_vision_mul(&mut context, &pass, (&ab, 0), (&bb, 0), n as u32).expect("encode");
    pass.commit_and_wait();
    let got = read_f16(&ab, n);
    let want: Vec<f32> = rounded(&a)
        .iter()
        .zip(rounded(&b))
        .map(|(x, y)| x * y)
        .collect();
    assert!(worst(&got, &want) <= bound(&want));
}

#[test]
fn axial_rope_refuses_a_head_it_cannot_split() {
    let mut context = MetalContext::new().expect("Metal device");
    let b = context.new_buffer_with_data(&half_of(&[0.0; 6]));
    let f = context.new_buffer_with_data(&[0.0f32; 3]);
    let pass = context.begin_pass();
    assert!(encode_vision_rope_axial(&mut context, &pass, (&b, 0), (&f, 0), 1, 1, 6).is_err());
    pass.commit_and_wait();
}
