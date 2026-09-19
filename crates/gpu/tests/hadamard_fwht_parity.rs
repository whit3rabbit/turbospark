//! Runs `hadamard_fwht_rows` on real Metal hardware against the CPU
//! reference in `turbospark_compute::kv_quant` (`rht_forward`/`rht_inverse`
//! per block segment) -- the activation-side half of the prism Hadamard
//! contract, `docs/BONSAI2.md`. Proves the shader compiles (a pipeline is
//! created), dispatches, and computes the same signed block transform prism's
//! bundled `fwht` does.
#![cfg(target_os = "macos")]

use half::f16;
use turbospark_gpu::{encode_hadamard_fwht, read_buffer_f16, MetalContext};

fn to_le(v: &[f16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect()
}

fn run_kernel(
    context: &mut MetalContext,
    src: &[f16],
    signs: &[f32],
    rows: u32,
    width: u32,
    block: u32,
    forward: bool,
) -> Vec<f16> {
    let src_buf = context.new_buffer_with_data(&to_le(src));
    let dst_buf = context.new_output_buffer((src.len() * 2) as u64);
    let signs_buf = context.new_buffer_with_data(signs);
    let pass = context.begin_pass();
    encode_hadamard_fwht(
        context,
        &pass,
        (&src_buf, 0),
        (&dst_buf, 0),
        (&signs_buf, 0),
        rows,
        width,
        block,
        forward,
    )
    .expect("dispatch succeeds");
    pass.commit_and_wait();
    read_buffer_f16(&dst_buf, 0, src.len())
}

/// The CPU reference, one `rht` call per `block`-sized segment of each row.
fn cpu_reference(
    src: &[f16],
    signs: &[f32],
    rows: usize,
    width: usize,
    block: usize,
    forward: bool,
) -> Vec<f32> {
    let mut out = Vec::with_capacity(src.len());
    for row in 0..rows {
        for seg in 0..width / block {
            let start = row * width + seg * block;
            let seg_x: Vec<f32> = src[start..start + block]
                .iter()
                .map(|v| v.to_f32())
                .collect();
            let seg_s = &signs[seg * block..(seg + 1) * block];
            let y = if forward {
                turbospark_compute::kv_quant::rht_forward(&seg_x, seg_s)
            } else {
                turbospark_compute::kv_quant::rht_inverse(&seg_x, seg_s)
            };
            out.extend(y);
        }
    }
    out
}

fn assert_parity(src: &[f16], signs: &[f32], rows: u32, width: u32, block: u32, forward: bool) {
    let mut context = MetalContext::new().expect("Metal device available on this machine");
    let gpu = run_kernel(&mut context, src, signs, rows, width, block, forward);
    // Round the CPU reference through f16, the storage both sides share: the
    // transform's output magnitude reaches a few units, where fp16's ~1e-3
    // relative quantum exceeds the FP16_REDUCTION bound on its own. The
    // comparison is then GPU-fp16 against CPU-fp16 at an f32 interior, which
    // is the claim the kernel makes.
    let cpu: Vec<f16> = cpu_reference(
        src,
        signs,
        rows as usize,
        width as usize,
        block as usize,
        forward,
    )
    .iter()
    .map(|&v| f16::from_f32(v))
    .collect();
    let err = turbospark_compute::max_abs_diff(
        &gpu.iter().map(|v| v.to_f32()).collect::<Vec<f32>>(),
        &cpu.iter().map(|v| v.to_f32()).collect::<Vec<f32>>(),
    );
    assert!(
        err < turbospark_compute::Tolerance::FP16_REDUCTION,
        "forward={forward} err = {err}"
    );
}

/// The real contract's shape: width 5120, block 1024, five segments per row.
#[test]
fn forward_matches_the_cpu_reference_at_the_real_width() {
    let signs: Vec<f32> = (0..5120)
        .map(|i| if i % 3 == 0 { -1.0 } else { 1.0 })
        .collect();
    let src: Vec<f16> = (0..5120)
        .map(|i| f16::from_f32(((i % 97) as f32 - 48.0) * 0.05))
        .collect();
    assert_parity(&src, &signs, 1, 5120, 1024, true);
}

#[test]
fn inverse_matches_the_cpu_reference_at_the_real_width() {
    let signs: Vec<f32> = (0..5120)
        .map(|i| if i % 3 == 0 { -1.0 } else { 1.0 })
        .collect();
    let src: Vec<f16> = (0..5120)
        .map(|i| f16::from_f32(((i % 89) as f32 - 44.0) * 0.05))
        .collect();
    assert_parity(&src, &signs, 1, 5120, 1024, false);
}

/// Forward and inverse are NOT inverses of each other (the sign conjugation
/// does not square to the identity); they are two DIFFERENT maps, and the
/// round trip through the wrong one must not accidentally agree.
#[test]
fn inverse_is_not_forward_on_the_same_input() {
    let signs: Vec<f32> = (0..1024)
        .map(|i| if i % 2 == 0 { -1.0 } else { 1.0 })
        .collect();
    let src: Vec<f16> = (0..1024)
        .map(|i| f16::from_f32(((i % 31) as f32 - 15.0) * 0.1))
        .collect();
    let mut context = MetalContext::new().expect("device");
    let fwd = run_kernel(&mut context, &src, &signs, 1, 1024, 1024, true);
    let inv = run_kernel(&mut context, &src, &signs, 1, 1024, 1024, false);
    let differing = fwd
        .iter()
        .zip(&inv)
        .filter(|(a, b)| (a.to_f32() - b.to_f32()).abs() > 1e-3)
        .count();
    assert!(
        differing > 512,
        "forward and inverse agree almost everywhere (only {differing} differ); the two \
         directions are not being distinguished"
    );
}

/// The FFN activation width (17408 = 17 blocks) and a wider multi-row
/// dispatch, the prefill shape.
#[test]
fn multi_row_matches_at_the_ffn_width() {
    let signs: Vec<f32> = (0..17408)
        .map(|i| if i % 5 == 0 { -1.0 } else { 1.0 })
        .collect();
    let src: Vec<f16> = (0..4 * 17408)
        .map(|i| f16::from_f32(((i % 71) as f32 - 35.0) * 0.02))
        .collect();
    assert_parity(&src, &signs, 4, 17408, 1024, true);
}

/// Block 512, the smallest width the bundled runtime validates.
#[test]
fn matches_at_block_512() {
    let signs: Vec<f32> = (0..5120)
        .map(|i| if i % 7 == 0 { -1.0 } else { 1.0 })
        .collect();
    let src: Vec<f16> = (0..5120)
        .map(|i| f16::from_f32(((i % 53) as f32 - 26.0) * 0.03))
        .collect();
    assert_parity(&src, &signs, 1, 5120, 512, true);
}
