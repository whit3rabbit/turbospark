#![cfg(target_os = "macos")]
//! `rmsnorm_bf16w_perhead` with `head_dim = hidden` must equal `rmsnorm_bf16w`
//! on each row, bit for bit. The batched Qwen prefill layer relies on it to
//! norm M rows in one dispatch instead of M. Rows differ in magnitude so a
//! reduction that spanned rows (one statistic for all of them) would show.

use half::f16;
use turbospark_gpu::{encode_rms_norm_bf16w, encode_rms_norm_bf16w_perhead, MetalContext};

fn f32_to_bf16_bits(v: f32) -> u16 {
    (v.to_bits() >> 16) as u16
}

fn bytes(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|b| b.to_le_bytes()).collect()
}

#[test]
fn perhead_at_hidden_width_is_bit_identical_to_the_whole_vector_norm_per_row() {
    let mut context = MetalContext::new().expect("Metal device available on this machine");
    let hidden = 5_120usize;
    let rows = 7usize;
    let eps = 1e-6f32;

    let x: Vec<u16> = (0..rows * hidden)
        .map(|i| {
            let row = (i / hidden) as f32;
            f16::from_f32((1.0 + row * row) * 0.37 * ((i as f32) * 0.013).sin()).to_bits()
        })
        .collect();
    let weight: Vec<u16> = (0..hidden)
        .map(|i| f32_to_bf16_bits(1.1 + 0.4 * ((i as f32) * 0.07).cos()))
        .collect();

    let x_buf = context.new_buffer_with_data(&bytes(&x));
    let w_buf = context.new_buffer_with_data(&bytes(&weight));
    let per_row = context.new_buffer_with_data(&bytes(&vec![0u16; rows * hidden]));
    let batched = context.new_buffer_with_data(&bytes(&vec![0u16; rows * hidden]));

    let pass = context.begin_pass();
    for r in 0..rows {
        let off = (r * hidden * 2) as u64;
        encode_rms_norm_bf16w(
            &mut context,
            &pass,
            (&x_buf, off),
            (&w_buf, 0),
            (&per_row, off),
            hidden as u32,
            eps,
        )
        .expect("per-row norm");
    }
    encode_rms_norm_bf16w_perhead(
        &mut context,
        &pass,
        (&x_buf, 0),
        (&w_buf, 0),
        (&batched, 0),
        rows as u32,
        hidden as u32,
        eps,
    )
    .expect("batched norm");
    pass.commit_and_wait();

    let n = rows * hidden;
    let a = turbospark_gpu::read_buffer_f16(&per_row, 0, n);
    let b = turbospark_gpu::read_buffer_f16(&batched, 0, n);
    let bits = |v: &[f16]| v.iter().map(|h| h.to_bits()).collect::<Vec<u16>>();
    assert_eq!(bits(&a), bits(&b));
    assert!(
        a.iter().any(|h| h.to_f32() != 0.0),
        "an all-zero output proves nothing"
    );
}
