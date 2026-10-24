//! The GDN causal conv must sum its taps in ONE order in decode and prefill.
//!
//! `gdn_conv_mix_decode` and `gdn_conv_mix_prefill` compute the same
//! depthwise conv (+ SiLU), and a speculative verify runs the prefill form
//! where sequential decode ran the decode form. If the two associate the four
//! tap products differently, the f32 sums differ in the last bit now and
//! then; rounding to half hides that almost always, so it surfaces in about
//! one element in ten thousand. On `qwen38-27b-mtp` it was the first
//! divergence between a verify row and the decode step it stands in for
//! (layer 33, one `conv_out` element), and it is what made greedy speculation
//! leave the sequential stream on prose (measured 2026-10-23,
//! `docs/INT4_SMALL_M_MMA.md`). Early in a sequence the tail rows are zero,
//! zero products make the order irrelevant, and the two stay exact -- which is
//! why the earlier probe saw an onset "at three keys".
//!
//! Fixture-sized random data rarely lands on a rounding boundary, so this
//! runs the real channel count with a full, nonzero tail over several trials.

#![cfg(target_os = "macos")]

use turbospark_gpu::{
    encode_gdn_conv_decode, encode_gdn_conv_prefill, read_buffer_f16, GdnShape, MetalContext,
};

const K: usize = 4;

fn shape() -> GdnShape {
    // Qwen3.8-27B's linear-attention widths: qkv_dim 2*16*128 + 48*128 = 10240.
    GdnShape {
        num_k_heads: 16,
        num_v_heads: 48,
        key_head_dim: 128,
        value_head_dim: 128,
        conv_kernel_size: K as u32,
    }
}

fn xorshift(seed: u64) -> impl FnMut() -> f32 {
    let mut s = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s % 20001) as f32 / 10000.0 - 1.0
    }
}

fn halves(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|x| half::f16::from_f32(*x).to_bits().to_le_bytes())
        .collect()
}

fn bf16s(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|x| ((x.to_bits() >> 16) as u16).to_le_bytes())
        .collect()
}

#[test]
fn decode_and_prefill_conv_agree_bit_for_bit_at_the_real_width() {
    let mut context = MetalContext::new().expect("Metal device");
    let c = shape().qkv_dim() as usize;
    let rows = 3usize;
    let mut differing = 0usize;
    for trial in 0..32u64 {
        let mut next = xorshift(0xC0_0000 + trial);
        let tail: Vec<f32> = (0..(K - 1) * c).map(|_| next() * 4.0).collect();
        let qkv: Vec<f32> = (0..rows * c).map(|_| next() * 4.0).collect();
        let w: Vec<f32> = (0..c * K).map(|_| next() * 0.5).collect();
        let w_buf = context.new_buffer_with_data(&bf16s(&w));
        let qkv_buf = context.new_buffer_with_data(&halves(&qkv));

        // Prefill: all rows from the entering tail.
        let tail_p = context.new_buffer_with_data(&halves(&tail));
        let out_p = context.new_output_buffer((rows * c * 2) as u64);
        // Decode: one row at a time, shifting its own copy of the tail.
        let tail_d = context.new_buffer_with_data(&halves(&tail));
        let out_d = context.new_output_buffer((rows * c * 2) as u64);
        let pass = context.begin_pass();
        encode_gdn_conv_prefill(
            &mut context,
            &pass,
            shape(),
            (&tail_p, 0),
            (&qkv_buf, 0),
            (&w_buf, 0),
            (&out_p, 0),
            rows as u32,
            1,
        )
        .unwrap();
        for r in 0..rows {
            encode_gdn_conv_decode(
                &mut context,
                &pass,
                shape(),
                (&tail_d, 0),
                (&qkv_buf, (r * c * 2) as u64),
                (&w_buf, 0),
                (&out_d, (r * c * 2) as u64),
                1,
            )
            .unwrap();
        }
        pass.commit_and_wait();
        let p = read_buffer_f16(&out_p, 0, rows * c);
        let d = read_buffer_f16(&out_d, 0, rows * c);
        let n = p
            .iter()
            .zip(&d)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        println!("trial {trial}: {n} of {} conv outputs differ", rows * c);
        differing += n;
    }
    assert_eq!(
        differing, 0,
        "decode and prefill GDN conv disagree: they sum the taps in different orders"
    );
}
