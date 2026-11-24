#![cfg(target_os = "macos")]
//! Attention-score soft cap (Gemma 2's `attn_logit_softcapping`): every score
//! `s` becomes `cap * tanh(s / cap)` after the scale and before the softmax.
//! Against a CPU reference computed here, on real hardware.
//!
//! The cap is a function constant that exists only for a dispatch that asks
//! for it, so two things are checked separately and both matter:
//! - a CAPPED dispatch matches the reference, with scores large enough that the
//!   cap changes the answer (a case where cap and no cap agree proves nothing);
//! - an UNCAPPED dispatch is bit-identical to what it was before the constant
//!   existed, i.e. `with_softcap(0.0)` is the identity on every other family.

use half::f16;
use turbospark_gpu::{AttentionScratch, MetalContext};

fn to_le_f16(v: &[f16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect()
}

struct Case {
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    seq_len: u32,
}

fn inputs(c: &Case, magnitude: f32) -> (Vec<f16>, Vec<f16>, Vec<f16>) {
    let q = (0..(c.num_q_heads * c.head_dim) as usize)
        .map(|i| f16::from_f32(((i as f32) * 0.19).sin() * magnitude))
        .collect();
    let kv = (c.seq_len * c.num_kv_heads * c.head_dim) as usize;
    let k = (0..kv)
        .map(|i| f16::from_f32(((i as f32) * 0.07).cos() * magnitude))
        .collect();
    let v = (0..kv)
        .map(|i| f16::from_f32(((i as f32) * 0.11).sin()))
        .collect();
    (q, k, v)
}

/// Plain softmax attention with an optional score cap, GQA-mapped, f64.
fn reference(c: &Case, q: &[f16], k: &[f16], v: &[f16], scale: f32, cap: f32) -> Vec<f32> {
    let hd = c.head_dim as usize;
    let group = (c.num_q_heads / c.num_kv_heads) as usize;
    let mut out = vec![0f32; (c.num_q_heads * c.head_dim) as usize];
    for h in 0..c.num_q_heads as usize {
        let kvh = h / group;
        let mut scores = Vec::new();
        for p in 0..c.seq_len as usize {
            let kb = (p * c.num_kv_heads as usize + kvh) * hd;
            let mut s = 0f64;
            for i in 0..hd {
                s += f64::from(q[h * hd + i].to_f32()) * f64::from(k[kb + i].to_f32());
            }
            s *= f64::from(scale);
            if cap != 0.0 {
                s = f64::from(cap) * (s / f64::from(cap)).tanh();
            }
            scores.push(s);
        }
        let m = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let denom: f64 = scores.iter().map(|s| (s - m).exp()).sum();
        for i in 0..hd {
            let mut acc = 0f64;
            for (p, s) in scores.iter().enumerate() {
                let vb = (p * c.num_kv_heads as usize + kvh) * hd;
                acc += (s - m).exp() / denom * f64::from(v[vb + i].to_f32());
            }
            out[h * hd + i] = acc as f32;
        }
    }
    out
}

fn run(c: &Case, q: &[f16], k: &[f16], v: &[f16], scale: f32, cap: f32) -> Vec<f32> {
    let mut context = MetalContext::new().expect("Metal device");
    let q_buf = context.new_buffer_with_data(&to_le_f16(q));
    let k_buf = context.new_buffer_with_data(&to_le_f16(k));
    let v_buf = context.new_buffer_with_data(&to_le_f16(v));
    let out_buf = context.new_output_buffer((c.num_q_heads * c.head_dim) as u64 * 2);
    let scratch = AttentionScratch::new(&context, c.num_q_heads, c.head_dim).with_softcap(cap);
    let pass = context.begin_pass();
    turbospark_gpu::encode_attention_decode(
        &mut context,
        &pass,
        (&q_buf, 0),
        &k_buf,
        &v_buf,
        &scratch,
        (&out_buf, 0),
        c.head_dim,
        c.num_q_heads,
        c.num_kv_heads,
        c.seq_len,
        0,
        0,
        scale,
        None,
    )
    .expect("encode");
    pass.commit_and_wait();
    let bytes = out_buf.contents() as *const u8;
    (0..(c.num_q_heads * c.head_dim) as usize)
        .map(|i| {
            // SAFETY: `out_buf` holds `num_q_heads * head_dim` halfs and the
            // pass above completed.
            let lo = unsafe { *bytes.add(i * 2) };
            let hi = unsafe { *bytes.add(i * 2 + 1) };
            f16::from_bits(u16::from_le_bytes([lo, hi])).to_f32()
        })
        .collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

const CASES: [Case; 3] = [
    // Gemma 2 2B's shape: 8 q heads over 4 kv, head_dim 256.
    Case {
        num_q_heads: 8,
        num_kv_heads: 4,
        head_dim: 256,
        seq_len: 37,
    },
    // Long enough for several chunks, so the cap is applied per chunk and the
    // combine must still agree.
    Case {
        num_q_heads: 8,
        num_kv_heads: 4,
        head_dim: 256,
        seq_len: 700,
    },
    // MHA, a non-power-of-two head width.
    Case {
        num_q_heads: 4,
        num_kv_heads: 4,
        head_dim: 96,
        seq_len: 129,
    },
];

#[test]
fn a_capped_dispatch_matches_the_cpu_reference_where_the_cap_matters() {
    let scale = 0.0625f32;
    let cap = 2.0f32;
    for c in &CASES {
        // The cap only changes the OUTPUT where several scores are both large
        // and comparable (a cap is monotone, so a single dominant score wins
        // either way). Pick the first magnitude, per shape, at which the
        // reference itself says the cap matters, so the check cannot pass
        // vacuously on a fixture that agrees with or without it.
        let mut picked = None;
        for magnitude in [1.0f32, 1.5, 2.0, 3.0, 4.0, 6.0] {
            let (q, k, v) = inputs(c, magnitude);
            let capped = reference(c, &q, &k, &v, scale, cap);
            let uncapped = reference(c, &q, &k, &v, scale, 0.0);
            if max_abs_diff(&capped, &uncapped) > 5e-3 {
                picked = Some((q, k, v, capped, uncapped));
                break;
            }
        }
        let (q, k, v, capped, uncapped) =
            picked.unwrap_or_else(|| panic!("no magnitude exercises the cap at seq {}", c.seq_len));
        let got = run(c, &q, &k, &v, scale, cap);
        let diff = max_abs_diff(&got, &capped);
        assert!(diff < 5e-3, "seq {}: capped diff {diff}", c.seq_len);
        // And it is NOT the uncapped answer.
        assert!(
            max_abs_diff(&got, &uncapped) > 2e-3,
            "seq {}: the dispatch ignored the cap",
            c.seq_len
        );
    }
}

#[test]
fn a_zero_cap_is_bit_identical_to_the_uncapped_dispatch() {
    let c = &CASES[0];
    let (q, k, v) = inputs(c, 1.0);
    let uncapped = {
        let mut context = MetalContext::new().expect("Metal device");
        let q_buf = context.new_buffer_with_data(&to_le_f16(&q));
        let k_buf = context.new_buffer_with_data(&to_le_f16(&k));
        let v_buf = context.new_buffer_with_data(&to_le_f16(&v));
        let out_buf = context.new_output_buffer((c.num_q_heads * c.head_dim) as u64 * 2);
        // No `with_softcap` call at all: the pre-existing construction.
        let scratch = AttentionScratch::new(&context, c.num_q_heads, c.head_dim);
        let pass = context.begin_pass();
        turbospark_gpu::encode_attention_decode(
            &mut context,
            &pass,
            (&q_buf, 0),
            &k_buf,
            &v_buf,
            &scratch,
            (&out_buf, 0),
            c.head_dim,
            c.num_q_heads,
            c.num_kv_heads,
            c.seq_len,
            0,
            0,
            0.0625,
            None,
        )
        .expect("encode");
        pass.commit_and_wait();
        let p = out_buf.contents() as *const u16;
        (0..(c.num_q_heads * c.head_dim) as usize)
            // SAFETY: as in `run`.
            .map(|i| unsafe { *p.add(i) })
            .collect::<Vec<u16>>()
    };
    let capped_zero = run(c, &q, &k, &v, 0.0625, 0.0);
    let as_bits: Vec<u16> = capped_zero
        .iter()
        .map(|x| f16::from_f32(*x).to_bits())
        .collect();
    assert_eq!(as_bits, uncapped);
}
