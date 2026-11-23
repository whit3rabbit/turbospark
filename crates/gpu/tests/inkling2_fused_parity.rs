#![cfg(target_os = "macos")]
//! Bit-parity tests for the fused Inkling 2 decode kernels (the oMLX 8dddad9a
//! launch-count work, re-implemented here). Each fused kernel must produce
//! exactly the bits of the chain of separate kernels it replaces, on real
//! Metal hardware, so the decode path, verify and DSpark stay bit-identical
//! whichever form runs. Closeness is not the contract: equality is.

use half::{bf16, f16};
use turbospark_gpu::{
    encode_inkling2_copy_half, encode_inkling2_depth_mix, encode_inkling2_depth_mix_norm,
    encode_inkling2_l2_clip, encode_inkling2_qkv_post, encode_inkling2_short_conv_decode,
    encode_inkling2_short_conv_residual, encode_residual_add, encode_rms_norm_bf16w,
    encode_rms_norm_no_scale_perhead, Inkling2ConvResidual, Inkling2QkvPostBuffers,
    Inkling2QkvPostShape, MetalContext, INKLING2_MIX_NORM_MAX_H,
};

fn lcg(state: &mut u32) -> u32 {
    *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *state
}

/// Uniform in [-1, 1).
fn unit(s: &mut u32) -> f32 {
    ((lcg(s) >> 8) as f32 / (1u32 << 23) as f32) - 1.0
}

fn le16(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|&x| f16::from_f32(x).to_bits().to_le_bytes())
        .collect()
}

fn leb16(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|&x| bf16::from_f32(x).to_bits().to_le_bytes())
        .collect()
}

fn bits(buffer: &metal::Buffer, n: usize) -> Vec<u16> {
    let ptr = buffer.contents() as *const u16;
    unsafe { std::slice::from_raw_parts(ptr, n) }.to_vec()
}

/// Per-head magnitudes that put some heads past the clip and some under it.
fn heads_row(s: &mut u32, heads: usize, dim: usize, base: f32, step: f32) -> Vec<f32> {
    (0..heads * dim)
        .map(|i| unit(s) * (base + step * (i / dim) as f32))
        .collect()
}

fn l2(v: &[f32]) -> f32 {
    v.iter().map(|a| a * a).sum::<f32>().sqrt()
}

/// One side's mutable state: the activations the kernels clip in place, the
/// conv tails and the KV row.
struct Side {
    q: metal::Buffer,
    r: metal::Buffer,
    tail_k: metal::Buffer,
    tail_v: metal::Buffer,
    k_out: metal::Buffer,
    v_out: metal::Buffer,
}

fn run_qkv_post(full: bool, steps: usize, tau_skew: f32) -> Result<(), String> {
    run_qkv_post_seeded(full, steps, tau_skew, 0x5eed, false, 1.0)
}

fn run_qkv_post_seeded(
    full: bool,
    steps: usize,
    tau_skew: f32,
    seed: u32,
    zero_tails: bool,
    spread: f32,
) -> Result<(), String> {
    let (heads, kvh, hd, d_rel, taps) = (24usize, 8usize, 128usize, 16usize, 4usize);
    let kvd = kvh * hd;
    let (qk_clip, rel_clip, eps) = (22.63f32, 4.0f32, 1e-6f32);
    let tau = if full { 1.37 } else { 1.0 };
    let mut s = seed;

    let mut context = MetalContext::new().expect("Metal device");
    let q_scale: Vec<f32> = (0..heads * hd)
        .map(|_| 0.8 + 0.4 * unit(&mut s).abs())
        .collect();
    let r_scale: Vec<f32> = (0..heads * d_rel)
        .map(|_| 0.8 + 0.4 * unit(&mut s).abs())
        .collect();
    let k_w: Vec<f32> = (0..kvd * taps).map(|_| unit(&mut s) * 0.8).collect();
    let v_w: Vec<f32> = (0..kvd * taps).map(|_| unit(&mut s) * 0.8).collect();
    let tail0: Vec<f32> = (0..(taps - 1) * kvd)
        .map(|_| if zero_tails { 0.0 } else { unit(&mut s) })
        .collect();
    let qs_buf = context.new_buffer_with_data(&leb16(&q_scale));
    let rs_buf = context.new_buffer_with_data(&leb16(&r_scale));
    let kw_buf = context.new_buffer_with_data(&leb16(&k_w));
    let vw_buf = context.new_buffer_with_data(&leb16(&v_w));

    let new_side = |context: &mut MetalContext| Side {
        q: context.new_output_buffer((heads * hd * 2) as u64),
        r: context.new_output_buffer((heads * d_rel * 2) as u64),
        tail_k: context.new_buffer_with_data(&le16(&tail0)),
        tail_v: context.new_buffer_with_data(&le16(&tail0)),
        k_out: context.new_output_buffer((kvd * 2) as u64),
        v_out: context.new_output_buffer((kvd * 2) as u64),
    };
    let a = new_side(&mut context);
    let b = new_side(&mut context);

    let (mut q_clipped, mut q_kept) = (0, 0);
    for _step in 0..steps {
        let q = heads_row(&mut s, heads, hd, 0.6, 0.12);
        let r = heads_row(&mut s, heads, d_rel, 0.4, 0.09);
        let mag: Vec<f32> = (0..2 * kvh)
            .map(|_| 2.5 * spread.powf(unit(&mut s)))
            .collect();
        let k_pre: Vec<f32> = (0..kvd).map(|i| unit(&mut s) * mag[i / hd]).collect();
        let v_pre: Vec<f32> = (0..kvd).map(|i| unit(&mut s) * mag[kvh + i / hd]).collect();
        let dk: Vec<f32> = (0..kvh * taps).map(|_| unit(&mut s) * 0.6).collect();
        let dv: Vec<f32> = (0..kvh * taps).map(|_| unit(&mut s) * 0.6).collect();
        for h in 0..heads {
            let scaled: Vec<f32> = (0..hd)
                .map(|i| q[h * hd + i] * q_scale[h * hd + i])
                .collect();
            if l2(&scaled) > qk_clip {
                q_clipped += 1;
            } else {
                q_kept += 1;
            }
        }
        for side in [&a, &b] {
            write(&side.q, &le16(&q));
            write(&side.r, &le16(&r));
        }
        let kp = context.new_buffer_with_data(&le16(&k_pre));
        let vp = context.new_buffer_with_data(&le16(&v_pre));
        let dk_buf = context.new_buffer_with_data(&le16(&dk));
        let dv_buf = context.new_buffer_with_data(&le16(&dv));

        // Side a: the six-dispatch chain the decode path ran before.
        let pass = context.begin_pass();
        let dyn_k = full.then_some(((&dk_buf, 0u64), kvh as u32));
        let dyn_v = full.then_some(((&dv_buf, 0u64), kvh as u32));
        encode_inkling2_short_conv_decode(
            &mut context,
            &pass,
            (&kp, 0),
            (&a.tail_k, 0),
            (&kw_buf, 0),
            dyn_k,
            (&a.k_out, 0),
            kvd as u32,
            taps as u32,
        )
        .unwrap();
        encode_inkling2_short_conv_decode(
            &mut context,
            &pass,
            (&vp, 0),
            (&a.tail_v, 0),
            (&vw_buf, 0),
            dyn_v,
            (&a.v_out, 0),
            kvd as u32,
            taps as u32,
        )
        .unwrap();
        encode_inkling2_l2_clip(
            &mut context,
            &pass,
            (&a.q, 0),
            Some(((&qs_buf, 0), hd as u32)),
            heads as u32,
            hd as u32,
            qk_clip,
            tau,
        )
        .unwrap();
        encode_inkling2_l2_clip(
            &mut context,
            &pass,
            (&a.k_out, 0),
            None,
            kvh as u32,
            hd as u32,
            qk_clip,
            1.0,
        )
        .unwrap();
        encode_rms_norm_no_scale_perhead(
            &mut context,
            &pass,
            (&a.v_out, 0),
            (&a.v_out, 0),
            kvh as u32,
            hd as u32,
            eps,
        )
        .unwrap();
        encode_inkling2_l2_clip(
            &mut context,
            &pass,
            (&a.r, 0),
            Some(((&rs_buf, 0), d_rel as u32)),
            heads as u32,
            d_rel as u32,
            rel_clip,
            tau,
        )
        .unwrap();

        // Side b: the fused kernel.
        encode_inkling2_qkv_post(
            &mut context,
            &pass,
            &Inkling2QkvPostBuffers {
                q: (&b.q, 0),
                q_scale: (&qs_buf, 0),
                r: (&b.r, 0),
                r_scale: (&rs_buf, 0),
                k_pre: (&kp, 0),
                v_pre: (&vp, 0),
                tail_k: (&b.tail_k, 0),
                tail_v: (&b.tail_v, 0),
                k_w: (&kw_buf, 0),
                v_w: (&vw_buf, 0),
                dyn_k: (&dk_buf, 0),
                dyn_v: (&dv_buf, 0),
                k_out: (&b.k_out, 0),
                v_out: (&b.v_out, 0),
            },
            Inkling2QkvPostShape {
                heads: heads as u32,
                kv_heads: kvh as u32,
                head_dim: hd as u32,
                d_rel: d_rel as u32,
                taps: taps as u32,
                dyn_heads: if full { kvh as u32 } else { 0 },
                qk_clip,
                rel_clip,
                tau: tau * tau_skew,
                rms_eps: eps,
            },
        )
        .unwrap();
        pass.commit_and_wait();

        for (x, y, n, what) in [
            (&a.q, &b.q, heads * hd, "q"),
            (&a.r, &b.r, heads * d_rel, "r"),
            (&a.k_out, &b.k_out, kvd, "k row"),
            (&a.v_out, &b.v_out, kvd, "v row"),
            (&a.tail_k, &b.tail_k, (taps - 1) * kvd, "k tail"),
            (&a.tail_v, &b.tail_v, (taps - 1) * kvd, "v tail"),
        ] {
            let (p, o) = (bits(x, n), bits(y, n));
            let diff = p.iter().zip(&o).filter(|(u, v)| u != v).count();
            if diff != 0 {
                return Err(format!("{what}: {diff} of {n} halves differ"));
            }
        }
    }
    assert!(
        spread != 1.0 || (q_clipped > 0 && q_kept > 0),
        "fixture must exercise both clip branches ({q_clipped}/{q_kept})"
    );
    Ok(())
}

fn write(buffer: &metal::Buffer, data: &[u8]) {
    let ptr = buffer.contents() as *mut u8;
    unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len()) };
}

#[test]
fn qkv_post_swa_is_bit_identical_to_the_chain() {
    run_qkv_post(false, 4, 1.0).unwrap();
}

#[test]
fn qkv_post_full_dynamic_taps_and_tau_is_bit_identical_to_the_chain() {
    run_qkv_post(true, 4, 1.0).unwrap();
}

/// The comparison can fail: a tau off by one part in 1e3 on the full layer
/// must show up as differing q bits. Guards against a harness that compares
/// a buffer with itself.
#[test]
fn qkv_post_parity_detects_a_tau_mutant() {
    assert!(run_qkv_post(true, 1, 1.001).is_err());
}

#[test]
fn qkv_post_refuses_out_of_contract_shapes() {
    let mut context = MetalContext::new().expect("Metal device");
    let x = context.new_buffer_with_data(&[0u8; 64]);
    let bufs = Inkling2QkvPostBuffers {
        q: (&x, 0),
        q_scale: (&x, 0),
        r: (&x, 0),
        r_scale: (&x, 0),
        k_pre: (&x, 0),
        v_pre: (&x, 0),
        tail_k: (&x, 0),
        tail_v: (&x, 0),
        k_w: (&x, 0),
        v_w: (&x, 0),
        dyn_k: (&x, 0),
        dyn_v: (&x, 0),
        k_out: (&x, 0),
        v_out: (&x, 0),
    };
    let ok = Inkling2QkvPostShape {
        heads: 24,
        kv_heads: 8,
        head_dim: 128,
        d_rel: 16,
        taps: 4,
        dyn_heads: 0,
        qk_clip: 1.0,
        rel_clip: 1.0,
        tau: 1.0,
        rms_eps: 1e-6,
    };
    let pass = context.begin_pass();
    for bad in [
        Inkling2QkvPostShape {
            head_dim: 100,
            ..ok
        },
        Inkling2QkvPostShape {
            head_dim: 512,
            ..ok
        },
        Inkling2QkvPostShape { taps: 1, ..ok },
        Inkling2QkvPostShape { dyn_heads: 3, ..ok },
    ] {
        assert!(encode_inkling2_qkv_post(&mut context, &pass, &bufs, bad).is_err());
    }
    pass.commit_and_wait();
}

// --- Depth mix + RMS norm ---------------------------------------------------

fn le32(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// The chain: copy `cur` into state row `n_states - 1`, depth mix, then one
/// `rmsnorm_bf16w` per query, against the fused kernel reading `cur` in
/// place. Both the norm output and the optional mix output must match.
fn run_mix_norm(n_states: usize, n_q: usize, write_mix: bool, eps_skew: f32) -> Result<(), String> {
    let h = 2304usize;
    let eps = 1e-6f32;
    let mut s = 0xd1ce_u32 ^ (n_states as u32 * 31 + n_q as u32);
    // States of differing magnitude, so the per-state RMS factor matters.
    let states: Vec<f32> = (0..n_states * h)
        .map(|i| unit(&mut s) * (0.3 + 0.4 * (i / h) as f32))
        .collect();
    let cur: Vec<f32> = (0..h).map(|_| unit(&mut s) * 1.7).collect();
    let queries: Vec<f32> = (0..n_q * h).map(|_| unit(&mut s) * 0.05).collect();
    let weights: Vec<Vec<f32>> = (0..n_q)
        .map(|_| (0..h).map(|_| 0.5 + unit(&mut s).abs()).collect())
        .collect();

    let mut context = MetalContext::new().expect("Metal device");
    let st_a = context.new_buffer_with_data(&le16(&states));
    let st_b = context.new_buffer_with_data(&le16(&states));
    let cur_buf = context.new_buffer_with_data(&le16(&cur));
    let q_buf = context.new_buffer_with_data(&le32(&queries));
    let w_bufs: Vec<metal::Buffer> = weights
        .iter()
        .map(|w| context.new_buffer_with_data(&leb16(w)))
        .collect();
    let mix_a = context.new_output_buffer((n_q * h * 2) as u64);
    let norm_a = context.new_output_buffer((n_q * h * 2) as u64);
    let mix_b = context.new_output_buffer((n_q * h * 2) as u64);
    let norm_b = context.new_output_buffer((n_q * h * 2) as u64);

    let pass = context.begin_pass();
    encode_inkling2_copy_half(
        &mut context,
        &pass,
        (&cur_buf, 0),
        (&st_a, ((n_states - 1) * h * 2) as u64),
        h as u32,
    )
    .unwrap();
    encode_inkling2_depth_mix(
        &mut context,
        &pass,
        (&st_a, 0),
        (&q_buf, 0),
        (&mix_a, 0),
        n_states as u32,
        n_q as u32,
        h as u32,
    )
    .unwrap();
    for (j, w) in w_bufs.iter().enumerate() {
        encode_rms_norm_bf16w(
            &mut context,
            &pass,
            (&mix_a, (j * h * 2) as u64),
            (w, 0),
            (&norm_a, (j * h * 2) as u64),
            h as u32,
            eps,
        )
        .unwrap();
    }
    let ws: Vec<(&metal::Buffer, u64)> = w_bufs.iter().map(|b| (b, 0u64)).collect();
    encode_inkling2_depth_mix_norm(
        &mut context,
        &pass,
        (&st_b, 0),
        (&cur_buf, 0),
        (&q_buf, 0),
        write_mix.then_some((&mix_b, 0u64)),
        &ws,
        (&norm_b, 0),
        n_states as u32,
        h as u32,
        eps * eps_skew,
    )
    .unwrap();
    pass.commit_and_wait();

    let cmp = |x: &metal::Buffer, y: &metal::Buffer, what: &str| -> Result<(), String> {
        let n = n_q * h;
        let (p, o) = (bits(x, n), bits(y, n));
        let diff = p.iter().zip(&o).filter(|(u, v)| u != v).count();
        if diff == 0 {
            Ok(())
        } else {
            Err(format!("{what}: {diff} of {n} halves differ"))
        }
    };
    cmp(&norm_a, &norm_b, "norm")?;
    if write_mix {
        cmp(&mix_a, &mix_b, "mix")?;
    }
    // The fused kernel must not write the state array.
    let (p, o) = (bits(&st_b, n_states * h), le16(&states));
    let want: Vec<u16> = o
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();
    if p != want {
        return Err("fused kernel wrote the state array".into());
    }
    Ok(())
}

#[test]
fn mix_norm_single_query_is_bit_identical_to_the_chain() {
    for n_states in [1usize, 2, 5, 13] {
        run_mix_norm(n_states, 1, false, 1.0).unwrap();
    }
}

#[test]
fn mix_norm_three_queries_with_their_own_norms_and_mix_output() {
    for n_states in [3usize, 9, 16] {
        run_mix_norm(n_states, 3, true, 1.0).unwrap();
    }
}

/// An eps mutant on the norm must change bits on states this small.
#[test]
fn mix_norm_parity_detects_an_eps_mutant() {
    assert!(run_mix_norm(4, 1, false, 1e6).is_err());
}

#[test]
fn mix_norm_refuses_out_of_contract_shapes() {
    let mut context = MetalContext::new().expect("Metal device");
    let x = context.new_buffer_with_data(&[0u8; 64]);
    let pass = context.begin_pass();
    let one = [(&x, 0u64)];
    let four = [(&x, 0u64); 4];
    for (ws, n, h) in [
        (&one[..], 0u32, 2304u32),
        (&one[..], 17, 2304),
        (&four[..], 2, 2304),
        (&[][..], 2, 2304),
        (&one[..], 2, 128),
        (&one[..], 2, INKLING2_MIX_NORM_MAX_H + 256),
    ] {
        assert!(encode_inkling2_depth_mix_norm(
            &mut context,
            &pass,
            (&x, 0),
            (&x, 0),
            (&x, 0),
            None,
            ws,
            (&x, 0),
            n,
            h,
            1e-6
        )
        .is_err());
    }
    pass.commit_and_wait();
}

/// Many seeds, zero starting tails (as at position 0), and per-head K/V
/// magnitudes from 0.05 to 30 times the base: the value ranges real
/// activations reach and a single fixture can miss.
#[test]
fn qkv_post_holds_over_seeds_zero_tails_and_wide_magnitudes() {
    for seed in 1..=24u32 {
        for full in [false, true] {
            run_qkv_post_seeded(full, 8, 1.0, seed * 7919, seed % 2 == 0, 12.0)
                .unwrap_or_else(|e| panic!("seed {seed} full {full}: {e}"));
        }
    }
}

// --- Short conv + residual ---------------------------------------------------

fn run_conv_residual(start_block: bool, steps: usize, seed: u32) {
    let (ch, taps) = (2304usize, 4usize);
    let mut s = seed;
    let mut context = MetalContext::new().expect("Metal device");
    let w: Vec<f32> = (0..ch * taps).map(|_| unit(&mut s) * 0.8).collect();
    let w_buf = context.new_buffer_with_data(&leb16(&w));
    let tail0: Vec<f32> = (0..(taps - 1) * ch).map(|_| unit(&mut s)).collect();
    let cur0: Vec<f32> = (0..ch).map(|_| unit(&mut s) * 3.0).collect();
    let (tail_a, tail_b) = (
        context.new_buffer_with_data(&le16(&tail0)),
        context.new_buffer_with_data(&le16(&tail0)),
    );
    let (cur_a, cur_b) = (
        context.new_buffer_with_data(&le16(&cur0)),
        context.new_buffer_with_data(&le16(&cur0)),
    );
    let (save_a, save_b) = (
        context.new_output_buffer((ch * 2) as u64),
        context.new_output_buffer((ch * 2) as u64),
    );
    let tmp = context.new_output_buffer((ch * 2) as u64);
    for _ in 0..steps {
        let x: Vec<f32> = (0..ch).map(|_| unit(&mut s) * 2.0).collect();
        let x_buf = context.new_buffer_with_data(&le16(&x));
        let pass = context.begin_pass();
        encode_inkling2_short_conv_decode(
            &mut context,
            &pass,
            (&x_buf, 0),
            (&tail_a, 0),
            (&w_buf, 0),
            None,
            (&tmp, 0),
            ch as u32,
            taps as u32,
        )
        .unwrap();
        if start_block {
            encode_inkling2_copy_half(&mut context, &pass, (&cur_a, 0), (&save_a, 0), ch as u32)
                .unwrap();
            encode_inkling2_copy_half(&mut context, &pass, (&tmp, 0), (&cur_a, 0), ch as u32)
                .unwrap();
        } else {
            encode_residual_add(&mut context, &pass, (&cur_a, 0), (&tmp, 0), ch as u32).unwrap();
        }
        let step = if start_block {
            Inkling2ConvResidual::StartBlock { save: (&save_b, 0) }
        } else {
            Inkling2ConvResidual::Add
        };
        encode_inkling2_short_conv_residual(
            &mut context,
            &pass,
            (&x_buf, 0),
            (&tail_b, 0),
            (&w_buf, 0),
            (&cur_b, 0),
            step,
            ch as u32,
            taps as u32,
        )
        .unwrap();
        pass.commit_and_wait();
        assert_eq!(bits(&cur_a, ch), bits(&cur_b, ch), "stream");
        assert_eq!(
            bits(&tail_a, ch * (taps - 1)),
            bits(&tail_b, ch * (taps - 1)),
            "tail"
        );
        if start_block {
            assert_eq!(bits(&save_a, ch), bits(&save_b, ch), "saved block");
        }
    }
}

#[test]
fn conv_residual_add_is_bit_identical_to_conv_then_add() {
    for seed in 1..=8u32 {
        run_conv_residual(false, 6, seed * 104_729);
    }
}

#[test]
fn conv_residual_start_block_is_bit_identical_to_conv_then_copies() {
    for seed in 1..=8u32 {
        run_conv_residual(true, 6, seed * 7_919);
    }
}

// --- Two-row decode GEMV -----------------------------------------------------

#[test]
fn bf16_gemv_multi2_is_bit_identical_to_the_one_row_kernel() {
    use turbospark_gpu::{
        encode_inkling2_bf16_gemv_multi, encode_inkling2_bf16_gemv_multi2, Inkling2GemvSegment,
    };
    let mut context = MetalContext::new().expect("Metal device");
    let mut s = 0x9e37_u32;
    // Odd row counts (a lone last row), three inputs, the Mini's widths.
    let segs_shape = [
        (3071usize, 2304usize, 0usize),
        (1025, 2304, 1),
        (17, 3072, 2),
        (1, 64, 0),
    ];
    let mut w = Vec::new();
    let mut offs = Vec::new();
    for &(r, c, _) in &segs_shape {
        offs.push(w.len() * 2);
        w.extend((0..r * c).map(|_| unit(&mut s) * 0.05));
    }
    let w_buf = context.new_buffer_with_data(&leb16(&w));
    let xs: Vec<metal::Buffer> = [2304usize, 2304, 3072]
        .iter()
        .map(|&c| {
            context.new_buffer_with_data(&le16(&(0..c).map(|_| unit(&mut s)).collect::<Vec<_>>()))
        })
        .collect();
    let run = |context: &mut MetalContext, two: bool| -> Vec<Vec<u16>> {
        let ys: Vec<metal::Buffer> = segs_shape
            .iter()
            .map(|&(r, _, _)| context.new_output_buffer((r * 2) as u64))
            .collect();
        let segs: Vec<Inkling2GemvSegment<'_>> = segs_shape
            .iter()
            .zip(&ys)
            .zip(&offs)
            .map(|((&(r, c, x), y), &o)| Inkling2GemvSegment {
                w_offset: o as u64,
                rows: r as u32,
                cols: c as u32,
                x,
                y: (y, 0),
            })
            .collect();
        let xr: Vec<(&metal::Buffer, u64)> = xs.iter().map(|b| (b, 0u64)).collect();
        let pass = context.begin_pass();
        if two {
            encode_inkling2_bf16_gemv_multi2(context, &pass, &w_buf, &xr, &segs).unwrap();
        } else {
            encode_inkling2_bf16_gemv_multi(context, &pass, &w_buf, &xr, &segs).unwrap();
        }
        pass.commit_and_wait();
        ys.iter()
            .zip(&segs_shape)
            .map(|(y, &(r, _, _))| bits(y, r))
            .collect()
    };
    let one = run(&mut context, false);
    let two = run(&mut context, true);
    assert_eq!(one, two, "two-row GEMV differs from the one-row kernel");
}
