#![cfg(target_os = "macos")]
//! Parity test for `inkling2_ops.metal` (router selection, L2 clip, short
//! convolution decode, depth mix) against `turbospark_compute::inkling2`, on
//! real Metal hardware.
//!
//! Each fixture is built so the branch it tests actually runs: the correction
//! bias changes WHICH experts win, some vectors are past the clip threshold and
//! some are not, the conv runs several steps so the tail carries state, and
//! the depth-mix states differ in magnitude so the RMS norm matters.

use half::{bf16, f16};
use turbospark_compute::inkling2::{block_attn_res_mix, l2_clip, route_sigmoid, short_conv};
use turbospark_gpu::{
    encode_inkling2_depth_mix, encode_inkling2_l2_clip, encode_inkling2_router_select,
    encode_inkling2_short_conv_decode, MetalContext,
};

fn lcg(state: &mut u32) -> u32 {
    *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *state
}

/// Uniform in [-1, 1).
fn unit(s: &mut u32) -> f32 {
    ((lcg(s) >> 8) as f32 / (1u32 << 23) as f32) - 1.0
}

fn f16s(v: &[f32]) -> Vec<f16> {
    v.iter().map(|&x| f16::from_f32(x)).collect()
}

fn bf16s(v: &[f32]) -> Vec<bf16> {
    v.iter().map(|&x| bf16::from_f32(x)).collect()
}

fn le16(v: &[f16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect()
}

fn leb16(v: &[bf16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect()
}

fn le32(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn read_f16(buffer: &metal::Buffer, n: usize) -> Vec<f32> {
    let ptr = buffer.contents() as *const u16;
    let bits = unsafe { std::slice::from_raw_parts(ptr, n) };
    bits.iter().map(|&b| f16::from_bits(b).to_f32()).collect()
}

fn read_u32(buffer: &metal::Buffer, n: usize) -> Vec<u32> {
    let ptr = buffer.contents() as *const u32;
    unsafe { std::slice::from_raw_parts(ptr, n) }.to_vec()
}

fn assert_close(got: &[f32], want: &[f32], abs: f32, rel: f32, what: &str) {
    assert_eq!(got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let tol = abs.max(w.abs() * rel);
        assert!((g - w).abs() <= tol, "{what}[{i}]: got {g} want {w}");
    }
}

// --- Router ---------------------------------------------------------------

#[test]
fn router_matches_reference_and_the_bias_changes_selection() {
    let ne = 128usize;
    // Logits are a permutation of an evenly spaced grid, so sigmoid scores
    // are distinct by a margin no FP32 exp difference can cross.
    let logits: Vec<f32> = (0..ne)
        .map(|e| -3.0 + 6.0 * ((e * 37) % ne) as f32 / ne as f32)
        .collect();
    // A bias on every ninth expert pulls some low scorers into the top 8.
    let bias: Vec<f32> = (0..ne)
        .map(|e| if e % 9 == 0 { 0.37 } else { 0.0 })
        .collect();
    let (want_idx, want_w) = route_sigmoid(&logits, &bias, 8, 8.0);
    let (plain_idx, _) = route_sigmoid(&logits, &vec![0.0; ne], 8, 8.0);
    assert_ne!(
        want_idx, plain_idx,
        "fixture: the bias must change the winners"
    );

    let mut context = MetalContext::new().expect("Metal device");
    let l_buf = context.new_buffer_with_data(&le32(&logits));
    let b_buf = context.new_buffer_with_data(&le32(&bias));
    let i_buf = context.new_output_buffer(8 * 4);
    let w_buf = context.new_output_buffer(8 * 2);
    let pass = context.begin_pass();
    encode_inkling2_router_select(
        &mut context,
        &pass,
        (&l_buf, 0),
        (&b_buf, 0),
        (&i_buf, 0),
        (&w_buf, 0),
        ne as u32,
        8.0,
    )
    .expect("router");
    pass.commit_and_wait();

    let got_idx: Vec<usize> = read_u32(&i_buf, 8).iter().map(|&i| i as usize).collect();
    assert_eq!(got_idx, want_idx, "rank order");
    assert_close(&read_f16(&w_buf, 8), &want_w, 1e-3, 2e-3, "router weight");
    let total: f32 = read_f16(&w_buf, 8).iter().sum();
    assert!(
        (total - 8.0).abs() < 1e-2,
        "weights sum to route_scale: {total}"
    );
}

#[test]
fn router_refuses_fewer_experts_than_top_k() {
    let mut context = MetalContext::new().expect("Metal device");
    let buf = context.new_buffer_with_data(&[0u8; 64]);
    let pass = context.begin_pass();
    let r = encode_inkling2_router_select(
        &mut context,
        &pass,
        (&buf, 0),
        (&buf, 0),
        (&buf, 0),
        (&buf, 0),
        7,
        8.0,
    );
    pass.commit_and_wait();
    assert!(r.is_err());
}

// --- L2 clip --------------------------------------------------------------

fn run_l2_clip(n_vec: usize, dim: usize, threshold: f32, post: f32, scale_stride: Option<usize>) {
    let mut s = 91u32;
    // Vector v has magnitude growing with v, so the early ones sit under the
    // threshold and the later ones are clipped.
    let x: Vec<f32> = (0..n_vec * dim)
        .map(|i| unit(&mut s) * (0.2 + 0.25 * (i / dim) as f32) * 2.0)
        .collect();
    let x16 = f16s(&x);
    let x_r: Vec<f32> = x16.iter().map(|v| v.to_f32()).collect();
    let scale = scale_stride.map(|stride| {
        let len = if stride == 0 { dim } else { n_vec * dim };
        let v: Vec<f32> = (0..len)
            .map(|_| 0.75 + 0.5 * (unit(&mut s) + 1.0) / 2.0)
            .collect();
        (bf16s(&v), stride)
    });

    let mut want = Vec::new();
    let (mut clipped, mut kept) = (0, 0);
    for v in 0..n_vec {
        let row = &x_r[v * dim..(v + 1) * dim];
        let sc: Option<Vec<f32>> = scale.as_ref().map(|(b, stride)| {
            let off = v * stride;
            b[off..off + dim].iter().map(|x| x.to_f32()).collect()
        });
        let scaled: Vec<f32> = match &sc {
            Some(c) => row.iter().zip(c).map(|(a, b)| a * b).collect(),
            None => row.to_vec(),
        };
        let norm = scaled.iter().map(|a| a * a).sum::<f32>().sqrt();
        if norm > threshold {
            clipped += 1;
        } else {
            kept += 1;
        }
        want.extend(
            l2_clip(row, sc.as_deref(), threshold)
                .iter()
                .map(|y| y * post),
        );
    }
    assert!(
        clipped > 0 && kept > 0,
        "fixture: both branches ({clipped}/{kept})"
    );

    let mut context = MetalContext::new().expect("Metal device");
    let x_buf = context.new_buffer_with_data(&le16(&x16));
    let s_buf = scale
        .as_ref()
        .map(|(b, _)| context.new_buffer_with_data(&leb16(b)));
    let pass = context.begin_pass();
    encode_inkling2_l2_clip(
        &mut context,
        &pass,
        (&x_buf, 0),
        s_buf
            .as_ref()
            .zip(scale.as_ref())
            .map(|(b, (_, stride))| ((b, 0u64), *stride as u32)),
        n_vec as u32,
        dim as u32,
        threshold,
        post,
    )
    .expect("l2 clip");
    pass.commit_and_wait();
    assert_close(&read_f16(&x_buf, n_vec * dim), &want, 2e-3, 2e-3, "l2 clip");
}

/// q: 24 heads of 128, per-head scale, the qk threshold, a full-attention tau.
#[test]
fn l2_clip_q_heads_with_per_vector_scale_and_tau() {
    run_l2_clip(24, 128, 22.627_417, 1.3, Some(128));
}

/// r: 16-wide relative states, one shared scale, the rel threshold.
#[test]
fn l2_clip_rel_with_shared_scale() {
    run_l2_clip(24, 16, 8.0, 1.0, Some(0));
}

/// k: no scale.
#[test]
fn l2_clip_without_scale() {
    run_l2_clip(8, 128, 22.627_417, 1.0, None);
}

// --- Short conv -----------------------------------------------------------

fn run_short_conv(ch: usize, taps: usize, heads: Option<usize>, steps: usize) {
    let mut s = 17u32;
    let rows: Vec<Vec<f32>> = (0..steps)
        .map(|_| {
            f16s(&(0..ch).map(|_| unit(&mut s)).collect::<Vec<_>>())
                .iter()
                .map(|v| v.to_f32())
                .collect()
        })
        .collect();
    let tail0: Vec<Vec<f32>> = (0..taps - 1)
        .map(|_| {
            f16s(&(0..ch).map(|_| unit(&mut s)).collect::<Vec<_>>())
                .iter()
                .map(|v| v.to_f32())
                .collect()
        })
        .collect();
    let w = bf16s(
        &(0..ch * taps)
            .map(|_| unit(&mut s) * 0.8)
            .collect::<Vec<_>>(),
    );
    let w_r: Vec<f32> = w.iter().map(|v| v.to_f32()).collect();
    let dyn_rows: Option<Vec<f16>> = heads.map(|h| {
        f16s(
            &(0..steps * h * taps)
                .map(|_| unit(&mut s) * 0.6)
                .collect::<Vec<_>>(),
        )
    });
    let dyn_r: Option<Vec<f32>> = dyn_rows
        .as_ref()
        .map(|d| d.iter().map(|v| v.to_f32()).collect());

    let mut tail = tail0.clone();
    let want = short_conv(&rows, &w_r, taps, &mut tail, dyn_r.as_deref().zip(heads));

    let mut context = MetalContext::new().expect("Metal device");
    let flat_tail: Vec<f32> = tail0.iter().flatten().copied().collect();
    let tail_buf = context.new_buffer_with_data(&le16(&f16s(&flat_tail)));
    let w_buf = context.new_buffer_with_data(&leb16(&w));
    let x_bufs: Vec<metal::Buffer> = rows
        .iter()
        .map(|r| context.new_buffer_with_data(&le16(&f16s(r))))
        .collect();
    let out_bufs: Vec<metal::Buffer> = (0..steps)
        .map(|_| context.new_output_buffer((ch * 2) as u64))
        .collect();
    let dyn_bufs: Option<Vec<metal::Buffer>> = dyn_rows.as_ref().zip(heads).map(|(d, h)| {
        d.chunks_exact(h * taps)
            .map(|c| context.new_buffer_with_data(&le16(c)))
            .collect()
    });
    // One step per pass: each step reads the tail the previous one wrote.
    for t in 0..steps {
        let pass = context.begin_pass();
        encode_inkling2_short_conv_decode(
            &mut context,
            &pass,
            (&x_bufs[t], 0),
            (&tail_buf, 0),
            (&w_buf, 0),
            dyn_bufs
                .as_ref()
                .zip(heads)
                .map(|(b, h)| ((&b[t], 0u64), h as u32)),
            (&out_bufs[t], 0),
            ch as u32,
            taps as u32,
        )
        .expect("short conv");
        pass.commit_and_wait();
    }
    for t in 0..steps {
        assert_close(
            &read_f16(&out_bufs[t], ch),
            &want[t],
            2e-3,
            2e-3,
            &format!("conv step {t}"),
        );
    }
    let want_tail: Vec<f32> = tail.iter().flatten().copied().collect();
    assert_eq!(
        read_f16(&tail_buf, ch * (taps - 1)),
        want_tail,
        "tail after {steps} steps"
    );
}

/// SWA K/V and the MLP / post-o-proj convs: static taps, K = 4.
#[test]
fn static_short_conv_over_several_steps() {
    run_short_conv(1024, 4, None, 6);
}

/// Full-attention K/V: per-token per-kv-head taps on top, 8 kv heads.
#[test]
fn dynamic_short_conv_over_several_steps() {
    run_short_conv(1024, 4, Some(8), 6);
}

// --- Depth mix ------------------------------------------------------------

#[test]
fn depth_mix_matches_reference() {
    let (n_states, n_q, h) = (5usize, 3usize, 2304usize);
    let mut s = 4242u32;
    // State magnitudes differ by up to 5x: without the RMS norm the largest
    // state would dominate both the scores and the sum.
    let states: Vec<Vec<f32>> = (0..n_states)
        .map(|k| {
            let v: Vec<f32> = (0..h)
                .map(|_| unit(&mut s) * (0.3 + 0.35 * k as f32))
                .collect();
            f16s(&v).iter().map(|x| x.to_f32()).collect()
        })
        .collect();
    let queries: Vec<Vec<f32>> = (0..n_q)
        .map(|_| {
            let v: Vec<f32> = (0..h).map(|_| unit(&mut s) * 0.15).collect();
            bf16s(&v).iter().map(|x| x.to_f32()).collect()
        })
        .collect();
    // Bias query 0 toward state 1 so its softmax is peaked, not uniform.
    let mut queries = queries;
    for (q, st) in queries[0].iter_mut().zip(&states[1]) {
        *q = bf16::from_f32(*q + st / 0.65 * 0.5).to_f32();
    }
    let want: Vec<f32> = block_attn_res_mix(&states, &queries).concat();

    let mut context = MetalContext::new().expect("Metal device");
    let st_buf = context.new_buffer_with_data(&le16(&f16s(&states.concat())));
    let q_buf = context.new_buffer_with_data(&leb16(&bf16s(&queries.concat())));
    let out_buf = context.new_output_buffer((n_q * h * 2) as u64);
    let pass = context.begin_pass();
    encode_inkling2_depth_mix(
        &mut context,
        &pass,
        (&st_buf, 0),
        (&q_buf, 0),
        (&out_buf, 0),
        n_states as u32,
        n_q as u32,
        h as u32,
    )
    .expect("depth mix");
    pass.commit_and_wait();
    let got = read_f16(&out_buf, n_q * h);
    assert_close(&got, &want, 3e-3, 3e-3, "depth mix");
    // The output of a uniform mix of normed random states is small; a peaked
    // one is near a single normed state (unit RMS). Prove query 0 is peaked.
    let rms0 = (want[..h].iter().map(|v| v * v).sum::<f32>() / h as f32).sqrt();
    assert!(rms0 > 0.8, "fixture: query 0 should be peaked, rms {rms0}");
}

#[test]
fn depth_mix_refuses_too_many_states() {
    let mut context = MetalContext::new().expect("Metal device");
    let buf = context.new_buffer_with_data(&[0u8; 64]);
    let pass = context.begin_pass();
    let r = encode_inkling2_depth_mix(
        &mut context,
        &pass,
        (&buf, 0),
        (&buf, 0),
        (&buf, 0),
        17,
        1,
        16,
    );
    pass.commit_and_wait();
    assert!(r.is_err());
}

// --- Relative bias table and decode attention -----------------------------

use turbospark_compute::inkling2::{attention_decode, rel_bias_table};
use turbospark_gpu::{
    encode_inkling2_attention_decode, encode_inkling2_rel_table, Inkling2AttnShape,
};

fn read_f32(buffer: &metal::Buffer, n: usize) -> Vec<f32> {
    let ptr = buffer.contents() as *const f32;
    unsafe { std::slice::from_raw_parts(ptr, n) }.to_vec()
}

/// One query at `pos` over `first..=pos`, Mini head shapes. `sink_scale`
/// sets how much the sink competes with the positions.
fn run_attention(n_pos: usize, first: usize, extent: usize, sink_scale: f32) {
    let (heads, kv_heads, hd, d_rel) = (24usize, 8usize, 128usize, 16usize);
    let pos = n_pos - 1;
    let mut s = 1234u32 + n_pos as u32;
    let r16 = f16s(&(0..heads * d_rel).map(|_| unit(&mut s)).collect::<Vec<_>>());
    let proj = bf16s(
        &(0..d_rel * extent)
            .map(|_| unit(&mut s) * 0.6)
            .collect::<Vec<_>>(),
    );
    // q and k near the clip radius so scores span a few nats.
    let q16 = f16s(
        &(0..heads * hd)
            .map(|_| unit(&mut s) * 0.35)
            .collect::<Vec<_>>(),
    );
    let k16 = f16s(
        &(0..n_pos * kv_heads * hd)
            .map(|_| unit(&mut s) * 0.35)
            .collect::<Vec<_>>(),
    );
    let v16 = f16s(
        &(0..n_pos * kv_heads * hd)
            .map(|_| unit(&mut s))
            .collect::<Vec<_>>(),
    );
    let sinks: Vec<f32> = (0..heads)
        .map(|h| sink_scale * (h as f32 / heads as f32 - 0.3))
        .collect();
    let f = |v: &[f16]| v.iter().map(|x| x.to_f32()).collect::<Vec<f32>>();
    let proj_r: Vec<f32> = proj.iter().map(|x| x.to_f32()).collect();
    let table = rel_bias_table(&f(&r16), &proj_r, heads, d_rel, extent);
    let want = attention_decode(
        &f(&q16),
        &f(&k16),
        &f(&v16),
        &table,
        &sinks,
        first,
        pos,
        heads,
        kv_heads,
        hd,
        extent,
    );

    let mut context = MetalContext::new().expect("Metal device");
    let r_buf = context.new_buffer_with_data(&le16(&r16));
    let p_buf = context.new_buffer_with_data(&leb16(&proj));
    let t_buf = context.new_output_buffer((heads * extent * 4) as u64);
    let q_buf = context.new_buffer_with_data(&le16(&q16));
    let k_buf = context.new_buffer_with_data(&le16(&k16));
    let v_buf = context.new_buffer_with_data(&le16(&v16));
    let s_buf = context.new_buffer_with_data(&le32(&sinks));
    let o_buf = context.new_output_buffer((heads * hd * 2) as u64);
    let shape = Inkling2AttnShape {
        heads: heads as u32,
        kv_heads: kv_heads as u32,
        head_dim: hd as u32,
        extent: extent as u32,
    };
    let pass = context.begin_pass();
    encode_inkling2_rel_table(
        &mut context,
        &pass,
        (&r_buf, 0),
        (&p_buf, 0),
        (&t_buf, 0),
        heads as u32,
        d_rel as u32,
        extent as u32,
    )
    .expect("rel table");
    encode_inkling2_attention_decode(
        &mut context,
        &pass,
        (&q_buf, 0),
        (&k_buf, 0),
        (&v_buf, 0),
        (&t_buf, 0),
        (&s_buf, 0),
        (&o_buf, 0),
        first as u32,
        pos as u32,
        shape,
    )
    .expect("attention");
    pass.commit_and_wait();
    assert_close(
        &read_f32(&t_buf, heads * extent),
        &table,
        1e-4,
        1e-4,
        "rel table",
    );
    assert_close(
        &read_f16(&o_buf, heads * hd),
        &want,
        2e-3,
        2e-3,
        "attention",
    );
}

/// Full layer: every position, some past the bias extent.
#[test]
fn attention_full_with_positions_past_the_extent() {
    run_attention(40, 0, 16, 2.0);
}

/// SWA layer: a window that starts after position 0.
#[test]
fn attention_sliding_window() {
    run_attention(70, 30, 64, 2.0);
}

/// Fewer positions than SIMD groups (the first token among them): the empty
/// groups must not contribute.
#[test]
fn attention_with_fewer_positions_than_groups() {
    run_attention(1, 0, 16, 2.0);
    run_attention(5, 0, 16, 2.0);
}

/// A sink that dominates: if the kernel dropped it, every output would be
/// several times too large.
#[test]
fn attention_dominant_sink() {
    run_attention(12, 0, 16, 12.0);
}
