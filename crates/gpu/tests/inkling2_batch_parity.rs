#![cfg(target_os = "macos")]
//! The multi-row (verify) Inkling 2 kernels against M calls of the
//! single-row kernel each mirrors, BIT FOR BIT, on real Metal hardware.
//!
//! This is the property the speculative verify relies on: a block of M rows
//! writes what M sequential decode calls write, so greedy speculation stays
//! lossless. Every fixture uses distinct per-row data, so a kernel that read
//! the wrong row (a stride bug) fails, and the conv and attention cases run
//! across a ring wrap.

use half::{bf16, f16};
use turbospark_gpu::*;

fn lcg(state: &mut u32) -> u32 {
    *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *state
}

fn unit(s: &mut u32) -> f32 {
    ((lcg(s) >> 8) as f32 / (1u32 << 23) as f32) - 1.0
}

fn rand_f16(s: &mut u32, n: usize, scale: f32) -> Vec<f16> {
    (0..n).map(|_| f16::from_f32(unit(s) * scale)).collect()
}

fn rand_bf16(s: &mut u32, n: usize, scale: f32) -> Vec<bf16> {
    (0..n).map(|_| bf16::from_f32(unit(s) * scale)).collect()
}

fn b16(v: &[f16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect()
}

fn bb16(v: &[bf16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect()
}

fn b32(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn bits16(b: &metal::Buffer, off: usize, n: usize) -> Vec<u16> {
    let p = b.contents() as *const u16;
    unsafe { std::slice::from_raw_parts(p.add(off), n) }.to_vec()
}

fn bits32(b: &metal::Buffer, off: usize, n: usize) -> Vec<u32> {
    let p = b.contents() as *const u32;
    unsafe { std::slice::from_raw_parts(p.add(off), n) }.to_vec()
}

const M: usize = 8;

fn run_gemv_rows(n: usize, cols_a: usize) {
    let mut ctx = MetalContext::new().expect("Metal device");
    let mut s = 7u32;
    let (rows_a, rows_b) = (40usize, 17usize);
    let wa = rand_bf16(&mut s, rows_a * cols_a, 0.5);
    let wb = rand_bf16(&mut s, rows_b * cols_a, 0.5);
    let mut wbytes = bb16(&wa);
    let off_b = wbytes.len();
    wbytes.extend(bb16(&wb));
    let w = ctx.new_buffer_with_data(&wbytes);
    // Two inputs, row strides larger than cols (as the cat buffer is).
    let xs = (cols_a * 2) as u32;
    let x0 = ctx.new_buffer_with_data(&b16(&rand_f16(&mut s, n * xs as usize, 1.0)));
    let x1 = ctx.new_buffer_with_data(&b16(&rand_f16(&mut s, n * xs as usize, 1.0)));
    let ya = ctx.new_output_buffer((n * rows_a * 2) as u64);
    let yb = ctx.new_output_buffer((n * 64 * 2) as u64);
    let pass = ctx.begin_pass();
    encode_inkling2_bf16_gemv_rows(
        &mut ctx,
        &pass,
        &w,
        &[((&x0, 0), xs), ((&x1, 0), xs)],
        &[
            Inkling2GemvRowsSegment {
                w_offset: 0,
                rows: rows_a as u32,
                cols: cols_a as u32,
                x: 0,
                y: (&ya, 0),
                y_stride: rows_a as u32,
            },
            Inkling2GemvRowsSegment {
                w_offset: off_b as u64,
                rows: rows_b as u32,
                cols: cols_a as u32,
                x: 1,
                y: (&yb, 0),
                y_stride: 64,
            },
        ],
        n as u32,
    )
    .unwrap();
    pass.commit_and_wait();
    let ra = ctx.new_output_buffer((rows_a * 2) as u64);
    let rb = ctx.new_output_buffer((rows_b * 2) as u64);
    for m in 0..n {
        let pass = ctx.begin_pass();
        let o = (m * xs as usize * 2) as u64;
        encode_inkling2_bf16_gemv_multi(
            &mut ctx,
            &pass,
            &w,
            &[(&x0, o), (&x1, o)],
            &[
                Inkling2GemvSegment {
                    w_offset: 0,
                    rows: rows_a as u32,
                    cols: cols_a as u32,
                    x: 0,
                    y: (&ra, 0),
                },
                Inkling2GemvSegment {
                    w_offset: off_b as u64,
                    rows: rows_b as u32,
                    cols: cols_a as u32,
                    x: 1,
                    y: (&rb, 0),
                },
            ],
        )
        .unwrap();
        pass.commit_and_wait();
        assert_eq!(
            bits16(&ya, m * rows_a, rows_a),
            bits16(&ra, 0, rows_a),
            "seg a row {m}"
        );
        assert_eq!(
            bits16(&yb, m * 64, rows_b),
            bits16(&rb, 0, rows_b),
            "seg b row {m}"
        );
    }
}

#[test]
fn gemv_rows_match_single_row_gemv() {
    // 8 rows take the verify kernel; above 8 the 32-accumulator twin. The
    // real hidden width too, where more terms reach a rounding boundary.
    for n in [1usize, 8, 9, 23, 32] {
        for cols in [256usize, 2304] {
            run_gemv_rows(n, cols);
        }
    }
}

#[test]
fn rms_rows_match_single_row_norms() {
    let mut ctx = MetalContext::new().expect("Metal device");
    let mut s = 11u32;
    let d = 2304usize;
    let stride = 2 * d;
    let x = ctx.new_buffer_with_data(&b16(&rand_f16(&mut s, M * stride, 3.0)));
    let w = ctx.new_buffer_with_data(&bb16(&rand_bf16(&mut s, d, 1.0)));
    let out = ctx.new_output_buffer((M * stride * 2) as u64);
    let pass = ctx.begin_pass();
    encode_inkling2_rms_rows(
        &mut ctx,
        &pass,
        Inkling2Rows::plain((&x, 0), stride as u32),
        Some((&w, 0)),
        Inkling2Rows::plain((&out, (d * 2) as u64), stride as u32),
        d as u32,
        1,
        1e-6,
        M as u32,
    )
    .unwrap();
    pass.commit_and_wait();
    let one = ctx.new_output_buffer((d * 2) as u64);
    let one_ns = ctx.new_output_buffer((d * 2) as u64);
    let out_ns = ctx.new_output_buffer((M * d * 2) as u64);
    let pass = ctx.begin_pass();
    encode_inkling2_rms_rows(
        &mut ctx,
        &pass,
        Inkling2Rows::plain((&x, 0), stride as u32),
        None,
        Inkling2Rows::plain((&out_ns, 0), d as u32),
        d as u32,
        1,
        1e-6,
        M as u32,
    )
    .unwrap();
    pass.commit_and_wait();
    for m in 0..M {
        let pass = ctx.begin_pass();
        let xo = (m * stride * 2) as u64;
        encode_rms_norm_bf16w(
            &mut ctx,
            &pass,
            (&x, xo),
            (&w, 0),
            (&one, 0),
            d as u32,
            1e-6,
        )
        .unwrap();
        encode_rms_norm_no_scale(&mut ctx, &pass, (&x, xo), (&one_ns, 0), d as u32, 1e-6).unwrap();
        pass.commit_and_wait();
        assert_eq!(
            bits16(&out, m * stride + d, d),
            bits16(&one, 0, d),
            "weighted row {m}"
        );
        assert_eq!(
            bits16(&out_ns, m * d, d),
            bits16(&one_ns, 0, d),
            "no-scale row {m}"
        );
    }
}

#[test]
fn per_head_rms_on_a_ring_matches_the_perhead_kernel() {
    let mut ctx = MetalContext::new().expect("Metal device");
    let mut s = 13u32;
    let (kvh, hd, ring, base) = (8usize, 128usize, 11usize, 7usize);
    let kvd = kvh * hd;
    let data = rand_f16(&mut s, ring * kvd, 2.0);
    let x = ctx.new_buffer_with_data(&b16(&data));
    let reference = ctx.new_buffer_with_data(&b16(&data));
    let pass = ctx.begin_pass();
    let rows = Inkling2Rows {
        buf: (&x, 0),
        stride: kvd as u32,
        ring_base: base as u32,
        ring: ring as u32,
    };
    encode_inkling2_rms_rows(
        &mut ctx, &pass, rows, None, rows, hd as u32, kvh as u32, 1e-6, M as u32,
    )
    .unwrap();
    pass.commit_and_wait();
    for m in 0..M {
        let row = (base + m) % ring;
        let pass = ctx.begin_pass();
        let o = (row * kvd * 2) as u64;
        encode_rms_norm_no_scale_perhead(
            &mut ctx,
            &pass,
            (&reference, o),
            (&reference, o),
            kvh as u32,
            hd as u32,
            1e-6,
        )
        .unwrap();
        pass.commit_and_wait();
    }
    assert_eq!(bits16(&x, 0, ring * kvd), bits16(&reference, 0, ring * kvd));
}

#[test]
fn depth_mix_rows_match_single_row_mix() {
    let mut ctx = MetalContext::new().expect("Metal device");
    let mut s = 17u32;
    let (h, n_states, n_q, max_states) = (2304usize, 9usize, 3usize, 16usize);
    let st_stride = max_states * h;
    let mut states = rand_f16(&mut s, M * st_stride, 1.0);
    // Distinct magnitudes so the norm matters.
    for (i, v) in states.iter_mut().enumerate() {
        *v = f16::from_f32(v.to_f32() * (1.0 + ((i / h) % 5) as f32));
    }
    let st = ctx.new_buffer_with_data(&b16(&states));
    let q: Vec<f32> = (0..n_q * h).map(|_| unit(&mut s)).collect();
    let qb = ctx.new_buffer_with_data(&b32(&q));
    let out = ctx.new_output_buffer((M * 4 * h * 2) as u64);
    let pass = ctx.begin_pass();
    encode_inkling2_depth_mix_rows(
        &mut ctx,
        &pass,
        (&st, 0),
        st_stride as u32,
        (&qb, 0),
        (&out, 0),
        (4 * h) as u32,
        n_states as u32,
        n_q as u32,
        h as u32,
        M as u32,
    )
    .unwrap();
    pass.commit_and_wait();
    let one = ctx.new_output_buffer((n_q * h * 2) as u64);
    for m in 0..M {
        let pass = ctx.begin_pass();
        encode_inkling2_depth_mix(
            &mut ctx,
            &pass,
            (&st, (m * st_stride * 2) as u64),
            (&qb, 0),
            (&one, 0),
            n_states as u32,
            n_q as u32,
            h as u32,
        )
        .unwrap();
        pass.commit_and_wait();
        assert_eq!(
            bits16(&out, m * 4 * h, n_q * h),
            bits16(&one, 0, n_q * h),
            "row {m}"
        );
    }
}

fn conv_case(dynamic: bool) {
    let mut ctx = MetalContext::new().expect("Metal device");
    let mut s = if dynamic { 19u32 } else { 23 };
    let (ch, taps, heads, ring, base) = (1024usize, 4usize, 8usize, 10usize, 6usize);
    let x = ctx.new_buffer_with_data(&b16(&rand_f16(&mut s, M * ch, 1.0)));
    let tail0 = rand_f16(&mut s, (taps - 1) * ch, 1.0);
    let tail = ctx.new_buffer_with_data(&b16(&tail0));
    let tail_ref = ctx.new_buffer_with_data(&b16(&tail0));
    let w = ctx.new_buffer_with_data(&bb16(&rand_bf16(&mut s, ch * taps, 0.7)));
    let dy = ctx.new_buffer_with_data(&b16(&rand_f16(&mut s, M * heads * taps, 0.5)));
    let out = ctx.new_buffer_with_data(&vec![0u8; ring * ch * 2]);
    let snap = ctx.new_output_buffer(((M + 1) * (taps - 1) * ch * 2) as u64);
    let pass = ctx.begin_pass();
    encode_inkling2_short_conv_rows(
        &mut ctx,
        &pass,
        (&x, 0),
        ch as u32,
        (&tail, 0),
        (&w, 0),
        dynamic.then_some(((&dy, 0), (heads * taps) as u32, heads as u32)),
        Inkling2Rows {
            buf: (&out, 0),
            stride: ch as u32,
            ring_base: base as u32,
            ring: ring as u32,
        },
        Some((&snap, 0)),
        ch as u32,
        taps as u32,
        M as u32,
    )
    .unwrap();
    pass.commit_and_wait();
    let one = ctx.new_output_buffer((ch * 2) as u64);
    let tn = (taps - 1) * ch;
    let start: Vec<u16> = tail0.iter().map(|v| v.to_bits()).collect();
    assert_eq!(
        bits16(&snap, 0, tn),
        start,
        "snapshot 0 is the block-start tail"
    );
    for m in 0..M {
        let pass = ctx.begin_pass();
        encode_inkling2_short_conv_decode(
            &mut ctx,
            &pass,
            (&x, (m * ch * 2) as u64),
            (&tail_ref, 0),
            (&w, 0),
            dynamic.then_some(((&dy, (m * heads * taps * 2) as u64), heads as u32)),
            (&one, 0),
            ch as u32,
            taps as u32,
        )
        .unwrap();
        pass.commit_and_wait();
        let row = (base + m) % ring;
        assert_eq!(
            bits16(&out, row * ch, ch),
            bits16(&one, 0, ch),
            "out row {m}"
        );
        let n = (taps - 1) * ch;
        assert_eq!(
            bits16(&snap, (m + 1) * n, n),
            bits16(&tail_ref, 0, n),
            "snapshot after row {m}"
        );
    }
    let n = (taps - 1) * ch;
    assert_eq!(bits16(&tail, 0, n), bits16(&tail_ref, 0, n), "final tail");
}

#[test]
fn static_conv_rows_match_sequential_decode() {
    conv_case(false);
}

#[test]
fn dynamic_conv_rows_match_sequential_decode() {
    conv_case(true);
}

#[test]
fn l2_clip_rows_match_single_row_clip() {
    let mut ctx = MetalContext::new().expect("Metal device");
    let mut s = 29u32;
    let (n_vec, dim, ring, base) = (8usize, 128usize, 9usize, 4usize);
    let stride = n_vec * dim;
    let data = rand_f16(&mut s, ring * stride, 4.0);
    let x = ctx.new_buffer_with_data(&b16(&data));
    let reference = ctx.new_buffer_with_data(&b16(&data));
    let sc = ctx.new_buffer_with_data(&bb16(&rand_bf16(&mut s, n_vec * dim, 2.0)));
    let post: Vec<f32> = (0..M).map(|m| 1.0 + 0.01 * m as f32).collect();
    let pass = ctx.begin_pass();
    encode_inkling2_l2_clip_rows(
        &mut ctx,
        &pass,
        Inkling2Rows {
            buf: (&x, 0),
            stride: stride as u32,
            ring_base: base as u32,
            ring: ring as u32,
        },
        Some(((&sc, 0), dim as u32)),
        n_vec as u32,
        dim as u32,
        22.6,
        &post,
    )
    .unwrap();
    pass.commit_and_wait();
    for (m, &p) in post.iter().enumerate() {
        let row = (base + m) % ring;
        let pass = ctx.begin_pass();
        encode_inkling2_l2_clip(
            &mut ctx,
            &pass,
            (&reference, (row * stride * 2) as u64),
            Some(((&sc, 0), dim as u32)),
            n_vec as u32,
            dim as u32,
            22.6,
            p,
        )
        .unwrap();
        pass.commit_and_wait();
    }
    assert_eq!(
        bits16(&x, 0, ring * stride),
        bits16(&reference, 0, ring * stride)
    );
}

#[test]
fn attention_rows_match_single_row_attention_across_a_wrap() {
    let mut ctx = MetalContext::new().expect("Metal device");
    let mut s = 31u32;
    let (heads, kvh, hd, d_rel, extent) = (24usize, 8usize, 128usize, 16usize, 64usize);
    // A window slot with block slack: window 40, ring 40 + 7, positions past
    // two wraps.
    for &(window, ring, pos0) in &[(40usize, 47usize, 101usize), (0, 200, 150)] {
        let kvd = kvh * hd;
        let k = ctx.new_buffer_with_data(&b16(&rand_f16(&mut s, ring * kvd, 1.0)));
        let v = ctx.new_buffer_with_data(&b16(&rand_f16(&mut s, ring * kvd, 1.0)));
        let q = ctx.new_buffer_with_data(&b16(&rand_f16(&mut s, M * heads * hd, 1.0)));
        let r = ctx.new_buffer_with_data(&b16(&rand_f16(&mut s, M * heads * d_rel, 1.0)));
        let proj = ctx.new_buffer_with_data(&bb16(&rand_bf16(&mut s, d_rel * extent, 0.5)));
        let sinks = ctx.new_buffer_with_data(&b16(&rand_f16(&mut s, M * heads, 1.0)));
        let table = ctx.new_output_buffer((M * heads * extent * 4) as u64);
        let out = ctx.new_output_buffer((M * heads * hd * 2) as u64);
        let shape = Inkling2AttnShape {
            heads: heads as u32,
            kv_heads: kvh as u32,
            head_dim: hd as u32,
            extent: extent as u32,
            ring: ring as u32,
        };
        let pass = ctx.begin_pass();
        encode_inkling2_rel_table_rows(
            &mut ctx,
            &pass,
            (&r, 0),
            (heads * d_rel) as u32,
            (&proj, 0),
            (&table, 0),
            heads as u32,
            d_rel as u32,
            extent as u32,
            M as u32,
        )
        .unwrap();
        encode_inkling2_attention_rows(
            &mut ctx,
            &pass,
            (&q, 0),
            (&k, 0),
            (&v, 0),
            (&table, 0),
            (&sinks, 0),
            (&out, 0),
            window as u32,
            pos0 as u32,
            shape,
            M as u32,
        )
        .unwrap();
        pass.commit_and_wait();
        let t1 = ctx.new_output_buffer((heads * extent * 4) as u64);
        let o1 = ctx.new_output_buffer((heads * hd * 2) as u64);
        for m in 0..M {
            let pos = pos0 + m;
            let first = if window == 0 {
                0
            } else {
                (pos + 1).saturating_sub(window)
            };
            let pass = ctx.begin_pass();
            encode_inkling2_rel_table(
                &mut ctx,
                &pass,
                (&r, (m * heads * d_rel * 2) as u64),
                (&proj, 0),
                (&t1, 0),
                heads as u32,
                d_rel as u32,
                extent as u32,
            )
            .unwrap();
            encode_inkling2_attention_decode(
                &mut ctx,
                &pass,
                (&q, (m * heads * hd * 2) as u64),
                (&k, 0),
                (&v, 0),
                (&t1, 0),
                (&sinks, (m * heads * 2) as u64),
                (&o1, 0),
                first as u32,
                pos as u32,
                shape,
            )
            .unwrap();
            pass.commit_and_wait();
            assert_eq!(
                bits32(&table, m * heads * extent, heads * extent),
                bits32(&t1, 0, heads * extent),
                "table row {m}"
            );
            assert_eq!(
                bits16(&out, m * heads * hd, heads * hd),
                bits16(&o1, 0, heads * hd),
                "window {window} row {m}"
            );
        }
    }
}

#[test]
fn router_rows_match_single_row_router_and_selection() {
    let mut ctx = MetalContext::new().expect("Metal device");
    let mut s = 37u32;
    let (ne, h) = (128usize, 2304usize);
    let w: Vec<f32> = (0..ne * h).map(|_| unit(&mut s) * 0.05).collect();
    let wb = ctx.new_buffer_with_data(&b32(&w));
    let bias: Vec<f32> = (0..ne)
        .map(|e| if e % 7 == 0 { 0.2 } else { 0.0 })
        .collect();
    let bb = ctx.new_buffer_with_data(&b32(&bias));
    let x = ctx.new_buffer_with_data(&b16(&rand_f16(&mut s, M * h, 1.0)));
    let logits = ctx.new_output_buffer((M * ne * 4) as u64);
    let sel = ctx.new_output_buffer((M * 8 * 4) as u64);
    let rw = ctx.new_output_buffer((M * 8 * 2) as u64);
    let pass = ctx.begin_pass();
    encode_inkling2_router_rows(
        &mut ctx,
        &pass,
        (&wb, 0),
        (&x, 0),
        h as u32,
        (&logits, 0),
        ne as u32,
        h as u32,
        M as u32,
    )
    .unwrap();
    encode_inkling2_router_select_rows(
        &mut ctx,
        &pass,
        (&logits, 0),
        (&bb, 0),
        (&sel, 0),
        (&rw, 0),
        ne as u32,
        8.0,
        M as u32,
    )
    .unwrap();
    pass.commit_and_wait();
    let l1 = ctx.new_output_buffer((ne * 4) as u64);
    let s1 = ctx.new_output_buffer(32);
    let w1 = ctx.new_output_buffer(16);
    for m in 0..M {
        let pass = ctx.begin_pass();
        encode_minimax_router(
            &mut ctx,
            &pass,
            (&wb, 0),
            (&x, (m * h * 2) as u64),
            (&l1, 0),
            ne as u32,
            h as u32,
        )
        .unwrap();
        encode_inkling2_router_select(
            &mut ctx,
            &pass,
            (&l1, 0),
            (&bb, 0),
            (&s1, 0),
            (&w1, 0),
            ne as u32,
            8.0,
        )
        .unwrap();
        pass.commit_and_wait();
        assert_eq!(
            bits32(&logits, m * ne, ne),
            bits32(&l1, 0, ne),
            "logits row {m}"
        );
        assert_eq!(bits32(&sel, m * 8, 8), bits32(&s1, 0, 8), "sel row {m}");
        assert_eq!(bits16(&rw, m * 8, 8), bits16(&w1, 0, 8), "weights row {m}");
    }
}

#[test]
fn control_combine_rows_match_single_row() {
    let mut ctx = MetalContext::new().expect("Metal device");
    let mut s = 41u32;
    let n = 2304usize;
    let cur = ctx.new_buffer_with_data(&b16(&rand_f16(&mut s, M * 3 * n, 1.0)));
    let msg = ctx.new_buffer_with_data(&b16(&rand_f16(&mut s, M * 4 * n, 1.0)));
    let g = ctx.new_buffer_with_data(&b16(&rand_f16(&mut s, M * n, 2.0)));
    let f = ctx.new_buffer_with_data(&b16(&rand_f16(&mut s, M * n, 2.0)));
    let out = ctx.new_output_buffer((M * 3 * n * 2) as u64);
    let pass = ctx.begin_pass();
    encode_inkling2_control_combine_rows(
        &mut ctx,
        &pass,
        (&cur, 0),
        (&msg, 0),
        (&g, 0),
        (&f, 0),
        (&out, 0),
        n as u32,
        [(3 * n) as u32, (4 * n) as u32, n as u32, (3 * n) as u32],
        M as u32,
    )
    .unwrap();
    pass.commit_and_wait();
    let one = ctx.new_output_buffer((n * 2) as u64);
    for m in 0..M {
        let pass = ctx.begin_pass();
        encode_inkling2_control_combine(
            &mut ctx,
            &pass,
            (&cur, (m * 3 * n * 2) as u64),
            (&msg, (m * 4 * n * 2) as u64),
            (&g, (m * n * 2) as u64),
            (&f, (m * n * 2) as u64),
            (&one, 0),
            n as u32,
        )
        .unwrap();
        pass.commit_and_wait();
        assert_eq!(bits16(&out, m * 3 * n, n), bits16(&one, 0, n), "row {m}");
    }
}
