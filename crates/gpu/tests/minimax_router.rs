#![cfg(target_os = "macos")]
use half::f16;
use turbospark_gpu::{encode_minimax_router, encode_rms_norm_bf16w, MetalContext};

#[test]
fn fp32_router_matches_independent_dot_products_with_offsets() {
    let mut c = MetalContext::new().unwrap();
    let hidden = 67usize;
    let mut w: Vec<f32> = (0..hidden * 3)
        .map(|i| ((i % 19) as f32 - 9.0) / 128.0)
        .collect();
    let mut x: Vec<f32> = (0..hidden).map(|i| ((i % 7) as f32 - 3.0) / 16.0).collect();
    // One lane must retain the unit term between large cancellations.
    // FP16 accumulation loses it before the SIMD reduction.
    w[..hidden].fill(0.0);
    for (col, weight) in [(0, 4096.0), (32, 1.0), (64, -4096.0)] {
        w[col] = weight;
        x[col] = 1.0;
    }
    let mut wb = vec![0u8; 16];
    wb.extend(w.iter().flat_map(|v| v.to_le_bytes()));
    let mut xb = vec![0u8; 8];
    xb.extend(
        x.iter()
            .flat_map(|v| f16::from_f32(*v).to_bits().to_le_bytes()),
    );
    let wb = c.new_buffer_with_data(&wb);
    let xb = c.new_buffer_with_data(&xb);
    let out = c.new_output_buffer(24);
    let pass = c.begin_pass();
    encode_minimax_router(
        &mut c,
        &pass,
        (&wb, 16),
        (&xb, 8),
        (&out, 4),
        3,
        hidden as u32,
    )
    .unwrap();
    pass.commit_and_wait();
    let got = turbospark_gpu::read_f32_buffer_at(&out, 1, 3);
    let expected: Vec<f32> = w
        .chunks(hidden)
        .map(|row| row.iter().zip(&x).map(|(a, b)| a * b).sum())
        .collect();
    assert_eq!(got, expected);
}

#[test]
fn whole_projection_norm_uses_all_heads_and_nonuniform_weights() {
    let mut c = MetalContext::new().unwrap();
    for n in [1024usize, 6144] {
        let x: Vec<f32> = (0..n)
            .map(|i| ((i % 128) as f32 - 63.0) / 32.0 * (1 + (i / 128) % 5) as f32)
            .collect();
        let w: Vec<f32> = (0..n).map(|i| 0.5 + (i % 17) as f32 / 16.0).collect();
        let xb = c.new_buffer_with_data(
            &x.iter()
                .flat_map(|v| f16::from_f32(*v).to_bits().to_le_bytes())
                .collect::<Vec<_>>(),
        );
        let wb = c.new_buffer_with_data(
            &w.iter()
                .flat_map(|v| ((*v).to_bits() >> 16).to_le_bytes()[..2].to_vec())
                .collect::<Vec<_>>(),
        );
        let pass = c.begin_pass();
        encode_rms_norm_bf16w(&mut c, &pass, (&xb, 0), (&wb, 0), (&xb, 0), n as u32, 1e-6).unwrap();
        pass.commit_and_wait();
        let got = turbospark_gpu::read_buffer_f16(&xb, 0, n);
        let rms = (x.iter().map(|v| v * v).sum::<f32>() / n as f32 + 1e-6).sqrt();
        for (i, v) in got.iter().enumerate() {
            let expected = x[i] * w[i] / rms;
            assert!(
                (v.to_f32() - expected).abs() < 0.004,
                "width {n}, element {i}"
            );
        }
    }
}

#[test]
fn minimax_six_query_heads_per_kv_head_match_scalar_attention() {
    let mut c = MetalContext::new().unwrap();
    let (hd, nq, nkv, seq) = (128usize, 48usize, 8usize, 33usize);
    let half_values = |values: Vec<f32>| values.into_iter().map(f16::from_f32).collect::<Vec<_>>();
    let q = half_values(
        (0..nq * hd)
            .map(|i| (i as f32 * 0.031).sin() * 0.6)
            .collect(),
    );
    let k = half_values(
        (0..seq * nkv * hd)
            .map(|i| (i as f32 * 0.17).sin() * 0.3)
            .collect(),
    );
    // Distinct KV-head means expose mis-grouping even when attention is diffuse.
    let v = half_values(
        (0..seq * nkv * hd)
            .map(|i| {
                0.5 + ((i / hd) % nkv) as f32 / 4.0
                    + (i / (nkv * hd)) as f32 / 32.0
                    + (i % 5) as f32 / 64.0
            })
            .collect(),
    );
    let floats = |values: &[f16]| values.iter().map(|v| v.to_f32()).collect::<Vec<_>>();
    let scale = 1.0 / (hd as f32).sqrt();
    let expected = turbospark_compute::causal_attention(
        &floats(&q),
        &floats(&k),
        &floats(&v),
        hd,
        nq,
        nkv,
        seq,
        None,
        Some(scale),
    );
    let got = turbospark_gpu::attention_decode(
        &mut c, &q, &k, &v, hd as u32, nq as u32, nkv as u32, seq as u32, scale,
    )
    .unwrap();
    assert_eq!(got.len(), expected.len());
    let error = turbospark_compute::max_abs_diff(&floats(&got), &expected);
    assert!(error < 0.004, "MiniMax GQA max error {error}");
}
