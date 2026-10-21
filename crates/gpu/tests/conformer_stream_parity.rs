//! Runs `conformer_stream.metal` on real Metal hardware against CPU
//! references that mirror `crates/audio`'s streaming FastConformer
//! (`NemotronEncoder::stream_layers`): SiLU, GLU into the extended conv
//! input, the causal depthwise conv, and cached relative-position
//! attention, batched over speakers with different cache fill.
#![cfg(target_os = "macos")]

use turbospark_gpu::{
    encode_conformer_copy, encode_conformer_depthwise_causal, encode_conformer_glu_extend,
    encode_conformer_relpos_attention, encode_conformer_silu, read_f32_buffer,
    ConformerRelPosAttention, F32View, MetalBuffer, MetalContext,
};

/// Deterministic values in `[-1, 1)` (splitmix64 on the flat index).
fn unit(i: usize, salt: u64) -> f32 {
    let mut z = (i as u64)
        .wrapping_add(salt)
        .wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    ((z >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
}

fn values(n: usize, salt: u64, scale: f32) -> Vec<f32> {
    (0..n).map(|i| unit(i, salt) * scale).collect()
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn upload(context: &MetalContext, data: &[f32]) -> MetalBuffer {
    context.new_buffer_with_data(data)
}

#[test]
fn silu_and_copy_match_cpu() {
    let mut context = MetalContext::new().expect("Metal device");
    let x = values(4097, 1, 8.0);
    let a = upload(&context, &x);
    let y = context.new_output_buffer((x.len() * 4) as u64);
    let z = context.new_output_buffer((x.len() * 4) as u64);
    let pass = context.begin_pass();
    encode_conformer_silu(
        &mut context,
        &pass,
        F32View::new(&a),
        F32View::new(&y),
        x.len() as u32,
    )
    .unwrap();
    encode_conformer_copy(
        &mut context,
        &pass,
        F32View::new(&y),
        F32View::new(&z),
        x.len() as u32,
    )
    .unwrap();
    pass.commit_and_wait();
    let want: Vec<f32> = x.iter().map(|v| v * (1.0 / (1.0 + (-v).exp()))).collect();
    let got = read_f32_buffer(&z, x.len());
    assert!(
        max_diff(&got, &want) < 1e-5,
        "silu {}",
        max_diff(&got, &want)
    );
}

#[test]
fn glu_extend_and_depthwise_match_cpu() {
    let mut context = MetalContext::new().expect("Metal device");
    let (speakers, rows, valid, channels, kernel) = (3usize, 14usize, 9usize, 40usize, 9usize);
    let ctx = kernel - 1;
    let width = ctx + rows;
    let pw = values(speakers * rows * 2 * channels, 2, 3.0);
    // The cache region is pre-filled; the GLU must leave it alone.
    let mut ext_init = vec![0.0f32; speakers * width * channels];
    for b in 0..speakers {
        for t in 0..ctx {
            for c in 0..channels {
                ext_init[(b * width + t) * channels + c] = unit((b * ctx + t) * channels + c, 3);
            }
        }
    }
    let w = values(channels * kernel, 4, 0.5);
    let bias = values(channels, 5, 0.5);
    let pw_buf = upload(&context, &pw);
    let ext_buf = upload(&context, &ext_init);
    let w_buf = upload(&context, &w);
    let bias_buf = upload(&context, &bias);
    let out = context.new_output_buffer((speakers * rows * channels * 4) as u64);
    let out_nb = context.new_output_buffer((speakers * rows * channels * 4) as u64);
    let pass = context.begin_pass();
    encode_conformer_glu_extend(
        &mut context,
        &pass,
        F32View::new(&pw_buf),
        F32View::new(&ext_buf),
        speakers as u32,
        rows as u32,
        valid as u32,
        channels as u32,
        ctx as u32,
    )
    .unwrap();
    for (bias_view, dst) in [(Some(F32View::new(&bias_buf)), &out), (None, &out_nb)] {
        encode_conformer_depthwise_causal(
            &mut context,
            &pass,
            F32View::new(&ext_buf),
            F32View::new(&w_buf),
            bias_view,
            F32View::new(dst),
            speakers as u32,
            rows as u32,
            channels as u32,
            kernel as u32,
        )
        .unwrap();
    }
    pass.commit_and_wait();

    let mut ext_want = ext_init.clone();
    for b in 0..speakers {
        for t in 0..rows {
            for c in 0..channels {
                let src = (b * rows + t) * 2 * channels;
                let value = if t < valid {
                    pw[src + c] / (1.0 + (-pw[src + channels + c]).exp())
                } else {
                    0.0
                };
                ext_want[(b * width + ctx + t) * channels + c] = value;
            }
        }
    }
    let ext_got = read_f32_buffer(&ext_buf, ext_want.len());
    assert!(max_diff(&ext_got, &ext_want) < 1e-6, "glu");
    for (with_bias, dst) in [(true, &out), (false, &out_nb)] {
        let mut want = vec![0.0f32; speakers * rows * channels];
        for b in 0..speakers {
            for t in 0..rows {
                for c in 0..channels {
                    let mut acc = if with_bias { bias[c] } else { 0.0 };
                    for k in 0..kernel {
                        acc += ext_want[(b * width + t + k) * channels + c] * w[c * kernel + k];
                    }
                    want[(b * rows + t) * channels + c] = acc;
                }
            }
        }
        let got = read_f32_buffer(dst, want.len());
        assert!(
            max_diff(&got, &want) < 1e-5,
            "depthwise bias={with_bias}: {}",
            max_diff(&got, &want)
        );
    }
}

/// CPU reference of `RelPosAttention::forward_cached`'s head loop over
/// stacked speakers.
#[allow(clippy::too_many_arguments)]
fn relpos_reference(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    pos: &[f32],
    bias_u: &[f32],
    bias_v: &[f32],
    firsts: &[u32],
    s: ConformerRelPosAttention,
) -> Vec<f32> {
    let (rows, valid, heads, hd) = (
        s.rows as usize,
        s.valid as usize,
        s.heads as usize,
        s.head_dim as usize,
    );
    let (frames, window) = (s.attn_frames as usize, s.window as usize);
    let d = heads * hd;
    let kv_rows = frames + rows;
    let mut out = vec![0.0f32; firsts.len() * rows * d];
    for (b, &first) in firsts.iter().enumerate() {
        let first = first as usize;
        let end = frames + valid;
        for h in 0..heads {
            for a in 0..valid {
                let qr = &q[(b * rows + a) * d + h * hd..(b * rows + a) * d + (h + 1) * hd];
                let mut scores: Vec<f32> = (first..end)
                    .map(|key| {
                        let kr = &k[(b * kv_rows + key) * d + h * hd..][..hd];
                        let row = window - 1 + key - frames - a;
                        let pr = &pos[row * d + h * hd..][..hd];
                        let mut content = 0.0f32;
                        let mut position = 0.0f32;
                        for i in 0..hd {
                            content += (qr[i] + bias_u[h * hd + i]) * kr[i];
                            position += (qr[i] + bias_v[h * hd + i]) * pr[i];
                        }
                        (content + position) / (hd as f32).sqrt()
                    })
                    .collect();
                let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for x in scores.iter_mut() {
                    *x = (*x - m).exp();
                    sum += *x;
                }
                for x in scores.iter_mut() {
                    *x /= sum;
                }
                for (j, p) in scores.iter().enumerate() {
                    let vr = &v[(b * kv_rows + first + j) * d + h * hd..][..hd];
                    for i in 0..hd {
                        out[(b * rows + a) * d + h * hd + i] += p * vr[i];
                    }
                }
            }
        }
    }
    out
}

fn run_relpos(s: ConformerRelPosAttention, firsts: &[u32]) -> (Vec<f32>, Vec<f32>) {
    let mut context = MetalContext::new().expect("Metal device");
    let speakers = firsts.len();
    let d = (s.heads * s.head_dim) as usize;
    let rows = s.rows as usize;
    let kv_rows = (s.attn_frames + s.rows) as usize;
    let q = values(speakers * rows * d, 10, 1.0);
    let k = values(speakers * kv_rows * d, 11, 1.0);
    let v = values(speakers * kv_rows * d, 12, 1.0);
    let pos = values((2 * s.window as usize - 1) * d, 13, 1.0);
    let bias_u = values(d, 14, 0.2);
    let bias_v = values(d, 15, 0.2);
    let bufs: Vec<MetalBuffer> = [&q, &k, &v, &pos, &bias_u, &bias_v]
        .iter()
        .map(|x| upload(&context, x))
        .collect();
    let firsts_buf = context.new_buffer_with_data(firsts);
    // Garbage in the output proves the kernel writes the zero rows.
    let out = upload(&context, &vec![7.0f32; speakers * rows * d]);
    let pass = context.begin_pass();
    encode_conformer_relpos_attention(
        &mut context,
        &pass,
        F32View::new(&bufs[0]),
        F32View::new(&bufs[1]),
        F32View::new(&bufs[2]),
        F32View::new(&bufs[3]),
        F32View::new(&bufs[4]),
        F32View::new(&bufs[5]),
        &firsts_buf,
        F32View::new(&out),
        s,
    )
    .unwrap();
    pass.commit_and_wait();
    let got = read_f32_buffer(&out, speakers * rows * d);
    let want = relpos_reference(&q, &k, &v, &pos, &bias_u, &bias_v, firsts, s);
    (got, want)
}

/// The Multitalker geometry (8 heads of 128, 70 cache rows, 14-row chunks,
/// an 84-row window) for three speakers whose caches hold 0, 14 and all
/// 70 valid rows, on a full chunk and on a partial final chunk.
#[test]
fn relpos_attention_matches_cpu() {
    for valid in [14u32, 9] {
        let s = ConformerRelPosAttention {
            speakers: 3,
            rows: 14,
            valid,
            heads: 8,
            head_dim: 128,
            attn_frames: 70,
            window: 84,
        };
        let (got, want) = run_relpos(s, &[70, 56, 0]);
        let diff = max_diff(&got, &want);
        assert!(diff < 1e-5, "relpos valid {valid}: max |d| {diff}");
    }
}

/// A window shorter than `attn_frames + valid` would make the kernel's
/// position row underflow, and more than `MAX_STREAM_KEYS` keys would
/// overrun its score array. Both are refused before anything is encoded.
#[test]
fn relpos_attention_refuses_windows_and_key_counts_it_cannot_index() {
    let mut context = MetalContext::new().expect("Metal device");
    let buf = upload(&context, &[0.0f32; 4]);
    let firsts = context.new_buffer_with_data(&[0u32]);
    let ok = ConformerRelPosAttention {
        speakers: 1,
        rows: 14,
        valid: 14,
        heads: 1,
        head_dim: 4,
        attn_frames: 70,
        window: 84,
    };
    let cases = [
        (
            "window under attn_frames + valid",
            ConformerRelPosAttention { window: 83, ..ok },
        ),
        (
            "keys over the score array",
            ConformerRelPosAttention {
                attn_frames: 250,
                window: 264,
                ..ok
            },
        ),
        (
            "key count overflows u32",
            ConformerRelPosAttention {
                attn_frames: u32::MAX,
                window: u32::MAX,
                ..ok
            },
        ),
    ];
    for (name, shape) in cases {
        let pass = context.begin_pass();
        let view = || F32View::new(&buf);
        let result = encode_conformer_relpos_attention(
            &mut context,
            &pass,
            view(),
            view(),
            view(),
            view(),
            view(),
            view(),
            &firsts,
            view(),
            shape,
        );
        assert!(result.is_err(), "{name} was accepted");
    }
}
