//! Runs `nemotron_diar.metal` on real Metal hardware against a CPU
//! reference of `ops::rope_neox` (the Nemotron-3 diarization encoder's
//! rotary), reading Q and K straight out of a fused QKV row layout.
#![cfg(target_os = "macos")]

use turbospark_gpu::{encode_nemotron_rope_neox, read_f32_buffer, F32View, MetalContext};

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

/// `ops::rope_neox` on row-major `[rows, heads * head_dim]` rows read at
/// `x_stride`, into a dense `[rows, heads * head_dim]` result.
#[allow(clippy::too_many_arguments)]
fn rope_reference(
    x: &[f32],
    cos: &[f32],
    sin: &[f32],
    rows: usize,
    heads: usize,
    head_dim: usize,
    rotary: usize,
    x_stride: usize,
) -> Vec<f32> {
    let width = heads * head_dim;
    let half = rotary / 2;
    let mut out = vec![0.0f32; rows * width];
    for t in 0..rows {
        for h in 0..heads {
            let src = &x[t * x_stride + h * head_dim..t * x_stride + (h + 1) * head_dim];
            let dst = &mut out[t * width + h * head_dim..t * width + (h + 1) * head_dim];
            dst.copy_from_slice(src);
            for d in 0..half {
                let (a, b) = (src[d], src[d + half]);
                let (c, s) = (cos[t * half + d], sin[t * half + d]);
                dst[d] = a * c - b * s;
                dst[d + half] = b * c + a * s;
            }
        }
    }
    out
}

fn run(rows: usize, heads: usize, head_dim: usize, rotary: usize, salt: u64) -> f32 {
    let mut context = MetalContext::new().expect("Metal device");
    let width = heads * head_dim;
    // A fused `[rows, 3 * width]` QKV layout; rotate the K third.
    let x_stride = 3 * width;
    let x: Vec<f32> = (0..rows * x_stride).map(|i| unit(i, salt)).collect();
    let half = rotary / 2;
    let cos: Vec<f32> = (0..rows * half).map(|i| unit(i, salt ^ 1)).collect();
    let sin: Vec<f32> = (0..rows * half).map(|i| unit(i, salt ^ 2)).collect();
    let x_buf = context.new_buffer_with_data(&x);
    let cos_buf = context.new_buffer_with_data(&cos);
    let sin_buf = context.new_buffer_with_data(&sin);
    let y_buf = context.new_output_buffer((rows * width * 4) as u64);
    let pass = context.begin_pass();
    encode_nemotron_rope_neox(
        &mut context,
        &pass,
        F32View::at(&x_buf, width as u64),
        F32View::new(&cos_buf),
        F32View::new(&sin_buf),
        F32View::new(&y_buf),
        rows as u32,
        heads as u32,
        head_dim as u32,
        rotary as u32,
        x_stride as u32,
        width as u32,
    )
    .expect("encode rope");
    pass.commit_and_wait();
    let gpu = read_f32_buffer(&y_buf, rows * width);
    let want = rope_reference(
        &x[width..],
        &cos,
        &sin,
        rows,
        heads,
        head_dim,
        rotary,
        x_stride,
    );
    gpu.iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max)
}

/// The diarizer's geometry (8 heads of 64, full rotary, a 542-row window)
/// and a partial rotary with pass-through lanes. Fast-math contraction may
/// fuse the multiply-adds, so the bound is a few ulps of values near 1.
#[test]
fn rope_neox_rows_matches_cpu_reference() {
    for &(rows, heads, head_dim, rotary) in &[(542, 8, 64, 64), (7, 3, 16, 8), (1, 1, 2, 2)] {
        let diff = run(rows, heads, head_dim, rotary, rows as u64);
        assert!(
            diff < 1e-6,
            "rope rows {rows} heads {heads} dim {head_dim} rotary {rotary}: max |d| {diff}"
        );
    }
}

#[test]
fn rope_neox_rows_rejects_short_strides() {
    let mut context = MetalContext::new().expect("Metal device");
    let buf = context.new_buffer_with_data(&[0.0f32; 64]);
    let pass = context.begin_pass();
    let err = encode_nemotron_rope_neox(
        &mut context,
        &pass,
        F32View::new(&buf),
        F32View::new(&buf),
        F32View::new(&buf),
        F32View::new(&buf),
        2,
        2,
        8,
        8,
        15,
        16,
    );
    assert!(err.is_err());
    pass.commit_and_wait();
}
