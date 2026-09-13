use turbospark_image::fixtures::{fixture_root, model_subpath, read_npy_file_f32, run_array_path};
use turbospark_image::vae::{
    conv2d, decoded_to_rgb8, encode_rgb8_png, group_norm, upsample_nearest_2x, ResnetBlock2D,
    VaeDecoder, GROUP_NORM_EPS, GROUP_NORM_GROUPS,
};

fn relative_l2(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut diff_norm_sq = 0.0f64;
    let mut a_norm_sq = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let diff = (*x as f64) - (*y as f64);
        diff_norm_sq += diff * diff;
        a_norm_sq += (*x as f64) * (*x as f64);
    }
    (diff_norm_sq.sqrt() / a_norm_sq.sqrt().max(1e-30)) as f32
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut max_diff = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        let diff = (x - y).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }
    max_diff
}

#[test]
fn test_vae_conv2d_basic() {
    // 1 channel in, 1 channel out, 3x3 input, 3x3 kernel, pad 1, stride 1
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let w = vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
    let b = vec![0.5];

    let out = conv2d(&x, 1, 3, 3, &w, Some(&b), 1, 3, 3, 1, 1).unwrap();
    assert_eq!(out.len(), 9);
    for i in 0..9 {
        assert!((out[i] - (x[i] + 0.5)).abs() < 1e-6);
    }
}

#[test]
fn test_vae_group_norm_basic() {
    // 32 channels, 2x2 spatial size
    let channels = 32;
    let (h, w) = (2, 2);
    let mut x = Vec::with_capacity(channels * h * w);
    for c in 0..channels {
        for _ in 0..h * w {
            x.push((c + 1) as f32);
        }
    }
    let weight = vec![1.0f32; channels];
    let bias = vec![0.0f32; channels];

    // For constant channel values within each 1-channel group, mean = val, var = 0
    let normed = group_norm(
        &x,
        channels,
        h,
        w,
        &weight,
        &bias,
        GROUP_NORM_GROUPS,
        GROUP_NORM_EPS,
    )
    .unwrap();

    assert_eq!(normed.len(), x.len());
    for &val in &normed {
        assert!(val.abs() < 1e-3);
    }
}

#[test]
fn test_vae_upsample_basic() {
    // 1 channel, 2x2 -> 4x4
    let x = vec![1.0, 2.0, 3.0, 4.0];
    let up = upsample_nearest_2x(&x, 1, 2, 2);
    assert_eq!(up.len(), 16);
    let expected = vec![
        1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 3.0, 3.0, 4.0, 4.0,
    ];
    assert_eq!(up, expected);
}

#[test]
fn test_vae_resnet_block_basic() {
    let channels = 32;
    let (h, w) = (4, 4);
    let x = vec![1.0f32; channels * h * w];

    let block = ResnetBlock2D {
        in_channels: channels,
        out_channels: channels,
        norm1_weight: vec![1.0; channels],
        norm1_bias: vec![0.0; channels],
        conv1_weight: vec![0.0; channels * channels * 9],
        conv1_bias: vec![0.0; channels],
        norm2_weight: vec![1.0; channels],
        norm2_bias: vec![0.0; channels],
        conv2_weight: vec![0.0; channels * channels * 9],
        conv2_bias: vec![0.0; channels],
        conv_shortcut_weight: None,
        conv_shortcut_bias: None,
    };

    let out = block.forward(&x, h, w).unwrap();
    assert_eq!(out.len(), x.len());
    // Since conv2 output is zero, out = shortcut = x
    for i in 0..out.len() {
        assert!((out[i] - x[i]).abs() < 1e-6);
    }
}

#[test]
fn test_decoded_rgb_and_png_contract() {
    // CHW values cover clamping, midpoint mapping, and channel interleaving.
    let decoded = vec![-1.0, 1.0, 0.0, 0.0, 1.5, -1.5];
    let rgb = decoded_to_rgb8(&decoded, 1, 2).expect("convert decoded RGB");
    assert_eq!(rgb, vec![0, 128, 255, 255, 128, 0]);

    let png = encode_rgb8_png(&rgb, 2, 1).expect("encode PNG");
    let decoder = png::Decoder::new(std::io::Cursor::new(png));
    let mut reader = decoder.read_info().expect("read PNG info");
    let mut buffer = vec![0; reader.output_buffer_size().expect("PNG output buffer size")];
    let info = reader.next_frame(&mut buffer).expect("decode PNG pixels");
    assert_eq!(
        (info.width, info.height, info.color_type),
        (2, 1, png::ColorType::Rgb)
    );
    assert_eq!(&buffer[..info.buffer_size()], rgb.as_slice());
}

#[test]
fn test_decoded_rgb_rejects_nonfinite_values() {
    let err = decoded_to_rgb8(&[f32::NAN, 0.0, 0.0], 1, 1).expect_err("non-finite decoded tensor");
    assert!(err.contains("non-finite"));
}

#[test]
#[ignore = "opt-in real-model forward test (reads 160 MB VAE weights, CPU compute)"]
fn test_vae_real_decode_parity() {
    let model_dir = model_subpath("vae");
    let weights_path = model_dir.join("diffusion_pytorch_model.safetensors");
    let latents_path = run_array_path("lighting", "final_latents.npy");
    let pixels_path = run_array_path("lighting", "decoded_pixels.npy");
    let mlx_pixels_path = fixture_root()
        .join("target")
        .join("ig0")
        .join("runs")
        .join("vae-compare")
        .join("mlx_decoded_pixels.npy");

    assert!(
        weights_path.exists(),
        "missing VAE checkpoint at {}",
        weights_path.display()
    );
    assert!(
        latents_path.exists(),
        "missing final latents at {}",
        latents_path.display()
    );

    println!("Loading VAE weights from {}...", weights_path.display());
    let sf =
        model_io::safetensors::SafetensorsFile::open(&weights_path).expect("open VAE safetensors");
    let decoder = VaeDecoder::from_safetensors(&sf).expect("build VaeDecoder from safetensors");

    let latents = read_npy_file_f32(&latents_path).expect("read final_latents.npy");
    assert_eq!(latents.shape, vec![1, 16, 128, 128]);

    println!("Running native VAE decode for 1024x1024 output...");
    let actual = decoder
        .decode(&latents.data, 128, 128)
        .expect("decode latents");
    assert_eq!(actual.len(), 3 * 1024 * 1024);

    if pixels_path.exists() {
        let expected = read_npy_file_f32(&pixels_path).expect("read decoded_pixels.npy");
        let max_err = max_abs_diff(&actual, &expected.data);
        let rel_err = relative_l2(&actual, &expected.data);
        println!("VAE decode vs Diffusers: max_abs={max_err:.4e}, rel_l2={rel_err:.4e}");
        // Frozen tolerances for fixed latent fixture: max_abs <= 6e-5, rel_l2 <= 3e-6
        assert!(
            max_err <= 6e-5,
            "VAE max_abs {max_err} exceeds frozen 6e-5 limit"
        );
        assert!(
            rel_err <= 3e-6,
            "VAE rel_l2 {rel_err} exceeds frozen 3e-6 limit"
        );
    }

    if mlx_pixels_path.exists() {
        let expected_mlx =
            read_npy_file_f32(&mlx_pixels_path).expect("read mlx_decoded_pixels.npy");
        let max_err_mlx = max_abs_diff(&actual, &expected_mlx.data);
        let rel_err_mlx = relative_l2(&actual, &expected_mlx.data);
        println!("VAE decode vs MFLUX: max_abs={max_err_mlx:.4e}, rel_l2={rel_err_mlx:.4e}");
        assert!(
            max_err_mlx <= 6e-5,
            "VAE max_abs vs MLX {max_err_mlx} exceeds frozen 6e-5 limit"
        );
        assert!(
            rel_err_mlx <= 3e-6,
            "VAE rel_l2 vs MLX {rel_err_mlx} exceeds frozen 3e-6 limit"
        );
    }
}
