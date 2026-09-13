use turbospark_image::fixtures::{
    fixture_root, read_npy_bool, read_npy_complex64, read_npy_f32, read_npy_file_complex64,
    read_npy_i64, read_npz_file, run_array_path,
};
use turbospark_image::patchify::{
    create_coordinate_grid, pad_with_ids, patchify_image, unpatchify,
};
use turbospark_image::rope::RopeEmbedder;
use turbospark_image::transformer::{AdaLnModulation, ZImageTransformerBlock};

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
#[ignore = "opt-in external synthetic reference fixture"]
fn test_rope_three_axis_synthetic_parity() {
    let npz_path = fixture_root()
        .join("target")
        .join("ig0")
        .join("runs")
        .join("blocks")
        .join("synthetic_blocks.npz");

    assert!(
        npz_path.exists(),
        "missing synthetic reference fixture at {}",
        npz_path.display()
    );

    let entries = read_npz_file(&npz_path).expect("read synthetic_blocks.npz");
    let embedder = RopeEmbedder::new(&[32, 48, 48], &[1536, 512, 512], 256.0);

    for tokens in [1, 16, 35] {
        let ids_raw = read_npy_i64(entries.get(&format!("ids_{tokens}.npy")).unwrap()).unwrap();
        assert_eq!(ids_raw.shape, vec![tokens, 3]);

        let mut coords = Vec::with_capacity(tokens);
        for row in ids_raw.data.chunks_exact(3) {
            coords.push([row[0] as i32, row[1] as i32, row[2] as i32]);
        }

        let computed_pairs = embedder.embed_ids(&coords).expect("embed_ids");
        assert_eq!(computed_pairs.len(), tokens * 64);
    }

    // Also compare against captured block_00_freqs.npy from real lighting run
    let freqs_path = run_array_path("lighting", "block_00_freqs.npy");
    if freqs_path.exists() {
        let captured = read_npy_file_complex64(&freqs_path).expect("read block_00_freqs.npy");
        // Shape is [1, 4128, 64]
        assert_eq!(captured.shape, vec![1, 4128, 64]);
        // Reconstruct position IDs for lighting:
        // 32 caption tokens starting at (1, 0, 0) with padding to 32 at (0, 0, 0)
        // 4096 image tokens starting at (33, 0, 0)
        let (_, cap_ids, _) =
            pad_with_ids(&vec![0.0f32; 27 * 4], 4, (32, 1, 1), (1, 0, 0), None).unwrap();
        assert_eq!(cap_ids.len(), 32);

        let img_grid = create_coordinate_grid((1, 64, 64), (33, 0, 0));
        assert_eq!(img_grid.len(), 4096);

        // Unified sequence order is [image, caption]
        let mut unified_ids = Vec::with_capacity(4128);
        unified_ids.extend_from_slice(&img_grid);
        unified_ids.extend_from_slice(&cap_ids);

        let computed = embedder
            .embed_ids(&unified_ids)
            .expect("embed lighting IDs");
        assert_eq!(computed.len(), captured.data.len());

        let mut max_abs_diff = 0.0f32;
        for (comp, capt) in computed.iter().zip(captured.data.iter()) {
            let r_diff = (comp.0 - capt.0).abs();
            let i_diff = (comp.1 - capt.1).abs();
            if r_diff > max_abs_diff {
                max_abs_diff = r_diff;
            }
            if i_diff > max_abs_diff {
                max_abs_diff = i_diff;
            }
        }
        // Frozen RoPE tolerance: <= 3e-6
        assert!(
            max_abs_diff <= 3e-6,
            "RoPE max absolute error {max_abs_diff} exceeds frozen 3e-6 limit"
        );
    }
}

#[test]
fn test_patchify_unpatchify_contract() {
    let (c, f, h, w) = (16, 1, 128, 128);
    let patch_size = 2;
    let f_patch_size = 1;

    let original: Vec<f32> = (0..c * f * h * w)
        .map(|i| ((i % 1000) as f32) * 0.001)
        .collect();

    let (patches, size, tokens) =
        patchify_image(&original, c, f, h, w, patch_size, f_patch_size).expect("patchify");

    assert_eq!(size, (1, 128, 128));
    assert_eq!(tokens, (1, 64, 64));
    assert_eq!(patches.len(), 4096 * 64);

    let rec = unpatchify(
        &patches,
        tokens.0,
        tokens.1,
        tokens.2,
        f_patch_size,
        patch_size,
        c,
    )
    .expect("unpatchify");

    assert_eq!(original.len(), rec.len());
    let diff = max_abs_diff(&original, &rec);
    assert_eq!(diff, 0.0f32, "patchify/unpatchify roundtrip error");
}

#[test]
#[ignore = "opt-in external synthetic reference fixture"]
fn test_transformer_synthetic_block_parity() {
    let npz_path = fixture_root()
        .join("target")
        .join("ig0")
        .join("runs")
        .join("blocks")
        .join("synthetic_blocks.npz");

    assert!(
        npz_path.exists(),
        "missing synthetic reference fixture at {}",
        npz_path.display()
    );

    let entries = read_npz_file(&npz_path).expect("read synthetic_blocks.npz");

    // Load block weights
    let get_f32 = |name: &str| -> Vec<f32> {
        let key = format!("{name}.npy");
        read_npy_f32(entries.get(&key).unwrap_or_else(|| panic!("missing {key}")))
            .unwrap()
            .data
    };

    let dim = 384;
    let num_heads = 3;
    let head_dim = 128;

    let block = ZImageTransformerBlock {
        dim,
        num_heads,
        head_dim,
        norm_eps: 1e-5,
        attention_norm1: get_f32("weight_attention_norm1.weight"),
        to_q: get_f32("weight_attention.to_q.weight"),
        to_k: get_f32("weight_attention.to_k.weight"),
        to_v: get_f32("weight_attention.to_v.weight"),
        norm_q: get_f32("weight_attention.norm_q.weight"),
        norm_k: get_f32("weight_attention.norm_k.weight"),
        to_out: get_f32("weight_attention.to_out.0.weight"),
        attention_norm2: get_f32("weight_attention_norm2.weight"),
        ffn_norm1: get_f32("weight_ffn_norm1.weight"),
        w1: get_f32("weight_feed_forward.w1.weight"),
        w2: get_f32("weight_feed_forward.w2.weight"),
        w3: get_f32("weight_feed_forward.w3.weight"),
        ffn_norm2: get_f32("weight_ffn_norm2.weight"),
        modulation: Some(AdaLnModulation::new(
            dim,
            get_f32("weight_adaLN_modulation.0.weight"),
            get_f32("weight_adaLN_modulation.0.bias"),
        )),
    };

    let embedder = RopeEmbedder::new(&[32, 48, 48], &[1536, 512, 512], 256.0);

    for tokens in [1, 16, 35] {
        let x = read_npy_f32(entries.get(&format!("x_{tokens}.npy")).unwrap())
            .unwrap()
            .data;
        let modulation = read_npy_f32(entries.get(&format!("modulation_{tokens}.npy")).unwrap())
            .unwrap()
            .data;
        let mask = read_npy_bool(entries.get(&format!("mask_{tokens}.npy")).unwrap())
            .unwrap()
            .data;
        let ids_raw = read_npy_i64(entries.get(&format!("ids_{tokens}.npy")).unwrap()).unwrap();
        let expected_torch = read_npy_f32(entries.get(&format!("torch_{tokens}.npy")).unwrap())
            .unwrap()
            .data;
        let expected_mlx = read_npy_f32(entries.get(&format!("mlx_{tokens}.npy")).unwrap())
            .unwrap()
            .data;

        let mut coords = Vec::with_capacity(tokens);
        for row in ids_raw.data.chunks_exact(3) {
            coords.push([row[0] as i32, row[1] as i32, row[2] as i32]);
        }
        let freqs = embedder.embed_ids(&coords).expect("embed_ids");

        let actual = block
            .forward(&x, Some(&mask), &freqs, Some(&modulation))
            .expect("block forward");

        let max_err_torch = max_abs_diff(&actual, &expected_torch);
        let rel_err_torch = relative_l2(&actual, &expected_torch);

        let max_err_mlx = max_abs_diff(&actual, &expected_mlx);
        let rel_err_mlx = relative_l2(&actual, &expected_mlx);

        println!(
            "Synthetic block tokens={tokens}: vs_torch(max_abs={max_err_torch:.4e}, rel_l2={rel_err_torch:.4e}), vs_mlx(max_abs={max_err_mlx:.4e}, rel_l2={rel_err_mlx:.4e})"
        );

        // Frozen tolerances for bounded FP32 synthetic blocks: max_abs <= 1e-5, rel_l2 <= 1e-6
        assert!(
            max_err_torch <= 1e-5,
            "tokens={tokens}: max_abs {max_err_torch} exceeds 1e-5"
        );
        assert!(
            rel_err_torch <= 1e-6,
            "tokens={tokens}: relative_l2 {rel_err_torch} exceeds 1e-6"
        );
        assert!(
            max_err_mlx <= 1e-5,
            "tokens={tokens}: max_abs vs MLX {max_err_mlx} exceeds 1e-5"
        );
        assert!(
            rel_err_mlx <= 1e-6,
            "tokens={tokens}: relative_l2 vs MLX {rel_err_mlx} exceeds 1e-6"
        );
    }
}

#[test]
#[ignore = "opt-in real-model forward test (reads transformer shard, CPU compute)"]
fn test_transformer_checkpoint_block_parity() {
    let npz_path = fixture_root()
        .join("target")
        .join("ig0")
        .join("runs")
        .join("checkpoint-block")
        .join("checkpoint_block.npz");
    let model_dir = turbospark_image::fixtures::model_subpath("transformer");
    let shard_path = model_dir.join("diffusion_pytorch_model-00001-of-00003.safetensors");

    assert!(
        npz_path.exists(),
        "missing checkpoint fixture at {}",
        npz_path.display()
    );
    assert!(
        shard_path.exists(),
        "missing checkpoint shard at {}",
        shard_path.display()
    );

    let entries = read_npz_file(&npz_path).expect("read checkpoint_block.npz");
    let sf = model_io::safetensors::SafetensorsFile::open(&shard_path)
        .expect("open transformer shard 1");

    let load_weight = |name: &str| -> Vec<f32> {
        let full_name = format!("layers.0.{name}");
        sf.load_as_f32(&full_name)
            .unwrap_or_else(|_| panic!("load tensor {full_name}"))
    };

    let dim = 3840;
    let num_heads = 30;
    let head_dim = 128;

    let block = ZImageTransformerBlock {
        dim,
        num_heads,
        head_dim,
        norm_eps: 1e-5,
        attention_norm1: load_weight("attention_norm1.weight"),
        to_q: load_weight("attention.to_q.weight"),
        to_k: load_weight("attention.to_k.weight"),
        to_v: load_weight("attention.to_v.weight"),
        norm_q: load_weight("attention.norm_q.weight"),
        norm_k: load_weight("attention.norm_k.weight"),
        to_out: load_weight("attention.to_out.0.weight"),
        attention_norm2: load_weight("attention_norm2.weight"),
        ffn_norm1: load_weight("ffn_norm1.weight"),
        w1: load_weight("feed_forward.w1.weight"),
        w2: load_weight("feed_forward.w2.weight"),
        w3: load_weight("feed_forward.w3.weight"),
        ffn_norm2: load_weight("ffn_norm2.weight"),
        modulation: Some(AdaLnModulation::new(
            dim,
            load_weight("adaLN_modulation.0.weight"),
            load_weight("adaLN_modulation.0.bias"),
        )),
    };

    let input = read_npy_f32(entries.get("input.npy").unwrap())
        .unwrap()
        .data;
    let mask = read_npy_bool(entries.get("mask.npy").unwrap())
        .unwrap()
        .data;
    let freqs_raw = read_npy_complex64(entries.get("freqs.npy").unwrap())
        .unwrap()
        .data;
    let modulation = read_npy_f32(entries.get("modulation.npy").unwrap())
        .unwrap()
        .data;
    let expected_torch = read_npy_f32(entries.get("torch_fp32.npy").unwrap())
        .unwrap()
        .data;
    let expected_mlx = read_npy_f32(entries.get("mlx_fp32.npy").unwrap())
        .unwrap()
        .data;

    let actual = block
        .forward(&input, Some(&mask), &freqs_raw, Some(&modulation))
        .expect("forward checkpoint block");

    let max_err_torch = max_abs_diff(&actual, &expected_torch);
    let rel_err_torch = relative_l2(&actual, &expected_torch);

    let max_err_mlx = max_abs_diff(&actual, &expected_mlx);
    let rel_err_mlx = relative_l2(&actual, &expected_mlx);

    println!(
        "Checkpoint block (tokens=64, dim=3840): vs_torch(max_abs={max_err_torch:.4e}, rel_l2={rel_err_torch:.4e}), vs_mlx(max_abs={max_err_mlx:.4e}, rel_l2={rel_err_mlx:.4e})"
    );

    // Frozen Phase 0 tolerance: max_abs <= 3e-5, relative_l2 <= 1e-6
    assert!(
        max_err_torch <= 3e-5,
        "checkpoint block max_abs vs torch {max_err_torch} exceeds 3e-5"
    );
    assert!(
        rel_err_torch <= 1e-6,
        "checkpoint block rel_l2 vs torch {rel_err_torch} exceeds 1e-6"
    );
    assert!(
        max_err_mlx <= 3e-5,
        "checkpoint block max_abs vs MLX {max_err_mlx} exceeds 3e-5"
    );
    assert!(
        rel_err_mlx <= 1e-6,
        "checkpoint block rel_l2 vs MLX {rel_err_mlx} exceeds 1e-6"
    );
}
