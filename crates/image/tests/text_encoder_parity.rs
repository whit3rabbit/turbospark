use turbospark_image::fixtures::{
    model_subpath, read_npy_file_f32, read_npy_file_i64, run_array_path,
};
use turbospark_image::text_encoder::{encode_tokens, ShardedSafetensors, HIDDEN_SIZE};

#[test]
#[ignore = "opt-in real-model forward test (reads 8 GB text encoder weights, CPU compute)"]
fn test_text_encoder_real_conditioning_parity() {
    let model_dir = model_subpath("text_encoder");
    if !model_dir.exists() {
        eprintln!(
            "NOTE: skipping test_text_encoder_real_conditioning_parity; model missing at {}",
            model_dir.display()
        );
        return;
    }

    println!(
        "Opening sharded safetensors from {}...",
        model_dir.display()
    );
    let shards = ShardedSafetensors::open(&model_dir).expect("open text encoder shards");

    for case_name in ["lighting", "empty", "unicode"] {
        let ids_path = run_array_path(case_name, "token_ids.npy");
        let mask_path = run_array_path(case_name, "attention_mask.npy");
        let cond_path = run_array_path(case_name, "conditioning.npy");

        if !ids_path.exists() || !mask_path.exists() || !cond_path.exists() {
            eprintln!(
                "NOTE: skipping case {case_name}; missing array files under target/ig0/runs/{case_name}"
            );
            continue;
        }

        let token_ids = read_npy_file_i64(&ids_path).expect("read token_ids");
        let mask = read_npy_file_i64(&mask_path).expect("read attention_mask");
        let captured = read_npy_file_f32(&cond_path).expect("read captured conditioning");

        let retained_tokens = mask.data.iter().filter(|&&m| m != 0).count();
        assert!(retained_tokens > 0);
        let unpadded_ids = &token_ids.data[..retained_tokens];

        println!(
            "Running native CPU forward for case '{case_name}' with {retained_tokens} tokens..."
        );
        let actual = encode_tokens(unpadded_ids, &shards).expect("encode_tokens forward pass");

        assert_eq!(
            actual.len(),
            retained_tokens * HIDDEN_SIZE,
            "actual output length mismatch"
        );
        assert_eq!(
            captured.data.len(),
            retained_tokens * HIDDEN_SIZE,
            "captured conditioning length mismatch"
        );

        let mut max_abs = 0.0f32;
        let mut sum_sq_diff = 0.0f64;
        let mut sum_sq_ref = 0.0f64;

        for (a, b) in actual.iter().zip(&captured.data) {
            let diff = (a - b).abs();
            if diff > max_abs {
                max_abs = diff;
            }
            sum_sq_diff += ((a - b) as f64).powi(2);
            sum_sq_ref += (*b as f64).powi(2);
        }

        let rel_l2 = (sum_sq_diff / sum_sq_ref).sqrt();
        println!(
            "Case '{case_name}': retained_tokens={retained_tokens}, max_abs={max_abs:.6e}, rel_l2={rel_l2:.6e}"
        );

        // Rust FP32 CPU vs PyTorch BF16 MPS: expected scale ~1e-2 rel-L2
        assert!(
            rel_l2 < 0.015,
            "case {case_name} rel_l2={rel_l2} exceeds 0.015 tolerance"
        );
    }
}
