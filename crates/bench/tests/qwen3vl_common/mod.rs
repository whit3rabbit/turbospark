//! Pin the Qwen3-VL 4B MLX INT4 artifact independently of family defaults.
pub fn install_dir() -> std::path::PathBuf {
    let dir = std::env::var_os("TURBOSPARK_QWEN3VL_INSTALL_DIR")
        .expect("set Qwen3-VL 4B MLX INT4 install directory");
    let dir = std::path::PathBuf::from(dir);
    let arch = repack::peek_manifest_arch(&dir).unwrap();
    assert_eq!(arch.family, model_io::ModelFamily::Qwen3Vl);
    assert_eq!(
        (
            arch.num_layers,
            arch.hidden_size,
            arch.num_heads,
            arch.num_full_kv_heads,
            arch.full_head_dim,
            arch.intermediate_size,
            arch.vocab_size,
            arch.num_experts
        ),
        (36, 2560, 32, 8, 128, 9_728, 151_936, 0),
        "gate requires the pinned Qwen3-VL 4B shape"
    );
    assert!(arch.tie_word_embeddings);
    assert_eq!(
        std::fs::metadata(dir.join("model_weights.bin"))
            .unwrap()
            .len(),
        2_262_985_728,
        "gate requires the witnessed MLX INT4 resident payload"
    );
    dir
}
