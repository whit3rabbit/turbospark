//! Pin the Qwen2.5 7B MLX INT4 artifact independently of family defaults.
pub fn install_dir() -> std::path::PathBuf {
    let dir = std::env::var_os("TURBOSPARK_QWEN2_DENSE_INSTALL_DIR")
        .expect("set Qwen2.5 7B MLX INT4 install directory");
    let dir = std::path::PathBuf::from(dir);
    let arch = repack::peek_manifest_arch(&dir).unwrap();
    assert_eq!(arch.family, model_io::ModelFamily::Qwen2Dense);
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
        (28, 3584, 28, 4, 128, 18_944, 152_064, 0),
        "gate requires the pinned Qwen2.5 7B shape"
    );
    assert!(!arch.tie_word_embeddings);
    assert_eq!(
        std::fs::metadata(dir.join("model_weights.bin"))
            .unwrap()
            .len(),
        4_284_312_576,
        "gate requires the witnessed MLX INT4 resident payload"
    );
    dir
}
