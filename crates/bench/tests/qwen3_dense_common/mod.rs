//! Pin the small dense regression artifact independently of family defaults.
pub fn install_dir() -> std::path::PathBuf {
    let dir = std::env::var_os("TURBOSPARK_QWEN3_DENSE_INSTALL_DIR")
        .expect("set Qwen3-0.6B Q8_0 install directory");
    let dir = std::path::PathBuf::from(dir);
    let arch = repack::peek_manifest_arch(&dir).unwrap();
    assert_eq!(arch.family, model_io::ModelFamily::Qwen3Dense);
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
        (28, 1024, 16, 8, 128, 3072, 151936, 0),
        "gate requires the pinned Qwen3-0.6B shape"
    );
    assert!(arch.tie_word_embeddings);
    assert_eq!(
        std::fs::metadata(dir.join("model_weights.bin"))
            .unwrap()
            .len(),
        633413632,
        "gate requires the witnessed Q8_0 resident payload"
    );
    dir
}
