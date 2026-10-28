//! Pin the LFM3-1.7B MLX 4-bit artifact (`LiquidAI/LFM3-1.7B-MLX-4bit` @
//! d0ca73dc) independently of family defaults.
pub fn install_dir() -> std::path::PathBuf {
    let dir = std::env::var_os("TURBOSPARK_LFM3_INSTALL_DIR")
        .expect("set the LFM3-1.7B MLX 4-bit install directory");
    let dir = std::path::PathBuf::from(dir);
    let arch = repack::peek_manifest_arch(&dir).unwrap();
    assert_eq!(
        arch,
        model_io::lfm3_1_7b(),
        "gate requires the LFM3-1.7B shape"
    );
    assert_eq!(
        std::fs::metadata(dir.join("model_weights.bin"))
            .unwrap()
            .len(),
        973_057_536,
        "gate requires the witnessed MLX 4-bit resident payload"
    );
    dir
}
