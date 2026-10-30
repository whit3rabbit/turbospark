//! Pin the two LFM3-1.7B artifacts independently of family defaults: the
//! MLX 4-bit conversion (`LiquidAI/LFM3-1.7B-MLX-4bit` @ d0ca73dc) and the
//! GGUF Q8_0 (`LiquidAI/LFM3-1.7B-GGUF` @ b3f2ae5d).
#![allow(dead_code)]

fn pinned(env: &str, what: &str, payload: u64) -> std::path::PathBuf {
    let dir = std::env::var_os(env).unwrap_or_else(|| panic!("set {env} to the {what} install"));
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
        payload,
        "gate requires the witnessed {what} resident payload"
    );
    dir
}

/// The MLX 4-bit install.
pub fn install_dir() -> std::path::PathBuf {
    pinned("TURBOSPARK_LFM3_INSTALL_DIR", "MLX 4-bit", 973_057_536)
}

/// The GGUF Q8_0 install.
pub fn q8_install_dir() -> std::path::PathBuf {
    pinned("TURBOSPARK_LFM3_Q8_INSTALL_DIR", "GGUF Q8_0", 1_837_264_384)
}
