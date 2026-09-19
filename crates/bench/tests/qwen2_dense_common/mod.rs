//! Pin the Qwen2.5 7B artifacts independently of family defaults.
//!
//! Three installs share one shape (the same 28-layer, 3584-hidden base
//! checkpoint) and differ in intake: the MLX INT4 conversion, the official
//! single-file Q3_K_M GGUF, and the single-file mradermacher Q4_K_M GGUF.
//! Each gate names its artifact by RESIDENT PAYLOAD BYTES, which is the one
//! thing an env var pointing at the wrong directory cannot fake.

pub fn install_dir() -> std::path::PathBuf {
    let (dir, arch) = pinned(
        "TURBOSPARK_QWEN2_DENSE_INSTALL_DIR",
        "Qwen2.5 7B MLX INT4",
        4_284_312_576,
    );
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
    dir
}

/// The official single-file Q3_K_M GGUF (Dense Qwen2 roadmap item): Q3_K
/// attention and FFN projections, Q4_K embedding table, Q6_K head, three
/// Q5_K tensors, F32 norms.
pub fn gguf_q3_k_m_install_dir() -> std::path::PathBuf {
    let (dir, arch) = pinned(
        "TURBOSPARK_QWEN2_GGUF_Q3KM_INSTALL_DIR",
        "Qwen2.5 7B GGUF Q3_K_M",
        3_801_820_160,
    );
    assert_eq!(
        (
            arch.num_layers,
            arch.hidden_size,
            arch.vocab_size,
            arch.num_experts
        ),
        (28, 3584, 152_064, 0),
        "gate requires the pinned Qwen2.5 7B shape"
    );
    dir
}

/// The single-file mradermacher Q4_K_M GGUF: the roadmap's second real
/// Qwen2 GGUF artifact (the official repo splits its Q4_K_M, and split GGUF
/// is out of scope for the dense qwen2 walk).
pub fn gguf_q4_k_m_install_dir() -> std::path::PathBuf {
    let (dir, arch) = pinned(
        "TURBOSPARK_QWEN2_GGUF_Q4KM_INSTALL_DIR",
        "Qwen2.5 7B GGUF Q4_K_M",
        4_679_677_824,
    );
    assert_eq!(
        (
            arch.num_layers,
            arch.hidden_size,
            arch.vocab_size,
            arch.num_experts
        ),
        (28, 3584, 152_064, 0),
        "gate requires the pinned Qwen2.5 7B shape"
    );
    dir
}

fn pinned(
    env: &str,
    label: &str,
    expected_bytes: u64,
) -> (std::path::PathBuf, model_io::ArchConfig) {
    let dir = std::env::var_os(env)
        .unwrap_or_else(|| panic!("set the {label} install directory ({env})"));
    let dir = std::path::PathBuf::from(dir);
    let arch = repack::peek_manifest_arch(&dir).unwrap();
    assert_eq!(arch.family, model_io::ModelFamily::Qwen2Dense, "{label}");
    assert_eq!(
        std::fs::metadata(dir.join("model_weights.bin"))
            .unwrap()
            .len(),
        expected_bytes,
        "{label}: gate requires the witnessed resident payload"
    );
    (dir, arch)
}
