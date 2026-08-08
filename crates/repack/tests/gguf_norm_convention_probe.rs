//! THROWAWAY probe: does the GGUF-derived install's norm convention match
//! the MLX-derived install's? Delete once settled.
//!
//!   TURBOSPARK_GEMMA4_INSTALL_DIR=~/models/gemma4.gturbo \
//!   TURBOSPARK_GEMMA4_GGUF_INSTALL_DIR=~/models/gemma4-gguf.gturbo \
//!     cargo test -p turbospark-repack --test gguf_norm_convention_probe --release -- --ignored --nocapture

use std::path::Path;

fn bf16_tensor(dir: &Path, name: &str) -> Vec<f32> {
    let index =
        model_io::load_resident_index(&dir.join("model_weights.bin")).expect("resident index");
    let e = index.entries.get(name).unwrap_or_else(|| {
        panic!(
            "{name} absent from {}; have e.g. {:?}",
            dir.display(),
            index.entries.keys().take(5).collect::<Vec<_>>()
        )
    });
    assert_eq!(e.dtype, 1, "{name} in {} is not BF16", dir.display());
    let bytes = std::fs::read(dir.join("model_weights.bin")).expect("weights");
    let start = e.file_offset as usize;
    bytes[start..start + e.size_bytes as usize]
        .chunks_exact(2)
        .map(|c| compute::bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect()
}

#[test]
#[ignore = "needs both installs"]
fn norms_agree_between_the_two_installs() {
    let mlx = std::path::PathBuf::from(std::env::var_os("TURBOSPARK_GEMMA4_INSTALL_DIR").unwrap());
    let gguf =
        std::path::PathBuf::from(std::env::var_os("TURBOSPARK_GEMMA4_GGUF_INSTALL_DIR").unwrap());

    for name in [
        "language_model.model.layers.0.input_layernorm.weight",
        "language_model.model.layers.0.post_attention_layernorm.weight",
        "language_model.model.layers.0.self_attn.q_norm.weight",
        "language_model.model.norm.weight",
        // The three "matched by shape" mappings.
        "language_model.model.layers.0.router.scale",
        "language_model.model.layers.0.router.per_expert_scale",
        "language_model.model.layers.0.layer_scalar",
        "language_model.model.layers.17.router.per_expert_scale",
    ] {
        let a = bf16_tensor(&mlx, name);
        let b = bf16_tensor(&gguf, name);
        assert_eq!(a.len(), b.len(), "{name} length");
        let n = a.len() as f32;
        let mean_a = a.iter().sum::<f32>() / n;
        let mean_b = b.iter().sum::<f32>() / n;
        let diff: Vec<f32> = a.iter().zip(&b).map(|(x, y)| y - x).collect();
        let mean_d = diff.iter().sum::<f32>() / n;
        let max_dev = diff.iter().fold(0.0f32, |m, &d| m.max((d - mean_d).abs()));
        let corr = compute::pearson(&a, &b);
        println!(
            "{name}\n  mlx mean {mean_a:+.5}  gguf mean {mean_b:+.5}  \
             mean(gguf-mlx) {mean_d:+.5}  max deviation from that constant {max_dev:.6}  \
             pearson {corr:+.5}"
        );
    }
}
