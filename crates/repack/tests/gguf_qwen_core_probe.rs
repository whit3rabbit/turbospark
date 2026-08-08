//! THROWAWAY probe, the Qwen sibling of `gguf_norm_convention_probe.rs`:
//! does the Q4_K_M GGUF install's resident core agree with the MLX one?
//!
//! Written because the real Qwen GGUF install decodes without erroring and
//! produces incoherent text, which is the signature of a resident tensor
//! that arrived with the right shape and the wrong values. The Gemma probe
//! answered the same question for that family (bit-identical, which killed
//! the norm "+1" hypothesis); the repack ALSO reported 131 Qwen tensors
//! losing bits on the way to BF16, so this file cannot assume the same.
//!
//!   MREFRUST_QWEN36_INSTALL_DIR=~/models/qwen36.gturbo \
//!   MREFRUST_QWEN36_GGUF_INSTALL_DIR=~/models/qwen36-gguf.gturbo \
//!     cargo test -p mrefrust-repack --test gguf_qwen_core_probe --release -- --ignored --nocapture

use std::path::Path;

fn bf16_tensor(dir: &Path, name: &str) -> Option<Vec<f32>> {
    let index =
        model_io::load_resident_index(&dir.join("model_weights.bin")).expect("resident index");
    let e = index.entries.get(name)?;
    if e.dtype != 1 {
        println!("  (skipped {name} in {}: dtype {})", dir.display(), e.dtype);
        return None;
    }
    let bytes = std::fs::read(dir.join("model_weights.bin")).expect("weights");
    let start = e.file_offset as usize;
    Some(
        bytes[start..start + e.size_bytes as usize]
            .chunks_exact(2)
            .map(|c| compute::bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
    )
}

#[test]
#[ignore = "needs both Qwen installs"]
fn the_resident_core_agrees_between_the_two_qwen_installs() {
    let mlx = std::path::PathBuf::from(std::env::var_os("MREFRUST_QWEN36_INSTALL_DIR").unwrap());
    let gguf =
        std::path::PathBuf::from(std::env::var_os("MREFRUST_QWEN36_GGUF_INSTALL_DIR").unwrap());

    for name in [
        "language_model.model.layers.0.input_layernorm.weight",
        "language_model.model.layers.0.post_attention_layernorm.weight",
        "language_model.model.norm.weight",
        // The gated-DeltaNet parameters, which are where a mismapped or
        // mis-transcoded tensor would hurt most: they drive a recurrence.
        "language_model.model.layers.0.linear_attn.A_log",
        "language_model.model.layers.0.linear_attn.dt_bias",
        "language_model.model.layers.0.linear_attn.norm.weight",
        "language_model.model.layers.0.linear_attn.conv1d.weight",
    ] {
        let (Some(a), Some(b)) = (bf16_tensor(&mlx, name), bf16_tensor(&gguf, name)) else {
            println!("{name}: absent from one install, skipped");
            continue;
        };
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
