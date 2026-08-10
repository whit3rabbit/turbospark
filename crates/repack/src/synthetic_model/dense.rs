//! Dense synthetic Gemma 4 install builders.

use model_io::ArchConfig;

use super::arch::{
    down_proj_name, embed_lm_head_name, gate_proj_name, k_proj_name, o_proj_name, q_proj_name,
    quantized_tensor, tiny_gemma4_arch, up_proj_name, FULL_HEAD_DIM, HIDDEN_SIZE,
    INTERMEDIATE_SIZE, NUM_HEADS,
};
use crate::gturbo_writer::{write_gturbo_install_with_resident_index, WriterError};
use crate::resident_writer::build_resident_weights_bin;

/// Writes a full tiny-Gemma4 `.gturbo` install to `dir` and returns the
/// `ArchConfig` it was built against (the caller needs this to open the
/// install back up, since `turbospark_model_io::load_manifest` validates
/// against a caller-supplied expected architecture rather than inferring
/// one for non-canonical shapes).
pub fn build_synthetic_gemma4_install(
    dir: &std::path::Path,
    vocab_size: i64,
    num_layers: i64,
    model_id: &str,
) -> Result<ArchConfig, WriterError> {
    let arch = tiny_gemma4_arch(vocab_size, num_layers);
    let hidden = HIDDEN_SIZE as usize;
    let inter = INTERMEDIATE_SIZE as usize;
    let qk_dim = (NUM_HEADS * FULL_HEAD_DIM) as usize;
    let vocab = vocab_size as usize;

    let mut specs = Vec::with_capacity(1 + num_layers as usize * 6);
    specs.push(quantized_tensor(&embed_lm_head_name(), vocab, hidden, 1));
    for l in 0..num_layers {
        let base = 1000u64 * (l as u64 + 1);
        specs.push(quantized_tensor(&q_proj_name(l), qk_dim, hidden, base + 1));
        specs.push(quantized_tensor(&k_proj_name(l), qk_dim, hidden, base + 2));
        specs.push(quantized_tensor(&o_proj_name(l), hidden, qk_dim, base + 3));
        specs.push(quantized_tensor(
            &gate_proj_name(l),
            inter,
            hidden,
            base + 4,
        ));
        specs.push(quantized_tensor(&up_proj_name(l), inter, hidden, base + 5));
        specs.push(quantized_tensor(
            &down_proj_name(l),
            hidden,
            inter,
            base + 6,
        ));
    }

    let resident_bytes = build_resident_weights_bin(&specs);
    write_gturbo_install_with_resident_index(dir, &arch, model_id, &resident_bytes)?;
    Ok(arch)
}

/// A dense tiny-Gemma4 install whose layers ALTERNATE sliding-window
/// (mask 0) and full attention (mask 1), with `sliding_window` positions
/// of window — the mixed attention-kind shape real Gemma 4 has (25 SWA +
/// 5 full layers), at toy size. Weights are identical to
/// [`build_synthetic_gemma4_install`] (same seeds).
pub fn build_synthetic_gemma4_swa_install(
    dir: &std::path::Path,
    vocab_size: i64,
    num_layers: i64,
    sliding_window: i64,
    model_id: &str,
) -> Result<ArchConfig, WriterError> {
    let mut arch = tiny_gemma4_arch(vocab_size, num_layers);
    arch.sliding_window = sliding_window;
    arch.full_attention_layer_mask = (0..num_layers).map(|l| (l % 2 == 1) as u8).collect();

    let hidden = HIDDEN_SIZE as usize;
    let inter = INTERMEDIATE_SIZE as usize;
    let qk_dim = (NUM_HEADS * FULL_HEAD_DIM) as usize;
    let vocab = vocab_size as usize;

    let mut specs = Vec::with_capacity(1 + num_layers as usize * 6);
    specs.push(quantized_tensor(&embed_lm_head_name(), vocab, hidden, 1));
    for l in 0..num_layers {
        let base = 1000u64 * (l as u64 + 1);
        specs.push(quantized_tensor(&q_proj_name(l), qk_dim, hidden, base + 1));
        specs.push(quantized_tensor(&k_proj_name(l), qk_dim, hidden, base + 2));
        specs.push(quantized_tensor(&o_proj_name(l), hidden, qk_dim, base + 3));
        specs.push(quantized_tensor(
            &gate_proj_name(l),
            inter,
            hidden,
            base + 4,
        ));
        specs.push(quantized_tensor(&up_proj_name(l), inter, hidden, base + 5));
        specs.push(quantized_tensor(
            &down_proj_name(l),
            hidden,
            inter,
            base + 6,
        ));
    }

    let resident_bytes = build_resident_weights_bin(&specs);
    write_gturbo_install_with_resident_index(dir, &arch, model_id, &resident_bytes)?;
    Ok(arch)
}
