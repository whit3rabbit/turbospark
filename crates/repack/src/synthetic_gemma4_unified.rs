//! A tiny `gemma4_unified` checkpoint: the dense Gemma 4 trunk fixture beside
//! the encoder-free image embedder and the audio projector, under the names
//! and quantization layout the MLX conversions use
//! (`docs/GEMMA4_UNIFIED_VISION_PHASE0.md`).
//!
//! The embedder's width (128) differs from the trunk's (64) so a swapped
//! projector shape cannot pass. The two vision matrices are INT4 over an INT8
//! trunk, so the per-tensor quantization path is exercised.

use model_io::ArchConfig;

use crate::gemma4_checkpoint::{
    drop_gemma4_unified_audio, parse_gemma4_config, parse_gemma4_quantization,
    parse_gemma4_unified_vision_config, write_gemma4_install,
};
use crate::ranged_download::MemoryRangeSource;
use crate::safetensors_header::parse_header;
use crate::synthetic_llama::SyntheticTowerTensor;
use crate::synthetic_real::dense_trunk_tensors;
use crate::synthetic_tensors::{
    assemble_safetensors, deterministic_row, int4_triple, u16_le, Tensor,
};

const TRUNK_HIDDEN: usize = 64;
const EMBED: usize = 128;
const PATCH: usize = 48 * 48 * 3;
const POSITIONS: usize = 8;

/// `(config.json, model.safetensors, the vision tensors as the runtime must
/// read them)`. The third lets a test run the forward pass independently:
/// matrices are dequantized and BF16 values rounded.
pub fn synthetic_gemma4_unified_checkpoint(
    vocab_size: i64,
    num_layers: i64,
    sliding_window: i64,
) -> (String, Vec<u8>, Vec<SyntheticTowerTensor>) {
    let layer_types: Vec<&str> = (0..num_layers)
        .map(|l| {
            if l % 6 == 5 {
                "full_attention"
            } else {
                "sliding_attention"
            }
        })
        .collect();
    let four_bit = serde_json::json!({"bits": 4, "group_size": 64});
    let config = serde_json::json!({
        "model_type": "gemma4_unified",
        "architectures": ["Gemma4UnifiedForConditionalGeneration"],
        // The image placeholder is 0 like the other synthetic trunk tests.
        "image_token_id": 0, "boi_token_id": 1, "eoi_token_id": 2, "video_token_id": 3,
        "quantization": {
            "bits": 8, "group_size": 64, "mode": "affine",
            "vision_embedder.patch_dense": four_bit,
            "embed_vision.embedding_projection": four_bit,
            "embed_audio.embedding_projection": four_bit,
        },
        "text_config": {
            "model_type": "gemma4_unified_text",
            "hidden_size": TRUNK_HIDDEN, "intermediate_size": 64,
            "num_attention_heads": 2, "num_key_value_heads": 2,
            "num_global_key_value_heads": 2, "head_dim": 32, "global_head_dim": 32,
            "vocab_size": vocab_size, "num_hidden_layers": num_layers,
            "sliding_window": sliding_window, "final_logit_softcapping": 30.0,
            "layer_types": layer_types, "tie_word_embeddings": true,
            "attention_k_eq_v": true, "enable_moe_block": false, "num_experts": null,
            "hidden_activation": "gelu_pytorch_tanh",
            "use_bidirectional_attention": "vision",
        },
        "vision_config": {
            "model_type": "gemma4_unified_vision", "mm_embed_dim": EMBED,
            "mm_posemb_size": POSITIONS, "model_patch_size": 48, "num_soft_tokens": 280,
            "output_proj_dims": EMBED, "patch_size": 16, "pooling_kernel_size": 3,
            "rms_norm_eps": 1e-6,
        },
        "audio_config": {"model_type": "gemma4_unified_audio"},
    })
    .to_string();

    let mut tensors = dense_trunk_tensors(vocab_size as usize, num_layers);
    let mut tower = Vec::new();
    // (name, shape, center, amplitude)
    for (i, (name, shape, center, amplitude)) in [
        ("patch_ln1.weight", vec![PATCH], 1.0f32, 0.2f32),
        ("patch_ln1.bias", vec![PATCH], 0.0, 0.1),
        ("patch_dense.bias", vec![EMBED], 0.0, 0.2),
        ("patch_ln2.weight", vec![EMBED], 1.0, 0.2),
        ("patch_ln2.bias", vec![EMBED], 0.0, 0.2),
        // Large against the normalized rows, so a position error shows.
        ("pos_embedding", vec![POSITIONS, 2, EMBED], 0.0, 1.0),
        ("pos_norm.weight", vec![EMBED], 1.0, 0.2),
        ("pos_norm.bias", vec![EMBED], 0.0, 0.2),
    ]
    .into_iter()
    .enumerate()
    {
        let name = format!("vision_embedder.{name}");
        let n: usize = shape.iter().product();
        let bits: Vec<u16> = deterministic_row(9_000 + i as u64, n)
            .iter()
            .map(|&j| compute::f32_to_bf16(center + j * amplitude))
            .collect();
        tower.push(SyntheticTowerTensor {
            name: name.clone(),
            shape: shape.clone(),
            values: bits.iter().map(|&b| compute::bf16_to_f32(b)).collect(),
        });
        tensors.push(Tensor {
            name,
            dtype: "BF16",
            shape: shape.iter().map(|&d| d as u64).collect(),
            bytes: u16_le(&bits),
        });
    }
    for (name, rows, cols, seed) in [
        ("vision_embedder.patch_dense.weight", EMBED, PATCH, 91u64),
        (
            "embed_vision.embedding_projection.weight",
            TRUNK_HIDDEN,
            EMBED,
            92,
        ),
    ] {
        let triple = int4_triple(name, rows, cols, seed);
        tower.push(SyntheticTowerTensor {
            name: name.to_string(),
            shape: vec![rows, cols],
            values: dequantize(&triple, rows, cols),
        });
        tensors.extend(triple);
    }
    tensors.extend(int4_triple(
        "embed_audio.embedding_projection.weight",
        TRUNK_HIDDEN,
        64,
        93,
    ));
    (config, assemble_safetensors(&tensors), tower)
}

/// `[weight, scales, biases]` to row-major f32.
fn dequantize(triple: &[Tensor], rows: usize, cols: usize) -> Vec<f32> {
    let groups = cols / 64;
    let plane = |t: &Tensor, i: usize| u16::from_le_bytes([t.bytes[2 * i], t.bytes[2 * i + 1]]);
    let mut out = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        let row = compute::quant::Int4AffineRow {
            packed: triple[0].bytes[r * cols / 2..(r + 1) * cols / 2].to_vec(),
            scales: (0..groups)
                .map(|g| plane(&triple[1], r * groups + g))
                .collect(),
            biases: (0..groups)
                .map(|g| plane(&triple[2], r * groups + g))
                .collect(),
        };
        out.extend(compute::quant::dequantize_int4_affine(&row, cols));
    }
    out
}

/// The checkpoint above through the real config parsers and repack walk.
pub fn build_synthetic_gemma4_unified_install(
    dir: &std::path::Path,
    vocab_size: i64,
    num_layers: i64,
    sliding_window: i64,
    model_id: &str,
) -> Result<(ArchConfig, Vec<SyntheticTowerTensor>), Box<dyn std::error::Error>> {
    let (config, blob, tower) =
        synthetic_gemma4_unified_checkpoint(vocab_size, num_layers, sliding_window);
    let mut arch = parse_gemma4_config(&config)?;
    arch.vision = parse_gemma4_unified_vision_config(&config)?;
    let quant = parse_gemma4_quantization(&config)?;
    let source = MemoryRangeSource::new(&blob);
    let mut header = parse_header(&blob, crate::safetensors_header::DEFAULT_MAX_HEADER_BYTES)?;
    drop_gemma4_unified_audio(&mut header);
    write_gemma4_install(dir, &arch, model_id, &header, &source, &quant)?;
    Ok((arch, tower))
}
