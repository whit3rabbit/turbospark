//! A tiny `unlimited_ocr2` install through the real streamed walk, in the
//! checkpoint's own tensor names, so CI covers the flow with no checkpoint.
//!
//! The tower is left out: its tensors are large at any useful shape, and the
//! flow takes image rows from the caller. The tower's gates are the kernel
//! parity tests and the real-model stages.

use model_io::{ArchConfig, ModelFamily};

use crate::gemma4_checkpoint::{write_unlimited_ocr2_install_streamed, Gemma4Shards};
use crate::ranged_download::MemoryRangeSource;
use crate::safetensors_header::{parse_header, DEFAULT_MAX_HEADER_BYTES};
use crate::synthetic_tensors::{
    assemble_safetensors, bf16_vector, deterministic_row, expert_int4_triple, int4_triple, u16_le,
    Tensor,
};
use crate::unlimited_ocr2_config::canonicalize_unlimited_ocr2_header;

const HIDDEN: usize = 128;
const HEADS: usize = 4;
const HEAD_DIM: usize = 32;
/// 64 so every down projection's input is a whole affine group, and the
/// shared expert's fused width is the dense lead's on no account.
const MOE_INTER: usize = 64;
const SHARED: usize = 2;
const DENSE_INTER: usize = 192;
const EXPERTS: usize = 8;
const TOP_K: i64 = 2;
const LAYERS: usize = 3;
/// In the tiny vocabulary, where a test can feed them.
pub const SYNTHETIC_UOCR2_IMAGE_TOKEN: i32 = 100;
pub const SYNTHETIC_UOCR2_PAGE_TOKEN: i32 = 101;

/// A tiny config. Every behavioural field is the real baseline's. The attention
/// scale is a binary fraction because the manifest round trip compares floats.
pub fn tiny_unlimited_ocr2_arch(vocab_size: i64) -> ArchConfig {
    let mut arch = model_io::unlimited_ocr2_12l();
    arch.hidden_size = HIDDEN as i64;
    arch.intermediate_size = (MOE_INTER * SHARED) as i64;
    arch.moe_intermediate_size = MOE_INTER as i64;
    arch.num_heads = HEADS as i64;
    arch.num_kv_heads = HEADS as i64;
    arch.num_full_kv_heads = HEADS as i64;
    arch.head_dim = HEAD_DIM as i64;
    arch.full_head_dim = HEAD_DIM as i64;
    arch.vocab_size = vocab_size;
    arch.num_layers = LAYERS as i64;
    arch.num_experts = EXPERTS as i64;
    arch.top_k_experts = TOP_K;
    arch.full_attention_layer_mask = vec![1u8; LAYERS];
    arch.attention_scale = 0.125;
    arch.dense_lead_intermediate_size = DENSE_INTER as i64;
    arch
}

fn bf16_matrix(name: &str, rows: usize, cols: usize, seed: u64) -> Tensor {
    let values: Vec<u16> = deterministic_row(seed, rows * cols)
        .iter()
        .map(|v| compute::f32_to_bf16(*v))
        .collect();
    Tensor {
        name: name.to_string(),
        dtype: "BF16",
        shape: vec![rows as u64, cols as u64],
        bytes: u16_le(&values),
    }
}

fn sparse_config() -> serde_json::Value {
    serde_json::json!({
        "training_stage": "dash",
        "use_sparse_attention": true,
        "dash_stage0": "head_cls",
        "dash_image_only": true,
        "dash_sigma": 1.0,
        "dash_topk_blocks": 12,
        "dash_scale_per_layer": vec![128.0; LAYERS],
        "dash_floor_per_layer": vec![0.0001; LAYERS],
        "image_token_id": SYNTHETIC_UOCR2_IMAGE_TOKEN,
        "page_token_id": SYNTHETIC_UOCR2_PAGE_TOKEN,
        "per_page_image_tokens": 273,
        "sparse_attention_freq": vec![1; LAYERS],
        "sparse_block_size": 32,
        "sparse_index_dim": HEAD_DIM,
        "sparse_local_block": 4,
        "sparse_main_rope_mode": "page2d",
        "sparse_main_rope_tok_freq_mode": "resample",
        "sparse_rope_page_mod": 2,
        "sparse_rope_page_pairs": 1,
        "sparse_rope_page_theta": 256.0,
        "sparse_rope_tok_theta": 10000.0,
    })
}

/// Write the install into `dir` and return its arch.
pub fn build_synthetic_unlimited_ocr2_install(
    dir: &std::path::Path,
    vocab_size: i64,
) -> Result<ArchConfig, Box<dyn std::error::Error>> {
    let arch = tiny_unlimited_ocr2_arch(vocab_size);
    debug_assert_eq!(arch.family, ModelFamily::UnlimitedOcr2);
    let vocab = vocab_size as usize;
    let mut ts: Vec<Tensor> = Vec::new();
    ts.extend(int4_triple(
        "language_model.model.embed_tokens.weight",
        vocab,
        HIDDEN,
        1,
    ));
    ts.extend(int4_triple(
        "language_model.lm_head.weight",
        vocab,
        HIDDEN,
        2,
    ));
    ts.push(bf16_vector(
        "language_model.model.norm.weight",
        HIDDEN,
        1.0,
        3,
    ));
    for l in 0..LAYERS {
        let p = format!("language_model.model.layers.{l}");
        let seed = 1000 * (l as u64 + 1);
        ts.push(bf16_vector(
            &format!("{p}.input_layernorm.weight"),
            HIDDEN,
            1.0,
            seed + 1,
        ));
        ts.push(bf16_vector(
            &format!("{p}.post_attention_layernorm.weight"),
            HIDDEN,
            1.0,
            seed + 2,
        ));
        for (i, proj) in ["q_proj", "k_proj", "v_proj", "o_proj"].iter().enumerate() {
            ts.extend(int4_triple(
                &format!("{p}.self_attn.{proj}.weight"),
                HIDDEN,
                HIDDEN,
                seed + 10 + i as u64,
            ));
        }
        let router = format!("{p}.self_attn.dash_router");
        ts.push(bf16_matrix(
            &format!("{router}.q_proj.weight"),
            HEAD_DIM,
            HIDDEN,
            seed + 20,
        ));
        ts.push(bf16_matrix(
            &format!("{router}.k_proj.weight"),
            HEAD_DIM,
            HIDDEN,
            seed + 21,
        ));
        ts.push(bf16_vector(
            &format!("{router}.q_norm.weight"),
            HEAD_DIM,
            0.0,
            seed + 22,
        ));
        ts.push(bf16_vector(
            &format!("{router}.k_norm.weight"),
            HEAD_DIM,
            0.0,
            seed + 23,
        ));
        ts.push(bf16_vector(
            &format!("{router}.head_cls"),
            HEAD_DIM,
            0.0,
            seed + 24,
        ));
        if l == 0 {
            for (i, (name, rows, cols)) in [
                ("gate_proj", DENSE_INTER, HIDDEN),
                ("up_proj", DENSE_INTER, HIDDEN),
                ("down_proj", HIDDEN, DENSE_INTER),
            ]
            .into_iter()
            .enumerate()
            {
                ts.extend(int4_triple(
                    &format!("{p}.mlp.{name}.weight"),
                    rows,
                    cols,
                    seed + 30 + i as u64,
                ));
            }
            continue;
        }
        ts.push(bf16_matrix(
            &format!("{p}.mlp.gate.weight"),
            EXPERTS,
            HIDDEN,
            seed + 40,
        ));
        let fused = MOE_INTER * SHARED;
        for (i, (name, rows, cols)) in [
            ("gate_proj", fused, HIDDEN),
            ("up_proj", fused, HIDDEN),
            ("down_proj", HIDDEN, fused),
        ]
        .into_iter()
        .enumerate()
        {
            ts.extend(int4_triple(
                &format!("{p}.mlp.shared_experts.{name}.weight"),
                rows,
                cols,
                seed + 50 + i as u64,
            ));
        }
        for (i, (name, rows, cols)) in [
            ("gate_proj", MOE_INTER, HIDDEN),
            ("up_proj", MOE_INTER, HIDDEN),
            ("down_proj", HIDDEN, MOE_INTER),
        ]
        .into_iter()
        .enumerate()
        {
            ts.extend(expert_int4_triple(
                &format!("{p}.mlp.switch_mlp.{name}.weight"),
                EXPERTS,
                rows,
                cols,
                seed + 60 + i as u64,
            ));
        }
    }

    let bytes = assemble_safetensors(&ts);
    let source = MemoryRangeSource::new(&bytes);
    let mut header = parse_header(&bytes, DEFAULT_MAX_HEADER_BYTES)?;
    canonicalize_unlimited_ocr2_header(&mut header)?;
    let shards = Gemma4Shards::single(&header, &source);
    let quant = crate::gemma4_checkpoint::parse_gemma4_quantization(
        r#"{"quantization": {"bits": 4, "group_size": 64, "mode": "affine"}}"#,
    )?;
    write_unlimited_ocr2_install_streamed(
        dir,
        &arch,
        "synthetic/unlimited-ocr2",
        std::slice::from_ref(&header),
        &shards,
        &quant,
        &sparse_config(),
        |_| {},
    )?;
    Ok(arch)
}
