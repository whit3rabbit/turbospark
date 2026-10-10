//! Builds a tiny Gemma 4 install with the REAL checkpoint pipeline: an
//! in-memory safetensors blob using the `mlx-community` tensor naming
//! (`language_model.` prefix, `.experts.switch_glu.` routed experts, INT8
//! router and shared MLP, BF16 norms/scales/scalars), pushed through
//! [`crate::write_gemma4_install`] — the exact path a downloaded
//! checkpoint takes. This is what `crates/runtime`'s real-Gemma-4
//! learned-weight decode flow is exercised against, since no trained
//! checkpoint exists in this environment. Weights are deterministic (the
//! same xorshift scheme as `synthetic_model.rs`) but not trained.

use model_io::ArchConfig;

use crate::gemma4_checkpoint::{write_gemma4_install, Gemma4Quant};
use crate::ranged_download::MemoryRangeSource;
use crate::safetensors_header::parse_header;
use crate::synthetic_model::tiny_gemma4_arch;
pub(crate) use crate::synthetic_tensors::{
    assemble_safetensors, bf16_vector, deterministic_row, expert_int4_triple, int4_triple,
    int8_triple, u16_le, Tensor,
};

const HIDDEN: usize = 64;
const NUM_HEADS: usize = 2;
const HEAD_DIM: usize = 32;
const INTER: usize = 64;

/// Writes a tiny "real" Gemma 4 `.gturbo` install (verbatim mlx-community
/// tensor naming, INT8 router + shared MLP, BF16 learned norms, streamed
/// routed experts) through the real checkpoint repack pipeline, and
/// returns the `ArchConfig` needed to open it. Layers alternate SWA
/// (mask 0) and full attention (mask 1).
pub fn build_synthetic_gemma4_real_install(
    dir: &std::path::Path,
    vocab_size: i64,
    num_layers: i64,
    num_experts: i64,
    top_k: i64,
    sliding_window: i64,
    model_id: &str,
) -> Result<ArchConfig, Box<dyn std::error::Error>> {
    build_synthetic_gemma4_real_install_at_shared_bits(
        dir,
        vocab_size,
        num_layers,
        num_experts,
        top_k,
        sliding_window,
        model_id,
        8,
    )
}

/// The same fixture with the SHARED EXPERT's three projections written at
/// `shared_bits` (8 or 4) instead of the default 8.
///
/// **THE DEFAULT IS 8 AND THE REAL CHECKPOINT IS 4**, which is not a bug in
/// either: `mlx-community/gemma-4-26b-a4b-it-4bit` declares
/// `sharedExpert.weightBits: 4` beside its 8-bit router, while this fixture
/// has carried an INT8 shared MLP since it was written and is the only
/// coverage in the repo of the INT8 resident GEMV inside a whole Gemma
/// forward pass. Flipping the default would silently retire that.
///
/// The 4-bit form exists because `encode_gemm_any` is INT4-affine only, so
/// the chunk driver's batched shared expert (`TURBOSPARK_BATCHED_GEMV`) is
/// unreachable on the 8-bit fixture -- it is refused BY NAME rather than
/// looped, correctly, which means the default fixture cannot gate the very
/// dispatches that seam moves on the real install. A fixture that cannot
/// see the property is worse than no fixture.
#[allow(clippy::too_many_arguments)]
pub fn build_synthetic_gemma4_real_install_at_shared_bits(
    dir: &std::path::Path,
    vocab_size: i64,
    num_layers: i64,
    num_experts: i64,
    top_k: i64,
    sliding_window: i64,
    model_id: &str,
    shared_bits: u32,
) -> Result<ArchConfig, Box<dyn std::error::Error>> {
    assert!(
        shared_bits == 8 || shared_bits == 4,
        "shared_bits must be 8 or 4; {shared_bits} has no resident GEMV pair here"
    );
    let mut arch = tiny_gemma4_arch(vocab_size, num_layers);
    arch.num_experts = num_experts;
    arch.top_k_experts = top_k;
    arch.moe_intermediate_size = INTER as i64;
    arch.sliding_window = sliding_window;
    arch.full_attention_layer_mask = (0..num_layers).map(|l| (l % 2 == 1) as u8).collect();

    let experts = num_experts as usize;
    let vocab = vocab_size as usize;
    let qk_dim = NUM_HEADS * HEAD_DIM;

    let mut ts: Vec<Tensor> = Vec::new();
    ts.extend(int4_triple(
        "language_model.model.embed_tokens.weight",
        vocab,
        HIDDEN,
        1,
    ));
    let mut overrides = std::collections::HashMap::new();
    for l in 0..num_layers as usize {
        let p = format!("language_model.model.layers.{l}");
        let seed = 1000 * (l as u64 + 1);
        ts.extend(int4_triple(
            &format!("{p}.self_attn.q_proj.weight"),
            qk_dim,
            HIDDEN,
            seed + 1,
        ));
        ts.extend(int4_triple(
            &format!("{p}.self_attn.k_proj.weight"),
            qk_dim,
            HIDDEN,
            seed + 2,
        ));
        // SWA layers carry a real v_proj; full layers reuse k_proj under
        // the K=V quirk (the tensor still ships in real checkpoints, and
        // shipping it keeps the fixture uniform).
        ts.extend(int4_triple(
            &format!("{p}.self_attn.v_proj.weight"),
            qk_dim,
            HIDDEN,
            seed + 7,
        ));
        ts.extend(int4_triple(
            &format!("{p}.self_attn.o_proj.weight"),
            HIDDEN,
            qk_dim,
            seed + 3,
        ));
        ts.push(bf16_vector(
            &format!("{p}.self_attn.q_norm.weight"),
            HEAD_DIM,
            1.0,
            seed + 20,
        ));
        ts.push(bf16_vector(
            &format!("{p}.self_attn.k_norm.weight"),
            HEAD_DIM,
            1.0,
            seed + 21,
        ));
        ts.extend(int8_triple(
            &format!("{p}.router.proj.weight"),
            experts,
            HIDDEN,
            seed + 4,
        ));
        overrides.insert(format!("{p}.router.proj"), 8u32);
        ts.push(bf16_vector(
            &format!("{p}.router.scale"),
            HIDDEN,
            1.0,
            seed + 22,
        ));
        ts.push(bf16_vector(
            &format!("{p}.router.per_expert_scale"),
            experts,
            1.0,
            seed + 23,
        ));
        for (i, role) in ["gate_proj", "up_proj", "down_proj"].iter().enumerate() {
            let (rows, cols) = if *role == "down_proj" {
                (HIDDEN, INTER)
            } else {
                (INTER, HIDDEN)
            };
            let triple = if shared_bits == 8 {
                int8_triple
            } else {
                int4_triple
            };
            ts.extend(triple(
                &format!("{p}.mlp.{role}.weight"),
                rows,
                cols,
                seed + 30 + i as u64,
            ));
            overrides.insert(format!("{p}.mlp.{role}"), shared_bits);
        }
        for (i, norm) in [
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "pre_feedforward_layernorm_2",
            "post_feedforward_layernorm",
            "post_feedforward_layernorm_1",
            "post_feedforward_layernorm_2",
        ]
        .iter()
        .enumerate()
        {
            ts.push(bf16_vector(
                &format!("{p}.{norm}.weight"),
                HIDDEN,
                1.0,
                seed + 40 + i as u64,
            ));
        }
        ts.push(bf16_vector(&format!("{p}.layer_scalar"), 1, 1.0, seed + 50));
        for (i, role) in ["gate_proj", "up_proj", "down_proj"].iter().enumerate() {
            let (rows, cols) = if *role == "down_proj" {
                (HIDDEN, INTER)
            } else {
                (INTER, HIDDEN)
            };
            ts.extend(expert_int4_triple(
                &format!("{p}.experts.switch_glu.{role}.weight"),
                experts,
                rows,
                cols,
                seed + 60 + i as u64,
            ));
        }
    }
    ts.push(bf16_vector(
        "language_model.model.norm.weight",
        HIDDEN,
        1.0,
        7,
    ));

    let blob = assemble_safetensors(&ts);
    let source = MemoryRangeSource::new(&blob);
    let header = parse_header(&blob, crate::safetensors_header::DEFAULT_MAX_HEADER_BYTES)?;
    let quant = Gemma4Quant {
        default_bits: 4,
        group_size: 64,
        bits_overrides: overrides,
    };
    write_gemma4_install(dir, &arch, model_id, &header, &source, &quant)?;
    Ok(arch)
}

/// Writes a tiny DENSE Gemma 4 install through the real checkpoint pipeline:
/// the same `language_model.` naming as [`build_synthetic_gemma4_real_install`]
/// with the routed half absent entirely -- no `router.*`, no
/// `.experts.switch_glu.`, a plain `mlp.{gate,up,down}_proj` triple, and only
/// the four per-layer norms the dense trunk carries (the `_1`/`_2` variants
/// are the MoE split's). This is the fixture for the dense arm of the real
/// Gemma 4 decode flow.
///
/// EVERYTHING QUANTIZED IS 8-BIT (`default_bits: 8`, no overrides), read off
/// the first real dense carrier, `cowWhySo/humanizer-12b-mlx-8bit`: that
/// checkpoint is INT8 affine at group 64 everywhere, embedding included, so
/// this fixture is the repo's only whole-forward coverage of the INT8
/// resident GEMV plus the INT8 embedding lookup. `v_proj` ships on sliding
/// layers only, matching the real inventory -- a full layer takes V from its
/// K projection under `attention_k_eq_v`, and the real 12B carries no
/// `v_proj` tensor for them at all.
///
/// The mask keeps the real 5:1 schedule (`full_attention` at layer index
/// 5, 11, ...), so a fixture with `num_layers >= 6` exercises both layer
/// kinds; with fewer layers everything is sliding except a final full one.
pub fn build_synthetic_gemma4_dense_real_install(
    dir: &std::path::Path,
    vocab_size: i64,
    num_layers: i64,
    sliding_window: i64,
    model_id: &str,
) -> Result<ArchConfig, Box<dyn std::error::Error>> {
    let mut arch = tiny_gemma4_arch(vocab_size, num_layers);
    arch.num_experts = 0;
    arch.top_k_experts = 0;
    arch.moe_intermediate_size = 0;
    arch.sliding_window = sliding_window;
    arch.full_attention_layer_mask = (0..num_layers).map(|l| (l % 6 == 5) as u8).collect();

    let vocab = vocab_size as usize;
    let qk_dim = NUM_HEADS * HEAD_DIM;

    let mut ts: Vec<Tensor> = Vec::new();
    ts.extend(int8_triple(
        "language_model.model.embed_tokens.weight",
        vocab,
        HIDDEN,
        1,
    ));
    for l in 0..num_layers as usize {
        let p = format!("language_model.model.layers.{l}");
        let full = l % 6 == 5;
        let seed = 1000 * (l as u64 + 1);
        ts.extend(int8_triple(
            &format!("{p}.self_attn.q_proj.weight"),
            qk_dim,
            HIDDEN,
            seed + 1,
        ));
        ts.extend(int8_triple(
            &format!("{p}.self_attn.k_proj.weight"),
            qk_dim,
            HIDDEN,
            seed + 2,
        ));
        // A sliding layer carries a real v_proj; a full layer takes V from
        // its K projection (attention_k_eq_v) and ships no v tensor.
        if !full {
            ts.extend(int8_triple(
                &format!("{p}.self_attn.v_proj.weight"),
                qk_dim,
                HIDDEN,
                seed + 7,
            ));
        }
        ts.extend(int8_triple(
            &format!("{p}.self_attn.o_proj.weight"),
            HIDDEN,
            qk_dim,
            seed + 3,
        ));
        ts.push(bf16_vector(
            &format!("{p}.self_attn.q_norm.weight"),
            HEAD_DIM,
            1.0,
            seed + 20,
        ));
        ts.push(bf16_vector(
            &format!("{p}.self_attn.k_norm.weight"),
            HEAD_DIM,
            1.0,
            seed + 21,
        ));
        for (i, role) in ["gate_proj", "up_proj", "down_proj"].iter().enumerate() {
            let (rows, cols) = if *role == "down_proj" {
                (HIDDEN, INTER)
            } else {
                (INTER, HIDDEN)
            };
            ts.extend(int8_triple(
                &format!("{p}.mlp.{role}.weight"),
                rows,
                cols,
                seed + 30 + i as u64,
            ));
        }
        // The DENSE norm inventory: four per layer, no MoE split variants.
        for (i, norm) in [
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ]
        .iter()
        .enumerate()
        {
            ts.push(bf16_vector(
                &format!("{p}.{norm}.weight"),
                HIDDEN,
                1.0,
                seed + 40 + i as u64,
            ));
        }
        // The real dense checkpoint carries `layer_scalar` on every layer
        // even with the MoE block off; ship it so the fixture matches the
        // inventory the classifier and any reader will meet.
        ts.push(bf16_vector(&format!("{p}.layer_scalar"), 1, 1.0, seed + 50));
    }
    ts.push(bf16_vector(
        "language_model.model.norm.weight",
        HIDDEN,
        1.0,
        7,
    ));

    let blob = assemble_safetensors(&ts);
    let source = MemoryRangeSource::new(&blob);
    let header = parse_header(&blob, crate::safetensors_header::DEFAULT_MAX_HEADER_BYTES)?;
    let quant = Gemma4Quant {
        default_bits: 8,
        group_size: 64,
        bits_overrides: std::collections::HashMap::new(),
    };
    write_gemma4_install(dir, &arch, model_id, &header, &source, &quant)?;
    Ok(arch)
}
