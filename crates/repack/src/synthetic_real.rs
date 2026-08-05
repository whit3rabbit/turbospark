//! Builds a tiny Gemma 4 install with the REAL checkpoint pipeline: an
//! in-memory safetensors blob using the `mlx-community` tensor naming
//! (`language_model.` prefix, `.experts.switch_glu.` routed experts, INT8
//! router and shared MLP, BF16 norms/scales/scalars), pushed through
//! [`crate::write_gemma4_install`] — the exact path a downloaded
//! checkpoint takes. This is what `crates/runtime`'s real-Gemma-4
//! learned-weight decode flow is exercised against, since no trained
//! checkpoint exists in this environment. Weights are deterministic (the
//! same xorshift scheme as `synthetic_model.rs`) but not trained.

use compute::{f32_to_bf16, quantize_int4_affine, quantize_int8_affine};
use model_io::ArchConfig;

use crate::gemma4_checkpoint::{write_gemma4_install, Gemma4Quant};
use crate::ranged_download::MemoryRangeSource;
use crate::safetensors_header::parse_header;
use crate::synthetic_model::tiny_gemma4_arch;

const HIDDEN: usize = 64;
const NUM_HEADS: usize = 2;
const HEAD_DIM: usize = 32;
const INTER: usize = 64;

struct Tensor {
    name: String,
    dtype: &'static str,
    shape: Vec<u64>,
    bytes: Vec<u8>,
}

fn assemble_safetensors(tensors: &[Tensor]) -> Vec<u8> {
    let mut header = serde_json::Map::new();
    let mut cursor = 0u64;
    for t in tensors {
        let end = cursor + t.bytes.len() as u64;
        header.insert(
            t.name.clone(),
            serde_json::json!({
                "dtype": t.dtype,
                "shape": t.shape,
                "data_offsets": [cursor, end],
            }),
        );
        cursor = end;
    }
    let header_json = serde_json::Value::Object(header).to_string().into_bytes();
    let mut out = Vec::new();
    out.extend_from_slice(&(header_json.len() as u64).to_le_bytes());
    out.extend_from_slice(&header_json);
    for t in tensors {
        out.extend_from_slice(&t.bytes);
    }
    out
}

fn deterministic_row(seed: u64, n: usize) -> Vec<f32> {
    let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(0x9E37_79B9);
    (0..n)
        .map(|i| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state = state.wrapping_add(i as u64);
            ((state % 2000) as f32 / 1000.0) - 1.0
        })
        .collect()
}

fn u16_le(values: &[u16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 2);
    for v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// A rank-2 INT4-quantized tensor triple (weight + scales + biases) in the
/// MLX safetensors shape: `U32 [rows, cols/8]` plus `BF16 [rows, cols/64]`
/// companions. The packed bytes are this port's own nibble layout (the two
/// layouts are LE-byte identical).
fn int4_triple(name: &str, rows: usize, cols: usize, seed: u64) -> Vec<Tensor> {
    let base = name.strip_suffix(".weight").unwrap();
    let mut packed = Vec::new();
    let mut scales = Vec::new();
    let mut biases = Vec::new();
    for r in 0..rows {
        let q = quantize_int4_affine(&deterministic_row(
            seed.wrapping_add(r as u64 * 97 + 1),
            cols,
        ));
        packed.extend_from_slice(&q.packed);
        scales.extend_from_slice(&q.scales);
        biases.extend_from_slice(&q.biases);
    }
    vec![
        Tensor {
            name: name.to_string(),
            dtype: "U32",
            shape: vec![rows as u64, (cols / 8) as u64],
            bytes: packed,
        },
        Tensor {
            name: format!("{base}.scales"),
            dtype: "BF16",
            shape: vec![rows as u64, (cols / 64) as u64],
            bytes: u16_le(&scales),
        },
        Tensor {
            name: format!("{base}.biases"),
            dtype: "BF16",
            shape: vec![rows as u64, (cols / 64) as u64],
            bytes: u16_le(&biases),
        },
    ]
}

/// INT8 sibling of [`int4_triple`]: `U32 [rows, cols/4]` packed bytes.
fn int8_triple(name: &str, rows: usize, cols: usize, seed: u64) -> Vec<Tensor> {
    let base = name.strip_suffix(".weight").unwrap();
    let mut packed = Vec::new();
    let mut scales = Vec::new();
    let mut biases = Vec::new();
    for r in 0..rows {
        let q = quantize_int8_affine(&deterministic_row(
            seed.wrapping_add(r as u64 * 97 + 1),
            cols,
        ));
        packed.extend_from_slice(&q.packed);
        scales.extend_from_slice(&q.scales);
        biases.extend_from_slice(&q.biases);
    }
    vec![
        Tensor {
            name: name.to_string(),
            dtype: "U32",
            shape: vec![rows as u64, (cols / 4) as u64],
            bytes: packed,
        },
        Tensor {
            name: format!("{base}.scales"),
            dtype: "BF16",
            shape: vec![rows as u64, (cols / 64) as u64],
            bytes: u16_le(&scales),
        },
        Tensor {
            name: format!("{base}.biases"),
            dtype: "BF16",
            shape: vec![rows as u64, (cols / 64) as u64],
            bytes: u16_le(&biases),
        },
    ]
}

/// Rank-3 expert-major INT4 triple: `U32 [experts, rows, cols/8]`.
fn expert_int4_triple(
    name: &str,
    experts: usize,
    rows: usize,
    cols: usize,
    seed: u64,
) -> Vec<Tensor> {
    let base = name.strip_suffix(".weight").unwrap();
    let mut packed = Vec::new();
    let mut scales = Vec::new();
    let mut biases = Vec::new();
    for e in 0..experts {
        for r in 0..rows {
            let row_seed = seed.wrapping_add(e as u64 * 10_000 + r as u64 * 97 + 1);
            let q = quantize_int4_affine(&deterministic_row(row_seed, cols));
            packed.extend_from_slice(&q.packed);
            scales.extend_from_slice(&q.scales);
            biases.extend_from_slice(&q.biases);
        }
    }
    vec![
        Tensor {
            name: name.to_string(),
            dtype: "U32",
            shape: vec![experts as u64, rows as u64, (cols / 8) as u64],
            bytes: packed,
        },
        Tensor {
            name: format!("{base}.scales"),
            dtype: "BF16",
            shape: vec![experts as u64, rows as u64, (cols / 64) as u64],
            bytes: u16_le(&scales),
        },
        Tensor {
            name: format!("{base}.biases"),
            dtype: "BF16",
            shape: vec![experts as u64, rows as u64, (cols / 64) as u64],
            bytes: u16_le(&biases),
        },
    ]
}

/// A BF16 vector near `center` with small deterministic jitter.
fn bf16_vector(name: &str, n: usize, center: f32, seed: u64) -> Tensor {
    let jitter = deterministic_row(seed, n);
    let bits: Vec<u16> = jitter
        .iter()
        .map(|&j| f32_to_bf16(center + j * 0.05))
        .collect();
    Tensor {
        name: name.to_string(),
        dtype: "BF16",
        shape: vec![n as u64],
        bytes: u16_le(&bits),
    }
}

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
            ts.extend(int8_triple(
                &format!("{p}.mlp.{role}.weight"),
                rows,
                cols,
                seed + 30 + i as u64,
            ));
            overrides.insert(format!("{p}.mlp.{role}"), 8u32);
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
