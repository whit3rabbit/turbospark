//! Quantize-on-install oracle for the Qwen MoE, read off REAL published bytes
//! with ranged reads (a few MB):
//!
//! ```sh
//! cargo test -p turbospark-repack --test mlx_quantize_parity_moe_network --release -- --ignored --nocapture
//! ```
//!
//! `Qwen/Qwen3.6-35B-A3B` ships the routed experts FUSED
//! (`experts.gate_up_proj [E, 2I, H]`, `experts.down_proj [E, H, I]`, no
//! `.weight` suffix). `mlx-community/Qwen3.6-35B-A3B-4bit` is that checkpoint
//! after `mlx_lm.convert`, with `switch_mlp.{gate,up,down}_proj` rank 3. This
//! checks, byte for byte, that quantizing the HF rows with the port gives the
//! published planes, which pins three things the offline fixture can only
//! assume: gate is the FIRST half of the fused rows and up the second, the
//! router and shared-expert gate are 8-bit, and the shared-expert projections
//! are 4-bit.

use turbospark_repack::{fetch_safetensors_header, HttpRangeSource, RangeSource};

const HF: &str =
    "https://huggingface.co/Qwen/Qwen3.6-35B-A3B/resolve/345d467b117d1fd5eda9b879a5988de9ae5f47a7";
const MLX: &str = "https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-4bit/resolve/38740b847e4cb78f352aba30aa41c76e08e6eb46";
/// This conversion rounds exact ties to even (the MLX CPU kernel); the dense
/// pair in `mlx_quantize_parity_network.rs` rounds them away from zero (the
/// Metal kernel). Everything else, scales and biases included, agrees.
const ROUNDING: compute::quant_mlx::Rounding = compute::quant_mlx::Rounding::HalfEven;
const MLX_SHARD: &str = "model-00001-of-00004.safetensors";

fn u16s(b: &[u8]) -> Vec<u16> {
    b.chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

fn rows(s: &HttpRangeSource, name: &str, first: u64, row_bytes: u64) -> Vec<u8> {
    let h = fetch_safetensors_header(s).expect("header");
    let (a, b) = h
        .absolute_range(name)
        .unwrap_or_else(|| panic!("{name} missing"));
    let from = a + first * row_bytes;
    assert!(from + row_bytes <= b, "{name} row {first} out of range");
    s.read_range(from, from + row_bytes).expect("row read")
}

struct Case {
    what: &'static str,
    hf_shard: &'static str,
    hf_name: &'static str,
    /// HF source row (already offset into the fused matrix).
    hf_row: u64,
    cols: u64,
    bits: u32,
    mlx_base: &'static str,
    /// Row index in the MLX tensor viewed as `[rows..., packed]`.
    mlx_row: u64,
}

fn mismatches(c: &Case) -> usize {
    let hf = HttpRangeSource::new(format!("{HF}/{}", c.hf_shard));
    let mx = HttpRangeSource::new(format!("{MLX}/{MLX_SHARD}"));
    let raw = rows(&hf, c.hf_name, c.hf_row, c.cols * 2);
    let row: Vec<f32> = u16s(&raw).into_iter().map(compute::bf16_to_f32).collect();
    let (packed, scales, biases) = match c.bits {
        4 => {
            let q = compute::quant_mlx::quantize_int4_affine_mlx_with(&row, ROUNDING);
            (q.packed, q.scales, q.biases)
        }
        8 => {
            let q = compute::quant_mlx::quantize_int8_affine_mlx_with(&row, ROUNDING);
            (q.packed, q.scales, q.biases)
        }
        b => panic!("bits {b}"),
    };
    let g = c.cols / 64;
    let w = rows(
        &mx,
        &format!("{}.weight", c.mlx_base),
        c.mlx_row,
        c.cols * c.bits as u64 / 8,
    );
    let sc = u16s(&rows(
        &mx,
        &format!("{}.scales", c.mlx_base),
        c.mlx_row,
        g * 2,
    ));
    let bi = u16s(&rows(
        &mx,
        &format!("{}.biases", c.mlx_base),
        c.mlx_row,
        g * 2,
    ));
    let (mp, ms, mb) = (
        packed.iter().zip(&w).filter(|(a, b)| a != b).count(),
        scales.iter().zip(&sc).filter(|(a, b)| a != b).count(),
        biases.iter().zip(&bi).filter(|(a, b)| a != b).count(),
    );
    if mp + ms + mb > 0 {
        let first = packed.iter().zip(&w).position(|(a, b)| a != b);
        println!(
            "    packed {mp} scales {ms} biases {mb}; first packed diff {:?}; scale0 ours {:#06x} theirs {:#06x}",
            first.map(|i| (i, packed[i], w[i])),
            scales[0],
            sc[0]
        );
    }
    mp + ms + mb
}

#[test]
#[ignore = "ranged reads off Qwen/Qwen3.6-35B-A3B and its mlx-community 4-bit conversion"]
fn fused_bf16_experts_and_gates_quantize_to_the_published_planes() {
    let (inter, hidden) = (512u64, 2048u64);
    let mut cases = Vec::new();
    // Two experts, first / middle / last row of each half.
    for e in [0u64, 255] {
        for r in [0u64, 1, 511] {
            cases.push(Case {
                what: "gate half (first rows of the fused matrix)",
                hf_shard: "model-00001-of-00026.safetensors",
                hf_name: "model.language_model.layers.0.mlp.experts.gate_up_proj",
                hf_row: e * 2 * inter + r,
                cols: hidden,
                bits: 4,
                mlx_base: "language_model.model.layers.0.mlp.switch_mlp.gate_proj",
                mlx_row: e * inter + r,
            });
            cases.push(Case {
                what: "up half (second rows of the fused matrix)",
                hf_shard: "model-00001-of-00026.safetensors",
                hf_name: "model.language_model.layers.0.mlp.experts.gate_up_proj",
                hf_row: e * 2 * inter + inter + r,
                cols: hidden,
                bits: 4,
                mlx_base: "language_model.model.layers.0.mlp.switch_mlp.up_proj",
                mlx_row: e * inter + r,
            });
        }
        for r in [0u64, 7, 2047] {
            cases.push(Case {
                what: "down_proj",
                hf_shard: "model-00002-of-00026.safetensors",
                hf_name: "model.language_model.layers.0.mlp.experts.down_proj",
                hf_row: e * hidden + r,
                cols: inter,
                bits: 4,
                mlx_base: "language_model.model.layers.0.mlp.switch_mlp.down_proj",
                mlx_row: e * hidden + r,
            });
        }
    }
    for r in [0u64, 100, 255] {
        cases.push(Case {
            what: "router mlp.gate (8-bit)",
            hf_shard: "model-00002-of-00026.safetensors",
            hf_name: "model.language_model.layers.0.mlp.gate.weight",
            hf_row: r,
            cols: hidden,
            bits: 8,
            mlx_base: "language_model.model.layers.0.mlp.gate",
            mlx_row: r,
        });
    }
    cases.push(Case {
        what: "shared_expert_gate (8-bit)",
        hf_shard: "model-00002-of-00026.safetensors",
        hf_name: "model.language_model.layers.0.mlp.shared_expert_gate.weight",
        hf_row: 0,
        cols: hidden,
        bits: 8,
        mlx_base: "language_model.model.layers.0.mlp.shared_expert_gate",
        mlx_row: 0,
    });
    for r in [0u64, 511] {
        cases.push(Case {
            what: "shared_expert.gate_proj (default 4-bit)",
            hf_shard: "model-00002-of-00026.safetensors",
            hf_name: "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight",
            hf_row: r,
            cols: hidden,
            bits: 4,
            mlx_base: "language_model.model.layers.0.mlp.shared_expert.gate_proj",
            mlx_row: r,
        });
    }
    let mut total = 0usize;
    for c in &cases {
        let m = mismatches(c);
        println!(
            "{:46} hf row {:7} -> mlx row {:6}  mismatched units {m}",
            c.what, c.hf_row, c.mlx_row
        );
        total += m;
    }
    assert_eq!(
        total, 0,
        "the fused-expert quantization does not reproduce the published planes"
    );
}
