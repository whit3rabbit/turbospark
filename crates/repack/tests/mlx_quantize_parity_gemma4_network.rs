//! Quantize-on-install oracle for the dense Gemma 4 text trunk, read off REAL
//! published bytes with ranged reads (a few MB):
//!
//! ```sh
//! cargo test -p turbospark-repack --test mlx_quantize_parity_gemma4_network --release -- --ignored --nocapture
//! ```
//!
//! `google/gemma-4-31B-it` is the HF transformers checkpoint and
//! `mlx-community/gemma-4-31b-it-4bit` its `mlx_lm.convert` conversion. This
//! pins, on bytes, what the view assumes: the matrices quantize to the
//! published planes, and the raw tensors (norms, `layer_scalar`) are the SAME
//! bytes in both, so Gemma 4 takes no norm shift.

use turbospark_repack::{fetch_safetensors_header, HttpRangeSource, RangeSource};

const HF: &str =
    "https://huggingface.co/google/gemma-4-31B-it/resolve/842da3794eaa0b77d5f08bae87a17459d91ff475";
const MLX: &str = "https://huggingface.co/mlx-community/gemma-4-31b-it-4bit/resolve/696d436c404745a59f30e4939a658162b0a9e57f";

/// This conversion rounds ties away from zero (the Metal kernel), measured: 0
/// mismatches under it. The Qwen3.6 MoE pair needs `HalfEven` instead.
const ROUNDING: compute::quant_mlx::Rounding = compute::quant_mlx::Rounding::HalfAway;

const HF_S1: &str = "model-00001-of-00002.safetensors";
const HF_S2: &str = "model-00002-of-00002.safetensors";
const MX_S1: &str = "model-00001-of-00004.safetensors";
const MX_S4: &str = "model-00004-of-00004.safetensors";

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

fn whole(s: &HttpRangeSource, name: &str) -> Vec<u8> {
    let h = fetch_safetensors_header(s).expect("header");
    let (a, b) = h
        .absolute_range(name)
        .unwrap_or_else(|| panic!("{name} missing"));
    s.read_range(a, b).expect("read")
}

struct Mat {
    /// Name below `model.language_model.` / `language_model.model.`.
    tail: &'static str,
    hf_shard: &'static str,
}

const MATS: &[Mat] = &[
    Mat {
        tail: "embed_tokens",
        hf_shard: HF_S1,
    },
    Mat {
        tail: "layers.0.self_attn.q_proj",
        hf_shard: HF_S1,
    },
    Mat {
        tail: "layers.0.self_attn.k_proj",
        hf_shard: HF_S1,
    },
    Mat {
        tail: "layers.0.self_attn.v_proj",
        hf_shard: HF_S1,
    },
    Mat {
        tail: "layers.0.self_attn.o_proj",
        hf_shard: HF_S1,
    },
    Mat {
        tail: "layers.0.mlp.gate_proj",
        hf_shard: HF_S1,
    },
    Mat {
        tail: "layers.0.mlp.up_proj",
        hf_shard: HF_S1,
    },
    Mat {
        tail: "layers.0.mlp.down_proj",
        hf_shard: HF_S1,
    },
    // A full-attention layer (no v_proj, attention_k_eq_v).
    Mat {
        tail: "layers.5.self_attn.q_proj",
        hf_shard: HF_S2,
    },
    Mat {
        tail: "layers.5.self_attn.k_proj",
        hf_shard: HF_S2,
    },
    Mat {
        tail: "layers.5.self_attn.o_proj",
        hf_shard: HF_S2,
    },
    Mat {
        tail: "layers.5.mlp.down_proj",
        hf_shard: HF_S2,
    },
];

#[test]
#[ignore = "ranged reads off google/gemma-4-31B-it and its mlx-community 4-bit conversion"]
fn bf16_matrices_quantize_to_the_published_planes() {
    let mx = HttpRangeSource::new(format!("{MLX}/{MX_S1}"));
    let mut total = 0usize;
    for m in MATS {
        let hf = HttpRangeSource::new(format!("{HF}/{}", m.hf_shard));
        let hf_name = format!("model.language_model.{}.weight", m.tail);
        let mx_base = format!("language_model.model.{}", m.tail);
        let info = fetch_safetensors_header(&hf).unwrap().tensors[&hf_name].clone();
        assert_eq!(info.dtype, "BF16", "{hf_name}");
        let (nrows, cols) = (info.shape[0], info.shape[1]);
        assert_eq!(cols % 64, 0, "{hf_name}");
        let mut mismatch = 0usize;
        for r in [0, 1, nrows / 2, nrows - 1] {
            let raw = rows(&hf, &hf_name, r, cols * 2);
            let row: Vec<f32> = u16s(&raw).into_iter().map(compute::bf16_to_f32).collect();
            let q = compute::quant_mlx::quantize_int4_affine_mlx_with(&row, ROUNDING);
            let w = rows(&mx, &format!("{mx_base}.weight"), r, cols / 2);
            let sc = u16s(&rows(&mx, &format!("{mx_base}.scales"), r, cols / 64 * 2));
            let bi = u16s(&rows(&mx, &format!("{mx_base}.biases"), r, cols / 64 * 2));
            mismatch += q.packed.iter().zip(&w).filter(|(a, b)| a != b).count()
                + q.scales.iter().zip(&sc).filter(|(a, b)| a != b).count()
                + q.biases.iter().zip(&bi).filter(|(a, b)| a != b).count();
        }
        println!("{hf_name:70} {nrows}x{cols}  mismatched units {mismatch}");
        total += mismatch;
    }
    assert_eq!(total, 0, "the port does not reproduce the published planes");
}

#[test]
#[ignore = "ranged reads off google/gemma-4-31B-it and its mlx-community 4-bit conversion"]
fn raw_tensors_are_the_same_bytes_in_hf_and_mlx() {
    // (tail, HF shard, MLX shard): no shift, no relabel.
    let raws = [
        ("layers.0.input_layernorm.weight", HF_S1, MX_S1),
        ("layers.0.post_attention_layernorm.weight", HF_S1, MX_S1),
        ("layers.0.pre_feedforward_layernorm.weight", HF_S1, MX_S1),
        ("layers.0.post_feedforward_layernorm.weight", HF_S1, MX_S1),
        ("layers.0.self_attn.q_norm.weight", HF_S1, MX_S1),
        ("layers.0.self_attn.k_norm.weight", HF_S1, MX_S1),
        ("layers.0.layer_scalar", HF_S1, MX_S1),
        ("layers.5.post_feedforward_layernorm.weight", HF_S2, MX_S1),
        ("norm.weight", HF_S1, MX_S4),
    ];
    for (tail, hs, ms) in raws {
        let hf = HttpRangeSource::new(format!("{HF}/{hs}"));
        let mx = HttpRangeSource::new(format!("{MLX}/{ms}"));
        let a = whole(&hf, &format!("model.language_model.{tail}"));
        let b = whole(&mx, &format!("language_model.model.{tail}"));
        assert_eq!(a, b, "{tail} differs between HF and MLX");
        println!("{tail:60} identical ({} bytes)", a.len());
    }
}
