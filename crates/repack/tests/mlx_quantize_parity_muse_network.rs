//! Quantize-on-install oracle for Muse Glimmer, read off REAL published bytes
//! with ranged reads (a few MB):
//!
//! ```sh
//! cargo test -p turbospark-repack --test mlx_quantize_parity_muse_network --release -- --ignored --nocapture
//! ```
//!
//! `meta-models/Muse-Glimmer-30B` is the official BF16 checkpoint and
//! `mlx-community/Muse-Glimmer-30B-4bit` its `mlx_lm.convert` conversion. This
//! pins, on bytes, what the view assumes: the text-trunk matrices (embedding,
//! untied head, attention incl. the output gate, FFN) quantize to the
//! published planes, and the CENTERED norms are the same bytes in both, so
//! Muse takes no norm shift.

use turbospark_repack::{fetch_safetensors_header, HttpRangeSource, RangeSource};

const HF: &str = "https://huggingface.co/meta-models/Muse-Glimmer-30B/resolve/cec94fcfd6512f2cfe708138c32533b346ea4f69";
const MLX: &str = "https://huggingface.co/mlx-community/Muse-Glimmer-30B-4bit/resolve/3e7677d7a40d348a3daba263a2b1c0aa41910710";

/// Tie rule of this conversion, measured (see the Gemma 4 and MoE oracles).
const ROUNDING: compute::quant_mlx::Rounding = compute::quant_mlx::Rounding::HalfAway;

fn get_text(url: &str) -> String {
    let r = reqwest::blocking::get(url).unwrap_or_else(|e| panic!("GET {url}: {e}"));
    assert!(r.status().is_success(), "GET {url}: HTTP {}", r.status());
    r.text().expect("text")
}

fn shard_of(base: &str, tensor: &str) -> String {
    let index: serde_json::Value =
        serde_json::from_str(&get_text(&format!("{base}/model.safetensors.index.json")))
            .expect("index");
    index["weight_map"][tensor]
        .as_str()
        .unwrap_or_else(|| panic!("{tensor} not in {base}"))
        .to_string()
}

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

/// `(HF name, MLX base)` of a quantized matrix.
fn matrix(tail: &str) -> (String, String) {
    match tail {
        "lm_head" => ("lm_head.weight".into(), "language_model.lm_head".into()),
        t => (
            format!("model.language_model.{t}.weight"),
            format!("language_model.model.{t}"),
        ),
    }
}

#[test]
#[ignore = "ranged reads off meta-models/Muse-Glimmer-30B and its mlx-community 4-bit conversion"]
fn bf16_matrices_quantize_to_the_published_planes() {
    let tails = [
        "embed_tokens",
        "lm_head",
        "layers.0.self_attn.q_proj",
        "layers.0.self_attn.k_proj",
        "layers.0.self_attn.v_proj",
        "layers.0.self_attn.gate_proj",
        "layers.0.self_attn.o_proj",
        "layers.0.mlp.gate_proj",
        "layers.0.mlp.up_proj",
        "layers.0.mlp.down_proj",
        // A full-attention (NoPE) layer.
        "layers.3.self_attn.q_proj",
        "layers.3.self_attn.gate_proj",
        "layers.3.mlp.down_proj",
    ];
    let mut total = 0usize;
    for tail in tails {
        let (hf_name, mx_base) = matrix(tail);
        let hf = HttpRangeSource::new(format!("{HF}/{}", shard_of(HF, &hf_name)));
        let mx = HttpRangeSource::new(format!(
            "{MLX}/{}",
            shard_of(MLX, &format!("{mx_base}.weight"))
        ));
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
        println!("{hf_name:62} {nrows}x{cols}  mismatched units {mismatch}");
        total += mismatch;
    }
    assert_eq!(total, 0, "the port does not reproduce the published planes");
}

#[test]
#[ignore = "ranged reads off meta-models/Muse-Glimmer-30B and its mlx-community 4-bit conversion"]
fn centered_norms_are_the_same_bytes_in_hf_and_mlx() {
    for tail in [
        "layers.0.input_layernorm.weight",
        "layers.0.post_attention_layernorm.weight",
        "layers.0.pre_feedforward_layernorm.weight",
        "layers.0.post_feedforward_layernorm.weight",
        "layers.3.input_layernorm.weight",
        "norm.weight",
    ] {
        let hf_name = format!("model.language_model.{tail}");
        let mx_name = format!("language_model.model.{tail}");
        let hf = HttpRangeSource::new(format!("{HF}/{}", shard_of(HF, &hf_name)));
        let mx = HttpRangeSource::new(format!("{MLX}/{}", shard_of(MLX, &mx_name)));
        let (a, b) = (whole(&hf, &hf_name), whole(&mx, &mx_name));
        assert_eq!(a, b, "{tail} differs between HF and MLX");
        println!("{tail:50} identical ({} bytes)", a.len());
    }
}
