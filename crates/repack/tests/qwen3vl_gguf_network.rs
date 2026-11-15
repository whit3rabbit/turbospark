//! Qwen3-VL's GGUF text trunk, re-read off the real published files.
//!
//! `Qwen/Qwen3-VL-4B-Instruct-GGUF` (llama.cpp's converter) and
//! `mlx-community/Qwen3-VL-4B-Instruct-4bit` (mlx-vlm's) are independent
//! conversions of `Qwen/Qwen3-VL-4B-Instruct`. Checked on their bytes:
//!
//! 1. the GGUF derives the same `ArchConfig` as the MLX config, and every one
//!    of its 398 tensor names maps;
//! 2. the norm tensors (input, post-attention, q/k, final) are the official
//!    BF16 values exactly, so this family takes no norm convention;
//! 3. the Q4_K `attn_q` rows match the official `q_proj` rows AS STORED, and
//!    a rotary permutation of them does not, so no row un-permute applies.
//!
//! ```sh
//! cargo test -p turbospark-repack --test qwen3vl_gguf_network --release -- --ignored --nocapture
//! ```

use model_io::ModelFamily;
use turbospark_repack::{
    arch_from_gguf, fetch_gguf_header, fetch_safetensors_header, map_gguf_name,
    parse_qwen3_vl_config, GgufMapping, HttpRangeSource, RangeSource,
};

const GGUF_REV: &str = "1cd86afb9a95c410a6038ab3b40d8b578c892266";
const MLX_REV: &str = "2fd8dacbdb8f1e54b8c005f081ec5bf79c56376b";
const HF_REV: &str = "ebb281ec70b05090aa6165b016eac8ec08e71b17";
const FILE: &str = "Qwen3VL-4B-Instruct-Q4_K_M.gguf";

fn gguf_url() -> String {
    format!("https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF/resolve/{GGUF_REV}/{FILE}")
}

fn get_text(url: &str) -> String {
    let r = reqwest::blocking::get(url).unwrap_or_else(|e| panic!("GET {url}: {e}"));
    assert!(r.status().is_success(), "GET {url}: HTTP {}", r.status());
    r.text().expect("text")
}

fn hf_base() -> String {
    format!("https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/resolve/{HF_REV}")
}

fn hf_tensor(name: &str) -> (HttpRangeSource, String) {
    let index: serde_json::Value = serde_json::from_str(&get_text(&format!(
        "{}/model.safetensors.index.json",
        hf_base()
    )))
    .expect("index");
    let shard = index["weight_map"][name]
        .as_str()
        .unwrap_or_else(|| panic!("{name} not in the HF index"))
        .to_string();
    (
        HttpRangeSource::new(format!("{}/{shard}", hf_base())),
        name.to_string(),
    )
}

fn hf_bytes(name: &str, first_row: u64, row_bytes: u64, rows: u64) -> Vec<u8> {
    let (s, name) = hf_tensor(name);
    let h = fetch_safetensors_header(&s).expect("hf header");
    let (a, _) = h.absolute_range(&name).expect("range");
    let from = a + first_row * row_bytes;
    s.read_range(from, from + rows * row_bytes)
        .expect("hf rows")
}

fn gguf_bytes(source: &HttpRangeSource, name: &str, from: u64, len: Option<u64>) -> Vec<u8> {
    let header = fetch_gguf_header(source).expect("header");
    let (a, b) = header
        .absolute_range(name)
        .unwrap_or_else(|| panic!("{name} missing"))
        .expect("range");
    let start = a + from;
    source
        .read_range(start, len.map_or(b, |l| start + l))
        .expect("gguf bytes")
}

fn bf16s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(2)
        .map(|c| compute::bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect()
}

fn f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[test]
#[ignore = "network: reads the GGUF header and the MLX config"]
fn the_gguf_derives_the_mlx_arch_and_every_name_maps() {
    let mlx = parse_qwen3_vl_config(&get_text(&format!(
        "https://huggingface.co/mlx-community/Qwen3-VL-4B-Instruct-4bit/resolve/{MLX_REV}/config.json"
    )))
    .expect("parses");
    let source = HttpRangeSource::new(gguf_url());
    let header = fetch_gguf_header(&source).expect("header");
    assert_eq!(header.architecture(), Some("qwen3vl"));
    let gguf = arch_from_gguf(&header).expect("derives");
    assert_eq!(gguf, mlx, "GGUF and MLX derive different configs");
    for name in header.tensors.keys() {
        match map_gguf_name(name, ModelFamily::Qwen3Vl) {
            Ok(GgufMapping::Resident(_)) => {}
            other => panic!("{name} maps to {other:?}"),
        }
    }
    println!(
        "{} tensors, all resident; arch == MLX",
        header.tensors.len()
    );
}

#[test]
#[ignore = "network: reads norms and 128 attn_q rows from the GGUF and the official BF16"]
fn norms_are_the_official_values_and_q_rows_need_no_unpermute() {
    let gguf = HttpRangeSource::new(gguf_url());
    // Norms: F32 in the GGUF, BF16 upstream, identical after widening.
    for (g, h) in [
        (
            "blk.0.attn_norm.weight",
            "model.language_model.layers.0.input_layernorm.weight",
        ),
        (
            "blk.0.ffn_norm.weight",
            "model.language_model.layers.0.post_attention_layernorm.weight",
        ),
        (
            "blk.0.attn_q_norm.weight",
            "model.language_model.layers.0.self_attn.q_norm.weight",
        ),
        (
            "blk.0.attn_k_norm.weight",
            "model.language_model.layers.0.self_attn.k_norm.weight",
        ),
        (
            "blk.17.attn_q_norm.weight",
            "model.language_model.layers.17.self_attn.q_norm.weight",
        ),
        ("output_norm.weight", "model.language_model.norm.weight"),
    ] {
        let ours = f32s(&gguf_bytes(&gguf, g, 0, None));
        let (s, name) = hf_tensor(h);
        let hdr = fetch_safetensors_header(&s).unwrap();
        let (a, b) = hdr.absolute_range(&name).unwrap();
        let theirs = bf16s(&s.read_range(a, b).unwrap());
        assert_eq!(ours, theirs, "{g} differs from the official {h}");
    }

    // attn_q rows: Q4_K, 4096 rows of 2560 columns, 128 rows = one head.
    const HEAD_DIM: usize = 128;
    let cols = 2560usize;
    let row_bytes = (cols / 256 * 144) as u64;
    let raw = gguf_bytes(
        &gguf,
        "blk.0.attn_q.weight",
        0,
        Some(HEAD_DIM as u64 * row_bytes),
    );
    let rows: Vec<Vec<f32>> = (0..HEAD_DIM)
        .map(|r| {
            compute::quant_gguf::dequantize_q4_k(
                &raw[r * row_bytes as usize..(r + 1) * row_bytes as usize],
                cols,
            )
        })
        .collect();
    let hf = hf_bytes(
        "model.language_model.layers.0.self_attn.q_proj.weight",
        0,
        (cols * 2) as u64,
        HEAD_DIM as u64,
    );
    let hf_rows: Vec<Vec<f32>> = (0..HEAD_DIM)
        .map(|r| bf16s(&hf[r * cols * 2..(r + 1) * cols * 2]))
        .collect();
    let err = |a: &[f32], b: &[f32]| {
        let scale = b.iter().map(|v| v.abs()).sum::<f32>() / b.len() as f32;
        a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum::<f32>() / b.len() as f32 / scale
    };
    let half = HEAD_DIM / 2;
    let (mut stored, mut permuted) = (0f32, 0f32);
    for i in 0..HEAD_DIM {
        stored += err(&rows[i], &hf_rows[i]);
        permuted += err(&rows[(i % half) * 2 + i / half], &hf_rows[i]);
    }
    let (stored, permuted) = (stored / HEAD_DIM as f32, permuted / HEAD_DIM as f32);
    println!("relative error: as stored {stored:.4}, rotary-permuted {permuted:.4}");
    assert!(
        stored < 0.15,
        "stored rows do not match the official q_proj: {stored}"
    );
    assert!(
        permuted > 0.5,
        "a permutation also matches: the check cannot tell them apart"
    );
}
