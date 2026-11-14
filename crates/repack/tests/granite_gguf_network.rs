//! Granite 4.x's GGUF, re-read off the real published files.
//!
//! `ibm-granite/granite-4.2-3b-GGUF` (llama.cpp's converter) against
//! `ibm-granite/granite-4.2-3b` (the official BF16). Checked on bytes:
//!
//! 1. the GGUF derives a config whose every shape field equals the official
//!    `config.json`, including the 1/head_dim attention scale (not Llama's
//!    1/sqrt), and every one of its 363 tensor names maps;
//! 2. the norms are the official BF16 values exactly (no convention);
//! 3. the Q4_K `attn_q` rows match the official rows once the Llama rotary
//!    un-permute is applied, and do not as stored.
//!
//! ```sh
//! cargo test -p turbospark-repack --test granite_gguf_network --release -- --ignored --nocapture
//! ```

use model_io::ModelFamily;
use turbospark_repack::{
    arch_from_gguf, fetch_gguf_header, fetch_safetensors_header, map_gguf_name, GgufMapping,
    HttpRangeSource, RangeSource,
};

const GGUF_REV: &str = "c40945d71cd90f249a56985e8155551a9188dc30";
const HF_REV: &str = "4e9c275e232f10211ec88644d106a1f6a1b60e06";
const FILE: &str = "granite-4.2-3b-Q4_K_M.gguf";

fn gguf_url() -> String {
    format!("https://huggingface.co/ibm-granite/granite-4.2-3b-GGUF/resolve/{GGUF_REV}/{FILE}")
}

fn hf_base() -> String {
    format!("https://huggingface.co/ibm-granite/granite-4.2-3b/resolve/{HF_REV}")
}

fn get_text(url: &str) -> String {
    let r = reqwest::blocking::get(url).unwrap_or_else(|e| panic!("GET {url}: {e}"));
    assert!(r.status().is_success(), "GET {url}: HTTP {}", r.status());
    r.text().expect("text")
}

fn hf_source(name: &str) -> HttpRangeSource {
    let index: serde_json::Value = serde_json::from_str(&get_text(&format!(
        "{}/model.safetensors.index.json",
        hf_base()
    )))
    .expect("index");
    let shard = index["weight_map"][name]
        .as_str()
        .unwrap_or_else(|| panic!("{name} not in the HF index"));
    HttpRangeSource::new(format!("{}/{shard}", hf_base()))
}

fn hf_rows(name: &str, first: u64, row_bytes: u64, rows: u64) -> Vec<u8> {
    let s = hf_source(name);
    let h = fetch_safetensors_header(&s).expect("hf header");
    let (a, _) = h.absolute_range(name).expect("range");
    let from = a + first * row_bytes;
    s.read_range(from, from + rows * row_bytes)
        .expect("hf rows")
}

fn gguf_bytes(source: &HttpRangeSource, name: &str, len: Option<u64>) -> Vec<u8> {
    let header = fetch_gguf_header(source).expect("header");
    let (a, b) = header
        .absolute_range(name)
        .unwrap_or_else(|| panic!("{name} missing"))
        .expect("range");
    source
        .read_range(a, len.map_or(b, |l| a + l))
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
#[ignore = "network: reads the GGUF header and the official config"]
fn the_gguf_derives_the_official_config_and_every_name_maps() {
    let cfg: serde_json::Value =
        serde_json::from_str(&get_text(&format!("{}/config.json", hf_base()))).expect("config");
    let source = HttpRangeSource::new(gguf_url());
    let header = fetch_gguf_header(&source).expect("header");
    assert_eq!(header.architecture(), Some("granite"));
    let a = arch_from_gguf(&header).expect("derives");
    let i = |k: &str| cfg[k].as_i64().unwrap_or_else(|| panic!("{k}"));
    assert_eq!(a.family, ModelFamily::Llama);
    assert_eq!(a.num_layers, i("num_hidden_layers"));
    assert_eq!(a.hidden_size, i("hidden_size"));
    assert_eq!(a.num_heads, i("num_attention_heads"));
    assert_eq!(a.num_kv_heads, i("num_key_value_heads"));
    assert_eq!(a.intermediate_size, i("intermediate_size"));
    assert_eq!(a.vocab_size, i("vocab_size"));
    assert_eq!(a.head_dim, i("hidden_size") / i("num_attention_heads"));
    assert_eq!(a.rope_theta, cfg["rope_theta"].as_f64().unwrap());
    assert_eq!(
        a.attention_scale,
        cfg["attention_multiplier"].as_f64().unwrap()
    );
    assert_eq!(
        a.tie_word_embeddings,
        cfg["tie_word_embeddings"].as_bool().unwrap()
    );
    assert_eq!(a.num_experts, 0);
    assert!(a.full_attention_layer_mask.iter().all(|&m| m == 1));
    for name in header.tensors.keys() {
        match map_gguf_name(name, ModelFamily::Llama) {
            Ok(GgufMapping::Resident(_)) => {}
            other => panic!("{name} maps to {other:?}"),
        }
    }
    println!(
        "{} tensors, all resident; config matches the official one",
        header.tensors.len()
    );
}

#[test]
#[ignore = "network: reads norms and 64 attn_q rows from the GGUF and the official BF16"]
fn norms_are_official_and_q_rows_need_the_llama_unpermute() {
    let gguf = HttpRangeSource::new(gguf_url());
    for (g, h) in [
        (
            "blk.0.attn_norm.weight",
            "model.layers.0.input_layernorm.weight",
        ),
        (
            "blk.0.ffn_norm.weight",
            "model.layers.0.post_attention_layernorm.weight",
        ),
        (
            "blk.20.attn_norm.weight",
            "model.layers.20.input_layernorm.weight",
        ),
        ("output_norm.weight", "model.norm.weight"),
    ] {
        let ours = f32s(&gguf_bytes(&gguf, g, None));
        let s = hf_source(h);
        let hdr = fetch_safetensors_header(&s).unwrap();
        let (a, b) = hdr.absolute_range(h).unwrap();
        assert_eq!(
            ours,
            bf16s(&s.read_range(a, b).unwrap()),
            "{g} differs from {h}"
        );
    }

    const HEAD_DIM: usize = 64;
    let cols = 2560usize;
    let row_bytes = cols / 256 * 144;
    let raw = gguf_bytes(
        &gguf,
        "blk.0.attn_q.weight",
        Some((HEAD_DIM * row_bytes) as u64),
    );
    let rows: Vec<Vec<f32>> = (0..HEAD_DIM)
        .map(|r| {
            compute::quant_gguf::dequantize_q4_k(&raw[r * row_bytes..(r + 1) * row_bytes], cols)
        })
        .collect();
    let hf = hf_rows(
        "model.layers.0.self_attn.q_proj.weight",
        0,
        (cols * 2) as u64,
        HEAD_DIM as u64,
    );
    let hf_rows_f: Vec<Vec<f32>> = (0..HEAD_DIM)
        .map(|r| bf16s(&hf[r * cols * 2..(r + 1) * cols * 2]))
        .collect();
    let err = |a: &[f32], b: &[f32]| {
        let scale = b.iter().map(|v| v.abs()).sum::<f32>() / b.len() as f32;
        a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum::<f32>() / b.len() as f32 / scale
    };
    let half = HEAD_DIM / 2;
    let (mut stored, mut fixed) = (0f32, 0f32);
    for i in 0..HEAD_DIM {
        stored += err(&rows[i], &hf_rows_f[i]);
        fixed += err(&rows[(i % half) * 2 + i / half], &hf_rows_f[i]);
    }
    let (stored, fixed) = (stored / HEAD_DIM as f32, fixed / HEAD_DIM as f32);
    println!("relative error: as stored {stored:.4}, llama-unpermuted {fixed:.4}");
    assert!(
        fixed < 0.15,
        "un-permuted rows do not match the official q_proj: {fixed}"
    );
    assert!(
        stored > 0.5,
        "the stored rows already match; no permutation is in the file"
    );
}
