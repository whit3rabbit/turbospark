//! Gemma 3 1B GGUF witnesses, re-read off the real published files.
//!
//! `ggml-org/gemma-3-1b-it-GGUF` (llama.cpp's converter) against
//! `mlx-community/gemma-3-1b-it-bf16`, a public bf16 copy of the gated
//! `google/gemma-3-1b-it` (HTTP 401 unauthenticated). Four things are checked
//! on bytes and on the config, none of which the offline tests can see:
//!
//! 1. the GGUF-derived `ArchConfig` agrees with the HF config's own numbers
//!    (`query_pre_attn_scalar`, `rope_local_base_freq`, `rope_theta`,
//!    `sliding_window`, `sliding_window_pattern`) and every one of the 340
//!    tensor names maps;
//! 2. every norm (four per layer, q/k per head, the final one), minus the
//!    converter's `+1` and rounded to BF16, is the HF tensor byte for byte;
//! 3. the Q8_0 `attn_q` and `attn_k` rows are the HF rows in the SAME order
//!    (llama.cpp applies NO rotary permutation to Gemma 3), and a Muse-style
//!    un-permute would break them;
//! 4. the tied head: no `output.weight`, and `token_embd` is the HF embedding
//!    (Q8_0 precision).
//!
//! What this does NOT prove: that the `mlx-community` copy equals the gated
//! original (it is a conversion, and its provenance is the uploader's), and
//! nothing here runs a forward pass.
//!
//! ```sh
//! cargo test -p turbospark-repack --test gemma3_gguf_network --release -- --ignored --nocapture
//! ```

use model_io::ModelFamily;
use turbospark_repack::{
    arch_from_gguf, fetch_gguf_header, fetch_safetensors_header, map_gguf_name, GgufMapping,
    HttpRangeSource, RangeSource,
};

const GGUF_REV: &str = "f9c28bcd85737ffc5aef028638d3341d49869c27";
const HF_REV: &str = "60444cd9d4be58ce6995614c6551b4d8db047483";
const Q4_K_M: &str = "gemma-3-1b-it-Q4_K_M.gguf";
const Q8_0: &str = "gemma-3-1b-it-Q8_0.gguf";

fn gguf_url(file: &str) -> String {
    format!("https://huggingface.co/ggml-org/gemma-3-1b-it-GGUF/resolve/{GGUF_REV}/{file}")
}

fn hf_url(file: &str) -> String {
    format!("https://huggingface.co/mlx-community/gemma-3-1b-it-bf16/resolve/{HF_REV}/{file}")
}

fn get_text(url: &str) -> String {
    let r = reqwest::blocking::get(url).unwrap_or_else(|e| panic!("GET {url}: {e}"));
    assert!(r.status().is_success(), "GET {url}: HTTP {}", r.status());
    r.text().expect("text")
}

fn f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn gguf_tensor(source: &HttpRangeSource, name: &str) -> Vec<u8> {
    let header = fetch_gguf_header(source).expect("header");
    let (a, b) = header
        .absolute_range(name)
        .unwrap_or_else(|| panic!("{name} missing"))
        .expect("range");
    source.read_range(a, b).expect("tensor bytes")
}

#[test]
#[ignore = "network: reads two GGUF headers and the HF config"]
fn the_gguf_derives_the_hf_numbers_and_every_name_maps() {
    let cfg: serde_json::Value =
        serde_json::from_str(&get_text(&hf_url("config.json"))).expect("config");
    assert_eq!(cfg["model_type"], "gemma3_text");
    let baseline = model_io::known_architecture(ModelFamily::Gemma3);

    for file in [Q4_K_M, Q8_0] {
        let source = HttpRangeSource::new(gguf_url(file));
        let header = fetch_gguf_header(&source).expect("fetch GGUF header");
        assert_eq!(header.architecture(), Some("gemma3"), "{file}");
        let gguf = arch_from_gguf(&header).expect("derives");
        assert_eq!(gguf, baseline, "{file}: GGUF disagrees with the baseline");

        // The constants the GGUF does not carry, against the HF config.
        let hf = |k: &str| cfg[k].as_f64().unwrap_or_else(|| panic!("config {k}"));
        assert_eq!(gguf.hidden_size as f64, hf("hidden_size"));
        assert_eq!(gguf.intermediate_size as f64, hf("intermediate_size"));
        assert_eq!(gguf.num_layers as f64, hf("num_hidden_layers"));
        assert_eq!(gguf.num_heads as f64, hf("num_attention_heads"));
        assert_eq!(gguf.num_kv_heads as f64, hf("num_key_value_heads"));
        assert_eq!(gguf.head_dim as f64, hf("head_dim"));
        assert_eq!(gguf.sliding_window as f64, hf("sliding_window"));
        assert_eq!(
            gguf.rope_theta,
            hf("rope_local_base_freq"),
            "local rope base"
        );
        assert_eq!(gguf.full_rope_theta, hf("rope_theta"), "global rope base");
        assert_eq!(gguf.attention_scale, hf("query_pre_attn_scalar").powf(-0.5));
        assert!(cfg["final_logit_softcapping"].is_null() && gguf.final_logit_softcap == 0.0);
        assert!(cfg["rope_scaling"].is_null());
        // sliding_window_pattern 6: every sixth layer is global.
        let pattern = hf("sliding_window_pattern") as usize;
        for (i, &kind) in gguf.full_attention_layer_mask.iter().enumerate() {
            assert_eq!(kind == 1, (i + 1) % pattern == 0, "{file} layer {i}");
        }
        assert!(gguf.tie_word_embeddings && !header.tensors.contains_key("output.weight"));

        let mut resident = 0usize;
        for name in header.tensors.keys() {
            match map_gguf_name(name, ModelFamily::Gemma3) {
                Ok(GgufMapping::Resident(_)) => resident += 1,
                other => panic!("{file}: {name} maps to {other:?}"),
            }
        }
        assert_eq!(resident, header.tensors.len(), "{file}");
        assert_eq!(resident, 340, "{file}");
        println!("{file}: {resident} tensors, arch == baseline == HF config");
    }
}

#[test]
#[ignore = "network: reads every norm of layers 0, 5 and 25 from the GGUF and the HF bf16"]
fn every_norm_minus_one_is_the_hf_norm_byte_for_byte() {
    let gguf = HttpRangeSource::new(gguf_url(Q4_K_M));
    let hf = HttpRangeSource::new(hf_url("model.safetensors"));
    let h = fetch_safetensors_header(&hf).expect("hf header");
    let want_of = |name: &str| {
        let (a, b) = h
            .absolute_range(name)
            .unwrap_or_else(|| panic!("{name} not in the HF header"));
        hf.read_range(a, b).expect("hf norm")
    };
    // What the walk does: subtract 1 in f32, then store as BF16.
    let walk = |ggml: &[u8]| -> Vec<u8> {
        f32s(ggml)
            .iter()
            .flat_map(|&v| compute::f32_to_bf16(v - 1.0).to_le_bytes())
            .collect()
    };
    let mut checked = 0;
    for layer in [0usize, 5, 25] {
        for (g, m) in [
            ("attn_norm", "input_layernorm"),
            ("post_attention_norm", "post_attention_layernorm"),
            ("ffn_norm", "pre_feedforward_layernorm"),
            ("post_ffw_norm", "post_feedforward_layernorm"),
            ("attn_q_norm", "self_attn.q_norm"),
            ("attn_k_norm", "self_attn.k_norm"),
        ] {
            let ggml = gguf_tensor(&gguf, &format!("blk.{layer}.{g}.weight"));
            let want = want_of(&format!("model.layers.{layer}.{m}.weight"));
            assert_eq!(walk(&ggml), want, "layer {layer} {m} differs from HF");
            // The unshifted value must NOT match: the +1 is really in the file.
            let raw: Vec<u8> = f32s(&ggml)
                .iter()
                .flat_map(|&v| compute::f32_to_bf16(v).to_le_bytes())
                .collect();
            assert_ne!(raw, want, "layer {layer} {m}: the file carries no +1");
            checked += 1;
        }
    }
    // Unlike Muse Glimmer, the FINAL norm is shifted too.
    let ggml = gguf_tensor(&gguf, "output_norm.weight");
    assert_eq!(walk(&ggml), want_of("model.norm.weight"), "final norm");
    println!("{checked} per-layer norms and the final norm equal HF after -1");
}

#[test]
#[ignore = "network: reads 256 attn_q and attn_k rows from the Q8_0 GGUF and the HF bf16"]
fn q_and_k_rows_are_in_the_hf_order_with_no_rotary_permutation() {
    const HEAD_DIM: usize = 256;
    let gguf = HttpRangeSource::new(gguf_url(Q8_0));
    let header = fetch_gguf_header(&gguf).expect("header");
    let hf = HttpRangeSource::new(hf_url("model.safetensors"));
    let h = fetch_safetensors_header(&hf).expect("hf header");

    let err = |ours: &[f32], theirs: &[f32]| {
        let scale = theirs.iter().map(|v| v.abs()).sum::<f32>() / theirs.len() as f32;
        let e = ours
            .iter()
            .zip(theirs)
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / theirs.len() as f32;
        e / scale
    };
    for (g, m) in [("attn_q", "q_proj"), ("attn_k", "k_proj")] {
        let name = format!("blk.0.{g}.weight");
        let info = header.tensors.get(&name).expect("tensor");
        assert_eq!(info.ggml_type, 8, "Q8_0 expected");
        let cols = info.dims[0] as usize;
        let row_bytes = cols / 32 * 34;
        let (a, _) = header.absolute_range(&name).unwrap().expect("range");
        let raw = gguf
            .read_range(a, a + (HEAD_DIM * row_bytes) as u64)
            .expect("rows");
        let rows: Vec<Vec<f32>> = (0..HEAD_DIM)
            .map(|r| {
                compute::quant_gguf::dequantize_q8_0(&raw[r * row_bytes..(r + 1) * row_bytes], cols)
            })
            .collect();
        let hf_name = format!("model.layers.0.self_attn.{m}.weight");
        let (s, _) = h.absolute_range(&hf_name).expect("hf range");
        let hf_raw = hf
            .read_range(s, s + (HEAD_DIM * cols * 2) as u64)
            .expect("hf rows");
        let hf_rows: Vec<Vec<f32>> = (0..HEAD_DIM)
            .map(|r| {
                hf_raw[r * cols * 2..(r + 1) * cols * 2]
                    .chunks_exact(2)
                    .map(|c| compute::bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                    .collect()
            })
            .collect();
        let half = HEAD_DIM / 2;
        let (mut stored, mut unpermuted) = (0f32, 0f32);
        for i in 0..HEAD_DIM {
            stored += err(&rows[i], &hf_rows[i]);
            // The Llama/Muse rule: install row i reads GGUF row (i % half) * 2 + i / half.
            unpermuted += err(&rows[(i % half) * 2 + i / half], &hf_rows[i]);
        }
        let (stored, unpermuted) = (stored / HEAD_DIM as f32, unpermuted / HEAD_DIM as f32);
        println!("{g}: relative error as stored {stored:.4}, un-permuted {unpermuted:.4}");
        assert!(
            stored < 0.02,
            "{g}: stored rows are not the HF rows: {stored}"
        );
        assert!(
            unpermuted > 0.5,
            "{g}: an un-permute also matches, so the file may be permuted"
        );
    }
}

#[test]
#[ignore = "network: reads the first 64 embedding rows from the Q8_0 GGUF and the HF bf16"]
fn the_tied_embedding_is_the_hf_embedding() {
    const ROWS: usize = 64;
    let gguf = HttpRangeSource::new(gguf_url(Q8_0));
    let header = fetch_gguf_header(&gguf).expect("header");
    assert!(!header.tensors.contains_key("output.weight"));
    let info = header.tensors.get("token_embd.weight").expect("embedding");
    assert_eq!(info.ggml_type, 8, "Q8_0 expected");
    let cols = info.dims[0] as usize;
    let row_bytes = cols / 32 * 34;
    let (a, _) = header
        .absolute_range("token_embd.weight")
        .unwrap()
        .expect("range");
    let raw = gguf
        .read_range(a, a + (ROWS * row_bytes) as u64)
        .expect("rows");
    let hf = HttpRangeSource::new(hf_url("model.safetensors"));
    let h = fetch_safetensors_header(&hf).expect("hf header");
    let (s, _) = h
        .absolute_range("model.embed_tokens.weight")
        .expect("hf range");
    let hf_raw = hf
        .read_range(s, s + (ROWS * cols * 2) as u64)
        .expect("hf rows");
    let (mut num, mut den) = (0f64, 0f64);
    for r in 0..ROWS {
        let ours =
            compute::quant_gguf::dequantize_q8_0(&raw[r * row_bytes..(r + 1) * row_bytes], cols);
        for (c, v) in ours.iter().enumerate() {
            let o = r * cols * 2 + c * 2;
            let theirs = compute::bf16_to_f32(u16::from_le_bytes([hf_raw[o], hf_raw[o + 1]]));
            num += (*v as f64 - theirs as f64).abs();
            den += theirs.abs() as f64;
        }
    }
    println!("embedding relative error {:.4}", num / den);
    assert!(num / den < 0.02, "embedding differs from HF: {}", num / den);
}
