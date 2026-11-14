//! Muse Glimmer's GGUF witnesses, re-read off the real published files.
//!
//! `ggml-org/Muse-Glimmer-30B-GGUF` (llama.cpp's converter) and
//! `mlx-community/Muse-Glimmer-30B-4bit` (mlx-vlm's) are independent
//! conversions of `meta-models/Muse-Glimmer-30B`. Three things are checked on
//! their bytes, none of which the offline tests can see:
//!
//! 1. the GGUF derives the same `ArchConfig` as the MLX config and the
//!    baseline, every one of its 731 tensor names maps, and the synthesized
//!    q/k norm vectors are the constants the flow assumes;
//! 2. the four per-layer norms, minus the converter's `+1` and rounded to
//!    BF16, are the MLX conversion's tensor byte for byte, and the final norm
//!    is identical unshifted;
//! 3. the Q8_0 `attn_q` rows, un-permuted the way the walk does, are the
//!    official BF16 `q_proj` rows (to Q8_0 precision), and left permuted they
//!    are not.
//!
//! ```sh
//! cargo test -p turbospark-repack --test muse_gguf_network --release -- --ignored --nocapture
//! ```

use model_io::ModelFamily;
use turbospark_repack::{
    arch_from_gguf, fetch_gguf_header, fetch_safetensors_header, map_gguf_name,
    muse_qk_norm_is_constant, parse_muse_glimmer_config, GgufMapping, HttpRangeSource, RangeSource,
};

const GGUF_REV: &str = "f7904564af4d624372d459825c1eaf23af73677c";
const MLX_REV: &str = "3e7677d7a40d348a3daba263a2b1c0aa41910710";
const HF_REV: &str = "cec94fcfd6512f2cfe708138c32533b346ea4f69";
const Q4_K_M: &str = "Muse-Glimmer-30B-Q4_K_M.gguf";
const Q8_0: &str = "Muse-Glimmer-30B-Q8_0.gguf";

fn gguf_url(file: &str) -> String {
    format!("https://huggingface.co/ggml-org/Muse-Glimmer-30B-GGUF/resolve/{GGUF_REV}/{file}")
}

fn mlx_url(file: &str) -> String {
    format!("https://huggingface.co/mlx-community/Muse-Glimmer-30B-4bit/resolve/{MLX_REV}/{file}")
}

fn get_text(url: &str) -> String {
    let r = reqwest::blocking::get(url).unwrap_or_else(|e| panic!("GET {url}: {e}"));
    assert!(r.status().is_success(), "GET {url}: HTTP {}", r.status());
    r.text().expect("text")
}

fn shard_of(index_url: &str, tensor: &str) -> String {
    let index: serde_json::Value = serde_json::from_str(&get_text(index_url)).expect("index");
    index["weight_map"][tensor]
        .as_str()
        .unwrap_or_else(|| panic!("{tensor} not in {index_url}"))
        .to_string()
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
#[ignore = "network: reads two GGUF headers, the MLX config and a few tensors"]
fn the_gguf_derives_the_mlx_arch_and_every_name_maps() {
    let mlx = parse_muse_glimmer_config(&get_text(&mlx_url("config.json"))).expect("parses");
    let baseline = model_io::known_architecture(ModelFamily::MuseGlimmer);
    assert_eq!(mlx, baseline, "the MLX config disagrees with the baseline");

    for file in [Q4_K_M, Q8_0] {
        let source = HttpRangeSource::new(gguf_url(file));
        let header = fetch_gguf_header(&source).expect("fetch GGUF header");
        assert_eq!(header.architecture(), Some("muse-glimmer"), "{file}");
        let gguf = arch_from_gguf(&header).expect("derives");
        assert_eq!(gguf, mlx, "{file}: GGUF and MLX derive different configs");

        let (mut resident, mut qk) = (0usize, Vec::new());
        for name in header.tensors.keys() {
            match map_gguf_name(name, ModelFamily::MuseGlimmer) {
                Ok(GgufMapping::Resident(_)) => resident += 1,
                Ok(GgufMapping::MuseQkNorm { is_q }) => qk.push((name.clone(), is_q)),
                other => panic!("{file}: {name} maps to {other:?}"),
            }
        }
        assert_eq!(qk.len(), 104, "{file}: one q and one k norm per layer");
        assert_eq!(resident + qk.len(), header.tensors.len(), "{file}");
        // A sliding layer, a full layer, and the last layer.
        for layer in [0, 3, 51] {
            for (suffix, is_q) in [("attn_q_norm", true), ("attn_k_norm", false)] {
                let bytes = gguf_tensor(&source, &format!("blk.{layer}.{suffix}.weight"));
                muse_qk_norm_is_constant(&bytes, gguf.full_head_dim as usize, is_q)
                    .unwrap_or_else(|e| panic!("{file} blk.{layer}.{suffix}: {e}"));
            }
        }
        println!(
            "{file}: {} tensors ({resident} resident, {} q/k norms), arch == MLX == baseline",
            header.tensors.len(),
            qk.len()
        );
    }
}

#[test]
#[ignore = "network: reads norms from the GGUF and the MLX conversion"]
fn per_layer_norms_minus_one_are_the_mlx_norms_and_the_final_norm_is_unshifted() {
    let gguf = HttpRangeSource::new(gguf_url(Q4_K_M));
    let index_url = mlx_url("model.safetensors.index.json");
    let mut checked = 0;
    for layer in [0usize, 3, 51] {
        for (g, m) in [
            ("attn_norm", "input_layernorm"),
            ("post_attention_norm", "post_attention_layernorm"),
            ("ffn_norm", "pre_feedforward_layernorm"),
            ("post_ffw_norm", "post_feedforward_layernorm"),
        ] {
            let ggml = f32s(&gguf_tensor(&gguf, &format!("blk.{layer}.{g}.weight")));
            let mlx_name = format!("language_model.model.layers.{layer}.{m}.weight");
            let shard = shard_of(&index_url, &mlx_name);
            let mx = HttpRangeSource::new(mlx_url(&shard));
            let h = fetch_safetensors_header(&mx).expect("mlx header");
            let (a, b) = h.absolute_range(&mlx_name).expect("range");
            let want = mx.read_range(a, b).expect("mlx norm");
            // What the walk does: subtract 1 in f32, then store as BF16.
            let got: Vec<u8> = ggml
                .iter()
                .flat_map(|&v| compute::f32_to_bf16(v - 1.0).to_le_bytes())
                .collect();
            assert_eq!(
                got, want,
                "layer {layer} {m} differs from the MLX conversion"
            );
            checked += 1;
        }
    }
    // The final norm is not shifted by the converter.
    let ggml = gguf_tensor(&gguf, "output_norm.weight");
    let shard = shard_of(&index_url, "language_model.model.norm.weight");
    let mx = HttpRangeSource::new(mlx_url(&shard));
    let h = fetch_safetensors_header(&mx).expect("mlx header");
    let (a, b) = h
        .absolute_range("language_model.model.norm.weight")
        .unwrap();
    let want = mx.read_range(a, b).unwrap();
    let got: Vec<u8> = f32s(&ggml)
        .iter()
        .flat_map(|&v| compute::f32_to_bf16(v).to_le_bytes())
        .collect();
    assert_eq!(got, want, "the final norm must pass through unshifted");
    println!("{checked} per-layer norms and the final norm match the MLX conversion exactly");
}

#[test]
#[ignore = "network: reads 128 attn_q rows from the Q8_0 GGUF and the official BF16"]
fn q_rows_unpermute_to_the_official_hf_layout() {
    const HEAD_DIM: usize = 128;
    let gguf = HttpRangeSource::new(gguf_url(Q8_0));
    let header = fetch_gguf_header(&gguf).expect("header");
    let info = header.tensors.get("blk.0.attn_q.weight").expect("attn_q");
    assert_eq!(info.ggml_type, 8, "Q8_0 expected");
    let cols = info.dims[0] as usize;
    let row_bytes = cols / 32 * 34;
    let (a, _) = header
        .absolute_range("blk.0.attn_q.weight")
        .unwrap()
        .expect("range");
    let raw = gguf
        .read_range(a, a + (HEAD_DIM * row_bytes) as u64)
        .expect("128 rows");
    let rows: Vec<Vec<f32>> = (0..HEAD_DIM)
        .map(|r| {
            compute::quant_gguf::dequantize_q8_0(&raw[r * row_bytes..(r + 1) * row_bytes], cols)
        })
        .collect();

    let hf_name = "model.language_model.layers.0.self_attn.q_proj.weight";
    let index_url = format!(
        "https://huggingface.co/meta-models/Muse-Glimmer-30B/resolve/{HF_REV}/model.safetensors.index.json"
    );
    let shard = shard_of(&index_url, hf_name);
    let hf = HttpRangeSource::new(format!(
        "https://huggingface.co/meta-models/Muse-Glimmer-30B/resolve/{HF_REV}/{shard}"
    ));
    let h = fetch_safetensors_header(&hf).expect("hf header");
    let (s, _) = h.absolute_range(hf_name).expect("range");
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

    // Mean absolute error relative to the HF row's own scale.
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
    let half = HEAD_DIM / 2;
    let (mut permuted, mut fixed) = (0f32, 0f32);
    for i in 0..HEAD_DIM {
        // The walk's rule: install row i reads GGUF row (i % half) * 2 + i / half.
        fixed += err(&rows[(i % half) * 2 + i / half], &hf_rows[i]);
        permuted += err(&rows[i], &hf_rows[i]);
    }
    let (fixed, permuted) = (fixed / HEAD_DIM as f32, permuted / HEAD_DIM as f32);
    println!("relative error: un-permuted {fixed:.4}, as stored {permuted:.4}");
    assert!(
        fixed < 0.02,
        "un-permuted rows do not match the official q_proj: {fixed}"
    );
    assert!(
        permuted > 0.5,
        "the stored rows already match; no permutation is in the file"
    );
}
