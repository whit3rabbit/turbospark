//! Quantize-on-install oracles, read off REAL published bytes with ranged
//! reads (a few MB, ~20 s):
//!
//! ```sh
//! cargo test -p turbospark-repack --test mlx_quantize_parity_network --release -- --ignored --nocapture
//! ```
//!
//! mlx-community publishes the same model twice, as `-bf16` and as `-4bit`,
//! both written by `mlx_lm.convert`. So the 4-bit file IS `mx.quantize` of
//! the bf16 file, and a port of it can be checked byte for byte instead of by
//! correlation. A correlation bar would pass the older min-bias quantizer
//! too, which is exactly the difference this has to catch.
//!
//! The second test pins the HF-to-MLX raw tensor relations (the norm `+1`
//! shift and anything else `sanitize` does) against the same weights in the
//! HF transformers layout (`Jackrong/...`, the base the mlx-community pair
//! was converted from), so the intake's shift list is read off bytes rather
//! than remembered.

use turbospark_repack::{fetch_safetensors_header, HttpRangeSource, RangeSource};

const BF16: &str = "https://huggingface.co/mlx-community/DeepSeek-V4-Pro-Qwen3.5-9B-bf16/resolve/d260920b984a466a72dc811c5238747dc4cf1cfb";
const Q4: &str = "https://huggingface.co/mlx-community/DeepSeek-V4-Pro-Qwen3.5-9B-4bit/resolve/ee25ba926687bb4548b16708affdd2a1a5456ae1";
const HF: &str = "https://huggingface.co/Jackrong/DeepSeek-V4-Pro-Qwen3.5-9B/resolve/7d7edca1b1e820d89127f9c8ac4b035e0f27652e";

/// `(name without the language_model. prefix, bf16 shard, 4-bit shard)`.
const LINEARS: &[(&str, &str, &str)] = &[
    (
        "model.embed_tokens.weight",
        "model-00001-of-00004.safetensors",
        "model-00001-of-00002.safetensors",
    ),
    (
        "lm_head.weight",
        "model-00004-of-00004.safetensors",
        "model-00002-of-00002.safetensors",
    ),
    (
        "model.layers.0.linear_attn.in_proj_qkv.weight",
        "model-00001-of-00004.safetensors",
        "model-00001-of-00002.safetensors",
    ),
    (
        "model.layers.0.linear_attn.in_proj_a.weight",
        "model-00001-of-00004.safetensors",
        "model-00001-of-00002.safetensors",
    ),
    (
        "model.layers.3.self_attn.q_proj.weight",
        "model-00001-of-00004.safetensors",
        "model-00001-of-00002.safetensors",
    ),
    (
        "model.layers.31.mlp.down_proj.weight",
        "model-00004-of-00004.safetensors",
        "model-00001-of-00002.safetensors",
    ),
];

fn src(base: &str, shard: &str) -> HttpRangeSource {
    HttpRangeSource::new(format!("{base}/{shard}"))
}

fn read_rows(s: &HttpRangeSource, name: &str, row: u64, row_bytes: u64) -> Vec<u8> {
    let h = fetch_safetensors_header(s).expect("header");
    let (start, end) = h
        .absolute_range(name)
        .unwrap_or_else(|| panic!("{name} missing"));
    let from = start + row * row_bytes;
    assert!(from + row_bytes <= end, "{name} row {row} out of range");
    s.read_range(from, from + row_bytes).expect("row read")
}

fn u16s(b: &[u8]) -> Vec<u16> {
    b.chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

#[test]
#[ignore = "ranged reads off two published mlx-community checkpoints"]
fn bf16_quantizes_to_the_published_4bit_bytes() {
    let mut total_mismatch = 0usize;
    for (name, bf_shard, q_shard) in LINEARS {
        let full = format!("language_model.{name}");
        let bs = src(BF16, bf_shard);
        let qs = src(Q4, q_shard);
        let bh = fetch_safetensors_header(&bs).unwrap();
        let info = &bh.tensors[&full];
        assert_eq!(info.dtype, "BF16", "{full}");
        let (rows, cols) = (info.shape[0], info.shape[1]);
        assert_eq!(cols % 64, 0, "{full} cols");
        let mut mismatch = 0usize;
        let sample = [0, 1, 2, rows / 2, rows - 1];
        for &r in &sample {
            let raw = read_rows(&bs, &full, r, cols * 2);
            let row: Vec<f32> = u16s(&raw).into_iter().map(compute::bf16_to_f32).collect();
            let q = compute::quant_mlx::quantize_int4_affine_mlx(&row);
            let w = read_rows(&qs, &full, r, cols / 2);
            let sname = full.replace(".weight", ".scales");
            let bname = full.replace(".weight", ".biases");
            let sc = u16s(&read_rows(&qs, &sname, r, cols / 64 * 2));
            let bi = u16s(&read_rows(&qs, &bname, r, cols / 64 * 2));
            let m = q.packed.iter().zip(&w).filter(|(a, b)| a != b).count()
                + q.scales.iter().zip(&sc).filter(|(a, b)| a != b).count()
                + q.biases.iter().zip(&bi).filter(|(a, b)| a != b).count();
            mismatch += m;
        }
        println!("{full:60} {rows}x{cols}  rows {sample:?}  mismatched units {mismatch}");
        total_mismatch += mismatch;
    }
    assert_eq!(
        total_mismatch, 0,
        "the port does not reproduce mx.quantize byte for byte"
    );
}

/// Raw (unquantized) trunk tensors and how the MLX file must relate to the
/// HF one. `true` means MLX = HF + 1 (the norm shift), `false` means the
/// values are identical.
const RAW: &[(&str, &str, &str, bool)] = &[
    (
        "model.layers.0.input_layernorm.weight",
        "model-00001-of-00004.safetensors",
        "model.safetensors-00004-of-00004.safetensors",
        true,
    ),
    (
        "model.layers.0.post_attention_layernorm.weight",
        "model-00001-of-00004.safetensors",
        "model.safetensors-00004-of-00004.safetensors",
        true,
    ),
    (
        "model.layers.3.self_attn.q_norm.weight",
        "model-00001-of-00004.safetensors",
        "model.safetensors-00004-of-00004.safetensors",
        true,
    ),
    (
        "model.layers.3.self_attn.k_norm.weight",
        "model-00001-of-00004.safetensors",
        "model.safetensors-00004-of-00004.safetensors",
        true,
    ),
    (
        "model.norm.weight",
        "model-00004-of-00004.safetensors",
        "model.safetensors-00004-of-00004.safetensors",
        true,
    ),
    (
        "model.layers.0.linear_attn.norm.weight",
        "model-00001-of-00004.safetensors",
        "model.safetensors-00004-of-00004.safetensors",
        false,
    ),
    (
        "model.layers.0.linear_attn.A_log",
        "model-00001-of-00004.safetensors",
        "model.safetensors-00004-of-00004.safetensors",
        false,
    ),
    (
        "model.layers.0.linear_attn.dt_bias",
        "model-00001-of-00004.safetensors",
        "model.safetensors-00004-of-00004.safetensors",
        false,
    ),
    (
        "model.layers.0.linear_attn.conv1d.weight",
        "model-00001-of-00004.safetensors",
        "model.safetensors-00004-of-00004.safetensors",
        false,
    ),
];

fn read_f32(s: &HttpRangeSource, name: &str) -> (String, Vec<u64>, Vec<f32>) {
    let h = fetch_safetensors_header(s).expect("header");
    let info = h
        .tensors
        .get(name)
        .unwrap_or_else(|| panic!("{name} missing"));
    let (a, b) = h.absolute_range(name).unwrap();
    let raw = s.read_range(a, b).unwrap();
    let vals = match info.dtype.as_str() {
        "BF16" => u16s(&raw).into_iter().map(compute::bf16_to_f32).collect(),
        "F32" => raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        d => panic!("{name}: dtype {d}"),
    };
    (info.dtype.clone(), info.shape.clone(), vals)
}

#[test]
#[ignore = "ranged reads off mlx-community bf16 and the HF base checkpoint"]
fn hf_raw_tensors_relate_to_mlx_as_sanitize_says() {
    // Which HF shard holds which tensor varies; find the name by trying
    // the shards the index lists for this repo.
    for (name, mlx_shard, hf_shard, shifted) in RAW {
        let mlx_name = format!("language_model.{name}");
        let hf_name = name.replacen("model.", "model.language_model.", 1);
        let (md, ms, mv) = read_f32(&src(BF16, mlx_shard), &mlx_name);
        let (hd, hs, hv) = read_f32(&src(HF, hf_shard), &hf_name);
        assert_eq!(mv.len(), hv.len(), "{name} element count");
        let max_eq = mv
            .iter()
            .zip(&hv)
            .map(|(m, h)| (m - h).abs())
            .fold(0f32, f32::max);
        let max_shift = mv
            .iter()
            .zip(&hv)
            .map(|(m, h)| (m - compute::bf16_to_f32(compute::f32_to_bf16(h + 1.0))).abs())
            .fold(0f32, f32::max);
        let mean_m = mv.iter().sum::<f32>() / mv.len() as f32;
        let mean_h = hv.iter().sum::<f32>() / hv.len() as f32;
        println!(
            "{name:52} mlx {md} {ms:?} mean {mean_m:+.4} | hf {hd} {hs:?} mean {mean_h:+.4} | max|m-h| {max_eq:.3e} max|m-(h+1)| {max_shift:.3e}"
        );
        if *shifted {
            assert_eq!(max_shift, 0.0, "{name}: MLX is not exactly bf16(HF + 1)");
        } else {
            assert!(max_eq < 1e-2, "{name}: MLX differs from HF ({max_eq})");
        }
    }
}
