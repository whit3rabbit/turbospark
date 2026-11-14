//! Quantize-on-install for the DENSE Gemma 4 text trunk, end to end through
//! the real Gemma 4 writers (streamed and one-shot).
//!
//! The HF spelling is `model.language_model.*` with the towers under
//! `model.vision_tower.*` / `model.embed_vision.*`; layer tensor names, norms
//! and `layer_scalar` carry the same bytes as the MLX conversion (read off
//! `google/gemma-4-31B-it` against `mlx-community/gemma-4-31b-it-4bit`), so
//! the install from HF BF16 must equal the install from the MLX-quantized
//! reference, file for file.

use std::path::{Path, PathBuf};

use model_io::ModelFamily;

use crate::gemma4_checkpoint::{
    write_gemma4_install, write_gemma4_install_streamed, Gemma4Quant, Gemma4Shards,
};
use crate::quantize_on_install::{
    canonicalize_for_family, quantize_on_install_shards_for, quantize_refusal,
};
use crate::ranged_download::{MemoryRangeSource, RangeSource};
use crate::safetensors_header::{parse_header, DEFAULT_MAX_HEADER_BYTES};
use crate::synthetic_tensors::{assemble_safetensors, Tensor};

use super::arch::tiny_gemma4_arch;

const MODEL_ID: &str = "synthetic/qoi-gemma4";
const HIDDEN: usize = 64;
const HEAD_DIM: usize = 32;
const NUM_HEADS: usize = 2;
const INTER: usize = 64;
const LAYERS: usize = 6;
const VOCAB: usize = 256;

fn arch() -> model_io::ArchConfig {
    let mut a = tiny_gemma4_arch(VOCAB as i64, LAYERS as i64);
    a.num_experts = 0;
    a.top_k_experts = 0;
    a.moe_intermediate_size = 0;
    a.sliding_window = 16;
    // The real 5:1 schedule: the last of six layers is full attention and
    // ships no v_proj (attention_k_eq_v).
    a.full_attention_layer_mask = (0..LAYERS).map(|l| (l % 6 == 5) as u8).collect();
    a
}

fn bf16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|&x| compute::f32_to_bf16(x).to_le_bytes())
        .collect()
}

/// BF16-representable, mixed sign, deterministic per `(seed, index)`.
fn values(seed: u64, n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let h = (i as u64 + 1)
                .wrapping_mul(2_654_435_761)
                .wrapping_add(seed.wrapping_mul(40_503));
            let x = ((h >> 5) % 2001) as f32 / 1000.0 - 1.0;
            compute::bf16_to_f32(compute::f32_to_bf16(x * 0.3))
        })
        .collect()
}

struct Mat {
    name: String,
    rows: usize,
    cols: usize,
    vals: Vec<f32>,
}

fn raw(name: &str, shape: Vec<u64>, vals: &[f32]) -> Tensor {
    Tensor {
        name: name.to_string(),
        dtype: "BF16",
        shape,
        bytes: bf16_bytes(vals),
    }
}

/// `(matrices, raw tensors)` in the MLX spelling.
fn weights() -> (Vec<Mat>, Vec<Tensor>) {
    let qk = NUM_HEADS * HEAD_DIM;
    let mut mats = Vec::new();
    let mut raws = Vec::new();
    let mut seed = 1u64;
    let mut mat = |name: String, rows: usize, cols: usize| {
        seed += 1;
        mats.push(Mat {
            vals: values(seed, rows * cols),
            name,
            rows,
            cols,
        });
    };
    mat(
        "language_model.model.embed_tokens.weight".into(),
        VOCAB,
        HIDDEN,
    );
    for l in 0..LAYERS {
        let p = format!("language_model.model.layers.{l}");
        mat(format!("{p}.self_attn.q_proj.weight"), qk, HIDDEN);
        mat(format!("{p}.self_attn.k_proj.weight"), qk, HIDDEN);
        if l % 6 != 5 {
            mat(format!("{p}.self_attn.v_proj.weight"), qk, HIDDEN);
        }
        mat(format!("{p}.self_attn.o_proj.weight"), HIDDEN, qk);
        mat(format!("{p}.mlp.gate_proj.weight"), INTER, HIDDEN);
        mat(format!("{p}.mlp.up_proj.weight"), INTER, HIDDEN);
        mat(format!("{p}.mlp.down_proj.weight"), HIDDEN, INTER);
        // Norm weights are served verbatim (no shift for Gemma 4), so use
        // values around 1..5 like the real ones.
        for (i, n) in [
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ]
        .iter()
        .enumerate()
        {
            let v: Vec<f32> = values(900 + i as u64 + l as u64, HIDDEN)
                .iter()
                .map(|x| 2.0 + x * 4.0)
                .collect();
            raws.push(raw(&format!("{p}.{n}.weight"), vec![HIDDEN as u64], &v));
        }
        for n in ["q_norm", "k_norm"] {
            let v: Vec<f32> = values(700 + l as u64, HEAD_DIM)
                .iter()
                .map(|x| 1.0 + x)
                .collect();
            raws.push(raw(
                &format!("{p}.self_attn.{n}.weight"),
                vec![HEAD_DIM as u64],
                &v,
            ));
        }
        raws.push(raw(&format!("{p}.layer_scalar"), vec![1], &[0.09]));
    }
    let v: Vec<f32> = values(5, HIDDEN).iter().map(|x| 3.0 + x).collect();
    raws.push(raw(
        "language_model.model.norm.weight",
        vec![HIDDEN as u64],
        &v,
    ));
    // A tower tensor: excluded by the walk, so only its NAME matters here.
    let tower = values(77, 64 * 64);
    raws.push(raw(
        "vision_tower.patch_embedder.input_proj.weight",
        vec![64, 64],
        &tower,
    ));
    (mats, raws)
}

fn quantize_row(row: &[f32]) -> (Vec<u8>, Vec<u16>, Vec<u16>) {
    let q = compute::quant_mlx::quantize_int4_affine_mlx(row);
    (q.packed, q.scales, q.biases)
}

/// `(BF16 checkpoint, mx.quantize reference)`, both MLX-spelled.
fn checkpoints() -> (Vec<Tensor>, Vec<Tensor>) {
    let (mats, raws) = weights();
    let mut bf16: Vec<Tensor> = raws
        .iter()
        .map(|t| Tensor {
            name: t.name.clone(),
            dtype: t.dtype,
            shape: t.shape.clone(),
            bytes: t.bytes.clone(),
        })
        .collect();
    let mut reference = raws;
    for m in &mats {
        bf16.push(raw(&m.name, vec![m.rows as u64, m.cols as u64], &m.vals));
        let (mut w, mut s, mut b) = (Vec::new(), Vec::new(), Vec::new());
        for r in 0..m.rows {
            let (p, sc, bi) = quantize_row(&m.vals[r * m.cols..(r + 1) * m.cols]);
            w.extend(p);
            s.extend(sc.iter().flat_map(|v| v.to_le_bytes()));
            b.extend(bi.iter().flat_map(|v| v.to_le_bytes()));
        }
        let base = m.name.strip_suffix(".weight").unwrap();
        reference.push(Tensor {
            name: m.name.clone(),
            dtype: "U32",
            shape: vec![m.rows as u64, (m.cols / 8) as u64],
            bytes: w,
        });
        for (suffix, bytes) in [("scales", s), ("biases", b)] {
            reference.push(Tensor {
                name: format!("{base}.{suffix}"),
                dtype: "BF16",
                shape: vec![m.rows as u64, (m.cols / 64) as u64],
                bytes,
            });
        }
    }
    (bf16, reference)
}

/// The HF transformers spelling of an MLX-layout checkpoint.
fn to_hf(ts: &[Tensor]) -> Vec<Tensor> {
    ts.iter()
        .map(|t| {
            let name = if let Some(r) = t.name.strip_prefix("language_model.model.") {
                format!("model.language_model.{r}")
            } else if t.name.starts_with("vision_tower.") {
                format!("model.{}", t.name)
            } else {
                t.name.clone()
            };
            Tensor {
                name,
                dtype: t.dtype,
                shape: t.shape.clone(),
                bytes: t.bytes.clone(),
            }
        })
        .collect()
}

fn quant() -> Gemma4Quant {
    Gemma4Quant {
        default_bits: 4,
        group_size: 64,
        ..Default::default()
    }
}

fn tmp(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("qoig4_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    d
}

fn files(dir: &Path) -> Vec<(PathBuf, Vec<u8>)> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).unwrap() {
            let p = e.unwrap().path();
            if p.is_dir() {
                stack.push(p);
            } else {
                out.push((
                    p.strip_prefix(dir).unwrap().to_path_buf(),
                    std::fs::read(&p).unwrap(),
                ));
            }
        }
    }
    out.sort();
    out
}

fn assert_same_install(a: &Path, b: &Path, what: &str) {
    let (fa, fb) = (files(a), files(b));
    let names = |f: &[(PathBuf, Vec<u8>)]| f.iter().map(|(p, _)| p.clone()).collect::<Vec<_>>();
    assert_eq!(names(&fa), names(&fb), "{what}: file sets differ");
    for ((p, x), (_, y)) in fa.iter().zip(&fb) {
        assert!(x == y, "{what}: {} differs", p.display());
    }
}

fn install_through_view(
    ts: &[Tensor],
    arch: &model_io::ArchConfig,
    tag: &str,
) -> (PathBuf, PathBuf) {
    let blob = assemble_safetensors(ts);
    let mut headers = vec![parse_header(&blob, DEFAULT_MAX_HEADER_BYTES).unwrap()];
    let layout = canonicalize_for_family(ModelFamily::Gemma4, &mut headers).unwrap();
    let src = MemoryRangeSource::new(&blob);
    let srcs: Vec<&dyn RangeSource> = vec![&src];
    let view =
        quantize_on_install_shards_for(ModelFamily::Gemma4, &headers, &srcs, 4, layout).unwrap();
    let streamed = tmp(&format!("{tag}_s"));
    let shards = Gemma4Shards::new(
        view.iter()
            .map(|s| (s.header(), s as &dyn RangeSource))
            .collect(),
    )
    .unwrap();
    write_gemma4_install_streamed(&streamed, arch, MODEL_ID, &shards, &quant(), |_| {}).unwrap();
    let oneshot = tmp(&format!("{tag}_o"));
    write_gemma4_install(
        &oneshot,
        arch,
        MODEL_ID,
        view[0].header(),
        &view[0],
        &quant(),
    )
    .unwrap();
    (streamed, oneshot)
}

#[test]
fn hf_bf16_installs_byte_identical_to_its_mlx_conversion() {
    let arch = arch();
    let (bf16, reference) = checkpoints();

    let ref_blob = assemble_safetensors(&reference);
    let ref_header = parse_header(&ref_blob, DEFAULT_MAX_HEADER_BYTES).unwrap();
    let ref_src = MemoryRangeSource::new(&ref_blob);
    let ref_dir = tmp("ref");
    write_gemma4_install_streamed(
        &ref_dir,
        &arch,
        MODEL_ID,
        &Gemma4Shards::single(&ref_header, &ref_src),
        &quant(),
        |_| {},
    )
    .unwrap();

    let (ms, mo) = install_through_view(&bf16, &arch, "mlx");
    assert_same_install(&ref_dir, &ms, "MLX-layout BF16, streamed writer");
    assert_same_install(&ref_dir, &mo, "MLX-layout BF16, one-shot writer");

    let hf = to_hf(&bf16);
    assert!(hf
        .iter()
        .any(|t| t.name.starts_with("model.language_model.")));
    assert!(hf.iter().any(|t| t.name.starts_with("model.vision_tower.")));
    let (hs, ho) = install_through_view(&hf, &arch, "hf");
    assert_same_install(&ref_dir, &hs, "HF-layout BF16, streamed writer");
    assert_same_install(&ref_dir, &ho, "HF-layout BF16, one-shot writer");

    for d in [ref_dir, ms, mo, hs, ho] {
        let _ = std::fs::remove_dir_all(d);
    }
}

#[test]
fn norms_are_not_shifted_for_gemma4() {
    // A view that wrongly shifted norms would change these bytes.
    let (bf16, _) = checkpoints();
    let hf = to_hf(&bf16);
    let blob = assemble_safetensors(&hf);
    let mut headers = vec![parse_header(&blob, DEFAULT_MAX_HEADER_BYTES).unwrap()];
    let layout = canonicalize_for_family(ModelFamily::Gemma4, &mut headers).unwrap();
    let src = MemoryRangeSource::new(&blob);
    let srcs: Vec<&dyn RangeSource> = vec![&src];
    let view =
        quantize_on_install_shards_for(ModelFamily::Gemma4, &headers, &srcs, 4, layout).unwrap();
    let name = "language_model.model.layers.0.input_layernorm.weight";
    let (a, b) = view[0].header().absolute_range(name).unwrap();
    let got = view[0].read_range(a, b).unwrap();
    let want = bf16.iter().find(|t| t.name == name).unwrap();
    assert_eq!(got, want.bytes);
}

#[test]
fn an_unknown_hf_name_outside_the_mapped_prefixes_is_refused() {
    let (bf16, _) = checkpoints();
    let mut hf = to_hf(&bf16);
    hf.push(raw("model.mystery.weight", vec![1], &[1.0]));
    let blob = assemble_safetensors(&hf);
    let mut headers = vec![parse_header(&blob, DEFAULT_MAX_HEADER_BYTES).unwrap()];
    let err = canonicalize_for_family(ModelFamily::Gemma4, &mut headers).unwrap_err();
    assert!(err.contains("model.mystery.weight"), "{err}");
}

#[test]
fn variants_this_view_does_not_model_are_refused_by_config() {
    let cfg = |tc: serde_json::Value, root: &str| serde_json::json!({ "model_type": root, "text_config": tc });
    let dense = serde_json::json!({
        "hidden_size_per_layer_input": 0, "enable_moe_block": false,
        "num_experts": null, "num_kv_shared_layers": 0
    });
    assert!(quantize_refusal(ModelFamily::Gemma4, &cfg(dense, "gemma4")).is_none());
    // E2B: per-layer embeddings and shared KV.
    let ple = serde_json::json!({ "hidden_size_per_layer_input": 256, "num_kv_shared_layers": 20 });
    assert!(quantize_refusal(ModelFamily::Gemma4, &cfg(ple, "gemma4"))
        .unwrap()
        .contains("per-layer"));
    // Shared KV alone.
    let kv = serde_json::json!({ "hidden_size_per_layer_input": 0, "num_kv_shared_layers": 4 });
    assert!(quantize_refusal(ModelFamily::Gemma4, &cfg(kv, "gemma4")).is_some());
    // 26B-A4B.
    let moe = serde_json::json!({ "enable_moe_block": true, "num_experts": 128 });
    assert!(quantize_refusal(ModelFamily::Gemma4, &cfg(moe, "gemma4"))
        .unwrap()
        .contains("MoE"));
    // 12B unified root.
    let d2 = serde_json::json!({ "enable_moe_block": false });
    assert!(
        quantize_refusal(ModelFamily::Gemma4, &cfg(d2, "gemma4_unified"))
            .unwrap()
            .contains("gemma4_unified")
    );
    // Another family is never affected.
    assert!(quantize_refusal(ModelFamily::QwenGdnDense, &serde_json::json!({})).is_none());
}
