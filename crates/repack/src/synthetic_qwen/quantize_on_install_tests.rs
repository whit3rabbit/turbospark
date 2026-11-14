//! Quantize-on-install, end to end through the REAL dense writers.
//!
//! One set of weights, three checkpoints, one expected install:
//!
//! - `reference`: the trunk MLX-packed by `compute::quant_mlx` (what
//!   `mlx_lm.convert` publishes), installed the way every MLX pull is.
//! - `bf16_mlx`: the same weights unquantized in the MLX layout (what
//!   `mlx-community/...-bf16` publishes), installed through the view.
//! - `bf16_hf`: the same weights in the HF transformers layout (renamed,
//!   norms zero-centred, conv1d `[C, 1, K]`), installed through the view.
//!
//! All three installs must be byte-identical, file for file. That one
//! assertion covers the virtual header, every served plane, the rename, the
//! norm shift, the conv1d relabel and both writers.

use std::path::{Path, PathBuf};

use crate::gemma4_checkpoint::{
    write_qwen_gdn_dense_install, write_qwen_gdn_dense_install_streamed, Gemma4Quant, Gemma4Shards,
};
use crate::quantize_on_install::{
    canonicalize_qwen35_hf_headers, check_norm_layout, quantize_on_install_shards, SourceLayout,
};
use crate::ranged_download::{MemoryRangeSource, RangeSource};
use crate::safetensors_header::{parse_header, DEFAULT_MAX_HEADER_BYTES};
use crate::synthetic_real::assemble_safetensors;
use crate::synthetic_tensors::Tensor;

use super::dense::{dense_checkpoint_tensors, tiny_qwen_gdn_dense_arch};

const MODEL_ID: &str = "synthetic/qoi";

fn u16s(b: &[u8]) -> Vec<u16> {
    b.chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

/// The fixture's MLX 4-bit checkpoint turned back into BF16 matrices (MLX
/// names, MLX-shifted F16 norms), plus the `mx.quantize` repack of those
/// same BF16 matrices.
fn checkpoints() -> (model_io::ArchConfig, Vec<Tensor>, Vec<Tensor>) {
    let arch = tiny_qwen_gdn_dense_arch(256, 4);
    let packed = dense_checkpoint_tensors(&arch, 4, false, false, false);
    let by_name: std::collections::HashMap<&str, &Tensor> =
        packed.iter().map(|t| (t.name.as_str(), t)).collect();
    let mut bf16 = Vec::new();
    let mut reference = Vec::new();
    for t in &packed {
        if t.name.ends_with(".scales") || t.name.ends_with(".biases") {
            continue;
        }
        if t.dtype != "U32" {
            bf16.push(Tensor {
                name: t.name.clone(),
                dtype: t.dtype,
                shape: t.shape.clone(),
                bytes: t.bytes.clone(),
            });
            reference.push(Tensor {
                name: t.name.clone(),
                dtype: t.dtype,
                shape: t.shape.clone(),
                bytes: t.bytes.clone(),
            });
            continue;
        }
        let base = t.name.strip_suffix(".weight").unwrap();
        let sc = by_name[format!("{base}.scales").as_str()];
        let bi = by_name[format!("{base}.biases").as_str()];
        assert_eq!(sc.dtype, "BF16");
        let rows = t.shape[0] as usize;
        let cols = t.shape[1] as usize * 8;
        let (sc, bi) = (u16s(&sc.bytes), u16s(&bi.bytes));
        let g = cols / 64;
        let mut dense = Vec::with_capacity(rows * cols * 2);
        let (mut qw, mut qs, mut qb) = (Vec::new(), Vec::new(), Vec::new());
        for r in 0..rows {
            let row = compute::quant::Int4AffineRow {
                packed: t.bytes[r * cols / 2..(r + 1) * cols / 2].to_vec(),
                scales: sc[r * g..(r + 1) * g].to_vec(),
                biases: bi[r * g..(r + 1) * g].to_vec(),
            };
            let vals: Vec<f32> = compute::quant::dequantize_int4_affine(&row, cols)
                .into_iter()
                .map(|v| compute::bf16_to_f32(compute::f32_to_bf16(v)))
                .collect();
            dense.extend(
                vals.iter()
                    .flat_map(|&v| compute::f32_to_bf16(v).to_le_bytes()),
            );
            let q = compute::quant_mlx::quantize_int4_affine_mlx(&vals);
            qw.extend(q.packed);
            qs.extend(q.scales.iter().flat_map(|v| v.to_le_bytes()));
            qb.extend(q.biases.iter().flat_map(|v| v.to_le_bytes()));
        }
        bf16.push(Tensor {
            name: t.name.clone(),
            dtype: "BF16",
            shape: vec![rows as u64, cols as u64],
            bytes: dense,
        });
        reference.push(Tensor {
            name: t.name.clone(),
            dtype: "U32",
            shape: t.shape.clone(),
            bytes: qw,
        });
        for (suffix, bytes) in [("scales", qs), ("biases", qb)] {
            reference.push(Tensor {
                name: format!("{base}.{suffix}"),
                dtype: "BF16",
                shape: vec![rows as u64, g as u64],
                bytes,
            });
        }
    }
    (arch, bf16, reference)
}

fn takes_shift(name: &str) -> bool {
    name.ends_with(".input_layernorm.weight")
        || name.ends_with(".post_attention_layernorm.weight")
        || name.ends_with(".self_attn.q_norm.weight")
        || name.ends_with(".self_attn.k_norm.weight")
        || name == "language_model.model.norm.weight"
}

/// The HF transformers spelling of an MLX-layout BF16 checkpoint.
fn to_hf(ts: &[Tensor]) -> Vec<Tensor> {
    ts.iter()
        .map(|t| {
            let bytes = if takes_shift(&t.name) {
                assert_eq!(t.dtype, "F16", "{}", t.name);
                // Exact: v is an F16 near 1, so v - 1 is exact in f32 and
                // representable in F16, and the view's +1 lands back on v.
                u16s(&t.bytes)
                    .into_iter()
                    .flat_map(|b| {
                        let v = compute::quant_1bit::f16_to_f32(b) - 1.0;
                        compute::quant_1bit::f32_to_f16(v).to_le_bytes()
                    })
                    .collect()
            } else {
                t.bytes.clone()
            };
            let shape = if t.name.ends_with(".conv1d.weight") {
                vec![t.shape[0], 1, t.shape[1]]
            } else {
                t.shape.clone()
            };
            let name = if let Some(tail) = t.name.strip_prefix("language_model.model.") {
                format!("model.language_model.{tail}")
            } else if let Some(tail) = t.name.strip_prefix("language_model.lm_head.") {
                format!("lm_head.{tail}")
            } else {
                t.name.clone()
            };
            Tensor {
                name,
                dtype: t.dtype,
                shape,
                bytes,
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
    let d = std::env::temp_dir().join(format!("qoi_{tag}_{}", std::process::id()));
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

/// Install `ts` through the quantized view with BOTH writers.
fn install_through_view(
    ts: &[Tensor],
    arch: &model_io::ArchConfig,
    tag: &str,
) -> (PathBuf, PathBuf) {
    let blob = assemble_safetensors(ts);
    let mut headers = vec![parse_header(&blob, DEFAULT_MAX_HEADER_BYTES).unwrap()];
    let layout = canonicalize_qwen35_hf_headers(&mut headers).unwrap();
    let src = MemoryRangeSource::new(&blob);
    let srcs: Vec<&dyn RangeSource> = vec![&src];
    check_norm_layout(&headers, &srcs, layout).unwrap();
    let view = quantize_on_install_shards(&headers, &srcs, 4, layout).unwrap();
    let streamed = tmp(&format!("{tag}_s"));
    let shards = Gemma4Shards::new(
        view.iter()
            .map(|s| (s.header(), s as &dyn RangeSource))
            .collect(),
    )
    .unwrap();
    write_qwen_gdn_dense_install_streamed(&streamed, arch, MODEL_ID, &shards, &quant(), |_| {})
        .unwrap();
    let oneshot = tmp(&format!("{tag}_o"));
    write_qwen_gdn_dense_install(
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
fn bf16_in_either_layout_installs_byte_identical_to_its_mlx_conversion() {
    let (arch, bf16, reference) = checkpoints();

    let ref_blob = assemble_safetensors(&reference);
    let ref_header = parse_header(&ref_blob, DEFAULT_MAX_HEADER_BYTES).unwrap();
    let ref_src = MemoryRangeSource::new(&ref_blob);
    let ref_dir = tmp("ref");
    write_qwen_gdn_dense_install_streamed(
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
    let (hs, ho) = install_through_view(&hf, &arch, "hf");
    assert_same_install(&ref_dir, &hs, "HF-layout BF16, streamed writer");
    assert_same_install(&ref_dir, &ho, "HF-layout BF16, one-shot writer");

    for d in [ref_dir, ms, mo, hs, ho] {
        let _ = std::fs::remove_dir_all(d);
    }
}

#[test]
fn hf_layout_with_pre_shifted_norms_is_refused() {
    // HF names but MLX norm values: a second +1 would be wrong, so refuse.
    let (_, bf16, _) = checkpoints();
    let mut hf = to_hf(&bf16);
    // Restore the MLX norm values under HF names.
    let mlx_norms: std::collections::HashMap<String, Vec<u8>> = bf16
        .iter()
        .filter(|t| takes_shift(&t.name))
        .map(|t| (t.name.clone(), t.bytes.clone()))
        .collect();
    for t in hf.iter_mut() {
        let mlx = t
            .name
            .replacen("model.language_model.", "language_model.model.", 1);
        if let Some(b) = mlx_norms.get(&mlx) {
            t.bytes = b.clone();
        }
    }
    let blob = assemble_safetensors(&hf);
    let mut headers = vec![parse_header(&blob, DEFAULT_MAX_HEADER_BYTES).unwrap()];
    let layout = canonicalize_qwen35_hf_headers(&mut headers).unwrap();
    assert_eq!(layout, SourceLayout::HfTransformers);
    let src = MemoryRangeSource::new(&blob);
    let err = check_norm_layout(&headers, &[&src as &dyn RangeSource], layout).unwrap_err();
    assert!(err.contains("second +1"), "{err}");
}
