//! Quantize-on-install for the Qwen MoE, end to end through the REAL MoE
//! writers (streamed and one-shot).
//!
//! Same shape as `quantize_on_install_tests`: one set of weights, an
//! MLX-converted reference and an HF-layout BF16 checkpoint with FUSED routed
//! experts (`experts.gate_up_proj`, `experts.down_proj`, the layout
//! `Qwen/Qwen3.6-35B-A3B` publishes), and the installs must be byte-identical.
//!
//! The reference quantizes the way `mlx_lm.convert` does for this family, read
//! off the published `mlx-community/Qwen3.6-35B-A3B-4bit` config: 8-bit for
//! `mlp.gate` and `mlp.shared_expert_gate`, 4-bit for everything else
//! (including the shared-expert projections, which the older fixture keeps at
//! 8-bit and which therefore must not be reused as the reference).

use std::path::{Path, PathBuf};

use crate::gemma4_checkpoint::{
    write_qwen_gdn_moe_install, write_qwen_gdn_moe_install_streamed, Gemma4Quant, Gemma4Shards,
};
use crate::quantize_on_install::{
    canonicalize_for_family, check_norm_layout, quantize_on_install_shards_for,
};
use crate::ranged_download::{MemoryRangeSource, RangeSource};
use crate::safetensors_header::{parse_header, DEFAULT_MAX_HEADER_BYTES};
use crate::synthetic_real::assemble_safetensors;
use crate::synthetic_tensors::Tensor;
use model_io::ModelFamily;

use super::moe::{tiny_qwen_gdn_moe_arch, trunk_tensors};

const MODEL_ID: &str = "synthetic/qoi-moe";

fn u16s(b: &[u8]) -> Vec<u16> {
    b.chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

fn bits_for(name: &str) -> u32 {
    if name.ends_with(".mlp.gate.weight") || name.ends_with(".mlp.shared_expert_gate.weight") {
        8
    } else {
        4
    }
}

fn quantize_row(row: &[f32], bits: u32) -> (Vec<u8>, Vec<u16>, Vec<u16>) {
    match bits {
        4 => {
            let q = compute::quant_mlx::quantize_int4_affine_mlx(row);
            (q.packed, q.scales, q.biases)
        }
        8 => {
            let q = compute::quant_mlx::quantize_int8_affine_mlx(row);
            (q.packed, q.scales, q.biases)
        }
        _ => unreachable!(),
    }
}

/// Dequantize a fixture triple back to BF16-representable floats, rows of
/// `cols`. Works for rank-2 and rank-3 weights (`rows` is the product of
/// every leading axis).
fn dequantize(w: &Tensor, sc: &Tensor, bi: &Tensor) -> (Vec<f32>, usize, usize) {
    let last = *w.shape.last().unwrap() as usize;
    let g = *sc.shape.last().unwrap() as usize;
    let cols = g * 64;
    let bits = last * 32 / cols;
    let rows: usize = w.shape[..w.shape.len() - 1].iter().product::<u64>() as usize;
    let (sc, bi) = (u16s(&sc.bytes), u16s(&bi.bytes));
    let row_bytes = cols * bits / 8;
    let mut out = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        let packed = w.bytes[r * row_bytes..(r + 1) * row_bytes].to_vec();
        let (s, b) = (
            sc[r * g..(r + 1) * g].to_vec(),
            bi[r * g..(r + 1) * g].to_vec(),
        );
        let vals = match bits {
            4 => compute::quant::dequantize_int4_affine(
                &compute::quant::Int4AffineRow {
                    packed,
                    scales: s,
                    biases: b,
                },
                cols,
            ),
            8 => compute::quant::dequantize_int8_affine(
                &compute::quant::Int8AffineRow {
                    packed,
                    scales: s,
                    biases: b,
                },
                cols,
            ),
            _ => unreachable!(),
        };
        out.extend(
            vals.into_iter()
                .map(|v| compute::bf16_to_f32(compute::f32_to_bf16(v))),
        );
    }
    (out, rows, cols)
}

fn bf16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|&x| compute::f32_to_bf16(x).to_le_bytes())
        .collect()
}

/// `(arch, bf16 MLX-layout checkpoint, mx.quantize reference, overrides)`.
fn checkpoints() -> (
    model_io::ArchConfig,
    Vec<Tensor>,
    Vec<Tensor>,
    std::collections::HashMap<String, u32>,
) {
    let arch = tiny_qwen_gdn_moe_arch(256, 2, 4);
    let (packed, _) = trunk_tensors(&arch, false);
    let by_name: std::collections::HashMap<&str, &Tensor> =
        packed.iter().map(|t| (t.name.as_str(), t)).collect();
    let mut bf16 = Vec::new();
    let mut reference = Vec::new();
    let mut overrides = std::collections::HashMap::new();
    for t in &packed {
        if t.name.ends_with(".scales") || t.name.ends_with(".biases") {
            continue;
        }
        if t.dtype != "U32" {
            for dst in [&mut bf16, &mut reference] {
                dst.push(Tensor {
                    name: t.name.clone(),
                    dtype: t.dtype,
                    shape: t.shape.clone(),
                    bytes: t.bytes.clone(),
                });
            }
            continue;
        }
        let base = t.name.strip_suffix(".weight").unwrap();
        let (vals, rows, cols) = dequantize(
            t,
            by_name[format!("{base}.scales").as_str()],
            by_name[format!("{base}.biases").as_str()],
        );
        let bits = bits_for(&t.name);
        if bits == 8 {
            overrides.insert(base.to_string(), 8);
        }
        let (mut qw, mut qs, mut qb) = (Vec::new(), Vec::new(), Vec::new());
        for r in 0..rows {
            let (p, s, b) = quantize_row(&vals[r * cols..(r + 1) * cols], bits);
            qw.extend(p);
            qs.extend(s.iter().flat_map(|v| v.to_le_bytes()));
            qb.extend(b.iter().flat_map(|v| v.to_le_bytes()));
        }
        // Keep the source's leading axes (rank 3 for the routed experts).
        let lead = &t.shape[..t.shape.len() - 1];
        let shape_of = |last: usize| {
            let mut s = lead.to_vec();
            s.push(last as u64);
            s
        };
        bf16.push(Tensor {
            name: t.name.clone(),
            dtype: "BF16",
            shape: shape_of(cols),
            bytes: bf16_bytes(&vals),
        });
        reference.push(Tensor {
            name: t.name.clone(),
            dtype: "U32",
            shape: shape_of(cols * bits as usize / 32),
            bytes: qw,
        });
        for (suffix, bytes) in [("scales", qs), ("biases", qb)] {
            reference.push(Tensor {
                name: format!("{base}.{suffix}"),
                dtype: "BF16",
                shape: shape_of(cols / 64),
                bytes,
            });
        }
    }
    (arch, bf16, reference, overrides)
}

fn takes_shift(name: &str) -> bool {
    name.ends_with(".input_layernorm.weight")
        || name.ends_with(".post_attention_layernorm.weight")
        || name.ends_with(".self_attn.q_norm.weight")
        || name.ends_with(".self_attn.k_norm.weight")
        || name == "language_model.model.norm.weight"
}

/// The HF transformers spelling: renamed trunk, zero-centred norms (BF16,
/// `v - 1`), conv1d `[C, 1, K]`, and the routed experts FUSED
/// (`experts.gate_up_proj [E, 2I, H]`, `experts.down_proj [E, H, I]`).
fn to_hf(ts: &[Tensor]) -> Vec<Tensor> {
    let mut out = Vec::new();
    let find = |n: &str| ts.iter().find(|t| t.name == n).unwrap();
    for t in ts {
        let rename = |n: &str| {
            if let Some(tail) = n.strip_prefix("language_model.model.") {
                format!("model.language_model.{tail}")
            } else if let Some(tail) = n.strip_prefix("language_model.lm_head.") {
                format!("lm_head.{tail}")
            } else {
                n.to_string()
            }
        };
        if let Some(p) = t.name.strip_suffix(".mlp.switch_mlp.gate_proj.weight") {
            // Fuse gate then up per expert.
            let up = find(&format!("{p}.mlp.switch_mlp.up_proj.weight"));
            let (e, i, h) = (
                t.shape[0] as usize,
                t.shape[1] as usize,
                t.shape[2] as usize,
            );
            let row = h * 2;
            let mut bytes = Vec::with_capacity(t.bytes.len() * 2);
            for x in 0..e {
                bytes.extend_from_slice(&t.bytes[x * i * row..(x + 1) * i * row]);
                bytes.extend_from_slice(&up.bytes[x * i * row..(x + 1) * i * row]);
            }
            out.push(Tensor {
                name: rename(&format!("{p}.mlp.experts.gate_up_proj")),
                dtype: "BF16",
                shape: vec![e as u64, 2 * i as u64, h as u64],
                bytes,
            });
            continue;
        }
        if t.name.ends_with(".mlp.switch_mlp.up_proj.weight") {
            continue;
        }
        if let Some(p) = t.name.strip_suffix(".mlp.switch_mlp.down_proj.weight") {
            out.push(Tensor {
                name: rename(&format!("{p}.mlp.experts.down_proj")),
                dtype: "BF16",
                shape: t.shape.clone(),
                bytes: t.bytes.clone(),
            });
            continue;
        }
        let bytes = if takes_shift(&t.name) {
            assert_eq!(t.dtype, "BF16", "{}", t.name);
            // The view's +1 in BF16 must land back on v, so pick values
            // where v - 1 is exact in BF16: the fixture's norms are 1.0 plus
            // small noise, and the roundtrip is asserted below by the install
            // comparison itself.
            u16s(&t.bytes)
                .into_iter()
                .flat_map(|b| {
                    let v = compute::bf16_to_f32(b) - 1.0;
                    compute::f32_to_bf16(v).to_le_bytes()
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
        out.push(Tensor {
            name: rename(&t.name),
            dtype: t.dtype,
            shape,
            bytes,
        });
    }
    out
}

fn quant(overrides: &std::collections::HashMap<String, u32>) -> Gemma4Quant {
    Gemma4Quant {
        default_bits: 4,
        group_size: 64,
        bits_overrides: overrides.clone(),
        ..Default::default()
    }
}

fn tmp(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("qoim_{tag}_{}", std::process::id()));
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

/// Install `ts` through the quantized view with BOTH writers, taking the
/// router/shared-gate widths from the VIEW, as `stream.rs` does.
fn install_through_view(
    ts: &[Tensor],
    arch: &model_io::ArchConfig,
    tag: &str,
) -> (PathBuf, PathBuf) {
    let blob = assemble_safetensors(ts);
    let mut headers = vec![parse_header(&blob, DEFAULT_MAX_HEADER_BYTES).unwrap()];
    let layout = canonicalize_for_family(ModelFamily::QwenGdnMoe, &mut headers).unwrap();
    let src = MemoryRangeSource::new(&blob);
    let srcs: Vec<&dyn RangeSource> = vec![&src];
    check_norm_layout(&headers, &srcs, layout).unwrap();
    let view = quantize_on_install_shards_for(ModelFamily::QwenGdnMoe, &headers, &srcs, 4, layout)
        .unwrap();
    let mut q = quant(&Default::default());
    for s in &view {
        for (base, b) in s.bits_overrides() {
            q.bits_overrides.insert(base.clone(), *b);
        }
    }
    let streamed = tmp(&format!("{tag}_s"));
    let shards = Gemma4Shards::new(
        view.iter()
            .map(|s| (s.header(), s as &dyn RangeSource))
            .collect(),
    )
    .unwrap();
    write_qwen_gdn_moe_install_streamed(&streamed, arch, MODEL_ID, &shards, &q, |_| {}).unwrap();
    let oneshot = tmp(&format!("{tag}_o"));
    write_qwen_gdn_moe_install(&oneshot, arch, MODEL_ID, view[0].header(), &view[0], &q).unwrap();
    (streamed, oneshot)
}

#[test]
fn hf_bf16_with_fused_experts_installs_byte_identical_to_its_mlx_conversion() {
    let (arch, bf16, reference, overrides) = checkpoints();
    assert!(
        !overrides.is_empty(),
        "the fixture must exercise INT8 gates"
    );

    let ref_blob = assemble_safetensors(&reference);
    let ref_header = parse_header(&ref_blob, DEFAULT_MAX_HEADER_BYTES).unwrap();
    let ref_src = MemoryRangeSource::new(&ref_blob);
    let ref_dir = tmp("ref");
    write_qwen_gdn_moe_install_streamed(
        &ref_dir,
        &arch,
        MODEL_ID,
        &Gemma4Shards::single(&ref_header, &ref_src),
        &quant(&overrides),
        |_| {},
    )
    .unwrap();

    let hf = to_hf(&bf16);
    assert!(hf
        .iter()
        .any(|t| t.name.ends_with(".mlp.experts.gate_up_proj") && t.shape.len() == 3));
    let (hs, ho) = install_through_view(&hf, &arch, "hf");
    assert_same_install(&ref_dir, &hs, "HF fused-expert BF16, streamed writer");
    assert_same_install(&ref_dir, &ho, "HF fused-expert BF16, one-shot writer");

    for d in [ref_dir, hs, ho] {
        let _ = std::fs::remove_dir_all(d);
    }
}

#[test]
fn only_four_bit_is_admitted_for_the_moe() {
    use crate::quantize_on_install::family_supports_bits;
    assert!(family_supports_bits(ModelFamily::QwenGdnMoe, 4));
    for bits in [2, 8] {
        assert!(!family_supports_bits(ModelFamily::QwenGdnMoe, bits));
    }
    assert!(family_supports_bits(ModelFamily::QwenGdnDense, 8));
}

#[test]
fn a_per_expert_checkpoint_is_refused_not_guessed() {
    let (_, bf16, _, _) = checkpoints();
    let mut hf = to_hf(&bf16);
    for t in hf.iter_mut() {
        if t.name.ends_with(".mlp.experts.down_proj") {
            t.name = t
                .name
                .replace(".experts.down_proj", ".experts.0.down_proj.weight");
            t.shape = t.shape[1..].to_vec();
            t.bytes
                .truncate(t.shape.iter().product::<u64>() as usize * 2);
        }
    }
    let blob = assemble_safetensors(&hf);
    let mut headers = vec![parse_header(&blob, DEFAULT_MAX_HEADER_BYTES).unwrap()];
    let layout = canonicalize_for_family(ModelFamily::QwenGdnMoe, &mut headers).unwrap();
    let src = MemoryRangeSource::new(&blob);
    let srcs: Vec<&dyn RangeSource> = vec![&src];
    let err =
        match quantize_on_install_shards_for(ModelFamily::QwenGdnMoe, &headers, &srcs, 4, layout) {
            Err(e) => e,
            Ok(_) => panic!("per-expert layout must be refused"),
        };
    assert!(err.contains("per-expert"), "{err}");
}
