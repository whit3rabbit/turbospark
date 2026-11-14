//! Quantize-on-install for Muse Glimmer, end to end through the real Muse
//! writers (streamed and one-shot).
//!
//! The HF spelling (`meta-models/Muse-Glimmer-30B`) is `model.language_model.*`
//! with a bare `lm_head.weight` and the vision side under `model.vision_*`.
//! The four per-layer norms and the final norm are byte-identical to the MLX
//! conversion (read off both repos), so there is no shift, and the install is
//! TEXT-ONLY: the view drops the vision tensors, which the HF checkpoint
//! carries and which MUST NOT reach the walk unquantized.

use std::path::{Path, PathBuf};

use model_io::ModelFamily;

use crate::gemma4_checkpoint::{
    write_muse_glimmer_install, write_muse_glimmer_install_streamed, Gemma4Quant, Gemma4Shards,
};
use crate::quantize_on_install::{
    canonicalize_for_family, quantize_on_install_shards_for, quantize_refusal,
};
use crate::ranged_download::{MemoryRangeSource, RangeSource};
use crate::safetensors_header::{parse_header, DEFAULT_MAX_HEADER_BYTES};
use crate::synthetic_muse::tiny_muse_glimmer_arch;
use crate::synthetic_tensors::{assemble_safetensors, Tensor};

const MODEL_ID: &str = "synthetic/qoi-muse";
const HIDDEN: usize = 128;
const HEAD_DIM: usize = 64;
const NUM_HEADS: usize = 4;
const NUM_KV_HEADS: usize = 2;
const INTER: usize = 128;
const LAYERS: usize = 4;
const VOCAB: usize = 256;

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
    let q_dim = NUM_HEADS * HEAD_DIM;
    let kv_dim = NUM_KV_HEADS * HEAD_DIM;
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
    mat("language_model.lm_head.weight".into(), VOCAB, HIDDEN);
    for l in 0..LAYERS {
        let p = format!("language_model.model.layers.{l}");
        mat(format!("{p}.self_attn.q_proj.weight"), q_dim, HIDDEN);
        mat(format!("{p}.self_attn.k_proj.weight"), kv_dim, HIDDEN);
        mat(format!("{p}.self_attn.v_proj.weight"), kv_dim, HIDDEN);
        mat(format!("{p}.self_attn.gate_proj.weight"), q_dim, HIDDEN);
        mat(format!("{p}.self_attn.o_proj.weight"), HIDDEN, q_dim);
        mat(format!("{p}.mlp.gate_proj.weight"), INTER, HIDDEN);
        mat(format!("{p}.mlp.up_proj.weight"), INTER, HIDDEN);
        mat(format!("{p}.mlp.down_proj.weight"), HIDDEN, INTER);
        // CENTERED norms (near 0), stored as they are in both layouts.
        for (i, n) in [
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ]
        .iter()
        .enumerate()
        {
            let v = values(900 + i as u64 + l as u64, HIDDEN);
            raws.push(raw(&format!("{p}.{n}.weight"), vec![HIDDEN as u64], &v));
        }
    }
    let v = values(5, HIDDEN);
    raws.push(raw(
        "language_model.model.norm.weight",
        vec![HIDDEN as u64],
        &v,
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

/// The HF transformers spelling of an MLX-layout checkpoint, plus the vision
/// tensors the HF checkpoint carries and the MLX text-only reference does not.
fn to_hf(ts: &[Tensor]) -> Vec<Tensor> {
    let mut out: Vec<Tensor> = ts
        .iter()
        .map(|t| {
            let name = if let Some(r) = t.name.strip_prefix("language_model.model.") {
                format!("model.language_model.{r}")
            } else if let Some(r) = t.name.strip_prefix("language_model.lm_head.") {
                format!("lm_head.{r}")
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
        .collect();
    for (name, shape) in [
        (
            "model.vision_tower.layers.0.attn.q_proj.weight",
            vec![64u64, 64],
        ),
        ("model.vision_tower.layers.0.attn.q_proj.bias", vec![64]),
        ("model.vision_adapter.fc1.weight", vec![64, 64]),
        ("model.vision_projection.weight", vec![64, 64]),
    ] {
        let n: usize = shape.iter().product::<u64>() as usize;
        out.push(raw(name, shape, &values(77, n)));
    }
    out
}

fn quant() -> Gemma4Quant {
    Gemma4Quant {
        default_bits: 4,
        group_size: 64,
        ..Default::default()
    }
}

fn tmp(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("qoimuse_{tag}_{}", std::process::id()));
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
    let layout = canonicalize_for_family(ModelFamily::MuseGlimmer, &mut headers).unwrap();
    let src = MemoryRangeSource::new(&blob);
    let srcs: Vec<&dyn RangeSource> = vec![&src];
    let view = quantize_on_install_shards_for(ModelFamily::MuseGlimmer, &headers, &srcs, 4, layout)
        .unwrap();
    assert!(
        view[0]
            .header()
            .tensors
            .keys()
            .all(|k| !k.contains("vision")),
        "the vision tensors must not reach the walk"
    );
    let streamed = tmp(&format!("{tag}_s"));
    let shards = Gemma4Shards::new(
        view.iter()
            .map(|s| (s.header(), s as &dyn RangeSource))
            .collect(),
    )
    .unwrap();
    write_muse_glimmer_install_streamed(&streamed, arch, MODEL_ID, &shards, &quant(), |_| {})
        .unwrap();
    let oneshot = tmp(&format!("{tag}_o"));
    write_muse_glimmer_install(
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
    let arch = tiny_muse_glimmer_arch(VOCAB as i64, LAYERS as i64);
    let (bf16, reference) = checkpoints();

    let ref_blob = assemble_safetensors(&reference);
    let ref_header = parse_header(&ref_blob, DEFAULT_MAX_HEADER_BYTES).unwrap();
    let ref_src = MemoryRangeSource::new(&ref_blob);
    let ref_dir = tmp("ref");
    write_muse_glimmer_install_streamed(
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
    assert!(hf.iter().any(|t| t.name == "lm_head.weight"));
    assert!(hf
        .iter()
        .any(|t| t.name.starts_with("model.vision_adapter.")));
    let (hs, ho) = install_through_view(&hf, &arch, "hf");
    assert_same_install(&ref_dir, &hs, "HF-layout BF16, streamed writer");
    assert_same_install(&ref_dir, &ho, "HF-layout BF16, one-shot writer");

    for d in [ref_dir, ms, mo, hs, ho] {
        let _ = std::fs::remove_dir_all(d);
    }
}

#[test]
fn the_centered_norms_are_served_untouched() {
    // A view that wrongly shifted norms (the qwen35 rule) would change these.
    let (bf16, _) = checkpoints();
    let hf = to_hf(&bf16);
    let blob = assemble_safetensors(&hf);
    let mut headers = vec![parse_header(&blob, DEFAULT_MAX_HEADER_BYTES).unwrap()];
    let layout = canonicalize_for_family(ModelFamily::MuseGlimmer, &mut headers).unwrap();
    let src = MemoryRangeSource::new(&blob);
    let srcs: Vec<&dyn RangeSource> = vec![&src];
    let view = quantize_on_install_shards_for(ModelFamily::MuseGlimmer, &headers, &srcs, 4, layout)
        .unwrap();
    for name in [
        "language_model.model.layers.0.input_layernorm.weight",
        "language_model.model.layers.3.post_feedforward_layernorm.weight",
        "language_model.model.norm.weight",
    ] {
        let (a, b) = view[0].header().absolute_range(name).unwrap();
        let got = view[0].read_range(a, b).unwrap();
        let want = bf16.iter().find(|t| t.name == name).unwrap();
        assert_eq!(got, want.bytes, "{name}");
    }
}

#[test]
fn only_the_plain_muse_glimmer_model_is_admitted() {
    let cfg = |root: &str| serde_json::json!({ "model_type": root });
    assert!(quantize_refusal(ModelFamily::MuseGlimmer, &cfg("muse_glimmer")).is_none());
    // The family string also reaches a drafter or wrapper.
    assert!(
        quantize_refusal(ModelFamily::MuseGlimmer, &cfg("muse_glimmer_assistant"))
            .unwrap()
            .contains("muse_glimmer")
    );
}
