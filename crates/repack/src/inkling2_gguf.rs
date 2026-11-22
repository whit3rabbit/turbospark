//! An Inkling 2 Mini GGUF (llama.cpp master's `inkling2_loop` or the older
//! converter's `inkling2`) as a `.gturbo` install the Inkling flow reads.
//!
//! - Names and converter transforms come from `inkling2_gguf_names`.
//! - Block-quantized matrices keep their ggml bytes under their GGUF resident
//!   tag (the shared resident GEMV and embedding lookup run them).
//! - Float tensors are decoded, un-transformed (transposed `rel_logits_proj`,
//!   squeezed convs, the per-core message queries recombined), and written F32
//!   where the flow reads F32 (router, its bias, every depth-mix and message
//!   query) and BF16 everywhere else, counting values BF16 cannot hold.
//! - Experts become `packed_experts` blobs: a 16-byte outer-scale record (the
//!   file's per-expert `.scale` for gate-up and down, or 1.0), then gate rows,
//!   up rows and down rows in their ggml block type. The fused
//!   `ffn_gate_up_exps` is gate rows first, then up rows. Sub-tensor dtypes
//!   are the lowercase ggml names; the flow runs them through the GGUF pairs'
//!   Inkling mode (`gpu::encode_inkling2_gguf_phase{1,2}`).
//!
//! The config is the canonical Mini's (`model_io::inkling2_mini`), checked
//! against the GGUF's own metadata and tensor shapes. A fine-tune of the Mini
//! installs; another shape is refused.

use std::collections::{BTreeMap, BTreeSet};
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use model_io::inkling2_config::{Inkling2Config, INKLING2_CONFIG_FILE};
use serde_json::Value;

use crate::gguf_checkpoint::dtype_tag_for_ggml_type;
use crate::gguf_header::{ggml_type_block, ggml_type_name, GgufHeader};
use crate::gturbo_writer::{ExpertBlob, LayerBlobs, StreamingGturboWriter, SubTensor};
use crate::inkling2_checkpoint::Inkling2RepackError;
use crate::inkling2_gguf_names::{gguf_to_hf, GgufTransform as T};
use crate::resident_writer::{
    write_resident_weights_bin_mixed, RawTensorSpec, ResidentEntrySpec, DTYPE_BF16, DTYPE_FP32,
};

use model_io::INKLING2_GGUF_ROUTED_SCHEME;

const GGML_F32: u32 = 0;
const GGML_F16: u32 = 1;
const GGML_BF16: u32 = 30;
const GGML_MXFP4: u32 = 39;
/// Headers carry the 200k-token vocabulary; anything this large is not one.
const MAX_HEADER_BYTES: u64 = 256 << 20;

fn err<T>(msg: impl Into<String>) -> Result<T, Inkling2RepackError> {
    Err(Inkling2RepackError(msg.into()))
}

/// What an intake wrote.
#[derive(Debug, Clone, PartialEq)]
pub struct Inkling2GgufReport {
    pub arch: String,
    pub layers: usize,
    pub resident_tensors: usize,
    pub expert_stride: u64,
    /// Per layer: (gate-up type, down type).
    pub expert_types: Vec<(String, String)>,
    /// Float values BF16 could not hold exactly (narrowed by rounding).
    pub lossy_values: usize,
    pub has_expert_scales: bool,
}

struct Gguf {
    file: std::fs::File,
    header: GgufHeader,
}

impl Gguf {
    fn open(path: &Path) -> Result<Self, Inkling2RepackError> {
        let ctx = |e: String| Inkling2RepackError(format!("{}: {e}", path.display()));
        let mut file = std::fs::File::open(path).map_err(|e| ctx(e.to_string()))?;
        let len = file.metadata().map_err(|e| ctx(e.to_string()))?.len();
        let mut want = (64u64 << 20).min(len);
        loop {
            let mut lead = vec![0u8; want as usize];
            file.seek(SeekFrom::Start(0))
                .and_then(|_| file.read_exact(&mut lead))
                .map_err(|e| ctx(e.to_string()))?;
            match crate::gguf_header::parse_header(&lead, MAX_HEADER_BYTES) {
                Ok(header) => return Ok(Self { file, header }),
                Err(crate::gguf_header::GgufHeaderError::TooShort { needed })
                    if needed > want && needed <= MAX_HEADER_BYTES.min(len) =>
                {
                    want = needed
                }
                Err(e) => return Err(ctx(e.to_string())),
            }
        }
    }

    fn read(&mut self, name: &str) -> Result<Vec<u8>, Inkling2RepackError> {
        let (start, end) = self
            .header
            .absolute_range(name)
            .ok_or_else(|| Inkling2RepackError(format!("GGUF has no tensor {name}")))?
            .map_err(|e| Inkling2RepackError(e.to_string()))?;
        let mut b = vec![0u8; (end - start) as usize];
        self.file
            .seek(SeekFrom::Start(start))
            .and_then(|_| self.file.read_exact(&mut b))
            .map_err(|e| Inkling2RepackError(format!("{name}: {e}")))?;
        Ok(b)
    }

    /// A float tensor's values as F32.
    fn floats(&mut self, name: &str) -> Result<Vec<f32>, Inkling2RepackError> {
        let ty = self.header.tensors[name].ggml_type;
        let b = self.read(name)?;
        Ok(match ty {
            GGML_F32 => b
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            GGML_F16 => b
                .chunks_exact(2)
                .map(|c| compute::quant_1bit::f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect(),
            GGML_BF16 => b
                .chunks_exact(2)
                .map(|c| compute::bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect(),
            other => return err(format!("{name}: ggml type {other} is not a float type")),
        })
    }
}

fn is_float(ty: u32) -> bool {
    matches!(ty, GGML_F32 | GGML_F16 | GGML_BF16)
}

/// Install names the flow reads as F32 (`families/inkling2/state.rs`).
fn wants_f32(hf: &str) -> bool {
    hf.ends_with(".mlp.gate.weight")
        || hf.ends_with(".mlp.gate.bias")
        || hf.starts_with("model.llm.attn_res.")
        || hf.starts_with("model.llm.middle_cycle_cross_pass_")
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// BF16 bytes and the count of values that did not survive exactly.
fn bf16_bytes(v: &[f32]) -> (Vec<u8>, usize) {
    let mut lossy = 0;
    let mut out = Vec::with_capacity(v.len() * 2);
    for &x in v {
        let b = compute::f32_to_bf16(x);
        if compute::bf16_to_f32(b) != x {
            lossy += 1;
        }
        out.extend_from_slice(&b.to_le_bytes());
    }
    (out, lossy)
}

fn shape4(dims: &[usize]) -> (u32, u32, u32, u32) {
    let mut d = [0u32; 4];
    for (i, &v) in dims.iter().enumerate() {
        d[i] = v as u32;
    }
    (d[0], d[1], d[2], d[3])
}

/// Checks a metadata key when the file has it.
fn check_meta(h: &GgufHeader, arch: &str, key: &str, want: u64) -> Result<(), Inkling2RepackError> {
    match h.metadata_u64(&format!("{arch}.{key}")) {
        Some(v) if v != want => err(format!(
            "GGUF {arch}.{key} is {v}; this intake takes the Inkling 2 Mini ({want})"
        )),
        _ => Ok(()),
    }
}

/// The `general.architecture` of a local GGUF when it is an Inkling 2 one
/// (`inkling2_loop` or `inkling2`), None for any other architecture. Reads
/// only the header. For install front ends deciding which intake a file needs.
pub fn inkling2_gguf_architecture(path: &Path) -> Result<Option<String>, Inkling2RepackError> {
    let g = Gguf::open(path)?;
    Ok(g.header
        .architecture()
        .filter(|a| matches!(*a, "inkling2_loop" | "inkling2"))
        .map(str::to_string))
}

/// The `manifest.quant` section: GGUF resident slots naming every block type
/// the install's resident matrices use, raw F32 router, the Inkling GGUF
/// routed scheme.
pub fn inkling2_gguf_manifest_quant(resident_types: &BTreeSet<String>, embed_type: &str) -> Value {
    let types: Vec<&String> = resident_types.iter().collect();
    let gguf = |dominant: &str, all: Vec<&String>| {
        serde_json::json!({
            "weightBits": 0, "scheme": "gguf", "scaleType": "none", "biasType": "none",
            "groupSize": 0, "ggmlType": dominant, "ggmlTypes": all
        })
    };
    let attention = gguf(types.first().map_or("q8_0", |t| t.as_str()), types.clone());
    serde_json::json!({
        "embedding": gguf(embed_type, vec![]),
        "attention": attention,
        "router": {"weightBits": 32, "scheme": "raw", "scaleType": "none", "biasType": "none", "groupSize": 0},
        "sharedExpert": attention,
        "routedExpert": {
            "weightBits": 0, "scheme": INKLING2_GGUF_ROUTED_SCHEME, "scaleType": "none",
            "biasType": "none", "groupSize": 0
        },
    })
}

/// Writes an install from a local GGUF.
pub fn write_inkling2_gguf_install(
    gguf: &Path,
    out: &Path,
    model_id: &str,
    mut progress: impl FnMut(&str),
) -> Result<Inkling2GgufReport, Inkling2RepackError> {
    let mut g = Gguf::open(gguf)?;
    let arch_name = g.header.architecture().unwrap_or("").to_string();
    if !matches!(arch_name.as_str(), "inkling2_loop" | "inkling2") {
        return err(format!(
            "GGUF architecture {arch_name:?} is not an Inkling 2 one"
        ));
    }
    let arch = model_io::inkling2_mini();
    let side = Inkling2Config::mini();
    let (layers, hidden, ffn, experts) = (
        arch.num_layers as usize,
        arch.hidden_size as usize,
        arch.moe_intermediate_size as usize,
        arch.num_experts as usize,
    );
    for (key, want) in [
        ("block_count", layers as u64),
        ("embedding_length", hidden as u64),
        ("expert_count", experts as u64),
        ("expert_used_count", arch.top_k_experts as u64),
        ("expert_feed_forward_length", ffn as u64),
    ] {
        check_meta(&g.header, &arch_name, key, want)?;
    }
    let core_start = side.core_start as usize;
    let core = (side.core_end - side.core_start) as usize;

    let mut resident = Vec::new();
    let mut resident_types = BTreeSet::new();
    let mut embed_type = String::from("bf16");
    let mut lossy = 0usize;
    let mut qs = vec![0.0f32; 2 * core * hidden];
    let mut kvs = vec![0.0f32; 2 * core * hidden];
    let mut msg_seen = vec![false; core];
    let mut expert_tensors: BTreeMap<String, (String, T)> = BTreeMap::new();
    let names: Vec<String> = g.header.tensors.keys().cloned().collect();
    for name in &names {
        if name.starts_with("v.") || name.starts_with("mm.") {
            continue;
        }
        let Some((hf, t)) = gguf_to_hf(name, core_start) else {
            return err(format!(
                "GGUF tensor {name} is not in the Inkling 2 name table"
            ));
        };
        let info = g.header.tensors[name].clone();
        let dims: Vec<usize> = info.dims.iter().map(|&d| d as usize).collect();
        match t {
            T::ExpertGateUpSplit | T::ExpertDown | T::ExpertScale => {
                expert_tensors.insert(name.clone(), (hf, t));
                continue;
            }
            T::MessageQueries { core_index } => {
                // `[hidden, 4]` in ggml order: rows [q[2i], kv[2i], kv[2i+1], q[2i+1]].
                if core_index >= core || dims != [hidden, 4] {
                    return err(format!(
                        "{name}: message queries {dims:?} for core index {core_index}"
                    ));
                }
                let v = g.floats(name)?;
                let row = |r: usize| &v[r * hidden..(r + 1) * hidden];
                let i = core_index;
                qs[2 * i * hidden..(2 * i + 1) * hidden].copy_from_slice(row(0));
                kvs[2 * i * hidden..(2 * i + 1) * hidden].copy_from_slice(row(1));
                kvs[(2 * i + 1) * hidden..(2 * i + 2) * hidden].copy_from_slice(row(2));
                qs[(2 * i + 1) * hidden..(2 * i + 2) * hidden].copy_from_slice(row(3));
                msg_seen[i] = true;
                continue;
            }
            _ => {}
        }
        if !is_float(info.ggml_type) {
            if t != T::None || dims.len() != 2 {
                return err(format!(
                    "{name}: a {:?} tensor must be a float, not ggml type {}",
                    t, info.ggml_type
                ));
            }
            if info.ggml_type == GGML_MXFP4 {
                return err(format!(
                    "{name}: MXFP4 resident matrices are not read by the Inkling flow"
                ));
            }
            let tag = dtype_tag_for_ggml_type(info.ggml_type).ok_or_else(|| {
                Inkling2RepackError(format!(
                    "{name}: ggml type {} has no resident tag",
                    info.ggml_type
                ))
            })?;
            let tn = ggml_type_name(info.ggml_type).unwrap_or("?").to_lowercase();
            if hf == "model.llm.embed.weight" {
                embed_type = tn.clone();
            } else {
                resident_types.insert(tn);
            }
            resident.push(ResidentEntrySpec::Raw(RawTensorSpec {
                name: hf,
                dtype: tag,
                bytes: g.read(name)?,
                // ggml dims are fastest first; the index stores rows first.
                shape: shape4(&dims.iter().rev().copied().collect::<Vec<_>>()),
            }));
            continue;
        }
        let mut v = g.floats(name)?;
        // Logical (install) shape, rows first.
        let mut shape: Vec<usize> = dims.iter().rev().copied().collect();
        match t {
            T::Transposed => {
                // Stored `[d_rel, extent]` in ggml order = logical `[extent, d_rel]`;
                // the install's `rel_logits_proj.proj` is `[d_rel, extent]`.
                let (extent, d_rel) = (shape[0], shape[1]);
                let mut tr = vec![0.0f32; v.len()];
                for e in 0..extent {
                    for d in 0..d_rel {
                        tr[d * extent + e] = v[e * d_rel + d];
                    }
                }
                v = tr;
                shape = vec![d_rel, extent];
            }
            T::SqueezedConv => {
                // `[ch, taps]` back to the install's `[ch, 1, taps]`; the bytes
                // are already in that order.
                if shape.len() != 2 || shape[1] != arch.short_conv_kernel_size as usize {
                    return err(format!("{name}: conv shape {shape:?}"));
                }
                shape = vec![shape[0], 1, shape[1]];
            }
            _ => {}
        }
        let (dtype, bytes) = if wants_f32(&hf) {
            (DTYPE_FP32, f32_bytes(&v))
        } else {
            let (b, l) = bf16_bytes(&v);
            lossy += l;
            (DTYPE_BF16, b)
        };
        resident.push(ResidentEntrySpec::Raw(RawTensorSpec {
            name: hf,
            dtype,
            bytes,
            shape: shape4(&shape),
        }));
    }
    // The message queries: per-core tensors (master) or the two global ones.
    let global_qs = resident.iter().any(|e| matches!(e, ResidentEntrySpec::Raw(r) if r.name == "model.llm.middle_cycle_cross_pass_queries"));
    if !global_qs {
        if let Some(i) = msg_seen.iter().position(|s| !s) {
            return err(format!(
                "GGUF has no message queries for core layer {}",
                core_start + i
            ));
        }
        for (n, v) in [
            ("model.llm.middle_cycle_cross_pass_queries", &qs),
            ("model.llm.middle_cycle_cross_pass_kv_queries", &kvs),
        ] {
            resident.push(ResidentEntrySpec::Raw(RawTensorSpec {
                name: n.into(),
                dtype: DTYPE_FP32,
                bytes: f32_bytes(v),
                shape: shape4(&[2 * core, hidden]),
            }));
        }
    } else if msg_seen.iter().any(|&s| s) {
        return err("GGUF has both conventions' message queries");
    }

    // Experts: per layer, gate-up and down types, the optional scales.
    let gguf_name = |l: usize, s: &str| format!("blk.{l}.{s}");
    let mut plans = Vec::with_capacity(layers);
    let mut has_scales = false;
    for l in 0..layers {
        let gu = gguf_name(l, "ffn_gate_up_exps.weight");
        let dn = gguf_name(l, "ffn_down_exps.weight");
        let gi = g
            .header
            .tensors
            .get(&gu)
            .ok_or_else(|| Inkling2RepackError(format!("GGUF has no {gu}")))?
            .clone();
        let di = g
            .header
            .tensors
            .get(&dn)
            .ok_or_else(|| Inkling2RepackError(format!("GGUF has no {dn}")))?
            .clone();
        if gi.dims != [hidden as u64, 2 * ffn as u64, experts as u64]
            || di.dims != [ffn as u64, hidden as u64, experts as u64]
        {
            return err(format!(
                "layer {l}: expert dims {:?} / {:?}",
                gi.dims, di.dims
            ));
        }
        for (n, i) in [(&gu, &gi), (&dn, &di)] {
            if is_float(i.ggml_type)
                || i.ggml_type == GGML_MXFP4
                || dtype_tag_for_ggml_type(i.ggml_type).is_none()
            {
                return err(format!(
                    "{n}: ggml type {} ({}) has no Inkling expert pair",
                    i.ggml_type,
                    ggml_type_name(i.ggml_type).unwrap_or("?")
                ));
            }
        }
        let rb = |ty: u32, n: usize| -> Result<usize, Inkling2RepackError> {
            let (elems, bytes) = ggml_type_block(ty)
                .ok_or_else(|| Inkling2RepackError(format!("ggml type {ty} has no block size")))?;
            if n as u64 % elems != 0 {
                return err(format!(
                    "layer {l}: width {n} is not a whole number of {elems}-element blocks"
                ));
            }
            Ok((n as u64 / elems * bytes) as usize)
        };
        let (gu_rb, dn_rb) = (rb(gi.ggml_type, hidden)?, rb(di.ggml_type, ffn)?);
        let scales = [
            gguf_name(l, "ffn_gate_up_exps.scale"),
            gguf_name(l, "ffn_down_exps.scale"),
        ];
        let present = scales
            .iter()
            .filter(|s| g.header.tensors.contains_key(*s))
            .count();
        if present == 1 {
            return err(format!(
                "layer {l}: only one of the two expert scale tensors"
            ));
        }
        has_scales |= present == 2;
        plans.push((
            gu,
            dn,
            gi.ggml_type,
            di.ggml_type,
            gu_rb,
            dn_rb,
            (present == 2).then_some(scales),
        ));
    }
    let used: BTreeSet<&String> = plans.iter().flat_map(|p| [&p.0, &p.1]).collect();
    for (name, (_, t)) in &expert_tensors {
        if *t != T::ExpertScale && !used.contains(name) {
            return err(format!("{name}: an expert tensor outside the layer plan"));
        }
    }
    let page = crate::GTURBO_PAGE_BYTES;
    let blob = |gu_rb: usize, dn_rb: usize| (16 + 2 * ffn * gu_rb + hidden * dn_rb) as u64;
    let stride = plans
        .iter()
        .map(|p| blob(p.4, p.5).div_ceil(page) * page)
        .max()
        .unwrap_or(0);

    let io = |e: crate::gturbo_writer::WriterError| Inkling2RepackError(e.to_string());
    let mut writer = StreamingGturboWriter::new(out, stride, experts).map_err(io)?;
    writer.set_quant(inkling2_gguf_manifest_quant(&resident_types, &embed_type));
    let mut expert_types = Vec::with_capacity(layers);
    for (l, (gu, dn, gt, dt, gu_rb, dn_rb, scales)) in plans.into_iter().enumerate() {
        let gub = g.read(&gu)?;
        let dnb = g.read(&dn)?;
        let (s13, s2) = match &scales {
            Some([a, b]) => {
                let (a, b) = (g.floats(a)?, g.floats(b)?);
                if a.len() != experts
                    || b.len() != experts
                    || a.iter().chain(&b).any(|v| !v.is_finite())
                {
                    return err(format!("layer {l}: expert scales malformed"));
                }
                (a, b)
            }
            None => (vec![1.0; experts], vec![1.0; experts]),
        };
        let (gn, dn_name) = (
            ggml_type_name(gt).unwrap().to_lowercase(),
            ggml_type_name(dt).unwrap().to_lowercase(),
        );
        let per = 2 * ffn * gu_rb;
        let blobs = (0..experts)
            .map(|e| {
                let w13 = &gub[e * per..(e + 1) * per];
                let mut outer = Vec::with_capacity(16);
                for v in [s13[e], s13[e], s2[e], 0.0f32] {
                    outer.extend_from_slice(&v.to_le_bytes());
                }
                let sub = |role: &str, bytes: Vec<u8>, dtype: &str, shape: Vec<u64>| SubTensor {
                    role: role.into(),
                    bytes,
                    dtype: dtype.into(),
                    shape,
                };
                ExpertBlob {
                    expert: e,
                    sub_tensors: vec![
                        sub("outer", outer, "F32", vec![4]),
                        sub(
                            "gate",
                            w13[..ffn * gu_rb].to_vec(),
                            &gn,
                            vec![ffn as u64, hidden as u64],
                        ),
                        sub(
                            "up",
                            w13[ffn * gu_rb..].to_vec(),
                            &gn,
                            vec![ffn as u64, hidden as u64],
                        ),
                        sub(
                            "down",
                            dnb[e * hidden * dn_rb..(e + 1) * hidden * dn_rb].to_vec(),
                            &dn_name,
                            vec![hidden as u64, ffn as u64],
                        ),
                    ],
                }
            })
            .collect();
        writer
            .write_layer(&LayerBlobs {
                layer: l,
                experts: blobs,
            })
            .map_err(io)?;
        progress(&format!("layer {l:2}: {gn} gate-up, {dn_name} down"));
        expert_types.push((gn, dn_name));
    }
    let side_path = out.join(INKLING2_CONFIG_FILE);
    std::fs::write(
        &side_path,
        serde_json::to_vec_pretty(&side).expect("sidecar serializes"),
    )
    .map_err(|e| Inkling2RepackError(format!("{}: {e}", side_path.display())))?;
    let n_resident = resident.len();
    writer
        .finish_streaming(&arch, model_id, |w| {
            write_resident_weights_bin_mixed(&resident, w)
        })
        .map_err(io)?;
    Ok(Inkling2GgufReport {
        arch: arch_name,
        layers,
        resident_tensors: n_resident,
        expert_stride: stride,
        expert_types,
        lossy_values: lossy,
        has_expert_scales: has_scales,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32_names_are_the_ones_the_flow_reads_as_f32() {
        for n in [
            "model.llm.layers.3.mlp.gate.weight",
            "model.llm.layers.3.mlp.gate.bias",
            "model.llm.attn_res.attn_queries.4",
            "model.llm.attn_res.output_query",
            "model.llm.middle_cycle_cross_pass_kv_queries",
        ] {
            assert!(wants_f32(n), "{n}");
        }
        for n in [
            "model.llm.layers.3.attn.q_l2_clip.scale",
            "model.llm.layers.3.attn_norm.weight",
            "model.llm.layers.3.attn.k_sconv.weight",
            "model.llm.layers.6.controllers.v.gain_weight",
            "model.llm.embed_norm.weight",
        ] {
            assert!(!wants_f32(n), "{n} is BF16 in the flow");
        }
    }

    #[test]
    fn bf16_narrowing_counts_values_it_cannot_hold() {
        let (b, lossy) = bf16_bytes(&[1.0, 0.5, -2.0, 1.0 + 1.0 / 1024.0]);
        assert_eq!(lossy, 1, "only 1 + 2^-10 needs more than 8 mantissa bits");
        assert_eq!(b.len(), 8);
        assert_eq!(bf16_bytes(&[0.0, 3.0]).1, 0);
    }

    #[test]
    fn non_inkling_architectures_are_refused() {
        let dir = std::env::temp_dir().join(format!("turbospark-ink-gguf-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("x.gguf");
        // A valid empty GGUF v3 header: no tensors, one KV naming another arch.
        let mut b = b"GGUF".to_vec();
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes());
        b.extend_from_slice(&1u64.to_le_bytes());
        let key = b"general.architecture";
        b.extend_from_slice(&(key.len() as u64).to_le_bytes());
        b.extend_from_slice(key);
        b.extend_from_slice(&8u32.to_le_bytes());
        let v = b"llama";
        b.extend_from_slice(&(v.len() as u64).to_le_bytes());
        b.extend_from_slice(v);
        std::fs::write(&p, &b).unwrap();
        let e = write_inkling2_gguf_install(&p, &dir.join("out"), "t", |_| {}).unwrap_err();
        assert!(e.0.contains("not an Inkling 2 one"), "{}", e.0);
    }
}
