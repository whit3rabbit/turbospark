//! A LookAtMyQuant packed release of Inkling 2 Mini (`inkling2-packed`,
//! `shinmegamiboson/Inkling-2-Mini-LookAtMyQuant-*`) as a `.gturbo` install
//! with the codes kept native. Spec and the checks behind every rule here:
//! `docs/LAMQ.md`.
//!
//! What the install holds:
//! - Coded attention linears, embedding buckets and head buckets as
//!   `DTYPE_LAMQ` (36) resident entries: payload = code stream, scales = f16
//!   per row, aux = `compute::lamq` header + signs (+ levels). The embedding
//!   and head keep their HF names as bucket-order entries (i32 token ids per
//!   packed position), with buckets at `<name>.b<kb>`.
//! - The shared trellis LUTs as raw FP16 entries `lamq.lut.<code>`.
//! - Each layer's expert signs as a `DTYPE_LAMQ` signs entry
//!   `model.llm.layers.<l>.mlp.experts.lamq_signs` (gate-up 2304, down 1152).
//! - The 499 steering tensors under their HF install names, in their release
//!   dtypes. Their F32 values are not exact BF16, so nothing is narrowed.
//! - Experts as `packed_experts` blobs: a 16-byte format record, then
//!   gate-up codes, gate-up scales, down codes, down scales. Code sub-tensors
//!   are padded to the layer's widest format so every expert of a layer has
//!   the same sub-offsets (the streamer reads `e * stride`). Gate is rows
//!   `[0, ffn)` of gate-up and up is `[ffn, 2 ffn)`, NOT the HF interleave.

use std::path::Path;

use compute::lamq::{encode_matrix_aux, LamqAuxHeader, LamqCode, LamqKind, LamqLinear, HASH_MULT};
use model_io::{SafetensorsFile, INKLING2_LAMQ_ROUTED_SCHEME, LAMQ_SCHEME};
use serde_json::Value;

use crate::gturbo_writer::{ExpertBlob, LayerBlobs, StreamingGturboWriter, SubTensor};
use crate::inkling2_checkpoint::{parse_inkling2_config, rewrite_static_w, Inkling2RepackError};
use crate::npz::{NpyDtype, NpzArchive};
use crate::resident_writer::{
    write_resident_weights_bin_mixed, RawTensorSpec, ResidentEntrySpec, ResidentTensorSpec,
    DTYPE_BF16, DTYPE_FP16, DTYPE_FP32,
};

/// `manifest.json` `format` of the release this intake reads.
pub const LAMQ_INKLING2_FORMAT: &str = "inkling2-packed-v1-sqyfr";
/// Sub-tensor dtype of the per-expert format record.
pub const INKLING2_LAMQ_FMT_DTYPE: &str = "inkling2_lamq_fmt";
/// Sub-tensor dtype of a LAMQ expert code stream.
pub const INKLING2_LAMQ_DTYPE: &str = "inkling2_lamq";
/// Bytes of the per-expert format record: gate-up code id, down code id,
/// gate-up row bytes, down row bytes (u32 each).
pub const INKLING2_LAMQ_FMT_BYTES: usize = 16;

/// Metadata files are small; anything larger is not this format.
const MAX_JSON_BYTES: u64 = 1 << 20;

/// The six coded linears of a layer: LAMQ meta key, HF install suffix.
const LINEARS: [(&str, &str); 6] = [
    ("self_attn.q_proj.linear", "attn.wq_du.weight"),
    ("self_attn.k_proj.linear", "attn.wk_dv.weight"),
    ("self_attn.v_proj.linear", "attn.wv_dv.weight"),
    (
        "self_attn.relative_position_bias.linear",
        "attn.wr_du.weight",
    ),
    ("self_attn.o_proj.gate_proj", "attn.gate_attn.weight"),
    ("self_attn.o_proj.linear", "attn.wo_ud.weight"),
];

fn err<T>(msg: impl Into<String>) -> Result<T, Inkling2RepackError> {
    Err(Inkling2RepackError(msg.into()))
}

fn npz_err(e: crate::npz::NpzError) -> Inkling2RepackError {
    Inkling2RepackError(e.0)
}

fn read_json(path: &Path) -> Result<Value, Inkling2RepackError> {
    let ctx = |e: String| Inkling2RepackError(format!("{}: {e}", path.display()));
    let len = std::fs::metadata(path)
        .map_err(|e| ctx(e.to_string()))?
        .len();
    if len > MAX_JSON_BYTES {
        return Err(ctx(format!("{len} bytes exceeds {MAX_JSON_BYTES}")));
    }
    let text = std::fs::read_to_string(path).map_err(|e| ctx(e.to_string()))?;
    serde_json::from_str(&text).map_err(|e| ctx(e.to_string()))
}

/// The install name of one steering tensor (`docs/LAMQ.md`, name map).
/// The six LAMQ-only renames come first, then mlx-lm's names, which
/// `inkling2_mlx::mlx_to_hf` already inverts. None for the merged message
/// queries, which the caller splits.
pub fn lamq_steering_to_hf(name: &str) -> Option<String> {
    if name == "model.message_input.query" {
        return None;
    }
    let mut n = format!("model.language_model.{}", name.strip_prefix("model.")?);
    for (from, to) in [
        (".self_attn.q_proj.norm.scale", ".self_attn.q_l2_clip.scale"),
        (
            ".self_attn.relative_position_bias.norm.scale",
            ".self_attn.rel_l2_clip.scale",
        ),
        (
            ".self_attn.relative_position_bias.proj",
            ".self_attn.rel_logits_proj.proj",
        ),
        (".self_attn.k_proj.sconv.", ".self_attn.k_sconv."),
        (".self_attn.v_proj.sconv.", ".self_attn.v_sconv."),
        ("embed_tokens.embed_norm.", "embed_norm."),
    ] {
        n = n.replace(from, to);
    }
    // `model.language_model.embed_norm.*` is not an mlx-lm name; it maps to
    // the HF `model.llm.embed_norm.*` directly.
    if let Some(rest) = n.strip_prefix("model.language_model.embed_norm.") {
        return Some(format!("model.llm.embed_norm.{rest}"));
    }
    let hf = crate::inkling2_mlx::mlx_to_hf(&n)?;
    // The install keeps a full-attention `static_w` rewritten as the conv
    // kernel's `_sconv.weight` (`rewrite_static_w`), so that is its name.
    Some(match hf.strip_suffix("_sconv.static_w") {
        Some(stem) => format!("{stem}_sconv.weight"),
        None => hf,
    })
}

fn dtype_tag(name: &str, dt: &str) -> Result<u8, Inkling2RepackError> {
    match dt {
        "BF16" => Ok(DTYPE_BF16),
        "F32" => Ok(DTYPE_FP32),
        other => err(format!(
            "{name}: steering dtype {other} is neither BF16 nor F32"
        )),
    }
}

fn rank4(shape: &[usize]) -> Result<(u32, u32, u32, u32), Inkling2RepackError> {
    if shape.len() > 4 {
        return err(format!("rank {} exceeds 4", shape.len()));
    }
    let mut d = [0u32; 4];
    for (i, &v) in shape.iter().enumerate() {
        d[i] = u32::try_from(v).map_err(|_| Inkling2RepackError(format!("dim {v} too large")))?;
    }
    Ok((d[0], d[1], d[2], d[3]))
}

fn u16s(b: &[u8]) -> Vec<u16> {
    b.chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

/// Steering tensors as raw resident entries under HF names.
fn steering_entries(
    st: &SafetensorsFile,
    core: usize,
    hidden: usize,
    taps: usize,
) -> Result<Vec<ResidentEntrySpec>, Inkling2RepackError> {
    let mut out = Vec::new();
    let mut names: Vec<String> = st.tensor_names().map(str::to_string).collect();
    names.sort_unstable();
    for name in &names {
        let d = st.descriptor(name).expect("listed");
        let b = st
            .raw_bytes(name)
            .map_err(|e| Inkling2RepackError(format!("{name}: {e}")))?;
        if name == "model.message_input.query" {
            // `[core, 4, hidden]` rows `[q0, kv0, kv1, q1]`, split back as the
            // MLX intake does.
            if d.dtype != "F32" || d.shape != [core, 4, hidden] {
                return err(format!(
                    "{name}: expected F32 [{core}, 4, {hidden}], got {} {:?}",
                    d.dtype, d.shape
                ));
            }
            let row =
                |i: usize, j: usize| &b[(i * 4 + j) * hidden * 4..(i * 4 + j + 1) * hidden * 4];
            let (mut qs, mut kvs) = (Vec::new(), Vec::new());
            for i in 0..core {
                qs.extend_from_slice(row(i, 0));
                qs.extend_from_slice(row(i, 3));
                kvs.extend_from_slice(row(i, 1));
                kvs.extend_from_slice(row(i, 2));
            }
            for (hf, bytes) in [
                ("model.llm.middle_cycle_cross_pass_queries", qs),
                ("model.llm.middle_cycle_cross_pass_kv_queries", kvs),
            ] {
                out.push(ResidentEntrySpec::Raw(RawTensorSpec {
                    name: hf.into(),
                    dtype: DTYPE_FP32,
                    bytes,
                    shape: rank4(&[2 * core, hidden])?,
                }));
            }
            continue;
        }
        let hf = lamq_steering_to_hf(name)
            .ok_or_else(|| Inkling2RepackError(format!("steering tensor {name} has no HF name")))?;
        if name.ends_with("sconv.static_weight") {
            if d.dtype != "BF16" || d.shape.len() != 2 || d.shape[0] != taps {
                return err(format!(
                    "{name}: expected BF16 [{taps}, ch], got {} {:?}",
                    d.dtype, d.shape
                ));
            }
            let ch = d.shape[1];
            out.push(ResidentEntrySpec::Raw(RawTensorSpec {
                name: hf,
                dtype: DTYPE_BF16,
                bytes: rewrite_static_w(b, taps, ch),
                shape: rank4(&[ch, 1, taps])?,
            }));
            continue;
        }
        out.push(ResidentEntrySpec::Raw(RawTensorSpec {
            name: hf,
            dtype: dtype_tag(name, &d.dtype)?,
            bytes: b.to_vec(),
            shape: rank4(&d.shape)?,
        }));
    }
    Ok(out)
}

/// The shared trellis LUTs, as f32 for validation and as raw FP16 entries.
struct Luts {
    tables: Vec<(LamqCode, Vec<f32>)>,
}

impl Luts {
    fn table(&self, code: LamqCode) -> Option<&[f32]> {
        self.tables
            .iter()
            .find(|(c, _)| *c == code)
            .map(|(_, t)| t.as_slice())
    }
}

fn read_luts(dir: &Path, out: &mut Vec<ResidentEntrySpec>) -> Result<Luts, Inkling2RepackError> {
    let mut z = NpzArchive::open(&dir.join("luts.npz")).map_err(npz_err)?;
    let mut tables = Vec::new();
    for code in [
        LamqCode::Ht2,
        LamqCode::Ht3,
        LamqCode::Rtr1,
        LamqCode::Rtr2,
        LamqCode::Rtr3,
        LamqCode::Rtr4,
    ] {
        let a = z.read(code.name()).map_err(npz_err)?;
        let shape: Vec<usize> = if code.g() == 1 {
            vec![code.table_rows()]
        } else {
            vec![code.table_rows(), code.g()]
        };
        a.expect(NpyDtype::F16, &shape)
            .map_err(|e| Inkling2RepackError(format!("luts.npz {}: {}", code.name(), e.0)))?;
        let bits = a.to_f16_bits().map_err(npz_err)?;
        let f: Vec<f32> = bits
            .iter()
            .map(|&b| compute::quant_1bit::f16_to_f32(b))
            .collect();
        if f.iter().any(|v| !v.is_finite()) {
            return err(format!("luts.npz {}: a non-finite entry", code.name()));
        }
        out.push(ResidentEntrySpec::Raw(RawTensorSpec {
            name: format!("lamq.lut.{}", code.name()),
            dtype: DTYPE_FP16,
            bytes: a.data.clone(),
            shape: rank4(&shape)?,
        }));
        tables.push((code, f));
    }
    if z.names().count() != 6 {
        return err("luts.npz: expected exactly the six trellis tables");
    }
    Ok(Luts { tables })
}

/// One coded matrix read from an npz under `prefix` (`lin_<safe>` or
/// `b<kb>`), validated, as a `DTYPE_LAMQ` matrix entry.
#[allow(clippy::too_many_arguments)]
fn matrix_entry(
    z: &mut NpzArchive,
    prefix: &str,
    name: String,
    code: LamqCode,
    m: usize,
    n: usize,
    out_rot: bool,
    luts: &Luts,
) -> Result<ResidentEntrySpec, Inkling2RepackError> {
    let ctx = |e: String| Inkling2RepackError(format!("{name} ({prefix}): {e}"));
    let codes = z.read(&format!("{prefix}_codes")).map_err(npz_err)?;
    let (want_dt, want_shape) = if code.is_trellis() {
        (
            NpyDtype::U32,
            vec![
                m,
                code.row_bytes(n)
                    .ok_or_else(|| ctx("width does not fit".into()))?
                    / 4,
            ],
        )
    } else {
        (
            NpyDtype::U8,
            vec![
                m * code
                    .row_bytes(n)
                    .ok_or_else(|| ctx("width does not fit".into()))?,
            ],
        )
    };
    codes.expect(want_dt, &want_shape).map_err(|e| ctx(e.0))?;
    let scales = z.read(&format!("{prefix}_scales")).map_err(npz_err)?;
    scales.expect(NpyDtype::F16, &[m]).map_err(|e| ctx(e.0))?;
    let scales = scales.to_f16_bits().map_err(npz_err)?;
    let signs_in = z.read(&format!("{prefix}_signs_in")).map_err(npz_err)?;
    signs_in.expect(NpyDtype::I8, &[n]).map_err(|e| ctx(e.0))?;
    let signs_in = signs_in.to_i8().map_err(npz_err)?;
    let signs_out = if out_rot {
        let s = z.read(&format!("{prefix}_signs_out")).map_err(npz_err)?;
        s.expect(NpyDtype::I8, &[m]).map_err(|e| ctx(e.0))?;
        Some(s.to_i8().map_err(npz_err)?)
    } else {
        if z.contains(&format!("{prefix}_signs_out")) {
            return Err(ctx("signs_out present without out_rot".into()));
        }
        None
    };
    let levels = if code.is_trellis() {
        None
    } else {
        let l = z.read(&format!("{prefix}_levels")).map_err(npz_err)?;
        l.expect(NpyDtype::F32, &[code.table_rows()])
            .map_err(|e| ctx(e.0))?;
        let l = l.to_f32().map_err(npz_err)?;
        if l.iter().any(|v| !v.is_finite()) {
            return Err(ctx("a non-finite level".into()));
        }
        Some(l)
    };
    if scales
        .iter()
        .any(|&s| !compute::quant_1bit::f16_to_f32(s).is_finite())
    {
        return Err(ctx("a non-finite scale".into()));
    }
    let table: &[f32] = match &levels {
        Some(l) => l,
        None => luts.table(code).ok_or_else(|| ctx("no LUT".into()))?,
    };
    LamqLinear {
        code,
        m,
        n,
        codes: &codes.data,
        scales: &scales,
        signs_in: &signs_in,
        signs_out: signs_out.as_deref(),
        table,
    }
    .validate()
    .map_err(ctx)?;
    let aux = encode_matrix_aux(
        code,
        n,
        m,
        &signs_in,
        signs_out.as_deref(),
        levels.as_deref(),
    );
    Ok(ResidentEntrySpec::Lamq(ResidentTensorSpec {
        name,
        packed: codes.data,
        scales,
        biases: u16s(&aux),
        rows: m as u32,
        cols: n as u32,
    }))
}

/// Checks a meta entry's declared parameters against its code.
fn check_meta_code(what: &str, m: &Value, code: LamqCode) -> Result<(), Inkling2RepackError> {
    let get = |k: &str| m.get(k).and_then(Value::as_u64);
    if get("k") != Some(u64::from(code.k())) {
        return err(format!(
            "{what}: k {:?} disagrees with {}",
            m.get("k"),
            code.name()
        ));
    }
    if code.is_trellis() {
        let hash = if code.hash_shift().is_some() {
            u64::from(HASH_MULT)
        } else {
            0
        };
        if get("g") != Some(code.g() as u64)
            || get("K") != Some(code.table_rows() as u64)
            || get("hash_mult") != Some(hash)
        {
            return err(format!(
                "{what}: g/K/hash_mult disagree with {}",
                code.name()
            ));
        }
    }
    Ok(())
}

fn code_of(what: &str, v: Option<&Value>) -> Result<LamqCode, Inkling2RepackError> {
    let s = v.and_then(Value::as_str).unwrap_or("");
    LamqCode::from_name(s).ok_or_else(|| Inkling2RepackError(format!("{what}: code {s:?} refused")))
}

/// An order entry plus its buckets (`embed` or `head`).
#[allow(clippy::too_many_arguments)]
fn bucketed(
    dir: &Path,
    stem: &str,
    hf: &str,
    hidden: usize,
    positions: usize,
    vocab: usize,
    luts: &Luts,
    out: &mut Vec<ResidentEntrySpec>,
) -> Result<(), Inkling2RepackError> {
    let meta = read_json(&dir.join(format!("{stem}_meta.json")))?;
    let buckets = meta
        .as_array()
        .ok_or_else(|| Inkling2RepackError(format!("{stem}_meta.json is not a list")))?;
    let mut z = NpzArchive::open(&dir.join(format!("{stem}_packed.npz"))).map_err(npz_err)?;
    let order = z.read("order").map_err(npz_err)?;
    order
        .expect(NpyDtype::I32, &[positions])
        .map_err(|e| Inkling2RepackError(format!("{stem} order: {}", e.0)))?;
    let ids = order.to_i32().map_err(npz_err)?;
    // A permutation of distinct token ids below the vocab: a repeated or
    // out-of-range id would silently alias two tokens.
    let mut seen = vec![false; vocab];
    for &t in &ids {
        let ok = usize::try_from(t)
            .ok()
            .filter(|&t| t < vocab)
            .filter(|&t| !std::mem::replace(&mut seen[t], true));
        if ok.is_none() {
            return err(format!(
                "{stem} order: token id {t} out of range or repeated"
            ));
        }
    }
    let header = LamqAuxHeader {
        kind: LamqKind::Order,
        code: None,
        out_rot: false,
        n: positions as u32,
        m: 0,
    };
    out.push(ResidentEntrySpec::Lamq(ResidentTensorSpec {
        name: hf.to_string(),
        packed: order.data,
        scales: Vec::new(),
        biases: u16s(&header.encode()),
        rows: positions as u32,
        cols: 1,
    }));
    let mut lo = 0usize;
    for (i, b) in buckets.iter().enumerate() {
        if b.get("kb").and_then(Value::as_u64) != Some(i as u64) {
            return err(format!("{stem} bucket {i}: kb out of order"));
        }
        let rows = b.get("rows").and_then(Value::as_u64).unwrap_or(0) as usize;
        let what = format!("{stem} bucket {i}");
        let code = code_of(&what, b.get("opt"))?;
        check_meta_code(&what, b, code)?;
        if rows == 0 {
            return err(format!("{what}: no rows"));
        }
        out.push(matrix_entry(
            &mut z,
            &format!("b{i}"),
            format!("{hf}.b{i}"),
            code,
            rows,
            hidden,
            false,
            luts,
        )?);
        lo += rows;
    }
    if lo != positions {
        return err(format!(
            "{stem}: buckets cover {lo} of {positions} positions"
        ));
    }
    if z.names().count()
        != 1 + buckets
            .iter()
            .map(|b| {
                if b["opt"].as_str().is_some_and(|o| o.starts_with("lm")) {
                    4
                } else {
                    3
                }
            })
            .sum::<usize>()
    {
        return err(format!("{stem}_packed.npz: unexpected members"));
    }
    Ok(())
}

/// What an intake wrote.
#[derive(Debug, Clone, PartialEq)]
pub struct Inkling2LamqReport {
    pub layers: usize,
    pub resident_tensors: usize,
    pub expert_stride: u64,
    /// Expert bytes before the per-layer padding, and after.
    pub expert_bytes_packed: u64,
    pub expert_bytes_written: u64,
}

/// One layer's experts as blobs, plus its signs entry.
fn layer_experts(
    z: &mut NpzArchive,
    meta: &Value,
    layer: usize,
    experts: usize,
    hidden: usize,
    ffn: usize,
) -> Result<(LayerBlobs, ResidentEntrySpec, u64), Inkling2RepackError> {
    let what = format!("layer {layer} experts");
    let signs =
        |name: &str, n: usize, z: &mut NpzArchive| -> Result<Vec<i8>, Inkling2RepackError> {
            let a = z.read(name).map_err(npz_err)?;
            a.expect(NpyDtype::I8, &[n])
                .map_err(|e| Inkling2RepackError(format!("{what} {name}: {}", e.0)))?;
            let s = a.to_i8().map_err(npz_err)?;
            if s.iter().any(|&v| v != 1 && v != -1) {
                return err(format!("{what} {name}: a sign is not +1/-1"));
            }
            Ok(s)
        };
    let gu_signs = signs("gu_signs_in", hidden, z)?;
    let dn_signs = signs("dn_signs_in", ffn, z)?;
    let mut payload: Vec<u8> = gu_signs.iter().map(|&s| s as u8).collect();
    payload.extend(dn_signs.iter().map(|&s| s as u8));
    let header = LamqAuxHeader {
        kind: LamqKind::Signs,
        code: None,
        out_rot: false,
        n: hidden as u32,
        m: ffn as u32,
    };
    let signs_entry = ResidentEntrySpec::Lamq(ResidentTensorSpec {
        name: format!("model.llm.layers.{layer}.mlp.experts.lamq_signs"),
        packed: payload,
        scales: Vec::new(),
        biases: u16s(&header.encode()),
        rows: (hidden + ffn) as u32,
        cols: 1,
    });

    // Per side: (rows, cols, map, formats with their codes and scales).
    struct Side {
        rows: usize,
        cols: usize,
        map: Vec<(LamqCode, usize)>,
        slabs: Vec<(LamqCode, Vec<u8>, Vec<u8>)>,
    }
    let mut sides = Vec::new();
    for (key, rows, cols) in [("gu", 2 * ffn, hidden), ("dn", hidden, ffn)] {
        let counts = meta["experts"][key]
            .as_object()
            .ok_or_else(|| Inkling2RepackError(format!("{what}: meta has no {key} counts")))?;
        let map_a = z.read(&format!("{key}_map")).map_err(npz_err)?;
        map_a
            .expect(NpyDtype::I16, &[experts, 2])
            .map_err(|e| Inkling2RepackError(format!("{what} {key}_map: {}", e.0)))?;
        let raw = map_a.to_i16().map_err(npz_err)?;
        let mut map = Vec::with_capacity(experts);
        let mut slabs = Vec::new();
        let mut total = 0usize;
        for (fmt, count) in counts {
            let code = LamqCode::from_name(fmt)
                .filter(|c| c.is_trellis())
                .ok_or_else(|| {
                    Inkling2RepackError(format!("{what}: expert format {fmt} refused"))
                })?;
            let count = count.as_u64().unwrap_or(0) as usize;
            let rb = code
                .row_bytes(cols)
                .ok_or_else(|| Inkling2RepackError(format!("{what}: {fmt} width {cols}")))?;
            let c = z.read(&format!("{key}_{fmt}_codes")).map_err(npz_err)?;
            if c.dtype != NpyDtype::U32 || c.data.len() != count * rows * rb {
                return err(format!("{what}: {key}_{fmt}_codes is {:?} {:?}, want {count} x {rows} rows of {rb} bytes", c.dtype, c.shape));
            }
            let s = z.read(&format!("{key}_{fmt}_scales")).map_err(npz_err)?;
            if s.dtype != NpyDtype::F16 || s.len() != count * rows {
                return err(format!(
                    "{what}: {key}_{fmt}_scales is {:?} {:?}, want {count} x {rows}",
                    s.dtype, s.shape
                ));
            }
            if s.to_f16_bits()
                .map_err(npz_err)?
                .iter()
                .any(|&b| !compute::quant_1bit::f16_to_f32(b).is_finite())
            {
                return err(format!("{what}: {key}_{fmt}: a non-finite scale"));
            }
            total += count;
            slabs.push((code, c.data, s.data));
        }
        if total != experts {
            return err(format!(
                "{what}: {key} counts sum to {total}, want {experts}"
            ));
        }
        // Every expert maps to a distinct, in-range slot of a listed format.
        let mut used: Vec<Vec<bool>> = slabs
            .iter()
            .map(|(_, _, s)| vec![false; s.len() / 2 / rows])
            .collect();
        for e in 0..experts {
            let (fid, slot) = (raw[2 * e], raw[2 * e + 1]);
            let code = LamqCode::from_fid(i32::from(fid)).ok_or_else(|| {
                Inkling2RepackError(format!("{what}: expert {e} {key} fid {fid} refused"))
            })?;
            let si = slabs
                .iter()
                .position(|(c, _, _)| *c == code)
                .ok_or_else(|| {
                    Inkling2RepackError(format!(
                        "{what}: expert {e} {key} uses {} with no slab",
                        code.name()
                    ))
                })?;
            let slot = usize::try_from(slot)
                .ok()
                .filter(|&s| s < used[si].len())
                .filter(|&s| !std::mem::replace(&mut used[si][s], true));
            let slot = slot.ok_or_else(|| {
                Inkling2RepackError(format!(
                    "{what}: expert {e} {key} slot out of range or reused"
                ))
            })?;
            map.push((code, slot));
        }
        sides.push(Side {
            rows,
            cols,
            map,
            slabs,
        });
    }
    // Two sign vectors and two maps, two arrays per format slab, and the
    // linears' own members.
    if z.names().count()
        != 4 + sides.iter().map(|s| 2 * s.slabs.len()).sum::<usize>() + count_linear_members(meta)?
    {
        return err(format!("layer {layer}: layer npz has unexpected members"));
    }

    let widest = |s: &Side| {
        s.slabs
            .iter()
            .map(|(c, _, _)| s.rows * c.row_bytes(s.cols).unwrap())
            .max()
            .unwrap_or(0)
    };
    let (gu_w, dn_w) = (widest(&sides[0]), widest(&sides[1]));
    let mut packed_bytes = 0u64;
    let mut blobs = Vec::with_capacity(experts);
    for e in 0..experts {
        let mut fmt = [0u8; INKLING2_LAMQ_FMT_BYTES];
        let mut subs = Vec::with_capacity(5);
        let mut parts = Vec::new();
        for (i, s) in sides.iter().enumerate() {
            let (code, slot) = s.map[e];
            let rb = code.row_bytes(s.cols).unwrap();
            let (_, codes, scales) = s.slabs.iter().find(|(c, _, _)| *c == code).unwrap();
            let span = s.rows * rb;
            let mut c = codes[slot * span..(slot + 1) * span].to_vec();
            packed_bytes += (span + 2 * s.rows) as u64;
            c.resize(if i == 0 { gu_w } else { dn_w }, 0);
            fmt[4 * i] = code.id();
            fmt[8 + 4 * i..12 + 4 * i].copy_from_slice(&(rb as u32).to_le_bytes());
            let sc = scales[slot * 2 * s.rows..(slot + 1) * 2 * s.rows].to_vec();
            parts.push((c, sc, s.rows, s.cols));
        }
        subs.push(SubTensor {
            role: "lamq_fmt".into(),
            bytes: fmt.to_vec(),
            dtype: INKLING2_LAMQ_FMT_DTYPE.into(),
            shape: vec![4],
        });
        for ((c, sc, rows, cols), role) in parts.into_iter().zip(["gate_up", "down"]) {
            subs.push(SubTensor {
                role: role.into(),
                bytes: c,
                dtype: INKLING2_LAMQ_DTYPE.into(),
                shape: vec![rows as u64, cols as u64],
            });
            subs.push(SubTensor {
                role: format!("{role}_scales"),
                bytes: sc,
                dtype: "F16".into(),
                shape: vec![rows as u64],
            });
        }
        blobs.push(ExpertBlob {
            expert: e,
            sub_tensors: subs,
        });
    }
    Ok((
        LayerBlobs {
            layer,
            experts: blobs,
        },
        signs_entry,
        packed_bytes,
    ))
}

/// npz members the coded linears of a layer own.
fn count_linear_members(meta: &Value) -> Result<usize, Inkling2RepackError> {
    let lin = meta["linears"]
        .as_object()
        .ok_or_else(|| Inkling2RepackError("layer meta has no linears".into()))?;
    Ok(lin
        .values()
        .map(|m| {
            3 + usize::from(m["out_rot"].as_bool().unwrap_or(false))
                + usize::from(m["fmt"].as_str().is_some_and(|f| f.starts_with("lm")))
        })
        .sum())
}

/// The `manifest.quant` section of a LAMQ install.
pub fn inkling2_lamq_manifest_quant() -> Value {
    let lamq = serde_json::json!({
        "weightBits": 0, "scheme": LAMQ_SCHEME, "scaleType": "none", "biasType": "none", "groupSize": 0
    });
    serde_json::json!({
        "embedding": lamq,
        "attention": lamq,
        "router": {"weightBits": 32, "scheme": "raw", "scaleType": "none", "biasType": "none", "groupSize": 0},
        "sharedExpert": lamq,
        "routedExpert": {
            "weightBits": 0, "scheme": INKLING2_LAMQ_ROUTED_SCHEME, "scaleType": "none",
            "biasType": "none", "groupSize": 0
        },
    })
}

/// Writes an install from a downloaded release directory.
pub fn write_inkling2_lamq_install(
    dir: &Path,
    out: &Path,
    model_id: &str,
    mut progress: impl FnMut(&str),
) -> Result<Inkling2LamqReport, Inkling2RepackError> {
    let manifest = read_json(&dir.join("manifest.json"))?;
    if manifest["format"].as_str() != Some(LAMQ_INKLING2_FORMAT) {
        return err(format!(
            "manifest.json format {:?} is not {LAMQ_INKLING2_FORMAT}",
            manifest["format"]
        ));
    }
    for (name, want) in [
        ("ht1", 0),
        ("ht2", 1),
        ("ht3", 2),
        ("rtr1", 3),
        ("rtr2", 4),
        ("rtr3", 5),
        ("rtr4", 6),
    ] {
        if manifest["fmt_ids"][name].as_i64() != Some(want) {
            return err(format!("manifest.json fmt_ids.{name} is not {want}"));
        }
    }
    let config = std::fs::read_to_string(dir.join("config.json"))
        .map_err(|e| Inkling2RepackError(format!("config.json: {e}")))?;
    let (arch, side) = parse_inkling2_config(&config)?;
    let layers = arch.num_layers as usize;
    let (hidden, ffn, experts) = (
        arch.hidden_size as usize,
        arch.moe_intermediate_size as usize,
        arch.num_experts as usize,
    );
    let core = (side.core_end - side.core_start) as usize;
    let taps = arch.short_conv_kernel_size as usize;

    let mut resident = Vec::new();
    let luts = read_luts(dir, &mut resident)?;
    let st = SafetensorsFile::open(&dir.join("steering.safetensors"))
        .map_err(|e| Inkling2RepackError(format!("steering.safetensors: {e}")))?;
    let steering = steering_entries(&st, core, hidden, taps)?;
    if manifest["steering_tensors"].as_u64() != Some(st.tensor_names().count() as u64) {
        return err("steering.safetensors tensor count disagrees with manifest.json");
    }
    resident.extend(steering);
    let vocab = arch.vocab_size as usize;
    bucketed(
        dir,
        "embed",
        "model.llm.embed.weight",
        hidden,
        vocab,
        vocab,
        &luts,
        &mut resident,
    )?;
    bucketed(
        dir,
        "head",
        "model.llm.unembed.weight",
        hidden,
        side.unpadded_vocab as usize,
        vocab,
        &luts,
        &mut resident,
    )?;

    // Pass 1: read every layer's linears and expert shapes (linears are
    // resident; expert blobs are written in pass 2 one layer at a time).
    let mut metas = Vec::with_capacity(layers);
    for l in 0..layers {
        let meta = read_json(&dir.join(format!("layer_{l:02}_meta.json")))?;
        if meta["layer"].as_u64() != Some(l as u64) {
            return err(format!(
                "layer_{l:02}_meta.json names layer {:?}",
                meta["layer"]
            ));
        }
        let lin = meta["linears"]
            .as_object()
            .ok_or_else(|| Inkling2RepackError(format!("layer {l}: no linears")))?;
        if lin.len() != LINEARS.len() {
            return err(format!(
                "layer {l}: {} linears, want {}",
                lin.len(),
                LINEARS.len()
            ));
        }
        let mut z =
            NpzArchive::open(&dir.join(format!("layer_{l:02}_packed.npz"))).map_err(npz_err)?;
        for (key, hf) in LINEARS {
            let m = lin
                .get(key)
                .ok_or_else(|| Inkling2RepackError(format!("layer {l}: no {key}")))?;
            let what = format!("layer {l} {key}");
            let code = code_of(&what, m.get("fmt"))?;
            check_meta_code(&what, m, code)?;
            let (rows, cols) = (
                m["m"].as_u64().unwrap_or(0) as usize,
                m["n"].as_u64().unwrap_or(0) as usize,
            );
            let safe = m["safe"].as_str().unwrap_or("");
            if safe != key.replace('.', "_") {
                return err(format!("{what}: safe name {safe:?}"));
            }
            let out_rot = m.get("out_rot").and_then(Value::as_bool).unwrap_or(false);
            resident.push(matrix_entry(
                &mut z,
                &format!("lin_{safe}"),
                format!("model.llm.layers.{l}.{hf}"),
                code,
                rows,
                cols,
                out_rot,
                &luts,
            )?);
        }
        metas.push(meta);
    }

    // Stride: the widest layer (code sub-tensors are padded per layer).
    let blob_bytes = |meta: &Value| -> Result<u64, Inkling2RepackError> {
        let mut total = INKLING2_LAMQ_FMT_BYTES;
        for (key, rows, cols) in [("gu", 2 * ffn, hidden), ("dn", hidden, ffn)] {
            let w = meta["experts"][key]
                .as_object()
                .ok_or_else(|| Inkling2RepackError(format!("meta has no {key} counts")))?
                .keys()
                .map(|f| {
                    LamqCode::from_name(f)
                        .and_then(|c| c.row_bytes(cols))
                        .ok_or_else(|| Inkling2RepackError(format!("expert format {f} refused")))
                })
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .max()
                .unwrap_or(0);
            total += rows * w + 2 * rows;
        }
        Ok(total as u64)
    };
    let page = crate::GTURBO_PAGE_BYTES;
    let mut stride = 0u64;
    for meta in &metas {
        stride = stride.max(blob_bytes(meta)?.div_ceil(page) * page);
    }

    let io = |e: crate::gturbo_writer::WriterError| Inkling2RepackError(e.to_string());
    let mut writer = StreamingGturboWriter::new(out, stride, experts).map_err(io)?;
    writer.set_quant(inkling2_lamq_manifest_quant());
    let (mut packed, mut written) = (0u64, 0u64);
    for (l, meta) in metas.iter().enumerate() {
        let mut z =
            NpzArchive::open(&dir.join(format!("layer_{l:02}_packed.npz"))).map_err(npz_err)?;
        let (blobs, signs, p) = layer_experts(&mut z, meta, l, experts, hidden, ffn)?;
        let per = blob_bytes(meta)?.div_ceil(page) * page;
        packed += p;
        written += per * experts as u64;
        writer.write_layer(&blobs).map_err(io)?;
        resident.push(signs);
        progress(&format!("layer {l:2}: {experts} experts, {per} bytes each"));
    }
    let side_path = out.join(model_io::inkling2_config::INKLING2_CONFIG_FILE);
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
    Ok(Inkling2LamqReport {
        layers,
        resident_tensors: n_resident,
        expert_stride: stride,
        expert_bytes_packed: packed,
        expert_bytes_written: written,
    })
}

#[cfg(test)]
#[path = "inkling2_lamq_tests.rs"]
mod tests;
