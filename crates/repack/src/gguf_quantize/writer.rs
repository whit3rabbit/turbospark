//! Streaming GGUF rewrite driven by a [`Recipe`].

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

use foundation::LogitValue as F16;

use super::imatrix::Imatrix;
use super::quantizers::Target;
use super::recipe::{Choice, Recipe};
use crate::gguf_header::types::{MAGIC, SUPPORTED_VERSION};
use crate::gguf_header::{
    ggml_type_block, ggml_type_name, parse_header, GgufHeader, GgufHeaderError, GgufTensorInfo,
    GgufValue, DEFAULT_MAX_HEADER_BYTES,
};

/// Elements converted per step. A multiple of every block size used here.
const CHUNK_ELEMS: u64 = 1 << 20;

/// Where a run is, reported after every batch of rows and every tensor.
#[derive(Debug, Clone)]
pub struct Progress {
    /// The tensor being written.
    pub tensor: String,
    pub tensors_done: u64,
    pub tensors_total: u64,
    /// Output bytes written so far, and the total the run will write.
    pub bytes_done: u64,
    pub bytes_total: u64,
}

/// Optional hooks for a run: progress reporting and cancellation. The progress
/// callback may be invoked from the calling thread only.
#[derive(Default)]
pub struct Control<'a> {
    pub progress: Option<&'a dyn Fn(&Progress)>,
    /// Checked between row batches; raising it ends the run with
    /// [`QuantizeError::Cancelled`] and removes the partial file.
    pub cancel: Option<&'a std::sync::atomic::AtomicBool>,
}

impl Control<'_> {
    fn cancelled(&self) -> bool {
        self.cancel
            .is_some_and(|c| c.load(std::sync::atomic::Ordering::Relaxed))
    }
}

#[derive(Debug)]
pub enum QuantizeError {
    Io(String),
    Header(GgufHeaderError),
    /// A tensor the recipe cannot handle, named, with the reason.
    Refused {
        tensor: String,
        reason: String,
    },
    /// The caller's cancel flag was raised. No output file is left behind.
    Cancelled,
}

impl std::fmt::Display for QuantizeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QuantizeError::Io(e) => write!(f, "gguf quantize: io: {e}"),
            QuantizeError::Header(e) => write!(f, "gguf quantize: header: {e}"),
            QuantizeError::Cancelled => write!(f, "gguf quantize: cancelled"),
            QuantizeError::Refused { tensor, reason } => {
                write!(f, "gguf quantize: tensor {tensor}: {reason}")
            }
        }
    }
}

impl std::error::Error for QuantizeError {}

fn io<E: std::fmt::Display>(e: E) -> QuantizeError {
    QuantizeError::Io(e.to_string())
}

/// What a run did, for the caller to print.
#[derive(Debug, Default, Clone)]
pub struct QuantizeReport {
    /// Output tensor count per ggml type name.
    pub by_type: BTreeMap<String, u64>,
    pub tensors: u64,
    pub bytes_in: u64,
    pub bytes_out: u64,
    /// Already-quantized tensors the default left alone (not re-quantized).
    pub kept_quantized: Vec<String>,
    /// Output block types that are not in `model_io::EXECUTABLE_GGUF_TYPES`,
    /// the gate for what a turbospark install can run. The file is still a valid
    /// GGUF (llama.cpp reads it); this only says turbospark's own runtime
    /// gate would not accept it as weights. Float types are not listed.
    pub not_in_executable_list: Vec<String>,
    /// Tensors quantized with importance weights.
    pub imatrix_used: u64,
    /// Tensors of an imatrix-aware type that had no entry in the imatrix and
    /// were quantized without weights (llama.cpp does the same, with a log).
    pub imatrix_missing: Vec<String>,
}

#[derive(Clone, Copy)]
enum Action {
    Copy,
    Convert(Target),
}

struct Planned {
    name: String,
    info: GgufTensorInfo,
    action: Action,
    out_type: u32,
    in_bytes: u64,
    out_bytes: u64,
    out_offset: u64,
}

fn is_float(ty: u32) -> bool {
    matches!(ty, 0 | 1 | 30)
}

/// A split GGUF (`...-00001-of-00005.gguf`) keeps its tensors in several
/// files and each header lists only its own. Quantizing one shard would write a
/// file that looks complete and is not, so refuse it by name.
fn refuse_split(h: &GgufHeader) -> Result<(), QuantizeError> {
    let count = h.metadata_u64("split.count").unwrap_or(1);
    if count > 1 {
        return Err(QuantizeError::Refused {
            tensor: "(whole file)".into(),
            reason: format!(
                "this is shard {} of a {count}-file split GGUF; quantizing one shard would \
                 write an incomplete model. Merge the shards first (llama.cpp's \
                 `llama-gguf-split --merge`) and quantize the merged file",
                h.metadata_u64("split.no").map_or(1, |n| n + 1)
            ),
        });
    }
    Ok(())
}

pub(super) fn read_header(f: &mut File, len: u64) -> Result<GgufHeader, QuantizeError> {
    let mut want = (1u64 << 20).min(len);
    loop {
        let mut buf = Vec::new();
        f.seek(SeekFrom::Start(0)).map_err(io)?;
        f.take(want).read_to_end(&mut buf).map_err(io)?;
        match parse_header(&buf, DEFAULT_MAX_HEADER_BYTES) {
            Ok(h) => return Ok(h),
            Err(GgufHeaderError::TooShort { needed }) if want < len => {
                want = needed.max(want.saturating_mul(2)).min(len);
            }
            Err(e) => return Err(QuantizeError::Header(e)),
        }
    }
}

fn plan_tensor(
    name: &str,
    info: &GgufTensorInfo,
    recipe: &Recipe,
    report: &mut QuantizeReport,
) -> Result<(Action, u32, u64), QuantizeError> {
    let refuse = |reason: String| QuantizeError::Refused {
        tensor: name.to_string(),
        reason,
    };
    let in_bytes = info.byte_size(name).map_err(QuantizeError::Header)?;
    let copy = (Action::Copy, info.ggml_type, in_bytes);

    // Norms, biases and other vectors stay as they are.
    if info.dims.len() < 2 {
        return Ok(copy);
    }
    let rule = recipe.rule_for(name);
    let chosen = rule.map_or(recipe.default, |r| r.choice);
    let target = match chosen {
        Choice::Keep => return Ok(copy),
        Choice::Type(t) => t,
    };
    if target.ggml_id() == info.ggml_type {
        return Ok(copy);
    }
    if !is_float(info.ggml_type) {
        let src = ggml_type_name(info.ggml_type).unwrap_or("?");
        if rule.is_some() {
            return Err(refuse(format!(
                "source is {src}; a rule asked for a different type, and re-quantizing a \
                 quantized tensor is refused (start from an F16/BF16/F32 source)"
            )));
        }
        report.kept_quantized.push(name.to_string());
        return Ok(copy);
    }
    // Rows are dims[0] long (ggml stores the fastest-varying dim first).
    let (be, _) = target.block();
    let target = if info.dims[0] % be as u64 == 0 {
        target
    } else {
        let fb = recipe.fallback;
        if info.dims[0] % fb.block().0 as u64 != 0 {
            return Err(refuse(format!(
                "row length {} fits neither {target:?} nor fallback {fb:?}",
                info.dims[0]
            )));
        }
        fb
    };
    if target.ggml_id() == info.ggml_type {
        return Ok(copy);
    }
    let (be, bb) = target.block();
    let out_bytes = info.element_count() / be as u64 * bb as u64;
    Ok((Action::Convert(target), target.ggml_id(), out_bytes))
}

// ---- header serialization ---------------------------------------------------

fn put_str(v: &mut Vec<u8>, s: &str) {
    v.extend_from_slice(&(s.len() as u64).to_le_bytes());
    v.extend_from_slice(s.as_bytes());
}

fn type_id(v: &GgufValue) -> u32 {
    match v {
        GgufValue::U8(_) => 0,
        GgufValue::I8(_) => 1,
        GgufValue::U16(_) => 2,
        GgufValue::I16(_) => 3,
        GgufValue::U32(_) => 4,
        GgufValue::I32(_) => 5,
        GgufValue::F32(_) => 6,
        GgufValue::Bool(_) => 7,
        GgufValue::String(_) => 8,
        GgufValue::Array(_) => 9,
        GgufValue::U64(_) => 10,
        GgufValue::I64(_) => 11,
        GgufValue::F64(_) => 12,
    }
}

fn put_value(out: &mut Vec<u8>, v: &GgufValue) {
    match v {
        GgufValue::U8(x) => out.push(*x),
        GgufValue::I8(x) => out.push(*x as u8),
        GgufValue::U16(x) => out.extend_from_slice(&x.to_le_bytes()),
        GgufValue::I16(x) => out.extend_from_slice(&x.to_le_bytes()),
        GgufValue::U32(x) => out.extend_from_slice(&x.to_le_bytes()),
        GgufValue::I32(x) => out.extend_from_slice(&x.to_le_bytes()),
        GgufValue::F32(x) => out.extend_from_slice(&x.to_bits().to_le_bytes()),
        GgufValue::Bool(x) => out.push(u8::from(*x)),
        GgufValue::String(s) => put_str(out, s),
        GgufValue::U64(x) => out.extend_from_slice(&x.to_le_bytes()),
        GgufValue::I64(x) => out.extend_from_slice(&x.to_le_bytes()),
        GgufValue::F64(x) => out.extend_from_slice(&x.to_bits().to_le_bytes()),
        GgufValue::Array(items) => {
            // Empty arrays lose their declared element type in the parser;
            // U32 is the neutral choice (see module docs).
            let et = items.first().map_or(4, type_id);
            out.extend_from_slice(&et.to_le_bytes());
            out.extend_from_slice(&(items.len() as u64).to_le_bytes());
            items.iter().for_each(|i| put_value(out, i));
        }
    }
}

fn serialize_header(metadata: &BTreeMap<String, GgufValue>, plan: &[Planned]) -> Vec<u8> {
    let mut h = Vec::new();
    h.extend_from_slice(&MAGIC.to_le_bytes());
    h.extend_from_slice(&SUPPORTED_VERSION.to_le_bytes());
    h.extend_from_slice(&(plan.len() as u64).to_le_bytes());
    h.extend_from_slice(&(metadata.len() as u64).to_le_bytes());
    for (k, v) in metadata {
        put_str(&mut h, k);
        h.extend_from_slice(&type_id(v).to_le_bytes());
        put_value(&mut h, v);
    }
    for p in plan {
        put_str(&mut h, &p.name);
        h.extend_from_slice(&(p.info.dims.len() as u32).to_le_bytes());
        p.info
            .dims
            .iter()
            .for_each(|d| h.extend_from_slice(&d.to_le_bytes()));
        h.extend_from_slice(&p.out_type.to_le_bytes());
        h.extend_from_slice(&p.out_offset.to_le_bytes());
    }
    h
}

// ---- data -------------------------------------------------------------------

fn to_f32(ty: u32, bytes: &[u8], out: &mut Vec<f32>) {
    out.clear();
    match ty {
        0 => out.extend(
            bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])),
        ),
        1 => out.extend(
            bytes
                .chunks_exact(2)
                .map(|b| f32::from(F16::from_bits(u16::from_le_bytes([b[0], b[1]])))),
        ),
        30 => out.extend(
            bytes
                .chunks_exact(2)
                .map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16)),
        ),
        _ => unreachable!("plan_tensor admits float sources only"),
    }
}

/// Types whose encoder reads importance weights (matches ggml's `quantize_*`).
fn uses_imatrix(t: Target) -> bool {
    !matches!(t, Target::F32 | Target::F16 | Target::Bf16 | Target::Q8_0)
}

/// `token_embd` and `output.weight` are exempt from llama.cpp's low-bit rule.
fn exempt_from_policy(name: &str) -> bool {
    name.ends_with("token_embd.weight") || name == "output.weight"
}

/// Refuse a very low-bit target with no importance weights, as ggml (hard
/// requirement) and llama.cpp (`tensor_requires_imatrix`) do.
fn check_imatrix_requirement(t: Target, name: &str, have: bool) -> Result<(), QuantizeError> {
    if have {
        return Ok(());
    }
    let hard = t.requires_imatrix();
    let policy = t.policy_requires_imatrix() && !exempt_from_policy(name);
    if hard || policy {
        return Err(QuantizeError::Refused {
            tensor: name.to_string(),
            reason: format!(
                "{t:?} needs an importance matrix entry for this tensor; without one the \
                 result is poor (llama.cpp refuses, ggml asserts). Pass --imatrix, or pick a \
                 recipe that gives this tensor a higher-precision type"
            ),
        });
    }
    Ok(())
}

/// llama.cpp's rule: an entry applies when it holds `ne0 * ne2 * ne3` values;
/// any other size is an error, except for `token_embd`, which is skipped.
fn imatrix_for<'a>(
    im: Option<&'a Imatrix>,
    name: &str,
    info: &GgufTensorInfo,
) -> Result<Option<&'a [f32]>, QuantizeError> {
    let Some(entry) = im.and_then(|i| i.get(name)) else {
        return Ok(None);
    };
    let want = info.dims[0] * info.dims.iter().skip(2).product::<u64>();
    if entry.len() as u64 == want {
        return Ok(Some(entry));
    }
    if name.ends_with("token_embd.weight") {
        return Ok(None);
    }
    Err(QuantizeError::Refused {
        tensor: name.to_string(),
        reason: format!(
            "imatrix has {} values but the tensor needs {want} (ne0 * ne2 * ne3)",
            entry.len()
        ),
    })
}

/// Convert and encode `nr` whole rows of `raw` on up to `threads` threads.
/// Rows are independent, so the bytes do not depend on the thread count.
fn encode_parallel(
    raw: &[u8],
    src_type: u32,
    target: Target,
    ne0: usize,
    nr: usize,
    im: Option<&[f32]>,
    threads: usize,
) -> Vec<u8> {
    let (_, elem_bytes) = ggml_type_block(src_type).expect("float source");
    let row_bytes = ne0 * elem_bytes as usize;
    let encode = |rows: &[u8]| -> Vec<u8> {
        let mut f = Vec::new();
        to_f32(src_type, rows, &mut f);
        target.encode_rows(&f, ne0, im)
    };
    let per = nr.div_ceil(threads.max(1)).max(1);
    if threads <= 1 || nr <= per {
        return encode(&raw[..nr * row_bytes]);
    }
    let parts: Vec<Vec<u8>> = std::thread::scope(|s| {
        let handles: Vec<_> = raw[..nr * row_bytes]
            .chunks(per * row_bytes)
            .map(|c| s.spawn(|| encode(c)))
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("encoder thread panicked"))
            .collect()
    });
    parts.concat()
}

fn write_tensor(
    src: &mut File,
    dst: &mut impl Write,
    p: &Planned,
    src_start: u64,
    im: Option<&[f32]>,
    threads: usize,
    batch: &mut dyn FnMut(u64) -> Result<(), QuantizeError>,
) -> Result<(), QuantizeError> {
    src.seek(SeekFrom::Start(src_start)).map_err(io)?;
    match p.action {
        Action::Copy => {
            let mut left = p.in_bytes;
            let mut buf = vec![0u8; 1 << 22];
            while left > 0 {
                let n = left.min(buf.len() as u64) as usize;
                src.read_exact(&mut buf[..n]).map_err(io)?;
                dst.write_all(&buf[..n]).map_err(io)?;
                left -= n as u64;
                batch(n as u64)?;
            }
        }
        Action::Convert(target) => {
            let (_, elem_bytes) = ggml_type_block(p.info.ggml_type).expect("float source");
            let ne0 = p.info.dims[0];
            let rows_per_expert = p.info.dims.get(1).copied().unwrap_or(1);
            let n_expert: u64 = p.info.dims.iter().skip(2).product();
            // Each thread gets about CHUNK_ELEMS elements per batch.
            let rows_per_batch = ((CHUNK_ELEMS / ne0).max(1)) * threads.max(1) as u64;
            let mut raw = Vec::new();
            // Each expert has its own imatrix slice, so batches never cross one.
            for e in 0..n_expert {
                let im_e = im.map(|m| &m[(e * ne0) as usize..((e + 1) * ne0) as usize]);
                let mut r = 0;
                while r < rows_per_expert {
                    let nr = rows_per_batch.min(rows_per_expert - r);
                    raw.resize((nr * ne0 * elem_bytes) as usize, 0);
                    src.read_exact(&mut raw).map_err(io)?;
                    let out = encode_parallel(
                        &raw,
                        p.info.ggml_type,
                        target,
                        ne0 as usize,
                        nr as usize,
                        im_e,
                        threads,
                    );
                    dst.write_all(&out).map_err(io)?;
                    batch(out.len() as u64)?;
                    r += nr;
                }
            }
        }
    }
    Ok(())
}

/// Rewrite `src` to `dst` under `recipe`. `dst` is written as `<dst>.partial`
/// and renamed on success, so a failed run leaves no half file at `dst`.
pub fn quantize_gguf(
    src_path: &Path,
    dst_path: &Path,
    recipe: &Recipe,
) -> Result<QuantizeReport, QuantizeError> {
    quantize_gguf_with(src_path, dst_path, recipe, None)
}

/// As [`quantize_gguf`], with optional importance weights.
pub fn quantize_gguf_with(
    src_path: &Path,
    dst_path: &Path,
    recipe: &Recipe,
    imatrix: Option<&Imatrix>,
) -> Result<QuantizeReport, QuantizeError> {
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    quantize_gguf_opts(src_path, dst_path, recipe, imatrix, threads)
}

/// As [`quantize_gguf_with`], with an explicit thread count. The output bytes
/// are the same for any count (rows are independent); only the time changes.
pub fn quantize_gguf_opts(
    src_path: &Path,
    dst_path: &Path,
    recipe: &Recipe,
    imatrix: Option<&Imatrix>,
    threads: usize,
) -> Result<QuantizeReport, QuantizeError> {
    quantize_gguf_ctl(
        src_path,
        dst_path,
        recipe,
        imatrix,
        threads,
        &Control::default(),
    )
}

/// As [`quantize_gguf_opts`], with progress reporting and cancellation.
pub fn quantize_gguf_ctl(
    src_path: &Path,
    dst_path: &Path,
    recipe: &Recipe,
    imatrix: Option<&Imatrix>,
    threads: usize,
    control: &Control,
) -> Result<QuantizeReport, QuantizeError> {
    let mut src = File::open(src_path).map_err(io)?;
    let len = src.metadata().map_err(io)?.len();
    let header = read_header(&mut src, len)?;
    refuse_split(&header)?;
    let mut report = QuantizeReport::default();

    // Stream in source order so reads are sequential.
    let mut order: Vec<(&String, &GgufTensorInfo)> = header.tensors.iter().collect();
    order.sort_by_key(|(_, i)| i.offset);

    let align = header.alignment;
    let round = |x: u64| x.div_ceil(align) * align;
    let mut plan = Vec::with_capacity(order.len());
    let mut cursor = 0u64;
    for (name, info) in order {
        let (action, out_type, out_bytes) = plan_tensor(name, info, recipe, &mut report)?;
        let in_bytes = info.byte_size(name).map_err(QuantizeError::Header)?;
        if let Action::Convert(t) = action {
            let entry = imatrix_for(imatrix, name, info)?;
            check_imatrix_requirement(t, name, entry.is_some())?;
            if uses_imatrix(t) && imatrix.is_some() {
                match entry {
                    Some(_) => report.imatrix_used += 1,
                    None if imatrix.is_some_and(|i| i.get(name).is_none()) => {
                        report.imatrix_missing.push(name.clone())
                    }
                    None => {}
                }
            }
        }
        let out_offset = round(cursor);
        cursor = out_offset + out_bytes;
        report.bytes_in += in_bytes;
        report.bytes_out += out_bytes;
        *report
            .by_type
            .entry(ggml_type_name(out_type).unwrap_or("?").to_string())
            .or_default() += 1;
        plan.push(Planned {
            name: name.clone(),
            info: info.clone(),
            action,
            out_type,
            in_bytes,
            out_bytes,
            out_offset,
        });
    }
    report.tensors = plan.len() as u64;
    for p in &plan {
        if matches!(p.out_type, 0 | 1 | 30) {
            continue;
        }
        let name = ggml_type_name(p.out_type)
            .unwrap_or("?")
            .to_ascii_lowercase();
        if !model_io::EXECUTABLE_GGUF_TYPES.contains(&name.as_str())
            && !report.not_in_executable_list.contains(&name)
        {
            report.not_in_executable_list.push(name);
        }
    }

    let mut metadata = header.metadata.clone();
    if let Choice::Type(t) = recipe.default {
        metadata.insert("general.file_type".into(), GgufValue::U32(t.file_type()));
    }
    let head = serialize_header(&metadata, &plan);

    let tmp = dst_path.with_extension("gguf.partial");
    let result = (|| -> Result<(), QuantizeError> {
        let mut dst = BufWriter::with_capacity(1 << 22, File::create(&tmp).map_err(io)?);
        dst.write_all(&head).map_err(io)?;
        let data_start = round(head.len() as u64);
        let mut pos = head.len() as u64;
        let pad = |dst: &mut BufWriter<File>, from: u64, to: u64| -> Result<(), QuantizeError> {
            dst.write_all(&vec![0u8; (to - from) as usize]).map_err(io)
        };
        pad(&mut dst, pos, data_start)?;
        pos = data_start;
        let bytes_total: u64 = plan.iter().map(|p| p.out_bytes).sum();
        let mut bytes_done = 0u64;
        let tensors_total = plan.len() as u64;
        for (n, p) in plan.iter().enumerate() {
            if control.cancelled() {
                return Err(QuantizeError::Cancelled);
            }
            pad(&mut dst, pos, data_start + p.out_offset)?;
            let src_start = header.data_region_start + p.info.offset;
            let im = match p.action {
                Action::Convert(t) if uses_imatrix(t) => imatrix_for(imatrix, &p.name, &p.info)?,
                _ => None,
            };
            let tensor_start = bytes_done;
            let mut batch = |written: u64| -> Result<(), QuantizeError> {
                bytes_done += written;
                if control.cancelled() {
                    return Err(QuantizeError::Cancelled);
                }
                if let Some(f) = control.progress {
                    f(&Progress {
                        tensor: p.name.clone(),
                        tensors_done: n as u64,
                        tensors_total,
                        bytes_done,
                        bytes_total,
                    });
                }
                Ok(())
            };
            write_tensor(&mut src, &mut dst, p, src_start, im, threads, &mut batch)?;
            bytes_done = tensor_start + p.out_bytes;
            if let Some(f) = control.progress {
                f(&Progress {
                    tensor: p.name.clone(),
                    tensors_done: n as u64 + 1,
                    tensors_total,
                    bytes_done,
                    bytes_total,
                });
            }
            pos = data_start + p.out_offset + p.out_bytes;
        }
        dst.flush().map_err(io)
    })();
    match result {
        Ok(()) => std::fs::rename(&tmp, dst_path).map_err(io)?,
        Err(e) => {
            let _ = std::fs::remove_file(&tmp);
            return Err(e);
        }
    }
    Ok(report)
}

#[cfg(test)]
mod split_tests {
    use super::*;
    use std::collections::BTreeMap;

    fn header(meta: &[(&str, u32)]) -> GgufHeader {
        GgufHeader {
            version: 3,
            metadata: meta
                .iter()
                .map(|(k, v)| (k.to_string(), GgufValue::U32(*v)))
                .collect::<BTreeMap<_, _>>(),
            tensors: BTreeMap::new(),
            alignment: 32,
            data_region_start: 0,
        }
    }

    #[test]
    fn a_split_shard_is_refused_and_a_whole_file_is_not() {
        assert!(refuse_split(&header(&[])).is_ok());
        assert!(refuse_split(&header(&[("split.count", 1)])).is_ok());
        let e = refuse_split(&header(&[("split.count", 5), ("split.no", 1)])).unwrap_err();
        let m = e.to_string();
        assert!(
            m.contains("shard 2 of a 5-file") && m.contains("--merge"),
            "{m}"
        );
    }
}
