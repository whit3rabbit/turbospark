//! Weight loading for Underdog Woof, mirroring `underdog_woof/runtime.py`
//! (upstream revision `be702960539777fc3f698e54335276a8391b00f9`).
//!
//! `lin_w` and `deq` there are the whole dequantization contract, and both are
//! plain f32 arithmetic, so the loader reproduces them bit for bit:
//!
//! - speech linear: `q.astype(f32) * s.astype(f32)[:, None]`, one IEEE f32
//!   multiply per element (`q` int8 `[out, in]`, `s` per-row f32), or a plain
//!   `w` array when the entry has no `q`. Bias `b` widens to f32.
//! - decoder tensor: a plain array widens to f32, or grouped int8
//!   (`q8`, `s`, `shape`, `axis`, `g`): groups of `g` run along the flattened
//!   non-`axis` dimensions, each with its own scale, then the axis moves back.
//!
//! The geometry is checked against the checkpoint's own `cfg` and refused when
//! it disagrees, instead of loading a shape nothing has been verified against.

use std::collections::BTreeMap;

use serde_json::Value;
use turbospark_model_io::hash_data;

use super::npz::Npz;
use crate::{Result, SpeechError};

/// The checkpoint's `speech.cfg` block.
#[derive(Debug, Clone, PartialEq)]
pub struct SpeechConfig {
    /// Transformer layers.
    pub layers: usize,
    /// Model width.
    pub d: usize,
    /// Attention heads.
    pub heads: usize,
    /// Feed-forward width (gate and up are packed, so the weight has `2 * ff` rows).
    pub ff: usize,
    /// Flow-head width.
    pub hw: usize,
    /// Flow-head depth (residual blocks).
    pub depth: usize,
    /// Latent channels per frame.
    pub latent: usize,
    pub rope_base: f32,
}

/// A dequantized linear: `w` is row-major `[rows, cols]` (HF `[out, in]`).
#[derive(Debug, Clone)]
pub struct Linear {
    pub rows: usize,
    pub cols: usize,
    pub w: Vec<f32>,
    pub b: Option<Vec<f32>>,
}

/// A dense f32 tensor of any rank, row-major.
#[derive(Debug, Clone)]
pub struct Dense {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

/// Everything the speech stage needs, dequantized to f32 at load.
#[derive(Debug, Clone)]
pub struct SpeechWeights {
    pub cfg: SpeechConfig,
    /// Per layer: packed qkv, attention output, packed gate+up, down.
    pub layers: Vec<[Linear; 4]>,
    pub ce: Linear,
    pub ada: Linear,
    /// Per flow-head block: up, down.
    pub hblocks: Vec<[Linear; 2]>,
    pub lat_in: Linear,
    pub h_inp: Linear,
    pub h_out: Linear,
    pub text_emb: Dense,
    pub bos: Dense,
}

/// One tensor under its frozen-reference name.
pub struct NamedTensor<'a> {
    pub name: String,
    pub shape: Vec<usize>,
    pub data: &'a [f32],
}

fn bad(field: &str, why: impl Into<String>) -> SpeechError {
    SpeechError::BadConfig {
        field: field.to_string(),
        why: why.into(),
    }
}

fn tensor_err(name: &str, why: impl Into<String>) -> SpeechError {
    SpeechError::Tensor {
        name: name.to_string(),
        why: why.into(),
    }
}

/// The archive key an `{"@": key}` node points at.
fn reference(node: &Value) -> Option<&str> {
    let map = node.as_object()?;
    if map.len() == 1 {
        map.get("@")?.as_str()
    } else {
        None
    }
}

fn field<'a>(node: &'a Value, key: &str, ctx: &str) -> Result<&'a Value> {
    node.get(key)
        .ok_or_else(|| tensor_err(ctx, format!("entry has no `{key}`")))
}

fn array_ref<'a>(node: &'a Value, key: &str, ctx: &str) -> Result<&'a str> {
    reference(field(node, key, ctx)?)
        .ok_or_else(|| tensor_err(ctx, format!("`{key}` is not an array reference")))
}

fn usize_of(node: &Value, field_name: &str) -> Result<usize> {
    node.as_u64()
        .and_then(|v| usize::try_from(v).ok())
        .ok_or_else(|| bad(field_name, "not a non-negative integer"))
}

impl SpeechConfig {
    fn from_tree(cfg: &Value) -> Result<Self> {
        let get = |k: &str| -> Result<usize> {
            usize_of(
                cfg.get(k)
                    .ok_or_else(|| bad("speech.cfg", format!("missing `{k}`")))?,
                k,
            )
        };
        let out = Self {
            layers: get("L")?,
            d: get("d")?,
            heads: get("H")?,
            ff: get("ff")?,
            hw: get("hw")?,
            depth: get("depth")?,
            latent: get("latent")?,
            rope_base: cfg
                .get("rope_base")
                .and_then(Value::as_f64)
                .ok_or_else(|| bad("speech.cfg", "missing `rope_base`"))?
                as f32,
        };
        if out.heads == 0 || out.d % out.heads != 0 {
            return Err(bad("speech.cfg", "d is not a multiple of H"));
        }
        Ok(out)
    }
}

/// Mirror of `lin_w`: `ctx` is the dotted name used in error messages.
pub fn lin_w(npz: &Npz, entry: &Value, ctx: &str) -> Result<Linear> {
    let (rows, cols, w) = if entry.get("q").is_some() {
        let q_name = array_ref(entry, "q", ctx)?;
        let s_name = array_ref(entry, "s", ctx)?;
        let q = npz.array(q_name)?;
        let s = npz.array(s_name)?.to_f32();
        if q.shape.len() != 2 || s.len() != q.shape[0] {
            return Err(tensor_err(
                ctx,
                format!(
                    "int8 weight {:?} does not match {} row scales",
                    q.shape,
                    s.len()
                ),
            ));
        }
        let (rows, cols) = (q.shape[0], q.shape[1]);
        let q = q.to_i8(q_name)?;
        let mut w = Vec::with_capacity(q.len());
        for (row, scale) in q.chunks_exact(cols).zip(&s) {
            w.extend(row.iter().map(|&v| f32::from(v) * scale));
        }
        (rows, cols, w)
    } else {
        let name = array_ref(entry, "w", ctx)?;
        let a = npz.array(name)?;
        if a.shape.len() != 2 {
            return Err(tensor_err(
                ctx,
                format!("weight is rank {}, expected 2", a.shape.len()),
            ));
        }
        (a.shape[0], a.shape[1], a.to_f32())
    };
    let b = match entry.get("b") {
        None => None,
        Some(node) => {
            let name =
                reference(node).ok_or_else(|| tensor_err(ctx, "`b` is not an array reference"))?;
            let b = npz.array(name)?.to_f32();
            if b.len() != rows {
                return Err(tensor_err(
                    ctx,
                    format!("bias has {} values for {rows} rows", b.len()),
                ));
            }
            Some(b)
        }
    };
    Ok(Linear { rows, cols, w, b })
}

/// Mirror of `deq`: a plain array widens to f32, a grouped-int8 dict dequantizes.
pub fn deq(npz: &Npz, node: &Value, ctx: &str) -> Result<Dense> {
    if let Some(name) = reference(node) {
        let a = npz.array(name)?;
        return Ok(Dense {
            shape: a.shape.clone(),
            data: a.to_f32(),
        });
    }
    let q_name = array_ref(node, "q8", ctx)?;
    let s_name = array_ref(node, "s", ctx)?;
    let shape: Vec<usize> = field(node, "shape", ctx)?
        .as_array()
        .ok_or_else(|| tensor_err(ctx, "`shape` is not an array"))?
        .iter()
        .map(|v| usize_of(v, "shape"))
        .collect::<Result<_>>()?;
    let axis = usize_of(field(node, "axis", ctx)?, "axis")?;
    let g = usize_of(field(node, "g", ctx)?, "g")?;
    if axis >= shape.len() || g == 0 {
        return Err(tensor_err(
            ctx,
            format!("axis {axis} / group {g} invalid for shape {shape:?}"),
        ));
    }
    let q = npz.array(q_name)?;
    let sc = npz.array(s_name)?;
    if q.shape.len() != 2 || sc.shape.len() != 2 || sc.shape[0] != q.shape[0] {
        return Err(tensor_err(
            ctx,
            format!("q8 {:?} and scales {:?} disagree", q.shape, sc.shape),
        ));
    }
    let (rows, n, ng) = (q.shape[0], q.shape[1], sc.shape[1]);
    let rest: Vec<usize> = (0..shape.len())
        .filter(|&d| d != axis)
        .map(|d| shape[d])
        .collect();
    if rows != shape[axis] || n != rest.iter().product::<usize>() || ng * g < n {
        return Err(tensor_err(
            ctx,
            format!(
                "q8 {:?} with {ng} groups of {g} does not fit shape {shape:?} on axis {axis}",
                q.shape
            ),
        ));
    }
    // Row-major strides of the output, then the offset of every flattened
    // non-axis position, so the axis can move back without a transpose pass.
    let mut stride = vec![1usize; shape.len()];
    for d in (0..shape.len() - 1).rev() {
        stride[d] = stride[d + 1] * shape[d + 1];
    }
    let rest_dims: Vec<usize> = (0..shape.len()).filter(|&d| d != axis).collect();
    let mut offset = vec![0usize; n];
    for (k, off) in offset.iter_mut().enumerate() {
        let mut rem = k;
        for &d in rest_dims.iter().rev() {
            *off += (rem % shape[d]) * stride[d];
            rem /= shape[d];
        }
    }
    let scales = sc.to_f32();
    let qv = q.to_i8(q_name)?;
    let mut data = vec![0f32; rows * n];
    for r in 0..rows {
        for k in 0..n {
            data[r * stride[axis] + offset[k]] = f32::from(qv[r * n + k]) * scales[r * ng + k / g];
        }
    }
    Ok(Dense { shape, data })
}

fn dense(npz: &Npz, node: &Value, ctx: &str) -> Result<Dense> {
    let name = reference(node).ok_or_else(|| tensor_err(ctx, "not an array reference"))?;
    let a = npz.array(name)?;
    Ok(Dense {
        shape: a.shape.clone(),
        data: a.to_f32(),
    })
}

fn expect_shape(l: &Linear, name: &str, rows: usize, cols: usize) -> Result<()> {
    if (l.rows, l.cols) != (rows, cols) {
        return Err(tensor_err(
            name,
            format!(
                "shape [{}, {}], expected [{rows}, {cols}] from speech.cfg",
                l.rows, l.cols
            ),
        ));
    }
    Ok(())
}

impl SpeechWeights {
    /// Load and dequantize the speech stage. Refuses a quantization scheme
    /// other than the verified round-to-nearest int8 and any geometry that
    /// disagrees with the checkpoint's own `cfg`.
    pub fn load(npz: &Npz) -> Result<Self> {
        let tree = npz.tree()?;
        match tree.get("int8_method").and_then(Value::as_str) {
            Some("rtn") => {}
            other => {
                return Err(SpeechError::Unsupported {
                    why: format!("int8_method {other:?}; only `rtn` has been verified"),
                })
            }
        }
        let sp = tree
            .get("speech")
            .ok_or_else(|| bad("__tree__", "no `speech` block"))?;
        let cfg = SpeechConfig::from_tree(sp.get("cfg").ok_or_else(|| bad("speech", "no cfg"))?)?;
        let (d, ff, hw, latent) = (cfg.d, cfg.ff, cfg.hw, cfg.latent);

        let layer_nodes = sp
            .get("layers")
            .and_then(Value::as_array)
            .ok_or_else(|| bad("speech.layers", "not a list"))?;
        if layer_nodes.len() != cfg.layers {
            return Err(bad(
                "speech.layers",
                format!("{} layers, cfg.L is {}", layer_nodes.len(), cfg.layers),
            ));
        }
        let mut layers = Vec::with_capacity(cfg.layers);
        for (i, node) in layer_nodes.iter().enumerate() {
            let items = node
                .as_array()
                .filter(|a| a.len() == 4)
                .ok_or_else(|| bad(&format!("speech.layers.{i}"), "expected 4 linears"))?;
            let mut four = Vec::with_capacity(4);
            for (j, e) in items.iter().enumerate() {
                four.push(lin_w(npz, e, &format!("layers.{i}.{j}"))?);
            }
            expect_shape(&four[0], &format!("layers.{i}.0"), 3 * d, d)?;
            expect_shape(&four[1], &format!("layers.{i}.1"), d, d)?;
            expect_shape(&four[2], &format!("layers.{i}.2"), 2 * ff, d)?;
            expect_shape(&four[3], &format!("layers.{i}.3"), d, ff)?;
            if four.iter().any(|l| l.b.is_some()) {
                return Err(tensor_err(
                    &format!("layers.{i}"),
                    "layer linears are bias-free in the reference",
                ));
            }
            layers.push(<[Linear; 4]>::try_from(four).expect("four linears"));
        }

        let one = |key: &str| lin_w(npz, sp.get(key).unwrap_or(&Value::Null), key);
        let ce = one("ce")?;
        let ada = one("ada")?;
        let lat_in = one("lat_in")?;
        let h_inp = one("h_inp")?;
        let h_out = one("h_out")?;
        expect_shape(&ce, "ce", hw + 1, d)?;
        expect_shape(&ada, "ada", 3 * hw * cfg.depth + 2 * hw, hw)?;
        expect_shape(&lat_in, "lat_in", d, latent)?;
        expect_shape(&h_inp, "h_inp", hw, latent)?;
        expect_shape(&h_out, "h_out", latent, hw)?;
        for (n, l) in [
            ("ce", &ce),
            ("ada", &ada),
            ("lat_in", &lat_in),
            ("h_inp", &h_inp),
            ("h_out", &h_out),
        ] {
            if l.b.is_none() {
                return Err(tensor_err(n, "missing bias"));
            }
        }

        let h_nodes = sp
            .get("hblocks")
            .and_then(Value::as_array)
            .ok_or_else(|| bad("speech.hblocks", "not a list"))?;
        if h_nodes.len() != cfg.depth {
            return Err(bad(
                "speech.hblocks",
                format!("{} blocks, cfg.depth is {}", h_nodes.len(), cfg.depth),
            ));
        }
        let mut hblocks = Vec::with_capacity(cfg.depth);
        for (i, node) in h_nodes.iter().enumerate() {
            let items = node
                .as_array()
                .filter(|a| a.len() == 2)
                .ok_or_else(|| bad(&format!("speech.hblocks.{i}"), "expected 2 linears"))?;
            let up = lin_w(npz, &items[0], &format!("hblocks.{i}.0"))?;
            let down = lin_w(npz, &items[1], &format!("hblocks.{i}.1"))?;
            expect_shape(&up, &format!("hblocks.{i}.0"), 2 * hw, hw)?;
            expect_shape(&down, &format!("hblocks.{i}.1"), hw, 2 * hw)?;
            if up.b.is_none() || down.b.is_none() {
                return Err(tensor_err(&format!("hblocks.{i}"), "missing bias"));
            }
            hblocks.push([up, down]);
        }

        let text_emb = dense(npz, sp.get("text_emb").unwrap_or(&Value::Null), "text_emb")?;
        let bos = dense(npz, sp.get("bos").unwrap_or(&Value::Null), "bos")?;
        if text_emb.shape.len() != 2 || text_emb.shape[1] != d || bos.shape != [d] {
            return Err(tensor_err(
                "text_emb",
                format!(
                    "{:?} / bos {:?} do not fit d = {d}",
                    text_emb.shape, bos.shape
                ),
            ));
        }
        Ok(Self {
            cfg,
            layers,
            ce,
            ada,
            hblocks,
            lat_in,
            h_inp,
            h_out,
            text_emb,
            bos,
        })
    }

    /// Every tensor under the name the frozen reference uses: `<path>.weight`,
    /// plus `<path>.bias` when there is one, then `text_emb` and `bos`.
    pub fn named_tensors(&self) -> Vec<NamedTensor<'_>> {
        fn pair<'a>(out: &mut Vec<NamedTensor<'a>>, path: String, l: &'a Linear) {
            out.push(NamedTensor {
                name: format!("{path}.weight"),
                shape: vec![l.rows, l.cols],
                data: &l.w,
            });
            if let Some(b) = &l.b {
                out.push(NamedTensor {
                    name: format!("{path}.bias"),
                    shape: vec![l.rows],
                    data: b,
                });
            }
        }
        let mut out = Vec::new();
        for (i, layer) in self.layers.iter().enumerate() {
            for (j, l) in layer.iter().enumerate() {
                pair(&mut out, format!("layers.{i}.{j}"), l);
            }
        }
        for (name, l) in [
            ("ce", &self.ce),
            ("ada", &self.ada),
            ("lat_in", &self.lat_in),
            ("h_inp", &self.h_inp),
            ("h_out", &self.h_out),
        ] {
            pair(&mut out, name.to_string(), l);
        }
        for (i, block) in self.hblocks.iter().enumerate() {
            for (j, l) in block.iter().enumerate() {
                pair(&mut out, format!("hblocks.{i}.{j}"), l);
            }
        }
        out.push(NamedTensor {
            name: "text_emb".into(),
            shape: self.text_emb.shape.clone(),
            data: &self.text_emb.data,
        });
        out.push(NamedTensor {
            name: "bos".into(),
            shape: self.bos.shape.clone(),
            data: &self.bos.data,
        });
        out
    }
}

/// What the frozen reference records per tensor: dtype is always float32.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorRecord {
    pub shape: Vec<usize>,
    pub bytes: usize,
    pub sha256: String,
}

/// SHA-256 over the contiguous little-endian f32 bytes, as `x.tobytes()` gives.
pub fn tensor_record(t: &NamedTensor<'_>) -> TensorRecord {
    let mut raw = Vec::with_capacity(t.data.len() * 4);
    for v in t.data {
        raw.extend_from_slice(&v.to_le_bytes());
    }
    TensorRecord {
        shape: t.shape.clone(),
        bytes: raw.len(),
        sha256: hash_data(&raw),
    }
}

/// Records for every tensor, keyed by name.
pub fn tensor_records(w: &SpeechWeights) -> BTreeMap<String, TensorRecord> {
    w.named_tensors()
        .iter()
        .map(|t| (t.name.clone(), tensor_record(t)))
        .collect()
}

/// SHA-256 of the canonical JSON the reference hashes:
/// `json.dumps(values, sort_keys=True, separators=(',', ':'))` with each value
/// `{"bytes", "dtype", "sha256", "shape"}` (sorted keys, float32 dtype).
pub fn tensor_values_digest(records: &BTreeMap<String, TensorRecord>) -> String {
    let mut s = String::from("{");
    for (i, (name, r)) in records.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        let shape = r
            .shape
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(",");
        s.push_str(&format!(
            "\"{name}\":{{\"bytes\":{},\"dtype\":\"float32\",\"sha256\":\"{}\",\"shape\":[{shape}]}}",
            r.bytes, r.sha256
        ));
    }
    s.push('}');
    hash_data(s.as_bytes())
}
