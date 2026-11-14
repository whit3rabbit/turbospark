//! Quantize-on-install: present an UNQUANTIZED safetensors checkpoint to the
//! MLX walk as the exact checkpoint `mlx_lm.convert` would have published.
//!
//! **Why a source layer and not a writer arm.** Every writer in
//! `gemma4_checkpoint` already ingests MLX-quantized shards, and repack's
//! both-writers rule (`crates/repack/AGENTS.md`) says a new arm must exist in
//! the streamed and the one-shot writer alike or a fixture tests a path no
//! real install takes. Rewriting the SOURCE instead (headers plus a
//! [`RangeSource`] that serves synthesized bytes) leaves every writer, the
//! MTP path and the vision path untouched, and makes the claim testable on
//! bytes: the virtual view of `mlx-community/...-bf16` must equal the real
//! `mlx-community/...-4bit` file tensor for tensor
//! (`tests/mlx_quantize_parity_network.rs`).
//!
//! What the view does, per tensor of the `language_model.` trunk:
//!
//! - A rank-2 float `.weight` whose columns are whole 64-element groups is
//!   replaced by the `U32` weight plus BF16 `.scales` / `.biases` triple
//!   [`compute::quant_mlx`] produces, which is `mx.quantize` byte for byte.
//! - For a checkpoint in the HF transformers layout, the five trunk RMSNorm
//!   kinds get `+1`, which mlx-vlm's `sanitize` bakes in and the runtime
//!   expects (`docs/MTP.md`). `linear_attn.norm` does NOT, and neither does
//!   anything under `mtp.`. Both relations were read off the real HF and MLX
//!   copies of the same weights, not remembered.
//! - Everything else (norms in MLX layout, `conv1d`, `A_log`, `dt_bias`, the
//!   vision tower, the MTP head) is served verbatim and takes the walk's own
//!   narrowing path.
//!
//! Only `QwenGdnDense` is admitted ([`family_admits_quantize_on_install`]).
//! Widening it needs that family's own real gate, not this module's say-so.

use std::cell::RefCell;
use std::collections::BTreeMap;

use model_io::ModelFamily;

use crate::ranged_download::{DownloadError, RangeSource};
use crate::safetensors_header::{SafetensorsHeader, TensorInfo};

/// The only group size quantize-on-install writes: the one the 4- and 8-bit
/// kernels dispatch.
pub const QUANTIZE_ON_INSTALL_GROUP: u32 = 64;

/// Bit widths quantize-on-install can write (both have kernels at group 64).
pub fn is_supported_quantize_bits(bits: u32) -> bool {
    matches!(bits, 4 | 8)
}

/// Families whose unquantized checkpoints may be quantized on install.
pub fn family_admits_quantize_on_install(family: ModelFamily) -> bool {
    family == ModelFamily::QwenGdnDense
}

/// Where a qwen3_5 checkpoint's tensors live.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceLayout {
    /// `language_model.model.*`, norms already shifted (mlx-community bf16).
    Mlx,
    /// `model.language_model.*` and a bare `lm_head.*` (a transformers
    /// export). Norms are zero-centred and need the shift.
    HfTransformers,
}

const HF_PREFIXES: [(&str, &str); 2] = [
    ("model.language_model.", "language_model.model."),
    ("lm_head.", "language_model.lm_head."),
];

/// Rename a qwen3_5 checkpoint's HF transformers trunk names onto the MLX
/// spelling the walk classifies, across ALL shards, and say which layout it
/// was. `model.visual.` is left for `canonicalize_vision_header`, and
/// `mtp.` is already the walk's spelling.
///
/// A checkpoint mixing the two layouts is refused: one half would get the
/// norm shift and the other would not.
pub fn canonicalize_qwen35_hf_headers(
    headers: &mut [SafetensorsHeader],
) -> Result<SourceLayout, String> {
    let any = |pred: &dyn Fn(&str) -> bool| {
        headers
            .iter()
            .any(|h| h.tensors.keys().any(|k| pred(k.as_str())))
    };
    let has_hf = any(&|k| HF_PREFIXES.iter().any(|(src, _)| k.starts_with(src)));
    let has_mlx = any(&|k| k.starts_with("language_model."));
    if has_hf && has_mlx {
        return Err(
            "checkpoint mixes the HF (model.language_model.*) and MLX (language_model.*) \
             tensor namespaces"
                .into(),
        );
    }
    if !has_hf {
        return Ok(SourceLayout::Mlx);
    }
    for header in headers.iter_mut() {
        let mut renamed = BTreeMap::new();
        for (name, info) in std::mem::take(&mut header.tensors) {
            let canonical = HF_PREFIXES
                .iter()
                .find_map(|(src, dst)| name.strip_prefix(src).map(|t| format!("{dst}{t}")))
                .unwrap_or(name);
            if renamed.insert(canonical.clone(), info).is_some() {
                return Err(format!("renaming creates duplicate tensor {canonical}"));
            }
        }
        header.tensors = renamed;
    }
    Ok(SourceLayout::HfTransformers)
}

/// Trunk RMSNorms that carry mlx-vlm's `+1`, matched on the canonical name.
fn takes_norm_shift(name: &str) -> bool {
    name.starts_with("language_model.")
        && (name.ends_with(".input_layernorm.weight")
            || name.ends_with(".post_attention_layernorm.weight")
            || name.ends_with(".self_attn.q_norm.weight")
            || name.ends_with(".self_attn.k_norm.weight")
            || name == "language_model.model.norm.weight")
}

/// A trunk matrix this view quantizes.
fn quantizes(name: &str, info: &TensorInfo) -> bool {
    name.starts_with("language_model.")
        && name.ends_with(".weight")
        && info.shape.len() == 2
        && matches!(info.dtype.as_str(), "BF16" | "F16" | "F32")
        && info.shape[1] % QUANTIZE_ON_INSTALL_GROUP as u64 == 0
}

fn elem_bytes(dtype: &str) -> Result<usize, String> {
    match dtype {
        "BF16" | "F16" => Ok(2),
        "F32" => Ok(4),
        d => Err(format!("unsupported float dtype {d}")),
    }
}

fn decode_into(dtype: &str, raw: &[u8], out: &mut Vec<f32>) {
    out.clear();
    match dtype {
        "BF16" => out.extend(
            raw.chunks_exact(2)
                .map(|c| compute::bf16_to_f32(u16::from_le_bytes([c[0], c[1]]))),
        ),
        "F16" => out.extend(
            raw.chunks_exact(2)
                .map(|c| compute::quant_1bit::f16_to_f32(u16::from_le_bytes([c[0], c[1]]))),
        ),
        "F32" => out.extend(
            raw.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])),
        ),
        _ => unreachable!("dtype checked by quantizes()"),
    }
}

/// The mean of every trunk `input_layernorm` / `post_attention_layernorm`
/// element, the cross-check on the layout decision. mlx layout sits near 1,
/// a zero-centred export near 0 (measured: +1.03 / -0.11 per kind against
/// +0.03 / -0.11 + 1 on the same weights). The name prefix DECIDES the shift;
/// this only refuses a checkpoint whose bytes contradict its names
/// (precedent: `fish_drama3`'s aggregate-mean refusal).
pub fn check_norm_layout(
    headers: &[SafetensorsHeader],
    sources: &[&dyn RangeSource],
    layout: SourceLayout,
) -> Result<f32, String> {
    let mut sum = 0f64;
    let mut n = 0usize;
    let mut buf = Vec::new();
    for (h, s) in headers.iter().zip(sources) {
        for (name, info) in &h.tensors {
            if !(name.starts_with("language_model.")
                && (name.ends_with(".input_layernorm.weight")
                    || name.ends_with(".post_attention_layernorm.weight")))
            {
                continue;
            }
            elem_bytes(&info.dtype).map_err(|e| format!("{name}: {e}"))?;
            let (a, b) = h
                .absolute_range(name)
                .ok_or_else(|| format!("{name}: range"))?;
            let raw = s.read_range(a, b).map_err(|e| format!("{name}: {e}"))?;
            decode_into(&info.dtype, &raw, &mut buf);
            sum += buf.iter().map(|&v| v as f64).sum::<f64>();
            n += buf.len();
        }
    }
    if n == 0 {
        return Err("no trunk layernorms found to check the norm layout against".into());
    }
    let mean = (sum / n as f64) as f32;
    let looks_shifted = mean > 0.5;
    match (layout, looks_shifted) {
        (SourceLayout::Mlx, true) | (SourceLayout::HfTransformers, false) => Ok(mean),
        (SourceLayout::Mlx, false) => Err(format!(
            "tensor names are the MLX layout (norms expected pre-shifted, mean near 1) but the \
             trunk layernorms average {mean:.3}; refusing rather than guessing the shift"
        )),
        (SourceLayout::HfTransformers, true) => Err(format!(
            "tensor names are the HF layout (norms expected zero-centred, mean near 0) but the \
             trunk layernorms average {mean:.3}; refusing rather than applying a second +1"
        )),
    }
}

#[derive(Debug, Clone)]
enum Item {
    /// Serve the source bytes at this absolute range.
    Verbatim { start: u64 },
    /// Serve the source tensor with every element `+1`, in its own dtype.
    Shifted { start: u64, end: u64, dtype: String },
    /// One plane of the quantized triple for `base`.
    Plane { base: String, plane: Plane },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Plane {
    Weight,
    Scales,
    Biases,
}

struct Quantized {
    weight: Vec<u8>,
    scales: Vec<u8>,
    biases: Vec<u8>,
}

/// One shard as the walk should see it.
pub struct QuantizedShard<'a> {
    header: SafetensorsHeader,
    inner: &'a dyn RangeSource,
    /// `(virtual absolute start, end, item)`, sorted by start.
    items: Vec<(u64, u64, Item)>,
    /// `base -> (source dtype, rows, cols, source abs start, end)`.
    sources: BTreeMap<String, (String, usize, usize, u64, u64)>,
    bits: u32,
    /// The last base quantized. The walk reads a triple's planes back to
    /// back, so one entry turns three quantizations into one.
    cache: RefCell<Option<(String, std::rc::Rc<Quantized>)>>,
}

impl<'a> QuantizedShard<'a> {
    /// The virtual header to register in `Gemma4Shards`.
    pub fn header(&self) -> &SafetensorsHeader {
        &self.header
    }

    /// The source tensors this shard quantizes, for the install log.
    pub fn quantized_count(&self) -> usize {
        self.sources.len()
    }

    fn planes(&self, base: &str) -> Result<std::rc::Rc<Quantized>, DownloadError> {
        if let Some((b, q)) = self.cache.borrow().as_ref() {
            if b == base {
                return Ok(q.clone());
            }
        }
        let (dtype, rows, cols, a, b) = self
            .sources
            .get(base)
            .cloned()
            .ok_or_else(|| DownloadError::Request(format!("no source for {base}")))?;
        let raw = self.inner.read_range(a, b)?;
        let q = std::rc::Rc::new(quantize_matrix(&dtype, &raw, rows, cols, self.bits));
        *self.cache.borrow_mut() = Some((base.to_string(), q.clone()));
        Ok(q)
    }
}

/// Quantize a row-major `rows x cols` matrix, rows split across threads.
fn quantize_matrix(dtype: &str, raw: &[u8], rows: usize, cols: usize, bits: u32) -> Quantized {
    let eb = if dtype == "F32" { 4 } else { 2 };
    let row_w = cols * bits as usize / 8;
    let row_g = cols / QUANTIZE_ON_INSTALL_GROUP as usize * 2;
    let mut weight = vec![0u8; rows * row_w];
    let mut scales = vec![0u8; rows * row_g];
    let mut biases = vec![0u8; rows * row_g];
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .min(rows.max(1));
    let per = rows.div_ceil(threads);
    std::thread::scope(|s| {
        for (((w, sc), bi), r) in weight
            .chunks_mut(per * row_w)
            .zip(scales.chunks_mut(per * row_g))
            .zip(biases.chunks_mut(per * row_g))
            .zip(raw.chunks(per * cols * eb))
        {
            s.spawn(move || {
                let mut row = Vec::with_capacity(cols);
                for (i, src) in r.chunks_exact(cols * eb).enumerate() {
                    decode_into(dtype, src, &mut row);
                    let (p, sv, bv) = if bits == 4 {
                        let q = compute::quant_mlx::quantize_int4_affine_mlx(&row);
                        (q.packed, q.scales, q.biases)
                    } else {
                        let q = compute::quant_mlx::quantize_int8_affine_mlx(&row);
                        (q.packed, q.scales, q.biases)
                    };
                    w[i * row_w..(i + 1) * row_w].copy_from_slice(&p);
                    for (j, (s16, b16)) in sv.iter().zip(&bv).enumerate() {
                        let o = i * row_g + j * 2;
                        sc[o..o + 2].copy_from_slice(&s16.to_le_bytes());
                        bi[o..o + 2].copy_from_slice(&b16.to_le_bytes());
                    }
                }
            });
        }
    });
    Quantized {
        weight,
        scales,
        biases,
    }
}

fn shift_bytes(dtype: &str, raw: &[u8]) -> Vec<u8> {
    match dtype {
        "BF16" => raw
            .chunks_exact(2)
            .flat_map(|c| {
                let v = compute::bf16_to_f32(u16::from_le_bytes([c[0], c[1]])) + 1.0;
                compute::f32_to_bf16(v).to_le_bytes()
            })
            .collect(),
        "F16" => raw
            .chunks_exact(2)
            .flat_map(|c| {
                let v = compute::quant_1bit::f16_to_f32(u16::from_le_bytes([c[0], c[1]])) + 1.0;
                compute::quant_1bit::f32_to_f16(v).to_le_bytes()
            })
            .collect(),
        "F32" => raw
            .chunks_exact(4)
            .flat_map(|c| (f32::from_le_bytes([c[0], c[1], c[2], c[3]]) + 1.0).to_le_bytes())
            .collect(),
        _ => unreachable!("dtype checked at build"),
    }
}

impl RangeSource for QuantizedShard<'_> {
    fn read_range(&self, start: u64, end_exclusive: u64) -> Result<Vec<u8>, DownloadError> {
        let idx = self.items.partition_point(|(s, _, _)| *s <= start);
        let bad = || DownloadError::InvalidRange {
            start,
            end_exclusive,
        };
        let (vs, ve, item) = idx.checked_sub(1).map(|i| &self.items[i]).ok_or_else(bad)?;
        if end_exclusive > *ve || end_exclusive < start {
            return Err(bad());
        }
        let (lo, hi) = ((start - vs) as usize, (end_exclusive - vs) as usize);
        let bytes = match item {
            Item::Verbatim { start: a } => {
                return self.inner.read_range(a + lo as u64, a + hi as u64)
            }
            Item::Shifted {
                start: a,
                end: b,
                dtype,
            } => shift_bytes(dtype, &self.inner.read_range(*a, *b)?),
            Item::Plane { base, plane } => {
                let q = self.planes(base)?;
                let v = match plane {
                    Plane::Weight => &q.weight,
                    Plane::Scales => &q.scales,
                    Plane::Biases => &q.biases,
                };
                return Ok(v[lo..hi].to_vec());
            }
        };
        Ok(bytes[lo..hi].to_vec())
    }
}

/// Build the quantized view of every shard. `headers` must already be
/// canonicalized (trunk renamed, vision renamed); `layout` comes from
/// [`canonicalize_qwen35_hf_headers`] and only decides the norm shift.
pub fn quantize_on_install_shards<'a>(
    headers: &[SafetensorsHeader],
    sources: &[&'a dyn RangeSource],
    bits: u32,
    layout: SourceLayout,
) -> Result<Vec<QuantizedShard<'a>>, String> {
    if !is_supported_quantize_bits(bits) {
        return Err(format!(
            "quantize-on-install writes 4- or 8-bit at group 64, not {bits}-bit"
        ));
    }
    let mut out = Vec::with_capacity(headers.len());
    for (h, &src) in headers.iter().zip(sources) {
        // A real header never carries an MLX companion next to a float
        // matrix; if one does, this is not the checkpoint the probe saw.
        if let Some(k) = h
            .tensors
            .keys()
            .find(|k| k.ends_with(".scales") || k.ends_with(".biases"))
        {
            return Err(format!(
                "{k}: an unquantized checkpoint should carry no MLX companion planes"
            ));
        }
        let mut tensors = BTreeMap::new();
        let mut items = Vec::new();
        let mut qsources = BTreeMap::new();
        // Virtual data region starts right after an empty header.
        let header_len = 0u64;
        let base_abs = 8 + header_len;
        let mut cursor = 0u64;
        let mut push = |name: String,
                        dtype: &str,
                        shape: Vec<u64>,
                        len: u64,
                        item: Item,
                        tensors: &mut BTreeMap<String, TensorInfo>,
                        items: &mut Vec<(u64, u64, Item)>| {
            tensors.insert(
                name,
                TensorInfo {
                    dtype: dtype.to_string(),
                    shape,
                    data_offsets: (cursor, cursor + len),
                },
            );
            items.push((base_abs + cursor, base_abs + cursor + len, item));
            cursor += len;
        };
        for (name, info) in &h.tensors {
            let (a, b) = h
                .absolute_range(name)
                .ok_or_else(|| format!("{name}: range overflows"))?;
            if quantizes(name, info) {
                let rows = info.shape[0] as usize;
                let cols = info.shape[1] as usize;
                let eb = elem_bytes(&info.dtype)?;
                if (b - a) as usize != rows * cols * eb {
                    return Err(format!("{name}: byte length does not match its shape"));
                }
                let base = name.strip_suffix(".weight").unwrap().to_string();
                let w_cols = (cols * bits as usize / 32) as u64;
                let g = (cols / QUANTIZE_ON_INSTALL_GROUP as usize) as u64;
                let r = rows as u64;
                push(
                    name.clone(),
                    "U32",
                    vec![r, w_cols],
                    r * w_cols * 4,
                    Item::Plane {
                        base: base.clone(),
                        plane: Plane::Weight,
                    },
                    &mut tensors,
                    &mut items,
                );
                for (suffix, plane) in [("scales", Plane::Scales), ("biases", Plane::Biases)] {
                    push(
                        format!("{base}.{suffix}"),
                        "BF16",
                        vec![r, g],
                        r * g * 2,
                        Item::Plane {
                            base: base.clone(),
                            plane,
                        },
                        &mut tensors,
                        &mut items,
                    );
                }
                qsources.insert(base, (info.dtype.clone(), rows, cols, a, b));
            } else if layout == SourceLayout::HfTransformers && takes_norm_shift(name) {
                elem_bytes(&info.dtype).map_err(|e| format!("{name}: {e}"))?;
                push(
                    name.clone(),
                    &info.dtype,
                    info.shape.clone(),
                    b - a,
                    Item::Shifted {
                        start: a,
                        end: b,
                        dtype: info.dtype.clone(),
                    },
                    &mut tensors,
                    &mut items,
                );
            } else {
                // mlx sanitize moves conv1d's kernel axis last: HF writes
                // `[C, 1, K]`, MLX `[C, K, 1]`. The middle axis is 1, so the
                // bytes are identical and only the label moves.
                let shape = if layout == SourceLayout::HfTransformers
                    && name.ends_with(".linear_attn.conv1d.weight")
                    && info.shape.len() == 3
                    && info.shape[1] == 1
                {
                    vec![info.shape[0], info.shape[2], 1]
                } else {
                    info.shape.clone()
                };
                push(
                    name.clone(),
                    &info.dtype,
                    shape,
                    b - a,
                    Item::Verbatim { start: a },
                    &mut tensors,
                    &mut items,
                );
            }
        }
        items.sort_by_key(|(s, _, _)| *s);
        out.push(QuantizedShard {
            header: SafetensorsHeader {
                tensors,
                metadata: h.metadata.clone(),
                header_len,
            },
            inner: src,
            items,
            sources: qsources,
            bits,
            cache: RefCell::new(None),
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ranged_download::MemoryRangeSource;

    /// Build a one-file safetensors blob from `(name, dtype, shape, bytes)`.
    fn blob(tensors: &[(&str, &str, Vec<u64>, Vec<u8>)]) -> (SafetensorsHeader, Vec<u8>) {
        let mut map = BTreeMap::new();
        let mut data = Vec::new();
        for (n, d, s, b) in tensors {
            let start = data.len() as u64;
            data.extend_from_slice(b);
            map.insert(
                n.to_string(),
                TensorInfo {
                    dtype: d.to_string(),
                    shape: s.clone(),
                    data_offsets: (start, data.len() as u64),
                },
            );
        }
        let mut file = vec![0u8; 8];
        file.extend(data);
        (
            SafetensorsHeader {
                tensors: map,
                metadata: None,
                header_len: 0,
            },
            file,
        )
    }

    fn bf16_bytes(v: &[f32]) -> Vec<u8> {
        v.iter()
            .flat_map(|&x| compute::f32_to_bf16(x).to_le_bytes())
            .collect()
    }

    fn matrix(rows: usize, cols: usize) -> Vec<f32> {
        (0..rows * cols)
            .map(|i| (((i * 37) % 101) as f32 - 50.0) / 400.0)
            .collect()
    }

    fn view(
        tensors: &[(&str, &str, Vec<u64>, Vec<u8>)],
    ) -> (Vec<SafetensorsHeader>, Vec<u8>, SourceLayout) {
        let (h, file) = blob(tensors);
        let mut hs = vec![h];
        let layout = canonicalize_qwen35_hf_headers(&mut hs).unwrap();
        (hs, file, layout)
    }

    #[test]
    fn a_matrix_becomes_the_mlx_triple_byte_for_byte() {
        let m = matrix(3, 128);
        let (hs, file, layout) = view(&[(
            "language_model.model.layers.0.mlp.up_proj.weight",
            "BF16",
            vec![3, 128],
            bf16_bytes(&m),
        )]);
        let src = MemoryRangeSource::new(&file);
        let shards = quantize_on_install_shards(&hs, &[&src], 4, layout).unwrap();
        let s = &shards[0];
        let h = s.header();
        let w = &h.tensors["language_model.model.layers.0.mlp.up_proj.weight"];
        assert_eq!((w.dtype.as_str(), w.shape.clone()), ("U32", vec![3, 16]));
        let (a, b) = h
            .absolute_range("language_model.model.layers.0.mlp.up_proj.weight")
            .unwrap();
        let got = s.read_range(a, b).unwrap();
        let (sa, sb) = h
            .absolute_range("language_model.model.layers.0.mlp.up_proj.scales")
            .unwrap();
        let got_s = s.read_range(sa, sb).unwrap();
        for r in 0..3 {
            let row: Vec<f32> = m[r * 128..(r + 1) * 128]
                .iter()
                .map(|&x| compute::bf16_to_f32(compute::f32_to_bf16(x)))
                .collect();
            let q = compute::quant_mlx::quantize_int4_affine_mlx(&row);
            assert_eq!(&got[r * 64..(r + 1) * 64], &q.packed[..]);
            let sc: Vec<u8> = q.scales.iter().flat_map(|v| v.to_le_bytes()).collect();
            assert_eq!(&got_s[r * 4..(r + 1) * 4], &sc[..]);
        }
    }

    #[test]
    fn hf_layout_renames_and_shifts_only_the_trunk_norms() {
        let norm = vec![0.25f32; 64];
        let (hs, file, layout) = view(&[
            (
                "model.language_model.layers.0.input_layernorm.weight",
                "BF16",
                vec![64],
                bf16_bytes(&norm),
            ),
            (
                "model.language_model.layers.0.linear_attn.norm.weight",
                "BF16",
                vec![64],
                bf16_bytes(&norm),
            ),
            ("mtp.norm.weight", "BF16", vec![64], bf16_bytes(&norm)),
            (
                "lm_head.weight",
                "BF16",
                vec![2, 64],
                bf16_bytes(&matrix(2, 64)),
            ),
        ]);
        assert_eq!(layout, SourceLayout::HfTransformers);
        let src = MemoryRangeSource::new(&file);
        let shards = quantize_on_install_shards(&hs, &[&src], 4, layout).unwrap();
        let s = &shards[0];
        let read = |n: &str| {
            let (a, b) = s.header().absolute_range(n).unwrap();
            s.read_range(a, b).unwrap()
        };
        assert_eq!(
            read("language_model.model.layers.0.input_layernorm.weight"),
            bf16_bytes(&[1.25f32; 64])
        );
        assert_eq!(
            read("language_model.model.layers.0.linear_attn.norm.weight"),
            bf16_bytes(&norm)
        );
        assert_eq!(read("mtp.norm.weight"), bf16_bytes(&norm));
        assert_eq!(
            s.header().tensors["language_model.lm_head.weight"].dtype,
            "U32"
        );
    }

    #[test]
    fn mlx_layout_norms_are_not_shifted() {
        let norm = vec![1.25f32; 64];
        let (hs, file, layout) = view(&[(
            "language_model.model.layers.0.input_layernorm.weight",
            "BF16",
            vec![64],
            bf16_bytes(&norm),
        )]);
        assert_eq!(layout, SourceLayout::Mlx);
        let src = MemoryRangeSource::new(&file);
        let shards = quantize_on_install_shards(&hs, &[&src], 4, layout).unwrap();
        let (a, b) = shards[0]
            .header()
            .absolute_range("language_model.model.layers.0.input_layernorm.weight")
            .unwrap();
        assert_eq!(shards[0].read_range(a, b).unwrap(), bf16_bytes(&norm));
    }

    #[test]
    fn a_mixed_namespace_is_refused() {
        let (h, _) = blob(&[
            (
                "model.language_model.norm.weight",
                "BF16",
                vec![64],
                vec![0; 128],
            ),
            (
                "language_model.model.norm.weight",
                "BF16",
                vec![64],
                vec![0; 128],
            ),
        ]);
        assert!(canonicalize_qwen35_hf_headers(&mut [h]).is_err());
    }

    #[test]
    fn a_norm_mean_that_contradicts_the_names_is_refused() {
        let (h, file) = blob(&[(
            "language_model.model.layers.0.input_layernorm.weight",
            "BF16",
            vec![64],
            bf16_bytes(&[0.02f32; 64]),
        )]);
        let src = MemoryRangeSource::new(&file);
        // MLX names, zero-centred bytes: someone skipped the shift.
        assert!(check_norm_layout(std::slice::from_ref(&h), &[&src], SourceLayout::Mlx).is_err());
        assert!(check_norm_layout(&[h], &[&src], SourceLayout::HfTransformers).is_ok());
    }

    #[test]
    fn eight_bit_planes_have_the_mlx_shapes() {
        let m = matrix(2, 64);
        let (hs, file, layout) = view(&[(
            "language_model.model.embed_tokens.weight",
            "BF16",
            vec![2, 64],
            bf16_bytes(&m),
        )]);
        let src = MemoryRangeSource::new(&file);
        let shards = quantize_on_install_shards(&hs, &[&src], 8, layout).unwrap();
        let h = shards[0].header();
        assert_eq!(
            h.tensors["language_model.model.embed_tokens.weight"].shape,
            vec![2, 16]
        );
        assert_eq!(
            h.tensors["language_model.model.embed_tokens.scales"].shape,
            vec![2, 1]
        );
        assert!(quantize_on_install_shards(&hs, &[&src], 6, layout).is_err());
    }
}
