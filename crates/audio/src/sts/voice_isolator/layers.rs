//! Layers of the voice-isolator network (`model.py` in the reference).
//!
//! Token tensors are row-major `[sequences, length, width]` f32. A dual-path
//! block reads the `[bands, frames, width]` grid as `bands` sequences along
//! time, then transposes it to `[frames, bands, width]` for the pass across
//! bands. Linear projections and non-causal attention dispatch to the
//! optional device backend ([`Exec`]); every other operation, the selective
//! scan included, runs on the host.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::rc::Rc;

use rayon::prelude::*;

use super::Loader;
use crate::backend::{
    AttentionShape, ComputeBackend, DType, DeviceWeight, ScanInputs, ScanShape, WeightData,
    WeightEncoding,
};
use crate::nn::{LayerNorm, Linear};
use crate::ops;
use crate::Result;

/// PyTorch's `LayerNorm` default epsilon, which every norm here uses.
pub(crate) const LN_EPS: f32 = 1e-5;
/// Rows per rayon task for the row-wise host operations.
const ROW_CHUNK: usize = 256;
/// Smallest projection (rows x input x output multiply-adds) sent to the
/// device; smaller ones run on the host. Overridable for measurement with
/// `TURBOSPARK_VOICE_ISOLATOR_DEVICE_MIN_MACS`.
fn device_min_macs() -> usize {
    static CELL: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CELL.get_or_init(|| {
        std::env::var("TURBOSPARK_VOICE_ISOLATOR_DEVICE_MIN_MACS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1 << 24)
    })
}

/// Device dispatch for the projections and non-causal attention.
///
/// A device error falls back to the host for that call and is counted, so a
/// test can assert that a device run really ran on the device.
#[derive(Default)]
pub(crate) struct Exec {
    backend: Option<Rc<dyn ComputeBackend>>,
    weights: RefCell<HashMap<usize, Rc<dyn DeviceWeight>>>,
    fallbacks: Cell<usize>,
    /// Wall time inside linear, attention and scan calls (either side).
    times: Cell<[std::time::Duration; 3]>,
}

impl Exec {
    fn timed<R>(&self, slot: usize, f: impl FnOnce() -> R) -> R {
        let started = std::time::Instant::now();
        let out = f();
        let mut t = self.times.get();
        t[slot] += started.elapsed();
        self.times.set(t);
        out
    }

    /// Seconds spent in linear, attention and scan calls since the backend
    /// was set.
    pub(crate) fn times(&self) -> [f64; 3] {
        self.times.get().map(|d| d.as_secs_f64())
    }

    pub(crate) fn new(backend: Option<Rc<dyn ComputeBackend>>) -> Self {
        Self {
            backend,
            ..Self::default()
        }
    }

    pub(crate) fn on_device(&self) -> bool {
        self.backend.is_some()
    }

    pub(crate) fn fallbacks(&self) -> usize {
        self.fallbacks.get()
    }

    fn device_weight(
        &self,
        backend: &dyn ComputeBackend,
        layer: &Linear,
    ) -> Result<Rc<dyn DeviceWeight>> {
        let key = layer.weight.as_ptr() as usize;
        if let Some(weight) = self.weights.borrow().get(&key) {
            return Ok(weight.clone());
        }
        let bytes: Vec<u8> = layer.weight.iter().flat_map(|v| v.to_le_bytes()).collect();
        let weight = backend.load_weight(WeightData {
            name: "voice_isolator_linear",
            shape: &[layer.output, layer.input],
            bytes: &bytes,
            encoding: WeightEncoding::F32,
            dtype: DType::F32,
            scales_dtype: None,
            offsets_dtype: None,
            scales: &[],
            offsets: &[],
            block_scales: &[],
            planned_rows: None,
            tiled_f32: false,
        })?;
        self.weights.borrow_mut().insert(key, weight.clone());
        Ok(weight)
    }

    /// `x [rows, layer.input] -> [rows, layer.output]`.
    pub(crate) fn linear(&self, layer: &Linear, x: &[f32], rows: usize) -> Vec<f32> {
        self.timed(0, || self.linear_inner(layer, x, rows))
    }

    fn linear_inner(&self, layer: &Linear, x: &[f32], rows: usize) -> Vec<f32> {
        // Below a few thousand multiply-adds per weight the device round
        // trip (copy in, dispatch, wait, copy out) costs more than the
        // host GEMM: a stream block's per-band layers see 4 rows.
        let device_worth = rows * layer.input * layer.output >= device_min_macs();
        if let Some(backend) = self.backend.as_ref().filter(|_| device_worth) {
            let out = self.device_weight(backend.as_ref(), layer).and_then(|w| {
                w.linear(
                    x,
                    layer.bias.as_deref(),
                    rows,
                    layer.input,
                    layer.output,
                    DType::F32,
                )
            });
            match out {
                Ok(out) => return out,
                Err(_) => self.fallbacks.set(self.fallbacks.get() + 1),
            }
        }
        layer.forward(x, rows)
    }

    /// The selective scan on the device, or `None` without one (or on a
    /// device error, counted). States above 16 stay on the host.
    pub(crate) fn scan(
        &self,
        inputs: ScanInputs<'_>,
        shape: ScanShape,
    ) -> Option<(Vec<f32>, Vec<f32>)> {
        let backend = self.backend.as_ref()?;
        if shape.states > 16 {
            return None;
        }
        match backend.selective_scan(inputs, shape) {
            Ok(out) => Some(out),
            Err(_) => {
                self.fallbacks.set(self.fallbacks.get() + 1);
                None
            }
        }
    }

    /// Non-causal self attention on the device over position-major
    /// `[batch, len, heads, dim]` inputs, or `None` without a device (or on
    /// a device error, counted).
    fn attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        batch: usize,
        len: usize,
        heads: usize,
        dim: usize,
    ) -> Option<Vec<f32>> {
        let backend = self.backend.as_ref()?;
        let shape = AttentionShape {
            batch,
            queries: len,
            keys: len,
            heads,
            kv_heads: heads,
            dim,
            kv_time_major: false,
            causal: false,
            offset: 0,
        };
        match backend.attention(q, k, v, shape, DType::F32) {
            Ok(out) => Some(out),
            Err(_) => {
                self.fallbacks.set(self.fallbacks.get() + 1);
                None
            }
        }
    }
}

/// Row-wise LayerNorm over `rows` rows, split across threads. Each row is
/// the same sequential reduction as [`ops::layernorm`].
pub(crate) fn layer_norm(norm: &LayerNorm, x: &[f32], rows: usize) -> Vec<f32> {
    let cols = norm.width();
    let mut y = x.to_vec();
    y.par_chunks_mut(cols * ROW_CHUNK).for_each(|chunk| {
        let n = chunk.len() / cols;
        norm.apply(chunk, n);
    });
    debug_assert_eq!(y.len(), rows * cols);
    y
}

/// Erf GELU in place, split across threads.
pub(crate) fn gelu(x: &mut [f32]) {
    x.par_chunks_mut(1 << 14).for_each(ops::gelu_erf);
}

/// `a + b` elementwise into `a`.
fn add_into(a: &mut [f32], b: &[f32]) {
    a.par_chunks_mut(1 << 14)
        .zip(b.par_chunks(1 << 14))
        .for_each(|(a, b)| a.iter_mut().zip(b).for_each(|(a, b)| *a += b));
}

/// `[n, m, w] -> [m, n, w]`.
pub(crate) fn swap_axes(x: &[f32], n: usize, m: usize, w: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    out.par_chunks_mut(n * w).enumerate().for_each(|(j, dst)| {
        for i in 0..n {
            dst[i * w..(i + 1) * w].copy_from_slice(&x[(i * m + j) * w..(i * m + j + 1) * w]);
        }
    });
    out
}

/// Which keys each query sees.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Mask {
    /// Every key.
    Full,
    /// Keys at or before the query.
    Causal,
    /// Keys at most `window - 1` frames before the query, and the query.
    Window(usize),
}

impl Mask {
    fn additive(self, len: usize) -> Option<Vec<f32>> {
        let allowed = |q: usize, k: usize| match self {
            Mask::Full => true,
            Mask::Causal => k <= q,
            Mask::Window(w) => k <= q && q - k < w,
        };
        if self == Mask::Full {
            return None;
        }
        Some(
            (0..len * len)
                .map(|i| {
                    if allowed(i / len, i % len) {
                        0.0
                    } else {
                        f32::NEG_INFINITY
                    }
                })
                .collect(),
        )
    }
}

/// Multi-head self attention over `seqs` sequences of `len` tokens whose
/// fused projection `qkv` is `[seqs * len, 3, heads, dim]`. Rotary positions
/// along the sequence when `rope`; the exclusive variant removes each
/// output's component along its own value. Returns `[seqs * len, heads *
/// dim]`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn attend(
    exec: &Exec,
    qkv: &[f32],
    seqs: usize,
    len: usize,
    heads: usize,
    dim: usize,
    rope: bool,
    mask: Mask,
    exclusive: bool,
) -> Vec<f32> {
    let width = heads * dim;
    let tokens = seqs * len;
    debug_assert_eq!(qkv.len(), tokens * 3 * width);
    // Position-major q, k, v: [seqs, len, heads, dim].
    let mut q = vec![0.0f32; tokens * width];
    let mut k = vec![0.0f32; tokens * width];
    let mut v = vec![0.0f32; tokens * width];
    q.par_chunks_mut(width)
        .zip(k.par_chunks_mut(width))
        .zip(v.par_chunks_mut(width))
        .enumerate()
        .for_each(|(t, ((q, k), v))| {
            let row = &qkv[t * 3 * width..(t + 1) * 3 * width];
            q.copy_from_slice(&row[..width]);
            k.copy_from_slice(&row[width..2 * width]);
            v.copy_from_slice(&row[2 * width..]);
        });
    if rope {
        let (cos, sin) = ops::rope_tables(len, dim, 10_000.0);
        let half = dim / 2;
        let rotate = |x: &mut [f32]| {
            x.par_chunks_mut(width)
                .enumerate()
                .for_each(|(token, row)| {
                    let p = token % len;
                    let (c, s) = (
                        &cos[p * half..(p + 1) * half],
                        &sin[p * half..(p + 1) * half],
                    );
                    for head in row.chunks_exact_mut(dim) {
                        let (lo, hi) = head.split_at_mut(half);
                        for d in 0..half {
                            let (a, b) = (lo[d], hi[d]);
                            lo[d] = a * c[d] - b * s[d];
                            hi[d] = a * s[d] + b * c[d];
                        }
                    }
                });
        };
        rotate(&mut q);
        rotate(&mut k);
    }
    let mut out = exec.timed(1, || {
        let device = if mask == Mask::Full {
            exec.attention(&q, &k, &v, seqs, len, heads, dim)
        } else {
            None
        };
        device.unwrap_or_else(|| host_attention(&q, &k, &v, seqs, len, heads, dim, mask))
    });
    if exclusive {
        out.par_chunks_mut(dim)
            .zip(v.par_chunks(dim))
            .for_each(|(o, v)| {
                let ov: f32 = o.iter().zip(v).map(|(a, b)| a * b).sum();
                let vv: f32 = v.iter().map(|a| a * a).sum();
                let coefficient = ov / vv.max(1e-6);
                o.iter_mut().zip(v).for_each(|(o, v)| *o -= coefficient * v);
            });
    }
    out
}

/// Host SDPA per (sequence, head), output position-major like the device.
#[allow(clippy::too_many_arguments)]
fn host_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seqs: usize,
    len: usize,
    heads: usize,
    dim: usize,
    mask: Mask,
) -> Vec<f32> {
    let width = heads * dim;
    let additive = mask.additive(len);
    let scale = 1.0 / (dim as f32).sqrt();
    let planes: Vec<Vec<f32>> = (0..seqs * heads)
        .into_par_iter()
        .map(|job| {
            let (s, h) = (job / heads, job % heads);
            let base = s * len * width + h * dim;
            let mut qh = vec![0.0f32; len * dim];
            for t in 0..len {
                qh[t * dim..(t + 1) * dim]
                    .copy_from_slice(&q[base + t * width..base + t * width + dim]);
            }
            ops::sdpa_strided(
                &qh,
                k,
                base,
                width,
                v,
                base,
                width,
                additive.as_deref(),
                len,
                len,
                dim,
                dim,
                scale,
            )
        })
        .collect();
    let mut out = vec![0.0f32; seqs * len * width];
    out.par_chunks_mut(len * width)
        .enumerate()
        .for_each(|(s, seq)| {
            for h in 0..heads {
                let plane = &planes[s * heads + h];
                for t in 0..len {
                    seq[t * width + h * dim..t * width + (h + 1) * dim]
                        .copy_from_slice(&plane[t * dim..(t + 1) * dim]);
                }
            }
        });
    out
}

/// A pre-norm transformer layer along time (`_TimeBlock`).
pub(crate) struct TimeBlock {
    norm1: LayerNorm,
    qkv: Linear,
    proj: Linear,
    norm2: LayerNorm,
    ffn_in: Linear,
    ffn_out: Linear,
    heads: usize,
    exclusive: bool,
    window: usize,
}

impl TimeBlock {
    pub(crate) fn load(
        l: &Loader,
        name: &str,
        width: usize,
        heads: usize,
        exclusive: bool,
        window: usize,
    ) -> Result<Self> {
        Ok(Self {
            norm1: l.norm(&format!("{name}.norm1"), width)?,
            qkv: l.linear(&format!("{name}.attn.qkv"), width, 3 * width, true)?,
            proj: l.linear(&format!("{name}.attn.proj"), width, width, true)?,
            norm2: l.norm(&format!("{name}.norm2"), width)?,
            ffn_in: l.linear(&format!("{name}.ffn.0"), width, 4 * width, true)?,
            ffn_out: l.linear(&format!("{name}.ffn.3"), 4 * width, width, true)?,
            heads,
            exclusive,
            window,
        })
    }

    /// The mask a forward over `len` frames applies.
    pub(crate) fn mask(&self, len: usize, causal: bool) -> Mask {
        if !causal {
            Mask::Full
        } else if self.window > 0 && self.window < len {
            Mask::Window(self.window)
        } else {
            Mask::Causal
        }
    }

    /// `x [seqs, len, width]` in place.
    pub(crate) fn forward(
        &self,
        exec: &Exec,
        x: &mut [f32],
        seqs: usize,
        len: usize,
        causal: bool,
    ) {
        let width = self.qkv.input;
        let rows = seqs * len;
        let h = layer_norm(&self.norm1, x, rows);
        let qkv = exec.linear(&self.qkv, &h, rows);
        let attended = attend(
            exec,
            &qkv,
            seqs,
            len,
            self.heads,
            width / self.heads,
            true,
            self.mask(len, causal),
            self.exclusive,
        );
        add_into(x, &exec.linear(&self.proj, &attended, rows));
        let h = layer_norm(&self.norm2, x, rows);
        let mut f = exec.linear(&self.ffn_in, &h, rows);
        gelu(&mut f);
        add_into(x, &exec.linear(&self.ffn_out, &f, rows));
    }
}

/// One attention block's time-attention past in a stream (`_AttentionState`
/// with `_AttentionCache`): keys and values `[bands, capacity, width]`, a key
/// rotated once at its slot (a slot's index is its rotary position).
pub(crate) struct KvCache {
    keys: Vec<f32>,
    values: Vec<f32>,
    end: usize,
    capacity: usize,
    /// The past a new frame may see before its block.
    frames: usize,
    /// The trained attention window (0: none).
    window: usize,
    cos: Vec<f32>,
    sin: Vec<f32>,
}

impl KvCache {
    pub(crate) fn new(
        bands: usize,
        width: usize,
        head_dim: usize,
        frames: usize,
        capacity: usize,
        window: usize,
    ) -> Self {
        let (cos, sin) = ops::rope_tables(capacity, head_dim, 10_000.0);
        Self {
            keys: vec![0.0; bands * capacity * width],
            values: vec![0.0; bands * capacity * width],
            end: 0,
            capacity,
            frames,
            window,
            cos,
            sin,
        }
    }

    /// Moves the last `frames` keys and values to the front, the keys
    /// rotated back by the distance they moved (`_compact`).
    fn compact(&mut self, bands: usize, width: usize, head_dim: usize) {
        let keep = self.end.min(self.frames);
        let shift = self.end - keep;
        if shift == 0 {
            return;
        }
        let half = head_dim / 2;
        let (c, s) = (
            &self.cos[shift * half..(shift + 1) * half],
            &self.sin[shift * half..(shift + 1) * half],
        );
        let cap = self.capacity;
        self.keys
            .par_chunks_mut(cap * width)
            .zip(self.values.par_chunks_mut(cap * width))
            .for_each(|(k, v)| {
                for slot in 0..keep {
                    let (dst, src) = (slot * width, (slot + shift) * width);
                    k.copy_within(src..src + width, dst);
                    v.copy_within(src..src + width, dst);
                    for head in k[dst..dst + width].chunks_exact_mut(head_dim) {
                        let (lo, hi) = head.split_at_mut(half);
                        for d in 0..half {
                            // apply_rope(x, cos, -sin)
                            let (a, b) = (lo[d], hi[d]);
                            lo[d] = a * c[d] + b * s[d];
                            hi[d] = -a * s[d] + b * c[d];
                        }
                    }
                }
            });
        let _ = bands;
        self.end = keep;
    }
}

impl TimeBlock {
    /// [`TimeBlock::forward`] over `count` new frames of every band, seeing
    /// the cached past (`streaming._attend` for one lane).
    pub(crate) fn forward_stream(
        &self,
        exec: &Exec,
        x: &mut [f32],
        bands: usize,
        count: usize,
        cache: &mut KvCache,
    ) {
        let width = self.qkv.input;
        let heads = self.heads;
        let dim = width / heads;
        let half = dim / 2;
        let rows = bands * count;
        let h = layer_norm(&self.norm1, x, rows);
        let qkv = exec.linear(&self.qkv, &h, rows);
        if cache.end + count > cache.capacity {
            cache.compact(bands, width, dim);
        }
        assert!(
            cache.end + count <= cache.capacity,
            "stream block larger than the cache headroom"
        );
        let first = cache.end;
        let cap = cache.capacity;
        let rotate = |row: &mut [f32], slot: usize, cos: &[f32], sin: &[f32]| {
            let (c, s) = (
                &cos[slot * half..(slot + 1) * half],
                &sin[slot * half..(slot + 1) * half],
            );
            for head in row.chunks_exact_mut(dim) {
                let (lo, hi) = head.split_at_mut(half);
                for d in 0..half {
                    let (a, b) = (lo[d], hi[d]);
                    lo[d] = a * c[d] - b * s[d];
                    hi[d] = a * s[d] + b * c[d];
                }
            }
        };
        // New keys (rotated at their slots) and values into the cache;
        // queries rotated at the same slots.
        let mut q = vec![0.0f32; rows * width];
        {
            let (cos, sin) = (&cache.cos, &cache.sin);
            let keys = &mut cache.keys;
            let values = &mut cache.values;
            keys.par_chunks_mut(cap * width)
                .zip(values.par_chunks_mut(cap * width))
                .zip(q.par_chunks_mut(count * width))
                .enumerate()
                .for_each(|(b, ((kb, vb), qb))| {
                    for i in 0..count {
                        let row =
                            &qkv[(b * count + i) * 3 * width..(b * count + i + 1) * 3 * width];
                        let slot = first + i;
                        let qrow = &mut qb[i * width..(i + 1) * width];
                        qrow.copy_from_slice(&row[..width]);
                        rotate(qrow, slot, cos, sin);
                        let krow = &mut kb[slot * width..(slot + 1) * width];
                        krow.copy_from_slice(&row[width..2 * width]);
                        rotate(krow, slot, cos, sin);
                        vb[slot * width..(slot + 1) * width].copy_from_slice(&row[2 * width..]);
                    }
                });
        }
        cache.end = first + count;
        let past = first.min(cache.frames);
        let lo = first - past;
        let keys_len = past + count;
        let window = cache.window;
        // Additive mask [count, keys_len]: itself and what precedes it,
        // within the trained window.
        let mask: Vec<f32> = (0..count * keys_len)
            .map(|i| {
                let (qi, kj) = (i / keys_len, i % keys_len);
                let (qslot, kslot) = (first + qi, lo + kj);
                let ok = kslot <= qslot && (window == 0 || kslot + window > qslot);
                if ok {
                    0.0
                } else {
                    f32::NEG_INFINITY
                }
            })
            .collect();
        let scale = 1.0 / (dim as f32).sqrt();
        let (keys, values) = (&cache.keys, &cache.values);
        let planes: Vec<Vec<f32>> = (0..bands * heads)
            .into_par_iter()
            .map(|job| {
                let (b, hd) = (job / heads, job % heads);
                let mut qh = vec![0.0f32; count * dim];
                for i in 0..count {
                    qh[i * dim..(i + 1) * dim].copy_from_slice(
                        &q[(b * count + i) * width + hd * dim
                            ..(b * count + i) * width + (hd + 1) * dim],
                    );
                }
                let base = b * cap * width + lo * width + hd * dim;
                ops::sdpa_strided(
                    &qh,
                    keys,
                    base,
                    width,
                    values,
                    base,
                    width,
                    Some(&mask),
                    count,
                    keys_len,
                    dim,
                    dim,
                    scale,
                )
            })
            .collect();
        let mut attended = vec![0.0f32; rows * width];
        for b in 0..bands {
            for hd in 0..heads {
                let plane = &planes[b * heads + hd];
                for i in 0..count {
                    let dst = (b * count + i) * width + hd * dim;
                    attended[dst..dst + dim].copy_from_slice(&plane[i * dim..(i + 1) * dim]);
                }
            }
        }
        if self.exclusive {
            for b in 0..bands {
                for i in 0..count {
                    let vrow = &values[b * cap * width + (first + i) * width
                        ..b * cap * width + (first + i + 1) * width];
                    let orow = &mut attended[(b * count + i) * width..(b * count + i + 1) * width];
                    for (o, v) in orow.chunks_exact_mut(dim).zip(vrow.chunks_exact(dim)) {
                        let ov: f32 = o.iter().zip(v).map(|(a, b)| a * b).sum();
                        let vv: f32 = v.iter().map(|a| a * a).sum();
                        let coefficient = ov / vv.max(1e-6);
                        o.iter_mut().zip(v).for_each(|(o, v)| *o -= coefficient * v);
                    }
                }
            }
        }
        add_into(x, &exec.linear(&self.proj, &attended, rows));
        let h = layer_norm(&self.norm2, x, rows);
        let mut f = exec.linear(&self.ffn_in, &h, rows);
        gelu(&mut f);
        add_into(x, &exec.linear(&self.ffn_out, &f, rows));
    }

    pub(crate) fn heads(&self) -> usize {
        self.heads
    }
}

/// A pre-norm transformer layer across bands (`_BandTransformer`).
pub(crate) struct BandBlock {
    norm1: LayerNorm,
    in_proj: Linear,
    out_proj: Linear,
    norm2: LayerNorm,
    linear1: Linear,
    linear2: Linear,
    heads: usize,
    exclusive: bool,
}

impl BandBlock {
    pub(crate) fn load(
        l: &Loader,
        name: &str,
        width: usize,
        heads: usize,
        exclusive: bool,
    ) -> Result<Self> {
        Ok(Self {
            norm1: l.norm(&format!("{name}.norm1"), width)?,
            in_proj: l.linear_named(
                &format!("{name}.self_attn.in_proj_weight"),
                Some(&format!("{name}.self_attn.in_proj_bias")),
                width,
                3 * width,
            )?,
            out_proj: l.linear(&format!("{name}.self_attn.out_proj"), width, width, true)?,
            norm2: l.norm(&format!("{name}.norm2"), width)?,
            linear1: l.linear(&format!("{name}.linear1"), width, 4 * width, true)?,
            linear2: l.linear(&format!("{name}.linear2"), 4 * width, width, true)?,
            heads,
            exclusive,
        })
    }

    /// `x [seqs, len, width]` in place, where a sequence is one frame's bands.
    pub(crate) fn forward(&self, exec: &Exec, x: &mut [f32], seqs: usize, len: usize) {
        let width = self.in_proj.input;
        let rows = seqs * len;
        let h = layer_norm(&self.norm1, x, rows);
        let qkv = exec.linear(&self.in_proj, &h, rows);
        let attended = attend(
            exec,
            &qkv,
            seqs,
            len,
            self.heads,
            width / self.heads,
            false,
            Mask::Full,
            self.exclusive,
        );
        add_into(x, &exec.linear(&self.out_proj, &attended, rows));
        let h = layer_norm(&self.norm2, x, rows);
        let mut f = exec.linear(&self.linear1, &h, rows);
        gelu(&mut f);
        add_into(x, &exec.linear(&self.linear2, &f, rows));
    }
}

/// The carried state of one Mamba layer across stream pieces: the last
/// `kernel - 1` conv inputs and the scan state, per sequence.
#[derive(Clone, Debug, Default)]
pub(crate) struct MambaState {
    /// `[seqs, kernel - 1, inner]`, oldest first.
    pub conv: Vec<f32>,
    /// `[seqs, inner, state]`.
    pub ssm: Vec<f32>,
}

/// One selective state-space layer (`Mamba`).
pub(crate) struct Mamba {
    in_proj: Linear,
    conv_w: Vec<f32>,
    conv_b: Vec<f32>,
    x_proj: Linear,
    dt_proj: Linear,
    /// `-exp(A_log)`, `[inner, state]`.
    a: Vec<f32>,
    d: Vec<f32>,
    out_proj: Linear,
    inner: usize,
    state: usize,
    kernel: usize,
    dt_rank: usize,
}

impl Mamba {
    pub(crate) fn load(
        l: &Loader,
        name: &str,
        width: usize,
        state: usize,
        kernel: usize,
        expand: usize,
    ) -> Result<Self> {
        let inner = expand * width;
        let dt_rank = width.div_ceil(16);
        let a_log = l.tensor(&format!("{name}.A_log"), &[inner, state])?;
        Ok(Self {
            in_proj: l.linear(&format!("{name}.in_proj"), width, 2 * inner, false)?,
            conv_w: l.tensor(&format!("{name}.conv1d.weight"), &[inner, 1, kernel])?,
            conv_b: l.tensor(&format!("{name}.conv1d.bias"), &[inner])?,
            x_proj: l.linear(&format!("{name}.x_proj"), inner, dt_rank + 2 * state, false)?,
            dt_proj: l.linear(&format!("{name}.dt_proj"), dt_rank, inner, true)?,
            a: a_log.iter().map(|v| -v.exp()).collect(),
            d: l.tensor(&format!("{name}.D"), &[inner])?,
            out_proj: l.linear(&format!("{name}.out_proj"), inner, width, false)?,
            inner,
            state,
            kernel,
            dt_rank,
        })
    }

    /// An empty carried state for `seqs` sequences.
    pub(crate) fn zero_state(&self, seqs: usize) -> MambaState {
        MambaState {
            conv: vec![0.0; seqs * (self.kernel - 1) * self.inner],
            ssm: vec![0.0; seqs * self.inner * self.state],
        }
    }

    /// `x [seqs, len, width] -> [seqs, len, width]`. With `state`, the
    /// sequences continue from it and it is advanced past this input; a
    /// fresh state (zeros) gives the stateless result exactly.
    pub(crate) fn forward(
        &self,
        exec: &Exec,
        x: &[f32],
        seqs: usize,
        len: usize,
        state: Option<&mut MambaState>,
    ) -> Vec<f32> {
        let rows = seqs * len;
        let inner = self.inner;
        let xz = exec.linear(&self.in_proj, x, rows);
        let km1 = self.kernel - 1;
        let carried = state.as_ref().map(|s| s.conv.clone());
        // Depthwise causal conv over u (the first half of xz), then SiLU.
        let mut u = vec![0.0f32; rows * inner];
        u.par_chunks_mut(len * inner)
            .enumerate()
            .for_each(|(s, seq)| {
                let history = |t: isize, c: usize| -> f32 {
                    if t >= 0 {
                        xz[(s * len + t as usize) * 2 * inner + c]
                    } else if let Some(conv) = &carried {
                        // t in -(k-1)..0 maps to the carried rows, oldest first.
                        conv[(s * km1 + (km1 as isize + t) as usize) * inner + c]
                    } else {
                        0.0
                    }
                };
                for t in 0..len {
                    let out = &mut seq[t * inner..(t + 1) * inner];
                    for (c, o) in out.iter_mut().enumerate() {
                        let mut acc = 0.0f32;
                        for k in 0..self.kernel {
                            acc += self.conv_w[c * self.kernel + k]
                                * history(t as isize + k as isize - km1 as isize, c);
                        }
                        let v = acc + self.conv_b[c];
                        *o = v / (1.0 + (-v).exp());
                    }
                }
            });
        let projected = exec.linear(&self.x_proj, &u, rows);
        let width_bc = self.dt_rank + 2 * self.state;
        let dt: Vec<f32> = projected
            .chunks_exact(width_bc)
            .flat_map(|row| row[..self.dt_rank].iter().copied())
            .collect();
        let mut delta = exec.linear(&self.dt_proj, &dt, rows);
        delta.par_chunks_mut(1 << 14).for_each(|chunk| {
            for v in chunk {
                // torch softplus: beta 1, linear above threshold 20.
                *v = if *v > 20.0 { *v } else { v.exp().ln_1p() };
            }
        });
        let ssm_in = state.as_ref().map(|s| s.ssm.as_slice());
        let device = exec.timed(2, || {
            exec.scan(
                ScanInputs {
                    u: &u,
                    delta: &delta,
                    a: &self.a,
                    bc: &projected,
                    d: &self.d,
                    z: &xz,
                    carried: ssm_in,
                },
                ScanShape {
                    seqs,
                    len,
                    channels: inner,
                    states: self.state,
                    bc_stride: width_bc,
                    b_offset: self.dt_rank,
                    c_offset: self.dt_rank + self.state,
                    z_stride: 2 * inner,
                    z_offset: inner,
                },
            )
        });
        let (y, last) = device.unwrap_or_else(|| {
            let mut y = vec![0.0f32; rows * inner];
            let mut last = vec![0.0f32; seqs * inner * self.state];
            y.par_chunks_mut(len * inner)
                .zip(last.par_chunks_mut(inner * self.state))
                .enumerate()
                .for_each(|(s, (ys, h))| {
                    if let Some(ssm) = ssm_in {
                        h.copy_from_slice(
                            &ssm[s * inner * self.state..(s + 1) * inner * self.state],
                        );
                    }
                    self.scan(
                        s,
                        len,
                        &u[s * len * inner..(s + 1) * len * inner],
                        &delta[s * len * inner..(s + 1) * len * inner],
                        &projected,
                        &xz,
                        h,
                        ys,
                    );
                });
            (y, last)
        });
        if let Some(st) = state {
            // Advance the conv history: the last k-1 inputs of carried + new.
            let mut conv = vec![0.0f32; seqs * km1 * inner];
            for s in 0..seqs {
                for j in 0..km1 {
                    // Position j of the new tail is time len - km1 + j.
                    let t = len as isize - km1 as isize + j as isize;
                    for c in 0..inner {
                        conv[(s * km1 + j) * inner + c] = if t >= 0 {
                            xz[(s * len + t as usize) * 2 * inner + c]
                        } else {
                            st.conv[(s * km1 + (km1 as isize + t) as usize) * inner + c]
                        };
                    }
                }
            }
            st.conv = conv;
            st.ssm = last;
        }
        exec.linear(&self.out_proj, &y, rows)
    }

    /// The scan of one sequence `s`: `h_t = exp(delta a) h + delta b u`,
    /// `y_t = c . h_t + d u`, gated by `silu(z)`. Summation order follows
    /// the reference CPU loop (`scan.selective_scan`).
    #[allow(clippy::too_many_arguments)]
    fn scan(
        &self,
        s: usize,
        len: usize,
        u: &[f32],
        delta: &[f32],
        projected: &[f32],
        xz: &[f32],
        h: &mut [f32],
        y: &mut [f32],
    ) {
        let (inner, n) = (self.inner, self.state);
        let width_bc = self.dt_rank + 2 * n;
        for t in 0..len {
            let row = &projected[(s * len + t) * width_bc..(s * len + t + 1) * width_bc];
            let b = &row[self.dt_rank..self.dt_rank + n];
            let c = &row[self.dt_rank + n..];
            let z = &xz[(s * len + t) * 2 * inner + inner..(s * len + t + 1) * 2 * inner];
            for ch in 0..inner {
                let step = delta[t * inner + ch];
                let x = u[t * inner + ch];
                let drive = step * x;
                let hs = &mut h[ch * n..(ch + 1) * n];
                let a = &self.a[ch * n..(ch + 1) * n];
                let mut out = 0.0f32;
                for i in 0..n {
                    hs[i] = (step * a[i]).exp() * hs[i] + drive * b[i];
                    out += hs[i] * c[i];
                }
                let g = z[ch];
                y[t * inner + ch] = (out + x * self.d[ch]) * (g / (1.0 + (-g).exp()));
            }
        }
    }
}

/// Forward and backward Mamba projected together (`BidirectionalMamba`);
/// forward only when the checkpoint is causal.
pub(crate) struct BiMamba {
    forward: Mamba,
    backward: Option<Mamba>,
    output_proj: Linear,
    norm: LayerNorm,
}

impl BiMamba {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn load(
        l: &Loader,
        name: &str,
        width: usize,
        state: usize,
        kernel: usize,
        expand: usize,
        causal: bool,
    ) -> Result<Self> {
        let backward = if causal {
            None
        } else {
            Some(Mamba::load(
                l,
                &format!("{name}.backward_blocks"),
                width,
                state,
                kernel,
                expand,
            )?)
        };
        Ok(Self {
            forward: Mamba::load(
                l,
                &format!("{name}.forward_blocks"),
                width,
                state,
                kernel,
                expand,
            )?,
            backward,
            output_proj: l.linear(
                &format!("{name}.output_proj"),
                if causal { width } else { 2 * width },
                width,
                true,
            )?,
            norm: l.norm(&format!("{name}.norm"), width)?,
        })
    }

    pub(crate) fn forward_layer(&self) -> &Mamba {
        &self.forward
    }

    /// `norm(output_proj(...))` of `x [seqs, len, width]`; `state` carries
    /// the forward scan of a causal layer across stream pieces.
    pub(crate) fn apply(
        &self,
        exec: &Exec,
        x: &[f32],
        seqs: usize,
        len: usize,
        state: Option<&mut MambaState>,
    ) -> Vec<f32> {
        let width = self.norm.width();
        let rows = seqs * len;
        let mut fwd = self.forward.forward(exec, x, seqs, len, state);
        add_into(&mut fwd, x);
        let joined = match &self.backward {
            None => fwd,
            Some(backward) => {
                let flipped = flip(x, seqs, len, width);
                let mut bwd = flip(
                    &backward.forward(exec, &flipped, seqs, len, None),
                    seqs,
                    len,
                    width,
                );
                add_into(&mut bwd, x);
                let mut joined = vec![0.0f32; rows * 2 * width];
                joined
                    .par_chunks_mut(2 * width)
                    .enumerate()
                    .for_each(|(r, row)| {
                        row[..width].copy_from_slice(&fwd[r * width..(r + 1) * width]);
                        row[width..].copy_from_slice(&bwd[r * width..(r + 1) * width]);
                    });
                joined
            }
        };
        let projected = exec.linear(&self.output_proj, &joined, rows);
        layer_norm(&self.norm, &projected, rows)
    }
}

/// Reverses every sequence of `x [seqs, len, width]` along `len`.
fn flip(x: &[f32], seqs: usize, len: usize, width: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    out.par_chunks_mut(len * width)
        .enumerate()
        .for_each(|(s, seq)| {
            for t in 0..len {
                let src = (s * len + (len - 1 - t)) * width;
                seq[t * width..(t + 1) * width].copy_from_slice(&x[src..src + width]);
            }
        });
    out
}

/// One block of the backbone.
pub(crate) enum Block {
    Attention { time: TimeBlock, band: BandBlock },
    Mamba { time: BiMamba, band: BiMamba },
}

/// The carried state of one block across stream pieces.
pub(crate) enum BlockState {
    Attention(KvCache),
    /// The causal time scan per band (`_MambaState`).
    Mamba(MambaState),
}

impl Block {
    /// A fresh stream state for `bands` bands.
    pub(crate) fn stream_state(
        &self,
        bands: usize,
        width: usize,
        frames: usize,
        capacity: usize,
        window: usize,
    ) -> BlockState {
        match self {
            Block::Attention { time, .. } => BlockState::Attention(KvCache::new(
                bands,
                width,
                width / time.heads(),
                frames,
                capacity,
                window,
            )),
            Block::Mamba { time, .. } => BlockState::Mamba(time.forward_layer().zero_state(bands)),
        }
    }

    /// `count` new frames `[bands, count, width]` through the block at its
    /// streaming position (`_attend` / `_scan`), in place.
    pub(crate) fn forward_stream(
        &self,
        exec: &Exec,
        x: &mut Vec<f32>,
        bands: usize,
        count: usize,
        state: &mut BlockState,
    ) {
        let width = x.len() / (bands * count);
        match (self, state) {
            (Block::Attention { time, band }, BlockState::Attention(cache)) => {
                time.forward_stream(exec, x, bands, count, cache);
                let mut across = swap_axes(x, bands, count, width);
                band.forward(exec, &mut across, count, bands);
                *x = swap_axes(&across, count, bands, width);
            }
            (Block::Mamba { time, band }, BlockState::Mamba(carried)) => {
                let t = time.apply(exec, x, bands, count, Some(carried));
                add_into(x, &t);
                let mut across = swap_axes(x, bands, count, width);
                let f = band.apply(exec, &across, count, bands, None);
                add_into(&mut across, &f);
                *x = swap_axes(&across, count, bands, width);
            }
            _ => unreachable!("stream state built from this block"),
        }
    }

    /// `x [bands, frames, width]` through the block, in place.
    pub(crate) fn forward(
        &self,
        exec: &Exec,
        x: &mut Vec<f32>,
        bands: usize,
        frames: usize,
        causal: bool,
    ) {
        let width = x.len() / (bands * frames);
        match self {
            Block::Attention { time, band } => {
                time.forward(exec, x, bands, frames, causal);
                let mut across = swap_axes(x, bands, frames, width);
                band.forward(exec, &mut across, frames, bands);
                *x = swap_axes(&across, frames, bands, width);
            }
            Block::Mamba { time, band } => {
                let t = time.apply(exec, x, bands, frames, None);
                add_into(x, &t);
                let mut across = swap_axes(x, bands, frames, width);
                let f = band.apply(exec, &across, frames, bands, None);
                add_into(&mut across, &f);
                *x = swap_axes(&across, frames, bands, width);
            }
        }
    }
}

/// Per-band heads (`_Decoder`): FiLM by the fixed controls, then
/// LayerNorm, Linear, GELU, Linear to four values per bin.
pub(crate) struct Decoder {
    scale: Vec<f32>,
    shift: Vec<f32>,
    heads: Vec<(LayerNorm, Linear, Linear)>,
}

impl Decoder {
    pub(crate) fn load(l: &Loader, width: usize, edges: &[(usize, usize)]) -> Result<Self> {
        let condition = l.linear("denoise_decoder.condition", 2, 2 * width, true)?;
        // The controls buffer is the constant [1, 0].
        let film = condition.forward(&[1.0, 0.0], 1);
        let mut heads = Vec::with_capacity(edges.len());
        for (i, &(start, end)) in edges.iter().enumerate() {
            let name = format!("denoise_decoder.heads.{i}");
            heads.push((
                l.norm(&format!("{name}.0"), width)?,
                l.linear(&format!("{name}.1"), width, width, true)?,
                l.linear(&format!("{name}.3"), width, 4 * (end - start), true)?,
            ));
        }
        Ok(Self {
            scale: film[..width].to_vec(),
            shift: film[width..].to_vec(),
            heads,
        })
    }

    /// `features [bands, frames, width]` -> raw `[frames, bins, 4]`, the
    /// bands' bins in order.
    pub(crate) fn forward(
        &self,
        exec: &Exec,
        features: &[f32],
        frames: usize,
        edges: &[(usize, usize)],
    ) -> Vec<f32> {
        let width = self.scale.len();
        let bins = edges.last().map_or(0, |e| e.1);
        let mut raw = vec![0.0f32; frames * bins * 4];
        for (b, ((norm, l1, l2), &(start, end))) in self.heads.iter().zip(edges).enumerate() {
            let mut f = features[b * frames * width..(b + 1) * frames * width].to_vec();
            for row in f.chunks_exact_mut(width) {
                for ((v, s), sh) in row.iter_mut().zip(&self.scale).zip(&self.shift) {
                    *v = *v * (1.0 + s) + sh;
                }
            }
            let h = layer_norm(norm, &f, frames);
            let mut h = exec.linear(l1, &h, frames);
            gelu(&mut h);
            let out = exec.linear(l2, &h, frames);
            let nb = end - start;
            for t in 0..frames {
                raw[(t * bins + start) * 4..(t * bins + end) * 4]
                    .copy_from_slice(&out[t * nb * 4..(t + 1) * nb * 4]);
            }
        }
        raw
    }
}
