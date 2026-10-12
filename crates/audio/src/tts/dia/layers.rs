//! Dia encoder/decoder building blocks (`layers.py`): DenseGeneral
//! projections (flat kernels transposed into crate linears), the gated
//! fused MLP, the timescale rotary embedding, attention with the
//! reference scale 1.0 and additive -1e9 masking, and the KV caches.
//!
//! Everything is portable f32, generation batch B = 2 fixed (the
//! unconditional and conditional rows travel together). Activations are
//! row-major `[tokens, width]`; attention tensors are head-major
//! `[2, heads, seq, head_dim]`.

use crate::nn::{Linear, RmsNorm};
use crate::{ops, Result, SpeechError};
use turbospark_model_io::safetensors::SafetensorsFile;

/// The MLP activation names the reference `get_activation_fn` supports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Act {
    Silu,
    Gelu,
    Relu,
    Linear,
}

impl Act {
    pub(crate) fn parse(name: &str) -> Result<Self> {
        match name {
            "silu" | "swish" => Ok(Self::Silu),
            "gelu" => Ok(Self::Gelu),
            "relu" => Ok(Self::Relu),
            "linear" => Ok(Self::Linear),
            other => Err(SpeechError::BadConfig {
                field: "mlp_activations".into(),
                why: format!("unsupported activation function: {other}"),
            }),
        }
    }

    fn apply(&self, row: &mut [f32]) {
        match self {
            Self::Silu => ops::silu(row),
            Self::Gelu => ops::gelu_erf(row),
            Self::Relu => row.iter_mut().for_each(|v| *v = v.max(0.0)),
            Self::Linear => {}
        }
    }
}

/// Loads a plain tensor from the checkpoint shard. The declared
/// `shape` fixes the element count only; the file keeps the MLX tree
/// shapes (`[rows, cols]` tables, `[out, in]` linears), and both orders
/// are flat row-major.
pub(crate) fn load_tensor(file: &SafetensorsFile, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
    let descriptor = file.descriptor(name).ok_or_else(|| SpeechError::Tensor {
        name: name.to_owned(),
        why: "tensor is missing".into(),
    })?;
    if descriptor.shape.iter().product::<usize>() != shape.iter().product::<usize>() {
        return Err(SpeechError::Tensor {
            name: name.to_owned(),
            why: format!("expected {shape:?} elements, got {:?}", descriptor.shape),
        });
    }
    crate::nn::load_tensor(file, name, &descriptor.shape)
}

/// A `DenseGeneral` kernel flattened to `[in, out]` row-major (the
/// out-feature axes concatenate with the last varying fastest) and
/// transposed into the crate `[out, in]` linear layout. Every dia
/// projection contracts the last input axis, and `o_proj`'s two-axis
/// contraction over `[heads, head_dim]` reduces to the same flat
/// row-major product.
pub(crate) fn load_dense_linear(
    file: &SafetensorsFile,
    name: &str,
    input: usize,
    output: usize,
) -> Result<Linear> {
    // The checkpoint stores the kernel with its full DenseGeneral shape
    // (`[in, out...]`); only the element count and the flat row-major
    // order matter here.
    let full = format!("{name}.weight");
    let descriptor = file.descriptor(&full).ok_or_else(|| SpeechError::Tensor {
        name: full.clone(),
        why: "tensor is missing".into(),
    })?;
    if descriptor.shape.iter().product::<usize>() != input * output {
        return Err(SpeechError::Tensor {
            name: full.clone(),
            why: format!(
                "expected {input}x{output} elements, got {:?}",
                descriptor.shape
            ),
        });
    }
    let kernel = load_tensor(file, &full, &descriptor.shape)?;
    let mut weight = vec![0.0f32; kernel.len()];
    for i in 0..input {
        for o in 0..output {
            weight[o * input + i] = kernel[i * output + o];
        }
    }
    Ok(Linear::new(weight, None, input, output))
}

/// The fused-gate MLP (`MlpBlock`): one DenseGeneral to
/// `[num_activations, n_hidden]`, per-part activations, an elementwise
/// product, and the down projection.
pub(crate) struct Mlp {
    pre_norm: Option<RmsNorm>,
    gate: Linear,
    up: Linear,
    wo: Linear,
    act_gate: Act,
    act_up: Act,
}

impl Mlp {
    pub(crate) fn load(
        file: &SafetensorsFile,
        prefix: &str,
        embed_dim: usize,
        intermediate_dim: usize,
        activations: [Act; 2],
        use_pre_norm: bool,
        eps: f32,
    ) -> Result<Self> {
        let (gate, up) = load_fused(
            file,
            &format!("{prefix}.wi_fused"),
            embed_dim,
            intermediate_dim,
        )?;
        Ok(Self {
            pre_norm: if use_pre_norm {
                Some(RmsNorm::new(
                    load_tensor(file, &format!("{prefix}.pre_norm.weight"), &[embed_dim])?,
                    eps,
                ))
            } else {
                None
            },
            gate,
            up,
            wo: load_dense_linear(file, &format!("{prefix}.wo"), intermediate_dim, embed_dim)?,
            act_gate: activations[0],
            act_up: activations[1],
        })
    }

    /// `x [rows, embed] -> [rows, embed]` in place.
    pub(crate) fn forward(&self, x: &mut [f32], rows: usize) {
        if let Some(norm) = &self.pre_norm {
            norm.apply(x, rows);
        }
        let hidden = self.gate.forward(x, rows);
        let mut up = self.up.forward(x, rows);
        let width = hidden.len() / rows.max(1);
        let mut hidden = hidden;
        for row in hidden.chunks_exact_mut(width) {
            self.act_gate.apply(row);
        }
        for row in up.chunks_exact_mut(width) {
            self.act_up.apply(row);
        }
        let mut product = vec![0.0f32; hidden.len()];
        for (p, (g, u)) in product.iter_mut().zip(hidden.iter().zip(&up)) {
            *p = g * u;
        }
        let out = self.wo.forward(&product, rows);
        x.copy_from_slice(&out);
    }
}

/// Splits the fused `[in, 2, hidden]` kernel into the two `[hidden,
/// in]` linears (activation part 0 gates, part 1 passes through).
fn load_fused(
    file: &SafetensorsFile,
    name: &str,
    embed_dim: usize,
    hidden: usize,
) -> Result<(Linear, Linear)> {
    // The fused kernel keeps its `[in, 2, hidden]` shape in the file.
    let full = format!("{name}.weight");
    let descriptor = file.descriptor(&full).ok_or_else(|| SpeechError::Tensor {
        name: full.clone(),
        why: "tensor is missing".into(),
    })?;
    if descriptor.shape.iter().product::<usize>() != embed_dim * 2 * hidden {
        return Err(SpeechError::Tensor {
            name: full.clone(),
            why: format!(
                "expected {}x2x{} elements, got {:?}",
                embed_dim, hidden, descriptor.shape
            ),
        });
    }
    let kernel = load_tensor(file, &full, &descriptor.shape)?;
    let mut gate_w = vec![0.0f32; hidden * embed_dim];
    let mut up_w = vec![0.0f32; hidden * embed_dim];
    for i in 0..embed_dim {
        for h in 0..hidden {
            gate_w[h * embed_dim + i] = kernel[(i * 2) * hidden + h];
            up_w[h * embed_dim + i] = kernel[(i * 2 + 1) * hidden + h];
        }
    }
    Ok((
        Linear::new(gate_w, None, embed_dim, hidden),
        Linear::new(up_w, None, embed_dim, hidden),
    ))
}

/// The `RotaryEmbedding`: per-dim timescales
/// `min * (max / min)^(2d / dim)`, sinusoid `position / timescale`,
/// rotating the first half of each head vector against the second.
#[derive(Debug, Clone)]
pub(crate) struct Rotary {
    min_timescale: f32,
    max_timescale: f32,
}

impl Rotary {
    pub(crate) fn new(min_timescale: i32, max_timescale: i32) -> Self {
        Self {
            min_timescale: min_timescale as f32,
            max_timescale: max_timescale as f32,
        }
    }

    /// `(cos, sin)` tables `[positions.len(), head_dim / 2]`, evaluated
    /// with the reference expression order (timescale powers, divide).
    pub(crate) fn tables(&self, positions: &[i32], head_dim: usize) -> (Vec<f32>, Vec<f32>) {
        let half = head_dim / 2;
        let mut cos = vec![0.0f32; positions.len() * half];
        let mut sin = vec![0.0f32; positions.len() * half];
        for (row, &position) in positions.iter().enumerate() {
            for d in 0..half {
                let fraction = (2.0 * d as f32) / head_dim as f32;
                let timescale =
                    self.min_timescale * (self.max_timescale / self.min_timescale).powf(fraction);
                let angle = position as f32 / timescale;
                cos[row * half + d] = angle.cos();
                sin[row * half + d] = angle.sin();
            }
        }
        (cos, sin)
    }
}

/// Rotates a `[2, heads, seq, head_dim]` buffer per batch row for
/// `positions`.
pub(crate) fn rope(
    rotary: &Rotary,
    x: &mut [f32],
    heads: usize,
    seq: usize,
    head_dim: usize,
    positions: &[i32],
) {
    let (cos, sin) = rotary.tables(positions, head_dim);
    let plane = heads * seq * head_dim;
    for b in 0..2 {
        ops::rope_neox(
            &mut x[b * plane..(b + 1) * plane],
            heads,
            seq,
            head_dim,
            &cos,
            &sin,
        );
    }
}

/// One attention KV store `[2, heads, max_len, head_dim]` (both
/// generation rows share the shape). Cross-attention caches are written
/// once by the precompute; self-attention caches append per decode step
/// or take a whole prefilled block.
pub(crate) struct KvStore {
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    heads: usize,
    max_len: usize,
    head_dim: usize,
    pub len: usize,
}

impl KvStore {
    pub(crate) fn new(heads: usize, max_len: usize, head_dim: usize) -> Self {
        let rows = heads * max_len * head_dim;
        Self {
            k: vec![0.0; 2 * rows],
            v: vec![0.0; 2 * rows],
            heads,
            max_len,
            head_dim,
            len: 0,
        }
    }

    /// `prefill_kv`: installs `k`/`v [2, heads, seq, head_dim]` into
    /// the first `seq` rows of each head plane.
    pub(crate) fn prefill(&mut self, k: &[f32], v: &[f32], seq: usize) {
        assert!(seq <= self.max_len, "KV cache prefill past capacity");
        for b in 0..2 {
            for h in 0..self.heads {
                let src = (b * self.heads + h) * seq * self.head_dim;
                let dst = self.base(b, h);
                self.k[dst..dst + seq * self.head_dim]
                    .copy_from_slice(&k[src..src + seq * self.head_dim]);
                self.v[dst..dst + seq * self.head_dim]
                    .copy_from_slice(&v[src..src + seq * self.head_dim]);
            }
        }
        self.len = seq;
    }

    /// Appends one position: `k`/`v [2, heads * head_dim]` (head-major).
    pub(crate) fn update(&mut self, k: &[f32], v: &[f32]) {
        assert!(self.len < self.max_len, "KV cache full");
        for b in 0..2 {
            for h in 0..self.heads {
                let src = (b * self.heads + h) * self.head_dim;
                let dst = self.base(b, h) + self.len * self.head_dim;
                self.k[dst..dst + self.head_dim].copy_from_slice(&k[src..src + self.head_dim]);
                self.v[dst..dst + self.head_dim].copy_from_slice(&v[src..src + self.head_dim]);
            }
        }
        self.len += 1;
    }

    /// Cache row base for `(row, head)` with the row stride the strided
    /// SDPA kernel needs.
    pub(crate) fn base(&self, row: usize, head: usize) -> usize {
        (row * self.heads + head) * self.max_len * self.head_dim
    }
}

/// `mx.repeat(k, groups, axis=heads)`: each kv head repeats `groups`
/// times consecutively, so query head `h` reads kv head `h / groups`.
fn repeat_kv(x: &[f32], kv_heads: usize, q_heads: usize, seq: usize, dim: usize) -> Vec<f32> {
    let groups = q_heads / kv_heads;
    let plane = seq * dim;
    let mut out = vec![0.0f32; 2 * q_heads * plane];
    for b in 0..2 {
        for qh in 0..q_heads {
            let src = (b * kv_heads + qh / groups) * plane;
            let dst = (b * q_heads + qh) * plane;
            out[dst..dst + plane].copy_from_slice(&x[src..src + plane]);
        }
    }
    out
}

/// `[2 * tokens, heads * head_dim]` -> `[2, heads, tokens, head_dim]`.
fn split_heads(x: &[f32], tokens: usize, heads: usize, head_dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; 2 * heads * tokens * head_dim];
    for b in 0..2 {
        for t in 0..tokens {
            for h in 0..heads {
                let src = (b * tokens + t) * heads * head_dim + h * head_dim;
                let dst = ((b * heads + h) * tokens + t) * head_dim;
                out[dst..dst + head_dim].copy_from_slice(&x[src..src + head_dim]);
            }
        }
    }
    out
}

/// `[2, heads, tokens, head_dim]` -> `[2 * tokens, heads * head_dim]`.
fn merge_heads(x: &[f32], tokens: usize, heads: usize, head_dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; 2 * tokens * heads * head_dim];
    for b in 0..2 {
        for t in 0..tokens {
            for h in 0..heads {
                let src = ((b * heads + h) * tokens + t) * head_dim;
                let dst = (b * tokens + t) * heads * head_dim + h * head_dim;
                out[dst..dst + head_dim].copy_from_slice(&x[src..src + head_dim]);
            }
        }
    }
    out
}

/// One self or cross attention module (`Attention`). The reference
/// scales scores by 1.0 (no head-dim normalization) and applies the
/// mask additively with -1e9.
pub(crate) struct Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    num_query_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    is_cross: bool,
    rotary: Rotary,
}

impl Attention {
    #[allow(
        clippy::too_many_arguments,
        reason = "attention construction carries the tensor geometry explicitly"
    )]
    pub(crate) fn load(
        file: &SafetensorsFile,
        prefix: &str,
        q_embed_dim: usize,
        kv_embed_dim: usize,
        num_query_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        is_cross: bool,
        out_embed_dim: usize,
        rotary: Rotary,
    ) -> Result<Self> {
        if num_query_heads % num_kv_heads != 0 {
            return Err(SpeechError::BadConfig {
                field: "heads".into(),
                why: format!(
                    "num_query_heads ({num_query_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
                ),
            });
        }
        Ok(Self {
            q: load_dense_linear(
                file,
                &format!("{prefix}.q_proj"),
                q_embed_dim,
                num_query_heads * head_dim,
            )?,
            k: load_dense_linear(
                file,
                &format!("{prefix}.k_proj"),
                kv_embed_dim,
                num_kv_heads * head_dim,
            )?,
            v: load_dense_linear(
                file,
                &format!("{prefix}.v_proj"),
                kv_embed_dim,
                num_kv_heads * head_dim,
            )?,
            o: load_dense_linear(
                file,
                &format!("{prefix}.o_proj"),
                num_query_heads * head_dim,
                out_embed_dim,
            )?,
            num_query_heads,
            num_kv_heads,
            head_dim,
            is_cross,
            rotary,
        })
    }

    /// Projects and ropes keys/values over `kv_x [2 * kv_len,
    /// kv_in_dim]` (the cross-attention precompute and the prefill
    /// self-attention path).
    pub(crate) fn project_kv(
        &self,
        kv_x: &[f32],
        kv_len: usize,
        kv_positions: &[i32],
    ) -> (Vec<f32>, Vec<f32>) {
        let mut k = split_heads(
            &self.k.forward(kv_x, 2 * kv_len),
            kv_len,
            self.num_kv_heads,
            self.head_dim,
        );
        let v = split_heads(
            &self.v.forward(kv_x, 2 * kv_len),
            kv_len,
            self.num_kv_heads,
            self.head_dim,
        );
        rope(
            &self.rotary,
            &mut k,
            self.num_kv_heads,
            kv_len,
            self.head_dim,
            kv_positions,
        );
        (k, v)
    }

    /// Multi-token pass. Self mode projects k/v from `kv_x` (equal to
    /// `x` for self attention) and, when `self_cache` is given, prefills
    /// it; cross mode reads the precomputed `cross_cache` (whose length
    /// defines the kv extent). `mask[b]` is the additive
    /// `[seq, kv_len]` bias. Returns `[2 * seq, out_dim]`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward(
        &self,
        x: &[f32],
        kv_x: Option<&[f32]>,
        seq: usize,
        positions: &[i32],
        mask: Option<&[Vec<f32>]>,
        self_cache: Option<&mut KvStore>,
        cross_cache: Option<&KvStore>,
    ) -> Result<Vec<f32>> {
        let mut q = split_heads(
            &self.q.forward(x, 2 * seq),
            seq,
            self.num_query_heads,
            self.head_dim,
        );
        rope(
            &self.rotary,
            &mut q,
            self.num_query_heads,
            seq,
            self.head_dim,
            positions,
        );

        let (attn_k, attn_v, kv_len);
        if self.is_cross {
            let cache = cross_cache.expect("cross attention needs its precomputed cache");
            attn_k = cache.k.clone();
            attn_v = cache.v.clone();
            kv_len = cache.len;
        } else {
            let kv_x = kv_x.expect("self attention needs kv input");
            let (k, v) = self.project_kv(kv_x, seq, positions);
            kv_len = seq;
            let k = repeat_kv(
                &k,
                self.num_kv_heads,
                self.num_query_heads,
                seq,
                self.head_dim,
            );
            let v = repeat_kv(
                &v,
                self.num_kv_heads,
                self.num_query_heads,
                seq,
                self.head_dim,
            );
            if let Some(cache) = self_cache {
                cache.prefill(&k, &v, seq);
            }
            attn_k = k;
            attn_v = v;
        }

        let mut out = vec![0.0f32; 2 * self.num_query_heads * seq * self.head_dim];
        for b in 0..2 {
            let mask_row: Option<&[f32]> = mask.map(|m| m[b].as_slice());
            for h in 0..self.num_query_heads {
                let head = (b * self.num_query_heads + h) * seq * self.head_dim;
                let q_head = &q[head..head + seq * self.head_dim];
                let k_base = (b * self.num_query_heads + h) * kv_len * self.head_dim;
                let k_head = &attn_k[k_base..k_base + kv_len * self.head_dim];
                let v_head = &attn_v[k_base..k_base + kv_len * self.head_dim];
                let attended = ops::sdpa(
                    q_head,
                    k_head,
                    v_head,
                    mask_row,
                    seq,
                    kv_len,
                    self.head_dim,
                    self.head_dim,
                    1.0,
                );
                out[head..head + seq * self.head_dim].copy_from_slice(&attended);
            }
        }
        let merged = merge_heads(&out, seq, self.num_query_heads, self.head_dim);
        Ok(self.o.forward(&merged, 2 * seq))
    }

    /// Single-token decode step. Self attention appends the roped k/v
    /// to its cache and attends rows `0..len` (`self_cache` required);
    /// cross attention reads its precomputed cache (`cross_cache`
    /// required). `cross_mask[b]` is the additive `[1, kv_len]` bias.
    /// Returns `[2, out_dim]`.
    pub(crate) fn step(
        &self,
        x: &[f32],
        position: i32,
        cross_mask: Option<&[Vec<f32>]>,
        mut self_cache: Option<&mut KvStore>,
        cross_cache: Option<&KvStore>,
    ) -> Vec<f32> {
        let mut q = split_heads(
            &self.q.forward(x, 2),
            1,
            self.num_query_heads,
            self.head_dim,
        );
        rope(
            &self.rotary,
            &mut q,
            self.num_query_heads,
            1,
            self.head_dim,
            &[position],
        );

        let (k_all, v_all, kv_len);
        if self.is_cross {
            let cache = cross_cache.expect("cross attention needs its precomputed cache");
            kv_len = cache.len;
            k_all = repeat_kv(
                &cache.k,
                self.num_kv_heads,
                self.num_query_heads,
                kv_len,
                self.head_dim,
            );
            v_all = repeat_kv(
                &cache.v,
                self.num_kv_heads,
                self.num_query_heads,
                kv_len,
                self.head_dim,
            );
        } else {
            // The cache stores the GQA-repeated planes (the reference
            // caches the repeated k/v), so the read-back needs no
            // further expansion.
            let cache: &mut KvStore = self_cache
                .as_deref_mut()
                .expect("self attention needs its cache");
            let (k, v) = self.project_kv(x, 1, &[position]);
            let k = repeat_kv(
                &k,
                self.num_kv_heads,
                self.num_query_heads,
                1,
                self.head_dim,
            );
            let v = repeat_kv(
                &v,
                self.num_kv_heads,
                self.num_query_heads,
                1,
                self.head_dim,
            );
            cache.update(&k, &v);
            kv_len = cache.len;
            k_all = cache.k.clone();
            v_all = cache.v.clone();
        }

        let mut out = vec![0.0f32; 2 * self.num_query_heads * self.head_dim];
        for b in 0..2 {
            let mask_row: Option<&[f32]> = cross_mask.map(|m| m[b].as_slice());
            for h in 0..self.num_query_heads {
                let q_row = &q[(b * self.num_query_heads + h) * self.head_dim
                    ..(b * self.num_query_heads + h + 1) * self.head_dim];
                let k_base = if self.is_cross {
                    (b * self.num_query_heads + h) * kv_len * self.head_dim
                } else {
                    self_cache.as_ref().expect("self cache").base(b, h)
                };
                let attended = ops::sdpa_strided(
                    q_row,
                    &k_all,
                    k_base,
                    self.head_dim,
                    &v_all,
                    k_base,
                    self.head_dim,
                    mask_row,
                    1,
                    kv_len,
                    self.head_dim,
                    self.head_dim,
                    1.0,
                );
                out[(b * self.num_query_heads + h) * self.head_dim
                    ..(b * self.num_query_heads + h + 1) * self.head_dim]
                    .copy_from_slice(&attended);
            }
        }
        // out is head-major [2, heads, head_dim]; the head axis maps
        // straight onto the output width for a single token.
        self.o.forward(&out, 2)
    }

    /// Key/value head count (equals the query head count for the MHA
    /// cross-attention modules).
    pub(crate) fn num_kv_heads(&self) -> usize {
        self.num_kv_heads
    }

    pub(crate) fn num_query_heads(&self) -> usize {
        self.num_query_heads
    }

    pub(crate) fn head_dim(&self) -> usize {
        self.head_dim
    }
}
