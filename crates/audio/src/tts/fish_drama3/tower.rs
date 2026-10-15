//! Drama 3's audio tower: samples codebooks 1..N-1 of a frame from the
//! backbone's final hidden state, attending over time.
//!
//! Port of mlx-audio `fish_drama3/tower.py` at [`super::REFERENCE_COMMIT`],
//! as the portable f32 reference. Unlike the S2 Pro fast decoder, every
//! block keeps a KV cache across frames, and all blocks run at the same
//! temporal position: the codebook index selects the block, not the rope
//! position. A frame row is `[main, c0, ..., c{N-1}]`, where `c0` is the main
//! token's offset in the semantic band.
//!
//! The 8-bit linears stay packed and dequantize inside each matvec, and the
//! BF16 embedding tables widen one row at a time, so the real tower holds
//! about its checkpoint size (1.4 GB) rather than 4x it in f32. A prefill is
//! run one row at a time through the whole stack, which computes the same
//! causal function the reference's whole-sequence pass does.

use std::ops::RangeInclusive;

use turbospark_model_io::safetensors::SafetensorsFile;

use crate::{Result, SpeechError};

/// `audio_decoder_config`, the fields the tower reads.
#[derive(Debug, Clone, PartialEq)]
pub struct TowerConfig {
    pub dim: usize,
    pub n_head: usize,
    pub n_local_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_codebooks: usize,
    pub shared_enrichment_layers: usize,
    pub vq_backbone_codebooks: usize,
    pub norm_eps: f32,
    pub rope_base: f32,
    pub rotary_percent: f32,
}

impl TowerConfig {
    /// Reads `audio_decoder_config`. Missing optional fields take the
    /// reference's dataclass defaults (eps 1e-6, rope base 1e7, rotary 0.25).
    pub fn from_json(v: &serde_json::Value) -> Result<Self> {
        let n = |k: &str| -> Result<usize> {
            v.get(k)
                .and_then(serde_json::Value::as_u64)
                .map(|x| x as usize)
                .ok_or_else(|| SpeechError::BadConfig {
                    field: format!("audio_decoder_config.{k}"),
                    why: "missing or not a non-negative integer".into(),
                })
        };
        let f = |k: &str, default: f32| -> f32 {
            v.get(k)
                .and_then(serde_json::Value::as_f64)
                .map_or(default, |x| x as f32)
        };
        let cfg = TowerConfig {
            dim: n("dim")?,
            n_head: n("n_head")?,
            n_local_heads: n("n_local_heads")?,
            head_dim: n("head_dim")?,
            intermediate_size: n("intermediate_size")?,
            vocab_size: n("vocab_size")?,
            num_codebooks: n("num_codebooks")?,
            shared_enrichment_layers: n("shared_enrichment_layers")?,
            vq_backbone_codebooks: n("vq_backbone_codebooks")?,
            norm_eps: f("norm_eps", 1e-6),
            rope_base: f("rope_base", 1e7),
            rotary_percent: f("rotary_percent", 0.25),
        };
        cfg.validate()?;
        Ok(cfg)
    }

    fn validate(&self) -> Result<()> {
        let bad = |why: String| {
            Err(SpeechError::BadConfig {
                field: "audio_decoder_config".into(),
                why,
            })
        };
        if self.n_local_heads == 0 || self.n_head % self.n_local_heads != 0 {
            return bad(format!(
                "n_head {} is not a multiple of n_local_heads {}",
                self.n_head, self.n_local_heads
            ));
        }
        let rotary = self.rotary_dims();
        if rotary % 2 != 0 || rotary > self.head_dim {
            return bad(format!(
                "rotary dims {rotary} must be even and fit head_dim"
            ));
        }
        if self.num_codebooks < 2 || self.vq_backbone_codebooks + 1 > self.num_codebooks {
            return bad(format!(
                "num_codebooks {} must exceed vq_backbone_codebooks {}",
                self.num_codebooks, self.vq_backbone_codebooks
            ));
        }
        Ok(())
    }

    /// Frame row width: the main token, then every codebook.
    pub fn frame_width(&self) -> usize {
        1 + self.num_codebooks
    }

    fn rotary_dims(&self) -> usize {
        (self.head_dim as f32 * self.rotary_percent) as usize
    }

    fn q_width(&self) -> usize {
        self.n_head * self.head_dim
    }

    fn kv_width(&self) -> usize {
        self.n_local_heads * self.head_dim
    }
}

fn bf16(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

fn bf16_vec(file: &SafetensorsFile, name: &str, want: &[usize]) -> Result<Vec<u16>> {
    let desc = file.descriptor(name).ok_or_else(|| SpeechError::Tensor {
        name: name.into(),
        why: "missing from checkpoint".into(),
    })?;
    if desc.dtype != "BF16" || desc.shape != want {
        return Err(SpeechError::Tensor {
            name: name.into(),
            why: format!(
                "expected BF16 {want:?}, got {} {:?}",
                desc.dtype, desc.shape
            ),
        });
    }
    file.load_as_u16(name).map_err(SpeechError::from)
}

fn f32_vec(file: &SafetensorsFile, name: &str, len: usize) -> Result<Vec<f32>> {
    let desc = file.descriptor(name).ok_or_else(|| SpeechError::Tensor {
        name: name.into(),
        why: "missing from checkpoint".into(),
    })?;
    if desc.shape != [len] {
        return Err(SpeechError::Tensor {
            name: name.into(),
            why: format!("expected [{len}], got {:?}", desc.shape),
        });
    }
    file.load_as_f32(name).map_err(SpeechError::from)
}

/// A BF16 table, widened one row at a time.
struct Table {
    rows: usize,
    cols: usize,
    bits: Vec<u16>,
}

impl Table {
    fn load(file: &SafetensorsFile, name: &str, rows: usize, cols: usize) -> Result<Self> {
        Ok(Table {
            rows,
            cols,
            bits: bf16_vec(file, name, &[rows, cols])?,
        })
    }

    fn add_row(&self, row: usize, out: &mut [f32]) {
        let row = row.min(self.rows - 1);
        for (o, &b) in out
            .iter_mut()
            .zip(&self.bits[row * self.cols..][..self.cols])
        {
            *o += bf16(b);
        }
    }
}

/// An MLX 8-bit affine linear (`[out, in]`, group 64), kept packed.
struct Q8Linear {
    rows: usize,
    cols: usize,
    q: Vec<u8>,
    scales: Vec<f32>,
    biases: Vec<f32>,
}

const GROUP: usize = 64;
/// Multiply-adds below which a matvec stays on one thread.
const PARALLEL_MACS: usize = 1 << 20;

impl Q8Linear {
    fn load(file: &SafetensorsFile, base: &str, rows: usize, cols: usize) -> Result<Self> {
        let weight = format!("{base}.weight");
        let desc = file
            .descriptor(&weight)
            .ok_or_else(|| SpeechError::Tensor {
                name: weight.clone(),
                why: "missing from checkpoint".into(),
            })?;
        // 8 bits: four values per U32 word, little-endian bytes in order.
        if desc.dtype != "U32" || desc.shape != [rows, cols / 4] || cols % GROUP != 0 {
            return Err(SpeechError::Unsupported {
                why: format!(
                    "{weight}: expected 8-bit group-{GROUP} U32 [{rows}, {}], got {} {:?}",
                    cols / 4,
                    desc.dtype,
                    desc.shape
                ),
            });
        }
        let q = file.raw_bytes(&weight).map_err(SpeechError::from)?.to_vec();
        let groups = cols / GROUP;
        let scales = bf16_vec(file, &format!("{base}.scales"), &[rows, groups])?;
        let biases = bf16_vec(file, &format!("{base}.biases"), &[rows, groups])?;
        Ok(Q8Linear {
            rows,
            cols,
            q,
            scales: scales.into_iter().map(bf16).collect(),
            biases: biases.into_iter().map(bf16).collect(),
        })
    }

    fn row(&self, r: usize, x: &[f32], group_sums: &[f32]) -> f32 {
        let groups = self.cols / GROUP;
        let row = &self.q[r * self.cols..][..self.cols];
        let mut acc = 0.0f32;
        for g in 0..groups {
            let dot: f32 = row[g * GROUP..][..GROUP]
                .iter()
                .zip(&x[g * GROUP..][..GROUP])
                .map(|(&q, &v)| f32::from(q) * v)
                .sum();
            let k = r * groups + g;
            acc += self.scales[k] * dot + self.biases[k] * group_sums[g];
        }
        acc
    }

    fn forward(&self, x: &[f32]) -> Vec<f32> {
        debug_assert_eq!(x.len(), self.cols);
        let group_sums: Vec<f32> = x.chunks_exact(GROUP).map(|g| g.iter().sum()).collect();
        let mut out = vec![0.0f32; self.rows];
        let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
        // Rows split across scoped threads above a size where the spawns pay
        // for themselves. Each row's sum is computed the same way either way,
        // so the result does not depend on the split.
        if threads < 2 || self.rows * self.cols < PARALLEL_MACS {
            for (r, o) in out.iter_mut().enumerate() {
                *o = self.row(r, x, &group_sums);
            }
            return out;
        }
        let per = self.rows.div_ceil(threads);
        std::thread::scope(|scope| {
            for (chunk, slice) in out.chunks_mut(per).enumerate() {
                let group_sums = &group_sums;
                scope.spawn(move || {
                    for (i, o) in slice.iter_mut().enumerate() {
                        *o = self.row(chunk * per + i, x, group_sums);
                    }
                });
            }
        });
        out
    }
}

/// `Fp32RMSNorm`: `x * rsqrt(mean(x^2) + eps) * w`.
fn rms_norm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let mean = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv = 1.0 / (mean + eps).sqrt();
    x.iter().zip(w).map(|(v, g)| v * inv * g).collect()
}

/// Interleaved ("traditional") rope over the first `dims` values of one head.
fn rope_interleaved(head: &mut [f32], dims: usize, base: f32, position: usize) {
    for i in 0..dims / 2 {
        let inv_freq = base.powf(-((2 * i) as f32) / dims as f32);
        let (sin, cos) = (position as f32 * inv_freq).sin_cos();
        let (a, b) = (head[2 * i], head[2 * i + 1]);
        head[2 * i] = a * cos - b * sin;
        head[2 * i + 1] = a * sin + b * cos;
    }
}

struct Block {
    attention_norm: Vec<f32>,
    wqkv: Q8Linear,
    wo: Q8Linear,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    w1: Q8Linear,
    w2: Q8Linear,
    w3: Q8Linear,
}

/// One block's KV rows, `kv_width` values each.
#[derive(Default, Clone)]
struct BlockCache {
    keys: Vec<f32>,
    values: Vec<f32>,
}

impl Block {
    fn load(file: &SafetensorsFile, p: &str, cfg: &TowerConfig) -> Result<Self> {
        let (dim, inter) = (cfg.dim, cfg.intermediate_size);
        let (qw, kvw) = (cfg.q_width(), cfg.kv_width());
        Ok(Block {
            attention_norm: f32_vec(file, &format!("{p}.attention_norm.weight"), dim)?,
            wqkv: Q8Linear::load(file, &format!("{p}.attention.wqkv"), 2 * qw + 2 * kvw, dim)?,
            wo: Q8Linear::load(file, &format!("{p}.attention.wo"), dim, qw)?,
            q_norm: f32_vec(file, &format!("{p}.attention.q_norm.weight"), cfg.head_dim)?,
            k_norm: f32_vec(file, &format!("{p}.attention.k_norm.weight"), cfg.head_dim)?,
            ffn_norm: f32_vec(file, &format!("{p}.ffn_norm.weight"), dim)?,
            w1: Q8Linear::load(file, &format!("{p}.feed_forward.w1"), inter, dim)?,
            w2: Q8Linear::load(file, &format!("{p}.feed_forward.w2"), dim, inter)?,
            w3: Q8Linear::load(file, &format!("{p}.feed_forward.w3"), inter, dim)?,
        })
    }

    /// One row at `position`, appending its K/V to `cache`.
    fn forward(
        &self,
        x: &[f32],
        cfg: &TowerConfig,
        position: usize,
        cache: &mut BlockCache,
    ) -> Vec<f32> {
        let (hd, nh, nkv) = (cfg.head_dim, cfg.n_head, cfg.n_local_heads);
        let (qw, kvw) = (cfg.q_width(), cfg.kv_width());
        let rotary = cfg.rotary_dims();

        let normed = rms_norm(x, &self.attention_norm, cfg.norm_eps);
        let qkv = self.wqkv.forward(&normed);
        // Rows are [all Q | all gates | K | V].
        let (q, rest) = qkv.split_at(qw);
        let (gate, rest) = rest.split_at(qw);
        let (k, v) = rest.split_at(kvw);

        let mut q: Vec<f32> = q
            .chunks_exact(hd)
            .flat_map(|h| rms_norm(h, &self.q_norm, cfg.norm_eps))
            .collect();
        let mut k: Vec<f32> = k
            .chunks_exact(hd)
            .flat_map(|h| rms_norm(h, &self.k_norm, cfg.norm_eps))
            .collect();
        for head in q.chunks_exact_mut(hd).chain(k.chunks_exact_mut(hd)) {
            rope_interleaved(head, rotary, cfg.rope_base, position);
        }
        cache.keys.extend_from_slice(&k);
        cache.values.extend_from_slice(v);

        let positions = cache.keys.len() / kvw;
        let scale = (hd as f32).powf(-0.5);
        let group = nh / nkv;
        let mut attn = vec![0.0f32; qw];
        for h in 0..nh {
            let kv = h / group;
            let qh = &q[h * hd..][..hd];
            let scores: Vec<f32> = (0..positions)
                .map(|p| {
                    let kp = &cache.keys[p * kvw + kv * hd..][..hd];
                    qh.iter().zip(kp).map(|(a, b)| a * b).sum::<f32>() * scale
                })
                .collect();
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let weights: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
            let total: f32 = weights.iter().sum();
            let out = &mut attn[h * hd..][..hd];
            for (p, w) in weights.iter().enumerate() {
                let vp = &cache.values[p * kvw + kv * hd..][..hd];
                for (o, val) in out.iter_mut().zip(vp) {
                    *o += w / total * val;
                }
            }
        }
        for (a, g) in attn.iter_mut().zip(gate) {
            *a *= 1.0 / (1.0 + (-g).exp());
        }
        let mut x: Vec<f32> = x
            .iter()
            .zip(self.wo.forward(&attn))
            .map(|(a, b)| a + b)
            .collect();

        let normed = rms_norm(&x, &self.ffn_norm, cfg.norm_eps);
        let a = self.w1.forward(&normed);
        let b = self.w3.forward(&normed);
        let act: Vec<f32> = a
            .iter()
            .zip(&b)
            .map(|(a, b)| a / (1.0 + (-a).exp()) * b)
            .collect();
        for (xi, d) in x.iter_mut().zip(self.w2.forward(&act)) {
            *xi += d;
        }
        x
    }
}

/// `TowerLayer`: mixes a code embedding (zero on text rows) into the hidden
/// state, then runs one block.
struct Layer {
    enorm: Vec<f32>,
    hnorm: Vec<f32>,
    eh_proj: Q8Linear,
    block: Block,
    final_layernorm: Vec<f32>,
}

impl Layer {
    fn load(file: &SafetensorsFile, p: &str, cfg: &TowerConfig) -> Result<Self> {
        let dim = cfg.dim;
        Ok(Layer {
            enorm: f32_vec(file, &format!("{p}.enorm.weight"), dim)?,
            hnorm: f32_vec(file, &format!("{p}.hnorm.weight"), dim)?,
            eh_proj: Q8Linear::load(file, &format!("{p}.eh_proj"), dim, 2 * dim)?,
            block: Block::load(file, &format!("{p}.transformer_layer"), cfg)?,
            final_layernorm: f32_vec(file, &format!("{p}.final_layernorm.weight"), dim)?,
        })
    }

    /// One row: `e` is this layer's input embedding
    /// ([`TowerEmbeddings::layer_input`]).
    fn forward(
        &self,
        hidden: &[f32],
        e: &[f32],
        cfg: &TowerConfig,
        position: usize,
        cache: &mut BlockCache,
    ) -> Vec<f32> {
        let mut cat = rms_norm(e, &self.enorm, cfg.norm_eps);
        cat.extend(rms_norm(hidden, &self.hnorm, cfg.norm_eps));
        let mixed = self.eh_proj.forward(&cat);
        let out = self.block.forward(&mixed, cfg, position, cache);
        rms_norm(&out, &self.final_layernorm, cfg.norm_eps)
    }
}

/// Every block's KV cache, all at the same temporal position.
#[derive(Clone)]
pub struct TowerCache {
    blocks: Vec<BlockCache>,
    position: usize,
}

impl TowerCache {
    /// Rows the cache holds.
    pub fn position(&self) -> usize {
        self.position
    }

    /// Drops every row at or after `position`.
    pub fn truncate(&mut self, position: usize) -> Result<()> {
        if position > self.position {
            return Err(SpeechError::Input {
                why: format!("cannot truncate {} tower rows to {position}", self.position),
            });
        }
        for block in &mut self.blocks {
            let width = block.keys.len().checked_div(self.position).unwrap_or(0);
            block.keys.truncate(position * width);
            block.values.truncate(position * width);
        }
        self.position = position;
        Ok(())
    }
}

/// The tower's embedding tables and the host-side arithmetic over them: the
/// part every tower implementation shares (this f32 reference and the
/// runtime's Metal tower).
pub struct TowerEmbeddings {
    cfg: TowerConfig,
    semantic: RangeInclusive<i32>,
    codebook: Table,
    /// One table per layer: the enrichment layers' `semantic_embedding`, then
    /// the residual layers' `codebook_embedding`.
    layers: Vec<Table>,
}

impl TowerEmbeddings {
    /// Loads the tables from the side file. `semantic` is the semantic token
    /// band, whose width must be the codebook vocabulary.
    pub fn load(
        file: &SafetensorsFile,
        cfg: &TowerConfig,
        semantic: RangeInclusive<i32>,
    ) -> Result<Self> {
        let band = (semantic.end() - semantic.start() + 1) as usize;
        if band != cfg.vocab_size {
            return Err(SpeechError::BadConfig {
                field: "semantic_start_token_id..semantic_end_token_id".into(),
                why: format!(
                    "band of {band} tokens, codebook vocabulary {}",
                    cfg.vocab_size
                ),
            });
        }
        let (rows, dim) = (cfg.vocab_size, cfg.dim);
        let mut layers = Vec::new();
        for i in 0..cfg.shared_enrichment_layers {
            let name = format!("tower.shared_layers.{i}.semantic_embedding.weight");
            layers.push(Table::load(file, &name, rows, dim)?);
        }
        for i in 0..cfg.num_codebooks - 1 {
            let name = format!("tower.residual_layers.{i}.codebook_embedding.weight");
            layers.push(Table::load(file, &name, rows, dim)?);
        }
        Ok(TowerEmbeddings {
            codebook: Table::load(
                file,
                "tower.codebook_embeddings.weight",
                rows * cfg.num_codebooks,
                dim,
            )?,
            layers,
            cfg: cfg.clone(),
            semantic,
        })
    }

    /// True for a token in the semantic band.
    pub fn is_semantic(&self, token: i32) -> bool {
        self.semantic.contains(&token)
    }

    /// A main token's offset in the semantic band, clamped as the reference
    /// clamps it (EOS maps to 0 and is masked by `semantic`).
    pub fn c0(&self, main: i32) -> i32 {
        (main - self.semantic.start()).clamp(0, self.cfg.vocab_size as i32 - 1)
    }

    /// Sum of the shared table's rows for codebooks `first..end` of `frame`.
    fn embedding_sum(&self, frame: &[i32], first: usize, end: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; self.cfg.dim];
        for book in first..end {
            let code = frame[1 + book].clamp(0, self.cfg.vocab_size as i32 - 1) as usize;
            self.codebook
                .add_row(code + book * self.cfg.vocab_size, &mut out);
        }
        out
    }

    /// The backbone input for `frame`, given its token's embedding row
    /// `text`. A semantic row adds its first residual codebooks and rescales;
    /// a text row keeps the token embedding.
    pub fn embed_backbone(&self, frame: &[i32], text: &[f32]) -> Vec<f32> {
        if !self.is_semantic(frame[0]) {
            return text.to_vec();
        }
        let count = self.cfg.vq_backbone_codebooks;
        let scale = ((1 + count) as f32).powf(-0.5);
        let sum = self.embedding_sum(frame, 1, 1 + count);
        text.iter()
            .zip(&sum)
            .map(|(t, s)| (t + s) * scale)
            .collect()
    }

    /// The codebooks of `frame` the backbone input left out, which the first
    /// enrichment layer adds; zero on a text row.
    pub fn unseen(&self, frame: &[i32]) -> Vec<f32> {
        if self.is_semantic(frame[0]) {
            let cfg = &self.cfg;
            self.embedding_sum(frame, 1 + cfg.vq_backbone_codebooks, cfg.num_codebooks)
        } else {
            vec![0.0f32; self.cfg.dim]
        }
    }

    /// Layer `layer`'s input embedding (enrichment layers first): its own row
    /// for `code` on a semantic row, zero otherwise, plus `extra`.
    pub fn layer_input(
        &self,
        layer: usize,
        code: i32,
        semantic: bool,
        extra: Option<&[f32]>,
    ) -> Vec<f32> {
        let mut e = vec![0.0f32; self.cfg.dim];
        if semantic {
            let row = code.clamp(0, self.cfg.vocab_size as i32 - 1) as usize;
            self.layers[layer].add_row(row, &mut e);
        }
        if let Some(extra) = extra {
            for (a, b) in e.iter_mut().zip(extra) {
                *a += b;
            }
        }
        e
    }

    /// Every layer's input for one row, in layer order, given how each
    /// residual layer's code is chosen: `labels` (prefill: the next row's
    /// codes) or `None` for decode, where only the enrichment layers and
    /// residual layer 0 are known up front.
    pub fn row_inputs(
        &self,
        frame: &[i32],
        c0: i32,
        semantic: bool,
        labels: Option<&[i32]>,
    ) -> Vec<Vec<f32>> {
        let unseen = self.unseen(frame);
        let shared = self.cfg.shared_enrichment_layers;
        let mut out: Vec<Vec<f32>> = (0..shared)
            .map(|i| self.layer_input(i, c0, semantic, (i == 0).then_some(unseen.as_slice())))
            .collect();
        match labels {
            Some(next) => {
                for i in 0..self.cfg.num_codebooks - 1 {
                    out.push(self.layer_input(shared + i, next[i + 1], semantic, None));
                }
            }
            None => out.push(self.layer_input(shared, c0, semantic, None)),
        }
        out
    }
}

/// A tower the generation loop can drive, holding its own cache.
pub trait TowerModel {
    fn config(&self) -> &TowerConfig;
    fn embeddings(&self) -> &TowerEmbeddings;
    /// Rows the cache holds.
    fn position(&self) -> usize;
    /// Drops every row at or after `position`.
    fn truncate(&mut self, position: usize) -> Result<()>;
    /// The semantic head's logits: the semantic band in order, then EOS.
    fn semantic_logits(&mut self, hidden: &[f32]) -> Result<Vec<f32>>;
    /// Teacher-forces prompt rows (see [`Tower::prefill`]).
    fn prefill(&mut self, hidden: &[f32], frames: &[&[i32]], labels: &[&[i32]]) -> Result<()>;
    /// One frame's codes (see [`Tower::residual_codes`]).
    fn residual_codes(
        &mut self,
        hidden: &[f32],
        main: i32,
        frame: &[i32],
        sample: &mut dyn FnMut(usize, &[f32]) -> i32,
    ) -> Result<Vec<i32>>;
}

/// The f32 reference tower with its own cache.
pub struct CpuTower {
    tower: Tower,
    cache: TowerCache,
}

impl CpuTower {
    pub fn new(tower: Tower) -> Self {
        let cache = tower.new_cache();
        CpuTower { tower, cache }
    }
}

impl TowerModel for CpuTower {
    fn config(&self) -> &TowerConfig {
        &self.tower.cfg
    }

    fn embeddings(&self) -> &TowerEmbeddings {
        &self.tower.embeddings
    }

    fn position(&self) -> usize {
        self.cache.position()
    }

    fn truncate(&mut self, position: usize) -> Result<()> {
        self.cache.truncate(position)
    }

    fn semantic_logits(&mut self, hidden: &[f32]) -> Result<Vec<f32>> {
        Ok(self.tower.semantic_logits(hidden))
    }

    fn prefill(&mut self, hidden: &[f32], frames: &[&[i32]], labels: &[&[i32]]) -> Result<()> {
        self.tower.prefill(hidden, frames, labels, &mut self.cache)
    }

    fn residual_codes(
        &mut self,
        hidden: &[f32],
        main: i32,
        frame: &[i32],
        sample: &mut dyn FnMut(usize, &[f32]) -> i32,
    ) -> Result<Vec<i32>> {
        self.tower
            .residual_codes(hidden, main, frame, &mut self.cache, sample)
    }
}

/// Where a row's codes come from.
enum Codes<'a, F> {
    /// Prefill: the following row's codes label this one.
    Labelled(&'a [i32]),
    /// Decode: `c0` is given, `c1..` are drawn from each head's logits.
    Sampled(i32, F),
}

/// The tower plus the semantic head that picks each frame's main token.
pub struct Tower {
    cfg: TowerConfig,
    embeddings: TowerEmbeddings,
    shared: Vec<Layer>,
    residual: Vec<Layer>,
    heads: Vec<Q8Linear>,
    semantic_head: Q8Linear,
}

impl Tower {
    /// Loads `tower.*` and `semantic_head.*` from the side file. `semantic` is
    /// the semantic token band (`semantic_start_token_id..=semantic_end_token_id`),
    /// whose width must be the codebook vocabulary.
    pub fn load(
        file: &SafetensorsFile,
        cfg: TowerConfig,
        semantic: RangeInclusive<i32>,
    ) -> Result<Self> {
        let embeddings = TowerEmbeddings::load(file, &cfg, semantic)?;
        let dim = cfg.dim;
        let shared = (0..cfg.shared_enrichment_layers)
            .map(|i| Layer::load(file, &format!("tower.shared_layers.{i}"), &cfg))
            .collect::<Result<_>>()?;
        let residual = (0..cfg.num_codebooks - 1)
            .map(|i| Layer::load(file, &format!("tower.residual_layers.{i}"), &cfg))
            .collect::<Result<_>>()?;
        let heads = (0..cfg.num_codebooks - 1)
            .map(|i| {
                Q8Linear::load(
                    file,
                    &format!("tower.codebook_heads.{i}"),
                    cfg.vocab_size,
                    dim,
                )
            })
            .collect::<Result<_>>()?;
        Ok(Tower {
            embeddings,
            shared,
            residual,
            heads,
            semantic_head: Q8Linear::load(file, "semantic_head", cfg.vocab_size + 1, dim)?,
            cfg,
        })
    }

    pub fn config(&self) -> &TowerConfig {
        &self.cfg
    }

    /// An empty cache for every block.
    pub fn new_cache(&self) -> TowerCache {
        TowerCache {
            blocks: vec![BlockCache::default(); self.shared.len() + self.residual.len()],
            position: 0,
        }
    }

    /// True for a token in the semantic band.
    pub fn is_semantic(&self, token: i32) -> bool {
        self.embeddings.is_semantic(token)
    }

    /// The embedding tables shared with every tower implementation.
    pub fn embeddings(&self) -> &TowerEmbeddings {
        &self.embeddings
    }

    /// The semantic head's logits for a hidden state: the semantic band in
    /// order, then EOS.
    pub fn semantic_logits(&self, hidden: &[f32]) -> Vec<f32> {
        self.semantic_head.forward(hidden)
    }

    /// See [`TowerEmbeddings::embed_backbone`].
    pub fn embed_backbone(&self, frame: &[i32], text: &[f32]) -> Vec<f32> {
        self.embeddings.embed_backbone(frame, text)
    }

    fn check(&self, hidden: &[f32], frame: &[i32]) -> Result<()> {
        if hidden.len() != self.cfg.dim || frame.len() != self.cfg.frame_width() {
            return Err(SpeechError::Input {
                why: format!(
                    "tower row needs {} hidden values and a {}-wide frame, got {} and {}",
                    self.cfg.dim,
                    self.cfg.frame_width(),
                    hidden.len(),
                    frame.len()
                ),
            });
        }
        Ok(())
    }

    fn run_row<F: FnMut(usize, &[f32]) -> i32>(
        &self,
        hidden: &[f32],
        frame: &[i32],
        semantic: bool,
        mut codes: Codes<'_, F>,
        cache: &mut TowerCache,
    ) -> Vec<i32> {
        let cfg = &self.cfg;
        let position = cache.position;
        let c0 = match &codes {
            Codes::Labelled(next) => next[1],
            Codes::Sampled(c0, _) => *c0,
        };
        let labels = match &codes {
            Codes::Labelled(next) => Some(*next),
            Codes::Sampled(..) => None,
        };
        // The enrichment layers' inputs (the first also sees this row's
        // codebooks the backbone input left out), then residual layer 0's,
        // or every residual layer's when labels are given.
        let inputs = self.embeddings.row_inputs(frame, c0, semantic, labels);
        let mut h = hidden.to_vec();
        for (i, layer) in self.shared.iter().enumerate() {
            h = layer.forward(&h, &inputs[i], cfg, position, &mut cache.blocks[i]);
        }
        let offset = self.shared.len();
        let mut out = vec![c0];
        for (i, layer) in self.residual.iter().enumerate() {
            let e = match inputs.get(offset + i) {
                Some(e) => e.clone(),
                // Decode: this layer's code was sampled by the previous head.
                None => self
                    .embeddings
                    .layer_input(offset + i, out[i], semantic, None),
            };
            h = layer.forward(&h, &e, cfg, position, &mut cache.blocks[offset + i]);
            if let Codes::Sampled(_, sample) = &mut codes {
                out.push(sample(i, &self.heads[i].forward(&h)));
            }
        }
        cache.position += 1;
        out
    }

    /// Teacher-forces the caches over prompt rows: row `t` (backbone hidden
    /// `hidden[t]`, frame `frames[t]`) is labelled by `next_frames[t]`.
    pub fn prefill(
        &self,
        hidden: &[f32],
        frames: &[&[i32]],
        next_frames: &[&[i32]],
        cache: &mut TowerCache,
    ) -> Result<()> {
        let dim = self.cfg.dim;
        if hidden.len() != frames.len() * dim || frames.len() != next_frames.len() {
            return Err(SpeechError::Input {
                why: format!(
                    "prefill needs one hidden row and one label per frame: {} values, {} frames, {} labels",
                    hidden.len(),
                    frames.len(),
                    next_frames.len()
                ),
            });
        }
        for ((h, frame), next) in hidden.chunks_exact(dim).zip(frames).zip(next_frames) {
            self.check(h, frame)?;
            self.check(h, next)?;
            let semantic = self.is_semantic(next[0]);
            self.run_row(
                h,
                frame,
                semantic,
                Codes::<fn(usize, &[f32]) -> i32>::Labelled(next),
                cache,
            );
        }
        Ok(())
    }

    /// Codes `c0..c{N-1}` of one frame, drawing `c1..` with
    /// `sample(head_index, logits)`. `frame` is the row the backbone was fed
    /// to produce `hidden`; `main` is this frame's main token.
    pub fn residual_codes(
        &self,
        hidden: &[f32],
        main: i32,
        frame: &[i32],
        cache: &mut TowerCache,
        sample: impl FnMut(usize, &[f32]) -> i32,
    ) -> Result<Vec<i32>> {
        self.check(hidden, frame)?;
        let c0 = self.embeddings.c0(main);
        let semantic = self.is_semantic(main);
        Ok(self.run_row(hidden, frame, semantic, Codes::Sampled(c0, sample), cache))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn golden() -> (serde_json::Value, SafetensorsFile) {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/fish_drama3");
        let text = std::fs::read_to_string(dir.join("tower.json")).expect("tower fixture");
        let golden: serde_json::Value = serde_json::from_str(&text).expect("fixture parses");
        assert_eq!(golden["commit"], super::super::REFERENCE_COMMIT);
        let file = SafetensorsFile::open(&dir.join("tower_side.safetensors")).expect("weights");
        (golden, file)
    }

    fn floats(v: &serde_json::Value) -> Vec<f32> {
        v.as_array()
            .expect("array")
            .iter()
            .map(|x| x.as_f64().expect("number") as f32)
            .collect()
    }

    fn ints(v: &serde_json::Value) -> Vec<i32> {
        v.as_array()
            .expect("array")
            .iter()
            .map(|x| x.as_i64().expect("int") as i32)
            .collect()
    }

    fn load(golden: &serde_json::Value, file: &SafetensorsFile) -> Tower {
        let cfg = TowerConfig::from_json(&golden["tower"]).expect("config");
        let band = golden["semantic_begin"].as_i64().unwrap() as i32
            ..=golden["semantic_end"].as_i64().unwrap() as i32;
        Tower::load(file, cfg, band).expect("tower loads")
    }

    /// Worst absolute error relative to the reference's largest magnitude.
    fn rel_err(got: &[f32], want: &[f32]) -> f32 {
        let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-6);
        got.iter()
            .zip(want)
            .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()))
            / scale
    }

    #[test]
    fn decode_matches_the_reference_after_a_teacher_forced_prefill() {
        let (golden, file) = golden();
        let tower = load(&golden, &file);
        let prompt: Vec<Vec<i32>> = golden["prompt"]
            .as_array()
            .unwrap()
            .iter()
            .map(ints)
            .collect();
        let hidden: Vec<f32> = golden["prompt_hidden"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(floats)
            .collect();
        let dim = tower.config().dim;
        let last = prompt.len() - 1;
        let frames: Vec<&[i32]> = prompt[..last].iter().map(Vec::as_slice).collect();
        let labels: Vec<&[i32]> = prompt[1..].iter().map(Vec::as_slice).collect();
        let mut cache = tower.new_cache();
        tower
            .prefill(&hidden[..last * dim], &frames, &labels, &mut cache)
            .unwrap();
        assert_eq!(cache.position(), last);

        let mut frame = prompt[last].clone();
        let mut worst = 0.0f32;
        for (s, step) in golden["steps"].as_array().unwrap().iter().enumerate() {
            let want_logits: Vec<Vec<f32>> = step["head_logits"]
                .as_array()
                .unwrap()
                .iter()
                .map(floats)
                .collect();
            let main = step["main"].as_i64().unwrap() as i32;
            let h = floats(&step["hidden"]);
            let mut got_logits = Vec::new();
            let codes = tower
                .residual_codes(&h, main, &frame, &mut cache, |_, logits| {
                    got_logits.push(logits.to_vec());
                    crate::tts::sampling::argmax(logits) as i32
                })
                .unwrap();
            for (head, (got, want)) in got_logits.iter().zip(&want_logits).enumerate() {
                let err = rel_err(got, want);
                worst = worst.max(err);
                eprintln!("step {s} head {head}: relative error {err:.2e}");
            }
            let mut got_frame = vec![main];
            got_frame.extend(&codes);
            assert_eq!(got_frame, ints(&step["frame"]), "step {s}: greedy codes");
            frame = got_frame;
        }
        // bf16 activations in the reference against f32 here.
        assert!(worst < 3e-2, "worst relative logit error {worst:.2e}");
    }

    #[test]
    fn backbone_input_mix_matches_the_reference() {
        let (golden, file) = golden();
        let tower = load(&golden, &file);
        let prompt: Vec<Vec<i32>> = golden["prompt"]
            .as_array()
            .unwrap()
            .iter()
            .map(ints)
            .collect();
        let text: Vec<Vec<f32>> = golden["embed_text"]
            .as_array()
            .unwrap()
            .iter()
            .map(floats)
            .collect();
        let want: Vec<Vec<f32>> = golden["embed_mixed"]
            .as_array()
            .unwrap()
            .iter()
            .map(floats)
            .collect();
        for ((frame, t), w) in prompt.iter().zip(&text).zip(&want) {
            let got = tower.embed_backbone(frame, t);
            let err = rel_err(&got, w);
            assert!(err < 1e-2, "frame {frame:?}: relative error {err:.2e}");
        }
    }
}
