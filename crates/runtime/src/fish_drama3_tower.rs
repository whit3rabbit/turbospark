//! Drama 3's audio tower on Metal, from the qwen36 flow's own kernels.
//!
//! The same model as `audio::tts::fish_drama3::tower::Tower` (the f32
//! reference, which this must match), run as f16 activations over the
//! 8-bit weights where they sit:
//!
//! - **Linears** use the INT8 affine GEMV (`dequant_int8_gemv_simd`) on the
//!   checkpoint's own packed rows, BF16 scales and biases. `wqkv` is three
//!   row ranges: the query and gate rows into scratch, and the K and V rows
//!   straight into this position's cache row, so no split or copy kernel runs.
//! - **Norms** use the centered RMSNorm kernels: each stored gamma (already
//!   `1 + w` in F32) is kept as BF16 `gamma - 1`, which the kernel adds back
//!   to 1 in f32, so a gamma near 1 keeps its precision.
//! - **Rope.** The tower rotates interleaved pairs `(2i, 2i + 1)` over the
//!   first quarter of each head. The qwen flow's `rope_neox_subdim` rotates
//!   pairs `(i, i + rotary / 2)` with the same frequency `theta^(-2i/rotary)`.
//!   Permuting each query and key head's first `rotary` rows to evens then
//!   odds (and the q/k norm gammas with them) turns one into the other.
//!   Attention scores are dot products over identically permuted q and k, so
//!   they are unchanged; values and gates are not permuted.
//! - **Attention** is the split-KV decode kernel over this tower's own K/V
//!   buffers, one pair per block, all blocks at the same position.
//!
//! A decode frame is ten command buffers: the semantic head, then the
//! enrichment layers with residual layer 0 and head 0, then one per remaining
//! residual layer, each ending in a host sample. A prefill row needs no
//! sample (its codes are the next row's), so many rows share one buffer.

use std::ops::RangeInclusive;

use audio::tts::fish_drama3::tower::{TowerConfig, TowerEmbeddings, TowerModel};
use audio::{Result, SpeechError};
use half::f16;
use model_io::safetensors::SafetensorsFile;

/// Prefill rows encoded into one command buffer.
const PREFILL_ROWS_PER_PASS: usize = 32;
/// Offsets into the weight arena are aligned for buffer binding.
const ALIGN: usize = 256;

fn gpu_err(e: gpu::GpuError) -> SpeechError {
    SpeechError::Unsupported {
        why: format!("metal tower: {e}"),
    }
}

fn tensor_err(name: &str, why: String) -> SpeechError {
    SpeechError::Tensor {
        name: name.to_string(),
        why,
    }
}

/// An 8-bit matrix in the arena.
#[derive(Clone, Copy, Debug)]
struct Mat {
    weights: u64,
    scales: u64,
    biases: u64,
    rows: usize,
    cols: usize,
}

struct LayerGpu {
    enorm: u64,
    hnorm: u64,
    eh_proj: Mat,
    attention_norm: u64,
    qg: Mat,
    k: Mat,
    v: Mat,
    q_norm: u64,
    k_norm: u64,
    wo: Mat,
    ffn_norm: u64,
    w1: Mat,
    w3: Mat,
    w2: Mat,
    final_norm: u64,
}

/// Lays out the weight arena. The first walk only measures, the second
/// writes each piece straight into the Metal buffer, so the host never
/// holds more than one matrix (the arena is about 1.1 GB on the real tower).
struct Arena {
    len: usize,
    target: Option<gpu::MetalBuffer>,
}

impl Arena {
    fn push(&mut self, len: usize, fill: impl FnOnce() -> Result<Vec<u8>>) -> Result<u64> {
        let at = self.len.next_multiple_of(ALIGN);
        if let Some(buffer) = &self.target {
            let bytes = fill()?;
            assert_eq!(bytes.len(), len, "arena piece changed size between walks");
            gpu::write_buffer_bytes(buffer, at, &bytes);
        }
        self.len = at + len;
        Ok(at as u64)
    }

    /// An F32 gamma `[len]`, stored as BF16 `gamma - 1` in `order`.
    fn norm(
        &mut self,
        file: &SafetensorsFile,
        name: &str,
        len: usize,
        order: &[usize],
    ) -> Result<u64> {
        let desc = file
            .descriptor(name)
            .ok_or_else(|| tensor_err(name, "missing from checkpoint".into()))?;
        if desc.shape != [len] {
            return Err(tensor_err(
                name,
                format!("expected [{len}], got {:?}", desc.shape),
            ));
        }
        self.push(order.len() * 2, || {
            let gamma = file.load_as_f32(name).map_err(SpeechError::from)?;
            Ok(order
                .iter()
                .flat_map(|&i| compute::f32_to_bf16(gamma[i] - 1.0).to_le_bytes())
                .collect())
        })
    }

    /// Rows `rows` (in that order) of an MLX 8-bit `[out, cols]` matrix.
    fn matrix(
        &mut self,
        file: &SafetensorsFile,
        base: &str,
        out: usize,
        cols: usize,
        rows: &[usize],
    ) -> Result<Mat> {
        let weight = format!("{base}.weight");
        let desc = file
            .descriptor(&weight)
            .ok_or_else(|| tensor_err(&weight, "missing from checkpoint".into()))?;
        if desc.dtype != "U32" || desc.shape != [out, cols / 4] || cols % 64 != 0 {
            return Err(SpeechError::Unsupported {
                why: format!(
                    "{weight}: expected 8-bit group-64 U32 [{out}, {}], got {} {:?}",
                    cols / 4,
                    desc.dtype,
                    desc.shape
                ),
            });
        }
        let packed = file.raw_bytes(&weight).map_err(SpeechError::from)?;
        let groups = cols / 64;
        let companion = |suffix: &str| -> Result<&[u8]> {
            let name = format!("{base}.{suffix}");
            let d = file
                .descriptor(&name)
                .ok_or_else(|| tensor_err(&name, "missing from checkpoint".into()))?;
            if d.dtype != "BF16" || d.shape != [out, groups] {
                return Err(tensor_err(
                    &name,
                    format!("expected BF16 [{out}, {groups}]"),
                ));
            }
            file.raw_bytes(&name).map_err(SpeechError::from)
        };
        let (scales, biases) = (companion("scales")?, companion("biases")?);
        let gather = |src: &[u8], width: usize| -> Result<Vec<u8>> {
            Ok(rows
                .iter()
                .flat_map(|&r| src[r * width..][..width].iter().copied())
                .collect())
        };
        let n = rows.len();
        Ok(Mat {
            weights: self.push(n * cols, || gather(packed, cols))?,
            scales: self.push(n * groups * 2, || gather(scales, groups * 2))?,
            biases: self.push(n * groups * 2, || gather(biases, groups * 2))?,
            rows: n,
            cols,
        })
    }
}

/// Head-dimension order that turns interleaved rope into the NeoX-subdim
/// kernel's pairing: evens, then odds, of the first `rotary`, then the rest.
fn rope_order(head_dim: usize, rotary: usize) -> Vec<usize> {
    let half = rotary / 2;
    (0..head_dim)
        .map(|j| match j {
            j if j < half => 2 * j,
            j if j < rotary => 2 * (j - half) + 1,
            j => j,
        })
        .collect()
}

/// The tower on Metal (see the module doc).
pub struct MetalTower {
    cfg: TowerConfig,
    embeddings: TowerEmbeddings,
    context: gpu::MetalContext,
    weights: gpu::MetalBuffer,
    layers: Vec<LayerGpu>,
    heads: Vec<Mat>,
    semantic_head: Mat,
    k_cache: Vec<gpu::MetalBuffer>,
    v_cache: Vec<gpu::MetalBuffer>,
    capacity: usize,
    position: usize,
    attn_scratch: gpu::AttentionScratch,
    // f16 scratch.
    h: [gpu::MetalBuffer; 2],
    cat: gpu::MetalBuffer,
    x: gpu::MetalBuffer,
    normed: gpu::MetalBuffer,
    qg: gpu::MetalBuffer,
    attn: gpu::MetalBuffer,
    delta: gpu::MetalBuffer,
    gate: gpu::MetalBuffer,
    up: gpu::MetalBuffer,
    act: gpu::MetalBuffer,
    logits: gpu::MetalBuffer,
    /// Layer inputs: one `dim` row per (row, layer) of a pass.
    e_rows: gpu::MetalBuffer,
    /// Backbone hidden states: one `dim` row per row of a pass.
    hidden_rows: gpu::MetalBuffer,
}

impl MetalTower {
    /// Loads the tower and semantic head from the side file onto a new Metal
    /// context, with KV room for `capacity` rows per block.
    pub fn load(
        file: &SafetensorsFile,
        cfg: TowerConfig,
        semantic: RangeInclusive<i32>,
        capacity: usize,
    ) -> Result<Self> {
        let embeddings = TowerEmbeddings::load(file, &cfg, semantic)?;
        let (dim, hd) = (cfg.dim, cfg.head_dim);
        let (qw, kvw) = (cfg.n_head * hd, cfg.n_local_heads * hd);
        let rotary = (hd as f32 * cfg.rotary_percent) as usize;
        if rotary % 2 != 0 || rotary > hd {
            return Err(SpeechError::BadConfig {
                field: "audio_decoder_config.rotary_percent".into(),
                why: format!("rotary dims {rotary} must be even and fit head_dim {hd}"),
            });
        }
        let order = rope_order(hd, rotary);
        let identity: Vec<usize> = (0..hd.max(dim)).collect();
        let all = |n: usize| -> Vec<usize> { (0..n).collect() };
        // wqkv rows are [all Q | all gates | K | V]; Q and K are permuted per
        // head, gates and V are not.
        let qg_rows: Vec<usize> = (0..cfg.n_head)
            .flat_map(|h| order.iter().map(move |&d| h * hd + d))
            .chain(qw..2 * qw)
            .collect();
        let k_rows: Vec<usize> = (0..cfg.n_local_heads)
            .flat_map(|h| order.iter().map(move |&d| 2 * qw + h * hd + d))
            .collect();
        let v_rows: Vec<usize> = (2 * qw + kvw..2 * qw + 2 * kvw).collect();

        let place = |arena: &mut Arena| -> Result<(Vec<LayerGpu>, Vec<Mat>, Mat)> {
            let mut layers = Vec::new();
            let names = (0..cfg.shared_enrichment_layers)
                .map(|i| format!("tower.shared_layers.{i}"))
                .chain((0..cfg.num_codebooks - 1).map(|i| format!("tower.residual_layers.{i}")));
            for p in names {
                let t = format!("{p}.transformer_layer");
                let inter = cfg.intermediate_size;
                let wqkv = format!("{t}.attention.wqkv");
                let rows = 2 * qw + 2 * kvw;
                layers.push(LayerGpu {
                    enorm: arena.norm(file, &format!("{p}.enorm.weight"), dim, &identity[..dim])?,
                    hnorm: arena.norm(file, &format!("{p}.hnorm.weight"), dim, &identity[..dim])?,
                    eh_proj: arena.matrix(
                        file,
                        &format!("{p}.eh_proj"),
                        dim,
                        2 * dim,
                        &all(dim),
                    )?,
                    attention_norm: arena.norm(
                        file,
                        &format!("{t}.attention_norm.weight"),
                        dim,
                        &identity[..dim],
                    )?,
                    qg: arena.matrix(file, &wqkv, rows, dim, &qg_rows)?,
                    k: arena.matrix(file, &wqkv, rows, dim, &k_rows)?,
                    v: arena.matrix(file, &wqkv, rows, dim, &v_rows)?,
                    q_norm: arena.norm(
                        file,
                        &format!("{t}.attention.q_norm.weight"),
                        hd,
                        &order,
                    )?,
                    k_norm: arena.norm(
                        file,
                        &format!("{t}.attention.k_norm.weight"),
                        hd,
                        &order,
                    )?,
                    wo: arena.matrix(file, &format!("{t}.attention.wo"), dim, qw, &all(dim))?,
                    ffn_norm: arena.norm(
                        file,
                        &format!("{t}.ffn_norm.weight"),
                        dim,
                        &identity[..dim],
                    )?,
                    w1: arena.matrix(
                        file,
                        &format!("{t}.feed_forward.w1"),
                        inter,
                        dim,
                        &all(inter),
                    )?,
                    w3: arena.matrix(
                        file,
                        &format!("{t}.feed_forward.w3"),
                        inter,
                        dim,
                        &all(inter),
                    )?,
                    w2: arena.matrix(
                        file,
                        &format!("{t}.feed_forward.w2"),
                        dim,
                        inter,
                        &all(dim),
                    )?,
                    final_norm: arena.norm(
                        file,
                        &format!("{p}.final_layernorm.weight"),
                        dim,
                        &identity[..dim],
                    )?,
                });
            }
            let vocab = cfg.vocab_size;
            let heads = (0..cfg.num_codebooks - 1)
                .map(|i| {
                    arena.matrix(
                        file,
                        &format!("tower.codebook_heads.{i}"),
                        vocab,
                        dim,
                        &all(vocab),
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            let semantic_head =
                arena.matrix(file, "semantic_head", vocab + 1, dim, &all(vocab + 1))?;
            Ok((layers, heads, semantic_head))
        };

        let mut measure = Arena {
            len: 0,
            target: None,
        };
        place(&mut measure)?;
        let context = gpu::MetalContext::new().map_err(gpu_err)?;
        let weights = context.new_output_buffer(measure.len as u64);
        let mut fill = Arena {
            len: 0,
            target: Some(weights.clone()),
        };
        let (layers, heads, semantic_head) = place(&mut fill)?;
        assert_eq!(fill.len, measure.len, "arena walks disagree");
        let vocab = cfg.vocab_size;
        let f16_buf = |n: usize| context.new_output_buffer((n * 2) as u64);
        let kv_bytes = (capacity * kvw * 2) as u64;
        let blocks = layers.len();
        let tower = MetalTower {
            k_cache: (0..blocks)
                .map(|_| context.new_output_buffer(kv_bytes))
                .collect(),
            v_cache: (0..blocks)
                .map(|_| context.new_output_buffer(kv_bytes))
                .collect(),
            attn_scratch: gpu::AttentionScratch::new(&context, cfg.n_head as u32, hd as u32),
            h: [f16_buf(dim), f16_buf(dim)],
            cat: f16_buf(2 * dim),
            x: f16_buf(dim),
            normed: f16_buf(dim),
            qg: f16_buf(2 * qw),
            attn: f16_buf(qw),
            delta: f16_buf(dim),
            gate: f16_buf(cfg.intermediate_size),
            up: f16_buf(cfg.intermediate_size),
            act: f16_buf(cfg.intermediate_size),
            logits: f16_buf(vocab + 1),
            e_rows: f16_buf(PREFILL_ROWS_PER_PASS * blocks * dim),
            hidden_rows: f16_buf(PREFILL_ROWS_PER_PASS * dim),
            weights,
            layers,
            heads,
            semantic_head,
            capacity,
            position: 0,
            context,
            embeddings,
            cfg,
        };
        Ok(tower)
    }

    fn resident(buffer: &gpu::MetalBuffer, m: Mat) -> gpu::Int8ResidentMatrix<'_> {
        gpu::Int8ResidentMatrix {
            buffer,
            weights_offset: m.weights,
            scales_offset: m.scales,
            biases_offset: m.biases,
            rows: m.rows,
            cols: m.cols,
        }
    }

    fn write_f16(
        buffer: &gpu::MetalBuffer,
        row: usize,
        width: usize,
        values: &[f32],
    ) -> Result<()> {
        if values
            .iter()
            .any(|v| !v.is_finite() || v.abs() > f16::MAX.to_f32())
        {
            return Err(SpeechError::Input {
                why: "tower input outside the f16 range".into(),
            });
        }
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|&v| f16::from_f32(v).to_le_bytes())
            .collect();
        gpu::write_buffer_bytes(buffer, row * width * 2, &bytes);
        Ok(())
    }

    fn read_logits(&self, n: usize) -> Vec<f32> {
        let mut out = vec![f16::ZERO; n];
        gpu::read_buffer_f16_into(&self.logits, 0, &mut out);
        out.iter().map(|v| v.to_f32()).collect()
    }

    /// Encodes one tower layer for one row at `position`: input embedding
    /// `e`, hidden in `h_in`, normed output to `h_out`.
    #[allow(clippy::too_many_arguments)]
    fn encode_layer(
        &mut self,
        pass: &gpu::PassEncoder,
        index: usize,
        e: (&gpu::MetalBuffer, u64),
        h_in: (&gpu::MetalBuffer, u64),
        h_out: (&gpu::MetalBuffer, u64),
        position: usize,
    ) -> std::result::Result<(), gpu::GpuError> {
        let cfg = &self.cfg;
        let (dim, hd, nh, nkv) = (cfg.dim, cfg.head_dim, cfg.n_head, cfg.n_local_heads);
        let (qw, kvw, inter) = (nh * hd, nkv * hd, cfg.intermediate_size);
        let eps = cfg.norm_eps;
        let rotary = (hd as f32 * cfg.rotary_percent) as u32;
        let theta = cfg.rope_base;
        let l = &self.layers[index];
        let w = &self.weights;
        let ctx = &mut self.context;
        let row = (position * kvw * 2) as u64;
        let (k_row, v_row) = ((&self.k_cache[index], row), (&self.v_cache[index], row));

        gpu::encode_rms_norm_bf16w_centered(
            ctx,
            pass,
            e,
            (w, l.enorm),
            (&self.cat, 0),
            dim as u32,
            eps,
        )?;
        gpu::encode_rms_norm_bf16w_centered(
            ctx,
            pass,
            h_in,
            (w, l.hnorm),
            (&self.cat, (dim * 2) as u64),
            dim as u32,
            eps,
        )?;
        let m = |m: Mat| Self::resident(w, m);
        gpu::encode_dequant_int8_gemv_resident(
            ctx,
            pass,
            &m(l.eh_proj),
            (&self.cat, 0),
            (&self.x, 0),
        )?;

        // Attention.
        gpu::encode_rms_norm_bf16w_centered(
            ctx,
            pass,
            (&self.x, 0),
            (w, l.attention_norm),
            (&self.normed, 0),
            dim as u32,
            eps,
        )?;
        gpu::encode_dequant_int8_gemv_resident(
            ctx,
            pass,
            &m(l.qg),
            (&self.normed, 0),
            (&self.qg, 0),
        )?;
        gpu::encode_dequant_int8_gemv_resident(ctx, pass, &m(l.k), (&self.normed, 0), k_row)?;
        gpu::encode_dequant_int8_gemv_resident(ctx, pass, &m(l.v), (&self.normed, 0), v_row)?;
        gpu::encode_rms_norm_bf16w_perhead_centered(
            ctx,
            pass,
            (&self.qg, 0),
            (w, l.q_norm),
            (&self.qg, 0),
            nh as u32,
            hd as u32,
            eps,
        )?;
        gpu::encode_rms_norm_bf16w_perhead_centered(
            ctx,
            pass,
            k_row,
            (w, l.k_norm),
            k_row,
            nkv as u32,
            hd as u32,
            eps,
        )?;
        gpu::encode_rope_neox_subdim(
            ctx,
            pass,
            (&self.qg, 0),
            position as u32,
            nh as u32,
            hd as u32,
            rotary,
            theta,
        )?;
        gpu::encode_rope_neox_subdim(
            ctx,
            pass,
            k_row,
            position as u32,
            nkv as u32,
            hd as u32,
            rotary,
            theta,
        )?;
        gpu::encode_attention_decode(
            ctx,
            pass,
            (&self.qg, 0),
            &self.k_cache[index],
            &self.v_cache[index],
            &self.attn_scratch,
            (&self.attn, 0),
            hd as u32,
            nh as u32,
            nkv as u32,
            (position + 1) as u32,
            0,
            0,
            (hd as f32).powf(-0.5),
            None,
        )?;
        gpu::encode_sigmoid_gate_mul(
            ctx,
            pass,
            (&self.attn, 0),
            (&self.qg, (qw * 2) as u64),
            qw as u32,
        )?;
        gpu::encode_dequant_int8_gemv_resident(
            ctx,
            pass,
            &m(l.wo),
            (&self.attn, 0),
            (&self.delta, 0),
        )?;
        gpu::encode_residual_add(ctx, pass, (&self.x, 0), (&self.delta, 0), dim as u32)?;

        // Feed-forward.
        gpu::encode_rms_norm_bf16w_centered(
            ctx,
            pass,
            (&self.x, 0),
            (w, l.ffn_norm),
            (&self.normed, 0),
            dim as u32,
            eps,
        )?;
        gpu::encode_dequant_int8_gemv_resident(
            ctx,
            pass,
            &m(l.w1),
            (&self.normed, 0),
            (&self.gate, 0),
        )?;
        gpu::encode_dequant_int8_gemv_resident(
            ctx,
            pass,
            &m(l.w3),
            (&self.normed, 0),
            (&self.up, 0),
        )?;
        gpu::encode_silu_mul(
            ctx,
            pass,
            (&self.gate, 0),
            (&self.up, 0),
            (&self.act, 0),
            inter as u32,
        )?;
        gpu::encode_dequant_int8_gemv_resident(
            ctx,
            pass,
            &m(l.w2),
            (&self.act, 0),
            (&self.delta, 0),
        )?;
        gpu::encode_residual_add(ctx, pass, (&self.x, 0), (&self.delta, 0), dim as u32)?;

        gpu::encode_rms_norm_bf16w_centered(
            ctx,
            pass,
            (&self.x, 0),
            (w, l.final_norm),
            h_out,
            dim as u32,
            eps,
        )
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

    fn check_room(&self, rows: usize) -> Result<()> {
        if self.position + rows > self.capacity {
            return Err(SpeechError::Input {
                why: format!(
                    "the tower is full: {} of {} rows used, {rows} more asked",
                    self.position, self.capacity
                ),
            });
        }
        Ok(())
    }
}

impl TowerModel for MetalTower {
    fn config(&self) -> &TowerConfig {
        &self.cfg
    }

    fn embeddings(&self) -> &TowerEmbeddings {
        &self.embeddings
    }

    fn position(&self) -> usize {
        self.position
    }

    /// Rows past `position` stay in the cache and are rewritten by the next
    /// rows; attention never reads past the current position.
    fn truncate(&mut self, position: usize) -> Result<()> {
        if position > self.position {
            return Err(SpeechError::Input {
                why: format!("cannot truncate {} tower rows to {position}", self.position),
            });
        }
        self.position = position;
        Ok(())
    }

    fn semantic_logits(&mut self, hidden: &[f32]) -> Result<Vec<f32>> {
        if hidden.len() != self.cfg.dim {
            return Err(SpeechError::Input {
                why: format!(
                    "semantic head needs {} values, got {}",
                    self.cfg.dim,
                    hidden.len()
                ),
            });
        }
        Self::write_f16(&self.hidden_rows, 0, self.cfg.dim, hidden)?;
        let rows = self.semantic_head.rows;
        gpu::autorelease_pool(|| -> Result<()> {
            let pass = self.context.begin_pass_labeled("drama3 semantic head");
            let matrix = Self::resident(&self.weights, self.semantic_head);
            gpu::encode_dequant_int8_gemv_resident(
                &mut self.context,
                &pass,
                &matrix,
                (&self.hidden_rows, 0),
                (&self.logits, 0),
            )
            .map_err(gpu_err)?;
            pass.commit_and_wait();
            Ok(())
        })?;
        Ok(self.read_logits(rows))
    }

    fn prefill(&mut self, hidden: &[f32], frames: &[&[i32]], labels: &[&[i32]]) -> Result<()> {
        let dim = self.cfg.dim;
        if hidden.len() != frames.len() * dim || frames.len() != labels.len() {
            return Err(SpeechError::Input {
                why: "prefill needs one hidden row and one label per frame".into(),
            });
        }
        self.check_room(frames.len())?;
        let blocks = self.layers.len();
        for start in (0..frames.len()).step_by(PREFILL_ROWS_PER_PASS) {
            let end = (start + PREFILL_ROWS_PER_PASS).min(frames.len());
            for r in start..end {
                let h = &hidden[r * dim..][..dim];
                self.check(h, frames[r])?;
                self.check(h, labels[r])?;
                let semantic = self.embeddings.is_semantic(labels[r][0]);
                let inputs =
                    self.embeddings
                        .row_inputs(frames[r], labels[r][1], semantic, Some(labels[r]));
                for (layer, e) in inputs.iter().enumerate() {
                    Self::write_f16(&self.e_rows, (r - start) * blocks + layer, dim, e)?;
                }
                Self::write_f16(&self.hidden_rows, r - start, dim, h)?;
            }
            gpu::autorelease_pool(|| -> Result<()> {
                let pass = self.context.begin_pass_labeled("drama3 tower prefill");
                for r in start..end {
                    let position = self.position + (r - start);
                    let row = (r - start) as u64;
                    // Layer 0 reads the backbone's row; each later layer the
                    // previous one's output, ping-ponged.
                    let hidden_rows = self.hidden_rows.clone();
                    let e_rows = self.e_rows.clone();
                    let h = self.h.clone();
                    for layer in 0..blocks {
                        let e = (&e_rows, ((row as usize * blocks + layer) * dim * 2) as u64);
                        let h_in = if layer == 0 {
                            (&hidden_rows, row * (dim * 2) as u64)
                        } else {
                            (&h[(layer + 1) % 2], 0)
                        };
                        let h_out = (&h[layer % 2], 0);
                        self.encode_layer(&pass, layer, e, h_in, h_out, position)
                            .map_err(gpu_err)?;
                    }
                }
                pass.commit_and_wait();
                Ok(())
            })?;
            self.position += end - start;
        }
        Ok(())
    }

    fn residual_codes(
        &mut self,
        hidden: &[f32],
        main: i32,
        frame: &[i32],
        sample: &mut dyn FnMut(usize, &[f32]) -> i32,
    ) -> Result<Vec<i32>> {
        self.check(hidden, frame)?;
        self.check_room(1)?;
        let dim = self.cfg.dim;
        let shared = self.cfg.shared_enrichment_layers;
        let vocab = self.cfg.vocab_size;
        let c0 = self.embeddings.c0(main);
        let semantic = self.embeddings.is_semantic(main);
        let position = self.position;
        let inputs = self.embeddings.row_inputs(frame, c0, semantic, None);
        for (layer, e) in inputs.iter().enumerate() {
            Self::write_f16(&self.e_rows, layer, dim, e)?;
        }
        Self::write_f16(&self.hidden_rows, 0, dim, hidden)?;

        let mut codes = vec![c0];
        let residual = self.cfg.num_codebooks - 1;
        let hidden_rows = self.hidden_rows.clone();
        let e_rows = self.e_rows.clone();
        let h = self.h.clone();
        let logits = self.logits.clone();
        let mut layer = 0usize;
        for head in 0..residual {
            if head > 0 {
                let e = self
                    .embeddings
                    .layer_input(shared + head, codes[head], semantic, None);
                Self::write_f16(&e_rows, shared + head, dim, &e)?;
            }
            // Pass 0 runs the enrichment layers and residual layer 0; every
            // later pass one residual layer. Each ends in its head.
            let last = shared + head;
            gpu::autorelease_pool(|| -> Result<()> {
                let pass = self.context.begin_pass_labeled("drama3 tower step");
                while layer <= last {
                    let e = (&e_rows, (layer * dim * 2) as u64);
                    let h_in = if layer == 0 {
                        (&hidden_rows, 0)
                    } else {
                        (&h[(layer + 1) % 2], 0)
                    };
                    self.encode_layer(&pass, layer, e, h_in, (&h[layer % 2], 0), position)
                        .map_err(gpu_err)?;
                    layer += 1;
                }
                let matrix = Self::resident(&self.weights, self.heads[head]);
                gpu::encode_dequant_int8_gemv_resident(
                    &mut self.context,
                    &pass,
                    &matrix,
                    (&h[last % 2], 0),
                    (&logits, 0),
                )
                .map_err(gpu_err)?;
                pass.commit_and_wait();
                Ok(())
            })?;
            let head_logits = self.read_logits(vocab);
            codes.push(sample(head, &head_logits));
        }
        self.position += 1;
        Ok(codes)
    }
}
