//! The Inkling 2 DSpark block drafter on Metal. PORT-LOCAL.
//!
//! The bundled `mtp.safetensors` (`mlx-community/Inkling-2-Mini-oQ4e-mtp`,
//! BF16): 3 layers, block 8 (the anchor and 7 mask rows), context K/V
//! projected from four target taps, the target's own embedding and LM head,
//! and an RNN Markov head that conditions each proposal on the previous one.
//! The reference is the CPU drafter in `examples/inkling2_dspark_accept.rs`
//! (ollama `mlxrunner/model/inkling2/draft.go`, oMLX `dspark.py`), with
//! ollama's window rule: local layers (0 and 2) see the last `window`
//! context positions, layer 1 all of them, and every layer sees the whole
//! block bidirectionally.
//!
//! One draft is one command buffer: the context rows the previous verify
//! committed, the block forward, the head over the 7 mask rows, and the 7
//! sequential Markov steps (each step's argmax feeds the next step's input on
//! the GPU). Only the 7 proposals are read back.
//!
//! The drafter only proposes; the target's verify decides every token, so a
//! drafter bug costs acceptance, never output.
//!
//! Where it lives: `<install>/dspark/{mtp.safetensors,config.json}` (the
//! drafter is not part of the target's checkpoint), or the directory named
//! by `TURBOSPARK_INKLING2_DSPARK`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use super::state::{RealInkling2State, TAG_BF16};
use super::verify::VERIFY_MAX_ROWS;
use super::DSPARK_TAP_ROWS;
use crate::real_forward::{RealForwardError, RealForwardRunner};
use crate::real_forward_utils::entry;

/// The drafter directory for an install, if one is present.
pub fn inkling2_dspark_dir(model_dir: &Path) -> Option<PathBuf> {
    let dir = std::env::var_os("TURBOSPARK_INKLING2_DSPARK")
        .map(PathBuf::from)
        .unwrap_or_else(|| model_dir.join("dspark"));
    (dir.join("mtp.safetensors").is_file() && dir.join("config.json").is_file()).then_some(dir)
}

#[derive(Clone, Debug)]
pub(crate) struct DsparkConfig {
    h: usize,
    layers: usize,
    heads: usize,
    kvh: usize,
    hd: usize,
    ffn: usize,
    eps: f32,
    theta: f32,
    rank: usize,
    /// Block rows: the anchor plus `block - 1` proposals.
    pub(crate) block: usize,
    window: usize,
    local: Vec<bool>,
    taps: usize,
}

impl DsparkConfig {
    fn load(dir: &Path, arch: &model_io::ArchConfig) -> Result<Self, RealForwardError> {
        let bad = |m: String| RealForwardError::Unsupported(format!("DSpark drafter: {m}"));
        let text = std::fs::read_to_string(dir.join("config.json"))
            .map_err(|e| bad(format!("read config.json: {e}")))?;
        let v: serde_json::Value =
            serde_json::from_str(&text).map_err(|e| bad(format!("config.json: {e}")))?;
        let m = &v["mtp_config"];
        let u = |k: &str| {
            m[k].as_u64()
                .map(|x| x as usize)
                .ok_or_else(|| bad(format!("mtp_config.{k} missing")))
        };
        let f = |k: &str| {
            m[k].as_f64()
                .ok_or_else(|| bad(format!("mtp_config.{k} missing")))
        };
        let layers = u("num_hidden_layers")?;
        let local_ids: Vec<usize> = m["local_layer_ids"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|x| x.as_u64().map(|x| x as usize))
                    .collect()
            })
            .unwrap_or_default();
        let taps: Vec<u64> = m["target_layer_ids"]
            .as_array()
            .map(|a| a.iter().filter_map(|x| x.as_u64()).collect())
            .unwrap_or_default();
        // The flow records the stream entering applications 5, 23 and 41 and
        // the final-norm output (`DSPARK_TAP_VISITS`); a drafter trained on
        // other taps would draft from the wrong features.
        if taps != [5, 23, 41, 48] {
            return Err(bad(format!(
                "target_layer_ids {taps:?}; the Inkling 2 flow records [5, 23, 41, 48]"
            )));
        }
        if m["markov_head_type"].as_str() != Some("rnn_hidden") {
            return Err(bad("markov_head_type is not rnn_hidden".into()));
        }
        let c = Self {
            h: u("hidden_size")?,
            layers,
            heads: u("num_attention_heads")?,
            kvh: u("num_key_value_heads")?,
            hd: u("head_dim")?,
            ffn: u("intermediate_size")?,
            eps: f("rms_norm_eps")? as f32,
            theta: f("rope_theta")? as f32,
            rank: u("markov_rank")?,
            block: u("block_size")?,
            window: u("sliding_window_size")?,
            local: (0..layers).map(|l| local_ids.contains(&l)).collect(),
            taps: taps.len(),
        };
        if c.h != arch.hidden_size as usize
            || c.block != VERIFY_MAX_ROWS
            || c.hd > 128
            || c.hd % 32 != 0
            || c.heads % c.kvh != 0
            || c.rank as u32 > gpu::DSPARK_MAX_RANK
            || c.taps != DSPARK_TAP_ROWS
        {
            return Err(bad(format!(
                "shape {c:?} is outside what the Metal drafter serves (block {VERIFY_MAX_ROWS}, \
                 the target's hidden size, head_dim <= 128, rank <= {})",
                gpu::DSPARK_MAX_RANK
            )));
        }
        Ok(c)
    }
}

/// Where the context taps for a position are: the last single-token pass or
/// rows of the last verify (cut to the rows the round kept).
#[derive(Clone, Copy, Debug)]
pub(crate) enum TapSource {
    None,
    Single(usize),
    Verify { base: usize, rows: usize },
}

pub(crate) struct DsparkState {
    pub(crate) cfg: DsparkConfig,
    w: gpu::MetalBuffer,
    offs: HashMap<String, u64>,
    /// Softmax of each `fuse_weight` row: layers, then the Markov head.
    alpha: Vec<[f32; 4]>,
    caps: Vec<usize>,
    kc: Vec<gpu::MetalBuffer>,
    vc: Vec<gpu::MetalBuffer>,
    /// FP16 `[rank][vocab]`: the LM head folded through `hidden_out`.
    fold_t: gpu::MetalBuffer,
    /// FP16 `[vocab]`: the LM head applied to `hidden_out.bias`.
    fold_b: gpu::MetalBuffer,
    /// Context rows written: positions `0 .. ctx_rows`.
    pub(crate) ctx_rows: usize,
    pub(crate) source: TapSource,
    // Scratch.
    cmix: gpu::MetalBuffer,
    ck: gpu::MetalBuffer,
    cv: gpu::MetalBuffer,
    emb: gpu::MetalBuffer,
    emb_n: gpu::MetalBuffer,
    x: gpu::MetalBuffer,
    xn: gpu::MetalBuffer,
    q: gpu::MetalBuffer,
    kb: gpu::MetalBuffer,
    vb: gpu::MetalBuffer,
    att: gpu::MetalBuffer,
    o: gpu::MetalBuffer,
    g: gpu::MetalBuffer,
    u: gpu::MetalBuffer,
    act: gpu::MetalBuffer,
    hid: gpu::MetalBuffer,
    base: gpu::MetalBuffer,
    feature: gpu::MetalBuffer,
    mstate: gpu::MetalBuffer,
    input: gpu::MetalBuffer,
    gts: gpu::MetalBuffer,
    latent: gpu::MetalBuffer,
    part_v: gpu::MetalBuffer,
    part_i: gpu::MetalBuffer,
    last: gpu::MetalBuffer,
    proposals: gpu::MetalBuffer,
}

fn softmax4(row: &[f32]) -> [f32; 4] {
    let mx = row.iter().cloned().fold(f32::MIN, f32::max);
    let e: Vec<f32> = row.iter().map(|v| (v - mx).exp()).collect();
    let s: f32 = e.iter().sum();
    [e[0] / s, e[1] / s, e[2] / s, e[3] / s]
}

impl DsparkState {
    /// Loads the drafter and folds the target head through the Markov
    /// projection. `max_ctx` sizes the full-attention layer's context cache.
    pub(crate) fn build(
        context: &mut gpu::MetalContext,
        weights: &gpu::ResidentGpuWeights,
        index: &model_io::ResidentIndex,
        arch: &model_io::ArchConfig,
        dir: &Path,
        max_ctx: usize,
    ) -> Result<Self, RealForwardError> {
        let bad = |m: String| RealForwardError::Unsupported(format!("DSpark drafter: {m}"));
        let cfg = DsparkConfig::load(dir, arch)?;
        let st = model_io::SafetensorsFile::open(&dir.join("mtp.safetensors"))
            .map_err(RealForwardError::Model)?;
        let mut bytes: Vec<u8> = Vec::new();
        let mut offs = HashMap::new();
        let mut fuse = Vec::new();
        for name in st.tensor_names() {
            let d = st.descriptor(name).expect("listed");
            if d.dtype != "BF16" {
                return Err(bad(format!("{name} is {}, expected BF16", d.dtype)));
            }
            let raw = st.raw_bytes(name).map_err(RealForwardError::Model)?;
            while bytes.len() % 256 != 0 {
                bytes.push(0);
            }
            offs.insert(name.to_string(), bytes.len() as u64);
            bytes.extend_from_slice(raw);
            if name == "fuse_weight" {
                fuse = raw
                    .chunks_exact(2)
                    .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                    .collect();
            }
        }
        let (h, kvd, rank) = (cfg.h, cfg.kvh * cfg.hd, cfg.rank);
        let want = |n: &str, elems: usize| -> Result<(), RealForwardError> {
            let d = st
                .descriptor(n)
                .ok_or_else(|| bad(format!("{n} missing")))?;
            if d.shape.iter().product::<usize>() != elems {
                return Err(bad(format!("{n} has shape {:?}", d.shape)));
            }
            Ok(())
        };
        want("fuse_weight", (cfg.layers + 1) * cfg.taps)?;
        want("mask_embed", h)?;
        want("norm.weight", h)?;
        want("markov_head.dflare_norm.weight", h)?;
        want("markov_head.hidden_out.weight", h * rank)?;
        want("markov_head.hidden_out.bias", h)?;
        want(
            "markov_head.markov_w1.weight",
            arch.vocab_size as usize * rank,
        )?;
        want("markov_head.rnn_proj.weight", 3 * rank * (2 * rank + 2 * h))?;
        want("markov_head.rnn_proj.bias", 3 * rank)?;
        for l in 0..cfg.layers {
            let p = |s: &str| format!("layers.{l}.{s}");
            for (s, n) in [
                ("dflare_norm.weight", h),
                ("input_layernorm.weight", h),
                ("post_attention_layernorm.weight", h),
                ("self_attn.q_norm.weight", cfg.hd),
                ("self_attn.k_norm.weight", cfg.hd),
                ("self_attn.k_norm_ctx.weight", cfg.hd),
                ("self_attn.q_proj.weight", cfg.heads * cfg.hd * h),
                ("self_attn.k_proj.weight", kvd * h),
                ("self_attn.v_proj.weight", kvd * h),
                ("self_attn.k_proj_ctx.weight", kvd * h),
                ("self_attn.v_proj_ctx.weight", kvd * h),
                ("self_attn.o_proj.weight", h * cfg.heads * cfg.hd),
                ("mlp.gate_proj.weight", cfg.ffn * h),
                ("mlp.up_proj.weight", cfg.ffn * h),
                ("mlp.down_proj.weight", h * cfg.ffn),
            ] {
                want(&p(s), n)?;
            }
        }
        let alpha: Vec<[f32; 4]> = fuse.chunks_exact(4).map(softmax4).collect();
        let w = context.new_buffer_with_data(&bytes);
        drop(bytes);

        let vocab = arch.vocab_size as usize;
        let half = |c: &gpu::MetalContext, n: usize| c.new_output_buffer((n.max(1) * 2) as u64);
        let f32b = |c: &gpu::MetalContext, n: usize| c.new_output_buffer((n.max(1) * 4) as u64);
        let caps: Vec<usize> = cfg
            .local
            .iter()
            .map(|&l| if l { max_ctx.min(cfg.window) } else { max_ctx })
            .collect();
        let kc = caps.iter().map(|&c| half(context, c * kvd)).collect();
        let vc = caps.iter().map(|&c| half(context, c * kvd)).collect();
        let b = cfg.block;
        let groups = (arch.vocab_size as u32).div_ceil(gpu::DSPARK_LOGITS_PER_GROUP) as usize;
        let mut s = Self {
            fold_t: half(context, rank * vocab),
            fold_b: half(context, vocab),
            cmix: half(context, b * h),
            ck: half(context, b * kvd),
            cv: half(context, b * kvd),
            emb: half(context, h),
            emb_n: half(context, h),
            x: f32b(context, b * h),
            xn: half(context, b * h),
            q: half(context, b * cfg.heads * cfg.hd),
            kb: half(context, b * kvd),
            vb: half(context, b * kvd),
            att: half(context, b * cfg.heads * cfg.hd),
            o: half(context, b * h),
            g: half(context, b * cfg.ffn),
            u: half(context, b * cfg.ffn),
            act: half(context, b * cfg.ffn),
            hid: half(context, b * h),
            base: half(context, (b - 1) * vocab),
            feature: half(context, h),
            mstate: f32b(context, rank),
            input: half(context, 2 * rank + 2 * h),
            gts: half(context, 3 * rank),
            latent: half(context, rank),
            part_v: f32b(context, groups),
            part_i: f32b(context, groups),
            last: f32b(context, 1),
            proposals: f32b(context, b),
            cfg,
            w,
            offs,
            alpha,
            caps,
            kc,
            vc,
            ctx_rows: 0,
            source: TapSource::None,
        };
        s.fold(context, weights, index, arch)?;
        Ok(s)
    }

    fn off(&self, name: &str) -> u64 {
        self.offs[name]
    }

    fn wv(&self, name: &str) -> (&gpu::MetalBuffer, u64) {
        (&self.w, self.off(name))
    }

    /// `fold_t[j] = head @ hidden_out[:, j]` and `fold_b = head @ hidden_out.bias`,
    /// once at open, on the GPU.
    fn fold(
        &mut self,
        context: &mut gpu::MetalContext,
        weights: &gpu::ResidentGpuWeights,
        index: &model_io::ResidentIndex,
        arch: &model_io::ArchConfig,
    ) -> Result<(), RealForwardError> {
        let g = RealForwardError::Gpu;
        let (h, rank, vocab) = (self.cfg.h, self.cfg.rank, arch.vocab_size as usize);
        let hot = context.new_output_buffer((rank * h * 2) as u64);
        let hb = context.new_output_buffer((h * 2) as u64);
        let pass = context.begin_pass_labeled("dspark fold");
        gpu::encode_dspark_transpose_bf16(
            context,
            &pass,
            self.wv("markov_head.hidden_out.weight"),
            (&hot, 0),
            h as u32,
            rank as u32,
        )
        .map_err(g)?;
        gpu::encode_dspark_bf16_to_half(
            context,
            &pass,
            self.wv("markov_head.hidden_out.bias"),
            (&hb, 0),
            h as u32,
        )
        .map_err(g)?;
        pass.commit_and_wait();
        let head = "model.llm.unembed.weight";
        let pass = context.begin_pass_labeled("dspark fold head");
        let mut j = 0;
        while j < rank {
            let n = (rank - j).min(VERIFY_MAX_ROWS);
            head_rows(
                context,
                &pass,
                weights,
                index,
                head,
                vocab,
                h,
                (&hot, (j * h * 2) as u64, h as u32),
                (&self.fold_t, (j * vocab * 2) as u64, vocab as u32),
                n,
            )?;
            j += n;
        }
        head_rows(
            context,
            &pass,
            weights,
            index,
            head,
            vocab,
            h,
            (&hb, 0, h as u32),
            (&self.fold_b, 0, vocab as u32),
            1,
        )?;
        pass.commit_and_wait();
        Ok(())
    }

    pub(crate) fn reset(&mut self) {
        self.ctx_rows = 0;
        self.source = TapSource::None;
    }
}

/// The target LM head over `n` FP16 rows: one multi-row dispatch on a BF16
/// head, the shared dispatcher per row on an MLX affine one.
#[allow(clippy::too_many_arguments)]
fn head_rows(
    context: &mut gpu::MetalContext,
    pass: &gpu::PassEncoder,
    weights: &gpu::ResidentGpuWeights,
    index: &model_io::ResidentIndex,
    name: &str,
    rows: usize,
    cols: usize,
    x: (&gpu::MetalBuffer, u64, u32),
    y: (&gpu::MetalBuffer, u64, u32),
    n: usize,
) -> Result<(), RealForwardError> {
    let e = entry(index, name)?;
    if e.dtype == TAG_BF16 {
        let o = weights.gpu_offset(e.file_offset - index.header.index_size);
        return gpu::encode_inkling2_bf16_gemv_rows(
            context,
            pass,
            weights.buffer(),
            &[((x.0, x.1), x.2)],
            &[gpu::Inkling2GemvRowsSegment {
                w_offset: o,
                rows: rows as u32,
                cols: cols as u32,
                x: 0,
                y: (y.0, y.1),
                y_stride: y.2,
            }],
            n as u32,
        )
        .map_err(RealForwardError::Gpu);
    }
    for m in 0..n as u64 {
        crate::real_forward_dispatch::encode_gemv_any(
            context,
            pass,
            weights,
            index,
            name,
            rows,
            cols,
            (x.0, x.1 + m * u64::from(x.2) * 2),
            (y.0, y.1 + m * u64::from(y.2) * 2),
        )?;
    }
    Ok(())
}

/// Drafter-weight GEMVs (all BF16, in the drafter's own buffer) over `n` rows.
fn dgemv(
    context: &mut gpu::MetalContext,
    pass: &gpu::PassEncoder,
    d: &DsparkState,
    x: (&gpu::MetalBuffer, u32),
    segs: &[(&str, usize, usize, &gpu::MetalBuffer, u32)],
    n: usize,
) -> Result<(), RealForwardError> {
    let v: Vec<gpu::Inkling2GemvRowsSegment<'_>> = segs
        .iter()
        .map(|&(name, rows, cols, y, ys)| gpu::Inkling2GemvRowsSegment {
            w_offset: d.off(name),
            rows: rows as u32,
            cols: cols as u32,
            x: 0,
            y: (y, 0),
            y_stride: ys,
        })
        .collect();
    gpu::encode_inkling2_bf16_gemv_rows(context, pass, &d.w, &[((x.0, 0), x.1)], &v, n as u32)
        .map_err(RealForwardError::Gpu)
}

impl RealForwardRunner {
    /// Opens the DSpark drafter for rounds of up to `block` proposals, at open
    /// (`DflashDraftPolicy::Fixed`). Refuses by name what it cannot serve.
    pub(crate) fn inkling2_open_dspark(&mut self, block: usize) -> Result<(), RealForwardError> {
        let unsupported = RealForwardError::Unsupported;
        let dir = inkling2_dspark_dir(&self.install_dir).ok_or_else(|| {
            unsupported(format!(
                "the DSpark drafter was asked for, but {} has no dspark/mtp.safetensors and \
                 config.json, and TURBOSPARK_INKLING2_DSPARK names no such directory",
                self.install_dir.display()
            ))
        })?;
        let st = self
            .real_inkling2
            .as_ref()
            .expect("inkling state built first");
        if st.resident.is_none() {
            return Err(unsupported(
                "the DSpark drafter needs resident expert residency: its verify is the batched \
                 Inkling forward, which has no streamed form (--expert-residency resident)"
                    .to_string(),
            ));
        }
        st.native_expert_types().map_err(unsupported)?;
        let max_ctx = st.max_ctx;
        let d = DsparkState::build(
            &mut self.context,
            &self.weights,
            &self.index,
            &self.arch,
            &dir,
            max_ctx,
        )?;
        if block == 0 || block > d.cfg.block - 1 {
            return Err(unsupported(format!(
                "the DSpark drafter proposes {} tokens per round; {block} was asked for",
                d.cfg.block - 1
            )));
        }
        let h = self.arch.hidden_size as usize;
        let st = self.real_inkling2.as_mut().expect("checked");
        if st.taps.is_none() {
            st.taps = Some(
                self.context
                    .new_buffer_with_data(&vec![0u8; DSPARK_TAP_ROWS * h * 2]),
            );
        }
        st.dspark = Some(d);
        Ok(())
    }

    /// Proposals per round of the open drafter (`block - 1`), or 0.
    pub(crate) fn inkling2_dspark_block(&self) -> usize {
        self.real_inkling2
            .as_ref()
            .and_then(|s| s.dspark.as_ref())
            .map_or(0, |d| d.cfg.block - 1)
    }

    /// `rewind_drafter`: the last verify's taps are valid through `position`.
    pub(crate) fn inkling2_dspark_rewind(&mut self, position: usize) -> Result<(), String> {
        let d = self
            .real_inkling2
            .as_mut()
            .and_then(|s| s.dspark.as_mut())
            .ok_or("no DSpark drafter is open")?;
        if let TapSource::Verify { base, rows } = d.source {
            if position + 1 < base {
                return Err(format!(
                    "DSpark rewind to {position} is before the verify at {base}"
                ));
            }
            d.source = TapSource::Verify {
                base,
                rows: rows.min(position + 1 - base),
            };
        }
        d.ctx_rows = d.ctx_rows.min(position + 1);
        Ok(())
    }

    /// Why Inkling 2 speculation cannot run on this runner, or `None`.
    pub(crate) fn inkling2_speculation_blocker(&self) -> Option<String> {
        let st = self.real_inkling2.as_ref()?;
        if st.dspark.is_none() {
            return Some(format!(
                "inkling2_loop speculates with its DSpark block drafter, which was not opened: \
                 put the drafter at {}/dspark (mtp.safetensors and config.json) or name it with \
                 TURBOSPARK_INKLING2_DSPARK, and select the dflash drafter with a block of 1 to 7",
                self.install_dir.display()
            ));
        }
        if self.prompt_vision.is_some() || self.arch.vision.is_active() {
            return Some(
                "the Inkling 2 verify embeds every row from the token table; image prompts \
                 are not served"
                    .to_string(),
            );
        }
        None
    }

    /// Whether the Inkling 2 DSpark drafter is open.
    pub(crate) fn inkling2_dspark_open(&self) -> bool {
        self.real_inkling2
            .as_ref()
            .is_some_and(|s| s.dspark.is_some())
    }

    /// The taps buffer, byte offset and position for `pos`, from wherever the
    /// last target pass left them.
    fn dspark_tap_row(
        st: &RealInkling2State,
        pos: usize,
        h: usize,
    ) -> Option<(gpu::MetalBuffer, u64)> {
        let d = st.dspark.as_ref()?;
        let row = (DSPARK_TAP_ROWS * h * 2) as u64;
        match d.source {
            TapSource::Single(p) if p == pos => st.taps.clone().map(|t| (t, 0)),
            TapSource::Verify { base, rows } if (base..base + rows).contains(&pos) => st
                .batch
                .as_ref()
                .map(|b| (b.taps.clone(), (pos - base) as u64 * row)),
            _ => None,
        }
    }

    /// Writes context rows `from .. to` (each from its own tap row) for every
    /// drafter layer into `pass`.
    fn dspark_write_context(
        &mut self,
        pass: &gpu::PassEncoder,
        from: usize,
        to: usize,
    ) -> Result<(), RealForwardError> {
        let g = RealForwardError::Gpu;
        let h = self.arch.hidden_size as usize;
        let st = self.real_inkling2.as_ref().expect("checked");
        let d = st.dspark.as_ref().expect("checked");
        let c = d.cfg.clone();
        let kvd = c.kvh * c.hd;
        let ts = (DSPARK_TAP_ROWS * h) as u32;
        let mut p = from;
        while p < to {
            // Rows from one source are contiguous in its tap buffer; take up
            // to a block of them per dispatch group.
            let (buf, off) = Self::dspark_tap_row(st, p, h).ok_or_else(|| {
                RealForwardError::Unsupported(format!(
                    "DSpark context row {p} was never captured by a target pass"
                ))
            })?;
            let mut n = 1;
            while p + n < to
                && n < VERIFY_MAX_ROWS
                && Self::dspark_tap_row(st, p + n, h).is_some_and(|(b2, o2)| {
                    b2.contents() == buf.contents() && o2 == off + (n as u64) * u64::from(ts) * 2
                })
            {
                n += 1;
            }
            for l in 0..c.layers {
                let lp = |s: &str| format!("layers.{l}.{s}");
                gpu::encode_dspark_tap_mix_norm(
                    &mut self.context,
                    pass,
                    (&buf, off),
                    d.wv(&lp("dflare_norm.weight")),
                    (&d.cmix, 0),
                    d.alpha[l],
                    h as u32,
                    ts,
                    c.eps,
                    n as u32,
                )
                .map_err(g)?;
                dgemv(
                    &mut self.context,
                    pass,
                    d,
                    (&d.cmix, h as u32),
                    &[
                        (
                            &lp("self_attn.k_proj_ctx.weight"),
                            kvd,
                            h,
                            &d.ck,
                            kvd as u32,
                        ),
                        (
                            &lp("self_attn.v_proj_ctx.weight"),
                            kvd,
                            h,
                            &d.cv,
                            kvd as u32,
                        ),
                    ],
                    n,
                )?;
                let ck = gpu::Inkling2Rows::plain((&d.ck, 0), kvd as u32);
                gpu::encode_inkling2_rms_rows(
                    &mut self.context,
                    pass,
                    ck,
                    Some(d.wv(&lp("self_attn.k_norm_ctx.weight"))),
                    ck,
                    c.hd as u32,
                    c.kvh as u32,
                    c.eps,
                    n as u32,
                )
                .map_err(g)?;
                gpu::encode_dspark_ctx_store(
                    &mut self.context,
                    pass,
                    (&d.ck, 0),
                    (&d.cv, 0),
                    (&d.kc[l], 0),
                    (&d.vc[l], 0),
                    c.kvh as u32,
                    c.hd as u32,
                    p as u32,
                    d.caps[l] as u32,
                    c.theta,
                    n as u32,
                )
                .map_err(g)?;
            }
            p += n;
        }
        Ok(())
    }

    /// `prime_drafter` for Inkling 2: context row `position` from the taps the
    /// single-token pass at `position` just recorded.
    pub(crate) fn inkling2_dspark_prime(
        &mut self,
        position: usize,
    ) -> Result<(), RealForwardError> {
        let ctx_rows = self
            .real_inkling2
            .as_ref()
            .and_then(|s| s.dspark.as_ref())
            .map(|d| d.ctx_rows)
            .ok_or_else(|| RealForwardError::Unsupported("no DSpark drafter is open".into()))?;
        if position < ctx_rows {
            return Ok(());
        }
        if position > ctx_rows {
            return Err(RealForwardError::Unsupported(format!(
                "DSpark priming at {position} skips context rows {ctx_rows}..{position}"
            )));
        }
        let pass = self.context.begin_pass_labeled("dspark prime");
        self.dspark_write_context(&pass, position, position + 1)?;
        pass.commit_and_wait();
        self.real_inkling2
            .as_mut()
            .and_then(|s| s.dspark.as_mut())
            .expect("checked")
            .ctx_rows = position + 1;
        Ok(())
    }

    /// One block draft anchored at `anchor` (the token at `base`, not yet
    /// fed): `block - 1` proposals for `base + 1 ..`.
    pub(crate) fn inkling2_dspark_draft(
        &mut self,
        anchor: i32,
        base: usize,
    ) -> Result<Vec<i32>, RealForwardError> {
        let g = RealForwardError::Gpu;
        let unsupported = RealForwardError::Unsupported;
        let (h, vocab) = (
            self.arch.hidden_size as usize,
            self.arch.vocab_size as usize,
        );
        let ctx_rows = {
            let st = self
                .real_inkling2
                .as_ref()
                .ok_or_else(|| unsupported("no Inkling state".into()))?;
            let d = st
                .dspark
                .as_ref()
                .ok_or_else(|| unsupported("no DSpark drafter is open".into()))?;
            d.ctx_rows
        };
        if base == 0 || ctx_rows > base {
            return Err(unsupported(format!(
                "DSpark draft at {base} with {ctx_rows} context rows"
            )));
        }
        let unpadded = self
            .real_inkling2
            .as_ref()
            .expect("checked")
            .cfg
            .unpadded_vocab;
        if anchor < 0 || anchor as u32 >= unpadded {
            return Err(unsupported(format!(
                "anchor {anchor} outside the vocabulary"
            )));
        }
        let pass = self.context.begin_pass_labeled("dspark draft");
        self.dspark_write_context(&pass, ctx_rows, base)?;
        let st = self.real_inkling2.as_ref().expect("checked");
        let (fbuf, foff) = Self::dspark_tap_row(st, base - 1, h).ok_or_else(|| {
            unsupported(format!(
                "DSpark feature row {} was never captured",
                base - 1
            ))
        })?;
        let d = st.dspark.as_ref().expect("checked");
        let c = d.cfg.clone();
        let (b, kvd, qd) = (c.block, c.kvh * c.hd, c.heads * c.hd);
        let ctx = &mut self.context;

        // Block input.
        crate::real_forward_dispatch::encode_embed_any(
            ctx,
            &pass,
            &self.weights,
            &self.index,
            "model.llm.embed.weight",
            (&d.emb, 0),
            anchor as u32,
            h as u32,
            vocab,
            1.0,
        )?;
        let en = entry(&self.index, "model.llm.embed_norm.weight")?;
        let en = (
            self.weights.buffer(),
            self.weights
                .gpu_offset(en.file_offset - self.index.header.index_size),
        );
        gpu::encode_inkling2_rms_rows(
            ctx,
            &pass,
            gpu::Inkling2Rows::plain((&d.emb, 0), h as u32),
            Some(en),
            gpu::Inkling2Rows::plain((&d.emb_n, 0), h as u32),
            h as u32,
            1,
            st.cfg.rms_eps,
            1,
        )
        .map_err(g)?;
        gpu::encode_dspark_init_x(
            ctx,
            &pass,
            (&d.emb_n, 0),
            d.wv("mask_embed"),
            (&d.x, 0),
            h as u32,
            b as u32,
        )
        .map_err(g)?;

        for l in 0..c.layers {
            let lp = |s: &str| format!("layers.{l}.{s}");
            gpu::encode_dspark_rms_f32(
                ctx,
                &pass,
                (&d.x, 0),
                d.wv(&lp("input_layernorm.weight")),
                (&d.xn, 0),
                h as u32,
                c.eps,
                b as u32,
            )
            .map_err(g)?;
            dgemv(
                ctx,
                &pass,
                d,
                (&d.xn, h as u32),
                &[
                    (&lp("self_attn.q_proj.weight"), qd, h, &d.q, qd as u32),
                    (&lp("self_attn.k_proj.weight"), kvd, h, &d.kb, kvd as u32),
                    (&lp("self_attn.v_proj.weight"), kvd, h, &d.vb, kvd as u32),
                ],
                b,
            )?;
            for (buf, nh, wn, stride) in [
                (&d.q, c.heads, "self_attn.q_norm.weight", qd),
                (&d.kb, c.kvh, "self_attn.k_norm.weight", kvd),
            ] {
                let r = gpu::Inkling2Rows::plain((buf, 0), stride as u32);
                gpu::encode_inkling2_rms_rows(
                    ctx,
                    &pass,
                    r,
                    Some(d.wv(&lp(wn))),
                    r,
                    c.hd as u32,
                    nh as u32,
                    c.eps,
                    b as u32,
                )
                .map_err(g)?;
                gpu::encode_dspark_rope_rows(
                    ctx,
                    &pass,
                    (buf, 0),
                    nh as u32,
                    c.hd as u32,
                    base as u32,
                    c.theta,
                    b as u32,
                )
                .map_err(g)?;
            }
            let first = if c.local[l] {
                base - base.min(c.window)
            } else {
                0
            };
            gpu::encode_dspark_attention(
                ctx,
                &pass,
                (&d.q, 0),
                (&d.kc[l], 0),
                (&d.vc[l], 0),
                (&d.kb, 0),
                (&d.vb, 0),
                (&d.att, 0),
                first as u32,
                base as u32,
                gpu::DsparkAttnShape {
                    heads: c.heads as u32,
                    kv_heads: c.kvh as u32,
                    head_dim: c.hd as u32,
                    cap: d.caps[l] as u32,
                    n_block: b as u32,
                },
            )
            .map_err(g)?;
            dgemv(
                ctx,
                &pass,
                d,
                (&d.att, qd as u32),
                &[(&lp("self_attn.o_proj.weight"), h, qd, &d.o, h as u32)],
                b,
            )?;
            gpu::encode_dspark_residual(ctx, &pass, (&d.x, 0), (&d.o, 0), (b * h) as u32)
                .map_err(g)?;
            gpu::encode_dspark_rms_f32(
                ctx,
                &pass,
                (&d.x, 0),
                d.wv(&lp("post_attention_layernorm.weight")),
                (&d.xn, 0),
                h as u32,
                c.eps,
                b as u32,
            )
            .map_err(g)?;
            dgemv(
                ctx,
                &pass,
                d,
                (&d.xn, h as u32),
                &[
                    (&lp("mlp.gate_proj.weight"), c.ffn, h, &d.g, c.ffn as u32),
                    (&lp("mlp.up_proj.weight"), c.ffn, h, &d.u, c.ffn as u32),
                ],
                b,
            )?;
            gpu::encode_dspark_silu_mul(
                ctx,
                &pass,
                (&d.g, 0),
                (&d.u, 0),
                (&d.act, 0),
                (b * c.ffn) as u32,
            )
            .map_err(g)?;
            dgemv(
                ctx,
                &pass,
                d,
                (&d.act, c.ffn as u32),
                &[(&lp("mlp.down_proj.weight"), h, c.ffn, &d.o, h as u32)],
                b,
            )?;
            gpu::encode_dspark_residual(ctx, &pass, (&d.x, 0), (&d.o, 0), (b * h) as u32)
                .map_err(g)?;
        }
        gpu::encode_dspark_rms_f32(
            ctx,
            &pass,
            (&d.x, 0),
            d.wv("norm.weight"),
            (&d.hid, 0),
            h as u32,
            c.eps,
            b as u32,
        )
        .map_err(g)?;
        head_rows(
            ctx,
            &pass,
            &self.weights,
            &self.index,
            "model.llm.unembed.weight",
            vocab,
            h,
            (&d.hid, (h * 2) as u64, h as u32),
            (&d.base, 0, vocab as u32),
            b - 1,
        )?;

        // Markov head: the feature is the position before the anchor.
        gpu::encode_dspark_tap_mix_norm(
            ctx,
            &pass,
            (&fbuf, foff),
            d.wv("markov_head.dflare_norm.weight"),
            (&d.feature, 0),
            d.alpha[c.layers],
            h as u32,
            (DSPARK_TAP_ROWS * h) as u32,
            c.eps,
            1,
        )
        .map_err(g)?;
        gpu::write_buffer_bytes(&d.mstate, 0, &vec![0u8; c.rank * 4]);
        gpu::write_buffer_bytes(&d.last, 0, &(anchor as u32).to_le_bytes());
        let in_len = 2 * c.rank + 2 * h;
        for i in 0..b - 1 {
            gpu::encode_dspark_markov_input(
                ctx,
                &pass,
                (&d.mstate, 0),
                d.wv("markov_head.markov_w1.weight"),
                (&d.last, 0),
                (&d.hid, ((i + 1) * h * 2) as u64),
                (&d.feature, 0),
                (&d.input, 0),
                c.rank as u32,
                h as u32,
            )
            .map_err(g)?;
            dgemv(
                ctx,
                &pass,
                d,
                (&d.input, in_len as u32),
                &[(
                    "markov_head.rnn_proj.weight",
                    3 * c.rank,
                    in_len,
                    &d.gts,
                    (3 * c.rank) as u32,
                )],
                1,
            )?;
            gpu::encode_dspark_markov_gate(
                ctx,
                &pass,
                (&d.gts, 0),
                d.wv("markov_head.rnn_proj.bias"),
                (&d.mstate, 0),
                (&d.latent, 0),
                c.rank as u32,
            )
            .map_err(g)?;
            let parts = gpu::encode_dspark_markov_logits(
                ctx,
                &pass,
                (&d.base, (i * vocab * 2) as u64),
                (&d.fold_b, 0),
                (&d.fold_t, 0),
                (&d.latent, 0),
                (&d.part_v, 0),
                (&d.part_i, 0),
                unpadded,
                vocab as u32,
                c.rank as u32,
            )
            .map_err(g)?;
            gpu::encode_dspark_markov_argmax(
                ctx,
                &pass,
                (&d.part_v, 0),
                (&d.part_i, 0),
                (&d.last, 0),
                (&d.proposals, 0),
                parts,
                i as u32,
            )
            .map_err(g)?;
        }
        pass.commit_and_wait();
        // The proposals are u32 token ids; read the words and reinterpret.
        let out: Vec<i32> = gpu::read_f32_buffer_at(&d.proposals, 0, b - 1)
            .into_iter()
            .map(|v| v.to_bits() as i32)
            .collect();
        let d = self
            .real_inkling2
            .as_mut()
            .and_then(|s| s.dspark.as_mut())
            .expect("checked");
        d.ctx_rows = base;
        Ok(out)
    }
}
