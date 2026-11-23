//! Inkling 2 (`inkling2_loop`) Metal decode flow. PORT-LOCAL.
//!
//! The graph is llama.cpp `src/models/inkling2.cpp` + `inkling2-loop.cpp` at
//! `29b52d398` (plus #32704), and the CPU oracle is
//! `compute::inkling2::forward`; `docs/INKLING2_PHASE0.md` lists every
//! convention. Per token: embed + `embed_norm`; layers 0..core_start once;
//! the core three times, each later pass restarting from the prelude output
//! with messages mixed from the previous pass's bank; the final layers once;
//! output depth mix, final norm, unembed, softcap 30.
//!
//! Each layer APPLICATION (48 per token) is: depth-mixed attention inputs,
//! loop controllers (later passes), norms, projections, short convs into the
//! slot's KV, L2 clips, relative-bias + sink attention, output gate, o-proj
//! and its conv, the block-residual step, the MoE input mix, then the F32
//! router GEMV. The host reads the router logits (one GPU-to-CPU sync per
//! application, the same as every family here), selects with
//! `route_sigmoid`, streams the experts for the PHYSICAL layer (so all
//! passes share one slot cache), and the routed pair plus `mlp_sconv` close
//! the layer.
//!
//! Exactness class: KL-bounded against the CPU reference (FP16 activations,
//! FP32 accumulation, FP32 router). Prefill is this decode path per token.

mod dspark;
mod lamq;
mod prefill;
mod state;
mod verify;

pub use dspark::inkling2_dspark_dir;

use half::f16;
use model_io::ResidentIndex;

use crate::real_forward::{RealForwardError, RealForwardRunner};
use crate::real_forward_utils::entry;
use foundation::LogitValue;
pub(crate) use state::RealInkling2State;
use state::{name, MAX_STATES};

type Buf<'a> = (&'a gpu::MetalBuffer, u64);

fn view<'a>(
    weights: &'a gpu::ResidentGpuWeights,
    index: &ResidentIndex,
    name: &str,
) -> Result<Buf<'a>, RealForwardError> {
    let e = entry(index, name)?;
    Ok((
        weights.buffer(),
        weights.gpu_offset(e.file_offset - index.header.index_size),
    ))
}

struct Ctx<'a> {
    context: &'a mut gpu::MetalContext,
    weights: &'a gpu::ResidentGpuWeights,
    index: &'a ResidentIndex,
    h: usize,
    eps: f32,
    /// A LookAtMyQuant install's coded entries (`lamq.rs`).
    lamq: Option<std::sync::Arc<lamq::LamqRuntime>>,
}

impl Ctx<'_> {
    fn gemv(
        &mut self,
        pass: &gpu::PassEncoder,
        name: &str,
        rows: usize,
        cols: usize,
        x: Buf<'_>,
        y: Buf<'_>,
    ) -> Result<(), RealForwardError> {
        self.gemv_group(pass, &[x], &[(name, rows, cols, 0, y)])
    }

    /// Several GEMVs: the BF16 ones in one dispatch
    /// (`gpu::encode_inkling2_bf16_gemv_multi`), each MLX affine one through
    /// the shared resident dispatcher (`encode_gemv_any`, tags 4 and 5). Each
    /// entry is `(weight, rows, cols, input index into xs, output)`.
    fn gemv_group(
        &mut self,
        pass: &gpu::PassEncoder,
        xs: &[Buf<'_>],
        segs: &[(&str, usize, usize, usize, Buf<'_>)],
    ) -> Result<(), RealForwardError> {
        let mut v = Vec::with_capacity(segs.len());
        let lq = self.lamq.clone();
        // The group's LAMQ matrices first, their input rotations batched into
        // one dispatch (the outputs are independent, so order does not
        // matter).
        if let Some(l) = lq.as_ref() {
            let coded: Vec<_> = segs.iter().copied().filter(|s| l.has(s.0)).collect();
            if !coded.is_empty() {
                l.gemv_batch(self.context, pass, self.weights.buffer(), xs, &coded)?;
            }
        }
        for &(name, rows, cols, x, y) in segs {
            if lq.as_ref().is_some_and(|l| l.has(name)) {
                continue;
            }
            if entry(self.index, name)?.dtype != state::TAG_BF16 {
                crate::real_forward_dispatch::encode_gemv_any(
                    self.context,
                    pass,
                    self.weights,
                    self.index,
                    name,
                    rows,
                    cols,
                    xs[x],
                    y,
                )?;
                continue;
            }
            let (_, o) = view(self.weights, self.index, name)?;
            v.push(gpu::Inkling2GemvSegment {
                w_offset: o,
                rows: rows as u32,
                cols: cols as u32,
                x,
                y,
            });
        }
        if v.is_empty() {
            return Ok(());
        }
        gpu::encode_inkling2_bf16_gemv_multi(self.context, pass, self.weights.buffer(), xs, &v)
            .map_err(RealForwardError::Gpu)
    }

    fn norm(
        &mut self,
        pass: &gpu::PassEncoder,
        x: Buf<'_>,
        weight: &str,
        out: Buf<'_>,
    ) -> Result<(), RealForwardError> {
        let w = view(self.weights, self.index, weight)?;
        gpu::encode_rms_norm_bf16w(self.context, pass, x, w, out, self.h as u32, self.eps)
            .map_err(RealForwardError::Gpu)
    }

    /// [`Self::mix`] then one [`Self::norm`] per query, in one dispatch
    /// (`gpu::encode_inkling2_depth_mix_norm`): `cur` is read in place, so
    /// state row `n_blocks` is not written (it is scratch: every reader
    /// copies `cur` there first). Query `j` is normed with `weights[j]` into
    /// row `j` of `out`. Bit-identical to the separate kernels.
    #[allow(clippy::too_many_arguments)]
    fn mix_norm(
        &mut self,
        pass: &gpu::PassEncoder,
        states: &gpu::MetalBuffer,
        n_blocks: usize,
        cur: Buf<'_>,
        query: Buf<'_>,
        weights: &[&str],
        out: Buf<'_>,
    ) -> Result<(), RealForwardError> {
        if n_blocks + 1 > MAX_STATES {
            return Err(RealForwardError::Unsupported(format!(
                "depth mix over {} states exceeds {MAX_STATES}",
                n_blocks + 1
            )));
        }
        let ws = weights
            .iter()
            .map(|w| view(self.weights, self.index, w))
            .collect::<Result<Vec<_>, _>>()?;
        gpu::encode_inkling2_depth_mix_norm(
            self.context,
            pass,
            (states, 0),
            cur,
            query,
            None,
            &ws,
            out,
            (n_blocks + 1) as u32,
            self.h as u32,
            self.eps,
        )
        .map_err(RealForwardError::Gpu)
    }

    fn copy(
        &mut self,
        pass: &gpu::PassEncoder,
        src: Buf<'_>,
        dst: Buf<'_>,
        n: usize,
    ) -> Result<(), RealForwardError> {
        gpu::encode_inkling2_copy_half(self.context, pass, src, dst, n as u32)
            .map_err(RealForwardError::Gpu)
    }

    /// Depth mix of `states[0..n_blocks]` plus `cur` (copied into row
    /// `n_blocks`) with `n_q` F32 query rows.
    #[allow(clippy::too_many_arguments)]
    fn mix(
        &mut self,
        pass: &gpu::PassEncoder,
        states: &gpu::MetalBuffer,
        n_blocks: usize,
        cur: Buf<'_>,
        query: Buf<'_>,
        n_q: usize,
        out: Buf<'_>,
    ) -> Result<(), RealForwardError> {
        if n_blocks + 1 > MAX_STATES {
            return Err(RealForwardError::Unsupported(format!(
                "depth mix over {} states exceeds {MAX_STATES}",
                n_blocks + 1
            )));
        }
        let h = self.h;
        self.copy(pass, cur, (states, (n_blocks * h * 2) as u64), h)?;
        gpu::encode_inkling2_depth_mix(
            self.context,
            pass,
            (states, 0),
            query,
            out,
            (n_blocks + 1) as u32,
            n_q as u32,
            h as u32,
        )
        .map_err(RealForwardError::Gpu)
    }

    /// Loop controller `ctrl` of core layer `layer` on `input` with `msg`.
    #[allow(clippy::too_many_arguments)]
    fn control(
        &mut self,
        pass: &gpu::PassEncoder,
        st: &RealInkling2State,
        layer: usize,
        ctrl: &str,
        input: Buf<'_>,
        msg: Buf<'_>,
        out: Buf<'_>,
    ) -> Result<(), RealForwardError> {
        let (h, rank) = (self.h, st.cfg.gate_rank as usize);
        let p = name(layer, &format!("controllers.{ctrl}"));
        let g = RealForwardError::Gpu;
        gpu::encode_rms_norm_no_scale(self.context, pass, input, (&st.cat, 0), h as u32, 1e-6)
            .map_err(g)?;
        gpu::encode_rms_norm_no_scale(
            self.context,
            pass,
            msg,
            (&st.cat, (h * 2) as u64),
            h as u32,
            1e-6,
        )
        .map_err(g)?;
        self.gemv(
            pass,
            &format!("{p}.down_weight"),
            rank,
            2 * h,
            (&st.cat, 0),
            (&st.gate_r, 0),
        )?;
        self.gemv(
            pass,
            &format!("{p}.gain_weight"),
            h,
            rank,
            (&st.gate_r, 0),
            (&st.gain, 0),
        )?;
        self.gemv(
            pass,
            &format!("{p}.forget_weight"),
            h,
            rank,
            (&st.gate_r, 0),
            (&st.forget, 0),
        )?;
        gpu::encode_inkling2_control_combine(
            self.context,
            pass,
            input,
            msg,
            (&st.gain, 0),
            (&st.forget, 0),
            out,
            h as u32,
        )
        .map_err(RealForwardError::Gpu)
    }
}

impl RealForwardRunner {
    /// The last token's DSpark context taps, `DSPARK_TAP_ROWS x hidden`, when
    /// the install opened as Inkling 2 with `TURBOSPARK_INKLING2_TAPS=1`.
    pub fn inkling2_taps(&self) -> Option<Vec<f16>> {
        let st = self.real_inkling2.as_ref()?;
        let t = st.taps.as_ref()?;
        let n = DSPARK_TAP_ROWS * self.arch.hidden_size as usize;
        let mut out = vec![f16::ZERO; n];
        gpu::read_buffer_f16_into(t, 0, &mut out);
        Some(out)
    }

    pub(crate) fn produce_real_inkling2(
        &mut self,
        token: i32,
        position: usize,
        logits: &mut [LogitValue],
    ) -> Result<(), RealForwardError> {
        let started = std::time::Instant::now();
        // The state is taken out for the token so its buffers can be borrowed
        // beside `self.context` and the streamers, and always put back.
        let mut st = self.real_inkling2.take().expect("caller checked is_some");
        // A single-token pass advances the conv tails in place, so the last
        // verify's tail snapshots no longer describe a rollback target.
        st.verify = None;
        let result = self.produce_real_inkling2_inner(&mut st, token, position, logits);
        if let Some(d) = st.dspark.as_mut() {
            d.source = if result.is_ok() && !self.skip_head {
                dspark::TapSource::Single(position)
            } else {
                // A headless pass skips the final-norm tap.
                dspark::TapSource::None
            };
        }
        self.real_inkling2 = Some(st);
        self.phases.calls += 1;
        self.phases.total_nanos += started.elapsed().as_nanos() as u64;
        result
    }

    fn produce_real_inkling2_inner(
        &mut self,
        st: &mut RealInkling2State,
        token: i32,
        position: usize,
        logits: &mut [LogitValue],
    ) -> Result<(), RealForwardError> {
        let arch = self.arch.clone();
        let cfg = st.cfg.clone();
        let (h, vocab) = (arch.hidden_size as usize, arch.vocab_size as usize);
        if position != self.kv.position() {
            return Err(RealForwardError::Unsupported(format!(
                "non-sequential position {position}; KV cache is at {}",
                self.kv.position()
            )));
        }
        if position >= st.max_ctx {
            return Err(RealForwardError::Unsupported(format!(
                "position {position} is past the {}-token Inkling 2 cache",
                st.max_ctx
            )));
        }
        if token < 0 || token as u32 >= cfg.unpadded_vocab {
            return Err(RealForwardError::Unsupported(format!(
                "token id {token} outside vocab {}",
                cfg.unpadded_vocab
            )));
        }
        let mut c = Ctx {
            context: &mut self.context,
            weights: &self.weights,
            index: &self.index,
            h,
            eps: cfg.rms_eps,
            lamq: st.lamq.clone(),
        };
        let mut pass = c.context.begin_pass_labeled("inkling2 attn+router");
        // BF16, MLX affine 4/8-bit on an MLX-sourced install, or a LAMQ
        // bucket row.
        if let Some(l) = c.lamq.clone() {
            l.embed(
                c.context,
                &pass,
                c.weights.buffer(),
                token as u32,
                (&st.y, 0),
            )?;
        } else {
            crate::real_forward_dispatch::encode_embed_any(
                c.context,
                &pass,
                c.weights,
                c.index,
                "model.llm.embed.weight",
                (&st.y, 0),
                token as u32,
                h as u32,
                vocab,
                1.0,
            )?;
        }
        c.norm(
            &pass,
            (&st.y, 0),
            "model.llm.embed_norm.weight",
            (&st.cur, 0),
        )?;

        let (core_start, core_end, layers) = (
            cfg.core_start as usize,
            cfg.core_end as usize,
            arch.num_layers as usize,
        );
        let core = core_end - core_start;
        let mut n_blocks = 0usize;
        let trace = trace_enabled();
        let t_token = std::time::Instant::now();
        let mut tr = TokenTrace::default();
        let mut apply = |c: &mut Ctx<'_>,
                         pass: &mut gpu::PassEncoder,
                         st: &mut RealInkling2State,
                         n_blocks: &mut usize,
                         slot: usize,
                         ip: usize,
                         tr: &mut TokenTrace,
                         msgs: bool|
         -> Result<(), RealForwardError> {
            let t0 = std::time::Instant::now();
            layer_attention(c, pass, st, &arch, &cfg, n_blocks, slot, ip, msgs, position)?;
            if st.resident.is_some() {
                // Every expert is resident and the router runs on the GPU,
                // so the application stays in this pass: no commit, no wait.
                layer_moe_resident(c, pass, st, &arch, &cfg, slot, ip)?;
                tr.encode_ms += t0.elapsed().as_secs_f64() * 1e3;
                return Ok(());
            }
            let committed =
                std::mem::replace(pass, c.context.begin_pass_labeled("inkling2 routed")).commit();
            let t1 = std::time::Instant::now();
            tr.gpu_ms += committed.wait_with_gpu_time() * 1e3;
            let t2 = std::time::Instant::now();
            let (hits, misses) = layer_moe(
                c,
                pass,
                st,
                &arch,
                &cfg,
                &mut self.streamers,
                &self.slot_buffers,
                slot,
                ip,
                tr,
            )?;
            tr.encode_ms += (t1 - t0).as_secs_f64() * 1e3;
            tr.wait_ms += (t2 - t1).as_secs_f64() * 1e3;
            tr.moe_host_ms += t2.elapsed().as_secs_f64() * 1e3;
            tr.hits += hits;
            tr.misses += misses;
            Ok(())
        };

        // DSpark context taps: the stream entering applications 5, 23 and 41,
        // then the final-norm output (`dspark_tap_row`). Off unless
        // `TURBOSPARK_INKLING2_TAPS=1` allocated the buffer.
        let tap = |c: &mut Ctx<'_>,
                   pass: &gpu::PassEncoder,
                   st: &RealInkling2State,
                   visit: usize| {
            match (st.taps.as_ref(), dspark_tap_row(visit)) {
                (Some(t), Some(row)) => c.copy(pass, (&st.cur, 0), (t, (row * h * 2) as u64), h),
                _ => Ok(()),
            }
        };
        let mut visit = 0usize;
        for ip in 0..core_start {
            tap(&mut c, &pass, st, visit)?;
            apply(&mut c, &mut pass, st, &mut n_blocks, ip, ip, &mut tr, false)?;
            visit += 1;
        }
        c.copy(&pass, (&st.cur, 0), (&st.prelude, 0), h)?;
        let prelude_blocks = n_blocks;
        for p in 0..cfg.passes as usize {
            if p > 0 {
                // The previous pass's bank: its blocks after the first core
                // block, then its output. Taken before the reset overwrites
                // those rows.
                let first = prelude_blocks + 1;
                let k = n_blocks.saturating_sub(first);
                if k > 0 {
                    c.copy(
                        &pass,
                        (&st.states, (first * h * 2) as u64),
                        (&st.bank, 0),
                        k * h,
                    )?;
                }
                c.copy(&pass, (&st.cur, 0), (&st.bank, (k * h * 2) as u64), h)?;
                c.copy(&pass, (&st.prelude, 0), (&st.cur, 0), h)?;
                n_blocks = prelude_blocks;
                st.bank_states = k + 1;
            }
            for ip in core_start..core_end {
                let i = ip - core_start;
                let slot = if p == 0 {
                    ip
                } else {
                    layers + (p - 1) * core + i
                };
                if p > 0 {
                    gpu::encode_inkling2_depth_mix(
                        c.context,
                        &pass,
                        (&st.bank, 0),
                        (&st.msg_queries[i], 0),
                        (&st.msgs, 0),
                        st.bank_states as u32,
                        4,
                        h as u32,
                    )
                    .map_err(RealForwardError::Gpu)?;
                }
                tap(&mut c, &pass, st, visit)?;
                apply(
                    &mut c,
                    &mut pass,
                    st,
                    &mut n_blocks,
                    slot,
                    ip,
                    &mut tr,
                    p > 0,
                )?;
                visit += 1;
            }
        }
        for ip in core_end..layers {
            tap(&mut c, &pass, st, visit)?;
            apply(&mut c, &mut pass, st, &mut n_blocks, ip, ip, &mut tr, false)?;
            visit += 1;
        }

        if !self.skip_head {
            let q = view(c.weights, c.index, "model.llm.attn_res.output_query")?;
            if fused_enabled() {
                c.mix_norm(
                    &pass,
                    &st.states,
                    n_blocks,
                    (&st.cur, 0),
                    q,
                    &["model.llm.norm.weight"],
                    (&st.mn, 0),
                )?;
            } else {
                c.mix(&pass, &st.states, n_blocks, (&st.cur, 0), q, 1, (&st.m, 0))?;
                c.norm(&pass, (&st.m, 0), "model.llm.norm.weight", (&st.mn, 0))?;
            }
            if let Some(t) = st.taps.as_ref() {
                c.copy(
                    &pass,
                    (&st.mn, 0),
                    (t, (DSPARK_TAPS_MIDDLE * h * 2) as u64),
                    h,
                )?;
            }
            if let Some(l) = c.lamq.clone() {
                l.head(
                    c.context,
                    &pass,
                    c.weights.buffer(),
                    (&st.mn, 0),
                    (&self.scratch.logits, 0),
                )?;
            } else {
                c.gemv(
                    &pass,
                    "model.llm.unembed.weight",
                    vocab,
                    h,
                    (&st.mn, 0),
                    (&self.scratch.logits, 0),
                )?;
            }
            gpu::encode_logit_softcap(
                c.context,
                &pass,
                (&self.scratch.logits, 0),
                arch.final_logit_softcap as f32,
                vocab as u32,
            )
            .map_err(RealForwardError::Gpu)?;
        }
        tr.gpu_ms += pass.commit_and_wait_with_gpu_time() * 1e3;
        if trace {
            tr.report(position, t_token.elapsed().as_secs_f64() * 1e3);
        }
        self.kv.advance();
        if self.skip_head {
            return Ok(());
        }
        if vocab != logits.len() {
            return Err(RealForwardError::Unsupported(format!(
                "vocab mismatch: model has {vocab}, caller expected {}",
                logits.len()
            )));
        }
        gpu::read_buffer_f16_into(&self.scratch.logits, 0, logits);
        mask_padding(logits, cfg.unpadded_vocab as usize);
        Ok(())
    }
}

/// Makes the padding rows past the real vocabulary unsampleable with
/// [`PAD_LOGIT`] (verify writes the same value). The most negative FINITE
/// f16, never -inf: `selection::select` refuses a score
/// vector with any non-finite entry, which is how every Inkling chat failed
/// while the logit gates (which drop these rows) passed. After the
/// sampler's max subtraction its weight underflows to exactly zero at any
/// temperature, and a greedy argmax never reaches it.
fn mask_padding(logits: &mut [f16], unpadded: usize) {
    let start = unpadded.min(logits.len());
    for l in &mut logits[start..] {
        *l = PAD_LOGIT;
    }
}

#[cfg(test)]
mod padding_tests {
    use super::*;
    use foundation::LogitsView;

    #[test]
    fn padding_is_finite_and_never_chosen() {
        let greedy = selection::ShapingConfig::new(0.0, 0, None, 1.0, Some(1)).unwrap();
        // Rows 6 and 7 are padding. The real rows are all very negative, so
        // a padding row left at a large value would win any comparison.
        let mut real = vec![f16::from_f32(-60000.0); 8];
        real[6] = f16::from_f32(3.0);
        real[7] = f16::from_f32(9.0);
        real[2] = f16::from_f32(1.0);
        mask_padding(&mut real, 6);
        assert!(real.iter().all(|l| l.is_finite()));
        assert_eq!(
            selection::select(LogitsView::new(&real), &greedy, &[], 0).unwrap(),
            2
        );
        let hot = selection::ShapingConfig::new(5.0, 0, None, 1.0, Some(7)).unwrap();
        for pos in 0..200 {
            let id = selection::select(LogitsView::new(&real), &hot, &[], pos).unwrap();
            assert!(id < 6, "padding row {id} sampled at position {pos}");
        }
    }
}

/// The attention half of one layer application, through the router GEMV.
#[allow(clippy::too_many_arguments)]
fn layer_attention(
    c: &mut Ctx<'_>,
    pass: &gpu::PassEncoder,
    st: &mut RealInkling2State,
    arch: &model_io::ArchConfig,
    cfg: &model_io::inkling2_config::Inkling2Config,
    n_blocks: &mut usize,
    slot: usize,
    ip: usize,
    msgs: bool,
    position: usize,
) -> Result<(), RealForwardError> {
    let g = RealForwardError::Gpu;
    let h = c.h;
    let (heads, kvh, hd) = (
        arch.num_heads as usize,
        arch.num_kv_heads as usize,
        arch.head_dim as usize,
    );
    let (kvd, taps, d_rel, extent) = (
        kvh * hd,
        arch.short_conv_kernel_size as u32,
        cfg.d_rel as usize,
        cfg.rel_extent,
    );
    let full = arch.full_attention_layer_mask[ip] == 1;
    let row = |b: &gpu::MetalBuffer, j: usize| -> (gpu::MetalBuffer, u64) {
        (b.clone(), (j * h * 2) as u64)
    };

    // Fused: the mix and its norms in one dispatch, when no controller sits
    // between them (layer 0 has no mix; message visits run the controllers
    // on the un-normed mix).
    let normed: [(gpu::MetalBuffer, u64); 3] = if fused_enabled() && ip > 0 && !msgs {
        let q = view(
            c.weights,
            c.index,
            &format!("model.llm.attn_res.attn_queries.{}", ip - 1),
        )?;
        let names: Vec<String> = if full {
            ["q", "k", "v"]
                .iter()
                .map(|w| name(ip, &format!("{w}_input_norm.weight")))
                .collect()
        } else {
            vec![name(ip, "attn_norm.weight")]
        };
        let ws: Vec<&str> = names.iter().map(String::as_str).collect();
        c.mix_norm(
            pass,
            &st.states,
            *n_blocks,
            (&st.cur, 0),
            q,
            &ws,
            (&st.normed, 0),
        )?;
        [0, 1, 2].map(|j| row(&st.normed, if full { j } else { 0 }))
    } else {
        // Attention inputs: the depth mix (1 query on SWA, 3 on full layers), or
        // the stream itself on layer 0.
        let n_q = if full { 3 } else { 1 };
        let sources: [(gpu::MetalBuffer, u64); 3] = if ip > 0 {
            let q = view(
                c.weights,
                c.index,
                &format!("model.llm.attn_res.attn_queries.{}", ip - 1),
            )?;
            c.mix(
                pass,
                &st.states,
                *n_blocks,
                (&st.cur, 0),
                q,
                n_q,
                (&st.mixed, 0),
            )?;
            [0, 1, 2].map(|j| row(&st.mixed, if full { j } else { 0 }))
        } else {
            [0, 1, 2].map(|_| (st.cur.clone(), 0))
        };
        let mut inputs = sources;
        if msgs {
            for (j, ctrl) in ["attn", "k", "v"].iter().enumerate() {
                let (sb, so) = inputs[j].clone();
                c.control(
                    pass,
                    st,
                    ip,
                    ctrl,
                    (&sb, so),
                    (&st.msgs, (j * h * 2) as u64),
                    (&st.inputs, (j * h * 2) as u64),
                )?;
                inputs[j] = row(&st.inputs, j);
            }
        }
        // Norms: SWA with messages norms each input; SWA without one norm feeds
        // all three; full layers have their own q/k/v input norms.
        if !full && !msgs {
            let (b, o) = inputs[0].clone();
            c.norm(
                pass,
                (&b, o),
                &name(ip, "attn_norm.weight"),
                (&st.normed, 0),
            )?;
            [0, 1, 2].map(|_| row(&st.normed, 0))
        } else {
            for (j, which) in ["q", "k", "v"].iter().enumerate() {
                let w = if full {
                    name(ip, &format!("{which}_input_norm.weight"))
                } else {
                    name(ip, "attn_norm.weight")
                };
                let (b, o) = inputs[j].clone();
                c.norm(pass, (&b, o), &w, (&st.normed, (j * h * 2) as u64))?;
            }
            [0, 1, 2].map(|j| row(&st.normed, j))
        }
    };
    let n = |j: usize| (&normed[j].0, normed[j].1);
    let a = |s: &str| name(ip, &format!("attn.{s}"));

    // Every projection of the attention input in one dispatch (the SWA
    // layer without messages reads one normed row for all of them).
    let names = [
        a("wq_du.weight"),
        a("wk_dv.weight"),
        a("wv_dv.weight"),
        a("wr_du.weight"),
        a("attn_sink_proj.weight"),
        a("gate_attn.weight"),
        a("k_sconv.weight_proj"),
        a("v_sconv.weight_proj"),
    ];
    let mut segs: Vec<(&str, usize, usize, usize, Buf<'_>)> = vec![
        (&names[0], heads * hd, h, 0, (&st.q, 0)),
        (&names[1], kvd, h, 1, (&st.k_pre, 0)),
        (&names[2], kvd, h, 2, (&st.v_pre, 0)),
        (&names[3], heads * d_rel, h, 0, (&st.r, 0)),
        (&names[4], heads, h, 0, (&st.sinks, 0)),
        (&names[5], heads * hd, h, 0, (&st.gate_attn, 0)),
    ];
    if full {
        segs.push((&names[6], kvh * taps as usize, h, 1, (&st.dk, 0)));
        segs.push((&names[7], kvh * taps as usize, h, 2, (&st.dv, 0)));
    }
    c.gemv_group(pass, &[n(0), n(1), n(2)], &segs)?;

    // Convs write straight into this slot's KV row for `position` (a ring
    // row on a sliding-window slot, which overwrites the position that just
    // left the window).
    let s = &st.slots[slot];
    let row_off = ((position % s.rows) * kvd * 2) as u64;
    let kw = view(c.weights, c.index, &a("k_sconv.weight"))?;
    let vw = view(c.weights, c.index, &a("v_sconv.weight"))?;
    let tau = if full {
        compute::inkling2::log_scale_tau(position, cfg.log_floor, cfg.log_alpha)
    } else {
        1.0
    };
    if fused_enabled() {
        // One dispatch for the convs, clips and V norm, bit-identical to the
        // chain below (`gpu/tests/inkling2_fused_parity.rs`).
        let qs = view(c.weights, c.index, &a("q_l2_clip.scale"))?;
        let rs = view(c.weights, c.index, &a("rel_l2_clip.scale"))?;
        gpu::encode_inkling2_qkv_post(
            c.context,
            pass,
            &gpu::Inkling2QkvPostBuffers {
                q: (&st.q, 0),
                q_scale: qs,
                r: (&st.r, 0),
                r_scale: rs,
                k_pre: (&st.k_pre, 0),
                v_pre: (&st.v_pre, 0),
                tail_k: (&s.tail_k, 0),
                tail_v: (&s.tail_v, 0),
                k_w: kw,
                v_w: vw,
                dyn_k: (&st.dk, 0),
                dyn_v: (&st.dv, 0),
                k_out: (&s.k, row_off),
                v_out: (&s.v, row_off),
            },
            gpu::Inkling2QkvPostShape {
                heads: heads as u32,
                kv_heads: kvh as u32,
                head_dim: hd as u32,
                d_rel: d_rel as u32,
                taps,
                dyn_heads: if full { kvh as u32 } else { 0 },
                qk_clip: cfg.qk_clip,
                rel_clip: cfg.rel_clip,
                tau,
                rms_eps: cfg.rms_eps,
            },
        )
        .map_err(g)?;
    } else {
        let dyn_k = full.then_some(((&st.dk, 0u64), kvh as u32));
        let dyn_v = full.then_some(((&st.dv, 0u64), kvh as u32));
        gpu::encode_inkling2_short_conv_decode(
            c.context,
            pass,
            (&st.k_pre, 0),
            (&s.tail_k, 0),
            kw,
            dyn_k,
            (&s.k, row_off),
            kvd as u32,
            taps,
        )
        .map_err(g)?;
        gpu::encode_inkling2_short_conv_decode(
            c.context,
            pass,
            (&st.v_pre, 0),
            (&s.tail_v, 0),
            vw,
            dyn_v,
            (&s.v, row_off),
            kvd as u32,
            taps,
        )
        .map_err(g)?;

        let qs = view(c.weights, c.index, &a("q_l2_clip.scale"))?;
        let rs = view(c.weights, c.index, &a("rel_l2_clip.scale"))?;
        gpu::encode_inkling2_l2_clip(
            c.context,
            pass,
            (&st.q, 0),
            Some((qs, hd as u32)),
            heads as u32,
            hd as u32,
            cfg.qk_clip,
            tau,
        )
        .map_err(g)?;
        gpu::encode_inkling2_l2_clip(
            c.context,
            pass,
            (&s.k, row_off),
            None,
            kvh as u32,
            hd as u32,
            cfg.qk_clip,
            1.0,
        )
        .map_err(g)?;
        gpu::encode_rms_norm_no_scale_perhead(
            c.context,
            pass,
            (&s.v, row_off),
            (&s.v, row_off),
            kvh as u32,
            hd as u32,
            cfg.rms_eps,
        )
        .map_err(g)?;
        gpu::encode_inkling2_l2_clip(
            c.context,
            pass,
            (&st.r, 0),
            Some((rs, d_rel as u32)),
            heads as u32,
            d_rel as u32,
            cfg.rel_clip,
            tau,
        )
        .map_err(g)?;
    }

    let proj = view(c.weights, c.index, &a("rel_logits_proj.proj"))?;
    gpu::encode_inkling2_rel_table(
        c.context,
        pass,
        (&st.r, 0),
        proj,
        (&st.table, 0),
        heads as u32,
        d_rel as u32,
        extent,
    )
    .map_err(g)?;
    let first = if full {
        0
    } else {
        (position + 1).saturating_sub(arch.sliding_window as usize)
    };
    let shape = gpu::Inkling2AttnShape {
        heads: heads as u32,
        kv_heads: kvh as u32,
        head_dim: hd as u32,
        extent,
        ring: s.rows as u32,
    };
    gpu::encode_inkling2_attention_decode(
        c.context,
        pass,
        (&st.q, 0),
        (&s.k, 0),
        (&s.v, 0),
        (&st.table, 0),
        (&st.sinks, 0),
        (&st.attn_o, 0),
        first as u32,
        position as u32,
        shape,
    )
    .map_err(g)?;
    gpu::encode_sigmoid_gate_mul(
        c.context,
        pass,
        (&st.attn_o, 0),
        (&st.gate_attn, 0),
        (heads * hd) as u32,
    )
    .map_err(g)?;
    c.gemv(
        pass,
        &a("wo_ud.weight"),
        h,
        heads * hd,
        (&st.attn_o, 0),
        (&st.o2, 0),
    )?;
    let ow = view(c.weights, c.index, &name(ip, "attn_sconv.weight"))?;
    let starts_block = ip % cfg.res_block as usize == 0;
    if fused_enabled() {
        // The conv and the residual step in one dispatch, bit-identical to
        // the conv then the add (or then the two copies) below.
        let step = if starts_block {
            gpu::Inkling2ConvResidual::StartBlock {
                save: (&st.states, (*n_blocks * h * 2) as u64),
            }
        } else {
            gpu::Inkling2ConvResidual::Add
        };
        gpu::encode_inkling2_short_conv_residual(
            c.context,
            pass,
            (&st.o2, 0),
            (&s.tail_o, 0),
            ow,
            (&st.cur, 0),
            step,
            h as u32,
            taps,
        )
        .map_err(g)?;
        if starts_block {
            *n_blocks += 1;
        }
    } else {
        gpu::encode_inkling2_short_conv_decode(
            c.context,
            pass,
            (&st.o2, 0),
            (&s.tail_o, 0),
            ow,
            None,
            (&st.attn_out, 0),
            h as u32,
            taps,
        )
        .map_err(g)?;

        // The first layer of a block starts a new stream; its input joins the
        // blocks. Otherwise the attention output is a residual.
        if ip % cfg.res_block as usize == 0 {
            c.copy(
                pass,
                (&st.cur, 0),
                (&st.states, (*n_blocks * h * 2) as u64),
                h,
            )?;
            *n_blocks += 1;
            c.copy(pass, (&st.attn_out, 0), (&st.cur, 0), h)?;
        } else {
            gpu::encode_residual_add(c.context, pass, (&st.cur, 0), (&st.attn_out, 0), h as u32)
                .map_err(g)?;
        }
    }

    // MoE input: depth mix, the FFN controller on later passes, the norm,
    // then the F32 router GEMV.
    let q = view(
        c.weights,
        c.index,
        &format!("model.llm.attn_res.mlp_queries.{ip}"),
    )?;
    if fused_enabled() && !msgs {
        c.mix_norm(
            pass,
            &st.states,
            *n_blocks,
            (&st.cur, 0),
            q,
            &[&name(ip, "mlp_norm.weight")],
            (&st.mn, 0),
        )?;
    } else {
        c.mix(pass, &st.states, *n_blocks, (&st.cur, 0), q, 1, (&st.m, 0))?;
    }
    if msgs {
        c.control(
            pass,
            st,
            ip,
            "mlp",
            (&st.m, 0),
            (&st.msgs, (3 * h * 2) as u64),
            (&st.y, 0),
        )?;
        c.copy(pass, (&st.y, 0), (&st.m, 0), h)?;
    }
    if !(fused_enabled() && !msgs) {
        c.norm(pass, (&st.m, 0), &name(ip, "mlp_norm.weight"), (&st.mn, 0))?;
    }
    let router = view(c.weights, c.index, &name(ip, "mlp.gate.weight"))?;
    gpu::encode_minimax_router(
        c.context,
        pass,
        router,
        (&st.mn, 0),
        (&st.router_logits, 0),
        arch.num_experts as u32,
        h as u32,
    )
    .map_err(g)
}

/// The routed half of one layer application, after the router sync.
#[allow(clippy::too_many_arguments)]
fn layer_moe(
    c: &mut Ctx<'_>,
    pass: &gpu::PassEncoder,
    st: &mut RealInkling2State,
    arch: &model_io::ArchConfig,
    cfg: &model_io::inkling2_config::Inkling2Config,
    streamers: &mut [Option<streaming::PreadExpertStreamer>],
    slot_buffers: &[Vec<gpu::MetalBuffer>],
    slot: usize,
    ip: usize,
    tr: &mut TokenTrace,
) -> Result<(usize, usize), RealForwardError> {
    let t0 = std::time::Instant::now();
    let g = RealForwardError::Gpu;
    let (h, ffn, experts, top_k) = (
        c.h,
        arch.moe_intermediate_size as usize,
        arch.num_experts as usize,
        arch.top_k_experts as usize,
    );
    let logits = gpu::read_f32_buffer_at(&st.router_logits, 0, experts);
    let (selected, weights) = compute::inkling2::route_sigmoid(
        &logits,
        &st.router_bias[ip],
        top_k,
        arch.routed_scaling_factor as f32,
    );
    let t1 = std::time::Instant::now();
    let streamer = streamers
        .get_mut(ip)
        .and_then(Option::as_mut)
        .ok_or_else(|| {
            RealForwardError::Unsupported(format!("layer {ip} has no expert streamer"))
        })?;
    let plan = streamer.plan_experts_cached(&selected, &std::collections::HashSet::new());
    let cache_slots = streamer
        .execute_expert_cache_plan(&plan)
        .map_err(|e| RealForwardError::Unsupported(format!("expert stream: {e}")))?;
    let t2 = std::time::Instant::now();
    tr.route_ms += (t1 - t0).as_secs_f64() * 1e3;
    tr.stream_ms += (t2 - t1).as_secs_f64() * 1e3;
    let mut routing16 = [f16::from_f32(0.0); gpu::MAX_STREAMED_EXPERTS];
    for (i, &w) in weights.iter().enumerate() {
        routing16[i] = f16::from_f32(w);
    }
    let bytes: Vec<u8> = routing16
        .iter()
        .flat_map(|v| v.to_bits().to_le_bytes())
        .collect();
    gpu::write_buffer_bytes(&st.routing_w, 0, &bytes);
    let blobs: Vec<(&gpu::MetalBuffer, u64)> = cache_slots
        .iter()
        .map(|&s| (&slot_buffers[ip][s], 0u64))
        .collect();
    let lq = st.lamq.clone().filter(|l| l.experts.is_some());
    let gguf = match st.kinds.get(ip) {
        Some(&state::ExpertKind::Gguf { gate_up, down }) => Some((gate_up, down)),
        _ => None,
    };
    let (source, function) = if lq.is_some() {
        (
            gpu::moe_inkling2_lamq_source(),
            gpu::MOE_INKLING2_LAMQ_REFLECT_FUNCTION,
        )
    } else if gguf.is_some() {
        (gpu::moe_gguf_source(), state::GGUF_REFLECT_FUNCTION)
    } else {
        (
            gpu::moe_inkling2_source(),
            gpu::MOE_INKLING2_REFLECT_FUNCTION,
        )
    };
    st.routed
        .bind_for(c.context, source, function, false, &blobs)
        .map_err(g)?;
    for &(b, _) in &blobs {
        pass.use_read_buffer(b);
    }
    if let Some(l) = lq {
        l.moe(
            c.context,
            pass,
            c.weights.buffer(),
            &gpu::LamqExperts::Streamed(&st.routed),
            ip,
            (&st.mn, 0),
            (&st.routing_w, 0),
            (&st.y, 0),
            h,
            ffn,
            top_k,
            (cfg.gate_softcap, cfg.up_softcap),
        )?;
        moe_tail(c, pass, st, arch, slot, ip)?;
        return Ok((plan.hits, plan.misses.len()));
    }
    if let Some((gate_up, down)) = gguf {
        gpu::encode_inkling2_gguf_phase1(
            c.context,
            pass,
            gate_up,
            &st.routed,
            &st.offsets[ip],
            (&st.mn, 0),
            (&st.acts, 0),
            h as u32,
            ffn as u32,
            top_k as u32,
            cfg.gate_softcap,
            cfg.up_softcap,
        )
        .map_err(g)?;
        gpu::encode_inkling2_gguf_phase2(
            c.context,
            pass,
            down,
            &st.routed,
            &st.offsets[ip],
            (&st.acts, 0),
            (&st.routing_w, 0),
            (&st.y, 0),
            h as u32,
            ffn as u32,
            top_k as u32,
        )
        .map_err(g)?;
        moe_tail(c, pass, st, arch, slot, ip)?;
        return Ok((plan.hits, plan.misses.len()));
    }
    let state::ExpertKind::Native(kind) = st.kinds[ip] else {
        unreachable!("GGUF layers returned above")
    };
    gpu::encode_inkling2_moe_phase1(
        c.context,
        pass,
        kind,
        &st.routed,
        &st.offsets[ip],
        (&st.mn, 0),
        (&st.acts, 0),
        h as u32,
        ffn as u32,
        top_k as u32,
        cfg.gate_softcap,
        cfg.up_softcap,
    )
    .map_err(g)?;
    gpu::encode_inkling2_moe_phase2(
        c.context,
        pass,
        kind,
        &st.routed,
        &st.offsets[ip],
        (&st.acts, 0),
        (&st.routing_w, 0),
        (&st.y, 0),
        h as u32,
        ffn as u32,
        top_k as u32,
    )
    .map_err(g)?;
    moe_tail(c, pass, st, arch, slot, ip)?;
    Ok((plan.hits, plan.misses.len()))
}

/// What the padding rows past `unpadded_vocab` read (see the decode readback).
pub(crate) const PAD_LOGIT: f16 = f16::MIN;

/// The DSpark drafter's target visits, `mtp_config.target_layer_ids` minus the
/// last (48, the final-norm output, stored as the row after these).
const DSPARK_TAP_VISITS: [usize; 3] = [5, 23, 41];
const DSPARK_TAPS_MIDDLE: usize = DSPARK_TAP_VISITS.len();
/// Rows of the per-token tap buffer: the three visits and the final norm.
pub(crate) const DSPARK_TAP_ROWS: usize = DSPARK_TAPS_MIDDLE + 1;

fn dspark_tap_row(visit: usize) -> Option<usize> {
    DSPARK_TAP_VISITS.iter().position(|&v| v == visit)
}

/// The expert sum's short conv and the residual add, after either MoE form.
fn moe_tail(
    c: &mut Ctx<'_>,
    pass: &gpu::PassEncoder,
    st: &RealInkling2State,
    arch: &model_io::ArchConfig,
    slot: usize,
    ip: usize,
) -> Result<(), RealForwardError> {
    let g = RealForwardError::Gpu;
    let h = c.h;
    let fw = view(c.weights, c.index, &name(ip, "mlp_sconv.weight"))?;
    let s = &st.slots[slot];
    if fused_enabled() {
        return gpu::encode_inkling2_short_conv_residual(
            c.context,
            pass,
            (&st.y, 0),
            (&s.tail_f, 0),
            fw,
            (&st.cur, 0),
            gpu::Inkling2ConvResidual::Add,
            h as u32,
            arch.short_conv_kernel_size as u32,
        )
        .map_err(g);
    }
    gpu::encode_inkling2_short_conv_decode(
        c.context,
        pass,
        (&st.y, 0),
        (&s.tail_f, 0),
        fw,
        None,
        (&st.y2, 0),
        h as u32,
        arch.short_conv_kernel_size as u32,
    )
    .map_err(g)?;
    gpu::encode_residual_add(c.context, pass, (&st.cur, 0), (&st.y2, 0), h as u32).map_err(g)
}

/// The routed half with every expert resident: the GPU router writes the
/// ranked ids and weights the indexed MoE pair reads, inside the same pass.
/// Same selection rule as `route_sigmoid` (`inkling2_router_select_k8`,
/// parity-tested against it in `inkling2_ops_parity.rs`).
#[allow(clippy::too_many_arguments)]
fn layer_moe_resident(
    c: &mut Ctx<'_>,
    pass: &gpu::PassEncoder,
    st: &RealInkling2State,
    arch: &model_io::ArchConfig,
    cfg: &model_io::inkling2_config::Inkling2Config,
    slot: usize,
    ip: usize,
) -> Result<(), RealForwardError> {
    let g = RealForwardError::Gpu;
    let (h, ffn, top_k) = (
        c.h,
        arch.moe_intermediate_size as usize,
        arch.top_k_experts as usize,
    );
    if top_k != gpu::INKLING2_ROUTER_TOP_K {
        return Err(RealForwardError::Unsupported(format!(
            "resident experts route top-{} on the GPU; this install selects {top_k}",
            gpu::INKLING2_ROUTER_TOP_K
        )));
    }
    let r = st.resident.as_ref().expect("caller checked resident");
    gpu::encode_inkling2_router_select(
        c.context,
        pass,
        (&st.router_logits, 0),
        (&r.bias[ip], 0),
        (&r.sel, 0),
        (&st.routing_w, 0),
        r.n_experts,
        arch.routed_scaling_factor as f32,
    )
    .map_err(g)?;
    let layer = gpu::Inkling2ResidentLayer {
        buffer: &r.layers[ip],
        stride: r.strides[ip],
        n_experts: r.n_experts,
        sel: (&r.sel, 0),
    };
    if let Some(l) = st.lamq.clone().filter(|l| l.experts.is_some()) {
        l.moe(
            c.context,
            pass,
            c.weights.buffer(),
            &gpu::LamqExperts::Resident(&layer),
            ip,
            (&st.mn, 0),
            (&st.routing_w, 0),
            (&st.y, 0),
            h,
            ffn,
            top_k,
            (cfg.gate_softcap, cfg.up_softcap),
        )?;
        return moe_tail(c, pass, st, arch, slot, ip);
    }
    let state::ExpertKind::Native(kind) = st.kinds[ip] else {
        return Err(RealForwardError::Unsupported(
            "resident experts have no GGUF kernels (state build refuses this)".into(),
        ));
    };
    gpu::encode_inkling2_moe_phase1_indexed(
        c.context,
        pass,
        kind,
        &layer,
        &st.offsets[ip],
        (&st.mn, 0),
        (&st.acts, 0),
        h as u32,
        ffn as u32,
        top_k as u32,
        cfg.gate_softcap,
        cfg.up_softcap,
    )
    .map_err(g)?;
    gpu::encode_inkling2_moe_phase2_indexed(
        c.context,
        pass,
        kind,
        &layer,
        &st.offsets[ip],
        (&st.acts, 0),
        (&st.routing_w, 0),
        (&st.y, 0),
        h as u32,
        ffn as u32,
        top_k as u32,
    )
    .map_err(g)?;
    moe_tail(c, pass, st, arch, slot, ip)
}

/// The fused decode kernels (oMLX 8dddad9a's launch-count work, ported):
/// on unless `TURBOSPARK_INKLING2_FUSED=0`, which restores the separate
/// kernels. Both forms are bit-identical; the switch exists for A/B timing
/// and to bisect a regression.
fn fused_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("TURBOSPARK_INKLING2_FUSED").as_deref() != Ok("0"))
}

/// `TURBOSPARK_INKLING2_TRACE=1` prints one line per token: where the
/// wall time goes between host encoding, the 48 commit-and-wait round trips,
/// the host routing and expert reads, and GPU execution. A diagnostic, not a
/// benchmark: it adds no syncs, but the host timers include scheduler noise.
fn trace_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("TURBOSPARK_INKLING2_TRACE").as_deref() == Ok("1"))
}

#[derive(Default)]
struct TokenTrace {
    /// Host time encoding each application's attention half.
    encode_ms: f64,
    /// Host time blocked in the per-application wait.
    wait_ms: f64,
    /// Host routing, expert plan, reads and MoE encoding.
    moe_host_ms: f64,
    /// Of `moe_host_ms`: router readback and host top-k.
    route_ms: f64,
    /// Of `moe_host_ms`: expert cache plan and the pread of misses.
    stream_ms: f64,
    /// Summed GPU execution of every command buffer of the token.
    gpu_ms: f64,
    hits: usize,
    misses: usize,
}

impl TokenTrace {
    fn report(&self, position: usize, wall_ms: f64) {
        eprintln!(
            "inkling2 trace pos {position} wall {wall_ms:.1} ms: gpu {:.1}, wait {:.1}, \
             encode {:.1}, moe host {:.1} (route {:.1}, stream {:.1}) ms; \
             experts {} hit / {} read",
            self.gpu_ms,
            self.wait_ms,
            self.encode_ms,
            self.moe_host_ms,
            self.route_ms,
            self.stream_ms,
            self.hits,
            self.misses
        );
    }
}
