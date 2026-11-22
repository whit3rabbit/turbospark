//! Inkling 2 multi-row target forward: the speculative verify. PORT-LOCAL.
//!
//! `M <= 8` consecutive tokens at positions `base .. base + M` in ONE command
//! buffer, every kernel in its multi-row form (`gpu::inkling2_batch`): a
//! dense weight row is read once for all rows, and the dispatch count is the
//! single-token path's, not `M` times it. Row `r` computes the arithmetic the
//! single-token flow (`mod.rs`) computes at position `base + r`, so the logits
//! equal `M` sequential `produce` calls bit for bit
//! (`tests/real_forward_inkling2_verify.rs`, real install).
//!
//! Resident expert residency only: the streamed path routes on the host
//! between two command buffers per application, and a batched form of it is
//! not built. Refused by name.
//!
//! State a rejected row leaves behind and how it is undone:
//! - KV rows: written at their own position (an SWA slot's ring has
//!   `window + VERIFY_SLACK` rows, so no row of the block lands on a position
//!   an earlier row's window still reads); rewinding the cursor is enough.
//! - Conv tails: every conv writes `M + 1` tail snapshots (block start, then
//!   after each row) into `BatchScratch::snaps`; a rollback keeping `k` rows
//!   copies snapshot `k` back ([`RealForwardRunner::inkling2_restore_tails`]).
//! - The position cursor: `KvCacheManager::rewind_cursor`.

use super::state::{name, ExpertKind, RealInkling2State, MAX_STATES, TAG_BF16};
use super::{view, Buf, Ctx, DSPARK_TAPS_MIDDLE, DSPARK_TAP_ROWS};
use crate::real_forward::{RealForwardError, RealForwardRunner};
use crate::real_forward_utils::entry;
use foundation::LogitValue;

/// Rows one verify serves (the DSpark block: the anchor and 7 proposals).
pub(crate) const VERIFY_MAX_ROWS: usize = gpu::INKLING2_BATCH_MAX_ROWS;
/// Extra rows on every sliding-window ring so a block's rows never overwrite
/// a position an earlier row of the same block still attends to.
pub(crate) const VERIFY_SLACK: usize = VERIFY_MAX_ROWS - 1;

const M: usize = VERIFY_MAX_ROWS;

/// The verify's per-row activations (FP16 unless noted), each `M` rows.
pub(crate) struct BatchScratch {
    cur: gpu::MetalBuffer,
    attn_out: gpu::MetalBuffer,
    prelude: gpu::MetalBuffer,
    /// `[M][MAX_STATES][h]`.
    states: gpu::MetalBuffer,
    bank: gpu::MetalBuffer,
    /// `[M][4][h]`.
    mixed: gpu::MetalBuffer,
    msgs: gpu::MetalBuffer,
    /// `[M][3][h]`.
    inputs: gpu::MetalBuffer,
    normed: gpu::MetalBuffer,
    /// `[M][2][h]`.
    cat: gpu::MetalBuffer,
    gate_r: gpu::MetalBuffer,
    gain: gpu::MetalBuffer,
    forget: gpu::MetalBuffer,
    q: gpu::MetalBuffer,
    k_pre: gpu::MetalBuffer,
    v_pre: gpu::MetalBuffer,
    r: gpu::MetalBuffer,
    sinks: gpu::MetalBuffer,
    gate_attn: gpu::MetalBuffer,
    dk: gpu::MetalBuffer,
    dv: gpu::MetalBuffer,
    /// F32 `[M][heads][extent]`.
    table: gpu::MetalBuffer,
    attn_o: gpu::MetalBuffer,
    o2: gpu::MetalBuffer,
    m: gpu::MetalBuffer,
    mn: gpu::MetalBuffer,
    /// F32 `[M][experts]`.
    router_logits: gpu::MetalBuffer,
    acts: gpu::MetalBuffer,
    routing_w: gpu::MetalBuffer,
    /// `u32[M][8]`.
    sel: gpu::MetalBuffer,
    y: gpu::MetalBuffer,
    y2: gpu::MetalBuffer,
    /// `[M][vocab]`.
    pub(crate) logits: gpu::MetalBuffer,
    /// `[M][DSPARK_TAP_ROWS][h]`: each row's DSpark context taps.
    pub(crate) taps: gpu::MetalBuffer,
    /// Per cache slot, the `M + 1` tail snapshots of its K, V, out and FFN
    /// convs (`gpu::encode_inkling2_short_conv_rows`).
    snaps: Vec<[gpu::MetalBuffer; 4]>,
}

impl BatchScratch {
    pub(crate) fn new(
        context: &mut gpu::MetalContext,
        st: &RealInkling2State,
        arch: &model_io::ArchConfig,
    ) -> Self {
        let (h, heads, kvh, hd) = (
            arch.hidden_size as usize,
            arch.num_heads as usize,
            arch.num_kv_heads as usize,
            arch.head_dim as usize,
        );
        let kvd = kvh * hd;
        let taps = arch.short_conv_kernel_size as usize;
        let (d_rel, extent, rank) = (
            st.cfg.d_rel as usize,
            st.cfg.rel_extent as usize,
            st.cfg.gate_rank as usize,
        );
        let (experts, ffn, vocab) = (
            arch.num_experts as usize,
            arch.moe_intermediate_size as usize,
            arch.vocab_size as usize,
        );
        let half = |c: &gpu::MetalContext, n: usize| c.new_output_buffer((M * n.max(1) * 2) as u64);
        let snap = |c: &mut gpu::MetalContext, ch: usize| {
            c.new_buffer_with_data(&vec![0u8; (M + 1) * (taps - 1) * ch * 2])
        };
        let snaps = (0..st.slots.len())
            .map(|_| {
                [
                    snap(context, kvd),
                    snap(context, kvd),
                    snap(context, h),
                    snap(context, h),
                ]
            })
            .collect();
        Self {
            cur: half(context, h),
            attn_out: half(context, h),
            prelude: half(context, h),
            states: half(context, MAX_STATES * h),
            bank: half(context, MAX_STATES * h),
            mixed: half(context, 4 * h),
            msgs: half(context, 4 * h),
            inputs: half(context, 3 * h),
            normed: half(context, 3 * h),
            cat: half(context, 2 * h),
            gate_r: half(context, rank),
            gain: half(context, h),
            forget: half(context, h),
            q: half(context, heads * hd),
            k_pre: half(context, kvd),
            v_pre: half(context, kvd),
            r: half(context, heads * d_rel),
            sinks: half(context, heads),
            gate_attn: half(context, heads * hd),
            dk: half(context, kvh * taps),
            dv: half(context, kvh * taps),
            table: context.new_output_buffer((M * heads * extent * 4) as u64),
            attn_o: half(context, heads * hd),
            o2: half(context, h),
            m: half(context, h),
            mn: half(context, h),
            router_logits: context.new_output_buffer((M * experts * 4) as u64),
            acts: half(context, gpu::INKLING2_MAX_SLOTS * ffn),
            routing_w: half(context, gpu::INKLING2_MAX_SLOTS),
            sel: context.new_output_buffer((M * gpu::INKLING2_MAX_SLOTS * 4) as u64),
            y: half(context, h),
            y2: half(context, h),
            logits: half(context, vocab),
            taps: half(context, DSPARK_TAP_ROWS * h),
            snaps,
        }
    }
}

/// What the last verify left: where it started and how many rows it fed.
/// Cleared by every single-token pass and by every rollback.
#[derive(Clone, Copy, Debug)]
pub(crate) struct InklingVerifyRecord {
    pub(crate) base: usize,
    pub(crate) rows: usize,
}

/// A row-strided view: buffer, byte offset of row 0, row stride in halfs.
type Rows<'a> = (&'a gpu::MetalBuffer, u64, u32);

fn plain<'a>(r: Rows<'a>) -> gpu::Inkling2Rows<'a> {
    gpu::Inkling2Rows::plain((r.0, r.1), r.2)
}

impl Ctx<'_> {
    /// Multi-row form of [`Ctx::gemv_group`]: BF16 segments in one
    /// `inkling2_bf16_gemv_rows` dispatch, each MLX affine one through the
    /// shared dispatcher once per row (the same kernel the single-token path
    /// runs, so the same bits). Each segment is `(weight, rows, cols, input
    /// index, output rows)`.
    fn gemv_rows(
        &mut self,
        pass: &gpu::PassEncoder,
        xs: &[Rows<'_>],
        segs: &[(&str, usize, usize, usize, Rows<'_>)],
        n: usize,
    ) -> Result<(), RealForwardError> {
        let mut v = Vec::with_capacity(segs.len());
        for &(nm, rows, cols, x, y) in segs {
            if entry(self.index, nm)?.dtype != TAG_BF16 {
                for m in 0..n {
                    crate::real_forward_dispatch::encode_gemv_any(
                        self.context,
                        pass,
                        self.weights,
                        self.index,
                        nm,
                        rows,
                        cols,
                        (xs[x].0, xs[x].1 + (m as u64) * u64::from(xs[x].2) * 2),
                        (y.0, y.1 + (m as u64) * u64::from(y.2) * 2),
                    )?;
                }
                continue;
            }
            let (_, o) = view(self.weights, self.index, nm)?;
            v.push(gpu::Inkling2GemvRowsSegment {
                w_offset: o,
                rows: rows as u32,
                cols: cols as u32,
                x,
                y: (y.0, y.1),
                y_stride: y.2,
            });
        }
        if v.is_empty() {
            return Ok(());
        }
        let xb: Vec<((&gpu::MetalBuffer, u64), u32)> =
            xs.iter().map(|&(b, o, s)| ((b, o), s)).collect();
        gpu::encode_inkling2_bf16_gemv_rows(
            self.context,
            pass,
            self.weights.buffer(),
            &xb,
            &v,
            n as u32,
        )
        .map_err(RealForwardError::Gpu)
    }

    fn norm_rows(
        &mut self,
        pass: &gpu::PassEncoder,
        x: Rows<'_>,
        weight: &str,
        out: Rows<'_>,
        n: usize,
    ) -> Result<(), RealForwardError> {
        let w = view(self.weights, self.index, weight)?;
        gpu::encode_inkling2_rms_rows(
            self.context,
            pass,
            plain(x),
            Some(w),
            plain(out),
            self.h as u32,
            1,
            self.eps,
            n as u32,
        )
        .map_err(RealForwardError::Gpu)
    }

    fn copy_rows(
        &mut self,
        pass: &gpu::PassEncoder,
        src: Rows<'_>,
        dst: Rows<'_>,
        len: usize,
        n: usize,
    ) -> Result<(), RealForwardError> {
        gpu::encode_inkling2_copy_rows(
            self.context,
            pass,
            (src.0, src.1),
            src.2,
            (dst.0, dst.1),
            dst.2,
            len as u32,
            n as u32,
        )
        .map_err(RealForwardError::Gpu)
    }

    /// [`Ctx::mix`] per row: `cur` joins each row's states at `n_blocks`.
    #[allow(clippy::too_many_arguments)]
    fn mix_rows(
        &mut self,
        pass: &gpu::PassEncoder,
        states: &gpu::MetalBuffer,
        n_blocks: usize,
        cur: Rows<'_>,
        query: Buf<'_>,
        n_q: usize,
        out: Rows<'_>,
        n: usize,
    ) -> Result<(), RealForwardError> {
        if n_blocks + 1 > MAX_STATES {
            return Err(RealForwardError::Unsupported(format!(
                "depth mix over {} states exceeds {MAX_STATES}",
                n_blocks + 1
            )));
        }
        let h = self.h;
        let ss = (MAX_STATES * h) as u32;
        self.copy_rows(pass, cur, (states, (n_blocks * h * 2) as u64, ss), h, n)?;
        gpu::encode_inkling2_depth_mix_rows(
            self.context,
            pass,
            (states, 0),
            ss,
            query,
            (out.0, out.1),
            out.2,
            (n_blocks + 1) as u32,
            n_q as u32,
            h as u32,
            n as u32,
        )
        .map_err(RealForwardError::Gpu)
    }

    /// [`Ctx::control`] per row.
    #[allow(clippy::too_many_arguments)]
    fn control_rows(
        &mut self,
        pass: &gpu::PassEncoder,
        b: &BatchScratch,
        rank: usize,
        layer: usize,
        ctrl: &str,
        input: Rows<'_>,
        msg: Rows<'_>,
        out: Rows<'_>,
        n: usize,
    ) -> Result<(), RealForwardError> {
        let h = self.h;
        let p = name(layer, &format!("controllers.{ctrl}"));
        let g = RealForwardError::Gpu;
        let cs = (2 * h) as u32;
        gpu::encode_inkling2_rms_rows(
            self.context,
            pass,
            plain(input),
            None,
            plain((&b.cat, 0, cs)),
            h as u32,
            1,
            1e-6,
            n as u32,
        )
        .map_err(g)?;
        gpu::encode_inkling2_rms_rows(
            self.context,
            pass,
            plain(msg),
            None,
            plain((&b.cat, (h * 2) as u64, cs)),
            h as u32,
            1,
            1e-6,
            n as u32,
        )
        .map_err(g)?;
        let down = format!("{p}.down_weight");
        self.gemv_rows(
            pass,
            &[(&b.cat, 0, cs)],
            &[(&down, rank, 2 * h, 0, (&b.gate_r, 0, rank as u32))],
            n,
        )?;
        let (gain, forget) = (format!("{p}.gain_weight"), format!("{p}.forget_weight"));
        self.gemv_rows(
            pass,
            &[(&b.gate_r, 0, rank as u32)],
            &[
                (&gain, h, rank, 0, (&b.gain, 0, h as u32)),
                (&forget, h, rank, 0, (&b.forget, 0, h as u32)),
            ],
            n,
        )?;
        gpu::encode_inkling2_control_combine_rows(
            self.context,
            pass,
            (input.0, input.1),
            (msg.0, msg.1),
            (&b.gain, 0),
            (&b.forget, 0),
            (out.0, out.1),
            h as u32,
            [input.2, msg.2, h as u32, out.2],
            n as u32,
        )
        .map_err(g)
    }
}

impl RealForwardRunner {
    /// The multi-row Inkling 2 forward over `tokens` at `base ..`: one logit
    /// row per token in `logits` (`tokens.len() * vocab`). Leaves the KV
    /// cursor at `base + tokens.len()` and an [`InklingVerifyRecord`] for the
    /// rollback.
    pub(crate) fn verify_real_inkling2(
        &mut self,
        tokens: &[i32],
        base: usize,
        logits: &mut [LogitValue],
    ) -> Result<(), RealForwardError> {
        let started = std::time::Instant::now();
        let mut st = self.real_inkling2.take().expect("caller checked is_some");
        let result = self.verify_real_inkling2_inner(&mut st, tokens, base, logits);
        self.real_inkling2 = Some(st);
        self.phases.calls += 1;
        self.phases.total_nanos += started.elapsed().as_nanos() as u64;
        result
    }

    fn verify_real_inkling2_inner(
        &mut self,
        st: &mut RealInkling2State,
        tokens: &[i32],
        base: usize,
        logits: &mut [LogitValue],
    ) -> Result<(), RealForwardError> {
        let unsupported = RealForwardError::Unsupported;
        let arch = self.arch.clone();
        let cfg = st.cfg.clone();
        let vocab = arch.vocab_size as usize;
        let n = tokens.len();
        if n == 0 || n > M {
            return Err(unsupported(format!(
                "an Inkling 2 verify takes 1 to {M} rows, got {n}"
            )));
        }
        if st.resident.is_none() {
            return Err(unsupported(
                "the Inkling 2 batched verify needs resident expert residency \
                 (--expert-residency resident); the streamed path routes on the host and has \
                 no multi-row form"
                    .to_string(),
            ));
        }
        st.native_expert_types().map_err(unsupported)?;
        if base != self.kv.position() {
            return Err(unsupported(format!(
                "verify at position {base}; KV cache is at {}",
                self.kv.position()
            )));
        }
        if base + n > st.max_ctx {
            return Err(unsupported(format!(
                "verify rows {base}..{} pass the {}-token Inkling 2 cache",
                base + n,
                st.max_ctx
            )));
        }
        if logits.len() != n * vocab {
            return Err(unsupported(format!(
                "verify logits hold {} values, expected {n} x {vocab}",
                logits.len()
            )));
        }
        if let Some(&t) = tokens
            .iter()
            .find(|&&t| t < 0 || t as u32 >= cfg.unpadded_vocab)
        {
            return Err(unsupported(format!(
                "token id {t} outside vocab {}",
                cfg.unpadded_vocab
            )));
        }
        if st.batch.is_none() {
            st.batch = Some(BatchScratch::new(&mut self.context, st, &arch));
        }
        let b = st.batch.take().expect("allocated above");
        let result = self.verify_encode(st, &b, tokens, base, logits);
        st.batch = Some(b);
        result?;
        st.verify = Some(InklingVerifyRecord { base, rows: n });
        if let Some(d) = st.dspark.as_mut() {
            d.source = super::dspark::TapSource::Verify { base, rows: n };
        }
        Ok(())
    }

    fn verify_encode(
        &mut self,
        st: &mut RealInkling2State,
        b: &BatchScratch,
        tokens: &[i32],
        base: usize,
        logits: &mut [LogitValue],
    ) -> Result<(), RealForwardError> {
        let arch = self.arch.clone();
        let cfg = st.cfg.clone();
        let (h, vocab) = (arch.hidden_size as usize, arch.vocab_size as usize);
        let n = tokens.len();
        let hs = h as u32;
        let mut c = Ctx {
            context: &mut self.context,
            weights: &self.weights,
            index: &self.index,
            h,
            eps: cfg.rms_eps,
            // Refused above when set; carried so the struct stays one shape.
            lamq: st.lamq.clone(),
        };
        let pass = c.context.begin_pass_labeled("inkling2 verify");
        for (m, &t) in tokens.iter().enumerate() {
            crate::real_forward_dispatch::encode_embed_any(
                c.context,
                &pass,
                c.weights,
                c.index,
                "model.llm.embed.weight",
                (&b.y, (m * h * 2) as u64),
                t as u32,
                hs,
                vocab,
                1.0,
            )?;
        }
        c.norm_rows(
            &pass,
            (&b.y, 0, hs),
            "model.llm.embed_norm.weight",
            (&b.cur, 0, hs),
            n,
        )?;

        let (core_start, core_end, layers) = (
            cfg.core_start as usize,
            cfg.core_end as usize,
            arch.num_layers as usize,
        );
        let core = core_end - core_start;
        let mut n_blocks = 0usize;
        let ts = (DSPARK_TAP_ROWS * h) as u32;
        let tap = |c: &mut Ctx<'_>, visit: usize| -> Result<(), RealForwardError> {
            match super::dspark_tap_row(visit) {
                Some(row) => c.copy_rows(
                    &pass,
                    (&b.cur, 0, hs),
                    (&b.taps, (row * h * 2) as u64, ts),
                    h,
                    n,
                ),
                None => Ok(()),
            }
        };
        let mut visit = 0usize;
        let apply = |c: &mut Ctx<'_>,
                     st: &RealInkling2State,
                     n_blocks: &mut usize,
                     slot: usize,
                     ip: usize,
                     msgs: bool|
         -> Result<(), RealForwardError> {
            layer_attention_rows(
                c, &pass, st, b, &arch, &cfg, n_blocks, slot, ip, msgs, base, n,
            )?;
            layer_moe_rows(c, &pass, st, b, &arch, &cfg, slot, ip, n)
        };
        for ip in 0..core_start {
            tap(&mut c, visit)?;
            apply(&mut c, st, &mut n_blocks, ip, ip, false)?;
            visit += 1;
        }
        c.copy(&pass, (&b.cur, 0), (&b.prelude, 0), n * h)?;
        let prelude_blocks = n_blocks;
        let ss = (MAX_STATES * h) as u32;
        let mut bank_states = 0usize;
        for p in 0..cfg.passes as usize {
            if p > 0 {
                let first = prelude_blocks + 1;
                let k = n_blocks.saturating_sub(first);
                if k > 0 {
                    c.copy_rows(
                        &pass,
                        (&b.states, (first * h * 2) as u64, ss),
                        (&b.bank, 0, ss),
                        k * h,
                        n,
                    )?;
                }
                c.copy_rows(
                    &pass,
                    (&b.cur, 0, hs),
                    (&b.bank, (k * h * 2) as u64, ss),
                    h,
                    n,
                )?;
                c.copy(&pass, (&b.prelude, 0), (&b.cur, 0), n * h)?;
                n_blocks = prelude_blocks;
                bank_states = k + 1;
            }
            for ip in core_start..core_end {
                let i = ip - core_start;
                let slot = if p == 0 {
                    ip
                } else {
                    layers + (p - 1) * core + i
                };
                if p > 0 {
                    gpu::encode_inkling2_depth_mix_rows(
                        c.context,
                        &pass,
                        (&b.bank, 0),
                        ss,
                        (&st.msg_queries[i], 0),
                        (&b.msgs, 0),
                        (4 * h) as u32,
                        bank_states as u32,
                        4,
                        hs,
                        n as u32,
                    )
                    .map_err(RealForwardError::Gpu)?;
                }
                tap(&mut c, visit)?;
                apply(&mut c, st, &mut n_blocks, slot, ip, p > 0)?;
                visit += 1;
            }
        }
        for ip in core_end..layers {
            tap(&mut c, visit)?;
            apply(&mut c, st, &mut n_blocks, ip, ip, false)?;
            visit += 1;
        }
        // The single-token flow keeps the last pass's bank count on the state;
        // it is rebuilt every token, so mirroring it keeps the two identical.
        st.bank_states = bank_states;

        let q = view(c.weights, c.index, "model.llm.attn_res.output_query")?;
        c.mix_rows(
            &pass,
            &b.states,
            n_blocks,
            (&b.cur, 0, hs),
            q,
            1,
            (&b.m, 0, hs),
            n,
        )?;
        c.norm_rows(
            &pass,
            (&b.m, 0, hs),
            "model.llm.norm.weight",
            (&b.mn, 0, hs),
            n,
        )?;
        c.copy_rows(
            &pass,
            (&b.mn, 0, hs),
            (&b.taps, (DSPARK_TAPS_MIDDLE * h * 2) as u64, ts),
            h,
            n,
        )?;
        c.gemv_rows(
            &pass,
            &[(&b.mn, 0, hs)],
            &[(
                "model.llm.unembed.weight",
                vocab,
                h,
                0,
                (&b.logits, 0, vocab as u32),
            )],
            n,
        )?;
        gpu::encode_logit_softcap(
            c.context,
            &pass,
            (&b.logits, 0),
            arch.final_logit_softcap as f32,
            (n * vocab) as u32,
        )
        .map_err(RealForwardError::Gpu)?;
        pass.commit_and_wait();
        for _ in 0..n {
            self.kv.advance();
        }
        gpu::read_buffer_f16_into(&b.logits, 0, logits);
        for row in logits.chunks_exact_mut(vocab) {
            for l in &mut row[cfg.unpadded_vocab as usize..] {
                *l = super::PAD_LOGIT;
            }
        }
        Ok(())
    }

    /// Rolls the Inkling conv tails back to the state after `keep` rows of
    /// the last verify (0 = its start) and the cursor to `base + keep`.
    pub(crate) fn inkling2_rollback_verify(&mut self, keep: usize) -> Result<(), String> {
        let st = self
            .real_inkling2
            .as_mut()
            .ok_or("inkling2 rollback without an Inkling state")?;
        let rec = st
            .verify
            .ok_or("inkling2 rollback needs a verify record; the last pass was not a verify")?;
        if keep > rec.rows {
            return Err(format!(
                "inkling2 rollback keeps {keep} rows of a {}-row verify",
                rec.rows
            ));
        }
        if self.kv.position() != rec.base + rec.rows {
            return Err(format!(
                "inkling2 rollback: cursor at {}, the verify ended at {}",
                self.kv.position(),
                rec.base + rec.rows
            ));
        }
        if keep < rec.rows {
            let b = st.batch.as_ref().expect("a verify allocated the scratch");
            let taps = self.arch.short_conv_kernel_size as usize - 1;
            let h = self.arch.hidden_size as usize;
            let kvd = (self.arch.num_kv_heads * self.arch.head_dim) as usize;
            let pass = self.context.begin_pass_labeled("inkling2 tail restore");
            for (slot, snaps) in b.snaps.iter().enumerate() {
                let s = &st.slots[slot];
                for (snap, (tail, ch)) in snaps.iter().zip([
                    (&s.tail_k, kvd),
                    (&s.tail_v, kvd),
                    (&s.tail_o, h),
                    (&s.tail_f, h),
                ]) {
                    let len = taps * ch;
                    gpu::encode_inkling2_copy_half(
                        &mut self.context,
                        &pass,
                        (snap, (keep * len * 2) as u64),
                        (tail, 0),
                        len as u32,
                    )
                    .map_err(|e| e.to_string())?;
                }
            }
            pass.commit_and_wait();
            self.kv.rewind_cursor(rec.rows - keep);
        }
        st.verify = None;
        if let Some(d) = st.dspark.as_mut() {
            if let super::dspark::TapSource::Verify { base, .. } = d.source {
                d.source = super::dspark::TapSource::Verify { base, rows: keep };
            }
        }
        Ok(())
    }
}

/// [`super::layer_attention`] over `n` rows at positions `base ..`.
#[allow(clippy::too_many_arguments)]
fn layer_attention_rows(
    c: &mut Ctx<'_>,
    pass: &gpu::PassEncoder,
    st: &RealInkling2State,
    b: &BatchScratch,
    arch: &model_io::ArchConfig,
    cfg: &model_io::inkling2_config::Inkling2Config,
    n_blocks: &mut usize,
    slot: usize,
    ip: usize,
    msgs: bool,
    base: usize,
    n: usize,
) -> Result<(), RealForwardError> {
    let g = RealForwardError::Gpu;
    let h = c.h;
    let hs = h as u32;
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
    let (s4, s3) = ((4 * h) as u32, (3 * h) as u32);

    let n_q = if full { 3 } else { 1 };
    // Each attention input as rows: (buffer, offset, stride).
    let mut inputs: [Rows<'_>; 3] = if ip > 0 {
        let q = view(
            c.weights,
            c.index,
            &format!("model.llm.attn_res.attn_queries.{}", ip - 1),
        )?;
        c.mix_rows(
            pass,
            &b.states,
            *n_blocks,
            (&b.cur, 0, hs),
            q,
            n_q,
            (&b.mixed, 0, s4),
            n,
        )?;
        [0, 1, 2].map(|j| (&b.mixed, ((if full { j } else { 0 }) * h * 2) as u64, s4))
    } else {
        [(&b.cur, 0, hs); 3]
    };
    if msgs {
        for (j, ctrl) in ["attn", "k", "v"].iter().enumerate() {
            let out = (&b.inputs, (j * h * 2) as u64, s3);
            c.control_rows(
                pass,
                b,
                cfg.gate_rank as usize,
                ip,
                ctrl,
                inputs[j],
                (&b.msgs, (j * h * 2) as u64, s4),
                out,
                n,
            )?;
            inputs[j] = out;
        }
    }
    let normed: [Rows<'_>; 3] = if !full && !msgs {
        c.norm_rows(
            pass,
            inputs[0],
            &name(ip, "attn_norm.weight"),
            (&b.normed, 0, s3),
            n,
        )?;
        [(&b.normed, 0, s3); 3]
    } else {
        for (j, which) in ["q", "k", "v"].iter().enumerate() {
            let w = if full {
                name(ip, &format!("{which}_input_norm.weight"))
            } else {
                name(ip, "attn_norm.weight")
            };
            c.norm_rows(pass, inputs[j], &w, (&b.normed, (j * h * 2) as u64, s3), n)?;
        }
        [0, 1, 2].map(|j| (&b.normed, (j * h * 2) as u64, s3))
    };
    let a = |s: &str| name(ip, &format!("attn.{s}"));
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
    let (qd, kt) = ((heads * hd) as u32, (kvh * taps as usize) as u32);
    let mut segs: Vec<(&str, usize, usize, usize, Rows<'_>)> = vec![
        (&names[0], heads * hd, h, 0, (&b.q, 0, qd)),
        (&names[1], kvd, h, 1, (&b.k_pre, 0, kvd as u32)),
        (&names[2], kvd, h, 2, (&b.v_pre, 0, kvd as u32)),
        (
            &names[3],
            heads * d_rel,
            h,
            0,
            (&b.r, 0, (heads * d_rel) as u32),
        ),
        (&names[4], heads, h, 0, (&b.sinks, 0, heads as u32)),
        (&names[5], heads * hd, h, 0, (&b.gate_attn, 0, qd)),
    ];
    if full {
        segs.push((&names[6], kt as usize, h, 1, (&b.dk, 0, kt)));
        segs.push((&names[7], kt as usize, h, 2, (&b.dv, 0, kt)));
    }
    c.gemv_rows(pass, &normed, &segs, n)?;

    let s = &st.slots[slot];
    let snaps = &b.snaps[slot];
    let ring = gpu::Inkling2Rows {
        buf: (&s.k, 0),
        stride: kvd as u32,
        ring_base: (base % s.rows) as u32,
        ring: s.rows as u32,
    };
    let vring = gpu::Inkling2Rows {
        buf: (&s.v, 0),
        ..ring
    };
    let kw = view(c.weights, c.index, &a("k_sconv.weight"))?;
    let vw = view(c.weights, c.index, &a("v_sconv.weight"))?;
    gpu::encode_inkling2_short_conv_rows(
        c.context,
        pass,
        (&b.k_pre, 0),
        kvd as u32,
        (&s.tail_k, 0),
        kw,
        full.then_some(((&b.dk, 0u64), kt, kvh as u32)),
        ring,
        (&snaps[0], 0),
        kvd as u32,
        taps,
        n as u32,
    )
    .map_err(g)?;
    gpu::encode_inkling2_short_conv_rows(
        c.context,
        pass,
        (&b.v_pre, 0),
        kvd as u32,
        (&s.tail_v, 0),
        vw,
        full.then_some(((&b.dv, 0u64), kt, kvh as u32)),
        vring,
        (&snaps[1], 0),
        kvd as u32,
        taps,
        n as u32,
    )
    .map_err(g)?;

    let tau: Vec<f32> = (0..n)
        .map(|r| {
            if full {
                compute::inkling2::log_scale_tau(base + r, cfg.log_floor, cfg.log_alpha)
            } else {
                1.0
            }
        })
        .collect();
    let ones = vec![1.0f32; n];
    let qs = view(c.weights, c.index, &a("q_l2_clip.scale"))?;
    let rs = view(c.weights, c.index, &a("rel_l2_clip.scale"))?;
    gpu::encode_inkling2_l2_clip_rows(
        c.context,
        pass,
        gpu::Inkling2Rows::plain((&b.q, 0), qd),
        Some((qs, hd as u32)),
        heads as u32,
        hd as u32,
        cfg.qk_clip,
        &tau,
    )
    .map_err(g)?;
    gpu::encode_inkling2_l2_clip_rows(
        c.context,
        pass,
        ring,
        None,
        kvh as u32,
        hd as u32,
        cfg.qk_clip,
        &ones,
    )
    .map_err(g)?;
    gpu::encode_inkling2_rms_rows(
        c.context,
        pass,
        vring,
        None,
        vring,
        hd as u32,
        kvh as u32,
        cfg.rms_eps,
        n as u32,
    )
    .map_err(g)?;
    gpu::encode_inkling2_l2_clip_rows(
        c.context,
        pass,
        gpu::Inkling2Rows::plain((&b.r, 0), (heads * d_rel) as u32),
        Some((rs, d_rel as u32)),
        heads as u32,
        d_rel as u32,
        cfg.rel_clip,
        &tau,
    )
    .map_err(g)?;

    let proj = view(c.weights, c.index, &a("rel_logits_proj.proj"))?;
    gpu::encode_inkling2_rel_table_rows(
        c.context,
        pass,
        (&b.r, 0),
        (heads * d_rel) as u32,
        proj,
        (&b.table, 0),
        heads as u32,
        d_rel as u32,
        extent,
        n as u32,
    )
    .map_err(g)?;
    let shape = gpu::Inkling2AttnShape {
        heads: heads as u32,
        kv_heads: kvh as u32,
        head_dim: hd as u32,
        extent,
        ring: s.rows as u32,
    };
    gpu::encode_inkling2_attention_rows(
        c.context,
        pass,
        (&b.q, 0),
        (&s.k, 0),
        (&s.v, 0),
        (&b.table, 0),
        (&b.sinks, 0),
        (&b.attn_o, 0),
        if full { 0 } else { arch.sliding_window as u32 },
        base as u32,
        shape,
        n as u32,
    )
    .map_err(g)?;
    gpu::encode_sigmoid_gate_mul(
        c.context,
        pass,
        (&b.attn_o, 0),
        (&b.gate_attn, 0),
        (n * heads * hd) as u32,
    )
    .map_err(g)?;
    c.gemv_rows(
        pass,
        &[(&b.attn_o, 0, qd)],
        &[(&a("wo_ud.weight"), h, heads * hd, 0, (&b.o2, 0, hs))],
        n,
    )?;
    let ow = view(c.weights, c.index, &name(ip, "attn_sconv.weight"))?;
    gpu::encode_inkling2_short_conv_rows(
        c.context,
        pass,
        (&b.o2, 0),
        hs,
        (&s.tail_o, 0),
        ow,
        None,
        gpu::Inkling2Rows::plain((&b.attn_out, 0), hs),
        (&snaps[2], 0),
        hs,
        taps,
        n as u32,
    )
    .map_err(g)?;

    let ss = (MAX_STATES * h) as u32;
    if ip % cfg.res_block as usize == 0 {
        c.copy_rows(
            pass,
            (&b.cur, 0, hs),
            (&b.states, (*n_blocks * h * 2) as u64, ss),
            h,
            n,
        )?;
        *n_blocks += 1;
        c.copy(pass, (&b.attn_out, 0), (&b.cur, 0), n * h)?;
    } else {
        gpu::encode_residual_add(
            c.context,
            pass,
            (&b.cur, 0),
            (&b.attn_out, 0),
            (n * h) as u32,
        )
        .map_err(g)?;
    }

    let q = view(
        c.weights,
        c.index,
        &format!("model.llm.attn_res.mlp_queries.{ip}"),
    )?;
    c.mix_rows(
        pass,
        &b.states,
        *n_blocks,
        (&b.cur, 0, hs),
        q,
        1,
        (&b.m, 0, hs),
        n,
    )?;
    if msgs {
        c.control_rows(
            pass,
            b,
            cfg.gate_rank as usize,
            ip,
            "mlp",
            (&b.m, 0, hs),
            (&b.msgs, (3 * h * 2) as u64, s4),
            (&b.y, 0, hs),
            n,
        )?;
        c.copy(pass, (&b.y, 0), (&b.m, 0), n * h)?;
    }
    c.norm_rows(
        pass,
        (&b.m, 0, hs),
        &name(ip, "mlp_norm.weight"),
        (&b.mn, 0, hs),
        n,
    )?;
    let router = view(c.weights, c.index, &name(ip, "mlp.gate.weight"))?;
    gpu::encode_inkling2_router_rows(
        c.context,
        pass,
        router,
        (&b.mn, 0),
        hs,
        (&b.router_logits, 0),
        arch.num_experts as u32,
        hs,
        n as u32,
    )
    .map_err(g)
}

/// [`super::layer_moe_resident`] over `n` rows.
#[allow(clippy::too_many_arguments)]
fn layer_moe_rows(
    c: &mut Ctx<'_>,
    pass: &gpu::PassEncoder,
    st: &RealInkling2State,
    b: &BatchScratch,
    arch: &model_io::ArchConfig,
    cfg: &model_io::inkling2_config::Inkling2Config,
    slot: usize,
    ip: usize,
    n: usize,
) -> Result<(), RealForwardError> {
    let g = RealForwardError::Gpu;
    let (h, ffn, top_k) = (
        c.h,
        arch.moe_intermediate_size as usize,
        arch.top_k_experts as usize,
    );
    let hs = h as u32;
    if top_k != gpu::INKLING2_ROUTER_TOP_K {
        return Err(RealForwardError::Unsupported(format!(
            "resident experts route top-{} on the GPU; this install selects {top_k}",
            gpu::INKLING2_ROUTER_TOP_K
        )));
    }
    let r = st.resident.as_ref().expect("verify checked resident");
    gpu::encode_inkling2_router_select_rows(
        c.context,
        pass,
        (&b.router_logits, 0),
        (&r.bias[ip], 0),
        (&b.sel, 0),
        (&b.routing_w, 0),
        r.n_experts,
        arch.routed_scaling_factor as f32,
        n as u32,
    )
    .map_err(g)?;
    let layer = gpu::Inkling2ResidentLayer {
        buffer: &r.layers[ip],
        stride: r.strides[ip],
        n_experts: r.n_experts,
        sel: (&b.sel, 0),
    };
    // Native rows only: `native_expert_types` refused every other install at
    // the verify entry.
    let kind = match st.kinds[ip] {
        ExpertKind::Native(t) => t,
        ExpertKind::Gguf { .. } => {
            return Err(RealForwardError::Unsupported(
                "GGUF experts have no batched verify".to_string(),
            ))
        }
    };
    gpu::encode_inkling2_moe_phase1_indexed_rows(
        c.context,
        pass,
        kind,
        &layer,
        &st.offsets[ip],
        (&b.mn, 0),
        hs,
        (&b.acts, 0),
        hs,
        ffn as u32,
        top_k as u32,
        cfg.gate_softcap,
        cfg.up_softcap,
        n as u32,
    )
    .map_err(g)?;
    gpu::encode_inkling2_moe_phase2_indexed_rows(
        c.context,
        pass,
        kind,
        &layer,
        &st.offsets[ip],
        (&b.acts, 0),
        (&b.routing_w, 0),
        (&b.y, 0),
        hs,
        ffn as u32,
        top_k as u32,
        n as u32,
    )
    .map_err(g)?;
    let fw = view(c.weights, c.index, &name(ip, "mlp_sconv.weight"))?;
    let s = &st.slots[slot];
    gpu::encode_inkling2_short_conv_rows(
        c.context,
        pass,
        (&b.y, 0),
        hs,
        (&s.tail_f, 0),
        fw,
        None,
        gpu::Inkling2Rows::plain((&b.y2, 0), hs),
        (&b.snaps[slot][3], 0),
        hs,
        arch.short_conv_kernel_size as u32,
        n as u32,
    )
    .map_err(g)?;
    gpu::encode_residual_add(c.context, pass, (&b.cur, 0), (&b.y2, 0), (n * h) as u32).map_err(g)
}
