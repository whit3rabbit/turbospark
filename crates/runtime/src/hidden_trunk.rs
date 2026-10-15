//! A headless `QwenGdnMoe` trunk, driven one input row at a time by the model
//! that owns it (Fish Audio Drama 3's text backbone).
//!
//! Wraps a [`RealForwardRunner`] opened as [`OpenRole::HeadlessTrunk`], so the
//! decode flow, kernels and routed-expert streaming are the qwen36 family's
//! own. What changes is the two ends of a pass:
//!
//! - **In**: a row is either a token id (the table lookup, unchanged) or a
//!   host row of `hidden_size` f32 values, narrowed to f16 and written over
//!   the lookup before the pass is committed. Positions stay plain 1-D rope
//!   positions.
//! - **Out**: the post-final-RMSNorm hidden state, read from `scratch.normed`
//!   after the pass's own wait (`hidden_out`). There is no head.
//!
//! [`HiddenTrunk::prefill`] runs the qwen MoE chunked driver's micro-batches
//! (`families/qwen/moe_prefill.rs`), one blocking router readback per layer
//! per 16 rows instead of per row, and every row's hidden state is
//! byte-identical to stepping it. The per-row loop stays as the reference
//! and the fallback where the driver refuses the install.
//!
//! The KV prefix record is always tainted: a token id does not identify a
//! host row, so prefix reuse must never match one. Facts and gates:
//! `docs/FISH_DRAMA3_PHASE0.md`.

use std::path::Path;

use half::f16;
use model_io::{ArchConfig, ExpertCacheSlots, ExpertResidency, KvQuant};

use crate::real_forward::{RealForwardRunner, RollbackPoint};
use crate::real_forward_layout::DTYPE_RAW_BF16;
use crate::real_forward_open::OpenRole;
use crate::real_forward_types::RealForwardError;
use crate::real_forward_utils::entry;

const EMBED_NAME: &str = "language_model.model.embed_tokens.weight";

/// One trunk input row.
#[derive(Clone, Copy, Debug)]
pub enum TrunkRow<'a> {
    /// A token id, looked up in the trunk's own embedding table.
    Token(u32),
    /// A host-computed input row of `hidden_size` values.
    Embedding(&'a [f32]),
}

/// A restore point taken by [`HiddenTrunk::snapshot`].
///
/// Valid until the next [`HiddenTrunk::reset`]: a reset releases the KV rows
/// the snapshot's prefix lives in, so restoring across one is refused rather
/// than attending over zeroed rows.
pub struct PrefixSnapshot {
    point: RollbackPoint,
    epoch: u64,
}

impl PrefixSnapshot {
    /// The position this snapshot restores to.
    pub fn position(&self) -> usize {
        self.point.position()
    }
}

/// A headless trunk (see the module doc).
pub struct HiddenTrunk {
    runner: RealForwardRunner,
    hidden: usize,
    max_context: usize,
    /// Bumped by every [`Self::reset`]; a snapshot from another epoch is
    /// refused.
    epoch: u64,
    /// The f16 readback buffer, reused across passes.
    readback: Vec<f16>,
    /// Whether [`Self::prefill`] may use the chunked driver.
    chunked: bool,
}

impl HiddenTrunk {
    /// Opens the trunk install in `dir`, which must carry a headless-trunk
    /// marker (`model_io::trunk_role`). Drafters, steering, KV quantization
    /// and parked sessions are off: none has a meaning for a trunk whose
    /// owner drives every row.
    pub fn open(
        dir: &Path,
        expecting: ArchConfig,
        max_context: usize,
        expert_cache_slots: ExpertCacheSlots,
        residency: ExpertResidency,
    ) -> Result<Self, RealForwardError> {
        let runner = RealForwardRunner::open_inner_with_role(
            dir,
            expecting,
            max_context,
            expert_cache_slots,
            None,
            crate::families::qwen::DraftPolicies::off(),
            crate::steering::SteeringPolicy::off(),
            1,
            KvQuant::Off,
            residency,
            OpenRole::HeadlessTrunk,
        )?;
        let hidden = runner.arch.hidden_size as usize;
        Ok(HiddenTrunk {
            runner,
            hidden,
            max_context,
            epoch: 0,
            readback: vec![f16::ZERO; hidden],
            chunked: true,
        })
    }

    /// Whether [`Self::prefill`] runs rows in micro-batches (the default,
    /// where the install allows it) or one [`Self::step`] at a time. The
    /// per-row loop is the reference the chunked driver must match.
    pub fn set_chunked_prefill(&mut self, chunked: bool) {
        self.chunked = chunked;
    }

    /// Whether the next [`Self::prefill`] will run in micro-batches.
    pub fn chunked_prefill(&self) -> bool {
        self.chunked
            && self.runner.qwen_moe_chunked_refusal().is_none()
            && !self.runner.routed_batch_prefill
            && !self.runner.batched_gemv_prefill
    }

    /// The trunk's architecture, as its manifest declares it.
    pub fn arch(&self) -> &ArchConfig {
        &self.runner.arch
    }

    /// Width of an input row and of a hidden state.
    pub fn hidden_size(&self) -> usize {
        self.hidden
    }

    /// The next position a row will be written at.
    pub fn position(&self) -> usize {
        self.runner.kv.position()
    }

    /// Rows this trunk can hold.
    pub fn max_context(&self) -> usize {
        self.max_context
    }

    /// The trunk's own embedding row for `token`, widened to f32.
    ///
    /// Read from the mapped resident table rather than a second copy. Only a
    /// raw BF16 table is supported (the Drama 3 storage), so the f32 values
    /// are exact and narrowing them back to f16 rounds the same way the
    /// GPU's lookup does.
    pub fn embedding_row(&self, token: u32, out: &mut [f32]) -> Result<(), RealForwardError> {
        let hidden = self.hidden;
        if out.len() != hidden {
            return Err(RealForwardError::Unsupported(format!(
                "embedding row needs {hidden} values, got a buffer of {}",
                out.len()
            )));
        }
        let index = &self.runner.index;
        let e = entry(index, EMBED_NAME)?;
        if e.dtype != DTYPE_RAW_BF16 || e.shape.1 as usize != hidden {
            return Err(RealForwardError::Unsupported(format!(
                "{EMBED_NAME}: host rows need a raw BF16 [rows, {hidden}] table, got dtype {} \
                 shape {:?}",
                e.dtype, e.shape
            )));
        }
        if token >= e.shape.0 {
            return Err(RealForwardError::Unsupported(format!(
                "token {token} is outside the {}-row embedding table",
                e.shape.0
            )));
        }
        let row_bytes = hidden * 2;
        let start = (e.file_offset - index.header.index_size) as usize + token as usize * row_bytes;
        let bytes = self
            .runner
            .weights
            .data()
            .get(start..start + row_bytes)
            .ok_or_else(|| {
                RealForwardError::Unsupported(format!(
                    "{EMBED_NAME}: row {token} lies outside the resident region"
                ))
            })?;
        for (value, pair) in out.iter_mut().zip(bytes.chunks_exact(2)) {
            *value = compute::bf16_to_f32(u16::from_le_bytes([pair[0], pair[1]]));
        }
        Ok(())
    }

    /// Feeds one row at [`Self::position`] and writes the post-norm hidden
    /// state into `hidden_out`.
    pub fn step(
        &mut self,
        row: TrunkRow<'_>,
        hidden_out: &mut [f32],
    ) -> Result<(), RealForwardError> {
        let hidden = self.hidden;
        if hidden_out.len() != hidden {
            return Err(RealForwardError::Unsupported(format!(
                "hidden output needs {hidden} values, got a buffer of {}",
                hidden_out.len()
            )));
        }
        let position = self.position();
        // Checked before the pass: the KV cursor's own check is an assert.
        if position >= self.max_context {
            return Err(RealForwardError::Unsupported(format!(
                "the trunk is full: {position} of {} rows used",
                self.max_context
            )));
        }
        let (token, host_embed) = match row {
            TrunkRow::Token(id) => (
                i32::try_from(id).map_err(|_| {
                    RealForwardError::Unsupported(format!("token id {id} does not fit i32"))
                })?,
                None,
            ),
            TrunkRow::Embedding(values) => (0, Some(encode_f16_row(values, hidden)?)),
        };
        let runner = &mut self.runner;
        // A per-row pass overwrites the scratch a recorded verify tape points
        // at, the same reason `LogitProducer::produce` clears it.
        runner.batched_tape = None;
        runner.batched_tape_row0 = None;
        runner.host_embed = host_embed;
        runner.hidden_out = true;
        let result = gpu::autorelease_pool(|| runner.produce_inner(token, position, &mut []));
        runner.host_embed = None;
        runner.hidden_out = false;
        runner.kv_prefix.taint();
        result?;
        gpu::read_buffer_f16_into(&runner.scratch.normed, 0, &mut self.readback);
        for (out, v) in hidden_out.iter_mut().zip(&self.readback) {
            *out = v.to_f32();
        }
        Ok(())
    }

    /// Feeds `rows` in order, writing each row's hidden state into the
    /// matching `hidden_size` slice of `hidden_out`.
    ///
    /// In micro-batches when [`Self::chunked_prefill`], else one
    /// [`Self::step`] per row. A row that fails leaves the trunk part way
    /// through its micro-batch, so after an error the owner restores a
    /// snapshot or resets.
    pub fn prefill(
        &mut self,
        rows: &[TrunkRow<'_>],
        hidden_out: &mut [f32],
    ) -> Result<(), RealForwardError> {
        let hidden = self.hidden;
        if hidden_out.len() != rows.len() * hidden {
            return Err(RealForwardError::Unsupported(format!(
                "{} rows need {} hidden values, got a buffer of {}",
                rows.len(),
                rows.len() * hidden,
                hidden_out.len()
            )));
        }
        if self.position() + rows.len() > self.max_context {
            return Err(RealForwardError::Unsupported(format!(
                "{} rows from position {} exceed the trunk's {} rows",
                rows.len(),
                self.position(),
                self.max_context
            )));
        }
        if rows.is_empty() {
            return Ok(());
        }
        if !self.chunked_prefill() {
            for (row, out) in rows.iter().zip(hidden_out.chunks_exact_mut(hidden)) {
                self.step(*row, out)?;
            }
            return Ok(());
        }
        let mut tokens = Vec::with_capacity(rows.len());
        let mut host = Vec::with_capacity(rows.len());
        for row in rows {
            match *row {
                TrunkRow::Token(id) => {
                    tokens.push(i32::try_from(id).map_err(|_| {
                        RealForwardError::Unsupported(format!("token id {id} does not fit i32"))
                    })?);
                    host.push(None);
                }
                TrunkRow::Embedding(values) => {
                    tokens.push(0);
                    host.push(Some(encode_f16_row(values, hidden)?));
                }
            }
        }
        let host_rows: Vec<Option<&[u8]>> = host.iter().map(|r| r.as_deref()).collect();
        let mut out = vec![f16::ZERO; rows.len() * hidden];
        let position = self.position();
        let runner = &mut self.runner;
        runner.batched_tape = None;
        runner.batched_tape_row0 = None;
        let result = gpu::autorelease_pool(|| {
            runner.prefill_hidden_rows_qwen_moe(&tokens, &host_rows, position, &mut out)
        });
        runner.kv_prefix.taint();
        result?;
        for (dst, v) in hidden_out.iter_mut().zip(&out) {
            *dst = v.to_f32();
        }
        Ok(())
    }

    /// A restore point at the current position.
    ///
    /// Waits for the recurrent-state copy to finish before returning, so a
    /// following [`Self::reset`] (a host write to the live state) cannot race
    /// the copy (`gpu::GdnStateManager::snapshot`'s contract).
    pub fn snapshot(&mut self) -> PrefixSnapshot {
        let point = self.runner.checkpoint();
        self.runner.context.wait_until_idle();
        PrefixSnapshot {
            point,
            epoch: self.epoch,
        }
    }

    /// Returns the trunk to `snapshot`. Rows after it are dropped; the next
    /// row is written at `snapshot.position()`.
    ///
    /// Refused after a [`Self::reset`] (the prefix's KV rows were released)
    /// and when the snapshot is ahead of the current position.
    pub fn restore(&mut self, snapshot: &PrefixSnapshot) -> Result<(), RealForwardError> {
        if snapshot.epoch != self.epoch {
            return Err(RealForwardError::Unsupported(
                "this snapshot was taken before the trunk was reset; its prefix rows are gone"
                    .to_string(),
            ));
        }
        if snapshot.position() > self.position() {
            return Err(RealForwardError::Unsupported(format!(
                "snapshot at {} is ahead of the trunk at {}",
                snapshot.position(),
                self.position()
            )));
        }
        self.runner.rollback(&snapshot.point);
        self.runner.kv_prefix.taint();
        Ok(())
    }

    /// Empties the trunk: position 0, zeroed recurrent state. Every earlier
    /// snapshot becomes unusable.
    pub fn reset(&mut self) {
        crate::producer::LogitProducer::reset(&mut self.runner);
        self.runner.kv_prefix.taint();
        self.epoch += 1;
    }
}

/// Narrows a host row to the f16 bytes the pass reads, refusing values f16
/// cannot carry rather than letting them become infinities in the residual.
fn encode_f16_row(values: &[f32], hidden: usize) -> Result<Vec<u8>, RealForwardError> {
    if values.len() != hidden {
        return Err(RealForwardError::Unsupported(format!(
            "an input row needs {hidden} values, got {}",
            values.len()
        )));
    }
    let mut bytes = Vec::with_capacity(hidden * 2);
    for (i, &v) in values.iter().enumerate() {
        if !v.is_finite() || v.abs() > f16::MAX.to_f32() {
            return Err(RealForwardError::Unsupported(format!(
                "input row value {i} is {v}, outside the f16 range"
            )));
        }
        bytes.extend_from_slice(&f16::from_f32(v).to_le_bytes());
    }
    Ok(bytes)
}
