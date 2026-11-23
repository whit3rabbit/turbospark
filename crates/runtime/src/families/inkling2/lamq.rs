//! LookAtMyQuant (LAMQ) installs of Inkling 2 (`docs/LAMQ.md`): the coded
//! linears, the bucketed embedding and head, and the LAMQ expert pair, on the
//! same flow as the HF and MLX installs.
//!
//! Every `DTYPE_LAMQ` (36) entry is parsed and checked once at open (its aux
//! header, sizes, alignment and table), and the GEMV paths bind codes, scales,
//! signs, levels and LUTs in place in the mapped resident buffer. Scratch
//! buffers are reused across consecutive dispatches, which is safe because
//! every Inkling pass is a serial compute encoder.

use std::collections::{BTreeSet, HashMap};

use compute::lamq::{parse_matrix_aux, LamqAuxHeader, LamqCode, LamqKind, LAMQ_AUX_HEADER_BYTES};
use model_io::ResidentIndex;

use crate::real_forward::RealForwardError;
use crate::real_forward_utils::entry;

/// `repack::resident_writer::DTYPE_LAMQ`.
pub(crate) const TAG_LAMQ: u8 = 36;
const TAG_FP16: u8 = 2;

type Buf<'a> = (&'a gpu::MetalBuffer, u64);
/// A row-strided view: buffer, byte offset of row 0, row stride in halfs.
type Rows<'a> = (&'a gpu::MetalBuffer, u64, u32);
/// Rows one multi-row prefill call takes (the panel, `PANEL_MAX_ROWS`).
const ROWS_MAX: u64 = gpu::INKLING2_BATCH_MAX_ROWS as u64;

fn bad(msg: String) -> RealForwardError {
    RealForwardError::Unsupported(msg)
}

/// One coded matrix: GPU offsets into the resident buffer.
#[derive(Clone, Debug)]
pub(crate) struct LamqMat {
    pub(crate) code: LamqCode,
    pub(crate) rows: u32,
    pub(crate) cols: u32,
    codes: u64,
    scales: u64,
    signs_in: u64,
    signs_out: Option<u64>,
    /// The shared FP16 LUT (trellis) or the matrix's own F32 levels.
    table: u64,
}

/// Expert-side state of a LAMQ install.
pub(crate) struct LamqExpertRt {
    /// Per physical layer: the blob sub-offsets. Uniform within a layer,
    /// different between layers (each layer pads to its own widest format).
    pub(crate) offsets: Vec<gpu::LamqExpertOffsets>,
    /// Per physical layer: GPU offsets of the gate-up (hidden) and down (ffn)
    /// sign vectors.
    signs: Vec<(u64, u64)>,
    luts: gpu::MetalBuffer,
    x_rot: gpu::MetalBuffer,
    acts: gpu::MetalBuffer,
    act_rot: gpu::MetalBuffer,
}

pub(crate) struct LamqRuntime {
    mats: HashMap<String, LamqMat>,
    factors: HashMap<u32, gpu::LamqRhtFactor>,
    embed: Vec<LamqMat>,
    /// Per token id: (bucket, row).
    embed_loc: Vec<(u16, u32)>,
    /// Head buckets with their first packed position.
    head: Vec<(u32, LamqMat)>,
    head_order: u64,
    head_count: u32,
    x_rot: gpu::MetalBuffer,
    /// One rotated input per slot of a batched group (`gemv_batch`).
    x_rot_batch: gpu::MetalBuffer,
    y32: gpu::MetalBuffer,
    head32: gpu::MetalBuffer,
    /// Multi-row scratch (`Self::gemv_rows`): packed FP16
    /// inputs and outputs, rotated F32 inputs, F32 outputs before their
    /// rotation, each `ROWS_MAX x widest`.
    x_pack: gpu::MetalBuffer,
    x_rot_rows: gpu::MetalBuffer,
    y32_rows: gpu::MetalBuffer,
    y_pack: gpu::MetalBuffer,
    pub(crate) experts: Option<LamqExpertRt>,
}

/// The aux (bias) bytes of a LAMQ entry, from the mapped resident data.
fn aux_bytes<'a>(
    weights: &'a gpu::ResidentGpuWeights,
    index: &ResidentIndex,
    e: &model_io::ResidentIndexEntry,
) -> Result<&'a [u8], RealForwardError> {
    let local = e
        .bias_offset
        .checked_sub(index.header.index_size)
        .ok_or_else(|| bad(format!("{}: aux before the resident region", e.name)))?
        as usize;
    weights
        .data()
        .get(local..local + e.bias_size as usize)
        .ok_or_else(|| bad(format!("{}: aux past the resident region", e.name)))
}

fn gpu_off(weights: &gpu::ResidentGpuWeights, index: &ResidentIndex, file_offset: u64) -> u64 {
    weights.gpu_offset(file_offset - index.header.index_size)
}

fn parse_mat(
    weights: &gpu::ResidentGpuWeights,
    index: &ResidentIndex,
    name: &str,
) -> Result<LamqMat, RealForwardError> {
    let e = entry(index, name)?;
    if e.dtype != TAG_LAMQ {
        return Err(bad(format!(
            "{name}: expected a LAMQ entry, found dtype {}",
            e.dtype
        )));
    }
    let aux =
        parse_matrix_aux(aux_bytes(weights, index, e)?).map_err(|m| bad(format!("{name}: {m}")))?;
    let code = aux.header.code.expect("a matrix has a code");
    let (rows, cols) = (e.shape.0, e.shape.1);
    let rb = code
        .row_bytes(cols as usize)
        .ok_or_else(|| bad(format!("{name}: {} cannot hold width {cols}", code.name())))?;
    if (aux.header.m, aux.header.n) != (rows, cols)
        || e.size_bytes as usize != rows as usize * rb
        || e.scale_size as usize != rows as usize * 2
    {
        return Err(bad(format!(
            "{name}: aux {}x{}, entry {rows}x{cols} with {} code and {} scale bytes",
            aux.header.m, aux.header.n, e.size_bytes, e.scale_size
        )));
    }
    let bias = gpu_off(weights, index, e.bias_offset);
    let signs_in = bias + LAMQ_AUX_HEADER_BYTES as u64;
    let signs_out = aux.header.out_rot.then_some(signs_in + u64::from(cols));
    let table = if code.is_trellis() {
        let lut = format!("lamq.lut.{}", code.name());
        let l = entry(index, &lut)?;
        if l.dtype != TAG_FP16 || l.size_bytes as usize != code.table_rows() * code.g() * 2 {
            return Err(bad(format!(
                "{lut}: not an FP16 table of {} values",
                code.table_rows() * code.g()
            )));
        }
        gpu_off(weights, index, l.file_offset)
    } else {
        let at = (LAMQ_AUX_HEADER_BYTES
            + cols as usize
            + if aux.header.out_rot { rows as usize } else { 0 })
        .div_ceil(4)
            * 4;
        bias + at as u64
    };
    let m = LamqMat {
        code,
        rows,
        cols,
        codes: gpu_off(weights, index, e.file_offset),
        scales: gpu_off(weights, index, e.scale_offset),
        signs_in,
        signs_out,
        table,
    };
    // The kernels read code words as u32, scales and LUTs as f16, levels f32.
    if m.codes % 4 != 0 || m.scales % 2 != 0 || m.table % if code.is_trellis() { 2 } else { 4 } != 0
    {
        return Err(bad(format!("{name}: misaligned in the resident buffer")));
    }
    Ok(m)
}

/// A bucket-order entry: its file offset and the token ids it holds.
fn order_entry(
    weights: &gpu::ResidentGpuWeights,
    index: &ResidentIndex,
    name: &str,
) -> Result<(u64, Vec<i32>), RealForwardError> {
    let e = entry(index, name)?;
    let h = LamqAuxHeader::parse(aux_bytes(weights, index, e)?)
        .map_err(|m| bad(format!("{name}: {m}")))?;
    if e.dtype != TAG_LAMQ || h.kind != LamqKind::Order || e.size_bytes != u64::from(h.n) * 4 {
        return Err(bad(format!("{name}: not a LAMQ bucket order")));
    }
    let local = (e.file_offset - index.header.index_size) as usize;
    let ids = weights.data()[local..local + e.size_bytes as usize]
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Ok((e.file_offset, ids))
}

fn buckets(
    weights: &gpu::ResidentGpuWeights,
    index: &ResidentIndex,
    stem: &str,
    hidden: u32,
) -> Result<Vec<LamqMat>, RealForwardError> {
    let mut out = Vec::new();
    while index
        .entries
        .contains_key(&format!("{stem}.b{}", out.len()))
    {
        let m = parse_mat(weights, index, &format!("{stem}.b{}", out.len()))?;
        if m.cols != hidden || m.signs_out.is_some() {
            return Err(bad(format!(
                "{stem}.b{}: a bucket is [rows, {hidden}] with no out_rot",
                out.len()
            )));
        }
        out.push(m);
    }
    if out.is_empty() || out.len() > usize::from(u16::MAX) {
        return Err(bad(format!("{stem}: no LAMQ buckets")));
    }
    Ok(out)
}

impl LamqRuntime {
    /// None for an install that is not LAMQ (its embedding is not tag 36).
    pub(crate) fn build(
        context: &mut gpu::MetalContext,
        weights: &gpu::ResidentGpuWeights,
        index: &ResidentIndex,
        hidden: usize,
        ffn: usize,
        vocab: usize,
        unpadded: usize,
    ) -> Result<Option<Self>, RealForwardError> {
        if entry(index, "model.llm.embed.weight")?.dtype != TAG_LAMQ {
            return Ok(None);
        }
        let h = hidden as u32;
        let mut mats = HashMap::new();
        let mut names: Vec<&String> = index.entries.keys().collect();
        names.sort();
        for n in names {
            let e = &index.entries[n];
            if e.dtype != TAG_LAMQ
                || n.contains(".weight.b")
                || n == "model.llm.embed.weight"
                || n == "model.llm.unembed.weight"
                || n.ends_with(".lamq_signs")
            {
                continue;
            }
            mats.insert(n.clone(), parse_mat(weights, index, n)?);
        }

        // Embedding: token -> (bucket, row) through the order.
        let embed = buckets(weights, index, "model.llm.embed.weight", h)?;
        let (_, order) = order_entry(weights, index, "model.llm.embed.weight")?;
        if order.len() != vocab || embed.iter().map(|b| b.rows as usize).sum::<usize>() != vocab {
            return Err(bad(format!(
                "embedding order or buckets do not cover {vocab} tokens"
            )));
        }
        let mut embed_loc = vec![(u16::MAX, 0u32); vocab];
        let (mut b, mut r) = (0usize, 0u32);
        for &t in &order {
            while r == embed[b].rows {
                b += 1;
                r = 0;
            }
            let slot = usize::try_from(t)
                .ok()
                .and_then(|t| embed_loc.get_mut(t))
                .filter(|s| s.0 == u16::MAX)
                .ok_or_else(|| {
                    bad(format!(
                        "embedding order: token {t} out of range or repeated"
                    ))
                })?;
            *slot = (b as u16, r);
            r += 1;
        }

        // Head: buckets over the unpadded vocab, scattered through its order.
        let head_b = buckets(weights, index, "model.llm.unembed.weight", h)?;
        let (order_at, order) = order_entry(weights, index, "model.llm.unembed.weight")?;
        if order.len() != unpadded
            || head_b.iter().map(|b| b.rows as usize).sum::<usize>() != unpadded
        {
            return Err(bad(format!(
                "head order or buckets do not cover {unpadded} tokens"
            )));
        }
        let mut seen = vec![false; vocab];
        for &t in &order {
            if !usize::try_from(t)
                .ok()
                .filter(|&t| t < vocab)
                .is_some_and(|t| !std::mem::replace(&mut seen[t], true))
            {
                return Err(bad(format!(
                    "head order: token {t} out of range or repeated"
                )));
            }
        }
        let mut lo = 0u32;
        let head = head_b
            .into_iter()
            .map(|m| {
                let at = lo;
                lo += m.rows;
                (at, m)
            })
            .collect();

        let mut widths: BTreeSet<u32> = [h, ffn as u32].into_iter().collect();
        for m in mats.values() {
            widths.insert(m.cols);
            if m.signs_out.is_some() {
                widths.insert(m.rows);
            }
        }
        let mut factors = HashMap::new();
        for w in widths {
            factors.insert(
                w,
                gpu::LamqRhtFactor::new(context, w as usize).map_err(RealForwardError::Gpu)?,
            );
        }
        let widest = *factors.keys().max().unwrap() as u64;
        Ok(Some(Self {
            mats,
            factors,
            embed,
            embed_loc,
            head,
            head_order: gpu_off(weights, index, order_at),
            head_count: unpadded as u32,
            x_rot: context.new_output_buffer(widest * 4),
            x_rot_batch: context.new_output_buffer(gpu::LAMQ_RHT_BATCH_MAX as u64 * widest * 4),
            y32: context.new_output_buffer(widest * 4),
            head32: context.new_output_buffer(unpadded as u64 * 4),
            x_pack: context.new_output_buffer(ROWS_MAX * widest * 2),
            x_rot_rows: context.new_output_buffer(ROWS_MAX * widest * 4),
            y32_rows: context.new_output_buffer(ROWS_MAX * widest * 4),
            y_pack: context.new_output_buffer(ROWS_MAX * widest * 2),
            experts: None,
        }))
    }

    pub(crate) fn has(&self, name: &str) -> bool {
        self.mats.contains_key(name)
    }

    fn view<'a>(&self, wb: &'a gpu::MetalBuffer, m: &LamqMat) -> gpu::LamqMatrixView<'a> {
        gpu::LamqMatrixView {
            code: m.code,
            rows: m.rows,
            cols: m.cols,
            codes: (wb, m.codes),
            scales: (wb, m.scales),
            table: (wb, m.table),
        }
    }

    fn factor(&self, n: u32) -> &gpu::LamqRhtFactor {
        &self.factors[&n]
    }

    /// The LAMQ matrix `name`, checked against the shape the flow wants.
    fn mat(&self, name: &str, rows: usize, cols: usize) -> Result<&LamqMat, RealForwardError> {
        let m = self
            .mats
            .get(name)
            .ok_or_else(|| bad(format!("{name}: not a LAMQ matrix")))?;
        if (m.rows as usize, m.cols as usize) != (rows, cols) {
            return Err(bad(format!(
                "{name}: LAMQ {}x{}, flow wants {rows}x{cols}",
                m.rows, m.cols
            )));
        }
        Ok(m)
    }

    /// `y = W x` for a coded linear: FP16 `x[cols]` in, FP16 `y[rows]` out.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn gemv(
        &self,
        context: &mut gpu::MetalContext,
        pass: &gpu::PassEncoder,
        wb: &gpu::MetalBuffer,
        name: &str,
        rows: usize,
        cols: usize,
        x: Buf<'_>,
        y: Buf<'_>,
    ) -> Result<(), RealForwardError> {
        let m = self.mat(name, rows, cols)?;
        gpu::encode_lamq_rht(
            context,
            pass,
            self.factor(m.cols),
            gpu::LamqRhtDir::In,
            x,
            true,
            (wb, m.signs_in),
            (&self.x_rot, 0),
            false,
            1,
        )
        .map_err(RealForwardError::Gpu)?;
        self.gemv_rotated(context, pass, wb, m, (&self.x_rot, 0), y)
    }

    /// Several coded linears of one width in one rotation dispatch, then
    /// their GEMVs: each `(name, rows, cols, input index into xs, output)`.
    /// The rotations are bit-identical to [`Self::gemv`]'s; they run on
    /// several cores at once instead of one after another. Falls back to one
    /// at a time when the widths differ or the group is too large.
    #[allow(clippy::type_complexity)]
    pub(crate) fn gemv_batch(
        &self,
        context: &mut gpu::MetalContext,
        pass: &gpu::PassEncoder,
        wb: &gpu::MetalBuffer,
        xs: &[Buf<'_>],
        segs: &[(&str, usize, usize, usize, Buf<'_>)],
    ) -> Result<(), RealForwardError> {
        let mats = segs
            .iter()
            .map(|&(name, rows, cols, _, _)| self.mat(name, rows, cols))
            .collect::<Result<Vec<_>, _>>()?;
        let cols = mats.first().map_or(0, |m| m.cols);
        let batchable = mats.len() <= gpu::LAMQ_RHT_BATCH_MAX
            && xs.len() <= 3
            && mats.iter().all(|m| m.cols == cols);
        if !batchable {
            for &(name, rows, c, x, y) in segs {
                self.gemv(context, pass, wb, name, rows, c, xs[x], y)?;
            }
            return Ok(());
        }
        let jobs: Vec<(u32, u64)> = segs
            .iter()
            .zip(&mats)
            .map(|(s, m)| (s.3 as u32, m.signs_in))
            .collect();
        gpu::encode_lamq_rht_in_batch(
            context,
            pass,
            self.factor(cols),
            xs,
            wb,
            &jobs,
            (&self.x_rot_batch, 0),
        )
        .map_err(RealForwardError::Gpu)?;
        for (i, (seg, m)) in segs.iter().zip(&mats).enumerate() {
            let slot = i as u64 * u64::from(cols) * 4;
            self.gemv_rotated(context, pass, wb, m, (&self.x_rot_batch, slot), seg.4)?;
        }
        Ok(())
    }

    /// One row of `m` over a rotated F32 input: trellis codes through the
    /// multi-row kernel at one row, so decode and the multi-row passes share
    /// one kernel's bits (`lamq_gemv_rows`, equal speed to `lamq_gemv` at one
    /// row); level codes through `lamq_gemv`, which they keep everywhere.
    #[allow(clippy::too_many_arguments)]
    fn gemv_one(
        &self,
        context: &mut gpu::MetalContext,
        pass: &gpu::PassEncoder,
        wb: &gpu::MetalBuffer,
        m: &LamqMat,
        x_rot: Buf<'_>,
        y: Buf<'_>,
        y_half: bool,
    ) -> Result<(), gpu::GpuError> {
        let view = self.view(wb, m);
        if m.code.is_trellis() {
            gpu::encode_lamq_gemv_rows(context, pass, &view, x_rot, m.cols, y, m.rows, 1, y_half)
        } else {
            gpu::encode_lamq_gemv(context, pass, &view, x_rot, y, y_half)
        }
    }

    /// Whether `name` is a trellis-coded matrix, the codes the multi-row
    /// GEMV serves (level codes stay one row at a time: the rows kernel is
    /// slower on them, `lamq_parity.rs` `gemv_rows_speed`).
    pub(crate) fn rows_capable(&self, name: &str) -> bool {
        self.mats.get(name).is_some_and(|m| m.code.is_trellis())
    }

    /// `y = W x` for `n` token rows of one trellis-coded linear (speculative
    /// verify and chunked prefill): the same rotations as decode (a counted
    /// rotation is bit-identical per vector) and one `lamq_gemv_rows`
    /// dispatch that decodes each chunk once for all the rows. Decode runs
    /// the same kernel at one row (`Self::gemv_rotated`), and a row's bits do
    /// not depend on the other rows, so every row gets decode's bits.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn gemv_rows(
        &self,
        context: &mut gpu::MetalContext,
        pass: &gpu::PassEncoder,
        wb: &gpu::MetalBuffer,
        name: &str,
        rows: usize,
        cols: usize,
        x: Rows<'_>,
        y: Rows<'_>,
        n: usize,
    ) -> Result<(), RealForwardError> {
        let g = RealForwardError::Gpu;
        let m = self.mat(name, rows, cols)?;
        if !m.code.is_trellis() || n == 0 || n as u64 > ROWS_MAX {
            return Err(bad(format!("{name}: no multi-row LAMQ GEMV for {n} rows")));
        }
        let (rows32, cols32, n32) = (m.rows, m.cols, n as u32);
        // Contiguous FP16 input rows for the counted rotation.
        let xin = if x.2 == cols32 {
            (x.0, x.1)
        } else {
            gpu::encode_inkling2_copy_rows(
                context,
                pass,
                (x.0, x.1),
                x.2,
                (&self.x_pack, 0),
                cols32,
                cols32,
                n32,
            )
            .map_err(g)?;
            (&self.x_pack, 0)
        };
        gpu::encode_lamq_rht(
            context,
            pass,
            self.factor(cols32),
            gpu::LamqRhtDir::In,
            xin,
            true,
            (wb, m.signs_in),
            (&self.x_rot_rows, 0),
            false,
            n32,
        )
        .map_err(g)?;
        let view = self.view(wb, m);
        match m.signs_out {
            None => gpu::encode_lamq_gemv_rows(
                context,
                pass,
                &view,
                (&self.x_rot_rows, 0),
                cols32,
                (y.0, y.1),
                y.2,
                n32,
                true,
            )
            .map_err(g),
            Some(so) => {
                gpu::encode_lamq_gemv_rows(
                    context,
                    pass,
                    &view,
                    (&self.x_rot_rows, 0),
                    cols32,
                    (&self.y32_rows, 0),
                    rows32,
                    n32,
                    false,
                )
                .map_err(g)?;
                let direct = y.2 == rows32;
                let out = if direct {
                    (y.0, y.1)
                } else {
                    (&self.y_pack, 0)
                };
                gpu::encode_lamq_rht(
                    context,
                    pass,
                    self.factor(rows32),
                    gpu::LamqRhtDir::Out,
                    (&self.y32_rows, 0),
                    false,
                    (wb, so),
                    out,
                    true,
                    n32,
                )
                .map_err(g)?;
                if !direct {
                    gpu::encode_inkling2_copy_rows(
                        context,
                        pass,
                        (&self.y_pack, 0),
                        rows32,
                        (y.0, y.1),
                        y.2,
                        rows32,
                        n32,
                    )
                    .map_err(g)?;
                }
                Ok(())
            }
        }
    }

    /// The GEMV of an already input-rotated F32 `x_rot`, plus the output
    /// rotation when the matrix has one.
    fn gemv_rotated(
        &self,
        context: &mut gpu::MetalContext,
        pass: &gpu::PassEncoder,
        wb: &gpu::MetalBuffer,
        m: &LamqMat,
        x_rot: Buf<'_>,
        y: Buf<'_>,
    ) -> Result<(), RealForwardError> {
        let g = RealForwardError::Gpu;
        match m.signs_out {
            None => self
                .gemv_one(context, pass, wb, m, x_rot, y, true)
                .map_err(g),
            Some(so) => {
                self.gemv_one(context, pass, wb, m, x_rot, (&self.y32, 0), false)
                    .map_err(g)?;
                gpu::encode_lamq_rht(
                    context,
                    pass,
                    self.factor(m.rows),
                    gpu::LamqRhtDir::Out,
                    (&self.y32, 0),
                    false,
                    (wb, so),
                    y,
                    true,
                    1,
                )
                .map_err(g)
            }
        }
    }

    /// The embedding row of `token` into FP16 `y[hidden]`.
    pub(crate) fn embed(
        &self,
        context: &mut gpu::MetalContext,
        pass: &gpu::PassEncoder,
        wb: &gpu::MetalBuffer,
        token: u32,
        y: Buf<'_>,
    ) -> Result<(), RealForwardError> {
        let g = RealForwardError::Gpu;
        let &(b, r) = self
            .embed_loc
            .get(token as usize)
            .ok_or_else(|| bad(format!("token {token} past the embedding")))?;
        let m = &self.embed[usize::from(b)];
        gpu::encode_lamq_decode_row(context, pass, &self.view(wb, m), r, (&self.y32, 0))
            .map_err(g)?;
        gpu::encode_lamq_rht(
            context,
            pass,
            self.factor(m.cols),
            gpu::LamqRhtDir::Out,
            (&self.y32, 0),
            false,
            (wb, m.signs_in),
            y,
            true,
            1,
        )
        .map_err(g)
    }

    /// The head: every bucket's logits for FP16 `x[hidden]`, scattered to
    /// token order into FP16 `logits` (rows past the unpadded vocab are not
    /// written; the caller masks them).
    pub(crate) fn head(
        &self,
        context: &mut gpu::MetalContext,
        pass: &gpu::PassEncoder,
        wb: &gpu::MetalBuffer,
        x: Buf<'_>,
        logits: Buf<'_>,
    ) -> Result<(), RealForwardError> {
        let g = RealForwardError::Gpu;
        for (lo, m) in &self.head {
            gpu::encode_lamq_rht(
                context,
                pass,
                self.factor(m.cols),
                gpu::LamqRhtDir::In,
                x,
                true,
                (wb, m.signs_in),
                (&self.x_rot, 0),
                false,
                1,
            )
            .map_err(g)?;
            gpu::encode_lamq_gemv(
                context,
                pass,
                &self.view(wb, m),
                (&self.x_rot, 0),
                (&self.head32, u64::from(*lo) * 4),
                false,
            )
            .map_err(g)?;
        }
        gpu::encode_lamq_scatter(
            context,
            pass,
            (&self.head32, 0),
            (wb, self.head_order),
            logits,
            self.head_count,
        )
        .map_err(g)
    }

    /// Reads and checks every expert's format record, and builds the expert
    /// state. `layout` is the install's packed layout.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn attach_experts(
        &mut self,
        context: &mut gpu::MetalContext,
        weights: &gpu::ResidentGpuWeights,
        index: &ResidentIndex,
        install_dir: &std::path::Path,
        layout: &model_io::PackedExpertsLayout,
        hidden: usize,
        ffn: usize,
        luts: &[(LamqCode, Vec<u16>)],
    ) -> Result<(), RealForwardError> {
        use std::io::{Read, Seek, SeekFrom};
        let sub = |s: &model_io::SubTensorEntry| {
            u32::try_from(s.offset).map_err(|_| bad("expert sub-tensor offset too large".into()))
        };
        fn get<'e>(
            e: &'e model_io::ExpertEntry,
            role: &str,
        ) -> Result<&'e model_io::SubTensorEntry, RealForwardError> {
            e.sub_tensors
                .get(role)
                .ok_or_else(|| bad(format!("LAMQ expert blob has no {role}")))
        }
        let of = |e: &model_io::ExpertEntry| -> Result<gpu::LamqExpertOffsets, RealForwardError> {
            Ok(gpu::LamqExpertOffsets {
                fmt: sub(get(e, "lamq_fmt")?)?,
                gu: sub(get(e, "gate_up")?)?,
                gu_s: sub(get(e, "gate_up_scales")?)?,
                dn: sub(get(e, "down")?)?,
                dn_s: sub(get(e, "down_scales")?)?,
            })
        };
        let mut signs = Vec::with_capacity(layout.layers.len());
        let mut per_layer = Vec::with_capacity(layout.layers.len());
        for (l, ll) in layout.layers.iter().enumerate() {
            let offsets = of(ll
                .experts
                .first()
                .ok_or_else(|| bad(format!("layer {l} has no experts")))?)?;
            let mut f = std::fs::File::open(install_dir.join("packed_experts").join(&ll.file))
                .map_err(|e| bad(format!("layer {l} experts: {e}")))?;
            for e in &ll.experts {
                // Uniform sub-offsets: the kernels bind one table per layer.
                for (role, want) in [
                    ("lamq_fmt", offsets.fmt),
                    ("gate_up", offsets.gu),
                    ("gate_up_scales", offsets.gu_s),
                    ("down", offsets.dn),
                    ("down_scales", offsets.dn_s),
                ] {
                    if sub(get(e, role)?)? != want {
                        return Err(bad(format!(
                            "layer {l} expert {}: {role} offset differs",
                            e.expert
                        )));
                    }
                }
                let mut rec = [0u8; 16];
                f.seek(SeekFrom::Start(e.offset + u64::from(offsets.fmt)))
                    .and_then(|_| f.read_exact(&mut rec))
                    .map_err(|err| bad(format!("layer {l} expert {}: {err}", e.expert)))?;
                let word = |i: usize| u32::from_le_bytes(rec[4 * i..4 * i + 4].try_into().unwrap());
                for (side, id, rb, rows, cols, cap) in [
                    (
                        "gate_up",
                        word(0),
                        word(2),
                        2 * ffn,
                        hidden,
                        get(e, "gate_up")?.size,
                    ),
                    ("down", word(1), word(3), hidden, ffn, get(e, "down")?.size),
                ] {
                    let code = u8::try_from(id)
                        .ok()
                        .and_then(LamqCode::from_id)
                        .filter(|c| c.is_trellis())
                        .ok_or_else(|| {
                            bad(format!(
                                "layer {l} expert {} {side}: code id {id} refused",
                                e.expert
                            ))
                        })?;
                    if code.row_bytes(cols) != Some(rb as usize)
                        || (rows * rb as usize) as u64 > cap
                    {
                        return Err(bad(format!(
                            "layer {l} expert {} {side}: row bytes {rb} wrong for {}",
                            e.expert,
                            code.name()
                        )));
                    }
                }
            }
            let s = entry(
                index,
                &format!("model.llm.layers.{l}.mlp.experts.lamq_signs"),
            )?;
            let hdr = LamqAuxHeader::parse(aux_bytes(weights, index, s)?)
                .map_err(|m| bad(format!("layer {l} signs: {m}")))?;
            if s.dtype != TAG_LAMQ
                || hdr.kind != LamqKind::Signs
                || (hdr.n as usize, hdr.m as usize) != (hidden, ffn)
                || s.size_bytes as usize != hidden + ffn
            {
                return Err(bad(format!("layer {l}: expert signs malformed")));
            }
            let local = (s.file_offset - index.header.index_size) as usize;
            if weights.data()[local..local + hidden + ffn]
                .iter()
                .any(|&b| b != 1 && b != 0xFF)
            {
                return Err(bad(format!("layer {l}: an expert sign is not +1/-1")));
            }
            let at = gpu_off(weights, index, s.file_offset);
            signs.push((at, at + hidden as u64));
            per_layer.push(offsets);
        }
        let pack: Vec<(LamqCode, &[u16])> = luts.iter().map(|(c, t)| (*c, t.as_slice())).collect();
        let luts = context
            .new_buffer_with_data(&gpu::lamq_lut_pack(&pack).map_err(RealForwardError::Gpu)?);
        let slots = gpu::INKLING2_LAMQ_MAX_SLOTS as u64;
        self.experts = Some(LamqExpertRt {
            offsets: per_layer,
            signs,
            luts,
            x_rot: context.new_output_buffer(hidden as u64 * 4),
            acts: context.new_output_buffer(slots * ffn as u64 * 4),
            act_rot: context.new_output_buffer(slots * ffn as u64 * 4),
        });
        Ok(())
    }

    /// The routed half for one layer application: rotate the input, phase 1,
    /// rotate the activations per slot, phase 2 into FP16 `y`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn moe(
        &self,
        context: &mut gpu::MetalContext,
        pass: &gpu::PassEncoder,
        wb: &gpu::MetalBuffer,
        experts: &gpu::LamqExperts<'_>,
        layer: usize,
        x: Buf<'_>,
        routing_w: Buf<'_>,
        y: Buf<'_>,
        hidden: usize,
        ffn: usize,
        top_k: usize,
        caps: (f32, f32),
    ) -> Result<(), RealForwardError> {
        let g = RealForwardError::Gpu;
        let ex = self
            .experts
            .as_ref()
            .ok_or_else(|| bad("LAMQ experts not attached".into()))?;
        let (gs, ds) = ex.signs[layer];
        gpu::encode_lamq_rht(
            context,
            pass,
            self.factor(hidden as u32),
            gpu::LamqRhtDir::In,
            x,
            true,
            (wb, gs),
            (&ex.x_rot, 0),
            false,
            1,
        )
        .map_err(g)?;
        gpu::encode_inkling2_lamq_phase1(
            context,
            pass,
            experts,
            &ex.offsets[layer],
            (&ex.luts, 0),
            (&ex.x_rot, 0),
            (&ex.acts, 0),
            hidden as u32,
            ffn as u32,
            top_k as u32,
            caps.0,
            caps.1,
        )
        .map_err(g)?;
        gpu::encode_lamq_rht(
            context,
            pass,
            self.factor(ffn as u32),
            gpu::LamqRhtDir::In,
            (&ex.acts, 0),
            false,
            (wb, ds),
            (&ex.act_rot, 0),
            false,
            top_k as u32,
        )
        .map_err(g)?;
        gpu::encode_inkling2_lamq_phase2(
            context,
            pass,
            experts,
            &ex.offsets[layer],
            (&ex.luts, 0),
            (&ex.act_rot, 0),
            routing_w,
            y,
            hidden as u32,
            ffn as u32,
            top_k as u32,
        )
        .map_err(g)
    }
}

/// The six trellis LUTs of an install, as f16 bits.
pub(crate) fn read_luts(
    weights: &gpu::ResidentGpuWeights,
    index: &ResidentIndex,
) -> Result<Vec<(LamqCode, Vec<u16>)>, RealForwardError> {
    [
        LamqCode::Ht2,
        LamqCode::Ht3,
        LamqCode::Rtr1,
        LamqCode::Rtr2,
        LamqCode::Rtr3,
        LamqCode::Rtr4,
    ]
    .into_iter()
    .map(|c| {
        let n = format!("lamq.lut.{}", c.name());
        let e = entry(index, &n)?;
        let want = c.table_rows() * c.g();
        if e.dtype != TAG_FP16 || e.size_bytes as usize != want * 2 {
            return Err(bad(format!("{n}: not an FP16 table of {want} values")));
        }
        let local = (e.file_offset - index.header.index_size) as usize;
        Ok((
            c,
            weights.data()[local..local + want * 2]
                .chunks_exact(2)
                .map(|b| u16::from_le_bytes([b[0], b[1]]))
                .collect(),
        ))
    })
    .collect()
}
