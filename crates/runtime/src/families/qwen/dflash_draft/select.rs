use foundation::{LogitValue, TokenId};
use model_io::ResidentIndex;

use crate::families::qwen::dflash::DFLASH_TOP_K;
use crate::real_forward::{RealForwardError, RealForwardRunner};
use crate::real_forward_utils::norm_view;

const PREDECESSOR_CODEBOOK: &str = "dflash.candidate_selector.predecessor_codebook";
const SUCCESSOR_CODEBOOK: &str = "dflash.candidate_selector.successor_codebook";

impl RealForwardRunner {
    /// The host half of the round: top-16 the mask rows' logits, gather the
    /// codebook rows, walk the greedy path.
    pub(crate) fn dflash_select(
        &self,
        anchor: TokenId,
        block: usize,
        proposals: &mut Vec<TokenId>,
    ) -> Result<(), RealForwardError> {
        let d = self.real_dflash.as_ref().expect("caller checked");
        let (rank, vocab) = (d.shape.rank, self.arch.vocab_size as usize);
        let pred = codebook_view(
            &self.weights,
            &self.index,
            PREDECESSOR_CODEBOOK,
            vocab,
            rank,
        )?;
        let succ = codebook_view(&self.weights, &self.index, SUCCESSOR_CODEBOOK, vocab, rank)?;

        let mut row_logits = vec![LogitValue::from_f32(0.0); vocab];
        let hproj = read_f16_rows(&d.hproj, block + 1, rank);
        let mut prev_token = anchor;
        proposals.reserve_exact(block);
        for step in 0..block {
            let row = step + 1;
            gpu::read_buffer_f16_into(&d.logits, row * vocab * 2, &mut row_logits);
            // A NON-FINITE ROW IS REFUSED, not walked. `top_k` compares
            // `v <= val[k-1]`, and every comparison against NaN is false, so
            // a NaN row is ADMITTED at every candidate; then `score >
            // best_score` is false at every one of them and the walk keeps
            // `cand[0]`, which is token id 0. That is exactly what an FP16
            // overflow in the block forward looked like for the life of
            // this drafter: eight zeros a round, a plausible-looking
            // `loses` table, and no error anywhere. It is also why the
            // bisect probe's rank-of-the-true-token read a PERFECT median 0
            // -- ranking counts `v > target`, which NaN also fails.
            if let Some(bad) = row_logits.iter().position(|v| !v.to_f32().is_finite()) {
                return Err(RealForwardError::Unsupported(format!(
                    "the DFlash2 draft pass produced a non-finite logit (row {row}, id {bad}); \
                     the drafter's residual stream has left FP16's range, which \
                     DFLASH_RESIDUAL_SCALE exists to prevent"
                )));
            }
            let (cand, unary) = top_k(&row_logits, DFLASH_TOP_K);
            // One predecessor row (the previous step's chosen token, or the
            // anchor at step 0) scores against every successor candidate.
            let pred_row = read_bf16_row(pred, prev_token)?;
            let proj = &hproj[row * rank..(row + 1) * rank];
            let mut best = 0usize;
            let mut best_score = f32::NEG_INFINITY;
            for (c, &candidate) in cand.iter().enumerate() {
                let succ_row = read_bf16_row(succ, candidate)?;
                let mut dot = 0.0f32;
                for r in 0..rank {
                    dot += pred_row[r] * proj[r] * succ_row[r];
                }
                let score = unary[c] + dot;
                if score > best_score {
                    best_score = score;
                    best = c;
                }
            }
            prev_token = cand[best];
            proposals.push(prev_token);
        }
        Ok(())
    }
}

/// A resident BF16 codebook validated against the model vocabulary and the
/// selector rank before host gathers can address it by token id.
#[derive(Clone, Copy)]
struct CodebookView<'a> {
    buffer: &'a gpu::MetalBuffer,
    offset: u64,
    rows: usize,
    rank: usize,
}

fn codebook_view<'a>(
    weights: &'a gpu::ResidentGpuWeights,
    index: &ResidentIndex,
    name: &str,
    rows: usize,
    rank: usize,
) -> Result<CodebookView<'a>, RealForwardError> {
    let elements = rows.checked_mul(rank).ok_or_else(|| {
        RealForwardError::Unsupported(format!("{name} element count overflows usize"))
    })?;
    let (buffer, offset) = norm_view(weights, index, name, elements)?;
    Ok(CodebookView {
        buffer,
        offset,
        rows,
        rank,
    })
}

/// The top-K candidates and their logits, by LINEAR SCAN with a running
/// K-list; ties prefer the lower token id (the reference walk's
/// tie-break). Sorting the whole vocab here would repeat
/// `selection::select`'s measured 18.9 ms mistake once per proposal step.
fn top_k(logits: &[LogitValue], k: usize) -> (Vec<TokenId>, Vec<f32>) {
    let mut idx = vec![0usize; k];
    let mut val = vec![f32::NEG_INFINITY; k];
    for (i, l) in logits.iter().enumerate() {
        let v = l.to_f32();
        // Strictly greater only: an equal logit keeps the earlier (lower)
        // token id, which is the tie-break the walk applies.
        if v <= val[k - 1] {
            continue;
        }
        let mut j = k - 1;
        idx[j] = i;
        val[j] = v;
        while j > 0 && (val[j - 1] < val[j] || (val[j - 1] == val[j] && idx[j - 1] > idx[j])) {
            idx.swap(j - 1, j);
            val.swap(j - 1, j);
            j -= 1;
        }
    }
    (idx.into_iter().map(|i| i as TokenId).collect(), val)
}

/// Reads `rows` rows of FP16 values off a GPU buffer into f32.
pub(crate) fn read_f16_rows(buffer: &gpu::MetalBuffer, rows: usize, width: usize) -> Vec<f32> {
    let raw = gpu::read_buffer_bytes(buffer, 0, rows * width * 2);
    raw.chunks_exact(2)
        .map(|c| LogitValue::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect()
}

/// Reads one BF16 codebook row as f32.
fn read_bf16_row(view: CodebookView<'_>, token: TokenId) -> Result<Vec<f32>, RealForwardError> {
    let row = usize::try_from(token).map_err(|_| {
        RealForwardError::Unsupported(format!("negative DFlash2 codebook token id {token}"))
    })?;
    if row >= view.rows {
        return Err(RealForwardError::Unsupported(format!(
            "DFlash2 codebook token id {token} is outside {} rows",
            view.rows
        )));
    }
    let row_bytes = view.rank.checked_mul(2).ok_or_else(|| {
        RealForwardError::Unsupported("DFlash2 codebook row size overflows usize".to_string())
    })?;
    let relative = row.checked_mul(row_bytes).ok_or_else(|| {
        RealForwardError::Unsupported("DFlash2 codebook row offset overflows usize".to_string())
    })?;
    let base = usize::try_from(view.offset).map_err(|_| {
        RealForwardError::Unsupported("DFlash2 codebook offset exceeds usize".to_string())
    })?;
    let offset = base.checked_add(relative).ok_or_else(|| {
        RealForwardError::Unsupported("DFlash2 codebook row address overflows usize".to_string())
    })?;
    let raw = gpu::read_buffer_bytes(view.buffer, offset, row_bytes);
    Ok(raw
        .chunks_exact(2)
        .map(|c| compute::bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect())
}
