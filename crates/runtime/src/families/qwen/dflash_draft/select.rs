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
            // The refusal and the top-K share ONE scan of the row: they were
            // two full passes over 248,320 values, seven times a round.
            let (cand, unary) = match top_k_finite(&row_logits, DFLASH_TOP_K) {
                Ok(found) => found,
                Err(bad) => {
                    return Err(RealForwardError::Unsupported(format!(
                        "the DFlash2 draft pass produced a non-finite logit (row {row}, id \
                         {bad}); the drafter's residual stream has left FP16's range, which \
                         DFLASH_RESIDUAL_SCALE exists to prevent"
                    )));
                }
            };
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

impl RealForwardRunner {
    /// The whole candidate lattice of the LAST draft pass, for tree-verify
    /// research (`crate::DflashLattice`). Read-only: it rereads the mask
    /// rows' logits and projections the selector just walked and changes no
    /// drafter or trunk state. Call it right after `dflash_draft_block` with
    /// the same anchor; the scores are the selector's own, so the lattice's
    /// greedy path is the proposals that call returned.
    pub fn dflash_lattice(
        &self,
        anchor: TokenId,
    ) -> Result<crate::DflashLattice, RealForwardError> {
        let d = self.real_dflash.as_ref().ok_or_else(|| {
            RealForwardError::Unsupported(
                "no DFlash2 state; open the install with a DFlash2 drafter".to_string(),
            )
        })?;
        let (rank, vocab, block) = (d.shape.rank, self.arch.vocab_size as usize, d.block);
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
        let mut candidates: Vec<Vec<TokenId>> = Vec::with_capacity(block);
        let mut edges: Vec<Vec<f32>> = Vec::with_capacity(block);
        for step in 0..block {
            let row = step + 1;
            gpu::read_buffer_f16_into(&d.logits, row * vocab * 2, &mut row_logits);
            let (cand, unary) = top_k_finite(&row_logits, DFLASH_TOP_K).map_err(|bad| {
                RealForwardError::Unsupported(format!(
                    "the DFlash2 draft pass produced a non-finite logit (row {row}, id {bad})"
                ))
            })?;
            let proj = &hproj[row * rank..(row + 1) * rank];
            let succ_rows: Vec<Vec<f32>> = cand
                .iter()
                .map(|&c| read_bf16_row(succ, c))
                .collect::<Result<_, _>>()?;
            // The predecessors: the anchor at depth 0, else every candidate
            // of the depth before.
            let prevs: Vec<TokenId> = if step == 0 {
                vec![anchor]
            } else {
                candidates[step - 1].clone()
            };
            let mut scores = Vec::with_capacity(prevs.len() * cand.len());
            for &prev in &prevs {
                let pred_row = read_bf16_row(pred, prev)?;
                for (c, succ_row) in succ_rows.iter().enumerate() {
                    // The selector's own expression and summation order.
                    let mut dot = 0.0f32;
                    for r in 0..rank {
                        dot += pred_row[r] * proj[r] * succ_row[r];
                    }
                    scores.push(unary[c] + dot);
                }
            }
            candidates.push(cand);
            edges.push(scores);
        }
        Ok(crate::DflashLattice { candidates, edges })
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
#[cfg(test)]
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

/// [`top_k`] plus the non-finite refusal in one scan. `Err(i)` is the FIRST
/// non-finite id, the same one a separate `position` pass would name. A
/// finite row returns exactly what [`top_k`] returns: the compare is the
/// same expression on the same values, and the finite check only reads.
fn top_k_finite(logits: &[LogitValue], k: usize) -> Result<(Vec<TokenId>, Vec<f32>), usize> {
    let mut idx = vec![0usize; k];
    let mut val = vec![f32::NEG_INFINITY; k];
    let mut bad: Option<usize> = None;
    for (i, l) in logits.iter().enumerate() {
        let v = l.to_f32();
        if !v.is_finite() {
            // Keep scanning cheaply is pointless: the row is refused.
            bad = Some(i);
            break;
        }
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
    match bad {
        Some(i) => Err(i),
        None => Ok((idx.into_iter().map(|i| i as TokenId).collect(), val)),
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn row(values: &[f32]) -> Vec<LogitValue> {
        values.iter().map(|&v| LogitValue::from_f32(v)).collect()
    }

    /// A deterministic row with plenty of exact ties, which is what the
    /// lower-id tie-break exists for: f16 has only 2^16 values.
    fn pseudo_random_row(n: usize, seed: u64) -> Vec<LogitValue> {
        let mut x = seed;
        (0..n)
            .map(|_| {
                x = x
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                LogitValue::from_f32(((x >> 40) % 64) as f32 * 0.25 - 8.0)
            })
            .collect()
    }

    #[test]
    fn the_fused_scan_returns_what_the_two_pass_form_returned() {
        for seed in 0..8u64 {
            let r = pseudo_random_row(5000 + seed as usize, seed);
            let fused = top_k_finite(&r, 16).expect("finite row");
            assert_eq!(fused, top_k(&r, 16), "seed {seed}");
        }
    }

    #[test]
    fn ties_keep_the_lower_token_id() {
        let r = row(&[1.0, 3.0, 3.0, 3.0, 0.5, 3.0]);
        let (ids, vals) = top_k_finite(&r, 3).unwrap();
        assert_eq!(ids, vec![1, 2, 3]);
        assert_eq!(vals, vec![3.0, 3.0, 3.0]);
    }

    #[test]
    fn the_first_non_finite_id_is_named() {
        let mut r = pseudo_random_row(1000, 3);
        r[417] = LogitValue::from_f32(f32::NAN);
        r[900] = LogitValue::from_f32(f32::INFINITY);
        assert_eq!(top_k_finite(&r, 16).unwrap_err(), 417);
        let mut r = pseudo_random_row(1000, 3);
        r[0] = LogitValue::from_f32(f32::NEG_INFINITY);
        assert_eq!(top_k_finite(&r, 16).unwrap_err(), 0);
    }
}
