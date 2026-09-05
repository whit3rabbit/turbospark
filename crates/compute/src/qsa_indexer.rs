//! CPU reference for `qwen4_exp`'s QSA (query-sparse attention) block
//! indexer (`docs/QWEN4_PHASE0.md` section 5). PORT-LOCAL: no decode flow
//! reads `self_attn.indexer.*` yet (`families/qwen4/mod.rs`'s own module
//! doc), and this is groundwork for that, not a wired kernel.
//!
//! **SCOPE, READ BEFORE EXTENDING.** Section 5's pseudocode is:
//!
//! ```text
//! qk        = index_qk_proj(hidden)                    // 2560 -> (4+1)*128 = 640
//! q, k_tok  = split(qk, [4*128, 1*128])
//! q         = rope(q_layernorm(q), current_positions)  // per-head norm, dim 128
//! k_tok     -> cached RAW, per layer, un-normed and un-roped
//!
//! blocks    = consecutive runs of compress_ratio=4 visible tokens
//! pooled    = mean(block keys)  in FP32
//! pooled    = rope(k_layernorm(pooled), position of the block's FIRST token)
//! scores    = relu(q @ pooled^T).sum(over the 4 heads) / sqrt(128)
//! chosen    = topk(scores, min(block_topk, num_complete_blocks))
//! selected  = the 4 tokens of each chosen block, PLUS the ragged tail
//! ```
//!
//! This module covers the BLOCK-POOLING, SCORING and SELECTION steps --
//! everything from `pooled = mean(...)` onward. It deliberately stops short
//! of `index_qk_proj` itself (a plain GEMV, no new kernel) and the two
//! per-head norms and RoPE applications, because those are already-existing
//! primitives in this crate (`rms_norm_centered`, `rope_neox_subdim`) rather
//! than anything new to write -- see "The norm and RoPE convention" below
//! for exactly how to call them, now resolved from source.
//!
//! **`index_kv_heads` is 1**, so one pooled key per block is shared across
//! every one of the 4 query heads (`index_n_heads`) -- a GQA shape, not a
//! per-head pool.
//!
//! ## The norm and RoPE convention (resolved from source, 2026-09-04)
//!
//! Read directly from `mlx-vlm`'s reference (`Blaizzy/mlx-vlm`,
//! `mlx_vlm/models/qwen4_exp/language.py` and `.../qwen3_5/language.py` at
//! `d68a25e71e84`, `mlx_vlm/models/rope_utils.py` at `3db7f1d3402f`), not
//! guessed:
//!
//! - **The norm is [`crate::rms_norm::rms_norm_centered`]**, applied
//!   PER HEAD (once per 128-wide slice, with the same weight vector reused
//!   across heads) -- `docs/QWEN4_PHASE0.md` item 9 already lists the
//!   indexer's `q_layernorm` and `k_layernorm` under "Every
//!   `Qwen4ExpTextRMSNorm` in the model is CENTERED," this module just
//!   connects that fact to the function that computes it.
//! - **The RoPE is [`crate::rope::rope_neox_subdim`] at `rotary_dim = 64`,
//!   `theta = 1e7`** -- the SAME two numbers the trunk's own QSA attention
//!   already dispatches (`families/qwen4/mod.rs`'s "QSA-as-dense-attention"
//!   section), because `Qwen4ExpQSAIndexer.__init__` in the reference is
//!   constructed with `rotary_emb` PASSED IN from the enclosing attention
//!   module (`self.indexer = Qwen4ExpQSAIndexer(config, self.rotary_emb)`)
//!   -- it is the identical Python object, not merely the same convention
//!   by coincidence. That object is `Qwen3_5RotaryEmbedding(int(head_dim *
//!   partial_rotary_factor), base=rope_theta, ..., style="interleaved")`,
//!   and despite mlx-vlm's own confusing name for it ("interleaved" is
//!   THEIR label, unrelated to this port's use of the same word for mrope
//!   section handling in `crates/gpu` Gotcha 11), `style="interleaved"`
//!   resolves to `_apply_interleaved_rotary_pos_emb_axis1`, which slices
//!   `cos`/`sin` to `rotary_dim = 64` elements and calls `rotate_half`
//!   (pair `i` with `i + rotary_dim/2`, the rest of the head passed through
//!   unrotated) -- exactly [`crate::rope::rope_neox_subdim`]'s contract,
//!   not [`crate::rope::rope_paired`]'s. This is independently confirmed by
//!   this port's own already-verified behavior: the trunk dispatches
//!   `rope_neox_subdim` for this exact checkpoint's QSA layers and its
//!   quality gate's frozen perplexity (8.7224) reproduces, so the shared
//!   object's convention is checked against real numbers on this side too,
//!   not only read off the reference.
//! - **Order, per section 5's pseudocode**: raw (un-normed, un-roped) ->
//!   per-head/per-block centered norm -> RoPE, for BOTH the query (at its
//!   own current position) and the pooled key (at its block's FIRST
//!   token's ABSOLUTE position, i.e. `key_start_position + block_index *
//!   compress_ratio` if the visible window does not start at position 0).
//! - **[`rope_neox_subdim`](crate::rope::rope_neox_subdim) takes ONE shared
//!   position for its whole call**, because every other caller in this
//!   crate rotates a batch of rows that share a position (one token's
//!   heads, or one prefill step's rows at one position each via
//!   `num_tokens`). The pooled keys do NOT share a position -- each block's
//!   pool sits at a different absolute token position -- so wiring this
//!   needs ONE CALL PER BLOCK (`num_tokens = 1, num_heads = 1`), not one
//!   batched call across all blocks. This is a real cost the eventual GPU
//!   kernel has to account for (likely one thread/threadgroup per block
//!   computing its own angle from its own position, rather than a
//!   host-precomputed shared cos/sin table).
//!
//! No new kernel-shaped function is added here for norm or RoPE, since
//! both already exist and are correct as called; [`the_full_indexer_chain_composes`]
//! in this module's tests demonstrates the composition end to end against a
//! hand-checked example.

/// Mean-pools consecutive runs of `compress_ratio` RAW (un-normed, un-roped)
/// key rows into one pooled row per complete block, in FP32 as section 5
/// requires ("pooled = mean(block keys) in FP32"). `keys` is
/// `[visible * head_dim]`, oldest token first. Returns one row per COMPLETE
/// block (`visible / compress_ratio`, rounded down) and never touches the
/// ragged tail -- the `visible % compress_ratio` trailing rows that form no
/// complete block, which section 5 says are ALWAYS selected rather than
/// pooled or scored at all.
pub fn pool_blocks_mean(keys: &[f32], head_dim: usize, compress_ratio: usize) -> Vec<f32> {
    assert!(compress_ratio > 0, "compress_ratio must be positive");
    assert!(head_dim > 0, "head_dim must be positive");
    assert_eq!(
        keys.len() % head_dim,
        0,
        "keys must be a whole number of head_dim-wide rows"
    );
    let visible = keys.len() / head_dim;
    let num_blocks = visible / compress_ratio;

    let mut pooled = vec![0.0f32; num_blocks * head_dim];
    let inv = 1.0 / compress_ratio as f32;
    for b in 0..num_blocks {
        let block_base = b * compress_ratio * head_dim;
        let out_base = b * head_dim;
        for t in 0..compress_ratio {
            let row_base = block_base + t * head_dim;
            for d in 0..head_dim {
                pooled[out_base + d] += keys[row_base + d] * inv;
            }
        }
    }
    pooled
}

/// `scores[b] = relu(sum_h(q[h] . pooled[b])) / sqrt(head_dim)`, summed over
/// the `num_heads` query heads against the ONE shared pooled key per block
/// (`index_kv_heads == 1`, so every head reads the same `pooled[b]` row).
/// `q` is `[num_heads * head_dim]`, already normed and roped; `pooled` is
/// `[num_blocks * head_dim]`, already normed and roped at its block's first
/// token's position (section 5: "rope(..., position of the block's FIRST
/// token)").
///
/// **The `relu` is OUTSIDE the head sum, matching section 5's own
/// parenthesization** (`relu(q @ pooled^T).sum(over heads)`, not
/// `relu(...).sum` per head then combined): the dot product `q @ pooled^T`
/// is computed per head first, summed across heads, and `relu` is applied
/// once to that total, not once per head before the sum. Applying it per
/// head would clip a head's own negative contribution to zero before the
/// heads combine, which is a different, smaller number whenever any head
/// disagrees in sign with the total.
pub fn score_blocks(q: &[f32], pooled: &[f32], num_heads: usize, head_dim: usize) -> Vec<f32> {
    assert!(num_heads > 0, "num_heads must be positive");
    assert!(head_dim > 0, "head_dim must be positive");
    assert_eq!(
        q.len(),
        num_heads * head_dim,
        "q must be num_heads * head_dim"
    );
    assert_eq!(
        pooled.len() % head_dim,
        0,
        "pooled must be a whole number of head_dim-wide rows"
    );
    let num_blocks = pooled.len() / head_dim;
    let inv_sqrt_d = 1.0 / (head_dim as f32).sqrt();

    let mut scores = vec![0.0f32; num_blocks];
    for (b, score) in scores.iter_mut().enumerate() {
        let block_base = b * head_dim;
        let mut total = 0.0f32;
        for h in 0..num_heads {
            let q_base = h * head_dim;
            let dot: f32 = (0..head_dim)
                .map(|d| q[q_base + d] * pooled[block_base + d])
                .sum();
            total += dot;
        }
        *score = total.max(0.0) * inv_sqrt_d;
    }
    scores
}

/// Selects which of `visible` token positions QSA attends to: the tokens of
/// the top `min(block_topk, num_complete_blocks)` scoring COMPLETE blocks
/// (each `compress_ratio` positions wide), plus the ragged tail (the
/// `visible % compress_ratio` trailing positions forming no complete
/// block), which is ALWAYS selected regardless of any block's score.
///
/// `scores` has one entry per complete block
/// (`visible / compress_ratio`, rounded down) -- the output of
/// [`score_blocks`]. Returns a `visible`-long boolean mask, oldest token
/// first, ANDed onto the causal mask by the caller.
///
/// **BELOW-BUDGET EXACTNESS (`docs/QWEN4_PHASE0.md` section 5's Q1
/// proof).** When `num_complete_blocks <= block_topk`, EVERY block is
/// chosen regardless of its score, so the mask is all-`true` and QSA is
/// bit-for-bit dense causal attention -- the ordering of ties or of scores
/// generally cannot matter, because nothing is excluded. This is not
/// special-cased in the implementation; it falls out of `chosen_count ==
/// num_complete_blocks` selecting the whole score vector regardless of
/// order, and [`below_budget_selects_every_token`] pins it as a property
/// rather than trusting the arithmetic to keep working.
///
/// A tie in score is broken by BLOCK INDEX (earlier block wins ties, via a
/// stable sort on `(-score, index)`), which matters only above budget and
/// only when two blocks score identically -- not specified by section 5
/// (which does not discuss ties at all) and therefore a choice this
/// function makes rather than one read off a reference; revisit if a
/// mlx-vlm cross-check ever pins a different tie-break.
pub fn select_blocks(
    scores: &[f32],
    visible: usize,
    compress_ratio: usize,
    block_topk: usize,
) -> Vec<bool> {
    assert!(compress_ratio > 0, "compress_ratio must be positive");
    let num_complete_blocks = visible / compress_ratio;
    assert_eq!(
        scores.len(),
        num_complete_blocks,
        "one score per complete block"
    );

    let mut mask = vec![false; visible];
    // The ragged tail is always visible, whatever the blocks decide.
    for slot in mask.iter_mut().skip(num_complete_blocks * compress_ratio) {
        *slot = true;
    }

    let chosen_count = block_topk.min(num_complete_blocks);
    let mut ranked: Vec<usize> = (0..num_complete_blocks).collect();
    ranked.sort_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .expect("score must not be NaN")
            .then(a.cmp(&b))
    });
    for &block in ranked.iter().take(chosen_count) {
        let base = block * compress_ratio;
        for slot in mask.iter_mut().skip(base).take(compress_ratio) {
            *slot = true;
        }
    }
    mask
}

#[cfg(test)]
mod tests {
    use super::{pool_blocks_mean, score_blocks, select_blocks};
    use crate::rms_norm::rms_norm_centered;
    use crate::rope::rope_neox_subdim;

    /// Composes the FULL indexer chain -- per-head centered norm, RoPE (one
    /// call per block, since blocks do not share a position), pooling,
    /// scoring, selection -- using ONLY functions this crate already ships
    /// ([`rms_norm_centered`], [`rope_neox_subdim`], and this module's own
    /// three), against a hand-checked example (values cross-computed in
    /// Python against the identical formulas). This is what proves the
    /// module doc's "norm and RoPE convention" section composes into a
    /// working reference rather than three functions that merely each pass
    /// their own isolated tests.
    ///
    /// 2 query heads, `head_dim = 8`, `rotary_dim = 4` (TWO rotation pairs
    /// per head -- deliberately more than one, see below), `theta = 10000`,
    /// `compress_ratio = 2`, 2 complete blocks and no ragged tail. Block 0
    /// pools to a UNIFORM key (all 1s) and block 1 to a SPARSE one that
    /// shares little direction with either query head, so block 0 must
    /// score higher and win at `block_topk = 1`.
    ///
    /// **`rotary_dim` MUST be at least 4 here, not 2.** At `rotary_dim = 2`
    /// there is only ONE rotation pair, and [`rope_neox_subdim`]'s pairing
    /// (`i` with `pairs + i`) and [`crate::rope::rope_paired`]'s (`2k` with
    /// `2k + 1`) both degenerate to the SAME single pair `(0, 1)` -- so a
    /// fixture at `rotary_dim = 2` cannot tell the two conventions apart at
    /// all, and a first draft of this test at that width let exactly that
    /// wrong-function mutation survive silently (AGENTS.md Gotcha 23's
    /// self-relative-fixture shape, caught by mutation-checking before this
    /// landed rather than by inspection).
    #[test]
    fn the_full_indexer_chain_composes() {
        let head_dim = 8;
        let num_heads = 2;
        let rotary_dim = 4;
        let theta = 10000.0f32;
        let eps = 1e-6;
        let compress_ratio = 2;

        #[rustfmt::skip]
        let raw_query: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ];
        let q_norm_w = vec![0.1f32; head_dim];
        let mut normed_q = vec![0.0f32; raw_query.len()];
        for h in 0..num_heads {
            let slice = &raw_query[h * head_dim..(h + 1) * head_dim];
            let n = rms_norm_centered(slice, &q_norm_w, eps);
            normed_q[h * head_dim..(h + 1) * head_dim].copy_from_slice(&n);
        }
        let query_position = 1;
        let roped_q = rope_neox_subdim(
            &normed_q,
            1,
            num_heads,
            head_dim,
            rotary_dim,
            query_position,
            theta,
        );

        // 2 complete blocks, no ragged tail: block 0's raw rows are both
        // all-1s (pools to itself), block 1's raw rows are one-hot-ish and
        // pool to something largely orthogonal to the query.
        #[rustfmt::skip]
        let raw_keys: Vec<f32> = vec![
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let k_norm_w = vec![0.1f32; head_dim];
        let pooled = pool_blocks_mean(&raw_keys, head_dim, compress_ratio);
        let num_blocks = pooled.len() / head_dim;
        let mut roped_pooled = vec![0.0f32; pooled.len()];
        let key_start_position = 0;
        for b in 0..num_blocks {
            let slice = &pooled[b * head_dim..(b + 1) * head_dim];
            let normed = rms_norm_centered(slice, &k_norm_w, eps);
            let block_position = key_start_position + b * compress_ratio;
            let roped =
                rope_neox_subdim(&normed, 1, 1, head_dim, rotary_dim, block_position, theta);
            roped_pooled[b * head_dim..(b + 1) * head_dim].copy_from_slice(&roped);
        }

        let scores = score_blocks(&roped_q, &roped_pooled, num_heads, head_dim);
        // Cross-checked in Python against the identical formulas; a wrong
        // RoPE pairing convention (rope_paired instead of rope_neox_subdim)
        // moves block 0's score to 5.904 and block 1's to 1.165 -- both well
        // outside this tolerance.
        assert!(
            (scores[0] - 5.749_187).abs() < 1e-3,
            "block 0 score: {}",
            scores[0]
        );
        assert!(
            (scores[1] - 2.496_774).abs() < 1e-3,
            "block 1 score: {}",
            scores[1]
        );

        let visible = raw_keys.len() / head_dim;
        let mask = select_blocks(&scores, visible, compress_ratio, 1);
        assert_eq!(
            mask,
            vec![true, true, false, false],
            "block 0 (higher score) selected, block 1 dropped"
        );
    }

    #[test]
    fn pools_the_mean_of_each_complete_block_and_drops_the_tail() {
        // 2 heads worth of head_dim=1 for readability: 5 visible "tokens",
        // compress_ratio=2 -> 2 complete blocks (rows 0-1, 2-3) plus a
        // 1-row ragged tail (row 4) that pooling must not touch.
        let keys = vec![1.0, 3.0, 5.0, 7.0, 100.0];
        let pooled = pool_blocks_mean(&keys, 1, 2);
        assert_eq!(pooled, vec![2.0, 6.0]);
    }

    #[test]
    fn pooling_averages_per_dimension_independently() {
        // head_dim=2, compress_ratio=2, one block: rows [1,10] and [3,20].
        let keys = vec![1.0, 10.0, 3.0, 20.0];
        let pooled = pool_blocks_mean(&keys, 2, 2);
        assert_eq!(pooled, vec![2.0, 15.0]);
    }

    #[test]
    fn scoring_sums_dot_products_across_heads_before_relu() {
        // head_dim=1, num_heads=2, one block. q = [3, -5], pooled = [2].
        // Per-head dots: 6, -10. Summed BEFORE relu: -4 -> relu = 0.
        // A per-head relu (wrong parenthesization) would give relu(6) +
        // relu(-10) = 6, a materially different and positive answer.
        let q = vec![3.0, -5.0];
        let pooled = vec![2.0];
        let scores = score_blocks(&q, &pooled, 2, 1);
        assert_eq!(scores, vec![0.0]);
    }

    #[test]
    fn scoring_matches_hand_computed_value_with_scale() {
        // head_dim=4, num_heads=1, one block. q . pooled = 1*1+2*1+3*1+4*1=10.
        // relu(10) / sqrt(4) = 10 / 2 = 5.
        let q = vec![1.0, 2.0, 3.0, 4.0];
        let pooled = vec![1.0, 1.0, 1.0, 1.0];
        let scores = score_blocks(&q, &pooled, 1, 4);
        assert!((scores[0] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn below_budget_selects_every_token() {
        // Q1's proof, restated as a property: at the exact boundary
        // (docs/QWEN4_PHASE0.md section 5: "visible <= 2051"),
        // floor(2051 / 4) = 512 == block_topk, so every block must be
        // chosen and the mask must be all-true -- REGARDLESS of the
        // scores, which are deliberately adversarial here (worst-scoring
        // block first) to prove the selection is not accidentally correct
        // because the scores happened to favor an early cutoff.
        let visible = 2051usize;
        let compress_ratio = 4usize;
        let block_topk = 512usize;
        let num_complete_blocks = visible / compress_ratio;
        assert_eq!(num_complete_blocks, block_topk);

        let scores: Vec<f32> = (0..num_complete_blocks).map(|i| i as f32).collect();
        let mask = select_blocks(&scores, visible, compress_ratio, block_topk);
        assert!(
            mask.iter().all(|&v| v),
            "every position must be visible at or under budget"
        );
    }

    #[test]
    fn one_block_over_budget_drops_exactly_the_lowest_scoring_one() {
        // One token past Q1's boundary: 2052 visible tokens is 513
        // complete blocks against a budget of 512, so exactly one block
        // (513 - 512) must be dropped, and it must be the LOWEST-scoring
        // one, not an arbitrary one (e.g. not simply "the last block" or
        // "the first block").
        let visible = 2052usize;
        let compress_ratio = 4usize;
        let block_topk = 512usize;
        let num_complete_blocks = visible / compress_ratio;
        assert_eq!(num_complete_blocks, 513);

        // Block 7 (arbitrary, not first or last) scores lowest.
        let mut scores: Vec<f32> = (0..num_complete_blocks).map(|i| 100.0 + i as f32).collect();
        scores[7] = -1.0;
        let mask = select_blocks(&scores, visible, compress_ratio, block_topk);

        let dropped_base = 7 * compress_ratio;
        for (pos, &visible) in mask
            .iter()
            .enumerate()
            .skip(dropped_base)
            .take(compress_ratio)
        {
            assert!(
                !visible,
                "block 7 (lowest score) must be dropped, position {pos}"
            );
        }
        let visible_count = mask.iter().filter(|&&v| v).count();
        assert_eq!(
            visible_count,
            block_topk * compress_ratio,
            "exactly block_topk blocks' worth of positions must remain visible"
        );
    }

    #[test]
    fn the_ragged_tail_is_always_selected_even_when_it_would_score_worst() {
        // 10 visible tokens, compress_ratio=4 -> 2 complete blocks (8
        // tokens) plus a 2-token ragged tail. Even at a budget of ZERO
        // (no block survives), the tail must still be visible, because
        // section 5 says the tail is selected unconditionally rather than
        // via the scoring/top-k path at all.
        let visible = 10usize;
        let compress_ratio = 4usize;
        let scores = vec![5.0, 9.0]; // 2 complete blocks
        let mask = select_blocks(&scores, visible, compress_ratio, 0);
        assert_eq!(
            mask,
            vec![false, false, false, false, false, false, false, false, true, true]
        );
    }

    #[test]
    fn selection_picks_the_highest_scoring_blocks_not_an_index_order() {
        // 4 complete blocks, budget for 2. The two HIGHEST-scoring blocks
        // are index 1 and 3 (deliberately not the first two, so a bug that
        // selects "the first `block_topk` blocks by index" rather than by
        // score would read a plausible-looking wrong mask instead of an
        // obviously broken one.
        let compress_ratio = 3usize;
        let visible = 4 * compress_ratio;
        let scores = vec![1.0, 9.0, 2.0, 8.0];
        let mask = select_blocks(&scores, visible, compress_ratio, 2);

        let block_selected = |b: usize| -> bool { mask[b * compress_ratio] };
        assert!(!block_selected(0));
        assert!(block_selected(1));
        assert!(!block_selected(2));
        assert!(block_selected(3));
    }

    #[test]
    fn no_complete_blocks_selects_only_the_tail() {
        // visible < compress_ratio: every token is ragged tail, no scores
        // to consult at all.
        let mask = select_blocks(&[], 3, 4, 512);
        assert_eq!(mask, vec![true, true, true]);
    }
}
