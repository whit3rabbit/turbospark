//! A block drafter's candidate lattice, as plain data.
//!
//! The DFlash2 selector ranks a short list of candidates per draft row and
//! scores every adjacent pair, then keeps only its greedy path
//! (`families/qwen/dflash_draft/select.rs`). Tree-verify research needs the
//! whole lattice: [`crate::RealForwardRunner::dflash_lattice`] reads it back
//! after a draft pass, without touching drafter or trunk state.

use foundation::TokenId;

/// One DFlash2 round's lattice.
///
/// Depth `d` predicts the token `d + 1` positions after the anchor.
#[derive(Clone, Debug, PartialEq)]
pub struct DflashLattice {
    /// `candidates[d]`: the selector's top-K at depth `d`, highest unary
    /// logit first.
    pub candidates: Vec<Vec<TokenId>>,
    /// Pair scores, the quantity the selector maximizes. `edges[0]` holds K
    /// scores from the anchor to each depth-0 candidate. For `d > 0`,
    /// `edges[d][p * K + c]` scores depth-`d` candidate `c` after depth-`d-1`
    /// candidate `p`.
    pub edges: Vec<Vec<f32>>,
}

impl DflashLattice {
    /// Candidates per depth.
    pub fn width(&self) -> usize {
        self.candidates.first().map_or(0, Vec::len)
    }

    /// The selector's greedy path, as candidate indices: the first maximum
    /// at every depth, the selector's own tie-break.
    pub fn greedy_path(&self) -> Vec<usize> {
        let k = self.width();
        let mut path = Vec::with_capacity(self.candidates.len());
        for (d, row) in self.edges.iter().enumerate() {
            let scores = if d == 0 {
                &row[..]
            } else {
                let p = path[d - 1];
                &row[p * k..(p + 1) * k]
            };
            let mut best = 0usize;
            for (c, &s) in scores.iter().enumerate() {
                if s > scores[best] {
                    best = c;
                }
            }
            path.push(best);
        }
        path
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_greedy_path_follows_the_best_edge_from_each_choice() {
        let l = DflashLattice {
            candidates: vec![vec![10, 11], vec![20, 21]],
            // From the anchor, candidate 1 wins. After candidate 1, 20 wins.
            edges: vec![vec![0.1, 0.9], vec![0.0, 5.0, 3.0, 1.0]],
        };
        assert_eq!(l.greedy_path(), vec![1, 0]);
    }

    #[test]
    fn ties_keep_the_first_candidate() {
        let l = DflashLattice {
            candidates: vec![vec![1, 2]],
            edges: vec![vec![0.5, 0.5]],
        };
        assert_eq!(l.greedy_path(), vec![0]);
    }
}
