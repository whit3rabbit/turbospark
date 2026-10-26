//! Tree verify for the DFlash2 drafter (`TURBOSPARK_DFLASH_TREE`,
//! `docs/KAPITSA_KERNELS.md` lever 4), the host halves as plain functions:
//! the best-first tree builder and the acceptance walk. Both are pinned to
//! `crates/bench`'s replay (`tree_replay::edge_best_first_nodes`,
//! `edge_accepted`), which is what priced the tree before any kernel
//! existed, so the shipped tree is the one the replay measured.

use foundation::TokenId;

/// One node of a DFlash2 tree: candidate `cand` at `depth` below the
/// anchor, under node `parent` (`None` for a child of the anchor).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DflashTreeNode {
    pub depth: usize,
    pub cand: usize,
    pub parent: Option<usize>,
}

/// KFlash's best-first builder: repeatedly take the frontier node with the
/// highest path probability, then add all its children. Parents precede
/// children, so every prefix is the best tree of its size.
///
/// `child_scores(depth, parent)` returns the selector's scores for the
/// depth-`depth` candidates after the `parent` node's candidate (after the
/// anchor when `None`). It is called only for the root and for each node as it is
/// EXPANDED, never for the whole lattice, so a 7-node tree scores at most 8
/// rows of 16 children. Probabilities are a softmax over one parent's
/// children in f64; ties prefer the shallower, then the lower candidate,
/// as the replay does.
pub fn dflash_best_first_tree(
    depth_max: usize,
    nodes: usize,
    mut child_scores: impl FnMut(usize, Option<DflashTreeNode>) -> Vec<f32>,
) -> Vec<DflashTreeNode> {
    let mut out: Vec<DflashTreeNode> = Vec::with_capacity(nodes);
    if depth_max == 0 || nodes == 0 {
        return out;
    }
    let mut frontier: Vec<(f64, DflashTreeNode)> = softmax(&child_scores(0, None))
        .into_iter()
        .enumerate()
        .map(|(cand, p)| {
            (
                p,
                DflashTreeNode {
                    depth: 0,
                    cand,
                    parent: None,
                },
            )
        })
        .collect();
    while out.len() < nodes && !frontier.is_empty() {
        let mut best = 0;
        for i in 1..frontier.len() {
            let (s, n) = frontier[i];
            let (bs, bn) = frontier[best];
            if s > bs || (s == bs && (n.depth, n.cand) < (bn.depth, bn.cand)) {
                best = i;
            }
        }
        let (score, node) = frontier.swap_remove(best);
        let id = out.len();
        out.push(node);
        if out.len() < nodes && node.depth + 1 < depth_max {
            for (cand, p) in softmax(&child_scores(node.depth + 1, Some(node)))
                .into_iter()
                .enumerate()
            {
                frontier.push((
                    score * p,
                    DflashTreeNode {
                        depth: node.depth + 1,
                        cand,
                        parent: Some(id),
                    },
                ));
            }
        }
    }
    out
}

fn softmax(scores: &[f32]) -> Vec<f64> {
    let m = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)) as f64;
    let e: Vec<f64> = scores.iter().map(|&s| (s as f64 - m).exp()).collect();
    let z: f64 = e.iter().sum();
    e.into_iter().map(|v| v / z).collect()
}

/// The accepted path through a verified tree: from the anchor, step into
/// the child whose token is the target's pick at the current node, while
/// there is one. `target(None)` is the pick at the anchor's row,
/// `target(Some(id))` at node `id`'s. Returns the accepted node ids in
/// depth order; the bonus token is `target` at the last of them (or at the
/// anchor when none was accepted).
pub fn dflash_tree_accept(
    tree: &[DflashTreeNode],
    tokens: &[TokenId],
    mut target: impl FnMut(Option<usize>) -> TokenId,
) -> Vec<usize> {
    let mut accepted = Vec::new();
    let mut at: Option<usize> = None;
    loop {
        let want = target(at);
        let hit = (0..tree.len()).find(|&id| tree[id].parent == at && tokens[id] == want);
        match hit {
            Some(id) => {
                accepted.push(id);
                at = Some(id);
            }
            None => return accepted,
        }
    }
}

/// A tree's slot parents for the verify pass: slot 0 is the anchor (-1),
/// node `i` sits at slot `i + 1` under its parent's slot.
pub fn dflash_tree_slot_parents(tree: &[DflashTreeNode]) -> Vec<i32> {
    std::iter::once(-1)
        .chain(tree.iter().map(|n| n.parent.map_or(0, |p| p as i32 + 1)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Scores where candidate `c` at any depth scores `-(c as f32)`, plus a
    /// boost for candidate 1 after the anchor so the runner-up branches.
    fn scores(depth: usize, parent: Option<DflashTreeNode>) -> Vec<f32> {
        let mut s: Vec<f32> = (0..4).map(|c| -(c as f32)).collect();
        if depth == 0 && parent.is_none() {
            s[1] = -0.2;
        }
        s
    }

    #[test]
    fn parents_precede_children_and_the_spine_is_greedy() {
        let tree = dflash_best_first_tree(4, 7, scores);
        assert_eq!(tree.len(), 7);
        for (i, n) in tree.iter().enumerate() {
            if let Some(p) = n.parent {
                assert!(p < i);
                assert_eq!(tree[p].depth + 1, n.depth);
            }
        }
        // The first node is the anchor's best child, the second the
        // runner-up (near-tied) at depth 0.
        assert_eq!((tree[0].depth, tree[0].cand), (0, 0));
        assert_eq!((tree[1].depth, tree[1].cand), (0, 1));
    }

    #[test]
    fn scores_are_asked_for_only_on_expansion() {
        let mut calls = 0;
        let tree = dflash_best_first_tree(7, 7, |d, p| {
            calls += 1;
            scores(d, p)
        });
        // The root, plus every expanded node but the last (its children
        // would be past the node budget).
        assert_eq!(calls, tree.len());
    }

    #[test]
    fn acceptance_walks_children_by_token() {
        let tree = [
            DflashTreeNode {
                depth: 0,
                cand: 0,
                parent: None,
            },
            DflashTreeNode {
                depth: 0,
                cand: 1,
                parent: None,
            },
            DflashTreeNode {
                depth: 1,
                cand: 0,
                parent: Some(1),
            },
        ];
        let tokens = [10, 11, 20];
        let picks = |at: Option<usize>| match at {
            None => 11,
            Some(1) => 20,
            _ => 99,
        };
        assert_eq!(dflash_tree_accept(&tree, &tokens, picks), vec![1, 2]);
        assert_eq!(dflash_tree_slot_parents(&tree), vec![-1, 0, 0, 2]);
    }
}
