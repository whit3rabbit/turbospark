//! Host-only replay of tree speculative verification (kapitsa's KFlash)
//! against a chain, at the SAME verify row budget.
//!
//! Why equal rows: our verify is a batched GEMV whose cost per row is
//! c(M) ~0.46 of a decode (`docs/SPECULATIVE_DECODING.md`), so extra rows
//! are not free the way KFlash's 8-row MMA tile is. A tree has to beat the
//! chain inside the rows the chain already pays for.
//!
//! The replay is exact on the token count: every accepted path equals the
//! reference stream's prefix, so a node is accepted iff its token is the
//! reference token at its position. No tree kernel is needed to count it.
//! It does NOT model a tree's own overhead (masked attention, per-branch
//! recurrent state, KV compaction; kapitsa measured 1-6% on dense models).
//!
//! Lattice shape. `depths[d]` holds the drafter's top-k at depth d, sorted by
//! probability, with `tokens[0]` the chain token. Only chain nodes have
//! children, because an MTP head conditions each step on the token it was
//! fed: a sibling's children would need another head step. That caps the
//! tree's gain at one token per round. A DFlash2 lattice (path-independent
//! rows plus pairwise scores) would lift the cap and needs its own builder.

/// The drafter's top-k at one depth, sorted by probability, highest first.
#[derive(Clone, Debug, PartialEq)]
pub struct DepthCandidates {
    /// Candidate token ids. `tokens[0]` is the chain's own proposal.
    pub tokens: Vec<i32>,
    /// Drafter probability of each candidate, conditional on the chain prefix.
    pub probs: Vec<f32>,
}

/// One round's lattice, drafted after the target committed `stream[i]`.
/// `depths[d]` predicts `stream[i + 1 + d]`.
#[derive(Clone, Debug, PartialEq)]
pub struct Lattice {
    /// Per-depth candidates, depth 0 first.
    pub depths: Vec<DepthCandidates>,
}

/// A node `(depth, rank)` of the lattice. Rank 0 is the chain token.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Node {
    /// Depth below the anchor (0 predicts the token right after it).
    pub depth: usize,
    /// Rank among that depth's candidates.
    pub rank: usize,
}

/// The chain's nodes for `nodes` verify rows past the anchor.
pub fn chain_nodes(lattice: &Lattice, nodes: usize) -> Vec<Node> {
    (0..nodes.min(lattice.depths.len()))
        .map(|depth| Node { depth, rank: 0 })
        .collect()
}

/// Best-first tree of `nodes` nodes, KFlash's builder: repeatedly take the
/// frontier node whose path probability is highest. A child never scores
/// above its parent, so parents precede children and the first n entries
/// are the best n-node tree for every n.
pub fn best_first_nodes(lattice: &Lattice, nodes: usize) -> Vec<Node> {
    let mut out = Vec::with_capacity(nodes);
    // (path probability, node). Linear scan: the frontier is a few dozen.
    let mut frontier: Vec<(f64, Node)> = Vec::new();
    let push_children = |frontier: &mut Vec<(f64, Node)>, depth: usize, prefix: f64| {
        if let Some(c) = lattice.depths.get(depth) {
            for (rank, &p) in c.probs.iter().enumerate() {
                frontier.push((prefix * p as f64, Node { depth, rank }));
            }
        }
    };
    push_children(&mut frontier, 0, 1.0);
    while out.len() < nodes && !frontier.is_empty() {
        // Highest score; ties go to the shallower, then lower-ranked node so
        // the build is deterministic.
        let mut best = 0;
        for i in 1..frontier.len() {
            let (s, n) = frontier[i];
            let (bs, bn) = frontier[best];
            if s > bs || (s == bs && (n.depth, n.rank) < (bn.depth, bn.rank)) {
                best = i;
            }
        }
        let (score, node) = frontier.swap_remove(best);
        out.push(node);
        if node.rank == 0 {
            push_children(&mut frontier, node.depth + 1, score);
        }
    }
    out
}

/// Accepted proposals (bonus excluded) for a tree anchored at `stream[i]`.
/// Walk from the anchor into the child whose token is the reference token;
/// only a rank-0 node can be walked through.
pub fn accepted(lattice: &Lattice, tree: &[Node], stream: &[i32], i: usize) -> usize {
    let mut n = 0;
    for depth in 0.. {
        let Some(&target) = stream.get(i + 1 + depth) else {
            break;
        };
        let hit = tree
            .iter()
            .find(|node| node.depth == depth && lattice.depths[depth].tokens[node.rank] == target);
        match hit {
            Some(node) => {
                n += 1;
                if node.rank != 0 {
                    break;
                }
            }
            None => break,
        }
    }
    n
}

/// The most any `nodes`-node tree over this lattice could accept.
pub fn oracle_accepted(lattice: &Lattice, nodes: usize, stream: &[i32], i: usize) -> usize {
    let mut n = 0;
    for (depth, c) in lattice.depths.iter().enumerate() {
        if n >= nodes {
            break;
        }
        let Some(&target) = stream.get(i + 1 + depth) else {
            break;
        };
        match c.tokens.iter().position(|&t| t == target) {
            Some(0) => n += 1,
            // A sibling hit costs one node and ends the path.
            Some(_) => return n + 1,
            None => break,
        }
    }
    n
}

/// Which tree a replay arm verifies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shape {
    /// The drafter's chain.
    Chain,
    /// KFlash's best-first tree.
    BestFirst,
    /// The best tree this lattice allows (upper bound, not buildable).
    Oracle,
}

/// Rounds and committed tokens of one replayed loop.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ReplayStats {
    /// Verify rounds run.
    pub rounds: usize,
    /// Tokens committed: accepted proposals plus one bonus per round.
    pub committed: usize,
}

impl ReplayStats {
    /// Committed tokens per round, the number a verify cost has to clear.
    pub fn committed_per_round(&self) -> f64 {
        self.committed as f64 / self.rounds.max(1) as f64
    }
}

/// Replays the speculative loop over `stream`, where `lattices[i]` is the
/// lattice drafted after `stream[i]`. Each arm walks its own anchors: a
/// round at `i` commits `a + 1` tokens and the next round starts at
/// `i + a + 1`. Stops when the next anchor has no lattice.
pub fn replay(
    lattices: &[Option<Lattice>],
    stream: &[i32],
    nodes: usize,
    shape: Shape,
) -> ReplayStats {
    let mut stats = ReplayStats::default();
    let mut i = 0;
    while let Some(Some(lattice)) = lattices.get(i) {
        let a = match shape {
            Shape::Chain => accepted(lattice, &chain_nodes(lattice, nodes), stream, i),
            Shape::BestFirst => accepted(lattice, &best_first_nodes(lattice, nodes), stream, i),
            Shape::Oracle => oracle_accepted(lattice, nodes, stream, i),
        };
        stats.rounds += 1;
        stats.committed += a + 1;
        i += a + 1;
    }
    stats
}

#[cfg(test)]
mod tests {
    use super::*;

    fn depth(tokens: &[i32], probs: &[f32]) -> DepthCandidates {
        DepthCandidates {
            tokens: tokens.to_vec(),
            probs: probs.to_vec(),
        }
    }

    #[test]
    fn with_one_candidate_the_tree_is_the_chain() {
        let l = Lattice {
            depths: vec![
                depth(&[5], &[0.9]),
                depth(&[6], &[0.8]),
                depth(&[7], &[0.7]),
            ],
        };
        assert_eq!(best_first_nodes(&l, 3), chain_nodes(&l, 3));
        let stream = [1, 5, 6, 9];
        assert_eq!(accepted(&l, &chain_nodes(&l, 3), &stream, 0), 2);
    }

    #[test]
    fn a_sibling_hit_adds_exactly_one_token_and_ends_the_path() {
        // Chain guesses 5 then 6; the target writes 8, the depth-0 runner-up.
        let l = Lattice {
            depths: vec![depth(&[5, 8], &[0.6, 0.4]), depth(&[6, 3], &[0.9, 0.1])],
        };
        let stream = [1, 8, 6, 2];
        assert_eq!(accepted(&l, &chain_nodes(&l, 2), &stream, 0), 0);
        let tree = best_first_nodes(&l, 2);
        // 0.6 (5), then 0.54 (5->6) beats 0.4 (8): the tree keeps the chain.
        assert_eq!(tree, chain_nodes(&l, 2));
        let tree = best_first_nodes(&l, 3);
        assert!(tree.contains(&Node { depth: 0, rank: 1 }));
        // 8 is accepted, and 6 under it is not walked: siblings are leaves.
        assert_eq!(accepted(&l, &tree, &stream, 0), 1);
    }

    #[test]
    fn a_low_confidence_chain_yields_its_rows_to_the_runner_up() {
        let l = Lattice {
            depths: vec![depth(&[5, 8], &[0.5, 0.45]), depth(&[6, 3], &[0.5, 0.5])],
        };
        // 0.5 (5), 0.45 (8), then 0.25 (5->6 or 5->3).
        assert_eq!(
            best_first_nodes(&l, 2),
            vec![Node { depth: 0, rank: 0 }, Node { depth: 0, rank: 1 }]
        );
    }

    #[test]
    fn parents_always_precede_children() {
        let l = Lattice {
            depths: (0..6)
                .map(|d| depth(&[d, 100 + d, 200 + d], &[0.7, 0.2, 0.1]))
                .collect(),
        };
        let tree = best_first_nodes(&l, 12);
        for (k, n) in tree.iter().enumerate() {
            if n.depth > 0 {
                let parent = Node {
                    depth: n.depth - 1,
                    rank: 0,
                };
                assert!(tree[..k].contains(&parent), "{n:?} before its parent");
            }
        }
        // Prefix property: every prefix is the n-node build.
        for n in 1..tree.len() {
            assert_eq!(best_first_nodes(&l, n), tree[..n]);
        }
    }

    #[test]
    fn a_replay_walks_its_own_anchors() {
        // Lattice i guesses stream[i+1] at depth 0 for even i only; depth 1
        // never hits. Chain of 2: even anchors accept 1 (commit 2), odd
        // anchors accept 0 (commit 1). From 0: 0 -> 2 -> 4 -> 6 -> stop.
        let stream: Vec<i32> = (0..8).collect();
        let lattices: Vec<Option<Lattice>> = (0..8)
            .map(|i| {
                if i >= 6 {
                    return None;
                }
                let guess = if i % 2 == 0 { i + 1 } else { -1 };
                Some(Lattice {
                    depths: vec![depth(&[guess, i + 1], &[0.9, 0.1]), depth(&[-1], &[0.9])],
                })
            })
            .collect();
        let chain = replay(&lattices, &stream, 2, Shape::Chain);
        assert_eq!(
            chain,
            ReplayStats {
                rounds: 3,
                committed: 6
            }
        );
        // A 3-node tree also holds the depth-0 runner-up, which is the
        // reference token at every anchor, so odd anchors would hit too;
        // but the walk never lands on one, and the count is the same.
        let tree = replay(&lattices, &stream, 3, Shape::BestFirst);
        assert_eq!(
            tree,
            ReplayStats {
                rounds: 3,
                committed: 6
            }
        );
        // From an odd anchor the tree's runner-up is what pays.
        let odd = &lattices[1].clone().unwrap();
        assert_eq!(accepted(odd, &chain_nodes(odd, 2), &stream, 1), 0);
        assert_eq!(accepted(odd, &best_first_nodes(odd, 3), &stream, 1), 1);
    }

    #[test]
    fn per_round_oracle_dominates_both_shapes() {
        let l = Lattice {
            depths: vec![
                depth(&[5, 8, 9], &[0.5, 0.3, 0.2]),
                depth(&[6, 3, 4], &[0.6, 0.3, 0.1]),
                depth(&[7, 2, 1], &[0.6, 0.3, 0.1]),
            ],
        };
        for stream in [
            [0, 5, 6, 7],
            [0, 8, 0, 0],
            [0, 5, 3, 0],
            [0, 5, 6, 2],
            [0, 1, 1, 1],
        ] {
            for nodes in 1..=5 {
                let o = oracle_accepted(&l, nodes, &stream, 0);
                assert!(o >= accepted(&l, &chain_nodes(&l, nodes), &stream, 0));
                assert!(o >= accepted(&l, &best_first_nodes(&l, nodes), &stream, 0));
            }
        }
    }
}
