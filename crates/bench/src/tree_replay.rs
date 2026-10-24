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
//! rows plus pairwise scores, `runtime::DflashLattice`) lifts the cap and
//! has its own builder below.

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

/// A drafter lattice the replay can score: the accepted proposal count
/// (bonus excluded) for a tree of `nodes` nodes of the given shape, anchored
/// at `stream[i]`.
pub trait ReplayLattice {
    /// Accepted proposals for one round.
    fn accepted_by(&self, shape: Shape, nodes: usize, stream: &[i32], i: usize) -> usize;
}

impl ReplayLattice for Lattice {
    fn accepted_by(&self, shape: Shape, nodes: usize, stream: &[i32], i: usize) -> usize {
        match shape {
            Shape::Chain => accepted(self, &chain_nodes(self, nodes), stream, i),
            Shape::BestFirst => accepted(self, &best_first_nodes(self, nodes), stream, i),
            Shape::Oracle => oracle_accepted(self, nodes, stream, i),
        }
    }
}

/// Replays the speculative loop over `stream`, where `lattices[i]` is the
/// lattice drafted after `stream[i]`. Each arm walks its own anchors: a
/// round at `i` commits `a + 1` tokens and the next round starts at
/// `i + a + 1`. Stops when the next anchor has no lattice.
pub fn replay<L: ReplayLattice>(
    lattices: &[Option<L>],
    stream: &[i32],
    nodes: usize,
    shape: Shape,
) -> ReplayStats {
    let mut stats = ReplayStats::default();
    let mut i = 0;
    while let Some(Some(lattice)) = lattices.get(i) {
        let a = lattice.accepted_by(shape, nodes, stream, i);
        stats.rounds += 1;
        stats.committed += a + 1;
        i += a + 1;
    }
    stats
}

// ---------------------------------------------------------------------------
// DFlash2 lattices (`runtime::DflashLattice`).
//
// Unlike the MTP shape above, every node can have children: the drafter's
// rows are path-independent, and the selector's pair scores give a
// conditional score for any candidate after any candidate of the row before.
// So a tree is not capped at one extra token per round here.

/// One node of a DFlash2 tree: candidate `cand` at `depth`, under `parent`
/// (an index into the same node list, `None` for a child of the anchor).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EdgeNode {
    /// Depth below the anchor.
    pub depth: usize,
    /// Candidate index at that depth.
    pub cand: usize,
    /// Parent node index, `None` under the anchor.
    pub parent: Option<usize>,
}

/// Conditional probabilities of the depth-`depth` candidates after the
/// given parent candidate (softmax over that parent's pair scores).
fn edge_probs(l: &runtime::DflashLattice, depth: usize, parent_cand: Option<usize>) -> Vec<f64> {
    let k = l.width();
    let scores: &[f32] = match parent_cand {
        None => &l.edges[depth][..],
        Some(p) => &l.edges[depth][p * k..(p + 1) * k],
    };
    let m = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)) as f64;
    let e: Vec<f64> = scores.iter().map(|&s| (s as f64 - m).exp()).collect();
    let z: f64 = e.iter().sum();
    e.into_iter().map(|v| v / z).collect()
}

/// The selector's greedy chain as tree nodes, `nodes` deep at most.
pub fn edge_chain_nodes(l: &runtime::DflashLattice, nodes: usize) -> Vec<EdgeNode> {
    l.greedy_path()
        .into_iter()
        .take(nodes)
        .enumerate()
        .map(|(depth, cand)| EdgeNode {
            depth,
            cand,
            parent: depth.checked_sub(1),
        })
        .collect()
}

/// KFlash's best-first builder over a DFlash2 lattice: take the frontier
/// node with the highest path probability, then add all its children.
/// Parents precede children, so every prefix is the best tree of its size.
pub fn edge_best_first_nodes(l: &runtime::DflashLattice, nodes: usize) -> Vec<EdgeNode> {
    let depth_max = l.candidates.len();
    let mut out: Vec<EdgeNode> = Vec::with_capacity(nodes);
    let mut frontier: Vec<(f64, EdgeNode)> = Vec::new();
    if depth_max == 0 {
        return out;
    }
    for (cand, p) in edge_probs(l, 0, None).into_iter().enumerate() {
        frontier.push((
            p,
            EdgeNode {
                depth: 0,
                cand,
                parent: None,
            },
        ));
    }
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
        if node.depth + 1 < depth_max {
            for (cand, p) in edge_probs(l, node.depth + 1, Some(node.cand))
                .into_iter()
                .enumerate()
            {
                frontier.push((
                    score * p,
                    EdgeNode {
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

/// Accepted proposals for a DFlash2 tree: from the anchor, step into the
/// child whose token is the reference token, as long as there is one.
pub fn edge_accepted(
    l: &runtime::DflashLattice,
    tree: &[EdgeNode],
    stream: &[i32],
    i: usize,
) -> usize {
    let mut parent: Option<usize> = None;
    let mut n = 0;
    for depth in 0..l.candidates.len() {
        let Some(&target) = stream.get(i + 1 + depth) else {
            break;
        };
        let hit = tree.iter().position(|node| {
            node.depth == depth && node.parent == parent && l.candidates[depth][node.cand] == target
        });
        match hit {
            Some(id) => {
                n += 1;
                parent = Some(id);
            }
            None => break,
        }
    }
    n
}

/// The most any `nodes`-node tree over this lattice could accept: the run
/// of leading depths whose reference token is among that depth's
/// candidates, one node per depth.
pub fn edge_oracle_accepted(
    l: &runtime::DflashLattice,
    nodes: usize,
    stream: &[i32],
    i: usize,
) -> usize {
    let mut n = 0;
    for (depth, row) in l.candidates.iter().enumerate() {
        if n >= nodes {
            break;
        }
        match stream.get(i + 1 + depth) {
            Some(t) if row.contains(t) => n += 1,
            _ => break,
        }
    }
    n
}

impl ReplayLattice for runtime::DflashLattice {
    fn accepted_by(&self, shape: Shape, nodes: usize, stream: &[i32], i: usize) -> usize {
        match shape {
            Shape::Chain => edge_accepted(self, &edge_chain_nodes(self, nodes), stream, i),
            Shape::BestFirst => edge_accepted(self, &edge_best_first_nodes(self, nodes), stream, i),
            Shape::Oracle => edge_oracle_accepted(self, nodes, stream, i),
        }
    }
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

    fn edge_lattice(rows: &[(&[i32], &[f32])]) -> runtime::DflashLattice {
        runtime::DflashLattice {
            candidates: rows.iter().map(|(c, _)| c.to_vec()).collect(),
            edges: rows.iter().map(|(_, e)| e.to_vec()).collect(),
        }
    }

    #[test]
    fn the_dflash_chain_is_the_selectors_greedy_path() {
        // Depth 0: candidate 1 (token 6) wins from the anchor. Depth 1 after
        // candidate 1: candidate 0 (token 7) wins.
        let l = edge_lattice(&[(&[5, 6], &[0.0, 2.0]), (&[7, 8], &[0.0, 9.0, 3.0, 0.0])]);
        let chain = edge_chain_nodes(&l, 2);
        assert_eq!(chain.iter().map(|n| n.cand).collect::<Vec<_>>(), vec![1, 0]);
        assert_eq!(edge_accepted(&l, &chain, &[0, 6, 7], 0), 2);
    }

    #[test]
    fn a_dflash_tree_can_go_deep_under_a_runner_up() {
        // The target writes 5 (the depth-0 runner-up) then 8, which is only
        // likely AFTER 5. An MTP lattice cannot reach depth 1 under a
        // sibling; this one can.
        let l = edge_lattice(&[(&[6, 5], &[1.0, 0.8]), (&[7, 8], &[5.0, 0.0, 0.0, 5.0])]);
        let stream = [0, 5, 8];
        assert_eq!(edge_accepted(&l, &edge_chain_nodes(&l, 2), &stream, 0), 0);
        let tree = edge_best_first_nodes(&l, 4);
        assert_eq!(edge_accepted(&l, &tree, &stream, 0), 2);
        assert_eq!(edge_oracle_accepted(&l, 4, &stream, 0), 2);
    }

    #[test]
    fn dflash_parents_precede_children_and_prefixes_are_builds() {
        let l = edge_lattice(&[
            (&[1, 2, 3], &[0.5, 0.2, 0.1]),
            (&[4, 5, 6], &[0.3, 0.1, 0.0, 0.2, 0.2, 0.2, 0.0, 0.9, 0.1]),
            (&[7, 8, 9], &[0.1; 9]),
        ]);
        let tree = edge_best_first_nodes(&l, 10);
        for (k, n) in tree.iter().enumerate() {
            if let Some(p) = n.parent {
                assert!(p < k, "{n:?} before its parent");
                assert_eq!(tree[p].depth + 1, n.depth);
            }
        }
        for n in 1..tree.len() {
            assert_eq!(edge_best_first_nodes(&l, n), tree[..n]);
        }
    }

    #[test]
    fn the_dflash_oracle_bounds_both_shapes() {
        let l = edge_lattice(&[
            (&[1, 2, 3], &[0.5, 0.2, 0.1]),
            (&[4, 5, 6], &[0.3, 0.1, 0.0, 0.2, 0.2, 0.2, 0.0, 0.9, 0.1]),
            (&[7, 8, 9], &[0.1; 9]),
        ]);
        for stream in [[0, 1, 4, 7], [0, 3, 5, 9], [0, 2, 6, 0], [0, 9, 9, 9]] {
            for nodes in 1..=8 {
                let o = edge_oracle_accepted(&l, nodes, &stream, 0);
                assert!(o >= edge_accepted(&l, &edge_chain_nodes(&l, nodes), &stream, 0));
                assert!(o >= edge_accepted(&l, &edge_best_first_nodes(&l, nodes), &stream, 0));
            }
        }
    }
}
