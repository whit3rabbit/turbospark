//! The shipped tree builder and acceptance walk (`runtime::dflash_tree`)
//! against the replay that priced them (`bench::tree_replay`), on random
//! lattices: the same nodes in the same order, and the same accepted count
//! against every reference stream. If these part, the measured 1.07-1.12x
//! committed/round no longer describes the tree the engine verifies.

use runtime::{dflash_best_first_tree, dflash_tree_accept, DflashLattice, DflashTreeNode};
use turbospark_bench::tree_replay::{edge_accepted, edge_best_first_nodes, EdgeNode};

fn lattice(seed: u64, depth: usize, k: usize) -> DflashLattice {
    let mut s = seed | 1;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    // Small token alphabet so siblings and cousins share tokens, the case
    // where a walk that matched by token alone would pick the wrong node.
    let candidates: Vec<Vec<i32>> = (0..depth)
        .map(|_| (0..k).map(|_| (next() % 9) as i32).collect())
        .collect();
    let edges: Vec<Vec<f32>> = (0..depth)
        .map(|d| {
            let n = if d == 0 { k } else { k * k };
            (0..n)
                .map(|_| (next() % 4000) as f32 / 500.0 - 4.0)
                .collect()
        })
        .collect();
    DflashLattice { candidates, edges }
}

fn shipped(l: &DflashLattice, nodes: usize) -> Vec<DflashTreeNode> {
    let k = l.width();
    dflash_best_first_tree(l.candidates.len(), nodes, |depth, parent| match parent {
        None => l.edges[depth].clone(),
        Some(p) => l.edges[depth][p.cand * k..(p.cand + 1) * k].to_vec(),
    })
}

fn as_edge(n: &DflashTreeNode) -> EdgeNode {
    EdgeNode {
        depth: n.depth,
        cand: n.cand,
        parent: n.parent,
    }
}

#[test]
fn the_shipped_builder_is_the_replays() {
    for seed in 1..200u64 {
        let l = lattice(seed, 7, 16);
        for nodes in [1usize, 3, 7, 15] {
            let got: Vec<EdgeNode> = shipped(&l, nodes).iter().map(as_edge).collect();
            assert_eq!(
                got,
                edge_best_first_nodes(&l, nodes),
                "seed {seed}, {nodes} nodes"
            );
        }
    }
}

#[test]
fn the_shipped_acceptance_is_the_replays() {
    for seed in 1..200u64 {
        let l = lattice(seed, 7, 4);
        let tree = shipped(&l, 7);
        let edges: Vec<EdgeNode> = tree.iter().map(as_edge).collect();
        let tokens: Vec<i32> = tree.iter().map(|n| l.candidates[n.depth][n.cand]).collect();
        // Reference streams drawn from the same alphabet, anchored at 0.
        for r in 0..20u64 {
            let mut s = (seed * 1000 + r) | 1;
            let stream: Vec<i32> = (0..9)
                .map(|_| {
                    s ^= s << 13;
                    s ^= s >> 7;
                    s ^= s << 17;
                    (s % 9) as i32
                })
                .collect();
            let want = edge_accepted(&l, &edges, &stream, 0);
            // The target's pick at a node is the stream's next token there.
            let got = dflash_tree_accept(&tree, &tokens, |at| {
                let depth = at.map_or(0, |id| tree[id].depth + 1);
                stream.get(1 + depth).copied().unwrap_or(-1)
            });
            assert_eq!(got.len(), want, "seed {seed}, stream {stream:?}");
        }
    }
}
