#![cfg(target_os = "macos")]
//! `encode_attention_decode_tree` (`FC_ATTN_TREE` on the GQA-shared decode
//! partial) against the per-row `encode_attention_decode` the chain verify
//! runs, at qwen38's attention shape (head_dim 256, 24 Q heads, 4 KV heads):
//!
//! 1. the identity path is the chain row bit for bit, at contexts on both
//!    sides of the launch's chunk-count step (16K positions);
//! 2. a branching node reads its ancestors' slots: its output is the chain
//!    row's over a cache where its root path was laid out consecutively;
//! 3. the identity map over the branching layout fails (2) exactly at the
//!    branch nodes, so the check separates a tree from a chain.

use half::f16;
use turbospark_gpu::{
    encode_attention_decode, encode_attention_decode_tree, read_buffer_f16, AttentionScratch,
    DecodeAttentionKernel, MetalBuffer, MetalContext,
};

const HEAD_DIM: u32 = 256;
const Q_HEADS: u32 = 24;
const KV_HEADS: u32 = 4;
const ROW: usize = (KV_HEADS * HEAD_DIM) as usize;
const Q_LEN: usize = (Q_HEADS * HEAD_DIM) as usize;
/// The depth-7 tree of `tree_verify_cost_bench.rs`: slot -> parent slot.
const BRANCHING: [i32; 8] = [-1, 0, 1, 2, 3, 0, 5, 2];

fn values(seed: u64, n: usize) -> Vec<f16> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            f16::from_f32(((s % 2000) as f32 / 1000.0 - 1.0) * 0.5)
        })
        .collect()
}

fn bytes(v: &[f16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect()
}

fn root_path(node: usize) -> Vec<u32> {
    let mut p = vec![node as u32];
    let mut at = BRANCHING[node];
    while at >= 0 {
        p.push(at as u32);
        at = BRANCHING[at as usize];
    }
    p.reverse();
    p
}

struct Fixture {
    context: MetalContext,
    scratch: AttentionScratch,
    q: MetalBuffer,
    out: MetalBuffer,
    scale: f32,
}

impl Fixture {
    fn new() -> Self {
        let context = MetalContext::new().expect("Metal device");
        let scratch = AttentionScratch::new(&context, Q_HEADS, HEAD_DIM)
            .with_decode_kernel(DecodeAttentionKernel::GqaShared);
        let q = context.new_buffer_with_data(&bytes(&values(0x51, Q_LEN)));
        let out = context.new_output_buffer((Q_LEN * 2) as u64);
        Self {
            context,
            scratch,
            q,
            out,
            scale: 1.0 / (HEAD_DIM as f32).sqrt(),
        }
    }

    fn chain(&mut self, k: &MetalBuffer, v: &MetalBuffer, seq_len: u32) -> Vec<u16> {
        let pass = self.context.begin_pass();
        encode_attention_decode(
            &mut self.context,
            &pass,
            (&self.q, 0),
            k,
            v,
            &self.scratch,
            (&self.out, 0),
            HEAD_DIM,
            Q_HEADS,
            KV_HEADS,
            seq_len,
            0,
            0,
            self.scale,
            None,
        )
        .unwrap();
        pass.commit_and_wait();
        self.read()
    }

    fn tree(&mut self, k: &MetalBuffer, v: &MetalBuffer, base: u32, path: &[u32]) -> Vec<u16> {
        let pass = self.context.begin_pass();
        let served = encode_attention_decode_tree(
            &mut self.context,
            &pass,
            (&self.q, 0),
            k,
            v,
            &self.scratch,
            (&self.out, 0),
            HEAD_DIM,
            Q_HEADS,
            KV_HEADS,
            0,
            base,
            base,
            path,
            0,
            self.scale,
        )
        .unwrap();
        assert!(served, "the GQA-shared scratch serves the tree");
        pass.commit_and_wait();
        self.read()
    }

    fn read(&self) -> Vec<u16> {
        read_buffer_f16(&self.out, 0, Q_LEN)
            .iter()
            .map(|x| x.to_bits())
            .collect()
    }
}

#[test]
fn the_identity_path_is_the_chain_row() {
    let mut f = Fixture::new();
    for base in [3u32, 291, 1500, 16_377, 16_385] {
        let total = base as usize + 8;
        let k = f
            .context
            .new_buffer_with_data(&bytes(&values(0xA0 + base as u64, total * ROW)));
        let v = f
            .context
            .new_buffer_with_data(&bytes(&values(0xB0 + base as u64, total * ROW)));
        for depth in [0usize, 3, 7] {
            let path: Vec<u32> = (0..=depth as u32).collect();
            let want = f.chain(&k, &v, base + depth as u32 + 1);
            let got = f.tree(&k, &v, base, &path);
            assert!(got == want, "base {base}, depth {depth}");
        }
    }
}

/// Per node: the tree over the branching slots against the chain over a
/// copy of the cache whose positions `base..` hold that node's root path.
/// Returns, per node, whether the tree's bits (or, with `ignore_parents`,
/// the identity map's) match.
fn branching_matches(f: &mut Fixture, base: u32, ignore_parents: bool) -> Vec<bool> {
    let total = base as usize + 8;
    let k_host = values(0xC1, total * ROW);
    let v_host = values(0xC2, total * ROW);
    let k = f.context.new_buffer_with_data(&bytes(&k_host));
    let v = f.context.new_buffer_with_data(&bytes(&v_host));
    (0..8)
        .map(|node| {
            let path = root_path(node);
            let lay = |host: &[f16]| {
                let mut out = host[..base as usize * ROW].to_vec();
                for &slot in &path {
                    let s = (base + slot) as usize * ROW;
                    out.extend_from_slice(&host[s..s + ROW]);
                }
                bytes(&out)
            };
            let (kc, vc) = (
                f.context.new_buffer_with_data(&lay(&k_host)),
                f.context.new_buffer_with_data(&lay(&v_host)),
            );
            let want = f.chain(&kc, &vc, base + path.len() as u32);
            let map: Vec<u32> = if ignore_parents {
                (0..path.len() as u32).collect()
            } else {
                path.clone()
            };
            f.tree(&k, &v, base, &map) == want
        })
        .collect()
}

#[test]
fn each_branching_node_reads_its_root_path() {
    let mut f = Fixture::new();
    for base in [40u32, 2_000] {
        let m = branching_matches(&mut f, base, false);
        assert!(m.iter().all(|&x| x), "base {base}: {m:?}");
    }
}

#[test]
fn the_identity_map_fails_the_branching_check_at_the_branches() {
    let mut f = Fixture::new();
    let m = branching_matches(&mut f, 40, true);
    assert_eq!(m, [true, true, true, true, true, false, false, false]);
}

/// The barrier override has no tree variant: the call reports it did not
/// serve, and the caller verifies the round as a chain.
#[test]
fn the_barrier_kernel_declines_the_tree() {
    let mut f = Fixture::new();
    let barrier = AttentionScratch::new(&f.context, Q_HEADS, HEAD_DIM)
        .with_decode_kernel(DecodeAttentionKernel::Barrier);
    let k = f.context.new_output_buffer((16 * ROW * 2) as u64);
    let pass = f.context.begin_pass();
    let served = encode_attention_decode_tree(
        &mut f.context,
        &pass,
        (&f.q, 0),
        &k,
        &k,
        &barrier,
        (&f.out, 0),
        HEAD_DIM,
        Q_HEADS,
        KV_HEADS,
        0,
        4,
        4,
        &[0, 1],
        0,
        f.scale,
    )
    .unwrap();
    pass.commit_and_wait();
    assert!(!served);
}
