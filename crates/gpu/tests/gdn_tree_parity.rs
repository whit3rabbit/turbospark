#![cfg(target_os = "macos")]
//! `encode_gdn_delta_prefill_lazy_tree` (`FC_GDN_TREE` on
//! `gdn_delta_step_prefill`) against the shipped lazy chain kernel, ported
//! from the bench prototype in `tree_verify_cost_bench.rs`:
//!
//! 1. with chain parents `(-1, 0, 1, ...)` its `y` and recorded `delta` are
//!    the shipped kernel's, bit for bit;
//! 2. on a branching tree each node's `y` and `delta` are the shipped
//!    kernel's over that node's root path, which is what makes the
//!    compacted accepted path's round-end fold the chain's;
//! 3. a kernel that ignored `parent` (computes the chain) fails (2), so the
//!    check can tell a tree from a chain.
//!
//! qwen38's linear-attention geometry; the key head dim is baked at 128
//! (the tree refuses anything else).

use half::f16;
use turbospark_compute::f32_to_bf16;
use turbospark_gpu::{
    encode_gdn_conv_prefill, encode_gdn_conv_prefill_tree, encode_gdn_delta_prefill_lazy,
    encode_gdn_delta_prefill_lazy_tree, read_buffer_f16, GdnShape, MetalBuffer, MetalContext,
};

const HK: usize = 16;
const HV: usize = 48;
const DK: usize = 128;
const DV: usize = 128;
const ROWS: usize = 8;
/// A best-first-shaped tree: a 4-deep spine, a runner-up at depth 0 with
/// its own child, and a sibling at depth 2. Row 0 is the anchor.
const BRANCHING: [i32; ROWS] = [-1, 0, 1, 2, 3, 0, 5, 2];

fn shape() -> GdnShape {
    GdnShape {
        num_k_heads: HK as u32,
        num_v_heads: HV as u32,
        key_head_dim: DK as u32,
        value_head_dim: DV as u32,
        conv_kernel_size: 4,
    }
}

fn deterministic(seed: u64, n: usize, scale: f32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (((state % 2000) as f32 / 1000.0) - 1.0) * scale
        })
        .collect()
}

fn half_bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|&x| f16::from_f32(x).to_bits().to_le_bytes())
        .collect()
}

fn bf16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|&x| f32_to_bf16(x).to_le_bytes())
        .collect()
}

struct HostRows {
    conv: Vec<f32>,
    a: Vec<f32>,
    b: Vec<f32>,
}

impl HostRows {
    fn new(rows: usize) -> Self {
        let c = shape().qkv_dim() as usize;
        Self {
            conv: deterministic(0xD1CE, rows * c, 0.25),
            a: deterministic(0xA000, rows * HV, 1.5),
            b: deterministic(0xB000, rows * HV, 1.0),
        }
    }

    fn gather(&self, path: &[usize]) -> HostRows {
        let c = shape().qkv_dim() as usize;
        let pick = |v: &[f32], w: usize| -> Vec<f32> {
            path.iter()
                .flat_map(|&r| v[r * w..(r + 1) * w].iter().copied())
                .collect()
        };
        HostRows {
            conv: pick(&self.conv, c),
            a: pick(&self.a, HV),
            b: pick(&self.b, HV),
        }
    }
}

struct Bufs {
    conv_out: MetalBuffer,
    a: MetalBuffer,
    b: MetalBuffer,
    a_log: MetalBuffer,
    dt_bias: MetalBuffer,
    state: MetalBuffer,
    y: MetalBuffer,
    tape: MetalBuffer,
    rows: usize,
}

impl Bufs {
    fn new(context: &MetalContext, host: &HostRows, rows: usize) -> Self {
        let s = shape();
        let state: Vec<u8> = deterministic(0x5EED, HV * DV * DK, 0.05)
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        Self {
            conv_out: context.new_buffer_with_data(&half_bytes(&host.conv)),
            a: context.new_buffer_with_data(&half_bytes(&host.a)),
            b: context.new_buffer_with_data(&half_bytes(&host.b)),
            a_log: context.new_buffer_with_data(&bf16_bytes(&deterministic(0xA100, HV, 1.25))),
            dt_bias: context.new_buffer_with_data(&bf16_bytes(&deterministic(0xA200, HV, 0.5))),
            state: context.new_buffer_with_data(&state),
            y: context.new_output_buffer((rows * s.value_dim() as usize * 2) as u64),
            tape: context.new_output_buffer((rows * s.value_dim() as usize * 4) as u64),
            rows,
        }
    }

    fn run(&self, context: &mut MetalContext, parent: Option<&[i32]>) {
        let pass = context.begin_pass();
        let args = (
            (&self.conv_out, 0u64),
            (&self.a, 0u64),
            (&self.b, 0u64),
            (&self.a_log, 0u64),
            (&self.dt_bias, 0u64),
        );
        match parent {
            Some(parent) => encode_gdn_delta_prefill_lazy_tree(
                context,
                &pass,
                shape(),
                args.0,
                args.1,
                args.2,
                args.3,
                args.4,
                &self.state,
                (&self.y, 0),
                (&self.tape, 0),
                parent,
            ),
            None => encode_gdn_delta_prefill_lazy(
                context,
                &pass,
                shape(),
                args.0,
                args.1,
                args.2,
                args.3,
                args.4,
                &self.state,
                (&self.y, 0),
                (&self.tape, 0),
                self.rows as u32,
            ),
        }
        .expect("gdn lazy prefill encodes");
        pass.commit_and_wait();
    }

    /// Row `row`'s `y` (fp16 bits) and recorded `delta` (fp32 bits).
    fn row(&self, row: usize) -> (Vec<u16>, Vec<u32>) {
        let w = shape().value_dim() as usize;
        let y = read_buffer_f16(&self.y, row * w * 2, w)
            .iter()
            .map(|v| v.to_bits())
            .collect();
        let ptr = self.tape.contents() as *const u32;
        // SAFETY: shared-storage tape of `rows * w` f32, waited on above.
        let tape = unsafe { std::slice::from_raw_parts(ptr.add(row * w), w) }.to_vec();
        (y, tape)
    }
}

fn path(parent: &[i32], node: usize) -> Vec<usize> {
    let mut p = vec![node];
    let mut at = parent[node];
    while at >= 0 {
        p.push(at as usize);
        at = parent[at as usize];
    }
    p.reverse();
    p
}

/// For each node, whether the tree run's row equals the chain kernel over
/// that node's root path.
fn root_path_matches(context: &mut MetalContext, host: &HostRows, tree: &Bufs) -> Vec<bool> {
    (0..ROWS)
        .map(|node| {
            let rows = path(&BRANCHING, node);
            let reference = Bufs::new(context, &host.gather(&rows), rows.len());
            reference.run(context, None);
            reference.row(rows.len() - 1) == tree.row(node)
        })
        .collect()
}

#[test]
fn chain_parents_are_the_shipped_kernel() {
    let mut context = MetalContext::new().expect("Metal device");
    let host = HostRows::new(ROWS);
    let chain: Vec<i32> = (0..ROWS as i32).map(|t| t - 1).collect();
    let shipped = Bufs::new(&context, &host, ROWS);
    shipped.run(&mut context, None);
    let tree = Bufs::new(&context, &host, ROWS);
    tree.run(&mut context, Some(&chain));
    for r in 0..ROWS {
        assert!(shipped.row(r) == tree.row(r), "row {r}");
    }
}

#[test]
fn each_node_is_the_shipped_kernel_over_its_root_path() {
    let mut context = MetalContext::new().expect("Metal device");
    let host = HostRows::new(ROWS);
    let tree = Bufs::new(&context, &host, ROWS);
    tree.run(&mut context, Some(&BRANCHING));
    let matches = root_path_matches(&mut context, &host, &tree);
    assert!(matches.iter().all(|&m| m), "per node: {matches:?}");
}

/// What a parent-ignoring kernel computes is the chain, so run the chain
/// and hand it to the same check: it must fail on the nodes whose parent
/// is not the row before (5, 6, 7), and pass on the spine (0-4). That the
/// spine still passes is the evidence the mutation is the parent and only
/// the parent.
#[test]
fn a_parent_ignoring_kernel_fails_the_root_path_check() {
    let mut context = MetalContext::new().expect("Metal device");
    let host = HostRows::new(ROWS);
    let ignoring = Bufs::new(&context, &host, ROWS);
    ignoring.run(&mut context, None);
    let matches = root_path_matches(&mut context, &host, &ignoring);
    assert_eq!(
        matches,
        [true, true, true, true, true, false, false, false],
        "the check must separate a tree from a chain exactly at the branches"
    );
}

#[test]
fn the_tree_refuses_what_its_registers_cannot_hold() {
    let mut context = MetalContext::new().expect("Metal device");
    let host = HostRows::new(9);
    let bufs = Bufs::new(&context, &host, 9);
    let pass = context.begin_pass();
    let nine: Vec<i32> = (0..9).map(|t| t - 1).collect();
    for parent in [&nine[..], &[0, 0][..], &[-1, 1][..], &[][..]] {
        let err = encode_gdn_delta_prefill_lazy_tree(
            &mut context,
            &pass,
            shape(),
            (&bufs.conv_out, 0),
            (&bufs.a, 0),
            (&bufs.b, 0),
            (&bufs.a_log, 0),
            (&bufs.dt_bias, 0),
            &bufs.state,
            (&bufs.y, 0),
            (&bufs.tape, 0),
            parent,
        );
        assert!(err.is_err(), "parent {parent:?} must be refused");
    }
    pass.commit_and_wait();
}

/// The conv half: raw qkv rows through `gdn_conv_mix_prefill`'s tree build
/// against its chain build, with a live tail (the committed history).
struct Conv {
    tail: MetalBuffer,
    weight: MetalBuffer,
}

impl Conv {
    fn new(context: &MetalContext) -> Self {
        let c = shape().qkv_dim() as usize;
        Self {
            tail: context.new_buffer_with_data(&half_bytes(&deterministic(0x7A11, 3 * c, 1.0))),
            weight: context.new_buffer_with_data(&bf16_bytes(&deterministic(0xC0DE, 4 * c, 0.5))),
        }
    }

    /// `rows` raw rows through the chain (`parent` None) or the tree build.
    fn run(&self, context: &mut MetalContext, raw: &[f32], parent: Option<&[i32]>) -> Vec<u16> {
        let c = shape().qkv_dim() as usize;
        let rows = raw.len() / c;
        let qkv = context.new_buffer_with_data(&half_bytes(raw));
        let out = context.new_output_buffer((rows * c * 2) as u64);
        let pass = context.begin_pass();
        match parent {
            Some(parent) => encode_gdn_conv_prefill_tree(
                context,
                &pass,
                shape(),
                (&self.tail, 0),
                (&qkv, 0),
                (&self.weight, 0),
                (&out, 0),
                parent,
            ),
            None => encode_gdn_conv_prefill(
                context,
                &pass,
                shape(),
                (&self.tail, 0),
                (&qkv, 0),
                (&self.weight, 0),
                (&out, 0),
                rows as u32,
                1,
            ),
        }
        .expect("conv encodes");
        pass.commit_and_wait();
        read_buffer_f16(&out, 0, rows * c)
            .iter()
            .map(|v| v.to_bits())
            .collect()
    }
}

#[test]
fn the_tree_conv_with_chain_parents_is_the_chain_conv() {
    let mut context = MetalContext::new().expect("Metal device");
    let conv = Conv::new(&context);
    let raw = deterministic(0x9A77, ROWS * shape().qkv_dim() as usize, 1.0);
    let chain: Vec<i32> = (0..ROWS as i32).map(|t| t - 1).collect();
    assert!(conv.run(&mut context, &raw, None) == conv.run(&mut context, &raw, Some(&chain)));
}

/// Each node's conv output is the chain conv's over its root path with the
/// same tail, which the conv-order parity test already holds to decode. A
/// parent-ignoring build (the chain) must fail on the branch nodes.
#[test]
fn each_tree_conv_node_is_the_chain_conv_over_its_root_path() {
    let mut context = MetalContext::new().expect("Metal device");
    let conv = Conv::new(&context);
    let c = shape().qkv_dim() as usize;
    let raw = deterministic(0x9A77, ROWS * c, 1.0);
    let tree = conv.run(&mut context, &raw, Some(&BRANCHING));
    let chain = conv.run(&mut context, &raw, None);
    let mut chain_matches = Vec::new();
    for node in 0..ROWS {
        let rows = path(&BRANCHING, node);
        let gathered: Vec<f32> = rows
            .iter()
            .flat_map(|&r| raw[r * c..(r + 1) * c].iter().copied())
            .collect();
        let reference = conv.run(&mut context, &gathered, None);
        let last = &reference[(rows.len() - 1) * c..];
        assert!(
            last == &tree[node * c..(node + 1) * c],
            "node {node} on path {rows:?}"
        );
        chain_matches.push(last == &chain[node * c..(node + 1) * c]);
    }
    assert_eq!(
        chain_matches,
        [true, true, true, true, true, false, false, false],
        "the check must separate a tree conv from a chain conv at the branches"
    );
}
