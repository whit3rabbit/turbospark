#![cfg(target_os = "macos")]
//! Parity for the port-local batched routed-expert pair over MXFP4 expert
//! blobs (`moe_prefill_batch_gguf.metal`, `docs/BATCHED_PREFILL.md` step 5).
//!
//! **The bar is BIT-IDENTITY with the MXFP4 DECODE pair run M times**
//! (`moe_phase1_gate_up_act_mxfp4` + `moe_phase2_down_reduce_k8_mxfp4`),
//! the same bar `moe_prefill_batch_parity.rs` holds for the affine pair.
//! A tolerance check against the CPU reference is NOT repeated here:
//! `moe_gguf_parity.rs` already holds both halves of that contract (the
//! MXFP4 unpack against `dequantize_mxfp4`, and the clamped SwiGLU against
//! ggml's `ggml_compute_forward_swiglu_oai_f32` written out from source),
//! and this file's kernels call those same MSL helpers. What only this
//! file can see is whether BATCHING them preserves the decode pair's bits.
//!
//! Four properties the affine sibling does not have to check, because
//! MXFP4 carries the family's expert MATH as well as its block layout:
//!
//!  - the per-expert gate/up biases are added BEFORE the activation, where
//!    the clamp reads them (`min(gate + b, limit)`, not
//!    `min(gate, limit) + b`);
//!  - the per-expert DOWN bias is added per slot INSIDE the routing weight,
//!    because it is that expert's own bias on that expert's own output;
//!  - `alpha` and `limit` reach the kernel at all, which
//!    `a_plain_activation_is_a_different_function` pins -- at
//!    `Mxfp4Activation::PLAIN` the kernel takes a different branch entirely,
//!    so a fixture that only ever ran `GPT_OSS` could not tell the uniform
//!    from a hardcoded constant;
//!  - and M=16 actually runs, which is this arm's whole reason to exist
//!    (`gpt-oss`'s 32 experts at top-4 keep its union under the slot count
//!    at every M, where both 128-expert families cap at M=8).
//!
//! Two permanent MUTATION cases make the test self-validating: permuting a
//! token's routing weights across ranks, and dropping the last route, must
//! each make the comparison FAIL.
//!
//! **Hand mutation-checked 2026-08-27, four kernel edits, and WHICH cases
//! each reddens is the part worth keeping** -- a mutation that reddens
//! everything says less than one that reddens exactly the cases able to see
//! it:
//!
//!  - reversing phase 2's rank-order accumulation reddens the REAL-SHAPE
//!    case and the plain-activation case, and NOT the two small `GPT_OSS`
//!    ones, whose shorter sums round identically under reordering. So the
//!    real-shape case is this file's order-sensitivity sentinel exactly as
//!    it is the affine sibling's, and weakening it weakens the Gotcha 27
//!    guard;
//!  - clamping the gate BEFORE adding its bias (`min(gate, limit) + b`, the
//!    plausible wrong order) reddens all three `GPT_OSS` parity cases and
//!    NEITHER `PLAIN` one -- correct, since `PLAIN` binds no bias and cannot
//!    see the question;
//!  - hoisting the down bias out of the routing weight reddens the same
//!    three and no others;
//!  - dropping the token stride from phase 1's `acts` index reddens all four
//!    parity cases, being a batching bug rather than an MXFP4 one.

use half::f16;
use metal::Buffer;
use turbospark_gpu::{
    bind_routed_blobs_wide_mxfp4, encode_moe_phase1_mxfp4, encode_moe_phase2_mxfp4,
    encode_moe_prefill_phase1_mxfp4, encode_moe_prefill_phase2_fused_mxfp4,
    new_routed_blobs_wide_mxfp4, MetalContext, MoeExpertOffsets, MoePrefillRoute, Mxfp4Activation,
    RoutedBlobsBuffer, RoutedBlobsWideBuffer, MAX_STREAMED_EXPERTS,
};

/// Random valid MXFP4 bytes at a fixed E8M0 exponent.
///
/// `e = 124` is `2^-4` and the codebook tops out at 12, so the largest
/// representable weight is 0.75 -- `moe_gguf_parity.rs`'s choice and its
/// reasoning, which is the dynamic-range trap: this chain SQUARES its
/// inputs (gate times up, then a second GEMV), so a fixture scaled for a
/// single GEMV produces `inf` on both sides and reads as a kernel bug.
fn mxfp4_row(cols: usize, seed: u64) -> Vec<u8> {
    assert_eq!(cols % 32, 0);
    let mut state = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    let mut byte = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        (state >> 33) as u8
    };
    // A one-byte slice rather than `push`, only because
    // `clippy::same_item_push` reads a constant push in a loop as a mistake.
    // The block header really is one fixed byte per block.
    const EXPONENT: [u8; 1] = [124];
    let mut out = Vec::with_capacity(cols / 32 * 17);
    for _ in 0..cols / 32 {
        out.extend_from_slice(&EXPONENT);
        for _ in 0..16 {
            out.push(byte());
        }
    }
    out
}

fn mxfp4_rows(rows: usize, cols: usize, seed: u64) -> Vec<Vec<u8>> {
    (0..rows)
        .map(|r| mxfp4_row(cols, seed.wrapping_add(r as u64 * 97 + 1)))
        .collect()
}

/// Biases spanning the clamp in BOTH directions, so `min(gate + b, limit)`
/// and `clamp(up + b, -limit, limit)` are exercised rather than inert. A
/// bias small against the weights would leave the clamp untaken and the
/// before-or-after-activation question unobservable.
fn bias_plane(n: usize, seed: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32) * 0.37 + seed).sin() * 9.0)
        .collect()
}

fn to_le(v: &[f16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 2);
    for x in v {
        out.extend_from_slice(&x.to_bits().to_le_bytes());
    }
    out
}

fn read_f16_bits(buffer: &Buffer, n: usize) -> Vec<u16> {
    let ptr = buffer.contents() as *const u16;
    unsafe { std::slice::from_raw_parts(ptr, n) }.to_vec()
}

struct Fixture {
    context: MetalContext,
    blobs: Vec<Buffer>,
    offsets: MoeExpertOffsets,
    x: Vec<f16>,
    routes: Vec<MoePrefillRoute>,
    weights: Vec<f32>,
    residual: Vec<f16>,
    d: usize,
    f: usize,
    top_k: usize,
    tokens: usize,
    act: Mxfp4Activation,
}

impl Fixture {
    fn new(
        d: usize,
        f: usize,
        top_k: usize,
        tokens: usize,
        experts: usize,
        act: Mxfp4Activation,
    ) -> Self {
        let context = MetalContext::new().expect("Metal device");
        let mut blobs = Vec::new();
        let mut offsets = None;
        for e in 0..experts {
            let seed = 9000 + 137 * e as u64;
            let gate = mxfp4_rows(f, d, seed + 1);
            let up = mxfp4_rows(f, d, seed + 2);
            let down = mxfp4_rows(d, f, seed + 3);

            let mut blob = Vec::new();
            let mut off = MoeExpertOffsets {
                gate_w: 0,
                gate_s: 0,
                gate_b: 0,
                up_w: 0,
                up_s: 0,
                up_b: 0,
                down_w: 0,
                down_s: 0,
                down_b: 0,
            };
            let push_rows = |rows: &[Vec<u8>], blob: &mut Vec<u8>| -> u32 {
                let at = blob.len() as u32;
                for r in rows {
                    blob.extend_from_slice(r);
                }
                at
            };
            off.gate_w = push_rows(&gate, &mut blob);
            off.up_w = push_rows(&up, &mut blob);
            off.down_w = push_rows(&down, &mut blob);
            // The bias planes as the repack walk packs `ffn_*_exps.bias`:
            // F32, per expert, contiguous. Written whatever `act.has_bias`
            // says, so the two arms differ ONLY in the uniform.
            let push_f32 = |v: &[f32], blob: &mut Vec<u8>| -> u32 {
                let at = blob.len() as u32;
                for x in v {
                    blob.extend_from_slice(&x.to_le_bytes());
                }
                at
            };
            off.gate_b = push_f32(&bias_plane(f, e as f32), &mut blob);
            off.up_b = push_f32(&bias_plane(f, e as f32 + 11.0), &mut blob);
            off.down_b = push_f32(&bias_plane(d, e as f32 + 23.0), &mut blob);

            offsets.get_or_insert(off);
            blobs.push(context.new_buffer_with_data(&blob));
        }

        let x: Vec<f16> = (0..tokens * d)
            .map(|i| f16::from_f32(((i as f32) * 0.17).sin() * 0.8))
            .collect();
        let mut routes = Vec::new();
        let mut weights = Vec::new();
        for t in 0..tokens {
            for r in 0..top_k {
                let expert = (t * top_k + r) % experts;
                routes.push(MoePrefillRoute {
                    token: t as u32,
                    rank: r as u32,
                    slot: expert as u32,
                });
                weights.push(0.05 + 0.09 * (((t * top_k + r) % 7) as f32));
            }
        }
        Self {
            context,
            blobs,
            offsets: offsets.unwrap(),
            x,
            routes,
            weights,
            residual: vec![f16::from_f32(0.0); tokens * d],
            d,
            f,
            top_k,
            tokens,
            act,
        }
    }

    /// The ORACLE: the MXFP4 decode pair, once per token, blobs bound in
    /// router ranking order exactly as the per-token routed loop binds them.
    fn decode_pair_sequential(&mut self) -> (Vec<u16>, Vec<u16>) {
        let (d, f, top_k, tokens) = (self.d, self.f, self.top_k, self.tokens);
        let mut acts_bits = vec![0u16; tokens * top_k * f];
        let mut y_bits = vec![0u16; tokens * d];

        let x_buf = self.context.new_buffer_with_data(&to_le(&self.x));
        // ZERO-FILLED, not merely allocated: phase 2 reduces all eight
        // slots unconditionally, so the padded ranks' acts rows must be
        // FINITE (crates/gpu Gotcha 3).
        let acts_buf = self
            .context
            .new_buffer_with_data(&vec![0u8; MAX_STREAMED_EXPERTS * f * 2]);
        let residual_buf = self.context.new_buffer_with_data(&to_le(&self.residual));
        let y_buf = self.context.new_output_buffer((d * 2) as u64);
        let routed = RoutedBlobsBuffer::new(&mut self.context, false).expect("arg buffer");

        for t in 0..tokens {
            let row: Vec<(usize, f32)> = (0..top_k)
                .map(|r| {
                    let route = self.routes[t * top_k + r];
                    (route.slot as usize, self.weights[t * top_k + r])
                })
                .collect();
            let mut routing16 = vec![f16::from_f32(0.0); MAX_STREAMED_EXPERTS];
            for (dispatch_slot, &(_, weight)) in row.iter().enumerate() {
                routing16[dispatch_slot] = f16::from_f32(weight);
            }
            let routing_buf = self.context.new_buffer_with_data(&to_le(&routing16));
            let blob_refs: Vec<(&Buffer, u64)> = row
                .iter()
                .map(|&(slot, _)| (&self.blobs[slot], 0u64))
                .collect();
            routed
                .bind(&mut self.context, false, &blob_refs)
                .expect("bind blobs");

            let pass = self.context.begin_pass();
            for &(slot, _) in &row {
                pass.use_read_buffer(&self.blobs[slot]);
            }
            encode_moe_phase1_mxfp4(
                &mut self.context,
                &pass,
                &routed,
                &self.offsets,
                (&x_buf, (t * d * 2) as u64),
                (&acts_buf, 0),
                d as u32,
                f as u32,
                top_k as u32,
                false,
                self.act,
            )
            .expect("decode phase1");
            encode_moe_phase2_mxfp4(
                &mut self.context,
                &pass,
                &routed,
                &self.offsets,
                (&acts_buf, 0),
                (&routing_buf, 0),
                (&residual_buf, (t * d * 2) as u64),
                (&y_buf, 0),
                d as u32,
                f as u32,
                false,
                self.act.has_bias,
            )
            .expect("decode phase2");
            pass.commit_and_wait();

            acts_bits[t * top_k * f..(t + 1) * top_k * f]
                .copy_from_slice(&read_f16_bits(&acts_buf, MAX_STREAMED_EXPERTS * f)[..top_k * f]);
            y_bits[t * d..(t + 1) * d].copy_from_slice(&read_f16_bits(&y_buf, d));
        }
        (acts_bits, y_bits)
    }

    /// The batched pair over the whole route list, one dispatch per kernel.
    fn batched_pair(
        &mut self,
        weights_override: Option<Vec<f32>>,
        route_count_override: Option<usize>,
    ) -> (Vec<u16>, Vec<u16>) {
        let (d, f, top_k, tokens) = (self.d, self.f, self.top_k, self.tokens);
        let weights = weights_override.unwrap_or_else(|| self.weights.clone());
        let route_count = route_count_override.unwrap_or(self.routes.len());

        let x_buf = self.context.new_buffer_with_data(&to_le(&self.x));
        let acts_buf = self
            .context
            .new_buffer_with_data(&vec![0u8; tokens * top_k * f * 2]);
        let y_buf = self.context.new_output_buffer((tokens * d * 2) as u64);
        let routing16: Vec<f16> = weights.iter().map(|&w| f16::from_f32(w)).collect();
        let routing_buf = self.context.new_buffer_with_data(&to_le(&routing16));
        let routes_buf = self
            .context
            .new_buffer_with_data(&MoePrefillRoute::bytes(&self.routes));
        let residual_buf = self.context.new_buffer_with_data(&to_le(&self.residual));

        let routed: RoutedBlobsWideBuffer =
            new_routed_blobs_wide_mxfp4(&mut self.context, false).expect("arg buffer");
        let blob_refs: Vec<(&Buffer, u64)> = self.blobs.iter().map(|b| (b, 0u64)).collect();
        bind_routed_blobs_wide_mxfp4(&routed, &mut self.context, false, &blob_refs)
            .expect("bind blobs");

        let pass = self.context.begin_pass();
        for blob in &self.blobs {
            pass.use_read_buffer(blob);
        }
        encode_moe_prefill_phase1_mxfp4(
            &mut self.context,
            &pass,
            &routed,
            &self.offsets,
            (&x_buf, 0),
            (&acts_buf, 0),
            (&routes_buf, 0),
            d as u32,
            f as u32,
            top_k as u32,
            route_count as u32,
            false,
            self.act,
        )
        .expect("batched phase1");
        encode_moe_prefill_phase2_fused_mxfp4(
            &mut self.context,
            &pass,
            &routed,
            &self.offsets,
            (&acts_buf, 0),
            (&routing_buf, 0),
            (&routes_buf, 0),
            (&residual_buf, 0),
            (&y_buf, 0),
            d as u32,
            f as u32,
            top_k as u32,
            tokens as u32,
            false,
            self.act.has_bias,
        )
        .expect("batched phase2");
        pass.commit_and_wait();

        (
            read_f16_bits(&acts_buf, tokens * top_k * f),
            read_f16_bits(&y_buf, tokens * d),
        )
    }
}

fn assert_bits_equal(name: &str, got: &[u16], want: &[u16]) {
    assert_eq!(got.len(), want.len(), "{name}: length");
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        assert_eq!(
            g, w,
            "{name}: element {i} differs (got {g:#06x} want {w:#06x})"
        );
    }
}

fn assert_bits_differ(name: &str, got: &[u16], want: &[u16]) {
    let differs = got.iter().zip(want.iter()).any(|(&g, &w)| g != w);
    assert!(
        differs,
        "{name}: mutation produced IDENTICAL output; the oracle cannot see it"
    );
}

/// Ragged routes over a 12-expert union at `gpt-oss`'s top-4, with the
/// family's activation and all three bias planes live: the wide binding,
/// the route list, the padded ranks and the rank-ordered fused reduce, all
/// bit-exact against M sequential decode passes.
#[test]
fn batched_mxfp4_pair_is_bit_identical_to_sequential_decode_pair() {
    let mut fixture = Fixture::new(128, 128, 4, 6, 12, Mxfp4Activation::GPT_OSS);
    let (want_acts, want_y) = fixture.decode_pair_sequential();
    let (got_acts, got_y) = fixture.batched_pair(None, None);
    assert_bits_equal("acts", &got_acts, &want_acts);
    assert_bits_equal("y", &got_y, &want_y);
}

/// **M=16, which is the reason this arm was built first.** `gpt-oss` has 32
/// experts at top-4, so its measured union stays under the slot count at
/// every M and the binding constraint is `MAX_BATCH_ROWS` rather than
/// `union(M) <= slot_count`. Both 128-expert families cap at M=8, so no
/// other batched-routed case in this repo reaches this width.
#[test]
fn the_batched_mxfp4_pair_runs_at_the_full_batch_width() {
    let mut fixture = Fixture::new(128, 64, 4, 16, 12, Mxfp4Activation::GPT_OSS);
    let (want_acts, want_y) = fixture.decode_pair_sequential();
    let (got_acts, got_y) = fixture.batched_pair(None, None);
    assert_bits_equal("acts", &got_acts, &want_acts);
    assert_bits_equal("y", &got_y, &want_y);
}

/// The REAL `gpt-oss` routed shape (D=2880, F=2880, top-4), which is where a
/// shape-dependent indexing bug lives and where the reduce is long enough
/// for a reordering to be visible at all -- the affine sibling records that
/// its small cases round identically under a reversed reduce, so the
/// real-shape case is the order-sensitivity sentinel. An edit that weakens
/// this case weakens the Gotcha 27 guard.
#[test]
fn batched_mxfp4_pair_is_bit_identical_at_the_real_gptoss_shape() {
    let mut fixture = Fixture::new(2880, 2880, 4, 3, 5, Mxfp4Activation::GPT_OSS);
    let (want_acts, want_y) = fixture.decode_pair_sequential();
    let (got_acts, got_y) = fixture.batched_pair(None, None);
    assert_bits_equal("acts", &got_acts, &want_acts);
    assert_bits_equal("y", &got_y, &want_y);
}

/// `Mxfp4Activation::PLAIN` takes the other branch of `moe_activate_mxfp4`
/// (`alpha <= 0` means "not gpt-oss") and binds no bias, so it is a
/// genuinely different function reaching the same kernel. Running only
/// `GPT_OSS` above would leave the uniforms indistinguishable from
/// hardcoded constants.
#[test]
fn the_batched_mxfp4_pair_is_bit_identical_under_the_plain_activation() {
    let mut fixture = Fixture::new(128, 128, 4, 6, 12, Mxfp4Activation::PLAIN);
    let (want_acts, want_y) = fixture.decode_pair_sequential();
    let (got_acts, got_y) = fixture.batched_pair(None, None);
    assert_bits_equal("acts", &got_acts, &want_acts);
    assert_bits_equal("y", &got_y, &want_y);
}

/// THE FIXTURE MUST DISCRIMINATE BEFORE THE FOUR CASES ABOVE MEAN ANYTHING.
/// The blobs carry bias planes either way, so if `has_bias` and the
/// activation parameters did NOT reach the kernel, the `GPT_OSS` and
/// `PLAIN` runs would agree and both parity cases would pass against a
/// kernel that ignored its uniforms entirely -- the trap AGENTS.md Gotchas
/// 48 and 50 record on two other axes. Same routes, same weights, same
/// activations, same bytes; only the uniform differs, and the outputs must
/// DIFFER.
#[test]
fn a_plain_activation_is_a_different_function() {
    let mut gpt_oss = Fixture::new(128, 128, 4, 6, 12, Mxfp4Activation::GPT_OSS);
    let (_, gpt_oss_y) = gpt_oss.batched_pair(None, None);
    let mut plain = Fixture::new(128, 128, 4, 6, 12, Mxfp4Activation::PLAIN);
    let (_, plain_y) = plain.batched_pair(None, None);
    assert_bits_differ("activation uniform", &gpt_oss_y, &plain_y);
}

/// MUTATION 1: swapping one token's routing weights across ranks changes
/// the rank-ordered reduce, so the comparison must fail.
#[test]
fn mutating_rank_weights_breaks_mxfp4_parity() {
    let mut fixture = Fixture::new(128, 128, 4, 6, 12, Mxfp4Activation::GPT_OSS);
    let (.., want_y) = fixture.decode_pair_sequential();
    let mut weights = fixture.weights.clone();
    weights.swap(2 * fixture.top_k, 2 * fixture.top_k + 1);
    let (.., got_y) = fixture.batched_pair(Some(weights), None);
    assert_bits_differ("y", &got_y, &want_y);
}

/// MUTATION 2: dropping the final route leaves its acts row unwritten
/// (zero), so the last token's output must differ.
#[test]
fn dropping_a_route_breaks_mxfp4_parity() {
    let mut fixture = Fixture::new(128, 128, 4, 6, 12, Mxfp4Activation::GPT_OSS);
    let (.., want_y) = fixture.decode_pair_sequential();
    let (.., got_y) = fixture.batched_pair(None, Some(fixture.routes.len() - 1));
    assert_bits_differ("y", &got_y, &want_y);
}
