#![cfg(target_os = "macos")]
//! Parity for the port-local batched routed-expert pair
//! (`moe_prefill_batch.metal`, `docs/BATCHED_PREFILL.md` steps 2 and 3).
//!
//! **The bar is BIT-IDENTITY with the vendored decode pair run M times**,
//! not tolerance against a CPU reference (that check is here too, for the
//! small case, because both GPU paths share the same MSL row helpers and
//! a shared bug there is invisible to a GPU-vs-GPU comparison). Ragged
//! routes exercise the wide binding (union above eight experts, experts
//! shared across tokens) and a `top_k < 8` case exercises the padded-rank
//! path.
//!
//! Two permanent MUTATION cases make the test self-validating: permuting
//! a token's routing weights across ranks, and dropping the last route,
//! must each make the comparison FAIL. If either mutation passes, the
//! oracle is not looking at what it claims to.
//!
//! Mutation-checked 2026-08-18 by hand as well: reversing the kernel's
//! rank-order accumulation flips exactly ONE bit in one token of the
//! REAL-SHAPE case and nothing in the small cases, whose 8-term sums
//! happen to round identically under reordering -- the real-shape case
//! is the order-sensitivity sentinel, and a future edit that weakens it
//! weakens the Gotcha 27 guard.
//!
//! **THE RESIDUAL SEED HAS THE SAME ASYMMETRY, measured 2026-08-21 when
//! it was added, and it is why there are TWO shared-expert cases rather
//! than one.** Moving the seed from the front of the reduce to the end --
//! the difference between what `qwen3_5`'s decode kernel does and what a
//! driver appending a residual add would do -- reddens
//! `a_shared_expert_residual_is_bit_identical_at_the_real_shape` and
//! NOTHING else, the 128-wide case included. So the small case cannot see
//! the property the seed exists for, and a future edit that drops the
//! real-shape case would leave the summation-order claim untested while
//! eight tests still passed. Three mutations were run: hardcoding the old
//! `0.0f` seed (reddens all three new cases and none of the six older
//! ones), appending instead of seeding (reddens the real-shape case
//! alone), and dropping the token stride from the residual index (reddens
//! both parity cases, not the discriminator).

use half::f16;
use metal::Buffer;
use turbospark_compute::quant::Int4AffineRow;
use turbospark_gpu::{
    encode_moe_phase1, encode_moe_phase2, encode_moe_prefill_phase1,
    encode_moe_prefill_phase2_fused, MoeExpertOffsets, MoePrefillRoute, RoutedBlobsBuffer,
    RoutedBlobsWideBuffer, MAX_STREAMED_EXPERTS,
};

fn deterministic_row(seed: u64, n: usize) -> Vec<f32> {
    let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(0x9E37_79B9);
    (0..n)
        .map(|i| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state = state.wrapping_add(i as u64);
            ((state % 2000) as f32 / 1000.0) - 1.0
        })
        .collect()
}

fn quantized_rows(rows: usize, cols: usize, seed: u64) -> Vec<Int4AffineRow> {
    (0..rows)
        .map(|r| {
            let row = deterministic_row(seed.wrapping_add(r as u64 * 97 + 1), cols);
            turbospark_compute::quantize_int4_affine(&row)
        })
        .collect()
}

fn u16_le(v: &[u16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 2);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

fn build_blob(
    gate: &[Int4AffineRow],
    up: &[Int4AffineRow],
    down: &[Int4AffineRow],
) -> (Vec<u8>, MoeExpertOffsets) {
    let mut blob = Vec::new();
    let mut offsets = [0u32; 9];
    let mut push = |i: usize, bytes: Vec<u8>, blob: &mut Vec<u8>| {
        offsets[i] = blob.len() as u32;
        blob.extend_from_slice(&bytes);
    };
    for (base, rows) in [(0usize, gate), (3, up), (6, down)] {
        let packed: Vec<u8> = rows.iter().flat_map(|r| r.packed.clone()).collect();
        let scales: Vec<u16> = rows.iter().flat_map(|r| r.scales.clone()).collect();
        let biases: Vec<u16> = rows.iter().flat_map(|r| r.biases.clone()).collect();
        push(base, packed, &mut blob);
        push(base + 1, u16_le(&scales), &mut blob);
        push(base + 2, u16_le(&biases), &mut blob);
    }
    (
        blob,
        MoeExpertOffsets {
            gate_w: offsets[0],
            gate_s: offsets[1],
            gate_b: offsets[2],
            up_w: offsets[3],
            up_s: offsets[4],
            up_b: offsets[5],
            down_w: offsets[6],
            down_s: offsets[7],
            down_b: offsets[8],
        },
    )
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
    context: turbospark_gpu::MetalContext,
    blobs: Vec<Buffer>,
    offsets: MoeExpertOffsets,
    /// `[tokens, D]` activations.
    x: Vec<f16>,
    /// `routes[t * top_k + r] = (t, r, expert)` with distinct experts
    /// per token, in router ranking (rank) order.
    routes: Vec<MoePrefillRoute>,
    /// `weights[t * top_k + r]`, the router weights behind those ranks.
    weights: Vec<f32>,
    /// `[tokens, D]`, phase 2's accumulator SEED. Zero for the families
    /// that add their shared expert elsewhere (Gemma, `llama`, `gpt-oss`);
    /// `qwen3_5`'s gated shared-expert output otherwise. Both sides of the
    /// comparison read it, so a non-zero one is what makes the seeding
    /// order observable at all -- see
    /// `the_residual_seed_is_what_a_zero_fixture_cannot_see`.
    residual: Vec<f16>,
    d: usize,
    f: usize,
    top_k: usize,
    tokens: usize,
}

impl Fixture {
    /// `experts` blobs; token t routes to experts `(t * top_k + r) *
    /// stride % experts`, distinct within a token whenever `experts >=
    /// top_k`, and the union exceeds eight for `experts > 8` -- the wide
    /// binding's reason to exist.
    fn new(d: usize, f: usize, top_k: usize, tokens: usize, experts: usize) -> Self {
        let context = turbospark_gpu::MetalContext::new().expect("Metal device");
        let mut blobs = Vec::new();
        let mut offsets = None;
        for e in 0..experts {
            let seed = 9000 + 137 * e as u64;
            let (blob, off) = build_blob(
                &quantized_rows(f, d, seed + 1),
                &quantized_rows(f, d, seed + 2),
                &quantized_rows(d, f, seed + 3),
            );
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
        }
    }

    /// Give phase 2 a NON-ZERO accumulator seed, the shape `qwen3_5`'s
    /// gated shared expert has. Deliberately spans several binades and
    /// both signs: a seed all of one magnitude cannot show that it is
    /// added FIRST rather than last, which is the only thing the seeding
    /// changes.
    fn with_residual(mut self) -> Self {
        let row = deterministic_row(0xD1CE, self.tokens * self.d);
        self.residual = row
            .iter()
            .enumerate()
            .map(|(i, &v)| f16::from_f32(v * if i % 3 == 0 { 8.0 } else { 0.125 }))
            .collect();
        self
    }

    /// The ORACLE: the vendored decode pair, once per token, blobs bound
    /// in router ranking order exactly as `families/gemma4/moe.rs` binds
    /// them. Returns `(acts_bits, y_bits)` -- `acts` laid out per token
    /// (`tokens * top_k * F`), `y` laid out per token (`tokens * D`).
    fn decode_pair_sequential(&mut self) -> (Vec<u16>, Vec<u16>) {
        let (d, f, top_k, tokens) = (self.d, self.f, self.top_k, self.tokens);
        let mut acts_bits = vec![0u16; tokens * top_k * f];
        let mut y_bits = vec![0u16; tokens * d];

        let x_buf = self.context.new_buffer_with_data(&to_le(&self.x));
        let acts_buf = self
            .context
            .new_output_buffer((MAX_STREAMED_EXPERTS * f * 2) as u64);
        // The decode kernel takes ONE row; the batched one takes all M.
        // Feeding the same values through both is what makes the seed's
        // position in the sum comparable.
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
            encode_moe_phase1(
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
            )
            .expect("decode phase1");
            encode_moe_phase2(
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
            )
            .expect("decode phase2");
            pass.commit_and_wait();

            acts_bits[t * top_k * f..(t + 1) * top_k * f]
                .copy_from_slice(&read_f16_bits(&acts_buf, MAX_STREAMED_EXPERTS * f)[..top_k * f]);
            y_bits[t * d..(t + 1) * d].copy_from_slice(&read_f16_bits(&y_buf, d));
        }
        (acts_bits, y_bits)
    }

    /// The batched pair over the whole route list in one dispatch per
    /// kernel. `weights_override` and `route_count_override` exist for
    /// the mutation cases only.
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
            .new_output_buffer((tokens * top_k * f * 2) as u64);
        let y_buf = self.context.new_output_buffer((tokens * d * 2) as u64);
        let routing16: Vec<f16> = weights.iter().map(|&w| f16::from_f32(w)).collect();
        let routing_buf = self.context.new_buffer_with_data(&to_le(&routing16));
        let routes_buf = self
            .context
            .new_buffer_with_data(&MoePrefillRoute::bytes(&self.routes));
        let residual_buf = self.context.new_buffer_with_data(&to_le(&self.residual));

        let routed = RoutedBlobsWideBuffer::new(&mut self.context, false).expect("arg buffer");
        let blob_refs: Vec<(&Buffer, u64)> = self.blobs.iter().map(|b| (b, 0u64)).collect();
        routed
            .bind(&mut self.context, false, &blob_refs)
            .expect("bind blobs");

        let pass = self.context.begin_pass();
        for blob in &self.blobs {
            pass.use_read_buffer(blob);
        }
        encode_moe_prefill_phase1(
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
        )
        .expect("batched phase1");
        encode_moe_prefill_phase2_fused(
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

/// Ragged routes over a 12-expert union at top_k=8: the wide binding, the
/// route list, and the rank-ordered fused reduce, bit-exact against M
/// sequential decode passes.
#[test]
fn batched_pair_is_bit_identical_to_sequential_decode_pair() {
    let mut fixture = Fixture::new(128, 128, 8, 6, 12);
    let (want_acts, want_y) = fixture.decode_pair_sequential();
    let (got_acts, got_y) = fixture.batched_pair(None, None);
    assert_bits_equal("acts", &got_acts, &want_acts);
    assert_bits_equal("y", &got_y, &want_y);
}

/// The real Gemma 4 routed shape (D=2816, F=704), decode-oracle only: a
/// shape-dependent indexing bug (the u16load path over many groups) is
/// what this size is for.
#[test]
fn batched_pair_is_bit_identical_at_the_real_gemma4_shape() {
    let mut fixture = Fixture::new(2816, 704, 8, 4, 10);
    let (want_acts, want_y) = fixture.decode_pair_sequential();
    let (got_acts, got_y) = fixture.batched_pair(None, None);
    assert_bits_equal("acts", &got_acts, &want_acts);
    assert_bits_equal("y", &got_y, &want_y);
}

/// top_k=4 exercises the padded-rank path: the decode oracle pads
/// routing weights with zeros past top_k while the batched kernel seeds
/// the padded ranks' partials with literal zeros. Both must land on the
/// same bits.
#[test]
fn batched_pair_pads_ranks_like_the_decode_kernel() {
    let mut fixture = Fixture::new(128, 64, 4, 5, 6);
    let (want_acts, want_y) = fixture.decode_pair_sequential();
    let (got_acts, got_y) = fixture.batched_pair(None, None);
    assert_bits_equal("acts", &got_acts, &want_acts);
    assert_bits_equal("y", &got_y, &want_y);
}

/// The SHARED-EXPERT shape: phase 2 seeded with a non-zero residual, which
/// is what `qwen3_5` passes (its gated shared-expert output) and what every
/// other family passes zeros for. The seed is added FIRST, before the eight
/// ranks, so it is not the same arithmetic as adding the shared expert to a
/// finished routed sum -- that is the whole reason the kernel takes it
/// rather than the driver appending a residual add.
#[test]
fn batched_pair_is_bit_identical_with_a_shared_expert_residual() {
    let mut fixture = Fixture::new(128, 128, 8, 6, 12).with_residual();
    let (want_acts, want_y) = fixture.decode_pair_sequential();
    let (got_acts, got_y) = fixture.batched_pair(None, None);
    assert_bits_equal("acts", &got_acts, &want_acts);
    assert_bits_equal("y", &got_y, &want_y);
}

/// The same at the real routed shape, where the 8-term sums are long
/// enough for reordering to be visible at all (the file header records
/// that the small cases round identically under a reversed reduce).
#[test]
fn a_shared_expert_residual_is_bit_identical_at_the_real_shape() {
    let mut fixture = Fixture::new(2816, 704, 8, 4, 10).with_residual();
    let (want_acts, want_y) = fixture.decode_pair_sequential();
    let (got_acts, got_y) = fixture.batched_pair(None, None);
    assert_bits_equal("acts", &got_acts, &want_acts);
    assert_bits_equal("y", &got_y, &want_y);
}

/// THE FIXTURE MUST DISCRIMINATE BEFORE THE TWO CASES ABOVE MEAN
/// ANYTHING. A zero residual is the additive identity, so with one the
/// seeded kernel and the `acc = 0.0f` kernel it replaced are the same
/// function and the parity cases would pass against either -- exactly the
/// trap AGENTS.md Gotchas 48 and 50 record on two other axes (a packing
/// fixture blind to field order, a norm fixture whose weights are near
/// zero). This asserts the seed reaches the output: same routes, same
/// weights, same activations, residual the only difference, and the
/// outputs must DIFFER.
///
/// It also pins the identity in the other direction, which is what says
/// the Gemma path's bytes did not move when the seed was added: a zeroed
/// residual reproduces the un-seeded result exactly.
#[test]
fn the_residual_seed_is_what_a_zero_fixture_cannot_see() {
    let mut zeroed = Fixture::new(128, 128, 8, 6, 12);
    let (_, zero_y) = zeroed.batched_pair(None, None);

    let mut seeded = Fixture::new(128, 128, 8, 6, 12).with_residual();
    let (_, seeded_y) = seeded.batched_pair(None, None);
    assert_bits_differ("residual seed", &seeded_y, &zero_y);

    // And the decode oracle agrees about the zero case, so "unchanged"
    // means unchanged against the vendored kernel and not merely
    // self-consistent.
    let (_, want_y) = zeroed.decode_pair_sequential();
    assert_bits_equal("zero residual", &zero_y, &want_y);
}

/// MUTATION 1: swapping one token's routing weights across ranks changes
/// the rank-ordered reduce, so the comparison must fail. A pass here
/// means the oracle is not sensitive to the thing it exists to check
/// (AGENTS.md Gotcha 27's whole failure mode).
#[test]
fn mutating_rank_weights_breaks_parity() {
    let mut fixture = Fixture::new(128, 128, 8, 6, 12);
    let (.., want_y) = fixture.decode_pair_sequential();
    let mut weights = fixture.weights.clone();
    // Token 2: swap the weights of ranks 0 and 1.
    weights.swap(2 * fixture.top_k, 2 * fixture.top_k + 1);
    let (.., got_y) = fixture.batched_pair(Some(weights), None);
    assert_bits_differ("y", &got_y, &want_y);
}

/// MUTATION 2: dropping the final route leaves its acts row unwritten
/// (zero), so the last token's output must differ.
#[test]
fn dropping_a_route_breaks_parity() {
    let mut fixture = Fixture::new(128, 128, 8, 6, 12);
    let (.., want_y) = fixture.decode_pair_sequential();
    let (.., got_y) = fixture.batched_pair(None, Some(fixture.routes.len() - 1));
    assert_bits_differ("y", &got_y, &want_y);
}

/// The CPU reference from the decode pair's own test, on the small case:
/// both GPU paths share their MSL helpers, so only an independent reader
/// can catch a bug they share.
#[test]
fn batched_pair_matches_the_cpu_reference_within_tolerance() {
    let (d, f, top_k, tokens, experts) = (128usize, 128usize, 8usize, 6usize, 12usize);
    let mut fixture = Fixture::new(d, f, top_k, tokens, experts);

    // Rebuild the same blobs on the CPU. The fixture quantized from
    // deterministic rows, so dequantizing through `run_ffn` needs those
    // rows again -- rebuild via the same seeds.
    let mut cpu_ffns = Vec::new();
    for e in 0..experts {
        let seed = 9000 + 137 * e as u64;
        cpu_ffns.push((
            quantized_rows(f, d, seed + 1),
            quantized_rows(f, d, seed + 2),
            quantized_rows(d, f, seed + 3),
        ));
    }

    let (.., got_y) = fixture.batched_pair(None, None);
    for t in 0..tokens {
        let x32: Vec<f32> = fixture.x[t * d..(t + 1) * d]
            .iter()
            .map(|v| v.to_f32())
            .collect();
        let mut expected = vec![0.0f32; d];
        for r in 0..top_k {
            let route = fixture.routes[t * top_k + r];
            let (gate, up, down) = &cpu_ffns[route.slot as usize];
            let out = turbospark_compute::run_ffn(gate, up, down, &x32, d, f);
            let w = fixture.weights[t * top_k + r];
            for (dst, o) in expected.iter_mut().zip(out.iter()) {
                *dst += w * o;
            }
        }
        for d2 in 0..d {
            let got = f16::from_bits(got_y[t * d + d2]).to_f32();
            let diff = (got - expected[d2]).abs();
            let tol = 5e-2_f32.max(expected[d2].abs() * 3e-2);
            assert!(
                diff <= tol,
                "t={t} d={d2}: got {got} want {} (diff {diff})",
                expected[d2]
            );
        }
    }
}
