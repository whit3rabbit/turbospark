#![cfg(target_os = "macos")]
//! Parity test for `moe_inkling2_lamq.metal` against `turbospark_compute::lamq`
//! rows and `inkling2::situ`, on real Metal hardware, at the Mini's real
//! dimensions (hidden 2304, expert width 1152, 8 slots).
//!
//! The eight slots use different formats (both hashed codes and row trellis
//! at several widths, on each side), so a kernel that took one slot's format
//! for another, or read the HF gate/up interleave instead of the halves,
//! fails. Phase 2 weights differ per slot and the reduction is checked in
//! slot order. Both the streamed (`RoutedBlobs`) and resident (`sel` list,
//! permuted) bindings run.

use half::f16;
use turbospark_compute::inkling2::situ;
use turbospark_compute::lamq::{LamqCode, LamqLinear};
use turbospark_gpu::{
    encode_inkling2_lamq_phase1, encode_inkling2_lamq_phase2, lamq_lut_pack,
    moe_inkling2_lamq_source, Inkling2ResidentLayer, LamqExpertOffsets, LamqExperts, MetalContext,
    RoutedBlobsBuffer, MOE_INKLING2_LAMQ_REFLECT_FUNCTION,
};

const D: usize = 2304;
const F: usize = 1152;
const SLOTS: usize = 8;
const CAPS: (f32, f32) = (4.0, 25.0);

fn xs(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

fn unit(s: &mut u32) -> f32 {
    (xs(s) >> 8) as f32 / (1u32 << 23) as f32 - 1.0
}

fn le32(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn read_f32(b: &metal::Buffer, n: usize) -> Vec<f32> {
    unsafe { std::slice::from_raw_parts(b.contents() as *const f32, n) }.to_vec()
}

fn read_f16(b: &metal::Buffer, n: usize) -> Vec<f32> {
    let bits = unsafe { std::slice::from_raw_parts(b.contents() as *const u16, n) };
    bits.iter().map(|&v| f16::from_bits(v).to_f32()).collect()
}

/// (gate-up code, down code) per slot.
const FORMATS: [(LamqCode, LamqCode); SLOTS] = [
    (LamqCode::Rtr1, LamqCode::Rtr1),
    (LamqCode::Ht2, LamqCode::Ht3),
    (LamqCode::Ht3, LamqCode::Ht2),
    (LamqCode::Rtr4, LamqCode::Rtr2),
    (LamqCode::Rtr2, LamqCode::Rtr3),
    (LamqCode::Ht2, LamqCode::Rtr1),
    (LamqCode::Rtr3, LamqCode::Rtr4),
    (LamqCode::Ht3, LamqCode::Ht3),
];

struct Expert {
    gu: (LamqCode, Vec<u8>, Vec<u16>),
    dn: (LamqCode, Vec<u8>, Vec<u16>),
}

fn random_side(
    code: LamqCode,
    rows: usize,
    cols: usize,
    s: &mut u32,
) -> (LamqCode, Vec<u8>, Vec<u16>) {
    let codes = (0..rows * code.row_bytes(cols).unwrap())
        .map(|_| xs(s) as u8)
        .collect();
    let scales = (0..rows)
        .map(|_| f16::from_f32(0.02 + 0.03 * (unit(s) + 1.0)).to_bits())
        .collect();
    (code, codes, scales)
}

fn trellis() -> [LamqCode; 6] {
    [
        LamqCode::Ht2,
        LamqCode::Ht3,
        LamqCode::Rtr1,
        LamqCode::Rtr2,
        LamqCode::Rtr3,
        LamqCode::Rtr4,
    ]
}

#[test]
fn lamq_expert_pair_matches_the_reference_streamed_and_resident() {
    let mut s = 0x5EED_1234u32;
    // LUTs: random f16, the reference sees the same rounded values.
    let luts: Vec<(LamqCode, Vec<u16>)> = trellis()
        .iter()
        .map(|&c| {
            (
                c,
                (0..c.table_rows() * c.g())
                    .map(|_| f16::from_f32(unit(&mut s)).to_bits())
                    .collect(),
            )
        })
        .collect();
    let lut_f32 = |c: LamqCode| -> Vec<f32> {
        luts.iter()
            .find(|(x, _)| *x == c)
            .unwrap()
            .1
            .iter()
            .map(|&b| f16::from_bits(b).to_f32())
            .collect()
    };
    let experts: Vec<Expert> = FORMATS
        .iter()
        .map(|&(g, d)| Expert {
            gu: random_side(g, 2 * F, D, &mut s),
            dn: random_side(d, D, F, &mut s),
        })
        .collect();

    // Blob layout with uniform offsets: widest format on each side.
    let gu_w = FORMATS
        .iter()
        .map(|(g, _)| 2 * F * g.row_bytes(D).unwrap())
        .max()
        .unwrap();
    let dn_w = FORMATS
        .iter()
        .map(|(_, d)| D * d.row_bytes(F).unwrap())
        .max()
        .unwrap();
    let off = LamqExpertOffsets {
        fmt: 0,
        gu: 16,
        gu_s: (16 + gu_w) as u32,
        dn: (16 + gu_w + 4 * F) as u32,
        dn_s: (16 + gu_w + 4 * F + dn_w) as u32,
    };
    let stride = (off.dn_s as usize + 2 * D).div_ceil(16384) * 16384;
    let blob = |e: &Expert| -> Vec<u8> {
        let mut b = vec![0u8; stride];
        b[0] = e.gu.0.id();
        b[4] = e.dn.0.id();
        b[8..12].copy_from_slice(&(e.gu.0.row_bytes(D).unwrap() as u32).to_le_bytes());
        b[12..16].copy_from_slice(&(e.dn.0.row_bytes(F).unwrap() as u32).to_le_bytes());
        b[off.gu as usize..off.gu as usize + e.gu.1.len()].copy_from_slice(&e.gu.1);
        for (i, v) in e.gu.2.iter().enumerate() {
            b[off.gu_s as usize + 2 * i..off.gu_s as usize + 2 * i + 2]
                .copy_from_slice(&v.to_le_bytes());
        }
        b[off.dn as usize..off.dn as usize + e.dn.1.len()].copy_from_slice(&e.dn.1);
        for (i, v) in e.dn.2.iter().enumerate() {
            b[off.dn_s as usize + 2 * i..off.dn_s as usize + 2 * i + 2]
                .copy_from_slice(&v.to_le_bytes());
        }
        b
    };

    let x: Vec<f32> = (0..D).map(|_| unit(&mut s)).collect();
    let act_in: Vec<f32> = (0..SLOTS * F).map(|_| unit(&mut s)).collect();
    let w: Vec<f32> = (0..SLOTS).map(|i| 0.05 + 0.1 * i as f32).collect();

    // Reference.
    let lin =
        |side: &(LamqCode, Vec<u8>, Vec<u16>), m: usize, n: usize, t: &[f32]| -> Vec<Vec<f32>> {
            let l = LamqLinear {
                code: side.0,
                m,
                n,
                codes: &side.1,
                scales: &side.2,
                signs_in: &[],
                signs_out: None,
                table: t,
            };
            (0..m).map(|r| l.row(r)).collect()
        };
    let dot = |a: &[f32], b: &[f32]| -> f64 {
        a.iter()
            .zip(b)
            .map(|(&p, &q)| f64::from(p) * f64::from(q))
            .sum()
    };
    let mut want_acts = vec![0.0f64; SLOTS * F];
    let mut want_y = vec![0.0f64; D];
    for (slot, e) in experts.iter().enumerate() {
        let gu = lin(&e.gu, 2 * F, D, &lut_f32(e.gu.0));
        for f in 0..F {
            let (g, u) = (dot(&gu[f], &x) as f32, dot(&gu[F + f], &x) as f32);
            want_acts[slot * F + f] = f64::from(situ(g, u, CAPS.0, CAPS.1));
        }
        let dn = lin(&e.dn, D, F, &lut_f32(e.dn.0));
        let a = &act_in[slot * F..(slot + 1) * F];
        for d in 0..D {
            want_y[d] += f64::from(f16::from_f32(w[slot]).to_f32()) * dot(&dn[d], a);
        }
    }

    let mut context = MetalContext::new().expect("Metal device");
    let pack: Vec<(LamqCode, &[u16])> = luts.iter().map(|(c, t)| (*c, t.as_slice())).collect();
    let lut_buf = context.new_buffer_with_data(&lamq_lut_pack(&pack).unwrap());
    let x_buf = context.new_buffer_with_data(&le32(&x));
    let a_buf = context.new_buffer_with_data(&le32(&act_in));
    let w_buf = context.new_buffer_with_data(
        &w.iter()
            .flat_map(|&v| f16::from_f32(v).to_bits().to_le_bytes())
            .collect::<Vec<_>>(),
    );
    let check = |acts: &[f32], y: &[f32], what: &str| {
        for (i, (&g, &r)) in acts.iter().zip(&want_acts).enumerate() {
            assert!(
                (f64::from(g) - r).abs() <= 1e-3 + 1e-3 * r.abs(),
                "{what} act {i}: {g} vs {r}"
            );
        }
        for (i, (&g, &r)) in y.iter().zip(&want_y).enumerate() {
            assert!(
                (f64::from(g) - r).abs() <= 5e-3 + 2e-3 * r.abs(),
                "{what} y {i}: {g} vs {r}"
            );
        }
    };

    // Streamed: one blob buffer per slot.
    let blobs: Vec<metal::Buffer> = experts
        .iter()
        .map(|e| context.new_buffer_with_data(&blob(e)))
        .collect();
    let source = moe_inkling2_lamq_source();
    let routed = RoutedBlobsBuffer::new_for(
        &mut context,
        source,
        MOE_INKLING2_LAMQ_REFLECT_FUNCTION,
        false,
    )
    .unwrap();
    let refs: Vec<(&metal::Buffer, u64)> = blobs.iter().map(|b| (b, 0u64)).collect();
    routed
        .bind_for(
            &mut context,
            source,
            MOE_INKLING2_LAMQ_REFLECT_FUNCTION,
            false,
            &refs,
        )
        .unwrap();
    let acts = context.new_buffer_with_data(&vec![0u8; SLOTS * F * 4]);
    let y = context.new_buffer_with_data(&vec![0u8; D * 2]);
    let ex = LamqExperts::Streamed(&routed);
    let pass = context.begin_pass();
    // Blobs reached through the argument buffer must be made resident by the
    // caller, as the runtime does for the NVFP4 pair.
    for b in &blobs {
        pass.use_read_buffer(b);
    }
    encode_inkling2_lamq_phase1(
        &mut context,
        &pass,
        &ex,
        &off,
        (&lut_buf, 0),
        (&x_buf, 0),
        (&acts, 0),
        D as u32,
        F as u32,
        SLOTS as u32,
        CAPS.0,
        CAPS.1,
    )
    .unwrap();
    encode_inkling2_lamq_phase2(
        &mut context,
        &pass,
        &ex,
        &off,
        (&lut_buf, 0),
        (&a_buf, 0),
        (&w_buf, 0),
        (&y, 0),
        D as u32,
        F as u32,
        SLOTS as u32,
    )
    .unwrap();
    pass.commit_and_wait();
    check(&read_f32(&acts, SLOTS * F), &read_f16(&y, D), "streamed");

    // Resident: 11 experts in one buffer, slots select a permutation of them.
    let n_experts = 11u32;
    let sel: Vec<u32> = vec![9, 2, 7, 0, 10, 4, 1, 6];
    let mut layer = vec![0u8; stride * n_experts as usize];
    for (slot, &e) in sel.iter().enumerate() {
        layer[e as usize * stride..(e as usize + 1) * stride]
            .copy_from_slice(&blob(&experts[slot]));
    }
    let layer_buf = context.new_buffer_with_data(&layer);
    let sel_buf =
        context.new_buffer_with_data(&sel.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());
    let resident = Inkling2ResidentLayer {
        buffer: &layer_buf,
        stride: stride as u64,
        n_experts,
        sel: (&sel_buf, 0),
    };
    let acts2 = context.new_buffer_with_data(&vec![0u8; SLOTS * F * 4]);
    let y2 = context.new_buffer_with_data(&vec![0u8; D * 2]);
    let ex = LamqExperts::Resident(&resident);
    let pass = context.begin_pass();
    encode_inkling2_lamq_phase1(
        &mut context,
        &pass,
        &ex,
        &off,
        (&lut_buf, 0),
        (&x_buf, 0),
        (&acts2, 0),
        D as u32,
        F as u32,
        SLOTS as u32,
        CAPS.0,
        CAPS.1,
    )
    .unwrap();
    encode_inkling2_lamq_phase2(
        &mut context,
        &pass,
        &ex,
        &off,
        (&lut_buf, 0),
        (&a_buf, 0),
        (&w_buf, 0),
        (&y2, 0),
        D as u32,
        F as u32,
        SLOTS as u32,
    )
    .unwrap();
    pass.commit_and_wait();
    let (a1, a2) = (read_f32(&acts, SLOTS * F), read_f32(&acts2, SLOTS * F));
    check(&a2, &read_f16(&y2, D), "resident");
    assert_eq!(
        a1, a2,
        "streamed and resident bindings compute the same bits"
    );
}

#[test]
fn lut_pack_and_shapes_out_of_contract_are_refused() {
    let t: Vec<u16> = vec![0; 4096];
    assert!(
        lamq_lut_pack(&[(LamqCode::Rtr1, &t)]).is_err(),
        "missing tables"
    );
    let short: Vec<u16> = vec![0; 100];
    let mut all: Vec<(LamqCode, &[u16])> = trellis()
        .iter()
        .map(|&c| (c, &t[..c.table_rows() * c.g()]))
        .collect();
    assert!(lamq_lut_pack(&all).is_ok());
    all[0].1 = &short;
    assert!(lamq_lut_pack(&all).is_err(), "wrong size");
}

/// Not a gate: GPU time of the resident expert pair per format, at the
/// Mini's shapes. 128 experts of one format sit in one buffer and the timed
/// dispatches cycle through 16 disjoint 8-expert selections, so the codes
/// come from DRAM, not the system cache; a primer pass brings the device up
/// first. Only the codes the 5.4 GB release uses for experts are timed. Run
/// with `cargo test --release -p turbospark-gpu --test
/// moe_inkling2_lamq_parity -- --ignored expert_pair_bandwidth --nocapture`.
#[test]
#[ignore]
fn expert_pair_bandwidth() {
    const EXPERTS: usize = 128;
    const REPS: usize = 8;
    let mut context = MetalContext::new().expect("Metal device");
    let mut s = 0xBE4C_0001u32;
    let luts: Vec<(LamqCode, Vec<u16>)> = trellis()
        .iter()
        .map(|&c| {
            (
                c,
                (0..c.table_rows() * c.g())
                    .map(|_| f16::from_f32(unit(&mut s)).to_bits())
                    .collect(),
            )
        })
        .collect();
    let pack: Vec<(LamqCode, &[u16])> = luts.iter().map(|(c, t)| (*c, t.as_slice())).collect();
    let lut_buf = context.new_buffer_with_data(&lamq_lut_pack(&pack).unwrap());
    let x: Vec<f32> = (0..D).map(|_| unit(&mut s)).collect();
    let act_in: Vec<f32> = (0..SLOTS * F).map(|_| unit(&mut s)).collect();
    let x_buf = context.new_buffer_with_data(&le32(&x));
    let a_buf = context.new_buffer_with_data(&le32(&act_in));
    let w_buf = context.new_buffer_with_data(&[0x3c00u16; SLOTS]);
    let acts = context.new_buffer_with_data(&vec![0u8; SLOTS * F * 4]);
    let y = context.new_buffer_with_data(&vec![0u8; D * 2]);
    let sets = EXPERTS / SLOTS;
    let sel: Vec<u32> = (0..EXPERTS as u32).collect();
    let sel_buf =
        context.new_buffer_with_data(&sel.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());
    for code in [LamqCode::Ht2, LamqCode::Ht3, LamqCode::Rtr1] {
        let (gu_rb, dn_rb) = (code.row_bytes(D).unwrap(), code.row_bytes(F).unwrap());
        let off = LamqExpertOffsets {
            fmt: 0,
            gu: 16,
            gu_s: (16 + 2 * F * gu_rb) as u32,
            dn: (16 + 2 * F * gu_rb + 4 * F) as u32,
            dn_s: (16 + 2 * F * gu_rb + 4 * F + D * dn_rb) as u32,
        };
        let stride = (off.dn_s as usize + 2 * D).div_ceil(16384) * 16384;
        let mut layer: Vec<u8> = (0..stride * EXPERTS).map(|_| xs(&mut s) as u8).collect();
        for e in 0..EXPERTS {
            let b = &mut layer[e * stride..];
            b[..16].fill(0);
            b[0] = code.id();
            b[4] = code.id();
            b[8..12].copy_from_slice(&(gu_rb as u32).to_le_bytes());
            b[12..16].copy_from_slice(&(dn_rb as u32).to_le_bytes());
            // Finite scales.
            for i in 0..2 * F {
                let o = off.gu_s as usize + 2 * i;
                b[o..o + 2].copy_from_slice(&f16::from_f32(0.03).to_bits().to_le_bytes());
            }
            for i in 0..D {
                let o = off.dn_s as usize + 2 * i;
                b[o..o + 2].copy_from_slice(&f16::from_f32(0.03).to_bits().to_le_bytes());
            }
        }
        let layer_buf = context.new_buffer_with_data(&layer);
        drop(layer);
        for phase in [1, 2] {
            let run = |context: &mut MetalContext| -> f64 {
                let pass = context.begin_pass();
                for _ in 0..REPS {
                    for set in 0..sets {
                        let resident = Inkling2ResidentLayer {
                            buffer: &layer_buf,
                            stride: stride as u64,
                            n_experts: EXPERTS as u32,
                            sel: (&sel_buf, (set * SLOTS * 4) as u64),
                        };
                        let ex = LamqExperts::Resident(&resident);
                        if phase == 1 {
                            encode_inkling2_lamq_phase1(
                                context,
                                &pass,
                                &ex,
                                &off,
                                (&lut_buf, 0),
                                (&x_buf, 0),
                                (&acts, 0),
                                D as u32,
                                F as u32,
                                SLOTS as u32,
                                CAPS.0,
                                CAPS.1,
                            )
                            .unwrap();
                        } else {
                            encode_inkling2_lamq_phase2(
                                context,
                                &pass,
                                &ex,
                                &off,
                                (&lut_buf, 0),
                                (&a_buf, 0),
                                (&w_buf, 0),
                                (&y, 0),
                                D as u32,
                                F as u32,
                                SLOTS as u32,
                            )
                            .unwrap();
                        }
                    }
                }
                pass.commit_and_wait_with_gpu_time()
            };
            run(&mut context); // primer
            let t = (0..3).map(|_| run(&mut context)).fold(f64::MAX, f64::min);
            let per = t / (REPS * sets) as f64;
            let bytes = if phase == 1 {
                SLOTS * 2 * F * gu_rb
            } else {
                SLOTS * D * dn_rb
            };
            println!(
                "{:5} phase {phase}: {:7.2} us/dispatch  {:6.1} GB/s codes  {:6.1} G weights/s",
                code.name(),
                per * 1e6,
                bytes as f64 / per / 1e9,
                (SLOTS * 2 * F * D / phase) as f64 / per / 1e9,
            );
        }
    }
}
