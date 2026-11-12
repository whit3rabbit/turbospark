//! Bench prototype for kapitsa's expert-sorted grouped GEMM MoE prefill
//! (docs/KAPITSA_KERNELS.md item 8), at the mechanism level: per expert,
//! ONE M-row GEMM over its routed token rows instead of one GEMV per route
//! instance, so each expert's weight bytes are read once per expert instead
//! of once per (token, route) pair. Uses the EXISTING int4 batch GEMM and
//! GEMV kernels against the same packed weights -- no new shader -- because
//! the question is the SCHEDULING (weight reads amortized across the
//! expert's sub-batch), not the tile.
//!
//! What this does NOT model: the routing permutation and row gather (host
//! work in a real integration, or a scatter kernel), the dedupe of
//! identical draft rows, and the combine order. Those bound what an
//! integration could add, not what the mechanism returns; the ratio below
//! is an upper bound on the kernel-side win, which is the conservative
//! direction for an adopt/skip decision.
//!
//! ```sh
//! cargo test -p turbospark-gpu --test moe_grouped_gemm_bench --release -- --ignored --nocapture
//! ```

#![cfg(target_os = "macos")]

use turbospark_gpu::{
    autorelease_pool, encode_dequant_int4_gemm_resident, encode_dequant_int4_gemv_resident,
    Int4ResidentMatrix, MetalBuffer, MetalContext,
};

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

fn half_bytes(v: &[half::f16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect()
}

/// One expert's packed INT4 weights, `[rows, cols]` at the shared-expert /
/// routed-expert prefill shape of Qwen 3.6 35B-A3B (moe_inter 512, hidden
/// 2048) and of the 27B dense FFN tail for the wide arm.
struct Expert {
    weights: MetalBuffer,
    weights_offset: u64,
    scales_offset: u64,
    biases_offset: u64,
    x: MetalBuffer,
    y: MetalBuffer,
    rows: usize,
    cols: usize,
}

impl Expert {
    fn new(context: &MetalContext, rows: usize, cols: usize, seed: u64) -> Self {
        let mut weights = Vec::new();
        let mut scales = Vec::new();
        let mut biases = Vec::new();
        for r in 0..rows {
            let q = turbospark_compute::quantize_int4_affine(&deterministic(
                seed + r as u64 * 97 + 1,
                cols,
                0.5,
            ));
            weights.extend_from_slice(&q.packed);
            scales.extend_from_slice(&q.scales);
            biases.extend_from_slice(&q.biases);
        }
        let weights_offset = 4u64;
        let scales_offset = weights_offset + weights.len() as u64;
        let biases_offset = scales_offset + (scales.len() * 2) as u64;
        let mut bytes = vec![0u8; biases_offset as usize + biases.len() * 2];
        bytes[weights_offset as usize..weights_offset as usize + weights.len()]
            .copy_from_slice(&weights);
        bytes[scales_offset as usize..scales_offset as usize + scales.len() * 2].copy_from_slice(
            &scales
                .iter()
                .flat_map(|s| s.to_le_bytes())
                .collect::<Vec<u8>>(),
        );
        bytes[biases_offset as usize..biases_offset as usize + biases.len() * 2].copy_from_slice(
            &biases
                .iter()
                .flat_map(|b| b.to_le_bytes())
                .collect::<Vec<u8>>(),
        );
        // 16 rows of x: the kernels read every byte regardless, and the
        // fixture measures bytes moved, not values.
        let x = context.new_output_buffer((cols * 2 * 16) as u64);
        let y = context.new_output_buffer((rows * 2 * 16) as u64);
        Self {
            weights: context.new_buffer_with_data(&bytes),
            weights_offset,
            scales_offset,
            biases_offset,
            x,
            y,
            rows,
            cols,
        }
    }

    fn matrix(&self) -> Int4ResidentMatrix<'_> {
        Int4ResidentMatrix {
            buffer: &self.weights,
            weights_offset: self.weights_offset,
            scales_offset: self.scales_offset,
            biases_offset: self.biases_offset,
            rows: self.rows,
            cols: self.cols,
        }
    }
}

/// `routes` route instances spread over `experts` experts, sub-batch
/// `rows_per_expert` rows each. GEMV arm: one 1-row call per route
/// instance, the incumbent's shape. GEMM arm: one `rows_per_expert`-row
/// call per expert, the grouped shape. Same expert weights in both arms;
/// the GEMM arm reads them once per expert instead of once per route.
#[test]
#[ignore = "benchmark: needs a real Metal device, reports rather than asserts"]
fn grouped_gemm_ratio_against_gemv_per_route() {
    const ROUNDS: usize = 5;
    // (label, experts, rows_per_expert, weight_rows, weight_cols). The row
    // counts bracket the slot caps the runtime's expert cache and streamed
    // mode put on a sub-batch (M = 2..8) plus a wide arm at the dense FFN
    // tail's shape.
    let cases: [(&str, usize, usize, usize, usize); 5] = [
        ("moe_inter512 h2048 g2 ", 64, 2, 512, 2048),
        ("moe_inter512 h2048 g4 ", 64, 4, 512, 2048),
        ("moe_inter512 h2048 g8 ", 64, 8, 512, 2048),
        ("moe_inter512 h2048 g16", 64, 16, 512, 2048),
        ("ffn_up 17408x5120 g8  ", 32, 8, 17408, 5120),
    ];

    let mut context = MetalContext::new().expect("Metal device");
    let mut expert_sets: Vec<Vec<Expert>> = Vec::new();
    for (i, (_, experts, _, rows, cols)) in cases.iter().enumerate() {
        expert_sets.push(
            (0..*experts)
                .map(|e| Expert::new(&context, *rows, *cols, 0xF0_000 + i as u64 * 31 + e as u64))
                .collect(),
        );
    }

    let weight_bytes = |r: usize, c: usize| (r * c / 2 + r * c / 64 * 4) as u64;

    // Warmup + fault-in.
    for (si, (_, _, g, _, _)) in cases.iter().enumerate() {
        let set = &expert_sets[si];
        autorelease_pool(|| {
            let pass = context.begin_pass();
            for e in set {
                let matrix = e.matrix();
                let x_row_bytes = (e.cols * 2) as u64;
                for m in 0..*g {
                    encode_dequant_int4_gemv_resident(
                        &mut context,
                        &pass,
                        &matrix,
                        (&e.x, m as u64 * x_row_bytes),
                        (&e.y, 0),
                    )
                    .unwrap();
                }
            }
            pass.commit_and_wait();
        });
    }

    println!("\ncase                     gemm/gemv  (5 interleaved rounds)");
    for (si, (label, experts, g, rows, cols)) in cases.iter().enumerate() {
        let set = &expert_sets[si];
        let routes = experts * g;
        let mut ratios = Vec::new();
        for _ in 0..ROUNDS {
            let a = autorelease_pool(|| {
                let pass = context.begin_pass();
                for _ in 0..*g {
                    // g rounds of the WHOLE expert set = `routes` GEMVs.
                    for e in set {
                        encode_dequant_int4_gemv_resident(
                            &mut context,
                            &pass,
                            &e.matrix(),
                            (&e.x, 0),
                            (&e.y, 0),
                        )
                        .unwrap();
                    }
                }
                pass.commit_and_wait_with_gpu_time()
            });
            let b = autorelease_pool(|| {
                let pass = context.begin_pass();
                for e in set {
                    encode_dequant_int4_gemm_resident(
                        &mut context,
                        &pass,
                        &e.matrix(),
                        (&e.x, 0),
                        (&e.y, 0),
                        *g,
                    )
                    .unwrap();
                }
                pass.commit_and_wait_with_gpu_time()
            });
            // Both arms serve the same `routes` route instances (the GEMM
            // batches each expert's g of them), so the deliverable is the
            // plain time ratio.
            ratios.push(b / a);
        }
        let mean: f64 = ratios.iter().sum::<f64>() / ratios.len() as f64;
        println!("{label}  {mean:.4}");
    }
    println!(
        "\ngemm/gemv < 1 means the grouped shape serves the same route\n\
         instances faster, because each expert's weights are read once, not\n\
         once per route. The batch GEMM's c(M) at these widths is 0.4-0.6,\n\
         so the ratio cannot reach the ideal 1/g. The adopt bar is whether\n\
         this survives the integration costs the bench does not model (route\n\
         permutation, row gather, combine order).\n"
    );
}
