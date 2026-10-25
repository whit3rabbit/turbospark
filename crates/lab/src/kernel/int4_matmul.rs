//! INT4 affine matmul at the install's own shapes: M one-row GEMVs against
//! the scalar batched GEMM and both Steel tiles, over a sweep of M.
//!
//! The port of `crates/gpu/tests/qwen38_int4_prefill_gemm_probe.rs` with the
//! shapes taken from `ArchConfig` instead of Qwen3.8's constants, and the
//! one-row GEMV added as the unit that c(M) is measured against.

use gpu::{
    autorelease_pool, encode_dequant_int4_gemm_resident, encode_dequant_int4_gemm_steel,
    encode_dequant_int4_gemv_resident, steel_supports, Int4ResidentMatrix, MetalContext, SteelTile,
    MAX_BATCH_ROWS,
};
use model_io::ArchConfig;
use serde_json::{json, Value};

use super::fit::{break_even, c_of_m, fit_affine};
use super::support::{copies_for, f16_activations, int4_blob, median};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Arm {
    /// M separate one-row GEMV dispatches: what M tokens cost unbatched.
    GemvPerRow,
    /// The scalar batched GEMM, `MAX_BATCH_ROWS` rows per dispatch.
    Scalar,
    Steel(SteelTile),
}

impl Arm {
    pub fn label(&self) -> String {
        match self {
            Arm::GemvPerRow => "gemv_per_row".into(),
            Arm::Scalar => "scalar_gemm".into(),
            Arm::Steel(SteelTile::M32K32N32) => "steel_m32".into(),
            Arm::Steel(SteelTile::M64K32N64) => "steel_m64".into(),
        }
    }
}

pub const ARMS: [Arm; 4] = [
    Arm::GemvPerRow,
    Arm::Scalar,
    Arm::Steel(SteelTile::M32K32N32),
    Arm::Steel(SteelTile::M64K32N64),
];

/// (rows, cols) pairs this install's INT4 projections run at, deduplicated,
/// restricted to the layout the resident kernels take (cols a multiple of
/// 64). Routed-expert shapes are added for MoE installs.
pub fn shapes(arch: &ArchConfig) -> Vec<(usize, usize)> {
    let mut out: Vec<(usize, usize)> = arch
        .decode_int4_gemv_shapes()
        .into_iter()
        .map(|(m, n)| (m as usize, n as usize))
        .collect();
    if arch.num_experts > 0 && arch.moe_intermediate_size > 0 {
        let (h, f) = (
            arch.hidden_size as usize,
            arch.moe_intermediate_size as usize,
        );
        out.push((2 * f, h));
        out.push((h, f));
    }
    if arch.has_linear_attention_layers() {
        // The gated-DeltaNet gate projections (`in_proj_a` and `in_proj_b`):
        // `num_v_heads` rows, so a handful of output rows against the full
        // hidden width. They are absent from `decode_int4_gemv_shapes`, and
        // they are exactly the shape where an 8-row verify pays launch and
        // occupancy latency instead of bandwidth (docs/KAPITSA_KERNELS.md,
        // lever 9), so a bench that skips them cannot see that cost.
        out.push((
            arch.linear_attention.num_v_heads as usize,
            arch.hidden_size as usize,
        ));
    }
    out.retain(|&(r, c)| r > 0 && c > 0 && c % 64 == 0);
    out.sort_unstable();
    out.dedup();
    out
}

/// Time every arm at every M for one shape. Returns the ledger record.
pub fn run_shape(
    context: &mut MetalContext,
    rows: usize,
    cols: usize,
    ms: &[usize],
    reps: usize,
) -> Result<Value, String> {
    let weight_bytes = rows * cols / 2 + rows * cols / 64 * 4;
    let copies = copies_for(weight_bytes, 256 << 20);
    let mats: Vec<_> = (0..copies)
        .map(|i| {
            let (data, scales, biases) = int4_blob(rows, cols, 0xC0FF_EE00 + i as u64);
            (context.new_buffer_with_data(&data), scales, biases)
        })
        .collect();
    // A Steel tile needs the row count to fit its tile, and some of the
    // install's real projections do not (the 48-row gated-DeltaNet gates):
    // callers of the engine fall back to the scalar GEMM there. The bench
    // records the arm as unsupported for that shape and times the rest.
    let supported: Vec<bool> = {
        let (buffer, scales, biases) = &mats[0];
        let w = Int4ResidentMatrix {
            buffer,
            weights_offset: 0,
            scales_offset: *scales,
            biases_offset: *biases,
            rows,
            cols,
        };
        ARMS.iter()
            .map(|arm| match arm {
                Arm::Steel(tile) => steel_supports(&w, *tile),
                _ => true,
            })
            .collect()
    };
    let max_m = *ms.iter().max().unwrap_or(&1);
    let x = context.new_buffer_with_data(&f16_activations(max_m * cols, 7));
    let y = context.new_output_buffer((max_m * rows * 2) as u64);
    let mut table = Vec::new();
    for &m in ms {
        let mut samples = vec![Vec::new(); ARMS.len()];
        for rep in 0..=reps {
            for (a, &arm) in ARMS.iter().enumerate() {
                if !supported[a] {
                    if rep > 0 {
                        samples[a].push(f64::NAN);
                    }
                    continue;
                }
                let t = autorelease_pool(|| -> Result<f64, String> {
                    let pass = context.begin_pass();
                    for (buffer, scales, biases) in &mats {
                        let w = Int4ResidentMatrix {
                            buffer,
                            weights_offset: 0,
                            scales_offset: *scales,
                            biases_offset: *biases,
                            rows,
                            cols,
                        };
                        match arm {
                            Arm::GemvPerRow => {
                                for r in 0..m {
                                    encode_dequant_int4_gemv_resident(
                                        context,
                                        &pass,
                                        &w,
                                        (&x, (r * cols * 2) as u64),
                                        (&y, (r * rows * 2) as u64),
                                    )
                                    .map_err(|e| e.to_string())?;
                                }
                            }
                            Arm::Scalar => {
                                let mut done = 0;
                                while done < m {
                                    let take = (m - done).min(MAX_BATCH_ROWS);
                                    encode_dequant_int4_gemm_resident(
                                        context,
                                        &pass,
                                        &w,
                                        (&x, (done * cols * 2) as u64),
                                        (&y, (done * rows * 2) as u64),
                                        take,
                                    )
                                    .map_err(|e| e.to_string())?;
                                    done += take;
                                }
                            }
                            Arm::Steel(tile) => {
                                encode_dequant_int4_gemm_steel(
                                    context,
                                    &pass,
                                    &w,
                                    (&x, 0),
                                    (&y, 0),
                                    m,
                                    tile,
                                )
                                .map_err(|e| e.to_string())?;
                            }
                        }
                    }
                    Ok(pass.commit_and_wait_with_gpu_time())
                })?;
                if rep > 0 {
                    samples[a].push(t * 1e3 / copies as f64);
                }
            }
        }
        let medians: Vec<f64> = samples.iter_mut().map(|s| median(s)).collect();
        table.push((m, medians));
    }

    // One-row GEMV time is the c(M) unit.
    let one_row = table
        .iter()
        .find(|(m, _)| *m == 1)
        .map(|(_, t)| t[0])
        .unwrap_or(table[0].1[0] / table[0].0 as f64);
    let flops = |m: usize| 2.0 * m as f64 * rows as f64 * cols as f64;
    let mut arms_json = serde_json::Map::new();
    let mut fits = Vec::new();
    for (a, arm) in ARMS.iter().enumerate() {
        if !supported[a] {
            fits.push(None);
            arms_json.insert(
                arm.label(),
                json!({"unsupported": "this shape does not fit the tile"}),
            );
            continue;
        }
        let points: Vec<(f64, f64)> = table.iter().map(|(m, t)| (*m as f64, t[a])).collect();
        let fit = fit_affine(&points);
        fits.push(fit);
        arms_json.insert(
            arm.label(),
            json!({
                "points": table.iter().map(|(m, t)| json!({
                    "m": m,
                    "ms": t[a],
                    "c_of_m": c_of_m(t[a], one_row, *m),
                    "tflops": flops(*m) / (t[a] * 1e-3) / 1e12,
                    "weight_gb_s": weight_bytes as f64 / (t[a] * 1e-3) / 1e9,
                })).collect::<Vec<_>>(),
                "fit": fit,
            }),
        );
    }
    let crossings: Vec<Value> = ARMS
        .iter()
        .enumerate()
        .skip(1)
        .filter_map(|(a, arm)| {
            let (r, c) = (fits[0]?, fits[a]?);
            Some(json!({"arm": arm.label(), "beats_gemv_per_row_above_m": break_even(&r, &c)}))
        })
        .collect();
    Ok(json!({
        "bench": "int4_matmul",
        "rows": rows,
        "cols": cols,
        "weight_bytes": weight_bytes,
        "copies": copies,
        "one_row_gemv_ms": one_row,
        "arms": arms_json,
        "break_even": crossings,
    }))
}

pub fn print_shape(rec: &Value) {
    println!(
        "int4 {} x {} ({:.1} MB), one-row GEMV {:.4} ms",
        rec["rows"],
        rec["cols"],
        rec["weight_bytes"].as_f64().unwrap_or(0.0) / 1e6,
        rec["one_row_gemv_ms"].as_f64().unwrap_or(f64::NAN)
    );
    let arms = rec["arms"].as_object().cloned().unwrap_or_default();
    let ms: Vec<u64> = arms
        .values()
        .next()
        .and_then(|a| a["points"].as_array())
        .map(|ps| ps.iter().filter_map(|p| p["m"].as_u64()).collect())
        .unwrap_or_default();
    print!("  {:<14}", "c(M) at M =");
    for m in &ms {
        print!("{m:>8}");
    }
    println!("   fit a + b*M (ms)");
    for (name, arm) in &arms {
        if let Some(why) = arm["unsupported"].as_str() {
            println!("  {name:<14}{why}");
            continue;
        }
        print!("  {name:<14}");
        for p in arm["points"].as_array().into_iter().flatten() {
            print!("{:>8.3}", p["c_of_m"].as_f64().unwrap_or(f64::NAN));
        }
        if let (Some(a), Some(b)) = (arm["fit"]["a"].as_f64(), arm["fit"]["b"].as_f64()) {
            print!("   {a:.4} + {b:.4}M");
        }
        println!();
    }
    for c in rec["break_even"].as_array().into_iter().flatten() {
        println!(
            "  {} beats per-row GEMV above M = {}",
            c["arm"].as_str().unwrap_or("?"),
            c["beats_gemv_per_row_above_m"]
                .as_f64()
                .map_or("never".to_string(), |m| format!("{m:.1}"))
        );
    }
}
