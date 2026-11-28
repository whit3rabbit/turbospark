//! Decode attention at the install's attention shapes: the barrier kernel
//! against the GQA-shared kernel, over context lengths, through the
//! production dispatcher (`AttentionScratch::with_decode_kernel`) rather than
//! hand-picked tuning parameters.
//!
//! The port of `crates/gpu/tests/qwen38_attention_decode_probe.rs` with the
//! head shapes and layer counts taken from `ArchConfig`. One case per
//! attention variant the install runs through `encode_attention_decode`:
//! full-attention layers, and sliding-window layers (ring cache, logical
//! window), each with the family's learned sinks when it has them. MLA,
//! compressed and gated-DeltaNet layers run other kernels and are named in
//! the case list's notes, not timed here.

use std::collections::BTreeMap;

use gpu::{
    attention_decode_gqa_supported, autorelease_pool, encode_attention_decode,
    encode_attention_decode_tq, encode_kv_quantize_tq, read_buffer_bytes, AttentionScratch,
    DecodeAttentionKernel, KvQuantTables, MetalContext, TqAttentionScratch,
    MAX_DECODE_ATTENTION_HEAD_DIM,
};
use model_io::{ArchConfig, KvQuant, ModelFamily};
use serde_json::json;

use super::support::{f16_bytes, median};
use super::{BenchOpts, Case, KernelBench, Subject};
use crate::bench::{fnv_hex, ArmCurve, Axis, BenchResult, Point};

/// The arms, barrier first: the kernel `TURBOSPARK_ATTN_DECODE_KERNEL` forces
/// with `barrier`, then the GQA-shared kernel where the shape is served, then
/// today's TurboQuant decode kernel (one threadgroup per q head, barriers per
/// position) at `--kv-bits` 4 and 3.5 on full-attention layers (KERN-02 step
/// a: price it before any rewrite). TurboQuant never serves a sliding layer.
pub const ARMS: [&str; 4] = ["barrier", "gqa", "tq4_barrier", "tq3.5_barrier"];

/// The TurboQuant arms and the `--kv-bits` value each runs.
const TQ_ARMS: [(&str, &str); 2] = [("tq4_barrier", "4"), ("tq3.5_barrier", "3.5")];

/// Gemma 4's two attention shapes, for `attn_decode_gemma4`, without a Gemma install
/// (KERN-03 step a). Full layers 16q over 2kv, sliding layers 8q over 4kv
/// with a 1024 window, head_dim 512 on both, 5 full and 25 sliding layers
/// (`crates/lab/families/gemma4.json`, `arch_baselines/gemma.rs`).
fn gemma4_shapes() -> Vec<(&'static str, AttnShape)> {
    vec![
        (
            "gemma4_full_attn",
            AttnShape {
                q_heads: 16,
                kv_heads: 2,
                head_dim: 512,
                layers: 5,
                window: 0,
                sinks: false,
            },
        ),
        (
            "gemma4_sliding_attn",
            AttnShape {
                q_heads: 8,
                kv_heads: 4,
                head_dim: 512,
                layers: 25,
                window: 1024,
                sinks: false,
            },
        ),
    ]
}

pub struct AttnDecode;

/// One attention variant's shape.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AttnShape {
    pub q_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub layers: usize,
    /// Sliding window in positions; 0 for full attention.
    pub window: u32,
    pub sinks: bool,
}

/// Families whose attention adds one learned sink logit per q head.
pub fn has_sinks(family: ModelFamily) -> bool {
    matches!(family, ModelFamily::GptOss | ModelFamily::Flint)
}

/// The attention variants this bench can drive, by case label.
pub fn shapes(arch: &ArchConfig) -> Vec<(&'static str, AttnShape)> {
    let count = |m: u8| {
        arch.full_attention_layer_mask
            .iter()
            .filter(|&&v| v == m)
            .count()
    };
    let sinks = has_sinks(arch.family);
    let mut out = Vec::new();
    let full_kv = if arch.num_full_kv_heads > 0 {
        arch.num_full_kv_heads
    } else {
        arch.num_kv_heads
    };
    let full_dim = if arch.full_head_dim > 0 {
        arch.full_head_dim
    } else {
        arch.head_dim
    };
    out.push((
        "full_attn",
        AttnShape {
            q_heads: arch.num_heads as u32,
            kv_heads: full_kv as u32,
            head_dim: full_dim as u32,
            layers: count(1),
            window: 0,
            sinks,
        },
    ));
    if arch.sliding_window > 0 {
        out.push((
            "sliding_attn",
            AttnShape {
                q_heads: arch.num_heads as u32,
                kv_heads: arch.num_kv_heads as u32,
                head_dim: arch.head_dim as u32,
                layers: count(0),
                window: arch.sliding_window as u32,
                sinks,
            },
        ));
    }
    out.retain(|(_, s)| {
        s.layers > 0
            && s.q_heads > 0
            && s.kv_heads > 0
            && s.head_dim > 0
            && s.q_heads % s.kv_heads == 0
            && s.head_dim <= MAX_DECODE_ATTENTION_HEAD_DIM
    });
    out
}

impl KernelBench for AttnDecode {
    fn name(&self) -> &'static str {
        "attn_decode"
    }

    fn describe(&self) -> &'static str {
        "decode attention per variant (full, sliding, sinks): barrier, GQA-shared and TurboQuant over context"
    }

    fn cases(&self, subject: &Subject, _opts: &BenchOpts) -> Result<Vec<Case>, String> {
        let arch = subject.arch()?;
        let cases: Vec<Case> = shapes(arch)
            .into_iter()
            .map(|(label, s)| Case {
                label: label.into(),
                params: json!({"q_heads": s.q_heads, "kv_heads": s.kv_heads,
                               "head_dim": s.head_dim, "layers": s.layers,
                               "window": s.window, "sinks": s.sinks}),
            })
            .collect();
        if cases.is_empty() {
            let mut why = "no full or sliding attention layers this kernel serves".to_string();
            if arch.has_mla_layers() {
                why.push_str("; MLA layers run the latent-attention kernels, which have no bench");
            }
            if arch.has_compressed_attention_layers() {
                why.push_str("; compressed attention has no bench");
            }
            return Err(why);
        }
        Ok(cases)
    }

    fn run(
        &self,
        context: &mut MetalContext,
        _subject: &Subject,
        case: &Case,
        opts: &BenchOpts,
    ) -> Result<BenchResult, String> {
        let p = &case.params;
        let u = |k: &str| p[k].as_u64().unwrap_or(0) as u32;
        let shape = AttnShape {
            q_heads: u("q_heads"),
            kv_heads: u("kv_heads"),
            head_dim: u("head_dim"),
            layers: u("layers") as usize,
            window: u("window"),
            sinks: p["sinks"] == true,
        };
        run_shape(context, &case.label, shape, opts)
    }
}

/// `attn_decode` at Gemma 4's shapes whatever the install (KERN-03 step a:
/// price the barrier kernel at head_dim 512, which the GQA-shared kernel does
/// not serve). The install only supplies the process; its own shapes are not
/// timed. Same arms and metrics as `attn_decode`.
pub struct AttnDecodeGemma4;

impl KernelBench for AttnDecodeGemma4 {
    fn name(&self) -> &'static str {
        "attn_decode_gemma4"
    }

    fn describe(&self) -> &'static str {
        "attn_decode at Gemma 4's head_dim 512 shapes (full 16q/2kv, sliding 8q/4kv), any install"
    }

    fn cases(&self, _subject: &Subject, _opts: &BenchOpts) -> Result<Vec<Case>, String> {
        Ok(gemma4_shapes()
            .into_iter()
            .map(|(label, s)| Case {
                label: label.into(),
                params: json!({"q_heads": s.q_heads, "kv_heads": s.kv_heads,
                               "head_dim": s.head_dim, "layers": s.layers,
                               "window": s.window, "sinks": s.sinks}),
            })
            .collect())
    }

    fn run(
        &self,
        context: &mut MetalContext,
        subject: &Subject,
        case: &Case,
        opts: &BenchOpts,
    ) -> Result<BenchResult, String> {
        AttnDecode.run(context, subject, case, opts)
    }
}

/// One timed lane: a dense FP16 kernel through the production dispatcher, or
/// today's TurboQuant kernel over the same rows quantized once at setup.
enum Lane {
    Dense(AttentionScratch),
    Tq {
        scratch: TqAttentionScratch,
        tables: KvQuantTables,
        packed: Vec<(gpu::MetalBuffer, gpu::MetalBuffer)>,
        /// Packed bytes per (position, kv head), K and V together.
        row_bytes: usize,
    },
}

pub fn run_shape(
    context: &mut MetalContext,
    label: &str,
    s: AttnShape,
    opts: &BenchOpts,
) -> Result<BenchResult, String> {
    let (nq, nkv, head_dim) = (s.q_heads, s.kv_heads, s.head_dim);
    let gqa = attention_decode_gqa_supported(head_dim, nq, nkv);
    let max_seq = *opts.contexts.iter().max().unwrap_or(&512);
    // A sliding layer's cache holds the window and wraps; a full layer's
    // holds every position.
    let rows_held = if s.window > 0 {
        s.window.min(max_seq)
    } else {
        max_seq
    } as usize;
    let kv_width = (nkv * head_dim) as usize;
    let q_width = (nq * head_dim) as usize;
    let q = context.new_buffer_with_data(&f16_bytes(
        (0..q_width).map(|i| ((i as f32) * 0.019).sin() * 0.5),
    ));
    // One K/V pair per layer of this variant, so a sample reads each layer's
    // cache once, as a decode step does.
    let kv: Vec<_> = (0..s.layers)
        .map(|layer| {
            let phase = layer as f32 * 0.37;
            let k = context.new_buffer_with_data(&f16_bytes(
                (0..rows_held * kv_width).map(|i| ((i as f32) * 0.0007 + phase).cos() * 0.3),
            ));
            let v = context.new_buffer_with_data(&f16_bytes(
                (0..rows_held * kv_width).map(|i| 0.7 + ((i as f32) * 0.0011 + phase).sin() * 0.3),
            ));
            (k, v)
        })
        .collect();
    // BF16 sink logits, one per q head, of the magnitude a trained sink has.
    let sinks = s.sinks.then(|| {
        let bytes: Vec<u8> = (0..nq)
            .flat_map(|h| {
                half::bf16::from_f32(1.0 + (h as f32 * 0.3).sin())
                    .to_bits()
                    .to_le_bytes()
            })
            .collect();
        context.new_buffer_with_data(&bytes)
    });
    let out = context.new_output_buffer((q_width * 2) as u64);
    let mut lanes: Vec<(&str, Lane)> = vec![(
        ARMS[0],
        Lane::Dense(
            AttentionScratch::new(context, nq, head_dim)
                .with_decode_kernel(DecodeAttentionKernel::Barrier),
        ),
    )];
    if gqa {
        lanes.push((
            ARMS[1],
            Lane::Dense(
                AttentionScratch::new(context, nq, head_dim)
                    .with_decode_kernel(DecodeAttentionKernel::GqaShared),
            ),
        ));
    }
    // TurboQuant only quantizes full-attention layers, over a linear cache.
    let tq_served = s.window == 0 && model_io::rht_supported(head_dim as i64);
    if tq_served {
        for (label, bits) in TQ_ARMS {
            let quant = KvQuant::parse(bits)?;
            let Some(tables) = KvQuantTables::new(context.device(), head_dim as usize, quant)
            else {
                continue;
            };
            let words = |b: u8| 1 + model_io::tq_packed_words(head_dim as i64, b) as usize;
            let (kw, vw) = (words(tables.k.bits), words(tables.v.bits));
            // The same FP16 K/V the dense arms read, quantized once by the
            // production kernel, so every arm attends over the same rows.
            let packed: Vec<_> = kv
                .iter()
                .map(|(k, v)| -> Result<_, String> {
                    let kq = context.new_output_buffer((rows_held * nkv as usize * kw * 4) as u64);
                    let vq = context.new_output_buffer((rows_held * nkv as usize * vw * 4) as u64);
                    autorelease_pool(|| -> Result<(), String> {
                        let pass = context.begin_pass();
                        for (src, dst, side) in [(k, &kq, &tables.k), (v, &vq, &tables.v)] {
                            encode_kv_quantize_tq(
                                context,
                                &pass,
                                (src, 0),
                                nkv * head_dim,
                                (dst, 0),
                                side,
                                head_dim,
                                nkv,
                                rows_held as u32,
                            )
                            .map_err(|e| e.to_string())?;
                        }
                        pass.commit_checked().map_err(|e| e.to_string())?.wait();
                        Ok(())
                    })?;
                    Ok((kq, vq))
                })
                .collect::<Result<_, _>>()?;
            lanes.push((
                label,
                Lane::Tq {
                    scratch: TqAttentionScratch::new(context, nq, head_dim),
                    tables,
                    packed,
                    row_bytes: (kw + vw) * 4,
                },
            ));
        }
    }
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut curves: Vec<Vec<Point>> = vec![Vec::new(); lanes.len()];
    for &seq in &opts.contexts {
        let (kv_start, ring) = if s.window > 0 {
            (
                seq.saturating_sub(s.window),
                if seq > s.window { s.window } else { 0 },
            )
        } else {
            (0, 0)
        };
        let read = (seq - kv_start) as f64;
        let mut samples = vec![Vec::new(); lanes.len()];
        let mut hashes = vec![None; lanes.len()];
        for rep in 0..=opts.reps {
            for (a, (_, lane)) in lanes.iter().enumerate() {
                let t = autorelease_pool(|| -> Result<f64, String> {
                    let pass = context.begin_pass();
                    match lane {
                        Lane::Dense(scratch) => {
                            for (k, v) in &kv {
                                encode_attention_decode(
                                    context,
                                    &pass,
                                    (&q, 0),
                                    k,
                                    v,
                                    scratch,
                                    (&out, 0),
                                    head_dim,
                                    nq,
                                    nkv,
                                    seq,
                                    kv_start,
                                    ring,
                                    scale,
                                    sinks.as_ref().map(|b| (b, 0)),
                                )
                                .map_err(|e| e.to_string())?;
                            }
                        }
                        Lane::Tq {
                            scratch,
                            tables,
                            packed,
                            ..
                        } => {
                            for (k, v) in packed {
                                encode_attention_decode_tq(
                                    context,
                                    &pass,
                                    (&q, 0),
                                    k,
                                    v,
                                    scratch,
                                    (&out, 0),
                                    head_dim,
                                    nq,
                                    nkv,
                                    seq,
                                    kv_start,
                                    scale,
                                    tables,
                                    sinks.as_ref().map(|b| (b, 0)),
                                )
                                .map_err(|e| e.to_string())?;
                            }
                        }
                    }
                    Ok(pass.commit_and_wait_with_gpu_time())
                })?;
                if rep > 0 {
                    samples[a].push(t * 1e3);
                }
                if opts.hash && rep == opts.reps {
                    hashes[a] = Some(fnv_hex(&read_buffer_bytes(&out, 0, q_width * 2)));
                }
            }
        }
        for (a, sm) in samples.iter_mut().enumerate() {
            // Bytes the arm's kernel reads: FP16 K and V rows, or the packed
            // TurboQuant rows (norm word plus index words, K and V).
            let row_bytes = match &lanes[a].1 {
                Lane::Dense(_) => kv_width as f64 * 4.0,
                Lane::Tq { row_bytes, .. } => (*row_bytes * nkv as usize) as f64,
            };
            let ms = median(sm);
            let mut p = Point::new(seq as f64, ms);
            p.gb_s = Some(s.layers as f64 * read * row_bytes / (ms * 1e-3) / 1e9);
            p.out_fnv = hashes[a].take();
            curves[a].push(p);
        }
    }
    let mut arms: Vec<ArmCurve> = lanes
        .iter()
        .zip(curves)
        .map(|((name, _), points)| ArmCurve {
            label: name.to_string(),
            points,
            fit: None,
            unsupported: None,
        })
        .collect();
    let mut notes = Vec::new();
    if !gqa {
        arms.push(ArmCurve::unsupported(
            "gqa",
            "the GQA-shared kernel does not serve this head shape",
        ));
    }
    if !tq_served {
        for (label, _) in TQ_ARMS {
            arms.push(ArmCurve::unsupported(
                label,
                "TurboQuant serves full-attention layers with an RHT-supported head_dim only",
            ));
        }
    }
    if s.window > 0 {
        notes.push(format!(
            "sliding window {}: reads min(context, window) positions from a ring",
            s.window
        ));
    }
    let mut subject = BTreeMap::new();
    for (k, v) in [
        ("q_heads", json!(nq)),
        ("kv_heads", json!(nkv)),
        ("head_dim", json!(head_dim)),
        ("layers", json!(s.layers)),
        ("window", json!(s.window)),
        ("sinks", json!(s.sinks)),
        ("gqa_supported", json!(gqa)),
    ] {
        subject.insert(k.to_string(), v);
    }
    Ok(BenchResult {
        bench: "attn_decode".into(),
        case: label.to_string(),
        subject,
        x: Axis::new("context", "positions", "log2"),
        layout: "fp16_kv_linear".into(),
        arms,
        break_even: Vec::new(),
        notes,
    })
}
