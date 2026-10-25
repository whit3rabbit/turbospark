//! Whisper's f32 projections at the install's own widths: the tiled GEMM
//! the encoder runs over its frames against the GEMV the decoder runs per
//! token, and the tiled GEMM at one row (why the decoder has a GEMV).
//!
//! Shapes come from the install's `config.json` (`d_model`, the FFN widths,
//! `max_source_positions`, `vocab_size`). WHISPER.md puts the encoder at
//! about 46 of 160 ms warm on this GEMM, near 250 GFLOPS, and names a
//! simdgroup-tiled kernel as the lever (`docs/SPEEDUP_PLAYBOOK.md`, item 3):
//! this bench is the curve that lever would move.

use std::collections::BTreeMap;

use gpu::{
    autorelease_pool, encode_whisper_gemv, encode_whisper_matmul_bias, read_buffer_bytes, F32View,
    MetalContext,
};
use serde_json::{json, Value};

use super::support::{copies_for, median};
use super::{BenchOpts, Case, KernelBench, Subject};
use crate::bench::{fnv_hex, ArmCurve, Axis, BenchResult, Point};

pub struct WhisperLinear;

/// (label, output rows n, input width k).
pub type Projection = (String, usize, usize);

/// The projections of one Whisper config, and its encoder frame count.
pub fn shapes(cfg: &Value) -> Result<(Vec<Projection>, usize), String> {
    let u = |k: &str| cfg[k].as_u64().map(|v| v as usize);
    let d = u("d_model").ok_or("config.json has no d_model")?;
    let frames = u("max_source_positions").unwrap_or(1500);
    let mut out = vec![("attn_proj".to_string(), d, d)];
    let enc = u("encoder_ffn_dim");
    let dec = u("decoder_ffn_dim");
    for (name, f) in [("encoder_ffn", enc), ("decoder_ffn", dec)] {
        if let Some(f) = f {
            if !out.iter().any(|(_, n, k)| (*n, *k) == (f, d)) {
                out.push((format!("{name}_up"), f, d));
                out.push((format!("{name}_down"), d, f));
            }
        }
    }
    if let Some(v) = u("vocab_size") {
        out.push(("logits".into(), v, d));
    }
    Ok((out, frames))
}

impl KernelBench for WhisperLinear {
    fn name(&self) -> &'static str {
        "whisper_linear"
    }

    fn describe(&self) -> &'static str {
        "Whisper f32 projections: encoder tiled GEMM over its frames, decoder GEMV, GEMM at one row"
    }

    fn cases(&self, subject: &Subject, _opts: &BenchOpts) -> Result<Vec<Case>, String> {
        let cfg = subject
            .audio_config
            .as_ref()
            .filter(|_| subject.family == "whisper")
            .ok_or("not a Whisper install")?;
        let (shapes, frames) = shapes(cfg)?;
        Ok(shapes
            .into_iter()
            .map(|(label, n, k)| Case {
                label: format!("{label}_{n}x{k}"),
                params: json!({"n": n, "k": k, "frames": frames,
                               "decoder_only": label == "logits"}),
            })
            .collect())
    }

    fn run(
        &self,
        context: &mut MetalContext,
        _subject: &Subject,
        case: &Case,
        opts: &BenchOpts,
    ) -> Result<BenchResult, String> {
        let u = |k: &str| case.params[k].as_u64().unwrap_or(0) as usize;
        run_shape(
            context,
            &case.label,
            u("n"),
            u("k"),
            u("frames"),
            case.params["decoder_only"] == true,
            opts,
        )
    }
}

fn f32_bytes(values: impl Iterator<Item = f32>) -> Vec<u8> {
    values.flat_map(|v| v.to_le_bytes()).collect()
}

pub fn run_shape(
    context: &mut MetalContext,
    label: &str,
    n: usize,
    k: usize,
    frames: usize,
    decoder_only: bool,
    opts: &BenchOpts,
) -> Result<BenchResult, String> {
    let weight_bytes = n * k * 4;
    let copies = copies_for(weight_bytes, 256 << 20);
    let ws: Vec<_> = (0..copies)
        .map(|c| {
            context.new_buffer_with_data(&f32_bytes(
                (0..n * k).map(|i| (((i + c * 7) % 23) as f32 - 11.0) / 512.0),
            ))
        })
        .collect();
    let bias = context.new_buffer_with_data(&f32_bytes((0..n).map(|i| (i % 5) as f32 / 64.0)));
    let rows = if decoder_only { 1 } else { frames.max(1) };
    let x = context.new_buffer_with_data(&f32_bytes(
        (0..rows * k).map(|i| ((i as f32) * 0.013).sin() * 0.5),
    ));
    let y = context.new_output_buffer((rows * n * 4) as u64);
    // (arm, m): the GEMV at one row; the tiled GEMM at one row and at the
    // encoder's frame count.
    let mut arms: Vec<(&str, usize)> = vec![("decoder_gemv", 1), ("tiled_gemm", 1)];
    if !decoder_only {
        arms.push(("tiled_gemm", frames));
    }
    let mut samples: Vec<Vec<f64>> = vec![Vec::new(); arms.len()];
    let mut hashes: Vec<Option<String>> = vec![None; arms.len()];
    for rep in 0..=opts.reps {
        for (a, &(arm, m)) in arms.iter().enumerate() {
            let t = autorelease_pool(|| -> Result<f64, String> {
                let pass = context.begin_pass();
                for w in &ws {
                    let r = if arm == "decoder_gemv" {
                        encode_whisper_gemv(
                            context,
                            &pass,
                            F32View::new(&x),
                            F32View::new(w),
                            Some(F32View::new(&bias)),
                            None,
                            F32View::new(&y),
                            k as u32,
                            n as u32,
                            k as u32,
                            false,
                            1.0,
                        )
                    } else {
                        encode_whisper_matmul_bias(
                            context,
                            &pass,
                            F32View::new(&x),
                            F32View::new(w),
                            Some(F32View::new(&bias)),
                            F32View::new(&y),
                            m as u32,
                            k as u32,
                            n as u32,
                            k as u32,
                            k as u32,
                            n as u32,
                            false,
                            1.0,
                        )
                    };
                    r.map_err(|e| e.to_string())?;
                }
                Ok(pass.commit_and_wait_with_gpu_time())
            })?;
            if rep > 0 {
                samples[a].push(t * 1e3 / copies as f64);
            }
            if opts.hash && rep == opts.reps {
                hashes[a] = Some(fnv_hex(&read_buffer_bytes(&y, 0, m * n * 4)));
            }
        }
    }
    let mut curves: BTreeMap<&str, Vec<Point>> = BTreeMap::new();
    for (a, &(arm, m)) in arms.iter().enumerate() {
        let ms = median(&mut samples[a]);
        let mut p = Point::new(m as f64, ms);
        p.gb_s = Some(weight_bytes as f64 / (ms * 1e-3) / 1e9);
        p.tflops = Some(2.0 * m as f64 * n as f64 * k as f64 / (ms * 1e-3) / 1e12);
        p.out_fnv = hashes[a].take();
        curves.entry(arm).or_default().push(p);
    }
    let mut subject = BTreeMap::new();
    subject.insert("n".into(), json!(n));
    subject.insert("k".into(), json!(k));
    subject.insert("frames".into(), json!(frames));
    subject.insert("copies".into(), json!(copies));
    Ok(BenchResult {
        bench: "whisper_linear".into(),
        case: label.to_string(),
        subject,
        x: Axis::new("m", "rows", "log2"),
        layout: "f32_linear_nk".into(),
        arms: curves
            .into_iter()
            .map(|(label, points)| ArmCurve {
                label: label.into(),
                points,
                fit: None,
                unsupported: None,
            })
            .collect(),
        break_even: Vec::new(),
        notes: Vec::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// openai/whisper-tiny.en's widths.
    #[test]
    fn tiny_en_shapes() {
        let cfg = json!({"d_model": 384, "encoder_ffn_dim": 1536, "decoder_ffn_dim": 1536,
                         "max_source_positions": 1500, "vocab_size": 51864});
        let (s, frames) = shapes(&cfg).unwrap();
        assert_eq!(frames, 1500);
        let pairs: Vec<(usize, usize)> = s.iter().map(|(_, n, k)| (*n, *k)).collect();
        assert_eq!(
            pairs,
            vec![(384, 384), (1536, 384), (384, 1536), (51864, 384)]
        );
        assert!(shapes(&json!({})).is_err());
    }

    /// Runs the real kernels once at whisper-tiny's attention width: the
    /// bench's plumbing, not a measurement.
    #[test]
    fn runs_on_metal() {
        let mut ctx = MetalContext::new().expect("Metal device");
        let opts = BenchOpts {
            reps: 1,
            hash: true,
            ..BenchOpts::default()
        };
        let r = run_shape(&mut ctx, "attn_proj_384x384", 384, 384, 64, false, &opts).unwrap();
        assert_eq!(r.arm("tiled_gemm").unwrap().points.len(), 2);
        assert!(r.arm("decoder_gemv").unwrap().points[0].out_fnv.is_some());
    }
}
