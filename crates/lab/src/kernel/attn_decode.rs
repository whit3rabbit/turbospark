//! Decode attention at the install's full-attention head shape: the barrier
//! kernel against the GQA-shared kernel, over context lengths, through the
//! production dispatcher (`AttentionScratch::with_decode_kernel`) rather than
//! hand-picked tuning parameters.
//!
//! The port of `crates/gpu/tests/qwen38_attention_decode_probe.rs` with the
//! head shape and layer count taken from `ArchConfig`.

use gpu::{
    attention_decode_gqa_supported, autorelease_pool, encode_attention_decode, AttentionScratch,
    DecodeAttentionKernel, MetalContext,
};
use model_io::ArchConfig;
use serde_json::{json, Value};

use super::support::{f16_bytes, median};

/// (q heads, kv heads, head dim, full-attention layer count), or `None` when
/// the install has no plain full-attention layers this bench can drive.
pub fn shape(arch: &ArchConfig) -> Option<(u32, u32, u32, usize)> {
    let layers = (0..arch.num_layers as usize)
        .filter(|&l| arch.layer_is_full(l))
        .count();
    let (q, kv, d) = (
        arch.num_heads as u32,
        arch.num_full_kv_heads.max(arch.num_kv_heads) as u32,
        arch.full_head_dim as u32,
    );
    (layers > 0 && q > 0 && kv > 0 && d > 0 && q % kv == 0).then_some((q, kv, d, layers))
}

pub fn run(
    context: &mut MetalContext,
    arch: &ArchConfig,
    contexts: &[u32],
    reps: usize,
) -> Result<Option<Value>, String> {
    let Some((nq, nkv, head_dim, layers)) = shape(arch) else {
        return Ok(None);
    };
    let gqa = attention_decode_gqa_supported(head_dim, nq, nkv);
    let max_seq = *contexts.iter().max().unwrap_or(&512) as usize;
    let kv_width = (nkv * head_dim) as usize;
    let q_width = (nq * head_dim) as usize;
    let q = context.new_buffer_with_data(&f16_bytes(
        (0..q_width).map(|i| ((i as f32) * 0.019).sin() * 0.5),
    ));
    // One K/V pair per full-attention layer, so a sample reads each layer's
    // cache once, as a decode step does.
    let kv: Vec<_> = (0..layers)
        .map(|layer| {
            let phase = layer as f32 * 0.37;
            let k = context.new_buffer_with_data(&f16_bytes(
                (0..max_seq * kv_width).map(|i| ((i as f32) * 0.0007 + phase).cos() * 0.3),
            ));
            let v = context.new_buffer_with_data(&f16_bytes(
                (0..max_seq * kv_width).map(|i| 0.7 + ((i as f32) * 0.0011 + phase).sin() * 0.3),
            ));
            (k, v)
        })
        .collect();
    let out = context.new_output_buffer((q_width * 2) as u64);
    let mut kernels = vec![("barrier", DecodeAttentionKernel::Barrier)];
    if gqa {
        kernels.push(("gqa", DecodeAttentionKernel::GqaShared));
    }
    let scratches: Vec<_> = kernels
        .iter()
        .map(|(_, k)| AttentionScratch::new(context, nq, head_dim).with_decode_kernel(*k))
        .collect();
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut points = Vec::new();
    for &seq in contexts {
        let mut samples = vec![Vec::new(); kernels.len()];
        for rep in 0..=reps {
            for (a, scratch) in scratches.iter().enumerate() {
                let t = autorelease_pool(|| -> Result<f64, String> {
                    let pass = context.begin_pass();
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
                            0,
                            0,
                            scale,
                            None,
                        )
                        .map_err(|e| e.to_string())?;
                    }
                    Ok(pass.commit_and_wait_with_gpu_time())
                })?;
                if rep > 0 {
                    samples[a].push(t * 1e3);
                }
            }
        }
        let kv_bytes = layers as f64 * seq as f64 * kv_width as f64 * 4.0;
        let row: Vec<Value> = kernels
            .iter()
            .zip(samples.iter_mut())
            .map(|((name, _), s)| {
                let ms = median(s);
                json!({"kernel": name, "ms": ms, "kv_gb_s": kv_bytes / (ms * 1e-3) / 1e9})
            })
            .collect();
        points.push(json!({"context": seq, "kernels": row}));
    }
    Ok(Some(json!({
        "bench": "attn_decode",
        "q_heads": nq,
        "kv_heads": nkv,
        "head_dim": head_dim,
        "layers": layers,
        "gqa_supported": gqa,
        "points": points,
    })))
}

pub fn print(rec: &Value) {
    println!(
        "attention decode: {} q / {} kv heads, head_dim {}, {} full layers{}",
        rec["q_heads"],
        rec["kv_heads"],
        rec["head_dim"],
        rec["layers"],
        if rec["gqa_supported"] == true {
            ""
        } else {
            " (GQA kernel does not serve this shape)"
        }
    );
    for p in rec["points"].as_array().into_iter().flatten() {
        let ks: Vec<String> = p["kernels"]
            .as_array()
            .into_iter()
            .flatten()
            .map(|k| {
                format!(
                    "{} {:.3} ms ({:.0} GB/s)",
                    k["kernel"].as_str().unwrap_or("?"),
                    k["ms"].as_f64().unwrap_or(f64::NAN),
                    k["kv_gb_s"].as_f64().unwrap_or(f64::NAN)
                )
            })
            .collect();
        println!("  ctx {:>6}: {}", p["context"], ks.join(" | "));
    }
}
