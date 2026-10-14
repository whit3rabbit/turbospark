//! One measured run of EmbeddingGemma 300m (the Gemma 3 one) on one
//! backend, for `crates/runtime/scripts/benchmark_embeddinggemma3.py`.
//!
//! ```text
//! embeddinggemma3_probe ids <cpu-model-dir> <texts.json> <ids.json>
//! embeddinggemma3_probe cpu <cpu-model-dir> <ids.json> <out.json>
//! embeddinggemma3_probe coreml_<cpu|gpu|ane|all> <coreml-dir> <ids.json> <out.json>
//! ```
//!
//! `ids` tokenizes the texts once with the CPU port (document prompt,
//! `<bos>`/`<eos>` included), so every arm embeds identical token ids and
//! the timing isolates the model. Each arm: open, one discarded warmup pass
//! over all ids, one measured pass, embeddings written to `out.json`.
//!
//! - `cpu`: the portable f32 port in `crates/embeddings`, on an MLX bf16
//!   checkpoint directory (dequantized to f32 at load).
//! - `coreml_*`: a sentence-transformers Core ML export with `input_ids`
//!   (int32) and `attention_mask` (int32 or fp16, whichever the model
//!   declares) in and a normalized `embedding` out, run through
//!   `turbospark-coreml` on the named compute units. The directory holds one
//!   `.mlpackage` or compiled `.mlmodelc`. Lengths are enumerated (64 or 256)
//!   or one fixed length. Right padding with id 0.
//!
//! macOS-only.

#![cfg(target_os = "macos")]

use std::path::Path;
use std::time::Instant;

use turbospark_coreml::{ComputeUnits, CoreMlModel, DType, FeatureKind, LoadOptions, Tensor};
use turbospark_embeddings::embeddinggemma::EmbeddingGemma;

type Res<T> = Result<T, String>;

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

fn read_json<T: serde::de::DeserializeOwned>(p: &str) -> Res<T> {
    serde_json::from_slice(&std::fs::read(p).map_err(|e| format!("{p}: {e}"))?)
        .map_err(|e| format!("{p}: {e}"))
}

fn write_out(
    path: &str,
    arm: &str,
    open_ms: f64,
    warmup_ms: f64,
    measured_ms: f64,
    v: &[Vec<f32>],
) -> Res<()> {
    let json = serde_json::json!({
        "arm": arm, "open_ms": open_ms, "warmup_ms": warmup_ms, "measured_ms": measured_ms,
        "texts": v.len(), "embeddings": v,
    });
    std::fs::write(path, json.to_string()).map_err(|e| e.to_string())?;
    eprintln!(
        "embeddinggemma3_probe arm={arm} open_ms={open_ms:.2} warmup_ms={warmup_ms:.2} \
         measured_ms={measured_ms:.2} texts={} texts_per_s={:.1}",
        v.len(),
        v.len() as f64 / (measured_ms / 1e3)
    );
    Ok(())
}

/// Runs `f` over every id list twice (warmup, measured) and records.
fn timed(
    arm: &str,
    open_ms: f64,
    ids: &[Vec<u32>],
    out: &str,
    f: impl Fn(&[u32]) -> Res<Vec<f32>>,
) -> Res<()> {
    let t = Instant::now();
    for i in ids {
        f(i)?;
    }
    let warmup_ms = ms(t);
    let t = Instant::now();
    let v = ids.iter().map(|i| f(i)).collect::<Res<Vec<_>>>()?;
    write_out(out, arm, open_ms, warmup_ms, ms(t), &v)
}

fn coreml(arm: &str, dir: &Path, ids: &[Vec<u32>], out: &str) -> Res<()> {
    let units = match arm {
        "coreml_cpu" => ComputeUnits::CpuOnly,
        "coreml_gpu" => ComputeUnits::CpuAndGpu,
        "coreml_ane" => ComputeUnits::CpuAndNeuralEngine,
        "coreml_all" => ComputeUnits::All,
        other => return Err(format!("unknown arm {other}")),
    };
    // Not recursive, so the `.compiled` cache next to a package is skipped.
    let package = std::fs::read_dir(dir)
        .map_err(|e| e.to_string())?
        .flatten()
        .map(|e| e.path())
        .find(|p| {
            p.extension()
                .is_some_and(|x| x == "mlpackage" || x == "mlmodelc")
        })
        .ok_or("no .mlpackage or .mlmodelc in the Core ML directory")?;
    let t = Instant::now();
    let m = CoreMlModel::open(&package, &dir.join(".compiled"), &LoadOptions::new(units))
        .map_err(|e| e.to_string())?;
    let open_ms = ms(t);
    // The export accepts a fixed set of lengths (64 and 256 here); use the
    // smallest that fits. A fixed-shape export reports one length.
    let lengths: Vec<usize> = match m
        .inputs()
        .iter()
        .find(|s| s.name == "input_ids")
        .map(|s| &s.kind)
    {
        Some(FeatureKind::MultiArray { shape: Some(s), .. }) => vec![s[1]],
        _ => vec![64, 256],
    };
    let mask_dtype = match m
        .inputs()
        .iter()
        .find(|s| s.name == "attention_mask")
        .map(|s| &s.kind)
    {
        Some(FeatureKind::MultiArray { dtype: Some(d), .. }) => *d,
        other => return Err(format!("attention_mask: unsupported declaration {other:?}")),
    };
    timed(arm, open_ms, ids, out, |ids| {
        let len = lengths
            .iter()
            .copied()
            .find(|&l| l >= ids.len())
            .ok_or_else(|| format!("{} tokens exceed the export's lengths", ids.len()))?;
        let mut input = vec![0i32; len];
        for (i, &id) in ids.iter().enumerate() {
            input[i] = id as i32;
        }
        let input = Tensor::i32(vec![1, len], input).map_err(|e| e.to_string())?;
        let ones = |i: usize| i < ids.len();
        let mask = match mask_dtype {
            DType::I32 => Tensor::i32(vec![1, len], (0..len).map(|i| ones(i) as i32).collect()),
            DType::F16 => Tensor::f16_from_f32(
                "attention_mask",
                vec![1, len],
                &(0..len).map(|i| ones(i) as u8 as f32).collect::<Vec<_>>(),
            ),
            DType::F32 => Tensor::f32(
                vec![1, len],
                (0..len).map(|i| ones(i) as u8 as f32).collect(),
            ),
        }
        .map_err(|e| e.to_string())?;
        let r = m
            .predict(&[("input_ids", &input), ("attention_mask", &mask)])
            .map_err(|e| e.to_string())?;
        r.get("embedding")
            .and_then(Tensor::to_f32_vec)
            .ok_or_else(|| "no float embedding output".into())
    })
}

fn main() -> Res<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let usage = "usage: embeddinggemma3_probe ids <cpu-dir> <texts.json> <ids.json> | cpu <cpu-dir> <ids.json> <out.json> | coreml_<units> <coreml-dir> <ids.json> <out.json>";
    if a.len() != 4 {
        return Err(usage.into());
    }
    match a[0].as_str() {
        "ids" => {
            let m = EmbeddingGemma::load(&a[1]).map_err(|e| e.to_string())?;
            let texts: Vec<String> = read_json(&a[2])?;
            let ids = texts
                .iter()
                .map(|t| m.tokenize(t, Some("document")).map_err(|e| e.to_string()))
                .collect::<Res<Vec<_>>>()?;
            std::fs::write(&a[3], serde_json::to_string(&ids).unwrap()).map_err(|e| e.to_string())
        }
        "cpu" => {
            let ids: Vec<Vec<u32>> = read_json(&a[2])?;
            let t = Instant::now();
            let m = EmbeddingGemma::load(&a[1]).map_err(|e| e.to_string())?;
            let open_ms = ms(t);
            timed("cpu", open_ms, &ids, &a[3], |i| {
                m.embed_ids(i).map_err(|e| e.to_string())
            })
        }
        arm if arm.starts_with("coreml_") => {
            let ids: Vec<Vec<u32>> = read_json(&a[2])?;
            coreml(arm, Path::new(&a[1]), &ids, &a[3])
        }
        _ => Err(usage.into()),
    }
}
