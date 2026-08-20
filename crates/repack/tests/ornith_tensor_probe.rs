//! Which installed tensor disagrees with the published BF16 checkpoint?
//!
//! The instrument AGENTS.md Gotcha 33 prescribes for "loads, decodes, never
//! errors, and the output is wrong": correlate every resident tensor against
//! an independent reading of the same weights, and let the ranking name the
//! culprit rather than guessing at conventions one at a time.
//!
//! ```sh
//! TURBOSPARK_ORNITH9B_INSTALL_DIR=~/models/ornith9b.gturbo \
//!   cargo test -p turbospark-repack --test ornith_tensor_probe --release -- --ignored --nocapture
//! ```
//!
//! **The reference is the BF16 safetensors repo, read by RANGE.** That is
//! what makes this cheap: a row of a `[out, in]` matrix is contiguous on both
//! sides (safetensors is row-major, and every GGUF block layout tiles along
//! the fastest-varying dim), so a few KB per tensor settles it. No download,
//! no second install -- which matters here because unlike Qwen 3.6 there is
//! no quantized MLX artifact of this model to compare against.
//!
//! **CORRELATION, NEVER EQUALITY, and PER ROW.** The two sides hold different
//! quantizations of the same trained weights, so agreement is a correlation
//! (the discipline `gguf_q4_k_network.rs` established). Per row rather than
//! pooled because a real checkpoint has all-zero rows inside a routed expert
//! and `pearson` returns 0.0 on a constant input BY CONTRACT -- pooling them
//! divides a good result by the number of dead rows and reads exactly like a
//! broken unpacker (Gotcha 30, which cost 45 minutes the first time).

use std::collections::BTreeMap;
use std::path::PathBuf;

use compute::{dequantize_q4_k, dequantize_q6_k, dequantize_q8_0, pearson};
use turbospark_repack::{HttpRangeSource, RangeSource, SafetensorsHeader};

/// The BF16 checkpoint the GGUF was converted from, pinned by revision.
const REF_BASE: &str = "https://huggingface.co/ornith-ai/Ornith-1.5-9B/resolve/98db59be66b580b0395b3dc8237b32eefcdfec22";

/// ggml type ids, for turning an install dtype tag back into a decoder.
const GGML_F32: u32 = 0;
const GGML_Q8_0: u32 = 8;
const GGML_Q4_K: u32 = 12;
const GGML_Q6_K: u32 = 14;

/// Rows sampled per tensor. Enough that a constant row or two cannot decide
/// the verdict, few enough that the whole probe is a handful of KB.
const ROWS: usize = 6;

struct Reference {
    shard_of: BTreeMap<String, String>,
    headers: BTreeMap<String, (SafetensorsHeader, HttpRangeSource)>,
}

impl Reference {
    fn new() -> Self {
        let body = reqwest::blocking::Client::builder()
            .timeout(None)
            .build()
            .expect("client")
            .get(format!("{REF_BASE}/model.safetensors.index.json"))
            .send()
            .expect("index")
            .text()
            .expect("index body");
        let json: serde_json::Value = serde_json::from_str(&body).expect("index json");
        let map = json["weight_map"].as_object().expect("weight_map");
        Self {
            shard_of: map
                .iter()
                .map(|(k, v)| (k.clone(), v.as_str().expect("shard").to_string()))
                .collect(),
            headers: BTreeMap::new(),
        }
    }

    /// One BF16 row of `name`, as f32.
    fn row(&mut self, name: &str, row: usize, cols: usize) -> Option<Vec<f32>> {
        let shard = self.shard_of.get(name)?.clone();
        if !self.headers.contains_key(&shard) {
            let source = HttpRangeSource::new(format!("{REF_BASE}/{shard}"));
            let header =
                turbospark_repack::fetch_safetensors_header(&source).expect("safetensors header");
            self.headers.insert(shard.clone(), (header, source));
        }
        let (header, source) = &self.headers[&shard];
        let info = header.tensors.get(name)?;
        assert_eq!(info.dtype, "BF16", "{name}: reference must be BF16");
        let base = header.data_region_start() + info.data_offsets.0;
        let start = base + (row * cols * 2) as u64;
        let bytes = source
            .read_range(start, start + (cols * 2) as u64)
            .expect("reference row");
        Some(
            bytes
                .chunks_exact(2)
                .map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16))
                .collect(),
        )
    }
}

/// One row of an installed tensor, dequantized to f32.
fn install_row(
    bytes: &[u8],
    entry: &model_io::ResidentIndexEntry,
    row: usize,
    cols: usize,
) -> Vec<f32> {
    let ggml = [GGML_F32, GGML_Q8_0, GGML_Q4_K, GGML_Q6_K]
        .into_iter()
        .find(|t| turbospark_repack::dtype_tag_for_ggml_type(*t) == Some(entry.dtype));
    let at = entry.file_offset as usize;
    match ggml {
        // Q4_K and Q6_K tile along the fastest-varying dim, so a row is a
        // contiguous run of whole superblocks.
        Some(GGML_Q4_K) => {
            let per = cols / 256 * 144;
            dequantize_q4_k(&bytes[at + row * per..at + (row + 1) * per], cols)
        }
        Some(GGML_Q6_K) => {
            let per = cols / 256 * 210;
            dequantize_q6_k(&bytes[at + row * per..at + (row + 1) * per], cols)
        }
        Some(GGML_Q8_0) => {
            let per = cols / 32 * 34;
            dequantize_q8_0(&bytes[at + row * per..at + (row + 1) * per], cols)
        }
        // Everything unquantized is BF16 in an install, whatever it was in
        // the source (repack Gotcha 9).
        _ => {
            let per = cols * 2;
            bytes[at + row * per..at + (row + 1) * per]
                .chunks_exact(2)
                .map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16))
                .collect()
        }
    }
}

/// `(install name tail, reference name tail)` for one layer. They differ:
/// this port's canonical prefix is `language_model.model.layers.N.` and the
/// published checkpoint's is `model.language_model.layers.N.` -- the
/// transformers 5.x spelling, with the two components swapped.
const LINEAR_TAILS: &[&str] = &[
    "linear_attn.in_proj_qkv.weight",
    "linear_attn.in_proj_z.weight",
    "linear_attn.in_proj_a.weight",
    "linear_attn.in_proj_b.weight",
    "linear_attn.out_proj.weight",
    "linear_attn.conv1d.weight",
    "linear_attn.norm.weight",
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
];

const FULL_TAILS: &[&str] = &[
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "mlp.gate_proj.weight",
    "mlp.down_proj.weight",
];

#[test]
#[ignore = "network: reads a few KB per tensor off the reference checkpoint"]
fn every_installed_tensor_agrees_with_the_published_checkpoint() {
    let dir = PathBuf::from(
        std::env::var_os("TURBOSPARK_ORNITH9B_INSTALL_DIR")
            .expect("TURBOSPARK_ORNITH9B_INSTALL_DIR must point at an Ornith-1.5-9B install"),
    );
    let path = dir.join("model_weights.bin");
    let index = model_io::load_resident_index(&path).expect("resident index");
    let bytes = std::fs::read(&path).expect("weights");
    let mut reference = Reference::new();

    // Layer 0 is LINEAR and layer 3 is FULL, on this architecture's
    // three-to-one hybrid. Probing only one kind would miss half the model,
    // and probing layer 0 alone is the mistake `qwen3moe`'s divergence test
    // made in a different form.
    let mut probes: Vec<(usize, &str)> = LINEAR_TAILS.iter().map(|t| (0usize, *t)).collect();
    probes.extend(FULL_TAILS.iter().map(|t| (3usize, *t)));

    let mut results: Vec<(f32, String, usize)> = Vec::new();
    for (layer, tail) in probes {
        let installed = format!("language_model.model.layers.{layer}.{tail}");
        let referenced = format!("model.language_model.layers.{layer}.{tail}");
        let Some(entry) = index.entries.get(&installed) else {
            println!("  {installed}: ABSENT from the install");
            continue;
        };
        // Shape is (rows, cols, _, _) with rank padded to 4; a rank-1 tensor
        // is one row.
        let (rows, cols) = if entry.shape.1 == 0 {
            (1usize, entry.shape.0 as usize)
        } else {
            (entry.shape.0 as usize, entry.shape.1 as usize)
        };

        let mut scores: Vec<f32> = Vec::new();
        let mut sampled = 0usize;
        for i in 0..rows {
            if sampled >= ROWS {
                break;
            }
            let step = (rows / ROWS.max(1)).max(1);
            if i % step != 0 {
                continue;
            }
            let a = install_row(&bytes, entry, i, cols);
            let Some(b) = reference.row(&referenced, i, cols) else {
                println!("  {referenced}: ABSENT from the reference");
                break;
            };
            // Gotcha 30: a constant row scores 0.0 by contract and must not
            // be averaged in as a disagreement.
            let constant = |v: &[f32]| v.iter().all(|x| (*x - v[0]).abs() < f32::EPSILON);
            if constant(&a) || constant(&b) {
                continue;
            }
            scores.push(pearson(&a, &b));
            sampled += 1;
        }
        if scores.is_empty() {
            println!("  {installed}: no non-constant row sampled");
            continue;
        }
        let mean = scores.iter().sum::<f32>() / scores.len() as f32;
        results.push((mean, format!("L{layer} {tail}"), scores.len()));
    }

    // WORST FIRST: the ranking is the diagnostic.
    //
    // NaN sorts to the TOP rather than panicking the comparator. A
    // non-finite correlation is the single most interesting outcome here --
    // it means a dequantized row is itself non-finite, which is a stronger
    // statement than "disagrees" and is exactly what the engine's sampler
    // reported. Losing it to an `.expect("finite")` would throw away the
    // finding this probe exists to produce.
    results.sort_by(|a, b| match (a.0.is_nan(), b.0.is_nan()) {
        (true, true) => std::cmp::Ordering::Equal,
        (true, false) => std::cmp::Ordering::Less,
        (false, true) => std::cmp::Ordering::Greater,
        (false, false) => a.0.partial_cmp(&b.0).expect("both finite"),
    });
    println!("\n  mean |pearson| per tensor, WORST FIRST:");
    for (score, name, n) in &results {
        let verdict = if score.is_nan() {
            "NON-FINITE"
        } else if *score > 0.9 {
            "ok"
        } else if *score > 0.5 {
            "SUSPECT"
        } else {
            "WRONG"
        };
        println!("    {score:>9.5}  n={n}  {name:<42} {verdict}");
    }

    // `is_nan() || < 0.9` rather than `!(>= 0.9)`: same set, but it states
    // that a NON-FINITE score is a failure in its own right rather than
    // relying on NaN's comparison behaviour to fall through a negation.
    let bad: Vec<&(f32, String, usize)> = results
        .iter()
        .filter(|(s, _, _)| s.is_nan() || *s < 0.9)
        .collect();
    assert!(
        bad.is_empty(),
        "{} tensor(s) disagree with the published checkpoint: {:?}",
        bad.len(),
        bad.iter().map(|(s, n, _)| (n, s)).collect::<Vec<_>>()
    );
}
