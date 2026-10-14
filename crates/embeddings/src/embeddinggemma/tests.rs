//! Parity and refusal tests. Fixtures come from
//! `tools/gen_embeddinggemma_fixtures.py`; see the family README for the
//! regeneration commands and the measured numbers behind each bound.

use std::path::PathBuf;
use std::time::Instant;

use serde_json::Value;

use super::backbone::{band_mask, GemmaNorm};
use super::config::GemmaConfig;
use super::*;
use crate::testutil::{max_abs_diff, model_dir, read_json, read_npy, testdata};

const FAMILY: &str = "embeddinggemma";

// Tiny random model vs HF f32. Measured max abs diff is reported by the
// test; the bound leaves headroom over f32 reduction-order noise only
// (hidden values are O(1..10), so 1e-4 abs is ~1e-5 relative) and is far
// below what a wrong mask, norm, or RoPE base produces (O(0.1..1)).
const TINY_HIDDEN_TOL: f32 = 5e-5;
const TINY_EMBED_TOL: f32 = 2e-6;

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| f64::from(*x) * f64::from(*y))
        .sum();
    let na: f64 = a.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
    dot / (na * nb)
}

fn tiny_cases() -> Vec<Vec<u32>> {
    read_json(FAMILY, "tiny_cases.json")["cases"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| {
            c["ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u32)
                .collect()
        })
        .collect()
}

/// Runs every tiny case through `model` and compares each traced hidden
/// state and the final embedding with the `variant` goldens. Returns the
/// worst (hidden, embedding) diffs.
fn tiny_parity(model: &EmbeddingGemma, variant: &str) -> (f32, f32) {
    let layers = model.config().num_hidden_layers();
    let (mut worst_h, mut worst_e) = (0.0f32, 0.0f32);
    for (i, ids) in tiny_cases().iter().enumerate() {
        let (shape, golden) = read_npy(FAMILY, &format!("tiny_{variant}_hidden_{i}.npy"));
        assert_eq!(
            shape,
            vec![layers + 1, ids.len(), model.config().hidden_size]
        );
        let plane = ids.len() * model.config().hidden_size;
        let mut seen = 0;
        let last = model
            .backbone()
            .forward_traced(ids, |k, h| {
                let d = max_abs_diff(h, &golden[k * plane..(k + 1) * plane]);
                assert!(
                    d < TINY_HIDDEN_TOL,
                    "{variant} case {i} state {k}: max abs diff {d}"
                );
                worst_h = worst_h.max(d);
                seen += 1;
            })
            .unwrap();
        assert_eq!(seen, layers + 1);
        let emb = model.head(&last, ids.len()).unwrap();
        let (_, want) = read_npy(FAMILY, &format!("tiny_{variant}_embed_{i}.npy"));
        let d = max_abs_diff(&emb, &want);
        assert!(
            d < TINY_EMBED_TOL,
            "{variant} case {i} embedding: max abs diff {d}"
        );
        worst_e = worst_e.max(d);
        assert_eq!(model.embed_ids(ids).unwrap(), emb);
    }
    eprintln!("tiny {variant}: worst hidden diff {worst_h:e}, worst embedding diff {worst_e:e}");
    (worst_h, worst_e)
}

#[test]
fn tiny_hf_layout_matches_reference() {
    let model = EmbeddingGemma::load(testdata(FAMILY, "tiny_hf")).unwrap();
    assert_eq!(model.layout(), Layout::SentenceTransformers);
    assert_eq!(model.dimension(), 64);
    assert_eq!(model.max_seq_length(), 32);
    tiny_parity(&model, "hf");
}

#[test]
fn tiny_mlx_q4_layout_matches_dequantized_reference() {
    let model = EmbeddingGemma::load(testdata(FAMILY, "tiny_mlx_q4")).unwrap();
    assert_eq!(model.layout(), Layout::Mlx);
    assert_eq!(model.config().quantization.map(|q| q.bits), Some(4));
    tiny_parity(&model, "q4");
}

#[test]
fn tiny_sequences_exceed_the_window() {
    // Guards the fixture: without a case longer than the window the band
    // mask is never exercised by the parity tests.
    let model = EmbeddingGemma::load(testdata(FAMILY, "tiny_hf")).unwrap();
    let w = model.config().band_width();
    assert!(tiny_cases().iter().any(|c| c.len() > 2 * w));
    assert!(model.config().layer_types.contains(&LayerType::Sliding));
    assert!(model.config().layer_types.contains(&LayerType::Full));
}

#[test]
fn band_mask_is_strict_and_symmetric() {
    let m = band_mask(5, 2);
    let open: Vec<Vec<usize>> = (0..5)
        .map(|q| (0..5).filter(|&k| m[q * 5 + k] == 0.0).collect())
        .collect();
    assert_eq!(
        open,
        vec![
            vec![0, 1],
            vec![0, 1, 2],
            vec![1, 2, 3],
            vec![2, 3, 4],
            vec![3, 4]
        ]
    );
    assert!(m.iter().all(|&v| v == 0.0 || v == f32::NEG_INFINITY));
}

#[test]
fn gemma_norm_scales_by_one_plus_weight() {
    let norm = GemmaNorm::from_raw(&[0.0, 1.0], 0.0);
    let mut x = vec![3.0, 4.0];
    norm.apply(&mut x);
    // rms = sqrt(12.5); scales (1, 2)
    let r = 12.5f32.sqrt();
    assert_eq!(x, vec![3.0 / r, 2.0 * 4.0 / r]);
}

#[test]
fn assemble_keeps_special_tokens_when_truncating() {
    assert_eq!(
        assemble(&[2], &[10, 11, 12], &[1], 10),
        vec![2, 10, 11, 12, 1]
    );
    assert_eq!(assemble(&[2], &[10, 11, 12], &[1], 4), vec![2, 10, 11, 1]);
    assert_eq!(assemble(&[2], &[], &[1], 4), vec![2, 1]);
}

#[test]
fn py_strip_matches_python_whitespace() {
    assert_eq!(py_strip("  \t a b \n"), "a b");
    assert_eq!(py_strip("\u{1c}x\u{1f}"), "x");
    assert_eq!(py_strip("\u{3000}x\u{a0}"), "x");
    assert_eq!(py_strip("\u{200b}x"), "\u{200b}x"); // ZWSP is not whitespace in Python
}

fn tiny_config() -> Value {
    serde_json::from_slice(&std::fs::read(testdata(FAMILY, "tiny_hf/config.json")).unwrap())
        .unwrap()
}

#[test]
fn config_parses_tiny_and_derives_scales() {
    let c = GemmaConfig::from_json(&tiny_config()).unwrap();
    assert_eq!(c.num_hidden_layers(), 3);
    assert_eq!(c.attn_scale(), (12.0f64).powf(-0.5) as f32);
    assert_eq!(c.embed_scale(), 8.0);
    // transformers halves the configured window for bidirectional models.
    assert_eq!(c.band_width(), 3);
    let mut v = tiny_config();
    v["sliding_window"] = serde_json::json!(512);
    assert_eq!(GemmaConfig::from_json(&v).unwrap().band_width(), 257);
}

#[test]
fn config_refuses_unsupported_values() {
    let cases: Vec<(&str, Value)> = vec![
        ("use_bidirectional_attention", Value::Bool(false)),
        ("attn_logit_softcapping", serde_json::json!(50.0)),
        ("final_logit_softcapping", serde_json::json!(30.0)),
        (
            "rope_scaling",
            serde_json::json!({"rope_type": "linear", "factor": 8.0}),
        ),
        ("hidden_activation", serde_json::json!("gelu")),
        ("attention_bias", Value::Bool(true)),
        ("model_type", serde_json::json!("gemma3")),
        (
            "layer_types",
            serde_json::json!(["sliding_attention", "full_attention"]),
        ),
        (
            "layer_types",
            serde_json::json!(["sliding_attention", "chunked_attention", "full_attention"]),
        ),
        ("num_key_value_heads", serde_json::json!(3)),
        ("head_dim", serde_json::json!(15)),
        ("sliding_window", serde_json::json!(0)),
        (
            "quantization",
            serde_json::json!({"group_size": 64, "bits": 6}),
        ),
        (
            "quantization",
            serde_json::json!({"group_size": 32, "bits": 4}),
        ),
        (
            "quantization",
            serde_json::json!({"group_size": 64, "bits": 4, "model.layers.0.mlp.up_proj": {"bits": 8}}),
        ),
    ];
    for (key, value) in cases {
        let mut v = tiny_config();
        v[key] = value.clone();
        assert!(
            GemmaConfig::from_json(&v).is_err(),
            "{key} = {value} was accepted"
        );
    }
    let mut v = tiny_config();
    v.as_object_mut()
        .unwrap()
        .remove("use_bidirectional_attention");
    assert!(GemmaConfig::from_json(&v).is_err());
    let mut v = tiny_config();
    v["quantization"] = serde_json::json!({"group_size": 64, "bits": 8});
    assert!(GemmaConfig::from_json(&v).is_ok());
}

/// Copies the tiny ST fixture to a scratch dir for a mutated-load test.
fn scratch_copy(tag: &str) -> PathBuf {
    let src = testdata(FAMILY, "tiny_hf");
    let dst = std::env::temp_dir().join(format!("tse-gemma-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dst);
    for entry in ["", "1_Pooling", "2_Dense", "3_Dense"] {
        std::fs::create_dir_all(dst.join(entry)).unwrap();
        for f in std::fs::read_dir(src.join(entry)).unwrap() {
            let f = f.unwrap();
            if f.file_type().unwrap().is_file() {
                std::fs::copy(f.path(), dst.join(entry).join(f.file_name())).unwrap();
            }
        }
    }
    dst
}

#[test]
fn load_refuses_excluded_prompt_pooling() {
    let dir = scratch_copy("pool");
    let path = dir.join("1_Pooling/config.json");
    let mut v: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    v["include_prompt"] = Value::Bool(false);
    std::fs::write(&path, serde_json::to_vec(&v).unwrap()).unwrap();
    let err = EmbeddingGemma::load(&dir).unwrap_err();
    assert!(matches!(err, EmbeddingError::Unsupported { .. }), "{err}");
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn text_api_without_tokenizer_is_an_error_and_ids_are_bounded() {
    let model = EmbeddingGemma::load(testdata(FAMILY, "tiny_hf")).unwrap();
    assert!(matches!(
        model.embed("hi"),
        Err(EmbeddingError::Tokenizer { .. })
    ));
    assert!(model.embed_ids(&[]).is_err());
    assert!(model.embed_ids(&[2, 999, 1]).is_err()); // out of vocab
    assert!(model.embed_ids(&vec![3; 65]).is_err()); // > max_position_embeddings
    assert!(model.prompt_names().any(|n| n == "query"));
}

#[test]
fn truncate_dim_renormalizes_and_is_validated() {
    let model = EmbeddingGemma::load(testdata(FAMILY, "tiny_hf")).unwrap();
    let ids = &tiny_cases()[1];
    let full = model.embed_ids(ids).unwrap();
    let model = model.with_truncate_dim(Some(16)).unwrap();
    assert_eq!(model.dimension(), 16);
    let short = model.embed_ids(ids).unwrap();
    assert_eq!(short, truncate_and_normalize(&full, 16));
    assert!(model.with_truncate_dim(Some(65)).is_err());
}

// ------------------------------------------------------------ real checkpoints

struct RealCase {
    prompt_name: Option<String>,
    text: String,
    ids: Vec<u32>,
}

fn real_cases() -> Vec<RealCase> {
    read_json(FAMILY, "real_cases.json")["cases"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| RealCase {
            prompt_name: c["prompt_name"].as_str().map(str::to_owned),
            text: c["text"].as_str().unwrap().to_owned(),
            ids: c["ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u32)
                .collect(),
        })
        .collect()
}

fn real_goldens(name: &str) -> Vec<Vec<f32>> {
    let (shape, data) = read_npy(FAMILY, name);
    data.chunks_exact(shape[1]).map(<[f32]>::to_vec).collect()
}

/// Embeds every real case (from the golden ids).
fn real_embeddings(model: &EmbeddingGemma, label: &str) -> Vec<Vec<f32>> {
    real_cases()
        .iter()
        .enumerate()
        .map(|(i, case)| {
            let t = Instant::now();
            let got = model.embed_ids(&case.ids).unwrap();
            eprintln!(
                "{label} case {i}: {} tokens in {:.2?}",
                case.ids.len(),
                t.elapsed()
            );
            got
        })
        .collect()
}

/// (max abs diff, min cosine) of `got` against the golden file `golden`.
fn real_parity(got: &[Vec<f32>], golden: &str, label: &str) -> (f32, f64) {
    let (mut worst_d, mut worst_c) = (0.0f32, 1.0f64);
    for (i, (got, want)) in got.iter().zip(real_goldens(golden)).enumerate() {
        let d = max_abs_diff(got, &want);
        let c = cosine(got, &want);
        eprintln!(
            "{label} vs {golden} case {i}: max abs diff {d:e}, 1 - cosine {:e}",
            1.0 - c
        );
        worst_d = worst_d.max(d);
        worst_c = worst_c.min(c);
    }
    eprintln!(
        "{label} vs {golden}: worst max abs diff {worst_d:e}, min cosine {worst_c:.9} (1 - cos {:e})",
        1.0 - worst_c
    );
    (worst_d, worst_c)
}

fn hf_dir() -> Option<PathBuf> {
    model_dir("TURBOSPARK_EMBEDDINGGEMMA_DIR", "embeddinggemma-300m")
}

#[test]
#[ignore = "needs the google/embeddinggemma-300m install"]
fn real_tokenizer_matches_sentence_transformers_ids() {
    let Some(dir) = hf_dir() else { return };
    let model = EmbeddingGemma::load(&dir).unwrap();
    assert_eq!(model.max_seq_length(), 2048);
    for (i, case) in real_cases().iter().enumerate() {
        let ids = model
            .tokenize(&case.text, case.prompt_name.as_deref())
            .unwrap();
        assert_eq!(ids, case.ids, "case {i}");
    }
}

/// Upstream f32 checkpoint vs the f32 reference. Bound set from the
/// measured worst case (see README) with ~10x headroom.
const HF_EMBED_TOL: f32 = 5e-6;

#[test]
#[ignore = "needs the google/embeddinggemma-300m install"]
fn real_hf_checkpoint_matches_sentence_transformers() {
    let Some(dir) = hf_dir() else { return };
    let model = EmbeddingGemma::load(&dir).unwrap();
    assert_eq!(model.layout(), Layout::SentenceTransformers);

    // Localize first: embedding, layer 0, last hidden for case 0.
    let case = &real_cases()[0];
    let (shape, trace) = read_npy(FAMILY, "real_trace_case0.npy");
    let plane = shape[1] * shape[2];
    let layers = model.config().num_hidden_layers();
    model
        .backbone()
        .forward_traced(&case.ids, |k, h| {
            let slot = match k {
                0 => 0,
                1 => 1,
                k if k == layers => 2,
                _ => return,
            };
            let d = max_abs_diff(h, &trace[slot * plane..(slot + 1) * plane]);
            let scale = trace[slot * plane..(slot + 1) * plane]
                .iter()
                .fold(0.0f32, |m, v| m.max(v.abs()));
            eprintln!("trace state {k}: max abs diff {d:e} (max |ref| {scale})");
        })
        .unwrap();

    let got = real_embeddings(&model, "hf");
    let (d, c) = real_parity(&got, "real_embed.npy", "hf");
    assert!(d < HF_EMBED_TOL, "max abs diff {d}");
    assert!(c > 0.99999, "min cosine {c}");

    let (_, mrl) = read_npy(FAMILY, "real_embed_mrl256.npy");
    let model = model.with_truncate_dim(Some(256)).unwrap();
    let got = model.embed_ids(&real_cases()[0].ids).unwrap();
    let d = max_abs_diff(&got, &mrl);
    eprintln!("hf matryoshka 256: max abs diff {d:e}");
    assert!(d < HF_EMBED_TOL);
}

/// Floors for one comparison: worst max abs diff and min cosine.
struct Floor {
    max_abs: f32,
    min_cos: f64,
}

fn assert_floor(got: &[Vec<f32>], golden: &str, label: &str, floor: &Floor) {
    let (d, c) = real_parity(got, golden, label);
    assert!(
        d < floor.max_abs,
        "{label} vs {golden}: max abs diff {d} >= {}",
        floor.max_abs
    );
    assert!(
        c > floor.min_cos,
        "{label} vs {golden}: min cosine {c} <= {}",
        floor.min_cos
    );
}

/// Gates per MLX conversion, in decreasing strictness:
/// 1. the reference pipeline on the same (numpy-dequantized) weights,
///    which isolates the port and is held to f32 noise;
/// 2. the checkpoint the conversion was made from (the QAT checkpoints
///    for 8-bit and 4-bit, upstream for F16), which measures the
///    conversion error;
/// 3. upstream, which for 8-bit and 4-bit is QAT drift plus quantization,
///    not a conversion error.
fn mlx_check(env: &str, name: &str, tag: &str, source: Option<(&str, Floor)>, upstream: Floor) {
    let Some(dir) = model_dir(env, name) else {
        return;
    };
    let model = EmbeddingGemma::load(&dir).unwrap();
    assert_eq!(model.layout(), Layout::Mlx);
    let got = real_embeddings(&model, name);
    let same = Floor {
        max_abs: SAME_WEIGHTS_TOL,
        min_cos: 0.99999,
    };
    assert_floor(&got, &format!("real_mlx_{tag}_embed.npy"), name, &same);
    if let Some((golden, floor)) = source {
        assert_floor(&got, golden, name, &floor);
    }
    assert_floor(&got, "real_embed.npy", name, &upstream);
}

/// Same-weights bound: the arithmetic is identical once the weights are
/// fixed, so this is the upstream f32 tolerance.
const SAME_WEIGHTS_TOL: f32 = HF_EMBED_TOL;

// Floors are the measured worst case (README) with margin. The "bf16"
// repo stores F16 converted from upstream, so it is checked against
// upstream only.
#[test]
#[ignore = "needs mlx-community/embeddinggemma-300m-bf16"]
fn real_mlx_bf16_matches_reference() {
    mlx_check(
        "TURBOSPARK_EMBEDDINGGEMMA_MLX_BF16_DIR",
        "mlx-embeddinggemma-300m-bf16",
        "bf16",
        None,
        Floor {
            max_abs: 5e-4,
            min_cos: 0.99999,
        },
    );
}

#[test]
#[ignore = "needs mlx-community/embeddinggemma-300m-8bit"]
fn real_mlx_8bit_matches_reference() {
    mlx_check(
        "TURBOSPARK_EMBEDDINGGEMMA_MLX_8BIT_DIR",
        "mlx-embeddinggemma-300m-8bit",
        "8bit",
        Some((
            "real_qat_q8_embed.npy",
            Floor {
                max_abs: MLX8_VS_QAT.0,
                min_cos: MLX8_VS_QAT.1,
            },
        )),
        Floor {
            max_abs: 5e-2,
            min_cos: 0.97,
        },
    );
}

#[test]
#[ignore = "needs mlx-community/embeddinggemma-300m-4bit"]
fn real_mlx_4bit_matches_reference() {
    mlx_check(
        "TURBOSPARK_EMBEDDINGGEMMA_MLX_4BIT_DIR",
        "mlx-embeddinggemma-300m-4bit",
        "4bit",
        Some((
            "real_qat_q4_embed.npy",
            Floor {
                max_abs: MLX4_VS_QAT.0,
                min_cos: MLX4_VS_QAT.1,
            },
        )),
        Floor {
            max_abs: 1e-1,
            min_cos: 0.90,
        },
    );
}

// MLX vs its QAT source: (max abs, min cosine). Measured worst over the
// 7 cases: 8-bit 3.5e-3 / 0.99965, 4-bit 3.8e-2 / 0.94744. Floors keep
// about 3x on the abs diff and 2.5x (8-bit) / 1.3x (4-bit) on 1 - cos.
const MLX8_VS_QAT: (f32, f64) = (1e-2, 0.999);
const MLX4_VS_QAT: (f32, f64) = (1e-1, 0.93);

fn qat_check(env: &str, name: &str, tag: &str) {
    let Some(dir) = model_dir(env, name) else {
        return;
    };
    let model = EmbeddingGemma::load(&dir).unwrap();
    assert_eq!(model.layout(), Layout::SentenceTransformers);
    for (i, case) in real_cases().iter().enumerate() {
        let ids = model
            .tokenize(&case.text, case.prompt_name.as_deref())
            .unwrap();
        assert_eq!(ids, case.ids, "{name} case {i}");
    }
    let got = real_embeddings(&model, name);
    let floor = Floor {
        max_abs: HF_EMBED_TOL,
        min_cos: 0.99999,
    };
    assert_floor(&got, &format!("real_qat_{tag}_embed.npy"), name, &floor);
    // Reported only: how far QAT training moved the embedding.
    real_parity(&got, "real_embed.npy", name);
}

#[test]
#[ignore = "needs google/embeddinggemma-300m-qat-q8_0-unquantized"]
fn real_qat_q8_matches_sentence_transformers() {
    qat_check(
        "TURBOSPARK_EMBEDDINGGEMMA_QAT_Q8_DIR",
        "embeddinggemma-300m-qat-q8_0-unquantized",
        "q8",
    );
}

#[test]
#[ignore = "needs google/embeddinggemma-300m-qat-q4_0-unquantized"]
fn real_qat_q4_matches_sentence_transformers() {
    qat_check(
        "TURBOSPARK_EMBEDDINGGEMMA_QAT_Q4_DIR",
        "embeddinggemma-300m-qat-q4_0-unquantized",
        "q4",
    );
}

#[test]
#[ignore = "needs the google/embeddinggemma-300m install; timing only"]
fn real_text_api_timing() {
    let Some(dir) = hf_dir() else { return };
    let t = Instant::now();
    let model = EmbeddingGemma::load(&dir).unwrap();
    eprintln!("load: {:.2?}", t.elapsed());
    let cases = real_cases();
    let goldens = real_goldens("real_embed.npy");
    for i in [0usize, 5] {
        let case = &cases[i];
        let _ = model.embed_ids(&case.ids[..4]).unwrap(); // warm threads
        let t = Instant::now();
        let got = match &case.prompt_name {
            Some(p) => model.embed_with_prompt(p, &case.text).unwrap(),
            None => model.embed(&case.text).unwrap(),
        };
        eprintln!(
            "embed case {i} ({} tokens): {:.2?}, cosine {:.9}",
            case.ids.len(),
            t.elapsed(),
            cosine(&got, &goldens[i])
        );
    }
}
