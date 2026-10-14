//! Parity and refusal tests. Fixtures come from `tools/gen_gte_fixtures.py`;
//! see the family README for regeneration and the measured numbers behind
//! each bound.

use std::path::PathBuf;
use std::time::Instant;

use serde_json::Value;

use super::config::truthy;
use super::encoder::{inv_freq, rope_tables};
use super::*;
use crate::testutil::{max_abs_diff, model_dir, read_json, read_npy, testdata};

const FAMILY: &str = "gte";

// Tiny random model vs the remote GteModel in f32. Post-LN keeps every
// state O(1), so these are f32 reduction-order noise with headroom (see
// README for the measured worst case); a wrong rotary layout, split
// order, or gate half moves states by O(0.1..1).
const TINY_HIDDEN_TOL: f32 = 1e-5;
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

fn ids_of(v: &Value) -> Vec<u32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as u32)
        .collect()
}

fn tiny_cases() -> Vec<Vec<u32>> {
    read_json(FAMILY, "tiny_cases.json")["cases"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| ids_of(&c["ids"]))
        .collect()
}

fn tiny() -> Gte {
    Gte::load(testdata(FAMILY, "tiny")).unwrap()
}

#[test]
fn tiny_matches_reference() {
    let model = tiny();
    let c = model.config().clone();
    assert_eq!(model.dimension(), 64);
    // No sentence_bert_config.json and no tokenizer_config.json.
    assert_eq!(model.max_seq_length(), c.max_position_embeddings);
    let (mut worst_h, mut worst_e) = (0.0f32, 0.0f32);
    for (i, ids) in tiny_cases().iter().enumerate() {
        let (shape, golden) = read_npy(FAMILY, &format!("tiny_hidden_{i}.npy"));
        assert_eq!(
            shape,
            vec![c.num_hidden_layers + 1, ids.len(), c.hidden_size]
        );
        let plane = ids.len() * c.hidden_size;
        let mut seen = 0;
        let last = model
            .encoder()
            .forward_traced(ids, |k, h| {
                let d = max_abs_diff(h, &golden[k * plane..(k + 1) * plane]);
                assert!(d < TINY_HIDDEN_TOL, "case {i} state {k}: max abs diff {d}");
                worst_h = worst_h.max(d);
                seen += 1;
            })
            .unwrap();
        assert_eq!(seen, c.num_hidden_layers + 1);
        let emb = model.head(&last, ids.len()).unwrap();
        let (_, want) = read_npy(FAMILY, &format!("tiny_embed_{i}.npy"));
        let d = max_abs_diff(&emb, &want);
        assert!(d < TINY_EMBED_TOL, "case {i} embedding: max abs diff {d}");
        worst_e = worst_e.max(d);
        assert_eq!(model.embed_ids(ids).unwrap(), emb);
    }
    eprintln!("gte tiny: worst hidden diff {worst_h:e}, worst embedding diff {worst_e:e}");
}

#[test]
fn tiny_reference_cross_checks_hold() {
    // The generator measured the unpadded (unpad_inputs) and eager paths
    // against the default one; both select layout or kernel, not math.
    let cross = &read_json(FAMILY, "tiny_cases.json")["cross_checks"];
    let cross = cross.as_object().unwrap();
    assert!(cross.contains_key("unpad_vs_padded"));
    for (k, v) in cross {
        let d = v.as_f64().unwrap();
        assert!(d < 1e-5, "{k}: {d}");
    }
}

/// Distance in f32 units in the last place (same-sign inputs).
fn ulps(a: f32, b: f32) -> u32 {
    (a.to_bits() as i64 - b.to_bits() as i64).unsigned_abs() as u32
}

#[test]
fn inv_freq_matches_reference_buffers() {
    let c = tiny().config().clone();
    let (_, want) = read_npy(FAMILY, "tiny_rotary_inv_freq.npy");
    let got = inv_freq(c.rope_theta, c.head_dim());
    assert_eq!(got.len(), want.len());
    for (g, w) in got.iter().zip(&want) {
        assert!(ulps(*g, *w) <= 1, "{g} vs {w}");
    }
    // The pinned config: theta 160000, head_dim 64.
    let (_, want) = read_npy(FAMILY, "real_rotary_inv_freq.npy");
    let got = inv_freq(160000.0, 64);
    let worst = got.iter().zip(&want).map(|(g, w)| ulps(*g, *w)).max();
    eprintln!("gte real inv_freq worst ulps vs torch: {worst:?}");
    assert!(worst <= Some(1));
}

#[test]
fn rope_tables_match_reference_rows() {
    let rows: Vec<usize> = read_json(FAMILY, "real_cases.json")["rotary_rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let inv = inv_freq(160000.0, 64);
    let seq = rows.iter().max().unwrap() + 1;
    let (cos, sin) = rope_tables(seq, &inv);
    let (_, want_cos) = read_npy(FAMILY, "real_rotary_cos_rows.npy");
    let (_, want_sin) = read_npy(FAMILY, "real_rotary_sin_rows.npy");
    let half = inv.len();
    let mut worst = 0.0f32;
    for (r, &t) in rows.iter().enumerate() {
        worst = worst
            .max(max_abs_diff(
                &cos[t * half..(t + 1) * half],
                &want_cos[r * half..(r + 1) * half],
            ))
            .max(max_abs_diff(
                &sin[t * half..(t + 1) * half],
                &want_sin[r * half..(r + 1) * half],
            ));
    }
    eprintln!("gte rope rows worst abs diff {worst:e}");
    // One f32 ulp of a value in [-1, 1] is at most 6e-8.
    assert!(worst <= 1.2e-7, "{worst}");
}

#[test]
fn rope_layout_is_rotate_half() {
    // Reference: q * cos + rotate_half(q) * sin with rotate_half(x) =
    // cat(-x2, x1) and cos/sin duplicated over both halves (modeling:158-179).
    let (dim, seq) = (8usize, 3usize);
    let inv = inv_freq(10.0, dim);
    let (cos, sin) = rope_tables(seq, &inv);
    let x: Vec<f32> = (0..seq * dim).map(|i| (i as f32 * 0.37).sin()).collect();
    let mut got = x.clone();
    turbospark_audio::ops::rope_neox(&mut got, 1, seq, dim, &cos, &sin);
    let half = dim / 2;
    for t in 0..seq {
        for d in 0..dim {
            let (c, s) = (cos[t * half + d % half], sin[t * half + d % half]);
            let rot = if d < half {
                -x[t * dim + d + half]
            } else {
                x[t * dim + d - half]
            };
            let want = x[t * dim + d] * c + rot * s;
            assert!((got[t * dim + d] - want).abs() < 1e-6, "t {t} d {d}");
        }
    }
}

#[test]
fn python_truthiness() {
    assert!(truthy(Some(&serde_json::json!("true"))));
    assert!(truthy(Some(&serde_json::json!("false"))));
    assert!(!truthy(Some(&serde_json::json!(""))));
    assert!(!truthy(Some(&Value::Null)));
    assert!(!truthy(None));
    assert!(truthy(Some(&serde_json::json!(1))));
    assert!(!truthy(Some(&serde_json::json!(0.0))));
}

fn tiny_config() -> Value {
    serde_json::from_slice(&std::fs::read(testdata(FAMILY, "tiny/config.json")).unwrap()).unwrap()
}

#[test]
fn config_accepts_pinned_flag_spellings() {
    let mut v = tiny_config();
    // As in the pinned checkpoint: strings, layout flags only.
    v["unpad_inputs"] = serde_json::json!("true");
    v["use_memory_efficient_attention"] = serde_json::json!("true");
    let c = GteConfig::from_json(&v).unwrap();
    assert_eq!(c.head_dim(), 16);
    assert_eq!(c.attn_scale(), 0.25);
    assert_eq!(c.matryoshka_dimensions, vec![16]);
    // Absent fields take the reference defaults.
    for key in [
        "pack_qkv",
        "rope_scaling",
        "layer_norm_type",
        "position_embedding_type",
    ] {
        let mut v = tiny_config();
        v.as_object_mut().unwrap().remove(key);
        assert!(GteConfig::from_json(&v).is_ok(), "{key} absent");
    }
}

#[test]
fn config_refuses_unsupported_values() {
    let cases: Vec<(&str, Value)> = vec![
        ("model_type", serde_json::json!("new")),
        ("position_embedding_type", serde_json::json!("absolute")),
        (
            "rope_scaling",
            serde_json::json!({"type": "ntk", "factor": 2.0}),
        ),
        ("hidden_act", serde_json::json!("silu")),
        ("layer_norm_type", serde_json::json!("rms_norm")),
        ("pack_qkv", Value::Bool(false)),
        ("pack_qkv", Value::Null),
        ("logn_attention_scale", Value::Bool(true)),
        ("logn_attention_scale", serde_json::json!("false")),
        ("logn_attention_clip1", Value::Bool(true)),
        ("type_vocab_size", serde_json::json!(0)),
        ("num_attention_heads", serde_json::json!(5)),
        ("num_attention_heads", serde_json::json!(64)), // head_dim 1, odd
        ("hidden_size", Value::Null),
        ("rope_theta", serde_json::json!(0)),
        ("layer_norm_eps", serde_json::json!(-1.0)),
        ("matryoshka_dimensions", serde_json::json!([0])),
        ("matryoshka_dimensions", serde_json::json!([65])),
    ];
    for (key, value) in cases {
        let mut v = tiny_config();
        v[key] = value.clone();
        assert!(
            GteConfig::from_json(&v).is_err(),
            "{key} = {value} was accepted"
        );
    }
}

/// Copies the tiny fixture to a scratch dir for a mutated-load test.
fn scratch_copy(tag: &str) -> PathBuf {
    let src = testdata(FAMILY, "tiny");
    let dst = std::env::temp_dir().join(format!("tse-gte-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dst);
    for entry in ["", "1_Pooling"] {
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
fn load_refuses_unimplemented_pipelines() {
    let dir = scratch_copy("pool");
    let path = dir.join("1_Pooling/config.json");
    let mut v: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    v["include_prompt"] = Value::Bool(false);
    std::fs::write(&path, serde_json::to_vec(&v).unwrap()).unwrap();
    let err = Gte::load(&dir).unwrap_err();
    assert!(matches!(err, EmbeddingError::Unsupported { .. }), "{err}");
    std::fs::remove_dir_all(&dir).unwrap();

    let dir = scratch_copy("dense");
    std::fs::create_dir_all(dir.join("2_Dense")).unwrap();
    std::fs::write(
        dir.join("2_Dense/config.json"),
        br#"{"in_features": 64, "out_features": 64, "bias": false}"#,
    )
    .unwrap();
    let mut modules: Value =
        serde_json::from_slice(&std::fs::read(dir.join("modules.json")).unwrap()).unwrap();
    modules.as_array_mut().unwrap().insert(
        2,
        serde_json::json!({"idx": 2, "name": "2", "path": "2_Dense",
            "type": "sentence_transformers.models.Dense"}),
    );
    modules[3]["idx"] = serde_json::json!(3);
    std::fs::write(
        dir.join("modules.json"),
        serde_json::to_vec(&modules).unwrap(),
    )
    .unwrap();
    let err = Gte::load(&dir).unwrap_err();
    assert!(matches!(err, EmbeddingError::Unsupported { .. }), "{err}");
    std::fs::remove_dir_all(&dir).unwrap();

    // A config that disagrees with the tensors fails at load.
    let dir = scratch_copy("shape");
    let mut v = tiny_config();
    v["intermediate_size"] = serde_json::json!(128);
    std::fs::write(dir.join("config.json"), serde_json::to_vec(&v).unwrap()).unwrap();
    assert!(matches!(
        Gte::load(&dir).unwrap_err(),
        EmbeddingError::Tensor { .. }
    ));
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn max_seq_length_follows_sentence_transformers() {
    let dir = scratch_copy("msl");
    std::fs::write(
        dir.join("tokenizer_config.json"),
        br#"{"model_max_length": 32}"#,
    )
    .unwrap();
    assert_eq!(Gte::load(&dir).unwrap().max_seq_length(), 32);
    std::fs::write(
        dir.join("tokenizer_config.json"),
        br#"{"model_max_length": 1e30}"#,
    )
    .unwrap();
    assert_eq!(Gte::load(&dir).unwrap().max_seq_length(), 64);
    std::fs::write(
        dir.join("sentence_bert_config.json"),
        br#"{"max_seq_length": 16}"#,
    )
    .unwrap();
    assert_eq!(Gte::load(&dir).unwrap().max_seq_length(), 16);
    std::fs::write(
        dir.join("sentence_bert_config.json"),
        br#"{"max_seq_length": 65}"#,
    )
    .unwrap();
    assert!(Gte::load(&dir).is_err());
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn text_api_without_tokenizer_is_an_error_and_ids_are_bounded() {
    let model = tiny();
    assert!(matches!(
        model.embed("hi"),
        Err(EmbeddingError::Tokenizer { .. })
    ));
    assert!(model.embed_ids(&[]).is_err());
    assert!(model.embed_ids(&[0, 96, 2]).is_err()); // out of vocab
    assert!(model.embed_ids(&[3; 65]).is_err()); // > max_position_embeddings
    assert_eq!(model.prompt_names().collect::<Vec<_>>(), vec!["query"]);
    assert_eq!(model.prompt("query"), Some("query: "));
}

#[test]
fn truncate_dim_renormalizes_and_is_validated() {
    let model = tiny();
    let ids = &tiny_cases()[3];
    let full = model.embed_ids(ids).unwrap();
    let model = model.with_truncate_dim(Some(16)).unwrap();
    assert_eq!(model.dimension(), 16);
    assert_eq!(
        model.embed_ids(ids).unwrap(),
        truncate_and_normalize(&full, 16)
    );
    assert!(model.with_truncate_dim(Some(65)).is_err());
}

// ------------------------------------------------------------ real checkpoint

struct RealCase {
    prompt_name: Option<String>,
    text: String,
    ids: Vec<u32>,
    tokenize_only: bool,
}

fn real_cases() -> Vec<RealCase> {
    read_json(FAMILY, "real_cases.json")["cases"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| RealCase {
            prompt_name: c["prompt_name"].as_str().map(str::to_owned),
            text: c["text"].as_str().unwrap().to_owned(),
            ids: ids_of(&c["ids"]),
            tokenize_only: c["tokenize_only"].as_bool().unwrap_or(false),
        })
        .collect()
}

fn real_dir() -> Option<PathBuf> {
    model_dir("TURBOSPARK_GTE_DIR", "snowflake-arctic-embed-m-v2.0")
}

#[test]
#[ignore = "needs the Snowflake/snowflake-arctic-embed-m-v2.0 install"]
fn real_tokenizer_matches_sentence_transformers_ids() {
    let Some(dir) = real_dir() else { return };
    let model = Gte::load(&dir).unwrap();
    assert_eq!(model.max_seq_length(), 8192);
    for (i, case) in real_cases().iter().enumerate() {
        let ids = model
            .tokenize(&case.text, case.prompt_name.as_deref())
            .unwrap();
        assert_eq!(ids, case.ids, "case {i}");
    }
}

/// Real checkpoint vs the f32 reference. Measured worst over the cases
/// (README): max abs diff under 6e-7 and 1 - cos under 1e-11, so the
/// bounds keep about 5x on the abs diff and 100x on 1 - cos.
const REAL_EMBED_TOL: f32 = 3e-6;
const REAL_MIN_COS: f64 = 0.999_999_999;

#[test]
#[ignore = "needs the Snowflake/snowflake-arctic-embed-m-v2.0 install"]
fn real_checkpoint_matches_sentence_transformers() {
    let Some(dir) = real_dir() else { return };
    let model = Gte::load(&dir).unwrap();
    let cases = real_cases();

    // Localize first: embedding LN, layer 0, last hidden for case 0.
    let (shape, trace) = read_npy(FAMILY, "real_trace_case0.npy");
    let plane = shape[1] * shape[2];
    let layers = model.config().num_hidden_layers;
    model
        .encoder()
        .forward_traced(&cases[0].ids, |k, h| {
            let slot = match k {
                0 => 0,
                1 => 1,
                k if k == layers => 2,
                _ => return,
            };
            let d = max_abs_diff(h, &trace[slot * plane..(slot + 1) * plane]);
            eprintln!("trace state {k}: max abs diff {d:e}");
        })
        .unwrap();

    let (shape, golden) = read_npy(FAMILY, "real_embed.npy");
    let mut g = golden.chunks_exact(shape[1]);
    let (mut worst_d, mut worst_c) = (0.0f32, 1.0f64);
    for (i, case) in cases.iter().enumerate().filter(|(_, c)| !c.tokenize_only) {
        let t = Instant::now();
        let got = model.embed_ids(&case.ids).unwrap();
        let want = g.next().unwrap();
        let (d, c) = (max_abs_diff(&got, want), cosine(&got, want));
        eprintln!(
            "case {i}: {} tokens in {:.2?}, max abs diff {d:e}, 1 - cos {:e}",
            case.ids.len(),
            t.elapsed(),
            1.0 - c
        );
        worst_d = worst_d.max(d);
        worst_c = worst_c.min(c);
    }
    eprintln!("gte real: worst max abs diff {worst_d:e}, min cosine {worst_c:.9}");
    assert!(worst_d < REAL_EMBED_TOL, "max abs diff {worst_d}");
    assert!(worst_c > REAL_MIN_COS, "min cosine {worst_c}");

    let (_, mrl) = read_npy(FAMILY, "real_embed_mrl256.npy");
    let model = model.with_truncate_dim(Some(256)).unwrap();
    let got = model.embed_ids(&cases[0].ids).unwrap();
    let d = max_abs_diff(&got, &mrl);
    eprintln!("matryoshka 256: max abs diff {d:e}");
    assert!(d < REAL_EMBED_TOL);
}

#[test]
#[ignore = "needs the Snowflake/snowflake-arctic-embed-m-v2.0 install; timing only"]
fn real_text_api_timing() {
    let Some(dir) = real_dir() else { return };
    let t = Instant::now();
    let model = Gte::load(&dir).unwrap();
    eprintln!("load: {:.2?}", t.elapsed());
    let cases = real_cases();
    let _ = model.embed_ids(&cases[0].ids).unwrap(); // warm threads
    for i in [0usize, 6] {
        let case = &cases[i];
        let t = Instant::now();
        let got = match &case.prompt_name {
            Some(p) => model.embed_with_prompt(p, &case.text).unwrap(),
            None => model.embed(&case.text).unwrap(),
        };
        eprintln!(
            "embed case {i} ({} tokens): {:.2?}",
            case.ids.len(),
            t.elapsed()
        );
        assert_eq!(got.len(), 768);
    }
}
