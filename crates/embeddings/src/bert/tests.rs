//! Parity and refusal tests. Fixtures come from
//! `tools/gen_bert_fixtures.py`; see the family README for the
//! regeneration commands and the measured numbers behind each bound.

use std::path::PathBuf;
use std::time::Instant;

use serde_json::Value;

use super::*;
use crate::pins;
use crate::testutil::{max_abs_diff, model_dir, read_json, read_npy, testdata};

const FAMILY: &str = "bert";

// Tiny random models vs HF f32 (sdpa). Hidden values are O(1) after each
// LayerNorm, so these bounds are f32 reduction-order noise plus the
// shared erf approximation (abs error < 1.5e-7), with about 10x headroom
// over the measured worst (README). A wrong position rule, LN order, or
// mask moves states by O(0.1..1).
const TINY_HIDDEN_TOL: f32 = 2e-5;
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

/// `(ids, mask)` for the `bert` or `xlmr` tiny cases.
fn tiny_cases(kind: &str) -> Vec<(Vec<u32>, Option<Vec<u32>>)> {
    read_json(FAMILY, "tiny_cases.json")["cases"][kind]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| {
            let mask = (!c["mask"].is_null()).then(|| ids_of(&c["mask"]));
            (ids_of(&c["ids"]), mask)
        })
        .collect()
}

/// CLS pooling, as every tiny checkpoint's `1_Pooling` sets it.
fn cls(dim: usize) -> PoolingConfig {
    let mut p = PoolingConfig::mean(dim);
    p.mean = false;
    p.cls = true;
    p
}

/// Every hidden state and the final embedding of each `kind` case vs the
/// `variant` goldens. Returns the worst (hidden, embedding) diffs.
fn tiny_parity(model: &BertEmbedder, variant: &str, kind: &str) -> (f32, f32) {
    let c = model.config();
    let (layers, h) = (c.num_hidden_layers, c.hidden_size);
    let (mut worst_h, mut worst_e) = (0.0f32, 0.0f32);
    for (i, (ids, mask)) in tiny_cases(kind).iter().enumerate() {
        let (shape, golden) = read_npy(FAMILY, &format!("tiny_{variant}_hidden_{i}.npy"));
        assert_eq!(shape, vec![layers + 1, ids.len(), h]);
        let plane = ids.len() * h;
        let mut seen = 0;
        let last = model
            .encoder()
            .forward_traced(ids, mask.as_deref(), |k, x| {
                let d = max_abs_diff(x, &golden[k * plane..(k + 1) * plane]);
                assert!(
                    d < TINY_HIDDEN_TOL,
                    "{variant} case {i} state {k}: max abs diff {d}"
                );
                worst_h = worst_h.max(d);
                seen += 1;
            })
            .unwrap();
        assert_eq!(seen, layers + 1);
        let ones = vec![1u32; ids.len()];
        let emb = model.head(&last, mask.as_deref().unwrap_or(&ones)).unwrap();
        let (_, want) = read_npy(FAMILY, &format!("tiny_{variant}_embed_{i}.npy"));
        let d = max_abs_diff(&emb, &want);
        assert!(
            d < TINY_EMBED_TOL,
            "{variant} case {i} embedding: max abs diff {d}"
        );
        worst_e = worst_e.max(d);
        if mask.is_none() {
            assert_eq!(model.embed_ids(ids).unwrap(), emb);
        }
    }
    eprintln!("tiny {variant}: worst hidden diff {worst_h:e}, worst embedding diff {worst_e:e}");
    (worst_h, worst_e)
}

#[test]
fn tiny_bert_matches_reference() {
    let model = BertEmbedder::load(testdata(FAMILY, "tiny_bert")).unwrap();
    assert_eq!(model.config().arch, Arch::Bert);
    assert_eq!(model.layout(), Layout::SentenceTransformers);
    assert_eq!(
        model.pooling_source(),
        &PoolingSource::Module("1_Pooling".into())
    );
    assert_eq!(model.pooling(), &cls(64));
    assert_eq!(model.dimension(), 64);
    // sentence_bert_config.json wins over the position table.
    assert_eq!(model.max_seq_length(), 16);
    assert!(model.encoder().word_table_is_mapped());
    tiny_parity(&model, "bert", "bert");
}

#[test]
fn tiny_xlmr_matches_reference() {
    let model = BertEmbedder::load(testdata(FAMILY, "tiny_xlmr")).unwrap();
    assert_eq!(model.config().arch, Arch::XlmRoberta);
    assert_eq!(model.layout(), Layout::SentenceTransformers);
    // No sentence_*_config.json: min(26 positions, model_max_length 20).
    assert_eq!(model.max_seq_length(), 20);
    assert_eq!(model.config().max_tokens(), 24);
    tiny_parity(&model, "xlmr", "xlmr");
}

/// The tiny MLX dirs have no `1_Pooling/` and do not fingerprint as a
/// known conversion, so plain `load` must refuse them.
fn tiny_mlx(name: &str) -> BertEmbedder {
    let dir = testdata(FAMILY, name);
    let err = BertEmbedder::load(&dir).unwrap_err();
    assert!(matches!(err, EmbeddingError::Unsupported { .. }), "{err}");
    let model = BertEmbedder::load_with_pooling(&dir, cls(64)).unwrap();
    assert_eq!(model.layout(), Layout::Mlx);
    assert_eq!(model.pooling_source(), &PoolingSource::Caller);
    model
}

#[test]
fn tiny_mlx_q4_matches_dequantized_reference() {
    let model = tiny_mlx("tiny_xlmr_mlx_q4");
    assert_eq!(model.config().quantization.map(|q| q.bits), Some(4));
    assert!(!model.encoder().word_table_is_mapped());
    tiny_parity(&model, "q4", "xlmr");
}

#[test]
fn tiny_mlx_q6_matches_dequantized_reference() {
    // 6-bit codes cross 32-bit word boundaries (64 values = 12 words).
    let model = tiny_mlx("tiny_xlmr_mlx_q6");
    assert_eq!(model.config().quantization.map(|q| q.bits), Some(6));
    tiny_parity(&model, "q6", "xlmr");
}

#[test]
fn tiny_mlx_bf16_matches_reference() {
    let model = tiny_mlx("tiny_xlmr_mlx_bf16");
    assert_eq!(model.config().quantization, None);
    assert!(model.encoder().word_table_is_mapped());
    tiny_parity(&model, "bf16", "xlmr");
}

#[test]
fn tiny_fixture_exercises_padding_and_the_last_position() {
    // Guards the fixture: without these cases the XLM-R position rule,
    // the key mask, and the top position row go untested.
    let model = BertEmbedder::load(testdata(FAMILY, "tiny_xlmr")).unwrap();
    let c = model.config();
    let cases = tiny_cases("xlmr");
    let pad = c.pad_token_id;
    assert!(cases
        .iter()
        .any(|(ids, m)| m.is_none() && ids[1..ids.len() - 1].contains(&pad)));
    assert!(cases
        .iter()
        .any(|(_, m)| m.as_ref().is_some_and(|m| m.contains(&0))));
    let top = cases
        .iter()
        .map(|(ids, _)| *c.position_ids(ids).iter().max().unwrap())
        .max()
        .unwrap();
    assert_eq!(top as usize, c.max_position_embeddings - 1);
    let bert = tiny_cases("bert");
    assert!(bert.iter().any(|(ids, m)| m.is_none() && ids.contains(&0)));
}

#[test]
fn xlmr_positions_skip_padding() {
    assert_eq!(xlmr_position_ids(&[0, 5, 9, 2], 1), vec![2, 3, 4, 5]);
    // A pad keeps position padding_idx and does not advance the count.
    assert_eq!(
        xlmr_position_ids(&[0, 1, 15, 1, 1, 30, 2], 1),
        vec![2, 1, 3, 1, 1, 4, 5]
    );
    assert_eq!(xlmr_position_ids(&[1, 1], 1), vec![1, 1]);
    assert_eq!(xlmr_position_ids(&[7, 0, 7], 0), vec![1, 0, 2]);
}

fn config_json(name: &str) -> Value {
    serde_json::from_slice(
        &std::fs::read(testdata(FAMILY, &format!("{name}/config.json"))).unwrap(),
    )
    .unwrap()
}

#[test]
fn bert_positions_count_from_zero_and_ignore_pad() {
    let c = BertConfig::from_json(&config_json("tiny_bert")).unwrap();
    assert_eq!(c.position_ids(&[2, 0, 15, 0, 3]), vec![0, 1, 2, 3, 4]);
    assert_eq!(c.max_tokens(), 24);
    let x = BertConfig::from_json(&config_json("tiny_xlmr")).unwrap();
    assert_eq!(x.position_ids(&[0, 1, 15, 2]), vec![2, 1, 3, 4]);
    let mut v = config_json("tiny_xlmr");
    v["max_position_embeddings"] = serde_json::json!(8194);
    assert_eq!(BertConfig::from_json(&v).unwrap().max_tokens(), 8192);
}

#[test]
fn config_refuses_unsupported_values() {
    let cases: Vec<(&str, Value)> = vec![
        ("model_type", serde_json::json!("roberta")),
        ("model_type", serde_json::json!("distilbert")),
        ("position_embedding_type", serde_json::json!("relative_key")),
        (
            "position_embedding_type",
            serde_json::json!("relative_key_query"),
        ),
        ("is_decoder", Value::Bool(true)),
        ("add_cross_attention", Value::Bool(true)),
        ("hidden_act", serde_json::json!("gelu_new")),
        ("hidden_act", serde_json::json!("relu")),
        ("num_attention_heads", serde_json::json!(5)),
        ("layer_norm_eps", serde_json::json!(0.0)),
        ("pad_token_id", serde_json::json!(96)),
        ("type_vocab_size", serde_json::json!(0)),
        ("matryoshka_dimensions", serde_json::json!([128, 65])),
        (
            "quantization",
            serde_json::json!({"group_size": 64, "bits": 5}),
        ),
        (
            "quantization",
            serde_json::json!({"group_size": 32, "bits": 4}),
        ),
        (
            "quantization",
            serde_json::json!({"group_size": 64, "bits": 4, "mode": "mxfp4"}),
        ),
        (
            "quantization",
            serde_json::json!({"group_size": 64, "bits": 4, "encoder.layer.0.output.dense": {"bits": 8}}),
        ),
    ];
    for base in ["tiny_bert", "tiny_xlmr"] {
        for (key, value) in &cases {
            let mut v = config_json(base);
            v[*key] = value.clone();
            assert!(
                BertConfig::from_json(&v).is_err(),
                "{base}: {key} = {value} was accepted"
            );
        }
        for key in ["hidden_act", "layer_norm_eps", "pad_token_id", "model_type"] {
            let mut v = config_json(base);
            v.as_object_mut().unwrap().remove(key);
            assert!(BertConfig::from_json(&v).is_err(), "{base}: missing {key}");
        }
        let mut v = config_json(base);
        v.as_object_mut().unwrap().remove("position_embedding_type");
        assert!(BertConfig::from_json(&v).is_ok());
    }
    // XLM-R with a position table too small for one token after the
    // padding offset.
    let mut v = config_json("tiny_xlmr");
    v["max_position_embeddings"] = serde_json::json!(2);
    assert!(BertConfig::from_json(&v).is_err());
}

/// The l-v2.0 config as shipped, built from the tiny XLM-R config.
fn arctic_l_v2_config() -> BertConfig {
    let mut v = config_json("tiny_xlmr");
    for (k, n) in [
        ("vocab_size", 250002),
        ("hidden_size", 1024),
        ("num_hidden_layers", 24),
        ("num_attention_heads", 16),
        ("intermediate_size", 4096),
        ("max_position_embeddings", 8194),
    ] {
        v[k] = serde_json::json!(n);
    }
    BertConfig::from_json(&v).unwrap()
}

fn prompts(pairs: &[(&str, &str)]) -> PromptConfig {
    PromptConfig {
        prompts: pairs
            .iter()
            .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
            .collect(),
        ..PromptConfig::default()
    }
}

#[test]
fn known_conversion_requires_an_exact_fingerprint() {
    let c = arctic_l_v2_config();
    let (repo, pooling) = known_conversion(&c, &prompts(&[("query", "query: ")])).unwrap();
    assert_eq!(repo, "Snowflake/snowflake-arctic-embed-l-v2.0");
    assert_eq!(pooling, cls(1024));
    // bge-m3 shape, no prompts: refused.
    assert!(known_conversion(&c, &prompts(&[])).is_none());
    assert!(known_conversion(&c, &prompts(&[("query", "query: "), ("x", "y")])).is_none());
    assert!(known_conversion(&c, &prompts(&[("query", "Query: ")])).is_none());
    let mut other = c.clone();
    other.max_position_embeddings = 514;
    assert!(known_conversion(&other, &prompts(&[("query", "query: ")])).is_none());
}

#[test]
fn known_upstreams_are_pinned_with_conversions() {
    for k in KNOWN_UPSTREAMS {
        let up = pins::pin(k.repo).expect("pinned");
        assert_eq!(up.family, pins::Family::Bert);
        assert!(up.converted_from.is_none());
        assert!(pins::pins_for(pins::Family::Bert).any(|p| p.converted_from == Some(k.repo)));
    }
}

#[test]
fn assemble_keeps_special_tokens_when_truncating() {
    assert_eq!(
        assemble(&[101], &[10, 11, 12], &[102], 10),
        vec![101, 10, 11, 12, 102]
    );
    assert_eq!(
        assemble(&[101], &[10, 11, 12], &[102], 4),
        vec![101, 10, 11, 102]
    );
    assert_eq!(assemble(&[0], &[], &[2], 4), vec![0, 2]);
}

#[test]
fn py_strip_matches_python_whitespace() {
    assert_eq!(py_strip("  \t a b \n"), "a b");
    assert_eq!(py_strip("\u{1c}x\u{1f}"), "x");
    assert_eq!(py_strip("\u{3000}x\u{a0}"), "x");
    assert_eq!(py_strip("\u{200b}x"), "\u{200b}x");
}

/// Copies a tiny fixture dir (top level and `1_Pooling`) to scratch.
fn scratch_copy(name: &str, tag: &str) -> PathBuf {
    let src = testdata(FAMILY, name);
    let dst = std::env::temp_dir().join(format!("tse-bert-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dst);
    for sub in ["", "1_Pooling"] {
        if !src.join(sub).is_dir() {
            continue;
        }
        std::fs::create_dir_all(dst.join(sub)).unwrap();
        for f in std::fs::read_dir(src.join(sub)).unwrap() {
            let f = f.unwrap();
            if f.file_type().unwrap().is_file() {
                std::fs::copy(f.path(), dst.join(sub).join(f.file_name())).unwrap();
            }
        }
    }
    dst
}

fn edit_json(path: &std::path::Path, f: impl FnOnce(&mut Value)) {
    let mut v: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    f(&mut v);
    std::fs::write(path, serde_json::to_vec(&v).unwrap()).unwrap();
}

#[test]
fn max_seq_length_follows_sentence_transformers() {
    let dir = scratch_copy("tiny_xlmr", "msl");
    // Tokenizer limit below the position table wins.
    assert_eq!(BertEmbedder::load(&dir).unwrap().max_seq_length(), 20);
    // Absent model_max_length is transformers' huge default: table size.
    std::fs::write(dir.join("tokenizer_config.json"), b"{}").unwrap();
    assert_eq!(BertEmbedder::load(&dir).unwrap().max_seq_length(), 26);
    std::fs::remove_file(dir.join("tokenizer_config.json")).unwrap();
    assert_eq!(BertEmbedder::load(&dir).unwrap().max_seq_length(), 26);
    // A later file in the reference's list is read when the first is absent.
    std::fs::write(
        dir.join("sentence_xlm-roberta_config.json"),
        br#"{"max_seq_length": 12, "do_lower_case": false}"#,
    )
    .unwrap();
    assert_eq!(BertEmbedder::load(&dir).unwrap().max_seq_length(), 12);
    std::fs::write(
        dir.join("sentence_xlm-roberta_config.json"),
        br#"{"max_seq_length": 12, "do_lower_case": true}"#,
    )
    .unwrap();
    assert!(BertEmbedder::load(&dir).is_err());
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn load_refuses_unsupported_pipelines() {
    let dir = scratch_copy("tiny_bert", "pipe");
    edit_json(&dir.join("1_Pooling/config.json"), |v| {
        v["include_prompt"] = Value::Bool(false)
    });
    let err = BertEmbedder::load(&dir).unwrap_err();
    assert!(matches!(err, EmbeddingError::Unsupported { .. }), "{err}");
    edit_json(&dir.join("1_Pooling/config.json"), |v| {
        v["include_prompt"] = Value::Bool(true);
        v["word_embedding_dimension"] = serde_json::json!(32);
    });
    assert!(BertEmbedder::load(&dir).is_err());
    edit_json(&dir.join("1_Pooling/config.json"), |v| {
        v["word_embedding_dimension"] = serde_json::json!(64)
    });
    edit_json(&dir.join("modules.json"), |v| {
        v.as_array_mut()
            .unwrap()
            .push(serde_json::json!({"idx": 3, "name": "3",
            "path": "3_Dense", "type": "sentence_transformers.models.Dense"}))
    });
    let err = BertEmbedder::load(&dir).unwrap_err();
    assert!(matches!(err, EmbeddingError::Unsupported { .. }), "{err}");
    std::fs::remove_file(dir.join("modules.json")).unwrap();
    assert!(BertEmbedder::load(&dir).is_err());
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn ids_are_bounded_and_text_needs_a_tokenizer() {
    let model = BertEmbedder::load(testdata(FAMILY, "tiny_xlmr")).unwrap();
    assert!(matches!(
        model.embed("hi"),
        Err(EmbeddingError::Tokenizer { .. })
    ));
    assert!(model.embed_ids(&[]).is_err());
    assert!(model.embed_ids(&[0, 96, 2]).is_err()); // out of vocab
    let mut ids = vec![5u32; 24];
    assert!(model.embed_ids(&ids).is_ok()); // positions 2..=25
    ids.push(5);
    assert!(model.embed_ids(&ids).is_err()); // position 26
                                             // Pads do not consume positions under the XLM-R rule.
    ids[3] = 1;
    assert!(model.embed_ids(&ids).is_ok());
    let enc = model.encoder();
    assert!(enc
        .forward_traced(&[0, 5, 2], Some(&[0, 0, 0]), |_, _| {})
        .is_err());
    assert!(enc
        .forward_traced(&[0, 5, 2], Some(&[1, 1]), |_, _| {})
        .is_err());
    let names: Vec<&str> = model.prompt_names().collect();
    assert_eq!(names, vec!["document", "query"]);
    assert_eq!(model.prompt("query"), Some("query: "));
    assert_eq!(model.prompt("document"), Some(""));
}

#[test]
fn truncate_dim_renormalizes_and_is_validated() {
    let model = BertEmbedder::load(testdata(FAMILY, "tiny_bert")).unwrap();
    let ids = &tiny_cases("bert")[1].0;
    let full = model.embed_ids(ids).unwrap();
    let model = model.with_truncate_dim(Some(16)).unwrap();
    assert_eq!(model.dimension(), 16);
    assert_eq!(
        model.embed_ids(ids).unwrap(),
        truncate_and_normalize(&full, 16)
    );
    assert!(model.with_truncate_dim(Some(65)).is_err());
}

// ------------------------------------------------------------ real checkpoints

struct RealCase {
    prompt_name: Option<String>,
    text: String,
}

fn real_doc() -> Value {
    read_json(FAMILY, "real_cases.json")
}

fn real_cases() -> Vec<RealCase> {
    real_doc()["cases"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| RealCase {
            prompt_name: c["prompt_name"].as_str().map(str::to_owned),
            text: c["text"].as_str().unwrap().to_owned(),
        })
        .collect()
}

/// sentence-transformers ids per case for upstream `tag`.
fn real_ids(tag: &str) -> Vec<Vec<u32>> {
    real_doc()["checkpoints"][tag]["ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(ids_of)
        .collect()
}

fn real_goldens(name: &str) -> Vec<Vec<f32>> {
    let (shape, data) = read_npy(FAMILY, name);
    data.chunks_exact(shape[1]).map(<[f32]>::to_vec).collect()
}

fn real_embeddings(model: &BertEmbedder, ids: &[Vec<u32>], label: &str) -> Vec<Vec<f32>> {
    ids.iter()
        .enumerate()
        .map(|(i, ids)| {
            let t = Instant::now();
            let got = model.embed_ids(ids).unwrap();
            eprintln!(
                "{label} case {i}: {} tokens in {:.2?}",
                ids.len(),
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

#[derive(Clone, Copy)]
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

/// Upstream f32 checkpoints (and every MLX conversion against its
/// same-weights golden) vs the f32 reference. Measured worst over all
/// ten checkpoints: 2.9e-7 abs, 1 - cos 3.2e-12 (README); the bounds keep
/// about 10x on the abs diff and 30x on 1 - cos.
const UPSTREAM: Floor = Floor {
    max_abs: 3e-6,
    min_cos: 1.0 - 1e-10,
};

/// `(tag, ~/models/embeddings name, env var)` for each upstream pin.
const UPSTREAMS: &[(&str, &str, &str)] = &[
    (
        "l_v2",
        "snowflake-arctic-embed-l-v2.0",
        "TURBOSPARK_ARCTIC_L_V2_DIR",
    ),
    (
        "m_v15",
        "snowflake-arctic-embed-m-v1.5",
        "TURBOSPARK_ARCTIC_M_V15_DIR",
    ),
    ("l", "snowflake-arctic-embed-l", "TURBOSPARK_ARCTIC_L_DIR"),
    ("m", "snowflake-arctic-embed-m", "TURBOSPARK_ARCTIC_M_DIR"),
    ("s", "snowflake-arctic-embed-s", "TURBOSPARK_ARCTIC_S_DIR"),
    (
        "xs",
        "snowflake-arctic-embed-xs",
        "TURBOSPARK_ARCTIC_XS_DIR",
    ),
];

fn upstream_dir(tag: &str) -> Option<PathBuf> {
    let (_, name, env) = UPSTREAMS.iter().find(|u| u.0 == tag).unwrap();
    model_dir(env, name)
}

/// Text API ids vs sentence-transformers for every case.
fn check_tokenizer(model: &BertEmbedder, want: &[Vec<u32>], label: &str) {
    for (i, (case, want)) in real_cases().iter().zip(want).enumerate() {
        let ids = model
            .tokenize(&case.text, case.prompt_name.as_deref())
            .unwrap();
        assert_eq!(&ids, want, "{label} case {i}");
    }
}

fn upstream_check(tag: &str) {
    let Some(dir) = upstream_dir(tag) else { return };
    let t = Instant::now();
    let model = BertEmbedder::load(&dir).unwrap();
    eprintln!("{tag}: load {:.2?}", t.elapsed());
    let entry = &real_doc()["checkpoints"][tag];
    assert_eq!(model.layout(), Layout::SentenceTransformers);
    assert_eq!(model.pooling(), &cls(model.config().hidden_size));
    assert_eq!(
        model.max_seq_length() as u64,
        entry["max_seq_length"].as_u64().unwrap()
    );
    for (name, p) in entry["prompts"].as_object().unwrap() {
        assert_eq!(model.prompt(name), p.as_str(), "{tag} prompt {name}");
    }
    let ids = real_ids(tag);
    check_tokenizer(&model, &ids, tag);

    let trace_name = format!("real_{tag}_trace_case0.npy");
    if testdata(FAMILY, &trace_name).is_file() {
        let (shape, trace) = read_npy(FAMILY, &trace_name);
        let plane = shape[1] * shape[2];
        let layers = model.config().num_hidden_layers;
        model
            .encoder()
            .forward_traced(&ids[0], None, |k, h| {
                let slot = match k {
                    0 => 0,
                    1 => 1,
                    k if k == layers => 2,
                    _ => return,
                };
                let r = &trace[slot * plane..(slot + 1) * plane];
                let scale = r.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                eprintln!(
                    "{tag} trace state {k}: max abs diff {:e} (max |ref| {scale})",
                    max_abs_diff(h, r)
                );
            })
            .unwrap();
    }

    let got = real_embeddings(&model, &ids, tag);
    assert_floor(&got, &format!("real_{tag}_embed.npy"), tag, &UPSTREAM);

    let mrl_name = format!("real_{tag}_mrl256.npy");
    if testdata(FAMILY, &mrl_name).is_file() {
        assert_eq!(model.matryoshka_dimensions(), &[256]);
        let (_, mrl) = read_npy(FAMILY, &mrl_name);
        let model = model.with_truncate_dim(Some(256)).unwrap();
        let got = model.embed_ids(&ids[0]).unwrap();
        let d = max_abs_diff(&got, &mrl);
        eprintln!("{tag} matryoshka 256: max abs diff {d:e}");
        assert!(d < UPSTREAM.max_abs);
    }
}

#[test]
#[ignore = "needs Snowflake/snowflake-arctic-embed-l-v2.0"]
fn real_l_v2_matches_sentence_transformers() {
    upstream_check("l_v2");
}

#[test]
#[ignore = "needs Snowflake/snowflake-arctic-embed-m-v1.5"]
fn real_m_v15_matches_sentence_transformers() {
    upstream_check("m_v15");
}

#[test]
#[ignore = "needs Snowflake/snowflake-arctic-embed-l"]
fn real_l_matches_sentence_transformers() {
    upstream_check("l");
}

#[test]
#[ignore = "needs Snowflake/snowflake-arctic-embed-m"]
fn real_m_matches_sentence_transformers() {
    upstream_check("m");
}

#[test]
#[ignore = "needs Snowflake/snowflake-arctic-embed-s"]
fn real_s_matches_sentence_transformers() {
    upstream_check("s");
}

#[test]
#[ignore = "needs Snowflake/snowflake-arctic-embed-xs"]
fn real_xs_matches_sentence_transformers() {
    upstream_check("xs");
}

#[test]
#[ignore = "needs Snowflake/snowflake-arctic-embed-l-v2.0"]
fn real_l_v2_truncates_at_8192_like_sentence_transformers() {
    let Some(dir) = upstream_dir("l_v2") else {
        return;
    };
    let model = BertEmbedder::load(&dir).unwrap();
    let entry = &real_doc()["checkpoints"]["l_v2"]["long_tokenize"];
    let want = ids_of(&entry["ids"]);
    let text = super_long_text(entry["sentences"].as_u64().unwrap() as usize);
    let ids = model.tokenize(&text, None).unwrap();
    assert_eq!(ids.len(), 8192);
    assert_eq!(ids, want);
    eprintln!(
        "l_v2 long text: {} tokens untruncated, cut to {}",
        entry["untruncated_len"],
        ids.len()
    );
}

/// The generator's `long_text`, needed only to rebuild the 8192-token
/// text without storing it.
fn super_long_text(sentences: usize) -> String {
    const SUBJECTS: [&str; 10] = [
        "The river",
        "A small robot",
        "Every library",
        "The old bridge",
        "Our neighbor",
        "This algorithm",
        "The orchestra",
        "A distant star",
        "The committee",
        "My grandmother",
    ];
    const VERBS: [&str; 8] = [
        "quietly measured",
        "carefully described",
        "suddenly remembered",
        "slowly rebuilt",
        "happily painted",
        "patiently explained",
        "rarely visited",
        "proudly announced",
    ];
    const OBJECTS: [&str; 9] = [
        "the northern valley",
        "seventeen blue lanterns",
        "an unfinished symphony",
        "the price of copper",
        "a forgotten recipe",
        "the winter schedule",
        "three hundred paper cranes",
        "the theory of tides",
        "a broken compass",
    ];
    const TAILS: [&str; 7] = [
        "before sunrise.",
        "after the long storm.",
        "during the festival.",
        "without telling anyone.",
        "for the second time.",
        "in complete silence.",
        "with remarkable precision.",
    ];
    (0..sentences)
        .map(|i| {
            format!(
                "{} {} {} {} (note {i})",
                SUBJECTS[i % 10],
                VERBS[(i * 3) % 8],
                OBJECTS[(i * 7) % 9],
                TAILS[(i * 5) % 7]
            )
        })
        .collect::<Vec<_>>()
        .join(" ")
}

#[test]
fn long_text_generator_matches_the_fixture_case() {
    // The real case 5 text is long_text(48); the Rust copy must agree so
    // the 8192-token tokenize test rebuilds the same input.
    assert_eq!(super_long_text(48), real_cases()[5].text);
}

/// Gates per MLX conversion of l-v2.0:
/// 1. the reference pipeline on the same (numpy-dequantized or widened)
///    weights, which isolates the port: held to the upstream bound;
/// 2. upstream l-v2.0, which measures the conversion, not the port.
fn mlx_check(tag: &str, bits: Option<u32>, vs_upstream: Floor) {
    let name = format!("mlx-snowflake-arctic-embed-l-v2.0-{tag}");
    let env = format!("TURBOSPARK_ARCTIC_L_V2_MLX_{}_DIR", tag.to_uppercase());
    let Some(dir) = model_dir(&env, &name) else {
        return;
    };
    let t = Instant::now();
    let model = BertEmbedder::load(&dir).unwrap();
    eprintln!("{name}: load {:.2?}", t.elapsed());
    assert_eq!(model.layout(), Layout::Mlx);
    assert_eq!(
        model.pooling_source(),
        &PoolingSource::ConvertedFrom("Snowflake/snowflake-arctic-embed-l-v2.0")
    );
    assert_eq!(model.config().quantization.map(|q| q.bits), bits);
    assert_eq!(model.max_seq_length(), 8192);
    let ids = real_ids("l_v2");
    check_tokenizer(&model, &ids, &name);
    let got = real_embeddings(&model, &ids, &name);
    assert_floor(&got, &format!("real_mlx_{tag}_embed.npy"), &name, &UPSTREAM);
    assert_floor(&got, "real_l_v2_embed.npy", &name, &vs_upstream);
}

// Floors vs upstream are the measured worst case (README) with margin.
#[test]
#[ignore = "needs mlx-community/snowflake-arctic-embed-l-v2.0-bf16"]
fn real_mlx_bf16_matches_reference() {
    mlx_check("bf16", None, MLX_VS_UPSTREAM[0]);
}

#[test]
#[ignore = "needs mlx-community/snowflake-arctic-embed-l-v2.0-8bit"]
fn real_mlx_8bit_matches_reference() {
    mlx_check("8bit", Some(8), MLX_VS_UPSTREAM[1]);
}

#[test]
#[ignore = "needs mlx-community/snowflake-arctic-embed-l-v2.0-6bit"]
fn real_mlx_6bit_matches_reference() {
    mlx_check("6bit", Some(6), MLX_VS_UPSTREAM[2]);
}

#[test]
#[ignore = "needs mlx-community/snowflake-arctic-embed-l-v2.0-4bit"]
fn real_mlx_4bit_matches_reference() {
    mlx_check("4bit", Some(4), MLX_VS_UPSTREAM[3]);
}

// bf16, 8bit, 6bit, 4bit vs upstream l-v2.0: conversion error, not the
// port. Measured worst over the 6 cases (max abs, 1 - cos): bf16 1.3e-3 /
// 4.7e-5, 8-bit 3.3e-3 / 5.2e-4, 6-bit 1.4e-2 / 8.6e-3, 4-bit 4.8e-2 /
// 9.8e-2. Floors keep about 3x on the abs diff and 4x (bf16, 8-bit),
// 3.5x (6-bit), 1.5x (4-bit) on 1 - cos.
const MLX_VS_UPSTREAM: [Floor; 4] = [
    Floor {
        max_abs: 5e-3,
        min_cos: 0.9998,
    },
    Floor {
        max_abs: 1e-2,
        min_cos: 0.998,
    },
    Floor {
        max_abs: 5e-2,
        min_cos: 0.97,
    },
    Floor {
        max_abs: 1.5e-1,
        min_cos: 0.85,
    },
];

#[test]
#[ignore = "needs the arctic-embed xs, m-v1.5, and l-v2.0 installs; timing only"]
fn real_timing() {
    for tag in ["xs", "m_v15", "l_v2"] {
        let Some(dir) = upstream_dir(tag) else {
            continue;
        };
        let t = Instant::now();
        let model = BertEmbedder::load(&dir).unwrap();
        eprintln!("{tag}: load {:.2?}", t.elapsed());
        let ids = real_ids(tag);
        let short = &ids[0];
        // 512 tokens: the BERT case 5 is already cut there; cut l-v2.0's.
        let long = assemble(
            &ids[5][..1],
            &ids[5][1..ids[5].len() - 1],
            &ids[5][ids[5].len() - 1..],
            512,
        );
        assert_eq!(long.len(), 512);
        let _ = model.embed_ids(&short[..3]).unwrap(); // warm threads
        for (label, x) in [("short", short), ("512", &long)] {
            let mut best = f64::MAX;
            for _ in 0..3 {
                let t = Instant::now();
                let _ = model.embed_ids(x).unwrap();
                best = best.min(t.elapsed().as_secs_f64());
            }
            eprintln!(
                "{tag} {label} ({} tokens): best of 3 {:.1} ms",
                x.len(),
                best * 1e3
            );
        }
    }
}
