//! Parity and refusal tests. Fixtures come from
//! `tools/gen_nomic_bert_fixtures.py`; see the family README for
//! regeneration and the measured numbers behind each bound.

use std::path::PathBuf;
use std::time::Instant;

use serde_json::Value;

use super::encoder::{inv_freq, ntk_base, rope_tables};
use super::*;
use crate::testutil::{max_abs_diff, model_dir, read_json, read_npy, testdata};

const FAMILY: &str = "nomic_bert";

/// The `query` prompt of the pinned config_sentence_transformers.json.
const QUERY_PROMPT: &str = "Represent this sentence for searching relevant passages: ";

// Tiny random model vs the remote NomicBertModel in f32. Post-LN keeps
// every state O(1); bounds are f32 reduction-order noise with headroom
// (README has the measured worst case). A wrong rotary layout, a missing
// NTK rescale, or swapped SwiGLU halves moves states by O(0.1..1).
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

fn tiny() -> NomicBert {
    NomicBert::load(testdata(FAMILY, "tiny")).unwrap()
}

#[test]
fn tiny_config_derives_padded_shapes() {
    let c = tiny().config().clone();
    // vocab 60 padded to a multiple of 64; n_inner 200 rounded to 256.
    assert_eq!(c.padded_vocab_size(), 64);
    assert_eq!(c.mlp_hidden(), 256);
    assert_eq!(c.attn_scale(), 0.25);
    assert_eq!(c.rotary_scaling_factor, Some(2.0));
    assert_eq!(c.max_trained_positions, 16);
}

#[test]
fn tiny_matches_reference_across_the_ntk_boundary() {
    let model = tiny();
    let c = model.config().clone();
    assert_eq!(model.max_seq_length(), 64);
    let cases = tiny_cases();
    // Guard the fixture: cases at and past max_trained_positions.
    assert!(cases.iter().any(|ids| ids.len() == c.max_trained_positions));
    assert!(cases
        .iter()
        .any(|ids| ids.len() > 2 * c.max_trained_positions));
    let (mut worst_h, mut worst_e) = (0.0f32, 0.0f32);
    for (i, ids) in cases.iter().enumerate() {
        let (shape, golden) = read_npy(FAMILY, &format!("tiny_hidden_{i}.npy"));
        assert_eq!(shape, vec![c.n_layer + 1, ids.len(), c.n_embd]);
        let plane = ids.len() * c.n_embd;
        let mut seen = 0;
        let last = model
            .encoder()
            .forward_traced(ids, |k, h| {
                let d = max_abs_diff(h, &golden[k * plane..(k + 1) * plane]);
                assert!(
                    d < TINY_HIDDEN_TOL,
                    "case {i} ({} tokens) state {k}: max abs diff {d}",
                    ids.len()
                );
                worst_h = worst_h.max(d);
                seen += 1;
            })
            .unwrap();
        assert_eq!(seen, c.n_layer + 1);
        let emb = model.head(&last, ids.len()).unwrap();
        let (_, want) = read_npy(FAMILY, &format!("tiny_embed_{i}.npy"));
        let d = max_abs_diff(&emb, &want);
        assert!(d < TINY_EMBED_TOL, "case {i} embedding: max abs diff {d}");
        worst_e = worst_e.max(d);
        assert_eq!(model.embed_ids(ids).unwrap(), emb);
    }
    eprintln!("nomic tiny: worst hidden diff {worst_h:e}, worst embedding diff {worst_e:e}");
}

/// Distance in f32 units in the last place (same-sign inputs).
fn ulps(a: f32, b: f32) -> u32 {
    (a.to_bits() as i64 - b.to_bits() as i64).unsigned_abs() as u32
}

#[test]
fn tiny_inv_freq_matches_reference_per_length() {
    let model = tiny();
    for (i, ids) in tiny_cases().iter().enumerate() {
        let (_, want) = read_npy(FAMILY, &format!("tiny_rotary_inv_freq_{i}.npy"));
        let got = model.encoder().inv_freq_for(ids.len());
        let worst = got
            .iter()
            .zip(&want)
            .map(|(g, w)| ulps(*g, *w))
            .max()
            .unwrap();
        assert!(worst <= 2, "case {i} ({} tokens): {worst} ulps", ids.len());
    }
}

#[test]
fn ntk_rescales_only_past_max_trained_positions() {
    let c = tiny().config().clone();
    // Hand-evaluated: base 1000, f 2, M 16, d 16 -> exponent 16 / 14.
    assert_eq!(ntk_base(&c, 1), 1000.0);
    assert_eq!(ntk_base(&c, 16), 1000.0);
    let want = 1000.0 * (2.0f64 * 17.0 / 16.0 - 1.0).powf(16.0 / 14.0);
    assert_eq!(ntk_base(&c, 17), want);
    assert!(ntk_base(&c, 64) > ntk_base(&c, 17));
    // Factor 1 (the class default) still rescales past M.
    let mut one = c.clone();
    one.rotary_scaling_factor = Some(1.0);
    assert_eq!(ntk_base(&one, 32), 1000.0 * 2.0f64.powf(16.0 / 14.0));
    // Scaling disabled (null or 0): never rescales.
    let mut off = c;
    off.rotary_scaling_factor = None;
    assert_eq!(ntk_base(&off, 64), 1000.0);
}

#[test]
fn real_rotary_tables_match_reference() {
    // Pinned config: base 1000, f 2, M 2048, head_dim 64.
    let real = read_json(FAMILY, "real_cases.json");
    let lengths: Vec<usize> = real["rotary"]["lengths"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let rows: Vec<usize> = real["rotary"]["rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let mut c = tiny().config().clone();
    c.n_embd = 768;
    c.n_head = 12;
    c.max_trained_positions = 2048;
    let (_, inv_want) = read_npy(FAMILY, "real_rotary_inv_freq.npy");
    let (_, cos_want) = read_npy(FAMILY, "real_rotary_cos_rows.npy");
    let (_, sin_want) = read_npy(FAMILY, "real_rotary_sin_rows.npy");
    let half = 32;
    let (mut worst_ulps, mut worst_trig) = (0u32, 0.0f32);
    for (li, &n) in lengths.iter().enumerate() {
        let inv = inv_freq(ntk_base(&c, n), 64);
        let want = &inv_want[li * half..(li + 1) * half];
        worst_ulps = worst_ulps.max(
            inv.iter()
                .zip(want)
                .map(|(g, w)| ulps(*g, *w))
                .max()
                .unwrap(),
        );
        let (cos, sin) = rope_tables(rows.iter().max().unwrap() + 1, &inv);
        for (r, &t) in rows.iter().enumerate() {
            let off = (li * rows.len() + r) * half;
            worst_trig = worst_trig
                .max(max_abs_diff(
                    &cos[t * half..(t + 1) * half],
                    &cos_want[off..off + half],
                ))
                .max(max_abs_diff(
                    &sin[t * half..(t + 1) * half],
                    &sin_want[off..off + half],
                ));
        }
    }
    eprintln!("nomic real rotary: worst inv_freq {worst_ulps} ulps, worst cos/sin abs diff {worst_trig:e}");
    // torch's f32 pow is a 1-ulp approximation, measured at most 2 ulps
    // after the reciprocal; angles up to 2047 * inv_freq then differ by
    // at most a few 1e-5 rad.
    assert!(worst_ulps <= 2);
    assert!(worst_trig < 1e-4, "{worst_trig}");
}

#[test]
fn rope_layout_is_rotate_half() {
    // Non-interleaved apply_rotary_emb: x * [cos, cos] + cat(-x2, x1) *
    // [sin, sin] (modeling:483-508).
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
fn reference_statefulness_is_recorded() {
    // The reference reuses a scaled rotary cache after a long input
    // (modeling:646-652); the port is stateless. The generator measured
    // how much a reused reference model drifts; keep it on record.
    let tiny = &read_json(FAMILY, "tiny_cases.json")["stateful_reference_drift"];
    assert!(tiny["max_abs"].as_f64().unwrap() > 1e-3);
    let real = &read_json(FAMILY, "real_cases.json")["stateful_reference_drift"];
    assert!(real["cosine"].as_f64().unwrap() < 0.9999);
}

fn tiny_config() -> Value {
    serde_json::from_slice(&std::fs::read(testdata(FAMILY, "tiny/config.json")).unwrap()).unwrap()
}

#[test]
fn config_refuses_unsupported_values() {
    let cases: Vec<(&str, Value)> = vec![
        ("model_type", serde_json::json!("bert")),
        ("activation_function", serde_json::json!("gelu")),
        ("activation_function", serde_json::json!("geglu")),
        ("rotary_emb_fraction", serde_json::json!(0.5)),
        ("rotary_emb_fraction", serde_json::json!(0.0)),
        ("rotary_emb_interleaved", Value::Bool(true)),
        ("rotary_emb_scale_base", serde_json::json!(512)),
        ("prenorm", Value::Bool(true)),
        ("qkv_proj_bias", Value::Bool(true)),
        ("mlp_fc1_bias", Value::Bool(true)),
        ("mlp_fc2_bias", Value::Bool(true)),
        ("causal", Value::Bool(true)),
        ("parallel_block", Value::Bool(true)),
        ("use_rms_norm", Value::Bool(true)),
        ("num_heads_kv", serde_json::json!(2)),
        ("type_vocab_size", serde_json::json!(0)),
        ("n_head", serde_json::json!(5)),
        ("n_head", serde_json::json!(32)), // head_dim 2
        ("rotary_scaling_factor", serde_json::json!(-1)),
        ("n_inner", Value::Null),
        ("pad_vocab_size_multiple", serde_json::json!(0)),
    ];
    for (key, value) in cases {
        let mut v = tiny_config();
        v[key] = value.clone();
        assert!(
            NomicBertConfig::from_json(&v).is_err(),
            "{key} = {value} was accepted"
        );
    }
    // Class defaults: biases default to true and rotary_emb_fraction to
    // 0, so a config that omits them is refused, not guessed.
    for key in ["qkv_proj_bias", "rotary_emb_fraction"] {
        let mut v = tiny_config();
        v.as_object_mut().unwrap().remove(key);
        assert!(NomicBertConfig::from_json(&v).is_err(), "{key} absent");
    }
    // Scaling factor: null and 0 disable, absent is the class default 1.
    let mut v = tiny_config();
    v["rotary_scaling_factor"] = Value::Null;
    assert_eq!(
        NomicBertConfig::from_json(&v)
            .unwrap()
            .rotary_scaling_factor,
        None
    );
    v["rotary_scaling_factor"] = serde_json::json!(0);
    assert_eq!(
        NomicBertConfig::from_json(&v)
            .unwrap()
            .rotary_scaling_factor,
        None
    );
    v.as_object_mut().unwrap().remove("rotary_scaling_factor");
    assert_eq!(
        NomicBertConfig::from_json(&v)
            .unwrap()
            .rotary_scaling_factor,
        Some(1.0)
    );
}

/// Copies the tiny fixture to a scratch dir for a mutated-load test.
fn scratch_copy(tag: &str) -> PathBuf {
    let src = testdata(FAMILY, "tiny");
    let dst = std::env::temp_dir().join(format!("tse-nomic-{tag}-{}", std::process::id()));
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
fn load_refuses_mismatched_checkpoints() {
    // Vocab rows must be the padded count.
    let dir = scratch_copy("vocab");
    let mut v = tiny_config();
    v["pad_vocab_size_multiple"] = serde_json::json!(1);
    std::fs::write(dir.join("config.json"), serde_json::to_vec(&v).unwrap()).unwrap();
    assert!(matches!(
        NomicBert::load(&dir).unwrap_err(),
        EmbeddingError::Tensor { .. }
    ));
    // MLP width must be the rounded one.
    let mut v = tiny_config();
    v["n_inner"] = serde_json::json!(300); // rounds to 512
    std::fs::write(dir.join("config.json"), serde_json::to_vec(&v).unwrap()).unwrap();
    assert!(matches!(
        NomicBert::load(&dir).unwrap_err(),
        EmbeddingError::Tensor { .. }
    ));
    // n_inner 129 also rounds to 256 and must load.
    let mut v = tiny_config();
    v["n_inner"] = serde_json::json!(129);
    std::fs::write(dir.join("config.json"), serde_json::to_vec(&v).unwrap()).unwrap();
    assert!(NomicBert::load(&dir).is_ok());
    std::fs::remove_dir_all(&dir).unwrap();

    let dir = scratch_copy("pool");
    let path = dir.join("1_Pooling/config.json");
    let mut v: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    v["include_prompt"] = Value::Bool(false);
    std::fs::write(&path, serde_json::to_vec(&v).unwrap()).unwrap();
    let err = NomicBert::load(&dir).unwrap_err();
    assert!(matches!(err, EmbeddingError::Unsupported { .. }), "{err}");
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
    assert!(model.embed_ids(&[1, 64, 2]).is_err()); // past the padded table
    assert!(model.embed_ids(&[1, 63, 2]).is_ok()); // a padding row is still a row
    assert!(model.embed_ids(&[3; 65]).is_err()); // > n_positions
    assert_eq!(model.prompt("query"), Some(QUERY_PROMPT));
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
    assert!(model.with_truncate_dim(Some(0)).is_err());
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

#[test]
fn pinned_tokenizer_matches_sentence_transformers_ids() {
    // testdata/nomic_bert/tokenizer.json is the pinned file (manifest
    // sha256), so this runs without the install. Includes a > 8192-token
    // text cut to max_seq_length with [SEP] kept.
    let tok = WrappedTokenizer::load(&testdata(FAMILY, "tokenizer.json")).unwrap();
    assert_eq!(
        (tok.prefix.as_slice(), tok.suffix.as_slice()),
        (&[101u32][..], &[102u32][..])
    );
    for (i, case) in real_cases().iter().enumerate() {
        let prompt = match case.prompt_name.as_deref() {
            Some("query") => QUERY_PROMPT,
            None => "",
            Some(other) => panic!("unknown prompt {other}"),
        };
        let ids = tok
            .encode(py_strip(&format!("{prompt}{}", case.text)), 8192)
            .unwrap();
        assert_eq!(ids, case.ids, "case {i}");
    }
}

fn real_dir() -> Option<PathBuf> {
    model_dir("TURBOSPARK_NOMIC_BERT_DIR", "snowflake-arctic-embed-m-long")
}

#[test]
#[ignore = "needs the Snowflake/snowflake-arctic-embed-m-long install"]
fn real_tokenizer_matches_sentence_transformers_ids() {
    let Some(dir) = real_dir() else { return };
    let model = NomicBert::load(&dir).unwrap();
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
#[ignore = "needs the Snowflake/snowflake-arctic-embed-m-long install"]
fn real_checkpoint_matches_sentence_transformers() {
    let Some(dir) = real_dir() else { return };
    let model = NomicBert::load(&dir).unwrap();
    let cases = real_cases();
    assert!(cases
        .iter()
        .any(|c| !c.tokenize_only && c.ids.len() > model.config().max_trained_positions));

    let (shape, trace) = read_npy(FAMILY, "real_trace_case0.npy");
    let plane = shape[1] * shape[2];
    let layers = model.config().n_layer;
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
    eprintln!("nomic real: worst max abs diff {worst_d:e}, min cosine {worst_c:.9}");
    assert!(worst_d < REAL_EMBED_TOL, "max abs diff {worst_d}");
    assert!(worst_c > REAL_MIN_COS, "min cosine {worst_c}");
}

#[test]
#[ignore = "needs the Snowflake/snowflake-arctic-embed-m-long install; timing only"]
fn real_text_api_timing() {
    let Some(dir) = real_dir() else { return };
    let t = Instant::now();
    let model = NomicBert::load(&dir).unwrap();
    eprintln!("load: {:.2?}", t.elapsed());
    let cases = real_cases();
    let _ = model.embed_ids(&cases[0].ids).unwrap(); // warm threads
    for i in [0usize, 4] {
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
