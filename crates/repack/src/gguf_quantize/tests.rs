//! Phase A tests. Oracles: the `compute` dequantizers (independent code from
//! the encoders here), a hand-derived Q4_0 fixture, and a parse-back of the
//! written file through the production header parser.
//!
//! Not covered: byte parity with `llama-quantize` (no golden files yet, so no
//! claim is made), and anything about model quality.

use std::path::PathBuf;

use compute::quant_gguf::{
    dequantize_q4_0, dequantize_q4_1, dequantize_q5_0, dequantize_q5_1, dequantize_q8_0,
};
use foundation::LogitValue as F16;

use super::*;
use crate::gguf_header::{parse_header, GgufValue};

/// Deterministic pseudo-random floats in about [-1, 1] with a few outliers.
fn data(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|i| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            let u = (s >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0;
            if i % 97 == 0 {
                u * 8.0
            } else {
                u
            }
        })
        .collect()
}

fn amax(b: &[f32]) -> f32 {
    b.iter().fold(0f32, |a, v| a.max(v.abs()))
}

/// The legacy block decoders against ggml's own `to_float` on ggml's own
/// quantized bytes (`scripts/ggml_dequant_fixture.c`). A round trip through
/// this port's encoder cannot catch a layout both sides misread; this can.
/// `compute`'s Q4_0/Q5_0/Q4_1/Q5_1 decoders are what the GPU kernels are held
/// to, so this is the root of their trust.
#[test]
fn legacy_decoders_match_ggml_to_float() {
    macro_rules! case {
        ($id:literal, $dq:expr) => {{
            let bytes: &[u8] = include_bytes!(concat!(
                "../../tests/fixtures/gguf_quantize/g_",
                $id,
                ".bin"
            ));
            let want: &[u8] = include_bytes!(concat!(
                "../../tests/fixtures/gguf_quantize/d_",
                $id,
                ".f32"
            ));
            let want: Vec<f32> = want
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            // 4 rows x 512; only the first 2048 elements were decoded.
            let got = $dq(bytes, want.len());
            assert_eq!(got.len(), want.len());
            let bad = got.iter().zip(&want).filter(|(a, b)| a != b).count();
            assert_eq!(bad, 0, "type id {}: {bad} elements differ from ggml", $id);
        }};
    }
    case!("2", dequantize_q4_0);
    case!("3", dequantize_q4_1);
    case!("6", dequantize_q5_0);
    case!("7", dequantize_q5_1);
}

#[test]
fn block_codecs_round_trip_within_one_step() {
    let x = data(32 * 64, 7);
    // (target, dequant, levels per side). One quantization step is amax/levels;
    // a slack of 1.1 covers the f16 rounding of the stored scale.
    type Dq = fn(&[u8], usize) -> Vec<f32>;
    let cases: [(Target, Dq, f32); 3] = [
        (Target::Q4_0, dequantize_q4_0, 8.0),
        (Target::Q5_0, dequantize_q5_0, 16.0),
        (Target::Q8_0, dequantize_q8_0, 127.0),
    ];
    for (t, dq, levels) in cases {
        let y = dq(&t.encode(&x), x.len());
        for (bx, by) in x.chunks(32).zip(y.chunks(32)) {
            let step = amax(bx) / levels * 1.1;
            for (a, b) in bx.iter().zip(by) {
                assert!((a - b).abs() <= step, "{t:?}: {a} vs {b}, step {step}");
            }
        }
    }
}

#[test]
fn q4_0_fixture_matches_hand_derived_bytes() {
    // x_j = j - 16. The max-magnitude element is j=0 (-16), so d = -16 / -8 = 2
    // and id = 0.5, giving q_j = min(15, trunc(j / 2 + 0.5)) = min(15, (j+1)/2).
    let x: Vec<f32> = (0..32).map(|j| j as f32 - 16.0).collect();
    let bytes = Target::Q4_0.encode(&x);
    assert_eq!(bytes.len(), 18);
    assert_eq!(&bytes[..2], &F16::from_f32(2.0).to_bits().to_le_bytes());
    let q = |j: usize| j.div_ceil(2).min(15) as u8;
    for k in 0..16 {
        assert_eq!(bytes[2 + k], q(k) | (q(16 + k) << 4), "byte {k}");
    }
}

#[test]
fn zero_block_encodes_to_zero_scale_and_decodes_to_zero() {
    for t in [Target::Q4_0, Target::Q5_0, Target::Q8_0] {
        let bytes = t.encode(&[0.0; 32]);
        // ggml computes d = 0 / -8 = -0.0, stored as f16 0x8000. Same bytes
        // here; only the magnitude matters to a decoder.
        assert_eq!(
            u16::from_le_bytes([bytes[0], bytes[1]]) & 0x7fff,
            0,
            "{t:?} scale"
        );
    }
    assert!(dequantize_q8_0(&Target::Q8_0.encode(&[0.0; 32]), 32)
        .iter()
        .all(|v| *v == 0.0));
}

#[test]
fn bf16_and_f16_round_trip() {
    let x = [1.0f32, -2.5, 0.15625, 3.0e-3];
    let bf = Target::Bf16.encode(&x);
    for (i, v) in x.iter().enumerate() {
        let back = f32::from_bits(u32::from(u16::from_le_bytes([bf[2 * i], bf[2 * i + 1]])) << 16);
        assert!((back - v).abs() <= v.abs() / 128.0);
    }
}

#[test]
fn recipe_rules_first_match_wins_and_layers_filter() {
    let r = Recipe::from_json(
        r#"{"default":"q8_0","rules":[
            {"match":"blk.*.ffn_*","type":"q4_0","layers":[2,3]},
            {"match":"blk.*.ffn_*","type":"q5_0"},
            {"match":"token_embd.weight","type":"keep"}]}"#,
    )
    .unwrap();
    let c = |n: &str| r.rule_for(n).map(|x| x.choice);
    assert_eq!(c("blk.2.ffn_up.weight"), Some(Choice::Type(Target::Q4_0)));
    assert_eq!(c("blk.9.ffn_up.weight"), Some(Choice::Type(Target::Q5_0)));
    assert_eq!(c("token_embd.weight"), Some(Choice::Keep));
    assert_eq!(c("output.weight"), None);
}

#[test]
fn recipe_rejects_unknown_type_field_and_empty_range() {
    assert!(Recipe::from_json(r#"{"default":"q9_9"}"#).is_err());
    assert!(Recipe::from_json(r#"{"default":"q8_0","typo":1}"#).is_err());
    assert!(Recipe::from_json(
        r#"{"default":"q8_0","rules":[{"match":"*","type":"q4_0","layers":[5,1]}]}"#
    )
    .is_err());
}

// ---- end to end -------------------------------------------------------------

struct T {
    name: &'static str,
    dims: Vec<u64>,
    ty: u32,
    bytes: Vec<u8>,
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Minimal GGUF v3 writer for fixtures (alignment 32).
fn build(path: &PathBuf, tensors: &[T]) {
    fn s(v: &mut Vec<u8>, t: &str) {
        v.extend_from_slice(&(t.len() as u64).to_le_bytes());
        v.extend_from_slice(t.as_bytes());
    }
    let mut h = Vec::new();
    h.extend_from_slice(b"GGUF");
    h.extend_from_slice(&3u32.to_le_bytes());
    h.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    h.extend_from_slice(&3u64.to_le_bytes());
    s(&mut h, "general.architecture");
    h.extend_from_slice(&8u32.to_le_bytes());
    s(&mut h, "llama");
    s(&mut h, "general.file_type");
    h.extend_from_slice(&4u32.to_le_bytes());
    h.extend_from_slice(&1u32.to_le_bytes());
    s(&mut h, "tokenizer.ggml.tokens");
    h.extend_from_slice(&9u32.to_le_bytes());
    h.extend_from_slice(&8u32.to_le_bytes());
    h.extend_from_slice(&2u64.to_le_bytes());
    s(&mut h, "a");
    s(&mut h, "b");
    let mut off = 0u64;
    let mut offs = Vec::new();
    for t in tensors {
        offs.push(off);
        off = (off + t.bytes.len() as u64).div_ceil(32) * 32;
    }
    for (t, o) in tensors.iter().zip(&offs) {
        s(&mut h, t.name);
        h.extend_from_slice(&(t.dims.len() as u32).to_le_bytes());
        t.dims
            .iter()
            .for_each(|d| h.extend_from_slice(&d.to_le_bytes()));
        h.extend_from_slice(&t.ty.to_le_bytes());
        h.extend_from_slice(&o.to_le_bytes());
    }
    h.resize(h.len().div_ceil(32) * 32, 0);
    let start = h.len();
    for (t, o) in tensors.iter().zip(&offs) {
        h.resize(start + *o as usize, 0);
        h.extend_from_slice(&t.bytes);
    }
    std::fs::write(path, h).unwrap();
}

fn scratch(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("ts-gguf-quant-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

#[test]
fn end_to_end_rewrite_applies_recipe_and_preserves_the_rest() {
    let dir = scratch("e2e");
    let (w, norm, ragged) = (data(64 * 4, 1), data(64, 2), data(30 * 2, 3));
    let src = dir.join("in.gguf");
    build(
        &src,
        &[
            T {
                name: "blk.0.ffn_up.weight",
                dims: vec![64, 4],
                ty: 0,
                bytes: f32_bytes(&w),
            },
            T {
                name: "blk.0.attn_norm.weight",
                dims: vec![64],
                ty: 0,
                bytes: f32_bytes(&norm),
            },
            T {
                name: "blk.0.odd.weight",
                dims: vec![30, 2],
                ty: 0,
                bytes: f32_bytes(&ragged),
            },
        ],
    );
    let recipe = Recipe::from_json(r#"{"default":"q8_0","fallback":"f16"}"#).unwrap();
    let dst = dir.join("out.gguf");
    let rep = quantize_gguf(&src, &dst, &recipe).unwrap();
    assert_eq!(rep.tensors, 3);

    let bytes = std::fs::read(&dst).unwrap();
    let h = parse_header(&bytes, 1 << 26).unwrap();
    // Types: weight -> Q8_0, norm untouched, ragged row -> fallback F16.
    assert_eq!(h.tensors["blk.0.ffn_up.weight"].ggml_type, 8);
    assert_eq!(h.tensors["blk.0.attn_norm.weight"].ggml_type, 0);
    assert_eq!(h.tensors["blk.0.odd.weight"].ggml_type, 1);
    // Metadata preserved (array included); file_type rewritten to Q8_0 = 7.
    assert_eq!(
        h.metadata["general.architecture"],
        GgufValue::String("llama".into())
    );
    assert_eq!(
        h.metadata["tokenizer.ggml.tokens"],
        GgufValue::Array(vec![
            GgufValue::String("a".into()),
            GgufValue::String("b".into())
        ])
    );
    assert_eq!(h.metadata["general.file_type"], GgufValue::U32(7));
    // Data: read every tensor back through its own header range and compare.
    let rng = |n: &str| {
        let (a, b) = h.absolute_range(n).unwrap().unwrap();
        &bytes[a as usize..b as usize]
    };
    let back = dequantize_q8_0(rng("blk.0.ffn_up.weight"), w.len());
    for (bx, by) in w.chunks(32).zip(back.chunks(32)) {
        assert!(bx
            .iter()
            .zip(by)
            .all(|(a, b)| (a - b).abs() <= amax(bx) / 127.0 * 1.1));
    }
    assert_eq!(rng("blk.0.attn_norm.weight"), &f32_bytes(&norm)[..]);
    let odd: Vec<f32> = rng("blk.0.odd.weight")
        .chunks_exact(2)
        .map(|b| f32::from(F16::from_bits(u16::from_le_bytes([b[0], b[1]]))))
        .collect();
    assert!(odd
        .iter()
        .zip(&ragged)
        .all(|(a, b)| (a - b).abs() <= b.abs() / 1000.0 + 1e-6));
    // Offsets obey the file's alignment.
    assert!(h.tensors.values().all(|t| t.offset % 32 == 0));
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn quantized_source_is_kept_by_default_and_refused_by_explicit_rule() {
    let dir = scratch("requant");
    let q = Target::Q8_0.encode(&data(64 * 2, 5));
    let src = dir.join("in.gguf");
    build(
        &src,
        &[T {
            name: "blk.0.ffn_up.weight",
            dims: vec![64, 2],
            ty: 8,
            bytes: q.clone(),
        }],
    );
    let dst = dir.join("out.gguf");

    let keep = Recipe::from_json(r#"{"default":"q4_0"}"#).unwrap();
    let rep = quantize_gguf(&src, &dst, &keep).unwrap();
    assert_eq!(rep.kept_quantized, vec!["blk.0.ffn_up.weight".to_string()]);
    let bytes = std::fs::read(&dst).unwrap();
    let h = parse_header(&bytes, 1 << 26).unwrap();
    assert_eq!(h.tensors["blk.0.ffn_up.weight"].ggml_type, 8);
    let (a, b) = h.absolute_range("blk.0.ffn_up.weight").unwrap().unwrap();
    assert_eq!(&bytes[a as usize..b as usize], &q[..]);

    let strict =
        Recipe::from_json(r#"{"default":"q8_0","rules":[{"match":"blk.*","type":"q4_0"}]}"#)
            .unwrap();
    let err = quantize_gguf(&src, &dir.join("o2.gguf"), &strict).unwrap_err();
    assert!(err.to_string().contains("blk.0.ffn_up.weight"), "{err}");
    assert!(!dir.join("o2.gguf").exists() && !dir.join("o2.gguf.partial").exists());
    std::fs::remove_dir_all(dir).unwrap();
}

// ---- byte parity with ggml --------------------------------------------------

mod golden {
    use super::*;

    const IN: &[u8] = include_bytes!("../../tests/fixtures/gguf_quantize/in.f32");
    const IM: &[u8] = include_bytes!("../../tests/fixtures/gguf_quantize/im.f32");

    fn floats(b: &[u8]) -> Vec<f32> {
        b.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    /// Number of differing bytes, and the first differing block index.
    fn diff(a: &[u8], b: &[u8], block: usize) -> (usize, Option<usize>) {
        assert_eq!(a.len(), b.len(), "length");
        let n = a.iter().zip(b).filter(|(x, y)| x != y).count();
        let first = a.iter().zip(b).position(|(x, y)| x != y).map(|p| p / block);
        (n, first)
    }

    /// Compare `got` to a golden, reporting where the bytes differ.
    fn check(target: Target, with_im: bool, golden: &[u8]) {
        let (x, im) = (floats(IN), floats(IM));
        let got = target.encode_rows(&x, 512, with_im.then_some(&im[..]));
        let (_, bb) = target.block();
        let (n, first) = diff(&got, golden, bb);
        let offs: Vec<(usize, usize)> = got
            .iter()
            .zip(golden)
            .enumerate()
            .filter(|(_, (a, b))| a != b)
            .map(|(i, _)| (i / bb, i % bb))
            .take(12)
            .collect();
        assert_eq!(
            n, 0,
            "{n} bytes differ, first in block {first:?}; (block, offset): {offs:?}"
        );
    }

    // The golden is chosen at compile time so a type that cannot run without
    // an imatrix needs no (nonexistent) imatrix-free file.
    macro_rules! parity {
        ($name:ident, $target:expr, $id:literal, true) => {
            parity!($name, $target, $id, true, #[test]);
        };
        ($name:ident, $target:expr, $id:literal, false) => {
            parity!($name, $target, $id, false, #[test]);
        };
        ($name:ident, $target:expr, $id:literal, true, $($attr:tt)*) => {
            $($attr)*
            fn $name() {
                check($target, true, include_bytes!(concat!("../../tests/fixtures/gguf_quantize/gi_", $id, ".bin")));
            }
        };
        ($name:ident, $target:expr, $id:literal, false, $($attr:tt)*) => {
            $($attr)*
            fn $name() {
                check($target, false, include_bytes!(concat!("../../tests/fixtures/gguf_quantize/g_", $id, ".bin")));
            }
        };
    }

    parity!(q4_0, Target::Q4_0, "2", false);
    parity!(q4_0_imatrix, Target::Q4_0, "2", true);
    parity!(q5_0_imatrix, Target::Q5_0, "6", true);
    parity!(q8_0_imatrix, Target::Q8_0, "8", true);
    parity!(q5_0, Target::Q5_0, "6", false);
    parity!(q4_1, Target::Q4_1, "3", false);
    parity!(q4_1_imatrix, Target::Q4_1, "3", true);
    parity!(q5_1, Target::Q5_1, "7", false);
    parity!(q5_1_imatrix, Target::Q5_1, "7", true);
    parity!(q8_0, Target::Q8_0, "8", false);
    parity!(q2_k, Target::Q2K, "10", false);
    parity!(q3_k, Target::Q3K, "11", false);
    parity!(q4_k, Target::Q4K, "12", false);
    parity!(q5_k, Target::Q5K, "13", false);
    parity!(q6_k, Target::Q6K, "14", false);
    parity!(q2_k_imatrix, Target::Q2K, "10", true);
    parity!(q3_k_imatrix, Target::Q3K, "11", true);
    parity!(q4_k_imatrix, Target::Q4K, "12", true);
    parity!(q5_k_imatrix, Target::Q5K, "13", true);
    parity!(q6_k_imatrix, Target::Q6K, "14", true);
    parity!(iq4_nl, Target::Iq4Nl, "20", false);
    parity!(iq4_nl_imatrix, Target::Iq4Nl, "20", true);
    parity!(iq4_xs, Target::Iq4Xs, "23", false);
    parity!(iq4_xs_imatrix, Target::Iq4Xs, "23", true);
    // IQ2_XXS and IQ2_XS refuse to run without an imatrix, in ggml and here.
    parity!(iq2_xxs_imatrix, Target::Iq2Xxs, "16", true);
    parity!(iq2_xs_imatrix, Target::Iq2Xs, "17", true);
    parity!(iq2_s, Target::Iq2S, "22", false);
    parity!(iq2_s_imatrix, Target::Iq2S, "22", true);
    parity!(iq3_xxs, Target::Iq3Xxs, "18", false);
    parity!(iq3_xxs_imatrix, Target::Iq3Xxs, "18", true);
    parity!(iq3_s, Target::Iq3S, "21", false);
    parity!(iq3_s_imatrix, Target::Iq3S, "21", true);
    parity!(iq1_s_imatrix, Target::Iq1S, "19", true);
    parity!(iq1_m, Target::Iq1M, "29", false);
    parity!(iq1_m_imatrix, Target::Iq1M, "29", true);
}

// ---- imatrix ------------------------------------------------------------------

mod imx {
    use super::*;
    use std::collections::HashMap;

    const IN: &[u8] = include_bytes!("../../tests/fixtures/gguf_quantize/in.f32");
    const IM: &[u8] = include_bytes!("../../tests/fixtures/gguf_quantize/im.f32");
    const G_Q4K: &[u8] = include_bytes!("../../tests/fixtures/gguf_quantize/g_12.bin");
    const GI_Q4K: &[u8] = include_bytes!("../../tests/fixtures/gguf_quantize/gi_12.bin");
    const GI_Q40: &[u8] = include_bytes!("../../tests/fixtures/gguf_quantize/gi_2.bin");

    fn fl(b: &[u8]) -> Vec<f32> {
        b.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    /// Legacy-format imatrix: (name, ncall, values) entries, optional trailer.
    fn legacy(entries: &[(&str, i32, Vec<f32>)], trailer: Option<(i32, &str)>) -> Vec<u8> {
        let mut v = (entries.len() as i32).to_le_bytes().to_vec();
        for (n, ncall, vals) in entries {
            v.extend_from_slice(&(n.len() as i32).to_le_bytes());
            v.extend_from_slice(n.as_bytes());
            v.extend_from_slice(&ncall.to_le_bytes());
            v.extend_from_slice(&(vals.len() as i32).to_le_bytes());
            vals.iter()
                .for_each(|x| v.extend_from_slice(&x.to_le_bytes()));
        }
        if let Some((calls, ds)) = trailer {
            v.extend_from_slice(&calls.to_le_bytes());
            v.extend_from_slice(&(ds.len() as i32).to_le_bytes());
            v.extend_from_slice(ds.as_bytes());
        }
        v
    }

    fn src_with(
        dir: &std::path::Path,
        name: &'static str,
        dims: Vec<u64>,
        vals: &[f32],
    ) -> PathBuf {
        let p = dir.join("in.gguf");
        build(
            &p,
            &[T {
                name,
                dims,
                ty: 0,
                bytes: f32_bytes(vals),
            }],
        );
        p
    }

    fn tensor_bytes(path: &PathBuf, name: &str) -> Vec<u8> {
        let bytes = std::fs::read(path).unwrap();
        let h = parse_header(&bytes, 1 << 26).unwrap();
        let (a, b) = h.absolute_range(name).unwrap().unwrap();
        bytes[a as usize..b as usize].to_vec()
    }

    #[test]
    fn low_bit_types_are_refused_without_an_imatrix_except_where_llama_cpp_allows() {
        let dir = scratch("imx-required");
        let x = fl(IN);
        let r = |ty: &str| Recipe::from_json(&format!(r#"{{"default":"{ty}"}}"#)).unwrap();

        // Hard requirement (ggml asserts): IQ2_XXS, even on token_embd.
        for name in ["blk.0.w.weight", "token_embd.weight"] {
            let src = src_with(&dir, name, vec![512, 4], &x);
            let e = quantize_gguf(&src, &dir.join("a.gguf"), &r("iq2_xxs")).unwrap_err();
            assert!(
                e.to_string().contains(name) && e.to_string().contains("importance matrix"),
                "{e}"
            );
        }
        // Policy requirement (llama.cpp): IQ3_XXS is refused on a block weight...
        let src = src_with(&dir, "blk.0.w.weight", vec![512, 4], &x);
        assert!(quantize_gguf(&src, &dir.join("b.gguf"), &r("iq3_xxs")).is_err());
        // ...but allowed on token_embd and output.weight.
        for name in ["token_embd.weight", "output.weight"] {
            let src = src_with(&dir, name, vec![512, 4], &x);
            let out = dir.join(format!("{name}.gguf"));
            quantize_gguf(&src, &out, &r("iq3_xxs")).unwrap();
            let bytes = std::fs::read(&out).unwrap();
            assert_eq!(
                parse_header(&bytes, 1 << 26).unwrap().tensors[name].ggml_type,
                18
            );
        }
        // With an imatrix entry the block weight is accepted.
        let im = Imatrix::from_entries(HashMap::from([("blk.0.w.weight".to_string(), fl(IM))]));
        quantize_gguf_with(
            &src_with(&dir, "blk.0.w.weight", vec![512, 4], &x),
            &dir.join("c.gguf"),
            &r("iq3_xxs"),
            Some(&im),
        )
        .unwrap();
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn legacy_divides_by_ncall_and_reads_trailer() {
        let b = legacy(
            &[
                ("a.weight", 4, vec![4.0, 8.0, 12.0]),
                ("b.weight", 0, vec![5.0]),
            ],
            Some((7, "wiki")),
        );
        let im = Imatrix::from_bytes(&b).unwrap();
        assert_eq!(im.get("a.weight").unwrap(), &[1.0, 2.0, 3.0]);
        assert_eq!(
            im.get("b.weight").unwrap(),
            &[5.0],
            "ncall 0 keeps the raw sum"
        );
        assert_eq!(
            (im.chunk_count, im.datasets.clone(), im.legacy),
            (7, vec!["wiki".to_string()], true)
        );
        // No trailer is fine; a truncated body and non-finite values are not.
        assert!(Imatrix::from_bytes(&legacy(&[("a", 1, vec![1.0])], None)).is_ok());
        assert!(Imatrix::from_bytes(&b[..20]).is_err());
        assert!(Imatrix::from_bytes(&legacy(&[("a", 1, vec![f32::NAN])], None)).is_err());
    }

    #[test]
    fn gguf_format_divides_per_expert_and_zero_count_becomes_one() {
        let dir = scratch("imx-gguf");
        let p = dir.join("im.gguf");
        build(
            &p,
            &[
                T {
                    name: "a.weight.in_sum2",
                    dims: vec![3, 2],
                    ty: 0,
                    bytes: f32_bytes(&[2.0, 4.0, 6.0, 9.0, 9.0, 9.0]),
                },
                T {
                    name: "a.weight.counts",
                    dims: vec![1, 2],
                    ty: 0,
                    bytes: f32_bytes(&[2.0, 0.0]),
                },
            ],
        );
        let im = Imatrix::load(&p).unwrap();
        assert_eq!(im.get("a.weight").unwrap(), &[1.0, 2.0, 3.0, 1.0, 1.0, 1.0]);
        // in_sum2 without counts is rejected.
        build(
            &p,
            &[T {
                name: "x.in_sum2",
                dims: vec![2],
                ty: 0,
                bytes: f32_bytes(&[1.0, 2.0]),
            }],
        );
        assert!(Imatrix::load(&p).is_err());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn file_rewrite_with_imatrix_is_byte_identical_to_ggml() {
        let dir = scratch("imx-e2e");
        let x = fl(IN);
        let src = src_with(&dir, "blk.0.ffn_up.weight", vec![512, 4], &x);
        let im = Imatrix::from_bytes(&legacy(&[("blk.0.ffn_up.weight", 1, fl(IM))], None)).unwrap();
        for (ty, with, golden) in [
            ("q4_k", true, GI_Q4K),
            ("q4_k", false, G_Q4K),
            ("q4_0", true, GI_Q40),
        ] {
            let r = Recipe::from_json(&format!(r#"{{"default":"{ty}"}}"#)).unwrap();
            let dst = dir.join(format!("out-{ty}-{with}.gguf"));
            let rep = quantize_gguf_with(&src, &dst, &r, with.then_some(&im)).unwrap();
            assert_eq!(rep.imatrix_used, u64::from(with));
            assert_eq!(
                tensor_bytes(&dst, "blk.0.ffn_up.weight"),
                golden,
                "{ty} imatrix={with}"
            );
        }
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn expert_tensors_use_one_imatrix_slice_per_expert() {
        let dir = scratch("imx-moe");
        let x = fl(IN);
        // dims [512, 2, 2]: two experts of two rows each, one slice per expert.
        let src = src_with(&dir, "blk.0.ffn_down_exps.weight", vec![512, 2, 2], &x);
        let s0 = fl(IM);
        let s1: Vec<f32> = s0.iter().rev().cloned().collect();
        let both: Vec<f32> = s0.iter().chain(&s1).cloned().collect();
        let im = Imatrix::from_entries(HashMap::from([(
            "blk.0.ffn_down_exps.weight".to_string(),
            both,
        )]));
        let r = Recipe::from_json(r#"{"default":"q4_k"}"#).unwrap();
        let dst = dir.join("o.gguf");
        quantize_gguf_with(&src, &dst, &r, Some(&im)).unwrap();
        let mut want = Target::Q4K.encode_rows(&x[..1024], 512, Some(&s0));
        want.extend(Target::Q4K.encode_rows(&x[1024..], 512, Some(&s1)));
        assert_eq!(tensor_bytes(&dst, "blk.0.ffn_down_exps.weight"), want);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn wrong_size_is_refused_except_token_embd_and_missing_is_reported() {
        let dir = scratch("imx-size");
        let x = fl(IN);
        let r = Recipe::from_json(r#"{"default":"q4_k"}"#).unwrap();
        let bad = Imatrix::from_entries(HashMap::from([(
            "blk.0.w.weight".to_string(),
            vec![1.0; 100],
        )]));
        let src = src_with(&dir, "blk.0.w.weight", vec![512, 4], &x);
        let e = quantize_gguf_with(&src, &dir.join("o1.gguf"), &r, Some(&bad)).unwrap_err();
        assert!(
            e.to_string().contains("blk.0.w.weight") && e.to_string().contains("100"),
            "{e}"
        );
        assert!(!dir.join("o1.gguf").exists());

        let src = src_with(&dir, "token_embd.weight", vec![512, 4], &x);
        let tok = Imatrix::from_entries(HashMap::from([(
            "token_embd.weight".to_string(),
            vec![1.0; 100],
        )]));
        let rep = quantize_gguf_with(&src, &dir.join("o2.gguf"), &r, Some(&tok)).unwrap();
        assert_eq!(
            rep.imatrix_used, 0,
            "token_embd mismatch is skipped, as in llama.cpp"
        );

        let other = Imatrix::from_entries(HashMap::from([("zzz".to_string(), vec![1.0; 512])]));
        let rep = quantize_gguf_with(&src, &dir.join("o3.gguf"), &r, Some(&other)).unwrap();
        assert_eq!(rep.imatrix_missing, vec!["token_embd.weight".to_string()]);
        std::fs::remove_dir_all(dir).unwrap();
    }
}

// ---- second, adversarial input ----------------------------------------------
//
// 3 rows x 768: heavy outliers, zeroed 32-wide sub-blocks inside a live block,
// a nearly constant sub-block, an all-negative sub-block, a tiny-magnitude block, a
// large-magnitude block, and near-zero importance columns. Goldens are again
// ggml's own output. Types that ggml refuses without an imatrix have no
// imatrix-free file.

mod golden2 {
    use super::*;

    const IN: &[u8] = include_bytes!("../../tests/fixtures/gguf_quantize/in2.f32");
    const IM: &[u8] = include_bytes!("../../tests/fixtures/gguf_quantize/im2.f32");

    fn floats(b: &[u8]) -> Vec<f32> {
        b.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    macro_rules! case {
        ($t:expr, $id:literal, plain) => {
            (
                $t,
                $id,
                Some(include_bytes!(concat!(
                    "../../tests/fixtures/gguf_quantize/g2_",
                    $id,
                    ".bin"
                )) as &[u8]),
                include_bytes!(concat!(
                    "../../tests/fixtures/gguf_quantize/gi2_",
                    $id,
                    ".bin"
                )) as &[u8],
            )
        };
        ($t:expr, $id:literal, imatrix_only) => {
            (
                $t,
                $id,
                None,
                include_bytes!(concat!(
                    "../../tests/fixtures/gguf_quantize/gi2_",
                    $id,
                    ".bin"
                )) as &[u8],
            )
        };
    }

    fn first_diff(a: &[u8], b: &[u8], block: usize) -> String {
        if a.len() != b.len() {
            return format!("length {} vs {}", a.len(), b.len());
        }
        let n = a.iter().zip(b).filter(|(x, y)| x != y).count();
        match a.iter().zip(b).position(|(x, y)| x != y) {
            None => String::new(),
            Some(p) => format!(
                "{n} bytes differ, first at block {} offset {}",
                p / block,
                p % block
            ),
        }
    }

    #[test]
    fn every_type_matches_ggml_on_the_adversarial_input() {
        let (x, im) = (floats(IN), floats(IM));
        let cases = [
            case!(Target::Q4_0, "2", plain),
            case!(Target::Q5_0, "6", plain),
            case!(Target::Q4_1, "3", plain),
            case!(Target::Q5_1, "7", plain),
            case!(Target::Q8_0, "8", plain),
            case!(Target::Q3K, "11", plain),
            case!(Target::Q4K, "12", plain),
            case!(Target::Q5K, "13", plain),
            case!(Target::Q6K, "14", plain),
            case!(Target::Iq4Nl, "20", plain),
            case!(Target::Iq4Xs, "23", plain),
            case!(Target::Iq3Xxs, "18", plain),
            case!(Target::Iq3S, "21", plain),
            case!(Target::Iq2S, "22", plain),
            case!(Target::Iq1M, "29", plain),
            case!(Target::Iq2Xxs, "16", imatrix_only),
            case!(Target::Iq2Xs, "17", imatrix_only),
            case!(Target::Iq1S, "19", imatrix_only),
        ];
        let mut bad = Vec::new();
        for (t, id, plain, with_im) in cases {
            let (_, bb) = t.block();
            if let Some(g) = plain {
                let d = first_diff(&t.encode_rows(&x, 768, None), g, bb);
                if !d.is_empty() {
                    bad.push(format!("{t:?} no imatrix (id {id}): {d}"));
                }
            }
            let d = first_diff(&t.encode_rows(&x, 768, Some(&im)), with_im, bb);
            if !d.is_empty() {
                bad.push(format!("{t:?} imatrix (id {id}): {d}"));
            }
        }
        assert!(bad.is_empty(), "{}", bad.join("\n"));
    }

    #[test]
    fn q2_k_without_imatrix() {
        let x = floats(IN);
        let g: &[u8] = include_bytes!("../../tests/fixtures/gguf_quantize/g2_10.bin");
        let d = first_diff(&Target::Q2K.encode_rows(&x, 768, None), g, 84);
        assert!(d.is_empty(), "{d}");
    }

    #[test]
    fn q2_k_with_imatrix() {
        let (x, im) = (floats(IN), floats(IM));
        let g: &[u8] = include_bytes!("../../tests/fixtures/gguf_quantize/gi2_10.bin");
        let d = first_diff(&Target::Q2K.encode_rows(&x, 768, Some(&im)), g, 84);
        assert!(d.is_empty(), "{d}");
    }
}

// ---- recipe from a reference GGUF ---------------------------------------------

mod reference {
    use super::*;

    fn header_only(path: &std::path::Path, tensors: &[(&'static str, Vec<u64>, u32)]) {
        let ts: Vec<T> = tensors
            .iter()
            .map(|(name, dims, ty)| T {
                name,
                dims: dims.clone(),
                ty: *ty,
                bytes: vec![0u8; 32],
            })
            .collect();
        build(&path.to_path_buf(), &ts);
    }

    #[test]
    fn reference_types_are_reproduced_on_a_float_copy() {
        let dir = scratch("ref");
        let reference = dir.join("ref.gguf");
        // The shape of a published mix: different types per layer of one role.
        header_only(
            &reference,
            &[
                ("blk.0.ffn_down.weight", vec![256, 2], 19), // IQ1_S
                ("blk.1.ffn_down.weight", vec![256, 2], 16), // IQ2_XXS
                ("blk.2.ffn_down.weight", vec![256, 2], 10), // Q2_K
                ("token_embd.weight", vec![256, 2], 10),
                ("output.weight", vec![256, 2], 11),      // Q3_K
                ("blk.0.attn_norm.weight", vec![256], 0), // 1-D, no rule
            ],
        );
        let r = recipe_from_gguf(&reference).unwrap();
        assert_eq!(r.rules.len(), 5);

        // The JSON form round-trips to the same rules.
        let back = Recipe::from_json(&r.to_json()).unwrap();
        let ids = |r: &Recipe| -> Vec<(String, Choice)> {
            r.rules
                .iter()
                .map(|x| (x.pattern.clone(), x.choice))
                .collect()
        };
        assert_eq!(ids(&r), ids(&back));

        // Apply it to a float source with the same names plus one the
        // reference does not know, which must stay as it is.
        let x = fl_data(512);
        let src = dir.join("src.gguf");
        let t = |name: &'static str, dims: Vec<u64>| T {
            name,
            dims,
            ty: 0,
            bytes: f32_bytes(&x[..dims_len(&[256, 2])]),
        };
        build(
            &src,
            &[
                t("blk.0.ffn_down.weight", vec![256, 2]),
                t("blk.1.ffn_down.weight", vec![256, 2]),
                t("blk.2.ffn_down.weight", vec![256, 2]),
                t("token_embd.weight", vec![256, 2]),
                t("output.weight", vec![256, 2]),
                t("blk.9.unknown.weight", vec![256, 2]),
            ],
        );
        // IQ1_S and IQ2_XXS need importance weights; give every tensor some.
        let ones = vec![1.0f32; 256];
        let im = Imatrix::from_entries(
            [
                "blk.0.ffn_down.weight",
                "blk.1.ffn_down.weight",
                "blk.2.ffn_down.weight",
                "token_embd.weight",
                "output.weight",
            ]
            .iter()
            .map(|n| (n.to_string(), ones.clone()))
            .collect(),
        );
        let dst = dir.join("out.gguf");
        quantize_gguf_with(&src, &dst, &r, Some(&im)).unwrap();
        let h = parse_header(&std::fs::read(&dst).unwrap(), 1 << 26).unwrap();
        let ty = |n: &str| h.tensors[n].ggml_type;
        assert_eq!(
            [
                ty("blk.0.ffn_down.weight"),
                ty("blk.1.ffn_down.weight"),
                ty("blk.2.ffn_down.weight"),
                ty("token_embd.weight"),
                ty("output.weight")
            ],
            [19, 16, 10, 10, 11]
        );
        assert_eq!(
            ty("blk.9.unknown.weight"),
            0,
            "a tensor the reference does not name is kept"
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn every_type_the_quantizer_writes_is_on_the_executable_list() {
        let dir = scratch("ref-exec");
        let x = super::data(512, 3);
        let src = dir.join("in.gguf");
        build(
            &src,
            &[T {
                name: "blk.0.w.weight",
                dims: vec![256, 2],
                ty: 0,
                bytes: f32_bytes(&x),
            }],
        );
        let run = |ty: &str| {
            let r = Recipe::from_json(&format!(r#"{{"default":"{ty}"}}"#)).unwrap();
            let dst = dir.join(format!("{ty}.gguf"));
            quantize_gguf(&src, &dst, &r)
                .unwrap()
                .not_in_executable_list
        };
        // The legacy block types gained kernels, so this used to assert that
        // q4_0 was flagged. Now NO producible type is outside the gate, which
        // is the invariant worth holding: a new Target must either ship with
        // kernels or this test names it. (`not_in_executable_list` stays in
        // the report for exactly that day.)
        for t in Target::ALL {
            let float = matches!(t, Target::F32 | Target::F16 | Target::Bf16);
            let listed = model_io::EXECUTABLE_GGUF_TYPES.contains(&t.name());
            assert!(
                float || listed,
                "{} is writable but not executable",
                t.name()
            );
        }
        for ty in ["q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "q4_k"] {
            assert!(run(ty).is_empty(), "{ty} should be accepted by the gate");
        }
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn unsupported_reference_types_are_refused_by_name() {
        let dir = scratch("ref-bad");
        let p = dir.join("ref.gguf");
        for (id, what) in [(9u32, "Q8_1"), (99, "unknown")] {
            header_only(&p, &[("blk.0.ffn_down.weight", vec![256, 2], id)]);
            let e = recipe_from_gguf(&p).unwrap_err().to_string();
            assert!(
                e.contains("blk.0.ffn_down.weight") && e.contains(what),
                "{e}"
            );
        }
        std::fs::remove_dir_all(dir).unwrap();
    }

    fn dims_len(d: &[usize]) -> usize {
        d.iter().product()
    }

    fn fl_data(n: usize) -> Vec<f32> {
        super::data(n, 11)
    }
}

/// Debug harness (not a regression test): compares this crate to ggml on rows
/// extracted from a real model. `GGUF_DBG` = `dir:tag:target:n_per_row`.
#[test]
#[ignore = "debug harness, needs GGUF_DBG"]
fn dbg_real_rows() {
    let Ok(spec) = std::env::var("GGUF_DBG") else {
        return;
    };
    if std::env::var("GGUF_TIE").as_deref() == Ok("apple") {
        set_tie_order(TieOrder::Apple);
    }
    let p: Vec<&str> = spec.split(':').collect();
    let (dir, tag, target, n) = (
        p[0],
        p[1],
        Target::parse(p[2]).unwrap(),
        p[3].parse::<usize>().unwrap(),
    );
    let rd = |s: &str| std::fs::read(format!("{dir}/{tag}.{s}")).unwrap();
    let fl = |b: Vec<u8>| -> Vec<f32> {
        b.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    };
    let (rows, im, gold) = (fl(rd("rows.f32")), fl(rd("im.f32")), rd("gold.bin"));
    let (be, bb) = target.block();
    let row_bytes = n / be * bb;
    let mut bad = 0;
    for (r, row) in rows.chunks(n).enumerate() {
        let mine = target.encode_rows(row, n, Some(&im));
        let g = &gold[r * row_bytes..(r + 1) * row_bytes];
        for b in 0..row_bytes / bb {
            if mine[b * bb..(b + 1) * bb] != g[b * bb..(b + 1) * bb] {
                bad += 1;
                let d: Vec<usize> = (0..bb)
                    .filter(|&i| mine[b * bb + i] != g[b * bb + i])
                    .collect();
                println!("row {r} block {b}: differing offsets {d:?}");
            }
        }
    }
    println!("differing blocks: {bad}");
}

// ---- threading ------------------------------------------------------------------

#[test]
fn output_does_not_depend_on_the_thread_count() {
    let dir = scratch("threads");
    // 37 rows (uneven split across threads) and an expert tensor [256, 5, 3].
    let plain = data(256 * 37, 21);
    let moe = data(256 * 5 * 3, 22);
    let src = dir.join("in.gguf");
    build(
        &src,
        &[
            T {
                name: "blk.0.attn_q.weight",
                dims: vec![256, 37],
                ty: 0,
                bytes: f32_bytes(&plain),
            },
            T {
                name: "blk.0.ffn_up_exps.weight",
                dims: vec![256, 5, 3],
                ty: 0,
                bytes: f32_bytes(&moe),
            },
        ],
    );
    let im = Imatrix::from_entries(std::collections::HashMap::from([
        (
            "blk.0.attn_q.weight".to_string(),
            data(256, 23).iter().map(|v| v.abs() + 0.01).collect(),
        ),
        (
            "blk.0.ffn_up_exps.weight".to_string(),
            data(256 * 3, 24).iter().map(|v| v.abs() + 0.01).collect(),
        ),
    ]));
    for ty in ["q4_k", "iq3_xxs", "iq4_nl"] {
        let r = Recipe::from_json(&format!(r#"{{"default":"{ty}"}}"#)).unwrap();
        let mut outs = Vec::new();
        for threads in [1, 2, 7] {
            let dst = dir.join(format!("{ty}-{threads}.gguf"));
            quantize_gguf_opts(&src, &dst, &r, Some(&im), threads).unwrap();
            outs.push(std::fs::read(&dst).unwrap());
        }
        assert!(
            outs[0] == outs[1] && outs[0] == outs[2],
            "{ty}: bytes depend on the thread count"
        );
    }
    std::fs::remove_dir_all(dir).unwrap();
}

// ---- provenance and freshness ---------------------------------------------------

mod fresh {
    use super::*;
    use std::cell::Cell;

    struct FakeHub {
        listing: Result<RepoListing, String>,
        calls: Cell<u32>,
    }

    impl Hub for FakeHub {
        fn listing(&self, _repo: &str) -> Result<RepoListing, String> {
            self.calls.set(self.calls.get() + 1);
            self.listing.clone()
        }
    }

    fn input(role: &str, name: &str, sha: &str, hf: Option<(&str, &str)>) -> InputRecord {
        InputRecord {
            role: role.into(),
            file_name: name.into(),
            size: 10,
            sha256: sha.into(),
            hf: hf.map(|(r, p)| HfFile {
                repo: r.into(),
                path: p.into(),
            }),
        }
    }

    fn prov(inputs: Vec<InputRecord>) -> Provenance {
        Provenance {
            schema: 1,
            tool: "t".into(),
            ggml_parity_commit: GGML_PARITY_COMMIT.into(),
            created_unix: 0,
            inputs,
            output_file_name: "o.gguf".into(),
            output_size: 1,
            output_sha256: "x".into(),
        }
    }

    fn rf(path: &str, size: u64, sha: Option<&str>) -> RepoFile {
        RepoFile {
            path: path.into(),
            size,
            sha256: sha.map(String::from),
        }
    }

    #[test]
    fn hf_references_parse_strictly() {
        assert_eq!(
            HfFile::parse("unsloth/Qwen3-0.6B-GGUF:UD-IQ2_XXS/a.gguf").unwrap(),
            HfFile {
                repo: "unsloth/Qwen3-0.6B-GGUF".into(),
                path: "UD-IQ2_XXS/a.gguf".into()
            }
        );
        for bad in [
            "noslash:file",
            "a/b",
            "a/b:",
            "a/b:../x",
            "a/b:/etc/passwd",
            "a/b/c:f",
            "a b/c:f",
            "a/b?x=1:f",
        ] {
            assert!(HfFile::parse(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn freshness_classifies_each_input() {
        let hub = FakeHub {
            listing: Ok(RepoListing {
                commit: "abc123".into(),
                files: vec![
                    rf("same.gguf", 10, Some("aaa")),
                    rf("changed.gguf", 10, Some("new")),
                    rf("plain.txt", 10, None),
                    rf("plain-resized.txt", 99, None),
                ],
            }),
            calls: Cell::new(0),
        };
        let p = prov(vec![
            input("source", "same.gguf", "aaa", Some(("o/r", "same.gguf"))),
            input(
                "reference",
                "changed.gguf",
                "old",
                Some(("o/r", "changed.gguf")),
            ),
            input("imatrix", "gone.dat", "ccc", Some(("o/r", "gone.dat"))),
            input("x", "plain.txt", "ddd", Some(("o/r", "plain.txt"))),
            input(
                "y",
                "plain-resized.txt",
                "eee",
                Some(("o/r", "plain-resized.txt")),
            ),
            input("z", "local.bin", "fff", None),
        ]);
        let f = check_freshness(&p, &hub);
        assert_eq!(f[0].status, Status::UpToDate);
        assert_eq!(
            f[1].status,
            Status::Changed {
                recorded: "old".into(),
                current: "new".into()
            }
        );
        assert_eq!(f[2].status, Status::Missing);
        assert_eq!(f[3].status, Status::SizeOnly { same: true });
        assert_eq!(f[4].status, Status::SizeOnly { same: false });
        assert_eq!(f[5].status, Status::Untracked);
        assert_eq!(f[0].commit.as_deref(), Some("abc123"));
        let flags: Vec<bool> = f.iter().map(Finding::needs_attention).collect();
        assert_eq!(flags, [false, true, true, false, true, false]);
        assert_eq!(
            hub.calls.get(),
            1,
            "one listing per repo, however many inputs"
        );
    }

    #[test]
    fn an_unreachable_hub_is_reported_not_treated_as_fresh() {
        let hub = FakeHub {
            listing: Err("offline".into()),
            calls: Cell::new(0),
        };
        let p = prov(vec![input(
            "source",
            "a.gguf",
            "aaa",
            Some(("o/r", "a.gguf")),
        )]);
        let f = check_freshness(&p, &hub);
        assert_eq!(f[0].status, Status::Unreachable("offline".into()));
        assert!(f[0].needs_attention());
    }

    #[test]
    fn recording_hashes_real_files_and_round_trips() {
        let dir = scratch("prov");
        let (src, out) = (dir.join("src.bin"), dir.join("out.bin"));
        std::fs::write(&src, b"source bytes").unwrap();
        std::fs::write(&out, b"output bytes").unwrap();
        let hf = HfFile::parse("o/r:src.bin").unwrap();
        let p = Provenance::record(&[("source", src.as_path(), Some(hf))], &out).unwrap();
        assert_eq!(p.inputs[0].sha256, model_io::hash_data(b"source bytes"));
        assert_eq!(p.output_sha256, model_io::hash_data(b"output bytes"));
        assert_eq!((p.inputs[0].size, p.output_size), (12, 12));
        let path = dir.join("out.provenance.json");
        write_provenance(&path, &p).unwrap();
        assert_eq!(read_provenance(&path).unwrap(), p);
        // An unknown schema is refused rather than guessed at.
        let bad = p.to_json().replace("\"schema\": 1", "\"schema\": 99");
        assert!(Provenance::from_json(&bad).is_err());
        std::fs::remove_dir_all(dir).unwrap();
    }
}

// ---- progress and cancellation ----------------------------------------------------

#[test]
fn progress_is_monotonic_and_cancel_leaves_no_file() {
    use std::cell::RefCell;
    use std::sync::atomic::{AtomicBool, Ordering};
    let dir = scratch("ctl");
    let src = dir.join("in.gguf");
    let x = data(256 * 8, 31);
    let t = |name: &'static str| T {
        name,
        dims: vec![256, 8],
        ty: 0,
        bytes: f32_bytes(&x),
    };
    build(
        &src,
        &[
            t("blk.0.a.weight"),
            t("blk.0.b.weight"),
            t("blk.0.c.weight"),
        ],
    );
    let r = Recipe::from_json(r#"{"default":"q8_0"}"#).unwrap();

    let seen = RefCell::new(Vec::<(u64, u64, u64)>::new());
    let progress = |p: &Progress| {
        seen.borrow_mut()
            .push((p.tensors_done, p.bytes_done, p.bytes_total))
    };
    let ctl = Control {
        progress: Some(&progress),
        cancel: None,
    };
    let dst = dir.join("ok.gguf");
    quantize_gguf_ctl(&src, &dst, &r, None, 2, &ctl).unwrap();
    let s = seen.borrow();
    assert!(
        s.windows(2).all(|w| w[0].0 <= w[1].0 && w[0].1 <= w[1].1),
        "{s:?}"
    );
    let last = s.last().unwrap();
    assert_eq!((last.0, last.1), (3, last.2), "ends complete: {last:?}");

    // Raise the flag from the first progress report: the run must stop.
    let flag = AtomicBool::new(false);
    let stop = |_: &Progress| flag.store(true, Ordering::Relaxed);
    let ctl = Control {
        progress: Some(&stop),
        cancel: Some(&flag),
    };
    let dst2 = dir.join("cancelled.gguf");
    let e = quantize_gguf_ctl(&src, &dst2, &r, None, 2, &ctl).unwrap_err();
    assert!(matches!(e, QuantizeError::Cancelled), "{e}");
    assert!(!dst2.exists() && !dir.join("cancelled.gguf.partial").exists());
    std::fs::remove_dir_all(dir).unwrap();
}

// ---- the types ggml added after the K and IQ families -------------------------

/// Q1_0, Q2_0, TQ1_0, TQ2_0, MXFP4 and NVFP4 against ggml's own bytes on an
/// adversarial input (`in3.f32`: ties at exact halves, zero blocks, powers of
/// two, E2M1-exact values, tiny and huge magnitudes, constants, an
/// all-negative block). None of these types takes an imatrix.
mod golden3 {
    use super::*;

    const IN: &[u8] = include_bytes!("../../tests/fixtures/gguf_quantize/in3.f32");

    fn floats(b: &[u8]) -> Vec<f32> {
        b.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    fn check(target: Target, golden: &[u8]) {
        let x = floats(IN);
        let got = target.encode_rows(&x, 768, None);
        assert_eq!(got.len(), golden.len(), "{target:?} length");
        let (_, bb) = target.block();
        let diffs: Vec<(usize, usize)> = got
            .iter()
            .zip(golden)
            .enumerate()
            .filter(|(_, (a, b))| a != b)
            .map(|(i, _)| (i / bb, i % bb))
            .take(12)
            .collect();
        assert!(
            diffs.is_empty(),
            "{target:?} differs from ggml at (block, offset) {diffs:?}"
        );
    }

    macro_rules! parity {
        ($name:ident, $target:expr, $id:literal) => {
            #[test]
            fn $name() {
                check(
                    $target,
                    include_bytes!(concat!(
                        "../../tests/fixtures/gguf_quantize/g3_",
                        $id,
                        ".bin"
                    )),
                );
            }
        };
    }

    parity!(q1_0, Target::Q1_0, "41");
    parity!(q2_0, Target::Q2_0, "42");
    parity!(tq1_0, Target::Tq1_0, "34");
    parity!(tq2_0, Target::Tq2_0, "35");
    parity!(mxfp4, Target::Mxfp4, "39");
    parity!(nvfp4, Target::Nvfp4, "40");

    /// Unsloth's IQ1_XS, IQ1_XXS and IQ1_XXXS against the FORK's own quantizer
    /// (built from branch `iq1-narrow-grids-b10775` and run through
    /// `scripts/ggml_quantize_fixture.c`, with `im2.f32`). The adversarial
    /// input has exactly equal values inside blocks, so the libc `qsort` tie
    /// order decides the result: the goldens are macOS, so Apple's order.
    #[test]
    fn iq1_narrow_types_match_the_forks_quantizer() {
        use super::super::iq1::{set_tie_order, TieOrder};
        const IM: &[u8] = include_bytes!("../../tests/fixtures/gguf_quantize/im2.f32");
        set_tie_order(TieOrder::Apple);
        let (x, im) = (floats(IN), floats(IM));
        for (target, golden) in [
            (
                Target::Iq1Xs,
                include_bytes!("../../tests/fixtures/gguf_quantize/gi3_64.bin") as &[u8],
            ),
            (
                Target::Iq1Xxs,
                include_bytes!("../../tests/fixtures/gguf_quantize/gi3_65.bin"),
            ),
            (
                Target::Iq1Xxxs,
                include_bytes!("../../tests/fixtures/gguf_quantize/gi3_66.bin"),
            ),
        ] {
            let got = target.encode_rows(&x, 768, Some(&im));
            assert_eq!(got.len(), golden.len(), "{target:?} length");
            let (_, bb) = target.block();
            let diffs: Vec<(usize, usize)> = got
                .iter()
                .zip(golden)
                .enumerate()
                .filter(|(_, (a, b))| a != b)
                .map(|(i, _)| (i / bb, i % bb))
                .take(12)
                .collect();
            assert!(
                diffs.is_empty(),
                "{target:?} differs from the fork at (block, offset) {diffs:?}"
            );
        }
    }

    /// Every new target round-trips through its decoder within the type's own
    /// resolution, so a quantizer and decoder that agree on a wrong layout
    /// (the byte-parity tests above rule that out for the encoder, not the
    /// decoder pairing) still get caught.
    #[test]
    fn each_new_target_decodes_close_to_its_input() {
        let x = floats(IN);
        let row = &x[..768];
        for (target, tol) in [
            (Target::Q2_0, 1.01),
            (Target::Tq2_0, 1.01),
            (Target::Tq1_0, 1.01),
            (Target::Mxfp4, 0.35),
            (Target::Nvfp4, 0.25),
        ] {
            let bytes = target.encode_rows(row, 768, None);
            let back = match target {
                Target::Q2_0 => compute::dequantize_q2_0(&bytes, 768),
                Target::Tq2_0 => compute::dequantize_tq2_0(&bytes, 768),
                Target::Tq1_0 => compute::dequantize_tq1_0(&bytes, 768),
                Target::Mxfp4 => compute::dequantize_mxfp4(&bytes, 768),
                _ => compute::dequantize_nvfp4(&bytes, 768),
            };
            let amax = row.iter().fold(0f32, |m, v| m.max(v.abs()));
            let err = row
                .iter()
                .zip(&back)
                .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
            // The ternary and 2-bit types resolve only a handful of levels, so
            // their bound is a fraction of the block's own scale, here the
            // row's amax; the FP4 types are tighter.
            assert!(err <= tol * amax, "{target:?}: error {err} vs amax {amax}");
        }
    }
}
