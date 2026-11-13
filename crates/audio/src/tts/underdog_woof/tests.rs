//! Tests for the Underdog Woof checkpoint reader and tensor-hash gate.
//!
//! Witness-free tests (synthetic archives, exhaustive f16, hand-checked
//! `deq` cases, the digest format) always run. The hash gate needs the real
//! `woof_int8.npz` and skips with a message when it is absent. Point it at
//! the file with `TURBOSPARK_UNDERDOG_WOOF_NPZ`; the fallbacks are
//! `~/models/underdog-woof/woof_int8.npz` and `~/models/woof_int8.npz`.
//!
//! Provenance of the pinned data:
//! - `testdata/tts_underdog_woof/tensor_hash_reference.json` is the frozen
//!   reference from the `cowWhySo/Underdog-Woof-MLX` repo (60 dequantized
//!   speech tensors, SHA-256 over contiguous little-endian f32 bytes), made by
//!   `verify_source_integrity` in its `mac_validate.py`.
//! - The `deq` and digest constants below were produced with NumPy using a
//!   verbatim copy of upstream `deq` and `json.dumps(sort_keys=True,
//!   separators=(',', ':'))`, so they do not share code with the Rust.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde_json::{json, Value};
use turbospark_model_io::hash_file;

use super::npz::{f16_to_f32, Npz};
use super::weights::{
    deq, lin_w, tensor_record, tensor_records, tensor_values_digest, NamedTensor, SpeechWeights,
    TensorRecord,
};
use super::SOURCE_WEIGHTS_SHA256;

// ---------------------------------------------------------------- archive builder

fn npy_header(descr: &str, shape: &[usize]) -> Vec<u8> {
    let shape_text = match shape.len() {
        0 => "()".to_string(),
        1 => format!("({},)", shape[0]),
        _ => format!(
            "({})",
            shape
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        ),
    };
    let mut header =
        format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape_text}, }}");
    // Pad so magic + version + length + header ends on a 64-byte boundary.
    while (10 + header.len() + 1) % 64 != 0 {
        header.push(' ');
    }
    header.push('\n');
    let mut out = b"\x93NUMPY\x01\x00".to_vec();
    out.extend_from_slice(&(header.len() as u16).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    out
}

fn npy(descr: &str, shape: &[usize], data: &[u8]) -> Vec<u8> {
    let mut out = npy_header(descr, shape);
    out.extend_from_slice(data);
    out
}

fn npy_f32(shape: &[usize], v: &[f32]) -> Vec<u8> {
    let data: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    npy("<f4", shape, &data)
}

fn npy_i8(shape: &[usize], v: &[i8]) -> Vec<u8> {
    let data: Vec<u8> = v.iter().map(|&x| x as u8).collect();
    npy("|i1", shape, &data)
}

fn npy_str(text: &str) -> Vec<u8> {
    let chars: Vec<u32> = text.chars().map(u32::from).collect();
    let data: Vec<u8> = chars.iter().flat_map(|c| c.to_le_bytes()).collect();
    npy(&format!("<U{}", chars.len()), &[], &data)
}

/// A ZIP archive of stored members. `method` lets a test claim compression.
fn build_zip(members: &[(&str, Vec<u8>)], method: u16) -> Vec<u8> {
    let mut out = Vec::new();
    let mut central = Vec::new();
    for (name, data) in members {
        let offset = out.len() as u32;
        let name = format!("{name}.npy");
        out.extend_from_slice(&0x0403_4b50u32.to_le_bytes());
        out.extend_from_slice(&[20, 0, 0, 0]);
        out.extend_from_slice(&method.to_le_bytes());
        out.extend_from_slice(&[0, 0, 0, 0, 0, 0, 0, 0]); // time, date, crc
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(&(name.len() as u16).to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(name.as_bytes());
        out.extend_from_slice(data);

        central.extend_from_slice(&0x0201_4b50u32.to_le_bytes());
        central.extend_from_slice(&[20, 0, 20, 0, 0, 0]);
        central.extend_from_slice(&method.to_le_bytes());
        central.extend_from_slice(&[0, 0, 0, 0, 0, 0, 0, 0]);
        central.extend_from_slice(&(data.len() as u32).to_le_bytes());
        central.extend_from_slice(&(data.len() as u32).to_le_bytes());
        central.extend_from_slice(&(name.len() as u16).to_le_bytes());
        central.extend_from_slice(&[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        central.extend_from_slice(&offset.to_le_bytes());
        central.extend_from_slice(name.as_bytes());
    }
    let cen_off = out.len() as u32;
    out.extend_from_slice(&central);
    out.extend_from_slice(&0x0605_4b50u32.to_le_bytes());
    out.extend_from_slice(&[0, 0, 0, 0]);
    out.extend_from_slice(&(members.len() as u16).to_le_bytes());
    out.extend_from_slice(&(members.len() as u16).to_le_bytes());
    out.extend_from_slice(&(central.len() as u32).to_le_bytes());
    out.extend_from_slice(&cen_off.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out
}

fn npz(members: &[(&str, Vec<u8>)]) -> Npz {
    Npz::from_bytes(build_zip(members, 0)).expect("synthetic archive")
}

// ---------------------------------------------------------------- reader

#[test]
fn archive_round_trips_every_dtype_and_the_string_tree() {
    let f16: Vec<u8> = [0x3C00u16, 0xC000, 0x0001]
        .iter()
        .flat_map(|h| h.to_le_bytes())
        .collect();
    let tree = r#"{"a": {"@": "x"}, "text": "café"}"#;
    let archive = npz(&[
        ("__tree__", npy_str(tree)),
        ("x", npy_f32(&[2, 2], &[1.0, -2.0, 0.5, 4.0])),
        ("q", npy_i8(&[3], &[-128, 0, 127])),
        ("h", npy("<f2", &[3], &f16)),
    ]);
    let x = archive.array("x").unwrap();
    assert_eq!(x.shape, vec![2, 2]);
    assert_eq!(x.to_f32(), vec![1.0, -2.0, 0.5, 4.0]);
    assert_eq!(
        archive.array("q").unwrap().to_f32(),
        vec![-128.0, 0.0, 127.0]
    );
    assert_eq!(
        archive.array("h").unwrap().to_f32(),
        vec![1.0, -2.0, 2f32.powi(-24)]
    );
    let parsed = archive.tree().unwrap();
    assert_eq!(parsed["text"], "caf\u{e9}");
    assert!(archive.array("missing").is_err());
}

#[test]
fn archive_refuses_compressed_zip64_truncated_and_unknown_dtype() {
    let members = [("x", npy_f32(&[1], &[1.0]))];
    let deflated = build_zip(&members, 8);
    let err = Npz::from_bytes(deflated)
        .err()
        .expect("deflate refused")
        .to_string();
    assert!(err.contains("compressed"), "{err}");

    let good = build_zip(&members, 0);
    assert!(Npz::from_bytes(good[..good.len() / 2].to_vec()).is_err());
    assert!(Npz::from_bytes(b"not a zip file at all".to_vec()).is_err());

    let f8 = npy("<f8", &[1], &1f64.to_le_bytes());
    let archive = npz(&[("d", f8)]);
    let err = archive.array("d").expect_err("f64 refused").to_string();
    assert!(err.contains("unsupported dtype <f8"), "{err}");

    // A payload whose length disagrees with its shape is refused, not read short.
    let short = npy("<f4", &[4], &[0u8; 8]);
    assert!(npz(&[("s", short)]).array("s").is_err());
}

#[test]
fn f16_widening_matches_the_exact_formula_for_all_65536_patterns() {
    for h in 0..=u16::MAX {
        let sign = if h & 0x8000 != 0 { -1.0f64 } else { 1.0 };
        let exp = i32::from((h >> 10) & 0x1F);
        let frac = f64::from(h & 0x3FF);
        let got = f16_to_f32(h);
        if exp == 0x1F {
            if frac == 0.0 {
                assert_eq!(got, sign as f32 * f32::INFINITY, "{h:#06x}");
            } else {
                assert!(got.is_nan(), "{h:#06x}");
            }
            continue;
        }
        let expected = if exp == 0 {
            sign * frac * 2f64.powi(-24)
        } else {
            sign * (1.0 + frac / 1024.0) * 2f64.powi(exp - 15)
        };
        // Every binary16 value is exactly representable in f32, and signed zero is kept.
        assert_eq!(f64::from(got), expected, "{h:#06x}");
        assert_eq!(got.is_sign_negative(), h & 0x8000 != 0, "{h:#06x}");
    }
    assert_eq!(f16_to_f32(0x7BFF), 65504.0);
    assert_eq!(f16_to_f32(0x0400), 2f32.powi(-14));
}

// ---------------------------------------------------------------- dequantization

#[test]
fn lin_w_multiplies_each_row_by_its_scale_and_widens_the_bias() {
    let archive = npz(&[
        ("q", npy_i8(&[2, 3], &[1, -2, 3, 4, 5, -6])),
        ("s", npy_f32(&[2], &[0.5, 0.25])),
        ("b", npy_f32(&[2], &[10.0, 20.0])),
        ("w", npy_f32(&[1, 2], &[7.0, 8.0])),
    ]);
    let entry = json!({"q": {"@": "q"}, "s": {"@": "s"}, "b": {"@": "b"}});
    let l = lin_w(&archive, &entry, "t").unwrap();
    assert_eq!((l.rows, l.cols), (2, 3));
    assert_eq!(l.w, vec![0.5, -1.0, 1.5, 1.0, 1.25, -1.5]);
    assert_eq!(l.b.as_deref(), Some(&[10.0, 20.0][..]));

    // No q: the plain `w` array is used as is, and a missing bias stays None.
    let plain = lin_w(&archive, &json!({"w": {"@": "w"}}), "t").unwrap();
    assert_eq!(plain.w, vec![7.0, 8.0]);
    assert!(plain.b.is_none());

    // A scale vector with the wrong length (6 values for 2 rows) is refused,
    // and so is a bias whose length is not the row count.
    assert!(lin_w(&archive, &json!({"q": {"@": "q"}, "s": {"@": "q"}}), "t").is_err());
    assert!(lin_w(&archive, &json!({"w": {"@": "w"}, "b": {"@": "q"}}), "t").is_err());
}

fn grouped(q: &[i8], q_shape: &[usize], s: &[f32], s_shape: &[usize]) -> Npz {
    npz(&[("q8", npy_i8(q_shape, q)), ("s", npy_f32(s_shape, s))])
}

fn grouped_node(shape: &[usize], axis: usize, g: usize) -> Value {
    json!({"q8": {"@": "q8"}, "s": {"@": "s"}, "shape": shape, "axis": axis, "g": g})
}

#[test]
fn deq_matches_the_numpy_reference_on_hand_checked_cases() {
    // A: shape (2, 3), axis 0, ragged last group (g = 2 over 3 columns).
    let a = grouped(
        &[1, 2, 3, 4, 5, 6],
        &[2, 3],
        &[0.5, 2.0, 1.0, 0.25],
        &[2, 2],
    );
    let r = deq(&a, &grouped_node(&[2, 3], 0, 2), "A").unwrap();
    assert_eq!(r.shape, vec![2, 3]);
    assert_eq!(r.data, vec![0.5, 1.0, 6.0, 4.0, 5.0, 1.5]);

    // B: shape (2, 2, 3), axis 1, so the quantized rows are the middle axis.
    let q: Vec<i8> = (-6..6).collect();
    let b = grouped(&q, &[2, 6], &[0.5, 0.25, 2.0, 1.0], &[2, 2]);
    let r = deq(&b, &grouped_node(&[2, 2, 3], 1, 3), "B").unwrap();
    assert_eq!(
        r.data,
        vec![-3.0, -2.5, -2.0, 0.0, 2.0, 4.0, -0.75, -0.5, -0.25, 3.0, 4.0, 5.0]
    );

    // C: shape (2, 4, 3), axis 2, the last axis.
    let q: Vec<i8> = (0..24).map(|i| (i * 5 - 60) as i8).collect();
    let c = grouped(&q, &[3, 8], &[0.5, 1.5, 0.25, 4.0, 1.0, 0.125], &[3, 2]);
    let r = deq(&c, &grouped_node(&[2, 4, 3], 2, 4), "C").unwrap();
    assert_eq!(
        r.data,
        vec![
            -30.0, -5.0, 20.0, -27.5, -3.75, 25.0, -25.0, -2.5, 30.0, -22.5, -1.25, 35.0, -60.0,
            0.0, 5.0, -52.5, 20.0, 5.625, -45.0, 40.0, 6.25, -37.5, 60.0, 6.875
        ]
    );
}

#[test]
fn deq_widens_plain_arrays_and_refuses_inconsistent_geometry() {
    let a = grouped(
        &[1, 2, 3, 4, 5, 6],
        &[2, 3],
        &[0.5, 2.0, 1.0, 0.25],
        &[2, 2],
    );
    // Wrong axis length, group count that cannot cover the columns, zero group.
    assert!(deq(&a, &grouped_node(&[3, 2], 0, 2), "x").is_err());
    assert!(deq(&a, &grouped_node(&[2, 3], 0, 1), "x").is_err());
    assert!(deq(&a, &grouped_node(&[2, 3], 0, 0), "x").is_err());
    assert!(deq(&a, &grouped_node(&[2, 3], 5, 2), "x").is_err());
    let plain = npz(&[("p", npy_f32(&[2], &[1.5, -2.5]))]);
    assert_eq!(
        deq(&plain, &json!({"@": "p"}), "p").unwrap().data,
        vec![1.5, -2.5]
    );
}

// ---------------------------------------------------------------- hashes

#[test]
fn tensor_record_and_digest_match_numpy_hashlib() {
    let a = NamedTensor {
        name: "x.weight".into(),
        shape: vec![2, 2],
        data: &[1.0, 2.0, 3.0, 4.0],
    };
    let b = NamedTensor {
        name: "x.bias".into(),
        shape: vec![2],
        data: &[0.5, -0.25],
    };
    let ra = tensor_record(&a);
    assert_eq!(ra.bytes, 16);
    assert_eq!(
        ra.sha256,
        "ad73b9acd6e4a74b2f5bb5386658ce3bb146cd040a1867646ab3b973fb6632b1"
    );
    let rb = tensor_record(&b);
    assert_eq!(
        rb.sha256,
        "53c0f587cb612b4908eff791773b2b8cd805eca616e5877a07b5a744c02e9c46"
    );
    let records: BTreeMap<String, TensorRecord> =
        [("x.weight".to_string(), ra), ("x.bias".to_string(), rb)]
            .into_iter()
            .collect();
    assert_eq!(
        tensor_values_digest(&records),
        "d0a6a508445cc9edcf674e59f362c9bd739807d55788b501fd1fba6c931715e1"
    );
}

struct Frozen {
    source_sha256: String,
    digest: String,
    tensors: BTreeMap<String, TensorRecord>,
}

fn frozen() -> Frozen {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("testdata/tts_underdog_woof/tensor_hash_reference.json");
    let v: Value = serde_json::from_str(&std::fs::read_to_string(path).expect("fixture")).unwrap();
    let tensors = v["tensors"]
        .as_object()
        .expect("tensors")
        .iter()
        .map(|(name, t)| {
            assert_eq!(t["dtype"], "float32", "{name}");
            let shape = t["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|d| d.as_u64().unwrap() as usize)
                .collect();
            let rec = TensorRecord {
                shape,
                bytes: t["bytes"].as_u64().unwrap() as usize,
                sha256: t["sha256"].as_str().unwrap().to_string(),
            };
            (name.clone(), rec)
        })
        .collect();
    Frozen {
        source_sha256: v["source_weight_sha256"].as_str().unwrap().to_string(),
        digest: v["tensor_values_digest"].as_str().unwrap().to_string(),
        tensors,
    }
}

/// Names whose record differs from the frozen one, or that exist on one side only.
fn changed_names(
    got: &BTreeMap<String, TensorRecord>,
    want: &BTreeMap<String, TensorRecord>,
) -> Vec<String> {
    let mut names: Vec<&String> = got.keys().chain(want.keys()).collect();
    names.sort();
    names.dedup();
    names
        .into_iter()
        .filter(|n| got.get(*n) != want.get(*n))
        .cloned()
        .collect()
}

#[test]
fn comparison_reports_exactly_the_changed_missing_and_extra_names() {
    let rec = |s: &str| TensorRecord {
        shape: vec![1],
        bytes: 4,
        sha256: s.into(),
    };
    let want: BTreeMap<_, _> = [("a".to_string(), rec("1")), ("b".to_string(), rec("2"))]
        .into_iter()
        .collect();
    assert!(changed_names(&want, &want).is_empty());
    let mut got = want.clone();
    got.insert("a".into(), rec("changed"));
    got.remove("b");
    got.insert("c".into(), rec("3"));
    assert_eq!(changed_names(&got, &want), vec!["a", "b", "c"]);
}

#[test]
fn frozen_reference_is_the_expected_60_tensor_set() {
    let f = frozen();
    assert_eq!(f.tensors.len(), 60);
    assert_eq!(f.source_sha256, SOURCE_WEIGHTS_SHA256);
    // 8 layers x 4 bias-free linears + ce/ada/lat_in/h_inp/h_out (+bias) + 4 blocks x 2 (+bias) + 2.
    assert_eq!(32 + 2 * 5 + 2 * 8 + 2, 60);
    assert!(f.tensors.contains_key("layers.7.3.weight"));
    assert!(!f.tensors.contains_key("layers.0.0.bias"));
    // The frozen digest is the digest of its own records, so the format is pinned
    // against the real data and not only against the two-tensor example.
    assert_eq!(tensor_values_digest(&f.tensors), f.digest);
}

// ---------------------------------------------------------------- the witness gate

fn installed_witness() -> Option<PathBuf> {
    let mut candidates = Vec::new();
    if let Some(p) = std::env::var_os("TURBOSPARK_UNDERDOG_WOOF_NPZ") {
        candidates.push(PathBuf::from(p));
    }
    if let Some(home) = std::env::var_os("HOME") {
        let home = PathBuf::from(home);
        candidates.push(home.join("models/underdog-woof/woof_int8.npz"));
        candidates.push(home.join("models/woof_int8.npz"));
    }
    candidates.into_iter().find(|p| p.is_file())
}

#[test]
fn installed_checkpoint_reproduces_all_60_frozen_tensor_hashes() {
    let Some(path) = installed_witness() else {
        eprintln!("skipping: set TURBOSPARK_UNDERDOG_WOOF_NPZ to woof_int8.npz");
        return;
    };
    let f = frozen();

    // 1. The file is the pinned upstream file.
    assert_eq!(
        hash_file(&path, 1 << 20).expect("hash"),
        SOURCE_WEIGHTS_SHA256
    );

    // 2. Every dequantized tensor hashes to the frozen value, and so does the whole set.
    let weights = SpeechWeights::load(&Npz::open(&path).unwrap()).expect("load");
    let c = &weights.cfg;
    assert_eq!(
        (c.layers, c.d, c.heads, c.ff, c.hw, c.depth, c.latent),
        (8, 448, 7, 1216, 320, 4, 32)
    );
    let got = tensor_records(&weights);
    assert_eq!(got.len(), 60);
    assert!(
        changed_names(&got, &f.tensors).is_empty(),
        "{:?}",
        changed_names(&got, &f.tensors)
    );
    assert_eq!(tensor_values_digest(&got), f.digest);

    // 3. Matrix-vector weights: 45 matrices, 22,747,584 values.
    let matrices = weights
        .layers
        .iter()
        .flatten()
        .chain(weights.hblocks.iter().flatten())
        .chain([
            &weights.ce,
            &weights.ada,
            &weights.lat_in,
            &weights.h_inp,
            &weights.h_out,
        ]);
    let (count, params) = matrices.fold((0, 0), |(n, p), l| (n + 1, p + l.rows * l.cols));
    assert_eq!((count, params), (45, 22_747_584));
}

#[test]
fn one_corrupted_int8_weight_is_caught_and_attributed_to_its_tensor() {
    let Some(path) = installed_witness() else {
        eprintln!("skipping: set TURBOSPARK_UNDERDOG_WOOF_NPZ to woof_int8.npz");
        return;
    };
    let f = frozen();
    let archive = Npz::open(&path).unwrap();
    // The payload of speech.layers.3.2.q starts with the .npy header. Corrupt the
    // last element: guaranteed past the header, inside the int8 data.
    let range = archive.member_range("speech.layers.3.2.q").expect("member");
    let at = range.end - 1;
    let mut bytes = archive.into_bytes();
    let before = bytes[at];
    bytes[at] = before.wrapping_add(1);
    assert_ne!(bytes[at], before, "the mutation must have applied");

    let weights = SpeechWeights::load(&Npz::from_bytes(bytes).unwrap()).expect("load");
    let got = tensor_records(&weights);
    assert_eq!(changed_names(&got, &f.tensors), vec!["layers.3.2.weight"]);
    assert_ne!(tensor_values_digest(&got), f.digest);
}
