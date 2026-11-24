//! Inkling 2 intake tests. The config test reads the real Mini `config.json`
//! (fixture, Apache-2.0); the install tests write a tiny checkpoint in the
//! HF layout and read the install back through the loaders the runtime uses.

use super::*;
use crate::synthetic_tensors::{assemble_safetensors, Tensor};
use compute::inkling2::{dequant_mxfp8_row, dequant_nvfp4_row, dequantize_mxfp8_blocks};
use model_io::inkling2_config::Inkling2Config;

const MINI_CONFIG: &str = include_str!("../tests/fixtures/inkling2/config.json");

/// The headline parser test: the real config yields exactly the baseline
/// and the sidecar, so a key wired to the wrong field cannot pass.
#[test]
fn the_real_mini_config_parses_to_the_baseline_and_sidecar() {
    let (arch, side) = parse_inkling2_config(MINI_CONFIG).unwrap();
    assert_eq!(arch, model_io::inkling2_mini());
    assert_eq!(side, Inkling2Config::mini());
}

#[test]
fn unimplemented_config_values_are_refused() {
    let mut v: Value = serde_json::from_str(MINI_CONFIG).unwrap();
    let refuse = |v: &Value| parse_inkling2_config(&v.to_string()).is_err();
    for (key, bad) in [
        ("q_bias", serde_json::json!(true)),
        ("n_shared_experts", serde_json::json!(1)),
        ("use_value_norm_no_gamma", serde_json::json!(false)),
        ("swa_num_key_value_heads", serde_json::json!(4)),
    ] {
        let mut c = v.clone();
        c["text_config"][key] = bad;
        assert!(refuse(&c), "{key}");
    }
    v["text_config"]["loop_config"]["is_varloop"] = serde_json::json!(true);
    assert!(refuse(&v));
    let mut c: Value = serde_json::from_str(MINI_CONFIG).unwrap();
    c["model_type"] = serde_json::json!("inkling2");
    assert!(refuse(&c));
}

// --- A tiny checkpoint in the HF layout -----------------------------------

const HIDDEN: usize = 128;
const FFN: usize = 64;
const EXPERTS: usize = 4;
const LAYERS: usize = 6;
const TAPS: usize = 4;
const KVD: usize = 32;

fn tiny_config() -> String {
    let mut v: Value = serde_json::from_str(MINI_CONFIG).unwrap();
    let t = &mut v["text_config"];
    t["hidden_size"] = HIDDEN.into();
    t["intermediate_size"] = FFN.into();
    t["num_hidden_layers"] = LAYERS.into();
    t["n_routed_experts"] = EXPERTS.into();
    t["num_experts_per_tok"] = 2.into();
    t["n_layers_per_attn_res_block"] = 2.into();
    t["local_layer_ids"] = serde_json::json!([0, 2, 4]);
    t["loop_config"]["core_start"] = 2.into();
    t["loop_config"]["core_end"] = 4.into();
    t["loop_config"]["num_passes"] = 2.into();
    t["vocab_size"] = 64.into();
    t["unpadded_vocab_size"] = 60.into();
    t["sliding_window_size"] = 16.into();
    t["rel_extent"] = 16.into();
    v["eos_token_id"] = 59.into();
    v.to_string()
}

struct Lcg(u32);
impl Lcg {
    fn byte(&mut self) -> u8 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (self.0 >> 24) as u8
    }
}

fn t(name: &str, dtype: &'static str, shape: &[usize], bytes: Vec<u8>) -> Tensor {
    Tensor {
        name: name.into(),
        dtype,
        shape: shape.iter().map(|&d| d as u64).collect(),
        bytes,
    }
}

fn f32s(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Layers 0-3 NVFP4, 4-5 MXFP8, with a full-attention `static_w` on layer 1,
/// a router per layer, and a few other resident tensors.
fn tiny_checkpoint(mut patch: impl FnMut(&mut Vec<Tensor>)) -> Vec<u8> {
    let mut r = Lcg(42);
    let mut v = Vec::new();
    let bytes = |r: &mut Lcg, n: usize, f: &dyn Fn(u8) -> u8| {
        (0..n).map(|_| f(r.byte())).collect::<Vec<u8>>()
    };
    v.push(t(
        "model.llm.embed.weight",
        "BF16",
        &[64, HIDDEN],
        bytes(&mut r, 64 * HIDDEN * 2, &|b| b),
    ));
    v.push(t(
        "model.llm.attn_res.output_query",
        "F32",
        &[HIDDEN],
        f32s(&vec![0.25; HIDDEN]),
    ));
    v.push(t("model.visual.backbone.dropme", "BF16", &[4], vec![0; 8]));
    // static_w[lag, ch] = lag * 32 + ch: below 128, so exact in BF16.
    let sw: Vec<u8> = (0..TAPS * KVD)
        .flat_map(|i| {
            let (lag, ch) = (i / KVD, i % KVD);
            let v = (lag * 32 + ch) as f32;
            ((v.to_bits() >> 16) as u16).to_le_bytes()
        })
        .collect();
    v.push(t(
        "model.llm.layers.1.attn.k_sconv.static_w",
        "BF16",
        &[TAPS, KVD],
        sw,
    ));
    for l in 0..LAYERS {
        let p = format!("model.llm.layers.{l}.mlp");
        v.push(t(
            &format!("{p}.gate.weight"),
            "BF16",
            &[EXPERTS, HIDDEN],
            bytes(&mut r, EXPERTS * HIDDEN * 2, &|b| b & 0x3F),
        ));
        v.push(t(
            &format!("{p}.gate.bias"),
            "F32",
            &[EXPERTS],
            f32s(&[0.1, 0.2, 0.3, 0.4]),
        ));
        let x = format!("{p}.experts");
        if l < 4 {
            let sc = |b: u8| 0x20 | (b & 0x0F);
            v.push(t(
                &format!("{x}.w13_weight"),
                "U8",
                &[EXPERTS, 2 * FFN, HIDDEN / 2],
                bytes(&mut r, EXPERTS * 2 * FFN * HIDDEN / 2, &|b| b),
            ));
            v.push(t(
                &format!("{x}.w13_weight_scale"),
                "F8_E4M3",
                &[EXPERTS, 2 * FFN, HIDDEN / 16],
                bytes(&mut r, EXPERTS * 2 * FFN * HIDDEN / 16, &sc),
            ));
            v.push(t(
                &format!("{x}.w2_weight"),
                "U8",
                &[EXPERTS, HIDDEN, FFN / 2],
                bytes(&mut r, EXPERTS * HIDDEN * FFN / 2, &|b| b),
            ));
            v.push(t(
                &format!("{x}.w2_weight_scale"),
                "F8_E4M3",
                &[EXPERTS, HIDDEN, FFN / 16],
                bytes(&mut r, EXPERTS * HIDDEN * FFN / 16, &sc),
            ));
            // w13 and w2 get different amax so a record that swapped them fails.
            for (n, amax) in [
                ("w13", [1.5f32, 2.25, 0.75, 3.0]),
                ("w2", [0.5, 1.25, 4.0, 2.0]),
            ] {
                v.push(t(
                    &format!("{x}.{n}_weight_amax"),
                    "F32",
                    &[EXPERTS],
                    f32s(&amax),
                ));
                v.push(t(
                    &format!("{x}.{n}_weight_outer_scale_denominator"),
                    "F32",
                    &[1],
                    f32s(&[2688.0]),
                ));
            }
        } else {
            let code = |b: u8| (b & 0x80) | (0x18 + (b & 0x3F) % 40);
            v.push(t(
                &format!("{x}.w13_weight"),
                "F8_E4M3",
                &[EXPERTS, 2 * FFN, HIDDEN],
                bytes(&mut r, EXPERTS * 2 * FFN * HIDDEN, &code),
            ));
            v.push(t(
                &format!("{x}.w13_weight_scale"),
                "U8",
                &[EXPERTS, 2 * FFN, HIDDEN / 32],
                bytes(&mut r, EXPERTS * 2 * FFN * HIDDEN / 32, &|b| 120 + b % 8),
            ));
            v.push(t(
                &format!("{x}.w2_weight"),
                "F8_E4M3",
                &[EXPERTS, HIDDEN, FFN],
                bytes(&mut r, EXPERTS * HIDDEN * FFN, &code),
            ));
            v.push(t(
                &format!("{x}.w2_weight_scale"),
                "U8",
                &[EXPERTS, HIDDEN, FFN / 32],
                bytes(&mut r, EXPERTS * HIDDEN * FFN / 32, &|b| 120 + b % 8),
            ));
        }
    }
    patch(&mut v);
    assemble_safetensors(&v)
}

fn scratch(tag: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("ts-inkling2-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn install(
    tag: &str,
    patch: impl FnMut(&mut Vec<Tensor>),
) -> Result<(std::path::PathBuf, Inkling2InstallReport), Inkling2RepackError> {
    let root = scratch(tag);
    let ckpt = root.join("model.safetensors");
    std::fs::write(&ckpt, tiny_checkpoint(patch)).unwrap();
    let src = SafetensorsFile::open(&ckpt).unwrap();
    let out = root.join("install");
    let report = write_inkling2_install(&src, &tiny_config(), &out, "tiny-inkling2", |_| {})?;
    Ok((out, report))
}

/// The bytes of expert `e`'s sub-tensor `role` in layer `l`.
fn blob_part(dir: &Path, l: usize, e: usize, role: &str) -> (String, Vec<u8>) {
    let layout = model_io::load_packed_experts_layout(dir, 1 << 26).expect("layout loads");
    let layer = &layout.layers[l];
    let entry = &layer.experts[e];
    let sub = &entry.sub_tensors[role];
    let file = std::fs::read(dir.join("packed_experts").join(&layer.file)).unwrap();
    let start = (entry.offset + sub.offset) as usize;
    (
        sub.dtype.clone(),
        file[start..start + sub.size as usize].to_vec(),
    )
}

#[test]
fn a_tiny_install_round_trips_every_expert_layout_and_convention() {
    let (dir, report) = install("roundtrip", |_| {}).unwrap();
    assert_eq!(
        (report.layers, report.nvfp4_layers, report.mxfp8_layers),
        (LAYERS, 4, 2)
    );

    // The manifest validates through the loader, quant section included.
    let (arch, side) = parse_inkling2_config(&tiny_config()).unwrap();
    model_io::load_manifest(&dir, &arch, 1 << 26).expect("manifest validates");
    assert_eq!(Inkling2Config::load(&dir, &arch).unwrap(), side);

    let src_bytes = tiny_checkpoint(|_| {});
    let path = dir.parent().unwrap().join("model.safetensors");
    assert_eq!(std::fs::read(&path).unwrap(), src_bytes);
    let src = SafetensorsFile::open(&path).unwrap();
    let raw = |n: &str| src.raw_bytes(n).unwrap().to_vec();

    // NVFP4 layer 2, expert 1: every gate, up and down row decodes to the HF
    // reference times the outer scale, and the record holds amax / den.
    let (l, e) = (2usize, 1usize);
    let x = format!("model.llm.layers.{l}.mlp.experts");
    let (w13, s13, w2, s2) = (
        raw(&format!("{x}.w13_weight")),
        raw(&format!("{x}.w13_weight_scale")),
        raw(&format!("{x}.w2_weight")),
        raw(&format!("{x}.w2_weight_scale")),
    );
    let (_, rec) = blob_part(&dir, l, e, "outer");
    let outer: Vec<f32> = rec
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert_eq!(
        outer,
        vec![2.25 / 2688.0, 2.25 / 2688.0, 1.25 / 2688.0, 0.0]
    );
    let check_nvfp4 = |role: &str,
                       rows: usize,
                       cols: usize,
                       w: &[u8],
                       s: &[u8],
                       pick: &dyn Fn(usize) -> usize| {
        let (dt, bytes) = blob_part(&dir, l, e, role);
        assert_eq!(dt, INKLING2_NVFP4_DTYPE);
        let got = compute::dequantize_nvfp4(&bytes, rows * cols);
        let src_rows = w.len() / EXPERTS / (cols / 2);
        for k in 0..rows {
            let r = e * src_rows + pick(k);
            let mut want = vec![0.0f32; cols];
            dequant_nvfp4_row(
                &w[r * cols / 2..(r + 1) * cols / 2],
                &s[r * cols / 16..(r + 1) * cols / 16],
                1.0,
                &mut want,
            );
            assert_eq!(&got[k * cols..(k + 1) * cols], &want[..], "{role} row {k}");
        }
    };
    check_nvfp4("gate", FFN, HIDDEN, &w13, &s13, &|k| 2 * k);
    check_nvfp4("up", FFN, HIDDEN, &w13, &s13, &|k| 2 * k + 1);
    check_nvfp4("down", HIDDEN, FFN, &w2, &s2, &|k| k);

    // MXFP8 layer 5, expert 3: bit-exact, and the record is all ones.
    let (l, e) = (5usize, 3usize);
    let x = format!("model.llm.layers.{l}.mlp.experts");
    let (w13, s13) = (
        raw(&format!("{x}.w13_weight")),
        raw(&format!("{x}.w13_weight_scale")),
    );
    let (_, rec) = blob_part(&dir, l, e, "outer");
    assert_eq!(rec, f32s(&[1.0, 1.0, 1.0, 0.0]));
    let (dt, gate) = blob_part(&dir, l, e, "gate");
    assert_eq!(dt, INKLING2_MXFP8_DTYPE);
    let got = dequantize_mxfp8_blocks(&gate, FFN * HIDDEN);
    for k in 0..FFN {
        let r = e * 2 * FFN + 2 * k;
        let mut want = vec![0.0f32; HIDDEN];
        dequant_mxfp8_row(
            &w13[r * HIDDEN..(r + 1) * HIDDEN],
            &s13[r * HIDDEN / 32..(r + 1) * HIDDEN / 32],
            &mut want,
        );
        for (g, w) in got[k * HIDDEN..(k + 1) * HIDDEN].iter().zip(&want) {
            assert_eq!(g.to_bits(), w.to_bits(), "mxfp8 gate row {k}");
        }
    }

    // Resident: no experts, no vision; the router widened exactly; static_w
    // rewritten to [ch, 1, taps] with tap taps-1 on the current row.
    let entries = crate::resident_reader::read_resident_entries(&dir).unwrap();
    let by_name = |n: &str| {
        entries.iter().find_map(|s| match s {
            ResidentEntrySpec::Raw(r) if r.name == n => Some(r.clone()),
            _ => None,
        })
    };
    assert!(entries.iter().all(|s| match s {
        ResidentEntrySpec::Raw(r) =>
            !r.name.contains(".mlp.experts.") && !r.name.contains("visual"),
        _ => false,
    }));
    let router = by_name("model.llm.layers.3.mlp.gate.weight").expect("router");
    assert_eq!(router.dtype, DTYPE_FP32);
    let src_router = raw("model.llm.layers.3.mlp.gate.weight");
    for (i, c) in router.bytes.chunks_exact(4).enumerate() {
        let want = u16::from_le_bytes([src_router[2 * i], src_router[2 * i + 1]]);
        assert_eq!(
            f32::from_le_bytes([c[0], c[1], c[2], c[3]]).to_bits(),
            u32::from(want) << 16
        );
    }
    assert!(by_name("model.llm.layers.1.attn.k_sconv.static_w").is_none());
    let conv = by_name("model.llm.layers.1.attn.k_sconv.weight").expect("rewritten conv");
    assert_eq!(conv.shape, (KVD as u32, 1, TAPS as u32, 0));
    for ch in 0..KVD {
        for j in 0..TAPS {
            let b = &conv.bytes[(ch * TAPS + j) * 2..(ch * TAPS + j) * 2 + 2];
            let v = f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16);
            assert_eq!(v, ((TAPS - 1 - j) * 32 + ch) as f32, "ch {ch} tap {j}");
        }
    }
    let q = by_name("model.llm.attn_res.output_query").expect("F32 query stays F32");
    assert_eq!(q.dtype, DTYPE_FP32);
}

#[test]
fn mxfp8_with_an_outer_scale_and_nan_scales_are_refused() {
    let extra = install("mxfp8-amax", |v| {
        v.push(t(
            "model.llm.layers.5.mlp.experts.w13_weight_amax",
            "F32",
            &[EXPERTS],
            f32s(&[1.0; EXPERTS]),
        ));
    });
    assert!(extra.unwrap_err().0.contains("must not carry"));

    let nan = install("nan-scale", |v| {
        let s = v
            .iter_mut()
            .find(|t| t.name == "model.llm.layers.0.mlp.experts.w2_weight_scale")
            .unwrap();
        s.bytes[7] = 0x7F;
    });
    assert!(nan.unwrap_err().0.contains("NaN"));
}
