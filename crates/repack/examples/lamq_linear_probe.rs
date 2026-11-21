//! Real-data check of the LAMQ input rotation and the hashed codes: runs each
//! coded attention linear of one layer through `compute::lamq::LamqLinear`
//! and through the BF16 HF install's matrix on the same input, and prints the
//! output cosine. Complements `lamq_embed_probe` (output rotation only).
//!
//! cargo run --release -p turbospark-repack --example lamq_linear_probe -- \
//!     ~/models/lamq-probe/ink54 ~/models/inkling2-mini-gturbo 0

use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;

use compute::lamq::{LamqCode, LamqLinear};
use turbospark_repack::npz::{NpyDtype, NpzArchive};

/// LAMQ linear name (meta key) -> HF install name suffix (`docs/LAMQ.md`).
const NAMES: [(&str, &str); 6] = [
    ("self_attn.q_proj.linear", "attn.wq_du.weight"),
    ("self_attn.k_proj.linear", "attn.wk_dv.weight"),
    ("self_attn.v_proj.linear", "attn.wv_dv.weight"),
    (
        "self_attn.relative_position_bias.linear",
        "attn.wr_du.weight",
    ),
    ("self_attn.o_proj.gate_proj", "attn.gate_attn.weight"),
    ("self_attn.o_proj.linear", "attn.wo_ud.weight"),
];

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [lamq, install, layer] = args.as_slice() else {
        eprintln!("usage: lamq_linear_probe <lamq release dir> <HF gturbo install> <layer>");
        std::process::exit(2);
    };
    let (lamq, install) = (PathBuf::from(lamq), PathBuf::from(install));
    let layer: usize = layer.parse().unwrap();
    let meta: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(lamq.join(format!("layer_{layer:02}_meta.json"))).unwrap(),
    )
    .unwrap();
    let mut luts = NpzArchive::open(&lamq.join("luts.npz")).unwrap();
    let mut npz = NpzArchive::open(&lamq.join(format!("layer_{layer:02}_packed.npz"))).unwrap();
    let weights = install.join("model_weights.bin");
    let idx = model_io::load_resident_index(&weights).unwrap();
    let mut f = std::fs::File::open(&weights).unwrap();

    for (lname, hf) in NAMES {
        let m = &meta["linears"][lname];
        let code = LamqCode::from_name(m["fmt"].as_str().unwrap()).unwrap();
        let (rows, cols) = (
            m["m"].as_u64().unwrap() as usize,
            m["n"].as_u64().unwrap() as usize,
        );
        let safe = m["safe"].as_str().unwrap();
        let codes = npz.read(&format!("lin_{safe}_codes")).unwrap();
        let codes = codes
            .bytes(if code.is_trellis() {
                NpyDtype::U32
            } else {
                NpyDtype::U8
            })
            .unwrap()
            .to_vec();
        let scales = npz
            .read(&format!("lin_{safe}_scales"))
            .unwrap()
            .to_f16_bits()
            .unwrap();
        let signs_in = npz
            .read(&format!("lin_{safe}_signs_in"))
            .unwrap()
            .to_i8()
            .unwrap();
        let signs_out = m["out_rot"].as_bool().unwrap_or(false).then(|| {
            npz.read(&format!("lin_{safe}_signs_out"))
                .unwrap()
                .to_i8()
                .unwrap()
        });
        let table: Vec<f32> = if code.is_trellis() {
            luts.read(code.name())
                .unwrap()
                .to_f16_bits()
                .unwrap()
                .into_iter()
                .map(compute::quant_1bit::f16_to_f32)
                .collect()
        } else {
            npz.read(&format!("lin_{safe}_levels"))
                .unwrap()
                .to_f32()
                .unwrap()
        };
        let lin = LamqLinear {
            code,
            m: rows,
            n: cols,
            codes: &codes,
            scales: &scales,
            signs_in: &signs_in,
            signs_out: signs_out.as_deref(),
            table: &table,
        };
        lin.validate().unwrap();

        let name = format!("model.llm.layers.{layer}.{hf}");
        let ent = &idx.entries[&name];
        assert_eq!(ent.dtype, 1, "{name}: the probe expects BF16");
        assert_eq!(
            (ent.shape.0 as usize, ent.shape.1 as usize),
            (rows, cols),
            "{name} shape"
        );
        let mut b = vec![0u8; rows * cols * 2];
        f.seek(SeekFrom::Start(ent.file_offset)).unwrap();
        f.read_exact(&mut b).unwrap();

        // Deterministic input with no structure the rotation could hide.
        let mut s = 0x9E37_79B9u32 ^ layer as u32;
        let x: Vec<f32> = (0..cols)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                (s % 20_001) as f32 / 10_000.0 - 1.0
            })
            .collect();
        let got = lin.forward(&x);
        let want: Vec<f64> = (0..rows)
            .map(|r| {
                (0..cols)
                    .map(|c| {
                        let o = 2 * (r * cols + c);
                        let w =
                            f32::from_bits(u32::from(u16::from_le_bytes([b[o], b[o + 1]])) << 16);
                        f64::from(w) * f64::from(x[c])
                    })
                    .sum()
            })
            .collect();
        let dot: f64 = got.iter().zip(&want).map(|(a, b)| a * b).sum();
        let na = got.iter().map(|a| a * a).sum::<f64>().sqrt();
        let nb = want.iter().map(|a| a * a).sum::<f64>().sqrt();
        println!(
            "layer {layer} {lname:<42} {:>5}{} {rows}x{cols}: output cosine {:.4}, norm ratio {:.4}",
            code.name(),
            if lin.signs_out.is_some() { "+out" } else { "    " },
            dot / (na * nb),
            na / nb
        );
    }
}
