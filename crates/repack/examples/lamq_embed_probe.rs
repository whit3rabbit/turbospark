//! Real-data check of the LAMQ CPU reference (`compute::lamq`): decodes
//! embedding rows from a LookAtMyQuant Inkling release and compares each with
//! the BF16 row of the same token in an HF install. It exercises `order`, the
//! bucket ranges, lm6 and rtr packing, the shared LUTs, the f16 scales, and
//! the Sylvester x Paley-I output rotation in one number per bucket.
//!
//! cargo run -p turbospark-repack --example lamq_embed_probe -- \
//!     ~/models/lamq-probe/ink54 ~/models/inkling2-mini-gturbo
//!
//! Prints cosine per bucket. A wrong rotation, transpose, bucket order or
//! token permutation puts cosine near 0; a right one gives the bucket's
//! quantization error.

use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;

use compute::lamq::{embed_row, LamqCode, LamqLinear};
use turbospark_repack::npz::{NpyDtype, NpzArchive};

fn bf16(b: [u8; 2]) -> f64 {
    f64::from(f32::from_bits(u32::from(u16::from_le_bytes(b)) << 16))
}

fn main() {
    let args: Vec<PathBuf> = std::env::args().skip(1).map(PathBuf::from).collect();
    let [lamq, install] = args.as_slice() else {
        eprintln!("usage: lamq_embed_probe <lamq release dir> <HF gturbo install>");
        std::process::exit(2);
    };
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(lamq.join("embed_meta.json")).unwrap())
            .unwrap();
    let mut luts = NpzArchive::open(&lamq.join("luts.npz")).unwrap();
    let mut e = NpzArchive::open(&lamq.join("embed_packed.npz")).unwrap();
    let order = e.read("order").unwrap().to_i32().unwrap();

    let weights = install.join("model_weights.bin");
    let idx = model_io::load_resident_index(&weights).unwrap();
    let ent = &idx.entries["model.llm.embed.weight"];
    let hidden = ent.shape.1 as usize;
    assert_eq!(ent.dtype, 1, "the probe expects the BF16 HF install");
    assert_eq!(ent.size_bytes as usize, ent.shape.0 as usize * hidden * 2);
    let mut f = std::fs::File::open(&weights).unwrap();
    let mut bf16_row = |tok: usize| -> Vec<f64> {
        let mut b = vec![0u8; hidden * 2];
        f.seek(SeekFrom::Start(ent.file_offset + (tok * hidden * 2) as u64))
            .unwrap();
        f.read_exact(&mut b).unwrap();
        b.chunks_exact(2).map(|c| bf16([c[0], c[1]])).collect()
    };

    let mut lo = 0usize;
    let mut worst = f64::INFINITY;
    for b in meta.as_array().unwrap() {
        let kb = b["kb"].as_u64().unwrap();
        let rows = b["rows"].as_u64().unwrap() as usize;
        let name = b["opt"].as_str().unwrap();
        let code = LamqCode::from_name(name).unwrap();
        let codes = e.read(&format!("b{kb}_codes")).unwrap();
        let codes_bytes = codes
            .bytes(if code.is_trellis() {
                NpyDtype::U32
            } else {
                NpyDtype::U8
            })
            .unwrap()
            .to_vec();
        let scales = e
            .read(&format!("b{kb}_scales"))
            .unwrap()
            .to_f16_bits()
            .unwrap();
        let signs = e.read(&format!("b{kb}_signs_in")).unwrap().to_i8().unwrap();
        let table: Vec<f32> = if code.is_trellis() {
            luts.read(name)
                .unwrap()
                .to_f16_bits()
                .unwrap()
                .into_iter()
                .map(compute::quant_1bit::f16_to_f32)
                .collect()
        } else {
            let lv = e.read(&format!("b{kb}_levels")).unwrap();
            match lv.dtype {
                NpyDtype::F32 => lv.to_f32().unwrap(),
                NpyDtype::F16 => lv
                    .to_f16_bits()
                    .unwrap()
                    .into_iter()
                    .map(compute::quant_1bit::f16_to_f32)
                    .collect(),
                d => panic!("levels dtype {d:?}"),
            }
        };
        let lin = LamqLinear {
            code,
            m: rows,
            n: hidden,
            codes: &codes_bytes,
            scales: &scales,
            signs_in: &signs,
            signs_out: None,
            table: &table,
        };
        lin.validate().unwrap();
        let mut cos_min = f64::INFINITY;
        let mut cos_sum = 0.0;
        let picks = [0, rows / 3, rows / 2, rows - 1];
        for &r in &picks {
            let tok = order[lo + r] as usize;
            let got = embed_row(&lin, r);
            let want = bf16_row(tok);
            let dot: f64 = got.iter().zip(&want).map(|(a, b)| a * b).sum();
            let na: f64 = got.iter().map(|a| a * a).sum::<f64>().sqrt();
            let nb: f64 = want.iter().map(|a| a * a).sum::<f64>().sqrt();
            let c = dot / (na * nb);
            cos_min = cos_min.min(c);
            cos_sum += c;
        }
        println!(
            "bucket {kb} {name:>4} rows {rows:>6} [{lo}, {}): cosine mean {:.4} min {:.4}",
            lo + rows,
            cos_sum / picks.len() as f64,
            cos_min
        );
        worst = worst.min(cos_min);
        lo += rows;
    }
    assert_eq!(lo, order.len(), "buckets cover every position");
    println!("worst cosine {worst:.4}");
}
