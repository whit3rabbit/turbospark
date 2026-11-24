//! Per-tensor weight error of a LAMQ Inkling 2 install against the BF16
//! install of the same model: for every coded matrix that has a same-named
//! BF16 tensor, `|| W_lamq x - W_bf16 x || / || W_bf16 x ||` over random
//! probes, with `W_lamq x` from the CPU reference (`LamqLinear::forward`,
//! rotations included). An outlier tensor points at a decode defect; uniform
//! error points at the quantization itself. Diagnostic, not a gate.
//!
//! usage: inkling2_lamq_weight_err <lamq install> <bf16 install> [probes]

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use compute::lamq::{parse_matrix_aux, LamqCode, LamqLinear};
use half::f16;
use model_io::{load_resident_index, ResidentIndexEntry};

const TAG_BF16: u8 = 1;
const TAG_FP16: u8 = 2;
const TAG_LAMQ: u8 = 36;

/// One expert matrix to compare: role, code, its codes, scales and input
/// signs, the reference weights, rows, cols, and the reference dtype.
type ExpertCase<'a> = (
    &'a str,
    LamqCode,
    &'a [u8],
    &'a [u16],
    &'a [i8],
    Vec<f32>,
    usize,
    usize,
    &'a str,
);

fn read_at(f: &mut File, off: u64, len: u64) -> Vec<u8> {
    let mut b = vec![0u8; len as usize];
    f.seek(SeekFrom::Start(off)).expect("seek");
    f.read_exact(&mut b).expect("read");
    b
}

fn u16s(b: &[u8]) -> Vec<u16> {
    b.chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

fn xorshift(s: &mut u64) -> f64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    // Approximately N(0, 1): sum of 12 uniforms minus 6.
    let mut acc = 0.0;
    for _ in 0..12 {
        *s ^= *s << 13;
        *s ^= *s >> 7;
        *s ^= *s << 17;
        acc += (*s >> 11) as f64 / (1u64 << 53) as f64;
    }
    acc - 6.0
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if !(3..=4).contains(&args.len()) {
        eprintln!("usage: {} <lamq install> <bf16 install> [probes]", args[0]);
        std::process::exit(2);
    }
    let probes: usize = args.get(3).map_or(4, |s| s.parse().expect("probes"));
    let (lq_path, bf_path) = (
        Path::new(&args[1]).join("model_weights.bin"),
        Path::new(&args[2]).join("model_weights.bin"),
    );
    let lq = load_resident_index(&lq_path).expect("lamq index");
    let bf = load_resident_index(&bf_path).expect("bf16 index");
    let (mut lf, mut bfile) = (File::open(&lq_path).unwrap(), File::open(&bf_path).unwrap());

    let lut = |code: LamqCode, lf: &mut File| -> Vec<f32> {
        let e = &lq.entries[&format!("lamq.lut.{}", code.name())];
        assert_eq!(e.dtype, TAG_FP16);
        u16s(&read_at(lf, e.file_offset, e.size_bytes))
            .into_iter()
            .map(|b| f16::from_bits(b).to_f32())
            .collect()
    };

    let mut names: Vec<&ResidentIndexEntry> = lq
        .entries
        .values()
        .filter(|e| {
            e.dtype == TAG_LAMQ && (e.name.ends_with(".weight") || e.name.contains(".weight.b"))
        })
        .collect();
    names.sort_by(|a, b| a.name.cmp(&b.name));
    println!("tensor\tcode\trows\tcols\trel_err\tcosine\tscale_ratio");
    // Bucketed embedding and head: bucket b's row r is token
    // order[first row of b + r] of the stem's BF16 matrix.
    let mut bucket_rows: std::collections::HashMap<String, Vec<usize>> = Default::default();
    for stem in ["model.llm.embed.weight", "model.llm.unembed.weight"] {
        let o = &lq.entries[stem];
        let ids: Vec<usize> = read_at(&mut lf, o.file_offset, o.size_bytes)
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as usize)
            .collect();
        let mut lo = 0usize;
        let mut b = 0;
        while let Some(be) = lq.entries.get(&format!("{stem}.b{b}")) {
            let rows = be.shape.0 as usize;
            bucket_rows.insert(be.name.clone(), ids[lo..lo + rows].to_vec());
            lo += rows;
            b += 1;
        }
        assert_eq!(lo, ids.len(), "{stem}: buckets cover the order");
    }
    let mut skipped = 0;
    for e in names {
        let row_map = bucket_rows.get(&e.name);
        let ref_name = match row_map {
            Some(_) => e.name[..e.name.find(".weight.b").unwrap() + 7].to_string(),
            None => e.name.clone(),
        };
        let Some(r) = bf.entries.get(&ref_name) else {
            skipped += 1;
            continue;
        };
        let (m, n) = (e.shape.0 as usize, e.shape.1 as usize);
        let ref_rows_ok = match row_map {
            Some(_) => r.shape.1 as usize == n,
            None => (r.shape.0 as usize, r.shape.1 as usize) == (m, n),
        };
        if r.dtype != TAG_BF16 || !ref_rows_ok {
            println!(
                "{}\tSHAPE/DTYPE MISMATCH lamq {m}x{n} bf16 dtype {} {}x{}",
                e.name, r.dtype, r.shape.0, r.shape.1
            );
            continue;
        }
        let aux = parse_matrix_aux(&read_at(&mut lf, e.bias_offset, e.bias_size)).expect("aux");
        let code = aux.header.code.expect("matrix code");
        let codes = read_at(&mut lf, e.file_offset, e.size_bytes);
        let scales = u16s(&read_at(&mut lf, e.scale_offset, e.scale_size));
        let table = if code.is_trellis() {
            lut(code, &mut lf)
        } else {
            aux.levels.clone().expect("levels")
        };
        let lin = LamqLinear {
            code,
            m,
            n,
            codes: &codes,
            scales: &scales,
            signs_in: &aux.signs_in,
            signs_out: aux.signs_out.as_deref(),
            table: &table,
        };
        lin.validate().expect("valid lamq linear");
        let bf16_row = |bfile: &mut File, t: usize| -> Vec<f32> {
            u16s(&read_at(
                bfile,
                r.file_offset + (t * n * 2) as u64,
                (n * 2) as u64,
            ))
            .into_iter()
            .map(|b| f32::from_bits(u32::from(b) << 16))
            .collect()
        };
        let w: Vec<f32> = match row_map {
            Some(ids) => ids.iter().flat_map(|&t| bf16_row(&mut bfile, t)).collect(),
            None => u16s(&read_at(&mut bfile, r.file_offset, r.size_bytes))
                .into_iter()
                .map(|b| f32::from_bits(u32::from(b) << 16))
                .collect(),
        };
        // PROFILE=<substring>: the error in the rotated (stored) domain by
        // column position, for matrices without an output rotation. Row r of
        // the stored matrix is apply_in(w_r): W x = W~ (H s x) / sqrt(n).
        if let Ok(pat) = std::env::var("PROFILE") {
            if e.name.contains(&pat) && aux.signs_out.is_none() {
                let rht = compute::lamq::Rht::new(n).expect("rht");
                let (mut by16, mut by16d) = (vec![0.0f64; 16], vec![0.0f64; 16]);
                let nch = n / 16;
                let (mut bych, mut bychd) = (vec![0.0f64; nch], vec![0.0f64; nch]);
                let (mut fit_num, mut fit_den) = (0.0f64, 0.0f64);
                for row in 0..m.min(256) {
                    let wr = rht.apply_in(&w[row * n..(row + 1) * n], &aux.signs_in);
                    let q = lin.row(row);
                    for j in 0..n {
                        let d = f64::from(q[j]) - wr[j];
                        by16[j % 16] += d * d;
                        by16d[j % 16] += wr[j] * wr[j];
                        bych[j / 16] += d * d;
                        bychd[j / 16] += wr[j] * wr[j];
                        fit_num += f64::from(q[j]) * wr[j];
                        fit_den += f64::from(q[j]) * f64::from(q[j]);
                    }
                }
                let r16: Vec<String> = (0..16)
                    .map(|j| format!("{:.3}", (by16[j] / by16d[j]).sqrt()))
                    .collect();
                println!("# {} col%16 rel err: {}", e.name, r16.join(" "));
                let rc: Vec<f64> = (0..nch).map(|c| (bych[c] / bychd[c]).sqrt()).collect();
                let even: f64 = rc.iter().step_by(2).sum::<f64>() / (nch / 2) as f64;
                let odd: f64 = rc.iter().skip(1).step_by(2).sum::<f64>() / (nch / 2) as f64;
                println!(
                    "# {} chunk rel err first {:.3} last {:.3} even {:.3} odd {:.3} max {:.3}; best scale {:.4}",
                    e.name,
                    rc[0],
                    rc[nch - 1],
                    even,
                    odd,
                    rc.iter().cloned().fold(0.0, f64::max),
                    fit_num / fit_den
                );
            }
        }
        let mut seed = 0x9E37_79B9_7F4A_7C15u64 ^ (m as u64 * 31 + n as u64);
        let (mut num, mut den, mut dot, mut qq) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for _ in 0..probes {
            let x: Vec<f32> = (0..n).map(|_| xorshift(&mut seed) as f32).collect();
            let yq = lin.forward(&x);
            for (row, &q) in yq.iter().enumerate() {
                let y: f64 = w[row * n..(row + 1) * n]
                    .iter()
                    .zip(&x)
                    .map(|(&a, &b)| f64::from(a) * f64::from(b))
                    .sum();
                num += (q - y) * (q - y);
                den += y * y;
                dot += q * y;
                qq += q * q;
            }
        }
        println!(
            "{}\t{}\t{m}\t{n}\t{:.4}\t{:.4}\t{:.4}",
            e.name,
            code.name(),
            (num / den).sqrt(),
            dot / (den.sqrt() * qq.sqrt()),
            (qq / den).sqrt()
        );
    }
    eprintln!("{skipped} coded tensors without a same-named BF16 tensor");

    // Routed experts: LAMQ gate / up / down against the reference install's
    // NVFP4 or MXFP8 expert times its outer scale (gate and up outer[0] and
    // [1], down outer[2]). LAMQ rotates the gate_up input with the layer's
    // signs[..hidden] and the down input with signs[hidden..], no output
    // rotation.
    let (lq_dir, bf_dir) = (Path::new(&args[1]), Path::new(&args[2]));
    let ll = model_io::load_packed_experts_layout(lq_dir, 1 << 26).expect("lamq layout");
    let bl = model_io::load_packed_experts_layout(bf_dir, 1 << 26).expect("ref layout");
    let (h, f) = (2304usize, 1152usize);
    println!("expert\tlayer\texpert\trole\tcode\tref\trel_err\tcosine\tscale_ratio");
    for l in [0usize, 7, 19, 20, 23] {
        let s_e = &lq.entries[&format!("model.llm.layers.{l}.mlp.experts.lamq_signs")];
        let signs: Vec<i8> = read_at(&mut lf, s_e.file_offset, s_e.size_bytes)
            .into_iter()
            .map(|b| b as i8)
            .collect();
        let mut lfile = File::open(lq_dir.join("packed_experts").join(&ll.layers[l].file)).unwrap();
        let mut rfile = File::open(bf_dir.join("packed_experts").join(&bl.layers[l].file)).unwrap();
        for e in [0usize, 31, 64, 127] {
            let le = &ll.layers[l].experts[e];
            let re = &bl.layers[l].experts[e];
            assert_eq!((le.expert, re.expert), (e, e));
            let lpart = |lfile: &mut File, role: &str| {
                let st = &le.sub_tensors[role];
                read_at(lfile, le.offset + st.offset, st.size)
            };
            let rpart = |rfile: &mut File, role: &str| {
                let st = &re.sub_tensors[role];
                (
                    read_at(rfile, re.offset + st.offset, st.size),
                    st.dtype.clone(),
                )
            };
            let fmt = lpart(&mut lfile, "lamq_fmt");
            let word = |i: usize| u32::from_le_bytes(fmt[4 * i..4 * i + 4].try_into().unwrap());
            let gu_code = LamqCode::from_id(word(0) as u8).expect("gu code");
            let dn_code = LamqCode::from_id(word(1) as u8).expect("dn code");
            let gu = lpart(&mut lfile, "gate_up");
            let gu_s = u16s(&lpart(&mut lfile, "gate_up_scales"));
            let dn = lpart(&mut lfile, "down");
            let dn_s = u16s(&lpart(&mut lfile, "down_scales"));
            let outer: Vec<f32> = rpart(&mut rfile, "outer")
                .0
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let dec = |bytes: &[u8], dtype: &str, n: usize, scale: f32| -> Vec<f32> {
                let v = if dtype.contains("nvfp4") {
                    compute::dequantize_nvfp4(bytes, n)
                } else {
                    compute::inkling2::dequantize_mxfp8_blocks(bytes, n)
                };
                v.into_iter().map(|x| x * scale).collect()
            };
            let (gb, gd) = rpart(&mut rfile, "gate");
            let (ub, ud) = rpart(&mut rfile, "up");
            let (db, dd) = rpart(&mut rfile, "down");
            let gu_rb = gu_code.row_bytes(h).unwrap();
            let cases: [ExpertCase<'_>; 3] = [
                (
                    "gate",
                    gu_code,
                    &gu[..f * gu_rb],
                    &gu_s[..f],
                    &signs[..h],
                    dec(&gb, &gd, f * h, outer[0]),
                    f,
                    h,
                    &gd,
                ),
                (
                    "up",
                    gu_code,
                    &gu[f * gu_rb..2 * f * gu_rb],
                    &gu_s[f..2 * f],
                    &signs[..h],
                    dec(&ub, &ud, f * h, outer[1]),
                    f,
                    h,
                    &ud,
                ),
                (
                    "down",
                    dn_code,
                    &dn[..h * dn_code.row_bytes(f).unwrap()],
                    &dn_s[..],
                    &signs[h..h + f],
                    dec(&db, &dd, h * f, outer[2]),
                    h,
                    f,
                    &dd,
                ),
            ];
            for (role, code, codes, scales, sg, w, m, n, rd) in cases {
                let table = lut(code, &mut lf);
                let lin = LamqLinear {
                    code,
                    m,
                    n,
                    codes,
                    scales,
                    signs_in: sg,
                    signs_out: None,
                    table: &table,
                };
                lin.validate().expect("valid expert linear");
                let mut seed = 0xA5A5_5A5A_1234_5678u64 ^ (l * 1000 + e) as u64;
                let (mut num, mut den, mut dot, mut qq) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
                for _ in 0..probes {
                    let x: Vec<f32> = (0..n).map(|_| xorshift(&mut seed) as f32).collect();
                    let yq = lin.forward(&x);
                    for (row, &q) in yq.iter().enumerate() {
                        let y: f64 = w[row * n..(row + 1) * n]
                            .iter()
                            .zip(&x)
                            .map(|(&a, &b)| f64::from(a) * f64::from(b))
                            .sum();
                        num += (q - y) * (q - y);
                        den += y * y;
                        dot += q * y;
                        qq += q * q;
                    }
                }
                println!(
                    "expert\t{l}\t{e}\t{role}\t{}\t{rd}\t{:.4}\t{:.4}\t{:.4}",
                    code.name(),
                    (num / den).sqrt(),
                    dot / (den.sqrt() * qq.sqrt()),
                    (qq / den).sqrt()
                );
            }
        }
    }
}
