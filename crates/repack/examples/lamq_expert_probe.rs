//! Real-data check of a LAMQ install's experts and of its resident entries,
//! read back through the install (not the release): each probed expert's
//! gate-up and down outputs against the HF install's NVFP4/MXFP8 expert on
//! the same input, and one coded linear per probed layer against its BF16
//! reference. Exercises the blob format record, the contiguous gate/up
//! halves, the down rotation and the `DTYPE_LAMQ` aux round trip.
//!
//! cargo run --release -p turbospark-repack --example lamq_expert_probe -- \
//!     ~/models/inkling2-mini-lamq54-gturbo ~/models/inkling2-mini-gturbo

use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use compute::lamq::{parse_matrix_aux, LamqCode, LamqLinear};

fn read_at(path: &Path, off: u64, len: usize) -> Vec<u8> {
    let mut f = std::fs::File::open(path).unwrap();
    let mut b = vec![0u8; len];
    f.seek(SeekFrom::Start(off)).unwrap();
    f.read_exact(&mut b).unwrap();
    b
}

fn input(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            (s % 20_001) as f32 / 10_000.0 - 1.0
        })
        .collect()
}

fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    dot / (a.iter().map(|x| x * x).sum::<f64>().sqrt()
        * b.iter().map(|x| x * x).sum::<f64>().sqrt())
}

fn matvec(w: &[f32], rows: usize, cols: usize, x: &[f32]) -> Vec<f64> {
    (0..rows)
        .map(|r| {
            (0..cols)
                .map(|c| f64::from(w[r * cols + c]) * f64::from(x[c]))
                .sum()
        })
        .collect()
}

fn f16_bits(b: &[u8]) -> Vec<u16> {
    b.chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [lamq, hf] = args.as_slice() else {
        eprintln!("usage: lamq_expert_probe <lamq install> <HF install>");
        std::process::exit(2);
    };
    let (lamq, hf) = (Path::new(lamq), Path::new(hf));
    let (h, f) = (2304usize, 1152usize);
    let lw = lamq.join("model_weights.bin");
    let li = model_io::load_resident_index(&lw).unwrap();
    let hw = hf.join("model_weights.bin");
    let hi = model_io::load_resident_index(&hw).unwrap();
    let lut = |code: LamqCode| -> Vec<f32> {
        let e = &li.entries[&format!("lamq.lut.{}", code.name())];
        f16_bits(&read_at(&lw, e.file_offset, e.size_bytes as usize))
            .into_iter()
            .map(compute::quant_1bit::f16_to_f32)
            .collect()
    };
    // The head: per-bucket GEMV on the rotated input, scattered to vocab order
    // through the order entry, against the BF16 unembed.
    {
        let oe = &li.entries["model.llm.unembed.weight"];
        assert_eq!(oe.dtype, 36);
        let order: Vec<usize> = read_at(&lw, oe.file_offset, oe.size_bytes as usize)
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()) as usize)
            .collect();
        let x = input(h, 777);
        let mut logits = vec![0.0f64; order.len()];
        let mut pos = 0usize;
        for kb in 0.. {
            let Some(e) = li.entries.get(&format!("model.llm.unembed.weight.b{kb}")) else {
                break;
            };
            let aux = parse_matrix_aux(&read_at(&lw, e.bias_offset, e.bias_size as usize)).unwrap();
            let code = aux.header.code.unwrap();
            let codes = read_at(&lw, e.file_offset, e.size_bytes as usize);
            let scales = f16_bits(&read_at(&lw, e.scale_offset, e.scale_size as usize));
            let table = aux.levels.clone().unwrap_or_else(|| lut(code));
            let m = e.shape.0 as usize;
            let lin = LamqLinear {
                code,
                m,
                n: h,
                codes: &codes,
                scales: &scales,
                signs_in: &aux.signs_in,
                signs_out: None,
                table: &table,
            };
            lin.validate().unwrap();
            for (r, v) in lin.forward(&x).into_iter().enumerate() {
                logits[pos + r] = v;
            }
            pos += m;
        }
        assert_eq!(pos, order.len());
        let r = &hi.entries["model.llm.unembed.weight"];
        let rows = order.len();
        let rb = read_at(&hw, r.file_offset, rows * h * 2);
        let mut want = vec![0.0f64; rows];
        let mut got = vec![0.0f64; rows];
        for (p, &tok) in order.iter().enumerate() {
            got[tok] = logits[p];
            want[tok] = (0..h)
                .map(|c| {
                    let o = 2 * (tok * h + c);
                    f64::from(f32::from_bits(
                        u32::from(u16::from_le_bytes([rb[o], rb[o + 1]])) << 16,
                    )) * f64::from(x[c])
                })
                .sum();
        }
        let top = |v: &[f64]| (0..v.len()).max_by(|&a, &b| v[a].total_cmp(&v[b])).unwrap();
        println!(
            "head: {rows} logits, cosine {:.4}, top-1 {} vs reference {}",
            cosine(&got, &want),
            top(&got),
            top(&want)
        );
    }

    let ll = model_io::load_packed_experts_layout(lamq, 1 << 26).unwrap();
    let hl = model_io::load_packed_experts_layout(hf, 1 << 26).unwrap();

    for layer in [0usize, 7, 20] {
        // One coded linear through the resident aux round trip.
        let name = format!("model.llm.layers.{layer}.attn.wv_dv.weight");
        let e = &li.entries[&name];
        assert_eq!(e.dtype, 36);
        let aux = parse_matrix_aux(&read_at(&lw, e.bias_offset, e.bias_size as usize)).unwrap();
        let code = aux.header.code.unwrap();
        let (m, n) = (e.shape.0 as usize, e.shape.1 as usize);
        let codes = read_at(&lw, e.file_offset, e.size_bytes as usize);
        let scales = f16_bits(&read_at(&lw, e.scale_offset, e.scale_size as usize));
        let table = aux.levels.clone().unwrap_or_else(|| lut(code));
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
        lin.validate().unwrap();
        let x = input(n, 11 + layer as u32);
        let r = &hi.entries[&name];
        let rb = read_at(&hw, r.file_offset, r.size_bytes as usize);
        let rw: Vec<f32> = rb
            .chunks_exact(2)
            .map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16))
            .collect();
        println!(
            "layer {layer:2} wv_dv {:>4}{}: cosine {:.4}",
            code.name(),
            if aux.header.out_rot { "+out" } else { "    " },
            cosine(&lin.forward(&x), &matvec(&rw, m, n, &x))
        );

        // Experts: the signs, then a few experts of different formats.
        let s = &li.entries[&format!("model.llm.layers.{layer}.mlp.experts.lamq_signs")];
        let sb = read_at(&lw, s.file_offset, s.size_bytes as usize);
        let signs: Vec<i8> = sb.iter().map(|&b| b as i8).collect();
        let (gu_s, dn_s) = (&signs[..h], &signs[h..]);
        let lfile = lamq.join("packed_experts").join(&ll.layers[layer].file);
        let hfile = hf.join("packed_experts").join(&hl.layers[layer].file);
        let mut seen = Vec::new();
        for ex in 0..128usize {
            let le = &ll.layers[layer].experts[ex];
            let fmt = read_at(&lfile, le.offset, 16);
            let key = (fmt[0], fmt[4]);
            if seen.contains(&key) && ex != 127 {
                continue;
            }
            seen.push(key);
            let (gc, dc) = (
                LamqCode::from_id(fmt[0]).unwrap(),
                LamqCode::from_id(fmt[4]).unwrap(),
            );
            let sub = |role: &str, size: Option<usize>| {
                let t = &le.sub_tensors[role];
                read_at(
                    &lfile,
                    le.offset + t.offset,
                    size.unwrap_or(t.size as usize),
                )
            };
            let gu_codes = sub("gate_up", Some(2 * f * gc.row_bytes(h).unwrap()));
            let dn_codes = sub("down", Some(h * dc.row_bytes(f).unwrap()));
            let (gu_sc, dn_sc) = (
                f16_bits(&sub("gate_up_scales", None)),
                f16_bits(&sub("down_scales", None)),
            );
            let (gt, dt) = (lut(gc), lut(dc));
            let gu = LamqLinear {
                code: gc,
                m: 2 * f,
                n: h,
                codes: &gu_codes,
                scales: &gu_sc,
                signs_in: gu_s,
                signs_out: None,
                table: &gt,
            };
            let dn = LamqLinear {
                code: dc,
                m: h,
                n: f,
                codes: &dn_codes,
                scales: &dn_sc,
                signs_in: dn_s,
                signs_out: None,
                table: &dt,
            };
            gu.validate().unwrap();
            dn.validate().unwrap();

            // The HF reference expert, dequantized with its outer scales.
            let he = &hl.layers[layer].experts[ex];
            let hsub = |role: &str| {
                let t = &he.sub_tensors[role];
                (
                    read_at(&hfile, he.offset + t.offset, t.size as usize),
                    t.dtype.clone(),
                )
            };
            let (outer, _) = hsub("outer");
            let o: Vec<f32> = outer
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            let deq = |role: &str, n: usize, scale: f32| -> Vec<f32> {
                let (b, dt) = hsub(role);
                let v = if dt == "inkling2_nvfp4" {
                    compute::dequantize_nvfp4(&b, n)
                } else {
                    compute::inkling2::dequantize_mxfp8_blocks(&b, n)
                };
                v.into_iter().map(|w| w * scale).collect()
            };
            let nv = he.sub_tensors["gate"].dtype == "inkling2_nvfp4";
            let (og, ou, od) = if nv {
                (o[0], o[1], o[2])
            } else {
                (1.0, 1.0, 1.0)
            };
            let (wg, wu, wd) = (
                deq("gate", f * h, og),
                deq("up", f * h, ou),
                deq("down", h * f, od),
            );

            let x = input(h, 101 + ex as u32);
            let y = gu.forward(&x);
            let (g_ref, u_ref) = (matvec(&wg, f, h, &x), matvec(&wu, f, h, &x));
            // The HF interleave read of the same rows, to show the halves are right.
            let g_inter: Vec<f64> = (0..f).map(|k| y[2 * k]).collect();
            let a = input(f, 202 + ex as u32);
            let d = dn.forward(&a);
            println!(
                "layer {layer:2} expert {ex:3} gu {:>4} dn {:>4}: gate {:.4} up {:.4} down {:.4} (interleaved gate read {:.4})",
                gc.name(),
                dc.name(),
                cosine(&y[..f], &g_ref),
                cosine(&y[f..], &u_ref),
                cosine(&d, &matvec(&wd, h, f, &a)),
                cosine(&g_inter, &g_ref)
            );
        }
    }
}
