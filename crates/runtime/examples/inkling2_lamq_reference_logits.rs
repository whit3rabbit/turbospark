//! The Inkling 2 CPU reference forward (`compute::inkling2::forward`, f32,
//! no Metal) on a LAMQ install's weights, decoded on demand through the
//! `LamqLinear` reference: coded linears, the bucketed head and embedding,
//! and the routed experts go through the source hooks; every uncoded tensor
//! is the install's own. Writes `n x 200058` f32 logits, a row per position,
//! for comparison with the Metal LAMQ path (`inkling2_logits`) on the same
//! install. Diagnostic, not a gate: f32 and slow.
//!
//! usage: inkling2_lamq_reference_logits <lamq install> <ids.txt> <out.f32>

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use compute::inkling2::forward::{Config, Dtype, Reference, WeightSource};
use compute::lamq::{parse_matrix_aux, LamqCode, LamqLinear, Rht};
use half::f16;
use model_io::{load_packed_experts_layout, load_resident_index, PackedExpertsLayout};

const TAG_BF16: u8 = 1;
const TAG_FP16: u8 = 2;
const TAG_F32: u8 = 3;
const TAG_LAMQ: u8 = 36;

fn u16s(b: &[u8]) -> Vec<u16> {
    b.chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

/// One coded matrix, owned.
struct Coded {
    code: LamqCode,
    m: usize,
    n: usize,
    codes: Vec<u8>,
    scales: Vec<u16>,
    signs_in: Vec<i8>,
    signs_out: Option<Vec<i8>>,
    table: Vec<f32>,
}

impl Coded {
    fn lin(&self) -> LamqLinear<'_> {
        LamqLinear {
            code: self.code,
            m: self.m,
            n: self.n,
            codes: &self.codes,
            scales: &self.scales,
            signs_in: &self.signs_in,
            signs_out: self.signs_out.as_deref(),
            table: &self.table,
        }
    }
}

/// `LamqLinear::forward` with the rows split across threads: the same
/// rotation, the same per-row f64 sums, so the same values.
fn forward_par(lin: &LamqLinear<'_>, x: &[f32]) -> Vec<f32> {
    let xr = Rht::new(lin.n)
        .expect("input Hadamard")
        .apply_in(x, lin.signs_in);
    let mut y = vec![0.0f64; lin.m];
    let threads = std::thread::available_parallelism().map_or(8, |n| n.get());
    let chunk = lin.m.div_ceil(threads).max(1);
    std::thread::scope(|s| {
        for (c, part) in y.chunks_mut(chunk).enumerate() {
            let xr = &xr;
            s.spawn(move || {
                for (i, o) in part.iter_mut().enumerate() {
                    *o = lin
                        .row(c * chunk + i)
                        .iter()
                        .zip(xr)
                        .map(|(&w, &v)| f64::from(w) * v)
                        .sum();
                }
            });
        }
    });
    let y = match lin.signs_out {
        Some(s) => Rht::new(lin.m).expect("output Hadamard").apply_out(&y, s),
        None => y,
    };
    y.into_iter().map(|v| v as f32).collect()
}

struct Source {
    dense: HashMap<String, (Dtype, Vec<usize>, Vec<u8>)>,
    coded: HashMap<String, Coded>,
    embed: Vec<Coded>,
    /// Token id -> (embedding bucket, row).
    embed_loc: Vec<(usize, usize)>,
    head: Vec<Coded>,
    /// Head position -> token id.
    head_order: Vec<usize>,
    padded_vocab: usize,
    hidden: usize,
    ffn: usize,
    signs: Vec<Vec<i8>>,
    luts: HashMap<LamqCode, Vec<f32>>,
    experts_dir: PathBuf,
    layout: PackedExpertsLayout,
}

impl Source {
    /// Expert `e` of layer `l`: (gate_up, down) as coded matrices.
    fn expert(&self, l: usize, e: usize) -> (Coded, Coded) {
        let ll = &self.layout.layers[l];
        let ee = &ll.experts[e];
        let mut f = File::open(self.experts_dir.join(&ll.file)).expect("expert file");
        let mut part = |role: &str| {
            let st = &ee.sub_tensors[role];
            let mut b = vec![0u8; st.size as usize];
            f.seek(SeekFrom::Start(ee.offset + st.offset)).unwrap();
            f.read_exact(&mut b).unwrap();
            b
        };
        let fmt = part("lamq_fmt");
        let word = |i: usize| u32::from_le_bytes(fmt[4 * i..4 * i + 4].try_into().unwrap());
        let (gc, dc) = (
            LamqCode::from_id(word(0) as u8).expect("gate_up code"),
            LamqCode::from_id(word(1) as u8).expect("down code"),
        );
        let (h, ff) = (self.hidden, self.ffn);
        let mut gu = part("gate_up");
        gu.truncate(2 * ff * gc.row_bytes(h).unwrap());
        let mut dn = part("down");
        dn.truncate(h * dc.row_bytes(ff).unwrap());
        let gu = Coded {
            code: gc,
            m: 2 * ff,
            n: h,
            codes: gu,
            scales: u16s(&part("gate_up_scales")),
            signs_in: self.signs[l][..h].to_vec(),
            signs_out: None,
            table: self.luts[&gc].clone(),
        };
        let dn = Coded {
            code: dc,
            m: h,
            n: ff,
            codes: dn,
            scales: u16s(&part("down_scales")),
            signs_in: self.signs[l][h..h + ff].to_vec(),
            signs_out: None,
            table: self.luts[&dc].clone(),
        };
        (gu, dn)
    }
}

impl WeightSource for Source {
    fn tensor(&self, name: &str) -> Option<(Dtype, &[usize], &[u8])> {
        self.dense
            .get(name)
            .map(|(d, s, b)| (*d, s.as_slice(), b.as_slice()))
    }

    fn matvec(&self, name: &str, x: &[f32]) -> Option<Vec<f32>> {
        if name == "model.llm.unembed.weight" {
            let mut logits = vec![0.0f32; self.padded_vocab];
            let mut pos = 0;
            for b in &self.head {
                for (i, v) in forward_par(&b.lin(), x).into_iter().enumerate() {
                    logits[self.head_order[pos + i]] = v;
                }
                pos += b.m;
            }
            return Some(logits);
        }
        self.coded.get(name).map(|c| forward_par(&c.lin(), x))
    }

    fn embed_row(&self, token: usize) -> Option<Vec<f32>> {
        let (b, r) = self.embed_loc[token];
        let lin = self.embed[b].lin();
        let row: Vec<f64> = lin.row(r).into_iter().map(f64::from).collect();
        let out = Rht::new(lin.n).unwrap().apply_out(&row, lin.signs_in);
        Some(out.into_iter().map(|v| v as f32).collect())
    }

    fn expert_gate_up(&self, ip: usize, e: usize, x: &[f32]) -> Option<(Vec<f32>, Vec<f32>)> {
        let (gu, _) = self.expert(ip, e);
        let y = forward_par(&gu.lin(), x);
        let f = self.ffn;
        Some((y[..f].to_vec(), y[f..].to_vec()))
    }

    fn expert_down(&self, ip: usize, e: usize, act: &[f32]) -> Option<Vec<f32>> {
        let (_, dn) = self.expert(ip, e);
        Some(forward_par(&dn.lin(), act))
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        eprintln!("usage: {} <lamq install> <ids.txt> <out.f32>", args[0]);
        std::process::exit(2);
    }
    let dir = Path::new(&args[1]);
    let cfg = Config::mini();
    let wpath = dir.join("model_weights.bin");
    let index = load_resident_index(&wpath).expect("index");
    let file = std::fs::read(&wpath).expect("read weights");
    let at = |off: u64, len: u64| &file[off as usize..(off + len) as usize];

    let mut luts = HashMap::new();
    for code in [
        LamqCode::Ht2,
        LamqCode::Ht3,
        LamqCode::Rtr1,
        LamqCode::Rtr2,
        LamqCode::Rtr3,
        LamqCode::Rtr4,
    ] {
        let e = &index.entries[&format!("lamq.lut.{}", code.name())];
        assert_eq!(e.dtype, TAG_FP16);
        luts.insert(
            code,
            u16s(at(e.file_offset, e.size_bytes))
                .into_iter()
                .map(|b| f16::from_bits(b).to_f32())
                .collect::<Vec<f32>>(),
        );
    }
    let coded_of = |e: &model_io::ResidentIndexEntry| -> Coded {
        let aux = parse_matrix_aux(at(e.bias_offset, e.bias_size)).expect("aux");
        let code = aux.header.code.expect("matrix code");
        let c = Coded {
            code,
            m: e.shape.0 as usize,
            n: e.shape.1 as usize,
            codes: at(e.file_offset, e.size_bytes).to_vec(),
            scales: u16s(at(e.scale_offset, e.scale_size)),
            signs_in: aux.signs_in,
            signs_out: aux.signs_out,
            table: if code.is_trellis() {
                luts[&code].clone()
            } else {
                aux.levels.expect("levels")
            },
        };
        c.lin().validate().expect("valid coded matrix");
        c
    };

    let (mut dense, mut coded) = (HashMap::new(), HashMap::new());
    for (name, e) in &index.entries {
        let shape: Vec<usize> = [e.shape.0, e.shape.1, e.shape.2, e.shape.3]
            .into_iter()
            .take_while(|&d| d != 0)
            .map(|d| d as usize)
            .collect();
        match e.dtype {
            TAG_BF16 | TAG_F32 => {
                let dt = if e.dtype == TAG_BF16 {
                    Dtype::Bf16
                } else {
                    Dtype::F32
                };
                dense.insert(
                    name.clone(),
                    (dt, shape, at(e.file_offset, e.size_bytes).to_vec()),
                );
            }
            TAG_LAMQ if name.ends_with(".weight") && e.shape.1 > 1 => {
                coded.insert(name.clone(), coded_of(e));
            }
            _ => {}
        }
    }
    // Full-attention layers: the install holds `static_w` rewritten as the
    // conv kernel `[ch, 1, taps]`; the reference reads `[taps, ch]` by lag.
    for &l in &cfg.full_layers {
        for x in ["k", "v"] {
            let a = format!("model.llm.layers.{l}.attn.{x}_sconv");
            let (dt, s, w) = dense[&format!("{a}.weight")].clone();
            assert_eq!(dt, Dtype::Bf16);
            let (ch, taps) = (s[0], s[2]);
            let mut sw = vec![0u8; w.len()];
            for c in 0..ch {
                for j in 0..taps {
                    let (src, dst) = ((c * taps + j) * 2, ((taps - 1 - j) * ch + c) * 2);
                    sw[dst..dst + 2].copy_from_slice(&w[src..src + 2]);
                }
            }
            dense.insert(format!("{a}.static_w"), (dt, vec![taps, ch], sw));
        }
    }

    let buckets = |stem: &str| -> (Vec<Coded>, Vec<usize>) {
        let o = &index.entries[stem];
        let order: Vec<usize> = at(o.file_offset, o.size_bytes)
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as usize)
            .collect();
        let mut b = Vec::new();
        while let Some(e) = index.entries.get(&format!("{stem}.b{}", b.len())) {
            b.push(coded_of(e));
        }
        assert_eq!(b.iter().map(|c| c.m).sum::<usize>(), order.len(), "{stem}");
        (b, order)
    };
    let (embed, eorder) = buckets("model.llm.embed.weight");
    let mut embed_loc = vec![(usize::MAX, 0); eorder.len()];
    let mut pos = 0;
    for (bi, b) in embed.iter().enumerate() {
        for r in 0..b.m {
            embed_loc[eorder[pos + r]] = (bi, r);
        }
        pos += b.m;
    }
    let (head, head_order) = buckets("model.llm.unembed.weight");

    let signs = (0..cfg.layers)
        .map(|l| {
            let e = &index.entries[&format!("model.llm.layers.{l}.mlp.experts.lamq_signs")];
            at(e.file_offset, e.size_bytes)
                .iter()
                .map(|&b| b as i8)
                .collect()
        })
        .collect();
    let source = Source {
        dense,
        coded,
        embed,
        embed_loc,
        head,
        head_order,
        padded_vocab: eorder.len().max(cfg.vocab),
        hidden: cfg.hidden,
        ffn: cfg.expert_ffn,
        signs,
        luts,
        experts_dir: dir.join("packed_experts"),
        layout: load_packed_experts_layout(dir, 1 << 26).expect("layout"),
    };
    drop(file);

    let ids: Vec<u32> = std::fs::read_to_string(&args[2])
        .expect("ids")
        .split_whitespace()
        .map(|t| t.parse().expect("token id"))
        .collect();
    let mut model = Reference::new(&source, cfg);
    let mut out = std::io::BufWriter::new(File::create(&args[3]).expect("create output"));
    for (i, &id) in ids.iter().enumerate() {
        let t = Instant::now();
        let logits = model.step(id);
        let (best, top) = logits
            .iter()
            .enumerate()
            .fold(
                (0, f32::NEG_INFINITY),
                |b, (j, &v)| if v > b.1 { (j, v) } else { b },
            );
        eprintln!(
            "pos {i:3} id {id:6} -> argmax {best:6} ({top:.4})  {:.2} s",
            t.elapsed().as_secs_f32()
        );
        for v in &logits {
            out.write_all(&v.to_le_bytes()).unwrap();
        }
    }
    out.flush().unwrap();
}
