//! Offline acceptance of the Inkling 2 DSpark drafter (`mtp.safetensors`)
//! against the real target, before any batched verify exists.
//!
//! ```text
//! TURBOSPARK_INKLING2_TAPS=1 cargo run --release -p turbospark-runtime \
//!     --example inkling2_dspark_accept -- <bf16 install> <mtp.safetensors> \
//!     <config.json with mtp_config> <prompt ids> <generate N>
//! ```
//!
//! The target decodes the prompt and `N` greedy tokens through the Metal flow,
//! recording each position's four context taps (the stream entering
//! applications 5, 23 and 41, and the final-norm output). Then, at every
//! generated position `t`, a CPU reference of the drafter (f32, Accelerate
//! sgemm) drafts the block `[token t, mask x 7]` from the taps of positions
//! `< t` and proposes 7 tokens greedily; a proposal is accepted while it equals
//! the target's own next greedy token. This is the acceptance a lossless greedy
//! verify would see, without its speed.
//!
//! The forward follows ollama `mlxrunner/model/inkling2/draft.go` and oMLX
//! `patches/inkling2/dspark.py`: per-layer softmax tap mixes, context K/V with
//! `k_norm_ctx` and NeoX RoPE, a bidirectional block over context plus block
//! (the local layers see the last `window` context rows, the others all of
//! them, as ollama), BF16-rounded residuals, the target LM head, and the RNN
//! Markov head.
//!
//! With `TURBOSPARK_INKLING2_DSPARK=<drafter dir>` the target also opens the
//! Metal drafter (`families/inkling2/dspark.rs`) and drafts at every generated
//! position from the same taps; the report then compares its proposals with
//! the CPU reference's (the drafter's parity gate) and gives both
//! acceptances. `<prompt ids>` may be a `.txt` file of prose instead, which is
//! tokenized without a template.

use std::collections::HashMap;
use std::path::Path;

use half::{bf16, f16};
use turbospark_runtime::{LogitProducer, RealForwardRunner, SpeculativeProducer};

#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    fn cblas_sgemm(
        order: i32,
        ta: i32,
        tb: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );
}

/// `y[n, out] = x[n, inp] @ w[out, inp]^T`.
fn matmul(x: &[f32], n: usize, w: &[f32], out: usize, inp: usize) -> Vec<f32> {
    assert_eq!(x.len(), n * inp);
    assert_eq!(w.len(), out * inp);
    let mut y = vec![0f32; n * out];
    // SAFETY: the buffers are the sizes asserted above; RowMajor=101,
    // NoTrans=111, Trans=112.
    unsafe {
        cblas_sgemm(
            101,
            111,
            112,
            n as i32,
            out as i32,
            inp as i32,
            1.0,
            x.as_ptr(),
            inp as i32,
            w.as_ptr(),
            inp as i32,
            0.0,
            y.as_mut_ptr(),
            out as i32,
        );
    }
    y
}

fn rms(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let d = w.len();
    let mut out = Vec::with_capacity(x.len());
    for row in x.chunks_exact(d) {
        let ms = row.iter().map(|v| v * v).sum::<f32>() / d as f32;
        let s = 1.0 / (ms + eps).sqrt();
        out.extend(row.iter().zip(w).map(|(v, g)| v * s * g));
    }
    out
}

fn round_bf16(x: &mut [f32]) {
    for v in x {
        *v = bf16::from_f32(*v).to_f32();
    }
}

/// NeoX (half-split) RoPE in place on `heads x dim` at `pos`.
fn rope(x: &mut [f32], dim: usize, pos: usize, theta: f32) {
    let half = dim / 2;
    for head in x.chunks_exact_mut(dim) {
        for i in 0..half {
            let f = (pos as f64) * (theta as f64).powf(-2.0 * i as f64 / dim as f64);
            let (s, c) = (f.sin() as f32, f.cos() as f32);
            let (a, b) = (head[i], head[i + half]);
            head[i] = a * c - b * s;
            head[i + half] = b * c + a * s;
        }
    }
}

struct Weights {
    t: HashMap<String, (Vec<usize>, Vec<f32>)>,
}

impl Weights {
    fn load(path: &Path) -> Self {
        let st = model_io::SafetensorsFile::open(path).expect("open mtp.safetensors");
        let mut t = HashMap::new();
        for name in st.tensor_names() {
            let d = st.descriptor(name).unwrap();
            assert_eq!(d.dtype, "BF16", "{name}: expected BF16");
            let b = st.raw_bytes(name).unwrap();
            let v = b
                .chunks_exact(2)
                .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                .collect();
            t.insert(name.to_string(), (d.shape.clone(), v));
        }
        Self { t }
    }
    fn get(&self, n: &str) -> &[f32] {
        &self.t.get(n).unwrap_or_else(|| panic!("missing {n}")).1
    }
}

struct Cfg {
    h: usize,
    layers: usize,
    heads: usize,
    kvh: usize,
    hd: usize,
    ffn: usize,
    eps: f32,
    theta: f32,
    rank: usize,
    taps: usize,
    vocab: usize,
    unpadded: usize,
    softcap: f32,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 6 {
        eprintln!(
            "usage: {} <bf16 install> <mtp.safetensors> <config.json> <prompt ids> <N>",
            args[0]
        );
        std::process::exit(2);
    }
    assert_eq!(
        std::env::var("TURBOSPARK_INKLING2_TAPS").as_deref(),
        Ok("1"),
        "set TURBOSPARK_INKLING2_TAPS=1"
    );
    let install = Path::new(&args[1]);
    let config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&args[3]).unwrap()).unwrap();
    let m = &config["mtp_config"];
    let text = &config["text_config"];
    let c = Cfg {
        h: m["hidden_size"].as_u64().unwrap() as usize,
        layers: m["num_hidden_layers"].as_u64().unwrap() as usize,
        heads: m["num_attention_heads"].as_u64().unwrap() as usize,
        kvh: m["num_key_value_heads"].as_u64().unwrap() as usize,
        hd: m["head_dim"].as_u64().unwrap() as usize,
        ffn: m["intermediate_size"].as_u64().unwrap() as usize,
        eps: m["rms_norm_eps"].as_f64().unwrap() as f32,
        theta: m["rope_theta"].as_f64().unwrap() as f32,
        rank: m["markov_rank"].as_u64().unwrap() as usize,
        taps: m["target_layer_ids"].as_array().unwrap().len(),
        vocab: text["vocab_size"].as_u64().unwrap() as usize,
        unpadded: text["unpadded_vocab_size"].as_u64().unwrap() as usize,
        softcap: text["final_logit_softcapping"].as_f64().unwrap() as f32,
    };
    let block = m["block_size"].as_u64().unwrap() as usize;
    let window = m["sliding_window_size"].as_u64().unwrap() as usize;
    let local_ids: Vec<u64> = m["local_layer_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap())
        .collect();
    let local: Vec<bool> = (0..c.layers as u64).map(|l| local_ids.contains(&l)).collect();
    let n_gen: usize = args[5].parse().unwrap();
    let raw = std::fs::read_to_string(&args[4]).unwrap();
    let prompt: Vec<i32> = if args[4].ends_with(".chat.txt") {
        // One user turn through the checkpoint's template.
        tokenizer::MfTokenizer::load_from_dir(install)
            .expect("tokenizer")
            .encode_text_continuation(&raw)
    } else if args[4].ends_with(".txt") {
        tokenizer::MfTokenizer::load_from_dir(install)
            .expect("tokenizer")
            .encode(&raw, false)
    } else {
        raw.split_whitespace().map(|t| t.parse().unwrap()).collect()
    };
    let metal = std::env::var_os("TURBOSPARK_INKLING2_DSPARK").is_some();

    // 1. Target: greedy decode with taps.
    let arch = model_io::inkling2_mini();
    let mut runner = if metal {
        RealForwardRunner::open_with_residency(
            install,
            arch.clone(),
            4096,
            turbospark_runtime::ExpertCacheSlots::Auto,
            turbospark_runtime::DraftPolicies {
                mtp: turbospark_runtime::MtpDraftPolicy::Off,
                dflash: turbospark_runtime::DflashDraftPolicy::Fixed(block - 1),
            },
            turbospark_runtime::SteeringPolicy::off(),
            1,
            turbospark_runtime::KvQuant::Off,
            turbospark_runtime::ExpertResidency::Resident,
        )
        .expect("open target with the Metal drafter")
    } else {
        RealForwardRunner::open_with_max_context(install, arch.clone(), 4096).expect("open target")
    };
    let mut tokens = prompt.clone();
    let mut taps: Vec<Vec<f32>> = Vec::new();
    let mut logits = vec![f16::ZERO; arch.vocab_size as usize];
    // Metal proposals per drafted position.
    let mut metal_props: std::collections::HashMap<usize, Vec<i32>> = Default::default();
    let mut metal_ms = 0f64;
    let t0 = std::time::Instant::now();
    let mut pos = 0usize;
    while pos < prompt.len() + n_gen {
        if metal && pos >= prompt.len() {
            let mut props = Vec::new();
            let t = std::time::Instant::now();
            runner
                .draft_block(tokens[pos], pos, block - 1, &mut props)
                .expect("metal draft");
            metal_ms += t.elapsed().as_secs_f64() * 1e3;
            metal_props.insert(pos, props);
        }
        runner
            .produce(tokens[pos], pos, &mut logits)
            .expect("produce");
        if metal && pos + 1 < prompt.len() {
            runner.prime_drafter(tokens[pos + 1], pos).expect("prime");
        }
        let t = runner.inkling2_taps().expect("taps enabled");
        taps.push(t.iter().map(|v| v.to_f32()).collect());
        if pos + 1 >= tokens.len() {
            let best = logits[..c.unpadded]
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.to_f32().total_cmp(&b.1.to_f32()))
                .unwrap()
                .0 as i32;
            tokens.push(best);
        }
        pos += 1;
    }
    eprintln!(
        "target: {} positions in {:.1} s",
        taps.len(),
        t0.elapsed().as_secs_f32()
    );
    drop(runner);

    // 2. Drafter and the target's embed, embed norm and head (BF16 install).
    let w = Weights::load(Path::new(&args[2]));
    let index = model_io::load_resident_index(&install.join("model_weights.bin")).unwrap();
    let file = std::fs::read(install.join("model_weights.bin")).unwrap();
    let target = |name: &str| -> Vec<f32> {
        let e = &index.entries[name];
        assert_eq!(e.dtype, 1, "{name}: the harness reads a BF16 install");
        file[e.file_offset as usize..(e.file_offset + e.size_bytes) as usize]
            .chunks_exact(2)
            .map(|b| bf16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32())
            .collect()
    };
    let embed = target("model.llm.embed.weight");
    let embed_norm = target("model.llm.embed_norm.weight");
    let head = target("model.llm.unembed.weight");
    let head = &head[..c.unpadded * c.h];
    // Fold the Markov latent projection into the head once: [V, rank] and [V].
    let hidden_out = w.get("markov_head.hidden_out.weight"); // [h, rank]
    let mut ho_t = vec![0f32; c.rank * c.h]; // [rank, h]
    for i in 0..c.h {
        for r in 0..c.rank {
            ho_t[r * c.h + i] = hidden_out[i * c.rank + r];
        }
    }
    let fold_w = matmul(head, c.unpadded, &ho_t, c.rank, c.h);
    let fold_b = matmul(
        head,
        c.unpadded,
        w.get("markov_head.hidden_out.bias"),
        1,
        c.h,
    );
    // Tap mixtures: softmax over each row of fuse_weight [layers + 1, taps].
    let fuse = w.get("fuse_weight");
    let alpha: Vec<Vec<f32>> = (0..=c.layers)
        .map(|i| {
            let row = &fuse[i * c.taps..(i + 1) * c.taps];
            let mx = row.iter().cloned().fold(f32::MIN, f32::max);
            let e: Vec<f32> = row.iter().map(|v| (v - mx).exp()).collect();
            let s: f32 = e.iter().sum();
            e.iter().map(|v| v / s).collect()
        })
        .collect();
    let mix = |tap: &[f32], a: &[f32], norm: &str| -> Vec<f32> {
        let mut x = vec![0f32; c.h];
        for (k, &ak) in a.iter().enumerate() {
            for (xi, &v) in x.iter_mut().zip(&tap[k * c.h..(k + 1) * c.h]) {
                *xi += ak * v;
            }
        }
        rms(&x, w.get(norm), c.eps)
    };
    let kvd = c.kvh * c.hd;
    let group = c.heads / c.kvh;
    let l = |i: usize, s: &str| format!("layers.{i}.{s}");

    // 3. Context K/V caches per layer, grown as t advances.
    let mut ctx_k: Vec<Vec<f32>> = vec![Vec::new(); c.layers];
    let mut ctx_v: Vec<Vec<f32>> = vec![Vec::new(); c.layers];
    let add_context = |row: usize, ck: &mut Vec<Vec<f32>>, cv: &mut Vec<Vec<f32>>| {
        for i in 0..c.layers {
            let x = mix(&taps[row], &alpha[i], &l(i, "dflare_norm.weight"));
            let mut k = matmul(&x, 1, w.get(&l(i, "self_attn.k_proj_ctx.weight")), kvd, c.h);
            k = rms(&k, w.get(&l(i, "self_attn.k_norm_ctx.weight")), c.eps);
            rope(&mut k, c.hd, row, c.theta);
            let v = matmul(&x, 1, w.get(&l(i, "self_attn.v_proj_ctx.weight")), kvd, c.h);
            ck[i].extend(k);
            cv[i].extend(v);
        }
    };

    let p = prompt.len();
    let mut accepted_hist = vec![0usize; block];
    let mut per_pos = vec![0usize; block - 1];
    let mut rounds = 0usize;
    let mut built = 0usize;
    let mut agree_prefix = vec![0usize; block];
    let mut agree_pos = 0usize;
    let mut metal_hist = vec![0usize; block];
    let t1 = std::time::Instant::now();
    for t in p..(p + n_gen).saturating_sub(block) {
        while built < t {
            add_context(built, &mut ctx_k, &mut ctx_v);
            built += 1;
        }
        // Block input: the anchor's normed embedding, then mask_embed rows.
        let anchor = tokens[t] as usize;
        let mut x = rms(&embed[anchor * c.h..(anchor + 1) * c.h], &embed_norm, c.eps);
        round_bf16(&mut x);
        for _ in 1..block {
            x.extend_from_slice(w.get("mask_embed"));
        }
        for i in 0..c.layers {
            let xn = rms(&x, w.get(&l(i, "input_layernorm.weight")), c.eps);
            let mut q = matmul(
                &xn,
                block,
                w.get(&l(i, "self_attn.q_proj.weight")),
                c.heads * c.hd,
                c.h,
            );
            q = rms(&q, w.get(&l(i, "self_attn.q_norm.weight")), c.eps);
            let mut k = matmul(
                &xn,
                block,
                w.get(&l(i, "self_attn.k_proj.weight")),
                kvd,
                c.h,
            );
            k = rms(&k, w.get(&l(i, "self_attn.k_norm.weight")), c.eps);
            let v = matmul(
                &xn,
                block,
                w.get(&l(i, "self_attn.v_proj.weight")),
                kvd,
                c.h,
            );
            for r in 0..block {
                rope(
                    &mut q[r * c.heads * c.hd..(r + 1) * c.heads * c.hd],
                    c.hd,
                    t + r,
                    c.theta,
                );
                rope(&mut k[r * kvd..(r + 1) * kvd], c.hd, t + r, c.theta);
            }
            // Context rows inside the window (local layers) or all of them,
            // then the block, all visible.
            let n_ctx = if local[i] { t.min(window) } else { t };
            let first = t - n_ctx;
            let keys: Vec<&[f32]> = (first..t)
                .map(|j| &ctx_k[i][j * kvd..(j + 1) * kvd])
                .chain((0..block).map(|r| &k[r * kvd..(r + 1) * kvd]))
                .collect();
            let vals: Vec<&[f32]> = (first..t)
                .map(|j| &ctx_v[i][j * kvd..(j + 1) * kvd])
                .chain((0..block).map(|r| &v[r * kvd..(r + 1) * kvd]))
                .collect();
            let scale = 1.0 / (c.hd as f32).sqrt();
            let mut attn = vec![0f32; block * c.heads * c.hd];
            for r in 0..block {
                for hq in 0..c.heads {
                    let kh = hq / group;
                    let qv = &q[(r * c.heads + hq) * c.hd..(r * c.heads + hq + 1) * c.hd];
                    let s: Vec<f32> = keys
                        .iter()
                        .map(|kr| {
                            qv.iter()
                                .zip(&kr[kh * c.hd..(kh + 1) * c.hd])
                                .map(|(a, b)| a * b)
                                .sum::<f32>()
                                * scale
                        })
                        .collect();
                    let mx = s.iter().cloned().fold(f32::MIN, f32::max);
                    let e: Vec<f32> = s.iter().map(|v| (v - mx).exp()).collect();
                    let sum: f32 = e.iter().sum();
                    let out = &mut attn[(r * c.heads + hq) * c.hd..(r * c.heads + hq + 1) * c.hd];
                    for (ej, vr) in e.iter().zip(&vals) {
                        for (o, &vv) in out.iter_mut().zip(&vr[kh * c.hd..(kh + 1) * c.hd]) {
                            *o += ej / sum * vv;
                        }
                    }
                }
            }
            let o = matmul(
                &attn,
                block,
                w.get(&l(i, "self_attn.o_proj.weight")),
                c.h,
                c.heads * c.hd,
            );
            round_bf16(&mut x);
            let mut r: Vec<f32> = x.iter().zip(&o).map(|(a, b)| a + b).collect();
            let ff = rms(&r, w.get(&l(i, "post_attention_layernorm.weight")), c.eps);
            let g = matmul(&ff, block, w.get(&l(i, "mlp.gate_proj.weight")), c.ffn, c.h);
            let u = matmul(&ff, block, w.get(&l(i, "mlp.up_proj.weight")), c.ffn, c.h);
            let act: Vec<f32> = g
                .iter()
                .zip(&u)
                .map(|(g, u)| g / (1.0 + (-g).exp()) * u)
                .collect();
            let d = matmul(
                &act,
                block,
                w.get(&l(i, "mlp.down_proj.weight")),
                c.h,
                c.ffn,
            );
            round_bf16(&mut r);
            x = r.iter().zip(&d).map(|(a, b)| a + b).collect();
        }
        let hidden = rms(&x, w.get("norm.weight"), c.eps);
        let masks = &hidden[c.h..];
        let base = matmul(masks, block - 1, head, c.unpadded, c.h);
        // Markov head over the 7 mask rows, conditioned on row t - 1's taps.
        let feature = mix(
            &taps[t - 1],
            &alpha[c.layers],
            "markov_head.dflare_norm.weight",
        );
        let w1 = w.get("markov_head.markov_w1.weight");
        let rnn_w = w.get("markov_head.rnn_proj.weight");
        let rnn_b = w.get("markov_head.rnn_proj.bias");
        let mut state = vec![0f32; c.rank];
        let mut last = anchor;
        let mut accepted = 0usize;
        let mut still = true;
        let mut cpu_props: Vec<i32> = Vec::with_capacity(block - 1);
        for i in 0..block - 1 {
            let mut input = state.clone();
            input.extend_from_slice(&w1[last * c.rank..(last + 1) * c.rank]);
            input.extend_from_slice(&masks[i * c.h..(i + 1) * c.h]);
            input.extend_from_slice(&feature);
            let gts = matmul(&input, 1, rnn_w, 3 * c.rank, input.len());
            let gts: Vec<f32> = gts.iter().zip(rnn_b).map(|(a, b)| a + b).collect();
            let r = c.rank;
            let latent: Vec<f32> = (0..r).map(|j| gts[2 * r + j].tanh()).collect();
            for j in 0..r {
                let keep = 1.0 / (1.0 + (-gts[j]).exp());
                state[j] = keep * state[j] + (1.0 - keep) * gts[r + j].tanh();
            }
            let bias = matmul(&latent, 1, &fold_w, c.unpadded, c.rank);
            let row = &base[i * c.unpadded..(i + 1) * c.unpadded];
            let best = (0..c.unpadded)
                .max_by(|&a, &b| {
                    let sa = c.softcap * ((row[a] + bias[a] + fold_b[a]) / c.softcap).tanh();
                    let sb = c.softcap * ((row[b] + bias[b] + fold_b[b]) / c.softcap).tanh();
                    sa.total_cmp(&sb)
                })
                .unwrap();
            if still && best as i32 == tokens[t + 1 + i] {
                accepted += 1;
                per_pos[i] += 1;
            } else {
                still = false;
            }
            last = best;
            cpu_props.push(best as i32);
        }
        if let Some(mp) = metal_props.get(&t) {
            let same = mp.iter().zip(&cpu_props).take_while(|(a, b)| a == b).count();
            agree_prefix[same] += 1;
            agree_pos += mp.iter().zip(&cpu_props).filter(|(a, b)| a == b).count();
            let acc = mp
                .iter()
                .enumerate()
                .take_while(|(i, &p)| p == tokens[t + 1 + i])
                .count();
            metal_hist[acc] += 1;
        }
        let _ = c.vocab;
        accepted_hist[accepted] += 1;
        rounds += 1;
    }
    let mean: f64 = accepted_hist
        .iter()
        .enumerate()
        .map(|(k, &n)| (k * n) as f64)
        .sum::<f64>()
        / rounds.max(1) as f64;
    eprintln!(
        "drafter: {rounds} rounds in {:.1} s; mean accepted {mean:.2} of {} (tokens per verify {:.2})",
        t1.elapsed().as_secs_f32(),
        block - 1,
        mean + 1.0
    );
    eprintln!("accepted-length histogram: {accepted_hist:?}");
    if metal {
        let mr: usize = metal_hist.iter().sum();
        let mmean = metal_hist
            .iter()
            .enumerate()
            .map(|(k, &n)| (k * n) as f64)
            .sum::<f64>()
            / mr.max(1) as f64;
        eprintln!(
            "metal drafter: {mr} rounds, {:.2} ms per draft; mean accepted {mmean:.2} of {} \
             (tokens per verify {:.2}); histogram {metal_hist:?}",
            metal_ms / metal_props.len().max(1) as f64,
            block - 1,
            mmean + 1.0
        );
        eprintln!(
            "metal vs cpu proposals: {agree_pos} of {} positions equal; agreeing-prefix \
             histogram {agree_prefix:?}",
            mr * (block - 1)
        );
    }
    eprintln!(
        "per-position acceptance: {:?}",
        per_pos
            .iter()
            .map(|&n| format!("{:.2}", n as f64 / rounds.max(1) as f64))
            .collect::<Vec<_>>()
    );
}
