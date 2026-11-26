#![cfg(target_os = "macos")]
//! The Metal Inkling 2 DFlash2 draft pass against an f32 CPU reference, on a
//! real target install and the published draft.
//!
//! ```text
//! INKLING2_INSTALL=~/.turbospark/models/text/inkling2-mini-mlx4.gturbo \
//! TURBOSPARK_INKLING2_DFLASH2=~/models/inkling2-dflash2 \
//! TURBOSPARK_INKLING2_TAPS=1 cargo test --release -p turbospark-runtime \
//!     --test real_forward_inkling2_dflash2_parity -- --ignored --nocapture
//! ```
//!
//! Both sides start from the same target taps and block input and run the
//! context write, the five layers and the final norm; the post-norm rows are
//! compared. The target head is left out so any install quantization works.
//! A wrong draft would not change output (the verify decides every token), so
//! this is what holds the port to the reference: the conventions follow
//! z-lab/dflash and llama.cpp (ggml-org/llama.cpp#33073).

use std::path::PathBuf;

use half::bf16;
use turbospark_runtime::{LogitProducer, RealForwardRunner, SpeculativeProducer};

const LAYERS: usize = 5;
const HEADS: usize = 32;
const KV_HEADS: usize = 8;
const HD: usize = 128;
const BLOCK: usize = 8;
const GROUP: usize = 16;
const EPS: f32 = 1e-6;
const THETA: f64 = 10000.0;
/// The draft's sliding window of context positions, before the block.
const WINDOW: usize = 2048;

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

struct Weights(model_io::SafetensorsFile);

impl Weights {
    fn get(&self, name: &str) -> Vec<f32> {
        self.0
            .raw_bytes(name)
            .unwrap_or_else(|_| panic!("missing {name}"))
            .chunks_exact(2)
            .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
            .collect()
    }

    /// `y[n, out] = x[n, inp] @ w[out, inp]^T`.
    fn mm(&self, name: &str, x: &[f32], n: usize) -> Vec<f32> {
        let w = self.get(name);
        let inp = x.len() / n;
        let out = w.len() / inp;
        let mut y = vec![0f32; n * out];
        // SAFETY: the three buffers are sized by the arithmetic above;
        // RowMajor=101, NoTrans=111, Trans=112.
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

    fn rms(&self, name: &str, x: &[f32]) -> Vec<f32> {
        let w = self.get(name);
        let mut out = Vec::with_capacity(x.len());
        for row in x.chunks_exact(w.len()) {
            let ms = row.iter().map(|v| v * v).sum::<f32>() / w.len() as f32;
            let s = 1.0 / (ms + EPS).sqrt();
            out.extend(row.iter().zip(&w).map(|(v, g)| v * s * g));
        }
        out
    }
}

fn rope(head: &mut [f32], pos: usize) {
    let half = HD / 2;
    for i in 0..half {
        let f = pos as f64 * THETA.powf(-2.0 * i as f64 / HD as f64);
        let (s, c) = (f.sin() as f32, f.cos() as f32);
        let (a, b) = (head[i], head[i + half]);
        head[i] = a * c - b * s;
        head[i + half] = b * c + a * s;
    }
}

/// DFlash2's two-tap grouped dynamic convolution, `side` 0 before a sublayer
/// and 1 after it. A tap never reads before the block's first row.
fn conv(x: &[f32], coeffs: &[f32], base: &[f32], side: usize, h: usize) -> Vec<f32> {
    let groups = h / GROUP;
    let stride = 2 * 2 * groups;
    let mut out = vec![0f32; x.len()];
    for row in 0..BLOCK {
        for tap in 0..2.min(row + 1) {
            for c in 0..h {
                let k = base[(side * 2 + tap) * h + c]
                    + coeffs[row * stride + (side * 2 + tap) * groups + c / GROUP];
                out[row * h + c] += k * x[(row - tap) * h + c];
            }
        }
    }
    out
}

/// Context K (roped, normed) and V rows for one layer.
type Context = (Vec<f32>, Vec<f32>);

fn context(w: &Weights, taps: &[Vec<f32>]) -> Vec<Context> {
    let n = taps.len();
    let flat: Vec<f32> = taps.concat();
    let fc = w.rms("hidden_norm.weight", &w.mm("fc.weight", &flat, n));
    (0..LAYERS)
        .map(|l| {
            let p = |s: &str| format!("layers.{l}.self_attn.{s}.weight");
            let mut k = w.rms(&p("k_norm"), &w.mm(&p("k_proj"), &fc, n));
            for (i, head) in k.chunks_exact_mut(HD).enumerate() {
                rope(head, i / KV_HEADS);
            }
            (k, w.mm(&p("v_proj"), &fc, n))
        })
        .collect()
}

/// The block forward: returns the post-final-norm rows.
fn draft(w: &Weights, ctx: &[Context], input: &[f32], base: usize) -> Vec<f32> {
    let h = input.len() / BLOCK;
    let mut x = input.to_vec();
    for (l, (ck, cv)) in ctx.iter().enumerate() {
        let p = |s: &str| format!("layers.{l}.{s}");
        let normed = w.rms(&p("input_layernorm.weight"), &x);
        let coeffs = w.mm(
            &p("attention_conv.kernel_projection.weight"),
            &normed,
            BLOCK,
        );
        let kernel = w.get(&p("attention_conv.base_kernel"));
        let pre = conv(&normed, &coeffs, &kernel, 0, h);
        let mut q = w.rms(
            &p("self_attn.q_norm.weight"),
            &w.mm(&p("self_attn.q_proj.weight"), &pre, BLOCK),
        );
        let mut k = w.rms(
            &p("self_attn.k_norm.weight"),
            &w.mm(&p("self_attn.k_proj.weight"), &pre, BLOCK),
        );
        let v = w.mm(&p("self_attn.v_proj.weight"), &pre, BLOCK);
        for (i, head) in q.chunks_exact_mut(HD).enumerate() {
            rope(head, base + i / HEADS);
        }
        for (i, head) in k.chunks_exact_mut(HD).enumerate() {
            rope(head, base + i / KV_HEADS);
        }
        let kvd = KV_HEADS * HD;
        let old = (ck.len() / kvd).saturating_sub(WINDOW) * kvd;
        let keys: Vec<&[f32]> = ck[old..]
            .chunks_exact(kvd)
            .chain(k.chunks_exact(kvd))
            .collect();
        let vals: Vec<&[f32]> = cv[old..]
            .chunks_exact(kvd)
            .chain(v.chunks_exact(kvd))
            .collect();
        let mut att = vec![0f32; BLOCK * HEADS * HD];
        for (i, out) in att.chunks_exact_mut(HD).enumerate() {
            let kh = (i % HEADS) / (HEADS / KV_HEADS);
            let span = kh * HD..(kh + 1) * HD;
            let query = &q[i * HD..(i + 1) * HD];
            let s: Vec<f32> = keys
                .iter()
                .map(|key| {
                    let dot: f32 = query
                        .iter()
                        .zip(&key[span.clone()])
                        .map(|(a, b)| a * b)
                        .sum();
                    dot / (HD as f32).sqrt()
                })
                .collect();
            let max = s.iter().cloned().fold(f32::MIN, f32::max);
            let e: Vec<f32> = s.iter().map(|v| (v - max).exp()).collect();
            let sum: f32 = e.iter().sum();
            for (weight, value) in e.iter().zip(&vals) {
                for (o, &vv) in out.iter_mut().zip(&value[span.clone()]) {
                    *o += weight / sum * vv;
                }
            }
        }
        let o = w.mm(&p("self_attn.o_proj.weight"), &att, BLOCK);
        let post = conv(&o, &coeffs, &kernel, 1, h);
        x.iter_mut().zip(&post).for_each(|(a, b)| *a += b);

        let normed = w.rms(&p("post_attention_layernorm.weight"), &x);
        let coeffs = w.mm(&p("mlp_conv.kernel_projection.weight"), &normed, BLOCK);
        let kernel = w.get(&p("mlp_conv.base_kernel"));
        let pre = conv(&normed, &coeffs, &kernel, 0, h);
        let gate = w.mm(&p("mlp.gate_proj.weight"), &pre, BLOCK);
        let up = w.mm(&p("mlp.up_proj.weight"), &pre, BLOCK);
        let act: Vec<f32> = gate
            .iter()
            .zip(&up)
            .map(|(g, u)| g / (1.0 + (-g).exp()) * u)
            .collect();
        let down = w.mm(&p("mlp.down_proj.weight"), &act, BLOCK);
        let post = conv(&down, &coeffs, &kernel, 1, h);
        x.iter_mut().zip(&post).for_each(|(a, b)| *a += b);
    }
    w.rms("norm.weight", &x)
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm(a) * norm(b))
}

#[test]
#[ignore = "needs INKLING2_INSTALL, TURBOSPARK_INKLING2_DFLASH2 and TURBOSPARK_INKLING2_TAPS=1"]
fn the_metal_draft_pass_matches_the_cpu_reference() {
    let env = |k: &str| std::env::var(k).unwrap_or_else(|_| panic!("set {k}"));
    let install = PathBuf::from(env("INKLING2_INSTALL"));
    let drafter = PathBuf::from(env("TURBOSPARK_INKLING2_DFLASH2"));
    assert_eq!(env("TURBOSPARK_INKLING2_TAPS"), "1");
    let tok = tokenizer::MfTokenizer::load_from_dir(&install).expect("tokenizer");
    let short = "Write a Rust function that parses a CSV line into a Vec<String>, handling \
                 quoted fields with embedded commas and escaped double quotes. Include unit tests."
        .to_string();
    // Past the window, so the context ring has wrapped.
    let long = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/INKLING2_PHASE0.md"),
    )
    .expect("long prompt");
    let long = format!(
        "{}\n\nSummarize the document above in five bullet points.",
        long.chars().take(9000).collect::<String>()
    );
    let arch = model_io::inkling2_mini();
    let mut runner = RealForwardRunner::open_with_residency(
        &install,
        arch.clone(),
        4096,
        turbospark_runtime::ExpertCacheSlots::Auto,
        turbospark_runtime::DraftPolicies {
            mtp: turbospark_runtime::MtpDraftPolicy::Off,
            dflash: turbospark_runtime::DflashDraftPolicy::Dflash2(BLOCK - 1),
        },
        turbospark_runtime::SteeringPolicy::off(),
        1,
        turbospark_runtime::KvQuant::Off,
        turbospark_runtime::ExpertResidency::Resident,
    )
    .expect("open with the DFlash2 drafter");

    let w = Weights(
        model_io::SafetensorsFile::open(&drafter.join("model.safetensors")).expect("draft"),
    );
    for (kind, text, past_window) in [("short", short, false), ("long", long, true)] {
        let ids = tok.encode_text_continuation(&text);
        assert_eq!(
            ids.len() > WINDOW + BLOCK,
            past_window,
            "{kind}: prompt length"
        );
        eprintln!("{kind}:");
        check(
            &mut runner,
            &w,
            &ids,
            arch.vocab_size as usize,
            arch.hidden_size as usize,
        );
    }
}

/// Drafts at the end of `ids` on Metal and on the CPU and compares the rows.
fn check(runner: &mut RealForwardRunner, w: &Weights, ids: &[i32], vocab: usize, h: usize) {
    // Walk the prompt, priming the drafter's context row by row, then draft
    // with the last prompt token as the anchor.
    runner.reset();
    let base = ids.len() - 1;
    let mut taps = Vec::new();
    let mut logits = vec![half::f16::ZERO; vocab];
    for pos in 0..base {
        runner.produce(ids[pos], pos, &mut logits).expect("produce");
        taps.push(
            runner
                .inkling2_taps()
                .expect("taps")
                .iter()
                .map(|v| v.to_f32())
                .collect::<Vec<f32>>(),
        );
        runner.prime_drafter(ids[pos + 1], pos).expect("prime");
    }
    let mut proposals = Vec::new();
    runner
        .draft_block(ids[base], base, BLOCK - 1, &mut proposals)
        .expect("draft");
    let (input, metal) = runner.inkling2_dflash2_block_rows().expect("DFlash2 rows");

    let cpu = draft(w, &context(w, &taps), &input, base);
    let cos: Vec<f32> = cpu
        .chunks_exact(h)
        .zip(metal.chunks_exact(h))
        .map(|(a, b)| cosine(a, b))
        .collect();
    eprintln!("{} context rows; per-row cosine {cos:?}", taps.len());
    eprintln!("proposal ids {proposals:?}");
    assert!(cos.iter().all(|&c| c > 0.999), "rows diverge: {cos:?}");
}
