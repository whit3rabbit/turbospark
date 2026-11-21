//! Runs the Inkling 2 Metal decode flow over a token id list and writes the
//! logits of every position, in the same format as the CPU reference driver
//! (`repack/examples/inkling2_reference_logits.rs`) and the llama.cpp dumper,
//! so the three compare position by position.
//!
//! ```text
//! cargo run --release -p turbospark-runtime --example inkling2_logits -- \
//!     <install dir> <ids.txt> <out.f32> [max context, default 4096] [stride]
//! ```
//!
//! `out.f32` is `rows x 200058` little-endian f32 (the unpadded vocabulary;
//! the padding rows are dropped). Without `stride` every position writes a
//! row; with it, only positions `p % stride == stride - 1` and the last 16,
//! the rule `~/models/oracles/dump_logits_sparse` uses for long prompts.

use std::io::Write;
use std::path::Path;
use std::time::Instant;

use turbospark_runtime::{LogitProducer, RealForwardRunner};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if !(4..=6).contains(&args.len()) {
        eprintln!(
            "usage: {} <install dir> <ids.txt> <out.f32> [max context] [stride]",
            args[0]
        );
        std::process::exit(2);
    }
    let max_ctx: usize = args
        .get(4)
        .map_or(4096, |s| s.parse().expect("max context"));
    let stride: usize = args.get(5).map_or(1, |s| s.parse().expect("stride"));
    let ids: Vec<i32> = std::fs::read_to_string(&args[2])
        .expect("read ids")
        .split_whitespace()
        .map(|t| t.parse().expect("token id"))
        .collect();
    let arch = model_io::inkling2_mini();
    let vocab = arch.vocab_size as usize;
    let unpadded = model_io::inkling2_config::Inkling2Config::mini().unpadded_vocab as usize;
    let t = Instant::now();
    let mut runner = RealForwardRunner::open_with_max_context(Path::new(&args[1]), arch, max_ctx)
        .unwrap_or_else(|e| panic!("open: {e}"));
    eprintln!("open {:.1} s", t.elapsed().as_secs_f32());
    let mut out = std::io::BufWriter::new(std::fs::File::create(&args[3]).expect("create output"));
    let mut logits = vec![half::f16::ZERO; vocab];
    let n = ids.len();
    let keep = |p: usize| p % stride == stride - 1 || p + 16 >= n;
    for (pos, &id) in ids.iter().enumerate() {
        let t = Instant::now();
        runner
            .produce(id, pos, &mut logits)
            .unwrap_or_else(|e| panic!("position {pos}: {e}"));
        if !keep(pos) {
            continue;
        }
        let (best, top) =
            logits[..unpadded]
                .iter()
                .enumerate()
                .fold((0, f32::NEG_INFINITY), |b, (j, v)| {
                    let v = v.to_f32();
                    if v > b.1 {
                        (j, v)
                    } else {
                        b
                    }
                });
        eprintln!(
            "pos {pos:3} id {id:6} -> argmax {best:6} ({top:.4})  {:.1} ms",
            t.elapsed().as_secs_f64() * 1e3
        );
        for v in &logits[..unpadded] {
            out.write_all(&v.to_f32().to_le_bytes())
                .expect("write logits");
        }
    }
    out.flush().expect("flush");
}
