//! Runs the Inkling 2 Mini CPU reference forward
//! (`turbospark_compute::inkling2::forward`) over a real checkpoint and writes
//! the logits of every position, for comparison with llama.cpp.
//!
//! ```text
//! cargo run --release -p turbospark-repack --example inkling2_reference_logits -- \
//!     <model.safetensors> <ids.txt> <out.f32>
//! ```
//!
//! `ids.txt` holds whitespace-separated token ids. `out.f32` gets
//! `n_tokens x 200058` little-endian f32 logits, row per position. This is a
//! measurement tool, not a runtime path: it is slow (f32, weights decoded on
//! every use) and needs the whole checkpoint memory-mapped.

use compute::inkling2::forward::{Config, Dtype, Reference, WeightSource};
use model_io::SafetensorsFile;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

struct Source(SafetensorsFile);

impl WeightSource for Source {
    fn tensor(&self, name: &str) -> Option<(Dtype, &[usize], &[u8])> {
        let d = self.0.descriptor(name)?;
        let dt = match d.dtype.as_str() {
            "BF16" => Dtype::Bf16,
            "F32" => Dtype::F32,
            "U8" => Dtype::U8,
            "F8_E4M3" => Dtype::F8E4M3,
            other => panic!("{name}: unsupported dtype {other}"),
        };
        let bytes = self.0.raw_bytes(name).expect("tensor bytes");
        Some((dt, d.shape.as_slice(), bytes))
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        eprintln!("usage: {} <model.safetensors> <ids.txt> <out.f32>", args[0]);
        std::process::exit(2);
    }
    let file = SafetensorsFile::open(Path::new(&args[1])).expect("open checkpoint");
    let ids: Vec<u32> = std::fs::read_to_string(&args[2])
        .expect("read ids")
        .split_whitespace()
        .map(|t| t.parse().expect("token id"))
        .collect();
    let source = Source(file);
    let mut model = Reference::new(&source, Config::mini());
    let mut out = std::io::BufWriter::new(std::fs::File::create(&args[3]).expect("create output"));
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
            out.write_all(&v.to_le_bytes()).expect("write logits");
        }
    }
    out.flush().expect("flush");
}
