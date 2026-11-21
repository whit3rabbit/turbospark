//! Installs Inkling 2 Mini from a LookAtMyQuant packed release
//! (`shinmegamiboson/Inkling-2-Mini-LookAtMyQuant-*`, `docs/LAMQ.md`), then
//! checks the result against an install made from the HF checkpoint.
//!
//! ```text
//! cargo run --release -p turbospark-repack --example inkling2_lamq_install -- \
//!     <lamq release dir> <reference hf install> <out dir>
//! ```
//!
//! Checks after writing: every reference resident name is present; the
//! reference's matrices are `DTYPE_LAMQ` (36) here with the same rows and
//! columns; every other name has the reference's shape and either its dtype or
//! F32 where the reference holds BF16 (the release's steering is F32, not
//! exact BF16, and is not narrowed); the extra names are exactly the LUTs,
//! the embedding and head buckets and the per-layer expert signs. The
//! runtime does not open the install yet (no LAMQ kernels).

use std::collections::BTreeMap;
use std::path::Path;
use std::time::Instant;

const DTYPE_BF16: u8 = 1;
const DTYPE_FP32: u8 = 3;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        eprintln!(
            "usage: {} <lamq release dir> <reference install> <out dir>",
            args[0]
        );
        std::process::exit(2);
    }
    let (src, reference, out) = (
        Path::new(&args[1]),
        Path::new(&args[2]),
        Path::new(&args[3]),
    );
    let t = Instant::now();
    let report = turbospark_repack::write_inkling2_lamq_install(
        src,
        out,
        "shinmegamiboson/Inkling-2-Mini-LookAtMyQuant",
        |line| println!("{line} ({:.0} s)", t.elapsed().as_secs_f64()),
    )
    .unwrap_or_else(|e| panic!("install failed: {e}"));
    turbospark_repack::copy_inkling2_tokenizer_files(src, out).expect("tokenizer files");
    println!("{report:?}");
    println!(
        "experts: {:.2} GB packed, {:.2} GB written (x{:.2} from per-layer padding)",
        report.expert_bytes_packed as f64 / 1e9,
        report.expert_bytes_written as f64 / 1e9,
        report.expert_bytes_written as f64 / report.expert_bytes_packed as f64
    );

    type Entries = BTreeMap<String, (u8, (u32, u32, u32, u32))>;
    let load = |dir: &Path| -> Entries {
        model_io::load_resident_index(&dir.join("model_weights.bin"))
            .expect("resident index")
            .entries
            .into_iter()
            .map(|(k, e)| (k, (e.dtype, e.shape)))
            .collect()
    };
    let (want, got) = (load(reference), load(out));
    let mut problems = Vec::new();
    let mut widened = 0;
    for (name, &(rdt, rshape)) in &want {
        match got.get(name) {
            None => problems.push(format!("missing {name}")),
            Some(&(36, gshape)) => {
                let (embed_or_head, (rows, cols)) = (
                    name == "model.llm.embed.weight" || name == "model.llm.unembed.weight",
                    (rshape.0, rshape.1),
                );
                // Order entries are (positions, 1); matrices keep rows x cols.
                if !embed_or_head && (gshape.0, gshape.1) != (rows, cols) {
                    problems.push(format!("{name}: LAMQ {gshape:?}, reference {rshape:?}"));
                }
            }
            Some(&(gdt, gshape)) => {
                if gshape != rshape {
                    problems.push(format!("{name}: shape {gshape:?}, reference {rshape:?}"));
                } else if gdt == DTYPE_FP32 && rdt == DTYPE_BF16 {
                    widened += 1;
                } else if gdt != rdt {
                    problems.push(format!("{name}: dtype {gdt}, reference {rdt}"));
                }
            }
        }
    }
    let extra: Vec<&String> = got.keys().filter(|k| !want.contains_key(*k)).collect();
    let expected_extra = |k: &str| {
        k.starts_with("lamq.lut.")
            || k.starts_with("model.llm.embed.weight.b")
            || k.starts_with("model.llm.unembed.weight.b")
            || k.ends_with(".mlp.experts.lamq_signs")
    };
    for k in &extra {
        if !expected_extra(k) {
            problems.push(format!("unexpected extra {k}"));
        }
    }
    let coded = got.values().filter(|(d, _)| *d == 36).count();
    println!(
        "reference {} names, install {} ({} extra, {coded} LAMQ entries, {widened} F32 where the reference is BF16)",
        want.len(),
        got.len(),
        extra.len()
    );
    if problems.is_empty() {
        println!("check: OK");
    } else {
        for p in &problems {
            println!("PROBLEM {p}");
        }
        std::process::exit(1);
    }
}
