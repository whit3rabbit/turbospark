//! Installs Inkling 2 Mini from a local GGUF (`inkling2_loop` or the older
//! `inkling2`), then checks the result against an install made from the HF
//! checkpoint.
//!
//! ```text
//! cargo run --release -p turbospark-repack --example inkling2_gguf_install -- \
//!     <model.gguf> <reference hf install> <out dir>
//! ```
//!
//! Checks after writing: the resident names equal the reference's exactly;
//! each has the reference's logical shape (rows x cols for block-quantized
//! matrices); each dtype is the reference's, or a GGUF block tag where the
//! reference holds a BF16 matrix. The tokenizer files are copied from the
//! reference (a GGUF carries its vocabulary in the header instead).

use std::collections::BTreeMap;
use std::path::Path;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        eprintln!(
            "usage: {} <model.gguf> <reference install> <out dir>",
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
    let report =
        turbospark_repack::write_inkling2_gguf_install(src, out, "inkling2-gguf", |line| {
            println!("{line} ({:.0} s)", t.elapsed().as_secs_f64())
        })
        .unwrap_or_else(|e| panic!("install failed: {e}"));
    turbospark_repack::copy_inkling2_tokenizer_files(reference, out).expect("tokenizer files");
    println!("{report:?}");

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
    for name in want.keys().filter(|k| !got.contains_key(*k)) {
        problems.push(format!("missing {name}"));
    }
    for name in got.keys().filter(|k| !want.contains_key(*k)) {
        problems.push(format!("extra {name}"));
    }
    let mut blocks = 0;
    for (name, &(gdt, gshape)) in &got {
        let Some(&(rdt, rshape)) = want.get(name) else {
            continue;
        };
        if gshape != rshape {
            problems.push(format!("{name}: shape {gshape:?}, reference {rshape:?}"));
        } else if gdt != rdt {
            if rdt == 1 && turbospark_repack::GGUF_BLOCK_DTYPES.contains(&gdt) {
                blocks += 1;
            } else {
                problems.push(format!("{name}: dtype {gdt}, reference {rdt}"));
            }
        }
    }
    println!(
        "reference {} names, install {}, {blocks} GGUF block matrices",
        want.len(),
        got.len()
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
