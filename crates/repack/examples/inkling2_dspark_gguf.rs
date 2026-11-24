//! Converts a DSpark drafter GGUF (unsloth's old layout) into the
//! `mtp.safetensors` + `config.json` pair an Inkling 2 install loads from
//! `<install>/dspark/`.
//!
//! ```text
//! cargo run --release -p turbospark-repack --example inkling2_dspark_gguf -- \
//!     <dspark-Inkling-2-Mini-Q8_0_Old.gguf> <install>/dspark
//! ```

use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: {} <drafter.gguf> <out dir>", args[0]);
        std::process::exit(2);
    }
    match turbospark_repack::write_inkling2_dspark_from_gguf(
        Path::new(&args[1]),
        Path::new(&args[2]),
    ) {
        Ok(r) => println!("{r:?}"),
        Err(e) => {
            eprintln!("drafter conversion failed: {e}");
            std::process::exit(1);
        }
    }
}
