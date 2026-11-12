//! `lfm3`'s witnesses, re-read off the real published files.
//!
//! The strongest Phase 0 check `docs/NEW_MODEL.md` names: two independently
//! produced artifacts of the same model -- llama.cpp's GGUF and LiquidAI's
//! MLX conversion -- must derive the SAME `ArchConfig` through code paths
//! that share nothing (`arch_from_gguf` against `parse_lfm3_config`), and
//! both must equal the hand-written baseline. Also: every GGUF tensor name
//! maps, and the real `rope_freqs.weight` bytes pass the zero-tail check.
//!
//! Headers and one 256-byte tensor only, a few MB per file. `#[ignore]`d and
//! opt-in like every other `*_network` test.
//!
//! ```sh
//! cargo test -p turbospark-repack --test lfm3_network --release -- --ignored --nocapture
//! ```

use model_io::ModelFamily;
use turbospark_repack::{
    arch_from_gguf, fetch_gguf_header, lfm3_rope_freqs_are_zero_tail, map_gguf_name,
    parse_lfm3_config, GgufMapping, HttpRangeSource, RangeSource,
};

const GGUF_REV: &str = "b3f2ae5d73cd3c91bf40e8ba752c648aa05dffdf";
const MLX_REV: &str = "d0ca73dcfa70ce18733754a492cdc1fb0123262d";
const GGUF_FILES: [&str; 2] = ["LFM3-1.7B-Q8_0.gguf", "LFM3-1.7B-Q4_K_M.gguf"];

fn gguf_url(file: &str) -> String {
    format!("https://huggingface.co/LiquidAI/LFM3-1.7B-GGUF/resolve/{GGUF_REV}/{file}")
}

#[test]
#[ignore = "network: reads two GGUF headers and the MLX config"]
fn both_sources_derive_the_baseline() {
    let baseline = model_io::lfm3_1_7b();

    let config_url =
        format!("https://huggingface.co/LiquidAI/LFM3-1.7B-MLX-4bit/resolve/{MLX_REV}/config.json");
    let response =
        reqwest::blocking::get(&config_url).unwrap_or_else(|e| panic!("GET {config_url}: {e}"));
    assert!(
        response.status().is_success(),
        "GET {config_url}: HTTP {}",
        response.status()
    );
    let text = response.text().expect("config text");
    let mlx = parse_lfm3_config(&text).expect("parses");
    assert_eq!(mlx, baseline, "the MLX config disagrees with the baseline");

    for file in GGUF_FILES {
        let url = gguf_url(file);
        let source = HttpRangeSource::new(&url);
        let header = fetch_gguf_header(&source).expect("fetch GGUF header");
        assert_eq!(header.architecture(), Some("lfm3"), "{file}");
        let gguf = arch_from_gguf(&header).expect("derives");
        assert_eq!(gguf, mlx, "{file}: GGUF and MLX derive different configs");

        let mut freqs = None;
        for name in header.tensors.keys() {
            match map_gguf_name(name, ModelFamily::Lfm3) {
                Ok(GgufMapping::ZeroTailRopeFreqs) => freqs = Some(name.clone()),
                Ok(GgufMapping::Resident(_)) => {}
                other => panic!("{file}: {name} maps to {other:?}"),
            }
        }
        let name = freqs.unwrap_or_else(|| panic!("{file}: no rope_freqs.weight"));
        let (start, end) = header
            .absolute_range(&name)
            .expect("present")
            .expect("range");
        let bytes = source.read_range(start, end).expect("rope_freqs bytes");
        lfm3_rope_freqs_are_zero_tail(&bytes, &gguf)
            .unwrap_or_else(|e| panic!("{file}: rope_freqs is not the zero-tail vector: {e}"));
        println!(
            "{file}: {} tensors, arch == baseline, rope_freqs verified",
            header.tensors.len()
        );
    }
}
