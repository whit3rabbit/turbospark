//! Do the head's bytes ON DISK match the published checkpoint?
//!
//! `mtp_quantize_network.rs` next door answers a NARROWER question than its
//! name suggests: it quantizes a freshly-fetched row and dequantizes it with
//! its own helper, so it validates `quantize_matrix_int4` against itself. That
//! is a self-consistency check, and AGENTS.md Gotcha 48 is explicit that a
//! self-consistent check proves nothing about a convention -- it passes
//! whenever the writer and the reader share a mistake.
//!
//! This one reads the INSTALLED tensor, dequantizes it the way the GEMV
//! kernel does, and correlates against the same row of the published BF16
//! shard. It is the only test that can see a walk which wrote well-formed,
//! distinct, non-zero bytes that are not the right bytes -- which is the exact
//! shape of the failure this head has already had once (`docs/MTP_SPECULATIVE.md`
//! step 1: the streamed writer classified `mtp.*` correctly and then never
//! read it).
//!
//! ```sh
//! TURBOSPARK_MTP_INSTALL_DIR=~/models/qwen38-27b-mtp.gturbo \
//!   cargo test -p turbospark-repack --test mtp_install_fidelity_network --release -- --ignored --nocapture
//! ```

use turbospark_repack::{fetch_safetensors_header, HttpRangeSource, RangeSource};

const REPO_BASE: &str = "https://huggingface.co/Qwen/Qwen3.8-27B/resolve/main";
const MTP_SHARD: &str = "model-00018-of-00018.safetensors";
const GROUP: usize = 64;

/// The four rows checked. `fc` first because it is the head's own tensor and
/// the one no trunk layer can stand in for.
const CHECKED: &[(&str, [u64; 2])] = &[
    ("mtp.fc.weight", [5120, 10240]),
    ("mtp.layers.0.self_attn.q_proj.weight", [12288, 5120]),
    ("mtp.layers.0.mlp.gate_proj.weight", [17408, 5120]),
    ("mtp.layers.0.mlp.down_proj.weight", [5120, 17408]),
];

/// Exactly what `dequant_int4_gemv_simd` does with the three planes: low
/// nibble first, one BF16 scale and bias per 64-element group.
fn dequantize_row(packed: &[u8], scales: &[u16], biases: &[u16], cols: usize) -> Vec<f32> {
    (0..cols)
        .map(|i| {
            let byte = packed[i / 2];
            let q = if i % 2 == 0 { byte & 0x0F } else { byte >> 4 };
            let g = i / GROUP;
            q as f32 * compute::bf16_to_f32(scales[g]) + compute::bf16_to_f32(biases[g])
        })
        .collect()
}

#[test]
#[ignore = "needs the install via TURBOSPARK_MTP_INSTALL_DIR plus a few MB of network"]
fn the_installed_head_matches_the_published_checkpoint() {
    let dir = std::path::PathBuf::from(
        std::env::var_os("TURBOSPARK_MTP_INSTALL_DIR").expect("TURBOSPARK_MTP_INSTALL_DIR"),
    );
    let path = dir.join("model_weights.bin");
    let index = model_io::load_resident_index(&path).expect("resident index");
    let source = HttpRangeSource::new(format!("{REPO_BASE}/{MTP_SHARD}"));
    let header = fetch_safetensors_header(&source).expect("shard header");

    // Row 1 rather than row 0: a leading zero row is common in real
    // checkpoints and `pearson` returns 0.0 on a constant input by contract
    // (AGENTS.md Gotcha 30).
    const ROW: usize = 1;
    println!("\n{:52} {:>12} {:>10}", "tensor", "cols", "pearson");
    let mut bad = Vec::new();
    for (name, shape) in CHECKED {
        let cols = shape[1] as usize;
        let groups = cols / GROUP;

        // -- The published side.
        let info = header.tensors.get(*name).expect("tensor in shard");
        assert_eq!(&info.shape, shape, "{name} changed shape");
        let (start, _) = header.absolute_range(name).expect("range resolves");
        let from = start + (ROW * cols * 2) as u64;
        let raw = source
            .read_range(from, from + (cols * 2) as u64)
            .expect("row read");
        let want: Vec<f32> = raw
            .chunks_exact(2)
            .map(|c| compute::bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect();

        // -- The installed side, read exactly as the kernel addresses it.
        let e = index.entries.get(*name).expect("tensor in install");
        let bytes = std::fs::read(&path).expect("read weights");
        let packed_at = e.file_offset as usize + ROW * cols / 2;
        let packed = &bytes[packed_at..packed_at + cols / 2];
        let plane = |off: u64| -> Vec<u16> {
            let at = off as usize + ROW * groups * 2;
            bytes[at..at + groups * 2]
                .chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect()
        };
        let got = dequantize_row(packed, &plane(e.scale_offset), &plane(e.bias_offset), cols);

        let r = compute::pearson(&want, &got);
        println!("{name:52} {cols:>12} {r:>10.5}");
        if r < 0.95 {
            bad.push((*name, r));
        }
    }
    assert!(
        bad.is_empty(),
        "the installed head does not match the published checkpoint: {bad:?}\n\
         A NEGATIVE or near-zero correlation here means the install's bytes are wrong \
         (walk or quantizer); a high one means the bytes are right and any bad drafting \
         is in the decode flow instead."
    );
}
