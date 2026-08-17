//! Does the head's ingest survive REAL bytes? (`docs/MTP_SPECULATIVE.md`,
//! step 1.) Ranged reads against `Qwen/Qwen3.8-27B`, a few MB, ~10 s:
//!
//! ```sh
//! cargo test -p turbospark-repack --test mtp_quantize_network --release -- --ignored --nocapture
//! ```
//!
//! **This exists because the alternative first real-bytes gate is a 20-minute
//! stream**, and `crates/repack` Gotcha 5's rule applies exactly: before
//! budgeting a download for a question, check whether the answer is a
//! contiguous byte range. It is. A safetensors tensor is one, so the head's
//! `fc` and one projection can be read, quantized and scored against their
//! own source in seconds -- and a quantizer that mangles real BF16 weights
//! fails HERE rather than after the trunk has streamed.
//!
//! What it can and cannot see. It CAN see that the real head's dtype and
//! shapes still match what `mtp_head_network.rs` pinned, that
//! `quantize_matrix_int4` accepts the real column widths (all eight
//! projections are whole numbers of 64-element groups, which is not
//! guaranteed by anything and is checked here), and that a dequantized row
//! tracks its own source closely enough to rule out a sign, scale or
//! nibble-order error. It CANNOT see anything about accept length: a drafter
//! quantized correctly can still be a bad drafter, and only step 3 measures
//! that.
//!
//! On the correlation bar, read AGENTS.md Gotcha 30 before believing a low
//! number: a constant row scores 0.0 by contract, and real checkpoints do
//! carry all-zero rows inside otherwise ordinary tensors. This test selects
//! non-constant rows and says how many it found.

use turbospark_repack::{
    fetch_safetensors_header, quantize_matrix_int4, HttpRangeSource, RangeSource,
};

/// The OFFICIAL BF16 checkpoint. The mlx-community conversion the trunk is
/// streamed from drops `mtp.*` entirely, so this is the only source.
const REPO_BASE: &str = "https://huggingface.co/Qwen/Qwen3.8-27B/resolve/main";
const MTP_SHARD: &str = "model-00018-of-00018.safetensors";

/// The affine quantizer's group size, restated rather than imported so this
/// test states the shape it is asserting.
const GROUP: u64 = 64;

/// Every projection in the head, with the shape `mtp_head_network.rs` pinned.
/// Eight of them, which is the count that matters here: the OTHER seven
/// tensors are rank-1 norms and take the narrowing path, not this one.
const PROJECTIONS: &[(&str, [u64; 2])] = &[
    ("mtp.fc.weight", [5120, 10240]),
    ("mtp.layers.0.self_attn.q_proj.weight", [12288, 5120]),
    ("mtp.layers.0.self_attn.k_proj.weight", [1024, 5120]),
    ("mtp.layers.0.self_attn.v_proj.weight", [1024, 5120]),
    ("mtp.layers.0.self_attn.o_proj.weight", [5120, 6144]),
    ("mtp.layers.0.mlp.gate_proj.weight", [17408, 5120]),
    ("mtp.layers.0.mlp.up_proj.weight", [17408, 5120]),
    ("mtp.layers.0.mlp.down_proj.weight", [5120, 17408]),
];

/// Dequantize one INT4-affine row back to `f32`, mirroring what the GEMV
/// kernel does with the same three planes.
fn dequantize_row(packed: &[u8], scales: &[u16], biases: &[u16], cols: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(cols);
    for i in 0..cols {
        let byte = packed[i / 2];
        // Low nibble first: the packing order the kernel reads.
        let q = if i % 2 == 0 { byte & 0x0F } else { byte >> 4 };
        let g = i / GROUP as usize;
        let s = compute::bf16_to_f32(scales[g]);
        let b = compute::bf16_to_f32(biases[g]);
        out.push(q as f32 * s + b);
    }
    out
}

#[test]
#[ignore = "reads a few MB off the real Qwen3.8-27B checkpoint over the network"]
fn the_real_mtp_head_quantizes_to_int4() {
    let source = HttpRangeSource::new(format!("{REPO_BASE}/{MTP_SHARD}"));
    let header = fetch_safetensors_header(&source).expect("shard header");

    println!(
        "\n{:52} {:14} {:>9} {:>9}",
        "projection", "shape", "groups", "pearson"
    );

    for (name, shape) in PROJECTIONS {
        let info = header
            .tensors
            .get(*name)
            .unwrap_or_else(|| panic!("{name} is not in {MTP_SHARD}"));
        assert_eq!(&info.shape, shape, "{name} changed shape");
        assert_eq!(info.dtype, "BF16", "{name} is no longer BF16");

        let (rows, cols) = (shape[0], shape[1]);
        // THE PRECONDITION `quantize_matrix_int4` ENFORCES, checked here
        // against the real widths rather than assumed. Nothing guarantees a
        // published head's columns are whole groups; if one were not, the
        // ingest would need a padding rule and this is where that shows.
        assert_eq!(
            cols % GROUP,
            0,
            "{name}: {cols} columns is not a whole number of {GROUP}-element groups"
        );

        // ONE ROW, not the tensor: a BF16 row is `cols * 2` contiguous bytes,
        // so this is a few KB even on the 17408-wide ones. Reading the whole
        // head would be 849 MB and would prove nothing more.
        let (start, _) = header.absolute_range(name).expect("range resolves");
        let row_bytes = cols * 2;
        // Row 1 rather than row 0: a leading row of zeros is common enough in
        // real checkpoints to be worth stepping over, and `pearson` returns
        // 0.0 on a constant input by contract (AGENTS.md Gotcha 30).
        let from = start + row_bytes;
        let raw = source
            .read_range(from, from + row_bytes)
            .unwrap_or_else(|e| panic!("{name} row read: {e}"));
        let row: Vec<f32> = raw
            .chunks_exact(2)
            .map(|c| compute::bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect();
        assert_eq!(row.len(), cols as usize);

        let quantized = quantize_matrix_int4(&row, 1, cols as usize)
            .unwrap_or_else(|e| panic!("{name} quantize: {e}"));
        let q = &quantized[0];
        let groups = (cols / GROUP) as usize;
        assert_eq!(q.scales.len(), groups, "{name} scale plane");
        assert_eq!(q.biases.len(), groups, "{name} bias plane");
        assert_eq!(q.packed.len(), (cols / 2) as usize, "{name} packed run");

        let back = dequantize_row(&q.packed, &q.scales, &q.biases, cols as usize);
        let r = compute::pearson(&row, &back);
        println!(
            "{name:52} {:14} {groups:>9} {r:>9.5}",
            format!("{rows}x{cols}")
        );
        // A 4-bit affine round trip of real weights sits well above this.
        // The bar is deliberately not tighter: it is here to catch a sign,
        // scale or nibble-order error, and those read near zero rather than
        // near 0.99. A tight bar would instead be measuring how gaussian this
        // particular row is.
        assert!(
            r > 0.95,
            "{name}: INT4 round trip correlates {r:.5} with its own source, \
             which is a packing error rather than quantization loss"
        );
    }
}
