#![cfg(target_os = "macos")]
//! **A0: can the routed experts be read IN PLACE out of an `mmap` instead of
//! `pread`-copied into a pinned slot?**
//!
//! Today the MoE decode path is
//!
//! ```text
//! packed_experts/layer_NN.bin  --pread memcpy-->  AlignedSlot  --wrap-->  MTLBuffer
//!    (already in page cache)      3.86 ms/token     (pinned)      no copy
//! ```
//!
//! and the middle step is the only reason the slot cache exists. Everything
//! needed to delete it is already here: `moe_decode.rs`'s own header says the
//! kernels read expert weights in place from "streamer slots or any other page
//! of memory", `RoutedBlobsBuffer::bind` already takes `(buffer, offset)`
//! pairs, `wrap_page_aligned_no_copy` already wraps arbitrary page-aligned
//! memory, and the experts are already stored one file per layer.
//!
//! **The whole idea turns on one number, and this file measures it.**
//! `AGENTS.md` Gotcha 19 says `newBufferWithBytesNoCopy` makes Metal PIN the
//! range, which would make a 12 GB mapping catastrophic. Gotcha 40 measured a
//! 4.07 GiB resident region reading 684 MiB of `phys_footprint`, and the
//! 2026-08-18 recommendation-engine work concluded outright that `counted`
//! CANNOT include the resident core. Those cannot both be right.
//!
//! So this REPORTS the footprint at four stages and asserts only what has to
//! hold for the report to mean anything. It deliberately does not assert an
//! outcome: asserting one would be assuming the answer it exists to find.
//! What it does assert is that the measurement is not degenerate -- a mapping
//! of the wrong size, or a GEMV that reads zeros or NaNs, would report a tidy
//! footprint table describing nothing (Gotcha 57's rule, and Gotcha 59's:
//! a non-finite row scores perfectly on the instruments that would check it).
//!
//! `#[ignore]`d and env-gated like every other real-install test here; an
//! unset var SKIPS with a note.

use std::path::PathBuf;

use gpu::{
    encode_dequant_int4_gemv_resident, read_buffer_f16, wrap_page_aligned_no_copy,
    Int4ResidentMatrix, MetalContext,
};
use turbospark_bench::memory::AppMemorySampler;

/// Which install to probe. Any MoE install works; the numbers in
/// `docs/EXPERT_RESIDENCY.md` are Gemma 4 26B-A4B's.
const INSTALL_VAR: &str = "TURBOSPARK_PROBE_INSTALL_DIR";

/// `load_packed_experts_layout`'s size cap. `layout.json` is a few hundred KB
/// on a 128-expert model.
const LAYOUT_MAX_BYTES: u64 = 64 * 1024 * 1024;

fn mib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

/// Signed MiB delta, because the interesting outcome is "it did not move" and
/// a `u64` subtraction the wrong way round would print a nonsense 1.8e13.
fn delta_mib(after: u64, before: u64) -> f64 {
    (after as f64 - before as f64) / (1024.0 * 1024.0)
}

#[test]
#[ignore = "needs a real MoE install; set TURBOSPARK_PROBE_INSTALL_DIR"]
fn mapping_the_expert_table_is_measured_against_phys_footprint() {
    let Ok(dir) = std::env::var(INSTALL_VAR) else {
        eprintln!("{INSTALL_VAR} unset; skipping");
        return;
    };
    let dir = PathBuf::from(shellexpand_tilde(&dir));
    // `load` takes the INSTALL dir and joins `packed_experts/layout.json`
    // itself; the layer basenames it returns are relative to that subdir.
    let experts_dir = dir.join("packed_experts");
    let layout = model_io::load_packed_experts_layout(&dir, LAYOUT_MAX_BYTES)
        .expect("packed_experts/layout.json should load");

    let mut sampler = AppMemorySampler::new();
    let baseline = sampler.sample().expect("phys_footprint should read");

    // --- stage 1: mmap every layer file -------------------------------
    //
    // Kept in a Vec that outlives every Metal buffer below: the wrap
    // creates no deallocator and ALIASES the mapping, so the mapping must
    // outlive the buffer (`resident_metal.rs`'s module docs).
    let mut mappings = Vec::with_capacity(layout.num_layers);
    let mut mapped_bytes = 0u64;
    for layer in &layout.layers {
        let path = experts_dir.join(&layer.file);
        let len = std::fs::metadata(&path)
            .unwrap_or_else(|e| panic!("{} should stat: {e}", path.display()))
            .len();
        let mapping = model_io::ResidentBuffer::map(&path, 0, len)
            .unwrap_or_else(|e| panic!("{} should map: {e:?}", path.display()));
        mapped_bytes += len;
        mappings.push(mapping);
    }
    let after_map = sampler.sample().expect("phys_footprint should read");

    // The measurement is only about the expert table if the expert table is
    // what got mapped. A layout that disagreed with the files on disk would
    // still print a footprint table.
    let expected = layout
        .layers
        .iter()
        .map(|l| l.expert_stride * layout.experts_per_layer as u64)
        .sum::<u64>();
    assert!(
        mapped_bytes >= expected,
        "mapped {mapped_bytes} bytes but the layout describes {expected}; \
         the probe is not measuring the expert table"
    );
    assert!(
        mapped_bytes > 1024 * 1024 * 1024,
        "expert table is only {:.1} MiB; this install is too small for the \
         question (the slot cache it would replace is sized in GiB)",
        mib(mapped_bytes)
    );

    // --- stage 2: wrap each mapping in a Metal buffer ------------------
    //
    // THE DECISIVE STAGE. One buffer per layer rather than one for the whole
    // table, which is what makes `maxBufferLength` a non-question: a layer
    // file is ~410 MiB on Gemma 4 where the table is 12.3 GB.
    let mut context = MetalContext::new().expect("Metal device available on this machine");
    let mut buffers = Vec::with_capacity(mappings.len());
    for mapping in &mappings {
        let bytes = mapping.mapped_bytes();
        let buffer = wrap_page_aligned_no_copy(context.device(), bytes.as_ptr(), bytes.len())
            .expect("newBufferWithBytesNoCopy should accept a page-aligned mmap");
        buffers.push(buffer);
    }
    let after_wrap = sampler.sample().expect("phys_footprint should read");

    // --- stage 3: read one expert through a real kernel ----------------
    //
    // Read-only on the weights BY CONSTRUCTION: the GEMV writes to `y` and
    // reads the matrix. Nothing here may write to a mapping -- it is a
    // read-only file map of the user's install.
    let entry = layout.expert(0, 0);
    let gate = entry.sub_tensors.get("gate").expect("expert 0 has a gate");
    let gate_scales = entry
        .sub_tensors
        .get("gate_scales")
        .expect("expert 0 has gate scales");
    let gate_biases = entry
        .sub_tensors
        .get("gate_biases")
        .expect("expert 0 has gate biases");
    // An AFFINE blob has nine sub-tensors and a GGUF one three, and that is
    // the documented way to tell them apart -- not the dtype string, which
    // the affine writers spell `u32` for a packed INT4 run rather than
    // `int4` (`crates/runtime` Gotcha 10). This probe dispatches the INT4
    // GEMV, so a GGUF install needs its own arm rather than a silently
    // wrong reading.
    assert_eq!(
        entry.sub_tensors.len(),
        9,
        "expected an affine expert (9 sub-tensors); this install has {} and \
         needs its own dispatch here",
        entry.sub_tensors.len()
    );

    // Shapes come from the install's own manifest rather than being typed in:
    // `gate` is [moe_intermediate, hidden] with hidden the reduction dim, so
    // this probe is not pinned to one checkpoint's numbers.
    let arch = repack::peek_manifest_arch(&dir).expect("manifest should peek");
    let (rows, cols) = (
        arch.moe_intermediate_size as usize,
        arch.hidden_size as usize,
    );

    // Cross-check the shapes against the blob's own byte counts before using
    // them: a packed INT4 run is one nibble per element, and the scale plane
    // carries one bf16 per group per row. If either disagrees, the layout is
    // not the matrix this dispatch is about to read.
    assert_eq!(
        gate.size as usize,
        rows * cols / 2,
        "gate is {} bytes, not the {}x{} INT4 matrix the manifest implies",
        gate.size,
        rows,
        cols
    );
    let groups_per_row = gate_scales.size as usize / (2 * rows);
    assert_eq!(
        cols / groups_per_row,
        64,
        "expected affine group 64; got {} from a {}-byte scale plane",
        cols / groups_per_row,
        gate_scales.size
    );

    // FP16 1.0 by its bit pattern, so this file needs no `half` dependency
    // for one constant. `read_buffer_f16` still hands back `half::f16`, whose
    // `to_f32` is inherent and needs no import.
    const FP16_ONE: u16 = 0x3C00;
    let x = context.new_buffer_with_data(&vec![FP16_ONE; cols]);
    let y = context.new_output_buffer((rows * 2) as u64);
    let base = entry.offset;
    gpu::autorelease_pool(|| {
        let pass = context.begin_pass();
        encode_dequant_int4_gemv_resident(
            &mut context,
            &pass,
            &Int4ResidentMatrix {
                buffer: &buffers[0],
                weights_offset: base + gate.offset,
                scales_offset: base + gate_scales.offset,
                biases_offset: base + gate_biases.offset,
                rows,
                cols,
            },
            (&x, 0),
            (&y, 0),
        )
        .expect("GEMV should encode against a mapped expert");
        pass.commit_and_wait();
    });
    let out = read_buffer_f16(&y, 0, rows);

    // Gotcha 59: a non-finite row is the outcome that scores perfectly on
    // every instrument that would check it, so it is refused by name here
    // rather than reported. An all-zero row is the other degenerate answer --
    // a mapping that read nothing would look exactly like a working one.
    assert!(
        out.iter().all(|v| v.to_f32().is_finite()),
        "the GEMV over mapped expert bytes produced a non-finite row; the GPU \
         did not read what the host mapped"
    );
    assert!(
        out.iter().any(|v| v.to_f32() != 0.0),
        "the GEMV over mapped expert bytes produced all zeros; a mapping that \
         faulted in nothing reads exactly like this"
    );
    let after_one_read = sampler.sample().expect("phys_footprint should read");

    // --- stage 4: touch every layer ------------------------------------
    //
    // One expert per layer, so the sweep spans all 30 mappings rather than
    // re-reading the pages stage 3 already faulted in.
    gpu::autorelease_pool(|| {
        for (layer_index, buffer) in buffers.iter().enumerate() {
            let entry = layout.expert(layer_index, 0);
            let g = entry.sub_tensors.get("gate").expect("gate");
            let gs = entry.sub_tensors.get("gate_scales").expect("gate scales");
            let gb = entry.sub_tensors.get("gate_biases").expect("gate biases");
            let pass = context.begin_pass();
            encode_dequant_int4_gemv_resident(
                &mut context,
                &pass,
                &Int4ResidentMatrix {
                    buffer,
                    weights_offset: entry.offset + g.offset,
                    scales_offset: entry.offset + gs.offset,
                    biases_offset: entry.offset + gb.offset,
                    rows,
                    cols,
                },
                (&x, 0),
                (&y, 0),
            )
            .expect("GEMV should encode");
            pass.commit_and_wait();
        }
    });
    let after_sweep = sampler.sample().expect("phys_footprint should read");

    println!("\n=== A0: mapped expert residency, phys_footprint ===");
    println!("install                {}", dir.display());
    println!(
        "expert table           {:.1} MiB across {} layer files",
        mib(mapped_bytes),
        layout.num_layers
    );
    println!("baseline               {:>10.1} MiB", mib(baseline));
    println!(
        "after mmap             {:>10.1} MiB  ({:+.1})",
        mib(after_map),
        delta_mib(after_map, baseline)
    );
    println!(
        "after Metal wrap       {:>10.1} MiB  ({:+.1})   <-- decisive",
        mib(after_wrap),
        delta_mib(after_wrap, after_map)
    );
    println!(
        "after one expert read  {:>10.1} MiB  ({:+.1})",
        mib(after_one_read),
        delta_mib(after_one_read, after_wrap)
    );
    println!(
        "after 1 expert x {:<3}   {:>10.1} MiB  ({:+.1})",
        layout.num_layers,
        mib(after_sweep),
        delta_mib(after_sweep, after_one_read)
    );
    println!(
        "total growth           {:>10.1} MiB against {:.1} MiB mapped",
        delta_mib(after_sweep, baseline),
        mib(mapped_bytes)
    );

    let growth = after_sweep.saturating_sub(baseline);
    let verdict = if growth * 2 > mapped_bytes {
        "PINNED: the wrap charges the mapping. Gotcha 19 is right and mapped \
         residency is a dead end for a table this size."
    } else if growth > mapped_bytes / 20 {
        "PARTIAL: growth tracks the pages touched, not the mapping. Mapped \
         residency works but is memory-sensitive; `auto` must budget for it."
    } else {
        "NOT CHARGED: the mapping is free on this counter. Gotcha 19 is wrong \
         as stated and mapped residency can delete the slot cache outright."
    };
    println!("\nverdict: {verdict}\n");
}

/// `~` expansion, because every install path in this repo's docs is written
/// with one and `PathBuf::from` does not expand it (the env var arrives
/// unexpanded when set inside a test harness rather than by a shell).
fn shellexpand_tilde(raw: &str) -> String {
    match raw.strip_prefix("~/") {
        Some(rest) => match std::env::var("HOME") {
            Ok(home) => format!("{home}/{rest}"),
            Err(_) => raw.to_string(),
        },
        None => raw.to_string(),
    }
}
