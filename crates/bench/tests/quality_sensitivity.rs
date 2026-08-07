#![cfg(target_os = "macos")]
//! Does the quality gate's perplexity number actually MOVE when the model
//! gets worse? ROADMAP Phase Q's sensitivity demonstration.
//!
//! WHY THIS EXISTS. `quality_gate.rs` freezes a perplexity and fails the
//! build if it drifts more than `PERPLEXITY_REL_TOLERANCE`. That is only
//! worth anything if real damage produces a drift bigger than the
//! tolerance, and until this test nothing in the repo had ever been run
//! against a deliberately degraded model: the gate's sensitivity was
//! asserted by construction. Phase S drops routed experts below 4 bits, so
//! the specific thing that must be detectable is a SMALL, UNIFORM,
//! quantization-shaped perturbation of the routed experts -- not a broken
//! model, which every existing smoke already catches.
//!
//! THE DAMAGE IS ONE QUANTIZATION LEVEL, applied to a strided subset. XOR
//! `0x01` into a byte of an int4 blob flips the low nibble's least
//! significant bit, moving that weight by exactly one of its sixteen
//! levels: the smallest error a requantization can make, and the direction
//! a sub-4-bit config errs in. It cannot produce a NaN or an infinity even
//! if it lands on an FP16 scale, because bit 0 of either scale byte is a
//! mantissa bit -- which matters, since a NaN would make this test pass for
//! the wrong reason (a destroyed model, not a degraded one).
//!
//! ONLY `packed_experts/` IS TOUCHED, so the resident core (attention,
//! embeddings, router, shared expert) is bit-identical and the measured
//! move is attributable to routed-expert weights alone.
//!
//! THE INSTALL IS NOT MODIFIED. It is APFS-cloned first (`cp -Rc`, which is
//! `clonefile`: 13 GB in about 8 ms, and only the pages this test writes
//! ever cost disk). The clone is removed on success; a panic leaves it
//! behind at the path printed below, on purpose, so a failure can be
//! investigated rather than swept up.
//!
//! Not run by default:
//!
//!   MREFRUST_GEMMA4_INSTALL_DIR=~/models/gemma4.gturbo \
//!     cargo test -p mrefrust-bench --test quality_sensitivity --release -- --ignored --nocapture

mod quality_common;

use std::fs::OpenOptions;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

/// Damage one 4 KiB page in every `DAMAGE_PAGE_STRIDE`.
///
/// A whole page rather than scattered bytes so the write pattern is
/// page-aligned (the expert layout is too, and a read-modify-write of
/// partial pages would cost far more I/O for the same effect).
///
/// MEASURED RESPONSE on the real Gemma 4 install, 2026-08-07, clean
/// perplexity 37.3105 (also in `docs/BENCHMARKS.md`):
///
/// | stride | expert bytes touched | perplexity | drift |
/// |---|---|---|---|
/// | 8 | 12.5% | 12,249,392 | model destroyed |
/// | 512 (this value) | 0.195% | 51.3597 | +37.7% |
/// | 8192 | 0.0122% | 41.2186 | +10.5% |
/// | 65536 | 0.0015% | 37.5118 | +0.54%, UNDER the gate's 2% band |
///
/// 512 is chosen for margin, not for being the smallest detectable damage:
/// it is 19x the gate's tolerance, so the assertion below cannot flake, and
/// it is still only one weight byte in 512. The 65536 row is the useful
/// bound in the other direction -- damage that small does NOT clear the
/// gate's band, so the gate's floor is somewhere between 0.0015% and
/// 0.0122% of expert bytes at one quantization level.
const DAMAGE_PAGE_STRIDE: u64 = 512;

const PAGE_BYTES: usize = 4096;

/// How much bigger the damaged perplexity must be, as a multiple of the
/// gate's own tolerance.
///
/// The claim being pinned is not "damage is detectable at all" but "damage
/// is detectable with margin", i.e. the gate would fail loudly rather than
/// sit just inside its band. 5x tolerance is 10%.
const MIN_DRIFT_MULTIPLE: f64 = 5.0;

fn install_dir() -> Option<PathBuf> {
    std::env::var_os("MREFRUST_GEMMA4_INSTALL_DIR").map(PathBuf::from)
}

#[test]
#[ignore = "needs a real ~14.6 GB Gemma 4 .gturbo install (MREFRUST_GEMMA4_INSTALL_DIR)"]
fn perplexity_moves_when_routed_experts_are_degraded() {
    let Some(dir) = install_dir() else {
        eprintln!(
            "quality_sensitivity: MREFRUST_GEMMA4_INSTALL_DIR is not set; skipping. \
             Point it at a repacked Gemma 4 .gturbo install to run the test."
        );
        return;
    };

    let clean = quality_common::measure_perplexity(&dir);
    eprintln!("quality_sensitivity: clean perplexity {clean:.4}");

    let damaged_dir = clone_install(&dir);
    eprintln!(
        "quality_sensitivity: cloned to {} (removed on success)",
        damaged_dir.display()
    );
    let (files, pages) = damage_experts(&damaged_dir);
    eprintln!(
        "quality_sensitivity: flipped one quantization level across {pages} pages \
         in {files} expert blobs (1 page in {DAMAGE_PAGE_STRIDE})"
    );

    let damaged = quality_common::measure_perplexity(&damaged_dir);
    let drift = (damaged - clean) / clean;
    eprintln!(
        "quality_sensitivity: damaged perplexity {damaged:.4}, {:.2}% above clean \
         (gate tolerance {:.0}%)",
        drift * 100.0,
        quality_common::PERPLEXITY_REL_TOLERANCE * 100.0
    );

    let floor = quality_common::PERPLEXITY_REL_TOLERANCE * MIN_DRIFT_MULTIPLE;
    assert!(
        drift >= floor,
        "shifting one in {DAMAGE_PAGE_STRIDE} pages of routed-expert weights by \
         a single quantization level moved perplexity only {:.2}% \
         ({damaged:.4} against {clean:.4}), under the {:.0}% this test requires. \
         Either the gate can no longer see quantization damage, or the damage \
         is not reaching the forward pass -- check that the routed path still \
         reads packed_experts/ before trusting any Phase S quality number.",
        drift * 100.0,
        floor * 100.0
    );

    std::fs::remove_dir_all(&damaged_dir).expect("the damaged clone should be removable");
}

/// APFS-clone `dir` into a fresh temp directory and return its path.
///
/// Shells out to `cp -Rc` rather than walking the tree in Rust: `-c` is the
/// `clonefile` flag, std has no binding for it, and copying 13 GB for real
/// would dominate this test's runtime and its disk use.
fn clone_install(dir: &Path) -> PathBuf {
    let target = std::env::temp_dir().join("mrefrust-quality-sensitivity-clone");
    if target.exists() {
        std::fs::remove_dir_all(&target).expect("a leftover clone should be removable");
    }
    let status = std::process::Command::new("cp")
        .arg("-Rc")
        .arg(dir)
        .arg(&target)
        .status()
        .expect("cp should run");
    assert!(
        status.success(),
        "cloning {} failed; a clone needs the source and {} on one APFS volume",
        dir.display(),
        target.display()
    );
    target
}

/// Flip one quantization level in every `DAMAGE_PAGE_STRIDE`-th page of
/// every expert blob. Returns (blobs touched, pages damaged).
fn damage_experts(install: &Path) -> (usize, u64) {
    let mut files = 0usize;
    let mut pages = 0u64;
    let entries = std::fs::read_dir(install.join("packed_experts"))
        .expect("the install should have a packed_experts directory");
    // Sorted so the damage is identical across runs; read_dir order is not
    // specified, and a golden number wants a deterministic input.
    let mut paths: Vec<PathBuf> = entries
        .map(|e| e.expect("directory entry").path())
        .filter(|p| p.extension().is_some_and(|e| e == "bin"))
        .collect();
    paths.sort();
    assert!(!paths.is_empty(), "no expert blobs to damage");

    for path in paths {
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .expect("expert blob should be writable in the clone");
        let len = file.metadata().expect("blob metadata").len();
        let mut page = vec![0u8; PAGE_BYTES];
        let mut offset = 0u64;
        while offset + PAGE_BYTES as u64 <= len {
            file.seek(SeekFrom::Start(offset)).expect("seek");
            file.read_exact(&mut page).expect("read page");
            for byte in page.iter_mut() {
                *byte ^= 0x01;
            }
            file.seek(SeekFrom::Start(offset)).expect("seek back");
            file.write_all(&page).expect("write page");
            pages += 1;
            offset += DAMAGE_PAGE_STRIDE * PAGE_BYTES as u64;
        }
        file.sync_all().expect("flush the damaged blob");
        files += 1;
    }
    (files, pages)
}
