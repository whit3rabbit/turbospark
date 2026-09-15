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
//! THE DAMAGE IS ONE QUANTIZATION LEVEL, applied to a strided subset of the
//! weight ranges identified by `packed_experts/layout.json`. XOR `0x01` into
//! a byte of an int4 weight flips the low nibble's least
//! significant bit, moving that weight by exactly one of its sixteen
//! levels: the smallest error a requantization can make, and the direction
//! a sub-4-bit config errs in. BF16 scales, biases, and padding are excluded,
//! so the stimulus cannot pass by severely corrupting a companion tensor.
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
//!   TURBOSPARK_GEMMA4_INSTALL_DIR=~/models/gemma4.gturbo \
//!     cargo test -p turbospark-bench --test quality_sensitivity --release -- --ignored --nocapture

mod quality_common;

use std::fs::OpenOptions;
use std::io::{Read, Seek, SeekFrom, Write};
use std::ops::Range;
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
use std::path::{Component, Path, PathBuf};

/// Damage one 4 KiB page in every `DAMAGE_PAGE_STRIDE`.
///
/// Page selection remains relative to each layer file, but only the portions
/// that overlap packed INT4 weight tensors are written. The previous measured
/// response also changed BF16 scales and biases and is intentionally not used
/// as evidence for this corrected stimulus.
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
    std::env::var_os("TURBOSPARK_GEMMA4_INSTALL_DIR").map(PathBuf::from)
}

#[test]
#[ignore = "needs a real ~14.6 GB Gemma 4 .gturbo install (TURBOSPARK_GEMMA4_INSTALL_DIR)"]
fn perplexity_moves_when_routed_experts_are_degraded() {
    let Some(dir) = install_dir() else {
        eprintln!(
            "quality_sensitivity: TURBOSPARK_GEMMA4_INSTALL_DIR is not set; skipping. \
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
    let target = std::env::temp_dir().join("turbospark-quality-sensitivity-clone");
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
    let layout = model_io::load_packed_experts_layout(
        install,
        model_io::PACKED_EXPERTS_LAYOUT_DEFAULT_MAX_BYTES,
    )
    .expect("packed_experts/layout.json should describe the damage ranges");
    let mut files = 0usize;
    let mut pages = 0u64;
    let install = install
        .canonicalize()
        .expect("the cloned install directory should resolve");
    let expert_dir = install
        .join("packed_experts")
        .canonicalize()
        .expect("the cloned packed_experts directory should resolve");
    assert_eq!(
        expert_dir.parent(),
        Some(install.as_path()),
        "packed_experts must resolve directly beneath the cloned install"
    );

    for layer in &layout.layers {
        let mut components = Path::new(&layer.file).components();
        assert!(
            matches!(components.next(), Some(Component::Normal(_))) && components.next().is_none(),
            "expert blob path must be a basename: {}",
            layer.file
        );
        let path = expert_dir
            .join(&layer.file)
            .canonicalize()
            .expect("expert blob should resolve in the clone");
        assert_eq!(
            path.parent(),
            Some(expert_dir.as_path()),
            "expert blob must resolve directly beneath the cloned packed_experts directory"
        );
        let metadata = path.metadata().expect("blob metadata");
        assert!(metadata.is_file(), "expert blob should be a regular file");
        let len = metadata.len();
        let mut weight_ranges = Vec::new();
        for expert in &layer.experts {
            assert!(
                expert.size <= layer.expert_stride,
                "expert {} exceeds its declared stride in {}",
                expert.expert,
                layer.file
            );
            let expert_end = expert
                .offset
                .checked_add(expert.size)
                .expect("expert range should not overflow");
            assert!(
                expert_end <= len,
                "expert {} exceeds the bounds of {}",
                expert.expert,
                layer.file
            );
            for (role, tensor) in &expert.sub_tensors {
                if matches!(role.as_str(), "gate" | "up" | "down") && tensor.dtype == "u32" {
                    let tensor_end = tensor
                        .offset
                        .checked_add(tensor.size)
                        .expect("tensor range should not overflow");
                    assert!(
                        tensor_end <= expert.size,
                        "{role} tensor exceeds expert {} in {}",
                        expert.expert,
                        layer.file
                    );
                    let start = expert
                        .offset
                        .checked_add(tensor.offset)
                        .expect("tensor file offset should not overflow");
                    let end = start
                        .checked_add(tensor.size)
                        .expect("tensor file range should not overflow");
                    assert!(
                        end <= len,
                        "{role} tensor exceeds the bounds of {}",
                        layer.file
                    );
                    weight_ranges.push(start..end);
                }
            }
        }
        assert_eq!(
            weight_ranges.len(),
            layout.experts_per_layer * 3,
            "{} should contain exactly three packed INT4 weight ranges per expert",
            layer.file
        );
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .custom_flags(libc::O_NOFOLLOW)
            .open(&path)
            .expect("expert blob should be writable in the clone");
        let opened_metadata = file.metadata().expect("opened blob metadata");
        assert!(
            opened_metadata.is_file()
                && opened_metadata.len() == len
                && opened_metadata.dev() == metadata.dev()
                && opened_metadata.ino() == metadata.ino(),
            "expert blob changed while it was being validated"
        );
        let mut page = vec![0u8; PAGE_BYTES];
        let mut offset = 0u64;
        while offset + PAGE_BYTES as u64 <= len {
            file.seek(SeekFrom::Start(offset)).expect("seek");
            file.read_exact(&mut page).expect("read page");
            let changed = damage_page(&mut page, offset, &weight_ranges);
            if changed {
                file.seek(SeekFrom::Start(offset)).expect("seek back");
                file.write_all(&page).expect("write page");
                pages += 1;
            }
            offset += DAMAGE_PAGE_STRIDE * PAGE_BYTES as u64;
        }
        file.sync_all().expect("flush the damaged blob");
        files += 1;
    }
    (files, pages)
}

fn damage_page(page: &mut [u8], offset: u64, weight_ranges: &[Range<u64>]) -> bool {
    let page_end = offset + page.len() as u64;
    let mut changed = false;
    for range in weight_ranges {
        let start = range.start.max(offset);
        let end = range.end.min(page_end);
        if start >= end {
            continue;
        }
        for byte in &mut page[(start - offset) as usize..(end - offset) as usize] {
            *byte ^= 0x01;
        }
        changed = true;
    }
    changed
}

#[cfg(test)]
mod tests {
    use super::damage_page;

    #[test]
    fn damage_page_changes_only_weight_intersections() {
        let mut page = [0u8; 16];
        assert!(damage_page(&mut page, 100, &[98..104, 108..112, 116..120]));
        assert_eq!(page, [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]);
    }

    #[test]
    fn damage_page_ignores_pages_outside_weight_ranges() {
        let mut page = [0u8; 8];
        assert!(!damage_page(&mut page, 100, &[0..100, 108..120]));
        assert_eq!(page, [0; 8]);
    }
}
