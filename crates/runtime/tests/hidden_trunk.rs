#![cfg(target_os = "macos")]
//! `HiddenTrunk` on a synthetic Fish Drama 3 install, built through the real
//! streamed Drama intake (`turbospark_repack::build_synthetic_fish_drama3_install`)
//! and run on real Metal.
//!
//! Weights are deterministic but untrained, so nothing here asserts on what a
//! hidden state means: only bit identity between paths that must agree,
//! finiteness, and the refusals. Facts and gates: `docs/FISH_DRAMA3_PHASE0.md`.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use turbospark_repack::{build_synthetic_fish_drama3_install, DRAMA3_TRUNK_DIR};
use turbospark_runtime::{
    ExpertCacheSlots, ExpertResidency, HiddenTrunk, RealForwardRunner, TrunkRow,
};

static COUNTER: AtomicU64 = AtomicU64::new(0);

const VOCAB: i64 = 256;
const EXPERTS: i64 = 8;

fn temp_dir() -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!(
        "turbospark-hidden-trunk-{}-{n}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Builds an install and returns (install root, trunk dir, trunk arch).
fn install() -> (PathBuf, PathBuf, model_io::ArchConfig) {
    let root = temp_dir();
    build_synthetic_fish_drama3_install(&root, VOCAB, EXPERTS, "synthetic/fish-drama3")
        .expect("drama install builds");
    let trunk = root.join(DRAMA3_TRUNK_DIR);
    // The manifest's own arch: the parser's attention scale for this
    // fixture's head_dim is not a binary fraction, so the manifest is the
    // one source both sides of the open's exact comparison agree on.
    let arch = turbospark_repack::peek_manifest_arch(&trunk).expect("manifest arch");
    (root, trunk, arch)
}

fn open(trunk: &Path, arch: &model_io::ArchConfig, max_context: usize) -> HiddenTrunk {
    HiddenTrunk::open(
        trunk,
        arch.clone(),
        max_context,
        ExpertCacheSlots::Fixed(16),
        ExpertResidency::Streamed,
    )
    .expect("headless trunk opens")
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

/// Deterministic token ids that cover the table, including 0 and the last.
fn tokens(n: usize) -> Vec<u32> {
    (0..n)
        .map(|i| match i {
            0 => 0,
            1 => VOCAB as u32 - 1,
            _ => ((i as u64 * 2_654_435_761) % VOCAB as u64) as u32,
        })
        .collect()
}

#[test]
fn a_chat_open_of_a_headless_trunk_is_refused_by_name() {
    let (root, trunk, arch) = install();
    let err = RealForwardRunner::open(&trunk, arch.clone())
        .err()
        .expect("a chat open must refuse a marked trunk")
        .to_string();
    assert!(err.contains("headless trunk"), "{err}");

    // Without the marker the trunk still fails closed: it has no head.
    std::fs::remove_file(trunk.join(model_io::trunk_role::TRUNK_ROLE_FILE)).unwrap();
    let err = RealForwardRunner::open(&trunk, arch)
        .err()
        .expect("an unmarked headless trunk has no head to open")
        .to_string();
    assert!(err.contains("lm_head"), "{err}");
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn a_headless_open_requires_the_marker() {
    let (root, trunk, arch) = install();
    std::fs::remove_file(trunk.join(model_io::trunk_role::TRUNK_ROLE_FILE)).unwrap();
    let err = HiddenTrunk::open(
        &trunk,
        arch,
        64,
        ExpertCacheSlots::Fixed(16),
        ExpertResidency::Streamed,
    )
    .err()
    .expect("no marker, no headless open")
    .to_string();
    assert!(err.contains("trunk_role.json"), "{err}");
    let _ = std::fs::remove_dir_all(&root);
}

/// The gate for the host-row seam: a row that IS the table row must give
/// the same hidden state, bit for bit, as the table lookup itself. Any
/// difference means the host write and the GPU gather disagree on rounding
/// or the write landed after the pass read `scratch.x`.
#[test]
fn token_rows_and_their_embedding_rows_give_bit_identical_hidden_states() {
    let (root, trunk, arch) = install();
    let mut by_token = open(&trunk, &arch, 128);
    let mut by_row = open(&trunk, &arch, 128);
    let hidden = by_token.hidden_size();
    let mut row = vec![0.0f32; hidden];
    let mut a = vec![0.0f32; hidden];
    let mut b = vec![0.0f32; hidden];
    for (step, id) in tokens(64).into_iter().enumerate() {
        by_token.step(TrunkRow::Token(id), &mut a).unwrap();
        by_row.embedding_row(id, &mut row).unwrap();
        by_row.step(TrunkRow::Embedding(&row), &mut b).unwrap();
        assert!(a.iter().all(|v| v.is_finite()), "step {step}");
        assert_eq!(bits(&a), bits(&b), "step {step}, token {id}");
    }
    assert_eq!(by_token.position(), 64);
    let _ = std::fs::remove_dir_all(&root);
}

/// Chunked prefill (the default) against stepping row by row and against
/// the per-row prefill: 37 rows cross two micro-batch edges (16 rows each)
/// with token and host rows mixed, every hidden state must agree bit for
/// bit, and so must the next step, which reads the state prefill left.
#[test]
fn prefill_matches_stepping_row_by_row() {
    let (root, trunk, arch) = install();
    let mut stepped = open(&trunk, &arch, 64);
    let mut chunked = open(&trunk, &arch, 64);
    let mut per_row = open(&trunk, &arch, 64);
    per_row.set_chunked_prefill(false);
    assert!(
        chunked.chunked_prefill(),
        "the fixture must take the chunked driver"
    );
    assert!(!per_row.chunked_prefill());
    let hidden = stepped.hidden_size();
    let ids = tokens(38);
    let (ids, next) = (&ids[..37], ids[37]);
    let mut table = vec![0.0f32; ids.len() * hidden];
    for (&id, row) in ids.iter().zip(table.chunks_exact_mut(hidden)) {
        stepped.embedding_row(id, row).unwrap();
    }
    // Every third row a host row: the same values as its table row, so the
    // host-row write lands in the right `scratch.x` row or the bits differ.
    let rows: Vec<TrunkRow> = ids
        .iter()
        .zip(table.chunks_exact(hidden))
        .enumerate()
        .map(|(i, (&id, row))| {
            if i % 3 == 1 {
                TrunkRow::Embedding(row)
            } else {
                TrunkRow::Token(id)
            }
        })
        .collect();
    let mut want = vec![0.0f32; ids.len() * hidden];
    for (row, out) in rows.iter().zip(want.chunks_exact_mut(hidden)) {
        stepped.step(*row, out).unwrap();
    }
    for (name, trunk) in [("chunked", &mut chunked), ("per row", &mut per_row)] {
        let mut got = vec![0.0f32; ids.len() * hidden];
        trunk.prefill(&rows, &mut got).unwrap();
        assert_eq!(trunk.position(), ids.len(), "{name}");
        for (r, (g, w)) in got
            .chunks_exact(hidden)
            .zip(want.chunks_exact(hidden))
            .enumerate()
        {
            assert_eq!(bits(g), bits(w), "{name}: row {r}");
        }
    }
    let mut a = vec![0.0f32; hidden];
    let mut b = vec![0.0f32; hidden];
    stepped.step(TrunkRow::Token(next), &mut a).unwrap();
    chunked.step(TrunkRow::Token(next), &mut b).unwrap();
    assert_eq!(bits(&a), bits(&b), "the step after a chunked prefill");
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn restoring_a_snapshot_replays_bit_identically() {
    let (root, trunk, arch) = install();
    let mut t = open(&trunk, &arch, 128);
    let hidden = t.hidden_size();
    let mut out = vec![0.0f32; hidden];
    let prefix = tokens(20);
    for &id in &prefix {
        t.step(TrunkRow::Token(id), &mut out).unwrap();
    }
    let snap = t.snapshot();
    assert_eq!(snap.position(), prefix.len());

    let tail: Vec<u32> = tokens(30).into_iter().rev().take(10).collect();
    let run = |t: &mut HiddenTrunk| -> Vec<u32> {
        let mut all = Vec::new();
        let mut h = vec![0.0f32; hidden];
        for &id in &tail {
            t.step(TrunkRow::Token(id), &mut h).unwrap();
            all.extend(bits(&h));
        }
        all
    };
    let first = run(&mut t);
    t.restore(&snap).unwrap();
    assert_eq!(t.position(), prefix.len());
    let second = run(&mut t);
    assert_eq!(first, second, "the recurrent state must restore exactly");

    // And a fresh trunk fed prefix + tail agrees with both.
    let mut fresh = open(&trunk, &arch, 128);
    for &id in &prefix {
        fresh.step(TrunkRow::Token(id), &mut out).unwrap();
    }
    assert_eq!(run(&mut fresh), first);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn a_reset_invalidates_snapshots_and_starts_clean() {
    let (root, trunk, arch) = install();
    let mut t = open(&trunk, &arch, 64);
    let hidden = t.hidden_size();
    let mut out = vec![0.0f32; hidden];
    for &id in &tokens(8) {
        t.step(TrunkRow::Token(id), &mut out).unwrap();
    }
    // Snapshot then reset straight away: the snapshot's copy must have
    // finished before the reset zeroes the live state.
    let snap = t.snapshot();
    t.reset();
    assert_eq!(t.position(), 0);
    let err = t.restore(&snap).expect_err("stale snapshot").to_string();
    assert!(err.contains("reset"), "{err}");

    let mut fresh = open(&trunk, &arch, 64);
    let mut a = vec![0.0f32; hidden];
    let mut b = vec![0.0f32; hidden];
    for &id in &tokens(5) {
        t.step(TrunkRow::Token(id), &mut a).unwrap();
        fresh.step(TrunkRow::Token(id), &mut b).unwrap();
        assert_eq!(bits(&a), bits(&b), "state survived the reset");
    }
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn bad_rows_and_a_full_trunk_are_refused_not_panicked() {
    let (root, trunk, arch) = install();
    let mut t = open(&trunk, &arch, 4);
    let hidden = t.hidden_size();
    let mut out = vec![0.0f32; hidden];

    let short = vec![0.0f32; hidden - 1];
    assert!(t.step(TrunkRow::Embedding(&short), &mut out).is_err());
    let mut nan = vec![0.0f32; hidden];
    nan[3] = f32::NAN;
    assert!(t.step(TrunkRow::Embedding(&nan), &mut out).is_err());
    let mut huge = vec![0.0f32; hidden];
    huge[0] = 1e6;
    assert!(t.step(TrunkRow::Embedding(&huge), &mut out).is_err());
    assert!(t.step(TrunkRow::Token(VOCAB as u32), &mut out).is_err());
    assert!(t.embedding_row(VOCAB as u32, &mut out).is_err());
    assert_eq!(t.position(), 0, "a refused row must not advance the trunk");

    for &id in &tokens(4) {
        t.step(TrunkRow::Token(id), &mut out).unwrap();
    }
    let err = t
        .step(TrunkRow::Token(1), &mut out)
        .expect_err("a full trunk refuses")
        .to_string();
    assert!(err.contains("full"), "{err}");
    let _ = std::fs::remove_dir_all(&root);
}

fn copy_dir(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for entry in std::fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        let target = to.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            copy_dir(&entry.path(), &target);
        } else {
            std::fs::copy(entry.path(), &target).unwrap();
        }
    }
}

/// The host's own `lm_head . hidden` for an INT4-affine head (low nibble
/// first, BF16 scale and bias per 64 values), read straight off the install.
fn host_logits(install: &Path, hidden: &[f32]) -> Vec<f32> {
    let weights = install.join("model_weights.bin");
    let index = model_io::load_resident_index(&weights).unwrap();
    let bytes = std::fs::read(&weights).unwrap();
    let e = &index.entries["language_model.lm_head.weight"];
    let (rows, cols) = (e.shape.0 as usize, e.shape.1 as usize);
    assert_eq!(cols, hidden.len());
    let groups = cols / 64;
    (0..rows)
        .map(|r| {
            let row = compute::Int4AffineRow {
                packed: bytes[e.file_offset as usize + r * cols / 2..][..cols / 2].to_vec(),
                scales: (0..groups)
                    .map(|g| {
                        let at = e.scale_offset as usize + (r * groups + g) * 2;
                        u16::from_le_bytes([bytes[at], bytes[at + 1]])
                    })
                    .collect(),
                biases: (0..groups)
                    .map(|g| {
                        let at = e.bias_offset as usize + (r * groups + g) * 2;
                        u16::from_le_bytes([bytes[at], bytes[at + 1]])
                    })
                    .collect(),
            };
            compute::dequantize_int4_affine(&row, cols)
                .iter()
                .zip(hidden)
                .map(|(w, h)| w * h)
                .sum()
        })
        .collect()
}

/// The hidden state a headless pass returns must be what the head reads:
/// on a headful install, `lm_head . hidden` (on the host) has to reproduce
/// the logits `produce` computes on the GPU. Bit identity between token and
/// host rows cannot catch a wrong buffer or a skipped final norm, since both
/// paths would be wrong together; this can.
#[test]
fn the_hidden_state_is_what_the_head_reads() {
    use turbospark_runtime::LogitProducer;

    let headful = temp_dir();
    let arch =
        turbospark_repack::build_synthetic_qwen_gdn_moe_install(&headful, 128, 4, 8, "tiny-qwen36")
            .expect("qwen install builds");
    let marked = temp_dir().join("marked");
    copy_dir(&headful, &marked);
    model_io::trunk_role::write_headless(&marked, "test").unwrap();

    let mut chat = RealForwardRunner::open(&headful, arch.clone()).expect("chat open");
    let mut trunk = open(&marked, &arch, 64);
    let hidden = trunk.hidden_size();
    let mut h = vec![0.0f32; hidden];
    let mut logits = vec![half::f16::ZERO; 128];
    chat.reset();
    for (position, id) in [5u32, 17, 99, 3, 64, 127, 0, 42].into_iter().enumerate() {
        chat.produce(id as i32, position, &mut logits).unwrap();
        trunk.step(TrunkRow::Token(id), &mut h).unwrap();
        let gpu: Vec<f32> = logits.iter().map(|v| v.to_f32()).collect();
        let host = host_logits(&marked, &h);
        let dot: f32 = gpu.iter().zip(&host).map(|(a, b)| a * b).sum();
        let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
        let cosine = dot / (norm(&gpu) * norm(&host));
        let argmax = |v: &[f32]| {
            v.iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0
        };
        assert!(cosine > 0.999, "position {position}: cosine {cosine}");
        assert_eq!(argmax(&gpu), argmax(&host), "position {position}");
    }
    let _ = std::fs::remove_dir_all(&headful);
    let _ = std::fs::remove_dir_all(marked.parent().unwrap());
}
