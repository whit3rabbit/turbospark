#![cfg(all(target_os = "macos", feature = "ane-prefill"))]
//! The ANE+Steel prefill split with a DFlash2 drafter open
//! (`TURBOSPARK_ANE_PREFILL_DFLASH=1`, `families/qwen/ane_split.rs`) must
//! leave the drafter where the token walk leaves it, as the Steel chunked
//! variant does (`real_forward_qwen35_dflash_chunked.rs`).
//!
//! The real programs are built for Qwen3.8-27B's shapes, so this runs the
//! split in `break` mode: every buffer break and handoff of the split, with
//! Steel taking every row, which needs a manifest and no Core ML program. The
//! manifest here is written for the fixture's own gate/up and down shapes.
//! What the split's encoder adds over Steel is the capture copy at its
//! 2048-row width and the capture's growth to it; a capture the encoder
//! skipped would leave the first draft reading stale rows, which the 1%
//! draft bound catches (mutation-checked by passing no drafter to the
//! split's post-FFN copy).
//!
//! The trunk is held to the plain open of the same install, under the same
//! split, bit for bit: the capture copies must not touch it.

use std::sync::atomic::{AtomicU64, Ordering};

use foundation::LogitValue;
use turbospark_repack::{build_synthetic_qwen_gdn_dense_install_with_dflash, peek_manifest_arch};
use turbospark_runtime::{
    AnePrefillMode, ChunkedPrefillRunner, DflashDraftPolicy, DraftPolicies, LogitProducer,
    MtpDraftPolicy, RealForwardRunner, SpeculativeProducer,
};

const VOCAB: i64 = 248_100;
/// 64 because the aux taps read trunk layers [5, 19, 33, 47, 61].
const LAYERS: i64 = 64;
const BLOCK: usize = 2;
/// The manifest's one token count: micro-batches of 128 to 256 rows split.
const SPLIT_TOKENS: usize = 256;
/// 150 and 256 split; 100 is under half the token count, so the split
/// declines it and Steel runs it at the split's width, capturing into the
/// widened buffer all the same.
const LENGTHS: [usize; 3] = [100, 150, 256];
const ANCHOR: i32 = 9;

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn temp_dir(tag: &str) -> std::path::PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!(
        "turbospark-dflash-ane-{tag}-{}-{n}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

/// A `break`-mode manifest for the fixture's FFN classes. The packages are
/// never opened in `break` mode.
fn write_manifest(dir: &std::path::Path, hidden: usize, inter: usize) {
    assert!(
        hidden > 32 && inter > 32,
        "the split needs ANE rows, a multiple of 32, inside each matrix"
    );
    let mat = |rows: usize| serde_json::json!({"rows": rows, "ane_rows": 32, "plane_rows": 32});
    let manifest = serde_json::json!({
        "format": 1,
        "programs": [
            {"class": "gateup", "k": hidden, "tokens": SPLIT_TOKENS,
             "matrices": [mat(inter), mat(inter)], "package": "gateup.mlpackage"},
            {"class": "down", "k": inter, "tokens": SPLIT_TOKENS,
             "matrices": [mat(hidden)], "package": "down.mlpackage"},
        ],
    });
    std::fs::write(dir.join("manifest.json"), manifest.to_string()).expect("manifest writes");
}

struct Rig {
    drafter: RealForwardRunner,
    plain: RealForwardRunner,
    _programs: std::path::PathBuf,
}

fn rig(tag: &str) -> Rig {
    // Read once per process, and before any runner reads it.
    std::env::set_var("TURBOSPARK_ANE_PREFILL_DFLASH", "1");
    let dir = temp_dir(tag);
    build_synthetic_qwen_gdn_dense_install_with_dflash(&dir, VOCAB, LAYERS, "dflash-ane", 4)
        .expect("the dense install with a DFlash2 drafter writes");
    let arch = peek_manifest_arch(&dir).expect("manifest peeks");
    let programs = temp_dir(&format!("{tag}-programs"));
    write_manifest(
        &programs,
        arch.hidden_size as usize,
        arch.intermediate_size as usize,
    );
    let open = |dflash| {
        let mut r = RealForwardRunner::open_with_options_and_speculation(
            &dir,
            arch.clone(),
            4096,
            16,
            DraftPolicies {
                mtp: MtpDraftPolicy::Off,
                dflash,
            },
        )
        .expect("the install opens");
        r.set_batched_gemv_prefill(true);
        r.set_prefill_steel(true);
        r.set_ane_prefill(AnePrefillMode::Break, Some(programs.clone()));
        r
    };
    let drafter = open(DflashDraftPolicy::Fixed(BLOCK));
    let plain = open(DflashDraftPolicy::Off);
    let _ = std::fs::remove_dir_all(&dir);
    Rig {
        drafter,
        plain,
        _programs: programs,
    }
}

fn prompt(n: usize) -> Vec<i32> {
    (0..n).map(|i| ((i * 7919 + 13) % 4096) as i32).collect()
}

fn first_draft(runner: &mut RealForwardRunner, n: usize) -> Vec<f32> {
    let mut proposals = Vec::new();
    runner
        .dflash_draft_block(ANCHOR, n, &mut proposals)
        .expect("the first round drafts off the prefill");
    let mut row = vec![LogitValue::from_f32(0.0); VOCAB as usize];
    let mut draft = Vec::new();
    for r in 0..=BLOCK {
        runner.dflash_probe_logits(r, &mut row).expect("probe");
        draft.extend(row.iter().map(|v| v.to_f32()));
    }
    draft
}

/// The token walk's first draft: the reference the chunked suite uses.
fn walk(runner: &mut RealForwardRunner, ids: &[i32]) -> Vec<f32> {
    runner.reset();
    let mut logits = vec![LogitValue::from_f32(0.0); VOCAB as usize];
    for (i, &token) in ids.iter().enumerate() {
        runner.produce(token, i, &mut logits).expect("produce");
        if i + 1 < ids.len() {
            runner.prime_drafter(ids[i + 1], i).expect("prime");
        }
    }
    first_draft(runner, ids.len())
}

#[test]
fn the_speculative_loop_is_asked_for_the_split_width() {
    let r = rig("chunk");
    assert_eq!(r.drafter.drafter_prefill_chunk(), 2048);
    // No drafter, no speculative prefill to size; the default stands.
    assert_eq!(
        r.plain.drafter_prefill_chunk(),
        foundation::DEFAULT_CHUNK_SIZE as usize
    );
}

#[test]
fn the_split_primes_the_drafter_within_the_steel_bound() {
    let mut r = rig("prime");
    for n in LENGTHS {
        let ids = prompt(n);
        let want = walk(&mut r.drafter, &ids);

        r.drafter.reset();
        let mut logits = vec![LogitValue::from_f32(0.0); VOCAB as usize];
        r.drafter
            .prefill_chunk_priming_drafter(&ids, 0, &mut logits, None)
            .expect("the split prefill priming the drafter");
        assert_eq!(
            r.drafter.dflash_handoff_state(),
            Some((n - 1, n - 1, 1)),
            "{n} tokens: the prefill must end on the last row, noted alone"
        );
        let trunk: Vec<f32> = logits.iter().map(|v| v.to_f32()).collect();
        let got = first_draft(&mut r.drafter, n);

        r.plain.reset();
        let mut plain = vec![LogitValue::from_f32(0.0); VOCAB as usize];
        r.plain
            .prefill_chunk(&ids, 0, &mut plain)
            .expect("plain split prefill");
        let plain: Vec<f32> = plain.iter().map(|v| v.to_f32()).collect();
        assert!(
            trunk == plain,
            "{n} tokens: priming the drafter moved the trunk's split logits"
        );

        let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let worst = got
            .iter()
            .zip(&want)
            .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(
            worst <= 0.01 * scale,
            "{n} tokens: draft logits moved {worst} against max {scale}"
        );
    }
}
