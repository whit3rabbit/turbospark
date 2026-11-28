#![cfg(target_os = "macos")]
//! Chunked prefill with an Inkling 2 block drafter open
//! (`SpeculativeProducer::prefill_chunk_priming_drafter`) must leave the
//! trunk and the drafter where the token-at-a-time walk leaves them.
//!
//! ```text
//! INKLING2_INSTALL=~/.turbospark/models/text/inkling2-mini-mlx4.gturbo \
//!     cargo test --release -p turbospark-runtime \
//!     --test real_forward_inkling2_drafter_chunked -- --ignored --nocapture
//! ```
//!
//! `INKLING2_SPEC_DRAFTER=dflash2` gates the DFlash2 drafter instead
//! (`TURBOSPARK_INKLING2_DFLASH2` or the install's `dflash2/`).
//!
//! Matrix-unit prefill is forced off, because only the exact GEMV arm is
//! bit-equal to the walk. The drafter's context is not readable from here, so
//! the oracle is the first round's proposals: a context row written from the
//! wrong tap, at the wrong position, or not at all changes them or errors.

use std::path::PathBuf;

use foundation::LogitValue;
use turbospark_runtime::{
    DflashDraftPolicy, DraftPolicies, LogitProducer, MtpDraftPolicy, RealForwardRunner,
    SpeculativeProducer,
};

const CHUNK: usize = 128;
/// Across the 64-row panel and the 128-token span, each side of a boundary.
const LENGTHS: [usize; 7] = [1, 5, 64, 65, 128, 129, 300];
const DEPTH: usize = 7;
/// Any in-vocab id: the draft is anchored on the token after the prompt.
const ANCHOR: i32 = 9;

fn first_round(
    runner: &mut RealForwardRunner,
    ids: &[i32],
    logits: &[LogitValue],
) -> (Vec<f32>, Vec<i32>) {
    let mut proposals = Vec::new();
    runner
        .draft_block(ANCHOR, ids.len(), DEPTH, &mut proposals)
        .expect("the first round drafts off the prefill");
    (logits.iter().map(|v| v.to_f32()).collect(), proposals)
}

#[test]
#[ignore = "needs INKLING2_INSTALL with a dspark/ drafter"]
fn chunked_prefill_primes_the_drafter_as_the_walk_does() {
    // Read once per process, so set before the first pass.
    std::env::set_var("TURBOSPARK_INKLING2_PREFILL_MMA", "0");
    let install = PathBuf::from(std::env::var("INKLING2_INSTALL").expect("INKLING2_INSTALL"));
    let arch = model_io::inkling2_mini();
    let vocab = arch.vocab_size as usize;
    let tok = tokenizer::MfTokenizer::load_from_dir(&install).expect("tokenizer");
    let dflash2 = std::env::var("INKLING2_SPEC_DRAFTER").is_ok_and(|v| v == "dflash2");
    let mut runner = RealForwardRunner::open_with_residency(
        &install,
        arch.clone(),
        2048,
        turbospark_runtime::ExpertCacheSlots::Auto,
        DraftPolicies {
            mtp: MtpDraftPolicy::Off,
            dflash: if dflash2 {
                DflashDraftPolicy::Dflash2(DEPTH)
            } else {
                DflashDraftPolicy::Fixed(DEPTH)
            },
        },
        turbospark_runtime::SteeringPolicy::off(),
        1,
        turbospark_runtime::KvQuant::Off,
        turbospark_runtime::ExpertResidency::Resident,
    )
    .expect("open with a block drafter");
    assert_eq!(
        runner.inkling2_drafter_name(),
        Some(if dflash2 { "DFlash2" } else { "DSpark" })
    );
    assert!(
        runner.supports_chunked_drafter_prefill(),
        "run without TURBOSPARK_DFLASH_CHUNKED_PREFILL=0"
    );

    let doc = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/INKLING2_PHASE0.md"),
    )
    .expect("prompt text");
    let text: String = doc.chars().take(4000).collect();
    let all = tok.encode(&text, false);
    assert!(all.len() >= LENGTHS[LENGTHS.len() - 1]);

    let mut logits = vec![LogitValue::from_f32(0.0); vocab];
    for n in LENGTHS {
        let ids = &all[..n];

        runner.reset();
        for (i, &token) in ids.iter().enumerate() {
            runner.produce(token, i, &mut logits).expect("produce");
            if i + 1 < n {
                runner.prime_drafter(ids[i + 1], i).expect("prime");
            }
        }
        let walk = first_round(&mut runner, ids, &logits);

        runner.reset();
        let mut start = 0;
        while start < n {
            let end = (start + CHUNK).min(n);
            runner
                .prefill_chunk_priming_drafter(
                    &ids[start..end],
                    start,
                    &mut logits,
                    ids.get(end).copied(),
                )
                .expect("chunked prefill priming the drafter");
            start = end;
        }
        let chunked = first_round(&mut runner, ids, &logits);

        assert!(walk.0 == chunked.0, "{n} tokens: final logits differ");
        assert_eq!(walk.1, chunked.1, "{n} tokens: first draft differs");
        eprintln!("{n} tokens: identical, draft {:?}", chunked.1);
    }
}
