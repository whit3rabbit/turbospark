#![cfg(target_os = "macos")]
//! The Metal buffer leak gate for Fish Audio Drama 3, against a REAL install.
//!
//! The decode hot path allocates no Metal buffer: weights are the resident
//! mapping, KV and activation scratch exist from open, and routed experts
//! stream into slots that exist from open. So the trunk context's allocation
//! count must be flat once the pipelines are built and the slots are
//! touched. Two phases, in one process because each opens the install:
//!
//! 1. The headless trunk alone: 5,000 `step`s of varied token ids, with a
//!    chunked prefill, a snapshot, a restore and a reset in the warm-up and
//!    again after it.
//! 2. The runner: whole multi-chunk requests, which also run the tower and the
//!    codec. The count covers the trunk's context only. The tower allocates
//!    at load alone (`MetalTower::load` is the only place it creates
//!    buffers). The codec allocates per call by design (one set of buffers per
//!    convolution or linear), so it has no flat count to assert: the footprint
//!    oracle (`fish_drama3_memory_oracle`) is its gate.
//!
//!   TURBOSPARK_FISH_DRAMA3_INSTALL_DIR=~/models/fish-drama3-4bit \
//!     cargo test -p turbospark-bench --test fish_drama3_leak_gate --release -- --ignored --nocapture

use std::path::Path;

use runtime::fish_drama3::{
    expected_trunk_arch, Drama3Request, FishDrama3Runner, DEFAULT_EXPERT_SLOTS, MAX_CONTEXT,
};
use runtime::{ExpertCacheSlots, ExpertResidency, HiddenTrunk, TrunkRow};

/// `audio::tts::fish_drama3::generate::TRUNK_DIR`.
const TRUNK_DIR: &str = "trunk";
const WARM_STEPS: usize = 200;
const TOTAL_STEPS: usize = 5_000;
const PREFILL_ROWS: usize = 40;

/// Token ids spread over the vocabulary, so the router meets many experts and
/// the slot cache both hits and misses. A fixed LCG: the walk is the same
/// every run.
struct Tokens {
    state: u64,
    vocab: u64,
}

impl Tokens {
    fn next(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.state >> 33) % self.vocab) as u32
    }
}

/// A prefill, a snapshot, a restore and `steps` steps from a reset trunk.
fn exercise(trunk: &mut HiddenTrunk, tokens: &mut Tokens, steps: usize) {
    let hidden = trunk.hidden_size();
    let mut out = vec![0.0f32; hidden];
    trunk.reset();
    let rows: Vec<TrunkRow<'_>> = (0..PREFILL_ROWS)
        .map(|_| TrunkRow::Token(tokens.next()))
        .collect();
    let mut prefill_out = vec![0.0f32; PREFILL_ROWS * hidden];
    trunk.prefill(&rows, &mut prefill_out).expect("prefill");
    let snapshot = trunk.snapshot();
    for _ in 0..3 {
        trunk
            .step(TrunkRow::Token(tokens.next()), &mut out)
            .expect("step");
    }
    trunk.restore(&snapshot).expect("restore");
    for step in 0..steps {
        trunk
            .step(TrunkRow::Token(tokens.next()), &mut out)
            .unwrap_or_else(|e| panic!("step {step}: {e}"));
    }
}

fn trunk_phase(dir: &Path) {
    let arch = expected_trunk_arch(dir).expect("the install's config.json gives the trunk arch");
    let mut trunk = HiddenTrunk::open(
        &dir.join(TRUNK_DIR),
        arch,
        MAX_CONTEXT,
        ExpertCacheSlots::Fixed(DEFAULT_EXPERT_SLOTS),
        ExpertResidency::Streamed,
    )
    .expect("the real trunk should open");
    let mut tokens = Tokens {
        state: 0x5eed,
        vocab: trunk.arch().vocab_size as u64,
    };

    exercise(&mut trunk, &mut tokens, WARM_STEPS);
    let warm = trunk.gpu_buffer_allocations();
    eprintln!("leak gate: trunk warm after {WARM_STEPS} steps: {warm} buffers");

    // Every 1,000 steps, so a creep shows where it starts, then one more
    // prefill, snapshot and restore on the warm trunk.
    let mut done = WARM_STEPS;
    while done < TOTAL_STEPS {
        let batch = 1_000.min(TOTAL_STEPS - done);
        exercise(&mut trunk, &mut tokens, batch);
        done += batch;
        let now = trunk.gpu_buffer_allocations();
        eprintln!("leak gate: trunk after {done} steps: {now} buffers");
        assert_eq!(
            now, warm,
            "the trunk allocated Metal buffers after warm-up (by step {done}): the decode path \
             must allocate none"
        );
    }
}

fn speak(runner: &mut FishDrama3Runner, text: &str, seed: u64) {
    let request = Drama3Request {
        text,
        reference: None,
        seed,
        sampling: Default::default(),
    };
    let mut emitted = 0usize;
    runner
        .synthesize(&request, &|| false, |pcm| {
            emitted += pcm.len();
            true
        })
        .expect("synthesize");
    assert!(emitted > 0, "the request produced no audio");
}

fn runner_phase(dir: &Path) {
    // Over two chunks, so a request snapshots the system block and restores it.
    let text = "The history of the lighthouse begins with fires lit on hilltops to guide \
        sailors home. Later, towers were built along dangerous coasts, and keepers tended oil \
        lamps through the night. Storms tested them every winter, and the keepers kept their \
        logbooks by candlelight. Today most lighthouses are automated, but many are preserved \
        as museums.";
    let mut runner = FishDrama3Runner::open(
        dir,
        ExpertCacheSlots::Fixed(DEFAULT_EXPERT_SLOTS),
        ExpertResidency::Streamed,
        MAX_CONTEXT,
    )
    .expect("the real install should open");
    // The first request compiles pipelines, uploads the codec and builds the
    // trunk's scratch.
    speak(&mut runner, text, 0);
    let warm = runner.gpu_buffer_allocations();
    eprintln!("leak gate: runner warm after one request: {warm} buffers");
    for seed in 1..=3u64 {
        speak(&mut runner, text, seed);
        let now = runner.gpu_buffer_allocations();
        eprintln!(
            "leak gate: runner after request {}: {now} buffers",
            seed + 1
        );
        assert_eq!(
            now, warm,
            "a warm request (seed {seed}) allocated trunk Metal buffers"
        );
    }
}

#[test]
#[ignore = "needs a real Drama 3 install via TURBOSPARK_FISH_DRAMA3_INSTALL_DIR"]
fn fish_drama3_decode_allocates_no_gpu_buffers() {
    let Some(dir) = std::env::var_os("TURBOSPARK_FISH_DRAMA3_INSTALL_DIR") else {
        eprintln!("skipping: TURBOSPARK_FISH_DRAMA3_INSTALL_DIR is not set");
        return;
    };
    let dir = Path::new(&dir);
    trunk_phase(dir);
    runner_phase(dir);
}
