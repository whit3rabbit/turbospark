#![cfg(target_os = "macos")]
//! The memory oracle for Fish Audio Drama 3, against a REAL install
//! (`~/models/fish-drama3-4bit`: the headless 35B-A3B trunk with streamed
//! experts, the Metal tower, the Metal codec).
//!
//! It does not use `oracle_common`: that body drives the text protocol
//! through a `LogitProducer`, and a Drama 3 request goes through
//! `FishDrama3Runner::synthesize`. The shape is the same, one whole-session
//! `phys_footprint` peak under a ceiling plus a warm-replay growth bound.
//!
//! ONE MODEL PER PROCESS (this file holds one `#[test]` for the reason
//! `oracle_common`'s module doc gives).
//!
//! Not run by default (needs the ~20 GB install; `--release` or the timing
//! lines are meaningless):
//!
//!   TURBOSPARK_FISH_DRAMA3_INSTALL_DIR=~/models/fish-drama3-4bit \
//!     cargo test -p turbospark-bench --test fish_drama3_memory_oracle --release -- --ignored --nocapture
//!
//! What it runs, 16 expert slots, streamed residency, 8192 trunk rows (the
//! runner's `MAX_CONTEXT`):
//!
//! 1. A discarded warm-up of [`TEXT`], then the measured run of it. The text
//!    is over two chunks long, so the generation loop prefills the system
//!    block once, snapshots it and restores it for every later chunk.
//! 2. One cloned-voice request: the first seconds of the measured clip as the
//!    reference. That adds the codec encoder's weights and a longer prompt.
//!    The reference is synthetic (the model's own output), so this measures
//!    memory and says nothing about voice quality.
//! 3. Warm replays of [`TEXT`]: up to four, stopping once one grows the
//!    footprint by no more than 8 MiB.
//!
//! The peak is sampled from a thread every 25 ms as well as at each emitted
//! piece, so a transient at a chunk boundary (snapshot or restore) is seen.

use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use runtime::fish_drama3::{
    open_bytes, Drama3Request, FishDrama3Runner, DEFAULT_EXPERT_SLOTS, MAX_CONTEXT,
};
use runtime::{ExpertCacheSlots, ExpertResidency};
use turbospark_bench::memory::{chip_brand_string, AppMemorySampler};

/// The whole-session `phys_footprint` ceiling, MiB: the measured peak plus 5
/// percent, the rule the other rows use. The first reading (2026-10-16, M4
/// Max, AC, release) was 6,106 MiB, 6,411 with the margin. `u64::MAX` means no
/// row has been measured yet, and the test then reports the peak without
/// asserting it.
const CEILING_MIB: u64 = 6_411;

/// Bytes of one generation chunk (`chunking::CHUNK_BYTES`). A text longer
/// than twice this has at least three chunks, so the snapshot is restored at
/// least twice.
const CHUNK_BYTES: usize = 300;

/// Warm replays must stop growing the footprint by this much.
const STEADY_STATE_SLACK_BYTES: u64 = 8 * 1_048_576;
const STEADY_STATE_ROUNDS: usize = 4;

/// A generation of tens of seconds in at least three chunks. The measured
/// audio length is printed, and the run must reach [`MIN_AUDIO_SECONDS`].
const TEXT: &str = "The history of the lighthouse begins with fires lit on hilltops to guide \
    sailors home. Later, towers were built along dangerous coasts, and keepers tended oil \
    lamps through the night. Storms tested them every winter, and the keepers kept their \
    logbooks by candlelight. Today most lighthouses are automated, but many are preserved as \
    museums. Visitors can climb the spiral stairs to look out over the water, and children \
    learn how the great lenses bent a small flame into a beam that reached the horizon. On \
    clear evenings the old beam still sweeps across the bay. Some of these towers have stood \
    for two hundred years.";

/// The first sentence of [`TEXT`], the transcript of the reference clip's
/// opening. Roughly right is enough: the reference only has to be a valid
/// voice prompt for a memory measurement.
const REFERENCE_TEXT: &str =
    "The history of the lighthouse begins with fires lit on hilltops to guide sailors home.";
const REFERENCE_SECONDS: usize = 6;
const MIN_AUDIO_SECONDS: usize = 25;

/// Footprint peak from a thread, so a transient between emitted pieces is
/// not missed. Stops and joins on drop.
struct PeakWatcher {
    peak: Arc<AtomicU64>,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl PeakWatcher {
    fn start() -> Self {
        let peak = Arc::new(AtomicU64::new(0));
        let stop = Arc::new(AtomicBool::new(false));
        let (p, s) = (peak.clone(), stop.clone());
        let thread = std::thread::spawn(move || {
            let mut sampler = AppMemorySampler::new();
            while !s.load(Ordering::Relaxed) {
                if let Some(bytes) = sampler.sample() {
                    p.fetch_max(bytes, Ordering::Relaxed);
                }
                std::thread::sleep(Duration::from_millis(25));
            }
        });
        PeakWatcher {
            peak,
            stop,
            thread: Some(thread),
        }
    }

    fn peak_mib(&self) -> f64 {
        self.peak.load(Ordering::Relaxed) as f64 / 1_048_576.0
    }
}

impl Drop for PeakWatcher {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

fn mib(bytes: u64) -> f64 {
    bytes as f64 / 1_048_576.0
}

/// One request, returning every sample it emitted.
fn speak(
    runner: &mut FishDrama3Runner,
    request: &Drama3Request<'_>,
    sampler: &mut AppMemorySampler,
) -> Vec<f32> {
    let mut samples = Vec::new();
    runner
        .synthesize(request, &|| false, |pcm| {
            samples.extend(pcm);
            sampler.sample();
            true
        })
        .expect("a Drama 3 request should synthesize");
    sampler.sample();
    samples
}

#[test]
#[ignore = "needs a real Drama 3 install via TURBOSPARK_FISH_DRAMA3_INSTALL_DIR"]
fn fish_drama3_peak_footprint_and_steady_state() {
    let Some(dir) = std::env::var_os("TURBOSPARK_FISH_DRAMA3_INSTALL_DIR") else {
        eprintln!("skipping: TURBOSPARK_FISH_DRAMA3_INSTALL_DIR is not set");
        return;
    };
    let dir = Path::new(&dir);
    assert!(
        TEXT.len() > 2 * CHUNK_BYTES,
        "the text must span at least three chunks so a snapshot is restored twice"
    );
    eprintln!(
        "fish_drama3_memory_oracle: chip {:?}, {DEFAULT_EXPERT_SLOTS} expert slots, streamed \
         residency, {MAX_CONTEXT} trunk rows, text {} bytes",
        chip_brand_string(),
        TEXT.len()
    );

    let watcher = PeakWatcher::start();
    let mut sampler = AppMemorySampler::new();
    let before = sampler.sample().expect("footprint sampling worked");

    let opened = Instant::now();
    let mut runner = FishDrama3Runner::open(
        dir,
        ExpertCacheSlots::Fixed(DEFAULT_EXPERT_SLOTS),
        ExpertResidency::Streamed,
        MAX_CONTEXT,
    )
    .expect("the real install should open");
    let rate = runner.sample_rate() as usize;
    let after_open = sampler.sample().expect("footprint sampling worked");
    eprintln!(
        "fish_drama3_memory_oracle: open {:.1}s, footprint {:.0} -> {:.0} MiB",
        opened.elapsed().as_secs_f64(),
        mib(before),
        mib(after_open)
    );

    let plain = Drama3Request {
        text: TEXT,
        reference: None,
        seed: 0,
        sampling: Default::default(),
    };

    // Frozen protocol: one discarded warm-up, then the measured run.
    speak(&mut runner, &plain, &mut sampler);
    eprintln!(
        "fish_drama3_memory_oracle: warm-up done, peak so far {:.1} MiB",
        watcher.peak_mib()
    );
    let measured = speak(&mut runner, &plain, &mut sampler);
    let seconds = measured.len() as f64 / rate as f64;
    eprintln!(
        "fish_drama3_memory_oracle: measured run {seconds:.1}s of audio, peak so far {:.1} MiB",
        watcher.peak_mib()
    );
    assert!(
        measured.len() >= MIN_AUDIO_SECONDS * rate,
        "the measured run produced {seconds:.1}s of audio, under the {MIN_AUDIO_SECONDS}s the \
         row is defined at: the peak would not cover several chunks"
    );

    let reference = &measured[..REFERENCE_SECONDS * rate];
    let cloned = Drama3Request {
        reference: Some((reference, rate as u32, REFERENCE_TEXT)),
        ..plain
    };
    let cloned_audio = speak(&mut runner, &cloned, &mut sampler);
    eprintln!(
        "fish_drama3_memory_oracle: cloned run {:.1}s of audio, peak so far {:.1} MiB",
        cloned_audio.len() as f64 / rate as f64,
        watcher.peak_mib()
    );
    assert!(
        cloned_audio.len() >= MIN_AUDIO_SECONDS * rate,
        "the cloned run produced under {MIN_AUDIO_SECONDS}s of audio"
    );

    // Steady state: everything the plain request touches is warm, so
    // replaying it must cost nothing. A leak per frame, per chunk or per
    // restore would grow here and never stop.
    let plain = Drama3Request {
        reference: None,
        ..cloned
    };
    let mut previous = sampler.sample().expect("footprint sampling worked");
    let mut growth = u64::MAX;
    let mut round = 0;
    while round < STEADY_STATE_ROUNDS && growth > STEADY_STATE_SLACK_BYTES {
        speak(&mut runner, &plain, &mut sampler);
        let now = sampler.sample().expect("footprint sampling worked");
        growth = now.saturating_sub(previous);
        previous = now;
        round += 1;
        eprintln!(
            "fish_drama3_memory_oracle: replay {round} -> {:.1} MiB (+{:.2} MiB)",
            mib(now),
            mib(growth)
        );
    }
    assert!(
        growth <= STEADY_STATE_SLACK_BYTES,
        "replaying an already-warm request {STEADY_STATE_ROUNDS} times never stopped growing \
         the footprint (last round +{:.2} MiB, slack {:.0} MiB): something accumulates per \
         frame, per chunk or per snapshot restore",
        mib(growth),
        mib(STEADY_STATE_SLACK_BYTES)
    );

    let peak = watcher
        .peak
        .load(Ordering::Relaxed)
        .max(sampler.peak_bytes().expect("footprint sampling worked"));
    let peak_mib = peak / 1_048_576;
    let estimate = open_bytes(dir).expect("the install has the files open_bytes sizes");
    eprintln!(
        "fish_drama3_memory_oracle: session peak {peak_mib} MiB; open_bytes estimate {:.0} MiB \
         (estimate minus peak {:+.0} MiB)",
        mib(estimate),
        mib(estimate) - peak as f64 / 1_048_576.0
    );
    if CEILING_MIB == u64::MAX {
        eprintln!("fish_drama3_memory_oracle: no ceiling recorded yet, peak not asserted");
        return;
    }
    assert!(
        peak_mib <= CEILING_MIB,
        "peak phys_footprint {peak_mib} MiB exceeds the {CEILING_MIB} MiB ceiling: Drama 3 \
         uses more memory than the row it was frozen at"
    );
}
