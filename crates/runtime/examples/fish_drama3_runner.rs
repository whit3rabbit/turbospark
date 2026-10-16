//! Fish Audio Drama 3 through `FishDrama3Runner`, the path a request takes:
//! the Metal trunk and tower generate here while the Metal codec decodes on
//! its own thread, and samples arrive in 4-frame then 7-frame pieces.
//!
//! ```sh
//! cargo run --release -p turbospark-runtime --example fish_drama3_runner -- \
//!     <install-dir> "<text>" <out.wav> [expert-slots=16] [seed=0] \
//!     [residency=streamed|mapped] [runs=1]
//! ```
//!
//! Prints, per run, the time to the first samples and the wall time against
//! the audio produced (real time is a factor of 1 or less). The first run
//! also uploads the codec's device weights.

use std::path::PathBuf;
use std::time::Instant;

use turbospark_runtime::fish_drama3::{Drama3Request, FishDrama3Runner, MAX_CONTEXT};
use turbospark_runtime::{ExpertCacheSlots, ExpertResidency};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let install = PathBuf::from(args.next().ok_or("install dir")?);
    let text = args.next().ok_or("text")?;
    let out = PathBuf::from(args.next().ok_or("out.wav")?);
    let slots: usize = args.next().map_or(Ok(16), |s| s.parse())?;
    let seed: u64 = args.next().map_or(Ok(0), |s| s.parse())?;
    let residency = match args.next().as_deref() {
        None | Some("streamed") => ExpertResidency::Streamed,
        Some("mapped") => ExpertResidency::Mapped,
        Some(other) => return Err(format!("residency {other:?}: streamed or mapped").into()),
    };
    let runs: usize = args.next().map_or(Ok(1), |s| s.parse())?;

    let open = Instant::now();
    let mut runner = FishDrama3Runner::open(
        &install,
        ExpertCacheSlots::Fixed(slots),
        residency,
        MAX_CONTEXT,
    )?;
    let rate = runner.sample_rate();
    eprintln!(
        "open {:.2}s ({slots} expert slots, {residency:?} residency)",
        open.elapsed().as_secs_f64()
    );

    let mut samples = Vec::new();
    for run in 0..runs {
        samples.clear();
        let started = Instant::now();
        let mut first = None;
        let request = Drama3Request {
            text: &text,
            reference: None,
            seed,
            sampling: Default::default(),
        };
        let emitted = runner.synthesize(&request, &|| false, |pcm| {
            first.get_or_insert_with(|| started.elapsed());
            samples.extend(pcm);
            true
        })?;
        let wall = started.elapsed().as_secs_f64();
        let audio = emitted as f64 / rate as f64;
        eprintln!(
            "run {run}: {audio:.2}s of audio in {wall:.2}s (real-time factor {:.2}), first \
             samples after {:.0} ms",
            wall / audio.max(1e-9),
            first.map_or(f64::NAN, |d| d.as_secs_f64() * 1e3)
        );
    }
    let wave = audio::Waveform {
        sample_rate: rate,
        channels: 1,
        samples,
    };
    std::fs::write(&out, audio::write_wav_f32(&wave))?;
    eprintln!("wrote {}", out.display());
    Ok(())
}
