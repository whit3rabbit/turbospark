//! Minimal library use of the MiniMax Music 3 Metal runner: open a converted
//! checkpoint, generate from a caption and lyrics with progress reporting,
//! and write a 16-bit stereo WAV.
//!
//! Usage: music3_generate <model-dir> <output.wav> [caption] [lyrics] [seconds] [steps] [seed]
//!
//! The runner is single-threaded (`!Send`): open and use it on one thread.
//! Cancelling a request from the callback returns `Ok(None)` and leaves the
//! runner usable.

#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use audio::music::minimax_music3::{Control, Progress, TextGenerateRequest};
    use std::io::Write;

    let mut args = std::env::args().skip(1);
    let model = std::path::PathBuf::from(args.next().ok_or("model directory required")?);
    let output = std::path::PathBuf::from(args.next().ok_or("output WAV path required")?);
    let caption = args
        .next()
        .unwrap_or_else(|| "upbeat acoustic folk".to_string());
    let lyrics = args.next().unwrap_or_else(|| "[instrumental]".to_string());
    let mut request = TextGenerateRequest::new(caption, lyrics);
    request.duration_seconds = Some(args.next().map_or(Ok(4.0), |v| v.parse())?);
    request.steps = Some(args.next().map_or(Ok(8), |v| v.parse())?);
    request.seed = Some(args.next().map_or(Ok(0), |v| v.parse())?);
    // Reject bad durations and step counts before any device work.
    request.validate()?;

    let runner = turbospark_runtime::Music3Runner::open(&model)?;
    let finished = runner.generate_text_with_progress(&request, |event| {
        match event {
            Progress::Tokenized { prompt_tokens } => eprintln!("prompt: {prompt_tokens} tokens"),
            Progress::ArFrame { emitted, target } if emitted % 25 == 0 || emitted == target => {
                eprint!("\rar frames {emitted}/{target}");
                let _ = std::io::stderr().flush();
            }
            Progress::FlowChunk { index, total } => eprintln!("\nflow chunk {}/{total}", index + 1),
            Progress::ArFrame { .. } => {}
            _ => {}
        }
        Control::Continue
    })?;
    let (generation, timings) = finished.ok_or("generation was cancelled")?;
    let wave = audio::Waveform::new(generation.sample_rate, 2, generation.waveform)?;
    std::fs::write(&output, audio::write_wav_i16(&wave))?;
    let seconds = generation.samples as f64 / f64::from(generation.sample_rate);
    println!(
        "wrote {seconds:.2} s of stereo audio to {} in {:.1} s",
        output.display(),
        timings.total.as_secs_f64()
    );
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("music3_generate requires macOS and a Metal device");
}
