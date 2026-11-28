//! Parakeet TDT end-to-end run: `cargo run -p turbospark-audio --example
//! parakeet_transcribe -- <model_dir> <wav> [--clean]`. Input audio is
//! converted to mono 16 kHz PCM before inference. `--clean` selects Fluid
//! Speech Pro's clean transcript style and is refused by other checkpoints.

use turbospark_audio::models::stt::{ParakeetTdt, TranscriptStyle};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let model_dir = std::path::PathBuf::from(args.next().expect("model dir"));
    let wav = args.next().expect("wav path");
    let style = match args.next().as_deref() {
        None => TranscriptStyle::Verbatim,
        Some("--clean") => TranscriptStyle::Clean,
        Some(other) => return Err(format!("unknown argument {other}").into()),
    };
    let input = turbospark_audio::read_wav_f32(std::path::Path::new(&wav))?;
    let samples = turbospark_audio::to_mono_resampled(
        &input,
        16_000,
        &turbospark_audio::MonoResampleStrategy::SincHann(Default::default()),
    )?;
    let model = ParakeetTdt::open_with_style(&model_dir, style)?;
    println!("TEXT: {}", model.transcribe(&samples, 16_000)?);
    Ok(())
}
