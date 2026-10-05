//! Nemotron 3.5 ASR end-to-end run: `cargo run -p turbospark-audio
//! --example nemotron_transcribe -- <model_dir> <wav>`. Input audio is
//! converted to mono 16 kHz PCM before inference.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let model_dir = std::path::PathBuf::from(args.next().expect("model dir"));
    let wav = args.next().expect("wav path");
    let language = args.next();
    let input = turbospark_audio::read_wav_f32(std::path::Path::new(&wav))?;
    let samples = turbospark_audio::to_mono_resampled(
        &input,
        16_000,
        &turbospark_audio::MonoResampleStrategy::SincHann(Default::default()),
    )?;
    let model = turbospark_audio::models::stt::NemotronAsr::open(&model_dir)?;
    println!(
        "TEXT: {}",
        model.transcribe_with_language(&samples, 16_000, language.as_deref())?
    );
    Ok(())
}
