//! Native CPU transcription from a Needle 3 Whistle checkpoint.

use std::path::PathBuf;

use turbospark_audio::models::stt::whistle::Whistle;
use turbospark_audio::read_wav_f32;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let usage = "usage: whistle_transcribe WHISTLE.cact AUDIO.wav [LANGUAGE]";
    let model_path = args.next().map(PathBuf::from).ok_or(usage)?;
    let audio_path = args.next().map(PathBuf::from).ok_or(usage)?;
    let language = args
        .next()
        .map(|s| s.into_string())
        .transpose()
        .map_err(|_| "language must be UTF-8")?;
    if args.next().is_some() {
        return Err(usage.into());
    }
    let waveform = read_wav_f32(&audio_path)?;
    if waveform.sample_rate != 16_000 || waveform.channels != 1 {
        return Err("expected a mono 16 kHz WAV".into());
    }
    let model = Whistle::load(&model_path)?;
    let transcript = model.transcribe(&waveform.samples, language.as_deref(), 318)?;
    println!("{}", transcript.text);
    eprintln!("language={}", transcript.language);
    if transcript.truncated {
        return Err("transcript reached the 318-token limit".into());
    }
    Ok(())
}
