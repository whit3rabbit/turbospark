use std::path::PathBuf;

use turbospark_audio::models::stt::granite_speech5_ctc::GraniteSpeech5Ctc;
use turbospark_audio::read_wav_f32;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let model_dir = args
        .next()
        .map(PathBuf::from)
        .ok_or("usage: granite5_transcribe MODEL_DIR AUDIO.wav")?;
    let audio_path = args
        .next()
        .map(PathBuf::from)
        .ok_or("usage: granite5_transcribe MODEL_DIR AUDIO.wav")?;
    let waveform = read_wav_f32(&audio_path)?;
    if waveform.sample_rate != 16_000 || waveform.channels != 1 {
        return Err(format!(
            "expected mono 16 kHz WAV, got {} Hz and {} channels",
            waveform.sample_rate, waveform.channels
        )
        .into());
    }
    let model = GraniteSpeech5Ctc::load(&model_dir)?;
    println!("{}", model.transcribe(&waveform.samples)?);
    Ok(())
}
