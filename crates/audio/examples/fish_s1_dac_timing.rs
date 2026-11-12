//! Fish S1 DAC CPU timing: `cargo run --release -p turbospark-audio --example
//! fish_s1_dac_timing -- <codec.safetensors> [seconds]`.
//!
//! Times the portable f32 encode and decode on a synthetic signal so the
//! Drama 3 Phase 0 budget (46.4 ms of audio per frame) rests on a measured
//! number. The signal is a deterministic chirp plus a harmonic stack, not
//! speech: codec cost depends on length, not content. Prints wall time and
//! real-time factor (audio seconds per wall second) for each direction.

use std::path::PathBuf;
use std::time::Instant;

use turbospark_audio::codec::fish_s1_dac::FishS1Dac;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let weights = PathBuf::from(args.next().expect("codec.safetensors path"));
    let seconds: f32 = args.next().map(|s| s.parse()).transpose()?.unwrap_or(10.0);

    let open_start = Instant::now();
    let model = FishS1Dac::open_weights(&weights)?;
    let open_s = open_start.elapsed().as_secs_f64();

    let rate = model.config.sample_rate as f32;
    let frame = model.config.frame_length();
    // Whole frames only: the encoder right-pads to a frame multiple, and a
    // partial frame would time padding that real requests do not pay.
    let frames = ((seconds * rate) as usize / frame).max(1);
    let n = frames * frame;
    let wave: Vec<f32> = (0..n)
        .map(|i| {
            let t = i as f32 / rate;
            let chirp = (std::f32::consts::TAU * (110.0 + 400.0 * t) * t).sin();
            let stack: f32 = (1..6)
                .map(|h| (std::f32::consts::TAU * 220.0 * h as f32 * t).sin() / h as f32)
                .sum();
            0.15 * chirp + 0.1 * stack
        })
        .collect();
    let audio_s = n as f64 / rate as f64;
    println!("open {open_s:.2} s, {frames} frames, {audio_s:.2} s of audio");

    let encode_start = Instant::now();
    let (codes, code_len) = model.encode(&wave)?;
    let encode_s = encode_start.elapsed().as_secs_f64();
    println!(
        "encode {encode_s:.2} s, {code_len} frames, rtf {:.3}x",
        audio_s / encode_s
    );

    let decode_start = Instant::now();
    let audio = model.decode(&codes)?;
    let decode_s = decode_start.elapsed().as_secs_f64();
    let finite = audio.iter().all(|v| v.is_finite());
    println!(
        "decode {decode_s:.2} s, {} samples, finite {finite}, rtf {:.3}x, {:.1} ms per frame",
        audio.len(),
        audio_s / decode_s,
        decode_s * 1000.0 / code_len as f64
    );
    Ok(())
}
