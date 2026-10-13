---
description: "Rust reference for turbospark-audio speech models: Parakeet and Whisper for speech-to-text, Silero VAD, Kokoro and MegaTTS3 for text-to-speech, and DeepFilterNet for enhancement."
icon: box
---
# Speech models

Speech roles follow one pattern. You open a converted model directory once, then call a method on the loaded model with samples at the rate it expects. Parakeet, Silero VAD, Kokoro, MegaTTS3, and DeepFilterNet all work this way. The examples below compile against this checkout, but none has run, because no speech model is installed locally. Read the family README before you point an example at a model folder, since each family needs its own files.

Every example here reads a file path from the command line and returns `Result` from `main`. Replace the argument handling with your own inputs in an application.

| Model | Input | Output |
|---|---|---|
| Parakeet TDT | Mono f32 at 16 kHz | Text, or per-token timestamps from `decode` |
| Whisper | Mono f32 at 16 kHz, up to 30 seconds | Token IDs, not text |
| Silero VAD | Mono f32 at 16 kHz in the example, with an 8 kHz branch in the code | Sample index ranges |
| Kokoro | Text | Mono f32 at 24 kHz, in segments |
| MegaTTS3 | Mono 24 kHz reference audio, plus text | Mono f32 at 24 kHz |
| DeepFilterNet | Mono f32 at 48 kHz | Enhanced mono f32 |

The repository's [examples folder](https://github.com/whit3rabbit/turbospark/tree/main/crates/audio/examples) has runnable starting points for other families, such as Moonshine, Granite, Nemotron, and Sortformer. Those are outside this page.

## Speech-to-text: Parakeet

`ParakeetTdt::open(model_dir: &Path)` loads the model. `transcribe(&self, samples: &[f32], sample_rate: u32) -> Result<String>` returns the text. The sample rate must match the model's configured rate, which is 16 kHz for the checkpoint in the example. A different rate returns `SpeechError::Input`, and so do empty or non-finite samples.

Use `decode` instead of `transcribe` when you need timing. It returns an `AlignedResult` with per-token waveform timestamps grouped into sentences. The [`ParakeetTdt` docs](https://docs.rs/turbospark-audio/latest/turbospark_audio/stt/struct.ParakeetTdt.html) list the methods.

```rust
use std::path::Path;

use turbospark_audio::stt::ParakeetTdt;
use turbospark_audio::{read_wav_f32, to_mono_resampled, MonoResampleStrategy};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let model_dir = args.next().ok_or("usage: <model_dir> <in.wav>")?;
    let input = args.next().ok_or("usage: <model_dir> <in.wav>")?;

    let wave = read_wav_f32(Path::new(&input))?;
    let samples = to_mono_resampled(
        &wave,
        16_000,
        &MonoResampleStrategy::SincHann(Default::default()),
    )?;

    let model = ParakeetTdt::open(Path::new(&model_dir))?;
    let text = model.transcribe(&samples, 16_000)?;
    println!("{text}");
    Ok(())
}
```

### Whisper stops at token IDs

`stt::whisper::WhisperModel::transcribe(&self, pcm_16k: &[f32], max_tokens: usize) -> Result<Vec<u32>>` runs the encoder and a greedy decoder, then returns token IDs. This crate does not turn those IDs into text, and it does not load the weight tensors from disk. The runtime crate has its own Whisper weight loader in `crates/runtime/src/whisper`. For Whisper text, use the TurboSpark server or the runtime, not this crate alone.

The method pads short input to 30 seconds. It also truncates longer input to its first 30 seconds with `Vec::resize`, and it does not warn you. Split long recordings into windows before you call it.

## Voice activity: Silero VAD

`SileroVad::load(dir: &Path)` loads the model. `detect(&self, samples: &[f32], sample_rate: u32) -> Result<Vec<(usize, usize)>>` returns speech segments as `(start, end)` sample indices. The example below converts them to seconds at 16 kHz.

For streaming, call `initial_state(sample_rate)` once, then `feed(&chunk, &mut state)` for each chunk. Each call returns one speech probability and advances the state. The chunk size is 512 samples at 16 kHz and 256 samples at 8 kHz.

```rust
use std::path::Path;

use turbospark_audio::vad::SileroVad;
use turbospark_audio::{read_wav_f32, to_mono_resampled, MonoResampleStrategy};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let model_dir = args.next().ok_or("usage: <model_dir> <in.wav>")?;
    let input = args.next().ok_or("usage: <model_dir> <in.wav>")?;

    let wave = read_wav_f32(Path::new(&input))?;
    let samples = to_mono_resampled(
        &wave,
        16_000,
        &MonoResampleStrategy::SincHann(Default::default()),
    )?;

    let vad = SileroVad::load(Path::new(&model_dir))?;
    for (start, end) in vad.detect(&samples, 16_000)? {
        println!(
            "{:.2}s to {:.2}s",
            start as f64 / 16_000.0,
            end as f64 / 16_000.0
        );
    }
    Ok(())
}
```

## Text-to-speech: Kokoro

`KokoroSynthesizer::open(dir: &Path)` reads `config.json` and the voice pack. Kokoro supports one voice, `af_heart`, in US English. The voice file may sit at `voices/af_heart.safetensors` or next to the weights as `af_heart.safetensors`.

`SynthesisRequest::new(text)` sets the language to `en-US`, the voice to `af_heart`, and the speed to 1.0. Its fields are public, so you can change the speed before you synthesize. `synthesize` calls `emit` once per frontend segment with 24 kHz mono samples. Return `false` from `emit` to stop early, which is not an error.

```rust
use std::path::Path;

use turbospark_audio::tts::kokoro::{KokoroSynthesizer, SynthesisRequest};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let model_dir = args.next().ok_or("usage: <model_dir> <out.f32>")?;
    let output = args.next().ok_or("usage: <model_dir> <out.f32>")?;

    let synth = KokoroSynthesizer::open(Path::new(&model_dir))?;
    let request = SynthesisRequest::new("Hello from TurboSpark.");

    let mut pcm: Vec<f32> = Vec::new();
    synth.synthesize(&request, |segment| {
        pcm.extend(segment);
        true
    })?;

    let bytes: Vec<u8> = pcm.iter().flat_map(|s| s.to_le_bytes()).collect();
    std::fs::write(&output, bytes)?;
    println!("{} samples at 24 kHz", pcm.len());
    Ok(())
}
```

The example writes raw little-endian f32 samples. If another tool needs a container, wrap the samples with `write_wav_f32` instead. The output rate is available as `turbospark_audio::tts::kokoro::OUTPUT_SAMPLE_RATE`, which is 24000.

## Text-to-speech with voice cloning: MegaTTS3

MegaTTS3 clones a voice from a reference clip. Cloning needs consent, and the crate does not check it. The HTTP route requires `consent: true`, and the CLI requires `--consent`. The Rust API has no consent field, so your application must collect consent and check it before it calls `prepare_prompt` or `synthesize`.

The reason is the encoder. The family uses a community WavVAE encoder whose provenance the project has not verified. The [MegaTTS3 README](https://github.com/whit3rabbit/turbospark/blob/main/crates/audio/src/tts/megatts3/README.md) and the [audio API](https://github.com/whit3rabbit/turbospark/blob/main/docs/AUDIO_API.md) describe the caller's statement that it has the right to use the reference voice. The example below reads that statement from an environment variable only to keep the program short. In an application, take it from an explicit user action.

```rust
use std::path::Path;

use turbospark_audio::conversion::{mixdown_interleaved_to_mono_average, MonoMixAccumulation};
use turbospark_audio::read_wav_f32;
use turbospark_audio::tts::megatts3::{resample_reference, MegaTts3, SeededRng, SynthParams};

/// The crate does not check consent. The HTTP and CLI layers do.
/// This gate mirrors that rule in the caller.
fn require_consent(consent: bool) -> Result<(), Box<dyn std::error::Error>> {
    if consent {
        Ok(())
    } else {
        Err("voice cloning needs consent from the speaker".into())
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let model_dir = args.next().ok_or("usage: <model_dir> <reference.wav> <out.f32>")?;
    let reference = args.next().ok_or("usage: <model_dir> <reference.wav> <out.f32>")?;
    let output = args.next().ok_or("usage: <model_dir> <reference.wav> <out.f32>")?;
    let consent = std::env::var("TURBOSPARK_DEMO_CONSENT").as_deref() == Ok("1");
    require_consent(consent)?;

    let wave = read_wav_f32(Path::new(&reference))?;
    let mono = mixdown_interleaved_to_mono_average(
        &wave.samples,
        wave.channels as usize,
        MonoMixAccumulation::Float64,
    )?;
    let prompt_audio = resample_reference(&mono, wave.sample_rate)?;

    let mut model = MegaTts3::open(Path::new(&model_dir))?;
    let mut rng = SeededRng::new(7);
    let prompt = model.prepare_prompt(&prompt_audio, &mut rng)?;

    let params = SynthParams::default();
    let pcm = model.synthesize(&prompt, "Hello from TurboSpark.", &params, &mut rng)?;

    let bytes: Vec<u8> = pcm.iter().flat_map(|s| s.to_le_bytes()).collect();
    std::fs::write(&output, bytes)?;
    println!("{} samples at 24 kHz", pcm.len());
    Ok(())
}
```

A few details matter here. `prepare_prompt` uses at most `MAX_PROMPT_SECONDS` (28) of the reference. `SynthParams::default()` runs 16 steps, which is the Balanced quality. `Quality::Fast.steps()` gives 8, and `Quality::Best.steps()` gives 32.

A decoder-only WavVAE checkpoint has no encoder. In that case `prepare_prompt` returns `SpeechError::Unsupported`. Call `can_clone()` first if you want to offer cloning only when it can work.

`synthesize` takes `&mut self`, so keep the model in a mutable binding. `synthesize_controlled` adds progress reporting and early cancellation through its callbacks. This page gives no timing for MegaTTS3, because the speed depends on the backend and the step count.

## Speech enhancement: DeepFilterNet

`DeepFilterNet::open(dir: &Path)` loads the model. `enhance(&self, samples: &[f32]) -> Result<Vec<f32>>` takes 48 kHz mono samples and returns the enhanced signal. The example rejects other formats instead of resampling them, so resample with `to_mono_resampled` first if your input is different.

```rust
use std::path::Path;

use turbospark_audio::read_wav_f32;
use turbospark_audio::sts::DeepFilterNet;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let model_dir = args.next().ok_or("usage: <model_dir> <in_48k.wav> <out.f32>")?;
    let input = args.next().ok_or("usage: <model_dir> <in_48k.wav> <out.f32>")?;
    let output = args.next().ok_or("usage: <model_dir> <in_48k.wav> <out.f32>")?;

    let wave = read_wav_f32(Path::new(&input))?;
    if wave.sample_rate != 48_000 || wave.channels != 1 {
        return Err("DeepFilterNet expects 48 kHz mono input".into());
    }

    let model = DeepFilterNet::open(Path::new(&model_dir))?;
    let enhanced = model.enhance(&wave.samples)?;
    let bytes: Vec<u8> = enhanced.iter().flat_map(|s| s.to_le_bytes()).collect();
    std::fs::write(&output, bytes)?;
    println!("{} samples", enhanced.len());
    Ok(())
}
```

## Errors you will see

Model calls return `SpeechError`. The messages name the failing field or tensor, so read the text before you change code. Common cases on this page:

- `BadConfig` when `config.json` or a required field is missing. The Kokoro example returns this for a missing directory.
- `ModelIo` when a weight file cannot be opened.
- `Input` when the sample rate or sample values are wrong, as Parakeet does.
- `Unsupported` when the checkpoint asks for a feature the port does not implement, such as cloning without an encoder.

The [`SpeechError` docs](https://docs.rs/turbospark-audio/latest/turbospark_audio/enum.SpeechError.html) list every variant.

## Other families in these roles

`stt`, `tts`, `vad`, and `sts` export more families than this page covers. The module lists are in [`crates/audio/src`](https://github.com/whit3rabbit/turbospark/tree/main/crates/audio/src), and the [model inventory](https://github.com/whit3rabbit/turbospark/blob/main/crates/audio/MODELS.md) records each family's verification status. Read the family README for its file layout and call pattern before you use it from Rust.
