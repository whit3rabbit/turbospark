---
description: "Overview of the turbospark-audio Rust crate: what it covers, which roles you can call from Rust today, and how errors work."
icon: box
---
# turbospark-audio

Use `turbospark-audio` when you need WAV decoding, resampling, mel features, or a direct call into one audio model from Rust. It is portable f32 code with no Cargo features and no target-specific gates. You can use its frontends and models without running the TurboSpark HTTP server.

Not every role is callable from Rust yet. The DSP primitives are the most complete surface. Speech and music each expose a few entry points. Language identification has no Rust API, and the codec families have no shared entry point. The role table below says which is which.

## Package and import

The package is `turbospark-audio`. Rust code imports it as `turbospark_audio`. External code needs only those two names. Other crates in this workspace may refer to crates under short aliases, but that is an internal detail.

## Add it to a project

Add the package from crates.io once the version you need is published. Until then, add a path dependency from a checkout of this repository:

```toml
[dependencies]
turbospark-audio = { path = "../turbospark/crates/audio" }
```

The path must point inside a full checkout. The crate depends on `turbospark-core`, `turbospark-model-io`, and `turbospark-tokenizer` through sibling paths, so copying `crates/audio` alone will not build. A git dependency on the repository is the other option. This page does not test that setup.

## Toolchain and platform

The workspace sets Rust 1.82 as the minimum version. The examples on these pages compiled with Rust 1.98.1, so the 1.82 floor is not verified here. `tokenizers` builds a C library through `onig_sys`, so a C compiler must be on the path.

The crate has no `target_os`, `target_arch`, or feature gates. Its CPU code does not use Metal or macOS APIs, and the [audio guide](https://github.com/whit3rabbit/turbospark/blob/main/crates/audio/AGENTS.md) requires that it stay buildable off macOS. This page does not include a Linux build result.

## Roles and what works from Rust

"Run" means an example executed in this checkout. "Compiled" means an example compiles against the crate, but it has not run because no speech or music model is installed locally.

| Role | Module | Rust entry points | Status from Rust |
|---|---|---|---|
| DSP primitives | `waveform`, `wav`, `conversion`, `resample`, `fft`, `stft`, `mel`, `nemo_mel`, `whisper`, `dsp` | `Waveform`, `read_wav_f32`, `to_mono_resampled`, `log_mel_spectrogram` | Run. See [DSP primitives](dsp-primitives.md). |
| Speech-to-text | `stt` | `ParakeetTdt::open`, `transcribe` | Parakeet compiled. Whisper stops at token IDs. See [Speech models](speech.md). |
| Text-to-speech | `tts` | `KokoroSynthesizer`, `MegaTts3` | Compiled. See [Speech models](speech.md). |
| Voice activity | `vad` | `SileroVad::load`, `detect` | Compiled. See [Speech models](speech.md). |
| Enhancement | `sts` | `DeepFilterNet::open`, `enhance` | Compiled. See [Speech models](speech.md). |
| Music generation | `music` | `minimax_music3::Model::load_converted`, `generate_text` | Compiled. See [Music and codecs](music-and-codecs.md). |
| Codecs | `codec` | Per-family types, no shared entry point | Not documented here. See [Music and codecs](music-and-codecs.md). |
| Language ID | `lid` | None | Not usable from Rust yet. |

Language identification is the clear gap. `src/lid/mod.rs` declares no submodules and exports nothing, so there is no call to make. The codec families each have their own types and loaders, but this set does not give them a common call pattern. Use the family READMEs for those.

Compiling is not the same as shipping. The macOS app gates each model family separately, and a family can compile here while the app keeps it unavailable. See the [audio workspace notes](https://github.com/whit3rabbit/turbospark/blob/main/docs/AUDIO_WORKSPACE.md) before you assume a model is product-ready.

## Smallest working example

This program reads a WAV file, mixes it to mono, resamples it to 16 kHz, computes an 80-band log-mel spectrogram, and writes the 16 kHz mono audio as 16-bit PCM. It ran on the fixture [`qwen3_forced_aligner_reference.wav`](https://github.com/whit3rabbit/turbospark/blob/main/crates/audio/testdata/qwen3_forced_aligner_reference.wav). That file reported 16000 Hz, 1 channel, and 44715 frames, and it produced 280 mel frames of 80 bins.

In an application, put this in `src/main.rs` after adding the dependency:

```rust
use std::path::Path;

use turbospark_audio::{
    log_mel_spectrogram, read_wav_f32, to_mono_resampled, write_wav_i16, MelSpectrogramOptions,
    MonoResampleStrategy, Waveform,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let input = args.next().ok_or("usage: dsp_demo <in.wav> <out.wav>")?;
    let output = args.next().ok_or("usage: dsp_demo <in.wav> <out.wav>")?;

    let wave = read_wav_f32(Path::new(&input))?;
    println!(
        "input: {} Hz, {} channel(s), {} frames",
        wave.sample_rate,
        wave.channels,
        wave.frame_count()
    );

    let strategy = MonoResampleStrategy::SincHann(Default::default());
    let mono_16k = to_mono_resampled(&wave, 16_000, &strategy)?;

    let mel = log_mel_spectrogram(&mono_16k, &MelSpectrogramOptions::default(), 1e-10)?;
    println!("log-mel: {} frames x {} bins", mel.len(), mel[0].len());

    let mono = Waveform::mono(16_000, mono_16k)?;
    std::fs::write(&output, write_wav_i16(&mono))?;
    println!("wrote {output}");
    Ok(())
}
```

## Errors

The crate has two error types, and you will handle both.

- `AudioError` covers WAV intake and DSP. Its messages name the value that failed. A non-WAV file, for example, reports `NotWav` with the first four bytes it found. An interleaved sample count that is not a whole number of frames reports `ShapeMismatch` with the expected and actual counts. See the [`AudioError` docs](https://docs.rs/turbospark-audio/latest/turbospark_audio/enum.AudioError.html).
- `SpeechError` covers models. Its variants are `BadConfig`, `Tensor`, `Input`, `Audio`, `Unsupported`, and `ModelIo`. Model open and inference calls return it. See the [`SpeechError` docs](https://docs.rs/turbospark-audio/latest/turbospark_audio/enum.SpeechError.html).

The crate defines `turbospark_audio::Result<T>` as `Result<T, SpeechError>`, and `turbospark_audio::AudioResult<T>` as `Result<T, AudioError>`. There is a `From<AudioError>` conversion into `SpeechError`, so `?` works inside model code that calls DSP functions. Parakeet, for example, rejects audio at the wrong rate with `SpeechError::Input`, and the message names the rate it expected.

## Where to go next

- [DSP primitives](dsp-primitives.md): WAV, resampling, STFT, and mel features.
- [Speech models](speech.md): Parakeet, Whisper, Silero VAD, Kokoro, MegaTTS3, and DeepFilterNet.
- [Music and codecs](music-and-codecs.md): MiniMax Music 3 and the codec layer.

For scope and the reasons behind this layering, read [Audio architecture](https://github.com/whit3rabbit/turbospark/blob/main/docs/AUDIO.md). The [audio HTTP API](https://github.com/whit3rabbit/turbospark/blob/main/docs/AUDIO_API.md) lists the served routes and their limits, which are separate from what the Rust crate exposes. The [model inventory](https://github.com/whit3rabbit/turbospark/blob/main/crates/audio/MODELS.md) records what each family has been verified to do.
