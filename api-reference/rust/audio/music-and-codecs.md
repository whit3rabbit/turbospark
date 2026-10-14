---
description: "Rust reference for turbospark-audio music generation with MiniMax Music 3, plus what the codec layer and language identification expose to Rust callers."
icon: box
---
# Music and codecs

Music generation has a Rust entry point. Its load path runs on the CPU reference, and the served route needs macOS and Metal. The codec layer is a set of per-family modules with no shared call pattern, so this page does not document one. Language identification has no Rust API yet.

## Music: MiniMax Music 3

`minimax_music3::Model::load_converted(dir: &Path)` loads a converted model tree at `Float32` precision. This is the CPU reference path. `load_converted_with_backend(dir, backend)` takes an `Rc<dyn ComputeBackend>` for a device backend, and the Metal runtime lives in `turbospark-runtime`, not in this crate. The [MiniMax Music 3 README](https://github.com/whit3rabbit/turbospark/blob/main/crates/audio/src/music/minimax_music3/README.md) describes the converted layout.

`TextGenerateRequest::new(caption, lyrics)` builds a request. Its optional fields are `duration_seconds: Option<f64>`, `steps: Option<usize>`, and `seed: Option<u64>`. The source defaults are 60 seconds, 30 steps, and seed 0. `generate_text` returns a `Generation`, whose `waveform` holds interleaved stereo samples, along with `samples`, `frames`, and `sample_rate`.

Read `sample_rate` from the `Generation` instead of hard coding a rate. The HTTP route reports 44.1 kHz stereo WAV, as [the audio API doc](https://github.com/whit3rabbit/turbospark/blob/main/docs/AUDIO_API.md) describes, but the Rust value is the source of truth for your code.

```rust
use std::path::Path;

use turbospark_audio::music::minimax_music3::{Model, TextGenerateRequest};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let model_dir = args.next().ok_or("usage: <converted_dir> <out.f32>")?;
    let output = args.next().ok_or("usage: <converted_dir> <out.f32>")?;

    let model = Model::load_converted(Path::new(&model_dir))?;
    let mut request = TextGenerateRequest::new("slow lofi piano", "[instrumental]");
    request.duration_seconds = Some(10.0);
    request.seed = Some(0);

    let generation = model.generate_text(&request)?;
    println!(
        "{} stereo frames at {} Hz",
        generation.samples, generation.sample_rate
    );
    let bytes: Vec<u8> = generation
        .waveform
        .iter()
        .flat_map(|s| s.to_le_bytes())
        .collect();
    std::fs::write(&output, bytes)?;
    Ok(())
}
```

The example compiles against this checkout but has not run, because no converted model is installed here. The [audio API doc](https://github.com/whit3rabbit/turbospark/blob/main/docs/AUDIO_API.md) says a full-size request takes minutes or longer, so use the async route for long clips. The repository records only short smoke runs of this model, so output quality is unqualified. Do not treat the example as evidence of quality or speed.

## Codecs

The `codec` module declares many families, including `snac`, `encodec`, `mimi`, `vocos`, `bigvgan`, and `wavvae`. Each family has its own types and loader, and the crate does not give them a shared trait or entry point. This set therefore documents no call pattern for codecs.

Two notes matter if you go there. Some families are building blocks for others. The [audio workspace guide](https://github.com/whit3rabbit/turbospark/blob/main/crates/audio/AGENTS.md) says MegaTTS3 is built over `wavvae`, for example. The [model inventory](https://github.com/whit3rabbit/turbospark/blob/main/crates/audio/MODELS.md) lists each family's source and verification status, and each family README describes its inputs. Start there before you write code against a codec type.

## Language identification

`lid` has no Rust API yet. `crates/audio/src/lid/mod.rs` contains module documentation and no submodules, and it exports nothing. There is nothing to call, so do not build against it until the module gains code. The [`lid` source folder](https://github.com/whit3rabbit/turbospark/tree/main/crates/audio/src/lid) has a README, but this page does not summarize it.

Do not confuse `codec::ecapa_tdnn` with a language identifier. That module lives under `codec`, and it is a codec family in this crate's layout.
