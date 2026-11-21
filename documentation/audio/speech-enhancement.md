---
description: Speech enhancement, voice isolation, live background-noise reduction, and vocal separation on device.
icon: wand-magic-sparkles
---

# Speech Enhancement and Separation

TurboSpark cleans speech in a recording (voice isolation, noise suppression), reduces background noise in live text, and separates vocals from music. Every result is a new file or take; the source audio is never changed.

## Supported Models

Install with `turbospark-model pull-audio <ALIAS>`:

| Alias | Task | Size | Engine | Notes |
| --- | --- | --- | --- | --- |
| `voice-isolator-v1` | Enhancement | 256 MB | Metal, CPU fallback | Inworld Voice Isolator v1, offline (Apache-2.0). 48 kHz mono. |
| `voice-isolator-v1-realtime` | Enhancement, live | 233 MB | Metal, CPU fallback; live streams also Core ML | The causal checkpoint: streams with 63 ms algorithmic latency. |
| `mossformer2-se-48k-fp16` | Enhancement | 111 MB | CPU | MossFormer2 SE 48K. 48 kHz mono. |
| `mel-roformer-vocals-v1` | Separation | 67 MB | CPU | Mel-Band RoFormer vocals (MIT; the training data is undisclosed). 44.1 kHz stereo vocals. |

Quality beyond parity with the reference implementations (PESQ, DNSMOS, SDR) is not measured. DeepFilterNet has source code but no runnable session.

## Command Line

```sh
# Clean speech; --background also writes what was removed
turbospark enhance --model voice-isolator-v1 --input noisy.wav --output voice.wav --background noise.wav

# Separate vocals; --background writes the instrumental
turbospark enhance --task separate --model mel-roformer-vocals-v1 \
  --input song.wav --output vocals.wav --background instrumental.wav
```

Input is any WAV from 8 kHz to 192 kHz. Enhancement mixes to mono and takes at most 661 seconds. Separation keeps mono or stereo, writes at most 360 seconds, and takes about 326 seconds of stereo 44.1 kHz input. `turbospark enhance` runs the `turbospark-enhance` binary; see the [CLI reference](../command-line-and-local-api/cli.md).

## In the Desktop App

* **Clean Up, Enhance speech**: reads the take at 48 kHz mono (at most 600 seconds) and saves the cleaned voice as a new take.
* **Clean Up, Separate stems**: reads the take at 44.1 kHz stereo (a mono take is upmixed, at most 326 seconds) and saves the vocals as a new take. The instrumental is not saved from the app; use `--background` on the command line.
* **Reduce background noise** (live text, off by default): appears when the `voice-isolator-v1-realtime` install is present. It cleans only the microphone feed to live text; the saved recording and the final transcription pass use the original audio. If cleaning falls behind real time for more than a second, live text switches back to the raw microphone for the rest of the take.

## For Developers

* Runtime tasks `enhancement` and `separation` open through `ts_audio_session_open`; live streams use `ts_audio_enhance_stream_*` (Swift `AudioEnhanceStream`). See `docs/SWIFT_BINDINGS.md` in the repository.
* Ports, parity and measured speed live in the family READMEs under `crates/audio/src/sts/` (`voice_isolator`, `mossformer2_se`, `mel_roformer`).
* The Core ML live step is a local export, not a catalog install yet: `crates/audio/src/sts/voice_isolator/COREML.md`.
