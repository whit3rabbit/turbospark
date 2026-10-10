# Qwen3-TTS in Audio

TurboSpark runs Qwen3-TTS through native Swift MLX. The Rust
`crates/audio/src/tts` module currently implements Kokoro, not Qwen3-TTS.
The Audio catalog installs Qwen checkpoints, but Rust AudioSession and the
HTTP Audio API do not execute this family. Their refusal remains explicit.

The implementation follows the
[official model roles](https://github.com/QwenLM/Qwen3-TTS) and the
[Qwen demo's input contracts](https://huggingface.co/spaces/Qwen/Qwen3-TTS/blob/main/app.py).
It provides three native destinations, sharing encrypted storage, playback,
presets, history, and export with the existing Audio workspace.

| Destination | Checkpoint | Inputs |
|---|---|---|
| Text-to-Speech | CustomVoice 0.6B or 1.7B | Text, language, preset speaker; style instructions on 1.7B |
| Voice Cloning | Base 0.6B or 1.7B | Text, language, reference audio, matching reference transcript |
| Voice Design | VoiceDesign 1.7B | Text, language, natural-language voice description |

Cloning also offers voice embedding only, without a transcript. The reference
documents lower similarity for that mode. It is an explicit choice, not a
fallback when transcript conditioning fails. Voice Design output can be sent
directly to Voice Cloning with its generated text as the reference transcript.

## Workflow

1. Open Audio, then choose a destination. Its model picker shows compatible
   variants. Download a pinned model or select a complete local model folder.
2. Write the text to speak. Blank lines divide sections; each generation
   creates a separate take. A single request is limited to 6,000 characters.
3. For Voice Cloning, import audio or choose a saved recording. MP3, WAV,
   M4A, FLAC, AIFF, and other macOS-decodable audio use AVFoundation. The
   original recording stays in the encrypted library.
4. Select a 1 to 30 second reference region and preview it. Changing the
   region clears its transcript because text from another region is invalid
   conditioning. Enter the exact words spoken in that region, or explicitly
   select voice embedding only.
5. For Voice Design, describe the voice's age, accent, pitch, pace, and emotion.
   CustomVoice 1.7B supports optional style instructions with a preset speaker.
6. Generate, listen, compare takes, and export WAV, M4A, FLAC, or AIFF through
   the existing audio export controls. Choose Use as reference voice to reuse
   a generated voice. Save a preset to retain the reference and settings.

All synthesis emits mono 24 kHz audio. Reference decoding mixes channels and
resamples to 24 kHz. Language Auto and the ten checkpoint languages are offered.
Qwen's autoregressive controls include temperature, seed, and token limit.
The Kokoro speed setting is hidden for Qwen because it is not a Qwen inference
control. A generation that reaches the token limit fails rather than saving
truncated speech.

## Models and loading

Five managed 8-bit profiles pin the Base and CustomVoice 0.6B/1.7B checkpoints
and VoiceDesign 1.7B. The immutable revisions, file sizes, and SHA-256 values
are in `crates/catalog/src/audio_assets.json`. Each plan includes the talker,
text tokenizer, speech codec configuration, and speech codec weights.
The app performs cancellable full byte verification before loading a managed
model. Tokenizer conversion uses temporary storage and does not alter the
verified checkpoint.

Local folders also accept BF16 and affine MLX 4-bit or 8-bit checkpoints with
group size 64. Select the matching model role and size; local precision may
differ from the managed profile. Other quantization schemes and incompatible
geometry are refused. Local folders have no managed receipt and are identified
by their configuration, not their name. They are saved as security-scoped bookmarks.
BF16 acceptance is a loader contract, not a claim of BF16 real-model validation.

The vendored source and local corrections are listed in
[UPSTREAM.md](../swift/TurboSparkApp/Vendor/Qwen3TTS/UPSTREAM.md).
The speaker frontend is checked against the pinned Python MLX reference.
The app owns a detached producer and holds the shared native heavyweight
permit through completion or cancellation drain. Profile teardown cancels and
drains that producer before reference asset keys expire. Models are released
after a speech job, then unused MLX GPU buffers are cleared while the job still
holds the heavyweight permit. Admission uses a conservative weight-based
estimate, not measured peak memory.

Reference clips are normalized encrypted asset references in saved takes,
presets, and per-mode preferences. Deleting an original library item does not
collect a reference still used by one of these records. Open profile exports
rewrite these references to portable asset paths and remove model bookmarks.

## Verification

Build the native archive before app tests, using the app's symbol-localized
archive when running the app suite:

```sh
make swift-test
make swift-test-app
cd swift/TurboSparkApp/Vendor/Qwen3TTS
swift test
```

The provider's real-model runner accepts a local model and writes a WAV:

```sh
cd swift/TurboSparkApp/Vendor/Qwen3TTS
swift run Qwen3TTSSmoke /path/to/Base /tmp/clone.wav \
  /path/to/reference.mp3 'Exact reference transcript' \
  'Text to speak with the cloned voice.'
```

Omit the reference arguments for CustomVoice or VoiceDesign. Optional positional
arguments after the target text select a speaker/style or voice description.
The app-worker gate imports a real reference into an isolated encrypted vault,
clones through the production worker, saves a take, and reopens its WAV export:

```sh
cd swift/TurboSparkApp
TURBOSPARK_TEST_QWEN3_TTS=/path/to/Base \
TURBOSPARK_TEST_VOICE_REFERENCE=/path/to/reference.mp3 \
  swift test --filter AudioVoiceTests
```

That opt-in test uses the supplied test recording's known transcript. Update
the fixture text when using a different recording. Ordinary tests are offline.
Network pin verification is explicit:

```sh
cargo test -p turbospark-catalog --test audio_catalog_network \
  pinned_audio_asset_graph_matches_immutable_upstream -- --ignored
```

The real token-exhaustion regression uses a CustomVoice checkpoint:

```sh
cd swift/TurboSparkApp/Vendor/Qwen3TTS
TURBOSPARK_TEST_QWEN3_TTS_CUSTOM=/path/to/CustomVoice swift test
```

Validation witnesses on 2026-10-09:

- All five managed 8-bit profiles passed immutable upstream asset verification
  and real native MLX generation: Base and CustomVoice at 0.6B and 1.7B,
  plus VoiceDesign 1.7B.
- Base 0.6B also passed embedding-only cloning and a local affine 4-bit
  checkpoint run. The Base cloning runs used the authorized `flavio.mp3`
  recording and its matching transcript.
- All seven runs reached the model's end token and produced finite, non-silent
  24 kHz PCM. Independent local Whisper transcription recovered the requested
  English sentence from each clip, allowing punctuation differences.
- The production app worker imported the MP3 into an isolated encrypted
  library, generated a take, saved its reference metadata, and reopened its
  WAV export. A built-app UI check also imported the MP3, cloned a sentence,
  exported a WAV, and exposed WAV, FLAC, AIFF, and M4A export choices. The
  reference transcript and local model selection survived quit and relaunch.
- The real app-worker test verified that no cached MLX GPU buffers remained
  after completion. Built-app cancellation returned to the previous completed
  take with an Interrupted status and no empty generated take.
- The speaker frontend passed its pinned Python golden. Regeneration preserved
  the numeric values exactly. Replacing the symmetric window with a periodic
  window made the parity test fail; the restored implementation passed.
- The real one-token budget test refused incomplete speech. Removing the
  exhaustion guard in an isolated package copy made that test fail; the
  original implementation passed.
- Real codec weights decoded zero-valued tokens with their complete frame
  lengths. Treating zero as padding in an isolated copy made the regression
  fail; all six provider tests passed on the original implementation.

Short real-model witnesses, finite PCM checks, frontend parity, persistence,
and ABI tests do not establish corpus quality, multilingual pronunciation,
speaker similarity, sustained peak memory, or release qualification. Keep
listening evaluation, long-script behavior, accessibility, and release gates
separate from these checks.
