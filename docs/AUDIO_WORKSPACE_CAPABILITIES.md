# Native Audio capability audit

This inventory covers 53 source families, including local Mel-RoFormer,
MossFormer2, SAM-Audio, and DialogueSidon additions. A declared portable method
is not a qualified Metal workflow. `audio_families.json` carries the same discovery
records returned by `ts_audio_capabilities_json`; pinned installation profiles
remain in the Audio catalog and require complete immutable asset plans.

The native worker owns every model and holds the shared text/image/audio permit
until execution stops. Whisper has per-token cancellation. Music has per-native
compute-operation cancellation. Kokoro stops between synthesized sentences.
Qwen's explicit experimental Metal mode stops between prefill/decode rows;
its audio encoder is still CPU and indivisible. Portable Qwen stops after the
current request. No unsupported family is silently routed to Whisper.
Model opening returns a retryable busy error when another heavy job owns the
device, because the caller does not yet have a session handle to cancel.
Loaded sessions queue execution with independently signalable cancellation.

All transcript timing is labeled: Whisper windows and Qwen clips are extents,
not measured alignment. Qwen language and generated-token log probabilities
are retained. Absent words, speakers, tags and confidence remain absent.
Existing STT calls and HTTP task spellings remain compatible; HTTP adapters
use the same native service without changing their public result shape.

| Family | Source operations | Native workspace status |
|---|---|---|
| [stt/canary](../crates/audio/src/stt/canary/mod.rs) | `transcribe`, `transcribe_with_options`, `decode`, `forward` | No native runtime, admission, cancellation and Swift adapter has been qualified for this family |
| [stt/cohere_asr](../crates/audio/src/stt/cohere_asr/mod.rs) | No runnable operation | Detection and explicit refusal only; no runnable decode implementation |
| [stt/fireredasr2](../crates/audio/src/stt/fireredasr2/mod.rs) | `transcribe`, `transcribe_with_options` | No native runtime, admission, cancellation and Swift adapter has been qualified for this family |
| [stt/fun_asr_nano](../crates/audio/src/stt/fun_asr_nano/mod.rs) | `transcribe` | No native runtime, admission, cancellation and Swift adapter has been qualified for this family |
| [stt/glmasr](../crates/audio/src/stt/glmasr/mod.rs) | `transcribe` | No native runtime, admission, cancellation and Swift adapter has been qualified for this family |
| [stt/granite_speech](../crates/audio/src/stt/granite_speech/mod.rs) | `transcribe`, `forward` | No native runtime, admission, cancellation and Swift adapter has been qualified for this family |
| [stt/granite_speech5_ctc](../crates/audio/src/stt/granite_speech5_ctc/mod.rs) | `transcribe` | No native runtime, admission, cancellation and Swift adapter has been qualified for this family |
| [stt/granite_speech_nar](../crates/audio/src/stt/granite_speech_nar/mod.rs) | `transcribe` | No native runtime, admission, cancellation and Swift adapter has been qualified for this family |
| [stt/higgs_audio_3](../crates/audio/src/stt/higgs_audio_3/mod.rs) | `transcribe`, `transcribe_with_options`, `forward` | No native runtime, admission, cancellation and Swift adapter has been qualified for this family |
| [stt/lasr_ctc](../crates/audio/src/stt/lasr_ctc/mod.rs) | `transcribe` | No native runtime, admission, cancellation and Swift adapter has been qualified for this family |
| [stt/mega_asr](../crates/audio/src/stt/mega_asr/mod.rs) | `transcribe`, `transcribe_with_options`, `forward` | No native runtime, admission, cancellation and Swift adapter has been qualified for this family |
| [stt/mms](../crates/audio/src/stt/mms/mod.rs) | `transcribe` | No native runtime, admission, cancellation and Swift adapter has been qualified for this family |
| [stt/moonshine](../crates/audio/src/stt/moonshine/mod.rs) | `transcribe`, `encode`, `decode` | No native runtime, admission, cancellation and Swift adapter has been qualified for this family |
| [stt/moss_music](../crates/audio/src/stt/moss_music/mod.rs) | `transcribe` | No native runtime, admission, cancellation and Swift adapter has been qualified for this family |
| [stt/moss_transcribe_diarize](../crates/audio/src/stt/moss_transcribe_diarize/mod.rs) | `transcribe`, `transcribe_with_options`, `forward` | No native runtime, admission, cancellation and Swift adapter has been qualified for this family |
| [stt/nemo](../crates/audio/src/stt/nemo/mod.rs) | No runnable operation | Codec or numerical component requires a task-specific native adapter |
| [stt/nemotron_asr](../crates/audio/src/stt/nemotron_asr/mod.rs) | `transcribe` | No native runtime, admission, cancellation and Swift adapter has been qualified for this family |
| [stt/parakeet](../crates/audio/src/stt/parakeet/mod.rs) | `transcribe`, `decode` | No native runtime, admission, cancellation and Swift adapter has been qualified for this family |
| [stt/phonon](../crates/audio/src/stt/phonon/mod.rs) | `transcribe`, `transcribe_with_options` | No native runtime, admission, cancellation and Swift adapter has been qualified for this family |
| [stt/qwen2_audio](../crates/audio/src/stt/qwen2_audio/mod.rs) | `transcribe`, `encode` | No native runtime, admission, cancellation and Swift adapter has been qualified for this family |
| [stt/qwen3_asr](../crates/audio/src/stt/qwen3_asr/mod.rs) | `transcribe`, `transcribe_with_options`, `transcribe_with_details`, `forward` | Packed Metal text decoder is explicit opt-in; audio encoder remains CPU; portable execution is a separate opt-in |
| [stt/qwen3_forced_aligner](../crates/audio/src/stt/qwen3_forced_aligner/mod.rs) | `align`, `align_with_language` | No native runtime, admission, cancellation and Swift adapter has been qualified for this family |
| [stt/sensevoice](../crates/audio/src/stt/sensevoice/mod.rs) | `transcribe`, `transcribe_with_options` | No native runtime, admission, cancellation and Swift adapter has been qualified for this family |
| [stt/vibevoice_asr](../crates/audio/src/stt/vibevoice_asr/mod.rs) | `transcribe` | No native runtime, admission, cancellation and Swift adapter has been qualified for this family |
| [stt/voxtral](../crates/audio/src/stt/voxtral/mod.rs) | `transcribe` | No native runtime, admission, cancellation and Swift adapter has been qualified for this family |
| [stt/voxtral_realtime](../crates/audio/src/stt/voxtral_realtime/mod.rs) | `transcribe` | No native runtime, admission, cancellation and Swift adapter has been qualified for this family |
| [stt/wav2vec](../crates/audio/src/stt/wav2vec/mod.rs) | `transcribe` | No native runtime, admission, cancellation and Swift adapter has been qualified for this family |
| [stt/whisper](../crates/audio/src/stt/whisper/mod.rs) | `transcribe`, `encode` | Implemented, checkpoint/quality unqualified |
| [tts/kokoro](../crates/audio/src/tts/kokoro/mod.rs) | `synthesize` | Portable synthesis is callable only with explicit diagnostic opt-in; Metal synthesis is not implemented |
| [music/minimax_music3](../crates/audio/src/music/minimax_music3/mod.rs) | `generate_text` | Implemented, checkpoint/quality unqualified |
| [sts/deepfilternet](../crates/audio/src/sts/deepfilternet/mod.rs) | `enhance` | No native runtime, admission, cancellation and Swift adapter has been qualified for this family |
| [sts/dialogue_sidon](../crates/audio/src/sts/dialogue_sidon/mod.rs) | `extract_features`, `normalize_chunk`, `conditioning`, `sample_latents`, `decode_latents` | Concurrent source in progress; no complete separation entry point, native adapter or checkpoint qualification |
| [sts/mel_roformer](../crates/audio/src/sts/mel_roformer/mod.rs) | `forward` | No native runtime, admission, cancellation and Swift adapter has been qualified for this family |
| [sts/mossformer2_se](../crates/audio/src/sts/mossformer2_se/mod.rs) | `enhance`, `enhance_with_mode`, `forward` | No native runtime, admission, cancellation and Swift adapter has been qualified for this family |
| [sts/sam_audio](../crates/audio/src/sts/sam_audio/mod.rs) | `separate`, `encode_text`, `audio_features`, `velocity` | No native runtime, admission, cancellation and Swift adapter has been qualified for this family |
| [vad/nemotron_diarization](../crates/audio/src/vad/nemotron_diarization/mod.rs) | `generate_stream`, `feed` | No native runtime, admission, cancellation and Swift adapter has been qualified for this family |
| [vad/silero_vad](../crates/audio/src/vad/silero_vad/mod.rs) | `detect`, `timestamps`, `feed` | No native runtime, admission, cancellation and Swift adapter has been qualified for this family |
| [vad/sortformer](../crates/audio/src/vad/sortformer/mod.rs) | `generate_stream`, `feed`, `forward` | No native runtime, admission, cancellation and Swift adapter has been qualified for this family |
| [codec/bigvgan](../crates/audio/src/codec/bigvgan/mod.rs) | `forward` | Codec or numerical component requires a task-specific native adapter |
| [codec/dacvae](../crates/audio/src/codec/dacvae/mod.rs) | `encode`, `decode` | Codec or numerical component requires a task-specific native adapter |
| [codec/descript](../crates/audio/src/codec/descript/mod.rs) | `encode`, `decode_codes`, `compress`, `decompress` | Codec or numerical component requires a task-specific native adapter |
| [codec/ecapa_tdnn](../crates/audio/src/codec/ecapa_tdnn/mod.rs) | `forward` | Codec or numerical component requires a task-specific native adapter |
| [codec/encodec](../crates/audio/src/codec/encodec/mod.rs) | `encode`, `decode_codes` | Codec or numerical component requires a task-specific native adapter |
| [codec/fish_s1_dac](../crates/audio/src/codec/fish_s1_dac/mod.rs) | `encode`, `decode` | Codec or numerical component requires a task-specific native adapter |
| [codec/higgs_audio](../crates/audio/src/codec/higgs_audio/mod.rs) | `encode`, `decode`, `decode_tokens`, `forward` | Codec or numerical component requires a task-specific native adapter |
| [codec/mimi](../crates/audio/src/codec/mimi/mod.rs) | `encode`, `decode` | Codec or numerical component requires a task-specific native adapter |
| [codec/mimo_audio_tokenizer](../crates/audio/src/codec/mimo_audio_tokenizer/mod.rs) | `encode`, `decode` | Codec or numerical component requires a task-specific native adapter |
| [codec/moss_audio_tokenizer](../crates/audio/src/codec/moss_audio_tokenizer/mod.rs) | `encode`, `decode` | Codec or numerical component requires a task-specific native adapter |
| [codec/nemotron_voicechat](../crates/audio/src/codec/nemotron_voicechat/mod.rs) | `encode`, `decode` | Codec or numerical component requires a task-specific native adapter |
| [codec/s3](../crates/audio/src/codec/s3/mod.rs) | `quantize`, `dequantize` | Codec or numerical component requires a task-specific native adapter |
| [codec/snac](../crates/audio/src/codec/snac/mod.rs) | `encode`, `decode` | Codec or numerical component requires a task-specific native adapter |
| [codec/stepaudio2](../crates/audio/src/codec/stepaudio2/mod.rs) | `decode`, `forward` | Codec or numerical component requires a task-specific native adapter |
| [codec/vocos](../crates/audio/src/codec/vocos/mod.rs) | `decode` | Codec or numerical component requires a task-specific native adapter |

## Input, output, and limits

- Transcription accepts finite interleaved mono f32 at 16 kHz. Individual C
  appends are at most 32,000 samples; a job is limited to 30 minutes. Family
  loaders may impose a smaller limit. Qwen3-ASR retains its 20-minute frontend
  ceiling. The app windows recordings before inference.
- Kokoro accepts plain US English, `af_heart`, and speed 0.5 through 2.0 through
  the native service. It returns 24 kHz mono. Other voices are refused. Its
  CPU runtime is an explicitly selected diagnostic path.
- MiniMax accepts caption/lyrics, duration 0 through 360 seconds (exclusive
  lower bound), steps 1 through 30, and an optional seed. It emits interleaved
  44.1 kHz stereo. PCM delivery is bounded to 32,000 samples per callback and
  six minutes total; the model still materializes its bounded final waveform.
- Alignment, diarization, detection, enhancement, separation, language ID,
  and codec signatures differ by family. Their linked READMEs own sample
  rates, layouts and evidence. Their operations are discoverable but unavailable
  as native jobs until dedicated adapters implement those contracts.
- Mel-RoFormer currently accepts planar stereo `[2, samples]` at 44.1 kHz, so
  a public interleaved PCM adapter remains required. MossFormer2 and
  DeepFilterNet are mono 48 kHz enhancement implementations. Source presence
  and reported fixtures do not establish native Metal support. Neither is
  promoted to a runnable product model by this integration.

## Evidence and remaining qualification

Discovery does not run checkpoints or validate historical README claims.
The catalog distinguishes a runnable implementation from qualification and
keeps the original source/evidence links. No newly fabricated asset hashes,
model timings or quality claims are introduced. Current-run tests and actual
checkpoint witnesses are reported separately by the implementation handoff.

The latest app rebuild attempt encountered seven compile errors in the concurrent
DialogueSidon addition: unsigned negation, mixed f32/f64 solver arithmetic and
collection, and an unresolved `sidon_filters` function. Its discovery entry
records declared intermediate operations only. It is unavailable, has no new
pinned install profile, and has not been included in a successfully rebuilt
native archive by this integration. Source changes after that failed attempt
require a fresh build and separate fixture/checkpoint qualification.

Kokoro Metal, the remaining family adapters, measured peak admission reserves,
full-duration MiniMax quality/performance, and every new pinned profile's
network-rot/install gate remain separate work. Memory admission uses current
process physical footprint plus conservative estimates; it is not a measured
maximum-memory guarantee. The worker never labels that estimate as measurement.


### Current native binding witnesses

The added contract tests execute the real native worker, C jobs, and Swift
PCM callbacks on the committed tiny Music Metal fixture. They verify borrowed
buffer copying, terminal states, close ordering, cancellation from a callback,
compute-time cancellation followed by reuse, and event-channel cleanup on unwind.
Catalog tests cover partial profile manifests and keep components unavailable.
Mel-RoFormer boundary tests reject malformed PCM, zero geometry, and checkpoint
linear dimensions that disagree with the configuration; existing numerical
fixtures remain a separate parity gate.

Local checkpoint worker tests also exercised:

| Checkpoint directory | Backend | Witness | Qualification limit |
|---|---|---|---|
| `distil-large-v3` | Whisper Metal | The 3.25-second reference speech produced the expected transcript; a second request cancelled | One local clip, not a fresh pinned `whisper-base` profile gate |
| `qwen3-asr-0.6b-8bit` | Packed Metal decoder and CPU encoder | Same expected transcript, emitted English metadata and finite token log probabilities | Explicit experimental mode; no corpus quality or measured peak-memory qualification |
| `kokoro-82m` | Portable CPU | Short English synthesis returned finite nonzero 24 kHz PCM through bounded callbacks | Diagnostic execution only; no Metal synthesis or task-quality claim |

The similarly named local `distil-whisper-large-v3` directory was refused because
it contained no tokenizer. Checkpoint folders were not re-identified against
managed pin receipts in these witnesses. Full-size MiniMax was not rerun, and
the [existing native BF16 parity divergence](../crates/audio/src/music/minimax_music3/README.md)
remains open. That source separately records successful controlled f32 parity;
this integration does not claim to refresh either comparison. Debug runs on a shared development
machine are not performance evidence.
