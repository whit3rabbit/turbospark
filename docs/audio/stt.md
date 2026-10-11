# Speech-to-Text (STT) Models

Speech-to-Text (STT) models transcribe spoken audio into written text, perform word- and character-level forced alignment, provide timestamps and speaker tags, or handle multimodal speech understanding.

All model implementations are unified under [`crates/audio/src/stt/`](file:///Users/whit3rabbit/Documents/GitHub/turbospark/crates/audio/src/stt/).

## Model Families

| Family | Directory | Architecture | Key Characteristics | Status |
|---|---|---|---|---|
| **Whisper & Distil-Whisper** | [`crates/audio/src/stt/whisper/`](file:///Users/whit3rabbit/Documents/GitHub/turbospark/crates/audio/src/stt/whisper/) | Encoder-Decoder Transformer | 80/128-band Slaney mel, 30s windowing, greedy autoregressive decoding; Metal and CPU runtime | Verified & Server Supported |
| **Qwen3-ASR** | [`crates/audio/src/stt/qwen3_asr/`](file:///Users/whit3rabbit/Documents/GitHub/turbospark/crates/audio/src/stt/qwen3_asr/) | Audio Tower + Qwen3 LM | Grouped-query attention, 128 mel bins, full-prefix/KV decoding; 8-bit quantized profile | Verified & Server Supported |
| **Whistle** | [`crates/audio/src/stt/whistle/`](file:///Users/whit3rabbit/Documents/GitHub/turbospark/crates/audio/src/stt/whistle/) | Conformer + Engram Decoder | Cactus Compute CQ2/CQ4 quantization, Needle `.cact` format, 8-block Conformer, tied CQ head | Verified (CPU) |
| **Moonshine** | [`crates/audio/src/stt/moonshine/`](file:///Users/whit3rabbit/Documents/GitHub/turbospark/crates/audio/src/stt/moonshine/) | Rotary Transformer | Variable-length encoder, low-resource edge transcription; CPU and experimental Metal runner | Verified |
| **Parakeet TDT** | [`crates/audio/src/stt/parakeet/`](file:///Users/whit3rabbit/Documents/GitHub/turbospark/crates/audio/src/stt/parakeet/) | FastConformer Transducer | Token-and-Duration Transducer (TDT), v2/v3 and base-3 ternary quantization (Redux) | Verified |
| **Granite Speech 5.0 TurboCTC** | [`crates/audio/src/stt/granite_speech5_ctc/`](file:///Users/whit3rabbit/Documents/GitHub/turbospark/crates/audio/src/stt/granite_speech5_ctc/) | Encoder-only CTC | 80-band HTK delta frontend, 470M parameters | In Progress |
| **Granite Speech 1B** | [`crates/audio/src/stt/granite_speech/`](file:///Users/whit3rabbit/Documents/GitHub/turbospark/crates/audio/src/stt/granite_speech/) | Conformer + Q-Former + Granite LM | ASR and speech translation over context-blocked Conformer, BLIP-2 QFormer, and Granite LLM | Single-clip parity |
| **Granite Speech 4.1 2B NAR** | [`crates/audio/src/stt/granite_speech_nar/`](file:///Users/whit3rabbit/Documents/GitHub/turbospark/crates/audio/src/stt/granite_speech_nar/) | Shaw Conformer + Bidirectional Editor | Non-autoregressive single-pass ASR with Q-Former projector and two CTC collapse passes | Single-clip parity |
| **MMS** | [`crates/audio/src/stt/mms/`](file:///Users/whit3rabbit/Documents/GitHub/turbospark/crates/audio/src/stt/mms/) | Wav2Vec2 + Adapters | Meta Massively Multilingual Speech supporting 1,000+ languages, 300M / 1B profiles | Verified |
| **Wav2Vec2 base CTC** | [`crates/audio/src/stt/wav2vec/`](file:///Users/whit3rabbit/Documents/GitHub/turbospark/crates/audio/src/stt/wav2vec/) | Post-norm Wav2Vec2 + CTC Head | Classic post-norm architecture with group-norm conv frontend and native CTC head | Single-clip parity |
| **Mega-ASR** | [`crates/audio/src/stt/mega_asr/`](file:///Users/whit3rabbit/Documents/GitHub/turbospark/crates/audio/src/stt/mega_asr/) | Qwen3-ASR + Quality Router | Routed robust ASR switching between base decode and LoRA-adapted path; 8-bit pre-merged profile | Single-clip parity |
| **Qwen3-ForcedAligner** | [`crates/audio/src/stt/qwen3_forced_aligner/`](file:///Users/whit3rabbit/Documents/GitHub/turbospark/crates/audio/src/stt/qwen3_forced_aligner/) | CTC Aligner | Precise word- and character-level timestamps for English and Chinese | Verified |
| **Nemotron ASR** | [`crates/audio/src/stt/nemotron_asr/`](file:///Users/whit3rabbit/Documents/GitHub/turbospark/crates/audio/src/stt/nemotron_asr/) | FastConformer + RNN-T | Nemotron 3.5 ASR architecture, relative positional attention, offline RNN-T decoding | Verified |
| **Canary** | [`crates/audio/src/stt/canary/`](file:///Users/whit3rabbit/Documents/GitHub/turbospark/crates/audio/src/stt/canary/) | FastConformer + Cross-Attention | Multilingual ASR and translation with prompt-driven language/punctuation control | Single-clip parity |
| **Phonon-1** | [`crates/audio/src/stt/phonon/`](file:///Users/whit3rabbit/Documents/GitHub/turbospark/crates/audio/src/stt/phonon/) | Qwen3-ASR + Quint5 Linear | Packed five-value (quint5) decoder weights unpacked to two 2-bit affine planes at load | Single-clip parity |
| **SenseVoice Small** | [`crates/audio/src/stt/sensevoice/`](file:///Users/whit3rabbit/Documents/GitHub/turbospark/crates/audio/src/stt/sensevoice/) | 50-block SANM CTC | Kaldi FBANK/LFR/CMVN frontend, language/emotion/event tags | Single-clip parity |
| **FireRedASR2-AED** | [`crates/audio/src/stt/fireredasr2/`](file:///Users/whit3rabbit/Documents/GitHub/turbospark/crates/audio/src/stt/fireredasr2/) | Conformer + Transformer | 80-bin FBANK, 16-block Conformer encoder, beam-search decoder | Single-clip parity |
| **Fun-ASR-Nano** | [`crates/audio/src/stt/fun_asr_nano/`](file:///Users/whit3rabbit/Documents/GitHub/turbospark/crates/audio/src/stt/fun_asr_nano/) | SANM + Audio Adaptor + Qwen3 LM | 80-bin FBANK, LFR 7x6, tied Qwen3 0.6B decoder | Single-clip parity |
| **GLM-ASR-Nano-2512** | [`crates/audio/src/stt/glmasr/`](file:///Users/whit3rabbit/Documents/GitHub/turbospark/crates/audio/src/stt/glmasr/) | Whisper + Llama | 128-bin log-mel, RoPE encoder, merge-four adaptor, affine 4-bit Llama decoder | Single-clip parity |
| **MOSS-Transcribe-Diarize** | [`crates/audio/src/stt/moss_transcribe_diarize/`](file:///Users/whit3rabbit/Documents/GitHub/turbospark/crates/audio/src/stt/moss_transcribe_diarize/) | Whisper + VQ Adaptor + Qwen3 LM | Timestamped transcription with speaker labels; 4-bit affine Qwen3 decoder | Single-clip parity |
| **MOSS-Music 8B Thinking** | [`crates/audio/src/stt/moss_music/`](file:///Users/whit3rabbit/Documents/GitHub/turbospark/crates/audio/src/stt/moss_music/) | Whisper + DeepStack + Qwen3 LM | Music understanding and lyrics transcription with explicit prompt markers | Single-clip parity |
| **Higgs Audio v3 STT** | [`crates/audio/src/stt/higgs_audio_3/`](file:///Users/whit3rabbit/Documents/GitHub/turbospark/crates/audio/src/stt/higgs_audio_3/) | Whisper + MLP Projector + Qwen3 LM | 128-band encoder, temporal downsampling, Qwen3 decoder with untied LM head | Single-clip parity |
| **LASR CTC / MedASR** | [`crates/audio/src/stt/lasr_ctc/`](file:///Users/whit3rabbit/Documents/GitHub/turbospark/crates/audio/src/stt/lasr_ctc/) | RoPE Conformer CTC | Medical domain CTC recognizer; raw-weights fp32 conversion | Single-clip parity |
| **VibeVoice-ASR** | [`crates/audio/src/stt/vibevoice_asr/`](file:///Users/whit3rabbit/Documents/GitHub/turbospark/crates/audio/src/stt/vibevoice_asr/) | Dual-tokenizer SConv1d + Qwen2 LM | Acoustic and semantic SConv1d encoders, two MLP connectors, Qwen2 1.5B decoder | Single-clip parity |
| **Qwen2-Audio** | [`crates/audio/src/stt/qwen2_audio/`](file:///Users/whit3rabbit/Documents/GitHub/turbospark/crates/audio/src/stt/qwen2_audio/) | Whisper Tower + Qwen2 7B LM | Conv + attention pooling audio encoder, 4-bit Qwen2 7B decoder in chat template | Single-clip parity |
| **Voxtral Mini 3B** | [`crates/audio/src/stt/voxtral/`](file:///Users/whit3rabbit/Documents/GitHub/turbospark/crates/audio/src/stt/voxtral/) | Whisper Large Tower + Mistral LM | 128-band mel, 32-layer audio tower, 4-frame window projector, Mistral text model | Single-clip parity |
| **Voxtral Realtime 4B** | [`crates/audio/src/stt/voxtral_realtime/`](file:///Users/whit3rabbit/Documents/GitHub/turbospark/crates/audio/src/stt/voxtral_realtime/) | Sliding-Window Encoder + Mistral LM | 128-band log-mel, 32-layer causal sliding-window encoder (4x downsample), 4-bit decoder | Single-clip parity |
| **Cohere Transcribe** | [`crates/audio/src/stt/cohere_asr/`](file:///Users/whit3rabbit/Documents/GitHub/turbospark/crates/audio/src/stt/cohere_asr/) | N/A | Gated official checkpoint; mirror decodes incoherently; explicit refusal | Documented Refusal |
| **NeMo Alignment** | [`crates/audio/src/stt/nemo/`](file:///Users/whit3rabbit/Documents/GitHub/turbospark/crates/audio/src/stt/nemo/) | Shared Alignment Helpers | Token, sentence, and chunk-merge alignment algorithms for NeMo transducers | Shared Component |

---

## Server & CLI Support for Speech-to-Text

TurboSpark includes end-to-end server and CLI support for Speech-to-Text, connecting native model runners to standardized OpenAI-compatible APIs and command-line workflows.

### 1. Launching the HTTP Server with STT Models

`turbospark serve` and `turbospark-server` accept `--stt-model` flags to attach speech-to-text models at startup:

```sh
# Attach an STT model alongside text generation
turbospark serve --model gemma4 --stt-model whisper-base-en --port 8080

# Run in audio-only mode (no text generation model required)
turbospark serve --stt-model whisper-base-en --stt-model qwen3-asr-06b-8bit --port 8080

# Attach by local filesystem path or installed catalog alias
turbospark serve --stt-model ~/models/whisper-base-en.gturbo --api-key "$KEY"
```

- **Supported Server Models**:
  - **Whisper Family**: `whisper-tiny-en`, `whisper-base-en`, `whisper-small-en`, `whisper-base`, and MLX 8-bit / 4-bit conversions (`whisper-tiny-en-mlx-8bit`, `whisper-base-en-mlx-4bit`, etc.). Runs on Metal by default on macOS with CPU fallback.
  - **Qwen3-ASR**: `qwen3-asr-06b-8bit` (pinned 0.6B 8-bit model). Runs portable CPU or experimental packed 8-bit Metal text decoder.
- **Dedicated Workers & Channels**:
  - Each loaded audio model runs on its own dedicated worker thread (`RealAudioProvider`), preventing cross-thread contention.
  - Bounded request channel (`STT_QUEUE = 4`) prevents unbounded buffering; additional concurrent requests receive HTTP 429 (`retry-after`).

### 2. HTTP Transcription Endpoints

#### POST `/v1/audio/transcriptions`
Accepts multipart form uploads or raw WAV bodies:

```sh
# Standard multipart upload with verbose JSON response
curl http://127.0.0.1:8080/v1/audio/transcriptions \
  -H "Authorization: Bearer $KEY" \
  -F file=@recording.wav \
  -F model=whisper-base-en \
  -F response_format=verbose_json
```

- **Input Specifications**:
  - Format: WAV only (PCM 8/16/24/32 bits or IEEE float32). The server automatically downmixes and resamples to 16 kHz mono.
  - Constraints: Up to 128 MiB and 30 minutes per request.
- **Response Formats (`response_format`)**:
  - `json` (default): `{"text": "..."}`
  - `verbose_json`: includes detailed segments, offsets, and language tag.
  - `text`, `srt`, `vtt`: plain text or subtitle format.
- **Streaming Mode (`stream=true`)**:
  - Returns Server-Sent Events / NDJSON: sends `start`, segment emissions, and `complete`.
- **Asynchronous Execution (`async=true`)**:
  - Returns HTTP 202 Accepted with a `job_id`. Query status at `GET /v1/audio/jobs/{id}` and fetch results at `GET /v1/audio/jobs/{id}/result`.

#### WebSocket Realtime Transcription: `GET /v1/audio/transcriptions/realtime`
Provides low-latency utterance streaming over WebSockets:

```sh
websocat -H="Authorization: Bearer $KEY" \
  "ws://127.0.0.1:8080/v1/audio/transcriptions/realtime?model=whisper-base-en"
```
Send binary frames of 16 kHz mono PCM (`s16le` or `f32le`), followed by `{"type":"commit"}` to trigger transcription.

### 3. Dynamic Model Lifecycle API

Audio models can be attached, queried, and unloaded at runtime without restarting the server:

- **List Models**: `GET /v1/audio/models` (models also appear in `GET /v1/models` with `capabilities: ["speech_to_text"]`).
- **Load Model**: `POST /v1/audio/models` with JSON `{"model_id": "<alias>"}`. Accepts installed catalog aliases only (path access is restricted for security).
- **Unload Model**: `DELETE /v1/audio/models/{id}` (returns HTTP 409 if active jobs are running).

### 4. CLI Discovery & Model Management

`turbospark-model` manages audio models in `~/.turbospark/models`:

```sh
# List bundled speech models (Whisper variants and Qwen3-ASR)
turbospark-model list-speech

# Pull and verify a speech model
turbospark-model pull whisper-base-en
turbospark-model pull qwen3-asr-06b-8bit

# Inspect install path
turbospark-model path whisper-base-en
```

### 5. In-Crate Standalone CLI Examples

The `crates/audio` crate includes runnable standalone examples for specific STT architectures:

```sh
# Whistle transcription (Needle .cact checkpoint)
cargo run --release -p turbospark-audio --example whistle_transcribe -- \
  /path/to/whistle.cact /path/to/mono-16000.wav [en]

# Parakeet TDT transcription
cargo run --release -p turbospark-audio --example parakeet_transcribe -- \
  /path/to/parakeet-model /path/to/audio.wav

# Moonshine edge transcription
cargo run --release -p turbospark-audio --example moonshine_transcribe -- \
  /path/to/moonshine-model /path/to/audio.wav

# Nemotron ASR transcription
cargo run --release -p turbospark-audio --example nemotron_transcribe -- \
  /path/to/nemotron-model /path/to/audio.wav

# Granite Speech 5 TurboCTC transcription
cargo run --release -p turbospark-audio --example granite5_transcribe -- \
  /path/to/granite-model /path/to/audio.wav
```
