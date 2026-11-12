# Whisper Runner

## Overview
The runtime path for the Whisper speech-to-text family. `WhisperRunner` opens a receipt-verified install directory, loads f32 weights, and transcribes PCM in 30-second windows with greedy decoding. It has two device paths: a CPU reference path (the parity partner) and a Metal engine built at open. Admission checks bound memory before weights are allocated. Windowing, decoding, and output are deterministic for the same PCM on a given device.

## Interface
- `WhisperRunner` [mod.rs:WhisperRunner]: open, transcribe, encode_window, using_metal, from_parts (synthetic tests, CPU only)
- `WhisperRunner::open` [mod.rs:open]: bounded reads, receipt and SHA-256 verification, tokenizer check, weight load, Metal try
- `transcribe` / `transcribe_cancellable` / `transcribe_with_progress` [mod.rs:transcribe]: buffered result with cancel and progress callbacks
- `WhisperTranscription`, `WhisperSegment` [mod.rs:WhisperTranscription]: segments with absolute second offsets plus language code
- `WhisperProgress` [mod.rs:WhisperProgress]: Window and Token checkpoints
- `WINDOW_STRIDE_SAMPLES` [mod.rs:WINDOW_STRIDE_SAMPLES]: 30 s window minus 1 s overlap
- `WindowDecoder`, `DecodedWindow`, `StopReason` [decode.rs:WindowDecoder]: incremental CPU greedy decoder
- `build_prompt`, `argmax_suppressed`, `detect_language` [decode.rs:build_prompt]: SOT grammar prefix, special-token masked argmax, language auto-detection
- `WhisperWeights`, `WhisperEncoderLayerOwned`, `WhisperDecoderLayerOwned` [weights.rs:WhisperWeights]: owned f32 tensors; `load_from_safetensors` dispatches HF or MLX layout
- `WhisperMetalEngine` [metal.rs:WhisperMetalEngine]: resident GPU weights, encoder, cross caches, `step_with_logits` (macOS only)
- `MetalWindowDecoder`, `detect_language_metal` [metal.rs:MetalWindowDecoder]: Metal decode and language detection
- Admission helpers (private) [admission.rs:admit]: `read_bounded`, `host_budget`, `estimated_open_bytes`, `receipt`, `verify_bytes`

## Invariants
- Dispatch is by `SpeechFamily::Whisper`, never by tensor name [mod.rs:family]
- Open is serialized by a static lock so concurrent opens do not spend the same headroom snapshot [mod.rs:open_with_budget]
- Admission runs before weight allocation, with config.json and tokenizer.json read under size limits (1 MiB, 16 MiB); the hard cap is 24 GiB and cannot be raised [admission.rs:MAX_OPEN_BYTES]
- Admission fails closed when process footprint or memory pressure is unreadable or not normal [admission.rs:budget]
- The receipt must match family, n_mels, and memory ceiling, and list config.json, tokenizer.json, and model.safetensors with matching sizes and hex SHA-256 [admission.rs:receipt]
- Tokenizer IDs must be below config vocab_size and consistent with config; special tokens come from the loaded (resolved) tokenizer [mod.rs:open_with_budget]
- Safetensors header is capped at 1 MiB and the file is SHA-256 verified before loading [mod.rs:open_with_budget]
- Missing tensors are errors naming the tensor, so partial checkpoints refuse at open [weights.rs:load_from_safetensors]
- Empty or non-finite PCM is rejected; English-only models accept only en, auto, or empty language [mod.rs:transcribe_controlled]
- Windows are zero-padded to 480000 samples, so encoder length is constant [mod.rs:transcribe_metal]
- Decode is greedy: ids above eot are masked, repeated 8 identical tokens stop, budget is max_target_positions minus prompt, capped at 440 [decode.rs:decode_with_progress]
- Metal engine requires d_model divisible by both head counts and exactly 2*seq mel frames; decoder position must equal the filled cache length [metal.rs:new]
- One Metal engine serves one transcription at a time behind a Mutex [mod.rs:WhisperRunner]
- Metal encode loops run inside `autorelease_pool` [metal.rs:encode_partial]
- A device path is used for a whole call; Metal build failure silently falls back to CPU [mod.rs:try_metal]

## Side Effects
- Reads config.json, tokenizer.json, model.safetensors, and the speech receipt from the install dir [mod.rs:open_with_budget]
- Reads host memory and process footprint via gpu [admission.rs:host_budget]
- Env `TURBOSPARK_WHISPER_DEVICE=cpu` forces the CPU path [mod.rs:try_metal]
- Env `TURBOSPARK_WHISPER_PROFILE=1` prints per-window phase timing to stderr [mod.rs:whisper_profile]
- Allocates Metal buffers, scratch, and command passes; scratch lives for the runner's lifetime [metal.rs:WhisperMetalEngine]
- Process-global static open lock [mod.rs:open_with_budget]

## Verification
- Run: `cargo test -p turbospark-runtime whisper`
