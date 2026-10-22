---
description: "Hardware execution backends: Metal GPU, Apple MLX, Core ML, and Portable CPU support per model family and modality."
icon: microchip
---

# Hardware Acceleration Matrix

TurboSpark is engineered to maximize Apple Silicon performance by routing each computational workload to the optimal hardware engine:
* **Native Metal GPU**: Handcrafted Metal Shading Language (MSL) compute kernels for dense GEMM/GEMV, quantized dequantization (INT4, Q4_K, Q8_0, MXFP4, 1-bit, 2-bit), fused INT4 FFN dispatches, split-KV attention, SWA rings, absorbed MLA attention, and MoE SSD streaming.
* **Apple MLX**: High-level Swift MLX integration used in the desktop app for diffusion image generation (Z-Image Turbo, Qwen-Image 2.1) and zero-shot voice design (Qwen3-TTS). Also provides safetensors intake and conversion.
* **Apple Core ML**: Native Apple Neural Engine (ANE) and GPU pipelines via `crates/coreml` on macOS: EmbeddingGemma 2 embeddings, Parakeet speech-to-text (Redux, Ultra, and the streaming Unified model), pyannote speaker diarization, Unlimited OCR, and the opt-in ANE+Steel prefill split for dense Qwen.
* **Portable CPU Reference**: Platform-independent Rust reference kernels in `crates/compute` and portable DSP in `crates/audio` used for cross-target verification, testing gates, and CPU fallback (`--cpu`).

---

## Model Families & Hardware Backend Matrix

| Family / Role | Native Metal GPU | Apple MLX | Apple Core ML | Portable CPU | Notes |
| --- | :---: | :---: | :---: | :---: | --- |
| **Gemma 4** (`gemma4`) | Yes (Primary) | Intake / safetensors | No | Yes (Reference) | MoE streaming on Metal; ~33-45 tok/s on M4 Max |
| **QwenGdnMoe** (`qwen36`) | Yes (Primary) | Intake / safetensors | No | Yes (Reference) | Batched GEMV prefill, streaming routed experts |
| **QwenGdnDense** (`qwen35`) | Yes (Primary) | Intake / safetensors | No | Yes (Reference) | Steel GEMM prefill, MTP speculation, 1-bit / 2-bit kernels |
| **Qwen4Exp** (`qwen4exp`) | Yes (Primary) | Intake (REAP-288) | No | Yes (Reference) | Hyper-connections, PLE embeddings, IQ2_XS GGUF |
| **DeepSeek2** (`deepseek2`) | Yes (Primary) | No (GGUF intake) | No | Yes (Reference) | Absorbed MLA latent attention, 64 MoE experts |
| **Llama / Mistral** (`llama`)| Yes (Primary) | No (GGUF intake) | No | Yes (Reference) | Standard dense GQA and SwiGLU |
| **Qwen3Moe** (`qwen3moe`) | Yes (Primary) | No (GGUF intake) | No | Yes (Reference) | Per-head Q/K norm before RoPE, MoE streaming |
| **Qwen3Dense** (`qwen3dense`)| Yes (Primary) | Intake / safetensors | No | Yes (Reference) | Dense FFN, Q/K norm |
| **Qwen2Dense** (`qwen2`) | Yes (Primary) | Intake / safetensors | No | Yes (Reference) | Q3_K and Q4_K resident Metal kernels |
| **GptOss** (`gptOss`) | Yes (Primary) | No (GGUF intake) | No | Yes (Reference) | MXFP4 dequant, attention sinks, YaRN rope |
| **MuseGlimmer** (`museGlimmer`)| Yes (Primary)| Intake / safetensors | No | Yes (Reference) | Centered layer norms (1+w), NoPE on full layers |
| **Spark25** (`spark2_5`) | Yes (Primary) | No (GGUF intake) | No | Yes (Reference) | Exact-erF GELU, scalar sigmoid gate, per-class RoPE |
| **MiniMaxM2** (`minimax_m2`) | Yes (Primary) | No (GGUF intake) | No | Yes (Reference) | Full GQA, sigmoid MoE |
| **Qwen3Vl** (`qwen3_vl`) | Yes (Primary) | Intake / safetensors | No | Yes (Reference) | Depth-24 SigLIP2 tower + 3 deepstack mergers |
| **DeepSeek-V4-Flash** | Scaffolded | Scaffolded | No | Scaffolded | Architecture ready for upcoming release |

---

## Multimodal, Image, Audio & Embeddings Matrix

| Task / Model | Native Metal GPU | Apple MLX | Apple Core ML | Portable CPU | Hardware Specifics |
| --- | :---: | :---: | :---: | :---: | --- |
| **Z-Image Turbo** (Image) | Yes (`turbospark-image`) | Yes (App default) | No | No | App uses Swift MLX (1 to 1.5 min on M4 Max); Metal CLI supported |
| **Qwen-Image 2.1** (Image)| No | Yes (App & HTTP) | No | No | Swift MLX mixed 4/8-bit (20s @ 512x512, 90s @ 1024x1024) |
| **Whisper** (Speech-to-Text)| Yes (Primary) | Yes (MLX quants) | No | Yes (Fallback) | Segment-level transcription & realtime WebSocket frames |
| **Qwen3-ASR** (STT) | Yes (Primary) | No | No | Yes (Fallback) | Compact 0.6B streaming speech recognition |
| **Parakeet Redux / Ultra** (STT) | No | No | Yes (macOS 15+) | No | Core ML batch transcription; Unified variant streams live text with its int8 encoder on the ANE |
| **Speaker diarization** | No | No | Yes (macOS 15+) | No | pyannote community-1 export, who-spoke-when turns |
| **Fish Audio Drama 3** (TTS & Clone) | Yes (only) | No | No | No | Headless Qwen3.5-MoE trunk with streamed experts, Metal tower and codec; 44.1 kHz |
| **OmniVoice** (TTS & Clone) | Yes (Default) | No | No | Yes (Fallback) | Multilingual zero-shot TTS, Metal backbone and heads with transparent CPU fallback |
| **Kokoro-82M** (TTS) | Pending | No | Install-only ANE export | Yes (Primary) | 24 kHz mono, English `af_heart`; the server and CLI run the portable CPU path |
| **Qwen3-TTS** (TTS & Clone)| No | Yes (App only) | No | No | 0.6B & 1.7B Base, CustomVoice, VoiceDesign in Swift MLX |
| **MegaTTS3** (Voice Clone) | Yes (Default on macOS)| No | No | Yes (`--cpu`) | Diffusion transformer + WavVAE decoder (5x faster on Metal) |
| **MiniMax Music 3** (Music)| Yes (Primary) | No | No | No | 4-bit Metal diffusion bit-identical to MLX; fast, exact and Turbo 8-step flow profiles |
| **Unlimited OCR** | No | No | Yes (macOS 15+) | No | Page images to text via `turbospark ocr` |
| **EmbeddingGemma 2** (Embed) | No | No | Yes (macOS ANE/GPU)| Yes (Fallback) | Apple Neural Engine acceleration via `crates/coreml` |
| **Arctic-Embed / BGE** | Yes (Primary) | Yes | No | Yes (Reference) | Dense text embeddings, float and base64 output |
| **OpenKind TypeSafe** (Decide)| Yes (via OpenKind) | No | No | Yes | Bundled `openkindd` loopback daemon (`laya:typed-decisions`) |

---

## Detailed Backend Characteristics

### Native Metal GPU (`crates/gpu`)
* **Primary Runtime**: All LLM text forward passes, MoE expert dispatch, and audio synthesis (MegaTTS3, OmniVoice, Fish Audio Drama 3, MiniMax Music) default to Metal on macOS.
* **Autorelease Pool Management**: Encodes command buffers within inner autorelease pools (`gpu::autorelease_pool`), so transient Metal command objects are deallocated every token cycle without process leaks.
* **Slot Cache**: MoE experts are dispatched to a pre-allocated GPU buffer pool. Expert weights stream from SSD directly into GPU-accessible unified memory.
* **JIT Warmup**: Kernel warmup is opt-in via `TURBOSPARK_METAL_KERNEL_WARMUP=1` (default disabled based on Qwen MoE startup experiments).

### Apple MLX (`swift/` & `crates/repack`)
* **Image Diffusion**: The desktop app runs image generation through Swift bindings to MLX. MLX pipelines provide unified memory caching and accelerated matrix operations for diffusion latents.
* **Qwen3-TTS**: Voice design and zero-shot voice cloning in the desktop app run via Swift MLX to leverage dynamic shape attention.
* **Checkpoint Intake**: MLX-quantized safetensors models are ingested directly via the Hugging Face intake path and repacked into `.gturbo` binaries.

### Apple Core ML (`crates/coreml`)
* **Apple Neural Engine (ANE)**: Core ML runs dedicated pipelines with zero CPU or GPU contention: EmbeddingGemma 2 embeddings, Parakeet speech-to-text, speaker diarization, Unlimited OCR, and (opt-in for dense Qwen) the ANE+Steel prefill split.
* **Compute Plan**: `turbospark_coreml::compute_plan` reports the device plan per op, so a Neural Engine placement claim can be checked on the machine that will run it.
* **In-Memory Model Loading**: Core ML artifacts are loaded and compiled via the Objective-C runtime boundary (`objc2-core-ml`), with a compile cache so the first-run compile cost is paid once per machine.

### Portable CPU Reference (`crates/compute`)
* **Cross-Platform Compatibility**: Pure Rust reference kernels that compile on Linux, macOS, and Windows.
* **Verification Baselines**: Used in continuous integration to assert numerical correctness and logit parity against GPU kernels.
* **Fallback Mode**: Activated explicitly with `--cpu` on tools like `turbospark-clone` when running on hardware without Metal acceleration.
