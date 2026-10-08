# Speech-to-Text (STT) Models

This directory contains Speech-to-Text (STT) and speech-text alignment model implementations organized by model family.

The [central inventory](../../MODELS.md#stt) tracks upstream source families, checkpoint profiles, and verification status.

## Model Families

- **[Whisper & Distil-Whisper](whisper/README.md)**: OpenAI Whisper encoder-decoder architecture with 80/128-band Slaney log-mel frontend and autoregressive greedy decoding.
- **[Moonshine](moonshine/README.md)**: Useful Sensors Moonshine encoder-decoder architecture optimized for resource-constrained edge transcription.
- **[Parakeet TDT](parakeet/README.md)**: FastConformer transducer with Token-and-Duration Transducer (TDT) decoding, supporting v2, v3, and Redux ternary quantization.
- **[SenseVoice Small](sensevoice/README.md)**: 50-block SANM CTC recognizer with language, emotion, and event tags.
- **[FireRedASR2-AED](fireredasr2/README.md)**: 16-block Conformer encoder and Transformer beam-search decoder for English ASR.
- **[Fun-ASR-Nano-2512](fun_asr_nano/README.md)**: SANM speech encoder and audio adaptor with a tied Qwen3 0.6B decoder.
- **[GLM-ASR-Nano-2512](glmasr/README.md)**: Whisper encoder with RoPE, merge-four audio adaptor, and affine 4-bit Llama decoder.
- **[Granite Speech 5.0 TurboCTC](granite_speech5_ctc/README.md)**: IBM Granite Speech encoder-only CTC model with precise HTK delta-feature frontend.
- **[Granite Speech 4.1 2B NAR](granite_speech_nar/README.md)**: single-pass non-autoregressive ASR over a Shaw-relative-position Conformer, a Q-Former window projector, and a bidirectional Granite editor with two CTC collapse passes.
- **[Granite Speech 1B](granite_speech/README.md)**: IBM Granite Speech ASR and speech translation over a context-blocked Conformer encoder, a BLIP-2 style QFormer projector, and the Granite LLM with its four multipliers.
- **[MMS (Massively Multilingual Speech)](mms/README.md)**: Meta MMS Wav2Vec2/adapter acoustic model supporting 1,000+ languages.
- **[Wav2Vec2 base CTC](wav2vec/README.md)**: the classic post-norm Wav2Vec2 family with a group-norm conv frontend and the checkpoint's own CTC head; the stable-layer-norm variant lives in the MMS port.
- **[Qwen3-ASR](qwen3_asr/README.md)**: Qwen3 ASR speech-to-text architecture with grouped-query attention and RoPE.
- **[Mega-ASR](mega_asr/README.md)**: routed robust ASR over the shared Qwen3-ASR backbone: an audio-quality router switches between the base decode path and a LoRA-adapted robust path, with an always-on-robust pre-merged 8-bit profile.
- **[Qwen3-ForcedAligner](qwen3_forced_aligner/README.md)**: CTC-based forced alignment producing word- and character-level timestamps for English and Chinese.
- **[Nemotron ASR](nemotron_asr/README.md)**: NVIDIA Nemotron 3.5 ASR architecture with FastConformer encoder and RNN-T decoder.
- **[Canary](canary/README.md)**: NVIDIA Canary-1B-v2 multilingual ASR and translation with a FastConformer encoder, cross-attention transformer decoder, and prompt-driven language/punctuation control.
- **[Phonon-1](phonon/README.md)**: Fermion Research English ASR over the Qwen3-ASR backbone with packed five-value (quint5) decoder linears materialized to two 2-bit affine planes at load.
- **[MOSS-Transcribe-Diarize](moss_transcribe_diarize/README.md)**: timestamped transcription with speaker labels over a whisper encoder, a Linear-SiLU-Linear-LayerNorm VQ adaptor, and the shared Qwen3 decoder with an affine 4-bit backbone.
- **[Higgs Audio v3 STT](higgs_audio_3/README.md)**: Boson AI transcription over a 128-band whisper-style encoder, an MLP projector with temporal downsample, and the shared Qwen3 decoder with an untied LM head.
- **[LASR CTC / MedASR](lasr_ctc/README.md)**: Google's medical-domain Conformer-style CTC transcriber over a RoPE encoder with scaled-residual feed-forward and convolution blocks, ported around the broken mlx-audio load path with a raw-weights fp32 MLX conversion as the pinned profile.

- **[VibeVoice-ASR](vibevoice_asr/README.md)**: Microsoft's dual-tokenizer transcriber (acoustic and semantic SConv1d encoders, two MLP connectors, and a Qwen2 1.5B decoder with inline speaker markers), pinned to the streaming checkpoint's offline single-window path with a scipy-compatible kaiser-best resampling frontend.

- **[Qwen2-Audio](qwen2_audio/README.md)**: Qwen2-Audio offline transcription over a whisper-family conv+attention-pooling encoder, a linear projector, and the 4-bit Qwen2 7B decoder spliced into an `<|AUDIO|>` chat prompt.

- **[Voxtral Realtime 4B](voxtral_realtime/README.md)**: Mistral's streaming speech-to-text family pinned to its offline buffered path: 128-band Slaney log-mel, a 32-layer causal sliding-window encoder with a 4x downsample adapter, and the time-conditioned Mistral decoder greedy-decoded over `[BOS]` + streaming-pad prompts with Tekken detokenization.

- **[Cohere Transcribe](cohere_asr/README.md)**: refused family; the official pin is access-gated and the public mirror decodes incoherently through the reference itself, so the module detects the distribution and refuses with the documented reason.

- **[MOSS-Music 8B Thinking](moss_music/README.md)**: music understanding and lyrics ASR over a conv-downsampling whisper-family encoder, deepstack mergers, and the shared Qwen3 decoder; pins an explicit transcription prompt because the upstream default describes music.

- **[Voxtral Mini 3B](voxtral/README.md)**: Mistral's speech understanding model over the whisper-large-v3 audio tower, a 4-frame window projector, and a llama-style text model.

## Layout and Conventions

The [shared NeMo alignment module](nemo/mod.rs) ports upstream
`mlx_audio/stt/models/nemo/alignment.py` (timestamped token, sentence, and
chunk-merge helpers) for the NeMo transducer families.

Each model family owns its subfolder containing:
- Architecture and decoder modules
- Pinned profile configs and weight loaders
- Family `README.md` documenting upstream references, URLs, benchmarks, and model parameters
- Test suites asserting numeric parity against pinned MLX / Python reference golden tensors
