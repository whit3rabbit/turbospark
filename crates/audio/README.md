# turbospark-audio

Unified portable audio crate for TurboSpark: DSP primitives, shared neural tensor operations, and audio model implementations organized by role (music, STT, TTS, VAD, STS, codec, LID).

## Architecture Overview

### 1. Signal Processing Primitives
- **`waveform`**: Interleaved f32 audio buffer abstraction with sample rate and channel count.
- **`wav`**: Strict RIFF/WAVE reader (PCM 8/16/24/32, IEEE float 32, extensible format) and writer.
- **`conversion`**: Mixdown, channel duplication, planar interchange, and mono-resample pipelines.
- **`resample`**: Linear and torchaudio-compatible sinc-Hann polyphase resamplers.
- **`fft` / `stft`**: Radix-2 / Bluestein real FFT, PyTorch-convention STFT and iSTFT with window-sum-of-squares normalization.
- **`mel`**: HTK and Slaney mel filterbank construction and power log-mel spectrograms.
- **`nemo_mel`**: NeMo-compatible pre-emphasis and centered Slaney power-mel frontend (Parakeet, Nemotron).
- **`whisper`**: OpenAI Whisper log-mel filterbank (80/128 bands) and 30-second windowing.
- **`dsp`**: Windowing functions (Hann), peak, RMS, gain, and peak normalization.

### 2. Neural Operations & Quantization
- **`ops`**: Pure Rust f32 neural tensor primitives (linear projection, RMSNorm, LayerNorm, SiLU, GELU, RoPE tables/rotations, attention SDPA, Conv1D, Conv2D, ConvTranspose1D).
- **`nn`**: shared owned-weight `Linear`, `LayerNorm`, and `RmsNorm` layer types over the `ops` kernels.
- **`lm`**: shared portable f32 LM backbones (KV caches, RoPE, dense, MoE, and Mamba2 hybrid families) ported from the mlx-audio `lm` subset.
- **`quant`**: MLX groupwise affine dequantization (2, 3, 4, 5, 6, 8-bit) for U32-packed safetensors checkpoints.

### 3. Models Organized by Role
- **`music`**: Music generation model families ([`minimax_music3`](src/music/minimax_music3/README.md)).
- **`stt`**: Speech-to-Text and alignment models ([`canary`](src/stt/canary/README.md), [`cohere_asr`](src/stt/cohere_asr/README.md), [`fireredasr2`](src/stt/fireredasr2/README.md), [`fun_asr_nano`](src/stt/fun_asr_nano/README.md), [`glmasr`](src/stt/glmasr/README.md), [`granite_speech`](src/stt/granite_speech/README.md), [`granite_speech5_ctc`](src/stt/granite_speech5_ctc/README.md), [`granite_speech_nar`](src/stt/granite_speech_nar/README.md), [`higgs_audio_3`](src/stt/higgs_audio_3/README.md), [`lasr_ctc`](src/stt/lasr_ctc/README.md), [`mega_asr`](src/stt/mega_asr/README.md), [`mms`](src/stt/mms/README.md), [`moonshine`](src/stt/moonshine/README.md), [`moss_music`](src/stt/moss_music/README.md), [`moss_transcribe_diarize`](src/stt/moss_transcribe_diarize/README.md), [`multitalker_asr`](src/stt/multitalker_asr/README.md), [`nemotron_asr`](src/stt/nemotron_asr/README.md), [`parakeet`](src/stt/parakeet/README.md), [`phonon`](src/stt/phonon/README.md), [`qwen2_audio`](src/stt/qwen2_audio/README.md), [`qwen3_asr`](src/stt/qwen3_asr/README.md), [`qwen3_forced_aligner`](src/stt/qwen3_forced_aligner/README.md), [`sensevoice`](src/stt/sensevoice/README.md), [`vibevoice_asr`](src/stt/vibevoice_asr/README.md), [`voxtral`](src/stt/voxtral/README.md), [`voxtral_realtime`](src/stt/voxtral_realtime/README.md), [`wav2vec`](src/stt/wav2vec/README.md), [`whisper`](src/stt/whisper/README.md), [`whistle`](src/stt/whistle/README.md)).
- **`tts`**: Text-to-Speech synthesis models ([`arktts`](src/tts/arktts/README.md), [`bailingmm`](src/tts/bailingmm/README.md), [`bark`](src/tts/bark/README.md), [`breeze_tts`](src/tts/breeze_tts/README.md), [`chatterbox`](src/tts/chatterbox/README.md), [`chatterbox_turbo`](src/tts/chatterbox_turbo/README.md), [`confucius4`](src/tts/confucius4/README.md), [`dia`](src/tts/dia/README.md), [`dramabox`](src/tts/dramabox/README.md), [`echo_tts`](src/tts/echo_tts/README.md), [`fish_drama3`](src/tts/fish_drama3/README.md), [`fish_qwen3_omni`](src/tts/fish_qwen3_omni/README.md), [`higgs_audio`](src/tts/higgs_audio/README.md), [`higgs_audio_v3`](src/tts/higgs_audio_v3/README.md), [`indextts`](src/tts/indextts/README.md), [`irodori_tts`](src/tts/irodori_tts/README.md), [`kitten_tts`](src/tts/kitten_tts/README.md), [`kokoro`](src/tts/kokoro/README.md), [`kugelaudio`](src/tts/kugelaudio/README.md), [`llama`](src/tts/llama/README.md), [`longcat_audiodit`](src/tts/longcat_audiodit/README.md), [`megatts3`](src/tts/megatts3/README.md), [`melotts`](src/tts/melotts/README.md), [`moss_tts`](src/tts/moss_tts/README.md), [`moss_tts_nano`](src/tts/moss_tts_nano/README.md), [`omnivoice`](src/tts/omnivoice/README.md), [`outetts`](src/tts/outetts/README.md), [`pocket_tts`](src/tts/pocket_tts/README.md), [`qwen3`](src/tts/qwen3/README.md), [`rumik_oss`](src/tts/rumik_oss/README.md), [`sesame`](src/tts/sesame/README.md), [`soprano`](src/tts/soprano/README.md), [`spark`](src/tts/spark/README.md), [`tada`](src/tts/tada/README.md), [`vibevoice`](src/tts/vibevoice/README.md), [`voxcpm`](src/tts/voxcpm/README.md), [`voxcpm2`](src/tts/voxcpm2/README.md), [`voxtral_tts`](src/tts/voxtral_tts/README.md), [`zonos2`](src/tts/zonos2/README.md)).
- **`vad`**: Voice activity detection and diarization ([`silero_vad`](src/vad/silero_vad/README.md), [`sortformer`](src/vad/sortformer/README.md), [`nemotron_diarization`](src/vad/nemotron_diarization/README.md)).
- **`sts`**: Speech-to-speech and enhancement models ([`deepfilternet`](src/sts/deepfilternet/README.md), [`dialogue_sidon`](src/sts/dialogue_sidon/README.md), [`mel_roformer`](src/sts/mel_roformer/README.md), [`mossformer2_se`](src/sts/mossformer2_se/README.md), [`sam_audio`](src/sts/sam_audio/README.md)).
- **`codec`**: Learned neural codecs and vocoders ([`bigvgan`](src/codec/bigvgan/README.md), [`dacvae`](src/codec/dacvae/README.md), [`descript`](src/codec/descript/README.md), [`ecapa_tdnn`](src/codec/ecapa_tdnn/README.md), [`encodec`](src/codec/encodec/README.md), [`fish_s1_dac`](src/codec/fish_s1_dac/README.md), [`higgs_audio`](src/codec/higgs_audio/README.md), [`mimi`](src/codec/mimi/README.md), [`mimo_audio_tokenizer`](src/codec/mimo_audio_tokenizer/README.md), [`moss_audio_tokenizer`](src/codec/moss_audio_tokenizer/README.md), [`nemotron_voicechat`](src/codec/nemotron_voicechat/README.md), [`s3`](src/codec/s3/README.md), [`snac`](src/codec/snac/README.md), [`spark_bicodec`](src/codec/spark_bicodec/README.md), [`stepaudio2`](src/codec/stepaudio2/README.md), [`vocos`](src/codec/vocos/README.md), [`wavvae`](src/codec/wavvae/README.md)).
- **`lid`**: Spoken language identification ([`lid`](src/lid/README.md)).

Each model family owns a dedicated folder containing its architecture, configuration, weight loader, and a detailed `README.md` covering background, URLs, benchmarks, and model metadata.

## Verification & Portability

The crate is portable plain f32 Rust, free of macOS- or Metal-specific dependencies, building and testing on any target architecture.

```sh
cargo test -p turbospark-audio
cargo check --target x86_64-unknown-linux-gnu -p turbospark-audio
```

Real-checkpoint gates for this crate's models live with their runners in other crates. The OmniVoice reference-transcript A/B (`cargo test --release -p turbospark-runtime --test omnivoice_reference_transcript -- --ignored --nocapture`) clones targets with no reference transcript, the exact one, an ASR transcript of the reference, and a stale one, and scores each clip with a CPU Parakeet oracle. It measures a real contract gap: the port follows mlx-audio, which has no ASR fallback for the reference, so `None` leaves the reference audio with no words, and a transcript that no longer matches the reference audio is spoken in place of the script. The gate asserts finite output only; the comparison is the measurement, recorded in [`src/tts/omnivoice/README.md`](src/tts/omnivoice/README.md).
