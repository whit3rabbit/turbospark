# STT

## Overview
Speech-to-text model families. `mod.rs` declares one submodule per family and re-exports each family's model type, profile type, and pinned profile constant [mod.rs:pub mod, pub use]. Several families share code: qwen3_asr supplies a decoder and tokenizer loader reused by other families, nemo supplies alignment types, and wav2vec supplies helpers reused by mms.

## Interface
- Re-exports such as `Canary`, `Parakeet*`, `WhisperModel`, `Qwen3Asr`, `Whistle` [mod.rs:pub use]
- `canary`: [./canary/EGM.md](./canary/EGM.md) - Canary-1B-v2 8-bit ASR/translation
- `cohere_asr`: [./cohere_asr/EGM.md](./cohere_asr/EGM.md) - Cohere Transcribe detection, load always refused
- `fireredasr2`: [./fireredasr2/EGM.md](./fireredasr2/EGM.md) - FireRedAsr2-AED Conformer beam-search ASR
- `fun_asr_nano`: [./fun_asr_nano/EGM.md](./fun_asr_nano/EGM.md) - Fun-ASR-Nano-2512 SANM + Qwen3 decoder
- `glmasr`: [./glmasr/EGM.md](./glmasr/EGM.md) - GLM-ASR-Nano 4-bit
- `granite_speech`: [./granite_speech/EGM.md](./granite_speech/EGM.md) - Granite Speech ASR/translation
- `granite_speech5_ctc`: [./granite_speech5_ctc/EGM.md](./granite_speech5_ctc/EGM.md) - Granite Speech 5 TurboCTC
- `granite_speech_nar`: [./granite_speech_nar/EGM.md](./granite_speech_nar/EGM.md) - Granite Speech 4.1 non-autoregressive
- `higgs_audio_3`: [./higgs_audio_3/EGM.md](./higgs_audio_3/EGM.md) - Higgs Audio v3 STT with VAD chunking
- `lasr_ctc`: [./lasr_ctc/EGM.md](./lasr_ctc/EGM.md) - LASR CTC (MedASR)
- `mega_asr`: [./mega_asr/EGM.md](./mega_asr/EGM.md) - Routed robust Qwen3-ASR
- `mms`: [./mms/EGM.md](./mms/EGM.md) - MMS English Wav2Vec2 CTC with adapters
- `moonshine`: [./moonshine/EGM.md](./moonshine/EGM.md) - Moonshine English ASR
- `moss_music`: [./moss_music/EGM.md](./moss_music/EGM.md) - MOSS-Music lyrics ASR
- `moss_transcribe_diarize`: [./moss_transcribe_diarize/EGM.md](./moss_transcribe_diarize/EGM.md) - MOSS transcribe + diarize
- `multitalker_asr`: [./multitalker_asr/EGM.md](./multitalker_asr/EGM.md) - Multitalker streaming RNN-T
- `nemo`: [./nemo/EGM.md](./nemo/EGM.md) - NeMo alignment data model and chunk merges
- `nemotron_asr`: [./nemotron_asr/EGM.md](./nemotron_asr/EGM.md) - Nemotron 3.5 ASR streaming FastConformer RNN-T
- `parakeet`: [./parakeet/EGM.md](./parakeet/EGM.md) - Parakeet TDT and Redux
- `phonon`: [./phonon/EGM.md](./phonon/EGM.md) - Phonon-1 packed quint5 STT
- `qwen2_audio`: [./qwen2_audio/EGM.md](./qwen2_audio/EGM.md) - Qwen2-Audio-7B 4-bit
- `qwen3_asr`: [./qwen3_asr/EGM.md](./qwen3_asr/EGM.md) - Qwen3-ASR (decoder shared by several families)
- `qwen3_forced_aligner`: [./qwen3_forced_aligner/EGM.md](./qwen3_forced_aligner/EGM.md) - Qwen3 forced aligner
- `sensevoice`: [./sensevoice/EGM.md](./sensevoice/EGM.md) - SenseVoice Small CTC
- `vibevoice_asr`: [./vibevoice_asr/EGM.md](./vibevoice_asr/EGM.md) - VibeVoice-ASR streaming 1.5B
- `voxtral`: [./voxtral/EGM.md](./voxtral/EGM.md) - Voxtral Mini 3B
- `voxtral_realtime`: [./voxtral_realtime/EGM.md](./voxtral_realtime/EGM.md) - Voxtral Mini 4B Realtime
- `wav2vec`: [./wav2vec/EGM.md](./wav2vec/EGM.md) - Wav2Vec2 CTC and shared helpers
- `whisper`: [./whisper/EGM.md](./whisper/EGM.md) - Whisper / Distil-Whisper
- `whistle`: [./whistle/EGM.md](./whistle/EGM.md) - Whistle (Needle 3) ASR

## Invariants
- Each family is pinned to specific checkpoint profiles; see each child contract. Module list and re-exports must stay in sync [mod.rs:pub mod].

## Side Effects
- None at this level; model loading side effects (file reads, temp files) are documented per child [mod.rs].

## Verification
- Run: `cargo test -p turbospark-audio stt::`
