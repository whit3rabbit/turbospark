# Multitalker Parakeet ASR

## Overview
Speaker-conditioned cache-aware streaming RNN-T. It reuses the Nemotron FastConformer encoder, predictor and joint, and adds NeMo's layer-0 speaker-kernel pre-hook (spk and background feed-forward kernels). Each speaker owns encoder caches and decoder state while weights are shared. A coupled scheduler gates speakers from diarizer predictions, steps each active speaker per chunk, and assembles SegLST sentences. A live path feeds PCM incrementally through a bounded-memory mel frontend. [mod.rs:MultitalkerAsr] [scheduler.rs:CoupledSession] [scheduler.rs:LiveSession]

## Interface
- `MultitalkerProfile`, `MULTITALKER_PARAKEET_STREAMING_V1` pinned checkpoint [mod.rs:MULTITALKER_PARAKEET_STREAMING_V1]
- `MultitalkerAsr` with `open`, `features`, `live_frontend`, `new_stream`, `pre_encode`, `apply_speaker_kernels`, `step`, `step_encoded`, `stream_weights`, `detokenize`, `chunk_rows`, `shift_frames`, `pre_encode_cache_frames`, `drop_extra_pre_encoded` [mod.rs:MultitalkerAsr]
- `SpeakerStream` (`tokens`, `decoded_frames`, `cache_len`), `Features` (`frame_major`, `channel_major`), `PreEncoded`, `Token`, `StepOutput` [mod.rs:SpeakerStream]
- `SpeakerEncoder` trait (device encoder hook) plus blanket impl for `&mut T` [mod.rs:SpeakerEncoder]
- `align_mask`, `strip_space_before_punctuation`; re-exports `StreamEncoderWeights`, `StreamLayerWeights` [mod.rs:align_mask] [mod.rs:strip_space_before_punctuation]
- `LiveFrontend` (`new`, `push`, `finish`, `frames_emitted`, `samples_received`) [frontend.rs:LiveFrontend]
- `CoupledConfig` (`frozen_e`), `DiarizationSource`, `ReplayedDiarization`, `zero_extra_speakers`, `gate`, `targets`, `StepReport`, `CoupledSession` (`new`, `run`, `seglst`, `diarization`, `with_encoder`), `LiveSession` (`new`, `append`, `finish`, `with_cancel`, `with_encoder`, `take_segment_updates`, `seglst`, `processed_frames`, `samples_received`, `diarization`) [scheduler.rs:CoupledSession] [scheduler.rs:LiveSession]
- `SeglstEntry`, `HypothesisView`, `SegmentUpdate`, `SentenceCursor`, `SentenceBuilder` (`new`, `update`, `changes`, `entries`) [seglst.rs:SentenceBuilder]

## Invariants
- Pinned checkpoint is repo `nvidia/multitalker-parakeet-streaming-0.6b-v1` at revision `8749fc71fd6e2d88ef230159bbf2aea69b524ee1`; `open` requires `model_type == "multitalker_asr"`. [mod.rs:MULTITALKER_PARAKEET_STREAMING_V1] [mod.rs:MultitalkerAsr::open]
- Only `ff` speaker kernels on layer 0 with a background kernel and `hidden == encoder_hidden` are accepted; anything else returns `Unsupported`. [mod.rs:check_speaker_kernels]
- Quantized weights (`...linear_q.scales` present) are refused; F32 only. [mod.rs:MultitalkerAsr::open]
- `supported_punctuation` is required in config.json and each entry must be one char, otherwise SegLST word split diverges from NeMo. [mod.rs:supported_punctuation]
- Speaker kernel order matters: the background kernel sees `x` already updated by the speaker kernel. Masks are left-padded with 1.0 (spk) and 0.0 (bg). [mod.rs:MultitalkerAsr::apply_speaker_kernels] [mod.rs:align_mask]
- `features` rejects sample rates other than the config rate and empty or non-finite PCM; mel uses constant padding, no normalization, `normalize_valid_frames: true`. [mod.rs:MultitalkerAsr::features] [mod.rs:MultitalkerAsr::mel_options]
- `step_encoded` rejects encoder output not equal to `rows * hidden` and advances `attn_len` capped at `attn_frames`. [mod.rs:MultitalkerAsr::step_encoded]
- Decoded rows are `valid` only when `keep_all`, else `min(valid, chunk)`. [mod.rs:MultitalkerAsr::decode_step]
- `LiveFrontend::new` requires `center`, positive hop, `pad_to == 0`, and `NemoMelNormalization::None`; frames withheld until `f*hop + fft/2` samples exist; `finish` needs at least two hops and zeroes frames at or past `samples/hop`. [frontend.rs:LiveFrontend::new] [frontend.rs:LiveFrontend::push] [frontend.rs:LiveFrontend::finish]
- `gate` breaks ties toward the lower speaker id, uses a strict `> threshold`, and returns ascending ids. [scheduler.rs:gate]
- Lowest missing speaker ids are allocated lazily up to the largest active id; inactive speakers keep state untouched. [scheduler.rs:Transcripts::transcribe]
- A device encoder must return one output per active speaker, else `SpeechError::Audio`. [scheduler.rs:Transcripts::transcribe]
- `LiveSession` refuses all work after `Finished` or `Failed`; any step error poisons it; non-finite PCM is rejected before consumption. [scheduler.rs:LiveSession::require_open] [scheduler.rs:LiveSession::guarded] [scheduler.rs:LiveSession::append]
- Batch plan drops a tail shorter than `subsampling_factor` (unless it is the only chunk); live `finish` mirrors that. [scheduler.rs:CoupledSession::run] [scheduler.rs:LiveSession::drain]
- Sentence rules port NeMo as written (word continuation split, `round3` half-to-even, capitalize first word); `entries` sorts by start time stably. [seglst.rs:SentenceBuilder::update] [seglst.rs:round3] [seglst.rs:SentenceBuilder::entries]

## Side Effects
- Reads `config.json` and maps `model.safetensors` in `open`. [mod.rs:MultitalkerAsr::open]
- Runs work on the kernels' rayon pool via `ops::in_pool` and `rayon::join` (diarizer one step ahead of ASR in batch `run`). [scheduler.rs:CoupledSession::run] [scheduler.rs:LiveSession::guarded]
- Optional device encoder holds per-speaker caches outside the session; caller resets between recordings. [mod.rs:SpeakerEncoder] [scheduler.rs:CoupledSession::with_encoder]
- No env vars or global statics.

## Verification
- Run: `cargo test -p turbospark-audio stt::multitalker_asr`
