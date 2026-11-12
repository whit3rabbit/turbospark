# SenseVoice Small CTC

## Overview
Offline SenseVoice Small speech recognizer: a Kaldi FBANK plus low-frame-rate stacking frontend (frontend.rs), four rich-tag query embeddings prepended to the features, a 50-block SANM encoder plus 20 timestamp-predictor blocks (sanm.rs, shared with Fun-ASR-Nano), and a CTC head decoded greedily and detokenized through a hand-parsed SentencePiece model (mod.rs). The first four CTC rows yield language, emotion, and event tags.

## Interface
- `SenseVoiceProfile`, `SENSEVOICE_SMALL` pinned repo and revision [mod.rs:SENSEVOICE_SMALL]
- `SenseVoiceOutput` (text, language, emotion, event, token_ids) [mod.rs:SenseVoiceOutput]
- `SenseVoiceSmall` with `load`, `profile`, `transcribe`, `transcribe_with_options` [mod.rs:SenseVoiceSmall]
- `frontend::FrontendConfig`, `frontend::extract_features` (`pub(super)`), `frontend::compute_fbank` (`pub(in crate::stt)`) [frontend.rs:extract_features]
- `sanm::SanmConfig`, `sanm::FsmnLayout`, `sanm::SanmEncoderLayer`, `sanm::add_sinusoidal_positions` (pub(crate)) [sanm.rs:SanmEncoderLayer]

## Invariants
- Config is refused unless every dimension equals the pinned profile (vocab 25055, input 560, output 512, 4 heads, 2048 units, 50 blocks, 20 tp blocks, kernel 11, shift 0, 16 kHz, 80 mels, LFR 7/6) [mod.rs:Config::load].
- Only `normalize_before=true` and the hamming window are accepted [mod.rs:Config::load].
- Language selection is limited to auto, zh, en, yue, ja, ko, nospeech; others return `SpeechError::Input` [mod.rs:transcribe_impl].
- Encoder input is 4 query rows (language, 1, 2, ITN 14 or 15) then features; CTC text is read from logits row offset 4 over `feature_frames` only [mod.rs:transcribe_impl].
- Feature ordering: FBANK, LFR stack with left pad `(lfr_m-1)/2` and right clamp, then `(x + mean) * istd` [frontend.rs:extract_features].
- Empty or non-finite audio and audio shorter than one 25 ms frame are rejected; CMVN length must equal `num_mels * lfr_m` [frontend.rs:extract_features].
- FBANK: per-frame mean removal, 0.97 pre-emphasis, hamming, energy floor 1e-8 before ln [frontend.rs:compute_fbank].
- `am.mvn` must give two vectors of width 560; SentencePiece piece count must equal vocab [mod.rs:parse_mvn, mod.rs:SenseVoiceSmall::load].
- SentencePiece decode skips control/unused kinds 3 and 5, maps kind 2 to U+FFFD, and unpacks byte pieces [mod.rs:decode_sentencepiece].
- CTC blank id is 0 and repeats collapse only when not separated by blank [mod.rs:greedy_ctc].
- FSMN left padding `(kernel-1)/2 + shift` must not exceed `kernel-1` (checked through the per-family `validate` fn); the residual skip applies only when input width equals output width [sanm.rs:SanmAttention::load, sanm.rs:SanmEncoderLayer::forward].
- Attention scores are O(rows^2) per head, so cost grows quadratically with audio length [sanm.rs:SanmAttention::forward].

## Side Effects
- Reads `config.json`, `model.safetensors`, `am.mvn`, `chn_jpn_yue_eng_ko_spectok.bpe.model` from the model dir; never downloads [mod.rs:SenseVoiceSmall::load].
- Builds an FFT plan and mel weights on every `compute_fbank` call; no caches, statics, env vars, threads, or GPU use [frontend.rs:compute_fbank].
- Optional stage trace is collected only by the crate-private `transcribe_impl` for tests [mod.rs:capture_stage].

## Verification
- Run: `cargo test -p turbospark-audio stt::sensevoice`
- Pinned-checkpoint test is `#[ignore]` and needs `TURBOSPARK_SENSEVOICE_MODEL_DIR` [mod.rs:pinned_checkpoint_matches_mlx_transcript_and_tags]
