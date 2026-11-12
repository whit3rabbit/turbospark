# Confucius4 TTS
## Overview
Multilingual zero-shot voice-cloning TTS. Ports mlx-audio `confucius4` (single batch, portable f32): w2v-BERT semantic condition, CAMPPlus style embedding, ECAPA-style prefix, GPT-2 text-to-semantic (T2S) decoder, S2A flow-matching mel estimator, then the shared BigVGAN vocoder. Tokenizer and reference-audio decode are caller concerns [mod.rs:Model::generate].

## Interface
- `Model` (open, open_with_config, sample_rate, compute_condition, prefix_embeddings, t2s_generate, t2s_step_logits, s2a_solve, vocode, generate) [mod.rs:Model]
- `ModelConfig` (from_json) [mod.rs:ModelConfig]
- `Generated` {samples, sample_rate, codes} [mod.rs:Generated]
- `ReferenceCondition` type alias [mod.rs:ReferenceCondition]
- `language_token(lang)` [mod.rs:language_token]
- `T2S`, `T2SConfig`, `BOS` (8192), `EOS` (8193), `sample_distribution` [t2s.rs:T2S] [t2s.rs:BOS] [t2s.rs:EOS] [t2s.rs:sample_distribution]
- `S2A`, `S2AConfig` (build_mu, forward, solve_euler) [s2a.rs:S2A]
- `W2VBert`, `W2VBertConfig` (hidden17) [w2vbert.rs:W2VBert]
- `Prefix`, `PrefixConfig`, `conv_same` (text_emb, cond_emb) [prefix.rs:Prefix] [prefix.rs:conv_same]
- `CampPlus`, `CampPlusConfig` (inference) [campplus.rs:CampPlus]
- `fbank_160`, `kaldi_fbank`, `slaney_mel`, `ref_mel` [dsp.rs:fbank_160] [dsp.rs:kaldi_fbank] [dsp.rs:slaney_mel] [dsp.rs:ref_mel]

## Invariants
- Stats tensors must hold at least `cond_dim` values, else open fails (indexing is `index % cond_dim`) [mod.rs:check_stats_len]
- Non-greedy T2S requires a finite temperature > 0 [t2s.rs:validate_temperature]
- Semantic tokens and positions must be non-negative and within vocab/max_positions [t2s.rs:T2S::semantic_row]
- T2S generation prefills [cond, text, BOS], stops on EOS, returns codes with BOS/EOS stripped and latent rows `1+Tt .. rows-2` [t2s.rs:T2S::generate]
- Text ids and S2A semantic codes must be non-negative and inside their tables [prefix.rs:Prefix::text_emb] [s2a.rs:S2A::build_mu]
- Reference audio shorter than 400 samples is refused [dsp.rs:fbank_160]
- S2A noise `z` is channel-major `[mels, rows]`; the prompt region of x is zero [s2a.rs:S2A::solve_euler]
- Fixed solve schedule: 26-point linspace over [0,1], cfg 0.7 [mod.rs:Model::s2a_solve]

## Side Effects
- Reads config.json and safetensors: w2vbert_mlx, t2s_model, s2a_mlx, bigvgan_mlx, campplus, w2v_stats, fbank_filters (npz replaced by safetensors) [mod.rs:Model::open_with_config]
- RNG: crate-local threefry keyed by `seed` for T2S sampling and `seed + 1` for the S2A noise plane [mod.rs:Model::generate] [mod.rs:Model::t2s_generate]
- No env vars, threads, or logging [mod.rs:Model]

## Verification
- Run: `cargo test -p turbospark-audio tts::confucius4`
