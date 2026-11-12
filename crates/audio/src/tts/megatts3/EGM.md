# MegaTTS3 Voice-Cloning TTS
## Overview
Port of ByteDance MegaTTS3 (English only): prompt aligner, G2P, duration LM, latent-diffusion DiT with three-row guidance, and the WavVAE codec, joined by the `MegaTts3` facade. Parity is checked against the PyTorch classes (no mlx-audio port exists). Output is 24 kHz mono.

## Interface
- `MegaTts3` (open, open_with_backend, prepare_prompt, synthesize, synthesize_controlled, can_clone) [facade.rs:MegaTts3]
- `SynthParams`, `Quality`, `VoicePrompt`, `SynthProgress` [facade.rs:SynthParams] [facade.rs:Quality]
- `plan_chunks`, `activation_reserve_bytes`, `resample_reference` [facade.rs:plan_chunks]
- `SAMPLE_RATE`, `MODEL_TYPE`, `MAX_PROMPT_SECONDS` [facade.rs:MAX_PROMPT_SECONDS]
- `Aligner`, `AlignerConfig`, `AlignedPrompt`, `align_tokens`, `split_ph_timestamp`, `pad_prompt_audio`, `resample_for_aligner`, `prompt_mel` [aligner.rs:Aligner] [aligner.rs:prompt_mel]
- `Dit`, `DitConfig`, `DitCondition` [dit.rs:Dit]
- `DurationLm`, `DurationConfig`, `finalize_durations`, `length_regulate` [duration.rs:DurationLm]
- `G2p` (phonemize, generate) [g2p.rs:G2p]
- `RelTransformer`, `RelTransformerConfig` [rel_transformer.rs:RelTransformer]
- `SamplerParams`, `sample`, `sample_controlled`, `sway_schedule`, `amo_step` [sampler.rs:sample_controlled]
- `Randomness` trait, `SeededRng`, `ReplayRng` [rng.rs:Randomness]
- `Crossfader`, `CROSSFADE_SECONDS` (0.16 s), `integrated_loudness`, `match_loudness`, `combine_segments` [postprocess.rs:Crossfader] [postprocess.rs:CROSSFADE_SECONDS]
- `ensure_english`, `normalize_english`, `chunk_english`, `cardinal_words`, `ordinal_words` [text.rs:normalize_english]

## Invariants
- Voice cloning is refused with `Unsupported` when the WavVAE has no encoder [facade.rs:prepare_prompt_with]
- Prompt audio must be non-empty and finite; only the first `MAX_PROMPT_SECONDS` (28) are used [facade.rs:prepare_prompt_with] [facade.rs:MAX_PROMPT_SECONDS]
- Root `config.json`, if present, must have `model_type == "megatts3"` and a `voice_clone` flag matching encoder presence [facade.rs:check_root_config]
- Random draw order is the reference order: one normal draw for the prompt, then per chunk two uniform and `1 + steps` normal draws [rng.rs:Randomness]
- Chunk latent frames are capped at `MAX_LATENT_FRAMES` (4096) [facade.rs:MAX_LATENT_FRAMES]
- Sampler steps must be in 1..=1000 [sampler.rs:sample_controlled]
- DiT tones outside {0, 4, 11..=15} reset to 3; mel2ph cut to a multiple of 8 [facade.rs:dit_inputs] [duration.rs:FRAMES_MULTIPLE]
- Config layer and width counts are bounded (e.g. layers 1..=64) [dit.rs:DitConfig] [aligner.rs:MAX_LAYERS]
- Duration LM accepts at most 6000 phones [duration.rs:MAX_TOKENS]
- `synthesize` output equals the concatenated `synthesize_controlled` stream bit for bit [facade.rs:synthesize_controlled]

## Side Effects
- Reads `model.safetensors` per submodel dir (`aligner_lm`, `diffusion_transformer`, `duration_lm`, `wavvae`, `g2p`) [facade.rs:open_with_backend] [aligner.rs:Aligner]
- Reads optional root `config.json` [facade.rs:check_root_config]
- Optional device dispatch through a caller-supplied `ComputeBackend`; uploaded weights cached in a RefCell map (not Send) [execution.rs:Execution]
- RNG is caller-supplied; `SeededRng` is SplitMix64, not torch's stream [rng.rs:SeededRng]
- No env vars, threads, or logging

## Verification
- Run: `cargo test -p turbospark-audio tts::megatts3`
