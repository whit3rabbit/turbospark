# Runtime Example Binaries

## Overview
Nineteen standalone Cargo example binaries for `turbospark-runtime`. They are real-model gates, benchmarks and bring-up tools for the audio (Fish Drama 3, MegaTTS3, Music 3, Moonshine, Qwen3-ASR, Whisper, Parakeet, Frozen E), Core ML (OCR, Parakeet, EmbeddingGemma) paths. They take model directories on the command line, need real checkpoints, and are not part of the unit test suite. Several are Metal or macOS-only.

## Interface
- `coreml_backend_probe` [coreml_backend_probe.rs]: one warm measured Core ML run (`parakeet|embed|ocr`), metric line on stderr, digest on stdout; driven by scripts/benchmark_coreml.py
- `embeddinggemma3_probe` [embeddinggemma3_probe.rs]: subcommands `ids`, `cpu`, `coreml_<cpu|gpu|ane|all>`; writes ids/embeddings JSON; driven by scripts/benchmark_embeddinggemma3.py
- `fish_drama3_say` [fish_drama3_say.rs]: text to 44.1 kHz WAV on the Metal trunk; args `<install-dir> <text> <out.wav> [expert-slots] [seed] [max-frames] [residency] [tower] [codec] [text-cap]`
- `fish_drama3_runner` [fish_drama3_runner.rs]: same through `FishDrama3Runner`; args `... [expert-slots] [seed] [residency] [runs]`; prints time to first samples and real-time factor
- `fish_drama3_parity` [fish_drama3_parity.rs]: subcommands `run`, `compare [--gate]`, `router-isolation` against an mlx-audio teacher-forced dump
- `fish_drama3_guard_sweep` [fish_drama3_guard_sweep.rs]: pause-guard recovery rates, args `<install-dir> <trapped|heldout> <off|hold:retry:retries>...`
- `fish_drama3_silence_probe` [fish_drama3_silence_probe.rs]: silence and pause-trap probe over a corpus, args `<install-dir> <corpus.txt> <out.jsonl> [seeds] [gate=on|off]`
- `fish_drama3_step_cost` [fish_drama3_step_cost.rs]: per-step phase breakdown text vs frame rows, args `<install-dir> [expert-slots] [steps] [rounds]`
- `fish_drama3_wer` [fish_drama3_wer.rs]: synthesize corpus, transcribe with CPU Parakeet, score WER; supports `--smoke`, `recipe=fish|reference`, `guard=...`
- `frozen_e_metal` [frozen_e_metal.rs]: meeting transcription gate; flags `--model --diar --clips --out [--ids] [--diar-device] [--asr-device]`
- `megatts3_bench` [megatts3_bench.rs]: MegaTTS3 clone jobs from JSON, one JSON line per request; flags `--jobs [--device metal|cpu] [--warmup]`
- `music3_bench` [music3_bench.rs]: time one Music 3 Metal request; flags `--caption --lyrics --output --duration --steps --seed --precision --warmup --flow-only --flow-solver --flow-profile --flow-uncond-*`
- `music3_generate` [music3_generate.rs]: minimal library use of the Music 3 runner, writes 16-bit stereo WAV
- `music3_backend_probe` [music3_backend_probe.rs]: dumps AR codes, hiddens and waveforms for the MLX probe; `--hiddens-file`, `--noise-file`
- `moonshine_backend_probe` [moonshine_backend_probe.rs]: portable vs Metal Moonshine; `--open-only`, `--warmup`
- `qwen3_asr_backend_probe` [qwen3_asr_backend_probe.rs]: CPU vs Metal Qwen3-ASR (backend is an explicit arg); `--warmup`
- `whisper_transcribe` [whisper_transcribe.rs]: whisper CPU reference gate, args `<model_dir> <wav> [language]`; compiles empty off macOS
- `parakeet_coreml_transcribe` [parakeet_coreml_transcribe.rs]: Core ML Parakeet transcription, args `<model_dir> <wav>`; macOS-only
- `ocr_coreml` [ocr_coreml.rs]: Core ML Unlimited OCR, args `<model_dir> <page.png>...`; one page uses the `single` prompt, several `multi`

## Invariants
- `fish_drama3_guard_sweep`, `_silence_probe`, `_parity` and `_wer` are gated `#![cfg(target_os = "macos")]`; `_say`, `_runner` and `_step_cost` carry no gate. All need an install dir holding config.json, tokenizer.json and codec.safetensors [fish_drama3_say.rs:main]
- Probe samples run one discarded warmup then one measured pass, so timing excludes pipeline compile [coreml_backend_probe.rs:main]
- Every embedding arm consumes identical token ids produced once by the `ids` subcommand [embeddinggemma3_probe.rs:main]
- The pause guard is off in the silence probe so draws are the raw model's [fish_drama3_silence_probe.rs:main]
- `music3_bench --flow-only` feeds synthetic hiddens: a timing aid with no quality meaning [music3_bench.rs:main]
- The Qwen3-ASR Metal decoder is chosen by argument, never by env switch [qwen3_asr_backend_probe.rs:main]
- The Music 3 runner is `!Send`: opened and used on one thread [music3_generate.rs:main]
- Parity `compare --gate` is the pass/fail mode; other modes report only [fish_drama3_parity.rs:main]

## Side Effects
- Writes WAVs, JSON or JSONL outputs to caller-given paths [fish_drama3_say.rs:main] [music3_bench.rs:main] [fish_drama3_silence_probe.rs:main]
- Creates and removes dump directories and files [fish_drama3_parity.rs:main]
- Sets `TURBOSPARK_ROUTER_TRACE` and `TURBOSPARK_ROUTER_HIST` in-process [fish_drama3_parity.rs:main]
- Reads env: `TURBOSPARK_DRAMA3_SERIAL` [fish_drama3_say.rs:main], `OCR_MAX_NEW_TOKENS` [ocr_coreml.rs:main], `TURBOSPARK_MOONSHINE_DEVICE` [moonshine_backend_probe.rs:main], `TURBOSPARK_MUSIC3_AR_ONLY`, `TURBOSPARK_MUSIC3_TRACE`, `TURBOSPARK_MUSIC3_TRACE_MAX_CALLS` [music3_backend_probe.rs:main], `TURBOSPARK_PARAKEET_COREML_ENCODER_UNITS` [parakeet_coreml_transcribe.rs:main]
- Uses Metal devices, Core ML and large resident model memory; no network access [fish_drama3_runner.rs:main]

## Verification
- Run: `cargo build -p turbospark-runtime --examples`
- Examples need real checkpoints, so a build is the only check without models.
