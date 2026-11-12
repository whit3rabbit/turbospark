# scripts

## Overview
Python dev tooling for the audio crate: parity-fixture generators for ASR/STT models against pinned reference stacks, benchmark drivers, and NeMo Frozen E trace tools. Nothing here ships or runs in `cargo test`.

## Interface
- Fixture generators `generate_*_fixture(s).py` for Canary, FireRedASR2, Fun-ASR-Nano, GLM-ASR, Granite Speech (and NAR, features), Higgs Audio 3, LASR CTC, Mega-ASR, MMS, MOSS Music/Transcribe, Multitalker, Parakeet, Phonon, Qwen2-Audio, Qwen3-ASR (encoder, features, forced aligner), SenseVoice, VibeVoice-ASR, Voxtral (and Realtime), Wav2Vec, and coupled diarizer goldens.
- Benchmarks: `benchmark_megatts3.py`, `benchmark_moonshine.py`, `benchmark_music3.py`, `benchmark_qwen3_asr.py`, `ab_music3_flow.py`, `spectral_distance.py`.
- NeMo tooling: `convert_multitalker_nemo.py`, `nemo_frozen_e_baseline.py`, `nemo_multitalker_trace.py`, `diff_multitalker_trace.py`; environment under `./nemo_218d0b4/` (pinned by its `pyproject.toml` and `uv.lock`, has its own README.md).

## Invariants
- Each generator's docstring names the pinned reference it is normative against; regenerate fixtures only with that pin [generate_canary_fixture.py module docstring].
- Benchmarks run one fresh process per sample and interleave variants [benchmark_music3.py, ab_music3_flow.py docstrings].

## Side Effects
- Fixture generators write golden files into the crate's fixture directories and load local checkpoints.
- `benchmark_music3.py` shells out via `subprocess.run` to `pmset -g batt`, git, `sysctl -n machdep.cpu.brand_string`, and `/usr/bin/time -l <binary>` per sample. It does not run cargo: the `music3_bench` binary (`turbospark-runtime --example music3_bench`) must be prebuilt by hand, as its module docstring says [benchmark_music3.py:run_sample].

## Verification
- Run: `python3 -I crates/audio/scripts/spectral_distance.py REFERENCE.wav CANDIDATE.wav`. The script has no argparse, so `--help` is read as a path and fails.
