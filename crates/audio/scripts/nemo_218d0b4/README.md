# NeMo Speech 218d0b4 reference environment

Dev-only Python environment for the Frozen E port (`stt/multitalker_asr` and
the coupled `vad/nemotron_diarization` mode). Nothing here ships or runs in
tests. `uv.lock` pins every resolved version, and NeMo itself is the exact
commit `218d0b443441b7d8aa1c832f47df310f186cf0a4`. Resolution is limited to macOS
arm64, where NeMo routes torch to the PyPI CPU wheel.

```sh
uv sync --project crates/audio/scripts/nemo_218d0b4 --python 3.12
```

Scripts that use it (all under `crates/audio/scripts/` unless noted):

| Script | Purpose |
|---|---|
| `convert_multitalker_nemo.py convert` | Multitalker `.nemo` to F32 safetensors in the layout `MultitalkerAsr::open` reads |
| `convert_multitalker_nemo.py convert-diar` | Nemotron-3 diarizer `.nemo` to the layout `NemotronDiarization::load` reads |
| `convert_multitalker_nemo.py check-diar` | Every HF `model.safetensors` diarizer tensor has an exact `.nemo` source |
| `nemo_frozen_e_baseline.py` | The unmodified NeMo multitalker driver on CPU with the Frozen E flags |
| `nemo_multitalker_trace.py` | The same run with per-step observers, writing the trace the Rust port is diffed against |
| `generate_coupled_diar_fixture.py`, `generate_multitalker_fixture.py` | Committed goldens from a trace |
| `diff_multitalker_trace.py` | First divergent stage between a Rust and a NeMo trace (standard library only) |
| `crates/bench/scripts/prepare_frozen_e_clips.py` | Rebuilds the frozen AMI clips and checks their sha256 |

Checkpoints, revisions and hashes are recorded in
`crates/audio/src/stt/multitalker_asr/README.md`.
