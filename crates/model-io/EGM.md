# turbospark-model-io Blueprint

## System Overview
Checkpoint manifests, architecture baselines, packed-expert and resident index layout, integrity receipts and memory sizing policy. See [AGENTS.md](./AGENTS.md) and [README.md](./README.md) for the human-maintained guidance.

## Subdirectories & System Boundaries
- src: [./src/EGM.md](./src/EGM.md) - all library code, with children `arch_baselines`, `arch_config`, `manifest`

## Global Invariants
- Library name is `turbospark_model_io`; depends on `turbospark-core` (imported as `foundation`), `serde`, `serde_json`, `sha2`, `memmap2`, `libc` [Cargo.toml:dependencies]

## Global Side Effects
- Disk reads and memory mapping of model installs; see [./src/EGM.md](./src/EGM.md)

## System Verification
- Run: `cargo test -p turbospark-model-io`
