# turbospark-repack

## Overview
Model intake crate: converts HF safetensors and GGUF checkpoints into `.gturbo` installs. Sources live in `src/` (see its contract) and bring-up binaries in `examples/`.

## Subdirectories & Boundaries
- src: [./src/EGM.md](./src/EGM.md) - crate root, per-family config parsers, resident writer, checkpoint walks, synthetic fixtures
- examples: [./examples/EGM.md](./examples/EGM.md) - Drama 3 intake, Flint local intake and rebalance, MiniCPM tower fixture examples
