# turbospark-cli Blueprint

## System Overview
Package `turbospark-cli` builds the user-facing binaries: `turbospark` (front door), `turbospark-check` (generation), and peer tools for models, image, music, clone, speak, steer and ocr. Decisions live in `catalog`, `invocation` and `runtime`; this crate parses, prints and chooses exit codes.

## Subdirectories & System Boundaries
- src: [./src/EGM.md](./src/EGM.md) - check binary, chat, daemon, agent launchers, and the sub-modules below.
- src/bin: [./src/bin/EGM.md](./src/bin/EGM.md) - binary entry points.
- src/bin/model_cmd: [./src/bin/model_cmd/EGM.md](./src/bin/model_cmd/EGM.md) - turbospark-model subcommands.
- src/bin/steer_cmd: [./src/bin/steer_cmd/EGM.md](./src/bin/steer_cmd/EGM.md) - turbospark-steer verbs.
- src/front: [./src/front/EGM.md](./src/front/EGM.md) - front-door model resolution and status.
- src/generate: [./src/generate/EGM.md](./src/generate/EGM.md) - generation session and vision.
- src/ui: [./src/ui/EGM.md](./src/ui/EGM.md) - terminal presentation.

## Global Invariants
- Output stays ASCII [ui/mod.rs module doc]; install and delete logic exists only in `turbospark-model` [front/models.rs module doc].

## Global Side Effects
- Model directory writes, Hugging Face downloads, daemon process and pid files, GPU model loads.

## System Verification
- Run: `cargo test -p turbospark-cli`
