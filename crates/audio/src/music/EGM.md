# music

## Overview
Music generation models: MiniMax Music 3 and ACE-Step 1.5 Turbo
(text2music on Metal through the runtime session, catalog and CLI).

## Interface
- `pub mod minimax_music3` [mod.rs]: see [minimax_music3](./minimax_music3/EGM.md).
- `pub mod ace_step15` [mod.rs]: see [ace_step15](./ace_step15/README.md).

## Invariants
- Only declares the modules; no logic [mod.rs].

## Side Effects
- None [mod.rs].

## Verification
- Run: `cargo test -p turbospark-audio music::`
