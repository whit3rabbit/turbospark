# catalog probe

## Overview
The header-only detection layer. Decides what the engine would make of an
arbitrary Hugging Face repository by reading a file list and headers (KB, not
GB). Gates run cheapest first: file list, architecture, quantization, expert
granularity (reported, gates nothing).

## Interface
- `probe(client, repo, want_file, sidecar_repo) -> Result<ProbeReport, String>` [src/probe/mod.rs:probe].
- `evaluate_gguf` for the GGUF half [src/probe/gguf.rs:evaluate_gguf].
- `evaluate_config` for the safetensors half [src/probe/safetensors.rs:evaluate_config].
- `Verdict` (with `is_runnable`), `TypeShare`, `ProbeReport` (with `refuse`, `slot_cache_bytes`) [src/probe/types.rs:Verdict, src/probe/types.rs:TypeShare, src/probe/types.rs:ProbeReport].
- `SLOT_COUNTS` (8, 16, 32) and `KNOWN_SIDECARS` (7 names) [src/probe/types.rs:SLOT_COUNTS, src/probe/types.rs:KNOWN_SIDECARS].

## Invariants
- Sidecars are discovered from the repository file list, not guessed [src/probe/mod.rs: module docs gate 1].
- F32/F16/BF16 are not checked against `model_io::EXECUTABLE_GGUF_TYPES`, since repack transcodes them [src/probe/mod.rs: module docs "Two traps"].
- A ggml type that cannot be sized is reported as unsized, never as zero bytes [src/probe/mod.rs: module docs "Two traps"].
- Expert granularity never refuses a repository [src/probe/mod.rs: module docs gate 4].

## Side Effects
- Network: file-list GET and ranged header reads through `Client` [src/probe/mod.rs:probe].
- No disk writes, no model download.

## Verification
- Run: `cargo test -p turbospark-catalog`
