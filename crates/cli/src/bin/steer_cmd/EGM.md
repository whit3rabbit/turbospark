# bin/steer_cmd

## Overview
Verbs behind the `turbospark-steer` binary: extract a steering direction from a model's own residual activations, and check its effect. Packs ship prompts and marker words, never a direction.

## Interface
- [extract.rs:ExtractOptions], [extract.rs:run]: capture positive and negative corpora and write a vector file.
- [check.rs:CheckOptions], [check.rs:run]: before/after generation readout counting pack markers.
- [packs.rs:TraitPack], [packs.rs:PackSide], [packs.rs:PackEval], [packs.rs:all], [packs.rs:by_id]: embedded trait packs.
- [walk.rs:ResolvedModel], [walk.rs:resolve_model], [walk.rs:render_prompt], [walk.rs:capture_side]: model resolution and the corpus walk.
- [capture_io.rs:read_capture], [capture_io.rs:read_positions]: read capture header and sidecar files.
- [mod.rs] re-exports the five modules.

## Invariants
- A model whose family does not dispatch steering is refused [walk.rs:resolve_model].
- A pack file name must equal its declared id [packs.rs:TraitPack::parse].
- One model open serves a whole corpus; `TURBOSPARK_RESID_CAPTURE` is set around each side and removed again [walk.rs:CAPTURE_ENV].
- Capture KV window is capped [walk.rs:MAX_CAPTURE_CONTEXT].
- Vector output is written through `.partial` plus rename so a failure never truncates an existing file [extract.rs:write_atomically].

## Side Effects
- Writes vector files and temp capture directories; deletes stale capture dirs [extract.rs:sweep_stale_capture_dirs], [extract.rs:run].
- Sets and unsets a process env var [walk.rs:CAPTURE_ENV].
- Loads a model onto the GPU and runs generation [walk.rs:capture_side], [check.rs:generate_text].

## Verification
- Run: `cargo test -p turbospark-cli`
