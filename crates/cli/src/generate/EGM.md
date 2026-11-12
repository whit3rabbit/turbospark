# generate

## Overview
Generation path for `turbospark-check`: opens a model session, renders prompts through the chat template, streams turns, and attaches vision inputs.

## Interface
- [mod.rs:try_generate], [mod.rs:Outcome]: entry point and Completed/Failed result.
- [mod.rs:render_prompt], [mod.rs:clamp_max_new], [mod.rs:stream_turn]: crate-internal prompt and streaming helpers.
- [session.rs:Session], [session.rs:open_session]: model, tokenizer, sampling config, resolved context.
- [format.rs:print_phases], [format.rs:print_footer], [format.rs:parse_messages_file], [format.rs:map_reasoning_effort], [format.rs:parse_role], [format.rs:role_name].
- [vision.rs:PreparedImage], [vision.rs:prepare_images], [vision.rs:attach] and per-family `attach_*` (minicpm, gemma4, clip, siglip, deepseek_ocr, unlimited_ocr2).

## Invariants
- Image-generation installs are rejected by text generate [session.rs:reject_image_install].
- Downstream code uses the resolved context window, never `request.max_context` [session.rs:Session] `max_context` field.
- Rate control is resolved once at open [session.rs:Session] `rate` field.
- Within the vision pipeline, images are decoded and preprocessed before the tower runs [vision.rs module doc]. In `--messages-file` mode the session opens first [mod.rs:run_messages_file], then [vision.rs:prepare_images] runs per turn [mod.rs:run_one_turn].
- `Outcome::Failed` yields a nonzero exit, but a failed turn inside `--chat` does not [mod.rs:Outcome].

## Side Effects
- Loads the model onto the GPU and allocates KV [session.rs:open_session].
- Reads `TURBOSPARK_PHASES`, `TURBOSPARK_DISPATCH_PROFILE`, `TURBOSPARK_PREFILL_CHUNK` [format.rs:print_phases], [mod.rs:resolve_chunk_tokens]. Also `TURBOSPARK_DEBUG_PROMPT_IDS` (prompt ids to stderr) [mod.rs:run_one_turn] and names `TURBOSPARK_EXPERT_DISK_IO` in a footer hint only; this crate never reads it [format.rs:print_phases].
- Streams to stdout and stderr; reads message and image files [format.rs:parse_messages_file], [vision.rs:prepare_images].

## Verification
- Run: `cargo test -p turbospark-cli`
