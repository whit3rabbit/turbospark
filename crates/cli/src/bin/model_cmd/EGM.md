# bin/model_cmd

## Overview
Subcommand implementations for the `turbospark-model` binary. Each module prints or drives something that the `catalog` crate decides. This layer picks column widths, prompts, progress bars and exit errors. It does not own verdict or install logic.

## Interface
- [mod.rs:list_speech], [mod.rs:list], [mod.rs:info], [mod.rs:probe], [mod.rs:path], [mod.rs:remove], [mod.rs:pull], [mod.rs:pull_image], [mod.rs:pull_vision]: catalog listing, probe, and install entry points.
- [mod.rs:INTERRUPTED] and [mod.rs:install_sigint_handler]: Ctrl-C flag shared by audio pull, drama3 pull and move.
- [audio.rs:list_audio], [audio.rs:pull_audio], [audio.rs:path_audio], [audio.rs:remove_audio]: audio model install management.
- [auth.rs:auth]: Hugging Face token inspect, set (stdin), and clear.
- [drama3.rs:pull_drama3]: Fish Audio Drama 3 install into a caller-chosen directory.
- [library.rs:library]: library list/add/remove/set-default.
- [transfer.rs:move_model]: move an install between libraries.
- [views.rs:ls], [views.rs:search], [views.rs:show], [views.rs:doctor], [views.rs:home], [views.rs:resolve], [views.rs:status_word]: user-facing views.
- [render.rs:entry], [render.rs:speech_entry], [render.rs:report], [render.rs:recommendations], [render.rs:human_bytes]: text rendering.
- [progress.rs:spinner], [progress.rs:byte_progress_bar], [progress.rs:count_progress_bar]: indicatif helpers.
- [destination.rs:Destination]: `--out`, library, or This Mac target for a pull.
- [image_source.rs:materialize]: stages a pinned image repo by calling `catalog::materialize_image_source`.

## Invariants
- Decisions live in `catalog`; this module only renders them [mod.rs module doc].
- Image staging refuses to overwrite an existing directory [image_source.rs:materialize].
- Moves use the absolute `/bin/df` so PATH cannot substitute a program [transfer.rs:DF].
- Move kinds are limited to text, vision, audio, image [transfer.rs:KINDS].
- A user-chosen `--alias` (`--repo` pulls, image and vision pulls) is ASCII-checked before network work [mod.rs:check_alias]; a curated positional alias is not. Destination is settled before any request [destination.rs module doc].

## Side Effects
- Installs and removals write and delete model directories (`std::fs::remove_dir_all`, `create_dir_all`) [mod.rs:remove], [mod.rs:pull_image], [audio.rs:remove_audio].
- Network downloads from Hugging Face through `catalog::Client` [mod.rs:pull], [drama3.rs:pull_drama3].
- Writes or clears the stored HF token [auth.rs:auth].
- Spawns `/bin/df` [transfer.rs:free_bytes].
- Installs a process-wide SIGINT handler (`libc::signal`) [mod.rs:install_sigint_handler].
- Reads token from stdin or terminal [auth.rs:read_token].

## Verification
- Run: `cargo test -p turbospark-cli`
