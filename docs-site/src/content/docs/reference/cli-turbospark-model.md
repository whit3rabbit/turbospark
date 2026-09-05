---
title: turbospark-model
description: Every command and flag of the turbospark-model binary, captured verbatim from --help.
---

<!-- generated: cli-help lane, signal: crates/cli/src/bin/model.rs -->

# turbospark-model

`turbospark-model` finds, inspects and installs models. It is sub-command
driven: `list`, `info`, `probe`, `recommend`, `pull`, `path` and `rm`.

Captured verbatim from `turbospark-model --help` (exit 0):

```text
turbospark-model -- find, inspect and install models

USAGE:
    turbospark-model <COMMAND> [OPTIONS]

COMMANDS:
    list [--filter TEXT]        curated models, marking the ones installed
    info <ALIAS>                one model in full, with its gate targets
    probe <REPO>[@REV]          what this engine makes of a Hugging Face repo,
                                reading headers only: no download
    recommend                   what this machine should run, ranked
    pull <ALIAS>                install a curated model
    pull --repo <REPO>[@REV]    install any repo the probe accepts
    path <ALIAS>                print an install directory, for scripts
    rm <ALIAS>                  delete an install

OPTIONS:
    --out <DIR>                 install here instead of the default store
    --alias <NAME>              name a --repo pull (required for one)
    --file <NAME.gguf>          pick one file where a repo offers several
    --sidecar-repo <REPO>[@REV] take tokenizer files from another repo. A GGUF
                                carries llama.cpp's tokenizer, not an HF
                                tokenizer.json, so a GGUF pull needs this
    --reuse-trunk-from <ALIAS>  for a row naming an mtp source: read an
                                already-installed alias's resident trunk back
                                off disk instead of re-streaming it, so only
                                the head crosses the network. Refused unless
                                that install's recorded repo and revision
                                match this row's exactly
    --filter <TEXT>             substring match for `list`
    --context <N>               window to fit against for `recommend` (4096)
    --budget <BYTES>            override the memory probe for `recommend`
    --load-guard <TIER|BYTES>   how much of the machine a session may commit:
                                off, relaxed (default), balanced, strict, or a
                                size that caps what the engine allocates. MUST
                                match what the session will open with.
    --discover [N]              also rank the N most-downloaded GGUF repos on
                                Hugging Face, filtered through the probe
    --probe                     read every curated row's header too, which is
                                what turns `recommend`'s unknowns into
                                arithmetic. Slower: one header per row
    --force                     install past a probe refusal
    --yes                       do not prompt before deleting
    --help                      print this text
    --version                   print the version

ENVIRONMENT:
    TURBOSPARK_HOME             the store root (default ~/.turbospark)
    HF_TOKEN                    for gated repositories

NOTE: an install streams multi-GB weights and CANNOT RESUME. A failure
restarts the walk from the beginning.
```

## Commands

| Command | Summary (verbatim from the COMMANDS block) |
|---|---|
| `list` | curated models, marking the ones installed |
| `info <ALIAS>` | one model in full, with its gate targets |
| `probe <REPO>[@REV]` | what this engine makes of a Hugging Face repo, reading headers only: no download |
| `recommend` | what this machine should run, ranked |
| `pull <ALIAS>` | install a curated model |
| `pull --repo <REPO>[@REV]` | install any repo the probe accepts |
| `path <ALIAS>` | print an install directory, for scripts |
| `rm <ALIAS>` | delete an install |

## Help behavior

There is no per-sub-command help text. Verified on 2026-09-05: every one of
the seven sub-commands accepts `--help`, prints this same top-level text, and
exits 0. The options block is shared across commands; each option's owning
command is named in its own description (`--filter` for `list`, `--context`
and `--budget` for `recommend`, and so on).

## Provenance

- Captured from the prebuilt release binary at
  `target/release/turbospark-model` on 2026-09-05; the binary was newer than
  `crates/cli/src/bin/model.rs` at capture time.
- `--version` prints `turbospark 0.1.0`.
- The catalog, the probe and the store layout are documented in `docs/MODELS.md`;
  `--reuse-trunk-from` is described further in `docs/MTP.md` and
  `docs/MTP_SPECULATIVE.md`. `docs/CLI.md` does not currently carry this flag
  (drift finding, see the [CLI Reference overview](./cli)).
