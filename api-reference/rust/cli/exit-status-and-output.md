---
description: "Exit codes and stdout/stderr routing for each turbospark-cli binary, including the cases where a failed run still exits 0."
icon: circle-check
---

# Exit status and output

Read this page before you script against the binaries. Exit codes and output streams differ by binary. In one case, a failed run exits `0`.

## Exit codes

| Binary | `0` | `1` | `2` |
| --- | --- | --- | --- |
| `turbospark-check` | A validated run, `--help`, or `--version`. Also any run that failed after parsing. | Never returned. | Any parse failure. |
| `turbospark-model` | Success, `--help`, or `--version`. | A run that failed: network error, probe refusal, store error, or declined confirmation. | Usage error: unknown command or option, missing or extra argument, or an option the command does not read. |
| `turbospark-image` | Success, or usage text printed for `--help` or no arguments. | Every error, including bad flags. | Never returned. |
| `turbospark-music` | Success. | A run that failed. | Usage error, including `--help`. |
| `turbospark-clone` | Success. | A run that failed. | Usage error, including `--help`. |
| `turbospark` | Success, or the child binary's `0`. | Daemon or agent error, or the child could not be started. Also a child killed by a signal. | Unknown command, or `humanize` with fewer than two arguments (its usage text still prints). |

`turbospark` passes the child binary's exit code through unchanged. If the child exits because of a signal, the front end returns `1`.

### turbospark-check runtime failures exit 0

`turbospark-check` decides its exit status from parsing alone. Generation runs after that decision and cannot change it. When the model cannot open, the tokenizer fails to load, or a request is refused at run time, the binary prints a `note:` line on stderr and exits `0`.

This is the most important fact on this page for scripts. `turbospark-check` has no exit code for a failed generation. Check stderr for these lines, and check stdout for the generation header:

- `note: not attempting real generation: ...`
- `note: ...` (for example, an image run that fails validation)
- `generation failed: ...`

The generation header is a useful marker for `--prompt`, `--messages-file`, and `--humanize-file`. Each of these prints its header on stdout just before generating, so a run that fails before that point has no header. The header text is:

- `generating (real forward pass):` for `--prompt`
- `generating (real forward pass, chat template applied):` for `--messages-file`
- `humanizing (real forward pass):` for `--humanize-file`

`--chat` prints no header. Its banner goes to stderr, so a failed chat start shows up only as a `note:` line.

These strings come from the source. They are not a versioned contract, so pin the version if a script relies on them.

## Stream routing

### turbospark-check

| Stream | What goes there |
| --- | --- |
| stdout | `--help` and `--version` text. On every validated run, the `resolved invocation:` block. Then the generation header and the generated text. Chat replies. The humanized rewrite. |
| stderr | Parse failure diagnostics, followed by the usage text. The `note:` and `generation failed:` lines. The statistics footer, unless `--quiet` is set. Chat prompts and slash-command output. Reasoning text. Humanize audit notes. Optional phase timing. |

Two details matter for pipes:

- The `resolved invocation:` block is printed to stdout before any generation. It is printed on every validated run, and no flag in this source suppresses it. A run that pipes stdout receives the block and the header before the generated text.
- `--humanize-file` writes its rewrite to stdout, but the resolved block and the `humanizing` header also go to stdout. Strip those lines before you use the output as text.

Parse failures print a Debug rendering of the failure, such as `UnknownOption { token: "--foo" }`, followed by the usage text. The wording is not stable. Match on the exit code and never on the message.

### turbospark-model

| Stream | What goes there |
| --- | --- |
| stdout | Command results: list tables, `info`, `path`, `probe` reports, `recommend`, the `installed` summary after `pull`, and `removed` after `rm`. |
| stderr | Errors (prefixed `error:` for run failures). Progress bars and `[pull]` stage lines from `indicatif`, which uses stderr by default. The delete prompt for `rm`. Usage text on usage errors. |

Use stdout for scripts. `turbospark-model path ALIAS` prints only the directory, so `$(turbospark-model path ALIAS)` works. If the alias is not installed, the command fails with exit `1` and prints nothing on stdout.

`probe` prints its report on stdout. A repo the engine refuses exits `1` after the report, so `turbospark-model probe REPO && turbospark-model pull --repo REPO --alias NAME` only pulls when the probe passes.

`rm` asks you to type the alias on stderr and reads the answer from stdin. A mismatch exits `1` and deletes nothing. A closed or empty stdin also exits `1`. Pass `--yes` (or `-y`) in scripts.

### turbospark-image

| Stream | What goes there |
| --- | --- |
| stdout | On a successful `generate`, the output path. On a successful `pack`, a `packed N files` line. On `--help` or no arguments, the usage text. |
| stderr | Errors, each prefixed `error:`. A bad flag prints only that line. A bad subcommand prints the error followed by the usage text. |

A successful `generate` prints only the output path, so `OUT=$(turbospark-image generate ...)` captures it.

### turbospark-music and turbospark-clone

| Stream | What goes there |
| --- | --- |
| stdout | The success summary: a sample count and the output path for music, and a duration, sample count, and the output path for clone. |
| stderr | Errors. For usage errors, the message followed by the usage text. The `--timings` output for music. |

### turbospark front end

The front end does not add output of its own for routed commands. Each child writes to the same stdout and stderr as if it had been run directly. The front end writes its own lines for `status`, `start`, `stop`, and `restart`. Those lines go to stdout. Their errors go to stderr with an `error:` prefix.

## Known quirks

These are behaviors the source produces. Each one can surprise a script.

- **`turbospark-clone --help` and `turbospark-music --help` exit `2`.** The usage text goes to stderr, because the parser treats `--help` as a usage error. `turbospark-model --help` and `turbospark-image --help` exit `0` and print to stdout.
- **`turbospark humanize --help` exits `2`.** The front end checks the argument count before it checks for `--help`. The usage text still prints to stdout.
- **`turbospark-check --help` wins only when reached first.** The parser scans left to right. An unknown option that appears before `--help` produces a parse failure instead.
- **`turbospark-image` exits `1` for usage errors.** It does not use `2`. A script that treats `2` as "bad invocation" will miss these.
- **`turbospark image pack` does not work through the front end.** The front end adds `generate` in front of every `image` argument that is not already `generate`, so `pack` reaches `turbospark-image generate pack` and fails. Run `turbospark-image pack` directly.
- **`turbospark-music` and `turbospark-clone` overwrite their output.** They do not check for an existing file. `turbospark-image generate` refuses to overwrite its `--output` and exits `1`.
- **`turbospark start` on a running server exits `0`.** It prints `turbospark server is already running` and does not restart the server. Use `restart` to change the model or port.
- **`turbospark status` exits `0` when the server is stopped.** Check the `Status:` line instead of the exit code.

## Limits

- The source defines no machine-readable output mode. None of the binaries prints JSON.
- No flag turns off the `resolved invocation:` block on `turbospark-check`.
- A run that fails inside `turbospark-check` after parsing has no distinct exit code. Script around the stderr and stdout markers above. Changing this would be a source change, not a documentation fix.

For the flags that trigger each of these paths, see [Commands overview](commands-overview.md). For runnable command lines, see [Examples](examples.md).
