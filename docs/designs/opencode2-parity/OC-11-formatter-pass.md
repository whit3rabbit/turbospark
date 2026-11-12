# OC-11: Formatter pass over agent edits

Finding: Section 8, "Formatters run over agent edits - Missing in
Swift" in [OPENCODE2_COMPARISON.md](../../OPENCODE2_COMPARISON.md).
Parity targets: OpenCode's
[formatter service](https://github.com/anomalyco/opencode/blob/dev/packages/core/src/formatter.ts)
and
[builtins table](https://github.com/anomalyco/opencode/blob/dev/packages/core/src/formatter/builtins.ts).

* Author: whit3rabbit
* Status: Draft
* Date: 2026-11-11

--------------------------------------------------------------------------------

## Overview

Run the user's own formatters over files the agent writes, when the
formatter's executable exists and the project enables the pass. A
style pass after each agent write when a formatter is available is small and high-leverage for coding
use; OpenCode runs configured formatters after every agent write, and
TurboSpark leaves style entirely to model discretion.

## Dependencies

* Registers on OC-10's post-edit hook in `AppModel+Tools.swift`
  rather than adding a second one. Formatting runs first, then
  OC-10 diagnostics.
* Reuses `LoginShellPath` (PATH capture, relative entries dropped),
  `ProcessExecutor` (timeout, capped output) and `PathContainment`
  (root check). `LoginShellPath` lives under `Tools/MCP/`; moving
  it to a shared folder is part of this change.

## The Problem

A search found no formatter pipeline. The only formatter-related code
is `ShellOutputFormatting`
([ShellOutputFormatting.swift](../../../swift/TurboSparkApp/Sources/TurboSparkApp/Tools/Terminal/ShellOutputFormatting.swift),
output display, reused here to strip ANSI from the error tail) and
`TerminalCommandClassifier`
([TerminalCommandClassifier.swift](../../../swift/TurboSparkApp/Sources/TurboSparkApp/Tools/Terminal/TerminalCommandClassifier.swift)),
which already lists prettier and rustfmt as auto-approvable build
commands when the model runs them itself. No code runs a formatter on
the app's behalf. The model may run a formatter tool if it thinks to; nothing guarantees
the pass. OpenCode resolves configured formatters per file extension,
checks executable availability, and formats files after agent writes.

## Goals

* A formatter registry matched by file extension. V1 ships one
  built-in row: swift-format (`.swift`, run as
  `swift-format format --in-place <temp copy>`). Other formatters
  (gofmt, clang-format, prettier, ruff, rustfmt) are user rows
  with an explicit executable and argument vector; the doc's
  examples show them but the app ships none. (author to confirm the
  command line)
* Availability check: the executable is resolved once to an
  absolute path using the PATH captured by `LoginShellPath` (one
  login-shell probe, cached, relative entries dropped). The
  formatter itself runs as an argument vector with no shell.
  Executables inside the project tree are refused in v1. A
  missing executable is a silent skip.
* Per-project toggle with a default off; enabling it is the user's
  explicit statement that the formatter may rewrite the agent's output
  on disk.
* The pass runs after a successful FileWrite, FileEdit, multiedit or
  apply_patch call. No other tool triggers it in v1. It runs through
  the existing hardened process path: argument vectors, no shell, a
  10 second timeout enforced by `ProcessExecutor`, process killed on
  expiry, error tail capped (for example 2 KB), cancelled when the
  turn is stopped. (author to confirm values)
* On failure the follow-up output gains a note with the formatter's
  error tail (capped, ANSI stripped via `ShellOutputFormatting`).
  Success adds no note.
* After a successful format the edit tool's result carries the
  post-format content, `FileSnapshotStore.recordSnapshot` is
  refreshed with it so the next edit is not rejected as stale, and
  the undo backup keeps the agent's pre-format write. (author to
  confirm)
* The pass formats only the path returned by a successful edit
  tool call in that turn, and only if
  `PathContainment.resolvedIfContained` accepts it under the
  project root (symlinks resolved). Anything else is skipped.

## Non-Goals

* No bundled formatter binaries.
* No v1 support for formatters that need project-wide context beyond
  the target file (no daemon-based servers, no whole-project passes).
* No auto-enablement by file extension alone: the project toggle is
  off until the user turns it on.
* No formatting for tools that only read files (no reformatting of
  files the agent only read).

## Alternatives Considered

(author to confirm all four bullets)

* **Leave style to the model** (status quo): rejected, because
  nothing guarantees the pass and style drift lands in review.
* **Share OC-10's post-edit seam versus a bespoke hook**: share.
  Both run after the same edit-tool success, so one seam avoids
  two write paths into `AppModel+Tools.swift`. OC-11 runs after
  OC-10's seam fires, and before diagnostics are collected, so
  diagnostics see the formatted file.
* **swift-format only versus user-configured rows**: ship
  swift-format plus user rows. A user row is the only way to
  cover prettier, ruff or rustfmt without bundling them.
* **Format at turn end versus per edit**: per edit. Turn-end
  batching adds a dirty-set and a coalescing rule for little gain
  on small files.

## Open Questions

* None open after review. (Registry contents and triggering tools are
  now settled in Goals.) (author to confirm)

## Critical User Journeys

### CUJ 1: style pass after edit

* **Given**: the project enables the pass; swift-format is installed;
  the agent writes a Swift file with bad style.
* **When**: the edit completes.
* **Then**: the file is formatted and the transcript shows the
  formatted content.

### CUJ 2: formatter missing

* **Given**: the pass enabled, but the formatter is not installed.
* **When**: an edit completes.
* **Then**: no pass runs, no error reaches the user, and the tool
  output carries no formatting note.

### CUJ 3: formatter fails

* **Given**: the pass enabled and the formatter errors on the file.
* **When**: the edit completes.
* **Then**: the file keeps the agent's content, the edit tool's
  follow-up output gains the error tail, and the turn continues.

## Technical Plan (Architecture)

```text
Edit tool success
  -> project toggle on? (no: stop)
  -> path contained in project root? (no: stop)
  -> FormatterRegistry.match(extension)
  -> executable resolved on login-shell PATH? (no: silent skip)
  -> FormatterRunner: temp copy, run, timeout
       exit 0 and file unchanged: replace original
       failure or changed file: keep original, add error tail
```

## Non-Functional Requirements

* **Performance**: the pass adds at most the formatter's own runtime;
  the 10 second timeout (kill on expiry) caps pathological cases.
  (author to confirm values)
* **Safety**: the pass is app-initiated, not a model tool call,
  and sits outside `AppToolPermissionEngine`. It runs only after
  an edit tool has already passed that ladder, never in read-only
  mode, and never from a command string the model supplies. The
  per-project toggle (default off) is the user's trust grant.
  User-configured rows live in the app archive, not in the
  repository, so a cloned repo cannot add a command. Formatter
  config files in the repo are trusted input once the toggle is
  on; this is a recorded residual risk. Argument vectors, no
  shell, absolute resolved executable, and a time bound apply.
  The first pass runs the user's shell rc to capture PATH, as the
  MCP first spawn already does. (author to confirm)
* **Localization**: project toggle copy and the note templates in 21
  languages in the same change.
* **Accessibility**: no new controls beyond the project toggle row;
  notes announce via the existing tool-output announcements.

## Risks and Mitigations

(author to confirm rows and the 10 s value)

| Risk | Mitigation |
| --- | --- |
| The model's next edit anchors on pre-format content | The tool result carries post-format content and the edit tool's recorded hash is refreshed (see Goals). |
| Formatter config in the repo executes code (for example a plugin) | Config is trusted input only because the user enabled the pass for this project. v1 ships swift-format, whose config is data. User rows are the user's own risk; the toggle copy says so. Default off. |
| Write races a user save | Format a temp copy, replace only on exit 0, skip with a failure note if the file hash changed. |
| Formatter hangs or floods output | `ProcessExecutor` timeout (10 s), kill on expiry, capped error tail, cancel on turn stop. |
| Path escapes the project root | `PathContainment` with symlink resolution before any run. |
| A planted binary in the repo is run | Executable resolved once to an absolute path from the login-shell PATH; relative PATH entries are dropped; no in-project executables in v1. |

## Detailed Implementation Plan

### Target Files to Create/Modify

1. `swift/TurboSparkApp/Sources/TurboSparkApp/Tools/Format/FormatterRegistry.swift` (Create)
   * **Rationale**: the extension-to-command registry and availability
     check.
2. `swift/TurboSparkApp/Sources/TurboSparkApp/Tools/Format/FormatterRunner.swift` (Create)
   * **Rationale**: the hardened run, time bound, and note/error-tail
     outputs.
3. `swift/TurboSparkApp/Sources/TurboSparkApp/State/AppModel+Tools.swift` (Modify)
   * **Rationale**: the post-write hook after edit-tool success.
4. `swift/TurboSparkApp/Sources/TurboSparkApp/Generation/ProjectPermissionsSectionView.swift` (Modify)
   * **Rationale**: the per-project formatting pass toggle row.
5. `swift/TurboSparkApp/Sources/TurboSparkApp/State/AppModel+Persistence.swift` (Modify)
   * **Rationale**: persist per-project pass records with the archive.
     A missing or unreadable key decodes to off.
6. `swift/TurboSparkApp/Localization/Localizable.xcstrings` (Modify)
   * **Rationale**: all new strings, 21 languages, same change.
7. `swift/TurboSparkApp/Tests/TurboSparkAppTests/FormatterRegistryTests.swift` (Create)
   * **Rationale**: matching, availability gating, silent skip when
     missing. Mutation check: invert the availability gate and assert
     its test fails.
8. `swift/TurboSparkApp/Tests/TurboSparkAppTests/FormatterRunnerTests.swift` (Create)
   * **Rationale**: timeout kills the process, a failing formatter keeps
     the original file, a path outside the root is skipped, a changed
     file hash skips the replace.

### Verification Plan

* **Build**: `make swift-lib`, then the app scheme build.
* **Tests**: the new suite plus localization gates and
  `make design-check`.
* **Manual**: enable the pass in a project with swift-format
  available. Have the agent write a badly styled file and confirm the
  pass, the note, and the silent-skip path.

--------------------------------------------------------------------------------

*Drafted with EGM*
