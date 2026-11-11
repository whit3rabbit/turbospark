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
guaranteed style pass per write is small and high-leverage for coding
use; OpenCode runs configured formatters after every agent write, and
TurboSpark leaves style entirely to model discretion.

## The Problem

No formatter pipeline exists. The only "formatter" matches are output
display helpers
([ShellOutputFormatting.swift](swift/TurboSparkApp/Sources/TurboSparkApp/Tools/Terminal/ShellOutputFormatting.swift))
and the terminal command classifier that recognizes prettier and
rustfmt as known commands
([TerminalCommandClassifier.swift](swift/TurboSparkApp/Sources/TurboSparkApp/Tools/Terminal/TerminalCommandClassifier.swift)).
The model may run a formatter tool if it thinks to; nothing guarantees
the pass. OpenCode resolves configured formatters per file extension,
checks executable availability, and formats files after agent writes.

## Goals

* A formatter registry: file extension or language to formatter
  command template. V1 rows: swift-format (toolchain), gofmt, and
  clang-format, plus user-configured rows (prettier, ruff, rustfmt) with
  explicit commands.
* Availability check before each pass: the executable must resolve on
  the login-shell PATH. Missing executable means silent skip, recorded
  in the pass log.
* Per-project toggle with a default off; enabling it is the user's
  explicit statement that the formatter may rewrite the agent's output
  on disk.
* The pass runs after a successful edit-tool write, through the
  existing hardened process path: argument vectors, no shell, bounded
  time.
* On failure the edit tool's follow-up output gains a short note with
  the formatter's error tail. On success a one-line "formatted by X"
  note. The file content shown in the transcript stays the post-format
  content.
* The pass never formats files the agent did not write in that turn,
  and never runs on files outside the project root.

## Non-Goals

* No bundled formatter binaries.
* No v1 support for formatters that need project-wide context beyond
  the target file (no daemon-based servers, no whole-project passes).
* No auto-enablement by file extension alone: the project toggle is
  off until the user turns it on.
* No format-on-agent-prompt for non-edit tools (no reformatting of
  files the agent only read).

## Open Questions

* **Registry contents for v1**: the three toolchain-adjacent rows
  (swift-format, gofmt, clang-format) plus user rows; or user rows only.
  Lean: include the three, since availability is checked and missing
  means skip.
* **Which tools trigger the pass**: the file-write and edit tools only,
  or any tool that returns a written-path result. Lean: write/edit
  tools only in v1.

## Critical User Journeys

### CUJ 1: style pass after edit

* **Given**: the project enables the pass; swift-format is installed;
  the agent writes a Swift file with bad style.
* **When**: the edit completes.
* **Then**: the file is formatted, and the transcript shows the
  formatted content with the "formatted by" note.

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
Edit tool success (path within project root)
        |
        v
FormatterRegistry.match(path extension/language)
  availability check (login-shell PATH)
        | match + available + project toggle on
        v
FormatterRunner (hardened process, bounded time)
  rewrite file in place
  success: note | failure: error tail into follow-up output
```

## Non-Functional Requirements

* **Performance**: the pass adds at most the formatter's own runtime;
  the bounded time caps pathological cases.
* **Safety**: the formatter is a user-chosen binary running on a file
  the agent already wrote under an approved edit; the project toggle
  plus the write permission cover the act. Argument vectors, no shell,
  and the time bound follow the existing process hardening rules.
* **Localization**: project toggle copy and the note templates in 21
  languages in the same change.
* **Accessibility**: no new controls beyond the project toggle row;
  notes announce via the existing tool-output announcements.

## Detailed Implementation Plan

### Target Files to Create/Modify

1. `swift/TurboSparkApp/Sources/TurboSparkApp/Tools/Format/FormatterRegistry.swift` (Create)
   * **Rationale**: the extension-to-command registry, availability
     check, and per-project enable records.
2. `swift/TurboSparkApp/Sources/TurboSparkApp/Tools/Format/FormatterRunner.swift` (Create)
   * **Rationale**: the hardened run, time bound, and note/error-tail
     outputs.
3. `swift/TurboSparkApp/Sources/TurboSparkApp/State/AppModel+Tools.swift` (Modify)
   * **Rationale**: the post-write hook after edit-tool success.
4. `swift/TurboSparkApp/Sources/TurboSparkApp/Generation/ProjectPermissionsSectionView.swift` (Modify)
   * **Rationale**: the per-project formatting pass toggle row.
5. `swift/TurboSparkApp/Sources/TurboSparkApp/State/AppModel+Persistence.swift` (Modify)
   * **Rationale**: persist per-project pass records with the archive.
6. `swift/TurboSparkApp/Localization/Localizable.xcstrings` (Modify)
   * **Rationale**: all new strings, 21 languages, same change.
7. `swift/TurboSparkApp/Tests/TurboSparkAppTests/FormatterRegistryTests.swift` (Create)
   * **Rationale**: matching, availability gating, silent skip when
     missing. Mutation check: invert the availability gate and assert
     its test fails.

### Verification Plan

* **Build**: `make swift-lib`, then the app scheme build.
* **Tests**: the new suite plus localization gates and
  `make design-check`.
* **Manual**: enable the pass in a project with swift-format
  available. Have the agent write a badly styled file and confirm the
  pass, the note, and the silent-skip path.

--------------------------------------------------------------------------------

*Drafted with EGM*
