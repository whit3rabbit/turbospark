Status: resolved

## Applied

* Applied: B1, B2, M1 (including the INDEX.md step 3 edit), M2, M3, M4, M5, M7, M8, M9, M10, N1, N2, N3, N4, N5, N6, N7. M6 was satisfied through N7's diagram body (temp copy, replace only on exit 0 and unchanged hash), which replaced the "rewrite file in place" text.
* Unapplied: N8 (ESCALATE, needs author and SHA), N9 (DECLINE).
* Note: M3 and M10 emptied the Open Questions section, so it now reads "None open after review" (author to confirm).

# Goldfish Annotated Review Report

**Target:** `docs/designs/opencode2-parity/OC-11-formatter-pass.md` (full file, 164 lines)
**Findings source:** `consolidated_review_report.md` in this folder (21 findings: Blocker 2, Major 10, Nit 9). The target file was not edited.

## Summary

| Category | Count | Items |
| --- | --- | --- |
| CONTEST | 0 | none |
| ESCALATE | 1 | N8 |
| PROPOSE FIX | 19 | B1, B2, M1-M10, N1, N2, N3, N4, N5, N6, N7 |
| DECLINE | 1 | N9 |

Verified-false reviewer claims: none. Every code claim was read and holds:

* `LoginShellPath` exists (`swift/TurboSparkApp/Sources/TurboSparkApp/Tools/MCP/LoginShellPath.swift`, `enum LoginShellPath`, `resolved()`); it drops relative PATH entries, which helps M5.
* `ProcessExecutor.run(... timeoutSeconds ...)` and `CappedOutputBuffer` exist (`Tools/Core/ProcessExecutor.swift`).
* `PathContainment.isContained` / `resolvedIfContained` exist (`State/PathContainment.swift`).
* `FileSnapshotStore.recordSnapshot` / `isStale` / `recordBackup` exist (`Tools/File/FileSnapshotStore.swift`), so the stale-hash risk in M4 is real.
* `TerminalCommandClassifier.buildCommands` lists prettier and rustfmt (`Tools/Terminal/TerminalCommandClassifier.swift`), as N1 says.
* The INDEX double listing is real: OC-11 sits in step 3 ("after OC-10's seam", INDEX line 95) while OC-10 is in step 7 (line 99).

Two nuances, neither a false claim:

* M1: OC-10 does name a "post-edit hook" in `State/AppModel+Tools.swift` (its file 7), but it never calls it a shared seam, and OC-11 does not depend on it in its own text.
* The Stage 5 caveat that reuse targets were "not re-verified" is now closed: all three exist. `LoginShellPath` lives under `Tools/MCP/`, so reusing it from `Tools/Format/` is a cross-folder use of a module-internal enum. State that in the Dependencies line.

Items that need the author, marked "(author to confirm)" below: the Alternatives wording, the 10 s timeout, the fixed swift-format command, and the Risks table rows. These are invented by me from the doc and the findings, not taken from the author.

Security items (M2, M3, M5, M6, M9, B2) all keep a conservative default: off by default, the toggle is the trust grant, no in-project executables, no shell, absolute executable resolved once.

## Prioritized Findings & AI Suggestions

### [Issue 1] [`design / structural / synthesizer`] - Severity: Blocker (B1)
*   **Affected Scope:** [Lines 1-164](../../../designs/opencode2-parity/OC-11-formatter-pass.md#L1-L164) (whole file, no Alternatives section)
*   **Discrepancy Description:** No Alternatives Considered section; the design reads as a single path.
*   **Rationale:** Reviewers cannot tell which options were weighed, so a reader cannot judge whether the registry, toggle and in-place rewrite are the right shape.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Valid. The doc has no such section (checked all headers). The alternatives below come from the doc's own Goals, Non-Goals and Open Questions plus the findings.
*   ***Proposed Edit:***
    *   Location: new `## Alternatives Considered` between `## Non-Goals` (ends line 66) and `## Open Questions` (line 68). (author to confirm all four bullets)
    *   Change: insert
        ```
        ## Alternatives Considered

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
        ```

### [Issue 2] [`design / structural / synthesizer`] - Severity: Blocker (B2)
*   **Affected Scope:** [Lines 117-128](../../../designs/opencode2-parity/OC-11-formatter-pass.md#L117-L128) (Non-Functional Requirements)
*   **Discrepancy Description:** No Risks and Mitigations section.
*   **Rationale:** The doc runs a user-chosen binary that rewrites files. Stale model view, code execution from formatter config, write race, hang and root escape all need a stated mitigation.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Valid. Security: the risk of running user-configured commands against repo content is a real finding, not a style gap. The mitigations below use what the repo already has (`ProcessExecutor` timeout and output cap, `PathContainment`, `FileSnapshotStore`, `LoginShellPath` dropping relative PATH entries).
*   ***Proposed Edit:***
    *   Location: new `## Risks and Mitigations` after `## Non-Functional Requirements` (ends line 128). (author to confirm rows and the 10 s value)
    *   Change: insert
        ```
        ## Risks and Mitigations

        | Risk | Mitigation |
        | --- | --- |
        | The model's next edit anchors on pre-format content | The tool result carries post-format content and the edit tool's recorded hash is refreshed (see Goals). |
        | Formatter config in the repo executes code (for example a plugin) | Config is trusted input only because the user enabled the pass for this project. v1 ships swift-format, whose config is data. User rows are the user's own risk; the toggle copy says so. Default off. |
        | Write races a user save | Format a temp copy, replace only on exit 0, skip with a failure note if the file hash changed. |
        | Formatter hangs or floods output | `ProcessExecutor` timeout (10 s), kill on expiry, capped error tail, cancel on turn stop. |
        | Path escapes the project root | `PathContainment` with symlink resolution before any run. |
        | A planted binary in the repo is run | Executable resolved once to an absolute path from the login-shell PATH; relative PATH entries are dropped; no in-project executables in v1. |
        ```

### [Issue 3] [`zoom-out / design / structural / synthesizer`] - Severity: Major (M1)
*   **Affected Scope:** [Lines 18-22](../../../designs/opencode2-parity/OC-11-formatter-pass.md#L18-L22), [134-141](../../../designs/opencode2-parity/OC-11-formatter-pass.md#L134-L141); `docs/designs/opencode2-parity/INDEX.md` lines 41, 83-84, 95, 99
*   **Discrepancy Description:** OC-10 post-edit seam and reuse targets not named; INDEX lists OC-11 in step 3 and step 7.
*   **Rationale:** Two specs hook the same edit success; ordering and reuse are unstated.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Valid. Verified: OC-10 file 7 modifies `State/AppModel+Tools.swift` ("the post-edit hook with the bounded wait"), the same file as OC-11 file 3. INDEX line 95 puts OC-11 in step 3 "after OC-10's seam" while OC-10 is only in step 7 (line 99); the sequence contradicts the dependency. Reuse targets verified present (see summary).
*   ***Proposed Edit:***
    *   Location 1: add `## Dependencies` after `## Overview` (ends line 22). Change:
        ```
        ## Dependencies

        * Registers on OC-10's post-edit hook in `AppModel+Tools.swift`
          rather than adding a second one. Formatting runs first, then
          OC-10 diagnostics.
        * Reuses `LoginShellPath` (PATH capture, relative entries dropped),
          `ProcessExecutor` (timeout, capped output) and `PathContainment`
          (root check). `LoginShellPath` lives under `Tools/MCP/`; moving
          it to a shared folder is part of this change.
        ```
    *   Location 2 (INDEX.md, separate file, author to approve): remove OC-11 from step 3 (line 95) and keep it in step 7 (line 99), so it reads "3. OC-02, OC-14 (medium; OC-02 after ZP-02)".

### [Issue 4] [`zoom-out / inventive / design / structural / style / synthesizer`] - Severity: Major (M2)
*   **Affected Scope:** [Lines 18-19](../../../designs/opencode2-parity/OC-11-formatter-pass.md#L18-L19), [45-47](../../../designs/opencode2-parity/OC-11-formatter-pass.md#L45-L47), [119-124](../../../designs/opencode2-parity/OC-11-formatter-pass.md#L119-L124) (Safety)
*   **Discrepancy Description:** Safety claim asserted with no permission-ladder position; user rows could plant commands; formatter config can execute code.
*   **Rationale:** The pass runs a user-chosen binary outside any stated gate.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Valid, and it is a security finding. Verified: `AppToolPermissionEngine.evaluate` is the ladder (category deny, high-risk gate, session approval); the doc never says where the pass sits. The write permission covers the file write, not a second process spawn. Conservative resolution: the pass is app-initiated, runs only after an edit tool already cleared the ladder in a mode that allows writes, is never reachable from a model-supplied command string, and the per-project toggle is the explicit trust grant. User rows are stored only in the app archive (not read from the repo), so a cloned repo cannot add one. Residual risk (formatter config in the repo is trusted once the toggle is on) is recorded, not hidden.
*   ***Proposed Edit:***
    *   Location: replace the `Safety` bullet (lines 121-124). (author to confirm)
    *   Change:
        ```
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
        ```

### [Issue 5] [`zoom-out / inventive / design / structural / style / synthesizer`] - Severity: Major (M3)
*   **Affected Scope:** [Lines 38-41](../../../designs/opencode2-parity/OC-11-formatter-pass.md#L38-L41), [70-73](../../../designs/opencode2-parity/OC-11-formatter-pass.md#L70-L73)
*   **Discrepancy Description:** Goals state what Open Questions leave open; v1 scope is wide; command lines are missing.
*   **Rationale:** An implementer cannot tell the v1 registry or the exact commands.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Valid. Goals (lines 38-41) already list swift-format, gofmt, clang-format while Open Question 1 (lines 70-73) asks whether to include them. Security-conservative resolution: fewer built-in rows, each a fixed command, user rows as examples.
*   ***Proposed Edit:***
    *   Location: replace the first Goals bullet (lines 38-41) and delete Open Question 1 (lines 70-73). (author to confirm the command line)
    *   Change:
        ```
        * A formatter registry matched by file extension. V1 ships one
          built-in row: swift-format (`.swift`, run as
          `swift-format format --in-place <temp copy>`). Other formatters
          (gofmt, clang-format, prettier, ruff, rustfmt) are user rows
          with an explicit executable and argument vector; the doc's
          examples show them but the app ships none.
        ```
        Remove the "Registry contents for v1" Open Question.

### [Issue 6] [`zoom-out / inventive / comprehension / design`] - Severity: Major (M4)
*   **Affected Scope:** [Lines 48-56](../../../designs/opencode2-parity/OC-11-formatter-pass.md#L48-L56), [99-100](../../../designs/opencode2-parity/OC-11-formatter-pass.md#L99-L100)
*   **Discrepancy Description:** What the model sees after the rewrite is unspecified; the next edit may anchor on stale content; undo and OC-10 ordering unstated.
*   **Rationale:** A silent rewrite desyncs the model's view from disk.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Valid. Verified: `ApplyPatchExecutor` line 9 and 257 record the post-write hash in `FileSnapshotStore` "or the NEXT tool call on the same file" fails the stale check (`isStale`, line 121). A formatter rewrite after that hash makes the next edit refuse as stale. `recordBackup` / `backupMatchesCurrent` guard undo "only while the file still matches" the post-write hash, so undo breaks too.
*   ***Proposed Edit:***
    *   Location: replace the last sentence of Goal 4 (lines 53-54, "The file content shown in the transcript stays the post-format content") and add one bullet. (author to confirm)
    *   Change:
        ```
        * After a successful format the edit tool's result carries the
          post-format content, `FileSnapshotStore.recordSnapshot` is
          refreshed with it so the next edit is not rejected as stale, and
          the undo backup keeps the agent's pre-format write.
        ```

### [Issue 7] [`inventive / comprehension / design / synthesizer`] - Severity: Major (M5)
*   **Affected Scope:** [Lines 42-44](../../../designs/opencode2-parity/OC-11-formatter-pass.md#L42-L44), [49-50](../../../designs/opencode2-parity/OC-11-formatter-pass.md#L49-L50), [119-124](../../../designs/opencode2-parity/OC-11-formatter-pass.md#L119-L124)
*   **Discrepancy Description:** "Login-shell PATH" appears to conflict with "no shell"; toolchain and project-local binaries are unresolved.
*   **Rationale:** A reader cannot tell whether a shell runs.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Valid as a comprehension gap. Verified: `LoginShellPath.capture` spawns an `-il` shell once to print PATH (markers in the file), and only PATH leaves it; `LoginShellPath.parse` drops relative entries. The formatter itself runs without a shell. Security: a relative entry like `node_modules/.bin` would let a cloned repo plant the binary, so keep them excluded and ban in-project executables.
*   ***Proposed Edit:***
    *   Location: replace Goal 2 (lines 42-44). Change:
        ```
        * Availability check: the executable is resolved once to an
          absolute path using the PATH captured by `LoginShellPath` (one
          login-shell probe, cached, relative entries dropped). The
          formatter itself runs as an argument vector with no shell.
          Executables inside the project tree are refused in v1. A
          missing executable is a silent skip.
        ```
        Note in Safety that the first pass runs the user's shell rc, as the MCP first spawn already does.

### [Issue 8] [`inventive / design / synthesizer`] - Severity: Major (M6)
*   **Affected Scope:** [Lines 48-53](../../../designs/opencode2-parity/OC-11-formatter-pass.md#L48-L53), [99-100](../../../designs/opencode2-parity/OC-11-formatter-pass.md#L99-L100)
*   **Discrepancy Description:** In-place rewrite is not atomic and can race with user saves.
*   **Rationale:** A partial write or a user save in the same window loses data.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Valid; the Architecture says "rewrite file in place" (line 113). Smallest safe change: format a temp copy and replace only when the formatter exits 0 and the original hash is unchanged. A failed run then cannot corrupt the file, which also makes CUJ 3's "file keeps the agent's content" true by construction.
*   ***Proposed Edit:***
    *   Location: replace "rewrite file in place" (line 113). Change: `copy to temp, format the copy, replace the original only on exit 0 and an unchanged file hash (else skip with a failure note)`.

### [Issue 9] [`comprehension / design / style / synthesizer`] - Severity: Major (M7)
*   **Affected Scope:** [Lines 49](../../../designs/opencode2-parity/OC-11-formatter-pass.md#L49), [112](../../../designs/opencode2-parity/OC-11-formatter-pass.md#L112), [119-120](../../../designs/opencode2-parity/OC-11-formatter-pass.md#L119-L120)
*   **Discrepancy Description:** "Bounded time" has no value, kill, or output cap.
*   **Rationale:** Not testable as written.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Valid. Verified: `ProcessExecutor.run` takes `timeoutSeconds` and caps output via `CappedOutputBuffer(capBytes:)`, so the doc can state values instead of a vague bound. The 10 s figure is the reviewer's example, so it needs the author's confirmation.
*   ***Proposed Edit:***
    *   Location: Goal 4 (line 49) and Performance bullet (lines 119-120). Change: replace "bounded time" with "a 10 second timeout enforced by `ProcessExecutor`, process killed on expiry, error tail capped (for example 2 KB), cancelled when the turn is stopped". (author to confirm values)

### [Issue 10] [`comprehension / design / style / inventive`] - Severity: Major (M8)
*   **Affected Scope:** [Lines 44](../../../designs/opencode2-parity/OC-11-formatter-pass.md#L44), [88-93](../../../designs/opencode2-parity/OC-11-formatter-pass.md#L88-L93), [118](../../../designs/opencode2-parity/OC-11-formatter-pass.md#L118)
*   **Discrepancy Description:** "Pass log" is undefined and contradicts CUJ 2.
*   **Rationale:** Line 44 records the skip in a log; CUJ 2 says nothing is surfaced.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Valid. No "pass log" exists anywhere in the repo (`FormatterRunner`/`FormatterRegistry` return no matches either), so this is an undefined artifact, not a reference to existing code. Resolved as in Issue 7: missing formatter is a silent skip.
*   ***Proposed Edit:***
    *   Location: line 43-44. Change: delete ", recorded in the pass log" (covered by the Issue 7 replacement).

### [Issue 11] [`comprehension / inventive / design / synthesizer`] - Severity: Major (M9)
*   **Affected Scope:** [Lines 55-56](../../../designs/opencode2-parity/OC-11-formatter-pass.md#L55-L56)
*   **Discrepancy Description:** "Files written this turn" and the project-root rule have no mechanism.
*   **Rationale:** An implementer must invent the dirty-set and the containment check.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Valid. Verified: `PathContainment.resolvedIfContained(_:in:)` resolves symlinks, which is the right check; a symlink in the repo pointing to a file outside the root is exactly the class of bug its header documents. The edit tools exist (see Issue 12), so "written this turn" can mean the path returned by a successful edit tool call, with no separate dirty-set.
*   ***Proposed Edit:***
    *   Location: replace Goal 6 (lines 55-56). Change:
        ```
        * The pass formats only the path returned by a successful edit
          tool call in that turn, and only if
          `PathContainment.resolvedIfContained` accepts it under the
          project root (symlinks resolved). Anything else is skipped.
        ```

### [Issue 12] [`comprehension / structural / style`] - Severity: Major (M10)
*   **Affected Scope:** [Lines 28, 31](../../../designs/opencode2-parity/OC-11-formatter-pass.md#L28), [48](../../../designs/opencode2-parity/OC-11-formatter-pass.md#L48), [74-76](../../../designs/opencode2-parity/OC-11-formatter-pass.md#L74-L76)
*   **Discrepancy Description:** Triggering tools are unnamed (named three ways); relative Swift links resolve to nowhere.
*   **Rationale:** Broken links and an open "which tools" question.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Valid on both counts. Tools verified: `FileWrite` and `FileEdit` (`Tools/File/FileReadWriteTools.swift`), `multiedit` (`MultiEditToolDefinitions.swift`) and `apply_patch` (`ApplyPatchTool.swift`) are the write tools. Links: the doc sits in `docs/designs/opencode2-parity/`, so `swift/TurboSparkApp/...` resolves to `docs/designs/opencode2-parity/swift/...`, which does not exist; it needs `../../../`. (OC-10 uses the same broken form; not in scope here.)
*   ***Proposed Edit:***
    *   Location 1: lines 28 and 31, prefix both Swift link targets with `../../../`.
    *   Location 2: replace the "Which tools trigger the pass" Open Question (lines 74-76) and Goal 4's "edit-tool write". Change: `The pass runs after a successful FileWrite, FileEdit, multiedit or apply_patch call. No other tool triggers it in v1.`

### [Issue 13] [`zoom-out / style`] - Severity: Nit (N1)
*   **Affected Scope:** [Lines 26-31](../../../designs/opencode2-parity/OC-11-formatter-pass.md#L26-L31)
*   **Discrepancy Description:** The ShellOutputFormatting citation is irrelevant; TerminalCommandClassifier already auto-approves formatters.
*   **Rationale:** Misleading evidence for "no pipeline" and a missed fact.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Half valid. The classifier fact is true (`buildCommands` includes `prettier` and `rustfmt`, auto-approvable under `.auto`). `ShellOutputFormatting` is not irrelevant once N2 is applied: it strips ANSI from the error tail. Keep it, re-labelled as a reuse target.
*   ***Proposed Edit:***
    *   Location: lines 26-31. Change: reword to "The only formatter-related code is `ShellOutputFormatting` (output display, reused here to strip ANSI from the error tail) and `TerminalCommandClassifier`, which already lists prettier and rustfmt as auto-approvable build commands when the model runs them itself. No code runs a formatter on the app's behalf." Say "found no pipeline by search" instead of "No formatter pipeline exists".

### [Issue 14] [`zoom-out / style / synthesizer`] - Severity: Nit (N2)
*   **Affected Scope:** [Lines 51-54](../../../designs/opencode2-parity/OC-11-formatter-pass.md#L51-L54), [99-100](../../../designs/opencode2-parity/OC-11-formatter-pass.md#L99-L100)
*   **Discrepancy Description:** Success note adds noise and localization cost; error tail uncapped.
*   **Rationale:** A note on every write costs tokens and 21 translations.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Valid and cheap. Failure-only note, capped, ANSI-stripped. CUJ 1 line 86 loses the "formatted by" note.
*   ***Proposed Edit:***
    *   Location: Goal 4 (lines 51-54), CUJ 1 line 86. Change: "On failure the follow-up output gains a note with the formatter's error tail (capped, ANSI stripped via `ShellOutputFormatting`). Success adds no note." CUJ 1 Then: "the file is formatted and the transcript shows the formatted content."

### [Issue 15] [`zoom-out / comprehension / design / style`] - Severity: Nit (N3)
*   **Affected Scope:** [Lines 125-128](../../../designs/opencode2-parity/OC-11-formatter-pass.md#L125-L128), [146-151](../../../designs/opencode2-parity/OC-11-formatter-pass.md#L146-L151), [155-157](../../../designs/opencode2-parity/OC-11-formatter-pass.md#L155-L157)
*   **Discrepancy Description:** Program constraints and test mutation detail are over-specified; the runner has no tests.
*   **Rationale:** `FormatterRunner` has the most dangerous behavior and no test file.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Valid on the missing test (file list has only `FormatterRegistryTests.swift`, item 7). Add one test file line; leave the localization and design-check lines as they are (the INDEX "Program constraints" section carries the same wording, and removing it from one doc breaks the pattern across the set).
*   ***Proposed Edit:***
    *   Location: after file 7 (line 151). Change: add `8. swift/TurboSparkApp/Tests/TurboSparkAppTests/FormatterRunnerTests.swift (Create)` with rationale "timeout kills the process, a failing formatter keeps the original file, a path outside the root is skipped, a changed file hash skips the replace."

### [Issue 16] [`style / structural`] - Severity: Nit (N4)
*   **Affected Scope:** [Lines 19-21](../../../designs/opencode2-parity/OC-11-formatter-pass.md#L19-L21), [65](../../../designs/opencode2-parity/OC-11-formatter-pass.md#L65), [70-73](../../../designs/opencode2-parity/OC-11-formatter-pass.md#L70-L73)
*   **Discrepancy Description:** Wording: "guaranteed", "Lean:", "format-on-agent-prompt", "toolchain-adjacent".
*   **Rationale:** Vague or unmeasurable phrases.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Valid for "guaranteed" (the pass is skipped when the formatter is missing, so it is not guaranteed) and "format-on-agent-prompt" (line 65 means non-edit tools). "Toolchain-adjacent" disappears with Issue 5. "Lean:" matches the convention in OC-10 (line 78), so leave it alone to stay consistent with the set.
*   ***Proposed Edit:***
    *   Location: line 19-20 "A guaranteed style pass per write" becomes "A style pass after each agent write when a formatter is available"; line 65 "No format-on-agent-prompt for non-edit tools" becomes "No formatting for tools that only read files".

### [Issue 17] [`design / synthesizer`] - Severity: Nit (N5)
*   **Affected Scope:** [Lines 144-145](../../../designs/opencode2-parity/OC-11-formatter-pass.md#L144-L145)
*   **Discrepancy Description:** No statement that a missing persistence key decodes to off.
*   **Rationale:** Old archives must not enable a code-running pass.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Valid and security relevant: the default must be off for every project that has no record. `AppJSONStore` quarantines corrupt files rather than overwriting, so a decode fallback must also resolve to off.
*   ***Proposed Edit:***
    *   Location: rationale of file 5 (line 145). Change: append "A missing or unreadable key decodes to off."

### [Issue 18] [`structural / synthesizer`] - Severity: Nit (N6)
*   **Affected Scope:** [Lines 134-136](../../../designs/opencode2-parity/OC-11-formatter-pass.md#L134-L136), [144-145](../../../designs/opencode2-parity/OC-11-formatter-pass.md#L144-L145)
*   **Discrepancy Description:** Registry and persistence both own per-project records.
*   **Rationale:** Two owners for one record.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Valid; file 1 (registry, "per-project enable records") and file 5 (persistence, "per-project pass records") overlap.
*   ***Proposed Edit:***
    *   Location: file 1 rationale (lines 135-136). Change: delete "and per-project enable records"; persistence (file 5) owns them.

### [Issue 19] [`structural`] - Severity: Nit (N7)
*   **Affected Scope:** [Lines 104-115](../../../designs/opencode2-parity/OC-11-formatter-pass.md#L104-L115)
*   **Discrepancy Description:** Diagram checks the toggle after the match; the failure branch is crammed.
*   **Rationale:** The toggle should be the first check because it is the trust grant.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Valid and security relevant: checking the toggle first means a disabled project never resolves an executable or probes the login shell.
*   ***Proposed Edit:***
    *   Location: replace the diagram body. Change:
        ```
        Edit tool success
          -> project toggle on? (no: stop)
          -> path contained in project root? (no: stop)
          -> FormatterRegistry.match(extension)
          -> executable resolved on login-shell PATH? (no: silent skip)
          -> FormatterRunner: temp copy, run, timeout
               exit 0 and file unchanged: replace original
               failure or changed file: keep original, add error tail
        ```

### [Issue 20] [`design / style`] - Severity: Nit (N8)
*   **Affected Scope:** [Lines 3-8](../../../designs/opencode2-parity/OC-11-formatter-pass.md#L3-L8), [19-23](../../../designs/opencode2-parity/OC-11-formatter-pass.md#L19-L23)
*   **Discrepancy Description:** Parity links point at the drifting `dev` branch; the motivation is uncited.
*   **Rationale:** A link to `dev` can change under the reader.
*   ***AI Suggestion:*** ESCALATE
*   ***Justification:*** The fix depends on a commit SHA (`7ce35879b8`) that lives in the external OpenCode repository, not in any file in this checkout. I could not read or verify it, and the sibling docs (OC-10 lines 8-9) also link to `dev`. The author needs to decide whether to pin links across the whole OC set and supply the SHA. No edit is proposed here.
*   ***Proposed Edit:***
    *   Location: none until the author confirms the SHA and the policy for sibling docs.
    *   Change: none.

### [Issue 21] [`inventive`] - Severity: Nit (N9)
*   **Affected Scope:** [Lines 51-52](../../../designs/opencode2-parity/OC-11-formatter-pass.md#L51-L52), [99-100](../../../designs/opencode2-parity/OC-11-formatter-pass.md#L99-L100)
*   **Discrepancy Description:** A broken formatter pays its full bound and appends an error tail on every write; optional per-formatter circuit breaker.
*   **Rationale:** Repeated waste on a persistently failing formatter.
*   ***AI Suggestion:*** DECLINE
*   ***Justification:*** Nit, optional by the finding's own wording, and the Stage 8 reviewer recommends rejecting it for v1. The 10 s timeout and capped tail (Issues 9 and 14) already bound the cost per write. Recorded for a later pass if real use shows repeated failures.
*   ***Proposed Edit:***
    *   Location: none.
    *   Change: none.

## Gate

I have compiled the audit findings into this file. Item 20 (N8) is the only ESCALATE and needs your decision. The INDEX.md ordering fix in Issue 3 touches a second file and needs your approval before it is applied.
